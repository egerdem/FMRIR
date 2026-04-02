import torch
import numpy as np
import os
import json
import glob
import hashlib
from tqdm import tqdm
import matplotlib
matplotlib.use('Qt5Agg', force=True)  # Same as eval_AUTOENCODER.py
from matplotlib import pyplot as plt
from inference import model_factory, load_model_and_config
from model_paths import MULTI_MODEL_PATHS

# Your model imports
from fm_utils import (
    ATF3DSampler, SetEncoder,
    CrossAttentionUNet3D, CrossAttentionUNet3D_RED3d,
    CFGVectorFieldODE_3D, CFGVectorFieldODE_3D_V2, EulerSimulator,
    get_model_info, print_model_info
)

# Reference model imports
import sys
sys.path.append('AUTOENCODER/src')
import AUTOENCODER.src.dataset as autoencoder_dataset
from AUTOENCODER.src.configs import config_FSMPAE_10026
import AUTOENCODER.src.utils as autoencoder_utils
from inference import calculate_lsd_unified

# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def load_reference_model(device):
    """Load FSMPAE predictions + PT ground truth with explicit mic-order alignment."""
    config = config_FSMPAE_10026.copy()
    config["use_processed_pt"] = True

    # Load model predictions
    pt_dir = 'AUTOENCODER/outputs/out_20250323_FSMPAE_10026'
    dataset_name = config['dataset'][0]
    pt_path = f'{pt_dir}/atf_mag/atf_mag_test_{dataset_name}.pt'

    atf_mag_est = torch.load(pt_path, weights_only=False)

    # Load PT-order GT from processed files (canonical FM order).
    idataset_pt = autoencoder_dataset.ATFdataset(config=config)
    data = idataset_pt.Data
    atf_mag_gt = data['test']['atf_mag'][dataset_name]

    # Build explicit AE->PT mic permutation from coordinates.
    # FSMPAE predictions are in AE/NPZ mic order; PT GT is in canonical FM order.
    data_root = config.get("data_dir", autoencoder_dataset._DATA_DIR)
    data_root = os.path.expanduser(data_root)
    dataset_dir = os.path.join(data_root, dataset_name)
    src_all = list(config['src_index'][dataset_name]['all'])
    if len(src_all) == 0:
        raise ValueError(f"Empty src_index['all'] for dataset '{dataset_name}'.")
    first_src = src_all[0]
    npz_path = os.path.join(dataset_dir, f"data_s{first_src + 1:04d}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Cannot build AE->PT mic mapping; missing NPZ file: {npz_path}")
    with np.load(npz_path) as npz_data:
        pos_mic_ae = npz_data['posMic'].astype(np.float64)  # [1331, 3] in AE order

    mic_pos_pt = data['test']['mic_position'][dataset_name][:, :, 0].cpu().numpy().astype(np.float64)  # [1331, 3]
    if pos_mic_ae.shape != mic_pos_pt.shape:
        raise ValueError(
            f"Mic coordinate shape mismatch while building mapping: "
            f"AE={pos_mic_ae.shape}, PT={mic_pos_pt.shape}"
        )

    # Robust hash map on rounded coordinates (avoids float representation mismatches).
    coord_decimals = 6
    pt_coord_to_idx = {}
    for i, c in enumerate(mic_pos_pt):
        key = tuple(np.round(c, coord_decimals))
        if key in pt_coord_to_idx:
            raise ValueError(f"Duplicate PT mic coordinate key detected at index {i}: {key}")
        pt_coord_to_idx[key] = i

    perm_ae_to_pt = np.empty(pos_mic_ae.shape[0], dtype=np.int64)
    for i, c in enumerate(pos_mic_ae):
        key = tuple(np.round(c, coord_decimals))
        if key not in pt_coord_to_idx:
            raise ValueError(
                f"Failed to map AE mic index {i} with coord {key} into PT grid coordinates. "
                "Mic-order alignment cannot be established."
            )
        perm_ae_to_pt[i] = pt_coord_to_idx[key]

    # Reorder FSMPAE predictions into PT canonical mic order.
    # perm_ae_to_pt maps AE index -> PT index, so for PT-ordered tensor we need inverse map.
    inv_perm_pt_to_ae = np.argsort(perm_ae_to_pt)
    atf_mag_est = atf_mag_est[inv_perm_pt_to_ae, :, :]

    fsmpae_model_freq_up_to = atf_mag_est.shape[1]
    print(f"Reference data loaded: {atf_mag_gt.shape} (Mic, Freq, Src)")
    print(f"FSMPAE model_freq_up_to: {fsmpae_model_freq_up_to}")
    print(
        f"FSMPAE predictions reordered to PT canonical mic order "
        f"(using {len(perm_ae_to_pt)}-mic coordinate mapping, inverse permutation)."
    )

    # Return FULL reference data (don't truncate here - let evaluation function handle it)
    return atf_mag_est, atf_mag_gt, config, data, fsmpae_model_freq_up_to, inv_perm_pt_to_ae

def evaluate_your_model(set_encoder, ode_3d, config, M_values, device, num_sources_eval=None,
                        guidance_scales=None, random_M_sampling=False, model_name=None,
                        normalize_coords=False, coord_mean=None, coord_std=None,
                        eval_freq_up_to=None, model_path=None):
    """
    Evaluate your 3D model.
    
    Args:
        guidance_scales: List of guidance scale values to evaluate. If None, defaults to [1.0, 2.0].
    """

    data_dir = "ir_fs2000_s8192_m1331_room4.0x6.0x3.0_rt200/"
    src_split = config['data']['src_splits']
    model_freq_up_to = config['model'].get('freq_up_to')
    freq_from  = config['model'].get('freq_from', 0)
    if eval_freq_up_to is None:
        eval_freq_up_to = model_freq_up_to
    if eval_freq_up_to > model_freq_up_to:
        raise ValueError(
            f"eval_freq_up_to={eval_freq_up_to} cannot be greater than "
            f"model_freq_up_to={model_freq_up_to} for model '{model_name}'."
        )
    print(f"  SFlow model_freq_up_to={model_freq_up_to}, eval_freq_up_to={eval_freq_up_to}")

    # Detect geo_conditioning from checkpoint config and parse room dims
    _geo = config.get('training', {}).get('geo_conditioning', False)
    _room_dims = None
    if _geo:
        import re as _re_g
        _cfg_dir = config.get('data', {}).get('data_dir', data_dir)
        _rm = _re_g.search(r'room(\d+\.?\d*)x(\d+\.?\d*)x(\d+\.?\d*)', _cfg_dir)
        if _rm:
            _room_dims = (float(_rm.group(1)), float(_rm.group(2)), float(_rm.group(3)))
            print(f"  Geo-conditioning active: room_dims={_room_dims}, coord_dim=9")
        else:
            print("  WARNING: geo_conditioning=True but room dims not found in data_dir. Using rel-only coords.")
            _geo = False
    
    # Load data
    train_sampler = ATF3DSampler(
        data_path=data_dir, mode='train', src_splits=src_split,
        normalize=True, freq_up_to=model_freq_up_to, freq_from=freq_from, model_name=model_name
    )
    test_sampler = ATF3DSampler(
        data_path=data_dir, mode='test', src_splits=src_split,
        normalize=False, freq_up_to=model_freq_up_to, freq_from=freq_from, model_name=model_name
    )
    test_sampler.cubes = (test_sampler.cubes - train_sampler.mean) / (train_sampler.std + 1e-8)
    
    # Limit evaluation to first N sources if specified
    total_sources = len(test_sampler)
    if num_sources_eval is not None:
        eval_sources = min(num_sources_eval, total_sources)
        print(f"Evaluating on first {eval_sources} sources (out of {total_sources})")
    else:
        eval_sources = total_sources
    
    grid_xyz = train_sampler.grid_xyz.to(device)
    spec_std = train_sampler.std.item()
    
    simulator = EulerSimulator(ode=ode_3d)
    results = {}
    
    # Load the SAME microphone selection strategy as reference model
    idx_mes_pos_path = "AUTOENCODER/ATF_interp/idx_mes_pos_s1024_m1331.npy"
    idx_mes_pos_mat = np.load(idx_mes_pos_path)
    print(f"Loaded reference microphone selection matrix: {idx_mes_pos_mat.shape}")
    print("Using source-specific microphone selection (different M=5 mics per source)")
    model_dir_for_cache = os.path.dirname(os.path.abspath(model_path)) if model_path else None

    # Precompute denormalized GT in [Mic, Freq, Src] for fast cached-metric evaluation.
    gt_denorm_mic = []
    for i_src in range(eval_sources):
        z_true = test_sampler.cubes[i_src].unsqueeze(0).to(device)  # [1,F,D,H,W] normalized
        z_true_denorm = z_true * spec_std + train_sampler.mean.item()
        z_true_flat = z_true_denorm[:, :eval_freq_up_to].view(eval_freq_up_to, -1).T.contiguous()  # [1331, F]
        gt_denorm_mic.append(z_true_flat.cpu())
    gt_denorm_mic = torch.stack(gt_denorm_mic, dim=2)  # [1331, F, S]

    def _compute_metrics_from_predictions(pred_mic_freq_src):
        # pred_mic_freq_src: [1331, F, S], denormalized dB domain
        lsd_scores_local = []
        per_source_local = {}
        m_fundamental_indices = [0, 272, 665, 937, 1330]

        for src_idx in range(eval_sources):
            pred = pred_mic_freq_src[:, :eval_freq_up_to, src_idx]
            gt = gt_denorm_mic[:, :eval_freq_up_to, src_idx]

            lsd_db = calculate_lsd_unified(pred, gt, freq_dim=1).item()
            mse = torch.mean((pred - gt) ** 2).item()
            gt_var = torch.var(gt).item()
            nmse_linear = mse / gt_var if gt_var > 0 else float('inf')
            nmse = 10 * np.log10(nmse_linear) if nmse_linear > 0 and nmse_linear != float('inf') else float('inf')

            pred_mf = pred[m_fundamental_indices, :]
            gt_mf = gt[m_fundamental_indices, :]
            lsd_m_fund_db = calculate_lsd_unified(pred_mf, gt_mf, freq_dim=1).item()
            mse_m_fund = torch.mean((pred_mf - gt_mf) ** 2).item()

            source_errors = {
                'lsd': lsd_db, 'mse': mse, 'nmse': nmse,
                'lsd_m_fund': lsd_m_fund_db, 'mse_m_fund': mse_m_fund
            }
            lsd_scores_local.append(source_errors)
            per_source_local[src_idx] = source_errors
        return lsd_scores_local, per_source_local

    for M in M_values:
        results[M] = {}
        print(f"Evaluating your model with M={M} microphones...")
        
        for w in guidance_scales:
            results[M][w] = {}  # Initialize the dictionary for this guidance scale
            print(f"  Using guidance scale w={w}")
            lsd_scores = []
            cache_meta = None
            cache_path = None
            if model_dir_for_cache is not None:
                cache_meta = _build_atf_cache_meta(
                    model_path=model_path,
                    guidance=w,
                    M=M,
                    num_sources_eval=eval_sources,
                    model_freq_up_to=model_freq_up_to,
                    eval_freq_up_to=eval_freq_up_to,
                    random_M_sampling=random_M_sampling,
                    config=config,
                )
                cache_path = os.path.join(model_dir_for_cache, _atf_cache_filename(cache_meta))
                loaded_path, cached_predictions = _load_matching_atf_cache(model_dir_for_cache, cache_meta)
                if cached_predictions is not None and w in cached_predictions:
                    print(f"  ⚡ Using cached predictions for M={M}, w={w} from {loaded_path}")
                    pred_cached = cached_predictions[w].float().cpu()
                    lsd_scores, per_source_cached = _compute_metrics_from_predictions(pred_cached)
                    results[M][w]['per_source_errors'] = per_source_cached
                else:
                    loaded_path = None

            if cache_meta is not None and 'per_source_errors' in results[M][w]:
                # Cache hit path: metrics already prepared from cached predictions.
                pass
            else:
                pred_cache_tensor = torch.zeros((1331, eval_freq_up_to, eval_sources), dtype=torch.float32)
            # Evaluate each source with this guidance scale
                for i in tqdm(range(eval_sources), desc=f"Your Model M={M}, w={w}"):
                    with torch.no_grad():
                        z_true = test_sampler.cubes[i].unsqueeze(0).to(device)
                        src_xyz = test_sampler.source_coords[i].unsqueeze(0).to(device)

                        # or choose randomly
                        if random_M_sampling:
                            source_specific_indices = torch.randperm(grid_xyz.shape[0])[:M]
                        else: # Use source-specific microphones (different M=5 for each source)
                            source_specific_indices = idx_mes_pos_mat[:M, i]  # First M mics for this source

                        obs_indices = torch.tensor(source_specific_indices, dtype=torch.long, device=device)

                        obs_xyz_abs = grid_xyz[obs_indices]
                        obs_coords_rel = (obs_xyz_abs - src_xyz)  # [M, 3]
                        if normalize_coords and coord_mean is not None and coord_std is not None:
                            _cm = coord_mean.to(device)
                            _cs = coord_std.to(device)
                            obs_coords_rel = (obs_coords_rel - _cm) / (_cs + 1e-8)
                        obs_coords_rel = obs_coords_rel.unsqueeze(0)  # [1, M, 3]

                        # Geo conditioning: append 6 wall distances if model was trained with --geo_conditioning
                        if _geo and _room_dims is not None:
                            _Lx, _Ly, _Lz = _room_dims
                            _half_min = min(_Lx, _Ly, _Lz) / 2.0
                            _d_walls = torch.stack([
                                src_xyz[:, 0],      _Lx - src_xyz[:, 0],
                                src_xyz[:, 1],      _Ly - src_xyz[:, 1],
                                src_xyz[:, 2],      _Lz - src_xyz[:, 2],
                            ], dim=1) / _half_min  # [1, 6]
                            obs_coords_rel = torch.cat([
                                obs_coords_rel,
                                _d_walls.unsqueeze(1).expand(-1, M, -1)  # [1, M, 6]
                            ], dim=-1)  # [1, M, 9]

                        z_flat = z_true.view(z_true.shape[1], -1)
                        obs_values = z_flat[:, obs_indices].transpose(0, 1).unsqueeze(0)
                        obs_mask = torch.ones(1, M, dtype=torch.bool, device=device)

                        # Inference
                        x0 = torch.randn_like(z_true)
                        y_tokens, pooled_context, freq_contexts = set_encoder(obs_coords_rel, obs_values, obs_mask)

                        ts = torch.linspace(0, 1, 11, device=device)
                        ts = ts.view(1, -1, 1, 1, 1, 1).expand(x0.shape[0], -1, -1, -1, -1, -1)

                        simulator.ode.guidance_scale = w

                        z_est = simulator.simulate(x0, ts, x0=x0, z_true=z_true, y_tokens=y_tokens,
                                                 obs_mask=obs_mask, pooled_context=pooled_context,
                                                 freq_contexts=freq_contexts,
                                                 paste_observations=True, obs_indices=obs_indices)

                        # Calculate MSE and LSD in denormalized (dB) domain
                        z_est_denorm = z_est * spec_std + train_sampler.mean.item()
                        z_true_denorm = z_true * spec_std + train_sampler.mean.item()
                        z_est_denorm_eval = z_est_denorm[:, :eval_freq_up_to]
                        z_true_denorm_eval = z_true_denorm[:, :eval_freq_up_to]
                        mse = torch.mean((z_est_denorm_eval - z_true_denorm_eval) ** 2).item()

                        # Calculate NMSE (Normalized MSE) in dB
                        z_true_var = torch.var(z_true_denorm_eval).item()
                        nmse_linear = mse / z_true_var if z_true_var > 0 else float('inf')
                        nmse = 10 * np.log10(nmse_linear) if nmse_linear > 0 and nmse_linear != float('inf') else float('inf')

                        # LSD directly on denormalized (dB domain) data
                        lsd_db = calculate_lsd_unified(
                            z_est_denorm_eval.squeeze(0), z_true_denorm_eval.squeeze(0), freq_dim=0
                        ).item()

                        # M_fundamental evaluation (5 specific positions: [0, 272, 665, 937, 1330])
                        m_fundamental_indices = [0, 272, 665, 937, 1330]

                        # Convert 3D cube to flat format [freq, 1331] for indexing
                        z_est_flat = z_est_denorm_eval.view(z_est_denorm_eval.shape[1], -1)  # [eval_freq, 1331]
                        z_true_flat = z_true_denorm_eval.view(z_true_denorm_eval.shape[1], -1)  # [eval_freq, 1331]

                        lsd_m_fund_db = calculate_lsd_unified(
                            z_est_flat[:, m_fundamental_indices].T,  # [5, freq] denormalized
                            z_true_flat[:, m_fundamental_indices].T,  # [5, freq] denormalized
                            freq_dim=1  # frequency is now dim=1
                        ).item()

                        mse_m_fund = torch.mean((z_est_flat[:, m_fundamental_indices] - z_true_flat[:, m_fundamental_indices]) ** 2).item()

                        # Store per-source errors
                        source_errors = {
                            'lsd': lsd_db, 'mse': mse, 'nmse': nmse,
                            'lsd_m_fund': lsd_m_fund_db, 'mse_m_fund': mse_m_fund
                        }
                        lsd_scores.append(source_errors)

                        # Store in per-source dictionary
                        if 'per_source_errors' not in results[M][w]:
                            results[M][w]['per_source_errors'] = {}
                        results[M][w]['per_source_errors'][i] = source_errors

                        # Save denormalized prediction in [1331, eval_freq, src]
                        pred_cache_tensor[:, :, i] = z_est_denorm_eval.squeeze(0).view(eval_freq_up_to, -1).T.cpu()

                if cache_path is not None and cache_meta is not None:
                    _save_atf_cache(cache_path, cache_meta, {w: pred_cache_tensor})
                    print(f"  ✅ Saved cache for M={M}, w={w} to {cache_path}")
        
            # Extract LSD, MSE, and NMSE values for this guidance scale
            if not lsd_scores and 'per_source_errors' in results[M][w]:
                # Cache-hit path populated only per_source_errors; materialize list view.
                lsd_scores = [results[M][w]['per_source_errors'][i] for i in range(eval_sources)]

            lsd_values = [score['lsd'] for score in lsd_scores]
            mse_values = [score['mse'] for score in lsd_scores]
            nmse_values = [score['nmse'] for score in lsd_scores]
            lsd_m_fund_values = [score['lsd_m_fund'] for score in lsd_scores]
            mse_m_fund_values = [score['mse_m_fund'] for score in lsd_scores]
            
            results[M][w].update({
                'lsd_mean': np.mean(lsd_values),
                'lsd_std': np.std(lsd_values), 
                'mse_mean': np.mean(mse_values),
                'mse_std': np.std(mse_values),
                'nmse_mean': np.mean(nmse_values),
                'nmse_std': np.std(nmse_values),
                'lsd_mean_m_fund': np.mean(lsd_m_fund_values),
                'lsd_std_m_fund': np.std(lsd_m_fund_values),
                'mse_mean_m_fund': np.mean(mse_m_fund_values),
                'mse_std_m_fund': np.std(mse_m_fund_values),
                'num_sources_eval': eval_sources,
                'model_freq_up_to': model_freq_up_to,
                'eval_freq_up_to': eval_freq_up_to
            })
    
    return results, idx_mes_pos_mat, grid_xyz


def evaluate_reference_model(atf_mag_est, atf_mag_gt, ref_config, num_sources_eval=None,
                             eval_freq_up_to=None, reference_name="Reference"):
    """
    Evaluate the reference AUTOENCODER model using the loaded data.
    Returns both aggregate metrics and per-source errors.
    """
    print(f"Evaluating {reference_name} model...")
    
    # Get the M value used by reference model
    dataset_name = ref_config['dataset'][0]
    ref_M = ref_config['num_mes_test']  # This is 5 for FSMPAE_10026
    print(f"Reference model uses M={ref_M} microphones")
    
    # Limit evaluation to first N sources if specified
    total_sources = atf_mag_gt.shape[2]
    if num_sources_eval is not None:
        eval_sources = min(num_sources_eval, total_sources)
        print(f"Evaluating on first {eval_sources} sources (out of {total_sources})")
    else:
        eval_sources = total_sources
        print(f"Evaluating on all {eval_sources} sources")
    
    # Calculate BOTH LSD and MSE for comprehensive comparison
    lsd_per_sample_full = []
    lsd_per_sample_matched = []
    mse_per_sample_full = []
    mse_per_sample_matched = []
    nmse_per_sample_full = []
    nmse_per_sample_matched = []
    # Add M_fundamental evaluation (5 specific positions)
    lsd_per_sample_m_fund = []
    mse_per_sample_m_fund = []
    
    # Store per-source errors in a dictionary
    per_source_errors = {}
    
    # 5 fundamental positions for PDF evaluation
    m_fundamental_indices = [0, 272, 665, 937, 1330]

    # If predictions have fewer freq bins than GT (e.g. EEAE with 20 bins), truncate GT to match
    pred_freq_bins = atf_mag_est.shape[1]
    eval_bins = min(eval_freq_up_to, pred_freq_bins) if eval_freq_up_to is not None else pred_freq_bins
    print(f"{reference_name} model_freq_up_to: {pred_freq_bins}, eval_freq_up_to: {eval_bins}")

    for src_idx in tqdm(range(eval_sources), desc="Reference Model"):
        # Full frequency range (truncated to match prediction if necessary)
        lsd_val_full = calculate_lsd_unified(
            atf_mag_est[:, :, src_idx],
            atf_mag_gt[:, :pred_freq_bins, src_idx],
            freq_dim=1
        )
        lsd_per_sample_full.append(lsd_val_full.item())

        # MSE for full frequency range
        mse_val_full = torch.mean((atf_mag_est[:, :, src_idx] - atf_mag_gt[:, :pred_freq_bins, src_idx]) ** 2).item()
        mse_per_sample_full.append(mse_val_full)
        
        # NMSE for full frequency range (in dB)
        gt_var_full = torch.var(atf_mag_gt[:, :pred_freq_bins, src_idx]).item()
        nmse_linear_full = mse_val_full / gt_var_full if gt_var_full > 0 else float('inf')
        nmse_val_full = 10 * np.log10(nmse_linear_full) if nmse_linear_full > 0 and nmse_linear_full != float('inf') else float('inf')
        nmse_per_sample_full.append(nmse_val_full)

        # M_fundamental evaluation on the shared eval range
        lsd_val_m_fund = calculate_lsd_unified(
            atf_mag_est[m_fundamental_indices, :eval_bins, src_idx],
            atf_mag_gt[m_fundamental_indices, :eval_bins, src_idx],
            freq_dim=1
        )
        mse_val_m_fund = torch.mean(
            (atf_mag_est[m_fundamental_indices, :eval_bins, src_idx] -
             atf_mag_gt[m_fundamental_indices, :eval_bins, src_idx]) ** 2
        ).item()
        
        lsd_per_sample_m_fund.append(lsd_val_m_fund.item())
        mse_per_sample_m_fund.append(mse_val_m_fund)
        
        # Store per-source errors
        per_source_errors[src_idx] = {
            'lsd_full': lsd_val_full.item(),
            'mse_full': mse_val_full,
            'nmse_full': nmse_val_full,
            'lsd_m_fund': lsd_val_m_fund.item(),
            'mse_m_fund': mse_val_m_fund
        }
        
        # Matched frequency range: cap at pred_freq_bins in case this reference model
        # has fewer bins than eval_freq_up_to (e.g. EEAE with 20 bins when eval is 30).
        if eval_freq_up_to is not None:
            matched_bins = eval_bins
            lsd_val_matched = calculate_lsd_unified(
                atf_mag_est[:, :matched_bins, src_idx],
                atf_mag_gt[:, :matched_bins, src_idx],
                freq_dim=1
            )
            lsd_per_sample_matched.append(lsd_val_matched.item())

            # MSE for matched frequency range
            mse_val_matched = torch.mean((atf_mag_est[:, :matched_bins, src_idx] - atf_mag_gt[:, :matched_bins, src_idx]) ** 2).item()
            mse_per_sample_matched.append(mse_val_matched)

            # NMSE for matched frequency range (in dB)
            gt_var_matched = torch.var(atf_mag_gt[:, :matched_bins, src_idx]).item()
            nmse_linear_matched = mse_val_matched / gt_var_matched if gt_var_matched > 0 else float('inf')
            nmse_val_matched = 10 * np.log10(nmse_linear_matched) if nmse_linear_matched > 0 and nmse_linear_matched != float('inf') else float('inf')
            nmse_per_sample_matched.append(nmse_val_matched)

            # Add matched frequency metrics to per-source errors
            per_source_errors[src_idx].update({
                'lsd_matched': lsd_val_matched.item(),
                'mse_matched': mse_val_matched,
                'nmse_matched': nmse_val_matched
            })
    
    result = {
        'lsd_mean': np.mean(lsd_per_sample_full), 
        'lsd_std': np.std(lsd_per_sample_full),
        'mse_mean': np.mean(mse_per_sample_full),
        'mse_std': np.std(mse_per_sample_full),
        'nmse_mean': np.mean(nmse_per_sample_full),
        'nmse_std': np.std(nmse_per_sample_full),
        'lsd_mean_m_fund': np.mean(lsd_per_sample_m_fund),
        'lsd_std_m_fund': np.std(lsd_per_sample_m_fund),
        'mse_mean_m_fund': np.mean(mse_per_sample_m_fund),
        'mse_std_m_fund': np.std(mse_per_sample_m_fund),
        'num_mics': ref_M,
        'num_sources_eval': eval_sources,
        # For backward compatibility
        'mean': np.mean(lsd_per_sample_full),
        'std': np.std(lsd_per_sample_full),
        # Add per-source errors
        'per_source_errors': per_source_errors
    }
    
    # Add matched frequency range results if available
    if eval_freq_up_to is not None and lsd_per_sample_matched:
        result['lsd_mean_matched_freq'] = np.mean(lsd_per_sample_matched)
        result['lsd_std_matched_freq'] = np.std(lsd_per_sample_matched)
        result['mse_mean_matched_freq'] = np.mean(mse_per_sample_matched)
        result['mse_std_matched_freq'] = np.std(mse_per_sample_matched)
        result['nmse_mean_matched_freq'] = np.mean(nmse_per_sample_matched)
        result['nmse_std_matched_freq'] = np.std(nmse_per_sample_matched)
        result['mean_matched_freq'] = np.mean(lsd_per_sample_matched)  # For backward compatibility
        
        matched_bins_actual = eval_bins
        print(f"{reference_name} LSD (full {pred_freq_bins} bins): {result['lsd_mean']:.4f} ± {result['lsd_std']:.4f} dB")
        print(f"{reference_name} MSE (full {pred_freq_bins} bins): {result['mse_mean']:.4f} ± {result['mse_std']:.4f}")
        print(f"{reference_name} NMSE (full {pred_freq_bins} bins): {result['nmse_mean']:.4f} ± {result['nmse_std']:.4f} dB")
        print(f"{reference_name} LSD (first {matched_bins_actual} bins): {result['lsd_mean_matched_freq']:.4f} ± {result['lsd_std_matched_freq']:.4f} dB")
        print(f"{reference_name} MSE (first {matched_bins_actual} bins): {result['mse_mean_matched_freq']:.4f} ± {result['mse_std_matched_freq']:.4f}")
        print(f"{reference_name} NMSE (first {matched_bins_actual} bins): {result['nmse_mean_matched_freq']:.4f} ± {result['nmse_std_matched_freq']:.4f} dB")
        print(f"{reference_name} LSD M_fund (first {matched_bins_actual} bins): {result['lsd_mean_m_fund']:.4f} ± {result['lsd_std_m_fund']:.4f} dB")
        print(f"{reference_name} MSE M_fund (first {matched_bins_actual} bins): {result['mse_mean_m_fund']:.4f} ± {result['mse_std_m_fund']:.4f}")
        print(
            f"✅ {reference_name} metrics evaluated on first {matched_bins_actual} bins "
            f"(requested eval_freq_up_to={eval_freq_up_to}, model_freq_up_to={pred_freq_bins})"
        )
    
    return result


def plot_atf_comparisons(atf_mag_est_ref, atf_mag_est_yours, atf_mag_gt, ref_config, freq_up_to, num_sources_eval, best_guidance=None, output_dir=None, atf_mag_est_eeae=None, grid_xyz=None, per_source_lsd=None):
    """
    Plot ATF comparisons with 3 methods: True, Reference, Your Model for multiple combinations
    
    Args:
        atf_mag_est_ref: Reference model predictions
        atf_mag_est_yours: Dictionary of your model predictions for each guidance scale
        atf_mag_gt: Ground truth ATF values
        ref_config: Reference model config
        freq_up_to: Number of frequency bins to use
        num_sources_eval: Number of sources to evaluate
        best_guidance: Optional, pre-computed best guidance scale. If None, will compute it.
    """
    # Get the correct number of sources to evaluate
    total_sources = atf_mag_gt.shape[2]  # Total available sources
    eval_sources = min(num_sources_eval, total_sources) if num_sources_eval is not None else total_sources
    print(f"Evaluating ATF plots for first {eval_sources} sources (out of {total_sources})")

    # Use provided best_guidance or compute it
    if best_guidance is None:
        # Select the best guidance scale for visualization (lowest average LSD)
        guidance_scales = list(atf_mag_est_yours.keys())
        best_guidance = guidance_scales[0]  # Default to first scale
        best_lsd = float('inf')
        
        for w in guidance_scales:
            # Make sure to use the same number of sources for comparison
            current_lsd = torch.mean((atf_mag_est_yours[w][:, :, :eval_sources] - 
                                    atf_mag_gt[:, :freq_up_to, :eval_sources]) ** 2).item()
            if current_lsd < best_lsd:
                best_lsd = current_lsd
                best_guidance = w
        print(f"Computed best guidance scale w={best_guidance} (LSD={best_lsd:.4f})")
    else:
        print(f"Using provided best guidance scale w={best_guidance}")
    
    atf_mag_est_yours_best = atf_mag_est_yours[best_guidance]

    # Create frequency axes for both models
    ref_freq_bins = ref_config['num_freq']  # 64 bins

    fs = ref_config['fs']  # 2000 Hz
    
    # Reference frequency axis (0 to 1000 Hz, 64 bins)
    freq_ref = np.arange(1, ref_freq_bins + 1) / ref_freq_bins * fs / 2
    
    # Your model frequency axis (0 to ~312 Hz, 20 bins)  
    freq_yours = np.arange(1, freq_up_to + 1) / freq_up_to * fs / 2
    
    print(f"Reference freq range: 0-{freq_ref[-1]:.0f} Hz ({ref_freq_bins} bins)")
    print(f"Your model freq range: 0-{freq_yours[-1]:.0f} Hz ({freq_up_to} bins)")
    fftlen_algn = 128
    freq_axis = np.arange(1, fftlen_algn // 2 + 1) / fftlen_algn * fs
    freq_axis = freq_axis[:freq_up_to]  # Ensure it matches model's frequency count

    print("be careful about frequency axis, there is redundacny")

    plt.rcParams["font.size"] = 18  # Same as eval_AUTOENCODER.py
    
    # Create output directory (same structure as inference_1d_atf.py)
    # output_dir = "artifacts/eval/atf_comparisons"
    # os.makedirs(output_dir, exist_ok=True)

    if atf_mag_est_yours is not None:
        # Multiple source and microphone combinations (similar to inference_1d_atf.py)
        total_sources_for_plots = min(num_sources_eval, atf_mag_gt.shape[2]) if num_sources_eval is not None else atf_mag_gt.shape[2]
        # --- Source selection: representative spread by LSD percentile ---
        # If per_source_lsd is provided, pick 10 sources spanning best/median/worst
        # thirds of the LSD distribution, so PDFs show a fair cross-section.
        # Fall back to first-10 if not available.
        if per_source_lsd is not None and len(per_source_lsd) >= 10:
            sorted_indices = sorted(range(len(per_source_lsd)), key=lambda i: per_source_lsd[i])
            n = len(sorted_indices)
            # 3 best, 4 median, 3 worst
            picks = (
                sorted_indices[:3] +
                sorted_indices[n//2 - 2 : n//2 + 2] +
                sorted_indices[-3:]
            )
            source_indices = sorted(picks)  # keep sorted for consistent filenames
            print(f"Source selection: 3 best / 4 median / 3 worst by LSD "
                  f"(LSD range {per_source_lsd[sorted_indices[0]]:.2f}–"
                  f"{per_source_lsd[sorted_indices[-1]]:.2f} dB)")
        else:
            source_indices = list(range(min(10, total_sources_for_plots)))
            print("Source selection: first 10 sources (no per_source_lsd provided)")

        # --- Mic selection: 5 spatially spread positions across the 11^3 cube ---
        # Indices [0, 272, 665, 937, 1330] are extreme corners + center.
        # Instead use positions covering different depths and lateral offsets.
        # These map to roughly: front-bottom-left, back-bottom-right,
        # center, front-top-right, back-top-left — evenly spread.
        mic_indices = [0, 272, 665, 937, 1330]  # keep original for PDF reproducibility
        
        plot_count = 0
        total_plots = len(source_indices)  # One PDF per source (each with 5 subplots)
        
        print(f"Generating {total_plots} ATF comparison PDFs (5 microphones per PDF)...")
        
        # Get microphone coordinates for titles
        # grid_xyz is passed in — it's a fixed 11x11x11 meshgrid, identical regardless of
        # source count, so we never need to reconstruct an ATF3DSampler just for this.
        if grid_xyz is None:
            raise ValueError("grid_xyz must be passed to plot_atf_comparisons to avoid cache invalidation")
        
        for src_idx in source_indices:
            fig, axes = plt.subplots(5, 1, figsize=(10, 4*5))
            plt.subplots_adjust(hspace=0.5)

            for i, mic_idx in enumerate(mic_indices):
                ax = axes[i]

                ax.plot(freq_axis, atf_mag_gt[mic_idx, :freq_up_to, src_idx], 'k--', label="True", linewidth=1.0)
                ax.plot(freq_axis, atf_mag_est_ref[mic_idx, :freq_up_to, src_idx], 'r-', label="FSMPAE", linewidth=0.9)
                if atf_mag_est_eeae is not None:
                    eeae_bins = atf_mag_est_eeae.shape[1]
                    ax.plot(freq_axis[:eeae_bins], atf_mag_est_eeae[mic_idx, :, src_idx], 'g-', label="EEAE", linewidth=0.9)
                ax.plot(freq_axis, atf_mag_est_yours_best[mic_idx, :, src_idx], 'b-', label="SF-Flow", linewidth=0.9)

                ax.set_xscale('log')
                ax.grid(True)
                ax.tick_params(labelsize=8)
                ax.set_xlabel("Frequency (Hz)", fontsize=8)
                ax.set_ylabel("Magnitude (dB)", fontsize=8)

                mic_coord = grid_xyz[mic_idx].numpy()
                ax.set_title(f"ATF ({mic_coord[0]:.2f} m, {mic_coord[1]:.2f} m, {mic_coord[2]:.2f} m)", fontsize=9)
                ax.legend(fontsize=7, loc='upper right', ncol=2)
                print(f"Plotting Source {src_idx+922}, Mic {mic_idx} (index {i+1}/5)")

            plt.tight_layout()

            filename = f"ATF_Comparison_src{src_idx+922:04d}_test.pdf"
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            plot_count += 1
            print(f"Saved {plot_count}/{len(source_indices)} plots: {filename}")
        
        print(f"All {len(source_indices)} ATF comparison PDFs saved to {output_dir}/")
    else:
        print("Your model predictions not available - skipping ATF plots")


def get_your_model_atf_predictions(set_encoder, ode_3d, config, device, atf_mag_gt, ref_config,
                                   model_freq_up_to, eval_freq_up_to, num_sources_eval,
                                   guidance_scales=None, single_guidance=None,
                                   random_M_sampling=False, model_name=None):
    """
    Extract ATF predictions from your model in the same format as reference model.
    Based on inference_1d_atf.py approach.
    
    Args:
        guidance_scales: List of guidance scales to evaluate. If single_guidance is provided, this is ignored.
        single_guidance: Optional, single guidance scale to evaluate. If provided, only this scale is used.
    """
    if single_guidance is not None:
        guidance_scales = [single_guidance]
    print("Generating ATF predictions from your 3D model...")

    # Load your data (same as in inference_1d_atf.py)
    data_path = "ir_fs2000_s8192_m1331_room4.0x6.0x3.0_rt200/"

    # Detect geo_conditioning from checkpoint config
    _geo_atf = config.get('training', {}).get('geo_conditioning', False)
    _room_dims_atf = None
    if _geo_atf:
        import re as _re_atf
        _cfg_dir_atf = config.get('data', {}).get('data_dir', data_path)
        _rm_atf = _re_atf.search(r'room(\d+\.?\d*)x(\d+\.?\d*)x(\d+\.?\d*)', _cfg_dir_atf)
        if _rm_atf:
            _room_dims_atf = (float(_rm_atf.group(1)), float(_rm_atf.group(2)), float(_rm_atf.group(3)))
        else:
            _geo_atf = False
    src_split = config['data']['src_splits']
    freq_from  = config['model'].get('freq_from', 0)

    # Load normalized data
    train_sampler = ATF3DSampler(
        data_path=data_path, mode='train', src_splits=src_split,
        normalize=True, freq_up_to=model_freq_up_to, freq_from=freq_from, model_name=model_name
    )
    test_sampler = ATF3DSampler(
        data_path=data_path, mode='test', src_splits=src_split,
        normalize=False, freq_up_to=model_freq_up_to, freq_from=freq_from, model_name=model_name
    )
    test_sampler.cubes = (test_sampler.cubes - train_sampler.mean) / (train_sampler.std + 1e-8)
    
    grid_xyz = train_sampler.grid_xyz.to(device)
    mean = train_sampler.mean.item()
    std = train_sampler.std.item()
    
    # Create simulator
    simulator = EulerSimulator(ode=ode_3d)
    
    # Initialize output array matching reference format [Guidance, Mic, Freq, Source]
    total_mics = atf_mag_gt.shape[0]
    total_sources = min(num_sources_eval or atf_mag_gt.shape[2], len(test_sampler))
    your_atf_predictions = {w: torch.zeros(total_mics, eval_freq_up_to, total_sources) for w in guidance_scales}
    
    # Fixed M and parameters (from inference_1d_atf.py)
    M = ref_config['num_mes_test']  # Use same M as reference (5)
    num_timesteps = 10
    
    # Load the SAME microphone selection strategy as reference model
    idx_mes_pos_path = "AUTOENCODER/ATF_interp/idx_mes_pos_s1024_m1331.npy"
    idx_mes_pos_mat = np.load(idx_mes_pos_path)
    print(f"Loaded reference microphone selection matrix: {idx_mes_pos_mat.shape}")
    print("Using source-specific microphone selection for ATF generation")
    print(f"Generating predictions for {total_sources} sources with M={M} microphones...")
    
    # Generate predictions for each source
    for src_idx in tqdm(range(total_sources), desc="Your Model ATF"):
        with torch.no_grad():
            # Get source data (same as inference_1d_atf.py)
            z_true = test_sampler.cubes[src_idx].unsqueeze(0).to(device)
            src_xyz = test_sampler.source_coords[src_idx].unsqueeze(0).to(device)
            
            # Create sparse observations - use SAME strategy as reference for fair comparison
            if random_M_sampling:
                print("Using random microphone selection for ATF generation")
                obs_indices = torch.randperm(grid_xyz.shape[0])[:M]  # Fallback to random
                print("odtü", obs_indices)

            else:
                # Use source-specific microphones (different M=5 for each source)
                source_specific_indices = idx_mes_pos_mat[:M, src_idx]  # First M mics for this source
                obs_indices = torch.tensor(source_specific_indices, dtype=torch.long, device=device)
                print("odtü", obs_indices)
            
            obs_xyz_abs = grid_xyz[obs_indices]
            obs_coords_rel = (obs_xyz_abs - src_xyz).unsqueeze(0)  # [1, M, 3]

            # Geo conditioning
            if _geo_atf and _room_dims_atf is not None:
                _Lx, _Ly, _Lz = _room_dims_atf
                _half_min = min(_Lx, _Ly, _Lz) / 2.0
                _d_walls = torch.stack([
                    src_xyz[:, 0],      _Lx - src_xyz[:, 0],
                    src_xyz[:, 1],      _Ly - src_xyz[:, 1],
                    src_xyz[:, 2],      _Lz - src_xyz[:, 2],
                ], dim=1) / _half_min  # [1, 6]
                obs_coords_rel = torch.cat([
                    obs_coords_rel,
                    _d_walls.unsqueeze(1).expand(-1, M, -1)
                ], dim=-1)  # [1, M, 9]
            
            z_flat = z_true.view(z_true.shape[1], -1)
            obs_values = z_flat[:, obs_indices].transpose(0, 1).unsqueeze(0)
            obs_mask = torch.ones(1, M, dtype=torch.bool, device=device)
            
            # Get conditioning tokens
            y_tokens, pooled_context, freq_contexts = set_encoder(obs_coords_rel, obs_values, obs_mask)

            # Generate prediction (same as inference_1d_atf.py)
            x0 = torch.randn_like(z_true)
            ts = torch.linspace(0, 1, num_timesteps + 1, device=device)
            ts = ts.view(1, -1, 1, 1, 1, 1).expand(x0.shape[0], -1, -1, -1, -1, -1)

            # Run inference for each guidance scale
            for w in guidance_scales:
                simulator.ode.guidance_scale = w
                x1_recon = simulator.simulate(x0, ts, x0=x0, z_true=z_true, y_tokens=y_tokens,
                                           obs_mask=obs_mask, pooled_context=pooled_context,
                                           freq_contexts=freq_contexts,
                                           paste_observations=True, obs_indices=obs_indices)
                
                # De-normalize (same as inference_1d_atf.py)
                gen_cube_denorm = (x1_recon * std + mean)
                
                # Convert 3D grid to microphone format
                # Extract ATF values at all microphone positions
                nx, ny, nz = 11, 11, 11  # Grid dimensions
                for mic_idx in range(total_mics):
                    # Convert flat microphone index to 3D coordinates (same as inference_1d_atf.py)
                    iz, iy, ix = np.unravel_index(mic_idx, (nz, ny, nx))
                    
                    # Extract frequency response at this microphone position
                    if iz < gen_cube_denorm.shape[2] and iy < gen_cube_denorm.shape[3] and ix < gen_cube_denorm.shape[4]:
                        your_atf_predictions[w][mic_idx, :, src_idx] = gen_cube_denorm[0, :eval_freq_up_to, iz, iy, ix].cpu()
    
    # print(f"Generated ATF predictions: {your_atf_predictions.shape} (Mic, Freq, Source)")
    return your_atf_predictions

# def get_fallback_reference_results():
#     """Get pre-computed reference results as fallback."""
#     reference_results = {
#         100: {'mean': 3.7072, 'std': 0.8607},
#         50: {'mean': 3.9413, 'std': 0.8662},
#         20: {'mean': 4.1927, 'std': 0.8633},
#         10: {'mean': 4.3775, 'std': 0.8779},
#         5: {'mean': 4.4037, 'std': 0.8952}
#     }
#     return reference_results


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
guidance_scales = [1]
M_values = [10,20,50]
num_sources_eval = 102  # Set to None to evaluate all 102 sources, or e.g. 30 for faster testing

random_M_sampling = False

# Set to True to generate distribution and ATF comparison PDFs after the summary table.
# Printing the table is always executed regardless of this flag.
GENERATE_PLOTS = False

# Set to True  → coord normalisation applied (correct, matches training pipeline for new runs).
# Set to False → no coord normalisation (legacy behaviour; needed to reproduce old Tokyo best 2.86 dB).
NORMALIZE_COORDS = True

# Fast debug switch: evaluate only reference methods (FSMPAE / optional EEAE), skip all SFlow inference.
REFERENCE_ONLY = False

# If set (e.g. 20), evaluate only first N bins while still running each model at its own trained freq.
# If None, defaults to the smallest model_freq_up_to across loaded SFlow + AE models.
EVAL_FREQ_UP_TO = None

# Add/remove EEAE runs to compare. Empty list disables EEAE evaluation.
# Example: EEAE_COMPARISONS = [10001, 10002, 10003]
EEAE_COMPARISONS = []


def get_dataset_version_from_data_dir(data_dir: str) -> str:
    """Parse dataset version from the data_dir path stored in config.
    Examples:
      'ir_fs2000_s1024_m1331_...' -> 'r1'
      'ir_fs2000_s8192_m1331_...' -> 'r4'
    Falls back to the model-name heuristic if pattern not found.
    """
    import re
    m = re.search(r's(\d+)', data_dir)
    if m:
        num_sources = int(m.group(1))
        mapping = {1024: 'r1', 2048: 'r2', 4096: 'r3', 8192: 'r4'}
        return mapping.get(num_sources, 'r1')
    return 'r1'  # safe fallback


def load_coord_stats(dataset_version='r1'):
    """Load cached coord normalisation statistics written by trainer-atf-3d.py."""
    cache_path = f"coord_stats_{dataset_version}.pt"
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Coord stats cache not found: {cache_path}. "
            "Run trainer-atf-3d.py once to generate it."
        )
    stats = torch.load(cache_path)
    return stats['mean'], stats['std']  # both are [3] tensors

def get_model_name(model_path):
    """Extract model name from path, including filename if multiple models in same directory.
    Works for both local artifacts/ paths and external/SSD paths."""
    # Use the parent directory name of the .pt file — works regardless of root path
    dir_name = os.path.basename(os.path.dirname(model_path))

    # Get filename without extension
    filename = os.path.basename(model_path).replace('.pt', '')

    # If filename is just "model", return directory name only (backward compatibility)
    if filename == "model":
        return dir_name
    else:
        # Include both directory and filename for unique identification
        return f"{dir_name}_{filename}"


def find_eeae_prediction_path(eeae_config_id: int) -> str:
    """Find EEAE prediction .pt path for a given config id (e.g., 10001)."""
    pattern = f"RESULTS/out_*_EEAE_{eeae_config_id}/atf_mag/atf_mag_test_*.pt"
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No EEAE prediction file found for config {eeae_config_id} with pattern: {pattern}"
        )
    if len(matches) > 1:
        print(f"WARNING: Multiple EEAE matches for {eeae_config_id}; using latest: {matches[-1]}")
    return matches[-1]


def _canonicalize_for_cache(obj):
    """Recursively convert values into JSON-stable, hashable forms."""
    if isinstance(obj, dict):
        return {str(k): _canonicalize_for_cache(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize_for_cache(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        if obj.numel() <= 64:
            return obj.detach().cpu().tolist()
        return {"__tensor_shape__": list(obj.shape), "__dtype__": str(obj.dtype)}
    return obj


def _build_atf_cache_meta(model_path, guidance, M, num_sources_eval, model_freq_up_to, eval_freq_up_to,
                          random_M_sampling, config):
    """
    Build a cache metadata dictionary that uniquely identifies an ATF inference run.
    Different M / guidance / eval bins / model checkpoint config -> different cache key.
    """
    model_path_abs = os.path.abspath(model_path)
    model_mtime = os.path.getmtime(model_path_abs) if os.path.exists(model_path_abs) else None
    return {
        "cache_version": 2,
        "model_path": model_path_abs,
        "model_mtime": model_mtime,
        "guidance": float(guidance),
        "M": int(M),
        "num_sources_eval": int(num_sources_eval),
        "model_freq_up_to": int(model_freq_up_to),
        "eval_freq_up_to": int(eval_freq_up_to),
        "random_M_sampling": bool(random_M_sampling),
        "num_timesteps": 10,  # fixed in get_your_model_atf_predictions
        "src_splits": config.get("data", {}).get("src_splits"),
        "freq_from": config.get("model", {}).get("freq_from", 0),
        "geo_conditioning": bool(config.get("training", {}).get("geo_conditioning", False)),
    }


def _atf_cache_filename(cache_meta):
    canonical = _canonicalize_for_cache(cache_meta)
    cache_hash = hashlib.sha1(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return f"atf_pred_cache_{cache_hash}.pt"


def _load_matching_atf_cache(model_dir, cache_meta):
    """Load matching cache if present in model directory."""
    model_dir = os.path.abspath(model_dir)
    exact_path = os.path.join(model_dir, _atf_cache_filename(cache_meta))
    candidates = [exact_path]
    if not os.path.exists(exact_path):
        candidates.extend(sorted(glob.glob(os.path.join(model_dir, "atf_pred_cache*.pt"))))

    for cache_path in candidates:
        if not os.path.exists(cache_path):
            continue
        try:
            payload = torch.load(cache_path, map_location='cpu')
        except Exception as e:
            print(f"  WARNING: Failed reading cache {cache_path}: {e}")
            continue

        # New-format cache with full metadata match
        if isinstance(payload, dict) and payload.get("metadata") == cache_meta and "atf_predictions" in payload:
            return cache_path, payload["atf_predictions"]

        # Backward compatibility: older cache payloads (best-effort match)
        if isinstance(payload, dict) and "atf_predictions" in payload:
            old_ok = (
                payload.get("guidance") == cache_meta["guidance"] and
                payload.get("M") == cache_meta["M"] and
                payload.get("num_sources_eval") == cache_meta["num_sources_eval"] and
                payload.get("model_freq_up_to") == cache_meta["model_freq_up_to"] and
                payload.get("eval_freq_up_to") == cache_meta["eval_freq_up_to"]
            )
            if old_ok:
                return cache_path, payload["atf_predictions"]

    return None, None


def _save_atf_cache(cache_path, cache_meta, atf_predictions):
    payload = {
        "metadata": cache_meta,
        "atf_predictions": {w: t.detach().cpu() for w, t in atf_predictions.items()},
    }
    torch.save(payload, cache_path)

# Get model names
MODEL_NAMES = [get_model_name(path) for path in MULTI_MODEL_PATHS]
MULTI_MODEL_MODE = len(MULTI_MODEL_PATHS) > 1

mode_title = '=== REFERENCE-ONLY EVALUATION ===' if REFERENCE_ONLY else (
    '=== MULTI-MODEL EVALUATION ===' if MULTI_MODEL_MODE else '=== SINGLE MODEL EVALUATION ==='
)
print(mode_title)
print(f"Device: {device}")
if not REFERENCE_ONLY:
    for i, (path, name) in enumerate(zip(MULTI_MODEL_PATHS, MODEL_NAMES)):
        print(f"  Model {i+1}: {name}")
print()

# Load all your models first (we choose eval_freq_up_to after seeing all methods)
if REFERENCE_ONLY:
    print("\n1. Skipping SFlow model loading (REFERENCE_ONLY=True)")
else:
    print("\n1. Loading your 3D Flow Matching models...")
all_your_results = {}
all_your_predictions = {}  # Store predictions to avoid reloading best model
all_model_info = {}  # Store model information
all_model_configs = {}
all_model_coord_stats = {}
sflow_model_freq_up_to = {}
all_model_paths = {}

if not REFERENCE_ONLY:
    for i, (model_path, model_name) in enumerate(zip(MULTI_MODEL_PATHS, MODEL_NAMES)):
        print(f"Loading model {i+1}/{len(MULTI_MODEL_PATHS)}: {model_name}")

        checkpoint, config, model_states_cfg = load_model_and_config(model_path, device)

        # Create and load models
        set_encoder, unet_3d, ode_3d, is_new_model = model_factory(config, model_states_cfg, device)

        model_freq = config['model'].get('freq_up_to')
        sflow_model_freq_up_to[model_name] = model_freq
        print(f"  {model_name} model_freq_up_to: {model_freq}")
        
        # Get model information
        set_encoder_info = get_model_info(set_encoder, "SetEncoder")
        unet_info = get_model_info(unet_3d, "UNet3D")
        
        # Calculate total model size (SetEncoder + UNet)
        total_params = set_encoder_info['total_params'] + unet_info['total_params']
        total_size_mb = set_encoder_info['model_size_mb'] + unet_info['model_size_mb']
        
        model_info = {
            'set_encoder': set_encoder_info,
            'unet': unet_info,
            'total_params': total_params,
            'total_params_str': f"{total_params:,}",
            'total_size_mb': total_size_mb,
            'total_size_str': f"{total_size_mb:.2f} MB"
        }
        all_model_info[model_name] = model_info
        
        # Print model info
        print(f"\n--- {model_name} Architecture ---")
        print_model_info(set_encoder, "SetEncoder")
        print_model_info(unet_3d, "UNet3D")
        print(f"=== Combined Model ===")
        print(f"Total parameters: {model_info['total_params_str']}")
        print(f"Total size: {model_info['total_size_str']}")
        print("=" * 20)
        
        # Load coord stats from the data_dir stored in this checkpoint's config
        _data_dir = config.get('data', {}).get('data_dir', '')
        dataset_version = get_dataset_version_from_data_dir(_data_dir)
        print(f"  Dataset version inferred from data_dir '{_data_dir}': {dataset_version}")
        try:
            _coord_mean, _coord_std = load_coord_stats(dataset_version)
            print(f"  Coord stats loaded: mean={_coord_mean}, std={_coord_std}")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}. Running without coord normalisation.")
            _coord_mean, _coord_std = None, None

        # Store model components for later plotting (avoid reloading best model)
        all_your_predictions[model_name] = (set_encoder, unet_3d, ode_3d, config)
        all_model_configs[model_name] = config
        all_model_coord_stats[model_name] = (_coord_mean, _coord_std)
        all_model_paths[model_name] = model_path

# Load reference methods and decide eval_freq_up_to
print("\n2. Loading reference AUTOENCODER model...")
atf_mag_est, atf_mag_gt, ref_config, ref_data, fsmpae_model_freq_up_to, ref_inv_perm_pt_to_ae = load_reference_model(device)

eeae_atf_est_by_id = {}
eeae_model_freq_up_to_by_id = {}
if EEAE_COMPARISONS:
    print(f"\n2b. Loading EEAE comparisons: {EEAE_COMPARISONS}")
    for eeae_id in EEAE_COMPARISONS:
        eeae_pt_path = find_eeae_prediction_path(eeae_id)
        print(f"  Loading EEAE {eeae_id} from {eeae_pt_path}...")
        eeae_atf_est = torch.load(eeae_pt_path, weights_only=False)
        # Align EEAE predictions to the same PT canonical mic order as reference GT.
        eeae_atf_est = eeae_atf_est[ref_inv_perm_pt_to_ae, :, :]
        eeae_atf_est_by_id[eeae_id] = eeae_atf_est
        eeae_model_freq_up_to_by_id[eeae_id] = eeae_atf_est.shape[1]
        print(f"  EEAE {eeae_id} results loaded: {eeae_atf_est.shape}")
        print(f"  EEAE {eeae_id} model_freq_up_to: {eeae_model_freq_up_to_by_id[eeae_id]}")
else:
    print("\n2b. EEAE comparisons disabled (EEAE_COMPARISONS is empty).")

all_freq_candidates = list(sflow_model_freq_up_to.values()) + [fsmpae_model_freq_up_to]
all_freq_candidates.extend(list(eeae_model_freq_up_to_by_id.values()))
if EVAL_FREQ_UP_TO is None:
    eval_freq_up_to = min(all_freq_candidates)
    print(f"\n[auto eval_freq_up_to] Using smallest available model_freq_up_to: {eval_freq_up_to}")
else:
    eval_freq_up_to = EVAL_FREQ_UP_TO
    min_available = min(all_freq_candidates)
    if eval_freq_up_to > min_available:
        raise ValueError(
            f"EVAL_FREQ_UP_TO={eval_freq_up_to} exceeds smallest available model_freq_up_to={min_available}. "
            "Set EVAL_FREQ_UP_TO <= smallest available value."
        )
    print(f"\n[user eval_freq_up_to] Using EVAL_FREQ_UP_TO={eval_freq_up_to}")

if REFERENCE_ONLY:
    print("\n3. Skipping SFlow evaluation (REFERENCE_ONLY=True)")
else:
    print("\n3. Evaluating SFlow models...")
    for i, (model_path, model_name) in enumerate(zip(MULTI_MODEL_PATHS, MODEL_NAMES)):
        print(f"Evaluating model {i+1}/{len(MULTI_MODEL_PATHS)}: {model_name}")
        set_encoder, unet_3d, ode_3d, config = all_your_predictions[model_name]
        _coord_mean, _coord_std = all_model_coord_stats[model_name]
        model_results, idx_mes_pos_mat, _grid_xyz = evaluate_your_model(
            set_encoder, ode_3d, config, M_values, device,
            num_sources_eval, guidance_scales,
            random_M_sampling=random_M_sampling,
            model_name=model_name,
            normalize_coords=NORMALIZE_COORDS,
            coord_mean=_coord_mean if NORMALIZE_COORDS else None,
            coord_std=_coord_std if NORMALIZE_COORDS else None,
            eval_freq_up_to=eval_freq_up_to,
            model_path=model_path,
        )
        all_your_results[model_name] = model_results
        all_model_grid_xyz = _grid_xyz  # fixed 11^3 room grid

print("\n4. Evaluating reference methods...")
ref_results = evaluate_reference_model(
    atf_mag_est, atf_mag_gt, ref_config, num_sources_eval,
    eval_freq_up_to=eval_freq_up_to, reference_name="FSMPAE 10026"
)
eeae_results_by_id = {}
for eeae_id, eeae_atf_est in eeae_atf_est_by_id.items():
    eeae_results_by_id[eeae_id] = evaluate_reference_model(
        eeae_atf_est, atf_mag_gt, ref_config, num_sources_eval,
        eval_freq_up_to=eval_freq_up_to, reference_name=f"EEAE {eeae_id}"
    )

# Print results
print("\n" + "="*80)
print("=== COMPARISON RESULTS ===")
print("="*80)
print(f"Evaluation freq range: 0-{eval_freq_up_to*ref_config['fs']//2//ref_config['num_freq']:.0f} Hz ({eval_freq_up_to} bins)")
print(f"Reference freq range: 0-{ref_config['fs']//2} Hz ({ref_config['num_freq']} bins)")
print(f"Sources evaluated: {ref_results['num_sources_eval']} (out of 102 total)")
print()
print("M_fundamental = 5 specific evaluation positions [0, 272, 665, 937, 1330] for PDFs")
print("Full cube = All 1331 spatial positions")
print(f"FAIR COMPARISON: All methods evaluated on first {eval_freq_up_to} bins (0-{eval_freq_up_to*ref_config['fs']//2//ref_config['num_freq']:.0f} Hz)")
print("-"*190)
print(f"{'Method':<45} | {'w':<4} | {'LSD M_fund':<12} | {'MSE M_fund':<12} | {'LSD Full':<12} | {'MSE Full':<12} | {'NMSE Full (dB)':<14} | {'Freq Range':<15}")
print("-"*190)

# Reference model - USE MATCHED FREQUENCY RANGE for fair comparison
ref_lsd_m_fund = ref_results['lsd_mean_m_fund']  # Now uses matched freq range
ref_mse_m_fund = ref_results['mse_mean_m_fund']  # Now uses matched freq range
# Use matched frequency range (first 20 bins) for fair comparison
ref_lsd_full_fair = ref_results.get('lsd_mean_matched_freq', ref_results['lsd_mean'])
ref_mse_full_fair = ref_results.get('mse_mean_matched_freq', ref_results['mse_mean'])
ref_nmse_full_fair = ref_results.get('nmse_mean_matched_freq', ref_results['nmse_mean'])

print(f"{'Reference FSMPAE 10026 (M=' + str(ref_results['num_mics']) + ' mics)':<45} | {'N/A':<4} | {ref_lsd_m_fund:.4f}     | {ref_mse_m_fund:.4f}     | {ref_lsd_full_fair:.4f}     | {ref_mse_full_fair:.4f}     | {ref_nmse_full_fair:.4f}       | {f'First {eval_freq_up_to} bins':<15}")

for eeae_id, eeae_results in eeae_results_by_id.items():
    eeae_lsd_m_fund = eeae_results['lsd_mean_m_fund']
    eeae_mse_m_fund = eeae_results['mse_mean_m_fund']
    eeae_lsd_full_fair = eeae_results.get('lsd_mean_matched_freq', eeae_results['lsd_mean'])
    eeae_mse_full_fair = eeae_results.get('mse_mean_matched_freq', eeae_results['mse_mean'])
    eeae_nmse_full_fair = eeae_results.get('nmse_mean_matched_freq', eeae_results['nmse_mean'])
    print(
        f"{'Reference EEAE ' + str(eeae_id) + ' (M=' + str(eeae_results['num_mics']) + ' mics)':<45} | "
        f"{'N/A':<4} | {eeae_lsd_m_fund:.4f}     | {eeae_mse_m_fund:.4f}     | "
        f"{eeae_lsd_full_fair:.4f}     | {eeae_mse_full_fair:.4f}     | {eeae_nmse_full_fair:.4f}       | "
        f"{f'First {eval_freq_up_to} bins':<15}"
    )

# All your models
COL_W = 45  # Method column width

for model_name, model_results in all_your_results.items():
    for M in M_values:
        label = f"{model_name} (M={M})"
        display = label[-COL_W:] if len(label) > COL_W else label

        # Print results for each guidance scale
        for w in guidance_scales:
            your_lsd_m_fund = model_results[M][w]['lsd_mean_m_fund']
            your_mse_m_fund = model_results[M][w]['mse_mean_m_fund']
            your_lsd_full = model_results[M][w]['lsd_mean']
            your_mse_full = model_results[M][w]['mse_mean']
            your_nmse_full = model_results[M][w]['nmse_mean']

            print(f"{display:<{COL_W}} | {w:<4.1f} | {your_lsd_m_fund:.4f}     | {your_mse_m_fund:.4f}     | {your_lsd_full:.4f}     | {your_mse_full:.4f}     | {your_nmse_full:.4f}       | {f'First {eval_freq_up_to} bins':<15}")
            print("-"*190)

# Find best model and guidance scale combination
if all_your_results:
    best_model = None
    best_guidance = None
    best_lsd = float('inf')
    best_results = {}  # Store best results for reuse

    for model_name, model_results in all_your_results.items():
        for M in M_values:
            for w in guidance_scales:
                if model_results[M][w]['lsd_mean'] < best_lsd:
                    best_lsd = model_results[M][w]['lsd_mean']
                    best_model = model_name
                    best_guidance = w
                    best_results = {
                        'model': best_model,
                        'guidance': best_guidance,
                        'lsd': best_lsd
                    }

    print("="*80)
    print(f"🏆 BEST MODEL: {best_model}")
    print(f"   Best Guidance Scale: {best_guidance}")
    print(f"   Best LSD: {best_lsd:.4f} dB")
    if best_model in all_model_info:
        best_model_info = all_model_info[best_model]
        print(f"   Model Parameters: {best_model_info['total_params_str']}")
        print(f"   Model Size: {best_model_info['total_size_str']}")
        print(f"   SetEncoder: {best_model_info['set_encoder']['total_params_str']} params")
        print(f"   UNet3D: {best_model_info['unet']['total_params_str']} params")
    # print(f"   Improvement over Reference: {ref_results['mean'] - best_lsd:+.4f} dB")
    print("="*80)
    print(f"Note: Ref models use M={ref_results['num_mics']} observation microphones")
    print(f"      Reference uses source-specific microphone selection")
    print(f"      Your models use SAME source-specific microphone selection")
    print(f"      (Different M=5 microphones for each source, as per reference)")
    print("="*80)

# Build/load cache for best SFlow model ATF predictions regardless of plotting.
best_model_atf_predictions = None
if all_your_results and best_model and best_model in all_your_predictions:
    print("\n5. Preparing cached ATF predictions for best SFlow model...")
    set_encoder_best, unet_3d_best, ode_3d_best, config_best = all_your_predictions[best_model]
    _best_w    = best_results['guidance']
    _atf_M     = ref_config['num_mes_test']          # M used inside get_your_model_atf_predictions (5)
    _best_model_path = all_model_paths[best_model]
    _best_model_dir = os.path.dirname(_best_model_path)
    _n_src     = num_sources_eval if num_sources_eval is not None else atf_mag_gt.shape[2]
    _best_model_freq_up_to = sflow_model_freq_up_to[best_model]
    _cache_meta = _build_atf_cache_meta(
        model_path=_best_model_path,
        guidance=_best_w,
        M=_atf_M,
        num_sources_eval=_n_src,
        model_freq_up_to=_best_model_freq_up_to,
        eval_freq_up_to=eval_freq_up_to,
        random_M_sampling=random_M_sampling,
        config=config_best,
    )
    _best_cache_path = os.path.join(_best_model_dir, _atf_cache_filename(_cache_meta))
    _loaded_path, _cached_predictions = _load_matching_atf_cache(_best_model_dir, _cache_meta)
    if _cached_predictions is not None:
        print(f"⚡ Loading cached ATF predictions from {_loaded_path}")
        best_model_atf_predictions = _cached_predictions
        if _best_w in best_model_atf_predictions:
            print(f"   Loaded predictions shape: {list(best_model_atf_predictions[_best_w].shape)}")
    else:
        print(f"🔄 No cache found — running ATF inference for {_n_src} sources...")
        best_model_atf_predictions = get_your_model_atf_predictions(
            set_encoder_best, ode_3d_best, config_best, device,
            atf_mag_gt, ref_config, _best_model_freq_up_to, eval_freq_up_to, num_sources_eval,
            single_guidance=_best_w, random_M_sampling=random_M_sampling, model_name=best_model
        )
        _save_atf_cache(_best_cache_path, _cache_meta, best_model_atf_predictions)
        print(f"✅ ATF predictions cached to {_best_cache_path}")

if GENERATE_PLOTS and all_your_results:
    # Plot distributions for each model individually and create combined plot
    ref_per_source = ref_results['per_source_errors']
    source_indices = list(range(len(ref_per_source)))
    ref_lsd = [ref_per_source[i]['lsd_matched'] for i in range(len(ref_per_source))]
    ref_mse = [ref_per_source[i]['mse_matched'] for i in range(len(ref_per_source))]

    # Prepare data for combined plot
    all_model_lsd = {}
    all_model_mse = {}
    colors = plt.cm.tab10(np.linspace(0, 1, len(MODEL_NAMES)))

    # Plot individual model distributions and collect data for combined plot
    for i, model_name in enumerate(MODEL_NAMES):
        if model_name in all_your_results:
            # Get best guidance for this model
            model_best_guidance = None
            model_best_lsd = float('inf')
            for w in guidance_scales:
                if all_your_results[model_name][M_values[0]][w]['lsd_mean'] < model_best_lsd:
                    model_best_lsd = all_your_results[model_name][M_values[0]][w]['lsd_mean']
                    model_best_guidance = w

            model_per_source = all_your_results[model_name][M_values[0]][model_best_guidance]['per_source_errors']
            model_lsd = [model_per_source[j]['lsd'] for j in range(len(model_per_source))]
            model_mse = [model_per_source[j]['mse'] for j in range(len(model_per_source))]

            # Store for combined plot
            all_model_lsd[model_name] = {'values': model_lsd, 'guidance': model_best_guidance, 'color': colors[i]}
            all_model_mse[model_name] = {'values': model_mse, 'guidance': model_best_guidance, 'color': colors[i]}

            # Save individual model plots - create unique subdirectory for each model
            base_model_dir = os.path.dirname(MULTI_MODEL_PATHS[i])
            # Use the filename (without extension) as subdirectory name for uniqueness
            model_filename = os.path.basename(MULTI_MODEL_PATHS[i]).replace('.pt', '')
            if model_filename == 'model':
                model_dir = base_model_dir  # Backward compatibility
            else:
                model_dir = os.path.join(base_model_dir, f"eval_{model_filename}")
            os.makedirs(model_dir, exist_ok=True)

            # Individual LSD plot
            plt.figure(figsize=(12, 6))
            ref_mean = np.mean(ref_lsd)
            model_mean = np.mean(model_lsd)
            plt.plot(source_indices, ref_lsd, 'r-', label=f'Reference (mean: {ref_mean:.4f} dB)', alpha=0.7)
            plt.plot(source_indices, model_lsd, 'b-', label=f'{model_name} w={model_best_guidance} (mean: {model_mean:.4f} dB)', alpha=0.7)
            plt.xlabel('Source Index')
            plt.ylabel('LSD Error (dB)')
            plt.title(f'LSD Distribution - {model_name}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(model_dir, 'lsd_distribution.pdf'), dpi=300, bbox_inches='tight')
            plt.close()

            # Individual MSE plot
            plt.figure(figsize=(12, 6))
            ref_mean_mse = np.mean(ref_mse)
            model_mean_mse = np.mean(model_mse)
            plt.plot(source_indices, ref_mse, 'r-', label=f'Reference (mean: {ref_mean_mse:.4f})', alpha=0.7)
            plt.plot(source_indices, model_mse, 'b-', label=f'{model_name} w={model_best_guidance} (mean: {model_mean_mse:.4f})', alpha=0.7)
            plt.xlabel('Source Index')
            plt.ylabel('MSE Error')
            plt.title(f'MSE Distribution - {model_name}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(model_dir, 'mse_distribution.pdf'), dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Individual distribution plots saved to {model_dir}/")

    # Create combined plots in parent directory
    parent_dir = os.path.dirname(os.path.dirname(MULTI_MODEL_PATHS[0]))  # Go up two levels
    os.makedirs(parent_dir, exist_ok=True)

    # Combined LSD plot
    plt.figure(figsize=(14, 8))
    ref_mean = np.mean(ref_lsd)
    plt.plot(source_indices, ref_lsd, 'r-', label=f'FSMPAE (mean: {ref_mean:.4f} dB)', alpha=0.8, linewidth=2)
    if eeae_results_by_id:
        eeae_colors = plt.cm.Set2(np.linspace(0, 1, len(eeae_results_by_id)))
        for cidx, (eeae_id, eeae_results) in enumerate(eeae_results_by_id.items()):
            eeae_per_source = eeae_results['per_source_errors']
            eeae_lsd = [
                eeae_per_source[i].get('lsd_matched', eeae_per_source[i]['lsd_full'])
                for i in range(len(eeae_per_source))
            ]
            eeae_mean_lsd = np.mean(eeae_lsd)
            plt.plot(
                source_indices, eeae_lsd, '-', color=eeae_colors[cidx],
                label=f'EEAE {eeae_id} (mean: {eeae_mean_lsd:.4f} dB)',
                alpha=0.8, linewidth=2
            )

    for model_name, data in all_model_lsd.items():
        model_mean = np.mean(data['values'])
        # Wrap long names: insert newline before parenthetical info
        wrapped = model_name.replace('_', '\n', 1) if len(model_name) > 30 else model_name
        plt.plot(source_indices, data['values'], '-', color=data['color'],
                label=f'{wrapped}\nw={data["guidance"]} mean={model_mean:.4f} dB', alpha=0.7)

    plt.xlabel('Source Index')
    plt.ylabel('LSD Error (dB)')
    plt.title('LSD Distribution Comparison - All Models')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(parent_dir, 'Zcombined_lsd_distribution.pdf'), dpi=300, bbox_inches='tight')
    plt.show()

    # Combined MSE plot
    plt.figure(figsize=(14, 8))
    ref_mean_mse = np.mean(ref_mse)
    plt.plot(source_indices, ref_mse, 'r-', label=f'Reference (mean: {ref_mean_mse:.4f})', alpha=0.8, linewidth=2)

    for model_name, data in all_model_mse.items():
        model_mean = np.mean(data['values'])
        plt.plot(source_indices, data['values'], '-', color=data['color'],
                label=f'{model_name} w={data["guidance"]} (mean: {model_mean:.4f})', alpha=0.7)

    plt.xlabel('Source Index')
    plt.ylabel('MSE Error')
    plt.title('MSE Distribution Comparison - All Models')
    plt.grid(True, alpha=0.3)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(parent_dir, 'Zcombined_mse_distribution.pdf'), dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nCombined distribution plots saved to {parent_dir}/")

    # Plot ATF comparisons using the best model (no reloading needed!)
    print("\n3. Generating ATF comparison plots...")
    print(f"Using best model for plots: {best_model}")

    if best_model and best_model in all_your_predictions and best_model_atf_predictions is not None:
        your_atf_predictions = best_model_atf_predictions
        _best_w = best_results['guidance']
        _best_model_dir = os.path.dirname(all_model_paths[best_model])

        # Build per_source_lsd list for representative source selection in PDF plots
        # _best_w already set above in cache block
        _best_M = min(all_your_results[best_model].keys())  # use smallest M for consistency
        _per_src_errors = all_your_results[best_model][_best_M][_best_w].get('per_source_errors', {})
        _per_source_lsd = [_per_src_errors[i]['lsd'] for i in sorted(_per_src_errors.keys())] if _per_src_errors else None

        # For ATF PDFs, overlay the first configured EEAE (if any) to avoid clutter.
        atf_mag_est_eeae_for_plot = None
        if eeae_atf_est_by_id:
            _first_eeae_id = next(iter(eeae_atf_est_by_id.keys()))
            atf_mag_est_eeae_for_plot = eeae_atf_est_by_id[_first_eeae_id]

        # Use the already computed best guidance scale
        plot_atf_comparisons(atf_mag_est, your_atf_predictions, atf_mag_gt, ref_config,
                            eval_freq_up_to, num_sources_eval, best_guidance=best_results['guidance'],
                            output_dir=_best_model_dir,
                            atf_mag_est_eeae=atf_mag_est_eeae_for_plot,
                            grid_xyz=all_model_grid_xyz,
                            per_source_lsd=_per_source_lsd)
    else:
        print("Could not find best model for plotting")


