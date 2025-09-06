import torch
import numpy as np
import os
import json
from tqdm import tqdm

# Your model imports
from fm_utils import (
    ATF3DSampler, SetEncoder, 
    CrossAttentionUNet3D, CrossAttentionUNet3D_RED3d, 
    CFGVectorFieldODE_3D, CFGVectorFieldODE_3D_V2, EulerSimulator
)

# Reference model imports
import sys
sys.path.append('AUTOENCODER/src')
import AUTOENCODER.src.dataset as autoencoder_dataset
from AUTOENCODER.src.configs import config_FSMPAE_10026
import AUTOENCODER.src.utils as autoencoder_utils

# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def calculate_lsd_unified(estimation, ground_truth, freq_dim=1):
    """
    Unified LSD calculation that works for both 3D spatial and microphone-based data.
    
    Args:
        estimation: Model prediction
        ground_truth: Ground truth
        freq_dim: Dimension along which frequency is stored
    
    Returns:
        LSD value in dB
    """
    squared_error = (estimation - ground_truth) ** 2
    lsd_per_position = torch.sqrt(torch.mean(squared_error, dim=freq_dim))
    return torch.mean(lsd_per_position)


def load_your_model(model_path, device):
    """Load your 3D Flow Matching model."""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})
    model_states_cfg = checkpoint['model_states']
    
    model_cfg = config['model']
    architecture = model_cfg.get('architecture_version')
    
    # Load models
    set_encoder = SetEncoder(
        num_freqs=model_cfg['freq_up_to'],
        d_model=model_cfg['d_model'],
        nhead=model_cfg['nhead'],
        num_layers=model_cfg['num_encoder_layers']
    ).to(device)
    
    if architecture == "v2_residual_context":
        unet_3d = CrossAttentionUNet3D_RED3d(
            in_channels=model_cfg['freq_up_to'],
            out_channels=model_cfg['freq_up_to'],
            channels=model_cfg['channels'],
            d_model=model_cfg['d_model'],
            nhead=model_cfg['nhead']
        ).to(device)
        ode_3d = CFGVectorFieldODE_3D_V2(unet=unet_3d, set_encoder=set_encoder)
    else:
        unet_3d = CrossAttentionUNet3D(
            in_channels=model_cfg['freq_up_to'],
            out_channels=model_cfg['freq_up_to'],
            channels=model_cfg['channels'],
            d_model=model_cfg['d_model'],
            nhead=model_cfg['nhead']
        ).to(device)
        ode_3d = CFGVectorFieldODE_3D(unet=unet_3d, set_encoder=set_encoder)
    
    # Load weights
    set_encoder.load_state_dict(model_states_cfg['set_encoder'])
    unet_3d.load_state_dict(model_states_cfg['unet'])
    set_encoder.eval()
    unet_3d.eval()
    
    return set_encoder, unet_3d, ode_3d, config


def load_reference_model(device, freq_up_to):
    """Load the reference AUTOENCODER model data and predictions."""
    # Use the exact same config as in eval_AUTOENCODER.py (no modifications!)
    config = config_FSMPAE_10026.copy()
    
    try:
        # Change to AUTOENCODER directory so dataset loading works correctly
        import os
        original_cwd = os.getcwd()
        os.chdir('AUTOENCODER')
        
        # Load dataset (this will use ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200/ in AUTOENCODER dir)
        idataset = autoencoder_dataset.ATFdataset(config=config)
        data = idataset.Data
        
        # Change back to original directory
        os.chdir(original_cwd)
        
        # Load model predictions
        pt_dir = 'AUTOENCODER/outputs/out_20250323_FSMPAE_10026'
        dataset_name = config['dataset'][0]
        pt_path = f'{pt_dir}/atf_mag/atf_mag_test_{dataset_name}.pt'
        
        if not os.path.exists(pt_path):
            print(f"Warning: Reference model predictions not found at {pt_path}")
            return None, None, None, None
        
        atf_mag_est = torch.load(pt_path, weights_only=False)
        atf_mag_gt = data['test']['atf_mag'][dataset_name]
        
        # Truncate to match your model's frequency range for fair comparison
        atf_mag_est_truncated = atf_mag_est[:, :freq_up_to, :]
        atf_mag_gt_truncated = atf_mag_gt[:, :freq_up_to, :]
        
        print(f"Reference data loaded: {atf_mag_gt_truncated.shape} (Mic, Freq, Src)")
        print(f"Using first {freq_up_to} frequency bins for comparison")
        
        return atf_mag_est_truncated, atf_mag_gt_truncated, config, data
        
    except Exception as e:
        print(f"Error loading reference model: {e}")
        print("Using pre-computed reference results instead...")
        return None, None, None, None


def evaluate_your_model(set_encoder, unet_3d, ode_3d, config, M_values, device):
    """Evaluate your 3D model."""
    data_dir = "ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200/"
    src_split = config['data']['src_splits']
    freq_up_to = config['model'].get('freq_up_to')
    
    # Load data
    train_sampler = ATF3DSampler(
        data_path=data_dir, mode='train', src_splits=src_split, 
        normalize=True, freq_up_to=freq_up_to
    )
    test_sampler = ATF3DSampler(
        data_path=data_dir, mode='test', src_splits=src_split, 
        normalize=False, freq_up_to=freq_up_to
    )
    test_sampler.cubes = (test_sampler.cubes - train_sampler.mean) / (train_sampler.std + 1e-8)
    
    grid_xyz = train_sampler.grid_xyz.to(device)
    spec_std = train_sampler.std.item()
    
    simulator = EulerSimulator(ode=ode_3d)
    results = {}
    
    for M in M_values:
        print(f"Evaluating your model with M={M} microphones...")
        lsd_scores = []
        
        for i in tqdm(range(len(test_sampler)), desc=f"Your Model M={M}"):
            with torch.no_grad():
                z_true = test_sampler.cubes[i].unsqueeze(0).to(device)
                src_xyz = test_sampler.source_coords[i].unsqueeze(0).to(device)
                
                # Create sparse observations
                obs_indices = torch.randperm(grid_xyz.shape[0])[:M]
                obs_xyz_abs = grid_xyz[obs_indices]
                obs_coords_rel = (obs_xyz_abs - src_xyz).unsqueeze(0)
                
                z_flat = z_true.view(z_true.shape[1], -1)
                obs_values = z_flat[:, obs_indices].transpose(0, 1).unsqueeze(0)
                obs_mask = torch.ones(1, M, dtype=torch.bool, device=device)
                
                # Inference
                x0 = torch.randn_like(z_true)
                y_tokens, pooled_context = set_encoder(obs_coords_rel, obs_values, obs_mask)
                
                ts = torch.linspace(0, 1, 11, device=device)
                ts = ts.view(1, -1, 1, 1, 1, 1).expand(x0.shape[0], -1, -1, -1, -1, -1)
                
                simulator.ode.guidance_scale = 1.0
                z_est = simulator.simulate(x0, ts, x0=x0, z_true=z_true, y_tokens=y_tokens,
                                         obs_mask=obs_mask, pooled_context=pooled_context,
                                         paste_observations=False, obs_indices=obs_indices)
                
                # Calculate LSD
                lsd_normalized = calculate_lsd_unified(z_est.squeeze(0), z_true.squeeze(0), freq_dim=0)
                lsd_db = lsd_normalized.item() * spec_std
                lsd_scores.append(lsd_db)
        
        results[M] = {'mean': np.mean(lsd_scores), 'std': np.std(lsd_scores)}
    
    return results


def evaluate_reference_model(atf_mag_est, atf_mag_gt, M_values=None):
    """Evaluate the reference AUTOENCODER model using the loaded data."""
    print("Evaluating reference AUTOENCODER model...")
    
    # Calculate LSD using the unified function
    # Data shape: [Microphone, Frequency, Source]
    lsd_per_sample = []
    
    for src_idx in tqdm(range(atf_mag_gt.shape[2]), desc="Reference Model"):
        lsd_val = calculate_lsd_unified(
            atf_mag_est[:, :, src_idx], 
            atf_mag_gt[:, :, src_idx], 
            freq_dim=1  # Frequency is dim=1 for reference data
        )
        lsd_per_sample.append(lsd_val.item())
    
    # Return single result (they use all microphones, not sparse sampling)
    return {'mean': np.mean(lsd_per_sample), 'std': np.std(lsd_per_sample)}


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

# Model paths
YOUR_MODEL_PATH = "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d512_head8_sigma0_lrWARM5k_e4_toe5_unet3_V2_layer_20250906-173025_iter300000/model.pt"

M_values = [5]

print("=== Unified Model Evaluation ===")
print(f"Device: {device}")

# Load and evaluate your model
print("\n1. Loading your 3D Flow Matching model...")
set_encoder, unet_3d, ode_3d, your_config = load_your_model(YOUR_MODEL_PATH, device)
your_freq_up_to = your_config['model']['freq_up_to']
print(f"Your model frequency range: {your_freq_up_to}")

your_results = evaluate_your_model(set_encoder, unet_3d, ode_3d, your_config, M_values, device)

# Load and evaluate reference model
print("\n2. Loading reference AUTOENCODER model...")
atf_mag_est, atf_mag_gt, ref_config, ref_data = load_reference_model(device, your_freq_up_to)

ref_results = evaluate_reference_model(atf_mag_est, atf_mag_gt)

# Print results
print("\n" + "="*60)
print("=== COMPARISON RESULTS ===")
print("="*60)
print(f"Frequency range used: {your_freq_up_to} bins (Reference uses 64 bins)")
print("-"*60)
print(f"{'Method':<30} | {'LSD (dB)':<15} | {'Std Dev':<10}")
print("-"*60)

print(f"{'Reference (All Mics)':<30} | {ref_results['mean']:.4f}        | {ref_results['std']:.4f}")

for M in M_values:
    print(f"{'Your Model (M=' + str(M) + ')':<30} | {your_results[M]['mean']:.4f}        | {your_results[M]['std']:.4f}")
    improvement = ref_results['mean'] - your_results[M]['mean']
    print(f"{'→ vs Reference':<30} | {improvement:+.4f}        | {'N/A':<10}")
    print("-"*60)

print("="*60)


