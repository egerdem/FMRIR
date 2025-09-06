import torch
import numpy as np
import matplotlib
matplotlib.use('Agg', force=True)  # Non-interactive backend for saving plots
from matplotlib import pyplot as plt
import os
import json
import random

# Import your necessary classes
from fm_utils import (
    ATF3DSampler, LSD,
    SetEncoder,
    CrossAttentionUNet3D, CrossAttentionUNet3D_RED3d, CFGVectorFieldODE_3D, CFGVectorFieldODE_3D_V2, EulerSimulator
)

# Set seed for reproducible results
SEED = 42  # You can use any integer you like
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # for GPU
np.random.seed(SEED)
random.seed(SEED)

# Ensure deterministic behavior for CUDA operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def model_factory(config, model_states_cfg, device):
    """
    Reads the config and returns the correctly instantiated and loaded models.
    """
    model_cfg = config['model']
    # Use the presence of the version key to decide which architecture to build
    architecture = model_cfg.get('architecture_version')

    # --- Instantiate models based on version ---
    set_encoder = SetEncoder(
        num_freqs=model_cfg['freq_up_to'],
        d_model=model_cfg['d_model'],
        nhead=model_cfg['nhead'],
        num_layers=model_cfg['num_encoder_layers']
    ).to(device)

    if architecture == "v2_residual_context":
        print("--- Creating (v2) architecture ---")
        unet_3d = CrossAttentionUNet3D_RED3d(
            in_channels=model_cfg['freq_up_to'],
            out_channels=model_cfg['freq_up_to'],
            channels=model_cfg['channels'],
            d_model=model_cfg['d_model'],
            nhead=model_cfg['nhead']
        ).to(device)
        ode_3d = CFGVectorFieldODE_3D_V2(unet=unet_3d, set_encoder=set_encoder)

    else:
        print("--- Creating v1 architecture: standard 3d unet ---")
        # Instantiate the old U-Net and ODE wrapper for old checkpoints
        unet_3d = CrossAttentionUNet3D(
            in_channels=model_cfg['freq_up_to'],
            out_channels=model_cfg['freq_up_to'],
            channels=model_cfg['channels'],
            d_model=model_cfg['d_model'],
            nhead=model_cfg['nhead']
        ).to(device)
        ode_3d = CFGVectorFieldODE_3D(unet=unet_3d, set_encoder=set_encoder)

    # --- Load weights ---
    set_encoder.load_state_dict(model_states_cfg['set_encoder'])
    unet_3d.load_state_dict(model_states_cfg['unet'])
    set_encoder.eval()
    unet_3d.eval()

    return set_encoder, unet_3d, ode_3d, architecture


def plot_1d_atf_comparison_multi_guidance(ax, freqs, gt_atf, gen_atf_dict, title):
    """Helper function to plot Ground Truth vs. Generated 1D ATF for multiple guidance levels."""
    # Plot ground truth
    ax.plot(freqs, gt_atf, label='Ground Truth', linewidth=2, color='black')
    
    # Plot generated ATFs for different guidance levels
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, (guidance, gen_atf) in enumerate(gen_atf_dict.items()):
        color = colors[i % len(colors)]
        ax.plot(freqs, gen_atf, label=f'Generated (w={guidance})', 
                linestyle='--', linewidth=1.5, color=color)
    
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xscale('log')
    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.legend()


def plot_1d_atf_comparison_multi_models(ax, freqs, gt_atf, model_results_dict, guidance_level, title):
    """Helper function to plot Ground Truth vs. Generated 1D ATF for multiple models at a specific guidance level."""
    # Plot ground truth
    ax.plot(freqs, gt_atf, label='Ground Truth', linewidth=2, color='black')
    
    # Plot generated ATFs for different models
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_results_dict)))
    linestyles = ['--', '-.', ':', '-']
    
    for i, (model_name, gen_atf) in enumerate(model_results_dict.items()):
        color = colors[i]
        linestyle = linestyles[i % len(linestyles)]
        ax.plot(freqs, gen_atf, label=f'{model_name}', 
                linestyle=linestyle, linewidth=1.5, color=color)
    
    ax.set_title(f"{title} (w={guidance_level})")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xscale('log')
    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.legend()


def get_model_name(model_path):
    return model_path.split("artifacts/")[1].split("/")[0]


def main():
    # --- Configuration ---
    # MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/ATF3D-CrossAttn-v1-freq20_M5to50_sigmaE5_UNET128_LRmin_e4_7_20250826-212533_iter100000/model.pt"
    # MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/ATF3D-CrossAttn-v1-freq64_M5to50_sigmaE5_UNET128_LRmin_e6dot6e4toe7_d128_20250827-185835_iter400000/model.pt"
    # MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/ATF3D-CrossAttn-v1-freq20_M5to50_20250825-201433_iter200000/modelCONVoldcheckpoint.pt"
    # MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/ATF3D-CrossAttn-v1-freq20_M5to50_sigmaE3_UNET256_20250826-192413_iter200000/model.pt"
    # MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts//ATF3D-CrossAttn-v1-freq20_M40to50_sigmaE5_enclayer3_UNET128_LRmin_e6dot6e4toe7_d256_20250827-213218_iter500000/model.pt"
    # MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/ATF3D-CrossAttn-v1-freq20_M5to50_sigmaE3_20250826-183304_iter200000/model_CONVoldcheckpoint.pt"
    MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/ATF3D-CrossAttn-v1-freq64_M5to50_sigmaE5_UNET128_LRmin_e6dot6e4toe7_d128_20250827-185835_iter400000/model.pt"
    MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/ATF3D-CrossAttn-v1-freq20_M5to100_sigmaE3_lr1e3to_e7_unet3_layer3_head3_d256_20250828-190043_iter300000/model.pt"
    MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/ATF3D-CrossAttn-v1-freq20_M5to50_sigmaE3_lr1e3to_e7_unet3_layer3_head8_d256_20250828-233343_iter50000/model.pt"
    MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/M5to6_freq20_layer3_d256_head8_sigmaE3_lr1e3to_e7_unet3_20250903-144259_iter300000/model.pt"
    MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/Loss169e3_M5to50_freq20_layer3_d256_head8_sigmaE3_lr1e3to_e7_unet3_20250904-191320_iter300000/model.pt"
    MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d256_head8_sigma0ZERO_lr1e3to_e7_unet3_20250904-203305_iter300000/model.pt"
    MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d256_head8_sigma0ZERO_lr1e3to_e7_unet3_20250904-203305_iter300000/model.pt"
    MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d256_head4_sigma0ZERO_lr1e4to_e7_unet3_20250904-214817_iter300000/model.pt"
    MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d256_head8_sigma0ZERO_lr1e4to_e7_unet3_20250904-222356_iter300000/model.pt"
    MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d512_head8_sigma0ZERO_lr1e4to_e7_unet3_20250904-225845_iter300000/model.pt"
    MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/M5to10_freq20_layer3_d512_head8_sigma0ZERO_lr1e4to_e7_unet3_20250905-140802_iter300000/model.pt"
    MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d512_head8_sigma0ZERO_lrWARM5k_e4_toe5_unet3_20250905-182733_iter300000/model.pt"
    # MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d512_head8_sigma0_lrWARM5k_e4_toe5_unet3_20250905-204240_iter300000/model.pt"
    # MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/M5to150_freq20_layer3_d512_head8_sigma0_lrWARM5k_e4_toe5_unet3_20250905-223838_iter300000/model.pt"
    MODEL_LOAD_PATH =  "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d512_head8_sigma0_lrWARM5k_e4_toe5_unet3_V2_layer_20250906-173025_iter300000/model.pt"
    
    # Support for multiple models - can be a single path or a list of paths
    MODEL_LOAD_PATH = [
        # "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d256_head4_sigma0ZERO_lr1e4to_e7_unet3_20250904-214817_iter300000/model.pt",
        # "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d256_head8_sigma0ZERO_lr1e4to_e7_unet3_20250904-222356_iter300000/model.pt",
        # "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d512_head8_sigma0ZERO_lr1e4to_e7_unet3_20250904-225845_iter300000/model.pt",
        # "/Users/ege/Projects/FMRIR/artifacts/M5to10_freq20_layer3_d512_head8_sigma0ZERO_lr1e4to_e7_unet3_20250905-140802_iter300000/model.pt",
        # "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer4_d256_head8_sigma0ZERO_lr1e4to_e7_unet3_20250905-154234_iter300000/model.pt",
        # "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d256_head8_sigma0ZERO_lrWARM5k_e4_toe7_unet3_20250905-165351_iter300000/model.pt",
        # "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d512_head8_sigma0ZERO_lrWARM5k_e4_toe7_unet3_20250905-173800_iter300000/model.pt",
        "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d512_head8_sigma0ZERO_lrWARM5k_e4_toe5_unet3_20250905-182733_iter300000/model.pt",
        # "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d512_head8_sigma1e3_lrWARM5k_e4_toe6_unet3_20250905-193258_iter300000/model.pt",
        "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d512_head8_sigma0_lrWARM5k_e4_toe5_unet3_20250905-204124_iter500000/model.pt",
        "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d512_head8_sigma0_lrWARM5k_e4_toe5_unet3_V2_layer_20250906-173025_iter300000/model.pt",
        "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d512_head8_sigma0_lrWARM5k_e4_toe5_unet4_layer3_20250906-191114_iter300000/model.pt"
    ]
    
    # Ensure MODEL_LOAD_PATH is always a list for uniform processing
    if isinstance(MODEL_LOAD_PATH, str):
        MODEL_LOAD_PATHS = [MODEL_LOAD_PATH]
    else:
        MODEL_LOAD_PATHS = MODEL_LOAD_PATH
        
    MODEL_NAMES = [get_model_name(path) for path in MODEL_LOAD_PATHS]
    MULTI_MODEL_MODE = len(MODEL_LOAD_PATHS) > 1
    
    print(f"{'=== MULTI-MODEL MODE ===' if MULTI_MODEL_MODE else '=== SINGLE MODEL MODE ==='}")
    for i, (path, name) in enumerate(zip(MODEL_LOAD_PATHS, MODEL_NAMES)):
        print(f"  Model {i+1}: {name}")
    print()

    data_path = "ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200/"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load Config from first model ---
    checkpoint = torch.load(MODEL_LOAD_PATHS[0], map_location=device)
    config = checkpoint.get('config', {})  # Use .get for safety
    freq_up_to = config['model'].get('freq_up_to')

    # --- Evaluation Configuration ---
    # 10 different sources from 922 to 1024 (indices 0 to 102 in test set)
    # source_indices = [0, 11, 22, 33, 44, 55, 66, 77, 88, 99]  # 10 sources
    source_indices = [0, 11, 22]  # 10 sources
    # 5 different random microphone positions from 0 to 1330
    mic_indices = [156, 423, 789, 1045, 1287, 665]  # 5 microphones
    
    M = 5  # Number of conditioning mics
    guidance = [1.0, 2.0]
    num_timesteps = 10
    lsd = LSD()

    # Create output directory
    output_dir = "artifacts/eval"
    if MULTI_MODEL_MODE:
        output_dir = os.path.join(output_dir, f"multi_model_comparison_{len(MODEL_LOAD_PATHS)}models")
    else:
        output_dir = os.path.join(output_dir, MODEL_NAMES[0])
    os.makedirs(output_dir, exist_ok=True)

    # --- Load Data Sampler (to get data and metadata) ---

    # --- 1. Data Loading ---
    # Create train sampler to get normalization stats and grid coordinates
    train_sampler = ATF3DSampler(
        data_path=data_path, mode='train', src_splits=config['data']['src_splits'], normalize=True, freq_up_to=freq_up_to
    )

    grid_xyz = train_sampler.grid_xyz.to(device)
    mean = train_sampler.mean.item()
    std = train_sampler.std.item()

    # Load the full test set for evaluation
    test_sampler = ATF3DSampler(
        data_path=data_path, mode='test',
        src_splits=config['data']['src_splits'],
        freq_up_to=config['model']['freq_up_to'],
        normalize=False  # Load raw, un-normalized data for plotting
    )
    # Normalize the test data using the stats from the training set
    test_sampler.cubes = (test_sampler.cubes - train_sampler.mean) / (train_sampler.std + 1e-8)

    print(f"Loaded Stats from 3D Training Set: Mean={mean:.4f}, Std={std:.4f}")
    print(f"Test set contains {len(test_sampler.cubes)} samples")

    # --- Load all models ---
    models_info = []
    for i, model_path in enumerate(MODEL_LOAD_PATHS):
        print(f"Loading model {i+1}/{len(MODEL_LOAD_PATHS)}: {MODEL_NAMES[i]}")
        checkpoint = torch.load(model_path, map_location=device)
        config_model = checkpoint.get('config', {})
        model_states_cfg = checkpoint['model_states']
        
        set_encoder, unet_3d, ode_3d, architecture = model_factory(config_model, model_states_cfg, device)
        simulator = EulerSimulator(ode=ode_3d)
        
        models_info.append({
            'name': MODEL_NAMES[i],
            'set_encoder': set_encoder,
            'unet_3d': unet_3d,
            'ode_3d': ode_3d,
            'simulator': simulator,
            'architecture': architecture
        })
    
    print(f"Loaded {len(models_info)} models successfully")

    # Get the frequency axis values from your data generation config
    data_gen_config_path = os.path.join(data_path, "config.json")
    with open(data_gen_config_path, 'r') as f:
        data_gen_config = json.load(f)
    fftlen_algn = data_gen_config['fftlen_algn']
    fs = data_gen_config['fs']
    freq_axis = np.arange(1, fftlen_algn // 2 + 1) / fftlen_algn * fs
    freq_axis = freq_axis[:freq_up_to]  # Ensure it matches model's frequency count

    # Fixed conditioning microphones for all evaluations
    # obs_indices = torch.tensor([763, 654, 398, 823, 947])  # Use only M=5 microphones
    obs_indices = torch.randperm(grid_xyz.shape[0])[:M]

    plot_count = 0
    if MULTI_MODEL_MODE:
        total_plots = len(source_indices) * len(mic_indices) * len(guidance)  # One plot per guidance level
    else:
        total_plots = len(source_indices) * len(mic_indices)  # One plot per source-mic pair
    
    print(f"Generating {total_plots} plots...")
    
    # Loop through all source and microphone combinations
    for source_idx in source_indices:
        # Get the ground truth cube and source position for this source
        z_true = test_sampler.cubes[source_idx].unsqueeze(0).to(device)
        src_xyz = test_sampler.source_coords[source_idx].unsqueeze(0).to(device)
        
        # z_true is already normalized from test_sampler.cubes
        # For plotting comparison later, we need the denormalized version
        z_true_denorm = (z_true * std + mean)
        
        # Prepare conditioning observations
        obs_xyz_abs = grid_xyz[obs_indices]
        obs_coords_rel = obs_xyz_abs - src_xyz
        z_flat = z_true.view(z_true.shape[1], -1)
        obs_values = z_flat[:, obs_indices].transpose(0, 1)
        
        # Batchify for the set encoder
        obs_coords_rel_batch = obs_coords_rel.unsqueeze(0)
        obs_values_batch = obs_values.unsqueeze(0)
        obs_mask = torch.ones(1, M, dtype=torch.bool, device=device)
        
        ts = torch.linspace(0, 1, num_timesteps + 1, device=device)
        ts = ts.view(1, -1, 1, 1, 1, 1).expand(1, -1, -1, -1, -1, -1)
        
        # Generate ATFs for all models and guidance levels
        # Use the same initial noise for all models and guidance levels for fair comparison
        x0 = torch.randn_like(z_true)
        
        # Store results for all models and guidance levels
        all_model_results = {}  # {model_name: {guidance: generated_cube}}
        
        for model_info in models_info:
            model_name = model_info['name']
            set_encoder = model_info['set_encoder']
            simulator = model_info['simulator']
            
            # Get conditioning tokens for this model
            y_tokens, pooled_context = set_encoder(obs_coords_rel_batch, obs_values_batch, obs_mask)
            
            gen_cubes_denorm = {}
            for guid in guidance:
                xt = x0.clone()  # Start from the same initial noise
                
                # Set the guidance scale on the ODE object
                simulator.ode.guidance_scale = guid
                
                # Simulation loop
                x1_recon = simulator.simulate(xt,
                                              ts,
                                              x0=x0,
                                              z_true=z_true,
                                              y_tokens=y_tokens,
                                              obs_mask=obs_mask,
                                              pooled_context=pooled_context,
                                              paste_observations=False,
                                              obs_indices=obs_indices
                                              )
                
                # De-normalize generated result
                gen_cubes_denorm[guid] = (x1_recon * std + mean)
            
            all_model_results[model_name] = gen_cubes_denorm
        
        # Loop through all microphones for this source
        for mic_idx in mic_indices:
            # Convert flat microphone index to 3D coordinates
            nx, ny, nz = 11, 11, 11
            iz, iy, ix = np.unravel_index(mic_idx, (nz, ny, nx))
            
            # Extract the 1D vector of frequencies for the chosen mic
            # Ground truth (same for all models and guidance levels)
            gt_atf_1d = z_true_denorm[0, :, iz, iy, ix].cpu().numpy()
            
            if MULTI_MODEL_MODE:
                # Multi-model comparison: create plots for each guidance level
                for guid in guidance:
                    # Create guidance-specific subdirectory
                    guidance_dir = os.path.join(output_dir, f"w{guid}")
                    os.makedirs(guidance_dir, exist_ok=True)
                    
                    # Extract results for this guidance level from all models
                    model_results_dict = {}
                    for model_name, model_results in all_model_results.items():
                        gen_atf_1d = model_results[guid][0, :, iz, iy, ix].cpu().numpy()
                        model_results_dict[model_name] = gen_atf_1d
                        
                        # Calculate and print LSD for this model
                        lsd_val = lsd(gt_atf_1d, gen_atf_1d, dim=1, mean=False)
                        lsd_mean = lsd_val.mean()
                        lsd_std = lsd_val.std()
                        print(f'{model_name} (w={guid}): LSD = {lsd_mean:.4f} +- {lsd_std:.4f} dB')
                    
                    # Create and save multi-model comparison plot
                    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                    plot_1d_atf_comparison_multi_models(ax, freq_axis, gt_atf_1d, model_results_dict, guid,
                                                       title=f"Multi-Model ATF Comparison: Source {source_idx+922}, Mic {mic_idx}")
                    plt.tight_layout()
                    
                    # Save plot in guidance-specific folder
                    filename = f"src{source_idx+922:04d}_mic{mic_idx:04d}_models.png"
                    filepath = os.path.join(guidance_dir, filename)
                    fig.savefig(filepath, dpi=150, bbox_inches='tight')
                    plt.close(fig)  # Close to save memory
                    
                    plot_count += 1
                    print(f"Saved plot {plot_count}: w{guid}/{filename}")
                    
            else:
                # Single model mode: create separate plots for each guidance level
                model_results = all_model_results[MODEL_NAMES[0]]
                
                for guid in guidance:
                    # Create guidance-specific subdirectory
                    guidance_dir = os.path.join(output_dir, f"w{guid}")
                    os.makedirs(guidance_dir, exist_ok=True)
                    
                    gen_atf_1d = model_results[guid][0, :, iz, iy, ix].cpu().numpy()
                    
                    # Calculate and print LSD
                    lsd_val = lsd(gt_atf_1d, gen_atf_1d, dim=1, mean=False)
                    lsd_mean = lsd_val.mean()
                    lsd_std = lsd_val.std()
                    print(f'LSD (w={guid}): {lsd_mean:.4f} +- {lsd_std:.4f} dB')

                    # Create and save individual guidance plot
                    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                    ax.plot(freq_axis, gt_atf_1d, label='Ground Truth', linewidth=2, color='black')
                    ax.plot(freq_axis, gen_atf_1d, label=f'Generated (w={guid})', 
                            linestyle='--', linewidth=1.5, color='blue')
                    
                    ax.set_title(f"ATF Comparison: Source {source_idx+922}, Mic {mic_idx} (w={guid})")
                    ax.set_xlabel("Frequency (Hz)")
                    ax.set_ylabel("Magnitude (dB)")
                    ax.set_xscale('log')
                    ax.grid(True, which="both", ls="-", alpha=0.5)
                    ax.legend()
                    plt.tight_layout()

                    # Save plot in guidance-specific folder
                    filename = f"src{source_idx+922:04d}_mic{mic_idx:04d}.png"
                    filepath = os.path.join(guidance_dir, filename)
                    fig.savefig(filepath, dpi=150, bbox_inches='tight')
                    plt.close(fig)  # Close to save memory

                    plot_count += 1
                    print(f"Saved plot {plot_count}: w{guid}/{filename}")
    
    print(f"All {total_plots} plots saved to {output_dir}/")


if __name__ == '__main__':
    main()