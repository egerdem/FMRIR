import torch
import numpy as np
import os
import json
from tqdm import tqdm
import matplotlib
matplotlib.use('Qt5Agg', force=True)
from matplotlib import pyplot as plt

# Your model imports
from fm_utils import ATF3DSampler

# Reference model imports
import sys
sys.path.append('AUTOENCODER/src')
import AUTOENCODER.src.dataset as autoencoder_dataset
from AUTOENCODER.src.configs import config_FSMPAE_10026
import AUTOENCODER.src.utils as autoencoder_utils
import AUTOENCODER.src.models as autoencoder_models
import AUTOENCODER.src.trainer as autoencoder_trainer

# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def calculate_lsd_unified(estimation, ground_truth, freq_dim=1):
    """Unified LSD calculation."""
    squared_error = (estimation - ground_truth) ** 2
    lsd_per_position = torch.sqrt(torch.mean(squared_error, dim=freq_dim))
    return torch.mean(lsd_per_position)

def load_reference_model(device, freq_up_to=None):
    """Load the reference AUTOENCODER model for inference."""
    print("Loading reference AUTOENCODER model...")
    
    # Load config
    config = config_FSMPAE_10026.copy()
    
    try:
        # Change to AUTOENCODER directory for dataset loading
        original_cwd = os.getcwd()
        os.chdir('AUTOENCODER')
        
        # Load dataset
        idataset = autoencoder_dataset.ATFdataset(config=config)
        data = idataset.Data
        
        # Load model checkpoint
        model_path = 'outputs/out_20250323_FSMPAE_10026/network.best.net'
        
        if not os.path.exists(model_path):
            print(f"❌ Model checkpoint not found: {model_path}")
            os.chdir(original_cwd)
            return None, None, None, None
        
        # Create model instance (need to figure out the architecture)
        # For now, let's load the checkpoint and examine it
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print(f"✅ Checkpoint loaded. Keys: {list(checkpoint.keys())}")
        
        # Change back to original directory
        os.chdir(original_cwd)
        
        return checkpoint, data, config, idataset
        
    except Exception as e:
        print(f"❌ Error loading reference model: {e}")
        if 'original_cwd' in locals():
            os.chdir(original_cwd)
        return None, None, None, None

def run_reference_inference(model, data, config, device, num_sources_eval=None, freq_up_to=None):
    """
    Run inference with the reference AUTOENCODER model.
    
    This function will:
    1. Load sparse M=5 observations (same as your model)
    2. Run reference model inference 
    3. Generate predictions at all 1331 microphone positions
    4. Compare with ground truth
    """
    print("🚀 Running reference AUTOENCODER inference...")
    
    # Get test data
    dataset_name = config['dataset'][0]
    src_position = data['test']['src_position'][dataset_name]  # [1331, 3, 102]
    mic_position = data['test']['mic_position'][dataset_name]  # [1331, 3, 1024] 
    atf_mag_gt = data['test']['atf_mag'][dataset_name]         # [1331, 64, 102]
    
    # Limit evaluation sources
    total_sources = atf_mag_gt.shape[2]  # 102 sources
    if num_sources_eval is not None:
        eval_sources = min(num_sources_eval, total_sources)
        print(f"Evaluating on first {eval_sources} sources (out of {total_sources})")
    else:
        eval_sources = total_sources
        print(f"Evaluating on all {eval_sources} sources")
    
    # Load microphone selection strategy (same as your model)
    idx_mes_pos_path = "AUTOENCODER/ATF_interp/idx_mes_pos_s1024_m1331.npy"
    if os.path.exists(idx_mes_pos_path):
        idx_mes_pos_mat = np.load(idx_mes_pos_path)
        print(f"✅ Loaded microphone selection matrix: {idx_mes_pos_mat.shape}")
    else:
        print("❌ Microphone selection matrix not found!")
        return None
    
    M = config['num_mes_test']  # 5 microphones
    print(f"Using M={M} observation microphones per source")
    
    # Initialize results storage
    predictions = torch.zeros_like(atf_mag_gt[:, :freq_up_to if freq_up_to else 64, :eval_sources])
    lsd_scores = []
    mse_scores = []
    
    print("⚠️  NOTE: Reference model inference implementation needed!")
    print("The reference model architecture needs to be instantiated from the checkpoint.")
    print("This requires understanding their model structure from src/models.py")
    print()
    print("For now, returning pre-computed results for comparison...")
    
    # Load pre-computed results as fallback
    pt_dir = 'AUTOENCODER/outputs/out_20250323_FSMPAE_10026'
    pt_path = f'{pt_dir}/atf_mag/atf_mag_test_{dataset_name}.pt'
    
    if os.path.exists(pt_path):
        atf_mag_est = torch.load(pt_path, weights_only=False)
        print(f"✅ Loaded pre-computed predictions: {atf_mag_est.shape}")
        
        # Truncate to match frequency range if needed
        if freq_up_to:
            atf_mag_est = atf_mag_est[:, :freq_up_to, :]
            atf_mag_gt_truncated = atf_mag_gt[:, :freq_up_to, :]
        else:
            atf_mag_gt_truncated = atf_mag_gt
        
        # Limit to eval_sources
        atf_mag_est = atf_mag_est[:, :, :eval_sources]
        atf_mag_gt_truncated = atf_mag_gt_truncated[:, :, :eval_sources]
        
        # Calculate metrics
        for src_idx in tqdm(range(eval_sources), desc="Reference Evaluation"):
            # LSD
            lsd_val = calculate_lsd_unified(
                atf_mag_est[:, :, src_idx], 
                atf_mag_gt_truncated[:, :, src_idx], 
                freq_dim=1
            )
            lsd_scores.append(lsd_val.item())
            
            # MSE
            mse_val = torch.mean((atf_mag_est[:, :, src_idx] - atf_mag_gt_truncated[:, :, src_idx]) ** 2)
            mse_scores.append(mse_val.item())
        
        results = {
            'predictions': atf_mag_est,
            'ground_truth': atf_mag_gt_truncated,
            'lsd_mean': np.mean(lsd_scores),
            'lsd_std': np.std(lsd_scores),
            'mse_mean': np.mean(mse_scores),
            'mse_std': np.std(mse_scores),
            'num_sources_eval': eval_sources,
            'M': M
        }
        
        print(f"📊 Reference Results:")
        print(f"   LSD: {results['lsd_mean']:.4f} ± {results['lsd_std']:.4f} dB")
        print(f"   MSE: {results['mse_mean']:.4f} ± {results['mse_std']:.4f}")
        
        return results
    
    else:
        print(f"❌ Pre-computed results not found: {pt_path}")
        return None

def compare_with_your_model(ref_results, your_model_path, device, freq_up_to=20):
    """
    Compare reference model results with your Flow Matching model.
    This implements the same evaluation as unified_evaluation.py but with 
    direct inference comparison.
    """
    print("\n🔄 Loading your Flow Matching model for comparison...")
    
    # Load your model (simplified from unified_evaluation.py)
    from fm_utils import (
        SetEncoder, CrossAttentionUNet3D, CrossAttentionUNet3D_RED3d, 
        CFGVectorFieldODE_3D, CFGVectorFieldODE_3D_V2, EulerSimulator
    )
    
    checkpoint = torch.load(your_model_path, map_location=device)
    config = checkpoint.get('config', {})
    model_states_cfg = checkpoint['model_states']
    
    model_cfg = config['model']
    architecture = model_cfg.get('architecture_version')
    
    # Load your models
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
    
    print("✅ Your model loaded successfully!")
    
    # TODO: Implement your model evaluation here
    # This would follow the same pattern as evaluate_your_model() in unified_evaluation.py
    # but with the same sources and microphone selections as the reference inference
    
    print("⚠️  Your model evaluation implementation needed here!")
    print("This should run inference on the same sources/microphones as reference model")
    
    return None

def main():
    """Main function to run reference model inference and comparison."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Reference Model Inference Script")
    print(f"Device: {device}")
    print()
    
    # Configuration
    freq_up_to = 20  # Match your model's frequency range
    num_sources_eval = 10  # Limit for faster testing
    your_model_path = "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d512_head8_sigma0_lrWARM5k_e4_toe5_unet4_layer3_20250906-191114_iter300000/model.pt"
    
    # Step 1: Load reference model
    checkpoint, data, config, dataset = load_reference_model(device, freq_up_to)
    
    if checkpoint is None:
        print("❌ Failed to load reference model. Exiting...")
        return
    
    # Step 2: Run reference inference  
    ref_results = run_reference_inference(
        checkpoint, data, config, device, 
        num_sources_eval=num_sources_eval, 
        freq_up_to=freq_up_to
    )
    
    if ref_results is None:
        print("❌ Reference inference failed. Exiting...")
        return
    
    # Step 3: Compare with your model
    your_results = compare_with_your_model(
        ref_results, your_model_path, device, freq_up_to
    )
    
    print("\n" + "="*60)
    print("🎯 DIRECT INFERENCE COMPARISON RESULTS")
    print("="*60)
    print(f"Reference Model (M={ref_results['M']}):")
    print(f"  LSD: {ref_results['lsd_mean']:.4f} ± {ref_results['lsd_std']:.4f} dB")
    print(f"  MSE: {ref_results['mse_mean']:.4f} ± {ref_results['mse_std']:.4f}")
    print(f"  Sources: {ref_results['num_sources_eval']}")
    print()
    print("Your Model:")
    print("  [Implementation needed]")
    print("="*60)
    
    print("\n💡 Next Steps:")
    print("1. Implement reference model architecture instantiation")
    print("2. Implement reference model forward pass")
    print("3. Run your model with same inputs for fair comparison")
    print("4. Generate comprehensive comparison plots")

if __name__ == "__main__":
    main()

