import matplotlib
matplotlib.use('Qt5Agg', force=True)   # or 'TkAgg'
from matplotlib import pyplot as plt
import torch
from torchvision import datasets, transforms
import os
import numpy as np
import random
from fm_utils import (ATFSliceSampler, FreqConditionalATFSampler, CFGVectorFieldODE, EulerSimulator,
                      ATFUNet)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 42  # You can use any integer you like
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED) # for GPU
np.random.seed(SEED)
random.seed(SEED)

# --- Data and Model Setup ---
# MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/ATFUNet_20250806-185407_iter20000-best-model/model60k.pt" #
# MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/find20_ATFUNet_20250808-174928_iter60k/model60k.pt" #
# MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/find20_noisegauss_ATFUNet_20250808-202859_iter20000-best-model/model.pt" #
# MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/ATFUNet_M30_holeloss_20250811-181215_iter100000/model.pt" #
# MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/ATFUNet_M30_holeloss_20250811-181215_iter100000/checkpoints/model_100000.pt"
# MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/ATFUNet_M30_holeloss_20250811-181215_iter100000/model.pt"
# MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/find20_holeloss_ATFUNet_20250809-192847_100kish/model_best_for100k.pt"
MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/ATFUNet_M5_holeloss_20250814-175237_iter100000-best-model/modelv2.pt"
# MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/ATFUnetFREQCOND_M50_20250815-182257_iter100000/model.pt"
# MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/FREQCOND_M50_Le4_20250818-154438_iter50000/model.pt"
# MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/FREQCOND_M50_Le4_20250818-154438_iter50000/checkpoints/ckpt_final_50000.pt"
# MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/FREQCOND_M50_Le4_sigma1e1_20250818-165410_iter50000/model.pt"
# MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/FREQCOND_M50_Le4_sigma1e1_20250818-165410_iter50000/checkpoints/ckpt_final_50000.pt"
# MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/FREQCOND_M50_LRe3_fbin64_20250818-192005_iter200000/model.pt"
# MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/FREQCOND_M50_Le4_sigma1e1_20250818-165410_iter50000/checkpoints/ckpt_final_200000.pt"
# MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/FREQCOND_M50_LRe4_fbin64_20250818-201558_iter100000/model.pt"
# MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/FREQCOND_M50_LRe4_fbin64_NOGAUASSIAN_20250818-210045_iter100000/model.pt"
# MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/FREQCOND_M50_LRe3_fbin20_NOGAUASSIAN_20250818-213142_iter50000/model.pt"
# MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/FREQCOND_M50_LRe3_fbin20_NOGAUASSIAN_20250818-213142_iter50000/checkpoints/ckpt_final_50000.pt"

# Extract artifact directory name after 'artifacts/' and before next '/'
MODEL_NAME = os.path.basename(os.path.dirname(MODEL_LOAD_PATH))
print(f"Model artifact: {MODEL_NAME}")

checkpoint = torch.load(MODEL_LOAD_PATH, map_location=device)

config = checkpoint.get('config', {}) # Use .get for safety
training_params = config.get('training', {})
FLAG_GAUSSIAN_MASK = training_params.get('flag_gaussian_mask')
sigma_train = training_params.get('sigma')

print("\n--- Automatically Configured from Loaded Model ---")
print(f"  FLAG_GAUSSIAN_MASK: {FLAG_GAUSSIAN_MASK}")
print(f"  Training Sigma: {sigma_train:.4f}")
print("--------------------------------------------------\n")

data_dir = config['data']['data_dir']
#override data_dir with local
data_dir = "ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200/"

src_split = config['data']['src_splits']

model_mode = config["training"].get('model_mode', "spatial")
freq_up_to = config['model'].get('freq_up_to')

model_cfg = config.get('model', {})

# ---- ADD THIS BLOCK FOR BACKWARD COMPATIBILITY ----
# For older models that didn't save freq_up_to in their config
if 'freq_up_to' not in model_cfg and model_mode == "spatial":
    print("WARNING: 'freq_up_to' not found in model config. Manually setting to 20.")
    model_cfg['freq_up_to'] = 20
# ---------------------------------------------------

freq_up_to = model_cfg.get('freq_up_to')

if model_mode == "freq_cond":
    print("freq_cond")
    config['model']["y_dim"] = 5
    config['model']["input_channels"] = 2
    config['model']["output_channels"] = 1

    temp_train_sampler = FreqConditionalATFSampler(
        data_path=data_dir, mode='train', src_splits=src_split,
        freq_up_to=freq_up_to
    )
else:
    print("SPATIAL")
    config['model']["y_dim"] = 4

    # Calculate stats from the training set to correctly de-normalize
    temp_train_sampler = ATFSliceSampler(
        data_path=data_dir, mode='train', src_splits=src_split,
        freq_up_to=config['model'].get('freq_up_to')
    )

    sample_spec, _ = temp_train_sampler.sample(1)
    freq_channels = sample_spec.shape[1]
    config['model']["input_channels"] = freq_channels + 1
    config['model']["output_channels"] = freq_channels + 1
    print(f"Using {config['model']['input_channels']} input channels and {config['model']['output_channels']} output channels for model.")


spec_mean = temp_train_sampler.slices.mean()
spec_std = temp_train_sampler.slices.std()
print(f"Loaded Stats from Training Set: Mean={spec_mean:.4f}, Std={spec_std:.4f}")

# Define the padding and transform for the test set
padding = (0, 0, 1, 1)  # Pad right and bottom for 12x12
transform = transforms.Compose([
    transforms.Pad(padding, padding_mode='reflect'),
    transforms.Normalize((spec_mean,), (spec_std,)),
])

if model_mode == "freq_cond":
    # Create the test sampler with frequency conditioning
    atf_test_sampler = FreqConditionalATFSampler(
        data_path=data_dir, mode='test',
        src_splits=src_split,
        transform=transform,
        freq_up_to=config['model'].get('freq_up_to')
    ).to(device)

else:
    # Create the test sampler
    atf_test_sampler = ATFSliceSampler(
        data_path=data_dir, mode='test',
        src_splits=src_split,
        transform=transform,
        freq_up_to=config['model'].get('freq_up_to')
    ).to(device)

print("\n--- Debugging Test Sampler ---")
if hasattr(atf_test_sampler, 'sample_info') and atf_test_sampler.sample_info is not None:
    # Get unique source IDs from the first column of sample_info
    unique_sources = torch.unique(atf_test_sampler.sample_info[:, 0])
    print(f"Available Source IDs in test set: {unique_sources.long().tolist()}")

    # Get unique z-heights from the second column of sample_info
    unique_z_heights = torch.unique(atf_test_sampler.sample_info[:, 1])
    print(f"Available Z-Heights in test set: {unique_z_heights.tolist()}")
    print("----------------------------\n")
else:
    print("Could not find 'sample_info' in the sampler to debug.")

# Dynamically compute input/output channels based on data
sample_spec, _ = temp_train_sampler.sample(1)

model_kwargs = {
    'channels': config['model']['channels'],
    'num_residual_layers': config['model']['num_residual_layers'],
    't_embed_dim': config['model']['t_embed_dim'],
    'y_dim': config['model']['y_dim'],
    'y_embed_dim': config['model']['y_embed_dim'],
    'input_channels': config['model']['input_channels'],
    'output_channels': config['model']['output_channels'],
}
atf_unet = ATFUNet(**model_kwargs).to(device)

# Print checkpoint info
is_best = checkpoint.get('is_best', False)
iter_info = f"iteration {checkpoint.get('iteration', '?')}"
if is_best:
    print(f"Loading BEST model state (from {iter_info})")
else:
    print(f"Loading latest checkpoint state (from {iter_info})")
print(f"Validation loss: {checkpoint.get('best_val_loss', '?'):.4f}")

atf_unet.load_state_dict(checkpoint['model_state_dict'])
atf_unet.eval()
print(f"--- Loaded model from {MODEL_LOAD_PATH} for inference ---")

# --- Inference Setup ---
ode_inference = CFGVectorFieldODE(net=atf_unet)
ode_inference.y_null.data = checkpoint['y_null'].to(device)
simulator = EulerSimulator(ode_inference)

# --- Visualization Parameters ---
guidance_scales = [1.0, 5]  # explore different guidance strengths
# Fixed total number of integration steps for accuracy/stability
num_timesteps = 100

# Layout: 5 examples (rows) x (2 + len(guidance_scales)) columns
num_examples = 9  # different random samples to show
num_cols = 2 + len(guidance_scales)  # GT, Input, then one per guidance scale
M = 5  # Number of sparse points to use as input


# --- Generate and Plot ---
fig, axes = plt.subplots(num_examples, num_cols, figsize=(4 * num_cols, 4 * num_examples), squeeze=False)

if model_mode == "freq_cond":
    fig.suptitle(f"Inpainting Results (M={M}) | {MODEL_NAME}", fontsize=16)
elif model_mode == "spatial":
    freq_idx_to_plot = 5  # Which frequency channel to visualize
    fig.suptitle(f"Inpainting Results (M={M}, Freq Idx={freq_idx_to_plot}) | {MODEL_NAME}", fontsize=16)

for row in range(num_examples):
    # 1. Get a random ground truth slice and its conditioning vector
    z_true, y_true = atf_test_sampler.sample(1)
    # z_true, y_true = atf_test_sampler.get_slice_by_id(src_id=930, z_height=0.0)
    if model_mode == "freq_cond":
        ffreq = float((y_true[0, -1] * 1000).item())
        print(f"ffreq: {ffreq:.0f} Hz, Conditioning Vector: {y_true}")
    else:
        print(f"Conditioning Vector: {y_true}")
    # 2. Create the sparse input
    B, _, H, W = z_true.shape
    mask = torch.zeros(B, 1, H, W, device=z_true.device)

    # Vectorized masking: Generate M random indices for each sample in the batch
    num_pixels = (H - 1) * (W - 1)
    indices = torch.multinomial(torch.ones(B, num_pixels), M, replacement=False).to(z_true.device)

    rows = indices // (W - 1)
    cols = indices % (W - 1)

    # Use advanced indexing to set the mask values for the entire batch at once
    batch_indices = torch.arange(B, device=z_true.device).view(-1, 1)
    mask[batch_indices, 0, rows, cols] = 1

    # Create the sparse input by broadcasting the mask
    x0_sparse = z_true * mask

    if FLAG_GAUSSIAN_MASK:
        noise = torch.randn_like(z_true) * sigma_train  # Gaussian prior
        x0_full = x0_sparse + (1 - mask) * noise  # fill the holes
        x0_model_input = torch.cat([x0_full, mask], dim=1)

    else:  # 3. Prepare the input for the model (data + mask channel)
        x0_model_input = torch.cat([x0_sparse, mask], dim=1)

    # 3. De-normalize for visualization
    z_true_denorm = (z_true * spec_std + spec_mean)
    x0_sparse_denorm = (x0_sparse * spec_std + spec_mean)
    z_plot = z_true_denorm[0, 0, :-1, :-1].detach().cpu().numpy()
    x0_plot_raw = x0_sparse_denorm[0, 0, :-1, :-1].detach().cpu().numpy()
    
    # Create proper sparse visualization: use NaN for missing values (where mask=0)
    mask_2d = mask[0, 0, :-1, :-1].detach().cpu().numpy()  # 2D mask for this frequency
    x0_plot = x0_plot_raw.copy()
    x0_plot[mask_2d == 0] = np.nan  # Missing values become NaN (transparent in imshow)

    # Per-row normalization based on ground-truth slice
    vmin = float(np.min(z_plot))
    vmax = float(np.max(z_plot))

    # 4. Plot ground truth and sparse input
    im_gt = axes[row, 0].imshow(z_plot, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[row, 0].set_title("Ground Truth" if row == 0 else "")
    axes[row, 0].axis('off')
    if model_mode == "freq_cond":
        axes[row, 0].text(0.02, 0.98, f"ffreq = {ffreq:.0f}", transform=axes[row, 0].transAxes,
                          va='top', ha='left', fontsize=10)

    # For sparse input, use a colormap that shows NaN as white/transparent
    cmap_sparse = plt.cm.viridis.copy()
    cmap_sparse.set_bad('white', alpha=0.3)  # NaN values appear as semi-transparent white
    
    im_sparse = axes[row, 1].imshow(x0_plot, origin='lower', cmap=cmap_sparse, vmin=vmin, vmax=vmax)
    axes[row, 1].set_title(f"Sparse Input (M={M})" if row == 0 else "")
    axes[row, 1].axis('off')

    # Store the last image for consistent colorbar reference
    last_im = im_sparse
    
    # 5. For each guidance scale, simulate and plot
    for g_idx, guidance_scale in enumerate(guidance_scales):
        simulator.ode.guidance_scale = guidance_scale
        ts = torch.linspace(0, 1, num_timesteps + 1, device=x0_model_input.device)
        ts = ts.view(1, -1, 1, 1, 1).expand(x0_model_input.shape[0], -1, -1, -1, -1)
        
        x1_recon = simulator.simulate(x0_model_input.clone(), ts, y=y_true)
        
        # Enforce constraint: restore known values from the original sparse input
        # Extract the non-mask channels from both tensors
        x1_recon_data = x1_recon[:, :-1]  # Remove mask channel
        x0_sparse_data = x0_sparse  # Original sparse data
        recon_mask = mask  # The mask indicating known locations
        
        # Keep known values fixed: use sparse input where mask=1, reconstruction elsewhere
        x1_recon_data = x0_sparse_data * recon_mask + x1_recon_data * (1 - recon_mask)
        
        # Reconstruct the full tensor with mask channel for consistency
        x1_recon = torch.cat([x1_recon_data, recon_mask], dim=1)

        x1_recon_denorm = (x1_recon * spec_std + spec_mean)
        x1_plot = x1_recon_denorm[0, 0, :-1, :-1].detach().cpu().numpy()

        col_idx = g_idx + 2  # +2 for GT and Input columns
        last_im = axes[row, col_idx].imshow(x1_plot, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[row, col_idx].set_title(f"w={guidance_scale}" if row == 0 else "")
        axes[row, col_idx].axis('off')

    # Add colorbar for this row using the last plotted image (all have same vmin/vmax)
    cbar = fig.colorbar(last_im, ax=axes[row, :], location='right', fraction=0.02, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Magnitude", fontsize=9)

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()