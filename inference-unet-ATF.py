import matplotlib
matplotlib.use('Qt5Agg', force=True)   # or 'TkAgg'
from matplotlib import pyplot as plt
import torch
from torchvision import datasets, transforms
import os
import numpy as np
import random
from fm_utils import (ATFSliceSampler, CFGVectorFieldODE, EulerSimulator,
                      ATFUNet)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 42  # You can use any integer you like
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED) # for GPU
np.random.seed(SEED)
random.seed(SEED)

# --- Configuration (should match your ATF training script) ---
config = {
    "data": {
        "data_dir": "ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200/",
        "src_splits": {
            "train": [0, 820],
            "valid": [820, 922],
            "test": [922, 1024],
            "all": [0, 1024]}
    },
    "model": {
        "channels": [32, 64, 128], "num_residual_layers": 2,
        "t_embed_dim": 40, "y_dim": 4, "y_embed_dim": 40,
        # Optional: if set, use only the first N frequency channels
        "freq_ind_up_to": 20
    }
}

# --- Data and Model Setup ---
# MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/ATFUNet_20250806-185407_iter20000-best-model/model60k.pt" #
# MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/find20_ATFUNet_20250808-174928_iter60k/model60k.pt" #
# MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/find20_noisegauss_ATFUNet_20250808-202859_iter20000-best-model/model.pt" #
# MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/find20_newlossonlyholes_ATFUNet_20250809-192847_/model60k.pt" #
MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/find20_newlossonlyholes_ATFUNet_20250809-192847_/model_best_for100k.pt" #

data_dir = config['data']['data_dir']
src_split = config['data']['src_splits']

# Calculate stats from the training set to correctly de-normalize
temp_train_sampler = ATFSliceSampler(
    data_path=data_dir, mode='train', src_splits=src_split,
    freq_ind_up_to=config['model'].get('freq_ind_up_to')
)
spec_mean = temp_train_sampler.slices.mean()
spec_std = temp_train_sampler.slices.std()
print(f"Loaded Stats from Training Set: Mean={spec_mean:.4f}, Std={spec_std:.4f}")

# Define the padding and transform for the test set
padding = (0, 0, 1, 1)  # Pad right and bottom for 12x12
transform = transforms.Compose([
    transforms.Pad(padding, padding_mode='reflect'),
    transforms.Normalize((spec_mean,), (spec_std,)),
])

# Create the test sampler
atf_test_sampler = ATFSliceSampler(
    data_path=data_dir, mode='test',
    src_splits=src_split,
    transform=transform,
    freq_ind_up_to=config['model'].get('freq_ind_up_to')
).to(device)

# Dynamically compute input/output channels based on data
sample_spec, _ = temp_train_sampler.sample(1)
freq_channels = sample_spec.shape[1]
input_channels = freq_channels + 1
output_channels = freq_channels + 1

model_kwargs = {
    'channels': config['model']['channels'],
    'num_residual_layers': config['model']['num_residual_layers'],
    't_embed_dim': config['model']['t_embed_dim'],
    'y_dim': config['model']['y_dim'],
    'y_embed_dim': config['model']['y_embed_dim'],
    'input_channels': input_channels,
    'output_channels': output_channels,
}
atf_unet = ATFUNet(**model_kwargs).to(device)
checkpoint = torch.load(MODEL_LOAD_PATH, map_location=device)

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
guidance_scales = [1.0, 3, 5]  # explore different guidance strengths
# Fixed total number of integration steps for accuracy/stability
num_timesteps = 100

# Layout: 5 examples (rows) x (2 + len(guidance_scales)) columns
num_examples = 5  # different random samples to show
num_cols = 2 + len(guidance_scales)  # GT, Input, then one per guidance scale
M = 50  # Number of sparse points to use as input
freq_idx_to_plot = 10  # Which frequency channel to visualize
FLAG_GAUSSIAN_MASK = False  # If True, use Gaussian noise to fill the holes

# --- Generate and Plot ---
fig, axes = plt.subplots(num_examples, num_cols, figsize=(4 * num_cols, 4 * num_examples), squeeze=False)
fig.suptitle(f"Inpainting Results (M={M}, Freq Idx={freq_idx_to_plot})", fontsize=16)

for row in range(num_examples):
    # 1. Get a random ground truth slice and its conditioning vector
    z_true, y_true = atf_test_sampler.sample(1)
    print(f"freq ind: {freq_idx_to_plot}, Conditioning Vector: {y_true}")
    
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
        sigma = 0.1
        noise = torch.randn_like(z_true) * sigma  # Gaussian prior
        x0_full = x0_sparse + (1 - mask) * noise  # fill the holes
        x0_model_input = torch.cat([x0_full, mask], dim=1)

    else:  # 3. Prepare the input for the model (data + mask channel)
        x0_model_input = torch.cat([x0_sparse, mask], dim=1)

    # 3. De-normalize for visualization
    z_true_denorm = (z_true * spec_std + spec_mean)
    x0_sparse_denorm = (x0_sparse * spec_std + spec_mean)
    z_plot = z_true_denorm[0, freq_idx_to_plot, :-1, :-1].detach().cpu().numpy()
    x0_plot = x0_sparse_denorm[0, freq_idx_to_plot, :-1, :-1].detach().cpu().numpy()

    # Per-row normalization based on ground-truth slice
    vmin = float(np.min(z_plot))
    vmax = float(np.max(z_plot))

    # 4. Plot ground truth and sparse input
    im_gt = axes[row, 0].imshow(z_plot, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[row, 0].set_title("Ground Truth" if row == 0 else "")
    axes[row, 0].axis('off')

    axes[row, 1].imshow(x0_plot, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[row, 1].set_title(f"Sparse Input (M={M})" if row == 0 else "")
    axes[row, 1].axis('off')

    # 5. For each guidance scale, simulate and plot
    for g_idx, guidance_scale in enumerate(guidance_scales):
        simulator.ode.guidance_scale = guidance_scale
        ts = torch.linspace(0, 1, num_timesteps + 1, device=x0_model_input.device)
        ts = ts.view(1, -1, 1, 1, 1).expand(x0_model_input.shape[0], -1, -1, -1, -1)
        x1_recon = simulator.simulate(x0_model_input.clone(), ts, y=y_true)

        x1_recon_denorm = (x1_recon * spec_std + spec_mean)
        x1_plot = x1_recon_denorm[0, freq_idx_to_plot, :-1, :-1].detach().cpu().numpy()

        col_idx = g_idx + 2  # +2 for GT and Input columns
        im = axes[row, col_idx].imshow(x1_plot, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[row, col_idx].set_title(f"w={guidance_scale}" if row == 0 else "")
        axes[row, col_idx].axis('off')

    # Add colorbar for this row
    cbar = fig.colorbar(im_gt, ax=axes[row, :], location='right', fraction=0.02, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Magnitude", fontsize=9)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()