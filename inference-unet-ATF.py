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
MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/find20_ATFUNet_20250808-174928_iter60k/model60k.pt" #
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

# --- Model Loading ---
if not os.path.exists(MODEL_LOAD_PATH):
    print(f"Model file not found at {MODEL_LOAD_PATH}.")
    exit()

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
atf_unet.load_state_dict(checkpoint['model_state_dict'])
atf_unet.eval()
print(f"--- Loaded model from {MODEL_LOAD_PATH} for inference ---")

# --- Inference Setup ---
ode_inference = CFGVectorFieldODE(net=atf_unet)
ode_inference.y_null.data = checkpoint['y_null'].to(device)
simulator = EulerSimulator(ode_inference)

# --- Visualization Parameters ---
# num_timesteps = 10
# guidance_scale = [1.0, 1.5, 2]
# num_plots = len(guidance_scales) + 2

guidance_scale = 3.0  # Fixed guidance for this visualization
timesteps_to_visualize = [x*10 for x in range(1,10)]
# timesteps_to_visualize = [10, 30, 50, 70, 90, 150, 200, 300]
# timesteps_to_visualize = [10, 200, 500, 1000, 2000]

num_plots = 5 # How many different random examples to show
M = 50  # Number of sparse points to use as input
freq_idx_to_plot = 5  # Which frequency channel to visualize

# --- Generate and Plot ---
# We have 2 fixed columns (True, Sparse) + one for each timestep we visualize
num_cols = 2 + len(timesteps_to_visualize)
fig, axes = plt.subplots(num_plots, num_cols, figsize=(4 * num_cols, 4 * num_plots), squeeze=False)
fig.suptitle(f"Inpainting Results (M={M}, Freq Idx={freq_idx_to_plot}, Guidance={guidance_scale})", fontsize=16)

for i in range(num_plots):
    # 1. Get a random ground truth slice and its conditioning vector
    z_true, y_true = atf_test_sampler.sample(1)
    print(f"freq ind: {freq_idx_to_plot}, Conditioning Vector: {y_true}")
    # 2. --- REPLICATE THE ROBUST MASKING LOGIC ---
    # Get the batch size, height, and width
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

    # 3. Prepare the input for the model (data + mask channel)
    x0_model_input = torch.cat([x0_sparse, mask], dim=1)

    # 4. De-normalize and plot the Ground Truth and Sparse Input once per row
    z_true_denorm = (z_true * spec_std + spec_mean)
    x0_sparse_denorm = (x0_sparse * spec_std + spec_mean)

    z_plot = z_true_denorm[0, freq_idx_to_plot, :-1, :-1].cpu().numpy()
    x0_plot = x0_sparse_denorm[0, freq_idx_to_plot, :-1, :-1].cpu().numpy()

    axes[i, 0].imshow(z_plot, origin='lower', cmap='viridis')
    axes[i, 0].set_title("Ground Truth")

    axes[i, 1].imshow(x0_plot, origin='lower', cmap='viridis')
    axes[i, 1].set_title(f"Sparse Input (M={M})")

    # 5. Loop through timesteps to generate and plot reconstructions
    simulator.ode.guidance_scale = guidance_scale
    
    # Efficient single-pass stepping: compute once up to max step and capture selected frames
    steps_set = set(timesteps_to_visualize)
    max_steps = max(timesteps_to_visualize)
    ts_lin = torch.linspace(0, 1, max_steps + 1, device=x0_model_input.device)
    frames = {}
    x_state = x0_model_input.clone()
    for step_idx in range(1, max_steps + 1):
        t_current = ts_lin[step_idx - 1].view(1, 1, 1, 1).expand(x_state.shape[0], -1, -1, -1)
        h = ts_lin[step_idx] - ts_lin[step_idx - 1]
        x_state = simulator.step(x_state, t_current, h, y=y_true)
        if step_idx in steps_set:
            frames[step_idx] = x_state.clone()

    for j, step in enumerate(timesteps_to_visualize):
        x1_recon = frames[step]

        # De-normalize and crop for visualization
        x1_recon_denorm = (x1_recon * spec_std + spec_mean)
        x1_plot = x1_recon_denorm[0, freq_idx_to_plot, :-1, :-1].cpu().numpy()

        # Plot the reconstruction in the correct column
        axes[i, j + 2].imshow(x1_plot, origin='lower', cmap='viridis')
        axes[i, j + 2].set_title(f"Timestep {step}")


    # Turn off axis for all plots in the row
    for ax in axes[i]:
        ax.axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()