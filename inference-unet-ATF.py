import matplotlib
matplotlib.use('Qt5Agg', force=True)   # or 'TkAgg'
from matplotlib import pyplot as plt
import torch
from torchvision import datasets, transforms
import os

from fm_utils import (ATFSliceSampler, CFGVectorFieldODE, EulerSimulator,
                      ATFUNet)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



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
        "t_embed_dim": 40, "y_dim": 4, "y_embed_dim": 40
    }
}
M = 40  # Number of sparse points to use as input

# --- Data and Model Setup ---
MODEL_LOAD_PATH = "experiments/ATFUNet_20250805-210109_iter100/checkpoints/ckpt_100.pt" #
data_dir = config['data']['data_dir']
src_split = config['data']['src_splits']

# Calculate stats from the training set to correctly de-normalize
temp_train_sampler = ATFSliceSampler(data_path=data_dir, mode='train', src_splits=src_split)
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
    transform=transform
).to(device)

# --- Model Loading ---
if not os.path.exists(MODEL_LOAD_PATH):
    print(f"Model file not found at {MODEL_LOAD_PATH}.")
    exit()

atf_unet = ATFUNet(**config['model']).to(device)
checkpoint = torch.load(MODEL_LOAD_PATH, map_location=device)
atf_unet.load_state_dict(checkpoint['model_state_dict'])
atf_unet.eval()
print(f"--- Loaded model from {MODEL_LOAD_PATH} for inference ---")

# --- Inference Setup ---
ode_inference = CFGVectorFieldODE(net=atf_unet)
ode_inference.y_null.data = checkpoint['y_null'].to(device)
simulator = EulerSimulator(ode_inference)

# --- Visualization Parameters ---
num_plots = 5
num_timesteps = 100
guidance_scales = [1.0, 2.0, 3.0]
freq_idx_to_plot = 10  # Which frequency channel to visualize

# --- Generate and Plot ---
# We have 2 fixed columns (True, Sparse) + one for each guidance scale
num_cols = 2 + len(guidance_scales)
fig, axes = plt.subplots(num_plots, num_cols, figsize=(4 * num_cols, 4 * num_plots), squeeze=False)
fig.suptitle(f"Inpainting Results (M={M}, Freq Idx={freq_idx_to_plot})", fontsize=16)

for i in range(num_plots):
    # 1. Get a random ground truth slice and its conditioning vector
    z_true, y_true = atf_test_sampler.sample(1)

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

    # 5. Loop through guidance scales to generate and plot reconstructions
    ts = torch.linspace(0, 1, num_timesteps).view(1, -1, 1, 1, 1).expand(1, -1, 1, 1, 1).to(device)

    for j, w in enumerate(guidance_scales):
        # Update the guidance scale in the simulator
        simulator.ode.guidance_scale = w

        # Simulate the ODE to generate the reconstruction
        x1_recon = simulator.simulate(x0_model_input, ts, y=y_true)

        # De-normalize and crop for visualization
        x1_recon_denorm = (x1_recon * spec_std + spec_mean)
        x1_plot = x1_recon_denorm[0, freq_idx_to_plot, :-1, :-1].cpu().numpy()

        # Plot the reconstruction in the correct column
        axes[i, j + 2].imshow(x1_plot, origin='lower', cmap='viridis')
        axes[i, j + 2].set_title(f"Recon (w={w:.1f})")

    # Turn off axis for all plots in the row
    for ax in axes[i]:
        ax.axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()