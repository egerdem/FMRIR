
from matplotlib import pyplot as plt
import torch
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import matplotlib
matplotlib.use('MacOSX')  # Use 'TkAgg' for Linux, 'MacOSX' for macOS, 'Qt5Agg' for Windows
import os

from fm_utils import (
    SpectrogramSampler, GaussianConditionalProbabilityPath, LinearAlpha,
    LinearBeta, CFGVectorFieldODE, EulerSimulator, SpecUNet
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

src_split = {
            "train": [0, 820],
            "valid": [820, 922],
            "test": [922, 1024],
            "all": [0, 1024]}

# --- Data and Model Setup ---
# MODEL_LOAD_PATH = "experiments/SpecUNet_20250730-112150/model.pt"
MODEL_LOAD_PATH = "experiments/SpecUNet_20250730-065144_5kv1/model.pt"

data_dir = "ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200/"
temp_sampler = SpectrogramSampler(data_path=data_dir, mode="test", src_splits=src_split)

spec_mean = temp_sampler.spectrograms.mean()
spec_std = temp_sampler.spectrograms.std()
print(f"Loaded Spectrogram Stats: Mean={spec_mean:.4f}, Std={spec_std:.4f}")

spec_test_sampler = SpectrogramSampler(
    data_path=data_dir, mode='test',
    src_splits=src_split,
    transform=transforms.Compose([
        transforms.Normalize((spec_mean,), (spec_std,)),
    ])
).to(device)

sample_spec, _ = spec_test_sampler.sample(1)
spec_shape = list(sample_spec.shape[1:])

path = GaussianConditionalProbabilityPath(
    p_data=spec_test_sampler,
    p_simple_shape=spec_shape,
    alpha=LinearAlpha(),
    beta=LinearBeta()
).to(device)

spec_unet = SpecUNet(
    channels=[32, 64, 128],
    num_residual_layers=2,
    t_embed_dim=40,
    y_dim=6,
    y_embed_dim=40,
).to(device)

# --- Load Model ---
if not os.path.exists(MODEL_LOAD_PATH):
    print(f"Model file not found at {MODEL_LOAD_PATH}. Please run the training script first.")
    exit()

print(f"--- Loading model from {MODEL_LOAD_PATH} for inference ---")
checkpoint = torch.load(MODEL_LOAD_PATH, map_location=device)
spec_unet.load_state_dict(checkpoint['model_state_dict'])

spec_unet.to(device)
spec_unet.eval()

# --- Inference / Generation ---
# Play with these!
num_plots = 5
num_timesteps = 100
guidance_scales = [1.0, 3.0, 5.0]
target_source_id = None  # Set to a specific source ID from the test set (e.g., 930) to generate specific samples
target_mic_start_id = None # Set to a specific mic ID (e.g., 100)

# --- Setup the ODE wrapper for inference ---
# Make sure to use the same y_dim and y_embed_dim as your trained model
ode_inference = CFGVectorFieldODE(
    net=spec_unet,
    guidance_scale=1.0,  # Will be updated in the loop
    y_dim=6,
    y_embed_dim=40
)

# Pass the learned null embedding from the trainer to the ODE
# The key should match what was saved during training
ode_inference.y_null.data = checkpoint['y_null'].to(device)

# --- Setup the simulator ---
simulator = EulerSimulator(ode_inference)

# --- Generate and Plot ---
num_cols = 1 + len(guidance_scales)
fig, axes = plt.subplots(num_plots, num_cols, figsize=(4 * num_cols, 4 * num_plots), squeeze=False)
fig.suptitle("Comparison: Ground Truth vs. Generated with Different Guidance", fontsize=14)

for i in range(num_plots):
    item_index = None
    if target_source_id is not None and target_mic_start_id is not None:
        # Find the specific sample if requested
        item_index = spec_test_sampler.find_sample_index(target_source_id, target_mic_start_id + i)
        if item_index is None:
            print(f"Sample for source {target_source_id}, mic {target_mic_start_id + i} not in test set. Skipping.")
            for ax in axes[i]:
                ax.axis('off')
            continue
    else:
        # Pick a random sample from the test set
        item_index = torch.randint(0, len(spec_test_sampler.spectrograms), (1,)).item()

    # Get the ground truth data and its metadata
    z_true, y_true, info = spec_test_sampler.get_item_by_idx(item_index)
    src_id, mic_id = info.squeeze().tolist()
    print(f"Plotting for Row {i+1} (Source ID: {src_id}, Mic ID: {mic_id})")

    # Plot Ground Truth in the first column
    z_true_denorm = z_true * spec_std + spec_mean
    axes[i, 0].imshow(z_true_denorm.squeeze().cpu().numpy(), cmap="viridis")
    axes[i, 0].set_title(f"True (S:{src_id} M:{mic_id})")
    axes[i, 0].axis("off")

    # Generate from the same coordinates with different guidances
    x0, _ = path.p_simple.sample(1)
    ts = torch.linspace(0, 1, num_timesteps).view(1, -1, 1, 1, 1).expand(1, -1, 1, 1, 1).to(device)

    for j, w in enumerate(guidance_scales):
        # Update guidance scale
        simulator.ode.guidance_scale = w

        # Simulate to generate the spectrograms
        x1_gen = simulator.simulate(x0, ts, y=y_true)

        # De-normalize for visualization
        x1_gen_denorm = x1_gen * spec_std + spec_mean
        
        # Plot Generated
        axes[i, j + 1].imshow(x1_gen_denorm.squeeze().cpu().numpy(), cmap="viridis")
        axes[i, j + 1].set_title(f"Gen (w={w:.1f})")
        axes[i, j + 1].axis("off")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()