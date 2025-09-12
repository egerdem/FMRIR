"""
Simple MNIST U-Net Inference Script
Loads trained model and runs inference using modules from the main training script.
"""

import torch
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
import matplotlib
matplotlib.use('Qt5Agg', force=True)  # or 'TkAgg'
from matplotlib import pyplot as plt
# Import all necessary classes from the main training script
from unet_mnist_lab3 import (
    device, path, MNISTUNet, CFGVectorFieldODE, EulerSimulator, DDPMScheduler, DDIMSampler2D
)
from tqdm import tqdm

def load_trained_model(model_path: str):
    """Load the trained model from a saved checkpoint"""
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    
    # Initialize model with saved config
    unet = MNISTUNet(
        channels=model_config['channels'],
        num_residual_layers=model_config['num_residual_layers'],
        t_embed_dim=model_config['t_embed_dim'],
        y_embed_dim=model_config['y_embed_dim']
    )
    
    # Load the trained weights
    unet.load_state_dict(checkpoint['model_state_dict'])
    unet.to(device)
    unet.eval()
    
    print(f"Model loaded successfully from {model_path}")
    return unet

# Load the trained model
model_name = 'trained_mnist_ddpm.pt'
unet = load_trained_model(model_name)

# Play with these!
samples_per_class = 10
num_timesteps = 300
guidance_scales = [1.0]

model = "flow"

if "ddpm" in model_name:
    model = "ddpm"
    # --- DDPM INFERENCE SETUP ---
    # 1. Initialize the scheduler and sampler
    ddpm_scheduler = DDPMScheduler(num_timesteps=1000)
    sampler = DDIMSampler2D(model=unet, scheduler=ddpm_scheduler)

# Graph
fig, axes = plt.subplots(1, len(guidance_scales), figsize=(10 * len(guidance_scales), 10))

y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.int64).repeat_interleave(samples_per_class).to(
        device)

for idx, w in enumerate(guidance_scales):

    if model == "flow":
        # Setup ode and simulator
        ode = CFGVectorFieldODE(unet, guidance_scale=w)
        simulator = EulerSimulator(ode)

        # Sample initial conditions
        num_samples = y.shape[0]
        x0, _ = path.p_simple.sample(num_samples) # (num_samples, 1, 32, 32)

        # Simulate
        ts = torch.linspace(0,1,num_timesteps).view(1, -1, 1, 1, 1).expand(num_samples, -1, 1, 1, 1).to(device)
        x1 = simulator.simulate(x0, ts, y=y)

    elif model == "ddpm":

        xt = torch.randn(y.shape[0], 1, 32, 32, device=device)  # Start from pure noise

        # 4. Create the discrete timestep schedule for the sampler
        timesteps = torch.linspace(ddpm_scheduler.num_timesteps - 1, 0, num_timesteps, dtype=torch.long,
                                   device=device)

        # 5. The backward denoising loop
        for t in tqdm(timesteps, desc=f"DDIM Sampling (w={w})"):
            xt = sampler.step(xt, t.item(), guidance_scale=w, y=y)

        # The final xt is the clean image
        x1 = xt

    # Plot
    grid = make_grid(x1, nrow=samples_per_class, normalize=True, value_range=(-1,1))

    if len(guidance_scales) == 1:
        axes.imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
        axes.axis("off")
        axes.set_title(f"Guidance: $w={w:.1f}$", fontsize=25)
    else:
        axes[idx].imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
        axes[idx].axis("off")
        axes[idx].set_title(f"Guidance: $w={w:.1f}$", fontsize=25)

plt.tight_layout()
plt.show()
