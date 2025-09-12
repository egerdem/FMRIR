"""
Simple MNIST U-Net Inference Script
Loads trained model and runs inference using modules from the main training script.
"""

import torch
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

# Import all necessary classes from the main training script
from unet_mnist_lab3 import (
    device, path, MNISTUNet, CFGVectorFieldODE, EulerSimulator
)

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
unet = load_trained_model('trained_mnist_unet.pt')

# Play with these!
samples_per_class = 10
num_timesteps = 100
guidance_scales = [1.0, 3.0, 5.0]

# Graph
fig, axes = plt.subplots(1, len(guidance_scales), figsize=(10 * len(guidance_scales), 10))

for idx, w in enumerate(guidance_scales):
    # Setup ode and simulator
    ode = CFGVectorFieldODE(unet, guidance_scale=w)
    simulator = EulerSimulator(ode)

    # Sample initial conditions
    y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.int64).repeat_interleave(samples_per_class).to(device)
    num_samples = y.shape[0]
    x0, _ = path.p_simple.sample(num_samples) # (num_samples, 1, 32, 32)

    # Simulate
    ts = torch.linspace(0,1,num_timesteps).view(1, -1, 1, 1, 1).expand(num_samples, -1, 1, 1, 1).to(device)
    x1 = simulator.simulate(x0, ts, y=y)

    # Plot
    grid = make_grid(x1, nrow=samples_per_class, normalize=True, value_range=(-1,1))
    axes[idx].imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
    axes[idx].axis("off")
    axes[idx].set_title(f"Guidance: $w={w:.1f}$", fontsize=25)
plt.show()
