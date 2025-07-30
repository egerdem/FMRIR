
from matplotlib import pyplot as plt
import torch
from torchvision import transforms
from torchvision.utils import make_grid
# import matplotlib
# matplotlib.use('MacOSX')  # Use 'TkAgg' for Linux, 'MacOSX' for macOS, 'Qt5Agg' for Windows
import os
import json
import time
import wandb

from fm_utils import (
    SpectrogramSampler, GaussianConditionalProbabilityPath, LinearAlpha,
    LinearBeta, CFGTrainer, SpecUNet
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Configuration ---
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
        "name": "SpecUNet",
        "channels": [32, 64, 128],
        "num_residual_layers": 2,
        "t_embed_dim": 40,
        "y_dim": 6,
        "y_embed_dim": 40
    },
    "training": {
        "num_iterations": 8,
        "batch_size": 250,
        "lr": 1e-3,
        "eta": 0.1
    },
    "experiments_dir": "experiments"
}

# --- Experiment Setup ---
timestamp = time.strftime("%Y%m%d-%H%M%S")
experiment_name = f"{config['model']['name']}_{timestamp}"
experiment_dir = os.path.join(config['experiments_dir'], experiment_name)
os.makedirs(experiment_dir, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(experiment_dir, "model.pt")
CONFIG_SAVE_PATH = os.path.join(experiment_dir, "config.json")

with open(CONFIG_SAVE_PATH, 'w') as f:
    json.dump(config, f, indent=4)
print(f"Experiment setup. Config saved to {CONFIG_SAVE_PATH}")

# --- Data Loading ---
data_cfg = config['data']

# ensures we only calculate normalization stats from data the model will be trained on.
# --- Instantiate Samplers for each split (only ONCE) ---
spec_train_sampler = SpectrogramSampler(
    data_path=data_cfg['data_dir'], mode='train', src_splits=data_cfg['src_splits']
).to(device)

spec_valid_sampler = SpectrogramSampler(
    data_path=data_cfg['data_dir'], mode='valid', src_splits=data_cfg['src_splits']
).to(device)

# --- Calculate stats from the single training sampler instance ---
spec_mean = spec_train_sampler.spectrograms.mean()
spec_std = spec_train_sampler.spectrograms.std()
print(f"\nCalculated Mean: {spec_mean:.4f}, Std: {spec_std:.4f} (from training set)")

# --- Define and apply the transform to the existing samplers ---
transform = transforms.Compose([
    transforms.Normalize((spec_mean,), (spec_std,)),
])
spec_train_sampler.transform = transform
spec_valid_sampler.transform = transform

sample_spec, _ = spec_train_sampler.sample(1)
spec_shape = list(sample_spec.shape[1:])

path = GaussianConditionalProbabilityPath(
    p_data=spec_train_sampler, #was [1, 32, 32], for mnist
    p_simple_shape=spec_shape,
    alpha=LinearAlpha(),
    beta=LinearBeta()
).to(device)


# --- Model and Trainer Initialization ---
model_cfg = config['model']
training_cfg = config['training']

spec_unet = SpecUNet(
    channels=model_cfg['channels'], # Same as MNIST version
    num_residual_layers=model_cfg['num_residual_layers'], # Same as MNIST version
    t_embed_dim=model_cfg['t_embed_dim'], # Same as MNIST version
    y_dim=model_cfg['y_dim'], # new: 6D coordinates for source and microphone positions
    y_embed_dim=model_cfg['y_embed_dim'], # Same as MNIST version
).to(device)

trainer = CFGTrainer(
    path=path,
    model=spec_unet,
    eta=training_cfg['eta'],
    y_dim=model_cfg['y_dim'],
)

# --- Training ---
print(f"\n--- Starting Training for experiment: {experiment_name} ---")
trainer.train(
    num_iterations=training_cfg['num_iterations'],
    device=device,
    lr=training_cfg['lr'],
    batch_size=training_cfg['batch_size'],
    validation_interval=1,
    valid_sampler=spec_valid_sampler,  # Pass the validation sampler**
    save_path=MODEL_SAVE_PATH,         # Pass the save path**
    config=config                      # Pass the config to be saved**
)

# --- Save the Model ---
# print(f"Saving model to {MODEL_SAVE_PATH}...")
# torch.save({
#     'model_state_dict': spec_unet.state_dict(),
#     'y_null': trainer.y_null,
#     'config': config # Save config with model for easy reference
# }, MODEL_SAVE_PATH)
# print("Model saved. You can now run inference using the model and config from the experiment directory.")
# print(f"Experiment directory: {experiment_dir}")

# **NEW: Finish the wandb run**
wandb.finish()
