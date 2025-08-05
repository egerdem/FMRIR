import torch
from torchvision import transforms
import os
import json
import time
import wandb

from fm_utils import (ATFSliceSampler, GaussianConditionalProbabilityPath, LinearAlpha,
                      LinearBeta, ATFInpaintingTrainer, ATFUNet)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandinit = False
resume_from_checkpoint = None
# resume_from_checkpoint = "/Users/ege/Projects/FMRIR/experiments/SpecUNet_20250804-140641/checkpoints/ckpt_10.pt"

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
        "name": "ATFUNet",
        "channels": [32, 64, 128],
        "num_residual_layers": 2,
        "t_embed_dim": 40,
        "y_dim": 4,
        "y_embed_dim": 40
    },
    "training": {
        "num_iterations": 1000,
        "batch_size": 250,
        "lr": 1e-3,
        "eta": 0.1
    },
    "experiments_dir": "experiments",
    "project_root": "/Users/ege/Projects/FMRIR"
}

start_iteration = 0
if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
    checkpoint_dir = os.path.dirname(resume_from_checkpoint)
    experiment_dir = os.path.dirname(checkpoint_dir)
    experiment_name = os.path.basename(experiment_dir)

    print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
    checkpoint = torch.load(resume_from_checkpoint, map_location=device)
    start_iteration = checkpoint['iteration']

    # Initialize wandb with the ID of the run you're resuming
    wandb.login(key="ec2cf1718868be26a8055412b556d952681ee0b6")
    run_id = checkpoint['wandb_run_id']
    wandb.init(project="FM-RIR", id=run_id, resume="must", config=config)

else:
    # --- Experiment Setup ---
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    experiment_name = f"{config['model']['name']}_{timestamp}_iter{str(config['training']['num_iterations'])}"
    experiment_dir = os.path.join(config['experiments_dir'], experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Initialize a new wandb run
    if wandinit:
        wandb.login(key="ec2cf1718868be26a8055412b556d952681ee0b6")
        run = wandb.init(project="FM-RIR", name=experiment_name, config=config)
        config['wandb_run_id'] = run.id

    CONFIG_SAVE_PATH = os.path.join(experiment_dir, "config.json")

    with open(CONFIG_SAVE_PATH, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Experiment setup. Config saved to {CONFIG_SAVE_PATH}")

MODEL_SAVE_PATH = os.path.join(experiment_dir, "model.pt")
CHECKPOINT_DIR = os.path.join(experiment_dir, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Data Loading ---
data_cfg = config['data']

# we calculate normalization stats from data the model will be trained on.
# --- Instantiate Samplers for each split (only ONCE) ---

atf_train_sampler = ATFSliceSampler(
    data_path=data_cfg['data_dir'], mode='train', src_splits=data_cfg['src_splits']).to(device)

atf_valid_sampler = ATFSliceSampler(
    data_path=data_cfg['data_dir'], mode='valid', src_splits=data_cfg['src_splits']).to(device)

# --- Calculate stats from the single training sampler instance ---
spec_mean = atf_train_sampler.slices.mean()
spec_std = atf_train_sampler.slices.std()
print(f"\nCalculated Mean: {spec_mean:.4f}, Std: {spec_std:.4f} (from training set)")

# --- Define and apply the transform to the existing samplers ---
# transform = transforms.Compose([
#     transforms.Normalize((spec_mean,), (spec_std,)),
# ])

# --- Define and apply the transform to the existing samplers ---
# We pad the 11x11 grid to 12x12 according to the torch documentation order:  left, top, right and bottom
padding = (0, 0, 1, 1) # right col and bottom row padded!

transform = transforms.Compose([
    transforms.Pad(padding, padding_mode='reflect'),
    transforms.Normalize((spec_mean,), (spec_std,)),
])

atf_train_sampler.transform = transform
atf_valid_sampler.transform = transform

atf_valid_sampler.plot()

sample_spec, _ = atf_train_sampler.sample(1) # The sample_spec will now have shape (1, 64, 11, 11)
# The shape for the path object should be [C, H, W]
atf_shape = list(sample_spec.shape[1:]) # This will be [64, 11, 11]
print(f"ATF Slice shape for Path: {atf_shape}")

path = GaussianConditionalProbabilityPath(
    p_data = atf_train_sampler,
    p_simple_shape = atf_shape,
    alpha = LinearAlpha(),
    beta = LinearBeta()
).to(device)

# --- Model and Trainer Initialization ---
model_cfg = config['model']
training_cfg = config['training']

atf_unet = ATFUNet(
    channels=model_cfg['channels'],  # Same as MNIST version
    num_residual_layers=model_cfg['num_residual_layers'],  # Same as MNIST version
    t_embed_dim=model_cfg['t_embed_dim'],  # Same as MNIST version
    y_dim=model_cfg['y_dim'],  # new: 6D coordinates for source and microphone positions
    y_embed_dim=model_cfg['y_embed_dim'],  # Same as MNIST version
).to(device)

trainer = ATFInpaintingTrainer(
    path=path,
    model=atf_unet,
    eta=training_cfg['eta'],
    M = 5, #Number of observation points / mic recordings
    y_dim=model_cfg['y_dim'],
)

if start_iteration > 0:
    atf_unet.load_state_dict(checkpoint['model_state_dict'])
    trainer.y_null.data = checkpoint['y_null'].to(device)

# --- Visualize the masking ---
trainer.visualize_masking(crop=True, sample_idx=15, freq_idx=20)

# --- Training ---
# print(f"\n--- Starting Training for experiment: {experiment_name} ---")
# trainer.train(
#     num_iterations=training_cfg['num_iterations'],
#     device=device,
#     lr=training_cfg['lr'],
#     batch_size=training_cfg['batch_size'],
#     valid_sampler=atf_valid_sampler,
#     save_path=MODEL_SAVE_PATH,
#     checkpoint_path=CHECKPOINT_DIR,
#     checkpoint_interval=5,  # Save a checkpoint every 1000 iterations
#     start_iteration=start_iteration,  # Start from 0 or the loaded iteration
#     config=config
# )
#
# # --- Finalizing the Run ---
# # Log the best model as a wandb Artifact for easy access later
# if os.path.exists(MODEL_SAVE_PATH):
#     print("Logging best model to W&B Artifacts...")
#     best_model_artifact = wandb.Artifact(f"{wandb.run.name}-best-model", type="model")
#     best_model_artifact.add_file(MODEL_SAVE_PATH)
#     wandb.log_artifact(best_model_artifact)
#
# wandb.finish()
# print("Training complete and wandb run finished.")