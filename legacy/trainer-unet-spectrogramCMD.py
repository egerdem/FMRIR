
import argparse
import os
import json
import time

import torch
from torchvision import transforms
import wandb

from fm_utils import (
    GaussianConditionalProbabilityPath, LinearAlpha,
    LinearBeta, CFGTrainer, SpecUNet, SpectrogramSampler
)

# --- Configuration ---
config = {
    "data": {
        "data_dir": "/home/eerdem/DATA/",
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

if __name__ == "__main__":

    # You can set this path manually or use argparse
    # resume_from_checkpoint = None  #

    parser = argparse.ArgumentParser(description="Train Flow Matching U-Net")
    parser.add_argument("--iterations", type=int, help="Number of training iterations")
    parser.add_argument("--batch_size", type=int, help="Training batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--data_dir", type=str, help="Path to the dataset")
    parser.add_argument("--exp_dir", type=str, help="Directory to save experiments")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    # --- Update config with command-line arguments if provided ---
    if args.iterations: config["training"]["num_iterations"] = args.iterations
    if args.batch_size: config["training"]["batch_size"] = args.batch_size
    if args.lr: config["training"]["lr"] = args.lr
    if args.data_dir: config["data"]["data_dir"] = args.data_dir
    if args.exp_dir: config["experiments_dir"] = args.exp_dir

    # --- Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Checkpoint and Experiment Directory Setup ---
    start_iteration = 0
    experiment_dir = ""

    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming training from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        start_iteration = checkpoint['iteration']
        config = checkpoint['config']  # Load the config from the checkpoint

        # Initialize wandb with the ID of the run you're resuming
        run_id = checkpoint.get('wandb_run_id') #was checkpoint['wandb_run_id']
        wandb.init(project="FM-RIR", id=run_id, resume="must", config=config)

    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        experiment_name = f"{config['model']['name']}_lr{config['training']['lr']}_bs{config['training']['batch_size']}_{timestamp}"
        experiment_dir = os.path.join(config['experiments_dir'], experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        # Initialize a new wandb run
        wandb.login(key="ec2cf1718868be26a8055412b556d952681ee0b6")
        run = wandb.init(project="FM-RIR", name=experiment_name, config=config)
        config['wandb_run_id'] = run.id  # Save run_id for resuming

    MODEL_SAVE_PATH = os.path.join(experiment_dir, "model.pt")
    CHECKPOINT_DIR = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

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

    if start_iteration > 0:
        spec_unet.load_state_dict(checkpoint['model_state_dict'])
        trainer.y_null.data = checkpoint['y_null'].to(device)

    # --- Training ---
    print(f"\n--- Starting Training for experiment: {experiment_name} ---")
    trainer.train(
        num_iterations=training_cfg['num_iterations'],
        device=device,
        lr=training_cfg['lr'],
        batch_size=training_cfg['batch_size'],
        valid_sampler=spec_valid_sampler,
        save_path=MODEL_SAVE_PATH,
        checkpoint_path=CHECKPOINT_DIR,
        checkpoint_interval=1000,  # Save a checkpoint every 1000 iterations
        start_iteration=start_iteration, # Start from 0 or the loaded iteration
        config=config
    )

    # --- Finalizing the Run ---
    # Log the best model as a wandb Artifact for easy access later
    if os.path.exists(MODEL_SAVE_PATH):
        print("Logging best model to W&B Artifacts...")
        best_model_artifact = wandb.Artifact(f"{wandb.run.name}-best-model", type="model")
        best_model_artifact.add_file(MODEL_SAVE_PATH)
        wandb.log_artifact(best_model_artifact)

    wandb.finish()
    print("Training complete and wandb run finished.")

    # --- Save the Model ---
    # print(f"Saving model to {MODEL_SAVE_PATH}...")
    # torch.save({
    #     'model_state_dict': spec_unet.state_dict(),
    #     'y_null': trainer.y_null,
    #     'config': config # Save config with model for easy reference
    # }, MODEL_SAVE_PATH)
    # print("Model saved. You can now run inference using the model and config from the experiment directory.")
    # print(f"Experiment directory: {experiment_dir}")

