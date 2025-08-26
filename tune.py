import optuna
import torch
import os
import argparse

# Import all the necessary classes from your project
from fm_utils import (
    ATF3DSampler,
    GaussianConditionalProbabilityPath,
    LinearAlpha,
    LinearBeta,
    SetEncoder,
    CrossAttentionUNet3D,
    ATF3DTrainer
)


# --- 1. Define the Objective Function ---
# This function wraps your entire training process
def objective(trial: optuna.trial.Trial):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- A. Define the Hyperparameter Search Space ---
    # Optuna will suggest a value for each of these in every trial
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    d_model = trial.suggest_categorical("d_model", [128, 256, 512])
    num_encoder_layers = trial.suggest_int("num_encoder_layers", 2, 4)

    # Dynamically define U-Net depth and channels
    num_unet_levels = trial.suggest_int("num_unet_levels", 3, 4)
    base_channels = [32, 64, 128, 256, 512]
    channels = base_channels[:num_unet_levels]

    # --- B. Standard Setup (copied from your trainer script) ---
    # This part is the same as your main() function, but uses the suggested params

    config = {
        "data": {"data_dir": "ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200/",
                 "src_splits": {"train": [0, 820], "valid": [820, 922]}},
        "model": {"channels": channels, "d_model": d_model, "nhead": 4, "num_encoder_layers": num_encoder_layers,
                  "freq_up_to": 20},
        "training": {"num_iterations": 50000, "batch_size": 4, "lr": lr, "M_range": [10, 50], "eta": 0.1, "sigma": 1e-5,
                     "validation_interval": 1000}
    }

    # Data Loading
    data_cfg = config['data']
    model_cfg = config['model']
    training_cfg = config['training']

    train_sampler = ATF3DSampler(data_path=data_cfg['data_dir'], mode='train', src_splits=data_cfg['src_splits'],
                                 freq_up_to=model_cfg['freq_up_to'], normalize=True)
    valid_sampler = ATF3DSampler(data_path=data_cfg['data_dir'], mode='valid', src_splits=data_cfg['src_splits'],
                                 freq_up_to=model_cfg['freq_up_to'], normalize=False)
    valid_sampler.cubes = (valid_sampler.cubes - train_sampler.mean) / (train_sampler.std + 1e-8)

    # Model and Trainer Initialization
    cube_shape = train_sampler.cubes.shape[1:]
    path = GaussianConditionalProbabilityPath(p_data=train_sampler, p_simple_shape=list(cube_shape),
                                              alpha=LinearAlpha(), beta=LinearBeta()).to(device)

    set_encoder = SetEncoder(num_freqs=cube_shape[0], d_model=model_cfg['d_model'], nhead=model_cfg['nhead'],
                             num_layers=model_cfg['num_encoder_layers']).to(device)
    unet_3d = CrossAttentionUNet3D(in_channels=cube_shape[0], out_channels=cube_shape[0],
                                   channels=model_cfg['channels'], d_model=model_cfg['d_model'],
                                   nhead=model_cfg['nhead']).to(device)

    trainer = ATF3DTrainer(path=path, model=unet_3d, set_encoder=set_encoder, eta=training_cfg['eta'],
                           M_range=training_cfg['M_range'], sigma=training_cfg['sigma'],
                           grid_xyz=train_sampler.grid_xyz)

    # --- C. Run Training and get the final score ---
    best_val_loss = trainer.train(
        num_iterations=training_cfg['num_iterations'],
        device=device,
        lr=training_cfg['lr'],
        batch_size=training_cfg['batch_size'],
        valid_sampler=valid_sampler,
        save_path="tuning_model.pt",  # Use a temporary path
        checkpoint_path="tuning_checkpoints",
        validation_interval=training_cfg['validation_interval'],
        checkpoint_interval=training_cfg['num_iterations']  # Don't save intermediate checkpoints
    )

    # --- D. Return the value for Optuna to minimize ---
    return best_val_loss


# --- 2. Create and Run the Study ---
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)  # Run 20 different hyperparameter combinations

    print("\n--- OPTIMIZATION FINISHED ---")
    print(f"Best validation loss: {study.best_value}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")