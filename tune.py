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
def objective(trial: optuna.trial.Trial, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- A. Define the Hyperparameter Search Space ---
    # Optuna will suggest a value for each of these in every trial
    # lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    lr = 1e-4
    d_model = trial.suggest_categorical("d_model", [128, 256])
    nhead = trial.suggest_categorical("nhead", [4, 8])
    num_encoder_layers = trial.suggest_int("num_encoder_layers", 3, 5)
    # Dynamically define U-Net depth and channels
    # num_unet_levels = trial.suggest_int("num_unet_levels", 3, 4)
    # num_unet_levels = 3
    # channels = base_channels[:num_unet_levels]
    channels = [32, 64, 128]


    print("\n" + "=" * 50)
    print(f"--- Starting Trial #{trial.number} ---")
    print(f"  - Learning Rate: {lr:.2e}")
    print(f"  - U-Net Depth: Channels: {channels}")
    print(f"  - SetEncoder d_model: {d_model}")
    print(f"  - SetEncoder Layers: {num_encoder_layers}")
    print("=" * 50 + "\n")

    # --- B. Standard Setup (copied from your trainer script) ---
    # This part is the same as your main() function, but uses the suggested params

    # Create experiment directory and name for this trial
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    trial_name = f"{args.model_name}_trial{trial.number}_{timestamp}_iter{args.num_iterations}"
    experiment_dir = os.path.join(args.experiments_dir, trial_name)
    os.makedirs(experiment_dir, exist_ok=True)

    config = {
        "data": {"data_dir": args.data_dir,
                 "src_splits": {
                     "train": [0, 820],
                     "valid": [820, 922]}},
        "model": {
            "name": args.model_name,
            "channels": channels,
            "d_model": d_model,
            "nhead": nhead,
            "num_encoder_layers": num_encoder_layers,
            "freq_up_to": args.freq_up_to},
        "training": {
            "num_iterations": args.num_iterations,
            "batch_size": args.batch_size,
            "lr": lr,
            "M_range": args.M_range,
            "eta": args.eta,
            "sigma": args.sigma,
            "validation_interval": args.validation_interval
        },
        "experiments_dir": args.experiments_dir
    }

    # Initialize wandb for this trial if enabled
    if args.wandb:
        import wandb
        wandb.login(key=args.wandb_key)
        run = wandb.init(project="FM-RIR-3D-TUNING", name=trial_name, config=config)
        config['wandb_run_id'] = run.id

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
    MODEL_SAVE_PATH = os.path.join(experiment_dir, "model_tuning.pt")
    CHECKPOINT_DIR = os.path.join(experiment_dir, "checkpoints_tuning")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    best_val_loss = trainer.train(
        num_iterations=training_cfg['num_iterations'],
        device=device,
        lr=training_cfg['lr'],
        batch_size=training_cfg['batch_size'],
        valid_sampler=valid_sampler,
        save_path=MODEL_SAVE_PATH,
        checkpoint_path=CHECKPOINT_DIR,
        validation_interval=training_cfg['validation_interval'],
        checkpoint_interval=training_cfg['num_iterations'],  # Don't save intermediate checkpoints
        start_iteration=0,
        config=config,
        resume_checkpoint_state=None
    )

    # Clean up wandb run for this trial
    if args.wandb:
        wandb.finish()

    # --- D. Return the value for Optuna to minimize ---
    return best_val_loss


# --- 2. Create and Run the Study ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D ATF Hyperparameter Tuning")
    
    # --- WandB ---
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--wandb_key', type=str, default="ec2cf1718868be26a8055412b556d952681ee0b6")

    # --- Data ---
    parser.add_argument('--data_dir', type=str, default="ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200/")

    # --- Model ---
    parser.add_argument('--model_name', default="TUNA_step1_ATF-3D-CrossAttn-UNet", type=str)
    # parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads.')
    parser.add_argument('--freq_up_to', type=int, default=20, help='Use only the first N frequency channels')

    # --- Training ---
    parser.add_argument('--num_iterations', type=int, default=800000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--M_range', type=lambda s: [int(item) for item in s.split(',')], default=[5, 50])
    parser.add_argument('--eta', type=float, default=0.1, help='Probability for CFG dropout.')
    parser.add_argument('--sigma', type=float, default=1e-5, help='Sigma for noise in the path.')
    parser.add_argument('--validation_interval', type=int, default=1000)

    # --- Tuning ---
    parser.add_argument('--n_trials', type=int, default=12, help='Number of optimization trials')
    
    # --- Paths ---
    parser.add_argument('--experiments_dir', type=str, default="~/FMRIR_experiments")

    args = parser.parse_args()
    
    # Create study
    study = optuna.create_study(direction="minimize")
    
    # Run optimization using lambda to pass args
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

    print("\n--- OPTIMIZATION FINISHED ---")
    print(f"Best validation loss: {study.best_value}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")