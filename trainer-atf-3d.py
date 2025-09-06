import torch
from torchvision import transforms
import os
import json
import time
import wandb
import argparse

from fm_utils import (ATF3DSampler, GaussianConditionalProbabilityPath,
    LinearAlpha, LinearBeta,
    SetEncoder, CrossAttentionUNet3D,ATF3DTrainer
                      )

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 1. Initial Config from Arguments ---
    # This creates a baseline config that can be used immediately
    config = {
        "data": {"data_dir": args.data_dir,
                 "src_splits": {"train": [0, 820], "valid": [820, 922], "test": [922, 1024]}},
        "model": {"name": args.model_name, "channels": args.channels, "d_model": args.d_model, "nhead": args.nhead,
                  "num_encoder_layers": args.num_encoder_layers, "freq_up_to": args.freq_up_to
                  "architecture_version": "v2_residual_context" },
        "training": {"num_iterations": args.num_iterations, "batch_size": args.batch_size, "lr": args.lr,
                     "warmup_iterations": args.warmup_iterations, "min_lr": args.min_lr,
                     "M_range": args.M_range, "eta": args.eta, "sigma": args.sigma,
                     "validation_interval": args.validation_interval},
        "experiments_dir": args.experiments_dir
    }

    # --- 2. Handle Resuming ---
    start_iteration = 0
    resume_checkpoint_state = None
    experiment_dir = None
    experiment_name = ""

    if args.resume_from_checkpoint:
        if os.path.exists(args.resume_from_checkpoint):
            print(f"RESUMING training from: {args.resume_from_checkpoint}")
            resume_checkpoint_state = torch.load(args.resume_from_checkpoint, map_location=device)

            # Load the old config, but then immediately update it with new args
            loaded_config = resume_checkpoint_state.get('config', {})
            loaded_config.update(config)
            config = loaded_config

            start_iteration = resume_checkpoint_state.get('iteration', 0)
            print(f"Resuming from iteration {start_iteration}")

            parent = os.path.dirname(args.resume_from_checkpoint)
            experiment_dir = os.path.dirname(parent) if os.path.basename(parent) == 'checkpoints' else parent
            experiment_name = os.path.basename(experiment_dir)

             # Initialize WandB if enabled and resume run
            if args.wandb:
                wandb.login(key=args.wandb_key)
                run_id = resume_checkpoint_state.get('wandb_run_id')
                wandb.init(project="FM-RIR-3D", id=run_id, resume="allow", config=config)
                print(f"Resuming W&B run ID: {run_id}")
        else:
            print(f"⚠️ Warning: resume path does not exist: {args.resume_from_checkpoint}")

        # --- Configuration ---
        # Create a config dict from args to save with the model
        config = {
            "data": {
                "data_dir": args.data_dir,
                "src_splits": {
                    "train": [0, 820],
                    "valid": [820, 922],
                    "test": [922, 1024]
                }
            },
            "model": {
                "name": args.model_name,
                "channels": args.channels,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_encoder_layers": args.num_encoder_layers,
                "freq_up_to": args.freq_up_to
            },
            "training": {
                "num_iterations": args.num_iterations,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "warmup_iterations": args.warmup_iterations,
                "min_lr": args.min_lr,
                "M_range": args.M_range,
                "eta": args.eta,
                "sigma": args.sigma,
                "validation_interval": args.validation_interval
            },
            "experiments_dir": args.experiments_dir
        }

    if experiment_dir is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        experiment_name = f"{args.model_name}_{timestamp}_iter{args.num_iterations}"
        experiment_dir = os.path.join(args.experiments_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        print("\n--- NEW EXPERIMENT ---")

        if args.wandb:
            wandb.login(key=args.wandb_key)
            run = wandb.init(project="FM-RIR-3D", name=experiment_name, config=config)
            config['wandb_run_id'] = run.id

        with open(os.path.join(experiment_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Experiment setup. Config saved to {experiment_dir}")

    MODEL_SAVE_PATH = os.path.join(experiment_dir, "model.pt")
    CHECKPOINT_DIR = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # --- Data Loading ---
    data_cfg = config['data']
    model_cfg = config['model']
    training_cfg = config['training']

    # 1. Create the training sampler. It will calculate and apply its own normalization.
    print("--- Loading Training Data ---")
    atf_train_sampler = ATF3DSampler(
        data_path=data_cfg['data_dir'],
        mode='train',
        src_splits=data_cfg['src_splits'],
        freq_up_to=model_cfg['freq_up_to'],
        normalize=True
    )

    # 2. Create the validation sampler, but load the data RAW (normalize=False).
    print("\n--- Loading Validation Data ---")
    atf_valid_sampler = ATF3DSampler(
        data_path=data_cfg['data_dir'],
        mode='valid',
        src_splits=data_cfg['src_splits'],
        freq_up_to=model_cfg['freq_up_to'],
        normalize=False
    )

    # 3. Manually apply the TRAINING stats to the VALIDATION data.
    print("Normalizing validation data using training set statistics...")
    atf_valid_sampler.cubes = (atf_valid_sampler.cubes - atf_train_sampler.mean) / (atf_train_sampler.std + 1e-8)

    # 4. Store the stats on the validation sampler object for consistency.
    atf_valid_sampler.mean = atf_train_sampler.mean
    atf_valid_sampler.std = atf_train_sampler.std

    # --- Model and Trainer Initialization ---

    # Get cube shape from the sampler for the probability path
    cube_shape = atf_train_sampler.cubes.shape[1:]

    path = GaussianConditionalProbabilityPath(
        p_data=atf_train_sampler,
        p_simple_shape=list(cube_shape),
        alpha=LinearAlpha(),
        beta=LinearBeta()
    ).to(device)

    # Instantiate the two models
    set_encoder = SetEncoder(
        num_freqs=cube_shape[0],  # 64
        d_model=model_cfg['d_model'],
        nhead=model_cfg['nhead'],
        num_layers=model_cfg['num_encoder_layers']
    ).to(device)

    unet_3d = CrossAttentionUNet3D(
        in_channels=cube_shape[0],
        out_channels=cube_shape[0],
        channels=model_cfg['channels'],
        d_model=model_cfg['d_model'],
        nhead=model_cfg['nhead']
    ).to(device)

    trainer = ATF3DTrainer(
        path=path,
        model=unet_3d,
        set_encoder=set_encoder,
        eta=training_cfg['eta'],
        M_range=training_cfg['M_range'],
        sigma=training_cfg['sigma'],
        grid_xyz=atf_train_sampler.grid_xyz
    )

    training_cfg['warmup_iterations'] = args.warmup_iterations
    training_cfg['min_lr'] = args.min_lr

    # --- Training ---
    print(f"\n--- Starting Training for experiment: {experiment_name} ---")
    trainer.train(
        num_iterations=training_cfg['num_iterations'],
        device=device,
        lr=training_cfg['lr'],
        warmup_iterations=training_cfg['warmup_iterations'],
        min_lr=training_cfg['min_lr'],
        batch_size=training_cfg['batch_size'],
        valid_sampler=atf_valid_sampler,
        save_path=MODEL_SAVE_PATH,
        checkpoint_path=CHECKPOINT_DIR,
        checkpoint_interval=args.checkpoint_interval,
        validation_interval=training_cfg['validation_interval'],
        start_iteration=start_iteration,
        config=config,
        resume_checkpoint_state=resume_checkpoint_state  #
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="3D ATF Trainer CMD")

    # --- Resuming ---
    parser.add_argument('--resume_from_checkpoint', type=str, help='Path to a checkpoint to resume from.')

    # --- WandB ---
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--wandb_key', type=str, default="ec2cf1718868be26a8055412b556d952681ee0b6")

    # --- Data ---
    parser.add_argument('--data_dir', type=str, default="ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200/")

    # --- Model ---
    parser.add_argument('--model_name', default="ZZZATF-3D-CrossAttn-UNet", type=str)
    parser.add_argument('--channels', type=lambda s: [int(item) for item in s.split(',')], default=[32, 64, 128])
    parser.add_argument('--d_model', type=int, default=512, help='Dimension for tokens and context.')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads.')
    parser.add_argument('--num_encoder_layers', type=int, default=3, help='Layers in the SetEncoder.')

    # --- Training ---
    parser.add_argument('--num_iterations', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)  # NOTE: Must be small for 3D models
    parser.add_argument('--lr', type=float, default=1e-4, help="now it is peak learning rate after warm-up.")
    parser.add_argument('--warmup_iterations', type=int, default=5000, help="Number of iterations for linear LR warm-up.")
    parser.add_argument('--min_lr', type=float, default=1e-7,
                        help="The minimum learning rate at the end of the cosine decay.")
    parser.add_argument('--M_range', type=lambda s: [int(item) for item in s.split(',')], default=[5, 50])
    parser.add_argument('--freq_up_to', type=int, default=20, help='Use only the first N frequency channels')
    parser.add_argument('--eta', type=float, help='Probability for CFG dropout.', default=0.1)
    parser.add_argument('--sigma', type=float, help='Sigma for noise in the path.', default=0)
    parser.add_argument('--checkpoint_interval', type=int, default=20000)
    parser.add_argument('--validation_interval', type=int, default=1000)

    # --- Paths ---
    parser.add_argument('--experiments_dir', type=str, default="experiments_3d")

    args = parser.parse_args()
    main(args)