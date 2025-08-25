import torch
from torchvision import transforms
import os
import json
import time
import wandb
import argparse

from fm_utils import (ATFSliceSampler, FreqConditionalATFSampler, GaussianConditionalProbabilityPath, LinearAlpha,
                      LinearBeta, ATFInpaintingTrainer, ATFUNet)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Overwrite config with args
    config = {
        "data": {
            "data_dir": args.data_dir,
            "src_splits": {
                "train": [0, 820],
                "valid": [820, 922],
                "test": [922, 1024],
                "all": [0, 1024]}
        },
        "model": {
            "name": args.model_name,
            "channels": args.channels,
            "num_residual_layers": args.num_residual_layers,
            "t_embed_dim": args.t_embed_dim,
            "y_embed_dim": args.y_embed_dim,
            "freq_up_to": args.freq_up_to
        },
        "training": {
            "num_iterations": args.num_iterations,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "M": args.M,
            "eta": args.eta,
            "sigma": args.sigma,
            "model_mode": args.model_mode,
            "flag_gaussian_mask": args.flag_gaussian_mask,
            "validation_interval": args.validation_interval
        },
        "experiments_dir": args.experiments_dir,
        "project_root": args.project_root
    }

    start_iteration = 0
    resume_checkpoint_state = None
    experiment_dir = None
    if args.resume_from_checkpoint:
        print(f"RESUMING training from: {args.resume_from_checkpoint}")
        if os.path.exists(args.resume_from_checkpoint):
            # Load once; pass state into trainer to avoid duplicate disk reads
            resume_checkpoint_state = torch.load(args.resume_from_checkpoint, map_location=device)

            # Determine experiment directory
            parent = os.path.dirname(args.resume_from_checkpoint)
            if os.path.basename(parent) == 'checkpoints':
                experiment_dir = os.path.dirname(parent)
            else:
                experiment_dir = parent

            experiment_name = os.path.basename(experiment_dir)

            # W&B resume (if available)
            wandb.login(key=args.wandb_key)
            run_id = resume_checkpoint_state.get('wandb_run_id') or resume_checkpoint_state.get('config', {}).get('wandb_run_id')
            if args.wandb:
                if run_id:
                    wandb.init(project="FM-RIR", id=run_id, resume="allow", config=config)
                    print(f"Resuming/attaching to wandb run ID: {run_id}")
                else:
                    wandb.init(project="FM-RIR", resume="allow", config=config)
        else:
            print(f"⚠️  Warning: resume path does not exist: {args.resume_from_checkpoint}")


    if experiment_dir is None:
        # --- Experiment Setup ---
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        experiment_name = f"{config['model']['name']}_{timestamp}_iter{str(config['training']['num_iterations'])}"
        experiment_dir = os.path.join(config['experiments_dir'], experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        print("\n NEW EXPERIMENT\n")

        # Initialize a new wandb run
        if args.wandb:
            wandb.login(key=args.wandb_key)
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

    if args.model_mode == "freq_cond":

        print("Using Frequency-Conditional model and sampler.")
        atf_train_sampler = FreqConditionalATFSampler(
            data_path=data_cfg['data_dir'], mode='train', src_splits=data_cfg['src_splits'], freq_up_to=args.freq_up_to
        ).to(device)
        atf_valid_sampler = FreqConditionalATFSampler(
            data_path=data_cfg['data_dir'], mode='valid', src_splits=data_cfg['src_splits'], freq_up_to=args.freq_up_to
        ).to(device)

        # For this mode, the model input has 2 channels (mag + mask) and y_dim is 5
        input_channels = 2
        output_channels = 1
        y_dim = 5

    else:  # "spatial" mode
        atf_train_sampler = ATFSliceSampler(
        data_path=data_cfg['data_dir'], mode='train', src_splits=data_cfg['src_splits'],
        freq_up_to=args.freq_up_to).to(device)

        atf_valid_sampler = ATFSliceSampler(
            data_path=data_cfg['data_dir'], mode='valid', src_splits=data_cfg['src_splits'],
            freq_up_to=args.freq_up_to).to(device)

        # For this mode, input channels = freqs+1 and y_dim is 4
        input_channels = args.freq_up_to + 1
        output_channels = args.freq_up_to
        y_dim = 4

    spec_mean = atf_train_sampler.slices.mean()
    spec_std = atf_train_sampler.slices.std()
    print(f"\nCalculated Mean: {spec_mean:.4f}, Std: {spec_std:.4f} (from training set)")

    padding = (0, 0, 1, 1)

    transform = transforms.Compose([
        transforms.Pad(padding, padding_mode='reflect'),
        transforms.Normalize((spec_mean,), (spec_std,)),
    ])

    atf_train_sampler.transform = transform
    atf_valid_sampler.transform = transform

    # atf_valid_sampler.plot()

    sample_spec, _ = atf_train_sampler.sample(1)
    atf_shape = list(sample_spec.shape[1:])
    print(f"ATF Slice shape for Path: {atf_shape}")
    # freq_channels = atf_shape[0]
    # input_channels = freq_channels + 1  # +1 for mask channel
    # output_channels = freq_channels + 1  # predict per-frequency plus mask output channel

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
        channels=model_cfg['channels'],
        num_residual_layers=model_cfg['num_residual_layers'],
        t_embed_dim=model_cfg['t_embed_dim'],
        y_dim=y_dim,
        y_embed_dim=model_cfg['y_embed_dim'],
        input_channels=input_channels,
        output_channels=output_channels,
    ).to(device)

    trainer = ATFInpaintingTrainer(
        path=path,
        model=atf_unet,
        eta=training_cfg['eta'],
        M=training_cfg['M'],
        y_dim=y_dim,
        sigma=training_cfg['sigma'],
        model_mode=args.model_mode,
        flag_gaussian_mask=args.flag_gaussian_mask
    )

    # NOTE: Model/optimizer/y_null restoration handled inside trainer via resume_checkpoint_state

    # --- Training ---
    print(f"\n--- Starting Training for experiment: {experiment_name} ---")
    trainer.train(
        num_iterations=training_cfg['num_iterations'],
        device=device,
        lr=training_cfg['lr'],
        batch_size=training_cfg['batch_size'],
        valid_sampler=atf_valid_sampler,
        save_path=MODEL_SAVE_PATH,
        checkpoint_path=CHECKPOINT_DIR,
        checkpoint_interval=args.checkpoint_interval,
        validation_interval=training_cfg['validation_interval'],
        start_iteration=start_iteration,
        config=config,
        resume_checkpoint_state=resume_checkpoint_state
    )

    # --- Finalizing the Run ---
    if args.wandb and os.path.exists(MODEL_SAVE_PATH):
        print("Logging best model to W&B Artifacts...")
        best_model_artifact = wandb.Artifact(f"{wandb.run.name}-best-model", type="model")
        best_model_artifact.add_file(MODEL_SAVE_PATH)
        wandb.log_artifact(best_model_artifact)
        wandb.finish()
    print("Training complete and wandb run finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="U-Net ATF Trainer CMD")

    # --- reprendre la formation ---
    parser.add_argument('--resume_from_checkpoint', type=str, default="/Users/ege/Projects/FMRIR/artifacts/ATFUNet_M5_holeloss_GaussFalse_LR5e3_n05_20250820-173718_iter500000/checkpoints/ckpt_final_50660.pt", help='Path to a checkpoint to resume training from.')
    parser.add_argument('--resume_from_iteration', type=int, help='Iteration to resume from if not in checkpoint.')
    parser.add_argument('--resume_run_id', type=str, help='WandB run ID to resume if not in checkpoint.')

    # --- WandB ---
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True, help='Enable or disable wandb logging.')
    parser.add_argument('--wandb_key', type=str, default="ec2cf1718868be26a8055412b556d952681ee0b6", help='WandB API key.')

    # --- Data ---
    parser.add_argument('--data_dir', type=str, default="ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200/", help='Directory of the data.')
    # --- Model ---
    parser.add_argument('--model_name', default="TRRRIAL", type=str, help='Name of the model.')

    parser.add_argument('--channels', type=lambda s: [int(item) for item in s.split(',')], default=[32, 64, 128], help='List of channels for the model, comma-separated.')
    parser.add_argument('--num_residual_layers', type=int, default=2, help='Number of residual layers.')
    parser.add_argument('--t_embed_dim', type=int, default=40, help='t embedding dimension.')
    # parser.add_argument('--y_dim', type=int, default=4, help='y dimension.')
    parser.add_argument('--y_embed_dim', type=int, default=40, help='y embedding dimension.')
    parser.add_argument('--freq_up_to', type=int, default=30, help='Use only the first N frequency channels; model uses N+1 channels with mask.')
    parser.add_argument("--model_mode", type=str, default="spatial",
                        choices=["spatial", "freq_cond"],
                        help="Model mode: 'spatial' for original model, 'freq_cond' for frequency-conditional model.")

    # --- Training ---
    parser.add_argument('--num_iterations', type=int, default=100000, help='Number of training iterations.')
    parser.add_argument('--batch_size', type=int, default=250, help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--M', type=int, default=30, help='Number of observation points.')
    parser.add_argument('--eta', type=float, default=0.1, help='Eta for inpainting.')
    parser.add_argument('--flag_gaussian_mask', type=bool, default=False)
    parser.add_argument('--sigma', type=float, default=0.0, help='Sigma for noise.')
    parser.add_argument('--checkpoint_interval', type=int, default=100000, help='Save a checkpoint every N iterations.')
    parser.add_argument('--validation_interval', type=int, default=20, help='Save a checkpoint every N iterations.')

    # --- Paths ---
    parser.add_argument('--experiments_dir', type=str, default="experiments", help='Directory for experiments.')
    # parser.add_argument('--project_root', type=str, default="/Users/ege/Projects/FMRIR", help='Project root directory.')
    parser.add_argument('--project_root', type=str, default="None", help='Project root directory.')

    args = parser.parse_args()
    main(args)

