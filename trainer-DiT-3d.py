import torch
from torchvision import transforms
import os
import json
import time
import wandb
import argparse
from tqdm import tqdm

from fm_utils import (model_factory,
                      ATF3DSampler, GaussianConditionalProbabilityPath,
                      LinearAlpha, LinearBeta, DiTTrainer3D, DiffusionTransformer3D
                                          )


def calculate_and_cache_coord_stats(train_sampler, cache_path="coord_stats.pt"):
    """
    Calculates and caches the mean and std of relative coordinates for the training set.
    """
    if os.path.exists(cache_path):
        print(f"Loading cached coordinate stats from {cache_path}")
        stats = torch.load(cache_path)
        return stats['mean'], stats['std']

    print("Calculating coordinate stats for the first time... (this may take a moment)")
    all_rel_coords = []

    # Iterate through all training sources to get all possible relative coordinates
    for i in tqdm(range(len(train_sampler.source_coords)), desc="Calculating Stats"):
        src_xyz = train_sampler.source_coords[i].unsqueeze(0)
        rel_coords = train_sampler.grid_xyz - src_xyz  # Shape [1331, 3]
        all_rel_coords.append(rel_coords)

    # Concatenate all relative coordinates into a single large tensor
    full_coords_tensor = torch.cat(all_rel_coords, dim=0)

    # Calculate mean and std along the sample dimension (dim=0)
    coord_mean = full_coords_tensor.mean(dim=0)
    coord_std = full_coords_tensor.std(dim=0)

    # Cache the results for future runs
    print(f"Saving coordinate stats to {cache_path}")
    torch.save({'mean': coord_mean, 'std': coord_std}, cache_path)

    return coord_mean, coord_std

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 1. Initial Config from Arguments ---
    # This creates a baseline config that can be used immediately
    config = {
        "data": {"data_dir": args.data_dir,
                 "src_splits": {"train": [0, 820], "valid": [820, 922], "test": [922, 1024]}},
        "model": {"name": args.model_name, "d_model": args.d_model, "nhead": args.nhead,
                  "num_encoder_layers": args.num_encoder_layers, "freq_up_to": args.freq_up_to,
                  "architecture_version": args.version, "setencoder_version": args.setencoder_version,
                  "patch_size": args.patch_size, "dit_depth": args.dit_depth},
        "training": {"num_iterations": args.num_iterations, "batch_size": args.batch_size, "lr": args.lr,
                     "warmup_iterations": args.warmup_iterations, "min_lr": args.min_lr,
                     "M_range": args.M_range, "eta": args.eta, "sigma": args.sigma, "loss_type": args.loss_type,
                     "validation_interval": args.validation_interval},
        "experiments_dir": args.experiments_dir
    }

    start_iteration = 0

    # --- 2. Handle Resuming ---
    experiment_dir = None
    experiment_name = ""

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

    # ### <<< NEW: Calculate (or load) coordinate statistics
    # The cache file will be created in the same directory you run the script from.
    coord_mean, coord_std = calculate_and_cache_coord_stats(atf_train_sampler)
    print(f"Using Coordinate Stats -> Mean: {coord_mean.numpy()}, Std: {coord_std.numpy()}")

    # --- Model and Trainer Initialization ---

    # Get cube shape from the sampler for the probability path
    cube_shape = atf_train_sampler.cubes.shape[1:]

    path = GaussianConditionalProbabilityPath(
        p_data=atf_train_sampler,
        p_simple_shape=list(cube_shape),
        alpha=LinearAlpha(),
        beta=LinearBeta()
    ).to(device)

    set_encoder, dit_net = model_factory(config, device)

    trainer = DiTTrainer3D(
        path=path,
        model=dit_net,  # model is the DiffusionTransformer3D
        set_encoder=set_encoder,
        eta=training_cfg['eta'],
        M_range=training_cfg['M_range'],
        sigma=training_cfg['sigma'],
        loss_type=training_cfg.get('loss_type', 'standard'),
        grid_xyz=atf_train_sampler.grid_xyz,
        coord_mean=coord_mean,
        coord_std=coord_std
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
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="3D Diffusion Transformer (DiT) ATF Trainer")

    # --- WandB ---
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--wandb_key', type=str, default="ec2cf1718868be26a8055412b556d952681ee0b6")

    parser.add_argument('--architecture', type=str, default="dit", choices=['unet', 'dit'],
                        help='Main model architecture to use.')

    # --- Data ---
    parser.add_argument('--data_dir', type=str, default="ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200/")

    # --- Model ---
    parser.add_argument('--model_name', default="ZZZDiT", type=str)
    parser.add_argument('--patch_size', type=int, default=4,
                        help='Size of each 3D patch (e.g., 4 means 4x4x4 patches).')
    parser.add_argument('--dit_depth', type=int, default=12, help='Number of DiT blocks (transformer layers).')

    # --- Shared & SetEncoder Arguments ---
    parser.add_argument('--d_model', type=int, default=512, help='Dimension for tokens and context.')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads.')
    parser.add_argument('--num_encoder_layers', type=int, default=3, help='Layers in the SetEncoder.')
    parser.add_argument('--setencoder_version', type=str, default="v3", help='setencoder architecture version, e.g. v12:merged feature, v3:pos embed')

    # --- Training ---
    parser.add_argument('--num_iterations', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=4)  # NOTE: Must be small for 3D models
    parser.add_argument('--lr', type=float, default=1e-4, help="now it is peak learning rate after warm-up.")
    parser.add_argument('--warmup_iterations', type=int, default=5000, help="Number of iterations for linear LR warm-up.")
    parser.add_argument('--min_lr', type=float, default=1e-7,
                        help="The minimum learning rate at the end of the cosine decay.")
    parser.add_argument('--M_range', type=lambda s: [int(item) for item in s.split(',')], default=[5, 50])
    parser.add_argument('--freq_up_to', type=int, default=20, help='Use only the first N frequency channels')
    parser.add_argument('--eta', type=float, help='Probability for CFG dropout.', default=0.1)
    parser.add_argument('--sigma', type=float, help='Sigma for noise in the path.', default=0)
    parser.add_argument('--loss_type', type=str, default='weighted', choices=['standard', 'weighted'],
                        help='Type of loss function for training: "standard" MSE or "weighted" perceptual MSE.')
    parser.add_argument('--checkpoint_interval', type=int, default=20000)
    parser.add_argument('--validation_interval', type=int, default=20)
    parser.add_argument('--version', type=str, default="v3_DiT")


    # --- Paths ---
    parser.add_argument('--experiments_dir', type=str, default="experiments_3d")

    args = parser.parse_args()
    main(args)