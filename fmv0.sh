#!/bin/bash -l
#SBATCH --job-name=ATF_M5_gf_lr5e3
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# --- (1) Load/activate environment ---
# If you use conda:
eval "$(conda shell.bash hook)"
#conda activate fmvenv     # <- change if your env name is different

# If you use venv instead, comment conda lines and uncomment:
source "$HOME/fmvenv/bin/activate"

# Optional: WandB credentials via env (safer than CLI)
# export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# --- (2) Go to your project directory (recommended) ---
cd "$HOME/FMRIR"

# Ensure experiments dir exists (in case you point to $HOME/FMRIR_experiments)
mkdir -p "$HOME/FMRIR_experiments"

# --- (3) Run training ---
# Note: using $HOME instead of ~ inside scripts is more robust
srun python trainer-unet-ATF-CMD.py \
  --model_name ATFUNet_M5_holeloss_GaussFalse_LR5e3 \
  --model_mode "spatial" \
  --flag_gaussian_mask False \
  --sigma 0.0 \
  --batch_size 250 \
  --M 5 \
  --validation_interval 20 \
  --eta 0.1 \
  --lr 5e-3 \
  --num_iterations 400000 \
  --data_dir "$HOME/DATASET" \
  --experiments_dir "$HOME/FMRIR_experiments"
