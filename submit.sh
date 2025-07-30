#!/bin/bash
#SBATCH --job-name=fm_rir_train
#SBATCH --output=fm_rir_%j.out
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# --- Setup Environment ---
module purge
module load anaconda3/2022.05
source activate your_env_name  # <-- IMPORTANT: Replace with your conda env name

# --- Define Paths and Parameters ---
DATA_ROOT="/home/ubuntu/SDN-Python/dataset"
EXP_ROOT="/home/ubuntu/SDN-Python/experiments"

# --- Option 1: Start a NEW training run ---
# Leave RESUME_CHECKPOINT empty to start from scratch
 RESUME_CHECKPOINT=""
 python trainer-unet-spectrogram.py \
     --iterations 50000 \
     --batch_size 250 \
     --lr 1e-3 \
     --data_dir ${DATA_ROOT} \
     --exp_dir ${EXP_ROOT}

# --- Option 2: RESUME a training run ---
# Set this to the path of the checkpoint you want to resume from.
# The script will automatically find the correct experiment folder and wandb run.
#RESUME_CHECKPOINT="${EXP_ROOT}/SpecUNet_lr0.001_bs250_20250731-013000/checkpoints/ckpt_5000.pt"
#python trainer-unet-spectrogram.py \
#    --resume_from ${RESUME_CHECKPOINT}

echo "Script finished."