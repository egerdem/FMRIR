#!/bin/bash -l
#SBATCH --job-name=ae_train
#SBATCH --partition=interruptible_gpu
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=0-15:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|h100|a100_80g|a100
#SBATCH --output=/scratch/users/%u/%j.out
#SBATCH --error=/scratch/users/%u/%j.err

source ~/fmvenv/bin/activate
cd ~/FMRIR

MODEL=EEAE
CFG=10002
RUN_DIR="outputs/out$(date +%Y%m%d)_${MODEL}${CFG}"
mkdir -p "$RUN_DIR"

python -u AUTOENCODER/ATF_interp/main.py \
    -m "$MODEL" \
    -c "$CFG" \
    --use_npz \
    --dataset_version r1 \
    -d ~/EGE/DATA \
    -a "$RUN_DIR" \
    > "$RUN_DIR/log_c${CFG}_j${SLURM_JOB_ID}.txt" 2>&1