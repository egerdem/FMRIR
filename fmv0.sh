#!/bin/bash -l
#SBATCH --job-name=ATF_M5_gf_lr5e3
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -p gpu
#SBATCH --output=/scratch/users/%x_%j.out

source "$HOME/fmvenv/bin/activate"

cd "$HOME/FMRIR" || exit

python trainer-atf-3d.py \
    --model_name "ATF3D-CrossAttn-v1-freq20_M5to50_sigmaE5_UNET128_LRmin_e5_7_d256" \
    --data_dir ~/DATA \
    --experiments_dir ~/FMRIR_experiments \
    --batch_size 4 \
    --num_iterations 400000 \
    --lr 1e-4 \
    --freq_up_to 20 \
    --channels 32,64,128 \
    --d_model 256 \
    --nhead 4 \
    --num_encoder_layers 3 \
    --M_range 5,50 \
    --eta 0.1 \
    --sigma 1e-5 \
    --validation_interval 50 \
    --checkpoint_interval 50000
