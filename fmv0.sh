#!/bin/bash -l
#SBATCH --job-name=ATF_M5_gf_lr5e3
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -p gpu
#SBATCH --output=/scratch/users/%x_%j.out

source "$HOME/fmvenv/bin/activate"

cd "$HOME/FMRIR" || exit

python trainer-atf-3d.py \
    --model_name "M5to50_freq20_layer4_d256_head8_sigma0_lrWARM5k_e4_toe6_unet4" \
    --data_dir ~/DATA \
    --experiments_dir ~/FMRIR_experiments \
    --batch_size 4 \
    --num_iterations 300000 \
    --version "v1_legacy" \
    --lr 1e-4 \
    --warmup_iterations 5000 \
    --min_lr 1e-6 \
    --channels 32,64,128,256 \
    --d_model 256 \
    --nhead 8 \
    --num_encoder_layers 4 \
    --M_range 5,50 \
    --eta 0.1 \
    --sigma 0 \
    --validation_interval 100 \
    --checkpoint_interval 100000
