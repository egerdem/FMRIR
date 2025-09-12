#!/bin/bash -l
#SBATCH --job-name=ATF_M5_gf_lr5e3
#SBATCH --time=2:00:00
#SBATCH --output=/users/k24037994/logs/%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -p gpu

# shellcheck disable=SC1090
source ~/fmvenv/bin/activate
cd ~/FMRIR

python -u trainer-atf-3d.py \
    --model_name "BIG8192DATA_VALSplit_M5to50_freq20_layer3_d512_eta0_head8_sigma0_lrWARM5k_2PHASEe4_toe5at500k_unet4v1_setv12_700k" \
    --data_dir ~/DATA \
    --experiments_dir ~/FMRIR_experiments \
    --batch_size 4 \
    --num_iterations 700000 \
    --lr 1e-4 \
    --min_lr 1e-5 \
    --warmup_iterations 5000 \
    --decay_iterations 500000 \
    --version "v1_legacy" \
    --setencoder_version "v12" \
    --freq_up_to 20 \
    --channels 32,64,128,256 \
    --d_model 512 \
    --nhead 8 \
    --num_encoder_layers 3 \
    --M_range 5,50 \
    --eta 0. \
    --sigma 0. \
    --FM_vs_Diff "score_matching" \
    --validation_interval 100 \
    --checkpoint_interval 100000
