#!/bin/bash

set -e  # stop on error

python trainer-atf-3d.py \
    --model_name "TRY_abl_r1_ICASSP_M5_50_freq20_layer3_d512_eta0_head8_sigma1e3_lrWARM5k_e4_toe5_decay300_unet4v1_setv3_300k" \
    --data_dir /home/ubuntu/EGE/DATA \
    --experiments_dir /home/ubuntu/EGE/FMRIR_experiments \
    --batch_size 4 \
    --num_iterations 300000 \
    --lr 1e-4 \
    --min_lr 1e-5 \
    --warmup_iterations 5000 \
    --decay_iterations 300000 \
    --version "v1_legacy" \
    --setencoder_version "v3" \
    --freq_up_to 20 \
    --channels 32,64,128,256 \
    --d_model 512 \
    --nhead 8 \
    --num_encoder_layers 3 \
    --M_range 5,50 \
    --eta 0. \
    --sigma 0. \
    --FM_vs_Diff "flow_matching" \
    --validation_interval 100 \
    --checkpoint_interval 100000