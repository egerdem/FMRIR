#!/bin/bash -l                                                                
#SBATCH --job-name=fmrir_train
#SBATCH --partition=interruptible_gpu
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=0-15:00
#SBATCH --gres=gpu:1
#SBATCH --constraint='(zen4|sapphirerapids)&(l40s|h100|a100_80g)'
#SBATCH --output=/scratch/users/%u/%j.out
#SBATCH --error=/scratch/users/%u/%j.err

source ~/fmvenv/bin/activate

python trainer-atf-3d.py \
    --geo_conditioning \
    --freq_ctx \
    --freq_up_to 20 \
    --M_range 5,10,20,50 \
    --channels 256,512,1024 \
    --decay_iterations 100000 \
    --num_iterations 400000 \
    --model_name "KCL_TimeMeanDefault_fctx_UnetV2res_freq20_Val102_Mval5_r1_M5_10_20_50_layer3_d512_eta0_head8_sigma0_lrWARM5k_e4_toe5_decay100_setv3_unet3_256to1024v2_400k" \
    --data_dir /scratch/users/k24037994/EGE/DATA/ir_fs2000_s8192_m1331_room4.0x6.0x3.0_rt200 \
    --experiments_dir /scratch/users/k24037994/EGE/FMRIR_experiments \
    --batch_size 4 \
    --lr 1e-4 \
    --min_lr 1e-5 \
    --warmup_iterations 5000 \
    --version v2_residual_context\
    --setencoder_version v3 \
    --d_model 512 \
    --nhead 8 \
    --num_encoder_layers 3 \
    --eta 0. \
    --sigma 0 \
    --FM_vs_Diff flow_matching \
    --validation_interval 100 \
    --checkpoint_interval 200000 \
    --save_start_iter 70000 \
    --idx_mes_pos_path idx_mes_pos_s8192_m1331.npy \
    --M_val_fixed 5 \
