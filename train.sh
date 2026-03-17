#!/bin/bash -l                                                                
#SBATCH --job-name=fmrir_train                                                
#SBATCH --partition=interruptible_gpu
#SBATCH --ntasks=4
#SBATCH --mem=16G
#SBATCH --time=0-10:00                                                 
#SBATCH --gres=gpu:1                                                                                                               
#SBATCH --output=/scratch/users/%u/%j.out
#SBATCH --error=/scratch/users/%u/%j.err

source ~/fmvenv/bin/activate

python trainer-atf-3d.py \
    --geo_conditioning \
    --freq_up_to 20 \
    --M_range 5,10,20,50 \
    --channels 128,256,512 \
    --decay_iterations 100000 \
    --num_iterations 400000 \
    --model_name "KCL_SWEEP_Val102_Mval5_r1_M5_10_20_50_freq20_layer3_d512_eta0_head8_sigma0_lrWARM5k_e4_toe5_decay100_unet3_128to512v1_setv3_400k" \
    --data_dir ~/EGE/DATA/ir_fs2000_s8192_m1331_room4.0x6.0x3.0_rt200 \
    --experiments_dir ~/EGE/FMRIR_experiments \
    --batch_size 4 \
    --lr 1e-4 \
    --min_lr 1e-5 \
    --warmup_iterations 5000 \
    --version v1_legacy \
    --setencoder_version v3 \
    --d_model 512 \
    --nhead 8 \
    --num_encoder_layers 3 \
    --eta 0. \
    --sigma 0 \
    --FM_vs_Diff flow_matching \
    --validation_interval 100 \
    --checkpoint_interval 200000 \
    --idx_mes_pos_path idx_mes_pos_s8192_m1331.npy \
    --M_val_fixed 5 \
    --sweep_M
