


python trainer-unet-ATF-CMD.py --resume_from_checkpoint /home/eerdem/DATA/artifacts/ATFUNet_20250806-185407_iter20000-best-model:v0/model.pt --resume_from_iteration 20000
--resume_run_id j30tdj4w  --sigma 0.0 --batch_size 250 --M 50 --validation_interval 20 --eta 0.1 --lr 1e-4 --num_iterations 40000 --checkpoint_interval 1000 --data_dir /home/eerdem/DATA


# import wandb
# wandb.login(key= "ec2cf1718868be26a8055412b556d952681ee0b6")
# run = wandb.init()
# artifact = run.use_artifact('ege-erdem-king-s-college-london/FM-RIR/ATFUNet_20250806-185407_iter20000-best-model:v0', type='model')
# artifact_dir = artifact.download()


#
#  ROSSINI
# first
#SLICE
python trainer-unet-ATF-CMD.py \
       --model_name ATFUNet_M5_holeloss_NOGAUSSION_LR5e3 \
        --model_mode "spatial" \
        --flag_gaussian_mask False \
        --sigma 0. \
        --batch_size 250 \
        --M 50 \
        --validation_interval 20 \
        --eta 0.1 \
        --lr 5e-3 \
        --num_iterations 300000 \
        --data_dir /home/eerdem/DATA \
        --experiments_dir ~/FMRIR_experiments
#3D UNET
python trainer-atf-3d.py \
    --data_dir ~/DATA \
    --experiments_dir ~/FMRIR_experiments \
    --batch_size 4 \
    --num_iterations 500000 \
    --lr 6e-5 \
    --freq_up_to 20 \
    --channels 32,64,128 \
    --d_model 256 \
    --nhead 4 \
    --num_encoder_layers 3 \
    --M_range 40,50 \
    --eta 0.1 \
    --sigma 1e-5 \
    --validation_interval 50 \
    --checkpoint_interval 50000
#resume

python trainer-atf-3d.py \
    --data_dir ~/DATA \
    --experiments_dir ~/FMRIR_experiments \
    --resume_from_checkpoint ~/FMRIR_experiments/M5to50_freq20_layer3_d512_head4_sigma1e3_lrWARM5k_e4_toe5_unet3_setv3_20250908-151143_iter300000/model.pt \
    --batch_size 4 \
    --num_iterations 300000 \
    --lr 1e-5 \
    --channels 32,64,128 \
    --d_model 512 \
    --nhead 4 \
    --num_encoder_layers 3 \
    --M_range 5,50 \
    --eta 0. \
    --sigma 0 \
    --loss_type "weighted" \
    --setencoder_version "v3" \
    --validation_interval 100 \
    --checkpoint_interval 50000

python trainer-atf-3d.py \
    --model_name "M5to50_freq20_layer3_d512_head8_eta0_sigma0_lrWARM5k_e4_toe6_unet4v1_setv12_700k" \
    --data_dir ~/DATA \
    --experiments_dir ~/FMRIR_experiments \
    --batch_size 4 \
    --num_iterations 300000 \
    --version "v1_legacy" \
    --freq_up_to 20 \
    --lr 1e-4 \
    --warmup_iterations 5000 \
    --min_lr 1e-6 \
    --channels 32,64,128,256 \
    --d_model 512 \
    --nhead 8 \
    --num_encoder_layers 3 \
    --M_range 5,50 \
    --eta 0.0 \
    --sigma 0 \
    --loss_type "standard" \
    --setencoder_version "v12" \
    --validation_interval 100 \
    --checkpoint_interval 100000
# --decay_iterations
# 500000 \

# FLOW MATCHING

python trainer-atf-3d.py \
    --model_name "*BIG_8192R4_ICASSP_M5_50_freq20_layer3_d512_eta0_head8_sigma0_lrWARM5k_e4_toe5_decay500_unet4v1_setv12_800k" \
    --data_dir ~/DATA \
    --experiments_dir ~/FMRIR_experiments \
    --batch_size 4 \
    --num_iterations 500000 \
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
    --M_sampling_mode "range"
    --eta 0. \
    --sigma 0 \
    --FM_vs_Diff "flow_matching" \
    --validation_interval 100 \
    --checkpoint_interval 200000

#SCORE DDPM

# python trainer-atf-3d.py \
#     --model_name "BIG_8192R4_DDOM_M5to50_freq20_layer3_d512_eta1e1_head8_sigma0_lrWARM5k_2PHASEe4_toe5at500k_unet4v1_setv12_800k" \
#     --data_dir ~/DATA \
#     --experiments_dir ~/FMRIR_experiments \
#     --batch_size 4 \
#     --num_iterations 800000 \
#     --lr 1e-4 \
#     --min_lr 1e-5 \
#     --warmup_iterations 5000 \
#     --decay_iterations 500000 \
#     --version "v1_legacy" \
#     --setencoder_version "v12" \
#     --freq_up_to 20 \
#     --channels 32,64,128,256 \
#     --d_model 512 \
#     --nhead 8 \
#     --num_encoder_layers 3 \
#     --M_range 5,50 \
#     --eta 0.1 \
#     --sigma 0 \
#     --FM_vs_Diff "score_matching" \
#     --validation_interval 100 \
#     --checkpoint_interval 100000

# python trainer-DiT-3d.py \
#     --model_name "M5to50_freq20_d512_head8_patch4_dept12_sigma0_lrWARM5k_e4_toe5_DiTNetv3_setv3" \
#     --data_dir ~/DATA \
#     --experiments_dir ~/FMRIR_experiments \
#     --batch_size 4 \
#     --num_iterations 300000 \
#     --version "v3_DiT" \
#     --freq_up_to 20 \
#     --lr 1e-4 \
#     --warmup_iterations 5000 \
#     --min_lr 1e-5 \
#     --patch_size 4 \
#     --dit_depth 12
#     --d_model 512 \
#     --nhead 8 \
#     --num_encoder_layers 3 \
#     --M_range 5,50 \
#     --eta 0.1 \
#     --sigma 0 \
#     --loss_type "standard" \
#     --setencoder_version "v3" \
#     --validation_interval 100 \
#     --checkpoint_interval 1000000

# python trainer-unet-ATF-CMD.py \
#     --model_mode "spatial" \
#     --flag_gaussian_mask False \
#     --sigma 0.0 \
#     --batch_size 250 \
#     --M 5 \
#     --validation_interval 20 \
#     --eta 0.1 \
#     --lr 1e-3 \
#     --num_iterations 350000 \
#     --resume_from_iteration 234579 \
#     --freq_up_to 20 \
#     --data_dir ~/DATA \
#     --experiments_dir ~/FMRIR_experiments \
#     --resume_from_checkpoint ~/FMRIR_experiments/ATF3D-CrossAttn-v1-freq20_M5to50_20250825-201433_iter200000/checkpoints/ckpt_final_200000.pt


# --data_dir /home/eerdem/DATA
# --experiments_dir ~/FMRIR_experiments
# conda activate fmvenv
# move checkpoint to local:
scp -r eerdem@rossini1.ap.nii.ac.jp:~/FMRIR_experiments/ATFUNet_M5_holeloss_GaussFalse_LR5e3_n05_20250820-173718_iter500000 /Users/ege/Projects/FMRIR/artifacts

#HPC CREATE
ssh k24037994@hpc.create.kcl.ac.uk
scp -r k24037994@hpc.create.kcl.ac.uk:~/FMRIR_experiments/ATF3D-CrossAttn-v1_20250824-173107_iter20000 /Users/ege/Projects/FMRIR/artifacts

# --data_dir ~/DATASET
# --experiments_dir ~/FMRIR_experiments
# source ~/fmvenv/bin/activate

# python trainer-unet-ATF-CMD.py \
#        --model_name ATFUNet_M5_holeloss_GaussFalse_LR5e3_n05 \
#         --model_mode "spatial" \
#         --flag_gaussian_mask False \
#         --sigma 0. \
#         --batch_size 250 \
#         --M 5 \
#         --validation_interval 20 \
#         --eta 0.05 \
#         --lr 5e-3 \
#         --num_iterations 500000 \
#         --freq_up_to 30 \
#         --data_dir ~/DATASET \
#         --experiments_dir ~/FMRIR_experiments


#LOCALDEN ROSSINIYE
scp -r /Users/ege/Projects/FMRIR/ir_fs2000_s8192_m1331_room4.0x6.0x3.0_rt200 eerdem@rossini1.ap.nii.ac.jp:~/DATA

scp -r /Users/ege/Projects/FMRIR/ir_fs2000_s8192_m1331_room4.0x6.0x3.0_rt200/processed_atf3d_valid_freqs20_r1.pt eerdem@rossini1.ap.nii.ac.jp:~/DATA
scp -r /Users/ege/Projects/FMRIR/ir_fs2000_s8192_m1331_room4.0x6.0x3.0_rt200/processed_atf3d_train_freqs20_r1.pt eerdem@rossini1.ap.nii.ac.jp:~/DATA

#LOCALDEN BELLINI
scp -r /Users/ege/Projects/FMRIR/ir_fs2000_s8192_m1331_room4.0x6.0x3.0_rt200/processed_atf3d_train_freqs20_r4.pt eerdem@bellini1.ap.nii.ac.jp:~/DATA
scp -r /Users/ege/Projects/FMRIR/ir_fs2000_s8192_m1331_room4.0x6.0x3.0_rt200/processed_atf3d_valid_freqs20_r4.pt eerdem@bellini1.ap.nii.ac.jp:~/DATA

scp -r /Users/ege/Projects/FMRIR/artifacts/ATF3D-CrossAttn-v1-freq20_M5to50_sigmaE3_20250826-183304_iter200000/checkpoints/ckpt_200000_CONV.pt eerdem@rossini1.ap.nii.ac.jp:~/FMRIR_experiments/ATF3D-CrossAttn-v1-freq20_M5to50_sigmaE3_20250826-183304_iter200000/checkpoints
#LOCALDEN KCL Create'e
scp -r /Users/ege/Projects/FMRIR/ir_fs2000_s8192_m1331_room4.0x6.0x3.0_rt200/processed_atf3d_train_freqs20_r3.pt k24037994@hpc.create.kcl.ac.uk:/users/k24037994/DATA

#ROSSINI'den locale
#
# scp -r eerdem@bellini1.ap.nii.ac.jp:~/FMRIR_experiments/M5to50_freq20_layer3_d512_head8_sigma1e3_lrWARM5k_e4_toe6_unet3_20250905-193258_iter300000 /Users/ege/Projects/FMRIR/artifacts

scp -r eerdem@rossini1.ap.nii.ac.jp:~/FMRIR_experiments/BIG_8192R4_ICASSP_M5_100_freq20_layer3_d512_eta0_head8_sigma0_lrWARM5k_e4_toe5_decay700_unet4v1_setv12_800k_20250916-205850_iter800000 /Users/ege/Projects/FMRIR/artifacts

#BELLİNİ'den locale
scp -r eerdem@bellini1.ap.nii.ac.jp:~/FMRIR_experiments/BIG_8192R4_DDOM_M5to50_freq20_layer3_d512_eta1e1_head8_sigma0_lrWARM5k_2PHASEe4_toe5at500k_unet4v1_setv12_800k_20250913-184207_iter800000 /Users/ege/Projects/FMRIR/artifacts

#Create'den locale

scp -r k24037994@hpc.create.kcl.ac.uk:~/FMRIR_experiments/M5to50_SCOREMATCH_freq20_layer3_d512_eta0_head8_sigma2e1_lrWARM20k_e4_toe5_unet4v1_setv3_20250909-061709_iter300000 /Users/ege/Projects/FMRIR/artifacts



