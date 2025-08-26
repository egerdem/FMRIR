


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
    --model_name "ATF3D-CrossAttn-v1-freq20_M5to50_sigmaE5_UNET256_d512n6" \
    --data_dir ~/DATA \
    --experiments_dir ~/FMRIR_experiments \
    --batch_size 4 \
    --num_iterations 100000 \
    --lr 1e-4 \
    --freq_up_to 20 \
    --channels 32,64,128,256 \
    --d_model 512 \
    --nhead 4 \
    --num_encoder_layers 6 \
    --M_range 5,50 \
    --eta 0.1 \
    --sigma 1e-5 \
    --validation_interval 50 \
    --checkpoint_interval 50000
#resume

python trainer-atf-3d.py \
    --data_dir ~/DATA \
    --experiments_dir ~/FMRIR_experiments \
    --resume_from_checkpoint ~/FMRIR_experiments/ATF3D-CrossAttn-v1-freq20_M5to50_sigmaE3_UNET256_20250826-192413_iter200000/checkpoints/ckpt_200000.pt \
    --batch_size 4 \
    --num_iterations 500000 \
    --lr 1e-4 \
    --freq_up_to 20 \
    --channels 32,64,128,256 \
    --d_model 256 \
    --nhead 4 \
    --num_encoder_layers 3 \
    --M_range 5,50 \
    --eta 0.1 \
    --sigma 1e-3 \
    --validation_interval 50 \
    --checkpoint_interval 50000

python trainer-unet-ATF-CMD.py \
    --model_mode "spatial" \
    --flag_gaussian_mask False \
    --sigma 0.0 \
    --batch_size 250 \
    --M 5 \
    --validation_interval 20 \
    --eta 0.1 \
    --lr 1e-3 \
    --num_iterations 350000 \
    --resume_from_iteration 234579 \
    --freq_up_to 20 \
    --data_dir ~/DATA \
    --experiments_dir ~/FMRIR_experiments \
    --resume_from_checkpoint ~/FMRIR_experiments/ATF3D-CrossAttn-v1-freq20_M5to50_20250825-201433_iter200000/checkpoints/ckpt_final_200000.pt


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

python trainer-unet-ATF-CMD.py \
       --model_name ATFUNet_M5_holeloss_GaussFalse_LR5e3_n05 \
        --model_mode "spatial" \
        --flag_gaussian_mask False \
        --sigma 0. \
        --batch_size 250 \
        --M 5 \
        --validation_interval 20 \
        --eta 0.05 \
        --lr 5e-3 \
        --num_iterations 500000 \
        --freq_up_to 30 \
        --data_dir ~/DATASET \
        --experiments_dir ~/FMRIR_experiments

python trainer-atf-3d.py \
    --model_name "ATF3D-CrossAttn-v1_freq64_origdata_m30to50" \
    --data_dir ~/DATASET \
    --experiments_dir ~/FMRIR_experiments \
    --batch_size 4 \
    --num_iterations 100000 \
    --lr 1e-4 \
    --channels 32,64,128 \
    --d_model 256 \
    --nhead 4 \
    --num_encoder_layers 3 \
    --M_range 30,50 \
    --eta 0.1 \
    --sigma 1e-4 \
    --validation_interval 50 \
    --checkpoint_interval 20000

#LOCALDEN ROSSINIYE
scp -r /Users/ege/Projects/FMRIR/artifacts/ATFUNet_M5_holeloss_20250814-175237_iter100000-best-model eerdem@rossini1.ap.nii.ac.jp:~/FMRIR_experiments
scp -r /Users/ege/Projects/FMRIR/ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200/processed_atf3d_train_freqs30.pt eerdem@rossini1.ap.nii.ac.jp:~/DATA
scp -r /Users/ege/Projects/FMRIR/ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200/processed_atf3d_valid_freqs30.pt eerdem@rossini1.ap.nii.ac.jp:~/DATA

scp -r /Users/ege/Projects/FMRIR/artifacts/ATF3D-CrossAttn-v1-freq20_M5to50_sigmaE3_20250826-183304_iter200000/checkpoints/ckpt_200000_CONV.pt eerdem@rossini1.ap.nii.ac.jp:~/FMRIR_experiments/ATF3D-CrossAttn-v1-freq20_M5to50_sigmaE3_20250826-183304_iter200000/checkpoints
#LOCALDEN KCL Create'e
scp -r /Users/ege/Projects/FMRIR/ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200/processed_atf3d_train.pt k24037994@hpc.create.kcl.ac.uk:/users/k24037994/DATASET

#ROSSINI'den locale
scp -r eerdem@rossini1.ap.nii.ac.jp:~/FMRIR_experiments/ATF3D-CrossAttn-v1-freq20_M5to50_sigmaE5_UNET256_d512n6_20250826-204427_iter100000 /Users/ege/Projects/FMRIR/artifacts

