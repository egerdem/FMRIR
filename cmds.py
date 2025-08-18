


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

#resume

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
    --data_dir /home/eerdem/DATA \
    --experiments_dir ~/FMRIR_experiments \
    --resume_from_checkpoint ~/FMRIR_experiments/ATFUNet_M5_holeloss_20250814-175237_iter100000-best-model/modelv2.pt


# --data_dir /home/eerdem/DATA
# --experiments_dir ~/FMRIR_experiments
# conda activate fmvenv
# move checkpoint to local:
scp -r eerdem@rossini1.ap.nii.ac.jp:~/FMRIR_experiments//ATFUNet_M5_holeloss_NOGAUSSION_LR5e3_20250818-224428_iter100000 /Users/ege/Projects/FMRIR/artifacts

#HPC CREATE
ssh k24037994@hpc.create.kcl.ac.uk

# --data_dir ~/DATASET
# --experiments_dir ~/FMRIR_experiments
# source ~/fmvenv/bin/activate

python trainer-unet-ATF-CMD.py \
       --model_name ATFUNet_M5_holeloss_GaussFalse_LR5e3 \
        --model_mode "spatial" \
        --flag_gaussian_mask False \
        --sigma 0. \
        --batch_size 250 \
        --M 5 \
        --validation_interval 20 \
        --eta 0.1 \
        --lr 5e-3 \
        --num_iterations 400000 \
        --data_dir ~/DATASET \
        --experiments_dir ~/FMRIR_experiments

#LOCALDEN ROSSINIYE
scp -r /Users/ege/Projects/FMRIR/artifacts/ATFUNet_M5_holeloss_20250814-175237_iter100000-best-model eerdem@rossini1.ap.nii.ac.jp:~/FMRIR_experiments

scp -r /Users/ege/Projects/FMRIR/artifacts/ATFUNet_M5_holeloss_20250814-175237_iter100000-best-model k24037994@hpc.create.kcl.ac.uk:/users/k24037994/FMRIR

