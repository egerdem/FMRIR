


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
       --model_name FREQCOND_M50_Le4_sigma1e1 \
                    --model_mode "freq_cond" \
                    --sigma 0.1 \
                    --batch_size 250 \
                    --M 50 \
                    --validation_interval 20 \
                    --eta 0.1 \
                    --lr 1e-4 \
                    --num_iterations 50000 \
                    --data_dir /home/eerdem/DATA \
                    --experiments_dir ~/FMRIR_experiments

#resume

python trainer-unet-ATF-CMD.py \
    --model_mode "freq_cond" \
    --sigma 0.0 \
    --batch_size 250 \
    --M 50 \
    --validation_interval 20 \
    --eta 0.1 \
    --lr 1e-4 \
    --num_iterations 100000 \
    --freq_up_to 64 \
    --data_dir /home/eerdem/DATA \
    --experiments_dir ~/FMRIR_experiments \

# --data_dir /home/eerdem/DATA
# --experiments_dir ~/FMRIR_experiments
# conda activate fmvenv
# move checkpoint to local:
scp -r eerdem@rossini1.ap.nii.ac.jp:/home/eerdem/FMRIR_experiments/FREQCOND_M50_LRe3_fbin64_20250818-192005_iter200000 /Users/ege/Projects/FMRIR/artifacts

#HPC CREATE
# --data_dir ~/DATASET
# --experiments_dir ~/FMRIR_experiments
# source ~/fmvenv/bin/activate