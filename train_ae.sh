#!/bin/bash -l
#SBATCH --job-name=ae_train
#SBATCH --partition=interruptible_gpu
#SBATCH --ntasks=2
#SBATCH --mem=16G
#SBATCH --time=0-10:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/users/%u/%j.out
#SBATCH --error=/scratch/users/%u/%j.err

source ~/fmvenv/bin/activate
cd ~/FMRIR

