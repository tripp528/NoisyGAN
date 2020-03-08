#!/bin/bash
# specify a partition
#SBATCH --partition=dggpu
# Request nodes
#SBATCH --nodes=1
# Request some processor cores
#SBATCH --ntasks=32
# Request GPUs
#SBATCH --gres=gpu:1
# Request memory 
#SBATCH --mem=16G
# Maximum runtime of 10 minutes
#SBATCH --time=10:00
# Name of this job
#SBATCH --job-name=ddsp
# Output of this job, stderr and stdout are joined by default
# %x=job-name %j=jobid
#SBATCH --output=./models/console/%x_%j.out
# change to the directory where you submitted this script
cd ${SLURM_SUBMIT_DIR}
# your job execution follows:
source activate ddsp
time python ~/scratch/NoisyGAN/script.py --iters=100
