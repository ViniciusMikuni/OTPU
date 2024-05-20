#!/bin/sh
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -n 32
#SBATCH --ntasks-per-node 4
#SBATCH --gpus-per-task 1
#SBATCH -t 04:30:00
#SBATCH -A m3246
#SBATCH --gpu-bind=none

module load tensorflow/2.15.0
export TF_CPP_MIN_LOG_LEVEL=2

echo srun python train.py
srun python train.py
