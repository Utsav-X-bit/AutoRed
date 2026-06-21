#!/bin/bash
#SBATCH --job-name=test_gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100-SXM4:4
#SBATCH --time=00:05:00
#SBATCH --output=test_gpus.out
#SBATCH --partition=airawatp

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
echo "SLURM_STEP_GPUS: $SLURM_STEP_GPUS"
