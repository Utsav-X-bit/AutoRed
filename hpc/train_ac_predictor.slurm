#!/bin/bash
#SBATCH --job-name=ac_predictor      # Job name
#SBATCH --output=logs/ac_predictor_%j.out
#SBATCH --error=logs/ac_predictor_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=airawatp
#SBATCH --time=1-00:00:00

mkdir -p logs
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Starting Access Code Type Predictor training..."
python scripts/training/train_access_code_predictor.py

echo "Training Completed."
