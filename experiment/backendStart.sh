#!/bin/bash
#SBATCH --job-name=AutoRedBackend
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/AutoRedBackend_%j.out
#SBATCH --error=logs/AutoRedBackend_%j.err
#SBATCH --partition=airawatp

source .venv/bin/activate

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export AUTORED_SERVER_MODE=1

python -m uvicorn server.main:app --host 0.0.0.0 --port 8001