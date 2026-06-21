#!/bin/bash
#SBATCH --job-name=AutoRed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=7-00:00:00
#SBATCH --output=AutoRed_Llama_3_8B_%j.out
#SBATCH --error=AutoRed_Llama_3_8B_%j.err
#SBATCH --partition=airawatp

ulimit -n 65535 2>/dev/null || echo "ulimit increase failed"

export AUTORED_SERVER_MODE=1

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1 

python -m uvicorn server.main:app --host 0.0.0.0 --port 8001 > llama3_8b_server.log 2>&1 &

cd ui && npm run dev > llama3_8b_frontend.log 2>&1 &