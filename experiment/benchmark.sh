#!/bin/bash
#SBATCH --job-name=Benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=7-00:00:00
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err
#SBATCH --partition=airawatp

AUTORED_SERVER_MODE=0

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

uv run experiment/llama_3_8b_verbose.py --mode benchmark --rounds 500 --dataset-size 5000 > benchmark.log