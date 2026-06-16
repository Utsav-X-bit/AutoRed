#!/bin/bash
#SBATCH --job-name=Benchmark_qlo1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=7-00:00:00
#SBATCH --output=benchmark_qlo_%j.out
#SBATCH --error=benchmark_qlo_%j.err
#SBATCH --partition=airawatp

source /nlsasfs/home/isea/isea13/AutoRed/.venv/bin/activate

cd /nlsasfs/home/isea/isea13/AutoRed

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

CUDA_VISIBLE_DEVICES=0 python experiment/llama_3_8b_verbose.py \
    --mode benchmark \
    --rounds 50 \
    --dataset-size 50 \
    --generator-path experiment/results/qlo_positive_v1 \
    --base-generator-path Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2 \
    --benchmark-output results/benchmarks/qlo_positive_v1_summary.json \
    2>&1 | tee logs/qlo_positive_v1_benchmark.log