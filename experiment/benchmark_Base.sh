#!/bin/bash
#SBATCH --job-name=Benchmark_Base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=7-00:00:00
#SBATCH --output=benchmark_base_%j.out
#SBATCH --error=benchmark_base_%j.err
#SBATCH --partition=airawatp

source /nlsasfs/home/isea/isea13/AutoRed/.venv/bin/activate

cd /nlsasfs/home/isea/isea13/AutoRed


CUDA_VISIBLE_DEVICES=0 python experiment/llama_3_8b_verbose.py \
    --mode benchmark \
    --rounds 50 \
    --dataset-size 50 \
    --generator-path Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2 \
    --benchmark-output results/benchmarks/baseline_generator_v1_summary.json \
    2>&1 | tee logs/baseline_generator_v1.log