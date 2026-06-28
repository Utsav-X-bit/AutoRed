#!/bin/bash
# =============================================================================
# AutoRed Super Oracle Multi-GPU Parallel Launcher (Best-of-N)
# =============================================================================
# Usage: sbatch hpc/super_oracle_benchmark.slurm [N_SAMPLES] [NUM_SCENARIOS]
#
# Examples:
#   sbatch hpc/super_oracle_benchmark.slurm 5 1000
#
# This script requests 4 A100 GPUs and launches 4 concurrent worker processes
# to run the Phase 3 Super Oracle (Best-of-N Search over States).
# =============================================================================

#SBATCH --job-name=SuperOracle_4GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100-SXM4:4
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/super_oracle_%j.out
#SBATCH --error=logs/super_oracle_%j.err
#SBATCH --partition=airawatp

# =============================================================================
# Configuration
# =============================================================================

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

N_SAMPLES=${1:-5}
NUM_SCENARIOS=${2:-1000}
NUM_GPUS=4

# Project root
PROJECT_ROOT="/nlsasfs/home/isea/isea11/slurmJobs/AutoRed"
cd "$PROJECT_ROOT"

# Activate virtual environment
source "$PROJECT_ROOT/.venv/bin/activate"

# =============================================================================
# Setup
# =============================================================================
mkdir -p "data"
mkdir -p "logs"

echo "============================================="
echo "AutoRed Super Oracle (Best-of-${N_SAMPLES}) 4-GPU Benchmark"
echo "============================================="
echo "Total Scenarios : $NUM_SCENARIOS"
echo "N Samples/Step  : $N_SAMPLES"
echo "GPUs            : $NUM_GPUS"
echo "============================================="

# =============================================================================
# Launch Workers
# =============================================================================
PIDS=()
for WORKER_ID in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_ID=$WORKER_ID
    WORKER_LOG="logs/super_oracle_worker_${WORKER_ID}.log"

    echo ""
    echo "[LAUNCH] Worker $WORKER_ID on GPU $GPU_ID (Best-of-${N_SAMPLES})"
    echo "         Log: $WORKER_LOG"

    # Launch worker on specific GPU
    env CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/dataset_tools/super_oracle.py \
        --n "$N_SAMPLES" \
        --scenarios "$NUM_SCENARIOS" \
        --worker-id "$WORKER_ID" \
        --num-workers "$NUM_GPUS" \
        > "$WORKER_LOG" 2>&1 &

    PIDS+=($!)

    # Stagger workers by 10s to avoid NFS I/O contention during model loading
    if [ $WORKER_ID -lt $((NUM_GPUS - 1)) ]; then
        echo "Waiting 10s before launching next worker..."
        sleep 10
    fi
done

echo ""
echo "[INFO] All $NUM_GPUS workers launched in background."
echo "Waiting for all workers to finish..."

# Wait for all background processes
FAIL=0
for PID in "${PIDS[@]}"; do
    wait $PID || let "FAIL+=1"
done

if [ "$FAIL" == "0" ]; then
    echo "✅ ALL WORKERS COMPLETED SUCCESSFULLY."
    
    # Optional: Merge JSONL files here if needed
    cat data/oracle_trajectories_v2_annotated_w*.jsonl > data/oracle_trajectories_v2_annotated.jsonl
    echo "Merged trajectories saved to data/oracle_trajectories_v2_annotated.jsonl"
else
    echo "❌ $FAIL WORKER(S) FAILED. Check logs for details."
    exit 1
fi
