# AutoRed QLoRA Training

## Overview

Fine-tunes `Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2` using QLoRA on AutoRed success datasets.

## Prerequisites

Install training dependencies:
```bash
pip install -r ../requirements_qlo.txt
```

Or on HPC:
```bash
uv pip install -r ../requirements_qlo.txt
```

## Dataset Variants

| Variant | Input | Output | Best For |
|---------|-------|--------|----------|
| A | Strategy | Attack | Simple baseline |
| B | Defense + Strategy | Attack | Context-aware |
| C | Defense + Response + Strategy | Attack | **Recommended** — matches AutoRed workflow |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| LoRA rank (r) | 64 |
| LoRA alpha | 128 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Quantization | NF4 4-bit |
| Batch size | 4 |
| Gradient accumulation | 8 (effective 32) |
| Learning rate | 2e-5 |
| Scheduler | Cosine with 5% warmup |
| Max length | 1024 tokens |

## Quick Start

### Local (GPU available)
```bash
# Verified dataset (138 samples)
python scripts/training/train_qlo.py \
    --dataset scripts/training/sft_data/variantc_verified_train.jsonl \
    --val_dataset scripts/training/sft_data/variantc_verified_val.jsonl \
    --output_dir experiment/results/qlo_verified_v1 \
    --epochs 10

# Positive dataset (291 samples)
python scripts/training/train_qlo.py \
    --dataset scripts/training/sft_data/variantc_positive_train.jsonl \
    --val_dataset scripts/training/sft_data/variantc_positive_val.jsonl \
    --output_dir experiment/results/qlo_positive_v1 \
    --epochs 6
```

### HPC (SLURM)
```bash
# Submit verified training
sbatch hpc/train_qlo_verified.slurm

# Submit positive training
sbatch hpc/train_qlo_positive.slurm
```

## Output

Training saves:
- `adapter_model.safetensors` — LoRA weights
- `adapter_config.json` — LoRA configuration
- `training_config.json` — Full training hyperparameters
- `train_metrics.json` — Training loss metrics

## Merging Adapters (Optional)

After training, merge LoRA adapters with base model:
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained(
    "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2",
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "experiment/results/qlo_verified_v1")
model = model.merge_and_unload()
model.save_pretrained("experiment/results/qlo_verified_merged")
```

## Evaluation

After training, benchmark the fine-tuned model against baseline:
```bash
# TODO: Add evaluation script
```

Compare against baseline metrics in `baseline_v1.md`.
