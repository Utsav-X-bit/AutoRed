# AutoRed — HPC Re-Setup Guide

**Last Updated:** 2026-07-09  
**Cluster:** NLS (Iowa State University)  
**Partitions:** `gpu`, `airawatp`  
**GPU:** NVIDIA A100-SXM4-40GB  
**Project Path:** `/nlsasfs/home/isea/isea13/AutoRed`

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Model Inventory & Download](#2-model-inventory--download)
3. [Missing Asset Audit & Recovery](#3-missing-asset-audit--recovery)
4. [Dataset Inventory](#4-dataset-inventory)
5. [Knowledge Base & RAG](#5-knowledge-base--rag)
6. [Training All Models (Step-by-Step)](#6-training-all-models-step-by-step)
7. [Benchmark Execution](#7-benchmark-execution)
8. [Server & Frontend](#8-server--frontend)
9. [Post-Processing & Analysis](#9-post-processing--analysis)
10. [SLURM Job Reference](#10-slurm-job-reference)
11. [Troubleshooting](#11-troubleshooting)
12. [Full Re-Setup Checklist](#12-full-re-setup-checklist)

---

## 1. Environment Setup

### 1.1 Clone & Navigate

```bash
# On the HPC login node (has internet access)
cd /nlsasfs/home/isea/isea13
git clone <your-repo-url> AutoRed
cd AutoRed
```

### 1.2 Create Virtual Environment

```bash
# Python ≥ 3.10 required
python3 -m venv .venv
source .venv/bin/activate
```

### 1.3 Install Core Dependencies

The project uses `uv` for fast installs on HPC. Install `uv` first if not available:

```bash
pip install uv
```

Then install the main dependencies:

```bash
# Core experiment + inference dependencies (PyTorch CUDA 12.4)
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124

# QLoRA/SFT/DPO training dependencies (on top of core)
uv pip install -r requirements_qlo.txt

# Server mode dependencies (FastAPI + Redis)
uv pip install -r requirements_server.txt

# vLLM backend (for vllm experiment variant)
uv pip install vllm

# Additional commonly needed packages
uv pip install faiss-cpu sentence-transformers scikit-learn
```

### 1.4 Verify Installation

```bash
python3 -c "
import torch
import transformers
import accelerate

print(f'torch: {torch.__version__}')
print(f'transformers: {transformers.__version__}')
print(f'accelerate: {accelerate.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

### 1.5 For QLoRA Training Verification

```bash
bash hpc/setup_qlo.sh
```

### 1.6 Directory Structure

```bash
mkdir -p logs results results/benchmarks experiment/results data models pre_trained
```

---

## 2. Model Inventory & Download

All models must be pre-downloaded on the **login node** (which has internet). Compute nodes run in offline mode (`TRANSFORMERS_OFFLINE=1`, `HF_HUB_OFFLINE=1`).

### 2.1 HuggingFace Models Required

| Model | HuggingFace ID | Size | Purpose |
|-------|---------------|------|---------|
| **Victim LLM** | `meta-llama/Meta-Llama-3-8B-Instruct` | ~16 GB | Model under attack (defended) |
| **Generator** | `Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2` | ~16 GB | Adversarial prompt generator (uncensored) |
| **Judge base** | `distilbert-base-uncased` | ~250 MB | Base for DistilBERT judge (PI reward model) |
| **Ranker base** | `microsoft/deberta-v3-base` | ~350 MB | Extraction ranking classifier |
| **RAG Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | ~80 MB | Defense embeddings for FAISS retrieval |
| **T5 Generator** | `t5-base` | ~850 MB | Original paper's generator (legacy) |
| **BERTScore** | `roberta-large` | ~1.3 GB | For evaluation metrics |

### 2.2 Download Script (Run on Login Node)

```bash
source .venv/bin/activate

python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

# Victim
print('Downloading Meta-Llama-3-8B-Instruct...')
AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')

# Generator
print('Downloading Llama-3.1-8B-Lexi-Uncensored-V2...')
AutoTokenizer.from_pretrained('Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2')
AutoModelForCausalLM.from_pretrained('Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2')

# Judge + classifiers base
print('Downloading distilbert-base-uncased...')
AutoTokenizer.from_pretrained('distilbert-base-uncased')
AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')

# Ranker base
print('Downloading deberta-v3-base...')
AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-base', num_labels=2)

# RAG embeddings
print('Downloading all-MiniLM-L6-v2...')
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Evaluation metrics base
print('Downloading roberta-large...')
AutoTokenizer.from_pretrained('roberta-large')

print('All models downloaded.')
"
```

### 2.3 Offline Environment Variables

All SLURM scripts should include:

```bash
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
```

---

## 3. Missing Asset Audit & Recovery

> [!IMPORTANT]
> This section documents every trained checkpoint, dataset, and file referenced by the experiment scripts, whether it exists locally, and how to create it if missing.

### 3.1 Trained Model Checkpoints

| Checkpoint | Expected Path | Status | How to Create |
|-----------|--------------|--------|---------------|
| **Judge (PI Reward Model)** | `pre_trained/pi_reward_model/` | ❌ **MISSING** | [§6.1 Train Judge](#61-judge-pi-reward-model) |
| **Strategy Predictor (MLP)** | `experiment/strategy_predictor.pth` | ✅ Present (209 KB) | [§6.2](#62-strategy-predictor-mlp) |
| **Feature/Label Vocab** | `experiment/feature_vocab.json`, `label_vocab.json` | ✅ Present | Generated by strategy predictor training |
| **Access Code Predictor** | `experiment/access_code_predictor/` | ⚠️ **EMPTY DIR** | [§6.3 Train Access Code Predictor](#63-access-code-predictor-distilbert-4-class) |
| **Defense Classifier** | `models/defense_classifier/` | ❌ **MISSING** | [§6.4 Train Defense Classifier](#64-defense-classifier-distilbert) |
| **Ranker (DeBERTa)** | `models/ranker_deberta_v1/` | ❌ **MISSING** | [§6.5 Train Ranker](#65-ranker-deberta) |
| **Planner (SFT adapter)** | `experiment/results/planner_sft_v4/` | ❌ **MISSING** | [§6.6 Train Planner](#66-planner-sft-qlora) |
| **Generator QLoRA (verified)** | `experiment/results/qlo_verified_v1/` | ❌ **MISSING** | [§6.7 Train Generator QLoRA](#67-generator-qlora-sft) |
| **Generator QLoRA (positive)** | `experiment/results/qlo_positive_v1/` | ❌ **MISSING** | [§6.7 Train Generator QLoRA](#67-generator-qlora-sft) |
| **Generator DPO adapter** | `experiment/generator_dpo_adapter/` | ❌ **MISSING** | [§6.8 Train Generator DPO](#68-generator-dpo) |

### 3.2 Datasets

| Dataset | Path | Status | How to Create |
|---------|------|--------|---------------|
| `autored_successes_v1.jsonl` | `data/` | ✅ Present (22 MB, 26K lines) | From benchmark post-processing |
| `autored_failures_v1.jsonl` | `data/` | ✅ Present (129 MB, 136K lines) | From benchmark post-processing |
| `autored_positive_v1.jsonl` | `data/` | ✅ Present (5.4 MB) | From analyze_dataset.py |
| `autored_verified_v1.jsonl` | `data/` | ✅ Present (3.5 MB) | From analyze_dataset.py |
| `autored_extractor_failures_v1.jsonl` | `data/` | ✅ Present (4.1 MB) | From analyze_dataset.py |
| `generator_sft_dataset.jsonl` | `data/` | ✅ Present (1.8 MB) | `build_generator_sft_dataset.py` |
| `generator_dpo_dataset.jsonl` | `data/` | ✅ Present (7.5 MB) | `build_dpo_dataset.py` |
| `strategy_matrix_raw_v1.jsonl` | `data/` | ✅ Present (28 MB) | `build_strategy_matrix.py` |
| `attack_transition_dataset.jsonl` | `data/` | ✅ Present (69 MB) | `mine_attack_transitions.py` |
| `sft_planner_v4.jsonl` | `data/` | ✅ Present (3.1 MB) | `build_oracle_sft_dataset.py` |
| `oracle_trajectories_v4.jsonl` | `data/` | ✅ Present (13 MB) | From oracle benchmark |
| `scored_trajectories_v4.jsonl` | `data/` | ✅ Present (14 MB) | From trajectory scoring |
| `defense_classifier_dataset-Part{1,2}.jsonl` | `data/` | ✅ Parts exist (85+85 MB) | `build_defense_classifier_dataset.py` |
| `defense_classifier_dataset.jsonl` **(merged)** | `data/` | ❌ **MISSING** | [Merge parts](#641-prepare-defense-classifier-dataset) |
| `access_code_classifier_dataset.jsonl` **(merged)** | `data/` | ❌ **MISSING** | [Merge parts](#631-prepare-access-code-classifier-dataset) |
| `access_code_classifier_dataset_part_{aa,ab,ac,ad}` | `data/` | ✅ Parts exist (150 MB total) | Split files need merging |
| `ranker_dataset_{train,val,test}_v1.jsonl` | `data/` | ❌ **MISSING** | [§6.5.1 Build ranker dataset](#651-build-ranker-dataset) |
| `benchmark_v1.jsonl` | `data/` | ✅ Present (275 KB) | From create_benchmark.py |
| `benchmark_v2.jsonl` | `data/` | ✅ Present (1.6 MB) | From freeze_benchmark.py |
| `garak_llama3-8B-Instruct_verified.jsonl` | `data/` | ✅ Present (62 MB) | External Garak data |
| `prompt_extraction_detection.jsonl` | project root | ✅ Present | TensorTrust dataset (judge training) |

### 3.3 SFT Training Data (in `scripts/training/sft_data/`)

| File | Status |
|------|--------|
| `variantc_verified_train.jsonl` / `val.jsonl` | ✅ Present |
| `variantc_positive_train.jsonl` / `val.jsonl` | ✅ Present |
| `varianta_*` / `variantb_*` | ✅ Present |
| `planner_v4_train.jsonl` / `val.jsonl` | ✅ Present |
| `generator_v4_train.jsonl` / `val.jsonl` | ✅ Present |

### 3.4 Other Files

| File | Path | Status |
|------|------|--------|
| `raw_dump_defenses.jsonl.bz2` | `experiment/` | ✅ Present (7.2 MB) |
| `oracle_v3_scenarios_5000.jsonl.bz2` | `experiment/` | ✅ Present (881 KB) |
| `autored_kb.db` | `data/` | ✅ Present (113 MB) |
| `data/rag/success_defenses.index` | `data/rag/` | ✅ Present (15 MB) |
| `data/rag/success_metadata.json` | `data/rag/` | ✅ Present (12 MB) |
| `ui/` (frontend) | `ui/` | ✅ Present (with node_modules) |
| `scripts/pi/pi_data/` | `scripts/pi/` | ✅ Present (train/test/val JSON) |
| `pre_trained/` directory | project root | ❌ **MISSING** (needs creating) |

### 3.5 Dependency Chain (Build Order)

The models have dependencies on each other and on datasets. Here is the correct build order:

```
Step 0: Verify all raw datasets exist (Section 3.2 ✅ items)
  │
  ├── Step 1: JUDGE — Train PI Reward Model
  │     Input:  prompt_extraction_detection.jsonl (✅ present, or downloads from GitHub)
  │     Output: pre_trained/pi_reward_model/
  │     Script: hpc/train_reward_model.py
  │     SLURM:  hpc/train_reward_model.slurm
  │
  ├── Step 2: STRATEGY PREDICTOR — Already trained ✅
  │     Checkpoint: experiment/strategy_predictor.pth
  │     (Retrain: hpc/train_strategy_predictor.slurm)
  │
  ├── Step 3: ACCESS CODE PREDICTOR — Merge dataset + train
  │     Input:  data/access_code_classifier_dataset_part_{aa,ab,ac,ad} → merge
  │     Output: experiment/access_code_predictor/
  │     Script: scripts/training/train_access_code_predictor.py
  │     SLURM:  hpc/train_ac_predictor.slurm
  │
  ├── Step 4: DEFENSE CLASSIFIER — Merge dataset + train
  │     Input:  data/defense_classifier_dataset-Part{1,2}.jsonl → merge
  │     Output: models/defense_classifier/
  │     Script: scripts/training/train_defense_classifier.py
  │     SLURM:  (none — run manually or create one)
  │
  ├── Step 5: RANKER — Build dataset + train
  │     Input:  data/autored_{verified,positive,failures}_v1.jsonl + results/
  │     Output: models/ranker_deberta_v1/
  │     Script: scripts/training/build_ranker_dataset.py → train_ranker.py
  │     SLURM:  hpc/train_ranker.slurm (builds dataset + trains)
  │
  ├── Step 6: PLANNER SFT — Train QLoRA adapter
  │     Input:  scripts/training/sft_data/planner_v4_{train,val}.jsonl (✅ present)
  │     Output: experiment/results/planner_sft_v4/
  │     Script: scripts/training/train_qlo.py
  │     SLURM:  hpc/train_planner_sft.slurm
  │
  ├── Step 7: GENERATOR QLoRA SFT — Train adapter
  │     Input:  scripts/training/sft_data/variantc_verified_{train,val}.jsonl (✅ present)
  │     Output: experiment/results/qlo_verified_v1/
  │     Script: scripts/training/train_qlo.py
  │     SLURM:  hpc/train_qlo_verified.slurm
  │
  └── Step 8: GENERATOR DPO — Train preference adapter (optional)
        Input:  data/generator_dpo_dataset.jsonl (✅ present)
        Output: experiment/generator_dpo_adapter/
        Script: scripts/training/train_dpo.py
        SLURM:  hpc/train_dpo.slurm
```

> [!NOTE]
> Steps 1–5 are **independent** and can run in parallel.
> Steps 6–8 are also independent of each other (they all fine-tune the base generator).
> The Judge (Step 1) is the **most critical** — the experiment scripts will not function without it.

---

## 4. Dataset Inventory

### 4.1 Core Datasets (in `data/`)

| File | Size | Lines | Description |
|------|------|-------|-------------|
| `autored_successes_v1.jsonl` | 22 MB | 26,741 | All successful attack attempts |
| `autored_failures_v1.jsonl` | 129 MB | 135,930 | All failed attack attempts |
| `autored_positive_v1.jsonl` | 5.4 MB | 4,853 | GT leaked OR verification success |
| `autored_verified_v1.jsonl` | 3.5 MB | — | Verification successes only |
| `autored_extractor_failures_v1.jsonl` | 4.1 MB | — | Extractor failure cases |

### 4.2 Training Datasets

| File | Size | Lines | Used By |
|------|------|-------|---------|
| `generator_sft_dataset.jsonl` | 1.8 MB | 2,238 | `train_generator_sft.py` |
| `generator_dpo_dataset.jsonl` | 7.5 MB | 3,627 | `train_dpo.py` |
| `sft_planner_v4.jsonl` | 3.1 MB | 2,352 | `train_planner_sft.slurm` |
| `attack_transition_dataset.jsonl` | 69 MB | 80,711 | Strategy learning |
| `strategy_matrix_raw_v1.jsonl` | 28 MB | — | `train_strategy_predictor.py` |
| `defense_classifier_dataset-Part{1,2}.jsonl` | 170 MB | — | `train_defense_classifier.py` (needs merge) |
| `access_code_classifier_dataset_part_{aa-ad}` | 150 MB | — | `train_access_code_predictor.py` (needs merge) |

### 4.3 SFT Training Data (in `scripts/training/sft_data/`)

| File | Purpose |
|------|---------|
| `variantc_verified_train/val.jsonl` | Variant C verified (recommended for QLoRA) |
| `variantc_positive_train/val.jsonl` | Variant C positive (larger dataset) |
| `varianta_*/variantb_*` | Simpler variants (less context) |
| `planner_v4_train/val.jsonl` | Planner SFT data |
| `generator_v4_train/val.jsonl` | Generator v4 SFT data |

### 4.4 Benchmark Scenarios

| File | Size | Description |
|------|------|-------------|
| `benchmark_v1.jsonl` | 275 KB | V1 defense scenarios (201 entries) |
| `benchmark_v2.jsonl` | 1.6 MB | V2 benchmark |
| `benchmark_dev_v2.jsonl` | 1.2 MB | V2 dev split |
| `benchmark_holdout_v2.jsonl` | 331 KB | V2 holdout split |

### 4.5 Oracle Trajectories

| File | Size | Lines |
|------|------|-------|
| `oracle_trajectories_v4.jsonl` | 13 MB | 4,506 |
| `scored_trajectories_v4.jsonl` | 14 MB | 4,506 |
| `oracle_trajectories_v3*.jsonl` (8 shards) | ~17 MB | — |
| `oracle_trajectories_v2_annotated*.jsonl` (4 shards) | ~3.5 MB | — |

---

## 5. Knowledge Base & RAG

### 5.1 SQLite Knowledge Base

- **File:** `data/autored_kb.db` (113 MB) — ✅ Present
- **Module:** `experiment/knowledge_base.py`
- **Tables:**
  - `trajectories` — 19 columns (session_id, scenario_id, defense_prompt, ground_truth, strategy, attack_string, victim_response, verifier_success, reward, etc.)
  - `state_snapshots` — 6 columns (state_id, session_id, attempt, state_json, hash, timestamp)
- **Indexes:** `idx_scenario`, `idx_success`, `idx_state`

### 5.2 RAG Components

| Component | File | Status |
|-----------|------|--------|
| FAISS Index | `data/rag/success_defenses.index` (15 MB) | ✅ Present |
| Metadata | `data/rag/success_metadata.json` (12 MB) | ✅ Present |
| Embedder | `all-MiniLM-L6-v2` | Needs HF download |
| Retriever Class | `DefenseRetriever` in experiment script | Built-in |

### 5.3 Rebuilding Knowledge Base (if needed)

```bash
# Convert benchmark JSON results into the KB
python scripts/dataset_tools/benchmark_json_to_kb.py

# Mine attack transitions from KB
python scripts/dataset_tools/mine_attack_transitions.py
```

### 5.4 Rebuilding RAG Index (if needed)

```bash
python scripts/dataset_tools/build_rag_index.py
```

---

## 6. Training All Models (Step-by-Step)

> [!IMPORTANT]
> This section provides the **complete training procedure** for every model in the AutoRed pipeline, including dataset preparation, exact commands, hyperparameters, and SLURM scripts.

### 6.1 Judge (PI Reward Model)

The Judge is a **DistilBERT binary classifier** that determines if a victim's response contains a prompt extraction attempt. This is the **most critical** missing model — the experiment pipeline cannot function without it.

#### Architecture
- **Base model:** `distilbert-base-uncased`
- **Task:** Binary classification (is_prompt_extraction: 0 or 1)
- **Max length:** 512 tokens

#### Training Data
- **Source:** [TensorTrust prompt extraction dataset](https://github.com/HumanCompatibleAI/tensor-trust-data)
- **File:** `prompt_extraction_detection.jsonl` (✅ present at project root, or auto-downloaded)
- **Format:** `{"llm_output": "...", "is_prompt_extraction": true/false}`
- **Split:** 80/20 random split (seed=42)

#### Training Script
- **Script:** `hpc/train_reward_model.py` (84 lines)
- **SLURM:** `hpc/train_reward_model.slurm`

#### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch size | 64 |
| Learning rate | 5e-5 |
| Optimizer | AdamW |
| Loss | CrossEntropyLoss |
| Max length | 512 |

#### Commands

```bash
# Option A: Submit via SLURM
mkdir -p logs pre_trained
sbatch hpc/train_reward_model.slurm

# Option B: Run directly on a GPU node
mkdir -p pre_trained/pi_reward_model
CUDA_VISIBLE_DEVICES=0 python hpc/train_reward_model.py
```

#### Output
```
pre_trained/pi_reward_model/
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── vocab.txt
```

> [!WARNING]
> The training script downloads data from GitHub if not found locally. On HPC compute nodes with no internet, ensure `prompt_extraction_detection.jsonl` exists at the project root **before** submitting the SLURM job. Copy it to the project root on the login node first:
> ```bash
> # On login node (has internet)
> wget -O prompt_extraction_detection.jsonl \
>   "https://github.com/HumanCompatibleAI/tensor-trust-data/raw/main/detecting-extractions/v1/prompt_extraction_detection.jsonl"
> ```

---

### 6.2 Strategy Predictor (MLP)

**Status:** ✅ Already trained — `experiment/strategy_predictor.pth` exists.

This is a custom 4-layer MLP that predicts which attack strategy to use based on defense features.

#### Architecture
```
Linear(input→256) → BN → ReLU → Dropout(0.3)
→ Linear(256→128) → BN → ReLU → Dropout(0.3)
→ Linear(128→64) → BN → ReLU → Dropout(0.2)
→ Linear(64→output)
```
- **Output classes:** 18 attack strategies

#### Training Data
- **Input:** `data/strategy_matrix_raw_v1.jsonl` (28 MB) — ✅ Present

#### Retraining (if needed)

```bash
# Via SLURM
sbatch hpc/train_strategy_predictor.slurm

# Or directly
CUDA_VISIBLE_DEVICES=0 python scripts/training/train_strategy_predictor.py
```

| Parameter | Value |
|-----------|-------|
| Batch size | 128 |
| Learning rate | 0.001 |
| Weight decay | 1e-4 |
| Epochs | 20 |
| Optimizer | Adam |
| Scheduler | ReduceLROnPlateau (patience=2, factor=0.5) |

#### Output
- `experiment/strategy_predictor.pth`
- `experiment/feature_vocab.json`
- `experiment/label_vocab.json`

---

### 6.3 Access Code Predictor (DistilBERT 4-class)

Classifies the type of access code in a defense prompt (TOKEN, MULTILINE, PHRASE, SENTENCE).

#### Architecture
- **Base model:** `distilbert-base-uncased`
- **Task:** 4-class classification
- **Labels:** TOKEN, MULTILINE, PHRASE, SENTENCE
- **Calibration:** Temperature scaling (LBFGS, lr=0.01, max_iter=50)

#### 6.3.1 Prepare Access Code Classifier Dataset

The training script expects a single merged JSONL file, but only split parts exist:

```bash
# Merge the split parts into a single file
cat data/access_code_classifier_dataset_part_aa \
    data/access_code_classifier_dataset_part_ab \
    data/access_code_classifier_dataset_part_ac \
    data/access_code_classifier_dataset_part_ad \
    > data/access_code_classifier_dataset.jsonl

echo "Merged $(wc -l < data/access_code_classifier_dataset.jsonl) lines"
```

#### Training Script
- **Script:** `scripts/training/train_access_code_predictor.py` (205 lines)
- **SLURM:** `hpc/train_ac_predictor.slurm`

#### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch size | 64 |
| Learning rate | 3e-5 |
| Max length | 128 |
| Class weighting | Smoothed (sqrt-based) |
| Early stopping | N/A |

#### Commands

```bash
# Step 1: Merge dataset (if not already done)
cat data/access_code_classifier_dataset_part_a{a,b,c,d} \
    > data/access_code_classifier_dataset.jsonl

# Step 2: Train via SLURM
sbatch hpc/train_ac_predictor.slurm

# Or directly
mkdir -p experiment/access_code_predictor
CUDA_VISIBLE_DEVICES=0 python scripts/training/train_access_code_predictor.py
```

#### Output
```
experiment/access_code_predictor/
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── (calibration files)
```

---

### 6.4 Defense Classifier (DistilBERT)

Classifies the type of defense prompt (translation, password, roleplay, conditional, conversation, etc.).

#### Architecture
- **Base model:** `distilbert-base-uncased`
- **Task:** Multi-class classification
- **Max length:** 128

#### 6.4.1 Prepare Defense Classifier Dataset

The training script expects `data/defense_classifier_dataset.jsonl` but only Part1 and Part2 exist:

```bash
# Merge Part1 and Part2 into a single file
cat data/defense_classifier_dataset-Part1.jsonl \
    data/defense_classifier_dataset-Part2.jsonl \
    > data/defense_classifier_dataset.jsonl

echo "Merged $(wc -l < data/defense_classifier_dataset.jsonl) lines"
```

> [!NOTE]
> To rebuild the defense classifier dataset from scratch (if parts are also missing):
> ```bash
> python scripts/dataset_tools/build_defense_classifier_dataset.py
> ```
> This reads from `experiment/raw_dump_defenses.jsonl.bz2` (✅ present).

#### Training Script
- **Script:** `scripts/training/train_defense_classifier.py` (126 lines)
- **SLURM:** None — submit manually or create one

#### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch size | 32 |
| Warmup steps | 500 |
| Weight decay | 0.01 |
| Max length | 128 |

#### Commands

```bash
# Step 1: Merge dataset
cat data/defense_classifier_dataset-Part{1,2}.jsonl \
    > data/defense_classifier_dataset.jsonl

# Step 2: Train (no SLURM script exists — run manually on GPU node)
mkdir -p models/defense_classifier
CUDA_VISIBLE_DEVICES=0 python scripts/training/train_defense_classifier.py

# Or create a SLURM job inline:
sbatch --job-name=defense_clf --gres=gpu:A100-SXM4:1 --time=2:00:00 \
    --partition=airawatp --output=logs/defense_clf_%j.out \
    --wrap="source .venv/bin/activate && \
    export TRANSFORMERS_OFFLINE=1 && \
    python scripts/training/train_defense_classifier.py"
```

#### Output
```
models/defense_classifier/
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── label_mapping.json
```

---

### 6.5 Ranker (DeBERTa)

Binary classifier that ranks extraction candidates to select the best one. Uses DeBERTa-v3-base.

#### Architecture
- **Base model:** `microsoft/deberta-v3-base`
- **Task:** Binary classification (num_labels=2)
- **Max length:** 512

#### 6.5.1 Build Ranker Dataset

The ranker training dataset does not exist and must be built first from existing data:

```bash
# Build the ranker dataset from success/failure data + benchmark results
python scripts/training/build_ranker_dataset.py \
    --data-dir data/ \
    --results-dir results/ \
    --output-dir data/
```

**Inputs required (all ✅ present):**
- `data/autored_verified_v1.jsonl`
- `data/autored_positive_v1.jsonl`
- `data/autored_extractor_failures_v1.jsonl`
- `data/autored_failures_v1.jsonl`
- `results/` directory (5033 benchmark JSON files ✅ present)

**Outputs:**
- `data/ranker_dataset_train_v1.jsonl`
- `data/ranker_dataset_val_v1.jsonl`
- `data/ranker_dataset_test_v1.jsonl`
- `data/ranker_dataset_v1_metadata.json`

#### Training Script
- **Script:** `scripts/training/train_ranker.py` (331 lines)
- **SLURM:** `hpc/train_ranker.slurm` (automatically builds dataset + trains)

#### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 5 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| FP16 | True |
| Early stopping | patience=2 |
| Loss | Class-weighted CrossEntropy |

#### Commands

```bash
# Recommended: Use the SLURM script (builds dataset + trains)
sbatch hpc/train_ranker.slurm

# Or manually:
# Step 1: Build dataset
python scripts/training/build_ranker_dataset.py \
    --data-dir data/ --results-dir results/ --output-dir data/

# Step 2: Train
CUDA_VISIBLE_DEVICES=0 python scripts/training/train_ranker.py \
    --train-file data/ranker_dataset_train_v1.jsonl \
    --val-file data/ranker_dataset_val_v1.jsonl \
    --test-file data/ranker_dataset_test_v1.jsonl \
    --output-dir models/ranker_deberta_v1/ \
    --model-name microsoft/deberta-v3-base \
    --epochs 5 --batch-size 16 --learning-rate 2e-5
```

#### Output
```
models/ranker_deberta_v1/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

---

### 6.6 Planner SFT (QLoRA)

Fine-tunes Llama-3.1-8B-Lexi with QLoRA to act as a strategy planner — selecting which strategy and primitives to use for each attack step.

#### Architecture
- **Base model:** `Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2`
- **Method:** QLoRA (4-bit NF4 quantization)
- **LoRA:** r=64, alpha=128, dropout=0.05

#### Training Data
- **Train:** `scripts/training/sft_data/planner_v4_train.jsonl` (3.4 MB) — ✅ Present
- **Val:** `scripts/training/sft_data/planner_v4_val.jsonl` (622 KB) — ✅ Present

> [!NOTE]
> To rebuild planner training data from scored trajectories:
> ```bash
> python scripts/dataset_tools/build_oracle_sft_dataset.py
> ```
> This reads `data/scored_trajectories_v4.jsonl` (✅ present) and writes to `scripts/training/sft_data/`.

#### Training Script
- **Script:** `scripts/training/train_qlo.py` (398 lines) — same script as generator QLoRA
- **SLURM:** `hpc/train_planner_sft.slurm`

#### Hyperparameters

| Parameter | Value |
|-----------|-------|
| LoRA r | 64 |
| LoRA alpha | 128 |
| LoRA dropout | 0.05 |
| Target modules | q/k/v/o/gate/up/down_proj |
| Quantization | NF4 4-bit, double quant, bf16 compute |
| Batch size | 2 |
| Gradient accumulation | 16 (effective 32) |
| Learning rate | 2e-5 (cosine + 5% warmup) |
| Max length | 1536 |
| Epochs | 5 |

#### Commands

```bash
# Via SLURM (recommended)
sbatch hpc/train_planner_sft.slurm planner 5

# Or directly
CUDA_VISIBLE_DEVICES=0 python scripts/training/train_qlo.py \
    --model_name "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2" \
    --dataset scripts/training/sft_data/planner_v4_train.jsonl \
    --val_dataset scripts/training/sft_data/planner_v4_val.jsonl \
    --output_dir experiment/results/planner_sft_v4 \
    --epochs 5 --batch_size 2 --gradient_accumulation 16 \
    --learning_rate 2e-5 --lora_r 64 --lora_alpha 128 \
    --max_length 1536 --device_map single --seed 42 \
    --run_name "planner_sft_v4_e5"
```

#### Output
```
experiment/results/planner_sft_v4/
├── adapter_model.safetensors   (LoRA weights)
├── adapter_config.json         (LoRA config)
├── training_config.json        (full hyperparameters)
└── train_metrics.json          (training loss)
```

---

### 6.7 Generator QLoRA (SFT)

Fine-tunes the generator to produce better adversarial prompts, using successful attack examples.

#### Architecture
- **Base model:** `Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2`
- **Method:** QLoRA (same as Planner but different data)

#### Training Data

| Dataset | File | Samples | Purpose |
|---------|------|---------|---------|
| **Verified** | `variantc_verified_train/val.jsonl` | 112 train / 26 val | High-quality verified attacks |
| **Positive** | `variantc_positive_train/val.jsonl` | ~233 train / ~58 val | Broader positive set |

Both ✅ present in `scripts/training/sft_data/`.

#### SLURM Scripts
- `hpc/train_qlo_verified.slurm` — trains on verified data (smaller, higher quality)
- `hpc/train_qlo_positive.slurm` — trains on positive data (larger)

#### Hyperparameters

| Parameter | Value |
|-----------|-------|
| LoRA r | 64 |
| LoRA alpha | 128 |
| LoRA dropout | 0.05 |
| Batch size | 4 |
| Gradient accumulation | 8 (effective 32) |
| Learning rate | 2e-5 |
| Max length | 1024 |
| Epochs | 10 (verified) / 6 (positive) |

#### Commands

```bash
# Verified dataset (recommended starting point)
sbatch hpc/train_qlo_verified.slurm

# Positive dataset (larger training set)
sbatch hpc/train_qlo_positive.slurm

# Or directly (verified example):
CUDA_VISIBLE_DEVICES=0 python scripts/training/train_qlo.py \
    --dataset scripts/training/sft_data/variantc_verified_train.jsonl \
    --val_dataset scripts/training/sft_data/variantc_verified_val.jsonl \
    --output_dir experiment/results/qlo_verified_v1 \
    --epochs 10 --batch_size 4 --gradient_accumulation 8 \
    --learning_rate 2e-5 --lora_r 64 --lora_alpha 128 \
    --lora_dropout 0.05 --max_length 1024 --seed 42 \
    --run_name "qlo_verified_variantc_v1"
```

#### Output
```
experiment/results/qlo_verified_v1/
├── adapter_model.safetensors
├── adapter_config.json
├── training_config.json
└── train_metrics.json
```

#### Previous Training Results (Reference)

| Metric | Value |
|--------|-------|
| Train samples | 112 |
| Val samples | 26 |
| Epochs | 10 |
| Optimizer steps | 40 |
| Final train loss | 1.808561 |
| Final eval loss | 1.792 |

---

### 6.8 Generator DPO

Preference optimization using chosen/rejected attack pairs. Runs **after** SFT.

#### Architecture
- **Base model:** `Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2`
- **Method:** DPO with QLoRA

#### Training Data
- **Input:** `data/generator_dpo_dataset.jsonl` (7.5 MB, 3,627 pairs) — ✅ Present
- **Format:** `{"prompt": [...], "chosen": [...], "rejected": [...]}`

#### SLURM Script
- `hpc/train_dpo.slurm`

#### Hyperparameters

| Parameter | Value |
|-----------|-------|
| DPO beta | 0.1 |
| LoRA r | 64 |
| Batch size | 2 |
| Gradient accumulation | 8 |
| Learning rate | 1e-5 |
| Epochs | 1 |
| Max prompt length | max_length // 2 |

#### Commands

```bash
# Via SLURM
sbatch hpc/train_dpo.slurm

# Or directly
CUDA_VISIBLE_DEVICES=0 python scripts/training/train_dpo.py \
    --model_name "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2" \
    --dataset data/generator_dpo_dataset.jsonl \
    --output_dir experiment/generator_dpo_adapter \
    --epochs 1 --batch_size 2 --gradient_accumulation 8 \
    --lora_r 64 --learning_rate 1e-5
```

#### Output
```
experiment/generator_dpo_adapter/
├── adapter_model.safetensors
├── adapter_config.json
└── (training logs)
```

---

### 6.9 Merging LoRA Adapters (Post-Training)

After training any QLoRA adapter, merge it into the base model for standalone inference or vLLM:

```bash
# Via SLURM
sbatch hpc/merge_lora.slurm

# Or directly
python scripts/training/merge_lora.py \
    --base-model Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2 \
    --adapter experiment/results/qlo_verified_v1 \
    --output experiment/results/qlo_verified_merged
```

> [!NOTE]
> The vLLM backend (`llama_3_8b_vllm.py`) auto-detects LoRA adapters and fuses them into `{path}_fused/` at runtime. Manual merge is not needed for vLLM benchmarks.

---

## 7. Benchmark Execution

### 7.1 Two Backends

| Backend | Script | When to Use |
|---------|--------|------------|
| **Transformers** | `experiment/llama_3_8b_verbose.py` | Default, well-tested, 7 strategies |
| **vLLM** | `experiment/llama_3_8b_vllm.py` | Faster inference, 17 strategies, LoRA auto-fuse |

### 7.2 Single Scenario Test

```bash
# Transformers backend
CUDA_VISIBLE_DEVICES=0 python experiment/llama_3_8b_verbose.py --mode single

# vLLM backend
CUDA_VISIBLE_DEVICES=0 python experiment/llama_3_8b_vllm.py --mode single
```

### 7.3 Multi-GPU Benchmark (Transformers)

```bash
sbatch hpc/autored_benchmark_4gpu.sh 1000 \
    "experiment/results/qlo_verified_v1" \
    "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2" \
    "" 1000 "results/benchmarks/qlo_verified_1000r"
```

**Arguments:** `NUM_ROUNDS GENERATOR_PATH BASE_MODEL_PATH DATASET_PATH DATASET_SIZE OUTPUT_DIR`

### 7.4 Multi-GPU Benchmark (vLLM)

```bash
sbatch hpc/autored_benchmark_4gpu_vllm.sh 1000 \
    "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2" \
    "" "" 1000 "results/benchmarks/benchmark_vllm_1000r"
```

### 7.5 CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | required | `single`, `benchmark`, `extractor_benchmark` |
| `--rounds` | 70 | Number of benchmark scenarios |
| `--dataset-size` | 1000 | Size of scenario pool to sample from |
| `--generator-path` | Lexi base | Path to generator (or LoRA adapter) |
| `--base-generator-path` | — | Base model if generator-path is a LoRA adapter |
| `--benchmark-output` | — | Output JSON file path |
| `--worker-id` | 0 | Worker ID for multi-GPU |
| `--num-workers` | 1 | Total workers for multi-GPU |
| `--dataset-path` | — | Custom dataset path (vLLM only) |
| `--extractor-ranker-path` | — | Custom ranker model path |

### 7.6 Evaluation After Training

```bash
# Baseline (no fine-tuning)
CUDA_VISIBLE_DEVICES=0 python experiment/llama_3_8b_verbose.py \
    --mode benchmark --rounds 1000 --dataset-size 1000 \
    --generator-path Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2 \
    --benchmark-output results/benchmarks/baseline_generator_v1_summary.json \
    2>&1 | tee logs/baseline_generator_v1.log

# QLoRA verified adapter
CUDA_VISIBLE_DEVICES=0 python experiment/llama_3_8b_verbose.py \
    --mode benchmark --rounds 1000 --dataset-size 1000 \
    --generator-path experiment/results/qlo_verified_v1 \
    --base-generator-path Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2 \
    --benchmark-output results/benchmarks/qlo_verified_v1_summary.json \
    2>&1 | tee logs/qlo_verified_v1_benchmark.log
```

---

## 8. Server & Frontend

### 8.1 Architecture

```
FastAPI Backend (port 8001) ←→ Vite Frontend (port 3000)
     │
     ├── server/main.py              (endpoints, lifespan)
     ├── server/experiment_server.py  (experiment orchestration)
     ├── server/models_server.py      (model loading)
     ├── server/schemas.py            (Pydantic schemas)
     ├── server/file_manager.py       (results I/O)
     ├── server/run_normalizer.py     (run normalization)
     └── server/websocket.py          (real-time updates)
```

### 8.2 Running Backend + Frontend on HPC

```bash
sbatch hpc/run_all.slurm
```

### 8.3 Port Forwarding

```bash
ssh -L 8001:localhost:8001 -L 3000:localhost:3000 username@nls.iastate.edu
```

### 8.4 Environment Variables

| Variable | Purpose |
|----------|---------|
| `AUTORED_SERVER_MODE=1` | Enable server mode |
| `AUTORED_LOAD_MODELS=1` | Force load models in server mode |

---

## 9. Post-Processing & Analysis

### 9.1 Extract Successes from Benchmark Results

```bash
python scripts/dataset_tools/autored_successes_logger.py \
    --mode post-process --input results/

python scripts/dataset_tools/autored_successes_logger.py --mode view
```

### 9.2 Dataset Analysis

```bash
python scripts/dataset_tools/analyze_dataset.py --mode all
python scripts/dataset_tools/analyze_dataset.py --mode build
python scripts/dataset_tools/analyze_dataset.py --mode features
python scripts/dataset_tools/analyze_dataset.py --mode strategies
```

### 9.3 Build Training Datasets

```bash
# Generator SFT dataset
python scripts/training/build_generator_sft_dataset.py

# DPO preference pairs
python scripts/training/build_dpo_dataset.py

# Ranker training data
python scripts/training/build_ranker_dataset.py \
    --data-dir data/ --results-dir results/ --output-dir data/

# Strategy predictor data
python scripts/training/build_strategy_predictor_dataset.py

# Planner/Generator v4 SFT data
python scripts/dataset_tools/build_oracle_sft_dataset.py
```

### 9.4 Merge Multi-GPU Results

```bash
python scripts/merge_benchmarks.py \
    --output results/benchmarks/merged_summary.json \
    --worker-results results/benchmarks/worker_*.json
```

---

## 10. SLURM Job Reference

| Script | Purpose | GPUs | Time | Partition |
|--------|---------|------|------|-----------|
| `run_all.slurm` | Backend + Frontend | 1 A100 | 24h | `gpu` |
| `autored_benchmark_4gpu.sh` | 4-GPU benchmark (transformers) | 4 A100 | 7d | `airawatp` |
| `autored_benchmark_4gpu_vllm.sh` | 4-GPU benchmark (vLLM) | 4 A100 | 7d | `airawatp` |
| `super_oracle_benchmark.sh` | Oracle benchmark pipeline | 4 A100 | — | `airawatp` |
| `train_reward_model.slurm` | **Judge** (PI reward model) | 1 GPU | 1d | `airawatp` |
| `train_strategy_predictor.slurm` | Strategy predictor MLP | 1 GPU | 1h | `airawatp` |
| `train_ac_predictor.slurm` | **Access code predictor** | 1 A100 | 7d | `airawatp` |
| `train_ranker.slurm` | **Ranker** (builds dataset + trains) | 1 A100 | 2h | `airawatp` |
| `train_qlo_verified.slurm` | **Generator QLoRA** (verified) | 1 A100 | 4h | `airawatp` |
| `train_qlo_positive.slurm` | Generator QLoRA (positive) | 1 A100 | 4h | `airawatp` |
| `train_planner_sft.slurm` | **Planner SFT** | 1 A100 | 4h | `airawatp` |
| `train_dpo.slurm` | Generator DPO | 1 A100 | 12h | `airawatp` |
| `train_generator_sft.slurm` | Generator SFT (older) | 1 A100 | 1h | `airawatp` |
| `merge_lora.slurm` | Merge LoRA into base | 1 A100 | — | `airawatp` |
| `run_server.slurm` | Backend only | 1 A100 | — | `gpu` |
| `audit_extractor.slurm` | Extractor auditing | 1 A100 | — | — |

```bash
# Submit jobs
mkdir -p logs
sbatch hpc/<script_name>

# Monitor
squeue -u $USER
sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS

# Logs
tail -f logs/<prefix>_<JOBID>.out
```

---

## 11. Troubleshooting

### 11.1 Common Issues

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce batch size or use `CUDA_VISIBLE_DEVICES` |
| `OSError: Can't load tokenizer` offline | Model not pre-downloaded. Run download on login node |
| `TRL/Transformers API mismatch` | `train_qlo.py` patches automatically; ensure TRL ≥ 0.9.0 |
| `device_map="auto"` shards across GPUs | Use `--device_map single` for single-GPU training |
| `vLLM torch.compile memory` | `VLLM_USE_V1=0` (already in vllm script) |
| `peft EmbeddingParallel error` | vLLM script includes monkey patch |
| `Judge identical logits` | Known for empty responses; not a blocker |
| `NFS I/O contention` | Workers staggered by 5s in multi-GPU scripts |

### 11.2 Path Differences

> [!WARNING]
> Two HPC user directories appear in the codebase:
> - `/nlsasfs/home/isea/isea11/slurmJobs/AutoRed` — in `llama_3_8b_verbose.py`, `train_dpo.slurm`
> - `/nlsasfs/home/isea/isea13/AutoRed` — in `train_qlo_verified.slurm`, `run_all.slurm`, `train_ranker.slurm`
>
> Ensure **all paths** point to your current project location. The vLLM variant uses **relative paths** and is more portable.

### 11.3 Quick Sanity Check

```bash
# Fastest verification — single scenario
CUDA_VISIBLE_DEVICES=0 python experiment/llama_3_8b_verbose.py --mode single

# 10-scenario probe
CUDA_VISIBLE_DEVICES=0 python experiment/llama_3_8b_verbose.py \
    --mode benchmark --rounds 10 --dataset-size 100 \
    --benchmark-output results/benchmarks/probe10.json
```

---

## 12. Full Re-Setup Checklist

### Environment
- [ ] Clone repository to HPC
- [ ] Create `.venv` and activate
- [ ] Install `requirements.txt` (with CUDA 12.4 index)
- [ ] Install `requirements_qlo.txt`
- [ ] Install `requirements_server.txt`
- [ ] Install `vllm` (if using vLLM backend)
- [ ] Install `faiss-cpu`, `sentence-transformers`, `scikit-learn`
- [ ] Create dirs: `logs/`, `results/`, `models/`, `pre_trained/`, `experiment/results/`

### HuggingFace Models (download on login node)
- [ ] `meta-llama/Meta-Llama-3-8B-Instruct`
- [ ] `Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2`
- [ ] `distilbert-base-uncased`
- [ ] `microsoft/deberta-v3-base`
- [ ] `sentence-transformers/all-MiniLM-L6-v2`
- [ ] `roberta-large`

### Dataset Preparation
- [ ] Merge access code classifier parts: `cat data/access_code_classifier_dataset_part_a{a,b,c,d} > data/access_code_classifier_dataset.jsonl`
- [ ] Merge defense classifier parts: `cat data/defense_classifier_dataset-Part{1,2}.jsonl > data/defense_classifier_dataset.jsonl`
- [ ] Ensure `prompt_extraction_detection.jsonl` exists at project root (for judge training)
- [ ] Build ranker dataset: `python scripts/training/build_ranker_dataset.py --data-dir data/ --results-dir results/ --output-dir data/`

### Train Models (submit to SLURM in this order)
- [ ] **Judge:** `sbatch hpc/train_reward_model.slurm` → `pre_trained/pi_reward_model/`
- [ ] **Access Code Predictor:** `sbatch hpc/train_ac_predictor.slurm` → `experiment/access_code_predictor/`
- [ ] **Defense Classifier:** Run manually → `models/defense_classifier/`
- [ ] **Ranker:** `sbatch hpc/train_ranker.slurm` → `models/ranker_deberta_v1/`
- [ ] **Planner SFT:** `sbatch hpc/train_planner_sft.slurm planner 5` → `experiment/results/planner_sft_v4/`
- [ ] **Generator QLoRA:** `sbatch hpc/train_qlo_verified.slurm` → `experiment/results/qlo_verified_v1/`
- [ ] **Generator DPO:** `sbatch hpc/train_dpo.slurm` → `experiment/generator_dpo_adapter/` (optional)

### Existing Checkpoints (verify these exist)
- [ ] `experiment/strategy_predictor.pth` (✅ already present)
- [ ] `experiment/feature_vocab.json` (✅ already present)
- [ ] `experiment/label_vocab.json` (✅ already present)

### Data Files (verify these exist)
- [ ] `experiment/raw_dump_defenses.jsonl.bz2` (✅ 7.2 MB)
- [ ] `experiment/oracle_v3_scenarios_5000.jsonl.bz2` (✅ 881 KB)
- [ ] `data/autored_kb.db` (✅ 113 MB)
- [ ] `data/rag/success_defenses.index` (✅ 15 MB)
- [ ] `data/rag/success_metadata.json` (✅ 12 MB)

### Validation
- [ ] Run single-scenario sanity check
- [ ] Submit first benchmark job
