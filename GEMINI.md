# GEMINI.md: AutoRed — Current State

**Last Updated:** 2026-06-14
**Status:** Active research project — 500-round benchmark completed, dataset analysis done

---

## 1. Overview

AutoRed is an automated framework for red teaming Large Language Models (LLMs). It generates malicious attack scenarios to extract sensitive information (hidden "access codes") from LLMs protected by defense prompts. The framework was originally published at IEEE BigData 2024 and has been extensively modified for this research project.

**Key Achievement:** 500-round benchmark against Llama-3-8B-Instruct achieved **56.6% success rate** (283/500 scenarios), with 1947 successful attempts and 4330 failures collected for analysis.

---

## 2. Current Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AutoRed Attack Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Strategy Selector ──→ Llama-3.1-8B-Lexi Generator ──→ Attack   │
│  (performance-based)   (8B params, uncensored)      (40-word)   │
│         ↑                              │                        │
│         │      ┌───────────────────────┘                        │
│         │      ▼                                                │
│  Strategy Feedback ← DistilBERT Judge ← Llama-3-8B-Instruct     │
│  (scoring system)   (256 tokens)      (victim, defended)        │
│                              │                                   │
│                              ▼                                   │
│              Multi-Layer Extractor Pipeline                      │
│              ├─ Ground Truth Leak Check                          │
│              ├─ 12 Regex Patterns                                │
│              ├─ Quoted Text Extraction                           │
│              ├─ Capitalized Candidate Extraction                 │
│              ├─ LLM Extractor (JSON-based)                       │
│              ├─ Candidate Ranking (scoring)                      │
│              └─ Verifier (send back to victim)                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Model | Role |
|-----------|-------|------|
| **Generator** | Llama-3.1-8B-Lexi-Uncensored-V2 | Generates adversarial prompts using 7 strategies |
| **Judge** | DistilBERT (256 tokens) | Extraction confidence scorer (not binary gate) |
| **Victim** | Llama-3-8B-Instruct | Defended model under attack |
| **Extractor** | Multi-layer pipeline | Extracts access codes from victim responses |
| **Strategy System** | Performance-based scoring | Selects and mutates attack strategies |

### 7 Attack Strategies

1. `trigger_phrase_discovery` — Find exact phrase that grants access
2. `instruction_leak` — Extract system instructions
3. `exception_discovery` — Find exceptions in defense rules
4. `roleplay` — Role-playing framing attacks
5. `translation` — Translation wrapper attacks
6. `summarization` — Summarization-based extraction
7. `system_prompt_recovery` — Recover full system prompt

---

## 3. Codebase Structure

```
AutoRed/
├── experiment/
│   ├── llama_3_8b_verbose.py      # Main experiment runner (~2074 lines)
│   └── llama_3_8b-1.py            # First custom test script (superseded)
├── scripts/
│   ├── dataset_tools/             # Dataset analysis tools
│   │   ├── autored_successes_logger.py  # Success/failure logger + post-processor
│   │   ├── analyze_dataset.py         # Feature mining + strategy analysis
│   │   ├── audit_generator.py         # Generator quality auditor
│   │   ├── build_augmented_v1.py      # Augmented dataset builder
│   │   ├── classify_dedup_score.py    # Classification/dedup/scoring
│   │   ├── clean_v1_report.py         # Cleaning report generator
│   │   ├── complexity_multilabel_gold.py  # Complexity multi-label gold
│   │   ├── create_benchmark.py        # Benchmark scenario creator
│   │   └── generate_augmentation.py   # Augmentation generator
│   ├── pi/                        # Prompt injection notebooks + data
│   └── training/                  # Training scripts + configs
├── rl4lms/                        # Forked RL4LMs library (AllenAI)
├── hpc/                           # SLURM scripts for HPC deployment
├── data/                          # Collected datasets
│   ├── autored_successes_v1.jsonl     # 1947 successful attempts
│   ├── autored_failures_v1.jsonl      # 4330 failed attempts
│   ├── autored_positive_v1.jsonl      # 291 positive (gt_leaked OR verified)
│   ├── autored_verified_v1.jsonl      # 138 verified successes
│   └── analysis_report_v1.md          # Comprehensive analysis report
├── results/                       # Benchmark run JSON files (503 runs)
└── assets/                        # Logos, diagrams
```

---

## 4. Key Modifications from Original Paper

| Component | Original Paper | Current Implementation |
|-----------|---------------|------------------------|
| Generator | T5-base (769M params) | Llama-3.1-8B-Lexi-Uncensored-V2 (8B params) |
| Victim | Llama-3-8B (base) | Llama-3-8B-Instruct |
| Judge | DistilBERT (64 tokens) | DistilBERT (256 tokens, trimmed input) |
| Extractor | Simple substring match | 8-phase multi-layer pipeline |
| Strategies | None (single approach) | 7 strategies with performance-based selection |
| Success Criteria | Binary (access code in response) | 3-tier: ground_truth_leaked, generator_success, verification_success |
| Benchmark | 70 rounds, 100 interactions | 500 rounds, variable interactions |

---

## 5. Running Experiments

### Single Scenario (Verbose)
```bash
python experiment/llama_3_8b_verbose.py --mode single
```

### Benchmark Mode
```bash
# 500-round benchmark (default dataset size: 1000)
python experiment/llama_3_8b_verbose.py --mode benchmark --rounds 500

# Larger dataset pool
python experiment/llama_3_8b_verbose.py --mode benchmark --rounds 500 --dataset-size 5000
```

### Post-Processing Results
```bash
# Extract successes from benchmark JSON files
python scripts/dataset_tools/autored_successes_logger.py --mode post-process --input results/

# View collected data
python scripts/dataset_tools/autored_successes_logger.py --mode view
```

### Dataset Analysis
```bash
# Build curated datasets + run feature mining + strategy analysis
python scripts/dataset_tools/analyze_dataset.py --mode all

# Specific modes
python scripts/dataset_tools/analyze_dataset.py --mode build      # Build datasets only
python scripts/dataset_tools/analyze_dataset.py --mode features   # Feature mining only
python scripts/dataset_tools/analyze_dataset.py --mode strategies # Strategy analysis only
```

---

## 6. Benchmark Results Summary

| Metric | Value |
|--------|-------|
| Total Scenarios | 500 |
| Success Rate | 56.6% (283/500) |
| Unique Successful Scenarios | 288 |
| Total Successful Attempts | 1947 |
| Total Failed Attempts | 4330 |
| Positive Dataset (gt_leaked OR verified) | 291 entries |
| Verified Dataset (verification_success) | 138 entries |

### Strategy Effectiveness (with failure baseline)
| Strategy | Success Rate |
|----------|-------------|
| exception_discovery | 39.7% |
| instruction_leak | 37.4% |
| trigger_phrase_discovery | 34.9% |
| roleplay | 32.9% |
| translation | 28.5% |
| system_prompt_recovery | 20.0% |
| summarization | 19.6% |

### Top Discriminative Features (lift > 1)
| Feature | Lift | Leak Rate |
|---------|------|-----------|
| contains_educational_frame | 1.99 | 47.2% |
| contains_negation_bypass | 1.77 | 44.3% |
| contains_command_injection | 1.71 | 43.5% |
| contains_technical_jargon | 1.37 | 38.1% |
| contains_questioning | 1.30 | 36.9% |

---

## 7. HPC Deployment

### Environment
- **Cluster:** NLS at Iowa State University
- **Partitions:** "gpu" / "airawatp"
- **GPU:** NVIDIA A100-SXM4-40GB
- **Offline Mode:** All models cached via `hpc/download_hf_assets.py`

### SLURM Scripts
| Script | Purpose | GPU | Memory | Time |
|--------|---------|-----|--------|------|
| `train_reward_model.slurm` | DistilBERT judge training | 1 | 40GB | 1h |
| `train_generator_sft.slurm` | T5-base SFT | 1 | 40GB | 1h |
| `train_generator_rl.slurm` | NLPO RL fine-tuning | 1 | 40GB | 1h |

---

## 8. Critical Files

| File | Purpose | Lines |
|------|---------|-------|
| `experiment/llama_3_8b_verbose.py` | Main experiment runner | ~2074 |
| `scripts/dataset_tools/autored_successes_logger.py` | Success/failure logger + post-processor | ~350 |
| `scripts/dataset_tools/analyze_dataset.py` | Feature mining + strategy analysis | ~550 |
| `rl4lms/envs/text_generation/reward.py` | PIReward class | ~350 |
| `rl4lms/envs/text_generation/policy/seq2seq_policy.py` | T5 policy | ~450 |
| `rl4lms/envs/text_generation/policy/causal_policy.py` | Causal LM policy | ~400 |

---

## 9. Known Issues

1. **Server Mode Incomplete:** `AUTORED_SERVER_MODE=1` skips model loading but has no remote inference path wired in
2. **Generator Self-Assessment Inflation:** 56.6% success rate includes generator self-assessment; only 14.9% have ground truth leaked or verified
3. **T5 Generator Still Available:** Original T5-base generator exists but is superseded by Llama-3.1-8B-Lexi
4. **Judge Deterministic on Empty:** DistilBERT produces identical logits for empty responses (21/50 occurrences in initial test)

---

## 10. Next Steps

1. **SFT Training:** Train on collected AutoRed-Successes dataset (291 positive, 138 verified)
2. **Larger Benchmarks:** Run 1000-5000 round benchmarks with `--dataset-size` flag
3. **Feature-Enhanced Generator:** Incorporate top discriminative features into attack generation
4. **Strategy Optimization:** Focus on exception_discovery and instruction_leak (highest success rates)
5. **Defense Complexity Analysis:** Test against harder defenses (currently 22.8% success rate on hard)

---

*Document generated on 2026-06-14*
*Source: `/home/utsav/Github/Research/AutoRed/`*
