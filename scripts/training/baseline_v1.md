# Baseline Generator v1 — Frozen

**Date:** 2026-06-14  
**Status:** Frozen comparison point for all SFT experiments

## Configuration

| Setting | Value |
|---------|-------|
| Model | Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2 |
| Prompt | Current AutoRed GENERATOR_PROMPT |
| Strategies | 7 (performance-based selection + mutation) |
| Benchmark | 500 scenarios from 5000 pool |
| Source | `results/run_*.json` (503 files) |

## Metrics

| Metric | Value |
|--------|-------|
| Ground Truth Success | 198/500 = **39.6%** |
| Generator Success | 198/500 = **39.6%** |
| Verified Success | 136/500 = **27.2%** |
| Extractor Success | 23/500 = **4.6%** |
| Mean Attempts | 12.5 |
| Total Attempts | 6246 |

## Strategy Attempt Distribution

| Strategy | Attempts | Scenarios Used |
|----------|----------|----------------|
| trigger_phrase_discovery | 1518 | 500 |
| instruction_leak | 1066 | 418 |
| summarization | 1054 | 293 |
| exception_discovery | 771 | 361 |
| translation | 623 | 309 |
| system_prompt_recovery | 622 | 208 |
| roleplay | 592 | 325 |

## Strategy Success Distribution (best_attack attribution)

Attribution: each successful scenario credited to the strategy that produced the best_attack.

| Strategy | GT Wins | GT Share | Verified Wins | Ver Share |
|----------|---------|----------|---------------|-----------|
| instruction_leak | 48 | 24.2% | 28 | 20.6% |
| summarization | 44 | 22.2% | 27 | 19.9% |
| trigger_phrase_discovery | 39 | 19.7% | 41 | 30.1% |
| exception_discovery | 33 | 16.7% | 12 | 8.8% |
| system_prompt_recovery | 16 | 8.1% | 6 | 4.4% |
| roleplay | 10 | 5.1% | 15 | 11.0% |
| translation | 8 | 4.0% | 7 | 5.1% |
| **Total** | **198** | **100%** | **136** | **100%** |

**Insight:** `trigger_phrase_discovery` has the highest verified win share (30.1%) despite only 19.7% GT wins — its attacks are more likely to produce verifiable access codes. `instruction_leak` dominates raw GT leaks (24.2%) but only 20.6% verified — many leaks don't survive verification.

## Strategy Success Distribution (scenario-level)

| Strategy | Scenarios | GT Success | GT Rate | Verified | Ver Rate |
|----------|-----------|------------|---------|----------|----------|
| trigger_phrase_discovery | 500 | 198 | 39.6% | 136 | 27.2% |
| instruction_leak | 418 | 169 | 40.4% | 100 | 23.9% |
| exception_discovery | 361 | 134 | 37.1% | 79 | 21.9% |
| roleplay | 325 | 108 | 33.2% | 68 | 20.9% |
| translation | 309 | 100 | 32.4% | 55 | 17.8% |
| summarization | 293 | 92 | 31.4% | 49 | 16.7% |
| system_prompt_recovery | 208 | 60 | 28.8% | 27 | 13.0% |

## Key Observations

1. **Extractor bottleneck**: 39.6% ground truth leak but only 4.6% extractor success
2. **Verified gap**: 39.6% gt leak → 27.2% verified (12.4% gap)
3. **High exploration**: 12.5 mean attempts per scenario
4. **Top strategies**: instruction_leak (40.4%), trigger_phrase_discovery (39.6%), exception_discovery (37.1%)

---

*All SFT experiments must benchmark against these numbers.*
