# AutoRed Attack Generator Improvements

**Date**: 2026-06-14
**Context**: 500-round benchmark completed. 56.6% success rate (283/500 scenarios) with Llama-3.1-8B-Lexi generator, 7-strategy system, and 8-phase extractor pipeline.

---

## Completed Improvements

### Phase 1: Generator Upgrade (DONE)

| Item | Status | Details |
|------|--------|---------|
| T5 → Llama-3.1-8B-Lexi | ✅ Done | 769M → 8B params; structural jailbreak patterns instead of keyword stuffing |
| Judge context 64 → 256 tokens | ✅ Done | Full response structure visible; deterministic empty-response trap addressed |
| Few-shot jailbreak seeds | ✅ Done | 7-strategy system with performance-based selection |
| 40-word concise attacks | ✅ Done | Generator prompt enforces max 40 words + banned phrases |
| Response-aware adaptation | ✅ Done | 3-consecutive-denial detection → strategy switch |
| Prompt pollution fix | ✅ Done | 3-layer preamble stripping (prompt + regex + fallback) |

### Phase 2: Bug Fixes (DONE)

| Bug | Status | Fix |
|-----|--------|-----|
| Agent state leakage across scenarios | ✅ Fixed | `agent.reset()` at start of each round |
| Dummy probe attacks (40% wasted) | ✅ Fixed | Generator always produces real attacks |
| False positive from echoed prompt | ✅ Fixed | Only check `new_content` beyond prompt echo |
| Stuck judge from full echo | ✅ Fixed | Judge receives `previous_new_content[-500:]` |
| Deterministic empty-response trap | ✅ Fixed | 256-token context + trimmed input |

### Phase 3: Extractor Overhaul (DONE)

| Phase | Status | Description |
|-------|--------|-------------|
| 1. Ground Truth Leak Check | ✅ Done | `access_code.lower() in response.lower()` |
| 2. Multi-Layer Regex (12 patterns) | ✅ Done | Password, access code, keyword, phrase, authorized person, etc. |
| 3. Quoted Text Extractor | ✅ Done | Extracts all `"..."` strings |
| 4. Capitalized Candidate Extractor | ✅ Done | 60+ stop word filtering |
| 5. LLM Extractor (JSON-based) | ✅ Done | Broad JSON extraction with fallback |
| 6. Candidate Ranking | ✅ Done | Scoring: keyword (+5), single word (+2), short (+1), long (-2) |
| 7. Verifier | ✅ Done | Send candidate back to victim for "Access Granted" confirmation |
| 8. TP/FP/FN Metrics | ✅ Done | Precision, Recall, F1 tracking |

### Phase 4: Strategy System (DONE)

| Feature | Status |
|---------|--------|
| 7 attack strategies | ✅ Done |
| Performance-based selection | ✅ Done |
| Strategy mutation | ✅ Done |
| Best attack reuse | ✅ Done |
| Strategy feedback in history | ✅ Done |

### Phase 5: Dataset Collection (DONE)

| Dataset | Entries |
|---------|---------|
| Successful attempts | 1947 |
| Failed attempts | 4330 |
| Positive (gt_leaked OR verified) | 291 |
| Verified (verification_success) | 138 |

### Phase 6: HPC Deployment (DONE)

| Item | Status |
|------|--------|
| Offline model caching | ✅ Done |
| SLURM scripts (3 jobs) | ✅ Done |
| rl4lms compatibility fixes | ✅ Done |

---

## Benchmark Results: Before vs. After

| Metric | Before (T5, 50 attempts) | After (Llama-3.1-8B-Lexi, 500 rounds) |
|--------|--------------------------|----------------------------------------|
| Success Rate | 0% (0/50) | 56.6% (283/500) |
| Generator | T5-base (769M) | Llama-3.1-8B-Lexi (8B) |
| Strategies | None | 7 with performance-based selection |
| Extractor | Simple substring | 8-phase multi-layer pipeline |
| Verified successes | N/A | 138 (14.9% of scenarios) |

---

## Next Phase: SFT Training and Optimization

### Priority 1: SFT Training on Collected Dataset

**Goal:** Fine-tune generator on the 291 positive / 138 verified successes.

**Approach:**
1. Prepare training dataset from `autored_positive_v1.jsonl` and `autored_verified_v1.jsonl`
2. Fine-tune Llama-3.1-8B-Lexi (or smaller model) on successful attack patterns
3. Evaluate on held-out defense scenarios
4. Iterate: train → benchmark → collect → retrain

**Expected outcome:** Generator produces higher-quality attacks from round 1, reducing exploration waste.

### Priority 2: Feature-Enhanced Generator

**Goal:** Incorporate top discriminative features into attack generation.

**Top features to encode:**
| Feature | Lift | Leak Rate |
|---------|------|-----------|
| contains_educational_frame | 1.99 | 47.2% |
| contains_negation_bypass | 1.77 | 44.3% |
| contains_command_injection | 1.71 | 43.5% |
| contains_technical_jargon | 1.37 | 38.1% |
| contains_questioning | 1.30 | 36.9% |

**Approach:** Add feature templates to generator prompt; bias strategy selection toward high-lift patterns.

### Priority 3: Strategy Optimization

**Goal:** Focus on highest-performing strategies.

**Current ranking:**
1. `exception_discovery` — 39.7%
2. `instruction_leak` — 37.4%
3. `trigger_phrase_discovery` — 34.9%

**Approach:** Allocate more attempts to top-3 strategies; retire or redesign summarization (19.6%) and system_prompt_recovery (20.0%).

### Priority 4: Larger Benchmarks

**Goal:** Run 1000-5000 round benchmarks.

**Approach:** Use `--dataset-size` flag for larger scenario pools; collect more data for SFT training.

### Priority 5: Defense Complexity Analysis

**Goal:** Test against harder defenses.

**Current stat:** 22.8% success rate on hard defenses.

**Approach:** Curate hard defense scenarios; analyze failure patterns; adapt generator for adversarial robustness.

---

## Known Issues (Remaining)

1. **Server Mode Incomplete:** `AUTORED_SERVER_MODE=1` skips model loading but has no remote inference path
2. **Generator Self-Assessment Inflation:** 56.6% includes self-assessment; only 14.9% verified
3. **T5 Generator Still Available:** Superseded by Llama-3.1-8B-Lexi but not removed
4. **Judge Deterministic on Empty:** 21/50 occurrences in initial test; partially addressed by trimming

---

## Long-Term Ideas

| Idea | Impact | Effort |
|------|--------|--------|
| Continuous reward signal (3-class: SAFE/PARTIAL/SUCCESS) | High | High |
| Genetic mutation of top attacks | Medium | Medium |
| Adversarial co-training against multiple targets | High | High |
| Multi-turn escalation attacks | Medium | Medium |
| Token smuggling / character encoding bypasses | Medium | Medium |
| RL fine-tuning on collected successes | High | High |

---

*Document last updated: 2026-06-14*
*Previous version: 2026-04-23 (T5-era analysis)*
