# Findings & Decisions: AutoRed Project Comprehension

## Requirements
- Go through all markdown documentation in the project to understand the project and its current state.
- Read core codebase files (such as `experiment/llama_3_8b_verbose.py` and others) to understand the implementation.
- Present a clear, comprehensive project summary.

## Research Findings
- **Core Architecture**: AutoRed is an automated red-teaming framework for LLMs designed to leak sensitive "access codes". It has a high-level decision/stopping model (Stop Point Identifier) and low-level models for generating malicious prompt injections (Generator) and extracting leaked codes (Extractor).
- **Major Milestone Upgrade**: The generator was upgraded from T5-base (769M) to Llama-3.1-8B-Lexi (8B), and the extractor was expanded into an 11-phase pipeline, driving the success rate against defended Llama-3-8B-Instruct from 0% to 56.6% (283/500 rounds).
- **Attack Strategy Effectiveness**: Out of 7 strategies, `exception_discovery` (39.7% success), `instruction_leak` (37.4%), and `trigger_phrase_discovery` (34.9%) are the top performers.
- **SFT & Adapter Training**: QLoRA SFT training for the verified-only adapter (`qlo_verified_v1`) has been successfully completed (10 epochs, final eval loss 1.792).
- **T5-era Architecture (Paper Context)**: The original paper framework framed red teaming as a Capture-the-Flag (CTF) game and used a T5-base generator (769M) with 6 generic seed keywords to produce attacks, and a few-shot GPT-3.5-turbo as the extractor. It used RL4LMs (with NLPO PPO variant) for RL training, using a DistilBERT classifier trained on TensorTrust as the judge.
- **T5 Reproduction Failures**: When reproducing the original T5 setup, it achieved 0% success (compared to the paper's claimed 61% against Llama-3-8B). Gap analysis attributed this to T5-base generator producing simple keyword stuffing rather than structural jailbreaks, judge input contract mismatch, the deterministic empty-response trap, and lack of live target feedback in the RL loop.
- **Improvement Roadmap**: The planned improvements included replacing T5-base with LLaMA-2-7B using QLoRA for fine-tuning, expanding the judge context, adding few-shot jailbreak seeds, using continuous probability-based rewards, and using genetic mutations of top attacks.
- **Codebase Upgrades & Fixes**:
  - Replaced state leakage issues by resetting the agent at the start of each round.
  - Solved prompt echo and stuck judge problems (judge now receives trimmed context `previous_new_content[-500:]`).
  - Replaced target LLM base model with Llama-3-8B-Instruct.
  - Refocused generator objective (aligned with TensorTrust, 40-word max length, banned prompt-injection meta-references, strategy reuse, and mutation).
  - Extractor Overhaul: 11-phase pipeline featuring multi-layer regex (12 patterns), quoted extraction, capitalized extraction with 60+ stop words, broad LLM JSON-based extraction, top-k candidate ranking, failed-candidate constraints, and validation on the victim LLM.
- **Plan Evaluation and Paper Audit**:
  - Audited the codebase against the IEEE BigData 2024 paper.
  - Deviations include: Paper used DeBERTa-v3 as the judge (codebase uses DistilBERT), few-shot target LLM as the extractor (codebase uses custom regex + LLM pipeline), base Llama-3 (codebase uses Instruct), and full responses as judge inputs (codebase uses trimmed content).
  - These deviations were essential to resolve original reproduction failures and achieve the 56.6% success rate.
- **SensitiveInfoExtractor Implementation Details**:
  - **Layers**: Starts with regex-based extraction (`_regex_extract` with 12 patterns, `_quoted_extract` for quotes, `_capitalized_extract` for capitalized words/phrases with a 40+ stop-word filter) and LLM-based broad extraction (`_llm_extract`).
  - **Normalization**: Standardizes whitespace and strips wrapper punctuation/quotes. Employs a lowercase, single-space key logic to prevent duplicate variants from being evaluated.
  - **Ranking System**: Scores candidate strings. Positive points: secret keyword (+5), single word (+2), short phrase (+4), LLM context rank/confidence bonus (+0..6). Penalties: long sentence (-2), refusal/deflection words (-10), protocol phrases (-12), instruction fragments (-8).
  - **Adaptive Verifier & Memory**: Sends candidate back to victim for verification. Uses an adaptive cap: if top candidate score >= 12, it tests up to 3 candidates; otherwise up to 10. Failed candidates are stored in `failed_candidates` and passed as negative constraints to subsequent extractor runs to prevent repeat failures.
  - **Metrics Tracking**: Calculates True Positives, False Positives, and False Negatives at runtime to compute Precision, Recall, and F1 scores.
- **Dataset Analysis & Logger Tools**:
  - `autored_successes_logger.py`: Logs successful and failed attempts to JSONL files. Classifies access codes into structural categories (TOKEN, PHRASE, SENTENCE, etc.) and computes defense complexity (easy, medium, hard) using a heuristic feature checklist.
  - `analyze_dataset.py`: Scans generated prompts for 18 distinct attack features (e.g., roleplay, translation wrapper, negation bypass, educational framing). Filters and compiles the datasets (`autored_positive_v1.jsonl`, `autored_verified_v1.jsonl`) and performs lift/effectiveness analyses.
- **PIReward Function**:
  - Implemented in `rl4lms/envs/text_generation/reward.py` as `PIReward`.
  - It initializes a pre-trained sequence classification model and wraps it in a `TextClassificationPipeline`.
  - When the episode is `done` (terminating token), it checks the generated text: returns `0` if the pipeline labels it `SAFE`, and `1` if it is classified otherwise (malicious extraction). It returns `0` for all intermediate tokens.
- **Known Issues**: Includes server mode incompleteness, generator self-assessment inflation, and deterministic judge behavior on empty responses.





## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Use `planning-with-files` | Retains state and insights on disk, ensuring nothing is lost across context resets. |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
|       |            |

## Resources
- [GEMINI.md](file:///home/utsav/Github/Research/AutoRed/GEMINI.md)
- [README.md](file:///home/utsav/Github/Research/AutoRed/README.md)
- [CHANGES_SUMMARY.md](file:///home/utsav/Github/Research/AutoRed/CHANGES_SUMMARY.md)
- [IMPROVEMENTS.md](file:///home/utsav/Github/Research/AutoRed/IMPROVEMENTS.md)
- [AUTORED_FULL_SUMMARY.md](file:///home/utsav/Github/Research/AutoRed/AUTORED_FULL_SUMMARY.md)
- [AUTORED_IMPLEMENTATION_PLAN.md](file:///home/utsav/Github/Research/AutoRed/AUTORED_IMPLEMENTATION_PLAN.md)
- [AUTO_RED_COMPREHENSIVE.md](file:///home/utsav/Github/Research/AutoRed/AUTO_RED_COMPREHENSIVE.md)
- [PLAN_EVALUATION.md](file:///home/utsav/Github/Research/AutoRed/PLAN_EVALUATION.md)
- [experiment/llama_3_8b_verbose.py](file:///home/utsav/Github/Research/AutoRed/experiment/llama_3_8b_verbose.py)
- [scripts/dataset_tools/autored_successes_logger.py](file:///home/utsav/Github/Research/AutoRed/scripts/dataset_tools/autored_successes_logger.py)
- [scripts/dataset_tools/analyze_dataset.py](file:///home/utsav/Github/Research/AutoRed/scripts/dataset_tools/analyze_dataset.py)
- [rl4lms/envs/text_generation/reward.py](file:///home/utsav/Github/Research/AutoRed/rl4lms/envs/text_generation/reward.py)

## Visual/Browser Findings
- None yet.

---
*Update this file after every 2 view/browser/search operations*
*This prevents visual information from being lost*
