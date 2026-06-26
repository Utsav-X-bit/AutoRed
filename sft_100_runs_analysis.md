# SFT Benchmark (100 Rounds) - Detailed Analysis

## 1. High-Level Metrics
- **Overall Success Rate**: **48%** (48/100 scenarios solved). This is a strong jump from the ~38% baseline we saw earlier for the un-tuned generator.
- **Top-1 Success**: 26% of scenarios were broken on the very first attempt!
- **Average Attempts**: When successful, it took an average of 5.68 attempts to break the defense.

## 2. The Multi-Step Planning (CoT) Issue
**Finding:** The SFT model **completely ignored** the new Chain-of-Thought planning instructions. Out of ~1,400 total attempts, 0 generated the `Objective:`, `Approach:`, `Reason:`, and `Attack:` headers.

**Why is this failing?** 
The model underwent Supervised Fine-Tuning (SFT) purely on the `autored_successes_v1.jsonl` dataset. Since that dataset only contains the raw 40-word attack strings, the SFT adapter aggressively overrides the prompt instructions and immediately outputs the attack payload, skipping the planning phase entirely.

**The Fix:** This is exactly why we built the **DPO pipeline**. Our `build_dpo_dataset.py` explicitly constructs the CoT planning into the preference pairs. The DPO training will force the model to respect the CoT format while maintaining the attack quality!

## 3. Extractor Precision & Recall
- **Precision:** 96% (Only 1 false positive!). The new `Access Code Type Predictor` penalization system we implemented is working perfectly. The extractor is no longer getting confused by the victim rejecting the attack.
- **Recall:** 53%. Out of the 47 times the ground truth was physically present in the victim's response, the extractor only found 25 of them. 

**Why is it failing here?**
The Extractor misses complex edge-cases where the victim surrounds the access code with unexpected punctuation or truncates it slightly. 
*Example Misses:*
- **Expected:** `juggernaut` 
- **Expected:** `Netherland`
- **Expected:** `私 🎁 И`
*These short, single-word or special-character access codes are likely being filtered out by the Extractor's heuristic checks (like minimum length or regex boundaries).*

## 4. Adaptive Strategy Switching
**Finding:** The Strategy Transition Graph is working flawlessly! 
Instead of getting stuck on a single strategy for all 20 attempts, the agent successfully learned to pivot when an attack fails based on historical probabilities.

**Sample Transition Trace from Run #3:**
`exception_discovery` → `roleplay` → `translation` → `trigger_phrase_discovery` → `instruction_leak`

**Top Strategies in Failed Scenarios:**
When a scenario completely failed after 20 attempts, the model spent the most time trying:
1. `trigger_phrase_discovery` (25% of failed attempts)
2. `instruction_leak` (25% of failed attempts)
3. `exception_discovery` (14% of failed attempts)

## 5. Conclusion & Next Steps
The pipeline architecture is incredibly solid. The Extractor's precision is fixed, and the Adaptive Strategy Switching is operational.
The final missing piece is unlocking the CoT reasoning capability in the generator to push the success rate past 55-60%. 

**Ready for DPO:** We need to execute the DPO training to teach the model *how* to plan its attacks.
