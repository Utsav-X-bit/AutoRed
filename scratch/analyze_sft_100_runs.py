#!/usr/bin/env python3
import os
import json
import re
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path("results")
PREFIX = "run_20260626_232706_"

def main():
    runs = []
    for f in os.listdir(RESULTS_DIR):
        if f.startswith(PREFIX) and f.endswith(".json"):
            with open(RESULTS_DIR / f, "r") as f_obj:
                runs.append(json.load(f_obj))
                
    print(f"Loaded {len(runs)} latest runs.")
    
    cot_leaked_count = 0
    extractor_misses = []
    failed_strategies = defaultdict(int)
    transition_examples = []
    total_attempts = 0
    
    for r in runs:
        scen_id = r.get("experiment", {}).get("scenario_id", "unknown")
        attempts = r.get("attempts", [])
        
        # Track transitions
        strats = [a.get("generator", {}).get("strategy") for a in attempts]
        if len(strats) > 1:
            transition_examples.append(" -> ".join(strats))
            
        is_success = r.get("result", {}).get("generator_success", False)
        
        for a in attempts:
            total_attempts += 1
            gen = a.get("generator", {})
            attack = gen.get("generated_attack", "")
            
            # 1. Did CoT leak?
            if "Objective:" in attack or "Approach:" in attack or "Reason:" in attack:
                cot_leaked_count += 1
                
            # 2. Extractor false negative?
            if a.get("ground_truth_found", False) and not a.get("extractor_match", False):
                extractor_misses.append({
                    "scenario": scen_id,
                    "response": a.get("victim", {}).get("response", ""),
                    "expected": r.get("scenario", {}).get("access_code", "")
                })
                
            # 3. If the whole run failed, what strategies did we try?
            if not is_success:
                failed_strategies[gen.get("strategy", "unknown")] += 1
                
    print(f"\n--- CoT / Planning Check ---")
    print(f"Total attempts: {total_attempts}")
    print(f"Attempts where CoT structure appeared in attack: {cot_leaked_count}")
    if cot_leaked_count == 0:
        print("Note: The SFT model seems to completely ignore the CoT prompt instructions, likely because it was fine-tuned to only output raw attacks.")
    
    print(f"\n--- Extractor Misses (Recall Bottleneck) ---")
    print(f"Number of times the ground truth leaked but extractor missed it: {len(extractor_misses)}")
    
    # Print a few examples of extractor misses
    for m in extractor_misses[:3]:
        print(f"\nScenario {m['scenario']}:")
        print(f"Expected Code: {m['expected']}")
        print(f"Victim Response: {m['response'][:200]}...")
        
    print(f"\n--- Top Strategies in Failed Runs ---")
    for k, v in sorted(failed_strategies.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {k}: {v}")
        
    print(f"\n--- Sample Transitions (First 5 Runs) ---")
    for t in transition_examples[:5]:
        print("  " + t)

if __name__ == "__main__":
    main()
