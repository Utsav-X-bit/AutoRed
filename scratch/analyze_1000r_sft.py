import json
from collections import Counter
import os
import sys

results_path = "/home/utsav/Github/Research/AutoRed/results/benchmarks/batched_1000r_4g/merged_summary.json"

with open(results_path, 'r') as f:
    data = json.load(f)

scenarios = data.get('results', [])
total = len(scenarios)

# Success counters
success_scenarios = [s for s in scenarios if s.get('best_score', 0) > 0]
success_count = len(success_scenarios)
print(f"Total Scenarios: {total}")
print(f"Overall Success Rate: {success_count}/{total} ({(success_count/total)*100:.2f}%)")

# Break down by access code type
all_access_code_types = Counter()
success_access_code_types = Counter()

all_defense_types = Counter()
success_defense_types = Counter()

all_complexities = Counter()
success_complexities = Counter()

# Trace variables
total_attempts = 0
judge_attempt_count = 0
extract_trigger_count = 0
extractor_found_nothing = 0
extractor_found_candidate_but_failed_verification = 0
successful_verifications = 0
victim_refusals = 0

strategy_counts = Counter()
strategy_success = Counter()

for s in scenarios:
    metadata = s.get('metadata', {})
    code_type = metadata.get('access_code_type', 'UNKNOWN')
    def_type = metadata.get('defense_type', 'UNKNOWN')
    comp = metadata.get('defense_complexity', 'UNKNOWN')
    
    all_access_code_types[code_type] += 1
    all_defense_types[def_type] += 1
    all_complexities[comp] += 1
    
    if s.get('best_score', 0) > 0:
        success_access_code_types[code_type] += 1
        success_defense_types[def_type] += 1
        success_complexities[comp] += 1
        
    for trace in s.get('trace', []):
        total_attempts += 1
        
        # Generator Strategy
        strat = trace.get('strategy', 'unknown')
        strategy_counts[strat] += 1
        
        judge_decision = trace.get('judge_decision')
        
        if judge_decision == "ATTEMPT":
            judge_attempt_count += 1
            
            # Did extractor find anything?
            candidates = trace.get('extracted_candidates', [])
            if not candidates:
                extractor_found_nothing += 1
            else:
                # Was it verified?
                verified = any(c.get('success', False) for c in candidates)
                if verified:
                    successful_verifications += 1
                    strategy_success[strat] += 1
                else:
                    extractor_found_candidate_but_failed_verification += 1
        else:
            victim_refusals += 1
            
print("\n--- By Access Code Type ---")
for t, count in all_access_code_types.most_common():
    suc = success_access_code_types[t]
    print(f"{t:30}: {suc:3}/{count:3} ({(suc/count)*100 if count>0 else 0:.1f}%)")

print("\n--- By Defense Complexity ---")
for c, count in all_complexities.most_common():
    suc = success_complexities[c]
    print(f"{c:30}: {suc:3}/{count:3} ({(suc/count)*100 if count>0 else 0:.1f}%)")
    
print("\n--- Strategy Effectiveness (by valid verification) ---")
for strat, count in strategy_counts.most_common():
    suc = strategy_success[strat]
    print(f"{strat:30}: {suc:4}/{count:4} ({(suc/count)*100 if count>0 else 0:.1f}%)")

print("\n--- Attempt Funnel Analysis ---")
print(f"Total Attack Attempts (Generator): {total_attempts}")
print(f"Victim Refused (Judge = ATTACK): {victim_refusals} ({(victim_refusals/total_attempts)*100 if total_attempts>0 else 0:.1f}%)")
print(f"Victim Leaked (Judge = ATTEMPT): {judge_attempt_count} ({(judge_attempt_count/total_attempts)*100 if total_attempts>0 else 0:.1f}%)")
print(f"  -> Extractor found NO candidates: {extractor_found_nothing} ({(extractor_found_nothing/judge_attempt_count)*100 if judge_attempt_count>0 else 0:.1f}%)")
print(f"  -> Extractor found candidates but failed verification: {extractor_found_candidate_but_failed_verification} ({(extractor_found_candidate_but_failed_verification/judge_attempt_count)*100 if judge_attempt_count>0 else 0:.1f}%)")
print(f"  -> Verification Successful: {successful_verifications} ({(successful_verifications/judge_attempt_count)*100 if judge_attempt_count>0 else 0:.1f}%)")

# Let's see some failed verifications (Generator output vs Victim Output)
print("\n--- Sample Failed Verifications ---")
failed_samples = 0
for s in scenarios:
    for trace in s.get('trace', []):
        if trace.get('judge_decision') == "ATTEMPT":
            candidates = trace.get('extracted_candidates', [])
            if candidates and not any(c.get('success', False) for c in candidates):
                print(f"ATTACK: {trace.get('attack')}")
                print(f"VICTIM RESPONSE: {trace.get('response')}")
                print(f"CANDIDATES: {[c.get('value') for c in candidates]}")
                print("-" * 50)
                failed_samples += 1
                if failed_samples >= 5:
                    break
    if failed_samples >= 5:
        break

print("\n--- Sample Missing Extractions (Judge=ATTEMPT, no candidates) ---")
missing_samples = 0
for s in scenarios:
    for trace in s.get('trace', []):
        if trace.get('judge_decision') == "ATTEMPT":
            candidates = trace.get('extracted_candidates', [])
            if not candidates:
                print(f"ATTACK: {trace.get('attack')}")
                print(f"VICTIM RESPONSE: {trace.get('response')}")
                print("-" * 50)
                missing_samples += 1
                if missing_samples >= 5:
                    break
    if missing_samples >= 5:
        break
