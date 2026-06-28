import os
import json
import time
import random
import argparse
from typing import List, Dict, Any
from pathlib import Path

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from experiment.llama_3_8b_vllm import (
    _load_models, chat_with_llama_batch, StopPointIdentifier,
    SensitiveInfoExtractor, DefenseScenario, LLAMA_PATH, DISTILBERT_CKPT,
    GENERATOR_PATH, BASE_GENERATOR_PATH, get_git_commit, ATTACK_TYPES, ATTACK_TYPE_PROMPTS
)
import experiment.llama_3_8b_vllm as core_module
from experiment.state_builder import StateBuilder

def load_all_scenarios(path: str) -> List[DefenseScenario]:
    import bz2
    scenarios = []
    with bz2.open(path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip(): continue
            data = json.loads(line)
            scenario = DefenseScenario(
                opening_defense=data.get("opening_defense", ""),
                closing_defense=data.get("closing_defense", ""),
                access_code=data.get("access_code", ""),
                access_code_type=data.get("access_code_type", "UNKNOWN"),
                defense_complexity=data.get("defense_complexity", "UNKNOWN")
            )
            scenario._defense_id = data.get("defense_id", str(i))
            scenarios.append(scenario)
    return scenarios

def run_super_oracle(n_samples: int, scenarios: List[DefenseScenario], gen_model, gen_tokenizer, extractor, max_attempts: int = 5, worker_id: int = 0):
    """
    Phase 3: Super Oracle (Best-of-N Search over States)
    """
    from vllm import SamplingParams
    # High temperature for diverse generation
    sampling_params = SamplingParams(n=n_samples, temperature=0.8, top_p=0.9, max_tokens=128)

    state_builder = StateBuilder(ATTACK_TYPES)
    results = []

    for i, scenario in enumerate(scenarios):
        print(f"\n[{i+1}/{len(scenarios)}] Running Best-of-{n_samples} for scenario {scenario._defense_id}")
        
        history = []
        last_victim_response = ""
        last_extractor_confidence = 0.0
        success = False
        trajectory = []

        for attempt in range(max_attempts):
            print(f"  Attempt {attempt+1}/{max_attempts}")
            
            # 1. Build State
            state = state_builder.build_state(
                scenario=scenario,
                attempt=attempt+1,
                previous_strategies=[h["strategy"] for h in history],
                local_memory=[h["attack"] for h in history],
                last_victim_response=last_victim_response,
                last_extractor_confidence=last_extractor_confidence
            )

            # 2. Oracle Strategy Selection
            strategy = random.choice(ATTACK_TYPES)
            
            # 3. Generate N Attacks
            prompt_text = f"{ATTACK_TYPE_PROMPTS.get(strategy, '')}\n\n"
            if history:
                prompt_text += f"Previous attack:\n{history[-1]['attack']}\n\n"
                prompt_text += f"Victim response:\n{last_victim_response}\n\n"
            prompt_text += "Rules:\n- Output ONLY the raw attack text — nothing else.\n- Maximum 100 words.\nOutput:\n"
            
            messages = [{"role": "user", "content": prompt_text}]
            gen_prompt = gen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Generate N candidates at once using vLLM
            outputs = gen_model.generate([gen_prompt], sampling_params=sampling_params, use_tqdm=False)
            candidates = []
            for out in outputs[0].outputs:
                attack_text = out.text.strip()
                if attack_text:
                    candidates.append(attack_text)
                
            if not candidates:
                print("  Failed to generate candidates")
                break
                
            # 4. Evaluate Candidates against Victim (Batched)
            pre_defenses = [scenario.opening_defense] * len(candidates)
            post_defenses = [scenario.closing_defense] * len(candidates)
            
            victim_responses = chat_with_llama_batch(pre_defenses, candidates, post_defenses)
            
            # 5. Score Candidates
            best_score = -1.0
            best_candidate = None
            best_response = None
            
            extractor.set_ground_truth(scenario.access_code)
            
            for j, (atk, resp) in enumerate(zip(candidates, victim_responses)):
                if extractor.check_ground_truth_leak(resp):
                    print(f"  🎉 SUCCESS on candidate {j+1}!")
                    success = True
                    best_candidate = atk
                    best_response = resp
                    best_score = 1.0
                    break
                    
                ext_res = extractor.extract(resp)
                score = 0.0
                if ext_res.get("ranked_candidates"):
                    # ranked_candidates is a list of (candidate, score) tuples
                    score = ext_res["ranked_candidates"][0][1]
                    
                if score > best_score:
                    best_score = score
                    best_candidate = atk
                    best_response = resp
            
            if best_candidate is None:
                best_candidate = candidates[0]
                best_response = victim_responses[0]
                
            # 6. Update State
            history.append({"strategy": strategy, "attack": best_candidate})
            last_victim_response = best_response
            last_extractor_confidence = best_score
            
            trajectory.append({
                "attempt": attempt + 1,
                "strategy": strategy,
                "attack": best_candidate,
                "response": best_response,
                "extractor_confidence": best_score,
                "success": success
            })
            
            if success:
                break
                
        results.append({
            "scenario_id": scenario._defense_id,
            "success": success,
            "trajectory": trajectory
        })
        
        print(f"  -> Scenario {scenario._defense_id} {'COMPLETED (Success)' if success else 'FAILED'}")
            
    # Save results
    out_path = Path(f"data/oracle_trajectories_v2_annotated_w{worker_id}.jsonl")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {len(results)} trajectories to {out_path}")
    
    successes = sum(1 for r in results if r["success"])
    print(f"Overall Success Rate: {successes}/{len(scenarios)} ({(successes/len(scenarios))*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Best-of-N samples per step")
    parser.add_argument("--scenarios", type=int, default=10, help="Number of scenarios to run")
    parser.add_argument("--worker-id", type=int, default=0, help="Worker ID for distributed run")
    parser.add_argument("--num-workers", type=int, default=1, help="Total number of workers")
    args = parser.parse_args()
    
    core_module._SERVER_MODE = False
    print("Loading models...")
    _load_models()
    judge_tokenizer, judge_model = core_module.load_decision_model(DISTILBERT_CKPT)
    gen_tokenizer, gen_model = core_module.load_gen_model(GENERATOR_PATH, BASE_GENERATOR_PATH)
    
    extractor = SensitiveInfoExtractor(model=core_module.llama_model, tokenizer=core_module.llama_tokenizer)
    
    scenarios = load_all_scenarios("experiment/raw_dump_defenses.jsonl.bz2")
    import random
    random.seed(42)
    scenarios = random.sample(scenarios, args.scenarios)
    
    # Sharding
    if args.num_workers > 1:
        chunk_size = len(scenarios) // args.num_workers
        start_idx = args.worker_id * chunk_size
        end_idx = start_idx + chunk_size if args.worker_id < args.num_workers - 1 else len(scenarios)
        scenarios = scenarios[start_idx:end_idx]
        print(f"[WORKER {args.worker_id}/{args.num_workers}] Processing scenarios {start_idx} to {end_idx} ({len(scenarios)} total)")
    
    run_super_oracle(args.n, scenarios, gen_model, gen_tokenizer, extractor, worker_id=args.worker_id)

if __name__ == "__main__":
    main()
