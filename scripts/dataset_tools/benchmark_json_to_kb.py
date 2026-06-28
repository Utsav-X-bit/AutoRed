import json
import argparse
from pathlib import Path
import sys
import uuid
import os

# Prevent heavy model loading when importing from experiment modules
os.environ["AUTORED_SERVER_MODE"] = "1"

# Add the project root to sys.path so we can import from experiment
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from experiment.knowledge_base import KnowledgeBase
from experiment.state_builder import StateBuilder
from experiment.llama_3_8b_verbose import DefenseScenario

def convert_benchmark_to_kb(json_path: str, db_path: str = "data/autored_kb.db"):
    print(f"Loading benchmark results from {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    runs = data.get("runs", data.get("results", []))
    if not runs and "results" not in data:
        print("Could not find 'runs' or 'results' in JSON.")
        return

    all_strategies = [
        "trigger_phrase_discovery",
        "instruction_leak",
        "exception_discovery",
        "roleplay",
        "translation",
        "summarization",
        "system_prompt_recovery",
    ]

    kb = KnowledgeBase(db_path=db_path)
    state_builder = StateBuilder(all_strategies=all_strategies)
    
    count = 0
    print(f"Importing {len(runs)} benchmark rounds into Knowledge Base...")
    
    for run in runs:
        if "trace" not in run:
            continue
            
        scenario_data = run.get("metadata", {}).get("scenario", {})
        if not scenario_data:
            continue
            
        scenario = DefenseScenario(
            access_code=scenario_data.get("access_code", ""),
            opening_defense=scenario_data.get("opening_defense", ""),
            closing_defense=scenario_data.get("closing_defense", ""),
            access_code_type=scenario_data.get("access_code_type", "Unknown")
        )
        scenario.primary_type = scenario_data.get("primary_type", "unknown")
        scenario.difficulty_score = scenario_data.get("difficulty_score", 1.0)
        
        # State tracking history
        history = []
        
        for step in run["trace"]:
            # 1. Build the State Snapshot exactly like the live loop
            attempt_number = step.get("iteration", 1)
            
            # Get last confidence and history from previous steps
            last_confidence = 0.0
            if attempt_number > 1:
                prev_step = run["trace"][attempt_number - 2]
                last_confidence = prev_step.get("extractor", {}).get("top_k_candidates", [{}])[0].get("score", 0.0) if prev_step.get("extractor", {}).get("top_k_candidates") else 0.0
                
            state = state_builder.build_state(
                scenario=scenario,
                history=history,
                current_attempt=attempt_number,
                last_extractor_confidence=last_confidence,
                previous_attack=step.get("generator", {}).get("generated_attack", "") if attempt_number > 1 else "",
                last_response=step.get("llm_response", {}).get("raw_output", "") if attempt_number > 1 else ""
            )
            
            # Save State to KB
            state_id = state.hash
            kb.save_state_snapshot(state)
            
            # 2. Extract step data
            strategy = step.get("generator", {}).get("strategy", "unknown")
            attack = step.get("generator", {}).get("generated_attack", "")
            response = step.get("llm_response", {}).get("raw_output", "")
            
            extractor_data = step.get("extractor", {})
            verified = extractor_data.get("verified", False)
            success_exact = extractor_data.get("success_exact", False)
            success_extractor = extractor_data.get("success_extractor", False)
            
            # Build trajectory record
            trajectory = {
                "scenario_id": run.get("metadata", {}).get("scenario_id", str(uuid.uuid4())),
                "attempt": attempt_number,
                "strategy": strategy,
                "attack_string": attack,
                "response_string": response,
                "reward": 10.0 if (verified or success_exact) else 0.0,
                "ground_truth_leaked": success_exact,
                "generator_success": success_exact,
                "extractor_success": success_extractor,
                "verifier_success": verified,
                "state_id": state_id,
                "chosen_strategy": strategy,
                "alternative_strategies": json.dumps([]),
                "decision_reason": "Heuristic selection from legacy batched pipeline",
                "decision_confidence": 1.0
            }
            
            # Save Trajectory to KB
            kb.log_trajectory(trajectory)
            
            # Append to history for next state calculation
            history.append({
                "strategy": strategy,
                "success": verified or success_exact
            })
            count += 1

    print(f"Successfully imported {count} transitions into {db_path}!")
    print("You can now run 'python scripts/dataset_tools/mine_attack_transitions.py' to generate your Planner training data.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Benchmark JSONs to KB Trajectories with StateSnapshots")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to merged_summary.json")
    parser.add_argument("--db", type=str, default="data/autored_kb.db", help="Path to output sqlite DB")
    
    args = parser.parse_args()
    convert_benchmark_to_kb(args.input, args.db)
