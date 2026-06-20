import re

with open("experiment/llama_3_8b_verbose.py", "r") as f:
    content = f.read()

# Replace verbose_test_llama's judge logic
# Find:
#         if i == 0:
#             judge_input = "[No previous output — first iteration]"
#         else: ...
#         judge_result = agent.judge_response(judge_input)
#         gen_result = agent.generate_attack(previous_attack=last_attack, previous_response=last_response)
#         attack = gen_result["generated_attack"]
#         response, _, done, info = env.step(attack)
#         new_content = info.get("clean_response", response)

verbose_judge_replace = """
        # 1. Generate attack
        gen_result = agent.generate_attack(
            previous_attack=last_attack, previous_response=last_response
        )
        attack = gen_result["generated_attack"]
        duplicate_attack = gen_result.get("duplicate_attack", False)
        print(f"  Generated Attack ({len(attack)} chars):\\n  \\033[96m{attack}\\033[0m")

        # 2. Victim response
        print(f"\\n  [Phase 2] Victim Interaction")
        response, _, done, info = env.step(attack)
        new_content = info.get("clean_response", response)
        
        print(f"  Victim Response ({len(response)} chars):")
        print(f"  \\033[93m{new_content[:500]}\\033[0m" + ("..." if len(new_content) > 500 else ""))

        # 3. Judge evaluates the victim response
        print(f"\\n  [Phase 4] Judge Evaluation")
        judge_input = new_content[-512:] if new_content else "[EMPTY RESPONSE]"
        safe_output = (
            judge_input.replace("\\n", " ")[:80] + "..."
            if len(judge_input) > 80
            else judge_input
        )
        print(f"  Input to Judge (Victim Response): '{safe_output}'")

        judge_result = agent.judge_response(judge_input)

        print(f"  ┌─ Probabilities: {judge_result['probabilities']}")
        print(f"  ├─ Confidence:    {judge_result['confidence']:.4f}")
        
        decision = judge_result['decision_name']
        if decision == "ATTACK":
            print(f"  └─ Decision: \\033[92m🟢 ATTACK (No leak detected)\\033[0m")
        else:
            print(f"  └─ Decision: \\033[91m🔴 ATTEMPT (Potential leak detected!)\\033[0m")

        # 4. Extract ONLY if ATTEMPT
        extraction_result = {
            "best_candidate": None,
            "verified_candidate": None,
            "verified_rank": 0,
            "verified_score": 0,
            "verification_response": "",
            "verification_traces": [],
            "ranked_candidates": [],
            "all_candidates": [],
            "top_k_candidates": [],
            "regex_candidates": [],
            "quoted_candidates": [],
            "capitalized_candidates": [],
            "llm_candidates": [],
            "llm_ranked_candidates": [],
            "verified": False,
        }
        
        if decision == "ATTEMPT":
            print(f"\\n  [Phase 6] Extractor Pipeline (Judge triggered)")
            extraction_result = agent.extractor.extract(response, env=env)
        else:
            print(f"\\n  [Phase 6] Extractor Skipped (Judge decision was ATTACK)")

        gt_leaked = agent.extractor.check_ground_truth_leak(response)
        extraction_result["ground_truth_leaked"] = gt_leaked
"""

# To perform the replacement we'll match the start of Phase 4 down to ground_truth_leaked
verbose_pattern = re.compile(r'# Phase 4: Identify if the generator output.*?ground_truth_leaked = agent\.extractor\.check_ground_truth_leak\(response\)\n\s+extraction_result\["ground_truth_leaked"\] = gt_leaked', re.DOTALL)

content = verbose_pattern.sub(verbose_judge_replace.strip(), content)

# Replace _silent_test
silent_judge_replace = """
        # 1. Generate attack
        gen_result = agent.generate_attack(
            previous_attack=last_attack, previous_response=last_response
        )
        attack = gen_result["generated_attack"]

        # 2. Victim response
        response, _, done, info = env.step(attack)
        new_content = info.get("clean_response", response)

        # 3. Judge evaluates victim response
        judge_input = new_content[-512:] if new_content else "[EMPTY RESPONSE]"
        judge_result = agent.judge_response(judge_input)
        decision = judge_result["decision_name"]

        # 4. Extractor ONLY if ATTEMPT
        if decision == "ATTEMPT":
            extraction_result = agent.extractor.extract(response, env=env)
        else:
            extraction_result = {
                "best_candidate": None,
                "verified_candidate": None,
                "verified_rank": 0,
                "verified_score": 0,
                "verification_response": "",
                "verification_traces": [],
                "ranked_candidates": [],
                "all_candidates": [],
                "top_k_candidates": [],
                "regex_candidates": [],
                "quoted_candidates": [],
                "capitalized_candidates": [],
                "llm_candidates": [],
                "llm_ranked_candidates": [],
                "verified": False,
            }

        gt_leaked = agent.extractor.check_ground_truth_leak(response)
        extraction_result["ground_truth_leaked"] = gt_leaked
"""

silent_pattern = re.compile(r'# Problem 4: Trim judge input.*?ground_truth_leaked = agent\.extractor\.check_ground_truth_leak\(response\)\n\s+extraction_result\["ground_truth_leaked"\] = gt_leaked', re.DOTALL)
content = silent_pattern.sub(silent_judge_replace.strip(), content)


# Replace _silent_test_batch
silent_batch_judge_replace = """
        # 1. Generator
        gen_results = generate_attack_batch(
            [agents[idx] for idx in active_indices],
            [last_attacks[idx] for idx in active_indices],
            [last_responses[idx] for idx in active_indices],
        )
        attacks = [r["generated_attack"] for r in gen_results]

        # 2. Victim
        pre_defenses = [envs[idx].scenario.opening_defense for idx in active_indices]
        post_defenses = [envs[idx].scenario.closing_defense for idx in active_indices]
        responses = chat_with_llama_batch(pre_defenses, attacks, post_defenses)

        new_contents = []
        gt_leaks = []
        judge_inputs = []
        for j, idx in enumerate(active_indices):
            env = envs[idx]
            env.current_step += 1
            resp = responses[j]
            env.last_response = resp
            if env.current_step >= env.max_steps:
                env.done = True
            clean_resp = strip_few_shot_patterns(resp)
            new_contents.append(clean_resp)
            gt_leaks.append(agents[idx].extractor.check_ground_truth_leak(resp))
            judge_inputs.append(clean_resp[-512:] if clean_resp else "[EMPTY RESPONSE]")

        # 3. Judge evaluates the Victim's response
        judge_results = agents[0].judge.predict_batch(judge_inputs)

        # 4. Extractor (ONLY if judge says ATTEMPT)
        extractors_to_run = []
        texts_to_extract = []
        envs_to_extract = []
        extract_indices = []

        for j, idx in enumerate(active_indices):
            if judge_results[j]["decision_name"] == "ATTEMPT":
                extractors_to_run.append(agents[idx].extractor)
                texts_to_extract.append(responses[j])
                envs_to_extract.append(envs[idx])
                extract_indices.append(j)

        batch_extraction_results = [None] * len(active_indices)

        if extractors_to_run:
            partial_extract_results = extract_batch(
                extractors_to_run, texts_to_extract, envs_to_extract, top_k=5
            )
            for j_ext, j_orig in enumerate(extract_indices):
                batch_extraction_results[j_orig] = partial_extract_results[j_ext]

        for j in range(len(active_indices)):
            if batch_extraction_results[j] is None:
                batch_extraction_results[j] = {
                    "best_candidate": None,
                    "verified_candidate": None,
                    "verified_rank": 0,
                    "verified_score": 0,
                    "verification_response": "",
                    "verification_traces": [],
                    "ranked_candidates": [],
                    "all_candidates": [],
                    "top_k_candidates": [],
                    "regex_candidates": [],
                    "quoted_candidates": [],
                    "capitalized_candidates": [],
                    "llm_candidates": [],
                    "llm_ranked_candidates": [],
                    "verified": False,
                }
"""

silent_batch_pattern = re.compile(r'judge_inputs = \[\]\n\s+for idx in active_indices:\n\s+if attempt == 0:.*?batch_extraction_results = extract_batch\(\n\s+\[agents\[idx\]\.extractor for idx in active_indices\],\n\s+responses, \[envs\[idx\] for idx in active_indices\], top_k=5\n\s+\)', re.DOTALL)
content = silent_batch_pattern.sub(silent_batch_judge_replace.strip(), content)

with open("experiment/llama_3_8b_verbose.py", "w") as f:
    f.write(content)

