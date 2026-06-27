import re

def apply_patch(filename):
    with open(filename, 'r') as f:
        content = f.read()
        
    old_extract_batch_start = '        llm_cands = ext.parse_llm_extract_output(raw_llm)'
    old_extract_batch_end = '        ranked = ext._rank_candidates(\n            unique_candidates, llm_rank_scores=llm_rank_scores\n        )'
    
    new_extract_batch = '''        llm_cands = ext.parse_llm_extract_output(raw_llm)
        
        # Build LLM confidence map
        llm_conf_map = {}
        for item in ext._last_llm_ranked_candidates:
            norm_val = ext._normalize(item.get("value", ""))
            if norm_val:
                key = ext._candidate_key(norm_val)
                llm_conf_map[key] = max(llm_conf_map.get(key, 0.0), item.get("confidence", 0.0))

        # Build Regex confidence map
        regex_conf_map = {}
        for c in regex_cands:
            key = ext._candidate_key(ext._normalize(c))
            regex_conf_map[key] = 1.0
        for c in quoted_cands:
            key = ext._candidate_key(ext._normalize(c))
            if key not in regex_conf_map: regex_conf_map[key] = 0.8
        for c in capped_cands:
            key = ext._candidate_key(ext._normalize(c))
            if key not in regex_conf_map: regex_conf_map[key] = 0.5

        # Merge all candidates and compute frequency
        all_candidates_raw = llm_cands + regex_cands + quoted_cands + capped_cands
        freq_map = {}
        for c in all_candidates_raw:
            key = ext._candidate_key(ext._normalize(c))
            freq_map[key] = freq_map.get(key, 0) + 1
        max_freq = max(freq_map.values()) if freq_map else 1

        seen = set()
        unique_candidates = []
        for c in all_candidates_raw:
            normalized = ext._normalize(c)
            candidate_key = ext._candidate_key(normalized)
            if candidate_key not in seen:
                seen.add(candidate_key)
                unique_candidates.append(normalized)

        all_regex = list(dict.fromkeys(regex_cands + quoted_cands + capped_cands))

        if not unique_candidates:
            batch_results.append(
                {
                    "best_candidate": None,
                    "verified_candidate": None,
                    "verified_rank": 0,
                    "verified_score": 0,
                    "verification_response": "",
                    "verification_traces": [],
                    "ranked_candidates": [],
                    "all_candidates": [],
                    "top_k_candidates": [],
                    "regex_candidates": all_regex,
                    "quoted_candidates": quoted_cands,
                    "capitalized_candidates": capped_cands,
                    "llm_candidates": llm_cands,
                    "llm_ranked_candidates": ext._last_llm_ranked_candidates,
                    "verified": False,
                }
            )
            continue

        ranked = ext._rank_candidates(
            unique_candidates, llm_conf_map, regex_conf_map, freq_map, max_freq
        )'''
    
    start_idx = content.find(old_extract_batch_start)
    if start_idx == -1:
        print("Failed to find start_idx in", filename)
        return
    end_idx = content.find(old_extract_batch_end, start_idx) + len(old_extract_batch_end)
    if content.find(old_extract_batch_end, start_idx) == -1:
        print("Failed to find end_idx in", filename)
        return
    
    content = content[:start_idx] + new_extract_batch + content[end_idx:]

    with open(filename, 'w') as f:
        f.write(content)

apply_patch('experiment/llama_3_8b_vllm.py')
# verbose doesn't have batched, but just in case
try:
    apply_patch('experiment/llama_3_8b_verbose.py')
except:
    pass
print("Successfully patched extract_batch!")
