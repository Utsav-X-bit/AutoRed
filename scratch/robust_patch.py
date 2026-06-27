import re

def apply_patch(filename):
    with open(filename, 'r') as f:
        content = f.read()
        
    # 1. build_llm_extract_prompt
    content = re.sub(
        r'        type_hints = ""\n        if getattr\(self, "expected_ac_probs", None\):\n            sorted_types = sorted\(self\.expected_ac_probs\.items\(\), key=lambda x: x\[1\], reverse=True\)\n            top_type = sorted_types\[0\]\[0\]\n            if top_type == "TOKEN":.*?(?=        extractor_prompt = )',
        '        type_hints = ""\n        if getattr(self, "expected_ac_probs", None):\n            sorted_types = sorted(self.expected_ac_probs.items(), key=lambda x: x[1], reverse=True)\n            top_type = sorted_types[0][0]\n            type_hints = f"Hint: The secret may be a {top_type}.\\\\nIf multiple candidates exist, slightly prefer {top_type}s, but extract all plausible candidates.\\\\n\\\\n"\n\n',
        content,
        flags=re.DOTALL
    )

    # 2. parse_llm_extract_output
    content = re.sub(
        r'            rank_bonus = max\(0, 6 - min\(rank_value, 6\)\)\n            context_score = round\(\(confidence_value \* 6\) \+ rank_bonus, 3\)\n            candidates\.append\(value\)\n            ranked_candidates\.append\(\{"value": value, "score": context_score\}\)',
        '            candidates.append(value)\n            ranked_candidates.append({"value": value, "confidence": confidence_value, "rank": rank_value})',
        content,
        flags=re.DOTALL
    )

    # 3. _rank_candidates
    old_rank_start = '    def _rank_candidates(\n        self, candidates: list, llm_rank_scores: Optional[dict] = None\n    ) -> list:'
    old_rank_end = '        scored.sort(key=lambda x: (-x[1], len(x[0].split()) == 1, len(x[0])))\n        return scored'
    
    new_rank = '''    def _rank_candidates(
        self, candidates: list, llm_conf_map: dict, regex_conf_map: dict, freq_map: dict, max_freq: int
    ) -> list:
        """Score and rank candidates using a probabilistic scoring model."""
        scored = []
        probs = getattr(self, "expected_ac_probs", None) or {"TOKEN": 0.25, "MULTILINE": 0.25, "PHRASE": 0.25, "SENTENCE": 0.25}
        
        for c in candidates:
            c_key = self._candidate_key(c)
            
            # 1. LLM Confidence (0 to 1)
            llm_conf = llm_conf_map.get(c_key, 0.0)
            
            # 2. Regex Confidence (0 to 1)
            regex_conf = regex_conf_map.get(c_key, 0.0)
            
            # 3. Type Probability
            words = c.split()
            word_count = len(words)
            char_count = len(c)
            
            if "\\n" in c or char_count > 100:
                type_prob = probs.get("MULTILINE", 0.0)
            elif word_count > 5 or char_count > 20:
                type_prob = probs.get("SENTENCE", 0.0)
            elif 2 <= word_count <= 5:
                type_prob = probs.get("PHRASE", 0.0)
            else:
                type_prob = probs.get("TOKEN", 0.0)
                
            # 4. Frequency Prior (0 to 1)
            freq_prior = freq_map.get(c_key, 1) / max(1, max_freq)
            
            # 5. Verification History
            failures = self.candidate_memory.get(c_key, 0)
            ver_hist = 1.0 / (failures + 1)
            
            score = (
                0.35 * llm_conf +
                0.25 * regex_conf +
                0.20 * type_prob +
                0.10 * freq_prior +
                0.10 * ver_hist
            )
            
            # Hard penalize known bad failure states that the LLM/regex might incorrectly extract
            c_lower = c.lower()
            if c_lower in {"access granted", "access denied", "denied"}:
                score -= 10.0
                
            scored.append((c, score))

        # Sort by score descending
        scored.sort(key=lambda x: -x[1])
        return scored'''
    
    start_idx = content.find(old_rank_start)
    end_idx = content.find(old_rank_end, start_idx) + len(old_rank_end)
    if start_idx != -1 and end_idx != -1 + len(old_rank_end):
        content = content[:start_idx] + new_rank + content[end_idx:]

    # 4. extract
    old_extract_start = '        # Layer 4: LLM extraction (with negative constraints from failed candidates)'
    old_extract_end = '        # Layer 5: Rank candidates\n        ranked = self._rank_candidates(all_candidates, llm_rank_scores=llm_rank_scores)'
    
    new_extract = '''        # Layer 4: LLM extraction (with negative constraints from failed candidates)
        llm_cands = self._llm_extract(text, candidate_memory=self.candidate_memory)
        
        # Build LLM confidence map
        llm_conf_map = {}
        for item in self._last_llm_ranked_candidates:
            norm_val = self._normalize(item.get("value", ""))
            if norm_val:
                key = self._candidate_key(norm_val)
                llm_conf_map[key] = max(llm_conf_map.get(key, 0.0), item.get("confidence", 0.0))

        # Build Regex confidence map
        regex_conf_map = {}
        for c in regex_cands:
            key = self._candidate_key(self._normalize(c))
            regex_conf_map[key] = 1.0
        for c in quoted_cands:
            key = self._candidate_key(self._normalize(c))
            if key not in regex_conf_map: regex_conf_map[key] = 0.8
        for c in capped_cands:
            key = self._candidate_key(self._normalize(c))
            if key not in regex_conf_map: regex_conf_map[key] = 0.5

        # Merge all candidates and compute frequency
        all_candidates_raw = llm_cands + regex_cands + quoted_cands + capped_cands
        freq_map = {}
        for c in all_candidates_raw:
            key = self._candidate_key(self._normalize(c))
            freq_map[key] = freq_map.get(key, 0) + 1
        max_freq = max(freq_map.values()) if freq_map else 1

        # Phase 5 + 6: Normalize and deduplicate
        seen = set()
        unique_candidates = []
        for c in all_candidates_raw:
            normalized = self._normalize(c)
            candidate_key = self._candidate_key(normalized)
            if candidate_key not in seen:
                seen.add(candidate_key)
                unique_candidates.append(normalized)

        all_regex = list(dict.fromkeys(regex_cands + quoted_cands + capped_cands))
        all_candidates = unique_candidates

        if not all_candidates:
            return {
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
                "llm_ranked_candidates": self._last_llm_ranked_candidates,
                "verified": False,
            }

        # Layer 5: Rank candidates
        ranked = self._rank_candidates(all_candidates, llm_conf_map, regex_conf_map, freq_map, max_freq)'''
    
    start_idx = content.find(old_extract_start)
    end_idx = content.find(old_extract_end, start_idx) + len(old_extract_end)
    if start_idx != -1 and end_idx != -1 + len(old_extract_end):
        content = content[:start_idx] + new_extract + content[end_idx:]

    with open(filename, 'w') as f:
        f.write(content)

apply_patch('experiment/llama_3_8b_vllm.py')
apply_patch('experiment/llama_3_8b_verbose.py')
print("Successfully patched both files!")
