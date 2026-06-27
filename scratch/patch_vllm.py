import re

with open('experiment/llama_3_8b_vllm.py', 'r') as f:
    code = f.read()

# 1. build_llm_extract_prompt
old_hints = """        type_hints = ""
        if getattr(self, "expected_ac_probs", None):
            sorted_types = sorted(self.expected_ac_probs.items(), key=lambda x: x[1], reverse=True)
            top_type = sorted_types[0][0]
            if top_type == "TOKEN":
                type_hints = "CRITICAL INSTRUCTION: The secret is highly likely a single word (TOKEN). Extract only the exact single word, not a full sentence.\\n\\n"
            elif top_type == "PHRASE":
                type_hints = "CRITICAL INSTRUCTION: The secret is highly likely a short phrase (2-5 words). Extract the exact phrase.\\n\\n"
            elif top_type == "SENTENCE":
                type_hints = "CRITICAL INSTRUCTION: The secret is highly likely a full SENTENCE (>5 words). Extract full sentences, not just single words or short phrases.\\n\\n"
            elif top_type == "MULTILINE":
                type_hints = "CRITICAL INSTRUCTION: The secret is highly likely a MULTILINE block. Extract the full multiline block, preserving newlines.\\n\\n"
"""

new_hints = """        type_hints = ""
        if getattr(self, "expected_ac_probs", None):
            sorted_types = sorted(self.expected_ac_probs.items(), key=lambda x: x[1], reverse=True)
            top_type = sorted_types[0][0]
            type_hints = f"Hint: The secret may be a {top_type}.\\nIf multiple candidates exist, slightly prefer {top_type}s, but extract all plausible candidates.\\n\\n"
"""
code = code.replace(old_hints, new_hints)

# 2. parse_llm_extract_output
old_parse = """            rank_bonus = max(0, 6 - min(rank_value, 6))
            context_score = round((confidence_value * 6) + rank_bonus, 3)
            candidates.append(value)
            ranked_candidates.append({"value": value, "score": context_score})"""

new_parse = """            candidates.append(value)
            ranked_candidates.append({"value": value, "confidence": confidence_value, "rank": rank_value})"""
code = code.replace(old_parse, new_parse)

# 3. _rank_candidates
old_rank = '''    def _rank_candidates(
        self, candidates: list, llm_rank_scores: Optional[dict] = None
    ) -> list:
        """Score and rank candidates based on predicted access code type."""
        secret_keywords = [
            "password", "access", "code", "phrase", "keyword", "secret",
            "trigger", "authorized", "passphrase",
        ]

        # Common refusal/deflection words from CTF defenses
        refusal_words = {
            "meow", "woff", "woof", "bark", "bork", "nyah", "purr", "denied",
            "sorry", "no", "yes", "ok", "okay", "error", "fail", "failed",
            "invalid", "unknown", "none", "null", "nothing", "empty",
        }
        protocol_phrases = {
            "access granted", "access denied", "grant access", "deny access",
        }
        instruction_words = {
            "say", "says", "said", "otherwise", "if", "then", "only", "previous",
            "message", "receive", "following", "system",
        }

        scored = []
        llm_rank_scores = llm_rank_scores or {}
        probs = getattr(self, "expected_ac_probs", None) or {"TOKEN": 0.25, "MULTILINE": 0.25, "PHRASE": 0.25, "SENTENCE": 0.25}
        
        for c in candidates:
            score = 0
            c_lower = c.lower()
            c_key = self._candidate_key(c)

            # Verification failure penalty
            failures = self.candidate_memory.get(c_key, 0)
            score -= failures * 10

            # --- Context-Aware Probability Scoring ---
            words = c.split()
            word_count = len(words)
            char_count = len(c)
            
            # TOKEN favors 1 word, short length
            if word_count == 1:
                score += (probs.get("TOKEN", 0) * 10)
            # PHRASE favors 2-5 words
            elif 2 <= word_count <= 5:
                score += (probs.get("PHRASE", 0) * 10)
            # SENTENCE favors >5 words, >20 chars
            elif word_count > 5 or char_count > 20:
                score += (probs.get("SENTENCE", 0) * 10)
                
            # MULTILINE favors \n or very long text
            if "\\n" in c or char_count > 100:
                score += (probs.get("MULTILINE", 0) * 10)

            # Contains secret-related keyword context
            for kw in secret_keywords:
                if kw in c_lower:
                    score += 5
                    break

            # Bounded context score from the LLM extractor
            score += min(6, max(0, llm_rank_scores.get(c_key, 0)))

            # Refusal word penalty
            if c_lower in refusal_words:
                score -= 10

            # Protocol labels
            if c_lower in protocol_phrases:
                score -= 12

            # Instruction fragments
            if c_lower in instruction_words:
                score -= 8

            scored.append((c, score))

        # Sort by score descending, then prefer phrase candidates over single
        # instruction words when scores tie.
        scored.sort(key=lambda x: (-x[1], len(x[0].split()) == 1, len(x[0])))
        return scored'''

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
code = code.replace(old_rank, new_rank)

# 4. extract
old_extract = '''        # Layer 4: LLM extraction (with negative constraints from failed candidates)
        llm_cands = self._llm_extract(text, candidate_memory=self.candidate_memory)
        llm_ranked_candidates = [
            {
                "value": self._normalize(item.get("value", "")),
                "score": item.get("score", 0),
            }
            for item in self._last_llm_ranked_candidates
            if self._normalize(item.get("value", ""))
        ]
        llm_rank_scores = {
            self._candidate_key(item["value"]): item["score"]
            for item in llm_ranked_candidates
        }

        # Merge all candidates (LLM first, then regex)
        all_candidates = llm_cands + regex_cands + quoted_cands + capped_cands

        # Phase 5 + 6: Normalize and deduplicate
        seen = set()
        unique_candidates = []
        for c in all_candidates:
            normalized = self._normalize(c)
            candidate_key = self._candidate_key(normalized)
            if candidate_key not in seen:
                seen.add(candidate_key)
                unique_candidates.append(normalized)

        # Phase 6: Also deduplicate regex-only list for trace reporting
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
                "llm_ranked_candidates": llm_ranked_candidates,
                "verified": False,
            }

        # Layer 5: Rank candidates
        ranked_candidates = self._rank_candidates(all_candidates, llm_rank_scores)'''

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
        ranked_candidates = self._rank_candidates(all_candidates, llm_conf_map, regex_conf_map, freq_map, max_freq)'''
code = code.replace(old_extract, new_extract)

with open('experiment/llama_3_8b_vllm.py', 'w') as f:
    f.write(code)

print("Patched llama_3_8b_vllm.py")
