# Graph Report - /home/utsav/Github/Research/AutoRed/experiment  (2026-04-23)

## Corpus Check
- Corpus is ~2,937 words - fits in a single context window. You may not need a graph.

## Summary
- 33 nodes · 37 edges · 8 communities detected
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]

## God Nodes (most connected - your core abstractions)
1. `verbose_test_llama()` - 5 edges
2. `inference_dec_model_verbose()` - 4 edges
3. `DecisionType` - 3 edges
4. `RedTeamingAgent` - 3 edges
5. `chat_with_llama()` - 3 edges
6. `DecisionType` - 3 edges
7. `inference_gen_model_verbose()` - 3 edges
8. `inference_dec_model()` - 2 edges
9. `RedTeamingAgent` - 2 edges
10. `print_summary_table()` - 2 edges

## Surprising Connections (you probably didn't know these)
- `inference_dec_model_verbose()` --calls--> `DecisionType`  [EXTRACTED]
  llama_3_8b_verbose.py → llama_3_8b_verbose.py  _Bridges community 0 → community 1_

## Communities

### Community 0 - "Community 0"
Cohesion: 0.22
Nodes (4): IntEnum, DecisionType, inference_dec_model(), DecisionType

### Community 1 - "Community 1"
Cohesion: 0.25
Nodes (8): chat_with_llama(), inference_dec_model_verbose(), inference_gen_model_verbose(), Generate a malicious prompt using the T5 generator.     Returns the generated at, Run the DistilBERT decision model.     Returns the decision (ATTACK or ATTEMPT), Run the AutoRed attack loop with FULL step-by-step logging.      Returns:, Send a prompt to Llama-3 and return the decoded response., verbose_test_llama()

### Community 2 - "Community 2"
Cohesion: 0.5
Nodes (1): AutoRed — Verbose Step-by-Step Red Teaming Experiment (Llama-3-8B) =============

### Community 3 - "Community 3"
Cohesion: 0.67
Nodes (1): RedTeamingAgent

### Community 4 - "Community 4"
Cohesion: 1.0
Nodes (2): Save the full trace to a JSON file for later analysis., save_trace()

### Community 5 - "Community 5"
Cohesion: 1.0
Nodes (2): print_summary_table(), Print a compact summary of all iterations.

### Community 6 - "Community 6"
Cohesion: 1.0
Nodes (1): RedTeamingAgent

### Community 7 - "Community 7"
Cohesion: 1.0
Nodes (2): analyze_attack_evolution(), Analyze how attacks evolve over iterations.

## Knowledge Gaps
- **8 isolated node(s):** `AutoRed — Verbose Step-by-Step Red Teaming Experiment (Llama-3-8B) =============`, `Send a prompt to Llama-3 and return the decoded response.`, `Generate a malicious prompt using the T5 generator.     Returns the generated at`, `Run the DistilBERT decision model.     Returns the decision (ATTACK or ATTEMPT)`, `Run the AutoRed attack loop with FULL step-by-step logging.      Returns:` (+3 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 4`** (2 nodes): `Save the full trace to a JSON file for later analysis.`, `save_trace()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 5`** (2 nodes): `print_summary_table()`, `Print a compact summary of all iterations.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 6`** (2 nodes): `RedTeamingAgent`, `.__init__()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 7`** (2 nodes): `analyze_attack_evolution()`, `Analyze how attacks evolve over iterations.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DecisionType` connect `Community 0` to `Community 1`, `Community 2`?**
  _High betweenness centrality (0.484) - this node is a cross-community bridge._
- **Why does `RedTeamingAgent` connect `Community 3` to `Community 0`?**
  _High betweenness centrality (0.123) - this node is a cross-community bridge._
- **What connects `AutoRed — Verbose Step-by-Step Red Teaming Experiment (Llama-3-8B) =============`, `Send a prompt to Llama-3 and return the decoded response.`, `Generate a malicious prompt using the T5 generator.     Returns the generated at` to the rest of the system?**
  _8 weakly-connected nodes found - possible documentation gaps or missing edges._