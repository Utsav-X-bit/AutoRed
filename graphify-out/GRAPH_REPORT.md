# Graph Report - /home/utsav/Github/Research/AutoRed  (2026-04-23)

## Corpus Check
- 83 files · ~57,424 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1053 nodes · 2905 edges · 33 communities detected
- Extraction: 53% EXTRACTED · 47% INFERRED · 0% AMBIGUOUS · INFERRED: 1359 edges (avg confidence: 0.59)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_RL Training Infrastructure|RL Training Infrastructure]]
- [[_COMMUNITY_Evaluation Metrics & Rewards|Evaluation Metrics & Rewards]]
- [[_COMMUNITY_A2C Algorithm|A2C Algorithm]]
- [[_COMMUNITY_Algorithm Utilities|Algorithm Utilities]]
- [[_COMMUNITY_Policy & Evaluation Core|Policy & Evaluation Core]]
- [[_COMMUNITY_AutoRed Framework|AutoRed Framework]]
- [[_COMMUNITY_Rollout Buffers|Rollout Buffers]]
- [[_COMMUNITY_Experiment Notebooks|Experiment Notebooks]]
- [[_COMMUNITY_Maskable Distributions|Maskable Distributions]]
- [[_COMMUNITY_Action Space|Action Space]]
- [[_COMMUNITY_Summarization & Reward Metrics|Summarization & Reward Metrics]]
- [[_COMMUNITY_CIDER Scorer|CIDER Scorer]]
- [[_COMMUNITY_Preference Reward|Preference Reward]]
- [[_COMMUNITY_AutoRed Visual Assets|AutoRed Visual Assets]]
- [[_COMMUNITY_KL Controllers|KL Controllers]]
- [[_COMMUNITY_Post Processors|Post Processors]]
- [[_COMMUNITY_HPC Asset Downloads|HPC Asset Downloads]]
- [[_COMMUNITY_HPC Setup Scripts|HPC Setup Scripts]]
- [[_COMMUNITY_HPC Model Downloads|HPC Model Downloads]]
- [[_COMMUNITY_ML Dependencies|ML Dependencies]]
- [[_COMMUNITY_Data Pool Rationale|Data Pool Rationale]]
- [[_COMMUNITY_HF Generation Rationale|HF Generation Rationale]]
- [[_COMMUNITY_Base Policy Rationale|Base Policy Rationale]]
- [[_COMMUNITY_Base Policy Rationale|Base Policy Rationale]]
- [[_COMMUNITY_Base Policy Rationale|Base Policy Rationale]]
- [[_COMMUNITY_Base Policy Rationale|Base Policy Rationale]]
- [[_COMMUNITY_Base Policy Rationale|Base Policy Rationale]]
- [[_COMMUNITY_Base Policy Rationale|Base Policy Rationale]]
- [[_COMMUNITY_Base Policy Rationale|Base Policy Rationale]]
- [[_COMMUNITY_Base Policy Rationale|Base Policy Rationale]]
- [[_COMMUNITY_Base Env Rationale|Base Env Rationale]]
- [[_COMMUNITY_Distributions Rationale|Distributions Rationale]]
- [[_COMMUNITY_WandB Dependency|WandB Dependency]]

## God Nodes (most connected - your core abstractions)
1. `RewardFunctionRegistry` - 67 edges
2. `RewardFunction` - 57 edges
3. `MeteorMetric` - 48 edges
4. `MetricRegistry` - 46 edges
5. `DataPoolRegistry` - 45 edges
6. `PolicyRegistry` - 45 edges
7. `AlgorithmRegistry` - 45 edges
8. `WrapperRegistry` - 45 edges
9. `Tracker` - 43 edges
10. `RougeLMax` - 42 edges

## Surprising Connections (you probably didn't know these)
- `rl4lms/ Directory (forked RL4LMs library)` --semantically_similar_to--> `RL4LMs Library (AllenAI)`  [INFERRED] [semantically similar]
  GEMINI.md → README.md
- `AUTO RED Framework (Poster)` --semantically_similar_to--> `AUTO RED Framework`  [INFERRED] [semantically similar]
  poster.pdf → AutoRed_ Automated Attack Scenario Generation Framework for Red Teaming of LLMs.AutoRed_AutomatedAttackScenarioGenerationFrameworkforRedTeamingofLLMs.pdf
- `Malicious Prompt Generator (Poster)` --semantically_similar_to--> `Malicious Prompt Generator`  [INFERRED] [semantically similar]
  poster.pdf → AutoRed_ Automated Attack Scenario Generation Framework for Red Teaming of LLMs.AutoRed_AutomatedAttackScenarioGenerationFrameworkforRedTeamingofLLMs.pdf
- `Sensitive Information Extractor (Poster)` --semantically_similar_to--> `Sensitive Information Extractor`  [INFERRED] [semantically similar]
  poster.pdf → AutoRed_ Automated Attack Scenario Generation Framework for Red Teaming of LLMs.AutoRed_AutomatedAttackScenarioGenerationFrameworkforRedTeamingofLLMs.pdf
- `AutoRed Overview and Architecture` --semantically_similar_to--> `AutoRed Framework`  [INFERRED] [semantically similar]
  GEMINI.md → README.md

## Hyperedges (group relationships)
- **AutoRed Three-Component Architecture** — readme_stop_point_identifier, readme_malicious_prompt_generator, readme_sensitive_information_extractor [EXTRACTED 1.00]
- **RL4LMs Text Evaluation Metrics Suite** — rl4lms_gemini_bleu, rl4lms_gemini_rouge, rl4lms_gemini_spice [EXTRACTED 1.00]
- **Experiment Target LLM Family** — experiment_gemini_gpt_35_turbo_experiment, experiment_gemini_llama_2_7b_experiment, experiment_gemini_llama_3_8b_experiment, experiment_gemini_mistral_experiment, experiment_gemini_internlm_experiment [EXTRACTED 1.00]
- **AUTO RED Three-Component Architecture** — autored_malicious_prompt_generator, autored_sensitive_information_extractor, autored_stop_point_identifier [EXTRACTED 1.00]
- **Two-Stage Prompt Generation Pipeline** — autored_supervised_fine_tuning_module, autored_reinforcement_learning_module, autored_malicious_prompt_generator [EXTRACTED 1.00]
- **Three Defense Strategy Recommendations** — autored_prompt_engineering_separators, autored_output_filtering, autored_alarm_triggers [EXTRACTED 1.00]
- **AutoRed Closed-Loop Attack Pipeline (generate → attack → observe → extract → decide → repeat)** — autored_model_malicious_prompt_generator, autored_model_attack_flow, autored_model_chatbot, autored_model_response_flow, autored_model_sensitive_info_extractor, autored_model_stop_point_identifier, autored_model_iterative_loop [EXTRACTED 1.00]
- **Logo Security Injection Metaphor (lock + syringe = attack on secure system)** — autored_logo_autored_logo, autored_logo_padlock_symbol, autored_logo_syringe_symbol [EXTRACTED 1.00]

## Communities

### Community 0 - "RL Training Infrastructure"
Cohesion: 0.03
Nodes (57): ActorCriticWarmStartMixin, compute_batched_rewards(), Unpacks vectorized dict observations into separate dict observations, TransitionInfo, unpack_observations(), wrap_onpolicy_alg(), _build_model_heads(), evaluate_actions() (+49 more)

### Community 1 - "Evaluation Metrics & Rewards"
Cohesion: 0.07
Nodes (83): BaseMetric, BatchedRewardFunction, Cider, Main Class to compute the CIDEr metric, PIExt, PIGen, prepare(), BaseMetric (+75 more)

### Community 2 - "A2C Algorithm"
Cohesion: 0.04
Nodes (46): A2C, Update policy using the currently gathered         rollout buffer (one gradient, Advantage Actor Critic (A2C)      Paper: https://arxiv.org/abs/1602.01783     Co, step(), EvaluateActionsOutput, Dataclass for the output of the method policy.evaluate_actions().     This is in, MaskableDictRolloutBuffer, MaskableDictRolloutBufferSamples (+38 more)

### Community 3 - "Algorithm Utilities"
Cohesion: 0.05
Nodes (49): conjugate_gradient_solver(), flat_grad(), quantile_huber_loss(), The quantile-regression loss, as described in the QR-DQN and TQC papers.     Par, Returns the gradients of the passed sequence of parameters into a flat gradient., Finds an approximate solution to a set of linear equations Ax = b      Sources:, tokenize_rewards(), BeamSampleDecoderOnlyOutput (+41 more)

### Community 4 - "Policy & Evaluation Core"
Cohesion: 0.05
Nodes (27): evaluate_on_samples(), generate_text(), get_batch(), If `inputs` is None and `name` is in both forward function and keyword arguments, main(), TrainerCallback, TrainerWarmStartMixin, build_alg() (+19 more)

### Community 5 - "AutoRed Framework"
Cohesion: 0.04
Nodes (68): Access Code (Sensitive Information), Agentic Workflow, Alarm Triggers for Suspicious Inputs, Attack Success Rate (ASR), AUTO RED Framework, Binary Sentence Classifier, Capture The Flag (CTF) Game, Curiosity-Driven RL for Red Teaming (+60 more)

### Community 6 - "Rollout Buffers"
Cohesion: 0.05
Nodes (23): :param action_masks: Masks applied to constrain the choice of possible actions., :param action_masks: Masks applied to constrain the choice of possible actions., Discrete, Env, Resets the environment and starts a new episode, A generic RL environment to generate textual sequences.         For eg: text gen, TextGenEnv, _concat() (+15 more)

### Community 7 - "Experiment Notebooks"
Cohesion: 0.06
Nodes (53): GPT-3.5-turbo Experiment Notebook, InternLM Experiment Notebook, Jupyter Notebook Environment, Llama 2 7B Experiment Notebook, Llama 3 8B Experiment Notebook, Mistral Experiment Notebook, Result Analysis Notebook, AutoRed Overview and Architecture (+45 more)

### Community 8 - "Maskable Distributions"
Cohesion: 0.07
Nodes (26): Distribution, apply_masking(), make_masked_proba_distribution(), MaskableBernoulliDistribution, MaskableCategorical, MaskableDistribution, MaskableMultiCategoricalDistribution, Code adapted from https://github.com/DLR-RM/stable-baselines3 (+18 more)

### Community 9 - "Action Space"
Cohesion: 0.09
Nodes (23): ABC, ActionSpace, add_sample(), BaseEnv, close(), A base class for all the environments, Args:             max_steps (int): max steps for each episode             reward, Resets the episode and returns an observation (+15 more)

### Community 10 - "Summarization & Reward Metrics"
Cohesion: 0.11
Nodes (8): batcher(), card_to_name(), get_neutral_idx(), name_to_card(), Code taken from https://github.com/tingofurro/summac, SummaCImager, is_number(), reward_increasing_numbers_in_text()

### Community 11 - "CIDER Scorer"
Cohesion: 0.14
Nodes (11): CiderScorer, cook_refs(), cook_test(), precook(), Compute term frequency for reference data.         This will be used to compute, Takes a string as input and returns an object that can be given to     either co, Takes a list of reference sentences for a single segment     and returns an obje, Takes a test sentence and returns an object that     encapsulates everything tha (+3 more)

### Community 12 - "Preference Reward"
Cohesion: 0.2
Nodes (9): get_model(), get_scores(), get_tokenizer(), main(), parse_args(), A (hopefully) Simple API for scoring commongen outputs.   {"input": "pyramids in, Inputs:       - a list of commongens to score, e.g.,:       - device: which torc, Optional args for main function, mostly just to test. (+1 more)

### Community 13 - "AutoRed Visual Assets"
Cohesion: 0.26
Nodes (15): AutoRed Logo (padlock + syringe), Padlock Symbol (security / protected system), Syringe Symbol (injection / attack / exploit), AutoRed's Attack (malicious prompts sent to Chatbot), AutoRed System (automated adversarial framework), Chatbot (LLM interface receiving prompts), Decision State d1 = extract (switch to extraction phase), Decision State d0 = generate (continue prompt generation) (+7 more)

### Community 14 - "KL Controllers"
Cohesion: 0.29
Nodes (2): kl_coeff(), KLController

### Community 15 - "Post Processors"
Cohesion: 0.5
Nodes (2): Returns first three sentences from the generated text, three_sentence_summary()

### Community 16 - "HPC Asset Downloads"
Cohesion: 0.83
Nodes (2): download(), main()

### Community 17 - "HPC Setup Scripts"
Cohesion: 0.83
Nodes (2): download(), main()

### Community 18 - "HPC Model Downloads"
Cohesion: 0.67
Nodes (1): main()

### Community 19 - "ML Dependencies"
Cohesion: 1.0
Nodes (3): bert-score, BLEURT (from google-research/bleurt.git), gem-metrics (from GEM-benchmark)

### Community 22 - "Data Pool Rationale"
Cohesion: 1.0
Nodes (1): A factory method to instantiate data pool

### Community 26 - "HF Generation Rationale"
Cohesion: 1.0
Nodes (1): r"""          Generates sequences of token ids for models with a language modeli

### Community 32 - "Base Policy Rationale"
Cohesion: 1.0
Nodes (1): Builds policy and value models         and sets self._policy_model and self._val

### Community 33 - "Base Policy Rationale"
Cohesion: 1.0
Nodes (1): Performs a forward pass on the policy and gets log_probs, entropy etc         co

### Community 34 - "Base Policy Rationale"
Cohesion: 1.0
Nodes (1): Performs a forward pass on the value network and gets values corresponding to ob

### Community 35 - "Base Policy Rationale"
Cohesion: 1.0
Nodes (1): Evaluates specified <observation, action>         and returns log_probs, values,

### Community 36 - "Base Policy Rationale"
Cohesion: 1.0
Nodes (1): Performs a forward pass on the reference policy and gets log_probs         corre

### Community 37 - "Base Policy Rationale"
Cohesion: 1.0
Nodes (1): Returns the first device of the policy. Used in the case of model parallel

### Community 38 - "Base Policy Rationale"
Cohesion: 1.0
Nodes (1): Returns the type of policy (causal or seq2seq)

### Community 39 - "Base Policy Rationale"
Cohesion: 1.0
Nodes (1): Extracts the prompt inputs and attention masks which is used as seed for generat

### Community 40 - "Base Env Rationale"
Cohesion: 1.0
Nodes (1): Takes a step with the given action and returns (next state, reward, done, info)

### Community 48 - "Distributions Rationale"
Cohesion: 1.0
Nodes (1): Eliminate ("mask out") chosen distribution outcomes by setting their probability

### Community 72 - "WandB Dependency"
Cohesion: 1.0
Nodes (1): wandb==0.12.15

## Knowledge Gaps
- **127 isolated node(s):** `A factory method to instantiate data pool`, `Creates a priority sampler          Args:             max_size (int): maximum si`, `Recursively splits the given object`, `MD5 hash of a dictionary.`, `Returns first three sentences from the generated text` (+122 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `KL Controllers`** (8 nodes): `kl_controllers.py`, `kl_coeff()`, `KLController`, `.get_state_dict()`, `.__init__()`, `.load_from_state_dict()`, `.step()`, `kl_controllers.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Post Processors`** (4 nodes): `post_processors.py`, `Returns first three sentences from the generated text`, `three_sentence_summary()`, `post_processors.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `HPC Asset Downloads`** (4 nodes): `download()`, `main()`, `download_hf_assets.py`, `download_hf_assets.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `HPC Setup Scripts`** (4 nodes): `setup_login_node.py`, `setup_login_node.py`, `download()`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `HPC Model Downloads`** (3 nodes): `main()`, `download_models.py`, `download_models.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Data Pool Rationale`** (1 nodes): `A factory method to instantiate data pool`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `HF Generation Rationale`** (1 nodes): `r"""          Generates sequences of token ids for models with a language modeli`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Base Policy Rationale`** (1 nodes): `Builds policy and value models         and sets self._policy_model and self._val`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Base Policy Rationale`** (1 nodes): `Performs a forward pass on the policy and gets log_probs, entropy etc         co`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Base Policy Rationale`** (1 nodes): `Performs a forward pass on the value network and gets values corresponding to ob`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Base Policy Rationale`** (1 nodes): `Evaluates specified <observation, action>         and returns log_probs, values,`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Base Policy Rationale`** (1 nodes): `Performs a forward pass on the reference policy and gets log_probs         corre`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Base Policy Rationale`** (1 nodes): `Returns the first device of the policy. Used in the case of model parallel`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Base Policy Rationale`** (1 nodes): `Returns the type of policy (causal or seq2seq)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Base Policy Rationale`** (1 nodes): `Extracts the prompt inputs and attention masks which is used as seed for generat`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Base Env Rationale`** (1 nodes): `Takes a step with the given action and returns (next state, reward, done, info)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Distributions Rationale`** (1 nodes): `Eliminate ("mask out") chosen distribution outcomes by setting their probability`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `WandB Dependency`** (1 nodes): `wandb==0.12.15`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `RewardFunctionRegistry` connect `Evaluation Metrics & Rewards` to `RL Training Infrastructure`, `A2C Algorithm`, `Policy & Evaluation Core`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **Why does `Tracker` connect `A2C Algorithm` to `RL Training Infrastructure`, `Evaluation Metrics & Rewards`, `Policy & Evaluation Core`?**
  _High betweenness centrality (0.051) - this node is a cross-community bridge._
- **Why does `RewardFunction` connect `Evaluation Metrics & Rewards` to `RL Training Infrastructure`, `Action Space`, `Policy & Evaluation Core`, `Rollout Buffers`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._
- **Are the 65 inferred relationships involving `RewardFunctionRegistry` (e.g. with `RewardFunction` and `BatchedRewardFunction`) actually correct?**
  _`RewardFunctionRegistry` has 65 INFERRED edges - model-reasoned connections that need verification._
- **Are the 38 inferred relationships involving `RewardFunction` (e.g. with `RewardIncreasingNumbers` and `RewardSentencesWithDates`) actually correct?**
  _`RewardFunction` has 38 INFERRED edges - model-reasoned connections that need verification._
- **Are the 43 inferred relationships involving `MeteorMetric` (e.g. with `Cider` and `Spice`) actually correct?**
  _`MeteorMetric` has 43 INFERRED edges - model-reasoned connections that need verification._
- **Are the 44 inferred relationships involving `MetricRegistry` (e.g. with `OnPolicyTrainer` and `SupervisedTrainer`) actually correct?**
  _`MetricRegistry` has 44 INFERRED edges - model-reasoned connections that need verification._