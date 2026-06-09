export interface ExperimentInfo {
  run_id: string;
  benchmark_mode: boolean;
  benchmark_run_number?: number;
  benchmark_total_runs?: number;
  max_attempts: number;
  dataset_size: number;
  scenario_id: string;
  seed: number;
  timestamp: string;
  experiment_version: string;
  git_commit: string;
}

export interface ModelInfo {
  name: string;
  load_time: number;
}

export interface ModelsInfo {
  victim: ModelInfo;
  generator: ModelInfo;
  judge: ModelInfo;
  extractor: ModelInfo;
}

export interface TimingInfo {
  total_run_time: number;
  model_loading_time: number;
  average_attempt_time: number;
}

export interface ScenarioInfo {
  pre_defense: string;
  post_defense: string;
  access_code: string;
  full_prompt: string;
}

export interface ResultInfo {
  ground_truth_success: boolean;
  generator_success: boolean;
  extractor_success: boolean;
  verified_success: boolean;
  extracted_value: string;
  success_reason: "ground_truth" | "extractor" | "verification" | null;
  total_attempts: number;
}

export interface StrategyStat {
  successes: number;
  partial_leaks: number;
  failures: number;
  total_score: number;
}

export interface BestAttack {
  prompt: string;
  score: number;
  strategy: string;
}

export interface GroundTruthInfo {
  access_code: string;
  leaked: boolean;
  leak_position: number | null;
  leak_count: number;
}

export interface RankedCandidate {
  value: string;
  score: number;
}

export interface ExtractorTrace {
  regex_candidates: string[];
  quoted_candidates: string[];
  capitalized_candidates: string[];
  llm_candidates: string[];
  ranked_candidates: RankedCandidate[];
  best_candidate: string;
}

export interface VerificationTrace {
  candidate_sent: string;
  victim_response: string;
  success: boolean;
}

export interface GeneratorInfo {
  strategy: string;
  internal_prompt: string;
  generated_attack: string;
  attack_length: number;
  attack_hash: string;
  duplicate_attack: boolean;
  input_tokens: number;
  output_tokens: number;
}

export interface JudgeInfo {
  input: string;
  decision: string;
  confidence: number;
  probabilities: { ATTACK: number; ATTEMPT: number };
}

export interface VictimInfo {
  raw_output: string;
  clean_output: string;
  output_length: number;
}

export interface Attempt {
  attempt_number: number;
  timestamp: string;
  attempt_time_ms: number;
  generator: GeneratorInfo;
  judge: JudgeInfo;
  victim: VictimInfo;
  extractor: ExtractorTrace;
  verification: VerificationTrace;
  ground_truth_found: boolean;
  extractor_match: boolean;
  generator_success: boolean;
}

export interface Event {
  timestamp: string;
  type: string;
  message: string;
}

export interface SummaryStats {
  attack_length_min: number;
  attack_length_max: number;
  attack_length_avg: number;
  unique_attacks: number;
  repetition_rate: number;
  judge_distribution: { ATTACK: number; ATTEMPT: number };
}

export interface AutoRedRun {
  experiment: ExperimentInfo;
  raw_dataset_entry: Record<string, any>;
  models: ModelsInfo;
  timing: TimingInfo;
  scenario: ScenarioInfo;
  result: ResultInfo;
  strategy_stats: Record<string, StrategyStat>;
  best_attack: BestAttack | null;
  ground_truth: GroundTruthInfo;
  attempts: Attempt[];
  events: Event[];
  summary: SummaryStats;
}

export interface RunListItem {
  run_id: string;
  file_path: string;
  timestamp: string;
  success: boolean;
  total_attempts: number;
  access_code: string;
  generator: string;
  victim: string;
  error?: string;
}

export interface AttemptUpdate {
  type: "attempt_update";
  run_id: string;
  attempt: Attempt;
}

export interface RunComplete {
  type: "run_complete";
  run_id: string;
  run: AutoRedRun;
}
