"""Compatibility normalization for AutoRed result JSON files."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _integer(value: Any, default: int = 0) -> int:
    return int(_number(value, default))


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return value if isinstance(value, str) else str(value)


def _list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _ranked_candidates(values: Iterable[Any]) -> List[Dict[str, Any]]:
    normalized = []
    for item in values:
        if isinstance(item, dict):
            value = _text(item.get("value"))
            score = _number(item.get("score"))
        elif isinstance(item, (list, tuple)) and item:
            value = _text(item[0])
            score = _number(item[1] if len(item) > 1 else 0)
        else:
            value = _text(item)
            score = 0.0
        if value:
            normalized.append({"value": value, "score": score})
    return normalized


def _judge_probabilities(value: Any) -> Dict[str, float]:
    probabilities = _dict(value)
    return {
        "ATTACK": _number(
            probabilities.get("ATTACK", probabilities.get("ATTACK (0)", 0))
        ),
        "ATTEMPT": _number(
            probabilities.get("ATTEMPT", probabilities.get("ATTEMPT (1)", 0))
        ),
    }


def normalize_extraction_result(result: Any) -> Dict[str, Any]:
    """Make extractor output safe for comparisons, JSON, and WebSocket clients."""
    result = _dict(result)
    best_candidate = result.get("best_candidate")
    raw_ranked = result.get("all_candidates")
    ranked_candidates = _ranked_candidates(
        raw_ranked if isinstance(raw_ranked, (list, tuple)) else []
    )

    def candidate_list(key: str) -> list:
        values = result.get(key)
        return list(values) if isinstance(values, (list, tuple)) else []

    return {
        "regex_candidates": candidate_list("regex_candidates"),
        "quoted_candidates": candidate_list("quoted_candidates"),
        "capitalized_candidates": candidate_list("capitalized_candidates"),
        "llm_candidates": candidate_list("llm_candidates"),
        "ranked_candidates": ranked_candidates,
        "best_candidate": best_candidate if isinstance(best_candidate, str) else "",
        "verified": bool(result.get("verified", False)),
    }


def _normalize_attempt(raw: Any, index: int) -> Dict[str, Any]:
    attempt = _dict(raw)
    generator = _dict(attempt.get("generator"))
    judge = _dict(attempt.get("judge"))
    victim = _dict(attempt.get("victim"))
    extractor = _dict(attempt.get("extractor"))
    verification = _dict(attempt.get("verification"))
    attack = _text(generator.get("generated_attack"))
    raw_output = _text(victim.get("raw_output"))

    return {
        "attempt_number": _integer(attempt.get("attempt_number"), index + 1),
        "timestamp": _text(attempt.get("timestamp")),
        "attempt_time_ms": _integer(attempt.get("attempt_time_ms")),
        "generator": {
            "strategy": _text(generator.get("strategy"), "unknown"),
            "internal_prompt": _text(generator.get("internal_prompt")),
            "generated_attack": attack,
            "attack_length": _integer(generator.get("attack_length"), len(attack)),
            "attack_hash": _text(generator.get("attack_hash")),
            "duplicate_attack": bool(generator.get("duplicate_attack", False)),
            "input_tokens": _integer(generator.get("input_tokens")),
            "output_tokens": _integer(generator.get("output_tokens")),
        },
        "judge": {
            "input": _text(judge.get("input")),
            "decision": _text(judge.get("decision")),
            "confidence": _number(judge.get("confidence")),
            "probabilities": _judge_probabilities(judge.get("probabilities")),
        },
        "victim": {
            "raw_output": raw_output,
            "clean_output": _text(victim.get("clean_output"), raw_output),
            "output_length": _integer(victim.get("output_length"), len(raw_output)),
        },
        "extractor": {
            "regex_candidates": [
                _text(value) for value in _list(extractor.get("regex_candidates"))
                if _text(value)
            ],
            "quoted_candidates": [
                _text(value) for value in _list(extractor.get("quoted_candidates"))
                if _text(value)
            ],
            "capitalized_candidates": [
                _text(value) for value in _list(extractor.get("capitalized_candidates"))
                if _text(value)
            ],
            "llm_candidates": [
                _text(value) for value in _list(extractor.get("llm_candidates"))
                if _text(value)
            ],
            "ranked_candidates": _ranked_candidates(
                _list(extractor.get("ranked_candidates"))
            ),
            "best_candidate": _text(extractor.get("best_candidate")),
        },
        "verification": {
            "candidate_sent": _text(verification.get("candidate_sent")),
            "victim_response": _text(verification.get("victim_response")),
            "success": bool(verification.get("success", False)),
        },
        "ground_truth_found": bool(attempt.get("ground_truth_found", False)),
        "extractor_match": bool(attempt.get("extractor_match", False)),
        "generator_success": bool(attempt.get("generator_success", False)),
    }


def _summary(attempts: List[Dict[str, Any]], raw_summary: Any) -> Dict[str, Any]:
    summary = _dict(raw_summary)
    lengths = [attempt["generator"]["attack_length"] for attempt in attempts]
    attacks = [attempt["generator"]["generated_attack"] for attempt in attempts]
    unique_attacks = len(set(attacks))
    total = len(attacks)
    judge_distribution = {"ATTACK": 0, "ATTEMPT": 0}
    for attempt in attempts:
        decision = attempt["judge"]["decision"]
        if decision in judge_distribution:
            judge_distribution[decision] += 1

    supplied_distribution = _dict(summary.get("judge_distribution"))
    return {
        "attack_length_min": _integer(
            summary.get("attack_length_min"), min(lengths, default=0)
        ),
        "attack_length_max": _integer(
            summary.get("attack_length_max"), max(lengths, default=0)
        ),
        "attack_length_avg": _number(
            summary.get("attack_length_avg"),
            sum(lengths) / len(lengths) if lengths else 0,
        ),
        "unique_attacks": _integer(summary.get("unique_attacks"), unique_attacks),
        "repetition_rate": _number(
            summary.get("repetition_rate"),
            (total - unique_attacks) / total if total else 0,
        ),
        "judge_distribution": {
            "ATTACK": _integer(
                supplied_distribution.get("ATTACK"), judge_distribution["ATTACK"]
            ),
            "ATTEMPT": _integer(
                supplied_distribution.get("ATTEMPT"), judge_distribution["ATTEMPT"]
            ),
        },
    }


def normalize_run(data: Any, fallback_run_id: str = "") -> Dict[str, Any]:
    """Return a complete, UI-safe AutoRedRun without discarding extra fields."""
    run = deepcopy(_dict(data))
    experiment = _dict(run.get("experiment"))
    scenario = _dict(run.get("scenario"))
    result = _dict(run.get("result"))
    timing = _dict(run.get("timing"))
    ground_truth = _dict(run.get("ground_truth"))
    raw_dataset_entry = _dict(run.get("raw_dataset_entry"))
    raw_models = _dict(run.get("models"))
    models = {}
    for model_name in ("victim", "generator", "judge", "extractor"):
        model = _dict(raw_models.get(model_name))
        models[model_name] = {
            "name": _text(model.get("name"), "unknown"),
            "load_time": _number(model.get("load_time")),
        }
    strategy_stats = {}
    for name, raw_stat in _dict(run.get("strategy_stats")).items():
        stat = _dict(raw_stat)
        strategy_stats[str(name)] = {
            "successes": _integer(stat.get("successes")),
            "partial_leaks": _integer(stat.get("partial_leaks")),
            "failures": _integer(stat.get("failures")),
            "total_score": _number(stat.get("total_score")),
        }
    raw_attempts = _list(run.get("attempts"))
    attempts = [_normalize_attempt(attempt, i) for i, attempt in enumerate(raw_attempts)]
    run_id = _text(experiment.get("run_id"), fallback_run_id)

    experiment.update({
        "run_id": run_id,
        "benchmark_mode": bool(experiment.get("benchmark_mode", False)),
        "max_attempts": _integer(experiment.get("max_attempts"), len(attempts)),
        "dataset_size": _integer(experiment.get("dataset_size")),
        "scenario_id": _text(
            experiment.get("scenario_id"),
            _text(raw_dataset_entry.get("defense_id"), "unknown"),
        ),
        "seed": _integer(experiment.get("seed"), 42),
        "timestamp": _text(experiment.get("timestamp")),
        "experiment_version": _text(experiment.get("experiment_version"), "unknown"),
        "git_commit": _text(experiment.get("git_commit"), "unknown"),
    })
    timing.update({
        "total_run_time": _number(timing.get("total_run_time")),
        "model_loading_time": _number(timing.get("model_loading_time")),
        "average_attempt_time": _number(timing.get("average_attempt_time")),
    })
    scenario.update({
        "pre_defense": _text(scenario.get("pre_defense")),
        "post_defense": _text(scenario.get("post_defense")),
        "access_code": _text(scenario.get("access_code")),
        "full_prompt": _text(scenario.get("full_prompt")),
    })
    result.update({
        "ground_truth_success": bool(result.get("ground_truth_success", False)),
        "generator_success": bool(result.get("generator_success", False)),
        "extractor_success": bool(result.get("extractor_success", False)),
        "verified_success": bool(result.get("verified_success", False)),
        "extracted_value": _text(result.get("extracted_value")),
        "success_reason": result.get("success_reason"),
        "total_attempts": _integer(result.get("total_attempts"), len(attempts)),
    })
    ground_truth.update({
        "access_code": _text(
            ground_truth.get("access_code"), scenario["access_code"]
        ),
        "leaked": bool(ground_truth.get("leaked", False)),
        "leak_count": _integer(ground_truth.get("leak_count")),
    })

    run.update({
        "experiment": experiment,
        "raw_dataset_entry": raw_dataset_entry,
        "models": models,
        "timing": timing,
        "scenario": scenario,
        "result": result,
        "strategy_stats": strategy_stats,
        "ground_truth": ground_truth,
        "attempts": attempts,
        "events": _list(run.get("events")),
        "summary": _summary(attempts, run.get("summary")),
    })
    return run
