import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .run_normalizer import normalize_run

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
BENCHMARKS_DIR = RESULTS_DIR / "benchmarks"


def _overall_success(result: Dict[str, Any]) -> bool:
    return any(
        bool(result.get(key))
        for key in ("ground_truth_success", "extractor_success", "verified_success")
    )


def ensure_results_dir():
    """Create results directory if it doesn't exist."""
    RESULTS_DIR.mkdir(exist_ok=True)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return None


def _is_benchmark_artifact(path: Path) -> bool:
    return "benchmarks" in path.parts


def _archive_date_from_timestamp(timestamp: str) -> Optional[str]:
    if not timestamp:
        return None
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).date().isoformat()
    except Exception:
        return None


def _trace_archives_for_timestamp(timestamp: str) -> List[Path]:
    date = _archive_date_from_timestamp(timestamp)
    if not date:
        return []
    day_root = RESULTS_DIR / date
    if not day_root.exists():
        return []
    return sorted(
        [p for p in day_root.iterdir() if p.is_dir() and list(p.glob("run_*.json"))],
        key=lambda p: p.stat().st_mtime,
    )


def _run_metadata_from_file(path: Path) -> Dict[str, Any]:
    raw = _load_json(path) or {}
    normalized = normalize_run(raw, path.stem)
    return {
        "run_id": normalized.get("experiment", {}).get("run_id", path.stem),
        "file_path": str(path),
        "timestamp": normalized.get("experiment", {}).get("timestamp", ""),
        "scenario_id": normalized.get("experiment", {}).get("scenario_id", ""),
        "success": _overall_success(normalized.get("result", {})),
        "verified_success": bool(normalized.get("result", {}).get("verified_success")),
        "total_attempts": normalized.get("result", {}).get("total_attempts", 0),
        "access_code": normalized.get("scenario", {}).get("access_code", ""),
        "generator": normalized.get("models", {}).get("generator", {}).get("name", ""),
        "victim": normalized.get("models", {}).get("victim", {}).get("name", ""),
        "benchmark_mode": normalized.get("experiment", {}).get("benchmark_mode", False),
        "worker_id": normalized.get("experiment", {}).get("worker_id"),
        "attempt_count": len(normalized.get("attempts", [])),
    }


def _summarize_trace_archive(path: Path) -> Dict[str, Any]:
    runs = []
    total_successes = 0
    verified_successes = 0
    total_attempts = 0
    for run_file in sorted(path.glob("run_*.json")):
        meta = _run_metadata_from_file(run_file)
        runs.append(meta)
        total_successes += int(meta["success"])
        verified_successes += int(meta["verified_success"])
        total_attempts += int(meta["total_attempts"])

    run_count = len(runs)
    return {
        "archive_id": f"{path.parent.name}/{path.name}",
        "date": path.parent.name,
        "path": str(path),
        "timestamp": runs[0]["timestamp"] if runs else "",
        "run_count": run_count,
        "success_rate": (total_successes / run_count) if run_count else 0.0,
        "verified_rate": (verified_successes / run_count) if run_count else 0.0,
        "avg_attempts_on_success": (
            total_attempts / total_successes if total_successes else 0.0
        ),
        "runs": runs,
    }


def list_trace_archives() -> List[Dict[str, Any]]:
    """List dated trace archives under results/YYYY-MM-DD/*."""
    ensure_results_dir()
    archives: List[Dict[str, Any]] = []
    for day_dir in sorted(
        [p for p in RESULTS_DIR.iterdir() if p.is_dir() and len(p.name) == 10 and p.name[4] == "-" and p.name[7] == "-"]
    ):
        for archive_dir in sorted([p for p in day_dir.iterdir() if p.is_dir() and list(p.glob("run_*.json"))]):
            archives.append(_summarize_trace_archive(archive_dir))
    return archives


def list_benchmarks(limit: Optional[int] = None, offset: int = 0) -> List[Dict[str, Any]]:
    """List benchmark summaries from results/benchmarks."""
    ensure_results_dir()
    if not BENCHMARKS_DIR.exists():
        return []

    benchmarks: List[Dict[str, Any]] = []
    for benchmark_dir in sorted([p for p in BENCHMARKS_DIR.iterdir() if p.is_dir()]):
        summary_file = benchmark_dir / "merged_summary.json"
        data = _load_json(summary_file)
        if not data:
            continue

        metadata = data.get("metadata", {})
        timestamp = metadata.get("timestamp", "")
        trace_archives = _trace_archives_for_timestamp(timestamp)
        benchmarks.append({
            "benchmark_id": benchmark_dir.name,
            "file_path": str(summary_file),
            "timestamp": timestamp,
            "total_rounds": data.get("total_rounds", 0),
            "total_successes": data.get("total_successes", 0),
            "verified_success": data.get("verified_success", 0),
            "success_rate": data.get("success_rate", 0.0),
            "avg_attempts_on_success": data.get("avg_attempts_on_success", 0.0),
            "top1_success": data.get("top1_success", 0),
            "top3_success": data.get("top3_success", 0),
            "top5_success": data.get("top5_success", 0),
            "extractor_metrics": data.get("extractor_metrics", {}),
            "worker_summaries": data.get("worker_summaries", []),
            "metadata": metadata,
            "trace_archive_count": len(trace_archives),
            "trace_archives": [archive.name for archive in trace_archives],
        })
    if offset:
        benchmarks = benchmarks[offset:]
    if limit is not None and limit >= 0:
        benchmarks = benchmarks[:limit]
    return benchmarks


def get_benchmark(benchmark_id: str) -> Optional[Dict[str, Any]]:
    """Load a benchmark summary and attach trace archive metadata."""
    summary_file = BENCHMARKS_DIR / benchmark_id / "merged_summary.json"
    data = _load_json(summary_file)
    if not data:
        return None

    metadata = data.get("metadata", {})
    timestamp = metadata.get("timestamp", "")
    trace_archives = [_summarize_trace_archive(path) for path in _trace_archives_for_timestamp(timestamp)]
    trace_runs = []
    for archive in trace_archives:
        trace_runs.extend(archive.get("runs", []))
    trace_runs.sort(key=lambda item: (item.get("timestamp", ""), item.get("run_id", "")))

    return {
        "benchmark_id": benchmark_id,
        "summary": data,
        "metadata": metadata,
        "worker_summaries": data.get("worker_summaries", []),
        "trace_archives": trace_archives,
        "trace_runs": trace_runs,
    }


def list_runs(limit: Optional[int] = None, offset: int = 0) -> List[Dict[str, Any]]:
    """List top-level run JSON files with metadata."""
    ensure_results_dir()
    runs: List[Dict[str, Any]] = []
    for f in sorted(RESULTS_DIR.glob("run_*.json"), key=os.path.getmtime, reverse=True):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = normalize_run(json.load(fp), f.stem)
            runs.append({
                "run_id": data.get("experiment", {}).get("run_id", f.stem),
                "file_path": str(f),
                "timestamp": data.get("experiment", {}).get("timestamp", ""),
                "scenario_id": data.get("experiment", {}).get("scenario_id", ""),
                "success": _overall_success(data.get("result", {})),
                "total_attempts": data.get("result", {}).get("total_attempts", 0),
                "access_code": data.get("scenario", {}).get("access_code", ""),
                "generator": data.get("models", {}).get("generator", {}).get("name", ""),
                "victim": data.get("models", {}).get("victim", {}).get("name", ""),
                "benchmark_mode": data.get("experiment", {}).get("benchmark_mode", False),
            })
        except (json.JSONDecodeError, KeyError) as e:
            runs.append({
                "run_id": f.stem,
                "file_path": str(f),
                "scenario_id": "",
                "error": str(e),
                "benchmark_mode": False,
            })
    if offset:
        runs = runs[offset:]
    if limit is not None and limit >= 0:
        runs = runs[:limit]
    return runs


def list_all_runs_recursive() -> List[Dict[str, Any]]:
    """List every run JSON, including dated trace archives."""
    ensure_results_dir()
    runs: List[Dict[str, Any]] = []
    for f in sorted(RESULTS_DIR.rglob("run_*.json"), key=os.path.getmtime, reverse=True):
        if _is_benchmark_artifact(f):
            continue
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = normalize_run(json.load(fp), f.stem)
            runs.append({
                "run_id": data.get("experiment", {}).get("run_id", f.stem),
                "file_path": str(f),
                "timestamp": data.get("experiment", {}).get("timestamp", ""),
                "scenario_id": data.get("experiment", {}).get("scenario_id", ""),
                "success": _overall_success(data.get("result", {})),
                "total_attempts": data.get("result", {}).get("total_attempts", 0),
                "access_code": data.get("scenario", {}).get("access_code", ""),
                "generator": data.get("models", {}).get("generator", {}).get("name", ""),
                "victim": data.get("models", {}).get("victim", {}).get("name", ""),
                "benchmark_mode": data.get("experiment", {}).get("benchmark_mode", False),
            })
        except (json.JSONDecodeError, KeyError) as e:
            runs.append({
                "run_id": f.stem,
                "file_path": str(f),
                "scenario_id": "",
                "error": str(e),
                "benchmark_mode": False,
            })
    return runs


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    """Load a specific run by run_id."""
    ensure_results_dir()
    candidates = list(RESULTS_DIR.glob("*.json")) + list(RESULTS_DIR.rglob("run_*.json"))
    seen: set[Path] = set()
    for path in sorted(candidates, key=os.path.getmtime, reverse=True):
        if path in seen or _is_benchmark_artifact(path):
            continue
        seen.add(path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = normalize_run(json.load(f), path.stem)
            if path.stem == run_id or data["experiment"]["run_id"] == run_id:
                return data
        except (json.JSONDecodeError, OSError, KeyError):
            continue
    return None


def upload_run(file_path: str) -> Dict[str, Any]:
    """Upload an external JSON file to results directory."""
    ensure_results_dir()
    src = Path(file_path)
    if not src.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(src, "r", encoding="utf-8") as f:
        data = normalize_run(json.load(f))

    run_id = data.get("experiment", {}).get("run_id", f"imported_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    safe_run_id = Path(run_id).name.replace(".json", "") or f"imported_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dest = RESULTS_DIR / f"{safe_run_id}.json"

    if dest.exists():
        timestamp = datetime.now().strftime("%H%M%S%f")[:-3]
        dest = RESULTS_DIR / f"{safe_run_id}_{timestamp}.json"

    data["experiment"]["run_id"] = dest.stem
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return {"run_id": dest.stem, "file_path": str(dest)}


def delete_run(run_id: str) -> bool:
    """Delete a top-level run JSON file."""
    for path in RESULTS_DIR.glob("*.json"):
        if path.stem == run_id:
            path.unlink()
            return True
    return False
