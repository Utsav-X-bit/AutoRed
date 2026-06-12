import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from .run_normalizer import normalize_run

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _overall_success(result: Dict[str, Any]) -> bool:
    return any(
        bool(result.get(key))
        for key in ("ground_truth_success", "extractor_success", "verified_success")
    )


def ensure_results_dir():
    """Create results directory if it doesn't exist."""
    RESULTS_DIR.mkdir(exist_ok=True)


def list_runs() -> List[Dict[str, Any]]:
    """List all run JSON files with metadata."""
    ensure_results_dir()
    runs = []
    for f in sorted(RESULTS_DIR.glob("run_*.json"), key=os.path.getmtime, reverse=True):
        try:
            with open(f, "r") as fp:
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
    for path in RESULTS_DIR.glob("*.json"):
        try:
            with open(path, "r") as f:
                data = normalize_run(json.load(f), path.stem)
            if path.stem == run_id or data["experiment"]["run_id"] == run_id:
                return data
        except (json.JSONDecodeError, OSError):
            continue
    return None


def upload_run(file_path: str) -> Dict[str, Any]:
    """Upload an external JSON file to results directory."""
    ensure_results_dir()
    src = Path(file_path)
    if not src.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(src, "r") as f:
        data = normalize_run(json.load(f))

    run_id = data.get("experiment", {}).get("run_id", f"imported_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    safe_run_id = Path(run_id).name.replace(".json", "") or f"imported_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dest = RESULTS_DIR / f"{safe_run_id}.json"

    # Avoid overwrite
    if dest.exists():
        timestamp = datetime.now().strftime("%H%M%S%f")[:-3]
        dest = RESULTS_DIR / f"{safe_run_id}_{timestamp}.json"

    data["experiment"]["run_id"] = dest.stem
    with open(dest, "w") as f:
        json.dump(data, f, indent=2)

    return {"run_id": dest.stem, "file_path": str(dest)}


def delete_run(run_id: str) -> bool:
    """Delete a run JSON file."""
    for path in RESULTS_DIR.glob("*.json"):
        if path.stem == run_id:
            path.unlink()
            return True
    return False
