import json
import os
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

RESULTS_DIR = Path("results")


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
                data = json.load(fp)
            runs.append({
                "run_id": data.get("experiment", {}).get("run_id", f.stem),
                "file_path": str(f),
                "timestamp": data.get("experiment", {}).get("timestamp", ""),
                "success": data.get("result", {}).get("ground_truth_success", False),
                "total_attempts": data.get("result", {}).get("total_attempts", 0),
                "access_code": data.get("scenario", {}).get("access_code", ""),
                "generator": data.get("models", {}).get("generator", {}).get("name", ""),
                "victim": data.get("models", {}).get("victim", {}).get("name", ""),
            })
        except (json.JSONDecodeError, KeyError) as e:
            runs.append({
                "run_id": f.stem,
                "file_path": str(f),
                "error": str(e),
            })
    return runs


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    """Load a specific run by run_id."""
    ensure_results_dir()
    # Try exact match first
    path = RESULTS_DIR / f"{run_id}.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    # Try glob pattern
    for p in RESULTS_DIR.glob(f"*{run_id}*"):
        with open(p, "r") as f:
            return json.load(f)
    return None


def upload_run(file_path: str) -> Dict[str, Any]:
    """Upload an external JSON file to results directory."""
    ensure_results_dir()
    src = Path(file_path)
    if not src.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(src, "r") as f:
        data = json.load(f)

    run_id = data.get("experiment", {}).get("run_id", f"imported_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    dest = RESULTS_DIR / f"{run_id}.json"

    # Avoid overwrite
    if dest.exists():
        timestamp = datetime.now().strftime("%H%M%S%f")[:-3]
        dest = RESULTS_DIR / f"{run_id}_{timestamp}.json"

    shutil.copy2(src, dest)

    return {"run_id": run_id, "file_path": str(dest)}


def delete_run(run_id: str) -> bool:
    """Delete a run JSON file."""
    path = RESULTS_DIR / f"{run_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False
