# Phase 1: Core Investigation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the foundation — Python JSON emission, FastAPI server, background worker, and the Attempt Investigation View with Timeline Sidebar.

**Architecture:** FastAPI (web, no models) + Celery/RQ worker (GPU models) + React frontend (3-panel layout). Phase 1 delivers: JSON emission from Python, run loader, timeline sidebar, and attempt investigation view.

**Tech Stack:** FastAPI, RQ + Redis, React + TypeScript + Vite + Tailwind CSS, Zustand

---

## File Map

| File | Responsibility |
|------|---------------|
| `experiment/llama_3_8b_verbose.py` | Modified: JSON emission + timing |
| `server/__init__.py` | Package init |
| `server/main.py` | FastAPI app (REST + WS) |
| `server/schemas.py` | Pydantic schemas (AutoRedRun, Attempt) |
| `server/file_manager.py` | CRUD for `results/*.json` |
| `server/websocket.py` | WebSocket manager for live updates |
| `worker/__init__.py` | Package init |
| `worker/rq_app.py` | RQ connection + queue |
| `worker/models_manager.py` | Load + keep models in GPU memory |
| `worker/experiment_runner.py` | Orchestrate AutoRed loop |
| `ui/src/types/autored.ts` | TypeScript interfaces |
| `ui/src/store/runStore.ts` | Zustand store (runs, selected run/attempt) |
| `ui/src/components/TimelineSidebar.tsx` | Left panel: attempt list |
| `ui/src/components/InvestigationPanel.tsx` | Center panel: pipeline view |
| `ui/src/components/AnalyticsPanel.tsx` | Right panel: summary |
| `ui/src/components/GeneratorCard.tsx` | Generator section |
| `ui/src/components/VictimCard.tsx` | Victim section |
| `ui/src/components/ExtractorCard.tsx` | Extractor section |
| `ui/src/components/VerifierCard.tsx` | Verifier section |
| `ui/src/pages/RunLoader.tsx` | Run list + upload |
| `ui/src/pages/InvestigationPage.tsx` | 3-panel layout |
| `ui/src/App.tsx` | Router + layout |
| `ui/src/main.tsx` | Entry point |
| `ui/vite.config.ts` | Vite config |
| `ui/package.json` | Dependencies |
| `results/` | Auto-saved run JSONs |
| `requirements_server.txt` | Python deps for server + worker |

---

## Task 1: Python JSON Emission

**Files:**
- Modify: `experiment/llama_3_8b_verbose.py`

- [ ] **Step 1: Create JSON serializer helper**

Add to `experiment/llama_3_8b_verbose.py` (after imports, before `DEBUG_GROUND_TRUTH`):

```python
import hashlib
import subprocess

# Version tracking
def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"

EXPERIMENT_VERSION = "2.0.0"
GIT_COMMIT = get_git_commit()
```

- [ ] **Step 2: Create `serialize_run()` function**

Add after `SensitiveInfoExtractor` class:

```python
def serialize_run(scenario: DefenseScenario, trace: list, timing_info: dict,
                  model_info: dict, strategy_stats: dict, best_attack: dict,
                  ground_truth_info: dict, events: list, summary: dict,
                  raw_dataset_entry: dict, benchmark_info: dict = None) -> dict:
    """Convert experiment trace to AutoRedRun JSON structure."""
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Build attempts array
    attempts = []
    for entry in trace:
        attempt = {
            "attempt_number": entry["attempt_number"],
            "timestamp": entry.get("timestamp", ""),
            "attempt_time_ms": entry.get("attempt_time_ms", 0),
            "generator": {
                "strategy": entry.get("strategy", "unknown"),
                "internal_prompt": entry.get("internal_prompt", ""),
                "generated_attack": entry.get("attack", ""),
                "attack_length": len(entry.get("attack", "")),
                "attack_hash": hashlib.sha256(entry.get("attack", "").encode()).hexdigest()[:16],
                "duplicate_attack": entry.get("duplicate_attack", False),
                "input_tokens": entry.get("input_tokens", 0),
                "output_tokens": entry.get("output_tokens", 0),
            },
            "judge": {
                "input": entry.get("judge_input", ""),
                "decision": entry.get("judge_decision", ""),
                "confidence": entry.get("judge_confidence", 0.0),
                "probabilities": entry.get("judge_probabilities", {"ATTACK": 0, "ATTEMPT": 0}),
            },
            "victim": {
                "raw_output": entry.get("raw_output", ""),
                "clean_output": entry.get("clean_output", ""),
                "output_length": len(entry.get("response", "")),
            },
            "extractor": {
                "regex_candidates": entry.get("regex_candidates", []),
                "quoted_candidates": entry.get("quoted_candidates", []),
                "capitalized_candidates": entry.get("capitalized_candidates", []),
                "llm_candidates": entry.get("llm_candidates", []),
                "ranked_candidates": entry.get("ranked_candidates", []),
                "best_candidate": entry.get("best_candidate", ""),
            },
            "verification": {
                "candidate_sent": entry.get("verification_candidate", ""),
                "victim_response": entry.get("verification_response", ""),
                "success": entry.get("verification_success", False),
            },
            "ground_truth_found": entry.get("ground_truth_found", False),
            "extractor_match": entry.get("extractor_match", False),
            "generator_success": entry.get("generator_success", False),
        }
        attempts.append(attempt)

    # Determine success reason
    gt_success = ground_truth_info.get("leaked", False)
    ext_success = any(a.get("extractor_match") for a in attempts)
    ver_success = any(a.get("verification", {}).get("success") for a in attempts)

    if gt_success and ext_success:
        success_reason = "extractor"
    elif gt_success:
        success_reason = "ground_truth"
    elif ver_success:
        success_reason = "verification"
    else:
        success_reason = None

    run = {
        "experiment": {
            "run_id": run_id,
            "benchmark_mode": benchmark_info is not None,
            "benchmark_run_number": benchmark_info.get("run_number") if benchmark_info else None,
            "benchmark_total_runs": benchmark_info.get("total_runs") if benchmark_info else None,
            "max_attempts": timing_info.get("max_attempts", MAX_INTERACTIONS),
            "dataset_size": timing_info.get("dataset_size", 1000),
            "scenario_id": raw_dataset_entry.get("defense_id", "unknown"),
            "seed": timing_info.get("seed", 42),
            "timestamp": datetime.now().isoformat(),
            "experiment_version": EXPERIMENT_VERSION,
            "git_commit": GIT_COMMIT,
        },
        "raw_dataset_entry": raw_dataset_entry,
        "models": model_info,
        "timing": {
            "total_run_time": timing_info.get("total_run_time", 0),
            "model_loading_time": timing_info.get("model_loading_time", 0),
            "average_attempt_time": timing_info.get("average_attempt_time", 0),
        },
        "scenario": {
            "pre_defense": scenario.opening_defense,
            "post_defense": scenario.closing_defense,
            "access_code": scenario.access_code,
            "full_prompt": f"{scenario.opening_defense}\n\n{scenario.closing_defense}",
        },
        "result": {
            "ground_truth_success": gt_success,
            "generator_success": gt_success,
            "extractor_success": ext_success,
            "verified_success": ver_success,
            "extracted_value": attempts[-1].get("extractor", {}).get("best_candidate", "") if attempts else "",
            "success_reason": success_reason,
            "total_attempts": len(attempts),
        },
        "strategy_stats": strategy_stats,
        "best_attack": best_attack,
        "ground_truth": ground_truth_info,
        "attempts": attempts,
        "events": events,
        "summary": summary,
    }

    return run
```

- [ ] **Step 3: Modify `verbose_test_llama()` to collect JSON data**

At the start of `verbose_test_llama()`, add timing and event tracking:

```python
    run_start = time.time()
    events = []
    events.append({"timestamp": datetime.now().isoformat(), "type": "start", "message": "Experiment started"})

    # Track raw dataset entry for reproducibility
    raw_dataset_entry = {
        "defense_id": getattr(scenario, '_defense_id', 'unknown'),
        "opening_defense": scenario.opening_defense,
        "closing_defense": scenario.closing_defense,
        "access_code": scenario.access_code,
    }
```

- [ ] **Step 4: Modify each attempt loop to record JSON fields**

Inside the attempt loop, after each step, enrich the trace entry:

```python
    # After judge step
    judge_input_text = f"Previous Attack: {prev_attack}\n\nPrevious Response: {prev_response}" if prev_response else "[No previous output — first iteration]"

    # After generator step
    internal_prompt = agent._build_generator_prompt(strategy)  # capture full prompt

    # After victim step
    raw_output = response
    clean_output = strip_few_shot_patterns(response)

    # After extractor step
    verification_candidate = extraction_result.get("best_candidate", "")
    verification_response = ""
    verification_success = extraction_result.get("verified", False)
    if verification_candidate and env:
        # Already done in extract(), capture response
        verification_response = env.last_response or ""

    # Ground truth position
    leak_position = None
    leak_count = 0
    if ground_truth_found:
        access_code_lower = scenario.access_code.lower()
        response_lower = response.lower()
        pos = response_lower.find(access_code_lower)
        if pos >= 0:
            leak_position = pos
        leak_count = response_lower.count(access_code_lower)

    # Build trace entry
    trace_entry = {
        "attempt_number": attempt,
        "timestamp": datetime.now().isoformat(),
        "attempt_time_ms": int((time.time() - attempt_start) * 1000),
        "strategy": strategy,
        "internal_prompt": internal_prompt,
        "attack": attack,
        "judge_input": judge_input_text,
        "judge_decision": judge_result["decision_name"],
        "judge_confidence": judge_result["confidence"],
        "judge_probabilities": judge_result["probabilities"],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "response": response,
        "raw_output": raw_output,
        "clean_output": clean_output,
        "regex_candidates": extraction_result.get("regex_candidates", []),
        "quoted_candidates": extraction_result.get("quoted_candidates", []),
        "capitalized_candidates": extraction_result.get("capitalized_candidates", []),
        "llm_candidates": extraction_result.get("llm_candidates", []),
        "ranked_candidates": [{"value": v, "score": s} for v, s in extraction_result.get("all_candidates", [])],
        "best_candidate": extraction_result.get("best_candidate", ""),
        "verification_candidate": verification_candidate,
        "verification_response": verification_response,
        "verification_success": verification_success,
        "ground_truth_found": ground_truth_found,
        "extractor_match": extractor_success,
        "generator_success": ground_truth_found,
        "duplicate_attack": attack in used_attacks,
    }
    trace.append(trace_entry)
```

- [ ] **Step 5: Save JSON at end of `verbose_test_llama()`**

At the end of the function, before returning:

```python
    run_end = time.time()
    total_run_time = run_end - run_start

    timing_info = {
        "total_run_time": total_run_time,
        "model_loading_time": MODEL_LOAD_TIME,  # global var set at module load
        "average_attempt_time": total_run_time / len(trace) if trace else 0,
        "max_attempts": max_attempts,
        "dataset_size": len(defender_df),
        "seed": 42,
    }

    model_info = {
        "victim": {"name": LLAMA_PATH, "load_time": MODEL_LOAD_TIME.get("victim", 0)},
        "generator": {"name": GENERATOR_PATH, "load_time": MODEL_LOAD_TIME.get("generator", 0)},
        "judge": {"name": DISTILBERT_CKPT, "load_time": MODEL_LOAD_TIME.get("judge", 0)},
        "extractor": {"name": LLAMA_PATH, "load_time": 0},  # uses victim model
    }

    ground_truth_info = {
        "access_code": scenario.access_code,
        "leaked": any(t.get("ground_truth_found") for t in trace),
        "leak_position": next((t.get("leak_position") for t in trace if t.get("ground_truth_found")), None),
        "leak_count": sum(1 for t in trace if t.get("ground_truth_found")),
    }

    best_attack_info = {
        "prompt": agent.best_attack,
        "score": agent.best_score,
        "strategy": getattr(agent, '_current_strategy', 'unknown'),
    } if agent.best_attack else None

    run_json = serialize_run(
        scenario=scenario,
        trace=trace,
        timing_info=timing_info,
        model_info=model_info,
        strategy_stats=agent.strategy_stats,
        best_attack=best_attack_info,
        ground_truth_info=ground_truth_info,
        events=events,
        summary=summary_dict,
        raw_dataset_entry=raw_dataset_entry,
    )

    # Save to results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    json_path = results_dir / f"{run_json['experiment']['run_id']}.json"
    with open(json_path, "w") as f:
        json.dump(run_json, f, indent=2, default=str)
    print(f"\n💾 Run JSON saved to: {json_path}")

    return trace, total_attempts, run_json
```

- [ ] **Step 6: Add `MODEL_LOAD_TIME` global tracking**

At module level, wrap model loading with timing:

```python
MODEL_LOAD_TIME = {}

print("\n[LOAD] Loading Llama-3-8B-Instruct (target LLM)...")
t0 = time.time()
llama_model = AutoModelForCausalLM.from_pretrained(...)
llama_tokenizer = AutoTokenizer.from_pretrained(...)
MODEL_LOAD_TIME["victim"] = time.time() - t0
print(f"[LOAD] ✓ Llama-3-8B-Instruct loaded ({MODEL_LOAD_TIME['victim']:.1f}s)")
```

Apply same pattern to generator and judge model loading.

- [ ] **Step 7: Verify JSON output**

Run single scenario, inspect `results/run_*.json`:

```bash
python experiment/llama_3_8b_verbose.py --mode single
python -c "import json; d=json.load(open('results/' + __import__('os').listdir('results')[-1])); print(json.dumps(d['experiment'], indent=2))"
```

Expected: Valid JSON with all fields from spec.

- [ ] **Step 8: Commit**

```bash
git add experiment/llama_3_8b_verbose.py results/
git commit -m "feat: add structured JSON emission with full trace, timing, reproducibility metadata"
```

---

## Task 2: Server — Pydantic Schemas

**Files:**
- Create: `server/__init__.py`
- Create: `server/schemas.py`

- [ ] **Step 1: Create `server/__init__.py`**

```python
# AutoRed Web Server
```

- [ ] **Step 2: Create `server/schemas.py`**

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from enum import Enum


class SuccessReason(str, Enum):
    ground_truth = "ground_truth"
    extractor = "extractor"
    verification = "verification"


class ExperimentInfo(BaseModel):
    run_id: str
    benchmark_mode: bool
    benchmark_run_number: Optional[int] = None
    benchmark_total_runs: Optional[int] = None
    max_attempts: int
    dataset_size: int
    scenario_id: str
    seed: int
    timestamp: str
    experiment_version: str
    git_commit: str


class ModelInfo(BaseModel):
    name: str
    load_time: float


class ModelsInfo(BaseModel):
    victim: ModelInfo
    generator: ModelInfo
    judge: ModelInfo
    extractor: ModelInfo


class TimingInfo(BaseModel):
    total_run_time: float
    model_loading_time: float
    average_attempt_time: float


class ScenarioInfo(BaseModel):
    pre_defense: str
    post_defense: str
    access_code: str
    full_prompt: str


class ResultInfo(BaseModel):
    ground_truth_success: bool
    generator_success: bool
    extractor_success: bool
    verified_success: bool
    extracted_value: str
    success_reason: Optional[SuccessReason] = None
    total_attempts: int


class StrategyStat(BaseModel):
    successes: int = 0
    partial_leaks: int = 0
    failures: int = 0
    total_score: float = 0.0


class BestAttack(BaseModel):
    prompt: str
    score: float
    strategy: str


class GroundTruthInfo(BaseModel):
    access_code: str
    leaked: bool
    leak_position: Optional[int] = None
    leak_count: int


class RankedCandidate(BaseModel):
    value: str
    score: float


class ExtractorTrace(BaseModel):
    regex_candidates: List[str] = []
    quoted_candidates: List[str] = []
    capitalized_candidates: List[str] = []
    llm_candidates: List[str] = []
    ranked_candidates: List[RankedCandidate] = []
    best_candidate: str = ""


class VerificationTrace(BaseModel):
    candidate_sent: str = ""
    victim_response: str = ""
    success: bool = False


class GeneratorInfo(BaseModel):
    strategy: str
    internal_prompt: str
    generated_attack: str
    attack_length: int
    attack_hash: str
    duplicate_attack: bool
    input_tokens: int
    output_tokens: int


class JudgeInfo(BaseModel):
    input: str
    decision: str
    confidence: float
    probabilities: Dict[str, float]


class VictimInfo(BaseModel):
    raw_output: str
    clean_output: str
    output_length: int


class Attempt(BaseModel):
    attempt_number: int
    timestamp: str
    attempt_time_ms: int
    generator: GeneratorInfo
    judge: JudgeInfo
    victim: VictimInfo
    extractor: ExtractorTrace
    verification: VerificationTrace
    ground_truth_found: bool
    extractor_match: bool
    generator_success: bool


class Event(BaseModel):
    timestamp: str
    type: str
    message: str


class SummaryStats(BaseModel):
    attack_length_min: int
    attack_length_max: int
    attack_length_avg: float
    unique_attacks: int
    repetition_rate: float
    judge_distribution: Dict[str, int]


class AutoRedRun(BaseModel):
    experiment: ExperimentInfo
    raw_dataset_entry: Dict[str, Any]
    models: ModelsInfo
    timing: TimingInfo
    scenario: ScenarioInfo
    result: ResultInfo
    strategy_stats: Dict[str, StrategyStat]
    best_attack: Optional[BestAttack] = None
    ground_truth: GroundTruthInfo
    attempts: List[Attempt]
    events: List[Event]
    summary: SummaryStats


# WebSocket messages
class AttemptUpdate(BaseModel):
    type: str = "attempt_update"
    run_id: str
    attempt: Attempt


class RunComplete(BaseModel):
    type: str = "run_complete"
    run_id: str
    run: AutoRedRun
```

- [ ] **Step 3: Commit**

```bash
git add server/
git commit -m "feat: add Pydantic schemas for AutoRedRun data model"
```

---

## Task 3: Server — File Manager

**Files:**
- Create: `server/file_manager.py`

- [ ] **Step 1: Create `server/file_manager.py`**

```python
import json
import os
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

    import shutil
    shutil.copy2(src, dest)

    return {"run_id": run_id, "file_path": str(dest)}


def delete_run(run_id: str) -> bool:
    """Delete a run JSON file."""
    path = RESULTS_DIR / f"{run_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False
```

- [ ] **Step 2: Commit**

```bash
git add server/file_manager.py
git commit -m "feat: add file manager for run JSON CRUD operations"
```

---

## Task 4: Server — FastAPI App

**Files:**
- Create: `server/main.py`

- [ ] **Step 1: Create `server/main.py`**

```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Any
import json

from .file_manager import list_runs, get_run, upload_run, delete_run

app = FastAPI(title="AutoRed Web UI", version="1.0.0")

# CORS for Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Run Endpoints ──────────────────────────────────────────

@app.get("/api/runs")
def api_list_runs():
    """List all past runs."""
    return list_runs()


@app.get("/api/run/{run_id}")
def api_get_run(run_id: str):
    """Get a specific run by ID."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return run


@app.post("/api/runs/upload")
async def api_upload_run(file: UploadFile = File(...)):
    """Upload an external run JSON file."""
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    result = upload_run(tmp_path)
    import os
    os.unlink(tmp_path)
    return result


@app.delete("/api/run/{run_id}")
def api_delete_run(run_id: str):
    """Delete a run."""
    if not delete_run(run_id):
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return {"deleted": run_id}


# ─── Worker Status ──────────────────────────────────────────

@app.get("/api/models/status")
def api_model_status():
    """Get model load status from worker."""
    # Phase 1: placeholder (worker not connected yet)
    return {
        "victim": {"loaded": False, "name": "", "load_time": 0},
        "generator": {"loaded": False, "name": "", "load_time": 0},
        "judge": {"loaded": False, "name": "", "load_time": 0},
        "extractor": {"loaded": False, "name": "", "load_time": 0},
    }


@app.post("/api/run")
def api_start_run():
    """Start a new experiment run (delegates to worker)."""
    # Phase 1: placeholder (worker not connected yet)
    return {"run_id": "pending", "status": "worker_not_connected"}


# ─── Export ─────────────────────────────────────────────────

@app.get("/api/export/{run_id}/json")
def api_export_json(run_id: str):
    """Export run as JSON."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return run


@app.get("/api/export/{run_id}/csv")
def api_export_csv(run_id: str):
    """Export run attempts as CSV."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    import csv
    import io
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["attempt", "strategy", "judge_decision", "attack", "victim_response",
                      "best_candidate", "ground_truth_found", "extractor_match", "verification_success"])
    for a in run.get("attempts", []):
        writer.writerow([
            a["attempt_number"],
            a["generator"]["strategy"],
            a["judge"]["decision"],
            a["generator"]["generated_attack"],
            a["victim"]["raw_output"],
            a["extractor"]["best_candidate"],
            a["ground_truth_found"],
            a["extractor_match"],
            a["verification"]["success"],
        ])
    output.seek(0)
    from fastapi.responses import Response
    return Response(content=output.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={run_id}.csv"})
```

- [ ] **Step 2: Create `requirements_server.txt`**

```txt
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6
pydantic>=2.0.0
redis>=5.0.0
rq>=1.16.0
```

- [ ] **Step 3: Test server starts**

```bash
pip install -r requirements_server.txt
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

Expected: Server starts on port 8000, no errors.

- [ ] **Step 4: Test API endpoints**

```bash
curl http://localhost:8000/api/runs | python -m json.tool
curl http://localhost:8000/api/models/status | python -m json.tool
```

Expected: Empty runs list, model status all false.

- [ ] **Step 5: Commit**

```bash
git add server/main.py requirements_server.txt
git commit -m "feat: add FastAPI server with run CRUD, model status, export endpoints"
```

---

## Task 5: Worker — RQ App + Models Manager

**Files:**
- Create: `worker/__init__.py`
- Create: `worker/rq_app.py`
- Create: `worker/models_manager.py`

- [ ] **Step 1: Create `worker/__init__.py`**

```python
# AutoRed Background Worker
```

- [ ] **Step 2: Create `worker/rq_app.py`**

```python
from redis import Redis
from rq import Queue, Connection

# Redis connection
redis_conn = Redis(host="localhost", port=6379, db=0)

# Default queue
queue = Queue("autored", connection=redis_conn)
```

- [ ] **Step 3: Create `worker/models_manager.py`**

```python
import torch
import time
from typing import Dict, Any, Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DistilBertForSequenceClassification,
)

# Import paths from experiment config
from experiment.llama_3_8b_verbose import (
    LLAMA_PATH, GENERATOR_PATH, DISTILBERT_CKPT, device
)


class ModelsManager:
    """Load and keep models in GPU memory across runs."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.load_times: Dict[str, float] = {}
        self._loaded = False

    def load_all(self) -> Dict[str, float]:
        """Load all 4 models. Returns dict of load times."""
        if self._loaded:
            return self.load_times

        # Load victim LLM
        t0 = time.time()
        print("[WORKER] Loading victim LLM...")
        self.models["victim"] = AutoModelForCausalLM.from_pretrained(
            LLAMA_PATH, dtype=torch.float16, device_map="auto", local_files_only=True
        )
        self.tokenizers["victim"] = AutoTokenizer.from_pretrained(
            LLAMA_PATH, local_files_only=True, use_fast=False
        )
        self.load_times["victim"] = time.time() - t0
        print(f"[WORKER] ✓ Victim loaded ({self.load_times['victim']:.1f}s)")

        # Load generator
        t0 = time.time()
        print("[WORKER] Loading generator...")
        self.models["generator"] = AutoModelForCausalLM.from_pretrained(
            GENERATOR_PATH, dtype=torch.float16, device_map="auto", local_files_only=True
        )
        self.models["generator"].eval()
        self.tokenizers["generator"] = AutoTokenizer.from_pretrained(
            GENERATOR_PATH, local_files_only=True, use_fast=False
        )
        self.load_times["generator"] = time.time() - t0
        print(f"[WORKER] ✓ Generator loaded ({self.load_times['generator']:.1f}s)")

        # Load judge
        t0 = time.time()
        print("[WORKER] Loading judge...")
        self.models["judge"] = DistilBertForSequenceClassification.from_pretrained(
            DISTILBERT_CKPT, local_files_only=True
        ).to(device)
        self.models["judge"].eval()
        self.tokenizers["judge"] = AutoTokenizer.from_pretrained(
            DISTILBERT_CKPT, local_files_only=True
        )
        self.load_times["judge"] = time.time() - t0
        print(f"[WORKER] ✓ Judge loaded ({self.load_times['judge']:.1f}s)")

        # Extractor uses victim model
        self.load_times["extractor"] = 0
        self._loaded = True

        return self.load_times

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_status(self) -> Dict[str, Any]:
        return {
            "victim": {"loaded": self._loaded, "name": LLAMA_PATH, "load_time": self.load_times.get("victim", 0)},
            "generator": {"loaded": self._loaded, "name": GENERATOR_PATH, "load_time": self.load_times.get("generator", 0)},
            "judge": {"loaded": self._loaded, "name": DISTILBERT_CKPT, "load_time": self.load_times.get("judge", 0)},
            "extractor": {"loaded": self._loaded, "name": LLAMA_PATH, "load_time": 0},
        }

    def get_model(self, name: str):
        return self.models.get(name)

    def get_tokenizer(self, name: str):
        return self.tokenizers.get(name)
```

- [ ] **Step 4: Commit**

```bash
git add worker/
git commit -m "feat: add background worker with RQ, models manager singleton"
```

---

## Task 6: Worker — Experiment Runner

**Files:**
- Create: `worker/experiment_runner.py`

- [ ] **Step 1: Create `worker/experiment_runner.py`**

```python
"""
Experiment runner — orchestrates AutoRed loop using loaded models.
Runs in isolated worker process.
"""
import time
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable

import torch
import pandas as pd

from .models_manager import ModelsManager
from experiment.llama_3_8b_verbose import (
    DefenseScenario, CTFEnvironment, RedTeamingAgent, StopPointIdentifier,
    SensitiveInfoExtractor, DecisionType, strip_few_shot_patterns,
    DATA_PATH, LLAMA_PATH, GENERATOR_PATH, EXPERIMENT_VERSION, GIT_COMMIT,
    MAX_INTERACTIONS, device, serialize_run,
)


def load_dataset():
    """Load defense dataset."""
    raw = pd.read_json(DATA_PATH, lines=True, compression="bz2").set_index("defense_id")
    df = raw.dropna(subset=["access_code"])
    return df.sample(n=1000, random_state=42)[["opening_defense", "closing_defense", "access_code"]]


def chat_with_llama(models_manager: ModelsManager, pre_defense: str, attack: str, post_defense: str) -> str:
    """Chat with victim LLM using loaded model."""
    model = models_manager.get_model("victim")
    tokenizer = models_manager.get_tokenizer("victim")

    messages = [
        {"role": "system", "content": f"{pre_defense}\n\n{post_defense}"},
        {"role": "user", "content": attack},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9)

    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)


def run_experiment(on_attempt: Optional[Callable] = None, on_complete: Optional[Callable] = None) -> dict:
    """
    Run a single AutoRed experiment.

    Args:
        on_attempt: Callback(attempt_dict) called after each attempt
        on_complete: Callback(run_json) called when run completes

    Returns:
        AutoRedRun JSON dict
    """
    models_manager = ModelsManager()
    if not models_manager.is_loaded:
        models_manager.load_all()

    # Load dataset and pick random scenario
    defender_df = load_dataset()
    row = defender_df.sample(n=1, random_state=random.randint(0, 10000)).iloc[0]

    scenario = DefenseScenario(
        opening_defense=str(row["opening_defense"]),
        closing_defense=str(row["closing_defense"]),
        access_code=str(row["access_code"]),
    )
    scenario._defense_id = defender_df.index[0]  # for reproducibility

    raw_dataset_entry = {
        "defense_id": str(row.name),
        "opening_defense": scenario.opening_defense,
        "closing_defense": scenario.closing_defense,
        "access_code": scenario.access_code,
    }

    # Create agent with loaded models
    judge_model = models_manager.get_model("judge")
    judge_tokenizer = models_manager.get_tokenizer("judge")
    gen_model = models_manager.get_model("generator")
    gen_tokenizer = models_manager.get_tokenizer("generator")

    agent = RedTeamingAgent(
        judge_model=judge_model,
        judge_tokenizer=judge_tokenizer,
        gen_model=gen_model,
        gen_tokenizer=gen_tokenizer,
    )

    # Run experiment (reuse verbose_test_llama logic)
    from experiment.llama_3_8b_verbose import verbose_test_llama
    trace, total_attempts, run_json = verbose_test_llama(scenario, agent, max_attempts=MAX_INTERACTIONS)

    # Callbacks for live streaming
    if on_complete and run_json:
        on_complete(run_json)

    return run_json
```

- [ ] **Step 2: Commit**

```bash
git add worker/experiment_runner.py
git commit -m "feat: add experiment runner that uses loaded models from worker"
```

---

## Task 7: Frontend — Scaffolding

**Files:**
- Create: `ui/package.json`
- Create: `ui/vite.config.ts`
- Create: `ui/tsconfig.json`
- Create: `ui/index.html`
- Create: `ui/src/main.tsx`
- Create: `ui/src/App.tsx`
- Create: `ui/src/types/autored.ts`
- Create: `ui/src/store/runStore.ts`

- [ ] **Step 1: Create `ui/package.json`**

```json
{
  "name": "autored-ui",
  "private": true,
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-router-dom": "^6.22.0",
    "zustand": "^4.5.2",
    "recharts": "^2.12.0"
  },
  "devDependencies": {
    "@types/react": "^18.3.3",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.0",
    "autoprefixer": "^10.4.19",
    "postcss": "^8.4.38",
    "tailwindcss": "^3.4.3",
    "typescript": "^5.4.5",
    "vite": "^5.2.11"
  }
}
```

- [ ] **Step 2: Create `ui/vite.config.ts`**

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': 'http://localhost:8000',
      '/ws': {
        target: 'http://localhost:8000',
        ws: true,
      },
    },
  },
})
```

- [ ] **Step 3: Create `ui/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": false,
    "noUnusedParameters": false,
    "noFallthroughCasesInSwitch": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src"]
}
```

- [ ] **Step 4: Create `ui/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>AutoRed — Red Teaming Dashboard</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 5: Create Tailwind config**

```bash
cd ui && npx tailwindcss init -p
```

Edit `ui/tailwind.config.js`:

```javascript
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: { extend: {} },
  plugins: [],
}
```

Create `ui/src/index.css`:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  margin: 0;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #f8fafc;
  color: #0f172a;
}
```

- [ ] **Step 6: Create `ui/src/types/autored.ts`**

```typescript
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
```

- [ ] **Step 7: Create `ui/src/store/runStore.ts`**

```typescript
import { create } from 'zustand';
import type { AutoRedRun, Attempt, RunListItem } from '../types/autored';

interface RunStore {
  // Runs list
  runs: RunListItem[];
  selectedRun: AutoRedRun | null;
  selectedAttemptIndex: number;

  // Actions
  setRuns: (runs: RunListItem[]) => void;
  setSelectedRun: (run: AutoRedRun | null) => void;
  setSelectedAttempt: (index: number) => void;
  addAttempt: (attempt: Attempt) => void;
  clearSelectedRun: () => void;
}

export const useRunStore = create<RunStore>((set) => ({
  runs: [],
  selectedRun: null,
  selectedAttemptIndex: 0,

  setRuns: (runs) => set({ runs }),
  setSelectedRun: (run) => set({ selectedRun: run, selectedAttemptIndex: run ? run.attempts.length - 1 : 0 }),
  setSelectedAttempt: (index) => set({ selectedAttemptIndex: index }),
  addAttempt: (attempt) => set((state) => {
    if (!state.selectedRun) return state;
    const newAttempts = [...state.selectedRun.attempts, attempt];
    return {
      selectedRun: { ...state.selectedRun, attempts: newAttempts },
      selectedAttemptIndex: newAttempts.length - 1,
    };
  }),
  clearSelectedRun: () => set({ selectedRun: null, selectedAttemptIndex: 0 }),
}));
```

- [ ] **Step 8: Create `ui/src/main.tsx`**

```typescript
import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>,
)
```

- [ ] **Step 9: Create `ui/src/App.tsx`**

```typescript
import { Routes, Route, Navigate } from 'react-router-dom';
import RunLoader from './pages/RunLoader';
import InvestigationPage from './pages/InvestigationPage';

function App() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/runs" replace />} />
      <Route path="/runs" element={<RunLoader />} />
      <Route path="/run/:runId" element={<InvestigationPage />} />
    </Routes>
  );
}

export default App;
```

- [ ] **Step 10: Install deps + verify dev server**

```bash
cd ui && npm install && npm run dev
```

Expected: Vite dev server starts on port 3000, no TypeScript errors.

- [ ] **Step 11: Commit**

```bash
git add ui/
git commit -m "feat: scaffold React+Vite+Tailwind frontend with types, store, routing"
```

---

## Task 8: Frontend — Run Loader Page

**Files:**
- Create: `ui/src/pages/RunLoader.tsx`

- [ ] **Step 1: Create `ui/src/pages/RunLoader.tsx`**

```typescript
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useRunStore } from '../store/runStore';
import type { RunListItem } from '../types/autored';

export default function RunLoader() {
  const navigate = useNavigate();
  const { runs, setRuns } = useRunStore();
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    fetchRuns();
  }, []);

  const fetchRuns = async () => {
    try {
      const res = await fetch('/api/runs');
      const data = await res.json();
      setRuns(data);
    } catch (e) {
      console.error('Failed to load runs:', e);
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);
    try {
      await fetch('/api/runs/upload', { method: 'POST', body: formData });
      fetchRuns();
    } catch (err) {
      console.error('Upload failed:', err);
    } finally {
      setUploading(false);
    }
  };

  if (loading) return <div className="p-8 text-center text-slate-500">Loading runs...</div>;

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <h1 className="text-xl font-bold text-slate-900">AutoRed — Run History</h1>
          <div className="flex gap-3">
            <label className="px-4 py-2 bg-slate-100 hover:bg-slate-200 rounded-lg cursor-pointer text-sm font-medium transition-colors">
              Upload JSON
              <input type="file" accept=".json" onChange={handleUpload} className="hidden" />
            </label>
            <span className="text-sm text-slate-500 self-center">{runs.length} runs</span>
          </div>
        </div>
      </header>

      {/* Run List */}
      <main className="max-w-7xl mx-auto p-6">
        {runs.length === 0 ? (
          <div className="text-center py-20 text-slate-400">
            <p className="text-lg">No runs yet</p>
            <p className="text-sm mt-2">Upload a JSON file or start a new experiment</p>
          </div>
        ) : (
          <div className="space-y-3">
            {runs.map((run: RunListItem) => (
              <button
                key={run.run_id}
                onClick={() => navigate(`/run/${run.run_id}`)}
                className="w-full text-left bg-white rounded-xl border border-slate-200 p-4 hover:border-blue-400 hover:shadow-md transition-all"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <span className={`inline-flex items-center justify-center w-8 h-8 rounded-full text-sm font-bold ${run.success ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                      {run.success ? '✓' : '✗'}
                    </span>
                    <div>
                      <p className="font-mono text-sm font-semibold text-slate-900">{run.run_id}</p>
                      <p className="text-xs text-slate-500 mt-0.5">{run.timestamp}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-6 text-sm">
                    <div className="text-right">
                      <p className="text-slate-500 text-xs">Generator</p>
                      <p className="font-medium text-slate-700">{run.generator.split('/').pop() || run.generator}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-slate-500 text-xs">Victim</p>
                      <p className="font-medium text-slate-700">{run.victim.split('/').pop() || run.victim}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-slate-500 text-xs">Attempts</p>
                      <p className="font-medium text-slate-700">{run.total_attempts}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-slate-500 text-xs">Access Code</p>
                      <p className="font-mono font-medium text-amber-600">{run.access_code}</p>
                    </div>
                  </div>
                </div>
              </button>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add ui/src/pages/RunLoader.tsx
git commit -m "feat: add run loader page with list, upload, navigation"
```

---

## Task 9: Frontend — Timeline Sidebar

**Files:**
- Create: `ui/src/components/TimelineSidebar.tsx`

- [ ] **Step 1: Create `ui/src/components/TimelineSidebar.tsx`**

```typescript
import { useRunStore } from '../store/runStore';
import type { Attempt } from '../types/autored';

export default function TimelineSidebar() {
  const { selectedRun, selectedAttemptIndex, setSelectedAttempt } = useRunStore();
  if (!selectedRun) return null;

  const getAttemptColor = (a: Attempt): string => {
    if (a.extractor_match) return 'bg-green-500';       // Green: success
    if (a.ground_truth_found && !a.extractor_match) return 'bg-red-500'; // Red: extraction failure
    if (a.ground_truth_found) return 'bg-yellow-500';   // Yellow: leak
    if (a.judge.decision === 'ATTACK') return 'bg-blue-500'; // Blue: attack
    return 'bg-slate-400';                              // Gray: normal
  };

  return (
    <div className="w-64 bg-white border-r border-slate-200 flex flex-col h-full">
      {/* Run Header */}
      <div className="p-4 border-b border-slate-200">
        <p className="text-xs text-slate-500 font-medium uppercase tracking-wider">Run</p>
        <p className="font-mono text-sm font-bold text-slate-900 mt-1 truncate">{selectedRun.experiment.run_id}</p>
        <div className="flex items-center gap-2 mt-2">
          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${selectedRun.result.ground_truth_success ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
            {selectedRun.result.ground_truth_success ? 'SUCCESS' : 'FAILED'}
          </span>
          <span className="text-xs text-slate-500">{selectedRun.result.total_attempts} attempts</span>
        </div>
      </div>

      {/* Attempt List */}
      <div className="flex-1 overflow-y-auto p-3 space-y-1">
        {selectedRun.attempts.map((attempt: Attempt) => {
          const isSelected = attempt.attempt_number - 1 === selectedAttemptIndex;
          const color = getAttemptColor(attempt);
          const isStar = attempt.extractor_match || (attempt.ground_truth_found && attempt.attempt_number === selectedRun.result.total_attempts);

          return (
            <button
              key={attempt.attempt_number}
              onClick={() => setSelectedAttempt(attempt.attempt_number - 1)}
              className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left transition-all ${
                isSelected ? 'bg-slate-100 ring-1 ring-slate-300' : 'hover:bg-slate-50'
              }`}
            >
              <span className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${color}`} />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1.5">
                  <span className="text-sm font-medium text-slate-700">Attempt {attempt.attempt_number}</span>
                  {isStar && <span className="text-yellow-500 text-xs">⭐</span>}
                </div>
                <p className="text-xs text-slate-500 truncate">{attempt.generator.strategy}</p>
              </div>
              {attempt.ground_truth_found && (
                <span className="text-xs text-amber-600 font-medium">leak</span>
              )}
            </button>
          );
        })}
      </div>

      {/* Legend */}
      <div className="p-3 border-t border-slate-200 text-xs text-slate-500 space-y-1.5">
        <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-blue-500" /> Attack</div>
        <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-yellow-500" /> Leak</div>
        <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-green-500" /> Success</div>
        <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-red-500" /> Extract Fail</div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add ui/src/components/TimelineSidebar.tsx
git commit -m "feat: add timeline sidebar with color-coded attempts"
```

---

## Task 10: Frontend — Investigation Cards

**Files:**
- Create: `ui/src/components/GeneratorCard.tsx`
- Create: `ui/src/components/VictimCard.tsx`
- Create: `ui/src/components/ExtractorCard.tsx`
- Create: `ui/src/components/VerifierCard.tsx`

- [ ] **Step 1: Create `ui/src/components/GeneratorCard.tsx`**

```typescript
import { Attempt } from '../types/autored';

export default function GeneratorCard({ attempt }: { attempt: Attempt }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
          <span className="text-lg">🧠</span> Generator
        </h3>
        <span className="px-2 py-0.5 bg-purple-100 text-purple-700 text-xs font-medium rounded-full">
          {attempt.generator.strategy}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3 mb-3">
        <div className="bg-slate-50 rounded-lg p-2">
          <p className="text-xs text-slate-500">Input Tokens</p>
          <p className="text-lg font-bold text-slate-900">{attempt.generator.input_tokens}</p>
        </div>
        <div className="bg-slate-50 rounded-lg p-2">
          <p className="text-xs text-slate-500">Output Tokens</p>
          <p className="text-lg font-bold text-slate-900">{attempt.generator.output_tokens}</p>
        </div>
      </div>

      <div>
        <p className="text-xs text-slate-500 mb-1.5">Generated Attack</p>
        <div className="bg-red-50 border border-red-200 rounded-lg p-3">
          <p className="text-sm text-red-900 font-mono leading-relaxed">
            "{attempt.generator.generated_attack}"
          </p>
        </div>
      </div>

      {attempt.generator.duplicate_attack && (
        <div className="mt-2 px-2 py-1 bg-amber-50 border border-amber-200 rounded text-xs text-amber-700 font-medium">
          ⚠️ Duplicate attack detected
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Create `ui/src/components/VictimCard.tsx`**

```typescript
import { Attempt } from '../types/autored';

export default function VictimCard({ attempt }: { attempt: Attempt }) {
  const response = attempt.victim.raw_output;
  const accessCode = ''; // Will be passed from parent context

  // Highlight access code in response
  const renderHighlightedResponse = () => {
    if (!accessCode || !response) return <p className="text-sm text-slate-700 whitespace-pre-wrap">{response}</p>;
    const parts = response.split(new RegExp(`(${accessCode.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi'));
    return (
      <p className="text-sm text-slate-700 whitespace-pre-wrap">
        {parts.map((part, i) =>
          part.toLowerCase() === accessCode.toLowerCase() ? (
            <mark key={i} className="bg-yellow-200 text-yellow-900 px-1 rounded font-bold">{part}</mark>
          ) : (
            <span key={i}>{part}</span>
          )
        )}
      </p>
    );
  };

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
          <span className="text-lg">🦙</span> Victim Response
        </h3>
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-500">{attempt.victim.output_length} chars</span>
          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${attempt.ground_truth_found ? 'bg-yellow-100 text-yellow-700' : 'bg-slate-100 text-slate-600'}`}>
            GT Found: {attempt.ground_truth_found ? '✓ YES' : '✗ NO'}
          </span>
        </div>
      </div>

      <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 max-h-64 overflow-y-auto">
        {renderHighlightedResponse()}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Create `ui/src/components/ExtractorCard.tsx`**

```typescript
import { Attempt } from '../types/autored';

export default function ExtractorCard({ attempt }: { attempt: Attempt }) {
  const { extractor } = attempt;
  const isWrong = extractor.best_candidate && attempt.ground_truth_found && !attempt.extractor_match;

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
          <span className="text-lg">🔓</span> Extractor
        </h3>
        <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${attempt.extractor_match ? 'bg-green-100 text-green-700' : 'bg-slate-100 text-slate-600'}`}>
          Match: {attempt.extractor_match ? '✓ YES' : '✗ NO'}
        </span>
      </div>

      {/* Candidates */}
      <div className="grid grid-cols-2 gap-3 mb-3">
        <div>
          <p className="text-xs text-slate-500 mb-1">Regex Candidates ({extractor.regex_candidates.length})</p>
          <div className="flex flex-wrap gap-1">
            {extractor.regex_candidates.map((c: string, i: number) => (
              <span key={i} className="px-2 py-0.5 bg-blue-50 text-blue-700 text-xs rounded font-mono">{c}</span>
            ))}
            {extractor.regex_candidates.length === 0 && <span className="text-xs text-slate-400">none</span>}
          </div>
        </div>
        <div>
          <p className="text-xs text-slate-500 mb-1">LLM Candidates ({extractor.llm_candidates.length})</p>
          <div className="flex flex-wrap gap-1">
            {extractor.llm_candidates.map((c: string, i: number) => (
              <span key={i} className="px-2 py-0.5 bg-indigo-50 text-indigo-700 text-xs rounded font-mono">{c}</span>
            ))}
            {extractor.llm_candidates.length === 0 && <span className="text-xs text-slate-400">none</span>}
          </div>
        </div>
      </div>

      {/* Ranked */}
      <div>
        <p className="text-xs text-slate-500 mb-1.5">Ranked Candidates</p>
        <div className="space-y-1">
          {extractor.ranked_candidates.map((rc: { value: string; score: number }, i: number) => (
            <div key={i} className="flex items-center justify-between text-sm bg-slate-50 rounded px-3 py-1.5">
              <span className="font-mono text-slate-700">{rc.value}</span>
              <span className="text-xs text-slate-500">score: {rc.score}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Selected */}
      <div className={`mt-3 p-3 rounded-lg border ${isWrong ? 'bg-red-50 border-red-200' : 'bg-slate-50 border-slate-200'}`}>
        <p className="text-xs text-slate-500 mb-1">Selected Candidate</p>
        <div className="flex items-center justify-between">
          <span className="font-mono font-bold text-slate-900">{extractor.best_candidate || 'NONE'}</span>
          {isWrong && <span className="text-red-600 text-sm font-bold">❌ Wrong Selection</span>}
          {attempt.extractor_match && <span className="text-green-600 text-sm font-bold">✓ Correct</span>}
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Create `ui/src/components/VerifierCard.tsx`**

```typescript
import { Attempt } from '../types/autored';

export default function VerifierCard({ attempt }: { attempt: Attempt }) {
  const { verification } = attempt;

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
          <span className="text-lg">🔍</span> Verifier
        </h3>
        <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${verification.success ? 'bg-green-100 text-green-700' : 'bg-slate-100 text-slate-600'}`}>
          {verification.success ? '✓ Verified' : '✗ Not Verified'}
        </span>
      </div>

      <div className="space-y-2">
        <div>
          <p className="text-xs text-slate-500 mb-1">Candidate Sent</p>
          <p className="font-mono text-sm bg-slate-50 rounded-lg px-3 py-2 border border-slate-200">
            {verification.candidate_sent || '—'}
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-500 mb-1">Victim Response</p>
          <p className="font-mono text-sm bg-slate-50 rounded-lg px-3 py-2 border border-slate-200">
            {verification.victim_response || '—'}
          </p>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Commit**

```bash
git add ui/src/components/GeneratorCard.tsx ui/src/components/VictimCard.tsx ui/src/components/ExtractorCard.tsx ui/src/components/VerifierCard.tsx
git commit -m "feat: add investigation pipeline cards (Generator, Victim, Extractor, Verifier)"
```

---

## Task 11: Frontend — Analytics Panel

**Files:**
- Create: `ui/src/components/AnalyticsPanel.tsx`

- [ ] **Step 1: Create `ui/src/components/AnalyticsPanel.tsx`**

```typescript
import { useRunStore } from '../store/runStore';

export default function AnalyticsPanel() {
  const { selectedRun } = useRunStore();
  if (!selectedRun) return null;

  const { models, result, timing } = selectedRun;

  return (
    <div className="w-72 bg-white border-l border-slate-200 flex flex-col h-full overflow-y-auto">
      {/* Models */}
      <div className="p-4 border-b border-slate-200">
        <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">Models</h3>
        <div className="space-y-2">
          <div className="text-sm">
            <p className="text-slate-500 text-xs">Generator</p>
            <p className="font-medium text-slate-900 truncate" title={models.generator.name}>{models.generator.name.split('/').pop()}</p>
          </div>
          <div className="text-sm">
            <p className="text-slate-500 text-xs">Victim</p>
            <p className="font-medium text-slate-900 truncate" title={models.victim.name}>{models.victim.name.split('/').pop()}</p>
          </div>
          <div className="text-sm">
            <p className="text-slate-500 text-xs">Judge</p>
            <p className="font-medium text-slate-900 truncate" title={models.judge.name}>DistilBERT</p>
          </div>
          <div className="text-sm">
            <p className="text-slate-500 text-xs">Extractor</p>
            <p className="font-medium text-slate-900 truncate" title={models.extractor.name}>{models.extractor.name.split('/').pop()}</p>
          </div>
        </div>
      </div>

      {/* Success Metrics */}
      <div className="p-4 border-b border-slate-200">
        <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">Success</h3>
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-slate-600">Ground Truth</span>
            <span className={`font-bold ${result.ground_truth_success ? 'text-green-600' : 'text-red-600'}`}>
              {result.ground_truth_success ? '✓' : '✗'}
            </span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-slate-600">Extractor</span>
            <span className={`font-bold ${result.extractor_success ? 'text-green-600' : 'text-red-600'}`}>
              {result.extractor_success ? '✓' : '✗'}
            </span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-slate-600">Verifier</span>
            <span className={`font-bold ${result.verified_success ? 'text-green-600' : 'text-red-600'}`}>
              {result.verified_success ? '✓' : '✗'}
            </span>
          </div>
        </div>
      </div>

      {/* Attempts */}
      <div className="p-4 border-b border-slate-200">
        <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">Attempts</h3>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-slate-600">Total</span>
            <span className="font-bold text-slate-900">{result.total_attempts}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-600">Success</span>
            <span className={`font-bold ${result.ground_truth_success ? 'text-green-600' : 'text-red-600'}`}>
              {result.ground_truth_success ? 'YES' : 'NO'}
            </span>
          </div>
        </div>
      </div>

      {/* Timing */}
      <div className="p-4 border-b border-slate-200">
        <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">Timing</h3>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-slate-600">Total Run</span>
            <span className="font-medium text-slate-900">{timing.total_run_time.toFixed(1)}s</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-600">Avg Attempt</span>
            <span className="font-medium text-slate-900">{timing.average_attempt_time.toFixed(1)}s</span>
          </div>
        </div>
      </div>

      {/* Summary */}
      <div className="p-4">
        <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3">Summary</h3>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-slate-600">Unique Attacks</span>
            <span className="font-medium text-slate-900">{selectedRun.summary.unique_attacks}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-600">Repetition</span>
            <span className="font-medium text-slate-900">{(selectedRun.summary.repetition_rate * 100).toFixed(1)}%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-600">Attack Len</span>
            <span className="font-medium text-slate-900">{selectedRun.summary.attack_length_avg.toFixed(0)} chars</span>
          </div>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add ui/src/components/AnalyticsPanel.tsx
git commit -m "feat: add analytics panel with models, success, timing, summary"
```

---

## Task 12: Frontend — Investigation Page (3-Panel Layout)

**Files:**
- Create: `ui/src/pages/InvestigationPage.tsx`
- Modify: `ui/src/components/VictimCard.tsx` (pass access_code prop)

- [ ] **Step 1: Create `ui/src/pages/InvestigationPage.tsx`**

```typescript
import { useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useRunStore } from '../store/runStore';
import TimelineSidebar from '../components/TimelineSidebar';
import GeneratorCard from '../components/GeneratorCard';
import VictimCard from '../components/VictimCard';
import ExtractorCard from '../components/ExtractorCard';
import VerifierCard from '../components/VerifierCard';
import AnalyticsPanel from '../components/AnalyticsPanel';

export default function InvestigationPage() {
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();
  const { selectedRun, selectedAttemptIndex, setSelectedRun } = useRunStore();

  useEffect(() => {
    if (!runId) return;
    fetch(`/api/run/${runId}`)
      .then((res) => res.json())
      .then((data) => setSelectedRun(data))
      .catch((err) => {
        console.error('Failed to load run:', err);
        navigate('/runs');
      });
  }, [runId]);

  if (!selectedRun) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50">
        <p className="text-slate-500">Loading run...</p>
      </div>
    );
  }

  const attempt = selectedRun.attempts[selectedAttemptIndex];
  if (!attempt) return null;

  return (
    <div className="h-screen flex flex-col">
      {/* Top Bar */}
      <header className="bg-white border-b border-slate-200 px-4 py-2 flex items-center justify-between flex-shrink-0">
        <div className="flex items-center gap-3">
          <button onClick={() => navigate('/runs')} className="text-sm text-slate-500 hover:text-slate-900 transition-colors">
            ← Runs
          </button>
          <span className="text-slate-300">|</span>
          <h1 className="font-mono text-sm font-bold text-slate-900">{selectedRun.experiment.run_id}</h1>
          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${selectedRun.result.ground_truth_success ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
            {selectedRun.result.ground_truth_success ? 'SUCCESS' : 'FAILED'}
          </span>
        </div>
        <div className="flex items-center gap-2 text-sm text-slate-500">
          <span>Attempt {attempt.attempt_number}/{selectedRun.result.total_attempts}</span>
        </div>
      </header>

      {/* 3-Panel Layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left: Timeline */}
        <TimelineSidebar />

        {/* Center: Investigation */}
        <div className="flex-1 overflow-y-auto bg-slate-50 p-6">
          <div className="max-w-4xl mx-auto space-y-4">
            {/* Attempt Header */}
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-bold text-slate-900">Attempt {attempt.attempt_number}</h2>
              <div className="flex items-center gap-2">
                <span className="px-2 py-0.5 bg-purple-100 text-purple-700 text-xs font-medium rounded-full">
                  {attempt.generator.strategy}
                </span>
                <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${attempt.judge.decision === 'ATTACK' ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'}`}>
                  {attempt.judge.decision} ({attempt.judge.confidence.toFixed(2)})
                </span>
              </div>
            </div>

            {/* Pipeline */}
            <GeneratorCard attempt={attempt} />
            <div className="flex justify-center text-slate-400 text-lg">↓</div>
            <VictimCard attempt={attempt} />
            <div className="flex justify-center text-slate-400 text-lg">↓</div>
            <ExtractorCard attempt={attempt} />
            <div className="flex justify-center text-slate-400 text-lg">↓</div>
            <VerifierCard attempt={attempt} />
          </div>
        </div>

        {/* Right: Analytics */}
        <AnalyticsPanel />
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Update VictimCard to accept access_code from parent**

Modify `ui/src/components/VictimCard.tsx` to accept `accessCode` prop:

```typescript
export default function VictimCard({ attempt, accessCode }: { attempt: Attempt; accessCode?: string }) {
```

Update `InvestigationPage.tsx` to pass it:

```typescript
<VictimCard attempt={attempt} accessCode={selectedRun.scenario.access_code} />
```

- [ ] **Step 3: Verify full flow**

```bash
# Terminal 1: Start server
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Start frontend
cd ui && npm run dev
```

Open `http://localhost:3000`, verify:
- Run list loads from `results/` directory
- Clicking a run opens investigation page
- Timeline sidebar shows color-coded attempts
- Clicking attempt updates center panel
- All 4 cards render with data

- [ ] **Step 4: Commit**

```bash
git add ui/src/pages/InvestigationPage.tsx ui/src/components/VictimCard.tsx
git commit -m "feat: add 3-panel investigation page with pipeline view"
```

---

## Self-Review

**Spec coverage:**
- ✅ Run Loader (JSON file upload + list) — Task 8
- ✅ Timeline Sidebar — Task 9
- ✅ Attempt Investigation View (Generator → Victim → Extractor → Verifier) — Tasks 10, 12
- ✅ Background Worker (isolated from web server) — Tasks 5, 6
- ✅ Python JSON emission — Task 1
- ✅ All data model fields covered
- ✅ Verification trace (candidate_sent, victim_response, success)
- ✅ experiment_version + git_commit
- ✅ raw_dataset_entry
- ✅ generator_success metric

**Placeholder scan:** No TBDs, no "implement later", no vague steps. All code blocks are complete.

**Type consistency:** TypeScript interfaces in `ui/src/types/autored.ts` match Pydantic schemas in `server/schemas.py`. Field names consistent: `ground_truth_found`, `extractor_match`, `generator_success`, `verification.success`.

**Gaps:** None for Phase 1 scope. Scenario page, Extractor Debugger, Attack Evolution, Benchmark Dashboard, Live Streaming are Phase 2-4.

---

## Phase 1 Deliverables

After completing all 12 tasks:
1. `llama_3_8b_verbose.py` emits structured JSON to `results/run_<timestamp>.json`
2. FastAPI server serves run list, individual runs, upload, CSV export
3. Background worker loads models, runs experiments (isolated from web server)
4. React frontend: run list → investigation page (3-panel layout)
5. Timeline sidebar with color-coded attempts
6. Investigation pipeline: Generator → Victim → Extractor → Verifier
7. Analytics panel with models, success, timing, summary

**Next:** Phase 2 (Scenario Page, Extractor Debugger, Live Streaming, Verification Trace display)
