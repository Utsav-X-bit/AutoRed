import os
# MUST set BEFORE any experiment imports to prevent double model loading
os.environ["AUTORED_SERVER_MODE"] = "1"

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Any
import json
import tempfile
import asyncio

from .file_manager import list_runs, get_run, upload_run, delete_run
from .websocket import ws_manager
from .models_server import server_models


# ─── Startup / Shutdown ──────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, keep in memory across runs."""
    print("[SERVER] Starting up — loading models...")
    load_times = server_models.load_all()
    total = sum(load_times.values())
    print(f"[SERVER] ✓ All models loaded in {total:.1f}s")
    print(f"[SERVER]   victim:    {load_times.get('victim', 0):.1f}s")
    print(f"[SERVER]   generator: {load_times.get('generator', 0):.1f}s")
    print(f"[SERVER]   judge:     {load_times.get('judge', 0):.1f}s")
    yield
    print("[SERVER] Shutting down — clearing models...")
    server_models.models.clear()
    server_models.tokenizers.clear()


app = FastAPI(title="AutoRed Web UI", version="1.0.0", lifespan=lifespan)

# CORS for Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track running state
_run_lock = asyncio.Lock()
_is_running = False


@app.get("/api/run/status")
def api_run_status():
    """Check if a run is currently in progress."""
    return {"running": _is_running}


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
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    result = upload_run(tmp_path)
    os.unlink(tmp_path)
    return result


@app.delete("/api/run/{run_id}")
def api_delete_run(run_id: str):
    """Delete a run."""
    if not delete_run(run_id):
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return {"deleted": run_id}


# ─── Model Status Endpoints ─────────────────────────────────

@app.get("/api/models/status")
def api_model_status():
    """Get model load status from server."""
    return server_models.get_status()


# ─── Experiment Endpoints ───────────────────────────────────

@app.post("/api/run")
async def api_start_run(scenario_id: str = "", max_attempts: int = 20):
    """Start a new experiment run with pre-loaded models and live WebSocket streaming.
    
    Returns immediately with run_id; experiment runs in background.
    Client should connect WebSocket to /ws/run/{run_id} before calling this endpoint.
    """
    if not server_models.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Server may be starting up.")

    # Generate run_id client-side or server-side
    run_id = f"run_{__import__('time').time():.0f}"

    # Start experiment in background
    asyncio.create_task(
        _run_experiment_task(run_id, scenario_id or None, max_attempts)
    )

    return {
        "run_id": run_id,
        "status": "started",
        "message": "Experiment started. Connect WebSocket to /ws/run/{run_id} for live updates.",
    }


async def _run_experiment_task(run_id: str, scenario_id: str, max_attempts: int):
    """Background task: run experiment, stream via WebSocket."""
    global _is_running
    async with _run_lock:
        _is_running = True
        try:
            from .experiment_server import run_experiment_server
            await run_experiment_server(
                run_id=run_id,
                scenario_id=scenario_id,
                max_attempts=max_attempts,
            )
        except Exception as e:
            print(f"[SERVER] Experiment {run_id} failed: {e}")
            import traceback
            traceback.print_exc()
            await ws_manager.send_run_complete(run_id, {"error": str(e), "experiment": {"run_id": run_id}})
        finally:
            _is_running = False


# ─── Export Endpoints ───────────────────────────────────────

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
    if not run.get("attempts"):
        output.write("no_attempts\n")
    else:
        fieldnames = [
            "attempt", "strategy", "attack", "judge_decision",
            "ground_truth_found", "extractor_match", "best_candidate",
            "verification_success"
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for a in run["attempts"]:
            row = {
                "attempt": a.get("attempt_number"),
                "strategy": a.get("generator", {}).get("strategy"),
                "attack": a.get("generator", {}).get("generated_attack")[:100],
                "judge_decision": a.get("judge", {}).get("decision"),
                "ground_truth_found": a.get("ground_truth_found"),
                "extractor_match": a.get("extractor_match"),
                "best_candidate": a.get("extractor", {}).get("best_candidate"),
                "verification_success": a.get("verification", {}).get("success"),
            }
            writer.writerow(row)

    from fastapi.responses import Response
    return Response(content=output.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={run_id}.csv"})


@app.get("/api/export/{run_id}/html")
def api_export_html(run_id: str):
    """Export run as styled HTML report."""
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AutoRed Report: {run_id}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #f8fafc; color: #0f172a; }}
h1 {{ font-size: 1.5rem; }} h2 {{ font-size: 1.2rem; margin-top: 1.5rem; }}
table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
th, td {{ border: 1px solid #e2e8f0; padding: 0.5rem; text-align: left; font-size: 0.875rem; }}
th {{ background: #f1f5f9; }}
.badge {{ display: inline-block; padding: 0.125rem 0.5rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; }}
.badge-green {{ background: #dcfce7; color: #166534; }}
.badge-red {{ background: #fee2e2; color: #991b1b; }}
.card {{ background: white; border: 1px solid #e2e8f0; border-radius: 0.5rem; padding: 1rem; margin: 0.5rem 0; }}
pre {{ background: #f1f5f9; padding: 0.75rem; border-radius: 0.375rem; overflow-x: auto; font-size: 0.8rem; }}
</style></head><body>
<h1>AutoRed Report: {run_id}</h1>
<p>Timestamp: {run.get('experiment', {}).get('timestamp', 'N/A')}</p>
<p>Version: {run.get('experiment', {}).get('experiment_version', 'N/A')}</p>
<p>Git: {run.get('experiment', {}).get('git_commit', 'N/A')}</p>

<h2>Result</h2>
<div class="card">
<p>Ground Truth: <span class="badge {'badge-green' if run.get('result', {}).get('ground_truth_success') else 'badge-red'}">{'✓' if run.get('result', {}).get('ground_truth_success') else '✗'}</span></p>
<p>Extractor: <span class="badge {'badge-green' if run.get('result', {}).get('extractor_success') else 'badge-red'}">{'✓' if run.get('result', {}).get('extractor_success') else '✗'}</span></p>
<p>Attempts: {run.get('result', {}).get('total_attempts', 0)}</p>
</div>

<h2>Attempts</h2>
<table>
<tr><th>#</th><th>Strategy</th><th>Judge</th><th>GT Found</th><th>Extractor</th><th>Best Candidate</th></tr>
"""
    for a in run.get("attempts", []):
        gt = '<span class="badge badge-green">✓</span>' if a.get("ground_truth_found") else '<span class="badge badge-red">✗</span>'
        ext = '<span class="badge badge-green">✓</span>' if a.get("extractor_match") else '<span class="badge badge-red">✗</span>'
        html += f'<tr><td>{a.get("attempt_number")}</td><td>{a.get("generator", {}).get("strategy")}</td><td>{a.get("judge", {}).get("decision")}</td><td>{gt}</td><td>{ext}</td><td>{a.get("extractor", {}).get("best_candidate", "N/A")}</td></tr>\n'

    html += """</table></body></html>"""

    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html)


# ─── WebSocket Endpoint ─────────────────────────────────────

@app.websocket("/ws/run/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: str):
    await ws_manager.connect(run_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(run_id, websocket)
