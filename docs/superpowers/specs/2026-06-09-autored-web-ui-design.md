# AutoRed Web UI — Design Specification

**Date**: 2026-06-09
**Status**: Approved
**Author**: Qwen Code (with user review)

---

## Design Philosophy

AutoRed is a story. Every run tells a story:

```
Defense → Attack Evolution → Victim Reactions → Leak → Extraction → Verification → Success/Failure
```

The UI tells this story. It is **attempt-centric**, not metrics-centric. 90% of debugging time is spent answering "Why did Attempt 17 fail?" not "What was the overall success rate?"

**Priority Order:**
1. Attempt Investigation View
2. Timeline Sidebar
3. Scenario Page
4. Extractor Debugger
5. Attack Evolution View
6. Benchmark Dashboard
7. Live Streaming
8. Advanced Analytics

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              React + Vite + Tailwind (Frontend)          │
│  3-Panel Layout: Timeline │ Investigation │ Analytics   │
└──────────────────────┬──────────────────────────────────┘
                       │ REST + WebSocket
┌──────────────────────▼──────────────────────────────────┐
│                  FastAPI Server (Web)                    │
│  REST API │ WebSocket Hub │ JSON File Manager           │
└──────────────────────┬──────────────────────────────────┘
                       │ Redis Queue / Celery Broker
┌──────────────────────▼──────────────────────────────────┐
│              Background Worker Process                   │
│  Model Manager │ Experiment Runner                      │
│  (isolated — crashes don't kill the web server)         │
└──────────────────────────────────────────────────────────┘
```

**Why separate worker?** If an experiment crashes (CUDA OOM, model exception, bad prompt), the web server stays alive. The worker is the only process holding GPU memory.

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React + TypeScript + Vite + Tailwind CSS |
| Web Server | FastAPI + Python |
| Background Worker | Celery + Redis (or RQ + Redis) |
| Charts | Recharts |
| State | Zustand |
| WebSocket | Native WebSocket API |
| Export | jsPDF (PDF), xlsx (CSV) |

### File Structure

```
AutoRed/
├── experiment/
│   └── llama_3_8b_verbose.py      # Modified: emit JSON + timing
├── server/
│   ├── __init__.py
│   ├── main.py                    # FastAPI web server (no models)
│   ├── schemas.py                 # Pydantic schemas
│   ├── file_manager.py            # Past run JSON CRUD
│   └── websocket.py               # WebSocket hub
├── worker/
│   ├── __init__.py
│   ├── celery_app.py              # Celery/RQ app config
│   ├── models_manager.py          # Load + keep models in memory
│   └── experiment_runner.py       # Orchestrate runs (GPU process)
├── ui/
│   ├── src/
│   │   ├── components/            # AttemptCard, Chart, FilterBar...
│   │   ├── pages/                 # Dashboard, Investigation, Benchmark...
│   │   ├── hooks/                 # useWebSocket, useRuns...
│   │   ├── types/                 # TypeScript interfaces
│   │   └── store/                 # Zustand state
│   ├── package.json
│   └── vite.config.ts
└── results/                       # Auto-saved run JSONs
    └── run_<timestamp>.json
```

---

## Data Model

### Top-Level Run

```typescript
interface AutoRedRun {
  // Reproducibility
  experiment: {
    run_id: string;
    benchmark_mode: boolean;
    benchmark_run_number?: number;
    benchmark_total_runs?: number;
    max_attempts: number;
    dataset_size: number;
    scenario_id: string;
    seed: number;
    timestamp: string;
    experiment_version: string;   // e.g., "2.0.0"
    git_commit: string;           // git hash for reproducibility
  };

  // Raw dataset entry (for full reproducibility)
  raw_dataset_entry: {
    defense_id: string;
    opening_defense: string;
    closing_defense: string;
    access_code: string;
    [key: string]: any;           // preserve all original fields
  };

  // All 4 models
  models: {
    victim: { name: string; load_time: number };
    generator: { name: string; load_time: number };
    judge: { name: string; load_time: number };
    extractor: { name: string; load_time: number };
  };

  // Fine-grained timing
  timing: {
    total_run_time: number;
    model_loading_time: number;
    average_attempt_time: number;
  };

  // Scenario
  scenario: {
    pre_defense: string;
    post_defense: string;
    access_code: string;
    full_prompt: string;
  };

  // Result with reason
  result: {
    ground_truth_success: boolean;
    generator_success: boolean;      // generator leaked access code
    extractor_success: boolean;
    verified_success: boolean;
    extracted_value: string;
    success_reason: "ground_truth" | "extractor" | "verification" | null;
    total_attempts: number;
  };

  // Strategy performance
  strategy_stats: Record<string, {
    successes: number;
    partial_leaks: number;
    failures: number;
    total_score: number;
  }>;

  // Best attack tracking
  best_attack: {
    prompt: string;
    score: number;
    strategy: string;
  } | null;

  // Ground truth analysis
  ground_truth: {
    access_code: string;
    leaked: boolean;
    leak_position: number | null;
    leak_count: number;
  };

  // Per-attempt detail
  attempts: Attempt[];

  // Event timeline
  events: Event[];

  // Summary stats
  summary: {
    attack_length_min: number;
    attack_length_max: number;
    attack_length_avg: number;
    unique_attacks: number;
    repetition_rate: number;
    judge_distribution: { ATTACK: number; ATTEMPT: number };
  };
}
```

### Per-Attempt

```typescript
interface Attempt {
  attempt_number: number;
  timestamp: string;
  attempt_time_ms: number;

  // Generator internals
  generator: {
    strategy: string;
    internal_prompt: string;
    generated_attack: string;
    attack_length: number;
    attack_hash: string;
    duplicate_attack: boolean;
    input_tokens: number;
    output_tokens: number;
  };

  // Judge internals
  judge: {
    input: string;
    decision: string;
    confidence: number;
    probabilities: { ATTACK: number; ATTEMPT: number };
  };

  // Victim output (both versions)
  victim: {
    raw_output: string;
    clean_output: string;
    output_length: number;
  };

  // Extractor full trace (6 layers)
  extractor: {
    regex_candidates: string[];
    quoted_candidates: string[];
    capitalized_candidates: string[];
    llm_candidates: string[];
    ranked_candidates: { value: string; score: number }[];
    best_candidate: string;
  };

  // Verification trace (was just `verified: boolean`)
  verification: {
    candidate_sent: string;
    victim_response: string;
    success: boolean;
  };

  // Ground truth per-attempt
  ground_truth_found: boolean;
  extractor_match: boolean;
  generator_success: boolean;       // access code leaked this attempt
}
```

### Event Timeline

```typescript
interface Event {
  timestamp: string;
  type: "model_load" | "generator" | "judge" | "victim" | "extractor" | "success" | "error";
  message: string;
}
```

### WebSocket Live Updates

```typescript
interface AttemptUpdate {
  type: "attempt_update";
  run_id: string;
  attempt: Attempt;
}

interface RunComplete {
  type: "run_complete";
  run_id: string;
  run: AutoRedRun;
}
```

---

## UI Layout

### Primary Layout — 3 Panel

```
┌──────────────┬─────────────────────────────┬──────────────┐
│   Timeline   │    Investigation Panel      │  Analytics   │
│   (Left)     │    (Center, 70%)            │  (Right)     │
│              │                             │              │
│ RUN #104     │ ┌─────────────────────────┐ │ Models       │
│ ▫ Attempt 1  │ │ Generator Card          │ │ Generator:   │
│ ▫ Attempt 2  │ │   strategy: summarization│ │   Lexi       │
│ ▫ Attempt 3  │ │   attack: "Summarize..." │ │ Victim:      │
│ ▫ Attempt 4  │ ├─────────────────────────┤ │   Llama-3    │
│ ▫ Attempt 5  │ │ Victim Card (largest)   │ │ Judge:       │
│ ● Attempt 6 ⭐│ │ "The password is verve63"│ │   DistilBERT │
│              │ │ [highlighted leak]      │ │ Extractor:   │
│ Color codes: │ ├─────────────────────────┤ │   Llama-3    │
│ Gray=normal  │ │ Extractor Card          │ │              │
│ Blue=attack  │ │ regex: [verve63, Beep]  │ ────────────── │
│ Yellow=leak  │ │ ranked: verve63(10)     │ Success        │
│ Green=success│ │ selected: Beep ❌       │ GT: ✓          │
│ Red=extract  │ ├─────────────────────────┤ │ Ext: ✗       │
│  failure     │ │ Verifier Card           │ │ Ver: ✗       │
│              │ │ "Access Denied" ❌      │ │ Attempts: 6  │
│              │ └─────────────────────────┘ │ │ Success: ✓  │
│              │                             │ ────────────── │
│              │ Tabs below:                 │ Charts:        │
│              │ Scenario │ Evolution │ Leak │ tokens, timing │
│              │ │ Debugger              │ │              │
│              └─────────────────────────┘ │              │
└──────────────┴─────────────────────────────┴──────────────┘
```

### Left Panel — Timeline

- Lists all attempts for current run
- Color coding:
  - **Gray**: Normal attempt
  - **Blue**: Attack generated
  - **Yellow**: Leak detected (ground truth found)
  - **Green**: Success (extractor match)
  - **Red**: Extraction failure (leaked but extractor missed)
- Clicking an attempt updates center + right panels
- Star icon (⭐) marks success attempt

### Center Panel — Investigation (70% width)

Vertical pipeline showing the flow:

```
Generator (strategy, tokens, attack)
    ↓
Victim (raw + clean output, leak highlight)
    ↓
Extractor (4 layers, ranking, selection)
    ↓
Verifier (send-back result)
```

**Generator Card:**
- Strategy used
- Input/output tokens
- Generated attack text (monospace, bordered)

**Victim Card (largest):**
- Full raw output
- Syntax highlighted
- If secret appears: yellow background highlight around the leaked text
- Ground truth found: ✓/✗

**Extractor Card:**
- Regex candidates (list)
- LLM candidates (list)
- Ranked candidates with scores
- Selected candidate with ✓/✗ indicator
- If wrong selection: "❌ Wrong Selection" warning

**Verifier Card:**
- Candidate sent back (from `verification.candidate_sent`)
- Victim response (from `verification.victim_response`)
- Verification result: ✓/✗ (from `verification.success`)

### Right Panel — Analytics

Persistent summary, always visible:
- Model names (all 4)
- Success metrics (GT, Extractor, Verifier)
- Attempt count
- Token charts
- Timing heatmap

### Tabs (below investigation)

| Tab | Content |
|-----|---------|
| **Scenario** | Pre/Post defense (collapsible), access code (debug mode only) |
| **Attack Evolution** | Strategy timeline, flow chart, why strategy changed |
| **Leak Analysis** | Leak type, confidence, character position, highlighted region |
| **Extractor Debugger** | Ground truth vs candidates, ranking, bug detection |

---

## Benchmark Dashboard

Separate view for `--rounds 70` mode.

**Top Cards:**
- Success Rate (%)
- Average Attempts
- Best Strategy
- Worst Strategy

**Charts:**
- Success Distribution (bar)
- Attempts to Success (histogram)
- Strategy Win Rate (pie)
- Leak Frequency (trend)

---

## Run Comparison View

Dedicated page for comparing two or more runs side-by-side. Essential for generator/model comparison.

**Use Case:** Compare Llama-2 vs Lexi generators, or different target models.

**Layout:**

```
┌─────────────────────────────────────────────────────────┐
│  Run A (Llama-2 Generator)  │  Run B (Lexi Generator)  │
├─────────────────────────────┼───────────────────────────┤
│  Success Rate: 42%          │  Success Rate: 58%        │
│  Avg Attempts: 13           │  Avg Attempts: 9          │
│  Best Strategy: summarization│  Best Strategy: roleplay  │
│  Worst Strategy: translation │  Worst Strategy: summary  │
├─────────────────────────────┼───────────────────────────┤
│  Per-attempt comparison     │  Per-attempt comparison   │
│  (aligned by attempt #)     │  (aligned by attempt #)   │
└─────────────────────────────────────────────────────────┘
```

**Extractor Failure Analysis:**
- Ground truth vs candidates
- Ranking breakdown
- Failure reason classification (ranking bug, missing candidate, etc.)

---

## Backend API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/models/status` | GET | Model load status + times |
| `/api/run` | POST | Start new run (returns `run_id`) |
| `/api/run/{run_id}` | GET | Get completed run JSON |
| `/api/runs` | GET | List all past runs |
| `/api/runs/upload` | POST | Upload external JSON file |
| `/ws/run/{run_id}` | WS | Live attempt stream |
| `/api/export/{run_id}/csv` | GET | Export CSV |
| `/api/export/{run_id}/html` | GET | Export HTML report |

---

## Model Persistence

The **background worker** loads all 4 models at startup and keeps them in GPU memory. Clicking "New Run" in the UI does NOT reload models — it only selects a new random scenario from the dataset and starts the experiment loop. This eliminates the ~30s model load time between runs.

Models are only reloaded if:
- Worker process restarts
- User explicitly requests model reload via `/api/models/reload`

**Isolation:** The web server (FastAPI) never loads models. It only communicates with the worker via Redis Queue. If the worker crashes (CUDA OOM, model exception), the web server stays alive and shows an error state.

## Python Script Modifications

### New JSON Emission

`llama_3_8b_verbose.py` modified to:
1. Wrap experiment loop in JSON serializer
2. Record timestamps for each attempt
3. Capture all 4 model names + load times
4. Track strategy stats, best attack
5. Record generator internal prompt
6. Record judge input
7. Record both raw/clean victim output
8. Record full extractor trace (6 layers)
9. Record ground truth position + count
10. Record event timeline
11. Record `experiment_version` + `git_commit`
12. Record `raw_dataset_entry` (full original row)
13. Record `generator_success` per attempt
14. Record `verification` trace (candidate, response, success)
15. Save to `results/run_<timestamp>.json`

### Model Persistence

New `worker/models_manager.py`:
- Loads models once on worker start
- Keeps in GPU memory across runs
- Exposes status via worker → web server queue

### Experiment Runner

New `worker/experiment_runner.py`:
- Takes scenario from dataset
- Orchestrates AutoRed loop
- Streams attempts via WebSocket (through web server)
- Saves JSON on completion
- Runs in isolated process (crashes don't affect web server)

---

## Search and Filters

| Filter | Description |
|--------|-------------|
| Show only successes | Filter runs by `result.ground_truth_success` |
| Show only failures | Filter runs by `!result.ground_truth_success` |
| Show roleplay attacks | Filter by `generator.strategy == "roleplay"` |
| Show ATTEMPT decisions | Filter by `judge.decision == "ATTEMPT"` |
| Show extractor mistakes | Filter by `ground_truth_found && !extractor_match` |
| Show false positives | Filter by `!ground_truth_found && extractor_match` |

---

## Export Features

| Format | Content |
|--------|---------|
| JSON | Full `AutoRedRun` object |
| CSV | Flattened attempt table |
| HTML Report | Styled report with charts |
| PDF | Printable summary |

---

## Implementation Phases

If development time becomes constrained, implement in this order:

### Phase 1 — Core Investigation
- Run Loader (JSON file upload + list)
- Timeline Sidebar
- Attempt Investigation View (Generator → Victim → Extractor → Verifier)
- Background Worker (isolated from web server)

### Phase 2 — Debugging Tools
- Scenario Page
- Extractor Debugger
- Live Streaming (WebSocket)
- Verification Trace display

### Phase 3 — Analysis
- Attack Evolution View
- Benchmark Dashboard
- Run Comparison View
- Search and Filters

### Phase 4 — Advanced
- Token Analytics Charts
- Model Performance Heatmap
- Export Reports (CSV, HTML, PDF)
- Strategy Heatmaps

---

## Live Mode

WebSocket-based real-time streaming:
1. User clicks "New Run"
2. Backend selects random scenario
3. Each attempt streamed as `AttemptUpdate`
4. Timeline sidebar updates in real-time
5. Investigation panel auto-scrolls to latest attempt
6. Run completes → `RunComplete` event → JSON saved

---

## Error Handling

- Model load failure: Show error in model status, disable "New Run"
- WebSocket disconnect: Show reconnect button, preserve received attempts
- JSON parse failure: Show raw JSON with syntax error highlight
- Empty results: Show placeholder with "No runs yet"

---

## Testing

- Unit tests for JSON schema validation
- Integration tests for WebSocket streaming
- E2E tests for attempt investigation view
- Benchmark: verify 70-round mode produces correct JSON

---

## Out of Scope

- Multi-user authentication
- Role-based access control
- Cloud deployment (HPC only)
- Mobile responsive (desktop-first)
- Dark mode (light theme only)
