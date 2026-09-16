# Cognitive Load Predictor System

This project is a working cognitive load assessment system, not just a UI mockup. It combines behavioral micro-tasks, feature extraction, weighted range validation, persistent logging, participant-specific baseline calibration, and an optional trainable classifier.

Version 2 adds an intelligence layer on top: temporal modeling across sessions, uncertainty estimation, grounded explanations, anomaly detection, forecasting, recommendations, an upgraded model comparison harness, difficulty normalization, and an HTTP API.

## What it does

The system runs four behavioral tasks:

- Reaction time: processing speed and alertness
- Task switching: mental flexibility and switch cost
- Working memory: short-term recall under load
- Selective attention: Stroop-style inhibitory control

Each assessment session stores raw trial-level data and produces:

- Task metrics such as latency, error rate, variability, switch cost, and interference cost
- A rule-based 0-100 score
- Weighted per-category validation scores based on which score ranges the overall result and task results fall into
- A cognitive load band: `Low`, `Manageable`, `Elevated`, or `High`
- A performance band: `Excellent`, `Good`, `Medium`, or `Risky`
- Optional model-based override only if you explicitly enable it

### V2 intelligence layer

Every session also gets (all computed from prior sessions only — no leakage):

- **History features**: change vs last session, rolling mean/std, EMA, and a trend direction (needs 3+ sessions)
- **Difficulty normalization**: how hard the task configuration was, with adjusted metrics (e.g., error rate normalized for the memory spans presented)
- **Data quality + anomalies**: robust z-score checks against personal history, internal-consistency checks (fast-but-inaccurate, impossible timing, repetitive responses), a 0-100 quality score, and a Normal/Review/Anomalous status. Flags, never deletes.
- **Uncertainty**: a confidence level with drivers, so a 0.36 and a 0.92 probability are never presented the same way. Never "High" confidence with fewer than 3 prior sessions.
- **Explanations**: an exact additive decomposition of the score into per-feature contributions (mathematically grounded, nothing fabricated), in plain English
- **Forecast**: a projection of the next session's band (needs 5+ sessions; otherwise an explicit INSUFFICIENT_HISTORY response)
- **Recommendations**: up to 4 deterministic, rule-based suggestions, each stating the exact basis that triggered it. Suggestions, not advice; no causal claims.

The interactive task runner now uses a fixed 1-minute response window for each prompt:

- Reaction time: up to 60 seconds after the green cue
- Task switching: up to 60 seconds per trial
- Working memory recall: up to 60 seconds per answer, with slower digit presentation
- Selective attention: up to 60 seconds per trial

The actual response duration is stored per trial in seconds and milliseconds, shown in the on-screen feedback, and included in the task scoring and final metrics.

## Why this is a real system

Without labeled data, no cognitive load system can honestly claim clinical-grade prediction. This project handles that correctly:

- It works immediately as a behavioral assessment and screening system
- It logs structured data for later analysis
- It supports participant-specific baseline calibration
- It can later train a softmax classifier from labeled sessions without changing the assessment code

That makes it a usable system for a project, pilot study, or internal research prototype.

## Project layout

```text
.
|-- run_assessment.py
|-- train_predictor.py
|-- recalibrate.py
|-- serve_api.py
|-- web/
|   |-- index.html          # UI shell (Geist-inspired, zero build step)
|   `-- static/
|       |-- styles.css      # design tokens, components, motion
|       `-- app.js          # hash router, views, API client
|-- scripts/
|   |-- smoke_test.py
|   |-- backfill_summary.py
|   `-- make_pitch_pdf.py
|-- src/cognitive_load/
|   |-- app.py
|   |-- analysis.py
|   |-- calibration.py
|   |-- config.py
|   |-- model.py
|   |-- simulation.py
|   |-- storage.py
|   |-- temporal.py      # V2 FR-1
|   |-- difficulty.py    # V2 FR-10 light
|   |-- anomaly.py       # V2 FR-8 + quality
|   |-- uncertainty.py   # V2 FR-3
|   |-- explain.py       # V2 FR-4
|   |-- forecast.py      # V2 FR-7
|   |-- recommendations.py # V2 FR-6
|   |-- ml.py            # V2 FR-2
|   |-- pipeline.py      # V2 orchestrator
|   `-- api/app.py       # V2 FR-9
|-- tests/
`-- data/
    |-- baselines/
    |-- models/
    |-- reports/
    `-- sessions/
```

## How to run

**One command does everything** — server, dashboard and the cognitive test (Windows):

```powershell
.\start.bat
```

It boots the API + UI, opens your browser on the test, and stays in the foreground (Ctrl+C or closing the window stops it). Take the four tasks (~8 min), and the session is scored and saved: score, band, confidence, exact score breakdown, data-quality flags and suggestions — then it is on your dashboard.

Prefer the command line? The same thing by hand:

```powershell
python serve_api.py --port 8765      # then open http://127.0.0.1:8765/#/test
```

The desktop battery (tkinter) still works and writes the identical trial schema:

```powershell
python run_assessment.py --participant-id P001
```

That opens the interactive task runner, saves the raw session, appends a summary row to CSV (including the V2 columns), and writes a text report with both the classic and V2 intelligence sections.

## Useful commands

Run a synthetic session without the GUI:

```powershell
python run_assessment.py --simulate manageable --participant-id TEST01
```

Train a classifier from labeled summary rows:

```powershell
python train_predictor.py --label-column self_report_load_band
```

Compare candidate models on a leakage-free chronological holdout and persist the winner (FR-2):

```powershell
python train_predictor.py --compare
```

Serve the HTTP API **and the web UI**, then open it:

```powershell
python serve_api.py
# open http://127.0.0.1:8000/ in a browser
```

The UI has four views: **Take test** (the real four-task cognitive battery in the browser — reaction time, task switching, working memory, selective attention; see below), **Overview** (participant picker, KPIs, forecast, recent sessions), **Session detail** (the full V2 report — uncertainty, exact explanation factors, data quality, recommendations with their stated bases), and **Simulate** (generate a synthetic session and analyze it live). It follows the Vercel/Geist design language: dark aurora-glass canvas, status as colored dots, one interactive accent, compositor-only motion with full `prefers-reduced-motion` support.

### The in-browser cognitive battery

`web/static/test.js` is a faithful port of the desktop battery: same four tasks, same trial counts (8 / 26 / 10 / 28), same pacing (from `TASK_TIMING`, mirrored in `web/static/config.js`) and the same trial schema — so a session taken in the browser scores identically to one taken in the desktop app. Timing uses `performance.now()`, the tab enters a quiet focus mode while a task is live, and the battery is keyboard-only.

When the battery finishes it POSTs to `POST /api/v1/assessments`, which builds the session server-side (server-owned ids and timestamps), runs the same pipeline as `run_assessment.py` (baseline-aware), and persists the session JSON, the text report and the V2-enriched CSV row.

API endpoints also remain callable directly:

```powershell
curl http://127.0.0.1:8000/api/v1/health
curl http://127.0.0.1:8000/api/v1/participants
curl http://127.0.0.1:8000/api/v1/participants/P001/history
```

`POST /api/v1/sessions/analyze` accepts a session JSON body and returns the full V2 payload; errors use `{"error": {"code", "message"}}` with 422 for insufficient history and 404 for unknown sessions.

`POST /api/v1/assessments` takes a browser-collected battery (`participant_id`, `tasks`, optional `questionnaire` / `self_report_load_band` / `baseline_session`), rejects incomplete batteries (`INCOMPLETE_BATTERY` → 400) and unsafe participant ids (`INVALID_PARTICIPANT` → 400), then scores and stores the session. It returns `{session_id, participant_id, overall_score, band, history_size, payload, report}`.

Rebuild the summary CSV from the stored session JSONs (fixes schema drift in old rows):

```powershell
python scripts/backfill_summary.py --dry-run
python scripts/backfill_summary.py
```

Use the trained model only when you want it:

```powershell
python run_assessment.py --participant-id P001 --use-trained-model
```

Build a participant-specific baseline from sessions marked as baseline:

```powershell
python recalibrate.py --participant-id P001
```

Run the smoke test:

```powershell
python scripts/smoke_test.py
```

## Put it on GitHub, then run it on another machine

**There is no API key and no cloud account involved.** Scoring is local: rule-based thresholds by
config, plus local scikit-learn model artifacts under `data/models/`. Nothing calls out to a network
service at runtime, so a clone needs no secrets — only Python and the pinned dependencies.

### Which model is actually running

The API applies the deployed model automatically when `data/models/softmax_predictor.json` exists
(the same path `run_assessment.py --use-trained-model` uses). The serving artifact is **plain JSON
weights, not a pickle**, so it transfers between machines without scikit-learn version risk — and it
is committed, so a clone uses it with no extra steps. Verify it on the new machine:

```powershell
curl http://127.0.0.1:8765/api/v1/health
# "scoring":"trained_model"      → the model is live
# "scoring":"rule_based"        → no artifact found, thresholds are scoring
# "model_stale":true            → artifact predates newer features; retrain it
```

Force the rule-based path for a reproducible comparison at any time with `?use_model=false` on
`/sessions/analyze`, `/sessions/analyze/enrich`, `/sessions/{id}/analysis` and `/assessments`.

To refresh the serving artifact from the current summary CSV (writes `softmax_predictor.json`):

```powershell
python train_predictor.py --label-column self_report_load_band   # fits the JSON weights
python train_predictor.py --compare                              # same + model comparison + registry
```

Worth knowing before you trust it: a model artifact carries the feature schema it was trained on. The
one currently in this repo was trained before three features existed, so it scores on 14 of 17
features — `/health` reports `model_stale: true` and names them. Retraining on the committed CSV
(22 rows) refreshes it, but with that few labeled rows the trained model is considerably more
extreme than the rule-based scorer; the honest default for a small study is `?use_model=false` until
there is more labeled data.

On the machine that has the project:

```powershell
git init
git add .
git commit -m "Cognitive Load Predictor V2"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

The `.gitignore` keeps `.venv/`, caches, agent tool state and the per-device raw session files out of
the repo. **`data/session_summary.csv` is committed** — it is your real participant data (ids, sleep,
stress, caffeine, scores). If the repo is public, that data becomes public with it; drop the file from
the commit or make the repo private if that matters.

On the other machine (Windows shown; macOS/Linux is the same minus `start.bat`):

```powershell
git clone https://github.com/<you>/<repo>.git
cd <repo>
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt   # macOS/Linux: .venv/bin/python -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest -q                          # expect: 117 passed
.\start.bat                                                    # or: python serve_api.py --port 8765
```

Then open `http://127.0.0.1:8765/#/test`. Notes that actually bite:

- **Never copy `.venv/` between machines or operating systems** — recreate it; it is gitignored anyway.
- Python 3.11+ (developed on 3.14). `requirements.txt` pins minimums, not exact versions.
- `start.bat` is Windows-only. On macOS/Linux run `.venv/bin/python serve_api.py --port 8765`.
- The first session for a new participant has no history, so trend and forecast stay gated (3 and 5
  prior sessions respectively) and the UI says so rather than inventing a number.
- **Raw sessions, reports and baselines are gitignored**, so on a fresh clone the dashboard shows your
  history from the CSV but old session detail pages will 404 — the raw trial JSONs aren't there. To
  carry them across, copy the `data/` folder by hand (or delete those lines from `.gitignore`).
- The Geist/Instrument fonts load from Google Fonts; offline, the UI falls back to system fonts.
- Retrain on the new machine any time with `python train_predictor.py --compare` (this also refreshes
  the serving artifact and the registry).

## Recommended study workflow

1. Collect 2-3 baseline sessions per participant using `--baseline-session`.
2. Recalibrate that participant with `recalibrate.py`.
3. Run regular sessions and record optional self-report load labels.
4. Train the model after enough labeled rows exist.
5. Use the saved classifier for future sessions.

## Output files

- Raw sessions: `data/sessions/<session_id>.json`
- Flat summary table: `data/session_summary.csv` (V1 + V2 columns)
- Text reports: `data/reports/<session_id>.txt` (classic + V2 intelligence sections)
- Participant baselines: `data/baselines/<participant_id>.json`
- Trained model: `data/models/softmax_predictor.json`
- V2 model artifacts: `data/models/<name>_v2.joblib` + `data/models/model_registry.json`

## Dependencies

```powershell
python -m pip install -r requirements.txt
python -m pip install pytest httpx   # for the test suite
```

`xgboost` is optional: when installed it joins the `--compare` model comparison.

## Honest scope

This remains a research prototype: behavioral proxies, rule-based scoring by default, and small-sample history. The V2 layer adds intelligence but makes no clinical claims — forecasts are projections from short histories, explanations are exact decompositions rather than causal analyses, and recommendations are suggestions with stated bases, not advice.
