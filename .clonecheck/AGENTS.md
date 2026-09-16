# Project Summary (for AI agents)

**Project:** Cognitive Load Predictor System — a Python research prototype that measures a person's cognitive load through behavioral micro-tasks, scores the performance, classifies it into a load band, and logs data for later model training. Explicitly a screening/research tool, not clinical-grade.

**Stack:** Python 3.10+ (`X | None` syntax), Tkinter GUI (stdlib), numpy, scikit-learn, joblib (FR-2 model artifacts), FastAPI + uvicorn (FR-9 HTTP service). No database — everything is JSON/CSV files. Test deps: pytest, httpx (starlette TestClient).

## Layout

```
run_assessment.py      # Main CLI entry point (GUI or simulation) — now runs the V2 pipeline
train_predictor.py     # Softmax training (default) + --compare FR-2 model comparison
recalibrate.py         # Builds participant baseline profiles
serve_api.py           # Serves the FR-9 HTTP API (uvicorn)
scripts/smoke_test.py  # End-to-end check (synthetic data + training)
scripts/backfill_summary.py  # Rebuild session_summary.csv from session JSONs (schema drift fix)
scripts/make_pitch_pdf.py    # Regenerates the non-technical V2 feature pitch PDF
src/cognitive_load/
  app.py          # Tkinter GUI: 4 interactive tasks, trial-level data collection
  analysis.py     # Metrics, features, scoring, bands, report formatting
  calibration.py  # Participant-specific baseline thresholds
  config.py       # ALL constants: weights, thresholds, timing, bands, CSV schema (incl. V2)
  model.py        # Softmax classifier (numpy-only gradient descent)
  simulation.py   # Synthetic session generator (4 profiles: low/manageable/elevated/high)
  storage.py      # Path helpers, JSON/CSV persistence
  temporal.py     # FR-1: deltas, rolling stats, EMA, trend (prior-sessions-only)
  difficulty.py   # FR-10 light: task-configuration difficulty + adjusted features
  anomaly.py      # FR-8 + §18: robust-z anomalies, consistency checks, session quality
  uncertainty.py  # FR-3: confidence level + drivers (history-size honesty cap)
  explain.py      # FR-4: exact additive contribution decomposition + narrative
  forecast.py     # FR-7: next-session forecast (INSUFFICIENT_HISTORY guard at 5 sessions)
  recommendations.py # FR-6: deterministic rules with stated basis (no causal claims)
  ml.py           # FR-2: chronological-split model comparison, V2Predictor, registry
  pipeline.py     # V2 orchestrator: one payload, report block, CSV enrichment
  api/app.py      # FR-9: FastAPI /api/v1 endpoints (injectable project root) + static UI mount
data/              # Outputs: sessions/, reports/, baselines/, models/, session_summary.csv
web/               # Static UI (no build step): index.html + static/{styles.css, app.js}
tests/             # pytest suite (107 tests)
```

## Core pipeline

1. **Collect** (`app.py` or `simulation.py`): 4 tasks — reaction time (8 trials), task switching (26 trials), working memory (10 trials, digit spans 3–7), Stroop (28 trials). Every trial has a **60s response window** (`TASK_TIMING`), stored in ms and seconds. Console questionnaire captures sleep/stress/caffeine/fatigue.
2. **Summarize** (`analysis.py`): trials → per-task metrics → 17 features (`FEATURE_COLUMNS` in `config.py`) → per-task 0–100 scores (linear interpolation between `excellent`/`poor` thresholds, weighted per feature) → overall score = 80% weighted task scores + 10% pause rate + 10% overall error rate.
3. **Classify**: overall score → load band (`Low` ≥80, `Manageable` ≥60, `Elevated` ≥40, `High` below) + performance band. A **range-validation vote** (overall 40% + each task × 0.6) refines this → `final_load_band`. Optional trained-model override with `--use-trained-model`.
4. **V2 intelligence pass** (`pipeline.py`, runs automatically in `run_assessment.py`): over prior sessions only (leakage guard — the current session never enters its own reference stats):
   - **FR-1 temporal**: delta/pct-delta vs last, rolling mean/std (k=3), EMA (α=0.3), trend direction (needs ≥3 sessions).
   - **FR-10 difficulty**: task-configuration difficulty score; error rate adjusted via `mult = 1 + K·(difficulty − 0.5)`.
   - **FR-8/§18 anomaly+quality**: robust z (median/MAD, |z|>3.5) vs history, fast-but-inaccurate / impossible-timing / repetitive-response checks; status Normal/Review/Anomalous; 0–100 quality score. **Flags, never deletes.**
   - **FR-3 uncertainty**: band-probability margin + history length + variance → Low/Moderate/High; never "High" with <3 sessions.
   - **FR-4 explanation**: exact additive decomposition of the score (rule path) or exact logit pulls (softmax path); plain-English narrative; model-agnostic ablation for tree models.
   - **FR-7 forecast**: blends last/EMA/rolling mean/trend → next-session band probabilities; explicit `INSUFFICIENT_HISTORY` below 5 sessions.
   - **FR-6 recommendations**: deterministic rules (band, contributors, trend, reliability, confidence); every item carries its `basis`; capped at 4.
5. **Persistence**: session JSON + V1 report + V2 report block + one CSV row with the V2 columns (`v2_*`, quality/anomaly/trend/forecast/top contributors).

## Model training (FR-2)

```bash
python train_predictor.py                # softmax benchmark (unchanged behavior)
python train_predictor.py --compare      # FR-2: chronological split, no shuffle
```

`--compare` enriches rows with temporal features (`enrich_rows_with_temporal`), splits chronologically (train = oldest), compares `logreg` / `random_forest` / `hist_gb` (+ softmax benchmark) and persists the winner as `data/models/<name>_v2.joblib` + `model_registry.json` entry. `xgboost` is an optional install that joins the comparison when present. `V2Predictor` provides `predict_proba(features_dict)` and deterministic ablation `feature_contributions`.

## HTTP API (FR-9)

```bash
python serve_api.py                       # http://127.0.0.1:8000
GET  /api/v1/health                        # + scoring/model_stale/missing_features (see below)
GET  /api/v1/participants                     # distinct ids + session counts + latest
GET  /api/v1/participants/{id}/history    # prior summary rows, oldest first
POST /api/v1/assessments                  # browser battery → score + persist (server-owned id/timestamps)
POST /api/v1/sessions/analyze             # body = session JSON → full V2 payload (includes overall_score)
POST /api/v1/sessions/analyze/enrich      # same + persists session JSON + CSV row
GET  /api/v1/sessions/{id}/analysis       # V2 payload for a stored session
POST /api/v1/forecast                     # {participant_id} → next-session forecast
```

The static UI is mounted at `/` (after the API router, so `/api/*` wins). It is a dependency-free vanilla JS app (`web/`): hash router (`#/`, `#/test`, `#/run`, `#/session/<id>`), aurora-glass tokens in `styles.css` (dark canvas, one accent, status dots), Vercel motion rules (transform/opacity only, explicit properties, 150/250/400ms tokens, staggered reveals, reduced-motion collapse). Browser E2E driver also exists at `e2e/dashboard.html` (older standalone test page).

`web/static/test.js` is the in-browser cognitive battery: a faithful port of `src/cognitive_load/app.py` (same tasks, trial counts, pacing and trial schema). Its timing constants live in `web/static/config.js` and **mirror `TASK_TIMING` in `src/cognitive_load/config.py` — change both together**; `test.js` validates that shape at load and warns if a response/recall window is missing, because a missing window used to resolve trials instantly and silently record phantom misses. It exposes `window.__clpTest = {state, start, press, recall}` as a headless driving seam. Results are POSTed to `/api/v1/assessments`.

Errors: `{"error": {"code", "message"}}` — `INSUFFICIENT_HISTORY` → 422, unknown session → 404, malformed/missing → 400.

**Model wiring:** the API applies `data/models/softmax_predictor.json` via `apply_model_prediction`
whenever it loads (`_apply_trained_model` in `api/app.py`), so a deployed artifact actually decides the
band — and `run_v2_pipeline` then switches to the model-based explanation. Historically only the CLI
flag `--use-trained-model` did this, which is why every API/browser result was silently rule-based even
with a model on disk. `/health` reports `scoring` (`trained_model` | `rule_based`), plus `model_stale`
and `missing_features` when the artifact's `feature_names` predate `FEATURE_COLUMNS` — a stale model
still loads and still predicts, just ignoring the features it never saw. Pass `?use_model=false` to
`/sessions/analyze`, `/sessions/analyze/enrich`, `/sessions/{id}/analysis` or `/assessments` to force
the rule-based path.

## Entry points

```bash
python run_assessment.py --participant-id P001                  # Interactive GUI session
python run_assessment.py --simulate manageable --participant-id TEST01   # Synthetic, no GUI
python run_assessment.py --participant-id P001 --baseline-session        # Mark as baseline
python run_assessment.py --participant-id P001 --use-trained-model       # Apply saved model
python train_predictor.py --label-column self_report_load_band  # Needs ≥8 labeled rows
python train_predictor.py --compare                             # FR-2 comparison + registry
python recalibrate.py --participant-id P001                     # From baseline-flagged sessions
python serve_api.py                                             # FR-9 HTTP service
start.bat                                                       # one command: server + browser + cognitive test
python scripts/backfill_summary.py --dry-run                    # Rebuild CSV from session JSONs
python scripts/smoke_test.py                                    # Asserts train accuracy ≥ 0.70
python -m pytest -q                                             # Full test suite
```

## Data flow (all under `data/`)

- `sessions/<session_id>.json` — raw trial-level data (**source of truth**)
- `session_summary.csv` — one flat row per session; schema = `SUMMARY_FIELDNAMES` (incl. V2 columns); **training input**
- `reports/<session_id>.txt` — human-readable report (V1 + V2 blocks)
- `baselines/<participant_id>.json` — per-participant feature stats (mean/std/min/max)
- `models/softmax_predictor.json` — V1 classifier; `models/<name>_v2.joblib` + `models/model_registry.json` — FR-2 artifacts

## Conventions & gotchas

- **`config.py` is the single source of truth** for scoring weights, thresholds, timing, bands, V2 constants, and the CSV schema. Change scoring behavior there, not in `analysis.py`.
- Thresholds are tuned to the 60s response window (e.g., "poor" RT = 60000 ms). Changing task timing requires retuning `SCORING_THRESHOLDS` and `PAUSE_THRESHOLD_*`.
- Root scripts add `src/` to `sys.path` manually; package modules use relative imports. The API package is `cognitive_load.api.app:app`.
- `storage.project_root()` assumes the `src/cognitive_load/` layout (parents[2] of storage.py).
- `append_summary_row` auto-migrates the CSV if the header doesn't match `SUMMARY_FIELDNAMES`. If *old rows* lack feature columns (schema drift), rebuild with `scripts/backfill_summary.py` instead of imputing.
- Load labels are normalized via `LABEL_MAP`: `low/manageable/elevated/high` or `1/2/3/4` (also `good/medium/risky/excellent`). ML code must normalize labels before `sklearn.metrics.log_loss` (it assumes lexicographic class order).
- Leakage guards: temporal/forecast features use **prior sessions only**; model comparison splits **chronologically**, never shuffled.
- The V2 pipeline never mutates the stored session; `run_v2_pipeline(session, summary, history_rows, baseline_profile=None, model=None)` returns one payload consumed by the report, CSV row, and API.
- GUI uses a token-guard pattern (`trial_token` + `root.after` scheduling) to prevent stale callbacks; all state lives on the `AssessmentApp` instance.
- Scoring treats higher = better performance; `direction: "inverse"` features (RT, cost, error/pause rates) are inverted before interpolation. Scores are clamped to [0, 100].
- The pitch PDF at `C:\Users\<user>\Downloads\Cognitive Load Predictor V2 - Feature Pitch.pdf` can be regenerated with `scripts/make_pitch_pdf.py` (reportlab, venv-only dep).
