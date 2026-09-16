"""FR-9 HTTP API: expose the V2 intelligence pipeline over /api/v1.

Endpoints
---------
GET  /api/v1/health                                - liveness + versions
GET  /api/v1/participants/{id}/history             - prior summary rows
POST /api/v1/assessments                           - score + persist a real battery
POST /api/v1/sessions/analyze                      - full V2 payload for a session
GET  /api/v1/sessions/{session_id}/analysis        - same, from stored session JSON
POST /api/v1/forecast                              - next-session forecast

Errors use a structured envelope: {"error": {"code": ..., "message": ...}}.
INSUFFICIENT_HISTORY maps to HTTP 422, unknown sessions to 404, malformed
bodies to 400.

The project root is a FastAPI dependency so tests can point the service at a
temporary data directory.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from ..analysis import apply_model_prediction, build_session_summary, build_summary_row, format_report
from ..calibration import load_baseline
from ..config import APP_VERSION, FEATURE_COLUMNS, MODEL_VERSION
from ..forecast import build_forecast_payload
from ..model import SoftmaxLoadModel, normalize_label
from ..pipeline import (
    format_v2_report,
    load_participant_history,
    run_v2_pipeline,
    update_summary_row,
)
from ..storage import (
    append_summary_row,
    default_baseline_path,
    default_model_path,
    load_summary_rows,
    project_root as storage_project_root,
    save_report,
    save_session,
)

API_VERSION = "v1"

# A complete battery needs trials for every task; partial batteries would score
# as if the missing tasks were perfect, so they are rejected outright.
REQUIRED_TASKS = ("reaction_time", "task_switching", "working_memory", "stroop")
PARTICIPANT_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,40}$")


def _json_body_schema(schema: dict, required: tuple[str, ...] = ()) -> dict:
    """OpenAPI requestBody doc for endpoints that parse a raw Request.

    Reading `await request.json()` keeps our structured error envelopes, but
    FastAPI can't infer the body from a bare Request — this documents it so
    Swagger UI renders a request-body editor (see fastapi discussion #11473).
    """
    doc = {**schema}
    if required:
        doc["required"] = list(required)
    return {
        "requestBody": {
            "content": {"application/json": {"schema": doc}},
            "required": True,
        }
    }


_ASSESSMENT_BODY_SCHEMA = {
    "type": "object",
    "properties": {
        "participant_id": {"type": "string", "example": "P001"},
        "duration_seconds": {"type": "integer", "description": "Wall-clock length of the battery."},
        "baseline_session": {"type": "boolean"},
        "self_report_load_band": {"type": "string", "description": "Optional low/manageable/elevated/high self-report."},
        "questionnaire": {
            "type": "object",
            "properties": {
                "sleep_hours": {"type": "number"},
                "stress_level": {"type": "integer"},
                "caffeine_mg": {"type": "integer"},
                "fatigue_level": {"type": "integer"},
                "notes": {"type": "string"},
            },
        },
        "tasks": {
            "type": "object",
            "description": (
                "Trial data collected by the browser battery: reaction_time, task_switching, "
                "working_memory and stroop, each shaped {trials: [...]} exactly like the desktop app."
            ),
            "required": list(REQUIRED_TASKS),
        },
    },
}

_SESSION_BODY_SCHEMA = {
    "type": "object",
    "properties": {
        "session_id": {"type": "string"},
        "participant_id": {"type": "string"},
        "started_at": {"type": "string"},
        "self_report_load_band": {"type": "string"},
        "tasks": {
            "type": "object",
            "description": "Trial-level task data: reaction_time, task_switching, working_memory, stroop.",
        },
    },
}


def project_root_dep() -> Path:
    """Overridable in tests via app.dependency_overrides."""
    return storage_project_root()


app = FastAPI(title="Cognitive Load Predictor", version=APP_VERSION)

# Dev convenience: browser-based dashboards (file:// or other ports) may call
# the API directly. Restrict origins before any production deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

router = __import__("fastapi").APIRouter(prefix=f"/api/{API_VERSION}")
app.include_router(router)

# Static UI (web/) at the root — mounted after the API router so /api/* wins.
_WEB_DIR = Path(__file__).resolve().parents[3] / "web"
if _WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_WEB_DIR), html=True), name="web")


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": {"code": code, "message": message}})


def _require_fields(body: dict, fields: tuple[str, ...]) -> str | None:
    for field in fields:
        if not body.get(field):
            return field
    return None


def _load_model(root: Path) -> SoftmaxLoadModel | None:
    path = default_model_path(root)
    if not path.exists():
        return None
    try:
        return SoftmaxLoadModel.load(path)
    except (OSError, json.JSONDecodeError, KeyError):
        return None


# --------------------------------------------------------------------- health


@router.get("/health")
def health(root: Path = Depends(project_root_dep)) -> dict:
    """Liveness plus which scorer is actually live.

    `scoring` is the honest answer to "is the trained model in use?" — check it
    after deploying to another machine instead of guessing from the file list.
    """
    model = _load_model(root)
    scored_features = list(model.feature_names) if model else []
    missing_features = [name for name in FEATURE_COLUMNS if name not in scored_features]
    return {
        "status": "ok",
        "app_version": APP_VERSION,
        "model_version": MODEL_VERSION,
        "api_version": API_VERSION,
        "scoring": "trained_model" if model else "rule_based",
        "classes": model.classes if model else [],
        # A model trained on an older feature schema still loads and still
        # predicts — it just quietly ignores the newer features. Say so.
        "model_stale": bool(missing_features),
        "missing_features": missing_features,
    }


# ------------------------------------------------------------------- history


@router.get("/participants")
def participants(root: Path = Depends(project_root_dep)) -> dict:
    """Distinct participant ids with session counts and latest activity."""
    rows = load_summary_rows(root=root)
    tally: dict[str, dict] = {}
    for row in rows:
        pid = (row.get("participant_id") or "").strip()
        if not pid:
            continue
        entry = tally.setdefault(pid, {"participant_id": pid, "sessions": 0, "latest": ""})
        entry["sessions"] += 1
        started = row.get("started_at") or ""
        if started > entry["latest"]:
            entry["latest"] = started
    participants = sorted(tally.values(), key=lambda item: item["participant_id"])
    return {"count": len(participants), "participants": participants}


@router.get("/participants/{participant_id}/history")
def participant_history(participant_id: str, root: Path = Depends(project_root_dep)) -> dict:
    rows = load_participant_history(participant_id, root=root)
    return {
        "participant_id": participant_id,
        "count": len(rows),
        "sessions": rows,
    }


# --------------------------------------------------------------- assessments


def _unique_session_id(root: Path, base: str) -> str:
    """Session ids are filenames — never overwrite an existing battery."""
    candidate = base
    suffix = 2
    sessions_dir = root / "data" / "sessions"
    while (sessions_dir / f"{candidate}.json").exists():
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def _build_session(participant_id: str, tasks: dict, body: dict, root: Path) -> dict:
    """Assemble the desktop-app session shape from browser-collected trials.

    The server owns the identifiers and timestamps so two submissions in the
    same second can never collide, whatever the client clock says.
    """
    finished_at = datetime.now()
    try:
        duration_seconds = max(0, int(body.get("duration_seconds") or 0))
    except (TypeError, ValueError):
        duration_seconds = 0
    started_at = finished_at - timedelta(seconds=duration_seconds)

    questionnaire = body.get("questionnaire") if isinstance(body.get("questionnaire"), dict) else {}
    session = {
        "session_id": _unique_session_id(root, f"{participant_id}-{finished_at.strftime('%Y%m%d-%H%M%S')}"),
        "participant_id": participant_id,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "duration_seconds": duration_seconds,
        "baseline_session": bool(body.get("baseline_session", False)),
        "questionnaire": questionnaire,
        "source": "web",
        "tasks": {name: {"trials": list(tasks[name]["trials"])} for name in REQUIRED_TASKS},
    }
    self_report = normalize_label(body.get("self_report_load_band"))
    if self_report:
        session["self_report_load_band"] = self_report
    return session


@router.post(
    "/assessments",
    response_model=None,
    openapi_extra=_json_body_schema(_ASSESSMENT_BODY_SCHEMA, ("participant_id", "tasks")),
)
async def create_assessment(
    request: Request,
    root: Path = Depends(project_root_dep),
    use_model: bool = True,
) -> JSONResponse | dict:
    """Score a session taken in the browser and persist it as history.

    This is the write path behind the in-browser battery: it runs the same
    pipeline as run_assessment.py (baseline awareness included) and stores the
    session JSON, the text report, and the V2-enriched summary row.
    """
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        return _error(400, "MALFORMED_BODY", "Request body must be a JSON assessment object.")

    if not isinstance(body, dict):
        return _error(400, "MALFORMED_BODY", "Request body must be a JSON assessment object.")

    participant_id = str(body.get("participant_id") or "").strip()
    if not participant_id:
        return _error(400, "MISSING_FIELD", "Assessment object is missing required field: participant_id.")
    if not PARTICIPANT_ID_PATTERN.match(participant_id):
        return _error(
            400,
            "INVALID_PARTICIPANT",
            "participant_id must be 1-40 characters of letters, digits, dot, dash or underscore.",
        )

    tasks = body.get("tasks")
    if not isinstance(tasks, dict):
        return _error(400, "MISSING_FIELD", "Assessment object is missing required field: tasks.")
    empty_tasks = [
        name
        for name in REQUIRED_TASKS
        if not isinstance(tasks.get(name), dict) or not tasks[name].get("trials")
    ]
    if empty_tasks:
        return _error(
            400,
            "INCOMPLETE_BATTERY",
            f"Assessment is missing trial data for: {', '.join(empty_tasks)}.",
        )

    session = _build_session(participant_id, tasks, body, root)
    baseline = load_baseline(default_baseline_path(participant_id, root))
    try:
        summary = build_session_summary(session, baseline_profile=baseline)
    except (KeyError, TypeError, ValueError) as error:
        return _error(400, "MALFORMED_SESSION", f"Session data could not be scored: {error}.")

    model = _load_model(root) if use_model else None
    summary = _apply_trained_model(summary, model)
    history = load_participant_history(participant_id, root=root, exclude_session_id=session["session_id"])
    try:
        payload = run_v2_pipeline(session, summary, history, baseline_profile=baseline, model=model)
    except (KeyError, TypeError, ValueError) as error:
        return _error(400, "MALFORMED_SESSION", f"Session data could not be analyzed: {error}.")

    session["summary"] = summary
    row = update_summary_row(build_summary_row(session, summary), payload)
    save_session(session, root)
    append_summary_row(row, root)
    report_text = format_report(session, summary) + "\n" + "\n".join(format_v2_report(payload))
    save_report(session["session_id"], report_text, root)

    return {
        "session_id": session["session_id"],
        "participant_id": participant_id,
        "overall_score": payload["overall_score"],
        "band": payload["band"],
        "history_size": payload["history_size"],
        "payload": payload,
        "report": report_text,
    }


# ------------------------------------------------------------------- analysis


def _apply_trained_model(summary: dict, model: SoftmaxLoadModel | None) -> dict:
    """Let the trained model decide the band when one is deployed.

    Mirrors run_assessment.py --use-trained-model. Without this step the model
    is loaded but never consulted, so every API/browser result silently stayed
    rule-based even with a model on disk (summary['prediction_method'] is what
    run_v2_pipeline checks before using model-based explanations).
    """
    if model is None:
        return summary
    return apply_model_prediction(summary, model.predict(summary["features"]))


def _analyze(session: dict, root: Path, use_model: bool = True) -> dict:
    session_id = session["session_id"]
    participant_id = session["participant_id"]
    history = load_participant_history(participant_id, root=root, exclude_session_id=session_id)
    baseline = load_baseline(default_baseline_path(participant_id, root))
    model = _load_model(root) if use_model else None
    summary = build_session_summary(session, baseline_profile=baseline)
    summary = _apply_trained_model(summary, model)
    return run_v2_pipeline(session, summary, history, baseline_profile=baseline, model=model)


@router.post("/sessions/analyze", response_model=None, openapi_extra=_json_body_schema(_SESSION_BODY_SCHEMA, ("session_id", "participant_id", "tasks")))
async def analyze_session(
    request: Request,
    root: Path = Depends(project_root_dep),
    use_model: bool = True,
) -> JSONResponse | dict:
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        return _error(400, "MALFORMED_BODY", "Request body must be a JSON session object.")

    if not isinstance(body, dict):
        return _error(400, "MALFORMED_BODY", "Request body must be a JSON session object.")
    missing = _require_fields(body, ("session_id", "participant_id", "tasks"))
    if missing:
        return _error(400, "MISSING_FIELD", f"Session object is missing required field: {missing}.")

    try:
        payload = _analyze(body, root, use_model=use_model)
    except KeyError as error:
        return _error(400, "MALFORMED_SESSION", f"Session data is incomplete: {error}.")
    except (TypeError, ValueError) as error:
        return _error(400, "MALFORMED_SESSION", f"Session data could not be scored: {error}.")
    return payload


@router.post("/sessions/analyze/enrich", response_model=None, openapi_extra=_json_body_schema(_SESSION_BODY_SCHEMA, ("session_id", "participant_id", "tasks")))
async def analyze_and_store(
    request: Request,
    root: Path = Depends(project_root_dep),
    use_model: bool = True,
) -> JSONResponse | dict:
    """Analyze a session AND persist it: session JSON + V2-enriched CSV row."""
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        return _error(400, "MALFORMED_BODY", "Request body must be a JSON session object.")
    missing = _require_fields(body, ("session_id", "participant_id", "tasks"))
    if missing:
        return _error(400, "MISSING_FIELD", f"Session object is missing required field: {missing}.")

    try:
        payload = _analyze(body, root, use_model=use_model)
    except KeyError as error:
        return _error(400, "MALFORMED_SESSION", f"Session data is incomplete: {error}.")
    except (TypeError, ValueError) as error:
        return _error(400, "MALFORMED_SESSION", f"Session data could not be scored: {error}.")

    row = update_summary_row(
        {
            "session_id": body["session_id"],
            "participant_id": body["participant_id"],
            "started_at": body.get("started_at", ""),
        },
        payload,
    )
    save_session(body, root)
    append_summary_row(row, root)
    return payload


@router.get("/sessions/{session_id}/analysis", response_model=None)
def stored_session_analysis(
    session_id: str,
    root: Path = Depends(project_root_dep),
    use_model: bool = True,
) -> JSONResponse | dict:
    session_path = root / "data" / "sessions" / f"{session_id}.json"
    if not session_path.exists():
        return _error(404, "SESSION_NOT_FOUND", f"No stored session with id {session_id}.")
    try:
        session = json.loads(session_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return _error(500, "SESSION_UNREADABLE", f"Stored session could not be read: {error}.")
    try:
        return _analyze(session, root, use_model=use_model)
    except (KeyError, TypeError, ValueError) as error:
        return _error(400, "MALFORMED_SESSION", f"Stored session could not be scored: {error}.")


# ------------------------------------------------------------------- forecast


@router.post("/forecast", response_model=None, openapi_extra=_json_body_schema({"type": "object", "properties": {"participant_id": {"type": "string"}}}, ("participant_id",)))
async def forecast(request: Request, root: Path = Depends(project_root_dep)) -> JSONResponse | dict:
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        return _error(400, "MALFORMED_BODY", "Request body must be JSON: {participant_id}.")

    participant_id = body.get("participant_id") if isinstance(body, dict) else None
    if not participant_id:
        return _error(400, "MISSING_FIELD", "participant_id is required.")

    history = load_participant_history(participant_id, root=root)
    payload = build_forecast_payload(history)
    if not payload["available"]:
        return _error(422, "INSUFFICIENT_HISTORY", payload["reason"])
    payload["participant_id"] = participant_id
    return payload
