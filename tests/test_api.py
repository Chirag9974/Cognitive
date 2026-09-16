import json

import pytest
from fastapi.testclient import TestClient

from cognitive_load.analysis import build_session_summary
from cognitive_load.api.app import app, project_root_dep
from cognitive_load.config import FEATURE_COLUMNS
from cognitive_load.pipeline import load_participant_history
from cognitive_load.simulation import generate_synthetic_session
from cognitive_load.storage import append_summary_row, ensure_project_layout, save_session


@pytest.fixture()
def client(tmp_path):
    """API client bound to a temporary data directory."""
    ensure_project_layout(tmp_path)
    app.dependency_overrides[project_root_dep] = lambda: tmp_path
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def _session(participant: str = "API1", profile: str = "manageable", seed: int = 1) -> dict:
    session = generate_synthetic_session(participant, profile=profile, seed=seed)
    session["started_at"] = f"2026-09-01T09:00:00"
    session["session_id"] = f"{participant}-sess"
    return session


def _seed_history(participant: str, scores: list[float], tmp_path) -> None:
    """Write prior summary rows so history-based endpoints have data."""
    for index, score in enumerate(scores):
        row = {
            "session_id": f"{participant}-h{index}",
            "participant_id": participant,
            "started_at": f"2026-08-{index + 1:02d}T09:00:00",
            "overall_score": score,
        }
        append_summary_row(row, tmp_path)


# health ----------------------------------------------------------------------


def test_health_returns_versions(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["api_version"] == "v1"
    assert body["model_version"] == "2.0.0"


# participants ------------------------------------------------------------------


def test_participants_lists_ids_with_counts(client, tmp_path):
    _seed_history("PA", [80.0, 75.0], tmp_path)
    _seed_history("PB", [70.0], tmp_path)
    response = client.get("/api/v1/participants")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    ids = {p["participant_id"]: p for p in body["participants"]}
    assert ids["PA"]["sessions"] == 2
    assert ids["PB"]["sessions"] == 1
    assert ids["PA"]["latest"].startswith("2026-08-02")


def test_participants_empty(client):
    response = client.get("/api/v1/participants")
    assert response.status_code == 200
    assert response.json() == {"count": 0, "participants": []}


def test_static_ui_served_at_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Cognitive Load Predictor" in response.text
    assert response.headers["content-type"].startswith("text/html")


def test_static_mount_does_not_shadow_api(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# history ----------------------------------------------------------------------


def test_participant_history_empty(client):
    response = client.get("/api/v1/participants/NOBODY/history")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 0
    assert body["sessions"] == []


def test_participant_history_returns_prior_rows(client, tmp_path):
    _seed_history("HIST", [80.0, 75.0, 70.0], tmp_path)
    response = client.get("/api/v1/participants/HIST/history")
    body = response.json()
    assert body["count"] == 3
    assert body["sessions"][0]["session_id"] == "HIST-h0"  # oldest first


# analyze ----------------------------------------------------------------------


def test_analyze_returns_full_v2_payload(client):
    response = client.post("/api/v1/sessions/analyze", json=_session())
    assert response.status_code == 200
    payload = response.json()
    for key in ("uncertainty", "explanation", "temporal", "difficulty", "quality", "anomaly", "forecast", "recommendations"):
        assert key in payload
    assert payload["recommendations"]["count"] == 0  # healthy synthetic session


def test_analyze_missing_field_returns_structured_error(client):
    session = _session()
    del session["tasks"]
    response = client.post("/api/v1/sessions/analyze", json=session)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "MISSING_FIELD"
    assert "tasks" in error["message"]


def test_analyze_malformed_body_returns_structured_error(client):
    response = client.post(
        "/api/v1/sessions/analyze",
        content=b"not-json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "MALFORMED_BODY"


def test_stored_session_analysis_404(client):
    response = client.get("/api/v1/sessions/GHOST/analysis")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "SESSION_NOT_FOUND"


def test_stored_session_analysis_roundtrip(client, tmp_path):
    session = _session("STORED")
    save_session(session, tmp_path)
    response = client.get("/api/v1/sessions/STORED-sess/analysis")
    assert response.status_code == 200
    assert response.json()["session_id"] == "STORED-sess"


# forecast ---------------------------------------------------------------------


def test_forecast_guard_maps_to_422(client, tmp_path):
    _seed_history("SHORT", [70.0, 71.0, 72.0], tmp_path)
    response = client.post("/api/v1/forecast", json={"participant_id": "SHORT"})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "INSUFFICIENT_HISTORY"
    assert "5" in error["message"]


def test_forecast_happy_path(client, tmp_path):
    _seed_history("FULL", [80.0, 79.0, 81.0, 80.5, 80.2], tmp_path)
    response = client.post("/api/v1/forecast", json={"participant_id": "FULL"})
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["band"] == "Low"
    assert "probabilities" in body


def test_forecast_missing_participant(client):
    response = client.post("/api/v1/forecast", json={})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "MISSING_FIELD"


# enrich (analyze + persist) ------------------------------------------------------


def test_analyze_enrich_persists_session_and_csv_row(client, tmp_path):
    session = _session("ENRICH")
    response = client.post("/api/v1/sessions/analyze/enrich", json=session)
    assert response.status_code == 200

    saved = json.loads((tmp_path / "data" / "sessions" / "ENRICH-sess.json").read_text(encoding="utf-8"))
    assert saved["session_id"] == "ENRICH-sess"

    csv_text = (tmp_path / "data" / "session_summary.csv").read_text(encoding="utf-8")
    assert "ENRICH-sess" in csv_text
    assert "v2_predicted_band" in csv_text  # V2 columns present in header

    # The persisted row is now available as history for the next analysis.
    history = load_participant_history("ENRICH", root=tmp_path)
    assert history and history[-1]["session_id"] == "ENRICH-sess"


# assessments (browser battery) ---------------------------------------------------


def _battery(participant: str = "WEB1", profile: str = "manageable", seed: int = 3) -> dict:
    """The payload shape the in-browser battery posts (trials only, no ids)."""
    session = generate_synthetic_session(participant, profile=profile, seed=seed)
    return {
        "participant_id": participant,
        "duration_seconds": 240,
        "tasks": session["tasks"],
    }


def test_assessment_scores_and_persists_the_battery(client, tmp_path):
    response = client.post("/api/v1/assessments", json=_battery("WEB1"))
    assert response.status_code == 200
    body = response.json()

    # Server owns the identity and timestamps.
    assert body["participant_id"] == "WEB1"
    assert body["session_id"].startswith("WEB1-")
    assert body["history_size"] == 0
    assert body["band"] == body["payload"]["band"]
    assert 0.0 <= body["overall_score"] <= 100.0

    payload = body["payload"]
    assert set(payload["probabilities"]) == {"Low", "Manageable", "Elevated", "High"}
    assert payload["recommendations"]["caveat"]  # always carries its caveat
    assert payload["quality"]["score"] > 0

    # Session JSON, text report and V2-enriched CSV row all land on disk.
    saved = json.loads((tmp_path / "data" / "sessions" / f"{body['session_id']}.json").read_text(encoding="utf-8"))
    assert saved["source"] == "web"
    assert set(saved["tasks"]) == {"reaction_time", "task_switching", "working_memory", "stroop"}
    assert saved["summary"]["overall_score"] == payload["overall_score"]
    assert (tmp_path / "data" / "reports" / f"{body['session_id']}.txt").exists()

    csv_text = (tmp_path / "data" / "session_summary.csv").read_text(encoding="utf-8")
    assert body["session_id"] in csv_text
    assert "v2_predicted_band" in csv_text


def test_assessment_second_battery_sees_first_as_history(client, tmp_path):
    first = client.post("/api/v1/assessments", json=_battery("WEB2", seed=4)).json()
    second = client.post("/api/v1/assessments", json=_battery("WEB2", seed=5))
    assert second.status_code == 200
    body = second.json()
    assert body["history_size"] == 1
    assert body["session_id"] != first["session_id"]
    assert body["payload"]["trend"]["direction"] in {"rising", "falling", "stable", "insufficient_history"}


def test_assessment_produces_suggestions_when_load_is_high(client):
    """A struggling battery must come back with actionable, grounded advice."""
    body = client.post("/api/v1/assessments", json=_battery("WEB8", profile="high", seed=7)).json()
    payload = body["payload"]
    assert payload["band"] in {"Elevated", "High"}
    items = payload["recommendations"]["items"]
    assert items, "a high-load session should yield recommendations"
    assert all(item["basis"] and item["action"] for item in items)
    assert len(items) <= 4


def test_assessment_ids_never_collide(client, tmp_path):
    """Two batteries posted within one second must not overwrite each other."""
    payload = _battery("WEB3", seed=6)
    ids = {client.post("/api/v1/assessments", json=payload).json()["session_id"] for _ in range(2)}
    assert len(ids) == 2
    saved = list((tmp_path / "data" / "sessions").glob("WEB3-*.json"))
    assert len(saved) == 2


def test_assessment_rejects_partial_battery(client):
    payload = _battery("WEB4")
    del payload["tasks"]["stroop"]
    response = client.post("/api/v1/assessments", json=payload)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "INCOMPLETE_BATTERY"
    assert "stroop" in error["message"]


def test_assessment_rejects_empty_trials(client):
    payload = _battery("WEB5")
    payload["tasks"]["working_memory"] = {"trials": []}
    response = client.post("/api/v1/assessments", json=payload)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INCOMPLETE_BATTERY"


def test_assessment_rejects_unsafe_participant_id(client):
    payload = _battery("WEB6")
    payload["participant_id"] = "../etc/passwd"
    response = client.post("/api/v1/assessments", json=payload)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_PARTICIPANT"


def test_assessment_requires_participant_and_tasks(client):
    assert client.post("/api/v1/assessments", json={"tasks": {}}).json()["error"]["code"] == "MISSING_FIELD"
    assert client.post("/api/v1/assessments", json={"participant_id": "X"}).json()["error"]["code"] == "MISSING_FIELD"


def test_assessment_malformed_body(client):
    response = client.post("/api/v1/assessments", content=b"not json", headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "MALFORMED_BODY"


def test_assessment_normalizes_self_report_band(client):
    payload = _battery("WEB7")
    payload["self_report_load_band"] = "ELEVATED"
    body = client.post("/api/v1/assessments", json=payload).json()
    assert body["session_id"].startswith("WEB7-")


# trained model wiring ------------------------------------------------------------


def _deploy_model(tmp_path, classes=("Low", "Manageable", "Elevated", "High"), features=None) -> None:
    """Write a valid softmax artifact so the API has a model to load.

    Uniform weights: the artifact is loadable and its probabilities are exactly
    knowable (1/n each), which keeps this test about *wiring*, not accuracy.
    """
    features = list(features) if features is not None else list(FEATURE_COLUMNS)
    artifact = {
        "feature_names": features,
        "classes": list(classes),
        "mean": [0.0] * len(features),
        "scale": [1.0] * len(features),
        "weights": [[0.0] * len(classes) for _ in features],
        "bias": [0.0] * len(classes),
    }
    path = tmp_path / "data" / "models" / "softmax_predictor.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact), encoding="utf-8")


def test_health_reports_rule_based_without_a_model(client):
    body = client.get("/api/v1/health").json()
    assert body["scoring"] == "rule_based"
    assert body["classes"] == []


def test_health_reports_the_trained_model_when_deployed(client, tmp_path):
    _deploy_model(tmp_path)
    body = client.get("/api/v1/health").json()
    assert body["scoring"] == "trained_model"
    assert body["classes"] == ["Low", "Manageable", "Elevated", "High"]
    assert body["model_stale"] is False


def test_health_flags_a_model_trained_on_an_older_feature_schema(client, tmp_path):
    """A stale artifact loads and predicts, silently ignoring new features."""
    _deploy_model(tmp_path, features=list(FEATURE_COLUMNS)[:-3])
    body = client.get("/api/v1/health").json()
    assert body["scoring"] == "trained_model"
    assert body["model_stale"] is True
    assert set(body["missing_features"]) == set(list(FEATURE_COLUMNS)[-3:])


def test_deployed_model_actually_decides_the_band(client, tmp_path):
    """A deployed artifact must change results, not just sit on disk."""
    _deploy_model(tmp_path)
    body = client.post("/api/v1/sessions/analyze", json=_session("MODELED")).json()
    assert body["prediction_method"] == "trained_model"
    assert set(body["probabilities"]) == {"Low", "Manageable", "Elevated", "High"}
    assert all(abs(value - 0.25) < 1e-6 for value in body["probabilities"].values())


def test_browser_battery_uses_the_deployed_model(client, tmp_path):
    _deploy_model(tmp_path)
    body = client.post("/api/v1/assessments", json=_battery("WEB9", seed=8)).json()
    assert body["payload"]["prediction_method"] == "trained_model"


def test_use_model_false_forces_rule_based_even_with_an_artifact(client, tmp_path):
    _deploy_model(tmp_path)
    body = client.post("/api/v1/sessions/analyze?use_model=false", json=_session("PLAIN")).json()
    assert body["prediction_method"] == "rule_based"


def test_stored_session_analysis_respects_use_model_flag(client, tmp_path):
    _deploy_model(tmp_path)
    save_session(_session("STORED2"), tmp_path)
    assert client.get("/api/v1/sessions/STORED2-sess/analysis").json()["prediction_method"] == "trained_model"
    off = client.get("/api/v1/sessions/STORED2-sess/analysis?use_model=false").json()
    assert off["prediction_method"] == "rule_based"


def test_assessment_records_baseline_aware_scoring(client, tmp_path):
    """The battery and the CLI must agree: both pass the personal baseline in."""
    baseline_path = tmp_path / "data" / "baselines" / "WEB10.json"
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps({"overall_score": 80.0, "features": {}}), encoding="utf-8")
    body = client.post("/api/v1/assessments", json=_battery("WEB10", seed=9)).json()
    assert body["session_id"].startswith("WEB10-")
    assert body["payload"]["prediction_method"] == "rule_based"
