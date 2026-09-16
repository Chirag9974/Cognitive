"""V2 pipeline orchestrator.

Runs the five intelligence features over one session and assembles:
- temporal history features + trend (FR-1)
- difficulty-normalized features (FR-10 light)
- anomaly flags + session quality (FR-8 / section 18)
- uncertainty statement (FR-3)
- grounded explanation (FR-4)

It never mutates the stored session; it only produces a payload that callers
attach to the summary and CSV row.
"""
from __future__ import annotations

from .anomaly import build_anomaly_report, score_session_quality
from .config import DEFAULT_LOAD_CLASSES, MODEL_VERSION, SUMMARY_FIELDNAMES
from .difficulty import build_difficulty_features
from .explain import build_explanation, format_explanation_lines
from .forecast import build_forecast_payload
from .model import SoftmaxLoadModel
from .recommendations import build_recommendations, format_recommendations_lines
from .storage import load_summary_rows
from .temporal import classify_trend, temporal_feature_set
from .uncertainty import assess_uncertainty, summarize_uncertainty

V2_PROBABILITY_COLUMNS = {
    "Low": "v2_probability_low",
    "Manageable": "v2_probability_manageable",
    "Elevated": "v2_probability_elevated",
    "High": "v2_probability_high",
}

TREND_DIRECTIONS = ["rising", "stable", "falling", "insufficient_history"]


def load_participant_history(
    participant_id: str,
    root=None,
    exclude_session_id: str | None = None,
    csv_path=None,
) -> list[dict]:
    """Prior summary rows for one participant, oldest first (leakage guard)."""
    rows = load_summary_rows(csv_path=csv_path, root=root)
    history = [
        row
        for row in rows
        if row.get("participant_id") == participant_id and row.get("session_id") != exclude_session_id
    ]
    return sorted(history, key=lambda row: (row.get("started_at") or "", row.get("session_id") or ""))


def _probabilities_from_summary(summary: dict) -> dict[str, float]:
    """Band probabilities for the current session.

    Prefers trained-model probabilities; otherwise normalizes the V1
    range-validation scores into a probability distribution.
    """
    model_probabilities = summary.get("model_probabilities")
    if model_probabilities:
        return {label: float(model_probabilities.get(label, 0.0)) for label in DEFAULT_LOAD_CLASSES}

    validation_scores = summary.get("validation_scores") or {}
    raw = {label: float(validation_scores.get(label, 0.0)) for label in DEFAULT_LOAD_CLASSES}
    total = sum(raw.values())
    if total <= 0:
        return {label: 0.25 for label in DEFAULT_LOAD_CLASSES}
    return {label: value / total for label, value in raw.items()}


def _v2_columns(payload: dict) -> dict:
    """Map the payload onto the exact V2 CSV column names."""
    uncertainty = payload["uncertainty"]
    contributions = payload["explanation"]["all_contributions"]

    columns = {
        "v2_prediction_method": payload["prediction_method"],
        "v2_predicted_band": uncertainty["top_label"],
        "v2_confidence": uncertainty["score"],
        "v2_uncertainty": uncertainty["level"],
        "v2_model_version": MODEL_VERSION,
        "session_quality_score": payload["quality"]["score"],
        "anomaly_score": payload["anomaly"]["anomaly_score"],
        "anomaly_status": payload["anomaly"]["status"],
        "trend_direction": payload["trend"]["direction"],
        "forecast_band": payload.get("forecast", {}).get("band", ""),
        "forecast_probability": payload.get("forecast", {}).get("probability", ""),
    }
    for index in (1, 2, 3):
        columns[f"top_contributor_{index}"] = (
            contributions[index - 1]["feature"] if len(contributions) >= index else ""
        )
    for label, column in V2_PROBABILITY_COLUMNS.items():
        columns[column] = round(payload["probabilities"].get(label, 0.0), 4)
    return columns


def update_summary_row(summary_row: dict, v2_payload: dict) -> dict:
    """Fill the V2 columns of a CSV row (only SUMMARY_FIELDNAMES survive)."""
    row = dict(summary_row)
    row.update(_v2_columns(v2_payload))
    return {field: row.get(field, "") for field in SUMMARY_FIELDNAMES}


def format_v2_report(payload: dict) -> list[str]:
    """Human-readable V2 block for the text report."""
    lines = ["", "=== V2 Intelligence Report ==="]
    lines.append(f"History: {payload['history_size']} prior session(s) for {payload['participant_id']}")

    trend = payload["trend"]
    if trend["direction"] == "insufficient_history":
        lines.append(f"Trend: not enough history ({trend['sessions_used']}/{trend['min_sessions']} sessions)")
    else:
        change = trend.get("score_change")
        arrow = {"rising": "worsening", "falling": "improving", "stable": "stable"}[trend["direction"]]
        lines.append(f"Trend: {trend['direction']} ({arrow} load), recent score change {change:+.1f}")

    difficulty = payload["difficulty"]
    lines.append(
        f"Task difficulty: {difficulty['score']:.0f}/100 "
        f"(memory {difficulty['components']['memory']:.2f}, switching {difficulty['components']['switching']:.2f}, "
        f"stroop {difficulty['components']['stroop']:.2f}) -> adjusted error rate "
        f"{difficulty['difficulty_adjusted_error_rate']:.3f} (raw {difficulty['raw_error_rate']:.3f})"
    )

    quality = payload["quality"]
    lines.append(f"Session quality: {quality['score']:.1f}/100")
    anomaly = payload["anomaly"]
    lines.append(f"Anomaly status: {anomaly['status']} (score {anomaly['anomaly_score']:.2f})")
    for reason in anomaly["reasons"]:
        lines.append(f"  - [{reason['severity']}] {reason['code']}: {reason['detail']}")
    if not anomaly["reasons"]:
        lines.append("  - no reliability concerns detected")

    lines.append(f"Uncertainty: {summarize_uncertainty(payload['uncertainty'])}")
    lines.append(f"Band probabilities: {payload['probabilities']}")
    lines.extend(format_explanation_lines(payload["explanation"]))

    forecast = payload.get("forecast")
    if forecast:
        lines.append("")
        if forecast.get("available"):
            lines.append(
                f"Forecast (next session): {forecast['band']} "
                f"(p={forecast['probability']:.2f}, projected score {forecast['projected_score']:.1f})"
            )
        else:
            lines.append(f"Forecast (next session): unavailable - {forecast['reason']}")

    recommendations = payload.get("recommendations")
    if recommendations:
        lines.extend(format_recommendations_lines(recommendations))
    return lines


def run_v2_pipeline(
    session: dict,
    summary: dict,
    history_rows: list[dict],
    baseline_profile: dict | None = None,
    model: SoftmaxLoadModel | None = None,
) -> dict:
    """Compute all five V2 intelligence payloads for one session.

    `history_rows` must contain only PRIOR sessions for this participant
    (see load_participant_history). The current session is never mixed into
    its own reference statistics.
    """
    features = summary["features"]
    history_features = [
        {name: row[name] for name in features if row.get(name) not in ("", None)}
        for row in history_rows
    ]

    # FR-1: temporal features and trend (prior sessions only).
    temporal_features = temporal_feature_set(history_rows, float(summary["overall_score"]))
    trend = classify_trend(history_rows)

    # FR-10 light: difficulty normalization.
    difficulty_built = build_difficulty_features(session, features)
    difficulty_payload = {
        "score": difficulty_built["task_difficulty_score"],
        "difficulty_adjusted_error_rate": difficulty_built["difficulty_adjusted_error_rate"],
        "components": difficulty_built["difficulty_components"],
        "raw_error_rate": float(features.get("overall_error_rate", 0.0)),
    }

    # FR-8 / section 18: anomaly + quality.
    anomaly_report = build_anomaly_report(session, history_features, features)
    quality = score_session_quality(session)

    # FR-3: uncertainty (the plain-English line is the same one the text
    # report uses, so the UI and the report never disagree).
    probabilities = _probabilities_from_summary(summary)
    uncertainty = assess_uncertainty(
        probabilities,
        session_count=len(history_rows),
        max_abs_robust_z=anomaly_report["max_abs_robust_z"],
    )
    uncertainty["statement"] = summarize_uncertainty(uncertainty)

    # FR-4: explanation (model path only when a trained model decided).
    explanation = build_explanation(
        summary,
        features,
        baseline_profile=baseline_profile,
        model=model if summary.get("prediction_method") == "trained_model" else None,
    )

    payload = {
        "participant_id": session["participant_id"],
        "session_id": session["session_id"],
        "history_size": len(history_rows),
        "overall_score": round(float(summary["overall_score"]), 2),
        "prediction_method": "trained_model" if summary.get("prediction_method") == "trained_model" else "rule_based",
        "temporal": temporal_features,
        "trend": trend,
        "difficulty": difficulty_payload,
        "quality": quality,
        "anomaly": anomaly_report,
        "probabilities": {label: round(value, 4) for label, value in probabilities.items()},
        "uncertainty": uncertainty,
        "explanation": explanation,
        "band": uncertainty["top_label"],
        "forecast": build_forecast_payload(history_rows),
    }
    payload["recommendations"] = build_recommendations(payload)
    return payload
