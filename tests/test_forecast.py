import pytest

from cognitive_load.config import FORECAST_MIN_SESSIONS
from cognitive_load.forecast import (
    WEIGHT_EMA,
    WEIGHT_LAST,
    WEIGHT_ROLLING_MEAN,
    WEIGHT_TREND,
    build_forecast_payload,
    forecast_next_band,
)


def _history(scores: list[float]) -> list[dict]:
    return [
        {
            "session_id": f"s{i}",
            "participant_id": "T",
            "started_at": f"2026-01-{i + 1:02d}T09:00:00",
            "overall_score": score,
        }
        for i, score in enumerate(scores)
    ]


# blending weights --------------------------------------------------------


def test_blend_weights_sum_to_one():
    assert WEIGHT_LAST + WEIGHT_EMA + WEIGHT_ROLLING_MEAN + WEIGHT_TREND == pytest.approx(1.0)


# insufficient-history guard -----------------------------------------------


def test_forecast_raises_below_min_sessions():
    with pytest.raises(ValueError, match="INSUFFICIENT_HISTORY"):
        forecast_next_band(_history([70, 72, 74, 76]))  # 4 < 5


def test_forecast_raises_on_empty_history():
    with pytest.raises(ValueError, match=str(FORECAST_MIN_SESSIONS)):
        forecast_next_band([])


def test_payload_degrades_gracefully_with_reason():
    payload = build_forecast_payload(_history([70.0, 71.0]))
    assert payload["available"] is False
    assert "INSUFFICIENT_HISTORY" in payload["reason"]
    assert payload["band"] == ""
    assert payload["projected_score"] is None


# projection behaviour ------------------------------------------------------


def test_stable_history_projects_onto_same_band():
    history = _history([80.0, 81.0, 80.5, 80.2, 80.8])
    forecast = forecast_next_band(history)
    assert forecast["available"] is True
    assert forecast["band"] == "Low"  # 80+ sits in the Low band
    assert 0.0 <= forecast["probability"] <= 1.0
    assert 75.0 <= forecast["projected_score"] <= 85.0


def test_improving_history_projects_above_its_smoothed_baseline():
    history = _history([60.0, 62.0, 64.0, 66.0, 68.0])
    forecast = forecast_next_band(history)
    # The uptrend must pull the projection above the smoothed statistics
    # (EMA/rolling mean lag behind the last score), even though the blend
    # keeps the projection conservative relative to the last score.
    assert forecast["projected_score"] > forecast["inputs"]["ema"]
    assert forecast["projected_score"] > forecast["inputs"]["rolling_mean"]


def test_band_probabilities_sum_to_one():
    history = _history([55.0, 57.0, 56.0, 58.0, 57.5])
    probabilities = forecast_next_band(history)["probabilities"]
    assert set(probabilities) == {"Low", "Manageable", "Elevated", "High"}
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_projected_score_never_leaves_score_range():
    history = _history([100.0, 100.0, 100.0, 100.0, 100.0])
    assert forecast_next_band(history)["projected_score"] <= 100.0
    history = _history([0.0, 0.0, 0.0, 0.0, 0.0])
    assert forecast_next_band(history)["projected_score"] >= 0.0


def test_forecast_is_deterministic():
    history = _history([70.0, 68.0, 71.0, 66.0, 69.0])
    first = forecast_next_band(history)
    second = forecast_next_band(history)
    assert first == second


def test_missing_scores_are_skipped_not_fatal():
    rows = _history([80.0, 81.0, 82.0, 83.0, 84.0, 85.0])
    rows[1]["overall_score"] = ""  # malformed row
    forecast = forecast_next_band(rows)
    assert forecast["inputs"]["sessions_used"] == 5  # 5 valid scores remain
