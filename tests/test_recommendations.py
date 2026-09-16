from cognitive_load.recommendations import (
    build_recommendations,
    format_recommendations_lines,
)


def _payload(**overrides) -> dict:
    base = {
        "band": "Manageable",
        "trend": {"direction": "stable", "score_change": 0.2, "sessions_used": 5},
        "anomaly": {"status": "Normal", "reasons": []},
        "quality": {"score": 95.0},
        "uncertainty": {"level": "Moderate", "score": 0.6, "top_label": "Manageable"},
        "explanation": {"raised_load": []},
    }
    base.update(overrides)
    return base


# happy paths -----------------------------------------------------------------


def test_healthy_session_yields_no_recommendations():
    result = build_recommendations(_payload())
    assert result["count"] == 0
    assert result["items"] == []
    assert "not clinical advice" in result["caveat"]


def test_high_band_fires_band_rule():
    result = build_recommendations(_payload(band="High"))
    titles = [item["title"] for item in result["items"]]
    assert "High load detected" in titles
    assert result["items"][0]["priority"] == 1


def test_raised_load_contributors_fire_task_rules():
    payload = _payload(
        explanation={
            "raised_load": [
                {
                    "feature": "switch_cost_ms",
                    "points": -6.0,
                    "detail": "a large task-switching cost",
                    "value_display": "400 ms",
                },
                {
                    "feature": "pause_rate",
                    "points": -3.0,
                    "detail": "frequent pauses or timeouts",
                    "value_display": "15%",
                },
            ]
        }
    )
    result = build_recommendations(payload)
    titles = [item["title"] for item in result["items"]]
    assert "Large task-switching cost" in titles
    assert "Frequent pauses or timeouts" in titles
    # Basis strings cite the same numbers as the explanation.
    switching = next(item for item in result["items"] if item["id"] == "contributor_switch_cost_ms")
    assert "6.0 score points" in switching["basis"]
    assert "400 ms" in switching["basis"]


def test_rising_trend_fires_trend_rule_with_basis():
    result = build_recommendations(_payload(trend={"direction": "rising", "score_change": -4.0, "sessions_used": 5}))
    trend_item = next(item for item in result["items"] if item["id"] == "trend_rising")
    assert "-4.0" in trend_item["basis"]


def test_falling_trend_gets_reassurance():
    result = build_recommendations(_payload(trend={"direction": "falling", "score_change": 3.0, "sessions_used": 5}))
    assert any(item["id"] == "trend_falling" for item in result["items"])


# reliability and confidence rules ---------------------------------------------


def test_anomalous_session_overrides_everything():
    result = build_recommendations(
        _payload(
            band="Low",
            anomaly={"status": "Anomalous", "reasons": [{"detail": "RT distribution irregular"}]},
            quality={"score": 55.0},
        )
    )
    assert result["items"][0]["id"] == "reliability"
    assert "Do not act on this result" in result["items"][0]["action"]


def test_low_quality_session_flags_caution():
    result = build_recommendations(_payload(quality={"score": 60.0}))
    assert any(item["id"] == "reliability" for item in result["items"])


def test_low_confidence_suggests_more_history():
    result = build_recommendations(_payload(uncertainty={"level": "Low", "score": 0.3, "top_label": "Low"}))
    assert any(item["id"] == "low_confidence" for item in result["items"])


# structure and rendering -------------------------------------------------------


def test_max_items_cap_is_respected():
    payload = _payload(
        band="High",
        trend={"direction": "rising", "score_change": -5.0, "sessions_used": 5},
        explanation={
            "raised_load": [
                {"feature": "switch_cost_ms", "points": -6.0, "detail": "d", "value_display": "v"},
                {"feature": "pause_rate", "points": -5.0, "detail": "d", "value_display": "v"},
                {"feature": "stroop_interference_ms", "points": -4.0, "detail": "d", "value_display": "v"},
            ]
        },
    )
    result = build_recommendations(payload)
    assert result["count"] <= 4


def test_output_is_deterministic():
    payload = _payload(band="Elevated")
    first = build_recommendations(payload)
    second = build_recommendations(payload)
    assert first == second


def test_format_lines_include_basis_and_caveat():
    result = build_recommendations(_payload(band="High"))
    lines = format_recommendations_lines(result)
    text = "\n".join(lines)
    assert "Basis:" in text
    assert "not clinical advice" in text


def test_format_lines_for_empty_list():
    lines = format_recommendations_lines(build_recommendations(_payload()))
    assert any("No specific recommendations" in line for line in lines)
