from cognitive_load.analysis import build_session_summary
from cognitive_load.pipeline import (
    format_v2_report,
    load_participant_history,
    run_v2_pipeline,
    update_summary_row,
)
from cognitive_load.config import SUMMARY_FIELDNAMES

# run_v2_pipeline --------------------------------------------------------------


def _payload(session, history):
    summary = build_session_summary(session)
    return run_v2_pipeline(session, summary, history), summary


def test_payload_contains_all_five_features(session_factory, history_factory):
    session = session_factory(profile="elevated", started_at="2026-09-15T09:00:00")
    history = history_factory(lambda i: "manageable", count=5)
    payload, _ = _payload(session, history)

    assert payload["history_size"] == 5
    assert payload["trend"]["direction"] in ("rising", "stable", "falling")
    assert payload["difficulty"]["score"] >= 0.0
    assert payload["quality"]["score"] > 0.0
    assert payload["anomaly"]["status"] in ("Normal", "Review", "Anomalous")
    assert payload["uncertainty"]["level"] in ("High", "Moderate", "Low")
    assert payload["explanation"]["narrative"]


def test_history_excludes_current_session(session_factory, history_factory):
    session = session_factory(profile="high", started_at="2026-09-15T09:00:00", session_id="TST01-CUR")
    history = history_factory(lambda i: "low", count=6)
    payload, _ = _payload(session, history)

    # All temporal stats come from prior sessions only.
    assert payload["temporal"]["temporal_session_count"] == 6
    scores = history_factory(lambda i: "low", count=6)
    assert payload["temporal"]["temporal_overall_score_rolling_mean"] is not None


def test_empty_history_still_works(session_factory):
    session = session_factory(profile="manageable", started_at="2026-09-15T09:00:00")
    payload, _ = _payload(session, [])

    assert payload["history_size"] == 0
    assert payload["temporal"]["temporal_session_count"] == 0
    assert payload["trend"]["direction"] == "insufficient_history"
    assert payload["uncertainty"]["level"] in ("High", "Moderate", "Low")
    assert payload["explanation"]["narrative"]  # explanation still produced


def test_degraded_history_raises_uncertainty_penalty(session_factory, history_factory):
    session = session_factory(profile="manageable", started_at="2026-09-15T09:00:00")
    summary = build_session_summary(session)

    rich = run_v2_pipeline(session, summary, history_factory(lambda i: "manageable", count=8))
    poor = run_v2_pipeline(session, summary, [])
    assert poor["uncertainty"]["score"] < rich["uncertainty"]["score"]


def test_worsening_history_shows_rising_trend(session_factory, history_factory):
    session = session_factory(profile="high", started_at="2026-09-15T09:00:00")
    history = history_factory(lambda i: ["low", "low", "manageable", "manageable"][i], count=4)
    payload, _ = _payload(session, history)
    assert payload["trend"]["direction"] == "rising"


# update_summary_row -----------------------------------------------------------


def test_summary_row_gets_v2_columns(session_factory, history_factory):
    session = session_factory(profile="elevated", started_at="2026-09-15T09:00:00", session_id="TST01-CSV")
    history = history_factory(lambda i: "manageable", count=4)
    payload, _ = _payload(session, history)

    row = update_summary_row({"session_id": "TST01-CSV", "participant_id": "TST01"}, payload)
    assert set(row.keys()) == set(SUMMARY_FIELDNAMES)
    assert row["anomaly_status"] in ("Normal", "Review", "Anomalous")
    assert 0.0 <= float(row["session_quality_score"]) <= 100.0
    assert row["trend_direction"] in ("rising", "stable", "falling", "insufficient_history")
    assert row["v2_uncertainty"] in ("High", "Moderate", "Low")


def test_top_contributors_filled(session_factory, history_factory):
    session = session_factory(profile="high", started_at="2026-09-15T09:00:00")
    history = history_factory(lambda i: "low", count=3)
    payload, _ = _payload(session, history)
    row = update_summary_row({}, payload)

    assert row["top_contributor_1"]
    assert row["top_contributor_2"]


# format_v2_report -------------------------------------------------------------


def test_report_block_renders_all_sections(session_factory, history_factory):
    session = session_factory(profile="high", started_at="2026-09-15T09:00:00")
    history = history_factory(lambda i: "manageable", count=5)
    payload, _ = _payload(session, history)

    text = "\n".join(format_v2_report(payload))
    for marker in ("V2 Intelligence Report", "Trend:", "Task difficulty:", "Session quality:",
                   "Anomaly status:", "Uncertainty:", "Top factors"):
        assert marker in text


def test_report_handles_empty_history(session_factory):
    session = session_factory(profile="low", started_at="2026-09-15T09:00:00")
    payload, _ = _payload(session, [])
    text = "\n".join(format_v2_report(payload))
    assert "not enough history" in text
