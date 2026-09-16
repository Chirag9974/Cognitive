from cognitive_load.temporal import (
    classify_trend,
    has_min_history,
    history_size,
    sort_history,
    temporal_feature_set,
    valid_scores,
)

# valid_scores / ordering ------------------------------------------------------


def test_valid_scores_sorts_chronologically_and_skips_missing():
    rows = [
        {"session_id": "b", "started_at": "2026-01-02", "overall_score": "55.5"},
        {"session_id": "a", "started_at": "2026-01-01", "overall_score": 60.0},
        {"session_id": "c", "started_at": "2026-01-03"},  # missing
        {"session_id": "d", "started_at": "2026-01-04", "overall_score": "oops"},
    ]
    assert valid_scores(rows) == [60.0, 55.5]
    ordered_ids = [r["session_id"] for r in sort_history(rows)]
    assert ordered_ids == ["a", "b", "c", "d"]


def test_history_size_and_minimum():
    rows = [{"started_at": f"2026-01-0{i}", "overall_score": 50 + i} for i in range(1, 5)]
    assert history_size(rows) == 4
    assert has_min_history(rows, 3)
    assert not has_min_history(rows[:2], 3)


# temporal_feature_set ---------------------------------------------------------


def _rows(scores):
    return [
        {"session_id": f"H{i:02d}", "started_at": f"2026-01-{i + 1:02d}", "overall_score": s}
        for i, s in enumerate(scores)
    ]


def test_delta_and_pct_use_only_last_prior_session():
    features = temporal_feature_set(_rows([50.0, 60.0]), current_score=66.0)
    assert features["temporal_overall_score_delta"] == 6.0
    assert features["temporal_overall_score_delta_pct"] == 10.0
    assert features["temporal_session_count"] == 2


def test_no_history_yields_nones_except_count():
    features = temporal_feature_set([], current_score=55.0)
    assert features["temporal_session_count"] == 0
    assert features["temporal_overall_score_delta"] is None
    assert features["temporal_overall_score_rolling_mean"] is None
    assert features["temporal_overall_score_ema"] is None


def test_ema_matches_manual_computation():
    scores = [40.0, 60.0, 80.0]
    features = temporal_feature_set(_rows(scores), current_score=80.0)
    ema = 40.0
    for v in scores[1:]:
        ema = 0.3 * v + 0.7 * ema
    assert features["temporal_overall_score_ema"] == pytest_approx(ema)


def pytest_approx(value, places=6):
    class _A:
        def __eq__(self, other):
            return round(other, places) == round(value, places)

    return _A()


def test_rolling_mean_uses_window_of_three():
    features = temporal_feature_set(_rows([10.0, 20.0, 30.0, 90.0]), current_score=95.0)
    assert features["temporal_overall_score_rolling_mean"] == pytest_approx((20.0 + 30.0 + 90.0) / 3)


def test_trend_is_none_below_min_sessions():
    features = temporal_feature_set(_rows([50.0, 52.0]), current_score=51.0)
    assert features["temporal_overall_score_trend"] is None


# classify_trend ---------------------------------------------------------------


def test_trend_rising_load_when_scores_falling():
    result = classify_trend(_rows([70.0, 65.0, 58.0]))
    assert result["direction"] == "rising"  # scores falling -> load rising
    assert result["score_change"] < 0


def test_trend_stable_within_threshold():
    result = classify_trend(_rows([60.0, 60.5, 60.2]))
    assert result["direction"] == "stable"


def test_trend_insufficient_history():
    result = classify_trend(_rows([60.0, 61.0]))
    assert result["direction"] == "insufficient_history"
    assert result["sessions_used"] == 2


def test_trend_never_includes_current_session():
    # The current session's score only enters via current_score, not the rows.
    rows = _rows([50.0, 50.0, 50.0])
    result = classify_trend(rows)
    assert result["direction"] == "stable"
    assert 50.0 in result["window"]
