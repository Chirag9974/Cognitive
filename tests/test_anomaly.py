from cognitive_load.anomaly import (
    build_anomaly_report,
    check_fast_but_inaccurate,
    check_impossible_timing,
    check_repetitive_responses,
    robust_z,
    score_session_quality,
)

# robust_z ---------------------------------------------------------------------


def test_robust_z_basic_and_insufficient_data():
    values = [10.0, 10.5, 10.0, 10.2, 10.1]
    assert robust_z(10.1, values) < 1.0
    assert robust_z(20.0, values) > 3.5
    assert robust_z(10.0, [10.0, 10.0]) is None  # too few values


def test_robust_z_zero_mad_extremes():
    values = [10.0, 10.0, 10.0, 10.0]
    assert robust_z(10.0, values) == 0.0
    assert robust_z(30.0, values) > 0  # inf, but a real number in comparisons


# heuristic checks -------------------------------------------------------------


def test_impossible_timing_flagged(session_factory):
    session = session_factory(profile="low")
    session["tasks"]["reaction_time"]["trials"][0]["response_ms"] = 90.0
    finding = check_impossible_timing(session)
    assert finding and finding["code"] == "impossible_timing"
    assert check_impossible_timing(session_factory(profile="low")) is None


def test_repetitive_responses_flagged(session_factory):
    session = session_factory(profile="low")
    for trial in session["tasks"]["stroop"]["trials"]:
        trial["pressed_key"] = "r"  # same answer on every trial
    finding = check_repetitive_responses(session)
    assert finding and "stroop" in finding["detail"]


def test_normal_session_has_no_repetition_flag(session_factory):
    assert check_repetitive_responses(session_factory(profile="high")) is None


def test_fast_but_inaccurate_flagged(session_factory):
    session = session_factory(profile="low")
    for trial in session["tasks"]["stroop"]["trials"]:
        trial["response_ms"] = 300.0
        trial["correct"] = False
    finding = check_fast_but_inaccurate(session)
    assert finding and "stroop" in finding["detail"]


# build_anomaly_report ---------------------------------------------------------


def _history_features(count=4, error_rate=0.10):
    return [
        {"overall_error_rate": error_rate, "reaction_mean_rt_ms": 480.0, "pause_rate": 0.05}
        for _ in range(count)
    ]


def test_report_normal_for_typical_session(session_factory, history_factory):
    session = session_factory(profile="manageable")
    from cognitive_load.analysis import build_session_summary

    summary = build_session_summary(session)
    report = build_anomaly_report(session, _history_features(), summary["features"])
    assert report["status"] in ("Normal", "Review")
    assert 0.0 <= report["anomaly_score"] <= 1.0


def test_report_escalates_to_anomalous_on_multiple_reasons(session_factory):
    session = session_factory(profile="low")
    # Plant two independent hard reasons.
    session["tasks"]["reaction_time"]["trials"][0]["response_ms"] = 90.0
    for trial in session["tasks"]["task_switching"]["trials"]:
        trial["pressed_key"] = "z"
    from cognitive_load.analysis import build_session_summary

    summary = build_session_summary(session)
    report = build_anomaly_report(session, _history_features(), summary["features"])
    assert report["anomaly_score"] >= 0.5
    assert report["status"] in ("Review", "Anomalous")


def test_report_flags_statistical_outlier_vs_history(session_factory):
    session = session_factory(profile="high")
    from cognitive_load.analysis import build_session_summary

    summary = build_session_summary(session)
    history = _history_features(count=6, error_rate=0.05)
    report = build_anomaly_report(session, history, summary["features"])
    outlier_reasons = [r for r in report["reasons"] if r["code"] == "statistical_outlier"]
    assert outlier_reasons  # a high-load session against low-error history must stand out


# score_session_quality --------------------------------------------------------


def test_quality_high_for_clean_session(session_factory):
    quality = score_session_quality(session_factory(profile="low"))
    assert quality["score"] >= 90.0
    assert quality["components"]["missing_data"] == 1.0


def test_quality_drops_with_misses(session_factory):
    session = session_factory(profile="manageable")
    for trial in session["tasks"]["reaction_time"]["trials"][:6]:
        trial["response_ms"] = None
        trial["missed"] = True
    quality = score_session_quality(session)
    clean = score_session_quality(session_factory(profile="manageable"))
    assert quality["score"] < clean["score"]


def test_quality_flags_switch_repeat_inconsistency(session_factory):
    session = session_factory(profile="manageable")
    for trial in session["tasks"]["task_switching"]["trials"]:
        if trial["is_switch"]:
            trial["correct"] = False
    quality = score_session_quality(session)
    assert quality["components"]["internal_consistency"] < 1.0
