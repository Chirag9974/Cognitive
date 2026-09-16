from cognitive_load.explain import (
    build_explanation,
    format_explanation_lines,
    rule_based_contributions,
)
from cognitive_load.analysis import build_session_summary

# rule_based_contributions -----------------------------------------------------


def test_contributions_sum_to_score_offset(session_factory):
    session = session_factory(profile="elevated")
    summary = build_session_summary(session)
    contributions = rule_based_contributions(summary["features"])
    total = sum(c["points"] for c in contributions)
    reconstructed = 50.0 + total
    assert abs(reconstructed - summary["overall_score"]) < 0.5  # exact additive accounting


def test_high_load_session_has_raised_load_contributors(session_factory):
    session = session_factory(profile="high")
    summary = build_session_summary(session)
    explanation = build_explanation(summary, summary["features"])
    assert explanation["raised_load"], "a high-load session must have load-raising factors"
    assert explanation["method"] == "rule_decomposition"


def test_every_contribution_carries_display_and_detail(session_factory):
    session = session_factory(profile="manageable")
    summary = build_session_summary(session)
    for c in rule_based_contributions(summary["features"]):
        assert c["value_display"], c["feature"]
        assert c["detail"], c["feature"]
        assert c["task"] in ("Reaction Time", "Task Switching", "Working Memory", "Selective Attention", "Global")
        assert c["impact"] in ("raised_load", "lowered_load")


def test_contributions_sorted_by_magnitude(session_factory):
    session = session_factory(profile="manageable")
    summary = build_session_summary(session)
    points = [abs(c["points"]) for c in rule_based_contributions(summary["features"])]
    assert points == sorted(points, reverse=True)


# build_explanation ------------------------------------------------------------


def test_no_single_factor_when_performance_even(session_factory):
    # Construct an artificial perfectly-even feature set (everything at neutral).
    summary = {"final_load_band": "Manageable", "rule_load_band": "Manageable", "overall_score": 50.0}
    features = {
        "reaction_mean_rt_ms": 30175.0, "reaction_rt_std_ms": 1020.0,
        "reaction_false_start_rate": 0.15, "switch_cost_ms": 6075.0,
        "switch_mean_rt_ms": 30600.0, "switch_accuracy": 0.755,
        "repeat_accuracy": 0.79, "working_memory_full_recall_rate": 0.55,
        "working_memory_max_span": 5.0, "working_memory_position_accuracy": 0.70,
        "working_memory_mean_response_ms": 32000.0, "stroop_interference_ms": 5060.0,
        "stroop_mean_rt_ms": 30600.0, "stroop_incongruent_accuracy": 0.695,
        "stroop_rt_std_ms": 10040.0, "pause_rate": 0.21, "overall_error_rate": 0.245,
    }
    explanation = build_explanation(summary, features)
    assert explanation["dominance"] == "no_single_factor"
    assert explanation["narrative"]


def test_narrative_sentences_are_plain_english(session_factory):
    session = session_factory(profile="high")
    summary = build_session_summary(session)
    explanation = build_explanation(summary, summary["features"])
    for sentence in explanation["narrative"]:
        assert isinstance(sentence, str) and len(sentence) > 15
    assert any("main" in s.lower() or "driven" in s.lower() for s in explanation["narrative"])


def test_report_lines_render(session_factory):
    session = session_factory(profile="elevated")
    summary = build_session_summary(session)
    explanation = build_explanation(summary, summary["features"])
    lines = format_explanation_lines(explanation)
    assert any("Top factors" in line for line in lines)
    assert any(line.startswith("- ") for line in lines)


def test_model_path_used_only_for_trained_predictions(session_factory):
    session = session_factory(profile="manageable")
    summary = build_session_summary(session)  # prediction_method == range_validation
    explanation = build_explanation(summary, summary["features"], model=None)
    assert explanation["method"] == "rule_decomposition"
    assert explanation["exact"] is True
