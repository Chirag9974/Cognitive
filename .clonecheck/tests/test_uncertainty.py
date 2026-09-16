import math

from cognitive_load.uncertainty import (
    assess_uncertainty,
    consistency_confidence,
    data_volume_confidence,
    margin_confidence,
    probability_margin,
    summarize_uncertainty,
)

UNIFORM = {"Low": 0.25, "Manageable": 0.25, "Elevated": 0.25, "High": 0.25}
DECISIVE = {"Low": 0.05, "Manageable": 0.10, "Elevated": 0.15, "High": 0.70}
SPLIT = {"Low": 0.10, "Manageable": 0.15, "Elevated": 0.38, "High": 0.37}


# components -------------------------------------------------------------------


def test_margin_extracts_top_two():
    label, prob, margin = probability_margin(DECISIVE)
    assert label == "High"
    assert prob == 0.70
    assert math.isclose(margin, 0.55)


def test_margin_confidence_monotonic():
    assert margin_confidence(0.9) > margin_confidence(0.5) > margin_confidence(0.1)
    assert margin_confidence(0.65) == 1.0
    assert margin_confidence(0.01) <= 0.1


def test_data_volume_confidence_grows_with_history():
    assert data_volume_confidence(0) < data_volume_confidence(2) < data_volume_confidence(5)
    assert data_volume_confidence(10) == 1.0


def test_consistency_confidence_downgrades_outliers():
    assert consistency_confidence(None) == 0.85
    assert consistency_confidence(0.5) == 1.0
    assert consistency_confidence(3.0) < consistency_confidence(0.5)


# assess_uncertainty -----------------------------------------------------------


def test_decisive_prediction_with_history_is_high():
    result = assess_uncertainty(DECISIVE, session_count=6, max_abs_robust_z=0.4)
    assert result["level"] == "High"
    assert result["top_label"] == "High"


def test_uniform_prediction_is_low():
    result = assess_uncertainty(UNIFORM, session_count=0, max_abs_robust_z=None)
    assert result["level"] == "Low"


def test_close_split_downgraded():
    decisive = assess_uncertainty(DECISIVE, session_count=5)
    split = assess_uncertainty(SPLIT, session_count=5)
    assert split["score"] < decisive["score"]
    assert split["level"] in ("Moderate", "Low")


def test_outlier_session_downgrades_level():
    calm = assess_uncertainty(DECISIVE, session_count=6, max_abs_robust_z=0.5)
    odd = assess_uncertainty(DECISIVE, session_count=6, max_abs_robust_z=2.8)
    assert odd["score"] < calm["score"]


def test_drivers_mention_low_history():
    result = assess_uncertainty(DECISIVE, session_count=1)
    assert any("1 prior session" in d for d in result["drivers"])


def test_probabilities_are_conserved():
    result = assess_uncertainty(SPLIT, session_count=4)
    assert result["top_label"] in SPLIT
    assert 0.0 <= result["score"] <= 1.0


# summary line -----------------------------------------------------------------


def test_summarize_uncertainty_mentions_level_and_band():
    line = summarize_uncertainty(assess_uncertainty(DECISIVE, session_count=5))
    assert "confidence" in line and "High" in line
