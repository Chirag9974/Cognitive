"""FR-4 explainability.

Two grounded explanation paths:

1. Rule-based path (always available): the overall score is an exact weighted
   sum of scored features, so each feature's contribution in score points is
   computed directly — nothing is approximated or invented.

2. Model path (when a trained softmax model made the prediction): logistic
   models are linear in their (z-scored) inputs, so each feature's pull on the
   winning class logit is exact.

The module reports only contributors it can verify and explicitly says when no
single factor dominates.
"""
from __future__ import annotations

from .analysis import score_feature
from .config import (
    SCORING_THRESHOLDS,
    TASK_LABELS,
    TASK_SCORE_FEATURES,
    TASK_SCORE_WEIGHTS,
)
from .model import SoftmaxLoadModel

NEUTRAL_SCORE = 50.0
DOMINANCE_EPSILON = 3.0  # score points; below this a factor "did not dominate"

_PAUSE_ERROR_WEIGHT = {
    "pause_rate": 0.10,
    "overall_error_rate": 0.10,
}

FEATURE_LABELS = {
    "reaction_mean_rt_ms": "Reaction speed",
    "reaction_rt_std_ms": "Reaction consistency",
    "reaction_false_start_rate": "False starts",
    "switch_cost_ms": "Switching cost",
    "switch_mean_rt_ms": "Switching speed",
    "switch_accuracy": "Switching accuracy",
    "repeat_accuracy": "Repetition accuracy",
    "working_memory_full_recall_rate": "Full-sequence recall",
    "working_memory_max_span": "Memory span reached",
    "working_memory_position_accuracy": "Item-position accuracy",
    "working_memory_mean_response_ms": "Recall speed",
    "stroop_interference_ms": "Stroop interference",
    "stroop_mean_rt_ms": "Stroop speed",
    "stroop_incongruent_accuracy": "Accuracy under conflict",
    "stroop_rt_std_ms": "Stroop consistency",
    "pause_rate": "Pauses / timeouts",
    "overall_error_rate": "Overall error rate",
}

_DIRECTION_PHRASES = {
    # feature: (phrase when the raw value is HIGH, phrase when LOW)
    "reaction_mean_rt_ms": ("responses were slow", "responses were fast"),
    "reaction_rt_std_ms": ("reaction times were erratic", "reaction times were steady"),
    "reaction_false_start_rate": ("frequent false starts", "few false starts"),
    "switch_cost_ms": ("a large task-switching cost", "a small task-switching cost"),
    "switch_mean_rt_ms": ("slow switching responses", "quick switching responses"),
    "switch_accuracy": ("switching accuracy was high", "switching accuracy was low"),
    "repeat_accuracy": ("repeat-trial accuracy was high", "repeat-trial accuracy was low"),
    "working_memory_full_recall_rate": ("many sequences recalled in full", "few sequences recalled in full"),
    "working_memory_max_span": ("a long memory span", "a short memory span"),
    "working_memory_position_accuracy": ("accurate item placement", "error-prone item placement"),
    "working_memory_mean_response_ms": ("slow recall responses", "quick recall responses"),
    "stroop_interference_ms": ("large interference from conflicting words", "small interference from conflicting words"),
    "stroop_mean_rt_ms": ("slow colour-naming responses", "quick colour-naming responses"),
    "stroop_incongruent_accuracy": ("strong accuracy under conflict", "weakened accuracy under conflict"),
    "stroop_rt_std_ms": ("erratic colour-naming times", "steady colour-naming times"),
    "pause_rate": ("frequent pauses or timeouts", "few pauses or timeouts"),
    "overall_error_rate": ("a high overall error rate", "a low overall error rate"),
}


def _feature_weight(feature_name: str) -> float:
    """Exact weight of a feature in the overall score (weights sum to 1.0).

    overall = 0.80 * sum(task_score * task_weight) + 0.10 * pause + 0.10 * error,
    so each task feature's end-to-end weight is 0.80 * task_weight * within_task_weight.
    """
    if feature_name in _PAUSE_ERROR_WEIGHT:
        return _PAUSE_ERROR_WEIGHT[feature_name]
    for task_name, weighted_features in TASK_SCORE_FEATURES.items():
        for candidate, weight in weighted_features:
            if candidate == feature_name:
                return 0.80 * TASK_SCORE_WEIGHTS[task_name] * weight
    return 0.0


def _task_of(feature_name: str) -> str:
    for task_name, weighted_features in TASK_SCORE_FEATURES.items():
        if any(candidate == feature_name for candidate, _ in weighted_features):
            return TASK_LABELS[task_name]
    return "Global"


def _format_value(feature_name: str, value: float) -> str:
    if feature_name.endswith(("_rate", "_accuracy")):
        return f"{value:.0%}"
    if feature_name.endswith("_ms"):
        return f"{value:,.0f} ms"
    if feature_name == "working_memory_max_span":
        return f"{value:.0f} items"
    return f"{value:.2f}"


def _reference_value(feature_name: str) -> float:
    """Raw feature value that scores exactly 50 (the neutral midpoint)."""
    threshold = SCORING_THRESHOLDS[feature_name]
    return (float(threshold["excellent"]) + float(threshold["poor"])) / 2.0


def rule_based_contributions(features: dict, baseline_profile: dict | None = None) -> list[dict]:
    """Exact per-feature contribution to the overall score, in score points.

    overall_score = 50 + sum(contribution_i), so the list is a complete,
    additive accounting of the result. Negative points = pushed the score down
    = raised the estimated load.
    """
    contributions = []
    for feature_name in SCORING_THRESHOLDS:
        if feature_name not in features:
            continue
        weight = _feature_weight(feature_name)
        if weight == 0.0:
            continue
        feature_score = score_feature(feature_name, float(features[feature_name]), baseline_profile)
        points = weight * (feature_score - NEUTRAL_SCORE)
        raw_value = float(features[feature_name])
        value_high = raw_value > _reference_value(feature_name)
        phrase = _DIRECTION_PHRASES[feature_name][0 if value_high else 1]

        contributions.append(
            {
                "feature": feature_name,
                "label": FEATURE_LABELS.get(feature_name, feature_name),
                "task": _task_of(feature_name),
                "points": round(points, 2),
                "impact": "raised_load" if points < 0 else "lowered_load",
                "value": round(raw_value, 4),
                "value_display": _format_value(feature_name, raw_value),
                "detail": phrase,
            }
        )

    contributions.sort(key=lambda item: abs(item["points"]), reverse=True)
    return contributions


def model_contributions(model: SoftmaxLoadModel, features: dict, top_n: int = 5) -> list[dict]:
    """Exact per-feature pull on the winning class for the softmax model.

    The model is linear in z-scored inputs, so each feature's pull on the
    winning logit is weight_i * z_i, in logit units. This is an exact
    decomposition of the winning score, not an approximation.
    """
    vector = {name: float(features[name]) for name in model.feature_names if name in features}
    if len(vector) < len(model.feature_names):
        return []

    top_label, top_probability = max(
        ((label, prob) for label, prob in model.predict_proba(features).items()),
        key=lambda item: item[1],
    )
    class_index = model.classes.index(top_label)

    contributions = []
    for name in model.feature_names:
        z = (vector[name] - model.mean[model.feature_names.index(name)]) / model.scale[
            model.feature_names.index(name)
        ]
        logit_pull = float(model.weights[model.feature_names.index(name)][class_index]) * z
        if logit_pull == 0.0:
            continue
        base_phrase = _DIRECTION_PHRASES.get(name, (None, None))
        phrase = base_phrase[0 if z > 0 else 1] if base_phrase else None
        contributions.append(
            {
                "feature": name,
                "label": FEATURE_LABELS.get(name, name),
                "task": _task_of(name),
                "points": round(logit_pull, 4),
                "impact": "raised_load" if logit_pull > 0 else "lowered_load",
                "value": round(vector[name], 4),
                "value_display": _format_value(name, vector[name]),
                "detail": phrase,
            }
        )

    contributions.sort(key=lambda item: abs(item["points"]), reverse=True)
    return contributions[:top_n]


def build_explanation(
    summary: dict,
    features: dict,
    baseline_profile: dict | None = None,
    model: SoftmaxLoadModel | None = None,
    max_items: int = 4,
) -> dict:
    """FR-4 payload: ranked contributions + plain-English narrative.

    Chooses the model path when a trained model produced the final band,
    otherwise the exact rule-based decomposition (always available).
    """
    used_model = model is not None and summary.get("prediction_method") == "trained_model"

    if used_model:
        contributions = model_contributions(model, features)
        unit = "logit"
    else:
        contributions = rule_based_contributions(features, baseline_profile)
        unit = "points"

    contributors = [c for c in contributions if c["impact"] == "raised_load"]
    supporters = [c for c in contributions if c["impact"] == "lowered_load"]

    top = contributions[0] if contributions else None
    if not top or abs(top["points"]) < DOMINANCE_EPSILON:
        dominance = "no_single_factor"
    else:
        runner_up_abs = abs(contributions[1]["points"]) if len(contributions) > 1 else 0.0
        dominance = "dominant" if abs(top["points"]) > 2.0 * max(runner_up_abs, 1e-9) else "leading"

    return {
        "method": "model_linear" if used_model else "rule_decomposition",
        "unit": unit,
        "exact": True,
        "dominance": dominance,
        "raised_load": contributors[:max_items],
        "lowered_load": supporters[:max_items],
        "all_contributions": contributions,
        "narrative": build_narrative(summary, contributors, supporters, dominance, unit),
    }


def build_narrative(
    summary: dict,
    contributors: list[dict],
    supporters: list[dict],
    dominance: str,
    unit: str,
) -> list[str]:
    """Plain-English sentences, each tied to a specific verified contributor."""
    band = summary.get("final_load_band", summary.get("rule_load_band", ""))
    score = summary.get("overall_score")
    sentences: list[str] = []

    if not contributors:
        sentences.append(
            f"No single measured behaviour pushed the result toward '{band}'; the score of "
            f"{score:.1f} reflects a broadly even performance across tasks."
        )
        if supporters:
            strengths = "; ".join(
                f"{c['detail']} ({c['value_display']})" for c in supporters[:2]
            )
            sentences.append(f"Working particularly well: {strengths}.")
        return sentences

    if dominance == "dominant":
        lead = contributors[0]
        sentences.append(
            f"The result is driven mainly by {lead['detail']} "
            f"({lead['value_display']}) in the {lead['task']} task."
        )
        rest = contributors[1:3]
        if rest:
            sentences.append("Also contributing: " + "; ".join(f"{c['detail']} ({c['value_display']})" for c in rest) + ".")
    else:
        listed = contributors[:3]
        sentences.append(
            "The main factors pushing the result up were: "
            + "; ".join(f"{c['detail']} ({c['value_display']})" for c in listed)
            + "."
        )

    if supporters:
        best = supporters[0]
        sentences.append(
            f"Working in the other direction, {best['detail']} ({best['value_display']}) "
            "partially offset the effect."
        )

    return sentences


def format_explanation_lines(explanation: dict) -> list[str]:
    """Render an explanation block for the text report."""
    lines = [f"Explanation method: {explanation['method']} (exact, no approximation)"]
    lines.extend(f"- {sentence}" for sentence in explanation["narrative"])

    unit_name = "score points" if explanation["unit"] == "points" else "logit units"
    top = explanation["all_contributions"][:5]
    if top:
        lines.append("Top factors (negative = raised load, in " + unit_name + "):")
        lines.extend(
            f"  * {c['label']:<28} {c['points']:>8.2f}   ({c['value_display']})" for c in top
        )
    return lines

