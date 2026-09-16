"""FR-6 recommendations: a deterministic rules layer.

Maps the V2 pipeline payload (band, top contributors, trend, quality, anomaly
status, uncertainty level) onto prioritized, plain-English actions. Every
recommendation carries its `basis`: the exact trigger that fired, in the same
terms as the rest of the report, so a reader can verify it. The layer is
deterministic — the same payload always produces the same list — and makes no
causal claims: actions are phrased as things to *consider*, not predicted
effects.
"""
from __future__ import annotations

from .config import RECOMMENDATION_MAX_ITEMS
from .explain import FEATURE_LABELS

# Thresholds the rules layer fires on (score points from the explanation).
MILD_CONTRIBUTION = 2.0        # |points| at/above this counts as a real push
STRONG_CONTRIBUTION = 5.0      # |points| at/above this is a leading factor

# Raw-value triggers, evaluated on the current session's features.
_ACCURACY_FEATURES = {
    "switch_accuracy",
    "repeat_accuracy",
    "working_memory_full_recall_rate",
    "working_memory_position_accuracy",
    "stroop_incongruent_accuracy",
}

# feature -> (short title, suggested action). Actions are behavioural
# suggestions, deliberately phrased as options rather than promises.
TASK_ADVICE: dict[str, tuple[str, str]] = {
    "reaction_mean_rt_ms": (
        "Slow reaction responses",
        "Consider a short warm-up round before assessed sessions, and check whether response timing feels different on this device or time of day.",
    ),
    "reaction_rt_std_ms": (
        "Erratic reaction timing",
        "Consider reducing distractions during the reaction task (notifications, background noise) to see whether timing steadies.",
    ),
    "reaction_false_start_rate": (
        "Frequent false starts",
        "Consider waiting for the cue rather than anticipating it — anticipation errors often shrink with a brief practice round.",
    ),
    "switch_cost_ms": (
        "Large task-switching cost",
        "Consider brief daily task-switching practice (alternating simple rules) and avoid multitasking immediately before assessed sessions.",
    ),
    "switch_mean_rt_ms": (
        "Slow switching responses",
        "Consider a short pause between sessions and blocks; switching speed often improves when sessions are not run back-to-back.",
    ),
    "repeat_accuracy": (
        "Lower repeat-trial accuracy",
        "Consider re-running the task-switching practice mode; repeat trials measure rule retention.",
    ),
    "working_memory_full_recall_rate": (
        "Few sequences recalled in full",
        "Consider practicing digit-span tasks at comfortable lengths and pacing recall deliberately rather than rushing.",
    ),
    "working_memory_max_span": (
        "Short memory span reached",
        "Consider chunking strategies (grouping digits in pairs) during the memory task.",
    ),
    "working_memory_position_accuracy": (
        "Error-prone item placement",
        "Consider slowing the recall phase — accuracy per position usually improves when recall is deliberate.",
    ),
    "working_memory_mean_response_ms": (
        "Slow recall responses",
        "Consider whether input method or fatigue is slowing entry; try the session at a different time of day.",
    ),
    "stroop_interference_ms": (
        "Large conflict interference",
        "Consider a brief colour-word practice round before assessed sessions to warm up selective attention.",
    ),
    "stroop_mean_rt_ms": (
        "Slow colour-naming responses",
        "Consider checking screen brightness and colour visibility, then retesting.",
    ),
    "stroop_incongruent_accuracy": (
        "Lower accuracy under conflict",
        "Consider prioritizing accuracy over speed in the Stroop task for a few sessions.",
    ),
    "stroop_rt_std_ms": (
        "Erratic colour-naming times",
        "Consider minimizing distractions during the attention task.",
    ),
    "pause_rate": (
        "Frequent pauses or timeouts",
        "Consider completing sessions in one sitting when possible; frequent pauses fragment the measurement.",
    ),
    "overall_error_rate": (
        "High overall error rate",
        "Consider retaking the session after a short break — global error spikes often reflect momentary distraction.",
    ),
}

CAVEAT = (
    "Recommendations are rule-based suggestions derived from session data; "
    "they are not clinical advice and do not claim causal effects."
)


def _title_for(feature: str) -> str:
    if feature in TASK_ADVICE:
        return TASK_ADVICE[feature][0]
    return FEATURE_LABELS.get(feature, feature)


def _band_rule(band: str) -> dict | None:
    if band == "High":
        return {
            "id": "band_high",
            "priority": 1,
            "title": "High load detected",
            "action": "Consider treating today as a high-load day: schedule demanding cognitive work for later, take a real break, and retake the assessment after resting.",
            "basis": "final load band is 'High'",
        }
    if band == "Elevated":
        return {
            "id": "band_elevated",
            "priority": 2,
            "title": "Elevated load detected",
            "action": "Consider planning demanding tasks for a period when you typically perform best, and re-assess after the next rest.",
            "basis": "final load band is 'Elevated'",
        }
    return None


def _contributor_rules(payload: dict) -> list[dict]:
    """One rule per significant raised-load contributor (max 2)."""
    explanation = payload["explanation"]
    items: list[dict] = []
    raised = [
        c
        for c in explanation.get("raised_load", [])
        if abs(c.get("points", 0.0)) >= MILD_CONTRIBUTION
    ]
    for contributor in raised[:2]:
        feature = contributor["feature"]
        advice = TASK_ADVICE.get(feature)
        if not advice:
            continue
        title, action = advice
        strong = abs(contributor["points"]) >= STRONG_CONTRIBUTION
        items.append(
            {
                "id": f"contributor_{feature}",
                "priority": 1 if strong else 3,
                "title": title,
                "action": action,
                "basis": (
                    f"{contributor['detail']} ({contributor['value_display']}) "
                    f"contributes {abs(contributor['points']):.1f} score points toward load"
                ),
            }
        )
    return items


def _trend_rule(payload: dict) -> dict | None:
    trend = payload.get("trend", {})
    if trend.get("direction") == "rising":
        return {
            "id": "trend_rising",
            "priority": 2,
            "title": "Load trending upward",
            "action": "Consider lighter sessions and more rest over the next few days, then watch whether the trend reverses.",
            "basis": (
                f"recent score change is {trend.get('score_change', 0):+.1f} "
                f"over the last {trend.get('sessions_used', 0)} sessions (worsening)"
            ),
        }
    if trend.get("direction") == "falling":
        return {
            "id": "trend_falling",
            "priority": 4,
            "title": "Load trending downward",
            "action": "Current routine seems to suit you — consider keeping it stable while you keep collecting sessions.",
            "basis": (
                f"recent score change is {trend.get('score_change', 0):+.1f} "
                f"over the last {trend.get('sessions_used', 0)} sessions (improving)"
            ),
        }
    return None


def _reliability_rule(payload: dict) -> dict | None:
    anomaly = payload.get("anomaly", {})
    quality = payload.get("quality", {})
    if anomaly.get("status") != "Normal" or quality.get("score", 100.0) < 70.0:
        reasons = [r["detail"] for r in anomaly.get("reasons", [])][:2]
        basis = (
            f"anomaly status is {anomaly.get('status')}"
            + (f": {'; '.join(reasons)}" if reasons else "")
            + f"; session quality {quality.get('score', 0):.0f}/100"
        )
        return {
            "id": "reliability",
            "priority": 1,
            "title": "Treat this session's results with caution",
            "action": "Do not act on this result. Address the flagged data issues and retake the session before drawing conclusions.",
            "basis": basis,
        }
    return None


def _uncertainty_rule(payload: dict) -> dict | None:
    if payload.get("uncertainty", {}).get("level") == "Low":
        return {
            "id": "low_confidence",
            "priority": 4,
            "title": "More history will sharpen this estimate",
            "action": "Consider completing sessions on a few different days (same conditions) so the tool can calibrate to your baseline.",
            "basis": f"confidence is Low ({payload['uncertainty'].get('score', 0):.2f})",
        }
    return None


def build_recommendations(payload: dict) -> dict:
    """Deterministic FR-6 rules over a run_v2_pipeline payload.

    Returns {"items": [...], "caveat": ...} with at most
    RECOMMENDATION_MAX_ITEMS items, sorted by priority.
    """
    band = payload.get("band") or payload.get("uncertainty", {}).get("top_label", "")
    candidates: list[dict] = []

    for rule in (
        _reliability_rule(payload),
        _band_rule(band),
        *_contributor_rules(payload),
        _trend_rule(payload),
        _uncertainty_rule(payload),
    ):
        if rule is not None:
            candidates.append(rule)

    # Deterministic order: priority, then rule id (stable tie-break).
    candidates.sort(key=lambda item: (item["priority"], item["id"]))
    items = candidates[:RECOMMENDATION_MAX_ITEMS]

    return {
        "items": items,
        "count": len(items),
        "caveat": CAVEAT,
    }


def format_recommendations_lines(recommendations: dict) -> list[str]:
    """Render the recommendations block for the text report."""
    lines = ["", "=== Recommendations (rule-based suggestions, not advice) ==="]
    if not recommendations["items"]:
        lines.append("- No specific recommendations for this session.")
        lines.append(f"- {recommendations['caveat']}")
        return lines
    for index, item in enumerate(recommendations["items"], start=1):
        lines.append(f"{index}. [{item['title']}] {item['action']}")
        lines.append(f"   Basis: {item['basis']}.")
    lines.append(f"- {recommendations['caveat']}")
    return lines
