"""FR-8 anomaly detection + section-18 session quality scoring.

Detects sessions that may be unreliable: statistically extreme feature values
(robust z via median/MAD), impossibly fast responses, fast-but-inaccurate
patterns, and repetitive responding. Sessions are flagged, never deleted or
modified; the caller decides what to do with the flags.
"""
from __future__ import annotations

from statistics import fmean, median

from .config import (
    ANOMALY_ANOMALOUS_SCORE,
    ANOMALY_REVIEW_SCORE,
    ANOMALY_ROBUST_Z_THRESHOLD,
    QUALITY_WEIGHTS,
)

# Median absolute deviation scaled to be comparable to a standard deviation.
_MAD_TO_STD = 1.4826

# Sentinel |z| used when MAD is 0 but the value differs from the median:
# the deviation is unbounded, so report a large finite, JSON-safe number.
_ZERO_MAD_SENTINEL = 10.0

# Tuned for the 60 s response windows used by every task in this app.
MIN_PLAUSIBLE_RT_MS = 200.0     # below this a human response is not plausible
FAST_INACCURATE_MIN_RATE = 0.4  # share of trials needed to call a pattern
QUALITY_MIN_TRIALS = 5

# Per-task speed floor below which a *correct decision* is implausible.
# The reaction-time task is excluded: fast presses are the point there.
FAST_RT_LIMITS_MS = {
    "task_switching": 350.0,
    "stroop": 350.0,
}

# Where the participant's answer lives in a trial record.
RESPONSE_KEYS = {
    "working_memory": "response",
    "stroop": "pressed_key",
    "task_switching": "pressed_key",
}

# Features inspected for statistical outliers (robust z vs personal history).
Z_SCORE_FEATURES = [
    "reaction_mean_rt_ms",
    "switch_cost_ms",
    "stroop_interference_ms",
    "working_memory_full_recall_rate",
    "overall_error_rate",
    "pause_rate",
]


def robust_z(value: float, values: list[float]) -> float | None:
    """Median/MAD robust z-score of `value` within `values`; None if undefined."""
    if len(values) < 3:
        return None
    med = median(values)
    mad = median([abs(v - med) for v in values])
    if mad == 0:
        if value == med:
            return 0.0
        return _ZERO_MAD_SENTINEL * (1 if value > med else -1)
    return (value - med) / (mad * _MAD_TO_STD)


def _trials(session: dict) -> list[tuple[str, dict]]:
    return [
        (task_name, trial)
        for task_name, task in session.get("tasks", {}).items()
        for trial in task.get("trials", [])
    ]


def check_impossible_timing(session: dict) -> dict | None:
    """Any response faster than humanly plausible."""
    for task_name, trial in _trials(session):
        rt = trial.get("response_ms")
        if rt is None or trial.get("missed"):
            continue
        if float(rt) < MIN_PLAUSIBLE_RT_MS:
            return {
                "code": "impossible_timing",
                "severity": "high",
                "detail":                f"{task_name}: response in {float(rt):.0f} ms is below the "
                f"{MIN_PLAUSIBLE_RT_MS:.0f} ms plausibility floor",
            }
    return None


def check_fast_but_inaccurate(session: dict) -> dict | None:
    """Implausibly quick answers paired with heavy errors — a random-clicking signature."""
    flagged = []
    for task_name, speed_limit_ms in FAST_RT_LIMITS_MS.items():
        trials = session.get("tasks", {}).get(task_name, {}).get("trials", [])
        if len(trials) < QUALITY_MIN_TRIALS:
            continue
        fast = [t for t in trials if t.get("response_ms") is not None and float(t["response_ms"]) < speed_limit_ms]
        if len(fast) / len(trials) >= FAST_INACCURATE_MIN_RATE:
            accuracy = sum(1 for t in trials if t.get("correct")) / len(trials)
            if accuracy <= 0.6:
                flagged.append(
                    f"{task_name}: {len(fast)}/{len(trials)} trials under {speed_limit_ms:.0f} ms "
                    f"with accuracy {accuracy:.0%}"
                )
    if flagged:
        return {"code": "fast_but_inaccurate", "severity": "medium", "detail": "; ".join(flagged)}
    return None



def check_repetitive_responses(session: dict) -> dict | None:
    """The same answer given for (nearly) every trial of a task.

    Uses the task-specific answer field (pressed_key / response). The
    reaction-time task is excluded: repeating one key there is correct play.
    """
    flagged = []
    for task_name, answer_key in RESPONSE_KEYS.items():
        trials = session.get("tasks", {}).get(task_name, {}).get("trials", [])
        if len(trials) < QUALITY_MIN_TRIALS:
            continue
        answers = [trial.get(answer_key) for trial in trials if trial.get(answer_key) is not None]
        if len(answers) >= QUALITY_MIN_TRIALS:
            distinct = set(map(_hashable, answers))
            if len(distinct) == 1:
                flagged.append(f"{task_name}: identical response on all {len(answers)} trials")
    if flagged:
        return {"code": "repetitive_responses", "severity": "medium", "detail": "; ".join(flagged)}
    return None


def _hashable(value):
    if isinstance(value, (list, dict)):
        return str(value)
    return value


def build_anomaly_report(session: dict, history_features: list[dict], current_features: dict) -> dict:
    """Full FR-8 report: heuristic reasons + robust-z outliers + 0-1 anomaly score.

    `history_features` = feature dicts of the participant's prior sessions only.
    """
    reasons: list[dict] = []

    for check in (check_impossible_timing, check_fast_but_inaccurate, check_repetitive_responses):
        finding = check(session)
        if finding:
            reasons.append(finding)

    max_abs_z = 0.0
    for feature_name in Z_SCORE_FEATURES:
        value = current_features.get(feature_name)
        if value is None:
            continue
        prior_values = [
            float(row[feature_name])
            for row in history_features
            if row.get(feature_name) not in (None, "")
        ]
        if not prior_values:
            continue
        z = robust_z(float(value), prior_values)
        if z is not None and abs(z) > max_abs_z:
            max_abs_z = abs(z)
            if abs(z) >= ANOMALY_ROBUST_Z_THRESHOLD:
                reasons.append(
                    {
                        "code": "statistical_outlier",
                        "severity": "medium",
                        "detail": f"{feature_name} is {abs(z):.1f} robust deviations from "
                        f"this participant's usual value",
                    }
                )

    heuristic_reasons = sum(1 for r in reasons if r["code"] != "statistical_outlier")
    outlier_reasons = len(reasons) - heuristic_reasons
    anomaly_score = min(1.0, 0.35 * heuristic_reasons + 0.20 * outlier_reasons)

    if anomaly_score >= ANOMALY_ANOMALOUS_SCORE:
        status = "Anomalous"
    elif anomaly_score >= ANOMALY_REVIEW_SCORE:
        status = "Review"
    else:
        status = "Normal"

    return {
        "anomaly_score": round(anomaly_score, 3),
        "status": status,
        "reasons": reasons,
        "max_abs_robust_z": round(max_abs_z, 2),
    }


def score_session_quality(session: dict) -> dict:
    """Section-18 quality score (0-100) from four equal-weight components."""
    trials = _trials(session)
    trial_count = len(trials)

    if trial_count == 0:
        return {"score": 0.0, "components": {name: 0.0 for name in QUALITY_WEIGHTS}}

    # 1) Response validity: plausible answers (not misses, not absurdly fast).
    bad = sum(
        1
        for _, trial in trials
        if trial.get("missed")
        or (trial.get("response_ms") is not None and float(trial["response_ms"]) < MIN_PLAUSIBLE_RT_MS)
    )
    response_validity = 1.0 - bad / trial_count

    # 2) Timing consistency: responded trials within humanly plausible band.
    responded = [
        float(trial["response_ms"])
        for _, trial in trials
        if trial.get("response_ms") is not None and not trial.get("missed")
    ]
    if responded:
        plausible = [rt for rt in responded if MIN_PLAUSIBLE_RT_MS <= rt <= 60000.0]
        timing_consistency = len(plausible) / len(responded)
    else:
        timing_consistency = 0.0

    # 3) Missing data: every trial answered.
    missing_data = 1.0 - sum(1 for _, trial in trials if trial.get("response_ms") is None) / trial_count

    # 4) Internal consistency: task-level accuracy does not contradict itself
    #    (switch accuracy wildly below repeat accuracy signals disengagement).
    tasks = session.get("tasks", {})
    inconsistencies = 0
    checked = 0
    switching = tasks.get("task_switching", {}).get("trials", [])
    if len(switching) >= QUALITY_MIN_TRIALS:
        sw = [t for t in switching if t.get("is_switch")]
        rp = [t for t in switching if not t.get("is_switch")]
        if sw and rp:
            checked += 1
            sw_acc = sum(1 for t in sw if t.get("correct")) / len(sw)
            rp_acc = sum(1 for t in rp if t.get("correct")) / len(rp)
            if rp_acc - sw_acc > 0.35:
                inconsistencies += 1
    stroop_trials = tasks.get("stroop", {}).get("trials", [])
    if len(stroop_trials) >= QUALITY_MIN_TRIALS:
        cong = [t for t in stroop_trials if t.get("congruent")]
        incong = [t for t in stroop_trials if not t.get("congruent")]
        if cong and incong:
            checked += 1
            cong_acc = sum(1 for t in cong if t.get("correct")) / len(cong)
            incong_acc = sum(1 for t in incong if t.get("correct")) / len(incong)
            if cong_acc - incong_acc > 0.5:
                inconsistencies += 1
    internal_consistency = 1.0 if not checked else 1.0 - inconsistencies / checked

    components = {
        "response_validity": round(response_validity, 4),
        "timing_consistency": round(timing_consistency, 4),
        "missing_data": round(missing_data, 4),
        "internal_consistency": round(internal_consistency, 4),
    }
    score = sum(QUALITY_WEIGHTS[name] * components[name] for name in QUALITY_WEIGHTS)
    return {"score": round(score * 100.0, 2), "components": components}
