from __future__ import annotations

from statistics import fmean, pstdev

from .calibration import resolve_threshold
from .config import (
    DEFAULT_LOAD_CLASSES,
    FEATURE_COLUMNS,
    PAUSE_THRESHOLD_CAP_MS,
    PAUSE_THRESHOLD_RATIO,
    SCORE_BANDS,
    TASK_LABELS,
    TASK_SCORE_FEATURES,
    TASK_SCORE_WEIGHTS,
    VALIDATION_COMPONENT_WEIGHTS,
)


def _mean(values: list[float], default: float = 0.0) -> float:
    return fmean(values) if values else default


def _std(values: list[float], default: float = 0.0) -> float:
    return pstdev(values) if len(values) > 1 else default


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _pause_threshold_ms(trial: dict) -> float:
    response_window_ms = float(trial.get("response_window_ms") or 60000.0)
    return min(PAUSE_THRESHOLD_CAP_MS, response_window_ms * PAUSE_THRESHOLD_RATIO)


def score_feature(feature_name: str, value: float, baseline_profile: dict | None = None) -> float:
    threshold = resolve_threshold(feature_name, baseline_profile)
    excellent = float(threshold["excellent"])
    poor = float(threshold["poor"])

    if threshold["direction"] == "inverse":
        if poor <= excellent:
            poor = excellent + 1.0
        raw_score = 100.0 * (poor - value) / (poor - excellent)
    else:
        if excellent <= poor:
            excellent = poor + 1.0
        raw_score = 100.0 * (value - poor) / (excellent - poor)

    return max(0.0, min(100.0, raw_score))


def summarize_reaction(trials: list[dict]) -> dict:
    valid_rts = [
        float(trial["response_ms"])
        for trial in trials
        if trial.get("response_ms") is not None and not trial.get("false_start") and not trial.get("missed")
    ]
    false_starts = sum(1 for trial in trials if trial.get("false_start"))
    misses = sum(1 for trial in trials if trial.get("missed"))
    return {
        "trial_count": len(trials),
        "valid_count": len(valid_rts),
        "mean_rt_ms": _mean(valid_rts),
        "rt_std_ms": _std(valid_rts),
        "false_start_rate": _rate(false_starts, len(trials)),
        "miss_rate": _rate(misses, len(trials)),
    }


def summarize_switching(trials: list[dict]) -> dict:
    switch_trials = [trial for trial in trials if trial.get("is_switch")]
    repeat_trials = [trial for trial in trials if not trial.get("is_switch")]
    responded_switch_rts = [
        float(trial["response_ms"])
        for trial in switch_trials
        if trial.get("response_ms") is not None and not trial.get("missed")
    ]
    responded_repeat_rts = [
        float(trial["response_ms"])
        for trial in repeat_trials
        if trial.get("response_ms") is not None and not trial.get("missed")
    ]
    correct_switch_rts = [float(trial["response_ms"]) for trial in switch_trials if trial.get("correct") and trial.get("response_ms")]
    correct_repeat_rts = [float(trial["response_ms"]) for trial in repeat_trials if trial.get("correct") and trial.get("response_ms")]

    switch_rt = _mean(responded_switch_rts)
    repeat_rt = _mean(responded_repeat_rts)
    accuracy = _rate(sum(1 for trial in trials if trial.get("correct")), len(trials))
    switch_accuracy = _rate(sum(1 for trial in switch_trials if trial.get("correct")), len(switch_trials))
    repeat_accuracy = _rate(sum(1 for trial in repeat_trials if trial.get("correct")), len(repeat_trials))

    return {
        "trial_count": len(trials),
        "mean_rt_ms": _mean(responded_switch_rts + responded_repeat_rts),
        "rt_std_ms": _std(responded_switch_rts + responded_repeat_rts),
        "accuracy": accuracy,
        "switch_accuracy": switch_accuracy,
        "repeat_accuracy": repeat_accuracy,
        "switch_cost_ms": max(0.0, switch_rt - repeat_rt) if switch_rt and repeat_rt else switch_rt,
    }


def summarize_working_memory(trials: list[dict]) -> dict:
    position_accuracies = [float(trial["position_accuracy"]) for trial in trials]
    full_correct_trials = [trial for trial in trials if trial.get("full_correct")]
    response_times = [float(trial["response_ms"]) for trial in trials if trial.get("response_ms") is not None]
    max_span = max((int(trial["sequence_length"]) for trial in full_correct_trials), default=0)

    return {
        "trial_count": len(trials),
        "full_recall_rate": _rate(len(full_correct_trials), len(trials)),
        "position_accuracy": _mean(position_accuracies),
        "max_span": float(max_span),
        "mean_response_ms": _mean(response_times),
    }


def summarize_stroop(trials: list[dict]) -> dict:
    congruent_trials = [trial for trial in trials if trial.get("congruent")]
    incongruent_trials = [trial for trial in trials if not trial.get("congruent")]

    congruent_response_rts = [
        float(trial["response_ms"])
        for trial in congruent_trials
        if trial.get("response_ms") is not None and not trial.get("missed")
    ]
    incongruent_response_rts = [
        float(trial["response_ms"])
        for trial in incongruent_trials
        if trial.get("response_ms") is not None and not trial.get("missed")
    ]
    congruent_rts = [float(trial["response_ms"]) for trial in congruent_trials if trial.get("correct") and trial.get("response_ms")]
    incongruent_rts = [float(trial["response_ms"]) for trial in incongruent_trials if trial.get("correct") and trial.get("response_ms")]
    all_response_rts = congruent_response_rts + incongruent_response_rts

    return {
        "trial_count": len(trials),
        "accuracy": _rate(sum(1 for trial in trials if trial.get("correct")), len(trials)),
        "incongruent_accuracy": _rate(
            sum(1 for trial in incongruent_trials if trial.get("correct")),
            len(incongruent_trials),
        ),
        "mean_rt_ms": _mean(all_response_rts),
        "rt_std_ms": _std(all_response_rts),
        "interference_ms": max(0.0, _mean(incongruent_rts) - _mean(congruent_rts)),
    }


def build_task_metrics(session: dict) -> dict:
    tasks = session["tasks"]
    return {
        "reaction_time": summarize_reaction(tasks["reaction_time"]["trials"]),
        "task_switching": summarize_switching(tasks["task_switching"]["trials"]),
        "working_memory": summarize_working_memory(tasks["working_memory"]["trials"]),
        "stroop": summarize_stroop(tasks["stroop"]["trials"]),
    }


def build_features(session: dict, task_metrics: dict) -> dict:
    timed_trials = []
    error_trials = []
    for task_name, task_payload in session["tasks"].items():
        for trial in task_payload["trials"]:
            timed_trials.append(trial)
            if task_name == "working_memory":
                error_trials.append(0 if trial.get("full_correct") else 1)
            else:
                error_trials.append(0 if trial.get("correct", True) else 1)

    pauses = sum(
        1
        for trial in timed_trials
        if (trial.get("response_ms") is not None and float(trial["response_ms"]) > _pause_threshold_ms(trial))
        or trial.get("missed")
    )

    features = {
        "reaction_mean_rt_ms": task_metrics["reaction_time"]["mean_rt_ms"],
        "reaction_rt_std_ms": task_metrics["reaction_time"]["rt_std_ms"],
        "reaction_false_start_rate": task_metrics["reaction_time"]["false_start_rate"],
        "switch_cost_ms": task_metrics["task_switching"]["switch_cost_ms"],
        "switch_mean_rt_ms": task_metrics["task_switching"]["mean_rt_ms"],
        "switch_accuracy": task_metrics["task_switching"]["switch_accuracy"],
        "repeat_accuracy": task_metrics["task_switching"]["repeat_accuracy"],
        "working_memory_full_recall_rate": task_metrics["working_memory"]["full_recall_rate"],
        "working_memory_max_span": task_metrics["working_memory"]["max_span"],
        "working_memory_position_accuracy": task_metrics["working_memory"]["position_accuracy"],
        "working_memory_mean_response_ms": task_metrics["working_memory"]["mean_response_ms"],
        "stroop_interference_ms": task_metrics["stroop"]["interference_ms"],
        "stroop_mean_rt_ms": task_metrics["stroop"]["mean_rt_ms"],
        "stroop_incongruent_accuracy": task_metrics["stroop"]["incongruent_accuracy"],
        "stroop_rt_std_ms": task_metrics["stroop"]["rt_std_ms"],
        "pause_rate": _rate(pauses, len(timed_trials)),
        "overall_error_rate": _rate(sum(error_trials), len(error_trials)),
    }
    return features


def build_task_scores(features: dict, baseline_profile: dict | None = None) -> dict:
    scores = {}
    for task_name, weighted_features in TASK_SCORE_FEATURES.items():
        task_score = 0.0
        for feature_name, weight in weighted_features:
            task_score += score_feature(feature_name, float(features[feature_name]), baseline_profile) * weight
        scores[task_name] = round(task_score, 2)
    return scores


def band_for_score(score: float) -> dict:
    for band in SCORE_BANDS:
        if band["min_score"] <= score <= band["max_score"]:
            return band
    return SCORE_BANDS[-1]


def derive_bands(overall_score: float) -> tuple[str, str]:
    band = band_for_score(overall_score)
    return band["label"], band["performance"]


def build_validation_summary(overall_score: float, task_scores: dict) -> dict:
    component_scores = {"overall_score": overall_score, **task_scores}
    validation_scores = {label: 0.0 for label in DEFAULT_LOAD_CLASSES}
    component_bands = {}

    for component_name, score in component_scores.items():
        band = band_for_score(float(score))
        component_bands[component_name] = {
            "score": round(float(score), 2),
            "band": band["label"],
            "weight": round(float(VALIDATION_COMPONENT_WEIGHTS[component_name]), 4),
        }
        validation_scores[band["label"]] += float(VALIDATION_COMPONENT_WEIGHTS[component_name])

    primary_band = band_for_score(overall_score)["label"]
    predicted_band = primary_band
    best_support = validation_scores[primary_band]
    for label in DEFAULT_LOAD_CLASSES:
        if validation_scores[label] > best_support:
            predicted_band = label
            best_support = validation_scores[label]

    return {
        "predicted_band": predicted_band,
        "confidence": round(best_support, 4),
        "scores": {label: round(score, 4) for label, score in validation_scores.items()},
        "components": component_bands,
    }


def build_session_summary(session: dict, baseline_profile: dict | None = None) -> dict:
    task_metrics = build_task_metrics(session)
    features = build_features(session, task_metrics)
    task_scores = build_task_scores(features, baseline_profile)

    pause_score = score_feature("pause_rate", features["pause_rate"], baseline_profile)
    error_score = score_feature("overall_error_rate", features["overall_error_rate"], baseline_profile)
    overall_score = (
        sum(task_scores[task_name] * TASK_SCORE_WEIGHTS[task_name] for task_name in TASK_SCORE_WEIGHTS)
        * 0.80
        + pause_score * 0.10
        + error_score * 0.10
    )
    overall_score = round(overall_score, 2)

    rule_load_band, performance_band = derive_bands(overall_score)
    validation_summary = build_validation_summary(overall_score, task_scores)

    return {
        "task_metrics": task_metrics,
        "features": {name: round(float(features[name]), 4) for name in FEATURE_COLUMNS},
        "task_scores": task_scores,
        "overall_score": overall_score,
        "rule_load_band": rule_load_band,
        "range_validation_band": validation_summary["predicted_band"],
        "final_load_band": validation_summary["predicted_band"],
        "performance_band": performance_band,
        "prediction_method": "range_validation",
        "validation_confidence": validation_summary["confidence"],
        "prediction_confidence": validation_summary["confidence"],
        "validation_scores": validation_summary["scores"],
        "validation_components": validation_summary["components"],
        "model_confidence": None,
    }


def apply_model_prediction(summary: dict, prediction: dict) -> dict:
    summary = dict(summary)
    summary["final_load_band"] = prediction["label"]
    summary["prediction_method"] = "trained_model"
    summary["prediction_confidence"] = round(float(prediction["confidence"]), 4)
    summary["model_confidence"] = round(float(prediction["confidence"]), 4)
    summary["model_probabilities"] = {
        label: round(float(probability), 4)
        for label, probability in prediction["probabilities"].items()
    }
    return summary


def build_summary_row(session: dict, summary: dict) -> dict:
    questionnaire = session.get("questionnaire", {})
    row = {
        "session_id": session["session_id"],
        "participant_id": session["participant_id"],
        "started_at": session["started_at"],
        "finished_at": session["finished_at"],
        "duration_seconds": session["duration_seconds"],
        "baseline_session": session.get("baseline_session", False),
        "sleep_hours": questionnaire.get("sleep_hours", ""),
        "stress_level": questionnaire.get("stress_level", ""),
        "caffeine_mg": questionnaire.get("caffeine_mg", ""),
        "fatigue_level": questionnaire.get("fatigue_level", ""),
        "notes": questionnaire.get("notes", ""),
        "self_report_load_band": session.get("self_report_load_band", ""),
        "rule_load_band": summary["rule_load_band"],
        "range_validation_band": summary.get("range_validation_band", summary["final_load_band"]),
        "final_load_band": summary["final_load_band"],
        "performance_band": summary["performance_band"],
        "prediction_method": summary["prediction_method"],
        "validation_confidence": summary.get("validation_confidence", ""),
        "prediction_confidence": summary.get("prediction_confidence", ""),
        "overall_score": summary["overall_score"],
        "reaction_score": summary["task_scores"]["reaction_time"],
        "task_switching_score": summary["task_scores"]["task_switching"],
        "working_memory_score": summary["task_scores"]["working_memory"],
        "stroop_score": summary["task_scores"]["stroop"],
        "low_validation_score": summary.get("validation_scores", {}).get("Low", ""),
        "manageable_validation_score": summary.get("validation_scores", {}).get("Manageable", ""),
        "elevated_validation_score": summary.get("validation_scores", {}).get("Elevated", ""),
        "high_validation_score": summary.get("validation_scores", {}).get("High", ""),
        "model_confidence": summary.get("model_confidence", ""),
    }

    row.update(summary["features"])
    return row


def format_report(session: dict, summary: dict) -> str:
    task_lines = []
    for task_name, label in TASK_LABELS.items():
        task_lines.append(
            f"- {label}: score={summary['task_scores'][task_name]:.2f}, metrics={summary['task_metrics'][task_name]}"
        )

    validation_line = (
        f"Range validation prediction={summary.get('range_validation_band', summary['final_load_band'])} "
        f"confidence={summary.get('validation_confidence', '')} "
        f"scores={summary.get('validation_scores', {})}"
    )

    model_line = "No trained model override was used."
    if summary["prediction_method"] == "trained_model":
        probabilities = summary.get("model_probabilities", {})
        model_line = f"Model prediction={summary['final_load_band']} confidence={summary['model_confidence']} probs={probabilities}"

    return "\n".join(
        [
            f"Session ID: {session['session_id']}",
            f"Participant: {session['participant_id']}",
            f"Started: {session['started_at']}",
            f"Finished: {session['finished_at']}",
            f"Duration seconds: {session['duration_seconds']}",
            f"Baseline session: {session.get('baseline_session', False)}",
            f"Rule-based load band: {summary['rule_load_band']}",
            f"Final load band: {summary['final_load_band']}",
            f"Performance band: {summary['performance_band']}",
            f"Overall score: {summary['overall_score']:.2f}",
            validation_line,
            model_line,
            "Task results:",
            *task_lines,
            "Features:",
            *(f"- {name}: {summary['features'][name]}" for name in FEATURE_COLUMNS),
        ]
    )
