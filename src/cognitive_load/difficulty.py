"""FR-10 (light) task difficulty normalization.

Records how demanding each session's task configuration was and derives a
difficulty-adjusted error rate so a harder variant is not mistaken for worse
performance. This is an engineered heuristic score, not an IRT/adaptive model.
"""
from __future__ import annotations

from .config import (
    CONGRUENT_RATE_EXPECTED,
    DIFFICULTY_ERROR_MULTIPLIER_K,
    MEMORY_SPAN_MAX,
    MEMORY_SPAN_MIN,
    SWITCH_RATE_EXPECTED,
)

# Expected full-recall rate per sequence length (engineering heuristic,
# linear from generous at MEMORY_SPAN_MIN to demanding at MEMORY_SPAN_MAX).
_RECALL_STEP = 0.60 / max(1, (MEMORY_SPAN_MAX - MEMORY_SPAN_MIN))
_RECALL_CEILING = 0.95
_RECALL_FLOOR = 0.35


def expected_recall_rate(span: int) -> float:
    span = max(MEMORY_SPAN_MIN, min(MEMORY_SPAN_MAX, int(span)))
    return max(_RECALL_FLOOR, min(_RECALL_CEILING, _RECALL_CEILING - (span - MEMORY_SPAN_MIN) * _RECALL_STEP))


def compute_task_difficulty(session: dict) -> dict:
    """Per-task difficulty components in [0, 1] (higher = harder) + overall 0-100 score."""
    tasks = session.get("tasks", {})

    # Working memory: harder when sequences are long relative to typical span.
    wm_trials = tasks.get("working_memory", {}).get("trials", [])
    if wm_trials:
        expected = [expected_recall_rate(int(trial.get("sequence_length", MEMORY_SPAN_MIN))) for trial in wm_trials]
        memory_difficulty = 1.0 - (sum(expected) / len(expected))
    else:
        memory_difficulty = 0.0

    # Task switching: a balanced mix (50/50 switch/repeat) is the canonical
    # configuration; any strong imbalance is treated as harder to compare.
    switch_trials = tasks.get("task_switching", {}).get("trials", [])
    if switch_trials:
        switch_rate = sum(1 for trial in switch_trials if trial.get("is_switch")) / len(switch_trials)
        switching_difficulty = min(1.0, abs(switch_rate - SWITCH_RATE_EXPECTED) * 2.0)
    else:
        switching_difficulty = 0.0

    # Stroop: harder when fewer easy congruent trials are mixed in.
    stroop_trials = tasks.get("stroop", {}).get("trials", [])
    if stroop_trials:
        congruent_rate = sum(1 for trial in stroop_trials if trial.get("congruent")) / len(stroop_trials)
        stroop_difficulty = max(0.0, CONGRUENT_RATE_EXPECTED - congruent_rate) / max(CONGRUENT_RATE_EXPECTED, 1e-9)
        stroop_difficulty = min(1.0, stroop_difficulty)
    else:
        stroop_difficulty = 0.0

    overall = (
        memory_difficulty * 0.45 + switching_difficulty * 0.25 + stroop_difficulty * 0.30
    )

    return {
        "memory": round(memory_difficulty, 4),
        "switching": round(switching_difficulty, 4),
        "stroop": round(stroop_difficulty, 4),
        "score": round(overall * 100.0, 2),
    }


def difficulty_multiplier(difficulty_score: float) -> float:
    """Error-rate multiplier for a session: >1 when harder than neutral, <1 when easier."""
    return max(0.25, 1.0 + DIFFICULTY_ERROR_MULTIPLIER_K * (difficulty_score / 100.0 - 0.5))


def build_difficulty_features(session: dict, features: dict) -> dict:
    """FR-10 feature columns: task_difficulty_score + difficulty_adjusted_error_rate.

    A harder-than-neutral session divides its raw error rate down; an easier
    session multiplies it up, making error rates comparable across configs.
    """
    difficulty = compute_task_difficulty(session)
    raw_error_rate = float(features.get("overall_error_rate", 0.0))
    adjusted = raw_error_rate / difficulty_multiplier(difficulty["score"])
    return {
        "task_difficulty_score": difficulty["score"],
        "difficulty_adjusted_error_rate": round(max(0.0, min(1.0, adjusted)), 4),
        "difficulty_components": difficulty,
    }
