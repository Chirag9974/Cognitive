"""FR-1 temporal cognitive modeling.

Derives per-participant temporal features from prior sessions only.
Every function that takes history treats it as *prior* sessions: the current
session's score is never mixed into the statistics it is compared against
(leakage guard).
"""
from __future__ import annotations

from statistics import fmean, pstdev

from .config import (
    EMA_ALPHA,
    ROLLING_WINDOW,
    TEMPORAL_MIN_SESSIONS,
    TEMPORAL_TREND_THRESHOLD,
)

SCORE_KEY = "overall_score"


def sort_history(rows: list[dict]) -> list[dict]:
    """Oldest -> newest by (started_at, session_id). Never mutates the input."""
    return sorted(rows, key=lambda row: (row.get("started_at") or "", row.get("session_id") or ""))


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN guard
        return None
    return number


def valid_scores(rows: list[dict]) -> list[float]:
    """Overall scores in chronological order, skipping malformed/missing ones."""
    return [s for row in sort_history(rows) if (s := _finite(row.get(SCORE_KEY))) is not None]


def history_size(rows: list[dict]) -> int:
    return len(valid_scores(rows))


def has_min_history(rows: list[dict], min_sessions: int = TEMPORAL_MIN_SESSIONS) -> bool:
    return history_size(rows) >= min_sessions


def temporal_feature_set(rows: list[dict], current_score: float) -> dict:
    """All FR-1 features for the current session, using prior sessions only.

    Returns None for features that lack enough history rather than guessing.
    """
    prior = valid_scores(rows)

    features: dict[str, float | int | None] = {
        "temporal_overall_score_delta": None,
        "temporal_overall_score_delta_pct": None,
        "temporal_overall_score_rolling_mean": fmean(prior[-ROLLING_WINDOW:]) if prior else None,
        "temporal_overall_score_rolling_std": pstdev(prior[-ROLLING_WINDOW:]) if len(prior) >= 2 else None,
        "temporal_overall_score_ema": None,
        "temporal_overall_score_trend": None,
        "temporal_session_count": len(prior),
    }

    if prior:
        last = prior[-1]
        features["temporal_overall_score_delta"] = current_score - last
        if last != 0:
            features["temporal_overall_score_delta_pct"] = 100.0 * (current_score - last) / abs(last)

    if len(prior) >= 2:
        ema = prior[0]
        for value in prior[1:]:
            ema = EMA_ALPHA * value + (1 - EMA_ALPHA) * ema
        features["temporal_overall_score_ema"] = ema

    if len(prior) >= TEMPORAL_MIN_SESSIONS:
        window = prior[-ROLLING_WINDOW:]
        half = len(window) // 2
        if half:
            first_half = fmean(window[:half])
            second_half = fmean(window[half:])
            # Performance change over the window (positive = scores improving).
            features["temporal_overall_score_trend"] = second_half - first_half

    return features


def classify_trend(
    rows: list[dict],
    threshold: float = TEMPORAL_TREND_THRESHOLD,
    min_sessions: int = TEMPORAL_MIN_SESSIONS,
) -> dict:
    """Direction of the *load* trend over the last ROLLING_WINDOW prior sessions.

    direction is expressed in load terms: falling scores mean rising load.
    """
    prior = valid_scores(rows)
    if len(prior) < min_sessions:
        return {
            "direction": "insufficient_history",
            "score_change": None,
            "sessions_used": len(prior),
            "min_sessions": min_sessions,
        }

    window = prior[-ROLLING_WINDOW:]
    direction = "stable"
    score_change = None

    if len(window) >= 2:
        half = len(window) // 2
        score_change = fmean(window[half:]) - fmean(window[:half])
        if score_change < -threshold:
            direction = "rising"    # scores falling -> load rising
        elif score_change > threshold:
            direction = "falling"   # scores improving -> load falling

    return {
        "direction": direction,
        "score_change": round(score_change, 2) if score_change is not None else None,
        "sessions_used": len(prior),
        "window": window,
    }


def insufficient_history_error(min_sessions: int, operation: str) -> str:
    return (
        f"INSUFFICIENT_HISTORY: {operation} needs at least {min_sessions} valid prior sessions."
    )
