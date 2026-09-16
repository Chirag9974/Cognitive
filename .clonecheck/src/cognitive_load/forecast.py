"""FR-7 next-session forecasting.

Projects the participant's next overall score from their PRIOR sessions only
(leakage guard: the session being analysed is never part of its own forecast).
The projection is a weighted blend of three simple, explainable statistics:

- last score            (where they are right now)
- EMA of prior scores   (smoothed recent level, ALPHA from config)
- rolling-window mean   (typical recent level)
- trend adjustment      (half-window slope over the rolling window)

The projected score maps onto the V1 score bands, and a probability
distribution over bands is derived from the distance to each band midpoint
(softmax over negative distances). Nothing is claimed beyond that: this is a
projection from a short behavioural history, not a clinical prediction.

History shorter than FORECAST_MIN_SESSIONS produces an explicit
INSUFFICIENT_HISTORY payload (never a guess).
"""
from __future__ import annotations

from .config import (
    EMA_ALPHA,
    FORECAST_MIN_SESSIONS,
    ROLLING_WINDOW,
    SCORE_BANDS,
)
from .temporal import insufficient_history_error, valid_scores

# Blending weights for the projected next score (sum to 1.0).
WEIGHT_LAST = 0.30
WEIGHT_EMA = 0.35
WEIGHT_ROLLING_MEAN = 0.25
WEIGHT_TREND = 0.10

# Softmax temperature in score points: higher = flatter band probabilities.
_BAND_PROBABILITY_TEMPERATURE = 12.0

_BAND_MIDPOINTS = {
    band["label"]: (band["min_score"] + band["max_score"]) / 2.0
    for band in SCORE_BANDS
}


def _ema(values: list[float]) -> float:
    result = values[0]
    for value in values[1:]:
        result = EMA_ALPHA * value + (1 - EMA_ALPHA) * result
    return result


def _trend_change(values: list[float]) -> float:
    """Mean(second half of window) - Mean(first half), like the FR-1 trend."""
    window = values[-ROLLING_WINDOW:]
    half = len(window) // 2
    if half == 0:
        return 0.0
    from statistics import fmean

    return fmean(window[half:]) - fmean(window[:half])


def _band_for_score(score: float) -> str:
    for band in SCORE_BANDS:
        if band["min_score"] <= score <= band["max_score"]:
            return band["label"]
    return SCORE_BANDS[-1]["label"]


def forecast_next_band(history_rows: list[dict]) -> dict:
    """Forecast the next session's load band from prior overall scores.

    Raises InsufficientHistoryError when fewer than FORECAST_MIN_SESSIONS
    valid prior scores exist (explicit guard per the PRD; never guesses).
    """
    prior = valid_scores(history_rows)
    if len(prior) < FORECAST_MIN_SESSIONS:
        raise ValueError(
            insufficient_history_error(FORECAST_MIN_SESSIONS, "next-session forecasting")
        )

    last = prior[-1]
    ema = _ema(prior)
    rolling_mean = sum(prior[-ROLLING_WINDOW:]) / len(prior[-ROLLING_WINDOW:])
    trend = _trend_change(prior)

    projected = (
        WEIGHT_LAST * last
        + WEIGHT_EMA * ema
        + WEIGHT_ROLLING_MEAN * rolling_mean
        + WEIGHT_TREND * (last + trend)
    )
    projected = max(0.0, min(100.0, projected))

    # Distance-to-midpoint softmax over the four bands.
    import math

    scores = {
        label: math.exp(-abs(projected - midpoint) / _BAND_PROBABILITY_TEMPERATURE)
        for label, midpoint in _BAND_MIDPOINTS.items()
    }
    total = sum(scores.values())
    probabilities = {label: value / total for label, value in scores.items()}
    top_band = max(probabilities, key=probabilities.get)

    return {
        "available": True,
        "band": top_band,
        "probability": round(probabilities[top_band], 4),
        "probabilities": {label: round(value, 4) for label, value in probabilities.items()},
        "projected_score": round(projected, 1),
        "method": "weighted_trend_ema_blend",
        "min_sessions": FORECAST_MIN_SESSIONS,
        "inputs": {
            "last_score": round(last, 1),
            "ema": round(ema, 1),
            "rolling_mean": round(rolling_mean, 1),
            "trend_change": round(trend, 1),
            "sessions_used": len(prior),
        },
    }


def build_forecast_payload(history_rows: list[dict]) -> dict:
    """Forecast payload that degrades gracefully: unavailable, with reason."""
    try:
        return forecast_next_band(history_rows)
    except ValueError as error:
        return {
            "available": False,
            "reason": str(error),
            "band": "",
            "probability": "",
            "probabilities": {},
            "projected_score": None,
            "method": "weighted_trend_ema_blend",
            "min_sessions": FORECAST_MIN_SESSIONS,
        }
