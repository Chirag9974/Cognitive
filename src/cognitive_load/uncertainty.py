"""FR-3 uncertainty estimation.

Turns model output + evidence availability into an explicit confidence
statement (High/Moderate/Low), a 0-1 confidence score, and the drivers behind
it — so "0.36 vs 0.92" are never presented the same way again.
"""
from __future__ import annotations

from .config import (
    UNCERTAINTY_HIGH_MARGIN,
    UNCERTAINTY_MAX_VARIANCE_Z,
    UNCERTAINTY_MODERATE_MARGIN,
)

MARGIN_WEIGHT = 0.55
DATA_WEIGHT = 0.25
CONSISTENCY_WEIGHT = 0.20
MIN_SESSIONS_FULL_DATA_CREDIT = 5


def probability_margin(probabilities: dict[str, float]) -> tuple[str, float, float]:
    """Return (top_label, top_probability, top1-top2 margin)."""
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        return "", 0.0, 0.0
    top_label, top_probability = ranked[0]
    margin = top_probability - (ranked[1][1] if len(ranked) > 1 else 0.0)
    return top_label, top_probability, margin


def margin_confidence(margin: float) -> float:
    """Map a top1-top2 margin onto a 0-1 confidence factor."""
    if margin >= UNCERTAINTY_HIGH_MARGIN:
        return 1.0
    if margin <= 0.05:
        return 0.05
    # Linear between the Low and High anchor points.
    return 0.05 + (margin - 0.05) * (0.95 / (UNCERTAINTY_HIGH_MARGIN - 0.05))


def data_volume_confidence(session_count: int) -> float:
    """0.3 with no history up to 1.0 with a full personal history."""
    if session_count <= 0:
        return 0.30
    if session_count >= MIN_SESSIONS_FULL_DATA_CREDIT:
        return 1.0
    return 0.30 + 0.70 * session_count / MIN_SESSIONS_FULL_DATA_CREDIT


def consistency_confidence(max_abs_robust_z: float | None) -> float:
    """Downgrade when the session itself is a behavioral outlier."""
    if max_abs_robust_z is None:
        return 0.85  # no history to compare against: mild, known uncertainty
    if max_abs_robust_z <= UNCERTAINTY_MAX_VARIANCE_Z:
        return 1.0
    # Beyond the tolerated variance, degrade linearly to 0.4 at 3x the limit.
    return max(0.40, 1.0 - (max_abs_robust_z - UNCERTAINTY_MAX_VARIANCE_Z) / (3.0 * UNCERTAINTY_MAX_VARIANCE_Z) * 0.60)


def assess_uncertainty(
    probabilities: dict[str, float],
    session_count: int,
    max_abs_robust_z: float | None = None,
) -> dict:
    """Full FR-3 assessment for one prediction."""
    top_label, top_probability, margin = probability_margin(probabilities)

    m_conf = margin_confidence(margin)
    d_conf = data_volume_confidence(session_count)
    c_conf = consistency_confidence(max_abs_robust_z)
    score = MARGIN_WEIGHT * m_conf + DATA_WEIGHT * d_conf + CONSISTENCY_WEIGHT * c_conf

    if score >= 0.75:
        level = "High"
    elif score >= 0.45:
        level = "Moderate"
    else:
        level = "Low"

    # Honesty cap: with almost no personal history, confidence can never be
    # 'High', however clean the margin looks.
    if session_count < 3 and level == "High":
        level = "Moderate"

    drivers = []
    if margin >= UNCERTAINTY_HIGH_MARGIN:
        drivers.append("the model strongly favours one band over the others")
    elif margin < UNCERTAINTY_MODERATE_MARGIN:
        drivers.append("two bands are close, so the split is not decisive")
    if session_count < MIN_SESSIONS_FULL_DATA_CREDIT:
        drivers.append(
            f"only {session_count} prior session(s) on record"
            if session_count != 1
            else "only 1 prior session on record"
        )
    if max_abs_robust_z is not None and max_abs_robust_z > UNCERTAINTY_MAX_VARIANCE_Z:
        drivers.append("this session's behaviour is unusual compared to recent sessions")

    return {
        "level": level,
        "score": round(score, 3),
        "top_label": top_label,
        "top_probability": round(top_probability, 4),
        "margin": round(margin, 4),
        "components": {
            "margin": round(m_conf, 3),
            "data_volume": round(d_conf, 3),
            "consistency": round(c_conf, 3),
        },
        "drivers": drivers,
        "session_count": session_count,
    }


def summarize_uncertainty(uncertainty: dict) -> str:
    """One-line plain-English confidence statement."""
    level = uncertainty.get("level", "Low")
    top = uncertainty.get("top_label", "")
    drivers = uncertainty.get("drivers", [])
    reason = f" ({'; '.join(drivers)})" if drivers else ""
    return f"{level} confidence that the load band is {top}{reason}"
