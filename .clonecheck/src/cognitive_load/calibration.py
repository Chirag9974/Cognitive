from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev

from .config import FEATURE_COLUMNS, SCORING_THRESHOLDS


def build_personal_baseline(summary_rows: list[dict], participant_id: str) -> dict:
    if not summary_rows:
        raise ValueError("No baseline rows were provided.")

    profile: dict[str, object] = {
        "participant_id": participant_id,
        "sample_size": len(summary_rows),
        "features": {},
    }

    for feature_name in FEATURE_COLUMNS:
        values = []
        for row in summary_rows:
            raw_value = row.get(feature_name, "")
            if raw_value in ("", None):
                continue
            try:
                values.append(float(raw_value))
            except ValueError:
                continue
        if not values:
            continue
        profile["features"][feature_name] = {
            "mean": mean(values),
            "std": max(pstdev(values), 1e-6),
            "min": min(values),
            "max": max(values),
        }

    return profile


def save_baseline(profile: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return output_path


def load_baseline(path: Path | None) -> dict | None:
    if not path or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_threshold(feature_name: str, baseline_profile: dict | None) -> dict:
    default_threshold = dict(SCORING_THRESHOLDS[feature_name])
    if not baseline_profile:
        return default_threshold

    feature_profiles = baseline_profile.get("features", {})
    if feature_name not in feature_profiles:
        return default_threshold

    feature_stats = feature_profiles[feature_name]
    baseline_mean = float(feature_stats["mean"])
    baseline_std = max(float(feature_stats["std"]), 1e-6)
    default_span = abs(default_threshold["poor"] - default_threshold["excellent"])
    adaptive_span = max(default_span * 0.35, baseline_std * 2.5)

    if default_threshold["direction"] == "inverse":
        return {
            **default_threshold,
            "excellent": baseline_mean,
            "poor": baseline_mean + adaptive_span,
        }

    poor_value = max(0.0, baseline_mean - adaptive_span)
    return {
        **default_threshold,
        "excellent": baseline_mean,
        "poor": poor_value,
    }
