from __future__ import annotations

import random
from datetime import datetime, timedelta

from .model import normalize_label


PROFILE_MAP = {
    "low": {
        "reaction_mean": 290,
        "reaction_std": 35,
        "false_start_rate": 0.03,
        "switch_accuracy": 0.95,
        "switch_cost": 65,
        "repeat_accuracy": 0.98,
        "memory_full": 0.82,
        "memory_pos": 0.93,
        "memory_span": 6,
        "stroop_acc": 0.95,
        "stroop_cost": 55,
        "stroop_std": 55,
    },
    "manageable": {
        "reaction_mean": 410,
        "reaction_std": 65,
        "false_start_rate": 0.07,
        "switch_accuracy": 0.88,
        "switch_cost": 130,
        "repeat_accuracy": 0.93,
        "memory_full": 0.64,
        "memory_pos": 0.80,
        "memory_span": 5,
        "stroop_acc": 0.87,
        "stroop_cost": 105,
        "stroop_std": 75,
    },
    "elevated": {
        "reaction_mean": 560,
        "reaction_std": 95,
        "false_start_rate": 0.13,
        "switch_accuracy": 0.76,
        "switch_cost": 205,
        "repeat_accuracy": 0.84,
        "memory_full": 0.43,
        "memory_pos": 0.68,
        "memory_span": 4,
        "stroop_acc": 0.77,
        "stroop_cost": 175,
        "stroop_std": 110,
    },
    "high": {
        "reaction_mean": 760,
        "reaction_std": 140,
        "false_start_rate": 0.20,
        "switch_accuracy": 0.58,
        "switch_cost": 320,
        "repeat_accuracy": 0.68,
        "memory_full": 0.22,
        "memory_pos": 0.51,
        "memory_span": 3,
        "stroop_acc": 0.58,
        "stroop_cost": 285,
        "stroop_std": 145,
    },
}


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _sample_rt(mean_ms: float, std_ms: float, minimum: float = 120.0) -> float:
    return round(_clamp(random.gauss(mean_ms, std_ms), minimum, mean_ms + std_ms * 4), 2)


def generate_synthetic_session(
    participant_id: str,
    profile: str = "manageable",
    baseline_session: bool = False,
    seed: int | None = None,
) -> dict:
    if seed is not None:
        random.seed(seed)
    if profile not in PROFILE_MAP:
        raise ValueError(f"Unknown simulation profile: {profile}")

    params = PROFILE_MAP[profile]
    started_at = datetime.now()

    reaction_trials = []
    for _ in range(8):
        false_start = random.random() < params["false_start_rate"]
        rt_value = None if false_start else _sample_rt(params["reaction_mean"], params["reaction_std"])
        reaction_trials.append(
            {
                "false_start": false_start,
                "missed": False,
                "response_ms": rt_value,
                "response_seconds": round(rt_value / 1000.0, 4) if rt_value is not None else None,
                "response_window_ms": 60000,
                "correct": not false_start,
            }
        )

    switching_trials = []
    previous_rule = "number"
    for _ in range(24):
        is_switch = random.random() < 0.5
        rule = "letter" if previous_rule == "number" and is_switch else "number" if is_switch else previous_rule
        previous_rule = rule
        correct = random.random() < (params["switch_accuracy"] if is_switch else params["repeat_accuracy"])
        base_rt = params["reaction_mean"] + (params["switch_cost"] if is_switch else 0)
        switching_trials.append(
            {
                "is_switch": is_switch,
                "correct": correct,
                "missed": False,
                "response_ms": _sample_rt(base_rt, params["reaction_std"] * 1.1),
                "response_window_ms": 60000,
            }
        )
        switching_trials[-1]["response_seconds"] = round(switching_trials[-1]["response_ms"] / 1000.0, 4)

    working_memory_trials = []
    for sequence_length in [3, 3, 4, 4, 5, 5, 6, 6, 7, 7]:
        full_correct = random.random() < params["memory_full"] * max(0.45, 1.1 - (sequence_length - 3) * 0.1)
        position_accuracy = _clamp(random.gauss(params["memory_pos"], 0.08), 0.0, 1.0)
        if full_correct:
            position_accuracy = max(position_accuracy, 1.0)
        working_memory_trials.append(
            {
                "sequence_length": sequence_length,
                "full_correct": full_correct,
                "position_accuracy": position_accuracy,
                "response_ms": _sample_rt(1400 + sequence_length * 260, 350, minimum=500),
                "response_window_ms": 60000,
                "correct": full_correct,
            }
        )
        working_memory_trials[-1]["response_seconds"] = round(working_memory_trials[-1]["response_ms"] / 1000.0, 4)

    stroop_trials = []
    for _ in range(28):
        congruent = random.random() < 0.4
        correct = random.random() < params["stroop_acc"]
        mean_rt = params["reaction_mean"] + (0 if congruent else params["stroop_cost"])
        stroop_trials.append(
            {
                "congruent": congruent,
                "correct": correct,
                "missed": False,
                "response_ms": _sample_rt(mean_rt, params["stroop_std"]),
                "response_window_ms": 60000,
            }
        )
        stroop_trials[-1]["response_seconds"] = round(stroop_trials[-1]["response_ms"] / 1000.0, 4)

    finished_at = started_at + timedelta(minutes=6, seconds=random.randint(5, 45))
    session_id = f"{participant_id}-{started_at.strftime('%Y%m%d-%H%M%S')}-{profile}"

    session = {
        "session_id": session_id,
        "participant_id": participant_id,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "duration_seconds": int((finished_at - started_at).total_seconds()),
        "baseline_session": baseline_session,
        "questionnaire": {
            "sleep_hours": round(random.uniform(5.5, 8.0), 1),
            "stress_level": random.randint(1, 5),
            "caffeine_mg": random.choice([0, 50, 100, 150]),
            "fatigue_level": random.randint(1, 5),
            "notes": f"synthetic profile={profile}",
        },
        "self_report_load_band": normalize_label(profile),
        "tasks": {
            "reaction_time": {"trials": reaction_trials},
            "task_switching": {"trials": switching_trials},
            "working_memory": {"trials": working_memory_trials},
            "stroop": {"trials": stroop_trials},
        },
    }
    return session
