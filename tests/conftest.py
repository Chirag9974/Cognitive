"""Shared fixtures: realistic session dicts mirroring the app's trial schema.

Key sets match the actual files in data/sessions (stroop/task_switching use
pressed_key, working_memory uses response, reaction_time has no answer key).
"""
from __future__ import annotations

import random

import pytest

from cognitive_load.analysis import build_session_summary

CURRENT_DAY = "2026-09-15"

PROFILES = {
    "low": dict(rt=380, switch_rt=1100, switch_acc=0.97, repeat_acc=0.99, span=7, full_recall=0.95, pos_acc=0.97, stroop_rt=950, stroop_acc=0.98, interference=90),
    "manageable": dict(rt=480, switch_rt=1500, switch_acc=0.92, repeat_acc=0.95, span=6, full_recall=0.75, pos_acc=0.90, stroop_rt=1150, stroop_acc=0.94, interference=160),
    "elevated": dict(rt=640, switch_rt=2200, switch_acc=0.82, repeat_acc=0.88, span=5, full_recall=0.45, pos_acc=0.78, stroop_rt=1500, stroop_acc=0.85, interference=280),
    "high": dict(rt=900, switch_rt=3200, switch_acc=0.65, repeat_acc=0.75, span=4, full_recall=0.20, pos_acc=0.60, stroop_rt=2100, stroop_acc=0.70, interference=420),
}


def _rt_task(rt_mean: float, n: int = 18) -> dict:
    rng = random.Random(42)
    return {
        "trials": [
            {
                "response_ms": round(max(250.0, rng.gauss(rt_mean, 45.0)), 1),
                "false_start": False,
                "missed": False,
                "correct": True,
            }
            for _ in range(n)
        ]
    }


def _switch_task(rt: float, switch_acc: float, repeat_acc: float, n: int = 20) -> dict:
    rng = random.Random(7)
    trials = []
    for i in range(n):
        is_switch = i % 2 == 0
        acc = switch_acc if is_switch else repeat_acc
        task_rt = rt * (1.25 if is_switch else 1.0)
        trials.append(
            {
                "is_switch": is_switch,
                "response_ms": round(max(300.0, rng.gauss(task_rt, 80.0)), 1),
                "correct": rng.random() < acc,
                "missed": False,
                "pressed_key": rng.choice(["z", "m"]),
                "rule": "letter" if i % 3 else "digit",
            }
        )
    return {"trials": trials}


def _wm_task(span: int, full_recall: float, position_acc: float, n: int = 6) -> dict:
    rng = random.Random(11)
    trials = []
    for i in range(n):
        length = 3 + (i % max(1, span - 2))  # ramp spans 3..span like the real app
        full = rng.random() < full_recall
        trials.append(
            {
                "sequence_length": length,
                "full_correct": full,
                "position_accuracy": round(position_acc if full else position_acc * 0.6, 3),
                "response_ms": round(max(800.0, rng.gauss(3500, 500)), 1),
                "response": "".join(rng.choice("123456789") for _ in range(length)),
                "correct": full,
                "missed": False,
                "target_sequence": "".join(rng.choice("123456789") for _ in range(length)),
            }
        )
    return {"trials": trials}


def _stroop_task(rt: float, acc: float, interference_ms: float, n: int = 20) -> dict:
    rng = random.Random(13)
    trials = []
    for _ in range(n):
        congruent = rng.random() < 0.4
        task_rt = rt if congruent else rt * (1 + interference_ms / max(rt, 1.0))
        trials.append(
            {
                "congruent": congruent,
                "response_ms": round(max(300.0, rng.gauss(task_rt, 90.0)), 1),
                "correct": rng.random() < acc,
                "missed": False,
                "pressed_key": rng.choice(["r", "g", "b"]),
                "word": "RED",
                "color_name": "blue",
            }
        )
    return {"trials": trials}


def make_session(
    participant_id: str = "TST01",
    profile: str = "manageable",
    started_at: str | None = None,
    session_id: str | None = None,
) -> dict:
    p = PROFILES[profile]
    started_at = started_at or f"{CURRENT_DAY}T10:00:00"
    return {
        "session_id": session_id or f"{participant_id}-TEST",
        "participant_id": participant_id,
        "started_at": started_at,
        "finished_at": started_at,
        "duration_seconds": 600,
        "baseline_session": False,
        "questionnaire": {},
        "tasks": {
            "reaction_time": _rt_task(p["rt"]),
            "task_switching": _switch_task(p["switch_rt"], p["switch_acc"], p["repeat_acc"]),
            "working_memory": _wm_task(p["span"], p["full_recall"], p["pos_acc"]),
            "stroop": _stroop_task(p["stroop_rt"], p["stroop_acc"], p["interference"]),
        },
    }


def make_summary(session: dict) -> dict:
    return build_session_summary(session)


def _row_from_summary(session: dict, summary: dict) -> dict:
    row = {
        "session_id": session["session_id"],
        "participant_id": session["participant_id"],
        "started_at": session["started_at"],
        "overall_score": summary["overall_score"],
    }
    row.update(summary["features"])
    return row


def make_history(
    scorer,
    count: int,
    participant_id: str = "TST01",
    start_day: int = 1,
) -> list[dict]:
    """CSV-style history rows (features + overall_score), oldest first."""
    rows = []
    for i in range(count):
        session = make_session(
            participant_id=participant_id,
            profile=scorer(i),
            started_at=f"2026-08-{start_day + i:02d}T09:00:00",
            session_id=f"{participant_id}-H{i:02d}",
        )
        rows.append(_row_from_summary(session, make_summary(session)))
    return rows


@pytest.fixture
def session_factory():
    return make_session


@pytest.fixture
def summary_factory():
    return make_summary


@pytest.fixture
def history_factory():
    return make_history
