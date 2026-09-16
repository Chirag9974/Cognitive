"""Rebuild data/session_summary.csv from the stored session JSONs.

Why this exists: the summary CSV schema grew over time (features such as
switch_mean_rt_ms, working_memory_mean_response_ms and stroop_mean_rt_ms were
added after the earliest sessions were recorded). Old rows therefore lack
columns that the V2 trainer needs. The session JSONs are the source of truth
and still contain the raw trials, so the correct fix is to recompute every row
with the current code — never to guess missing values.

Usage:
    python scripts/backfill_summary.py [--dry-run]

This overwrites data/session_summary.csv. A timestamped backup is written
next to it first.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cognitive_load.analysis import build_session_summary, build_summary_row
from cognitive_load.config import SUMMARY_FIELDNAMES
from cognitive_load.pipeline import load_participant_history, run_v2_pipeline, update_summary_row
from cognitive_load.storage import project_root


def rebuild(root: Path, dry_run: bool = False) -> int:
    sessions_dir = root / "data" / "sessions"
    csv_path = root / "data" / "session_summary.csv"

    session_files = sorted(sessions_dir.glob("*.json"))
    if not session_files:
        print("No session JSONs found; nothing to do.")
        return 1

    rows = []
    errors = 0
    for session_file in session_files:
        session = json.loads(session_file.read_text(encoding="utf-8"))
        # Sessions recorded before the questionnaire existed may lack it.
        session.setdefault("questionnaire", {})
        try:
            summary = build_session_summary(session)
        except Exception as error:  # keep going: one bad file must not stop the rebuild
            print(f"  ! {session_file.name}: could not summarize ({error})")
            errors += 1
            continue

        history = load_participant_history(
            session["participant_id"], root=root, exclude_session_id=session["session_id"]
        )
        payload = run_v2_pipeline(session, summary, history)
        rows.append(update_summary_row(build_summary_row(session, summary), payload))

    rows.sort(key=lambda row: (row.get("started_at") or "", row.get("session_id") or ""))

    print(f"Rebuilt {len(rows)} rows from {len(session_files)} sessions ({errors} unreadable).")
    features_missing = sum(
        1
        for row in rows
        if any(not str(row.get(name, "")).strip() for name in SUMMARY_FIELDNAMES if row.get(name) is not None)
    )
    if dry_run:
        print("Dry run: CSV not written.")
        return 0

    if csv_path.exists():
        backup = csv_path.with_suffix(".csv.bak")
        shutil.copy2(csv_path, backup)
        print(f"Backup written: {backup.name}")

    import csv as csv_module

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv_module.DictWriter(csv_file, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Rewrote {csv_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild the summary CSV from session JSONs.")
    parser.add_argument("--dry-run", action="store_true", help="Report counts without writing the CSV.")
    args = parser.parse_args()
    return rebuild(project_root(), dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
