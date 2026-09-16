from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cognitive_load.calibration import build_personal_baseline, save_baseline
from cognitive_load.storage import default_baseline_path, load_summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a participant-specific baseline profile.")
    parser.add_argument("--participant-id", required=True, help="Participant identifier.")
    return parser.parse_args()


def is_truthy(value: str | bool | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def main() -> int:
    args = parse_args()
    rows = [
        row
        for row in load_summary_rows(root=ROOT)
        if row.get("participant_id") == args.participant_id and is_truthy(row.get("baseline_session"))
    ]
    if not rows:
        raise SystemExit("No baseline rows found for that participant.")

    baseline_profile = build_personal_baseline(rows, args.participant_id)
    output_path = save_baseline(baseline_profile, default_baseline_path(args.participant_id, ROOT))
    print(f"Built baseline from {baseline_profile['sample_size']} sessions.")
    print(f"Saved baseline: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
