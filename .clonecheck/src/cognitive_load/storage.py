from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_DATA_DIRS, SUMMARY_FIELDNAMES, project_root_from


def project_root() -> Path:
    return project_root_from(__file__)


def ensure_project_layout(root: Path | None = None) -> Path:
    root = root or project_root()
    for relative_dir in PROJECT_DATA_DIRS:
        (root / relative_dir).mkdir(parents=True, exist_ok=True)
    return root


def save_session(session: dict, root: Path | None = None) -> Path:
    root = ensure_project_layout(root)
    output_path = root / "data" / "sessions" / f"{session['session_id']}.json"
    output_path.write_text(json.dumps(session, indent=2), encoding="utf-8")
    return output_path


def save_report(session_id: str, report_text: str, root: Path | None = None) -> Path:
    root = ensure_project_layout(root)
    output_path = root / "data" / "reports" / f"{session_id}.txt"
    output_path.write_text(report_text, encoding="utf-8")
    return output_path


def append_summary_row(summary_row: dict, root: Path | None = None) -> Path:
    root = ensure_project_layout(root)
    output_path = root / "data" / "session_summary.csv"
    if output_path.exists():
        with output_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            existing_fieldnames = reader.fieldnames or []
            existing_rows = list(reader)
        if existing_fieldnames != SUMMARY_FIELDNAMES:
            with output_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDNAMES)
                writer.writeheader()
                for existing_row in existing_rows:
                    writer.writerow({field: existing_row.get(field, "") for field in SUMMARY_FIELDNAMES})
    write_header = not output_path.exists()

    row = {field: summary_row.get(field, "") for field in SUMMARY_FIELDNAMES}
    with output_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return output_path


def load_summary_rows(csv_path: Path | None = None, root: Path | None = None) -> list[dict]:
    if csv_path is None:
        root = root or project_root()
        csv_path = root / "data" / "session_summary.csv"
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def default_model_path(root: Path | None = None) -> Path:
    root = root or project_root()
    return root / "data" / "models" / "softmax_predictor.json"


def default_baseline_path(participant_id: str, root: Path | None = None) -> Path:
    root = root or project_root()
    return root / "data" / "baselines" / f"{participant_id}.json"
