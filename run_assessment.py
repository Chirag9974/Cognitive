from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cognitive_load.analysis import apply_model_prediction, build_session_summary, build_summary_row, format_report
from cognitive_load.app import run_interactive_assessment
from cognitive_load.calibration import load_baseline
from cognitive_load.model import SoftmaxLoadModel, normalize_label
from cognitive_load.pipeline import (
    format_v2_report,
    load_participant_history,
    run_v2_pipeline,
    update_summary_row,
)
from cognitive_load.simulation import generate_synthetic_session
from cognitive_load.storage import (
    append_summary_row,
    default_baseline_path,
    default_model_path,
    save_report,
    save_session,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the cognitive load assessment.")
    parser.add_argument("--participant-id", required=True, help="Participant identifier.")
    parser.add_argument(
        "--simulate",
        choices=["low", "manageable", "elevated", "high"],
        help="Generate a synthetic session instead of launching the GUI.",
    )
    parser.add_argument(
        "--baseline-session",
        action="store_true",
        help="Mark this session as a baseline calibration session.",
    )
    parser.add_argument(
        "--skip-questionnaire",
        action="store_true",
        help="Skip the short console questionnaire before the assessment.",
    )
    parser.add_argument(
        "--self-report-load",
        help="Optional self-report label. Accepted values include low/manageable/elevated/high or 1/2/3/4.",
    )
    parser.add_argument(
        "--use-trained-model",
        action="store_true",
        help="Use a saved trained model to override the default range-validation prediction.",
    )
    return parser.parse_args()


def _prompt_float(prompt: str, default: float | None = None) -> float | None:
    try:
        raw_value = input(prompt).strip()
    except EOFError:
        return default
    if not raw_value:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _prompt_int(prompt: str, default: int | None = None) -> int | None:
    try:
        raw_value = input(prompt).strip()
    except EOFError:
        return default
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def collect_questionnaire(skip: bool) -> dict:
    if skip:
        return {}
    print("Pre-task questionnaire")
    return {
        "sleep_hours": _prompt_float("Sleep hours last night [blank to skip]: ", default=None),
        "stress_level": _prompt_int("Stress level 1-5 [blank to skip]: ", default=None),
        "caffeine_mg": _prompt_int("Approx caffeine mg today [blank to skip]: ", default=None),
        "fatigue_level": _prompt_int("Fatigue level 1-5 [blank to skip]: ", default=None),
        "notes": input("Notes/context [blank to skip]: ").strip(),
    }


def prompt_self_report(existing_value: str | None = None) -> str | None:
    if existing_value:
        return normalize_label(existing_value)
    try:
        raw_value = input(
            "Optional self-report load label [low/manageable/elevated/high or 1/2/3/4, blank to skip]: "
        ).strip()
    except EOFError:
        return None
    return normalize_label(raw_value)


def maybe_apply_model(summary: dict, enabled: bool) -> dict:
    if not enabled:
        return summary
    model_path = default_model_path(ROOT)
    if not model_path.exists():
        return summary
    model = SoftmaxLoadModel.load(model_path)
    return apply_model_prediction(summary, model.predict(summary["features"]))


def build_session(args: argparse.Namespace) -> dict | None:
    if args.simulate:
        return generate_synthetic_session(
            participant_id=args.participant_id,
            profile=args.simulate,
            baseline_session=args.baseline_session,
        )
    questionnaire = collect_questionnaire(args.skip_questionnaire)
    return run_interactive_assessment(
        participant_id=args.participant_id,
        questionnaire=questionnaire,
        baseline_session=args.baseline_session,
    )


def main() -> int:
    args = parse_args()
    session = build_session(args)
    if not session:
        print("Assessment cancelled.")
        return 1

    self_report = prompt_self_report(args.self_report_load or session.get("self_report_load_band"))
    if self_report:
        session["self_report_load_band"] = self_report

    baseline_profile = load_baseline(default_baseline_path(session["participant_id"], ROOT))
    summary = build_session_summary(session, baseline_profile=baseline_profile)
    summary = maybe_apply_model(summary, enabled=args.use_trained_model)
    session["summary"] = summary

    # V2 intelligence pass: temporal, difficulty, anomaly/quality, uncertainty,
    # explanation, forecast, recommendations (prior sessions only, no leakage).
    trained_model = None
    if args.use_trained_model:
        model_path = default_model_path(ROOT)
        if model_path.exists():
            trained_model = SoftmaxLoadModel.load(model_path)
    history = load_participant_history(
        session["participant_id"], root=ROOT, exclude_session_id=session["session_id"]
    )
    v2_payload = run_v2_pipeline(
        session, summary, history, baseline_profile=baseline_profile, model=trained_model
    )
    summary_row = update_summary_row(build_summary_row(session, summary), v2_payload)

    session_path = save_session(session, ROOT)
    report_text = format_report(session, summary) + "\n" + "\n".join(format_v2_report(v2_payload))
    report_path = save_report(session["session_id"], report_text, ROOT)
    summary_csv_path = append_summary_row(summary_row, ROOT)

    print(report_text)
    print(f"\nSaved raw session: {session_path}")
    print(f"Saved report: {report_path}")
    print(f"Updated summary CSV: {summary_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
