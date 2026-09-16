from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cognitive_load.ml import compare_models, enrich_rows_with_temporal, save_v2_predictor
from cognitive_load.model import SoftmaxLoadModel, train_softmax_model
from cognitive_load.storage import default_model_path, load_summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the cognitive load classifier.")
    parser.add_argument(
        "--label-column",
        default="self_report_load_band",
        help="Summary CSV column containing labels.",
    )
    parser.add_argument(
        "--participant-id",
        help="Optional participant filter. Leave empty to use all rows.",
    )
    parser.add_argument("--epochs", type=int, default=2500, help="Gradient descent epochs.")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="V2 mode: compare sklearn candidates against the softmax benchmark on a chronological holdout, and persist the winner.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.7,
        help="Chronological train fraction for --compare (default 0.7).",
    )
    return parser.parse_args()


def run_softmax_training(rows: list[dict], args: argparse.Namespace) -> int:
    model, metrics = train_softmax_model(rows, label_column=args.label_column, epochs=args.epochs)
    output_path = model.save(default_model_path(ROOT))

    print("Training complete.")
    print(f"Rows used: {metrics['rows_used']}")
    print(f"Classes: {', '.join(metrics['classes'])}")
    print(f"Training accuracy: {metrics['train_accuracy']}")
    print(f"Saved model: {output_path}")
    return 0


def run_v2_comparison(rows: list[dict], args: argparse.Namespace) -> int:
    rows = enrich_rows_with_temporal(rows)
    comparison = compare_models(
        rows,
        label_column=args.label_column,
        train_fraction=args.train_fraction,
    )
    split = comparison["split"]
    print(f"Chronological split: {split['n_train']} train (oldest) / {split['n_test']} test (newest)")
    print("\n=== Model comparison (leakage-free holdout) ===")
    for result in comparison["results"]:
        print(
            f"  {result['model_name']:14} acc={result['accuracy']:.3f}  "
            f"log_loss={result['log_loss']:.3f}  (n_train={result['n_train']}, n_test={result['n_test']})"
        )
    winner = comparison["winner"]
    print(f"Winner: {winner}")

    if winner == "softmax":
        model, metrics = train_softmax_model(rows, label_column=args.label_column, epochs=args.epochs)
        output_path = model.save(default_model_path(ROOT))
        print(f"Saved softmax model: {output_path}")
    else:
        artifact_path, registry_path = save_v2_predictor(comparison, winner, ROOT / "data" / "models")
        print(f"Saved {winner} artifact: {artifact_path}")
        print(f"Updated registry: {registry_path}")
    return 0


def main() -> int:
    args = parse_args()
    rows = load_summary_rows(root=ROOT)
    if args.participant_id:
        rows = [row for row in rows if row.get("participant_id") == args.participant_id]

    try:
        if args.compare:
            return run_v2_comparison(rows, args)
        return run_softmax_training(rows, args)
    except ValueError as error:
        print(f"Training aborted: {error}")
        print(
            "Hint: training needs labeled rows (self_report_load_band) covering at least two "
            "bands. Run more sessions with --self-report-load, or check data/session_summary.csv."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
