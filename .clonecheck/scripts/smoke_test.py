from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cognitive_load.analysis import build_session_summary, build_summary_row
from cognitive_load.model import train_softmax_model
from cognitive_load.simulation import generate_synthetic_session


def main() -> int:
    rows = []
    for profile in ["low", "manageable", "elevated", "high"]:
        for index in range(8):
            session = generate_synthetic_session(
                participant_id=f"SIM-{profile.upper()}",
                profile=profile,
                seed=index,
            )
            summary = build_session_summary(session)
            rows.append(build_summary_row(session, summary))

    model, metrics = train_softmax_model(rows, label_column="self_report_load_band", epochs=2200)

    probe = generate_synthetic_session(participant_id="PROBE", profile="elevated", seed=999)
    probe_summary = build_session_summary(probe)
    prediction = model.predict(probe_summary["features"])

    assert metrics["rows_used"] == 32
    assert metrics["train_accuracy"] >= 0.70
    assert prediction["label"] in metrics["classes"]

    print("Smoke test passed.")
    print(f"Training accuracy: {metrics['train_accuracy']}")
    print(f"Probe prediction: {prediction['label']} with confidence {prediction['confidence']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
