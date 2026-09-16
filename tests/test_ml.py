import json

import numpy as np
import pytest

from cognitive_load.config import FEATURE_COLUMNS, FEATURE_COLUMNS_V2
from cognitive_load.ml import (
    build_v2_feature_row,
    build_v2_matrix,
    chronological_split,
    compare_models,
    load_v2_predictor,
    save_v2_predictor,
)
from cognitive_load.analysis import build_session_summary
from cognitive_load.simulation import generate_synthetic_session


def _make_rows(n_blocks: int = 4, per_block: int = 6) -> list[dict]:
    """Synthetic labeled rows with rotating bands (all four classes present)."""
    rotation = ["low", "manageable", "manageable", "elevated", "elevated", "high"]
    rows = []
    for block in range(n_blocks):
        for j in range(per_block):
            profile = rotation[(block + j) % len(rotation)]
            session = generate_synthetic_session(f"SYN{block}", profile=profile, seed=block * 10 + j)
            session["started_at"] = f"2026-08-{block + 1:02d}T{9 + j:02d}:00:00"
            session["session_id"] = f"SYN{block}-{j:02d}"
            summary = build_session_summary(session)
            rows.append(
                {
                    "session_id": session["session_id"],
                    "participant_id": f"SYN{block}",
                    "started_at": session["started_at"],
                    "self_report_load_band": session["self_report_load_band"],
                    "overall_score": summary["overall_score"],
                    **summary["features"],
                }
            )
    return rows


# chronological split ---------------------------------------------------------


def test_chronological_split_preserves_time_order():
    rows = _make_rows()
    train, test = chronological_split(rows)
    assert train[-1]["started_at"] <= test[0]["started_at"]
    assert len(train) + len(test) == len(rows)


def test_chronological_split_does_not_shuffle():
    rows = _make_rows(1, 10)
    train, test = chronological_split(rows)
    assert train == rows[: len(train)]
    assert test == rows[len(train) :]


# V2 feature assembly ----------------------------------------------------------


def test_build_v2_feature_row_skips_rows_missing_base_features():
    assert build_v2_feature_row({"session_id": "x"}) is None


def test_build_v2_feature_row_fills_neutral_extras():
    rows = _make_rows(1, 1)
    vector = build_v2_feature_row(rows[0])
    for name in FEATURE_COLUMNS_V2:
        assert name in vector
    assert vector["temporal_session_count"] == 0.0  # neutral fallback
    assert vector["task_difficulty_score"] == 0.0


def test_build_v2_matrix_filters_unlabeled_rows():
    rows = _make_rows(2, 3)
    rows[0]["self_report_load_band"] = ""
    vectors, labels = build_v2_matrix(rows)
    assert len(vectors) == len(labels) == len(rows) - 1
    assert all(name in vectors[0] for name in FEATURE_COLUMNS_V2)


# comparison harness -------------------------------------------------------------


def test_compare_models_ranks_by_accuracy_and_returns_winner():
    comparison = compare_models(_make_rows())
    assert {r["model_name"] for r in comparison["results"]} >= {"logreg", "random_forest", "hist_gb", "softmax"}
    accuracies = [r["accuracy"] for r in comparison["results"]]
    assert accuracies == sorted(accuracies, reverse=True)
    assert comparison["winner"] == comparison["results"][0]["model_name"]
    assert comparison["split"]["method"] == "chronological"


def test_compare_models_rejects_single_class_training_split():
    rows = _make_rows()
    for row in rows:
        row["self_report_load_band"] = "Low"  # collapse to one class
    with pytest.raises(ValueError, match="fewer than two classes"):
        compare_models(rows)


def test_softmax_benchmark_uses_normalized_labels_no_warning():
    # self_report_load_band may be lowercase; the benchmark must normalize
    # before computing log_loss (sklearn assumes lexicographic label order).
    import warnings

    rows = _make_rows()
    for row in rows:
        row["self_report_load_band"] = row["self_report_load_band"].lower()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        comparison = compare_models(rows)
    softmax = next(r for r in comparison["results"] if r["model_name"] == "softmax")
    assert softmax["log_loss"] < 100.0  # misaligned columns would explode


# persistence ---------------------------------------------------------------------


def test_save_and_load_v2_predictor_roundtrip(tmp_path):
    comparison = compare_models(_make_rows())
    name = comparison["winner"]
    artifact_path, registry_path = save_v2_predictor(comparison, name, tmp_path)

    assert artifact_path.exists() and artifact_path.suffix == ".joblib"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["entries"], "registry must contain an entry"
    entry = registry["entries"][-1]
    assert entry["model_name"] == name
    assert entry["model_version"] == "2.0.0"
    assert entry["feature_schema_version"] == "2.0"

    predictor = load_v2_predictor(artifact_path)
    assert predictor.model_name == name
    assert predictor.feature_names == FEATURE_COLUMNS_V2

    rows = _make_rows(1, 2)
    vector = build_v2_feature_row(rows[0])
    probabilities = predictor.predict_proba(vector)
    assert set(probabilities) == set(predictor.classes)
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_v2_predictor_ablation_contributions_are_deterministic(tmp_path):
    comparison = compare_models(_make_rows())
    artifact_path, _ = save_v2_predictor(comparison, comparison["winner"], tmp_path)
    predictor = load_v2_predictor(artifact_path)
    vector = build_v2_feature_row(_make_rows(1, 1)[0])
    first = predictor.feature_contributions(vector)
    second = predictor.feature_contributions(vector)
    assert first == second
    if first:
        assert all(-1.0 <= item["delta"] <= 1.0 for item in first)


def test_registry_appends_not_overwrites(tmp_path):
    comparison = compare_models(_make_rows())
    save_v2_predictor(comparison, comparison["winner"], tmp_path)
    second = save_v2_predictor(comparison, "logreg", tmp_path)
    registry = json.loads(second[1].read_text(encoding="utf-8"))
    assert len(registry["entries"]) == 2
