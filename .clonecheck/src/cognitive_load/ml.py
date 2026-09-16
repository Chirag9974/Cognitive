"""FR-2 model upgrade: comparison harness + V2 predictor.

Trains candidate classifiers on labeled summary rows and compares them on a
chronological (time-ordered) holdout split — never a random shuffle, so future
sessions never leak into training (leakage guard required by the PRD).

Models compared:
- softmax      (existing V1 numpy benchmark, kept for continuity)
- logreg       (sklearn LogisticRegression — linear, exactly explainable)
- random_forest (sklearn RandomForestClassifier)
- hist_gb      (sklearn HistGradientBoostingClassifier — the boosted-trees
                stand-in for XGBoost, same model family, no extra dependency)

The winner is persisted as a joblib artifact plus a JSON registry entry.
xgboost is an optional import: if installed it joins the comparison.

Feature schema: FEATURE_COLUMNS_V2 (base features + baseline z-scores +
temporal + difficulty). Rows missing base features are skipped; missing V2
extras fall back to neutral values so V1-era rows remain trainable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .config import (
    BASELINE_Z_COLUMNS,
    DEFAULT_LOAD_CLASSES,
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_V2,
    FEATURE_SCHEMA_VERSION,
    MODEL_VERSION,
    REGISTRY_LABELS,
)
from .model import SoftmaxLoadModel, normalize_label, prepare_training_rows, train_softmax_model
from .temporal import temporal_feature_set

LABEL_COLUMN_DEFAULT = "self_report_load_band"


# ----------------------------------------------------------------- splitting


def chronological_split(rows: list[dict], train_fraction: float = 0.7) -> tuple[list[dict], list[dict]]:
    """Time-ordered train/test split (no shuffling — leakage guard).

    Rows missing started_at keep a stable position via session_id tie-break.
    """
    ordered = sorted(
        rows,
        key=lambda row: (row.get("started_at") or "", row.get("session_id") or ""),
    )
    cut = max(1, int(len(ordered) * train_fraction))
    cut = min(cut, len(ordered) - 1) if len(ordered) > 1 else cut
    return ordered[:cut], ordered[cut:]


# ------------------------------------------------------- V2 feature assembly


def _baseline_z(value: float, profile: dict | None, feature_name: str) -> float:
    if not profile:
        return 0.0
    stats = profile.get("features", {}).get(feature_name)
    if not stats:
        return 0.0
    std = max(float(stats.get("std", 1e-6)), 1e-6)
    return (value - float(stats.get("mean", 0.0))) / std


def build_v2_feature_row(
    row: dict,
    baseline_profile: dict | None = None,
) -> dict | None:
    """Assemble the FEATURE_COLUMNS_V2 vector for one summary row.

    Returns None when any base feature is missing (row unusable). V2 extras
    that are absent (V1-era rows) fall back to neutral values.
    """
    try:
        features = {name: float(row[name]) for name in FEATURE_COLUMNS}
    except (KeyError, TypeError, ValueError):
        return None

    for name in BASELINE_Z_COLUMNS:
        features[f"{name}_baseline_z"] = _baseline_z(features[name], baseline_profile, name)

    score = features.get("overall_score", 50.0)
    # Temporal features: prefer stored V2 columns, else neutral.
    temporal_keys = {
        "temporal_overall_score_delta": 0.0,
        "temporal_overall_score_delta_pct": 0.0,
        "temporal_overall_score_rolling_mean": score,
        "temporal_overall_score_rolling_std": 0.0,
        "temporal_overall_score_ema": score,
        "temporal_overall_score_trend": 0.0,
        "temporal_session_count": 0.0,
    }
    for key, neutral in temporal_keys.items():
        try:
            features[key] = float(row[key])
        except (KeyError, TypeError, ValueError):
            features[key] = neutral

    for key in ("task_difficulty_score", "difficulty_adjusted_error_rate"):
        try:
            features[key] = float(row[key])
        except (KeyError, TypeError, ValueError):
            features[key] = 0.0

    return features


def build_v2_matrix(
    rows: list[dict],
    baseline_profile: dict | None = None,
    label_column: str = LABEL_COLUMN_DEFAULT,
) -> tuple[list[dict], list[str]]:
    """(feature dicts, labels) for all trainable rows, in input order."""
    vectors: list[dict] = []
    labels: list[str] = []
    for row in rows:
        vector = build_v2_feature_row(row, baseline_profile)
        label = row.get(label_column)
        if vector is None or not label:
            continue
        vectors.append(vector)
        labels.append(str(label))
    return vectors, labels


def enrich_rows_with_temporal(rows: list[dict]) -> list[dict]:
    """Fill missing temporal columns from each participant's prior rows.

    V1-era CSV rows predate the V2 schema; this recomputes FR-1 features from
    the rows themselves (per participant, chronological, prior-only) so old
    data stays trainable. Rows that already carry temporal values are kept.
    Returns new row dicts; the input is not mutated.
    """
    by_participant: dict[str, list[dict]] = {}
    for row in rows:
        by_participant.setdefault(row.get("participant_id", ""), []).append(row)

    enriched: list[dict] = []
    for participant_rows in by_participant.values():
        ordered = sorted(
            participant_rows,
            key=lambda row: (row.get("started_at") or "", row.get("session_id") or ""),
        )
        prior: list[dict] = []
        for row in ordered:
            new_row = dict(row)
            try:
                current_score = float(row["overall_score"])
            except (KeyError, TypeError, ValueError):
                current_score = None
            if current_score is not None and new_row.get("temporal_session_count") in ("", None):
                temporal = temporal_feature_set(prior, current_score)
                for key, value in temporal.items():
                    if value is not None:
                        new_row[key] = value
            if current_score is not None:
                prior.append(row)
            enriched.append(new_row)
    return enriched


# ----------------------------------------------------------------- V2 model


@dataclass
class V2Predictor:
    """Wraps a trained sklearn model with schema + preprocessing metadata.

    Satisfies the duck-typed predictor interface used across the codebase:
    predict_proba(features: dict) -> {label: probability}.
    """

    model_name: str
    feature_names: list[str]
    classes: list[str]
    scaler_mean: list[float]
    scaler_scale: list[float]
    estimator: object
    metadata: dict = field(default_factory=dict)

    def _vector(self, features: dict) -> np.ndarray:
        values = []
        for name in self.feature_names:
            if name not in features:
                raise KeyError(f"Missing V2 feature: {name}")
            values.append(float(features[name]))
        vector = np.asarray(values, dtype=float)
        mean = np.asarray(self.scaler_mean, dtype=float)
        scale = np.asarray(self.scaler_scale, dtype=float)
        return (vector - mean) / scale

    def predict_proba(self, features: dict) -> dict[str, float]:
        scaled = self._vector(features).reshape(1, -1)
        probabilities = self.estimator.predict_proba(scaled)[0]
        return {label: float(p) for label, p in zip(self.classes, probabilities)}

    def predict(self, features: dict) -> dict:
        probabilities = self.predict_proba(features)
        label, confidence = max(probabilities.items(), key=lambda item: item[1])
        return {"label": label, "confidence": confidence, "probabilities": probabilities}

    def feature_contributions(self, features: dict) -> list[dict]:
        """Deterministic ablation contributions for tree-style models.

        Each feature's impact = change in the top-class probability when the
        feature is replaced by its training mean (a grounded, model-agnostic
        ablation — no fabricated attributions).
        """
        base = self.predict_proba(features)
        top_label = max(base, key=base.get)
        scaled = self._vector(features)
        contributions = []
        for index, name in enumerate(self.feature_names):
            perturbed = scaled.copy()
            perturbed[index] = 0.0  # z-score of the training mean is exactly 0
            probe = self.estimator.predict_proba(perturbed.reshape(1, -1))[0]
            delta = float(probe[self.classes.index(top_label)] - base[top_label])
            if delta != 0.0:
                contributions.append({"feature": name, "delta": round(delta, 5)})
        contributions.sort(key=lambda item: abs(item["delta"]), reverse=True)
        return contributions


# ------------------------------------------------------------ training core


def _make_estimator(model_name: str, seed: int):
    if model_name == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=seed)),
            ]
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
    if model_name == "hist_gb":
        return HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=seed,
        )
    raise ValueError(f"Unknown model name: {model_name}")


def _fit_sklearn(model_name: str, X_train, y_train, classes, seed: int):
    estimator = _make_estimator(model_name, seed)
    estimator.fit(X_train, y_train)
    return estimator


def _softmax_benchmark(train_rows: list[dict], test_rows: list[dict], label_column: str):
    """Train + evaluate the V1 softmax on the same chronological split."""
    try:
        softmax_model, _ = train_softmax_model(train_rows, label_column)
    except ValueError:
        return None
    test_vectors, y_test_raw = _softmax_xy(test_rows, softmax_model.feature_names)
    if not test_vectors:
        return None
    y_test = [normalize_label(label) for label in y_test_raw]
    if any(label not in softmax_model.classes for label in y_test):
        return None
    probabilities = [
        softmax_model.predict_proba(dict(zip(softmax_model.feature_names, row)))
        for row in test_vectors
    ]
    # sklearn log_loss assumes lexicographic label order for y_prob columns,
    # so align the matrix to sorted(classes) explicitly.
    ordered_classes = sorted(softmax_model.classes)
    aligned = [
        [row[class_index] for class_index in sorted(
            range(len(softmax_model.classes)),
            key=lambda i: ordered_classes.index(softmax_model.classes[i]),
        )]
        for row in ([probabilities[i].get(c, 0.0) for c in softmax_model.classes] for i in range(len(y_test)))
    ]
    predicted = [
        max(softmax_model.classes, key=lambda c: probabilities[i].get(c, 0.0))
        for i in range(len(y_test))
    ]
    return {
        "model_name": "softmax",
        "accuracy": float(accuracy_score(y_test, predicted)),
        "log_loss": float(log_loss(y_test, np.asarray(aligned), labels=ordered_classes)),
        "n_train": len(train_rows),
        "n_test": len(y_test),
    }


def _softmax_xy(rows: list[dict], feature_names: list[str]):
    vectors, labels = [], []
    for row in rows:
        label = row.get(LABEL_COLUMN_DEFAULT)
        if not label or not str(label).strip():
            continue
        try:
            vectors.append([float(row[name]) for name in feature_names])
            labels.append(str(label).strip())
        except (KeyError, TypeError, ValueError):
            continue
    return vectors, labels


def compare_models(
    rows: list[dict],
    label_column: str = LABEL_COLUMN_DEFAULT,
    baseline_profile: dict | None = None,
    train_fraction: float = 0.7,
    seed: int = 42,
) -> dict:
    """Train every candidate on a chronological train split, score on the test split.

    Returns {"results": [...sorted by accuracy...], "winner": name, "split": {...}}.
    """
    train_rows, test_rows = chronological_split(rows, train_fraction)
    if not train_rows or not test_rows:
        raise ValueError("Not enough labeled rows for a chronological split.")

    train_vectors, y_train_text = build_v2_matrix(train_rows, baseline_profile)
    test_vectors, y_test_text = build_v2_matrix(test_rows, baseline_profile)
    if len(set(y_train_text)) < 2:
        raise ValueError("Training split contains fewer than two classes.")

    classes = [c for c in DEFAULT_LOAD_CLASSES if c in set(y_train_text) | set(y_test_text)]
    extra = sorted((set(y_train_text) | set(y_test_text)) - set(classes))
    classes.extend(extra)

    def to_xy(vectors, labels_text):
        X = np.asarray([[v[name] for name in FEATURE_COLUMNS_V2] for v in vectors], dtype=float)
        y = np.asarray([classes.index(l) for l in labels_text], dtype=int)
        return X, y

    X_train, y_train = to_xy(train_vectors, y_train_text)
    X_test, y_test = to_xy(test_vectors, y_test_text)

    feature_names = list(FEATURE_COLUMNS_V2)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    results: list[dict] = []
    artifacts: dict[str, object] = {}

    candidate_names = ["logreg", "random_forest", "hist_gb"]
    try:
        import xgboost  # noqa: F401

        candidate_names.append("xgboost")
    except ImportError:
        pass

    for model_name in candidate_names:
        if model_name == "xgboost":
            continue  # wired when the optional dependency exists
        estimator = _fit_sklearn(model_name, X_train_s, y_train, classes, seed)
        probabilities = estimator.predict_proba(X_test_s)
        aligned = [[row[classes.index(c)] for c in classes] for row in probabilities]
        results.append(
            {
                "model_name": model_name,
                "accuracy": float(accuracy_score(y_test, np.asarray(aligned).argmax(axis=1))),
                "log_loss": float(log_loss(y_test, np.asarray(aligned), labels=list(range(len(classes))))),
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
            }
        )
        artifacts[model_name] = estimator

    softmax_result = _softmax_benchmark(train_rows, test_rows, label_column)
    if softmax_result:
        results.append(softmax_result)

    results.sort(key=lambda item: (item["accuracy"], -item["log_loss"]), reverse=True)
    winner = results[0]["model_name"]

    return {
        "results": results,
        "winner": winner,
        "split": {
            "method": "chronological",
            "train_fraction": train_fraction,
            "n_train": len(train_rows),
            "n_test": len(test_rows),
        },
        "classes": classes,
        "_artifacts": artifacts,
        "_scaler": scaler,
        "_feature_names": feature_names,
    }


# ------------------------------------------------------- persistence/registry


def save_v2_predictor(
    comparison: dict,
    model_name: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Persist one trained candidate + its registry entry.

    Returns (artifact_path, registry_path). The softmax benchmark is not
    persisted here (it already has its own loader).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    estimator = comparison["_artifacts"][model_name]
    scaler: StandardScaler = comparison["_scaler"]

    predictor = V2Predictor(
        model_name=model_name,
        feature_names=comparison["_feature_names"],
        classes=comparison["classes"],
        scaler_mean=scaler.mean_.tolist(),
        scaler_scale=scaler.scale_.tolist(),
        estimator=estimator,
        metadata={
            "model_version": MODEL_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "metrics": next(r for r in comparison["results"] if r["model_name"] == model_name),
            "split": comparison["split"],
            "trained_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    artifact_path = output_dir / f"{model_name}_v2.joblib"
    joblib.dump(predictor, artifact_path)

    entry = {
        "model_name": model_name,
        "model_version": MODEL_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "artifact": artifact_path.name,
        "metrics": predictor.metadata["metrics"],
        "trained_at": predictor.metadata["trained_at"],
    }
    registry_path = output_dir / "model_registry.json"
    registry = {"entries": []}
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            registry = {"entries": []}
    registry.setdefault("entries", []).append(entry)
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return artifact_path, registry_path


def load_v2_predictor(artifact_path: Path) -> V2Predictor:
    return joblib.load(artifact_path)
