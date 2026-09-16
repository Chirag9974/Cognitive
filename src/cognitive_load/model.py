from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DEFAULT_LOAD_CLASSES, FEATURE_COLUMNS


LABEL_MAP = {
    "1": "Low",
    "low": "Low",
    "excellent": "Low",
    "2": "Manageable",
    "manageable": "Manageable",
    "good": "Manageable",
    "3": "Elevated",
    "elevated": "Elevated",
    "medium": "Elevated",
    "4": "High",
    "high": "High",
    "risky": "High",
}


def normalize_label(raw_label: str | None) -> str | None:
    if raw_label is None:
        return None
    cleaned = str(raw_label).strip().lower()
    if not cleaned:
        return None
    return LABEL_MAP.get(cleaned, str(raw_label).strip())


def rows_to_matrix(rows: list[dict], feature_names: list[str]) -> np.ndarray:
    matrix = []
    for row in rows:
        matrix.append([float(row[feature_name]) for feature_name in feature_names])
    return np.asarray(matrix, dtype=float)


@dataclass
class SoftmaxLoadModel:
    feature_names: list[str]
    classes: list[str]
    mean: list[float]
    scale: list[float]
    weights: list[list[float]]
    bias: list[float]

    def predict_proba(self, features: dict) -> dict[str, float]:
        vector = np.asarray([float(features[name]) for name in self.feature_names], dtype=float)
        mean = np.asarray(self.mean, dtype=float)
        scale = np.asarray(self.scale, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        bias = np.asarray(self.bias, dtype=float)

        scaled = (vector - mean) / scale
        logits = scaled @ weights + bias
        logits -= np.max(logits)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        return {label: float(probabilities[index]) for index, label in enumerate(self.classes)}

    def predict(self, features: dict) -> dict:
        probabilities = self.predict_proba(features)
        label, confidence = max(probabilities.items(), key=lambda item: item[1])
        return {
            "label": label,
            "confidence": confidence,
            "probabilities": probabilities,
        }

    def save(self, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "feature_names": self.feature_names,
                    "classes": self.classes,
                    "mean": self.mean,
                    "scale": self.scale,
                    "weights": self.weights,
                    "bias": self.bias,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return output_path

    @classmethod
    def load(cls, input_path: Path) -> "SoftmaxLoadModel":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        return cls(**payload)


def prepare_training_rows(rows: list[dict], label_column: str, feature_names: list[str] | None = None) -> tuple[list[dict], list[str]]:
    feature_names = feature_names or FEATURE_COLUMNS
    prepared = []
    labels = []

    for row in rows:
        label = normalize_label(row.get(label_column))
        if not label:
            continue
        try:
            candidate = {feature_name: float(row[feature_name]) for feature_name in feature_names}
        except (KeyError, TypeError, ValueError):
            continue
        prepared.append(candidate)
        labels.append(label)

    return prepared, labels


def train_softmax_model(
    rows: list[dict],
    label_column: str,
    feature_names: list[str] | None = None,
    epochs: int = 2500,
    learning_rate: float = 0.08,
    l2_penalty: float = 1e-4,
) -> tuple[SoftmaxLoadModel, dict]:
    feature_names = feature_names or FEATURE_COLUMNS
    prepared_rows, labels = prepare_training_rows(rows, label_column, feature_names)
    if len(prepared_rows) < 8:
        raise ValueError("At least 8 labeled rows are required for training.")

    classes = [label for label in DEFAULT_LOAD_CLASSES if label in labels]
    remaining = [label for label in sorted(set(labels)) if label not in classes]
    classes.extend(remaining)

    X = rows_to_matrix(prepared_rows, feature_names)
    y = np.asarray([classes.index(label) for label in labels], dtype=int)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0
    X_scaled = (X - mean) / scale

    sample_count, feature_count = X_scaled.shape
    class_count = len(classes)
    weights = np.zeros((feature_count, class_count), dtype=float)
    bias = np.zeros(class_count, dtype=float)
    targets = np.eye(class_count, dtype=float)[y]

    for _ in range(epochs):
        logits = X_scaled @ weights + bias
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        error = probabilities - targets
        weight_gradient = (X_scaled.T @ error) / sample_count + l2_penalty * weights
        bias_gradient = error.mean(axis=0)

        weights -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient

    model = SoftmaxLoadModel(
        feature_names=feature_names,
        classes=classes,
        mean=mean.tolist(),
        scale=scale.tolist(),
        weights=weights.tolist(),
        bias=bias.tolist(),
    )

    predictions = [model.predict(row)["label"] for row in prepared_rows]
    accuracy = sum(pred == true for pred, true in zip(predictions, labels)) / len(labels)
    metrics = {
        "rows_used": len(prepared_rows),
        "feature_count": len(feature_names),
        "classes": classes,
        "train_accuracy": round(float(accuracy), 4),
    }
    return model, metrics
