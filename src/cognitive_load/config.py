from __future__ import annotations

from pathlib import Path

APP_NAME = "Cognitive Load Predictor"
APP_VERSION = "2.0.0"
MODEL_VERSION = "2.0.0"
FEATURE_SCHEMA_VERSION = "2.0"

# ---------------------------------------------------------------- V2 constants

# --- Temporal modeling (PRD FR-1) ---
TEMPORAL_MIN_SESSIONS = 3          # minimum valid history for trend/temporal features
FORECAST_MIN_SESSIONS = 5          # minimum valid history for next-session forecasting
ROLLING_WINDOW = 3                 # rolling mean/std window (k)
EMA_ALPHA = 0.3                    # exponential moving average smoothing factor
TEMPORAL_TREND_THRESHOLD = 1.5     # |mean delta| in score points for a trend to count

# --- Anomaly detection (PRD FR-8) ---
ANOMALY_ROBUST_Z_THRESHOLD = 3.5   # |robust z| (median/MAD) for a single-feature reason
ANOMALY_IFOREST_THRESHOLD = 0.62   # isolation-forest decision score for Anomalous
ANOMALY_IFOREST_MIN_SESSIONS = 10  # minimum history rows to fit IsolationForest
ANOMALY_REVIEW_SCORE = 0.5         # anomaly_score at/above this => status Review
ANOMALY_ANOMALOUS_SCORE = 0.75     # anomaly_score at/above this => status Anomalous

# --- Session quality score (PRD section 18) ---
QUALITY_WEIGHTS = {
    "response_validity": 0.25,
    "timing_consistency": 0.25,
    "missing_data": 0.25,
    "internal_consistency": 0.25,
}

# --- Uncertainty (PRD FR-3) ---
UNCERTAINTY_HIGH_MARGIN = 0.60       # top1-top2 probability margin for High confidence
UNCERTAINTY_MODERATE_MARGIN = 0.25   # margin for Moderate
UNCERTAINTY_MAX_VARIANCE_Z = 1.0     # tolerated behavioral-variance z before downgrade

# --- Difficulty normalization (PRD FR-10, light) ---
MEMORY_SPAN_MIN = 3
MEMORY_SPAN_MAX = 7
SWITCH_RATE_EXPECTED = 0.5
CONGRUENT_RATE_EXPECTED = 0.4
DIFFICULTY_ERROR_MULTIPLIER_K = 0.4  # error-rate multiplier strength: mult = 1 + K*(difficulty-0.5)

# --- Recommendations (PRD FR-6) ---
RECOMMENDATION_MAX_ITEMS = 4

TASK_LABELS = {
    "reaction_time": "Reaction Time",
    "task_switching": "Task Switching",
    "working_memory": "Working Memory",
    "stroop": "Selective Attention",
}

TASK_SCORE_WEIGHTS = {
    "reaction_time": 0.25,
    "task_switching": 0.30,
    "working_memory": 0.25,
    "stroop": 0.20,
}

TASK_TIMING = {
    "reaction_time": {
        "response_window_ms": 60000,
        "feedback_ms": 900,
    },
    "task_switching": {
        "response_window_ms": 60000,
        "feedback_ms": 1000,
    },
    "working_memory": {
        "digit_display_ms": 900,
        "digit_gap_ms": 350,
        "recall_window_ms": 60000,
        "feedback_ms": 1200,
    },
    "stroop": {
        "response_window_ms": 60000,
        "feedback_ms": 1000,
    },
}

PAUSE_THRESHOLD_RATIO = 0.50
PAUSE_THRESHOLD_CAP_MS = 30000.0

SCORING_THRESHOLDS = {
    "reaction_mean_rt_ms": {"direction": "inverse", "excellent": 350.0, "poor": 60000.0},
    "reaction_rt_std_ms": {"direction": "inverse", "excellent": 40.0, "poor": 20000.0},
    "reaction_false_start_rate": {"direction": "inverse", "excellent": 0.00, "poor": 0.30},
    "switch_cost_ms": {"direction": "inverse", "excellent": 150.0, "poor": 12000.0},
    "switch_mean_rt_ms": {"direction": "inverse", "excellent": 1200.0, "poor": 60000.0},
    "switch_accuracy": {"direction": "direct", "excellent": 0.96, "poor": 0.55},
    "repeat_accuracy": {"direction": "direct", "excellent": 0.98, "poor": 0.60},
    "working_memory_full_recall_rate": {"direction": "direct", "excellent": 0.90, "poor": 0.20},
    "working_memory_max_span": {"direction": "direct", "excellent": 7.0, "poor": 3.0},
    "working_memory_position_accuracy": {"direction": "direct", "excellent": 0.95, "poor": 0.45},
    "working_memory_mean_response_ms": {"direction": "inverse", "excellent": 4000.0, "poor": 60000.0},
    "stroop_interference_ms": {"direction": "inverse", "excellent": 120.0, "poor": 10000.0},
    "stroop_mean_rt_ms": {"direction": "inverse", "excellent": 1200.0, "poor": 60000.0},
    "stroop_incongruent_accuracy": {"direction": "direct", "excellent": 0.94, "poor": 0.45},
    "stroop_rt_std_ms": {"direction": "inverse", "excellent": 80.0, "poor": 20000.0},
    "pause_rate": {"direction": "inverse", "excellent": 0.02, "poor": 0.40},
    "overall_error_rate": {"direction": "inverse", "excellent": 0.04, "poor": 0.45},
}

FEATURE_COLUMNS = list(SCORING_THRESHOLDS.keys())

TASK_SCORE_FEATURES = {
    "reaction_time": [
        ("reaction_mean_rt_ms", 0.50),
        ("reaction_rt_std_ms", 0.30),
        ("reaction_false_start_rate", 0.20),
    ],
    "task_switching": [
        ("switch_cost_ms", 0.30),
        ("switch_mean_rt_ms", 0.20),
        ("switch_accuracy", 0.30),
        ("repeat_accuracy", 0.20),
    ],
    "working_memory": [
        ("working_memory_full_recall_rate", 0.35),
        ("working_memory_max_span", 0.25),
        ("working_memory_position_accuracy", 0.20),
        ("working_memory_mean_response_ms", 0.20),
    ],
    "stroop": [
        ("stroop_interference_ms", 0.25),
        ("stroop_mean_rt_ms", 0.20),
        ("stroop_incongruent_accuracy", 0.35),
        ("stroop_rt_std_ms", 0.20),
    ],
}

DEFAULT_LOAD_CLASSES = ["Low", "Manageable", "Elevated", "High"]

SCORE_BANDS = [
    {"label": "Low", "performance": "Excellent", "min_score": 80.0, "max_score": 100.0},
    {"label": "Manageable", "performance": "Good", "min_score": 60.0, "max_score": 79.9999},
    {"label": "Elevated", "performance": "Medium", "min_score": 40.0, "max_score": 59.9999},
    {"label": "High", "performance": "Risky", "min_score": 0.0, "max_score": 39.9999},
]

VALIDATION_COMPONENT_WEIGHTS = {
    "overall_score": 0.40,
    "reaction_time": TASK_SCORE_WEIGHTS["reaction_time"] * 0.60,
    "task_switching": TASK_SCORE_WEIGHTS["task_switching"] * 0.60,
    "working_memory": TASK_SCORE_WEIGHTS["working_memory"] * 0.60,
    "stroop": TASK_SCORE_WEIGHTS["stroop"] * 0.60,
}

SUMMARY_FIELDNAMES = [
    "session_id",
    "participant_id",
    "started_at",
    "finished_at",
    "duration_seconds",
    "baseline_session",
    "sleep_hours",
    "stress_level",
    "caffeine_mg",
    "fatigue_level",
    "notes",
    "self_report_load_band",
    "rule_load_band",
    "range_validation_band",
    "final_load_band",
    "performance_band",
    "prediction_method",
    "validation_confidence",
    "prediction_confidence",
    "overall_score",
    "reaction_score",
    "task_switching_score",
    "working_memory_score",
    "stroop_score",
    "reaction_mean_rt_ms",
    "reaction_rt_std_ms",
    "reaction_false_start_rate",
    "switch_cost_ms",
    "switch_mean_rt_ms",
    "switch_accuracy",
    "repeat_accuracy",
    "working_memory_full_recall_rate",
    "working_memory_max_span",
    "working_memory_position_accuracy",
    "working_memory_mean_response_ms",
    "stroop_interference_ms",
    "stroop_mean_rt_ms",
    "stroop_incongruent_accuracy",
    "stroop_rt_std_ms",
    "pause_rate",
    "overall_error_rate",
    "low_validation_score",
    "manageable_validation_score",
    "elevated_validation_score",
    "high_validation_score",
    "model_confidence",
    # --- V2 columns ---
    "v2_prediction_method",
    "v2_predicted_band",
    "v2_probability_low",
    "v2_probability_manageable",
    "v2_probability_elevated",
    "v2_probability_high",
    "v2_confidence",
    "v2_uncertainty",
    "v2_model_version",
    "session_quality_score",
    "anomaly_score",
    "anomaly_status",
    "trend_direction",
    "forecast_band",
    "forecast_probability",
    "top_contributor_1",
    "top_contributor_2",
    "top_contributor_3",
]

PROJECT_DATA_DIRS = [
    "data",
    "data/baselines",
    "data/models",
    "data/reports",
    "data/sessions",
]


def project_root_from(file_path: str | Path) -> Path:
    return Path(file_path).resolve().parents[2]


# ------------------------------------------------------------ V2 feature schema

# Baseline deviations from the personal baseline (FR-2 model input).
BASELINE_Z_COLUMNS = [
    "reaction_mean_rt_ms",
    "switch_cost_ms",
    "stroop_interference_ms",
    "working_memory_full_recall_rate",
    "overall_error_rate",
    "pause_rate",
]

# V2 feature schema: base features + baseline z + temporal (FR-1/FR-2).
FEATURE_COLUMNS_V2 = [
    *FEATURE_COLUMNS,
    *[f"{name}_baseline_z" for name in BASELINE_Z_COLUMNS],
    "temporal_overall_score_delta",
    "temporal_overall_score_delta_pct",
    "temporal_overall_score_rolling_mean",
    "temporal_overall_score_rolling_std",
    "temporal_overall_score_ema",
    "temporal_overall_score_trend",
    "temporal_session_count",
    "task_difficulty_score",
    "difficulty_adjusted_error_rate",
]

MODEL_REGISTRY_FILENAME = "model_registry.json"
REGISTRY_LABELS = ["model_name", "model_version", "feature_schema_version"]

class InsufficientHistoryError(ValueError):
    """Raised when temporal/forecast features need more history than available."""


ERROR_CODES = {
    "INSUFFICIENT_HISTORY": "At least {min_sessions} valid sessions are required for this operation.",
}
