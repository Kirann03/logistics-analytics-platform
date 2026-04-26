from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.ml_model import (
    TrainedPredictionModels,
    load_model_artifacts,
    predict_with_models as _predict_with_models,
    save_model_artifacts,
    train_prediction_models as _train_prediction_models,
)


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


@st.cache_resource(show_spinner=False)
def train_prediction_models(orders) -> TrainedPredictionModels:
    required_files = [
        "lead_time_regressor.joblib",
        "lead_time_lower_regressor.joblib",
        "lead_time_upper_regressor.joblib",
        "delay_classifier.joblib",
        "delay_threshold.joblib",
        "metrics.joblib",
        "feature_columns.joblib",
        "feature_importance.joblib",
        "training_summary.joblib",
        "context_tables.joblib",
    ]
    if MODEL_DIR.exists() and all((MODEL_DIR / filename).exists() for filename in required_files):
        return load_model_artifacts(MODEL_DIR)

    models = _train_prediction_models(orders, perform_cross_validation=False)
    save_model_artifacts(models, MODEL_DIR)
    return models


def predict_with_models(orders, inputs: dict) -> dict:
    models = train_prediction_models(orders)
    return _predict_with_models(models, orders, inputs)


__all__ = [
    "TrainedPredictionModels",
    "predict_with_models",
    "save_model_artifacts",
    "train_prediction_models",
]
