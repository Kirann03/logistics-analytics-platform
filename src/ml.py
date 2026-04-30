from src.backend_service import _dataset_model
from src.ml_model import (
    TrainedPredictionModels,
    predict_with_models as _predict_with_models,
    save_model_artifacts,
)

def predict_with_models(orders, inputs: dict) -> dict:
    models = _dataset_model("default")
    return _predict_with_models(models, orders, inputs)

__all__ = [
    "TrainedPredictionModels",
    "predict_with_models",
    "save_model_artifacts",
]
