from __future__ import annotations

from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any

from catboost import CatBoostClassifier, CatBoostRegressor, Pool
import joblib
import numpy as np
import pandas as pd

from src.lookup import LOCATION_COORDINATES


PRIORITY_FROM_MODE = {
    "Same Day": "Critical",
    "First Class": "Expedited",
    "Second Class": "Standard",
    "Standard Class": "Economy",
}

SEASON_FROM_MONTH = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Autumn",
    10: "Autumn",
    11: "Autumn",
}

NUMERIC_FEATURES = [
    "units",
    "route_distance_km",
    "order_month",
    "order_week",
    "order_dayofweek",
    "is_weekend",
    "is_holiday",
    "route_frequency",
    "route_frequency_ratio",
    "route_avg_lead_time",
    "route_delay_rate",
    "avg_lead_time_state",
    "avg_lead_time_region",
    "avg_lead_time_ship_mode",
    "avg_lead_time_factory",
    "moving_avg_lead_time",
    "rolling_7_shipment_avg",
    "rolling_30_shipment_avg",
    "trend_slope",
    "load_per_shipment",
    "units_bucket_code",
]

CATEGORICAL_FEATURES = [
    "region",
    "state",
    "ship_mode",
    "factory",
    "route",
    "factory_region_route",
    "destination_country",
    "units_bucket",
    "priority_band",
    "season",
    "region_ship_mode",
    "state_units_bucket",
    "route_ship_mode",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class TrainedPredictionModels:
    regressor: CatBoostRegressor
    lower_regressor: CatBoostRegressor
    upper_regressor: CatBoostRegressor
    classifier: CatBoostClassifier
    delay_threshold: int
    metrics: dict[str, float]
    feature_importance: pd.DataFrame
    feature_columns: list[str]
    training_summary: dict[str, object]
    context_tables: dict[str, object]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
    return 2 * radius * asin(sqrt(a))


def _holiday_flag(series: pd.Series) -> pd.Series:
    month_day = series.dt.strftime("%m-%d")
    holiday_days = {
        "01-01",
        "07-01",
        "07-04",
        "11-11",
        "11-28",
        "12-24",
        "12-25",
        "12-26",
        "12-31",
    }
    return month_day.isin(holiday_days).astype(int)


def _units_bucket(series: pd.Series) -> pd.Series:
    return pd.cut(
        series.fillna(0),
        bins=[-np.inf, 5, 20, np.inf],
        labels=["Small", "Medium", "Large"],
    ).astype(str)


def _compute_route_distance(frame: pd.DataFrame) -> pd.Series:
    has_coords = frame[["factory_lat", "factory_lon", "dest_lat", "dest_lon"]].notna().all(axis=1)
    distance = pd.Series(np.nan, index=frame.index, dtype="float64")
    distance.loc[has_coords] = frame.loc[has_coords].apply(
        lambda row: haversine_km(
            float(row["factory_lat"]),
            float(row["factory_lon"]),
            float(row["dest_lat"]),
            float(row["dest_lon"]),
        ),
        axis=1,
    )
    return distance


def _apply_time_and_shipping_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["destination_country"] = enriched["destination_country"].fillna("Unknown")
    enriched["priority_band"] = enriched["ship_mode"].map(PRIORITY_FROM_MODE).fillna("Standard")
    enriched["order_month"] = enriched["order_date"].dt.month.astype(int)
    enriched["order_week"] = enriched["order_date"].dt.isocalendar().week.astype(int)
    enriched["order_dayofweek"] = enriched["order_date"].dt.dayofweek.astype(int)
    enriched["is_weekend"] = enriched["order_dayofweek"].isin([5, 6]).astype(int)
    enriched["season"] = enriched["order_month"].map(SEASON_FROM_MONTH).fillna("Unknown")
    enriched["is_holiday"] = _holiday_flag(enriched["order_date"])
    enriched["route"] = enriched["factory"] + " -> " + enriched["state"]
    enriched["factory_region_route"] = enriched["factory"] + " -> " + enriched["region"]
    enriched["route_ship_mode"] = enriched["route"] + " | " + enriched["ship_mode"]
    enriched["region_ship_mode"] = enriched["region"] + " | " + enriched["ship_mode"]
    enriched["units_bucket"] = _units_bucket(enriched["units"])
    enriched["state_units_bucket"] = enriched["state"] + " | " + enriched["units_bucket"]
    enriched["units_bucket_code"] = enriched["units_bucket"].map({"Small": 1, "Medium": 2, "Large": 3}).fillna(0)
    enriched["route_distance_km"] = _compute_route_distance(enriched)
    enriched["load_per_shipment"] = enriched["units"].clip(lower=1)
    return enriched


def _apply_historical_aggregates(frame: pd.DataFrame, delay_threshold: int) -> pd.DataFrame:
    enriched = frame.sort_values("order_date").copy()
    enriched["delay_flag"] = (enriched["lead_time_days"] >= delay_threshold).astype(int)

    route_stats = enriched.groupby("route").agg(
        route_frequency=("order_id", "count"),
        route_avg_lead_time=("lead_time_days", "mean"),
        route_delay_rate=("delay_flag", "mean"),
    )
    state_avg = enriched.groupby("state")["lead_time_days"].mean()
    region_avg = enriched.groupby("region")["lead_time_days"].mean()
    ship_mode_avg = enriched.groupby("ship_mode")["lead_time_days"].mean()
    factory_avg = enriched.groupby("factory")["lead_time_days"].mean()

    enriched["route_frequency"] = enriched["route"].map(route_stats["route_frequency"])
    enriched["route_frequency_ratio"] = enriched["route_frequency"] / max(len(enriched), 1)
    enriched["route_avg_lead_time"] = enriched["route"].map(route_stats["route_avg_lead_time"])
    enriched["route_delay_rate"] = enriched["route"].map(route_stats["route_delay_rate"])
    enriched["avg_lead_time_state"] = enriched["state"].map(state_avg)
    enriched["avg_lead_time_region"] = enriched["region"].map(region_avg)
    enriched["avg_lead_time_ship_mode"] = enriched["ship_mode"].map(ship_mode_avg)
    enriched["avg_lead_time_factory"] = enriched["factory"].map(factory_avg)
    enriched["moving_avg_lead_time"] = (
        enriched.groupby("route")["lead_time_days"].transform(lambda s: s.shift(1).expanding().mean())
    )
    enriched["rolling_7_shipment_avg"] = (
        enriched.groupby("route")["lead_time_days"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    )
    enriched["rolling_30_shipment_avg"] = (
        enriched.groupby("route")["lead_time_days"].transform(lambda s: s.shift(1).rolling(30, min_periods=1).mean())
    )
    enriched["trend_slope"] = enriched["rolling_7_shipment_avg"] - enriched["rolling_30_shipment_avg"]

    global_mean = float(enriched["lead_time_days"].mean())
    enriched["moving_avg_lead_time"] = enriched["moving_avg_lead_time"].fillna(global_mean)
    enriched["rolling_7_shipment_avg"] = enriched["rolling_7_shipment_avg"].fillna(global_mean)
    enriched["rolling_30_shipment_avg"] = enriched["rolling_30_shipment_avg"].fillna(global_mean)
    enriched["trend_slope"] = enriched["trend_slope"].fillna(0.0)
    return enriched


def prepare_training_frame(orders: pd.DataFrame, delay_threshold: int | None = None) -> tuple[pd.DataFrame, int]:
    frame = orders.copy()
    frame = frame.drop_duplicates(subset=["order_id"]).copy()
    frame = frame[frame["lead_time_days"].notna()].copy()
    frame = frame[frame["lead_time_days"] >= 0].copy()
    frame["order_date"] = pd.to_datetime(frame["order_date"], errors="coerce")
    frame["ship_date"] = pd.to_datetime(frame["ship_date"], errors="coerce")
    frame = frame.dropna(
        subset=["region", "state", "ship_mode", "factory", "order_date", "ship_date", "units", "lead_time_days"]
    ).copy()

    threshold = delay_threshold or int(frame["lead_time_days"].quantile(0.75))
    frame = _apply_time_and_shipping_features(frame)
    frame = _apply_historical_aggregates(frame, threshold)
    return frame, threshold


def build_feature_frame(inputs: dict) -> pd.DataFrame:
    order_date = pd.to_datetime(inputs["order_date"])
    state_meta = LOCATION_COORDINATES.get(inputs["state"], {})
    dest_lat = state_meta.get("lat")
    dest_lon = state_meta.get("lon")
    destination_country = state_meta.get("country", "Unknown")

    route_distance_km = np.nan
    if all(value is not None for value in [inputs.get("factory_lat"), inputs.get("factory_lon"), dest_lat, dest_lon]):
        route_distance_km = haversine_km(
            float(inputs["factory_lat"]),
            float(inputs["factory_lon"]),
            float(dest_lat),
            float(dest_lon),
        )
    if float(inputs.get("distance") or 0) > 0:
        route_distance_km = float(inputs["distance"])

    units_bucket = _units_bucket(pd.Series([inputs["units"]])).iloc[0]
    return pd.DataFrame(
        [
            {
                "region": inputs["region"],
                "state": inputs["state"],
                "ship_mode": inputs["ship_mode"],
                "factory": inputs["factory"],
                "route": f"{inputs['factory']} -> {inputs['state']}",
                "factory_region_route": f"{inputs['factory']} -> {inputs['region']}",
                "destination_country": destination_country,
                "units_bucket": units_bucket,
                "priority_band": inputs.get("priority", PRIORITY_FROM_MODE.get(inputs["ship_mode"], "Standard")),
                "season": SEASON_FROM_MONTH.get(int(order_date.month), "Unknown"),
                "region_ship_mode": f"{inputs['region']} | {inputs['ship_mode']}",
                "state_units_bucket": f"{inputs['state']} | {units_bucket}",
                "route_ship_mode": f"{inputs['factory']} -> {inputs['state']} | {inputs['ship_mode']}",
                "units": int(inputs["units"]),
                "route_distance_km": route_distance_km,
                "order_month": int(order_date.month),
                "order_week": int(order_date.isocalendar().week),
                "order_dayofweek": int(order_date.dayofweek),
                "is_weekend": int(order_date.dayofweek in [5, 6]),
                "is_holiday": int(_holiday_flag(pd.Series([order_date])).iloc[0]),
                "load_per_shipment": int(inputs["units"]),
                "units_bucket_code": {"Small": 1, "Medium": 2, "Large": 3}.get(units_bucket, 0),
            }
        ]
    )


def attach_context_features(feature_frame: pd.DataFrame, context_tables: dict[str, object]) -> pd.DataFrame:
    frame = feature_frame.copy()
    route = frame.at[0, "route"]
    state = frame.at[0, "state"]
    region = frame.at[0, "region"]
    ship_mode = frame.at[0, "ship_mode"]
    factory = frame.at[0, "factory"]

    global_mean = float(context_tables["global_mean"])
    global_delay_rate = float(context_tables["global_delay_rate"])
    route_frequency_map = context_tables["route_frequency_map"]
    route_avg_map = context_tables["route_avg_map"]
    route_delay_map = context_tables["route_delay_map"]
    state_avg_map = context_tables["state_avg_map"]
    region_avg_map = context_tables["region_avg_map"]
    ship_mode_avg_map = context_tables["ship_mode_avg_map"]
    factory_avg_map = context_tables["factory_avg_map"]
    total_rows = int(context_tables["total_rows"])

    route_frequency = int(route_frequency_map.get(route, 0))
    route_avg = float(route_avg_map.get(route, global_mean))
    frame["route_frequency"] = route_frequency
    frame["route_frequency_ratio"] = route_frequency / max(total_rows, 1)
    frame["route_avg_lead_time"] = route_avg
    frame["route_delay_rate"] = float(route_delay_map.get(route, global_delay_rate))
    frame["avg_lead_time_state"] = float(state_avg_map.get(state, global_mean))
    frame["avg_lead_time_region"] = float(region_avg_map.get(region, global_mean))
    frame["avg_lead_time_ship_mode"] = float(ship_mode_avg_map.get(ship_mode, global_mean))
    frame["avg_lead_time_factory"] = float(factory_avg_map.get(factory, global_mean))
    frame["moving_avg_lead_time"] = frame["route_avg_lead_time"]
    frame["rolling_7_shipment_avg"] = frame["route_avg_lead_time"]
    frame["rolling_30_shipment_avg"] = frame["route_avg_lead_time"]
    frame["trend_slope"] = 0.0
    return frame


def prepare_model_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    matrix = frame[MODEL_FEATURES].copy()
    for feature in CATEGORICAL_FEATURES:
        matrix[feature] = matrix[feature].fillna("Unknown").astype(str)
    for feature in NUMERIC_FEATURES:
        matrix[feature] = pd.to_numeric(matrix[feature], errors="coerce")
    return matrix


def aggregate_feature_importance(model) -> pd.DataFrame:
    raw_importance = model.get_feature_importance()
    importance = pd.DataFrame(
        {
            "feature": [feature.replace("_", " ").title() for feature in MODEL_FEATURES],
            "importance_score": raw_importance[: len(MODEL_FEATURES)],
        }
    )
    total = float(importance["importance_score"].sum()) or 1.0
    importance["impact_percent"] = (importance["importance_score"] / total * 100).round(1)
    return importance.sort_values("impact_percent", ascending=False)[["feature", "impact_percent"]]


def summarize_training_profile(training_frame: pd.DataFrame) -> dict[str, Any]:
    numeric_ranges = {}
    for feature in NUMERIC_FEATURES:
        series = pd.to_numeric(training_frame[feature], errors="coerce").dropna()
        if series.empty:
            continue
        numeric_ranges[feature] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0) or 0.0),
            "p05": float(series.quantile(0.05)),
            "p95": float(series.quantile(0.95)),
        }

    category_vocab = {}
    for feature in CATEGORICAL_FEATURES:
        counts = training_frame[feature].fillna("Unknown").astype(str).value_counts()
        category_vocab[feature] = {
            "top_values": counts.head(25).index.tolist(),
            "coverage": float(counts.head(25).sum() / max(counts.sum(), 1)),
        }

    lead_time = training_frame["lead_time_days"]
    return {
        "numeric_ranges": numeric_ranges,
        "category_vocab": category_vocab,
        "lead_time_distribution": {
            "mean": float(lead_time.mean()),
            "std": float(lead_time.std(ddof=0) or 0.0),
            "p10": float(lead_time.quantile(0.10)),
            "p90": float(lead_time.quantile(0.90)),
        },
    }


def build_prediction_shap(model: CatBoostClassifier, feature_frame: pd.DataFrame) -> pd.DataFrame:
    cat_features = [feature for feature in CATEGORICAL_FEATURES if feature in feature_frame.columns]
    pool = Pool(feature_frame, cat_features=cat_features)
    shap_values = model.get_feature_importance(data=pool, type="ShapValues")
    contributions = shap_values[0][:-1]
    explanation = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "shap_value": contributions,
            "abs_value": np.abs(contributions),
            "direction": np.where(contributions >= 0, "Increase Risk", "Reduce Risk"),
            "feature_value": [str(feature_frame.iloc[0][feature]) for feature in MODEL_FEATURES],
        }
    )
    return explanation.sort_values("abs_value", ascending=False)


def detect_feature_drift(feature_frame: pd.DataFrame, training_profile: dict[str, Any]) -> dict[str, Any]:
    numeric_alerts = []
    for feature, profile in training_profile["numeric_ranges"].items():
        if feature not in feature_frame.columns:
            continue
        value = pd.to_numeric(feature_frame.iloc[0][feature], errors="coerce")
        if pd.isna(value):
            continue
        severity = 0.0
        if value < profile["p05"] or value > profile["p95"]:
            width = max(profile["p95"] - profile["p05"], 1e-6)
            severity = abs(float(value) - float(profile["mean"])) / width
            numeric_alerts.append(
                {
                    "feature": feature.replace("_", " ").title(),
                    "value": float(value),
                    "baseline": f"{profile['p05']:.2f} - {profile['p95']:.2f}",
                    "severity": round(float(severity), 2),
                }
            )

    categorical_alerts = []
    for feature, profile in training_profile["category_vocab"].items():
        if feature not in feature_frame.columns:
            continue
        value = str(feature_frame.iloc[0][feature])
        if value not in profile["top_values"]:
            categorical_alerts.append(
                {
                    "feature": feature.replace("_", " ").title(),
                    "value": value,
                    "baseline": "Unseen / low-frequency category",
                    "severity": 1.0,
                }
            )

    max_numeric = max([item["severity"] for item in numeric_alerts], default=0.0)
    max_categorical = max([item["severity"] for item in categorical_alerts], default=0.0)
    drift_score = round(max(max_numeric, max_categorical), 2)
    if drift_score >= 1.5:
        status = "High"
    elif drift_score >= 0.75:
        status = "Medium"
    else:
        status = "Low"

    return {
        "status": status,
        "score": drift_score,
        "numeric_alerts": numeric_alerts,
        "categorical_alerts": categorical_alerts,
    }


def _load_sklearn_training_tools():
    try:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            mean_absolute_error,
            precision_score,
            r2_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
    except Exception as exc:  # pragma: no cover - optional local dependency failure
        raise RuntimeError(
            "Training utilities from scikit-learn are unavailable in this environment. "
            "Prediction can still run from saved model artifacts, but retraining requires a working scikit-learn installation."
        ) from exc

    return {
        "accuracy_score": accuracy_score,
        "f1_score": f1_score,
        "mean_absolute_error": mean_absolute_error,
        "precision_score": precision_score,
        "r2_score": r2_score,
        "recall_score": recall_score,
        "roc_auc_score": roc_auc_score,
        "KFold": KFold,
        "StratifiedKFold": StratifiedKFold,
        "train_test_split": train_test_split,
    }


def train_prediction_models(
    orders: pd.DataFrame,
    *,
    perform_cross_validation: bool = False,
    random_state: int = 42,
) -> TrainedPredictionModels:
    sklearn_tools = _load_sklearn_training_tools()
    accuracy_score = sklearn_tools["accuracy_score"]
    f1_score = sklearn_tools["f1_score"]
    mean_absolute_error = sklearn_tools["mean_absolute_error"]
    precision_score = sklearn_tools["precision_score"]
    r2_score = sklearn_tools["r2_score"]
    recall_score = sklearn_tools["recall_score"]
    roc_auc_score = sklearn_tools["roc_auc_score"]
    KFold = sklearn_tools["KFold"]
    StratifiedKFold = sklearn_tools["StratifiedKFold"]
    train_test_split = sklearn_tools["train_test_split"]

    training_frame, delay_threshold = prepare_training_frame(orders)
    X = prepare_model_matrix(training_frame)
    y_reg = training_frame["lead_time_days"]
    y_clf = training_frame["delay_flag"]
    cat_feature_indices = [X.columns.get_loc(column) for column in CATEGORICAL_FEATURES]

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X,
        y_reg,
        y_clf,
        test_size=0.2,
        random_state=random_state,
        stratify=y_clf,
    )

    regressor = CatBoostRegressor(
        iterations=650,
        depth=8,
        learning_rate=0.045,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
    )
    lower_regressor = CatBoostRegressor(
        iterations=500,
        depth=8,
        learning_rate=0.04,
        loss_function="Quantile:alpha=0.1",
        eval_metric="Quantile:alpha=0.1",
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
    )
    upper_regressor = CatBoostRegressor(
        iterations=500,
        depth=8,
        learning_rate=0.04,
        loss_function="Quantile:alpha=0.9",
        eval_metric="Quantile:alpha=0.9",
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
    )
    classifier = CatBoostClassifier(
        iterations=700,
        depth=8,
        learning_rate=0.04,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=random_state,
        verbose=False,
        auto_class_weights="Balanced",
        allow_writing_files=False,
    )

    regressor.fit(
        X_train,
        y_reg_train,
        cat_features=cat_feature_indices,
        eval_set=(X_test, y_reg_test),
        use_best_model=True,
        early_stopping_rounds=60,
    )
    lower_regressor.fit(
        X_train,
        y_reg_train,
        cat_features=cat_feature_indices,
        eval_set=(X_test, y_reg_test),
        use_best_model=True,
        early_stopping_rounds=50,
    )
    upper_regressor.fit(
        X_train,
        y_reg_train,
        cat_features=cat_feature_indices,
        eval_set=(X_test, y_reg_test),
        use_best_model=True,
        early_stopping_rounds=50,
    )
    classifier.fit(
        X_train,
        y_clf_train,
        cat_features=cat_feature_indices,
        eval_set=(X_test, y_clf_test),
        use_best_model=True,
        early_stopping_rounds=60,
    )

    reg_predictions = regressor.predict(X_test)
    clf_predictions = classifier.predict(X_test)
    clf_probabilities = classifier.predict_proba(X_test)[:, 1]

    metrics = {
        "mae": float(mean_absolute_error(y_reg_test, reg_predictions)),
        "r2": float(r2_score(y_reg_test, reg_predictions)),
        "accuracy": float(accuracy_score(y_clf_test, clf_predictions)),
        "precision": float(precision_score(y_clf_test, clf_predictions, zero_division=0)),
        "recall": float(recall_score(y_clf_test, clf_predictions, zero_division=0)),
        "f1": float(f1_score(y_clf_test, clf_predictions, zero_division=0)),
        "auc": float(roc_auc_score(y_clf_test, clf_probabilities)),
    }

    if perform_cross_validation:
        cv_cls = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        recall_scores = []
        f1_scores = []
        auc_scores = []
        for train_idx, valid_idx in cv_cls.split(X, y_clf):
            X_fold_train, X_fold_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_fold_train, y_fold_valid = y_clf.iloc[train_idx], y_clf.iloc[valid_idx]
            fold_model = CatBoostClassifier(
                iterations=450,
                depth=8,
                learning_rate=0.04,
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=random_state,
                verbose=False,
                auto_class_weights="Balanced",
                allow_writing_files=False,
            )
            fold_model.fit(
                X_fold_train,
                y_fold_train,
                cat_features=cat_feature_indices,
                eval_set=(X_fold_valid, y_fold_valid),
                use_best_model=True,
                early_stopping_rounds=40,
            )
            fold_pred = fold_model.predict(X_fold_valid)
            fold_prob = fold_model.predict_proba(X_fold_valid)[:, 1]
            recall_scores.append(recall_score(y_fold_valid, fold_pred, zero_division=0))
            f1_scores.append(f1_score(y_fold_valid, fold_pred, zero_division=0))
            auc_scores.append(roc_auc_score(y_fold_valid, fold_prob))

        cv_reg = KFold(n_splits=5, shuffle=True, random_state=random_state)
        mae_scores = []
        for train_idx, valid_idx in cv_reg.split(X):
            X_fold_train, X_fold_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_fold_train, y_fold_valid = y_reg.iloc[train_idx], y_reg.iloc[valid_idx]
            fold_reg = CatBoostRegressor(
                iterations=420,
                depth=8,
                learning_rate=0.045,
                loss_function="RMSE",
                eval_metric="RMSE",
                random_seed=random_state,
                verbose=False,
                allow_writing_files=False,
            )
            fold_reg.fit(
                X_fold_train,
                y_fold_train,
                cat_features=cat_feature_indices,
                eval_set=(X_fold_valid, y_fold_valid),
                use_best_model=True,
                early_stopping_rounds=40,
            )
            fold_reg_pred = fold_reg.predict(X_fold_valid)
            mae_scores.append(mean_absolute_error(y_fold_valid, fold_reg_pred))

        metrics["cv_recall"] = float(np.mean(recall_scores))
        metrics["cv_f1"] = float(np.mean(f1_scores))
        metrics["cv_auc"] = float(np.mean(auc_scores))
        metrics["cv_mae"] = float(np.mean(mae_scores))

    importance = aggregate_feature_importance(classifier)
    training_profile = summarize_training_profile(training_frame)
    context_tables = {
        "global_mean": float(training_frame["lead_time_days"].mean()),
        "global_delay_rate": float(training_frame["delay_flag"].mean()),
        "route_frequency_map": training_frame.groupby("route")["order_id"].count().to_dict(),
        "route_avg_map": training_frame.groupby("route")["lead_time_days"].mean().to_dict(),
        "route_delay_map": training_frame.groupby("route")["delay_flag"].mean().to_dict(),
        "state_avg_map": training_frame.groupby("state")["lead_time_days"].mean().to_dict(),
        "region_avg_map": training_frame.groupby("region")["lead_time_days"].mean().to_dict(),
        "ship_mode_avg_map": training_frame.groupby("ship_mode")["lead_time_days"].mean().to_dict(),
        "factory_avg_map": training_frame.groupby("factory")["lead_time_days"].mean().to_dict(),
        "total_rows": int(len(training_frame)),
        "training_profile": training_profile,
    }
    training_summary = {
        "rows_used": int(len(training_frame)),
        "delay_rate": float(training_frame["delay_flag"].mean()),
        "feature_columns": MODEL_FEATURES,
    }
    return TrainedPredictionModels(
        regressor=regressor,
        lower_regressor=lower_regressor,
        upper_regressor=upper_regressor,
        classifier=classifier,
        delay_threshold=delay_threshold,
        metrics=metrics,
        feature_importance=importance,
        feature_columns=MODEL_FEATURES.copy(),
        training_summary=training_summary,
        context_tables=context_tables,
    )


def predict_with_models(models: TrainedPredictionModels, orders: pd.DataFrame, inputs: dict) -> dict:
    feature_frame = prepare_model_matrix(attach_context_features(build_feature_frame(inputs), models.context_tables))
    expected = float(models.regressor.predict(feature_frame)[0])
    lower_bound = float(models.lower_regressor.predict(feature_frame)[0])
    upper_bound = float(models.upper_regressor.predict(feature_frame)[0])
    probability = float(models.classifier.predict_proba(feature_frame)[0][1])
    shap_explanation = build_prediction_shap(models.classifier, feature_frame)
    drift_report = detect_feature_drift(feature_frame, models.context_tables["training_profile"])

    if probability < 0.35:
        risk = "Low"
    elif probability < 0.65:
        risk = "Medium"
    else:
        risk = "High"

    route_mask = (
        (orders["factory"] == inputs["factory"])
        & (orders["state"] == inputs["state"])
        & (orders["ship_mode"] == inputs["ship_mode"])
    )
    state_mode_mask = (orders["state"] == inputs["state"]) & (orders["ship_mode"] == inputs["ship_mode"])
    evidence = int(max(route_mask.sum(), state_mode_mask.sum(), (orders["state"] == inputs["state"]).sum(), 1))
    confidence = min(
        97.0,
        52
        + models.metrics["accuracy"] * 14
        + models.metrics["auc"] * 12
        + models.metrics["recall"] * 10
        + np.log1p(evidence) * 3.5,
    )

    return {
        "delay_probability": probability,
        "risk": risk,
        "expected_lead_time": expected,
        "lead_time_lower": min(lower_bound, upper_bound),
        "lead_time_upper": max(lower_bound, upper_bound),
        "confidence": confidence,
        "delay_threshold": models.delay_threshold,
        "samples": {
            "route": int(route_mask.sum()),
            "state_mode": int(state_mode_mask.sum()),
            "state": int((orders["state"] == inputs["state"]).sum()),
            "region": int((orders["region"] == inputs["region"]).sum()),
            "mode": int((orders["ship_mode"] == inputs["ship_mode"]).sum()),
        },
        "importance": models.feature_importance.copy(),
        "shap_explanation": shap_explanation,
        "drift_report": drift_report,
        "model_metrics": models.metrics.copy(),
    }


def save_model_artifacts(models: TrainedPredictionModels, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(models.regressor, output_path / "lead_time_regressor.joblib")
    joblib.dump(models.lower_regressor, output_path / "lead_time_lower_regressor.joblib")
    joblib.dump(models.upper_regressor, output_path / "lead_time_upper_regressor.joblib")
    joblib.dump(models.classifier, output_path / "delay_classifier.joblib")
    joblib.dump(models.feature_columns, output_path / "feature_columns.joblib")
    joblib.dump(models.feature_importance, output_path / "feature_importance.joblib")
    joblib.dump(models.training_summary, output_path / "training_summary.joblib")
    joblib.dump(models.context_tables, output_path / "context_tables.joblib")
    joblib.dump(models.metrics, output_path / "metrics.joblib")
    joblib.dump(models.delay_threshold, output_path / "delay_threshold.joblib")


def load_model_artifacts(model_dir: str | Path) -> TrainedPredictionModels:
    model_path = Path(model_dir)
    return TrainedPredictionModels(
        regressor=joblib.load(model_path / "lead_time_regressor.joblib"),
        lower_regressor=joblib.load(model_path / "lead_time_lower_regressor.joblib"),
        upper_regressor=joblib.load(model_path / "lead_time_upper_regressor.joblib"),
        classifier=joblib.load(model_path / "delay_classifier.joblib"),
        delay_threshold=joblib.load(model_path / "delay_threshold.joblib"),
        metrics=joblib.load(model_path / "metrics.joblib"),
        feature_importance=joblib.load(model_path / "feature_importance.joblib"),
        feature_columns=joblib.load(model_path / "feature_columns.joblib"),
        training_summary=joblib.load(model_path / "training_summary.joblib"),
        context_tables=joblib.load(model_path / "context_tables.joblib"),
    )
