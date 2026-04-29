from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from src.analytics import (
    build_anomaly_table,
    build_canada_analytics,
    build_cost_saving_estimator,
    build_factory_summary,
    build_monthly_forecast,
    build_monthly_trend,
    build_profitability_view,
    build_region_bottlenecks,
    build_recommendation_actions,
    build_route_concentration,
    build_route_summary,
    build_seasonality_decomposition,
    build_shipping_category_summary,
    build_ship_mode_summary,
    build_sla_tracker,
    build_state_bottlenecks,
    build_state_summary,
    build_transition_matrix,
)
from src.data import DataBundle, load_data_bundle, load_uploaded_data_bundle
from src.ml_model import (
    load_model_artifacts,
    predict_with_models as run_model_prediction,
    prepare_training_frame,
    save_model_artifacts,
    summarize_training_profile,
    train_prediction_models,
    TrainedPredictionModels,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"

_DATASETS: dict[str, DataBundle] = {}
_DASHBOARD_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
_MODEL_CACHE: dict[str, Any] = {}


def _normalize_json_ready(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return dataframe_records(value)
    if isinstance(value, pd.Series):
        return [_normalize_json_ready(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {key: _normalize_json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_json_ready(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    prepared = frame.copy()
    for column in prepared.columns:
        if pd.api.types.is_datetime64_any_dtype(prepared[column]):
            prepared[column] = prepared[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return [
        {column: _normalize_json_ready(value) for column, value in row.items()}
        for row in prepared.to_dict(orient="records")
    ]


def matrix_payload(frame: pd.DataFrame) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {"index": [], "columns": [], "values": []}
    return {
        "index": [str(item) for item in frame.index.tolist()],
        "columns": [str(item) for item in frame.columns.tolist()],
        "values": [[_normalize_json_ready(item) for item in row] for row in frame.to_numpy().tolist()],
    }


@lru_cache(maxsize=4)
def get_default_bundle() -> DataBundle:
    bundle = load_data_bundle(ROOT)
    _DATASETS.setdefault("default", bundle)
    return bundle


def register_uploaded_dataset(file_bytes: bytes, filename: str) -> dict[str, Any]:
    digest = hashlib.md5(file_bytes).hexdigest()[:12]
    dataset_id = f"upload-{digest}"
    if dataset_id not in _DATASETS:
        _DATASETS[dataset_id] = load_uploaded_data_bundle(ROOT, file_bytes, filename)
    return dataset_metadata(dataset_id)


def get_bundle(dataset_id: str) -> DataBundle:
    if dataset_id == "default":
        return get_default_bundle()
    if dataset_id not in _DATASETS:
        raise KeyError(f"Unknown dataset: {dataset_id}")
    return _DATASETS[dataset_id]


def dataset_metadata(dataset_id: str) -> dict[str, Any]:
    bundle = get_bundle(dataset_id)
    orders = bundle.orders.copy()
    states_by_region: dict[str, list[str]] = {}
    for region, group in orders.groupby("region"):
        states_by_region[str(region)] = sorted(group["state"].dropna().astype(str).unique().tolist())
    return {
        "dataset_id": dataset_id,
        "data_source": bundle.data_source,
        "validation_messages": list(bundle.validation_messages),
        "row_count": int(len(orders)),
        "date_min": orders["order_date"].min().date().isoformat(),
        "date_max": orders["order_date"].max().date().isoformat(),
        "regions": sorted(orders["region"].dropna().astype(str).unique().tolist()),
        "ship_modes": sorted(orders["ship_mode"].dropna().astype(str).unique().tolist()),
        "states_by_region": states_by_region,
        "factories": sorted(orders["factory"].dropna().astype(str).unique().tolist()),
        "prediction_defaults": {
            "median_units": int(max(1, orders["units"].median())),
            "min_lead_time": int(max(1, orders["lead_time_days"].min())),
            "max_lead_time": int(max(1, orders["lead_time_days"].max())),
            "priorities": ["Standard", "Expedited", "Critical", "Economy"],
        },
    }


def apply_filters(dataset_id: str, filters: dict[str, Any]) -> tuple[pd.DataFrame, int]:
    bundle = get_bundle(dataset_id)
    orders = bundle.orders.copy()
    start_date = pd.to_datetime(filters.get("start_date") or orders["order_date"].min()).date()
    end_date = pd.to_datetime(filters.get("end_date") or orders["order_date"].max()).date()
    regions = filters.get("regions") or sorted(orders["region"].dropna().unique().tolist())
    states = filters.get("states") or sorted(orders[orders["region"].isin(regions)]["state"].dropna().unique().tolist())
    modes = filters.get("modes") or sorted(orders["ship_mode"].dropna().unique().tolist())
    delay_threshold = int(filters.get("delay_threshold") or 7)

    filtered = orders.copy()
    filtered = filtered[filtered["order_date"].dt.date.between(start_date, end_date)]
    filtered = filtered[filtered["region"].isin(regions)]
    filtered = filtered[filtered["state"].isin(states)]
    filtered = filtered[filtered["ship_mode"].isin(modes)]
    filtered = filtered.copy()
    filtered["delay_flag"] = filtered["lead_time_days"] > delay_threshold
    return filtered, delay_threshold


def _build_compare_summary(filtered_orders: pd.DataFrame, column: str) -> pd.DataFrame:
    return filtered_orders.groupby(column, as_index=False).agg(
        shipments=("order_id", "count"),
        avg_lead_time=("lead_time_days", "mean"),
        delay_rate=("delay_flag", "mean"),
        avg_sales=("sales", "mean"),
        total_sales=("sales", "sum"),
    )


def _build_alert_payload(route_summary: pd.DataFrame, factory_summary: pd.DataFrame, monthly_trend: pd.DataFrame) -> dict[str, Any]:
    high_delay = route_summary.sort_values(["delay_rate", "shipments"], ascending=[False, False]).head(3)
    factory_watch = factory_summary.sort_values(["delay_rate", "avg_lead_time"], ascending=[False, False]).head(3)
    spike_message = "No significant lead-time spike detected."
    if len(monthly_trend) >= 2:
        latest = float(monthly_trend.iloc[-1]["avg_lead_time"])
        previous_avg = float(monthly_trend.iloc[:-1]["avg_lead_time"].mean())
        if previous_avg and latest > previous_avg * 1.1:
            spike_message = f"Red flag: latest period lead time is {((latest - previous_avg) / previous_avg) * 100:.1f}% above the prior average."
    return {
        "high_delay": dataframe_records(high_delay[["route_label", "delay_rate", "avg_lead_time", "shipments"]]),
        "factory_watch": dataframe_records(factory_watch[["factory", "shipments", "avg_lead_time", "delay_rate"]]),
        "spike_message": spike_message,
    }


def _explain_anomalies(anomalies: pd.DataFrame, route_summary: pd.DataFrame, filtered_orders: pd.DataFrame) -> pd.DataFrame:
    if anomalies.empty:
        return anomalies
    route_context = route_summary[["route_label", "shipments"]].rename(columns={"shipments": "route_shipments"})
    explained = anomalies.merge(route_context, on="route_label", how="left")
    reasons: list[str] = []
    for _, row in explained.iterrows():
        parts = []
        if float(row.get("lead_time_zscore") or 0) >= 2:
            parts.append("extreme lead-time deviation")
        if float(row.get("route_shipments") or 9999) <= 5:
            parts.append("low route frequency")
        if str(row.get("ship_mode", "")).lower() in {"standard class", "second class"}:
            parts.append("slower shipping mode")
        reasons.append(" + ".join(parts) if parts else "combined operational variation")
    explained["anomaly_reason"] = reasons
    return explained


def dashboard_overview(dataset_id: str, filters: dict[str, Any]) -> dict[str, Any]:
    cache_key = (dataset_id, json.dumps(filters, sort_keys=True, default=str))
    if cache_key in _DASHBOARD_CACHE:
        return _DASHBOARD_CACHE[cache_key]

    bundle = get_bundle(dataset_id)
    filtered_orders, delay_threshold = apply_filters(dataset_id, filters)
    route_summary = build_route_summary(filtered_orders)
    state_summary = build_state_summary(filtered_orders)
    ship_mode_summary = build_ship_mode_summary(filtered_orders)
    shipping_category_summary = build_shipping_category_summary(filtered_orders)
    region_bottlenecks = build_region_bottlenecks(filtered_orders)
    state_bottlenecks = build_state_bottlenecks(filtered_orders)
    factory_summary = build_factory_summary(filtered_orders)
    monthly_trend = build_monthly_trend(filtered_orders)
    anomalies = build_anomaly_table(filtered_orders)
    route_concentration = build_route_concentration(route_summary)
    actions = build_recommendation_actions(route_summary, state_bottlenecks, factory_summary, ship_mode_summary)
    sla_route, sla_factory = build_sla_tracker(filtered_orders)
    cost_risk = build_cost_saving_estimator(route_summary)
    profitability = build_profitability_view(route_summary)
    seasonality = build_seasonality_decomposition(monthly_trend)
    forecast = build_monthly_forecast(monthly_trend, periods=6)
    transition = build_transition_matrix(filtered_orders)
    canada_analytics = build_canada_analytics(filtered_orders)
    explained_anomalies = _explain_anomalies(anomalies, route_summary, filtered_orders)

    orders_needed = filtered_orders[[
        column for column in [
            "order_id", "order_date", "ship_date", "ship_mode", "customer_id", "city", "state", "region",
            "product_name", "sales", "cost", "gross_profit", "units", "lead_time_days", "delay_flag",
            "route_label", "order_month", "route_distance_km", "factory", "dest_lat", "dest_lon",
            "factory_lat", "factory_lon", "destination_country", "state_code"
        ] if column in filtered_orders.columns
    ]].copy()

    payload = {
        "meta": {
            "data_source": bundle.data_source,
            "validation_messages": list(bundle.validation_messages),
            "filtered_count": int(len(filtered_orders)),
            "delay_threshold": delay_threshold,
        },
        "filtered_orders": dataframe_records(orders_needed),
        "route_summary": dataframe_records(route_summary),
        "state_summary": dataframe_records(state_summary),
        "ship_mode_summary": dataframe_records(ship_mode_summary),
        "shipping_category_summary": dataframe_records(shipping_category_summary),
        "region_bottlenecks": dataframe_records(region_bottlenecks),
        "state_bottlenecks": dataframe_records(state_bottlenecks),
        "factory_summary": dataframe_records(factory_summary),
        "monthly_trend": dataframe_records(monthly_trend),
        "alerts": _build_alert_payload(route_summary, factory_summary, monthly_trend),
        "comparative": {
            "region": dataframe_records(_build_compare_summary(filtered_orders, "region")),
            "factory": dataframe_records(_build_compare_summary(filtered_orders, "factory")),
            "ship_mode": dataframe_records(_build_compare_summary(filtered_orders, "ship_mode")),
        },
        "route_concentration": dataframe_records(route_concentration),
        "actions": actions,
        "sla_route": dataframe_records(sla_route),
        "sla_factory": dataframe_records(sla_factory),
        "cost_risk": dataframe_records(cost_risk),
        "profitability": dataframe_records(profitability),
        "seasonality": dataframe_records(seasonality),
        "forecast": dataframe_records(forecast),
        "transition": matrix_payload(transition),
        "anomalies": dataframe_records(explained_anomalies),
        "canada_analytics": dataframe_records(canada_analytics),
    }
    _DASHBOARD_CACHE[cache_key] = payload
    return payload




def _build_context_tables(orders: pd.DataFrame, delay_threshold: int) -> dict:
    training_frame, _ = prepare_training_frame(orders, delay_threshold=delay_threshold)
    profile = summarize_training_profile(training_frame)
    return {
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
        "training_profile": profile,
    }

def _dataset_model(dataset_id: str):
    if dataset_id in _MODEL_CACHE:
        return _MODEL_CACHE[dataset_id]

    if "default" in _MODEL_CACHE:
        default_model = _MODEL_CACHE["default"]
    else:
        bundle = get_default_bundle()
        if MODEL_DIR.exists() and (MODEL_DIR / "delay_classifier.joblib").exists():
            default_model = load_model_artifacts(MODEL_DIR)
        else:
            default_model = train_prediction_models(bundle.orders, perform_cross_validation=False)
            save_model_artifacts(default_model, MODEL_DIR)
        _MODEL_CACHE["default"] = default_model

    if dataset_id != "default":
        bundle = get_bundle(dataset_id)
        context_tables = _build_context_tables(bundle.orders, default_model.delay_threshold)
        uploaded_training_frame, _ = prepare_training_frame(bundle.orders, delay_threshold=default_model.delay_threshold)
        assembled_model = TrainedPredictionModels(
            regressor=default_model.regressor,
            lower_regressor=default_model.lower_regressor,
            upper_regressor=default_model.upper_regressor,
            classifier=default_model.classifier,
            delay_threshold=default_model.delay_threshold,
            metrics=default_model.metrics,
            feature_importance=default_model.feature_importance,
            feature_columns=default_model.feature_columns,
            training_summary={
                **default_model.training_summary,
                "rows_used": int(len(uploaded_training_frame)),
                "delay_rate": float(uploaded_training_frame["delay_flag"].mean()),
                "feature_columns": default_model.feature_columns,
                "training_profile": summarize_training_profile(uploaded_training_frame),
            },
            context_tables=context_tables,
        )
        _MODEL_CACHE[dataset_id] = assembled_model
        return assembled_model

    return default_model


def prediction_options(dataset_id: str) -> dict[str, Any]:
    bundle = get_bundle(dataset_id)
    orders = bundle.orders
    return {
        "regions": sorted(orders["region"].dropna().unique().tolist()),
        "states": sorted(orders["state"].dropna().unique().tolist()),
        "ship_modes": sorted(orders["ship_mode"].dropna().unique().tolist()),
        "factories": sorted(orders["factory"].dropna().unique().tolist()),
        "priorities": ["Standard", "Expedited", "Critical", "Economy"],
        "defaults": {
            "median_units": int(max(1, orders["units"].median())),
            "min_lead_time": int(max(1, orders["lead_time_days"].min())),
            "max_lead_time": int(max(1, orders["lead_time_days"].max())),
        },
    }


def _estimate_shipping_cost(orders: pd.DataFrame, inputs: dict) -> float:
    mode_subset = orders[orders["ship_mode"] == inputs["ship_mode"]]
    mode_cost = float(mode_subset["cost"].mean()) if not mode_subset.empty else float(orders["cost"].mean())
    priority_multiplier = {
        "Economy": 0.92,
        "Standard": 1.0,
        "Expedited": 1.22,
        "Critical": 1.45,
    }[inputs["priority"]]
    distance_multiplier = 1 + max(float(inputs.get("distance") or 0), 0) / 12000
    return float(mode_cost * max(inputs["units"], 1) * priority_multiplier * distance_multiplier)


def _calculate_route_benchmark(orders: pd.DataFrame, inputs: dict, prediction: dict) -> dict[str, Any]:
    similar = orders[(orders["state"] == inputs["state"]) & (orders["ship_mode"] == inputs["ship_mode"])]
    if similar.empty:
        similar = orders[orders["state"] == inputs["state"]]
    if similar.empty:
        similar = orders
    percentile = float((similar["lead_time_days"] <= prediction["expected_lead_time"]).mean() * 100)
    return {
        "similar_cases": int(len(similar)),
        "historical_avg": float(similar["lead_time_days"].mean()),
        "historical_best": float(similar["lead_time_days"].min()),
        "historical_worst": float(similar["lead_time_days"].max()),
        "percentile": percentile,
    }


def _make_recommendations(orders: pd.DataFrame, inputs: dict, prediction: dict) -> list[str]:
    options = prediction_options(inputs["dataset_id"])
    mode_scores = []
    for mode in options["ship_modes"]:
        result = predict_payload(inputs["dataset_id"], {**inputs, "ship_mode": mode}, int(prediction.get("delay_threshold") or 7), include_recommendations=False)
        mode_scores.append({
            "ship_mode": mode,
            "probability": result["delay_probability"],
            "lead_time": result["expected_lead_time"],
            "cost": result["estimated_cost"],
        })
    mode_df = pd.DataFrame(mode_scores).sort_values(["probability", "lead_time", "cost"])
    best_mode = mode_df.iloc[0]
    unit_target = int(max(1, orders["units"].quantile(0.5)))
    safer_states = (
        orders.groupby("state", as_index=False)
        .agg(avg_lead_time=("lead_time_days", "mean"), shipments=("order_id", "count"))
        .query("shipments >= 5")
        .sort_values("avg_lead_time")
        .head(3)["state"]
        .tolist()
    )
    recommendations = [
        f"Best shipping mode suggestion: use {best_mode['ship_mode']} for the lowest modeled delay risk.",
        f"Ideal shipment size: keep units close to {unit_target} where possible for safer historical performance.",
        f"Cost vs speed tradeoff: fastest low-risk mode is estimated at {best_mode['lead_time']:.1f} days and ${best_mode['cost']:.2f}.",
    ]
    if prediction["risk"] == "High":
        recommendations.append("Priority upgrade suggestion: upgrade to Expedited or Critical priority before dispatch.")
    if safer_states:
        recommendations.append(f"Safer route reference: historically safer destination states include {', '.join(safer_states)}.")
    return recommendations


def predict_payload(dataset_id: str, inputs: dict[str, Any], delay_threshold: int | None = None, include_recommendations: bool = True) -> dict[str, Any]:
    bundle = get_bundle(dataset_id)
    orders = bundle.orders.copy()
    models = _dataset_model(dataset_id)
    factory_coords = orders.loc[orders["factory"] == inputs["factory"], ["factory_lat", "factory_lon"]].dropna().head(1)
    enriched_inputs = dict(inputs)
    enriched_inputs["dataset_id"] = dataset_id
    if not factory_coords.empty:
        enriched_inputs["factory_lat"] = float(factory_coords.iloc[0]["factory_lat"])
        enriched_inputs["factory_lon"] = float(factory_coords.iloc[0]["factory_lon"])
    prediction = run_model_prediction(models, orders, enriched_inputs)
    prediction["estimated_cost"] = _estimate_shipping_cost(orders, inputs)
    prediction["delay_threshold"] = int(delay_threshold or prediction.get("delay_threshold") or models.delay_threshold)
    prediction["route_benchmark"] = _calculate_route_benchmark(orders, inputs, prediction)
    if include_recommendations:
        prediction["recommendations"] = _make_recommendations(orders, enriched_inputs, prediction)
    prediction = _normalize_json_ready(prediction)
    return json.loads(json.dumps(prediction))


def prediction_performance(dataset_id: str) -> dict[str, Any]:
    model = _dataset_model(dataset_id)
    return {key: _normalize_json_ready(value) for key, value in model.metrics.items()}
