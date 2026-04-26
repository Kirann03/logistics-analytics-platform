from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover - optional dependency / blocked native wheels
    KMeans = None

try:
    from sklearn.ensemble import IsolationForest
except Exception:  # pragma: no cover - optional dependency / blocked native wheels
    IsolationForest = None

try:
    from statsmodels.tsa.seasonal import STL
except Exception:  # pragma: no cover - optional dependency
    STL = None

try:
    from prophet import Prophet
except Exception:  # pragma: no cover - optional dependency
    Prophet = None


def minmax(series: pd.Series) -> pd.Series:
    if series.nunique() <= 1:
        return pd.Series(1.0, index=series.index)
    return (series - series.min()) / (series.max() - series.min())


def bottleneck_score(avg_lead_time: pd.Series, shipments: pd.Series, delay_rate: pd.Series) -> pd.Series:
    return (
        0.5 * minmax(avg_lead_time)
        + 0.3 * minmax(shipments)
        + 0.2 * minmax(delay_rate)
    ) * 100


def build_route_summary(orders: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "shipments": ("order_id", "count"),
        "avg_lead_time": ("lead_time_days", "mean"),
        "median_lead_time": ("lead_time_days", "median"),
        "lead_time_std": ("lead_time_days", "std"),
        "delay_rate": ("delay_flag", "mean"),
        "avg_sales": ("sales", "mean"),
        "avg_cost": ("cost", "mean"),
        "total_sales": ("sales", "sum"),
        "total_gross_profit": ("gross_profit", "sum"),
        "avg_units": ("units", "mean"),
    }
    if "route_distance_km" in orders.columns:
        aggregations["avg_distance_km"] = ("route_distance_km", "mean")

    summary = (
        orders.groupby(["route_label", "factory", "state", "region"], as_index=False)
        .agg(**aggregations)
        .fillna({"lead_time_std": 0.0, "avg_distance_km": 0.0})
    )
    summary["route_rank"] = summary.rank(
        method="dense", ascending=True, numeric_only=True
    )["avg_lead_time"].astype(int)
    summary["efficiency_score"] = (100 * (1 - minmax(summary["avg_lead_time"]))).round(1)
    summary["gross_margin"] = np.where(
        summary["total_sales"] > 0,
        summary["total_gross_profit"] / summary["total_sales"],
        0.0,
    )
    summary["revenue_at_risk"] = summary["delay_rate"] * summary["avg_sales"] * summary["shipments"]
    return summary.sort_values(
        ["avg_lead_time", "lead_time_std", "shipments"],
        ascending=[True, True, False],
    )


def build_state_summary(orders: pd.DataFrame) -> pd.DataFrame:
    summary = (
        orders.groupby(["state", "state_code", "region"], as_index=False)
        .agg(
            shipments=("order_id", "count"),
            avg_lead_time=("lead_time_days", "mean"),
            delay_rate=("delay_flag", "mean"),
            total_sales=("sales", "sum"),
            total_gross_profit=("gross_profit", "sum"),
        )
        .sort_values(["avg_lead_time", "shipments"], ascending=[False, False])
    )
    summary["revenue_at_risk"] = summary["delay_rate"] * (summary["total_sales"] / summary["shipments"].clip(lower=1)) * summary["shipments"]
    return summary


def build_ship_mode_summary(orders: pd.DataFrame) -> pd.DataFrame:
    return (
        orders.groupby("ship_mode", as_index=False)
        .agg(
            shipments=("order_id", "count"),
            avg_lead_time=("lead_time_days", "mean"),
            median_lead_time=("lead_time_days", "median"),
            delay_rate=("delay_flag", "mean"),
            avg_cost=("cost", "mean"),
            avg_sales=("sales", "mean"),
            total_gross_profit=("gross_profit", "sum"),
        )
        .sort_values("avg_lead_time")
    )


def build_shipping_category_summary(orders: pd.DataFrame) -> pd.DataFrame:
    categorized = orders.copy()
    categorized["shipping_category"] = np.where(
        categorized["ship_mode"].eq("Standard Class"),
        "Standard shipping",
        "Expedited shipping",
    )
    return (
        categorized.groupby(["shipping_category", "ship_mode"], as_index=False)
        .agg(
            shipments=("order_id", "count"),
            avg_lead_time=("lead_time_days", "mean"),
            lead_time_std=("lead_time_days", "std"),
            delay_rate=("delay_flag", "mean"),
            avg_cost=("cost", "mean"),
        )
        .fillna({"lead_time_std": 0.0})
        .sort_values(["shipping_category", "avg_lead_time"])
    )


def build_region_bottlenecks(orders: pd.DataFrame) -> pd.DataFrame:
    summary = (
        orders.groupby("region", as_index=False)
        .agg(
            shipments=("order_id", "count"),
            avg_lead_time=("lead_time_days", "mean"),
            delay_rate=("delay_flag", "mean"),
            total_sales=("sales", "sum"),
        )
    )
    summary["bottleneck_score"] = bottleneck_score(summary["avg_lead_time"], summary["shipments"], summary["delay_rate"])
    summary["revenue_at_risk"] = summary["delay_rate"] * (summary["total_sales"] / summary["shipments"].clip(lower=1)) * summary["shipments"]
    return summary.sort_values(["bottleneck_score", "avg_lead_time"], ascending=[False, False])


def build_state_bottlenecks(orders: pd.DataFrame) -> pd.DataFrame:
    summary = (
        orders.groupby(["state", "region"], as_index=False)
        .agg(
            shipments=("order_id", "count"),
            avg_lead_time=("lead_time_days", "mean"),
            delay_rate=("delay_flag", "mean"),
            total_sales=("sales", "sum"),
            destination_country=("destination_country", "first"),
        )
    )
    summary["bottleneck_score"] = (
        0.45 * minmax(summary["avg_lead_time"])
        + 0.35 * minmax(summary["shipments"])
        + 0.20 * minmax(summary["delay_rate"])
    ) * 100
    return summary.sort_values(["bottleneck_score", "avg_lead_time"], ascending=[False, False])


def build_canada_analytics(orders: pd.DataFrame) -> pd.DataFrame:
    canada = orders[orders["destination_country"].eq("Canada")].copy()
    if canada.empty:
        return pd.DataFrame(columns=["state", "shipments", "avg_lead_time", "delay_rate", "factory_coverage", "bottleneck_score"])
    summary = (
        canada.groupby("state", as_index=False)
        .agg(
            shipments=("order_id", "count"),
            avg_lead_time=("lead_time_days", "mean"),
            delay_rate=("delay_flag", "mean"),
            factory_coverage=("factory", "nunique"),
        )
    )
    summary["bottleneck_score"] = (
        0.5 * minmax(summary["avg_lead_time"])
        + 0.25 * minmax(summary["shipments"])
        + 0.15 * minmax(summary["delay_rate"])
        + 0.10 * (1 - minmax(summary["factory_coverage"]))
    ) * 100
    return summary.sort_values(["bottleneck_score", "avg_lead_time"], ascending=[False, False])


def build_city_summary(orders: pd.DataFrame, state: str | None = None) -> pd.DataFrame:
    scoped = orders.copy()
    if state:
        scoped = scoped[scoped["state"] == state].copy()
    if scoped.empty:
        return pd.DataFrame(columns=["city", "state", "shipments", "avg_lead_time", "delay_rate", "total_sales"])
    return (
        scoped.groupby(["city", "state"], as_index=False)
        .agg(
            shipments=("order_id", "count"),
            avg_lead_time=("lead_time_days", "mean"),
            delay_rate=("delay_flag", "mean"),
            total_sales=("sales", "sum"),
        )
        .sort_values(["shipments", "avg_lead_time"], ascending=[False, False])
    )


def build_sla_tracker(orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sla_days = {
        "Same Day": 1,
        "First Class": 3,
        "Second Class": 5,
        "Standard Class": 7,
    }
    scoped = orders.copy()
    scoped["sla_days"] = scoped["ship_mode"].map(sla_days).fillna(7)
    scoped["sla_compliant"] = scoped["lead_time_days"] <= scoped["sla_days"]

    route_sla = (
        scoped.groupby(["route_label", "factory", "ship_mode"], as_index=False)
        .agg(
            shipments=("order_id", "count"),
            avg_lead_time=("lead_time_days", "mean"),
            sla_target=("sla_days", "mean"),
            sla_compliance=("sla_compliant", "mean"),
        )
        .sort_values(["sla_compliance", "avg_lead_time"], ascending=[True, False])
    )
    factory_sla = (
        scoped.groupby("factory", as_index=False)
        .agg(
            shipments=("order_id", "count"),
            avg_lead_time=("lead_time_days", "mean"),
            sla_compliance=("sla_compliant", "mean"),
        )
        .sort_values(["sla_compliance", "avg_lead_time"], ascending=[True, False])
    )
    return route_sla, factory_sla


def build_cost_saving_estimator(route_summary: pd.DataFrame) -> pd.DataFrame:
    estimator = route_summary.copy()
    estimator["revenue_at_risk"] = estimator["delay_rate"] * estimator["avg_sales"] * estimator["shipments"]
    estimator["recovered_revenue_10pct"] = estimator["revenue_at_risk"] * 0.10
    estimator["recovered_revenue_25pct"] = estimator["revenue_at_risk"] * 0.25
    return estimator.sort_values(["revenue_at_risk", "avg_lead_time"], ascending=[False, False])


def build_profitability_view(route_summary: pd.DataFrame) -> pd.DataFrame:
    profitability = route_summary.copy()
    profitability["profit_per_shipment"] = profitability["total_gross_profit"] / profitability["shipments"].clip(lower=1)
    profitability["profitability_risk_score"] = (
        0.5 * minmax(profitability["total_gross_profit"])
        + 0.3 * minmax(profitability["avg_lead_time"])
        + 0.2 * minmax(profitability["delay_rate"])
    ) * 100
    return profitability.sort_values(["profitability_risk_score", "total_gross_profit"], ascending=[False, False])


def build_monthly_trend(orders: pd.DataFrame) -> pd.DataFrame:
    trend = (
        orders.groupby("order_month", as_index=False)
        .agg(
            shipments=("order_id", "count"),
            avg_lead_time=("lead_time_days", "mean"),
            delay_rate=("delay_flag", "mean"),
            total_sales=("sales", "sum"),
            total_gross_profit=("gross_profit", "sum"),
        )
        .sort_values("order_month")
    )
    trend["rolling_avg_lead_time"] = trend["avg_lead_time"].rolling(window=3, min_periods=1).mean()
    return trend


def build_seasonality_decomposition(monthly_trend: pd.DataFrame) -> pd.DataFrame:
    if monthly_trend.empty:
        return pd.DataFrame(columns=["order_month", "trend_component", "seasonal_component", "residual_component", "method"])
    series = monthly_trend.set_index("order_month")["avg_lead_time"].astype(float)
    if len(series) >= 6 and STL is not None:
        result = STL(series, period=min(12, max(2, len(series) // 2)), robust=True).fit()
        return pd.DataFrame(
            {
                "order_month": series.index,
                "trend_component": result.trend.values,
                "seasonal_component": result.seasonal.values,
                "residual_component": result.resid.values,
                "method": "STL",
            }
        )
    trend_component = series.rolling(window=3, min_periods=1).mean()
    seasonal_component = series - trend_component
    residual_component = series - trend_component - seasonal_component
    return pd.DataFrame(
        {
            "order_month": series.index,
            "trend_component": trend_component.values,
            "seasonal_component": seasonal_component.values,
            "residual_component": residual_component.values,
            "method": "Rolling fallback",
        }
    )


def build_monthly_forecast(monthly_trend: pd.DataFrame, periods: int = 6) -> pd.DataFrame:
    if monthly_trend.empty:
        return pd.DataFrame(columns=["ds", "forecast_lead_time", "forecast_delay_rate", "model_type"])
    scoped = monthly_trend.rename(columns={"order_month": "ds", "avg_lead_time": "lead_time_y", "delay_rate": "delay_y"}).copy()
    scoped["ds"] = pd.to_datetime(scoped["ds"])
    future_dates = pd.date_range(scoped["ds"].max() + pd.offsets.MonthBegin(1), periods=periods, freq="MS")

    if Prophet is not None and len(scoped) >= 6:
        lead_model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
        lead_model.fit(scoped[["ds", "lead_time_y"]].rename(columns={"lead_time_y": "y"}))
        lead_fcst = lead_model.predict(pd.DataFrame({"ds": future_dates}))[["ds", "yhat"]]

        delay_model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
        delay_model.fit(scoped[["ds", "delay_y"]].rename(columns={"delay_y": "y"}))
        delay_fcst = delay_model.predict(pd.DataFrame({"ds": future_dates}))[["ds", "yhat"]]

        forecast = lead_fcst.merge(delay_fcst, on="ds", suffixes=("_lead", "_delay"))
        forecast["model_type"] = "Prophet"
        forecast = forecast.rename(columns={"yhat_lead": "forecast_lead_time", "yhat_delay": "forecast_delay_rate"})
        return forecast

    x = np.arange(len(scoped))
    lead_coef = np.polyfit(x, scoped["lead_time_y"].astype(float), deg=1)
    delay_coef = np.polyfit(x, scoped["delay_y"].astype(float), deg=1)
    future_x = np.arange(len(scoped), len(scoped) + periods)
    return pd.DataFrame(
        {
            "ds": future_dates,
            "forecast_lead_time": np.polyval(lead_coef, future_x),
            "forecast_delay_rate": np.clip(np.polyval(delay_coef, future_x), 0, 1),
            "model_type": "Linear fallback",
        }
    )


def build_transition_matrix(orders: pd.DataFrame) -> pd.DataFrame:
    if orders.empty:
        return pd.DataFrame()
    route_modes = (
        orders.groupby(["route_label", "ship_mode"], as_index=False)
        .agg(shipments=("order_id", "count"), avg_lead_time=("lead_time_days", "mean"))
    )
    pivot = route_modes.pivot_table(index="route_label", columns="ship_mode", values="shipments", fill_value=0)
    return pivot.sort_index()


def build_delay_timeline(orders: pd.DataFrame, delay_threshold: int) -> pd.DataFrame:
    timeline = orders.copy()
    timeline["delay_bucket"] = np.select(
        [
            timeline["delay_flag"],
            timeline["lead_time_days"] >= max(delay_threshold - 2, 1),
        ],
        ["Delayed", "Near Threshold"],
        default="On Time",
    )
    return timeline


def build_factory_summary(orders: pd.DataFrame) -> pd.DataFrame:
    summary = (
        orders.groupby("factory", as_index=False)
        .agg(
            shipments=("order_id", "count"),
            avg_lead_time=("lead_time_days", "mean"),
            delay_rate=("delay_flag", "mean"),
            total_sales=("sales", "sum"),
            total_gross_profit=("gross_profit", "sum"),
            avg_cost=("cost", "mean"),
            states_served=("state", "nunique"),
        )
        .sort_values(["avg_lead_time", "shipments"], ascending=[True, False])
    )
    summary["performance_score"] = (
        100
        * (
            0.5 * (1 - minmax(summary["avg_lead_time"]))
            + 0.3 * (1 - minmax(summary["delay_rate"]))
            + 0.2 * minmax(summary["states_served"])
        )
    ).round(1)
    return summary


def build_anomaly_table(orders: pd.DataFrame) -> pd.DataFrame:
    anomalies = orders.copy()
    numeric_cols = [column for column in ["lead_time_days", "units", "sales", "cost", "route_distance_km"] if column in anomalies.columns]
    if IsolationForest is not None and len(anomalies) >= 20 and numeric_cols:
        matrix = anomalies[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(anomalies[numeric_cols].median(numeric_only=True))
        model = IsolationForest(contamination=0.07, random_state=42)
        anomaly_scores = model.fit_predict(matrix)
        anomalies["anomaly_flag"] = anomaly_scores == -1
        anomalies["lead_time_zscore"] = (anomalies["lead_time_days"] - anomalies["lead_time_days"].mean()) / max(anomalies["lead_time_days"].std(), 1)
        anomalies["severity"] = np.where(anomalies["anomaly_flag"], "Critical", "Normal")
        anomalies["anomaly_model"] = "Isolation Forest"
        return anomalies.sort_values(["anomaly_flag", "lead_time_days"], ascending=[False, False])

    mean = orders["lead_time_days"].mean()
    std = max(orders["lead_time_days"].std(), 1)
    anomalies["lead_time_zscore"] = (anomalies["lead_time_days"] - mean) / std
    anomalies["severity"] = np.select(
        [
            anomalies["lead_time_zscore"] >= 2.0,
            anomalies["lead_time_zscore"] >= 1.25,
        ],
        ["Critical", "Watch"],
        default="Normal",
    )
    anomalies["anomaly_model"] = "Z-Score"
    return anomalies.sort_values("lead_time_zscore", ascending=False)


def build_route_clusters(route_summary: pd.DataFrame) -> pd.DataFrame:
    if route_summary.empty:
        return route_summary.copy()
    clustered = route_summary.copy()
    features = route_summary[["avg_lead_time", "delay_rate", "shipments"]].copy()
    if "avg_distance_km" in route_summary.columns:
        features["avg_distance_km"] = route_summary["avg_distance_km"].fillna(route_summary["avg_distance_km"].median())

    if KMeans is not None and len(route_summary) >= 3:
        cluster_count = min(4, max(2, len(route_summary) // 5 or 2))
        model = KMeans(n_clusters=cluster_count, n_init=10, random_state=42)
        clustered["cluster_label"] = model.fit_predict(features.fillna(features.median(numeric_only=True)))
    else:
        risk_band = pd.qcut(clustered["avg_lead_time"].rank(method="first"), q=min(3, len(clustered)), labels=False, duplicates="drop")
        clustered["cluster_label"] = risk_band.fillna(0).astype(int)

    label_map = {}
    for label, subset in clustered.groupby("cluster_label"):
        if subset["avg_lead_time"].mean() <= clustered["avg_lead_time"].quantile(0.33):
            label_map[label] = "Efficient cluster"
        elif subset["delay_rate"].mean() >= clustered["delay_rate"].quantile(0.67):
            label_map[label] = "High-risk cluster"
        else:
            label_map[label] = "Balanced cluster"
    clustered["cluster_name"] = clustered["cluster_label"].map(label_map)
    clustered["cluster_method"] = "KMeans" if KMeans is not None and len(route_summary) >= 3 else "Quantile fallback"
    return clustered


def build_route_concentration(route_summary: pd.DataFrame) -> pd.DataFrame:
    concentration = route_summary.sort_values("shipments", ascending=False).copy()
    total = max(concentration["shipments"].sum(), 1)
    concentration["shipment_share"] = concentration["shipments"] / total
    concentration["cumulative_share"] = concentration["shipment_share"].cumsum()
    concentration["dominant_factory_share"] = concentration.groupby("factory")["shipments"].transform("sum") / total
    concentration["concentration_risk"] = np.where(
        concentration["dominant_factory_share"] > 0.40,
        "High",
        np.where(concentration["dominant_factory_share"] > 0.25, "Watch", "Balanced"),
    )
    return concentration


def top_and_bottom_routes(route_summary: pd.DataFrame, size: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    fastest = route_summary.head(size).copy()
    slowest = route_summary.tail(size).copy().sort_values(
        ["avg_lead_time", "lead_time_std", "shipments"],
        ascending=[False, False, False],
    )
    return fastest, slowest


def build_analysis_snapshot(
    orders: pd.DataFrame,
    route_summary: pd.DataFrame,
    region_bottlenecks: pd.DataFrame,
    shipping_category_summary: pd.DataFrame,
) -> dict[str, str]:
    fastest_route = route_summary.iloc[0]
    slowest_route = route_summary.sort_values(
        ["avg_lead_time", "lead_time_std", "shipments"],
        ascending=[False, False, False],
    ).iloc[0]
    top_region = region_bottlenecks.iloc[0]
    fastest_mode = shipping_category_summary.sort_values("avg_lead_time").iloc[0]
    return {
        "portfolio_lead_time": f"{orders['lead_time_days'].mean():.1f} days",
        "fastest_route": f"{fastest_route['route_label']} ({fastest_route['avg_lead_time']:.1f} days)",
        "slowest_route": f"{slowest_route['route_label']} ({slowest_route['avg_lead_time']:.1f} days)",
        "top_region_bottleneck": f"{top_region['region']} ({top_region['avg_lead_time']:.1f} days, {int(top_region['shipments'])} shipments)",
        "fastest_mode": f"{fastest_mode['ship_mode']} ({fastest_mode['avg_lead_time']:.1f} days)",
    }


def build_recommendation_actions(
    route_summary: pd.DataFrame,
    state_bottlenecks: pd.DataFrame,
    factory_summary: pd.DataFrame,
    ship_mode_summary: pd.DataFrame,
) -> list[str]:
    slowest_route = route_summary.sort_values(["avg_lead_time", "shipments"], ascending=[False, False]).iloc[0]
    bottleneck_state = state_bottlenecks.iloc[0]
    best_factory = factory_summary.sort_values("performance_score", ascending=False).iloc[0]
    fastest_mode = ship_mode_summary.sort_values("avg_lead_time").iloc[0]
    return [
        f"Prioritize intervention on {slowest_route['route_label']} because it has the highest average lead time in the filtered view.",
        f"Monitor {bottleneck_state['state']} closely; it combines high lead time, shipment volume, and delay rate.",
        f"Use {best_factory['factory']} as the benchmark factory because it has the strongest performance score.",
        f"Prefer {fastest_mode['ship_mode']} when service speed is the priority.",
    ]


def generate_executive_summary(
    filtered_orders: pd.DataFrame,
    route_summary: pd.DataFrame,
    state_summary: pd.DataFrame,
    ship_mode_summary: pd.DataFrame,
    delay_threshold: int,
) -> tuple[str, str]:
    best_route = route_summary.iloc[0]
    worst_route = route_summary.sort_values("efficiency_score").iloc[0]
    top_state = state_summary.sort_values("delay_rate", ascending=False).iloc[0]
    fastest_mode = ship_mode_summary.sort_values("avg_lead_time").iloc[0]
    revenue_at_risk = float(route_summary["revenue_at_risk"].sum()) if "revenue_at_risk" in route_summary.columns else 0.0

    summary = (
        f"Across {len(filtered_orders):,} filtered orders, the portfolio averages "
        f"{filtered_orders['lead_time_days'].mean():.1f} days from order to shipment. "
        f"The strongest route is {best_route['route_label']} with an efficiency score of "
        f"{best_route['efficiency_score']:.1f}, while {worst_route['route_label']} is the main "
        f"performance drag. {top_state['state']} currently shows the highest delay pressure, and "
        f"{fastest_mode['ship_mode']} is the fastest ship mode on observed lead time. "
        f"Estimated revenue currently exposed to delays is ${revenue_at_risk:,.0f}."
    )
    recommendations = (
        f"Prioritize root-cause review for {worst_route['route_label']}, monitor states with the "
        f"highest delay rates, and use the {delay_threshold}-day threshold to escalate delayed "
        f"orders earlier in operations reviews."
    )
    return summary, recommendations


def describe_filters(
    start_date: date,
    end_date: date,
    regions: list[str],
    states: list[str],
    modes: list[str],
) -> str:
    return (
        f"Viewing {len(regions)} region(s), {len(states)} state(s), and {len(modes)} ship mode(s) "
        f"from {start_date.isoformat()} to {end_date.isoformat()}."
    )
