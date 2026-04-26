from __future__ import annotations

import json
from pathlib import Path

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from branca.colormap import LinearColormap
from folium.plugins import HeatMap
from streamlit.components.v1 import html

from src.analytics import (
    build_canada_analytics,
    build_city_summary,
    build_delay_timeline,
    build_route_clusters,
    build_route_summary,
    build_state_bottlenecks,
    describe_filters,
    top_and_bottom_routes,
)
from src.alerts import trigger_sla_alerts
from src.api_client import get_dashboard_overview, records_to_frame
from src.common import base_layout, render_header, render_metric, render_metric_sparkline, render_section_heading

PRESET_PATH = Path(__file__).resolve().parent.parent / "saved_filters.json"

def render_folium_map(map_object: folium.Map, height: int = 620) -> None:
    html(map_object.get_root().render(), height=height, scrolling=False)


def _load_presets() -> dict[str, dict]:
    if PRESET_PATH.exists():
        return json.loads(PRESET_PATH.read_text(encoding="utf-8"))
    return {}


def _save_presets(presets: dict[str, dict]) -> None:
    PRESET_PATH.write_text(json.dumps(presets, indent=2), encoding="utf-8")


def build_heatmap_leaflet(destination_map: pd.DataFrame) -> folium.Map:
    center_lat = destination_map["dest_lat"].mean()
    center_lon = destination_map["dest_lon"].mean()
    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=3,
        tiles="CartoDB positron",
        control_scale=True,
    )

    heat_data = destination_map[["dest_lat", "dest_lon", "avg_lead_time"]].values.tolist()
    HeatMap(
        heat_data,
        name="Lead Time Heatmap",
        min_opacity=0.35,
        radius=32,
        blur=22,
        gradient={
            0.2: "#1d4ed8",
            0.45: "#38bdf8",
            0.7: "#f1bb7b",
            1.0: "#c4683c",
        },
    ).add_to(fmap)

    scale = LinearColormap(
        colors=["#1d4ed8", "#38bdf8", "#f1bb7b", "#c4683c"],
        vmin=float(destination_map["avg_lead_time"].min()),
        vmax=float(destination_map["avg_lead_time"].max()),
    )
    scale.caption = "Average Lead Time (days)"
    scale.add_to(fmap)

    for _, row in destination_map.iterrows():
        folium.CircleMarker(
            location=[row["dest_lat"], row["dest_lon"]],
            radius=max(6, min(18, row["shipments"] ** 0.5)),
            color="#fff7ef",
            weight=1,
            fill=True,
            fill_opacity=0.82,
            fill_color=scale(row["avg_lead_time"]),
            popup=folium.Popup(
                (
                    f"<b>{row['state']}</b><br>"
                    f"Country: {row['destination_country']}<br>"
                    f"Shipments: {int(row['shipments'])}<br>"
                    f"Average lead time: {row['avg_lead_time']:.1f} days<br>"
                    f"Delay rate: {row['delay_rate']:.1%}"
                ),
                max_width=280,
            ),
            tooltip=f"{row['state']}: {row['avg_lead_time']:.1f} days",
        ).add_to(fmap)

    bounds = destination_map[["dest_lat", "dest_lon"]].dropna().values.tolist()
    if bounds:
        fmap.fit_bounds(bounds, padding=(24, 24))

    return fmap


def build_network_leaflet(
    route_network: pd.DataFrame,
    factories: pd.DataFrame,
    destination_map: pd.DataFrame,
    country_label: str,
) -> folium.Map:
    fmap = folium.Map(
        location=[46, -101],
        zoom_start=3,
        tiles="CartoDB Positron",
        control_scale=True,
    )

    for _, row in route_network.iterrows():
        folium.PolyLine(
            locations=[
                [row["factory_lat"], row["factory_lon"]],
                [row["dest_lat"], row["dest_lon"]],
            ],
            color="#d97706",
            weight=max(2.2, min(7.5, row["shipments"] / 14)),
            opacity=0.62,
            tooltip=(
                f"{row['factory']} -> {row['state']} | "
                f"Shipments: {int(row['shipments'])} | "
                f"Lead time: {row['avg_lead_time']:.1f} days"
            ),
        ).add_to(fmap)

    for _, row in factories.iterrows():
        folium.CircleMarker(
            location=[row["factory_lat"], row["factory_lon"]],
            radius=7,
            color="#ffffff",
            weight=1.5,
            fill=True,
            fill_color="#ff8a5b",
            fill_opacity=0.95,
            popup=folium.Popup(f"<b>{row['factory']}</b><br>Factory origin", max_width=220),
            tooltip=row["factory"],
        ).add_to(fmap)

    for _, row in destination_map.iterrows():
        folium.CircleMarker(
            location=[row["dest_lat"], row["dest_lon"]],
            radius=max(4, min(13, row["shipments"] ** 0.45)),
            color="#ffffff",
            weight=1.2,
            fill=True,
            fill_color="#2563eb",
            fill_opacity=0.82,
            popup=folium.Popup(
                (
                    f"<b>{row['state']}</b><br>"
                    f"Country: {row['destination_country']}<br>"
                    f"Shipments: {int(row['shipments'])}<br>"
                    f"Average lead time: {row['avg_lead_time']:.1f} days"
                ),
                max_width=250,
            ),
        ).add_to(fmap)

    bounds = []
    bounds.extend(destination_map[["dest_lat", "dest_lon"]].dropna().values.tolist())
    bounds.extend(factories[["factory_lat", "factory_lon"]].dropna().values.tolist())
    if bounds:
        fmap.fit_bounds(bounds, padding=(36, 36))

    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 20px;
        left: 20px;
        z-index: 9999;
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(0,0,0,0.12);
        border-radius: 12px;
        padding: 10px 12px;
        box-shadow: 0 8px 18px rgba(0,0,0,0.18);
        font-size: 12px;
        color: #1f2937;
        min-width: 180px;
    ">
        <div style="font-weight:700; margin-bottom:6px;">{country_label} Network</div>
        {"<div style='margin-bottom:6px;'>US factory origins connected to Canadian destinations.</div>" if country_label == "Canada" else ""}
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
            <span style="width:10px; height:10px; border-radius:999px; background:#ff8a5b; display:inline-block;"></span>
            <span>Factory origin</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
            <span style="width:10px; height:10px; border-radius:999px; background:#2563eb; display:inline-block;"></span>
            <span>Destination state/province</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
            <span style="width:16px; height:3px; background:#d97706; display:inline-block;"></span>
            <span>Shipment route</span>
        </div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    return fmap


def build_volume_bottleneck_leaflet(destination_map: pd.DataFrame) -> folium.Map:
    fmap = folium.Map(
        location=[47, -96],
        zoom_start=3,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    max_shipments = max(float(destination_map["shipments"].max()), 1.0)
    max_lead_time = max(float(destination_map["avg_lead_time"].max()), 1.0)
    min_lead_time = float(destination_map["avg_lead_time"].min())

    for _, row in destination_map.iterrows():
        lead_ratio = (row["avg_lead_time"] - min_lead_time) / max(max_lead_time - min_lead_time, 1)
        color = "#2d9bf0" if lead_ratio < 0.45 else "#f59e0b" if lead_ratio < 0.75 else "#dc2626"
        radius = 4 + (float(row["shipments"]) / max_shipments) ** 0.55 * 18
        folium.CircleMarker(
            location=[row["dest_lat"], row["dest_lon"]],
            radius=radius,
            color="#1d4ed8",
            weight=1.2,
            fill=True,
            fill_color=color,
            fill_opacity=0.62,
            popup=folium.Popup(
                (
                    f"<b>{row['state']}</b><br>"
                    f"Country: {row['destination_country']}<br>"
                    f"Route volume: {int(row['shipments'])}<br>"
                    f"Avg lead time: {row['avg_lead_time']:.1f} days<br>"
                    f"Delay rate: {row['delay_rate']:.1%}"
                ),
                max_width=280,
            ),
            tooltip=(
                f"{row['state']} | Volume: {int(row['shipments'])} | "
                f"Lead time: {row['avg_lead_time']:.1f} days"
            ),
        ).add_to(fmap)

    bounds = destination_map[["dest_lat", "dest_lon"]].dropna().values.tolist()
    if bounds:
        fmap.fit_bounds(bounds, padding=(24, 24))

    legend_html = """
    <div style="
        position: fixed;
        bottom: 20px;
        left: 20px;
        z-index: 9999;
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(0,0,0,0.12);
        border-radius: 12px;
        padding: 10px 12px;
        box-shadow: 0 8px 18px rgba(0,0,0,0.18);
        font-size: 12px;
        color: #1f2937;
        min-width: 220px;
    ">
        <div style="font-weight:700; margin-bottom:6px;">Volume and Bottleneck Map</div>
        <div>Bubble size = route volume</div>
        <div style="margin-top:4px;"><span style="color:#2d9bf0;">●</span> Lower lead time</div>
        <div><span style="color:#f59e0b;">●</span> Moderate bottleneck</div>
        <div><span style="color:#dc2626;">●</span> High bottleneck</div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
    return fmap


def render_filters(metadata: dict) -> dict:
    presets = _load_presets()
    with st.sidebar:
        st.markdown("## Dashboard Filters")
        st.caption("Refine the analysis with executive filters for time window, geography, service mode, and SLA threshold.")

        preset_names = ["None"] + sorted(presets.keys())
        selected_preset = st.selectbox("Saved filter sets", preset_names, index=0)
        preset = presets.get(selected_preset, {}) if selected_preset != "None" else {}

        date_min = pd.to_datetime(metadata["date_min"]).date()
        date_max = pd.to_datetime(metadata["date_max"]).date()
        default_start = pd.to_datetime(preset.get("start_date", date_min)).date() if preset else date_min
        default_end = pd.to_datetime(preset.get("end_date", date_max)).date() if preset else date_max
        date_range = st.date_input("Date Range Filter", value=(default_start, default_end), min_value=date_min, max_value=date_max)
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = end_date = date_range

        regions = metadata["regions"]
        selected_regions = st.multiselect("Region Selector", regions, default=preset.get("regions", regions))
        states_by_region = metadata.get("states_by_region", {})
        state_pool = sorted({state for region in (selected_regions or regions) for state in states_by_region.get(region, [])})
        selected_states = st.multiselect("State Selector", state_pool, default=preset.get("states", state_pool))
        ship_modes = metadata["ship_modes"]
        selected_modes = st.multiselect("Ship Mode Filter", ship_modes, default=preset.get("modes", ship_modes))
        delay_threshold = st.slider("Lead-Time Threshold Slider", min_value=1, max_value=30, value=int(preset.get("delay_threshold", 7)))

        st.markdown("### SLA Alerting")
        enable_alerting = st.checkbox("Enable SLA breach alerting", value=False)
        alert_threshold = st.slider("Alert delay-rate threshold", min_value=0.10, max_value=1.00, value=0.40, step=0.05)
        st.session_state["dashboard_alerting"] = {"enabled": enable_alerting, "threshold": alert_threshold}

        preset_name = st.text_input("Preset name", value="")
        if st.button("Save current filters", use_container_width=True) and preset_name.strip():
            presets[preset_name.strip()] = {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "regions": selected_regions,
                "states": selected_states,
                "modes": selected_modes,
                "delay_threshold": delay_threshold,
            }
            _save_presets(presets)
            st.success(f"Saved preset: {preset_name.strip()}")

        st.markdown("---")
        st.caption(describe_filters(start_date, end_date, selected_regions, selected_states, selected_modes))

    return {"start_date": start_date.isoformat(), "end_date": end_date.isoformat(), "regions": selected_regions, "states": selected_states, "modes": selected_modes, "delay_threshold": delay_threshold}


def _format_delta(current: float, previous: float) -> str:
    if previous == 0:
        return "No prior period"
    change = ((current - previous) / abs(previous)) * 100
    arrow = "?" if change > 0 else "?" if change < 0 else "?"
    return f"{arrow} {abs(change):.1f}% vs last period"



def render_kpi_intelligence(monthly: pd.DataFrame, route_summary: pd.DataFrame) -> None:
    render_section_heading("00", "KPI Intelligence Panel", "Executive snapshot of service performance, shipment volume, on-time delivery, and revenue exposure with period-over-period movement.")
    if monthly.empty:
        st.info("No KPI trend data available for the selected filters.")
        return
    monthly = monthly.sort_values("order_month").copy()
    if "on_time_rate" not in monthly.columns and "delay_rate" in monthly.columns:
        monthly["on_time_rate"] = 1 - monthly["delay_rate"]
    latest = monthly.iloc[-1]
    previous = monthly.iloc[-2] if len(monthly) > 1 else None
    spark = monthly.tail(6)
    revenue_impact = float(route_summary.get("revenue_at_risk", pd.Series(dtype=float)).sum())
    cards = [
        ("Avg Lead Time", f"{latest['avg_lead_time']:.1f} days", _format_delta(latest['avg_lead_time'], previous['avg_lead_time']) if previous is not None else "Latest period", spark['avg_lead_time'].tolist(), "#f59e0b"),
        ("Delay Rate", f"{latest['delay_rate']:.1%}", _format_delta(latest['delay_rate'], previous['delay_rate']) if previous is not None else "Latest period", spark['delay_rate'].tolist(), "#dc2626"),
        ("On-Time Delivery", f"{latest['on_time_rate']:.1%}", _format_delta(latest['on_time_rate'], previous['on_time_rate']) if previous is not None else "Latest period", spark['on_time_rate'].tolist(), "#0f62fe"),
        ("Total Shipments", f"{int(latest['shipments']):,}", _format_delta(latest['shipments'], previous['shipments']) if previous is not None else "Latest period", spark['shipments'].tolist(), "#38bdf8"),
        ("Revenue Impact", f"${revenue_impact:,.0f}", "Revenue currently exposed to delays", spark['shipments'].tolist(), "#9333ea"),
    ]
    for column, card in zip(st.columns(5), cards):
        with column:
            render_metric_sparkline(*card)


def render_dashboard_storyline(
    filtered_orders: pd.DataFrame,
    route_summary: pd.DataFrame,
    ship_mode_summary: pd.DataFrame,
) -> None:
    best_route = route_summary.sort_values(["efficiency_score", "shipments"], ascending=[False, False]).iloc[0]
    risk_route = route_summary.sort_values(["delay_rate", "avg_lead_time"], ascending=[False, False]).iloc[0]
    best_mode = ship_mode_summary.sort_values("avg_lead_time").iloc[0]
    revenue_risk = float(route_summary.get("revenue_at_risk", pd.Series(dtype=float)).sum())
    st.markdown(
        f"""
        <div class="premium-band">
            <div class="premium-story">
                <div class="premium-story-title">Executive Narrative</div>
                <div class="premium-story-copy">
                    Within the current filtered portfolio, <strong>{best_route['route_label']}</strong> stands out as the strongest-performing route,
                    while <strong>{risk_route['route_label']}</strong> remains the most material service-risk focus area.
                    <strong>{best_mode['ship_mode']}</strong> is currently the fastest observed shipping mode,
                    and the network is carrying an estimated <strong>${revenue_risk:,.0f}</strong> in revenue exposure associated with delay behaviour.
                </div>
                <div class="premium-kpi-row">
                    <div class="premium-mini">
                        <div class="premium-mini-label">Portfolio Lead Time</div>
                        <div class="premium-mini-value">{filtered_orders['lead_time_days'].mean():.1f} days</div>
                    </div>
                    <div class="premium-mini">
                        <div class="premium-mini-label">Delay Pressure</div>
                        <div class="premium-mini-value">{filtered_orders['delay_flag'].mean():.1%}</div>
                    </div>
                    <div class="premium-mini">
                        <div class="premium-mini-label">Fastest Mode</div>
                        <div class="premium-mini-value">{best_mode['ship_mode']}</div>
                    </div>
                </div>
            </div>
            <div class="premium-highlight">
                <div class="premium-highlight-title">Operational Focus</div>
                <div class="premium-highlight-copy">
                    Start with the KPI panel to understand portfolio movement, move through route and map diagnostics
                    to isolate pressure points, and then use drill-down and prediction to evaluate corrective actions
                    with quantified confidence, scenario comparison, and explainable model outputs.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_comparative_analysis(comparative_payload: dict[str, pd.DataFrame]) -> None:
    render_section_heading("01A", "Comparative Analysis", "Compare multiple regions, factories, or ship modes at once with richer metric breakdowns, ranked insights, and an executive-ready donut view.")
    compare_map = {"Region Comparison": ("region", "region"), "Factory Comparison": ("factory", "factory"), "Ship Mode Comparison": ("ship_mode", "ship_mode")}
    metric_map = {"Average Lead Time": "avg_lead_time", "Delay Rate": "delay_rate", "Average Sales": "avg_sales", "Shipment Volume": "shipments"}
    compare_label = st.selectbox("Comparison Type", list(compare_map.keys()))
    payload_key, column = compare_map[compare_label]
    compare_df = comparative_payload.get(payload_key, pd.DataFrame()).copy()
    if compare_df.empty:
        st.info("No comparison data available for the current filters.")
        return
    metric_label = st.selectbox("Primary Comparison Metric", list(metric_map.keys()))
    metric_column = metric_map[metric_label]
    options = sorted(compare_df[column].dropna().astype(str).unique().tolist())
    default_selection = options[: min(4, len(options))]
    selected_values = st.multiselect("Selections", options, default=default_selection, help="Choose up to 4 items for a focused comparison.")
    if len(selected_values) < 2:
        st.info("Select at least two items to compare meaningfully.")
        return
    compare_df = compare_df[compare_df[column].isin(selected_values)].copy()
    compare_df = compare_df.sort_values(metric_column, ascending=(metric_column not in {"shipments", "avg_sales", "total_sales"}))
    best_row = compare_df.iloc[0]
    worst_row = compare_df.iloc[-1]
    if metric_column in {"avg_lead_time", "delay_rate"}:
        pct_diff = ((worst_row[metric_column] - best_row[metric_column]) / max(abs(worst_row[metric_column]), 1e-9)) * 100
        summary_text = f"{best_row[column]} performs best on {metric_label.lower()} and is {abs(pct_diff):.1f}% better than {worst_row[column]}."
    else:
        pct_diff = ((best_row[metric_column] - worst_row[metric_column]) / max(abs(worst_row[metric_column]), 1e-9)) * 100
        summary_text = f"{best_row[column]} leads on {metric_label.lower()} by {abs(pct_diff):.1f}% compared with {worst_row[column]}."
    st.info(summary_text)
    k1, k2, k3 = st.columns(3)
    with k1:
        render_metric("Best Performer", str(best_row[column]), f"Strongest {metric_label.lower()} in the selected comparison set.")
    with k2:
        render_metric("Comparison Spread", f"{abs(pct_diff):.1f}%", f"Gap between {best_row[column]} and {worst_row[column]}.")
    with k3:
        render_metric("Items Compared", f"{len(compare_df)}", "Number of selected entities in this comparison.")
    pie_values = compare_df[metric_column].copy()
    if metric_column in {"avg_lead_time", "delay_rate"}:
        pie_values = pie_values.max() - pie_values + (0.01 if metric_column == "delay_rate" else 1.0)
    donut = go.Figure(data=[go.Pie(labels=compare_df[column], values=pie_values, hole=0.58, sort=False, pull=[0.08 if i == 0 else 0.02 for i in range(len(compare_df))], marker=dict(colors=["#0f62fe", "#38bdf8", "#f59e0b", "#dc2626"][: len(compare_df)], line=dict(color="rgba(255,255,255,0.55)", width=2)), customdata=compare_df[["shipments", "avg_lead_time", "delay_rate", "avg_sales", "total_sales"]], hovertemplate=("<b>%{label}</b><br>" + f"{metric_label}: %{{value:.2f}}<br>" + "Shipments: %{customdata[0]}<br>Avg lead time: %{customdata[1]:.1f} days<br>Delay rate: %{customdata[2]:.1%}<br>Avg sales: %{customdata[3]:.2f}<br>Total sales: %{customdata[4]:.2f}<extra></extra>"), texttemplate="%{label}<br>%{percent}", textposition="outside")])
    donut.add_annotation(text=f"<b>{metric_label}</b><br>Comparison Mix", x=0.5, y=0.5, showarrow=False, font=dict(size=17))
    donut.update_layout(title=f"{compare_label} Donut Analysis")
    c1, c2 = st.columns([0.58, 0.42])
    with c1:
        st.plotly_chart(base_layout(donut, 520), use_container_width=True)
    with c2:
        st.dataframe(compare_df, use_container_width=True, hide_index=True)


def render_alert_system(route_summary: pd.DataFrame, factory_summary: pd.DataFrame, monthly_trend: pd.DataFrame) -> None:
    render_section_heading("01B", "Alert System", "Highlights high-delay routes, sudden lead-time spikes, and underperforming factories using warning badges and red-flag messaging.")
    high_delay = route_summary.sort_values(["delay_rate", "shipments"], ascending=[False, False]).head(3)
    factory_watch = factory_summary.sort_values(["delay_rate", "avg_lead_time"], ascending=[False, False]).head(3)
    spike_message = "No significant lead-time spike detected."
    if len(monthly_trend) >= 2:
        latest = float(monthly_trend.iloc[-1]["avg_lead_time"])
        previous_avg = float(monthly_trend.iloc[:-1]["avg_lead_time"].mean())
        if previous_avg and latest > previous_avg * 1.1:
            spike_message = f"Red flag: latest period lead time is {((latest - previous_avg) / previous_avg) * 100:.1f}% above the prior average."
    a, b, c = st.columns(3)
    with a:
        st.error("High Delay Routes")
        st.dataframe(high_delay[["route_label", "delay_rate", "avg_lead_time", "shipments"]], use_container_width=True, hide_index=True)
    with b:
        st.warning("Lead-Time Spike")
        st.write(spike_message)
    with c:
        st.error("Underperforming Factories")
        st.dataframe(factory_watch[["factory", "shipments", "avg_lead_time", "delay_rate"]], use_container_width=True, hide_index=True)
    alerting = st.session_state.get("dashboard_alerting", {"enabled": False, "threshold": 0.4})
    if alerting.get("enabled"):
        if st.button("Trigger SLA breach alert now", use_container_width=True):
            result = trigger_sla_alerts(route_summary, float(alerting.get("threshold", 0.4)))
            if result.get("sent"):
                st.success(f"Alert processed for {result['count']} route(s). {result['message']}")
            else:
                st.info(result.get("message", "No alert was sent."))


def build_anomaly_explainer(anomalies: pd.DataFrame, route_summary: pd.DataFrame, filtered_orders: pd.DataFrame) -> pd.DataFrame:
    route_context = route_summary[["route_label", "shipments"]].rename(columns={"shipments": "route_shipments"})
    distance_context = filtered_orders.groupby("route_label", as_index=False).agg(
        avg_distance=("dest_lat", "count")
    )
    explained = anomalies.merge(route_context, on="route_label", how="left")
    explained = explained.merge(distance_context, on="route_label", how="left")

    reasons = []
    for _, row in explained.iterrows():
        parts = []
        if row.get("lead_time_zscore", 0) >= 2:
            parts.append("extreme lead-time deviation")
        if row.get("route_shipments", 9999) <= 5:
            parts.append("low route frequency")
        if str(row.get("ship_mode", "")).lower() in {"standard class", "second class"}:
            parts.append("slower shipping mode")
        reasons.append(" + ".join(parts) if parts else "combined operational variation")
    explained["anomaly_reason"] = reasons
    return explained


def render_route_overview(route_summary: pd.DataFrame) -> None:
    render_section_heading(
        "01",
        "Route Efficiency Overview",
        "Average lead time by route, a route performance leaderboard, and pinned route comparison.",
    )

    top10, bottom10 = top_and_bottom_routes(route_summary, size=10)
    col1, col2 = st.columns(2)
    fig_lead = px.bar(
        route_summary.head(15).sort_values("avg_lead_time", ascending=False),
        x="avg_lead_time",
        y="route_label",
        orientation="h",
        color="lead_time_std",
        color_continuous_scale=["#2563eb", "#d28a54", "#a64b2a"],
        title="Average Lead Time by Route",
        labels={"avg_lead_time": "Average Lead Time (days)", "route_label": "Route"},
        hover_data={"shipments": True, "lead_time_std": ":.1f"},
    )
    with col1:
        st.plotly_chart(base_layout(fig_lead, 500), use_container_width=True)

    leaderboard = top10.copy()
    leaderboard.insert(0, "rank", range(1, len(leaderboard) + 1))
    with col2:
        st.markdown("### Route Performance Leaderboard")
        st.dataframe(
            leaderboard[["rank", "route_label", "shipments", "avg_lead_time", "delay_rate", "efficiency_score"]],
            use_container_width=True,
            hide_index=True,
        )

    t1, t2, t3 = st.tabs(["Top 10 Most Efficient Routes", "Bottom 10 Least Efficient Routes", "Pinned Route Comparison"])
    with t1:
        st.dataframe(
            top10[["route_label", "factory", "state", "region", "shipments", "avg_lead_time", "lead_time_std", "efficiency_score"]],
            use_container_width=True,
            hide_index=True,
        )
    with t2:
        st.dataframe(
            bottom10[["route_label", "factory", "state", "region", "shipments", "avg_lead_time", "lead_time_std", "efficiency_score"]],
            use_container_width=True,
            hide_index=True,
        )
    with t3:
        route_options = route_summary["route_label"].tolist()
        selected_routes = st.multiselect("Select two routes", route_options, default=route_options[:2], max_selections=2)
        if len(selected_routes) == 2:
            comparison = route_summary[route_summary["route_label"].isin(selected_routes)][["route_label", "avg_lead_time", "delay_rate", "avg_cost", "efficiency_score", "shipments"]]
            st.dataframe(comparison, use_container_width=True, hide_index=True)
            faster = comparison.sort_values("avg_lead_time").iloc[0]
            slower = comparison.sort_values("avg_lead_time").iloc[-1]
            difference = ((slower["avg_lead_time"] - faster["avg_lead_time"]) / max(slower["avg_lead_time"], 1e-9)) * 100
            st.info(f"{faster['route_label']} is {abs(difference):.1f}% faster than {slower['route_label']} in the filtered view.")
        else:
            st.info("Select exactly two routes to compare them side by side.")



def render_geography(
    state_summary: pd.DataFrame,
    region_bottlenecks: pd.DataFrame,
    state_bottlenecks: pd.DataFrame,
    filtered_orders: pd.DataFrame,
) -> None:
    render_section_heading(
        "02",
        "Geographic Shipping Map",
        "Shipping geography across both the United States and Canada, including heatmaps, route networks, and route clusters.",
    )

    destination_map = (
        filtered_orders.groupby(["state", "destination_country", "dest_lat", "dest_lon"], as_index=False)
        .agg(
            shipments=("order_id", "count"),
            avg_lead_time=("lead_time_days", "mean"),
            delay_rate=("delay_flag", "mean"),
        )
        .dropna(subset=["dest_lat", "dest_lon"])
    )
    route_network = (
        filtered_orders.groupby(
            [
                "factory",
                "state",
                "factory_lat",
                "factory_lon",
                "dest_lat",
                "dest_lon",
                "destination_country",
            ],
            as_index=False,
        )
        .agg(
            shipments=("order_id", "count"),
            avg_lead_time=("lead_time_days", "mean"),
        )
        .dropna(subset=["factory_lat", "factory_lon", "dest_lat", "dest_lon"])
        .sort_values(["shipments", "avg_lead_time"], ascending=[False, False])
    )
    factories = filtered_orders[["factory", "factory_lat", "factory_lon"]].dropna().drop_duplicates().sort_values("factory")
    canada_summary = build_canada_analytics(filtered_orders)
    route_clusters = build_route_clusters(build_route_summary(filtered_orders))

    usa_tab, canada_tab, cluster_tab = st.tabs(["USA", "Canada", "Route Clusters"])
    with usa_tab:
        usa_destinations = destination_map[destination_map["destination_country"] == "United States"].copy()
        usa_routes = route_network[route_network["destination_country"] == "United States"].head(80).copy()
        col1, col2 = st.columns([1.25, 0.75])
        with col1:
            st.markdown("### USA Heatmap")
            render_folium_map(build_heatmap_leaflet(usa_destinations), height=560)
        with col2:
            fig_region = px.bar(
                region_bottlenecks.sort_values("avg_lead_time"),
                x="avg_lead_time",
                y="region",
                orientation="h",
                color="shipments",
                color_continuous_scale=["#dbeafe", "#2563eb"],
                title="Regional Bottleneck Visualization",
                labels={"avg_lead_time": "Average Lead Time (days)", "region": "Region"},
                hover_data={"delay_rate": ":.1%"},
            )
            st.plotly_chart(base_layout(fig_region, 560), use_container_width=True)
        st.markdown("### USA Network")
        render_folium_map(build_network_leaflet(usa_routes, factories, usa_destinations, "USA"), height=620)

    with canada_tab:
        canada_destinations = destination_map[destination_map["destination_country"] == "Canada"].copy()
        col1, col2 = st.columns([1.2, 0.8])
        with col1:
            st.markdown("### Canada Province Bottleneck Map")
            render_folium_map(build_volume_bottleneck_leaflet(canada_destinations), height=620)
        with col2:
            st.markdown("### Canada Analytics")
            st.dataframe(canada_summary, use_container_width=True, hide_index=True)

    with cluster_tab:
        st.markdown("### Route Archetype Clusters")
        if route_clusters.empty:
            st.info("Not enough route diversity for clustering in the current filter selection.")
        else:
            fig_cluster = px.scatter(
                route_clusters,
                x="avg_lead_time",
                y="delay_rate",
                size="shipments",
                color="cluster_name",
                hover_name="route_label",
                title="Shipment Clustering by Route Performance",
            )
            fig_cluster.update_yaxes(tickformat=".0%")
            st.plotly_chart(base_layout(fig_cluster, 500), use_container_width=True)
            st.dataframe(
                route_clusters[["route_label", "cluster_name", "shipments", "avg_lead_time", "delay_rate"]],
                use_container_width=True,
                hide_index=True,
            )

    a, b = st.columns(2)
    with a:
        st.markdown("### Congestion-Prone Regions")
        st.dataframe(region_bottlenecks, use_container_width=True, hide_index=True)
    with b:
        st.markdown("### Congestion-Prone States")
        st.dataframe(state_bottlenecks.head(10), use_container_width=True, hide_index=True)



def render_ship_modes(ship_mode_summary: pd.DataFrame, shipping_category_summary: pd.DataFrame) -> None:
    render_section_heading(
        "03",
        "Ship Mode Comparison",
        "Lead time comparison by shipping method with a supporting cost-time view to explain operational trade-offs.",
    )

    col1, col2 = st.columns([1.05, 0.95])
    fig_modes = px.bar(
        ship_mode_summary,
        x="ship_mode",
        y="avg_lead_time",
        color="delay_rate",
        color_continuous_scale=["#2563eb", "#d28a54", "#a64b2a"],
        title="Lead Time Comparison by Shipping Method",
        labels={"avg_lead_time": "Average Lead Time (days)", "ship_mode": "Ship Mode"},
        hover_data={"shipments": True, "median_lead_time": ":.1f"},
    )
    with col1:
        st.plotly_chart(base_layout(fig_modes, 450), use_container_width=True)

    fig_tradeoff = px.scatter(
        shipping_category_summary,
        x="avg_cost",
        y="avg_lead_time",
        size="shipments",
        color="shipping_category",
        symbol="ship_mode",
        title="Cost-Time Trade-off (Standard vs Expedited)",
        labels={"avg_cost": "Average Cost", "avg_lead_time": "Average Lead Time (days)"},
    )
    with col2:
        st.plotly_chart(base_layout(fig_tradeoff, 450), use_container_width=True)

    x, y = st.columns(2)
    with x:
        st.markdown("### Lead Time by Shipping Method")
        st.dataframe(ship_mode_summary, use_container_width=True, hide_index=True)
    with y:
        st.markdown("### Standard vs Expedited Shipping")
        st.dataframe(shipping_category_summary, use_container_width=True, hide_index=True)


def render_drilldown(
    filtered_orders: pd.DataFrame,
    route_summary: pd.DataFrame,
    delay_threshold: int,
) -> None:
    render_section_heading(
        "04",
        "Route Drill-Down",
        "State-level performance insights, city-level breakdowns, and order-level shipment timelines driven by the selected route.",
    )

    state_view = build_state_bottlenecks(filtered_orders)

    top_col, top_table = st.columns([1.1, 0.9])
    fig_state = px.bar(
        state_view.head(12).sort_values("avg_lead_time"),
        x="avg_lead_time",
        y="state",
        orientation="h",
        color="delay_rate",
        color_continuous_scale=["#2563eb", "#d28a54", "#a64b2a"],
        title="State-Level Performance Insights",
        labels={"avg_lead_time": "Average Lead Time (days)", "state": "State"},
        hover_data={"shipments": True, "region": True, "total_sales": ":,.2f"},
    )
    with top_col:
        st.plotly_chart(base_layout(fig_state, 440), use_container_width=True)
    with top_table:
        st.dataframe(
            state_view.head(12)[["state", "region", "shipments", "avg_lead_time", "delay_rate", "bottleneck_score"]],
            use_container_width=True,
            hide_index=True,
        )

    selected_state = st.selectbox("High-volume state for city drill-down", state_view.head(12)["state"].tolist())
    city_summary = build_city_summary(filtered_orders, selected_state)
    st.markdown("### City-Level Performance Insights")
    st.dataframe(city_summary.head(15), use_container_width=True, hide_index=True)

    route_options = route_summary.sort_values(["efficiency_score", "shipments"], ascending=[False, False])["route_label"].tolist()
    selected_route = st.selectbox("Route Selector", route_options)
    route_orders = filtered_orders.loc[filtered_orders["route_label"] == selected_route].sort_values("order_date").copy()

    timeline = build_delay_timeline(route_orders, delay_threshold)
    timeline_daily = (
        timeline.assign(order_day=timeline["order_date"].dt.date)
        .groupby("order_day", as_index=False)
        .agg(
            avg_lead_time=("lead_time_days", "mean"),
            max_lead_time=("lead_time_days", "max"),
            shipments=("order_id", "count"),
            delayed_orders=("delay_flag", "sum"),
            total_sales=("sales", "sum"),
        )
        .sort_values("order_day")
    )

    fig_timeline = px.line(
        timeline_daily,
        x="order_day",
        y="avg_lead_time",
        markers=True,
        title="Average Lead Time Trend",
        labels={"order_day": "Order Date", "avg_lead_time": "Lead Time (days)"},
        hover_data={"shipments": True, "max_lead_time": ":.1f", "delayed_orders": True, "total_sales": ":,.2f"},
    )
    fig_timeline.update_traces(
        line=dict(color="#f1bb7b", width=3),
        marker=dict(size=8, color="#f1bb7b", line=dict(color="#ffffff", width=1)),
    )
    st.plotly_chart(base_layout(fig_timeline, 500), use_container_width=True)

    detail_columns = [
        "order_id",
        "order_date",
        "ship_date",
        "ship_mode",
        "customer_id",
        "city",
        "state",
        "region",
        "product_name",
        "sales",
        "units",
        "lead_time_days",
        "delay_flag",
    ]
    st.dataframe(
        route_orders[[column for column in detail_columns if column in route_orders.columns]],
        use_container_width=True,
        hide_index=True,
    )



def render_operational_intelligence(
    route_summary: pd.DataFrame,
    factory_summary: pd.DataFrame,
    monthly_trend: pd.DataFrame,
    route_concentration: pd.DataFrame,
    actions: list[str],
    sla_route: pd.DataFrame,
    sla_factory: pd.DataFrame,
    cost_risk: pd.DataFrame,
    profitability: pd.DataFrame,
    seasonality: pd.DataFrame,
    forecast: pd.DataFrame,
    transition_payload: dict,
    anomalies: pd.DataFrame,
) -> None:
    render_section_heading("05", "Operational Intelligence", "Extra analytics for executive decisions: trends, forecasting, SLA compliance, profitability, concentration risk, anomaly detection, and recommended actions.")
    col1, col2 = st.columns(2)
    with col1:
        fig_trend = px.line(monthly_trend, x="order_month", y=["avg_lead_time", "rolling_avg_lead_time"], markers=True, title="Monthly Lead-Time Trend", labels={"order_month": "Order Month", "value": "Lead Time (days)", "variable": "Metric"})
        st.plotly_chart(base_layout(fig_trend, 420), use_container_width=True)
    with col2:
        fig_factory = px.bar(factory_summary.sort_values("performance_score"), x="performance_score", y="factory", orientation="h", color="avg_lead_time", color_continuous_scale=["#2563eb", "#38bdf8", "#f59e0b"], title="Factory Performance Score", labels={"performance_score": "Performance Score", "factory": "Factory"}, hover_data={"shipments": True, "delay_rate": ":.1%", "states_served": True})
        st.plotly_chart(base_layout(fig_factory, 420), use_container_width=True)
    col3, col4 = st.columns(2)
    with col3:
        fig_concentration = px.area(route_concentration.head(30), x="route_label", y="cumulative_share", title="Route Concentration Curve", labels={"route_label": "Route", "cumulative_share": "Cumulative Shipment Share"})
        fig_concentration.update_yaxes(tickformat=".0%")
        fig_concentration.update_xaxes(showticklabels=False)
        st.plotly_chart(base_layout(fig_concentration, 380), use_container_width=True)
        st.dataframe(route_concentration[["route_label", "factory", "shipment_share", "dominant_factory_share", "concentration_risk"]].head(12), use_container_width=True, hide_index=True)
    with col4:
        st.markdown("### Recommended Actions")
        for action in actions:
            st.write(f"- {action}")
        html_report = "<html><body><h1>Executive Summary</h1><ul>" + "".join(f"<li>{item}</li>" for item in actions) + "</ul></body></html>"
        st.download_button("Download Executive Summary HTML", html_report.encode("utf-8"), "executive_summary.html", "text/html", use_container_width=True)
    tabs = st.tabs(["SLA Compliance", "Cost Saving", "Profitability", "Seasonality & Forecast", "Transition Matrix", "Anomaly Watchlist"])
    with tabs[0]:
        left, right = st.columns(2)
        with left:
            st.markdown("### Route SLA Scorecard")
            st.dataframe(sla_route.head(20), use_container_width=True, hide_index=True)
        with right:
            st.markdown("### Factory SLA Scorecard")
            st.dataframe(sla_factory.head(20), use_container_width=True, hide_index=True)
    with tabs[1]:
        st.dataframe(cost_risk[["route_label", "shipments", "delay_rate", "revenue_at_risk", "recovered_revenue_10pct", "recovered_revenue_25pct"]].head(20), use_container_width=True, hide_index=True)
    with tabs[2]:
        st.dataframe(profitability[["route_label", "total_gross_profit", "profit_per_shipment", "gross_margin", "avg_lead_time", "delay_rate", "profitability_risk_score"]].head(20), use_container_width=True, hide_index=True)
    with tabs[3]:
        p1, p2 = st.columns(2)
        with p1:
            if not seasonality.empty:
                fig_season = px.line(seasonality, x="order_month", y=["trend_component", "seasonal_component", "residual_component"], title=f"Seasonality Decomposition ({seasonality['method'].iloc[0]})")
                st.plotly_chart(base_layout(fig_season, 420), use_container_width=True)
        with p2:
            if not forecast.empty:
                fig_forecast = px.line(forecast, x="ds", y=["forecast_lead_time", "forecast_delay_rate"], title=f"Forward Forecast ({forecast['model_type'].iloc[0]})")
                st.plotly_chart(base_layout(fig_forecast, 420), use_container_width=True)
    with tabs[4]:
        if transition_payload.get("values"):
            fig_transition = px.imshow(transition_payload["values"], x=transition_payload["columns"], y=transition_payload["index"], aspect="auto", color_continuous_scale=["#dbeafe", "#2563eb"], title="Route ? Ship Mode Transition Matrix")
            st.plotly_chart(base_layout(fig_transition, 520), use_container_width=True)
    with tabs[5]:
        if not anomalies.empty:
            columns = [column for column in ["order_id", "route_label", "factory", "state", "region", "ship_mode", "lead_time_days", "lead_time_zscore", "severity", "anomaly_model", "anomaly_reason"] if column in anomalies.columns]
            st.dataframe(anomalies[columns].head(20), use_container_width=True, hide_index=True)


def render_dashboard_page(dataset_ref: dict) -> None:
    filters = render_filters(dataset_ref)
    overview = get_dashboard_overview(dataset_ref["dataset_id"], json.dumps(filters, sort_keys=True))
    filtered_orders = records_to_frame(overview["filtered_orders"])
    delay_threshold = int(overview["meta"]["delay_threshold"])
    if filtered_orders.empty:
        st.error("No records match the current filters. Widen the date, region, state, or ship mode selection.")
        return
    route_summary = records_to_frame(overview["route_summary"])
    state_summary = records_to_frame(overview["state_summary"])
    ship_mode_summary = records_to_frame(overview["ship_mode_summary"])
    shipping_category_summary = records_to_frame(overview["shipping_category_summary"])
    region_bottlenecks = records_to_frame(overview["region_bottlenecks"])
    state_bottlenecks = records_to_frame(overview["state_bottlenecks"])
    factory_summary = records_to_frame(overview["factory_summary"])
    monthly_trend = records_to_frame(overview["monthly_trend"])
    route_concentration = records_to_frame(overview["route_concentration"])
    sla_route = records_to_frame(overview["sla_route"])
    sla_factory = records_to_frame(overview["sla_factory"])
    cost_risk = records_to_frame(overview["cost_risk"])
    profitability = records_to_frame(overview["profitability"])
    seasonality = records_to_frame(overview["seasonality"])
    forecast = records_to_frame(overview["forecast"])
    anomalies = records_to_frame(overview["anomalies"])
    comparative = {key: records_to_frame(value) for key, value in overview["comparative"].items()}

    render_header(overview["meta"]["data_source"], overview["meta"]["validation_messages"], int(overview["meta"]["filtered_count"]))
    render_dashboard_storyline(filtered_orders, route_summary, ship_mode_summary)
    render_kpi_intelligence(monthly_trend, route_summary)
    render_route_overview(route_summary)
    render_comparative_analysis(comparative)
    render_alert_system(route_summary, factory_summary, monthly_trend)
    render_geography(state_summary, region_bottlenecks, state_bottlenecks, filtered_orders)
    render_ship_modes(ship_mode_summary, shipping_category_summary)
    render_drilldown(filtered_orders, route_summary, delay_threshold)
    render_operational_intelligence(route_summary, factory_summary, monthly_trend, route_concentration, overview["actions"], sla_route, sla_factory, cost_risk, profitability, seasonality, forecast, overview["transition"], anomalies)
