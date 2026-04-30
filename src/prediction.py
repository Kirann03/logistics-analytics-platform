from __future__ import annotations

from datetime import date
import json
import os
import re
from urllib import request as urllib_request

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.api_client import get_prediction_options as api_get_prediction_options, get_prediction_performance, infer_prediction
from src.common import base_layout, render_metric, render_section_heading

def get_prediction_options(dataset_id: str) -> dict[str, list[str]]:
    return api_get_prediction_options(dataset_id)


def estimate_shipping_cost(prediction: dict) -> float:
    return float(prediction.get("estimated_cost", 0.0))


def predict_shipment(dataset_id: str, inputs: dict, delay_threshold: int | None = None) -> dict:
    payload = dict(inputs)
    payload["order_date"] = str(payload["order_date"])
    if delay_threshold is not None:
        payload["delay_threshold"] = int(delay_threshold)
    return infer_prediction(dataset_id, payload)


def make_recommendations(prediction: dict) -> list[str]:
    return prediction.get("recommendations", [])


def prediction_alert(prediction: dict) -> tuple[str, str]:
    if prediction["risk"] == "High":
        return "High risk warning", "Critical delay alert: review ship mode, priority, route, and shipment size before approval."
    if prediction["risk"] == "Medium":
        return "Medium risk warning", "Suggested action: compare an expedited scenario before confirming shipment."
    return "Low risk", "Shipment appears acceptable based on current historical patterns."


def risk_color(risk: str) -> str:
    return {"Low": "#0f62fe", "Medium": "#f59e0b", "High": "#dc2626"}.get(risk, "#0f62fe")


def build_mode_decision_table(dataset_id: str, options: dict, inputs: dict, delay_threshold: int) -> pd.DataFrame:
    rows = []
    for mode in options["ship_modes"]:
        result = predict_shipment(dataset_id, {**inputs, "ship_mode": mode}, delay_threshold)
        rows.append({
            "ship_mode": mode,
            "risk": result["risk"],
            "delay_probability": result["delay_probability"],
            "expected_lead_time": result["expected_lead_time"],
            "estimated_cost": result["estimated_cost"],
            "confidence": result["confidence"],
            "speed_rank": 0,
            "cost_rank": 0,
        })
    table = pd.DataFrame(rows)
    table["speed_rank"] = table["expected_lead_time"].rank(method="dense")
    table["cost_rank"] = table["estimated_cost"].rank(method="dense")
    table["decision_score"] = ((1 - table["delay_probability"]) * 45 + (1 - table["speed_rank"] / table["speed_rank"].max()) * 30 + (1 - table["cost_rank"] / table["cost_rank"].max()) * 25).round(1)
    return table.sort_values("decision_score", ascending=False)


def build_priority_sensitivity(dataset_id: str, inputs: dict, delay_threshold: int) -> pd.DataFrame:
    rows = []
    for priority in ["Economy", "Standard", "Expedited", "Critical"]:
        result = predict_shipment(dataset_id, {**inputs, "priority": priority}, delay_threshold)
        rows.append({
            "priority": priority,
            "delay_probability": result["delay_probability"],
            "expected_lead_time": result["expected_lead_time"],
            "estimated_cost": result["estimated_cost"],
            "risk": result["risk"],
        })
    return pd.DataFrame(rows)


def calculate_route_benchmark(prediction: dict) -> dict:
    return prediction.get("route_benchmark", {})


def parse_nlp_shipment(text: str, options: dict) -> dict:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        payload = {
            "model": os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
            "max_tokens": 300,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Extract logistics shipment fields as JSON with keys region, state, ship_mode, units, priority, distance, factory. "
                        "Return only JSON. Allowed options: "
                        + json.dumps(options)
                        + " Shipment text: "
                        + text
                    ),
                }
            ],
        }
        try:
            req = urllib_request.Request(
                "https://api.anthropic.com/v1/messages",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                method="POST",
            )
            with urllib_request.urlopen(req, timeout=20) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            content = response_payload.get("content", [])
            if content:
                extracted = json.loads(content[0].get("text", "{}"))
                return {key: value for key, value in extracted.items() if value not in (None, "", [])}
        except Exception:
            pass

    parsed = {}
    lower = text.lower()
    for ship_mode in options["ship_modes"]:
        if ship_mode.lower() in lower:
            parsed["ship_mode"] = ship_mode
    for region in options["regions"]:
        if region.lower() in lower:
            parsed["region"] = region
    for state in options["states"]:
        if state.lower() in lower:
            parsed["state"] = state
    for factory in options["factories"]:
        if factory.lower().replace("\u2019", "'") in lower.replace("\u2019", "'"):
            parsed["factory"] = factory
    unit_match = re.search(r"(\d+)\s*(units|unit|pcs|pieces)", lower)
    if unit_match:
        parsed["units"] = int(unit_match.group(1))
    distance_match = re.search(r"(\d+)\s*(miles|mile|mi|km)", lower)
    if distance_match:
        parsed["distance"] = float(distance_match.group(1))
    if "critical" in lower:
        parsed["priority"] = "Critical"
    elif "expedited" in lower or "urgent" in lower:
        parsed["priority"] = "Expedited"
    elif "economy" in lower:
        parsed["priority"] = "Economy"
    return parsed



def render_decision_card(label: str, value: str, note: str, pill: str | None = None) -> None:
    pill_html = f"<span class='risk-pill' style='background:{risk_color(pill)}'>{pill}</span>" if pill else value
    st.markdown(
        f"""
        <div class="decision-card">
            <div class="decision-label">{label}</div>
            <div class="decision-value">{pill_html}</div>
            <div class="decision-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_decision_cockpit(
    inputs: dict,
    prediction: dict,
    sla_days: int,
    compact: bool = False,
) -> None:
    benchmark = calculate_route_benchmark(prediction)
    sla_gap = prediction["expected_lead_time"] - sla_days
    approval = "Approve" if prediction["risk"] == "Low" and sla_gap <= 0 else "Review" if prediction["risk"] != "High" else "Escalate"
    cards = [
        ("Decision Status", approval, "Auto-generated from risk and SLA fit.", None),
        ("Risk Level", prediction["risk"], "Operational delay classification.", prediction["risk"]),
        ("SLA Gap", f"{sla_gap:+.1f} days", f"Target SLA is {sla_days} days.", None),
        ("Route Percentile", f"{benchmark.get('percentile', 0):.0f}th", f"Compared with {benchmark.get('similar_cases', 0):,} similar shipments.", None),
    ]
    columns = [*st.columns(4)] if not compact else [*st.columns(2), *st.columns(2)]
    for column, (label, value, note, pill) in zip(columns, cards):
        with column:
            render_decision_card(label, value, note, pill)


def render_advanced_panels(dataset_id: str, options: dict, inputs: dict, prediction: dict) -> None:
    mode_table = build_mode_decision_table(dataset_id, options, inputs, prediction["delay_threshold"])
    priority_table = build_priority_sensitivity(dataset_id, inputs, prediction["delay_threshold"])
    benchmark = calculate_route_benchmark(prediction)

    st.markdown("""
        <div class="panel-shell">
            <div class="panel-title">Advanced Decision Analytics</div>
            <div class="panel-copy">
                Compare shipping modes on cost, lead time, and delay behavior, then inspect how service priority
                changes the risk curve for the same shipment profile.
            </div>
        </div>
        """, unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        fig_mode = px.scatter(mode_table, x="estimated_cost", y="expected_lead_time", size="decision_score", color="risk", hover_name="ship_mode", title="Cost vs Speed Decision Matrix", labels={"estimated_cost": "Estimated Cost", "expected_lead_time": "Expected Lead Time"}, color_discrete_map={"Low": "#0f62fe", "Medium": "#f59e0b", "High": "#dc2626"})
        fig_mode.update_traces(marker=dict(opacity=0.88, line=dict(color="rgba(255,255,255,0.36)", width=1.2)))
        st.markdown('<div class="panel-shell tight"><div class="panel-title">Cost vs Speed Decision Matrix</div></div>', unsafe_allow_html=True)
        st.plotly_chart(base_layout(fig_mode, 460), use_container_width=True)
    with col2:
        fig_priority = px.line(priority_table, x="priority", y="delay_probability", markers=True, title="Priority Upgrade Sensitivity", labels={"delay_probability": "Delay Probability", "priority": "Shipment Priority"})
        fig_priority.update_yaxes(tickformat=".0%")
        fig_priority.update_traces(line=dict(width=4, color="#60a5fa"), marker=dict(size=9))
        st.markdown('<div class="panel-shell tight"><div class="panel-title">Priority Upgrade Sensitivity</div></div>', unsafe_allow_html=True)
        st.plotly_chart(base_layout(fig_priority, 460), use_container_width=True)
    st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)
    col3, col4 = st.columns([0.6, 0.4], gap="large")
    with col3:
        st.markdown('<div class="panel-shell tight"><div class="panel-title">Shipping Mode Decision Table</div><div class="prediction-note">Ranked summary of mode choices using risk, speed, cost, confidence, and decision score.</div></div>', unsafe_allow_html=True)
        st.dataframe(mode_table[["ship_mode", "risk", "delay_probability", "expected_lead_time", "estimated_cost", "confidence", "decision_score"]], use_container_width=True, hide_index=True)
    with col4:
        st.markdown('<div class="panel-shell tight"><div class="panel-title">Route Benchmark</div><div class="prediction-note">Historical context for matching shipments on this destination and service pattern.</div></div>', unsafe_allow_html=True)
        render_metric("Similar Cases", f"{benchmark.get('similar_cases',0):,}", "Historical shipments matched to this scenario.")
        render_metric("Historical Average", f"{benchmark.get('historical_avg',0):.1f} days", "Average lead time across similar cases.")
        render_metric("Historical Range", f"{benchmark.get('historical_best',0):.0f} - {benchmark.get('historical_worst',0):.0f} days", "Observed best-to-worst historical delivery window.")


def render_risk_gauge(prediction: dict) -> None:
    probability = prediction["delay_probability"] * 100
    color = "#2563eb" if prediction["risk"] == "Low" else "#d28a54" if prediction["risk"] == "Medium" else "#c4683c"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability,
            number={"suffix": "%"},
            title={"text": "Delay Probability", "font": {"size": 22, "color": "#f3efe7"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 35], "color": "rgba(37,99,235,0.52)"},
                    {"range": [35, 65], "color": "rgba(210,138,84,0.48)"},
                    {"range": [65, 100], "color": "rgba(196,104,60,0.50)"},
                ],
                "threshold": {
                    "line": {"color": "#f3efe7", "width": 3},
                    "thickness": 0.8,
                    "value": probability,
                },
            },
        )
    )
    fig.update_traces(number={"font": {"size": 54, "color": "#f3efe7"}})
    st.plotly_chart(base_layout(fig, 300), use_container_width=True)


def render_model_performance(dataset_id: str) -> None:
    metrics = get_prediction_performance(dataset_id)
    st.markdown("### Model Performance")
    top = st.columns(6)
    cards = [
        ("Lead-Time MAE", f"{metrics['mae']:.1f} days", "Average absolute prediction error on holdout data."),
        ("Lead-Time R²", f"{metrics['r2']:.2f}", "Explained variance on unseen validation shipments."),
        ("Risk Accuracy", f"{metrics['accuracy']:.1%}", "Classifier accuracy on holdout delay labels."),
        ("Risk Precision", f"{metrics['precision']:.1%}", "Share of predicted delays that were truly delayed."),
        ("Risk Recall", f"{metrics['recall']:.1%}", "Ability to catch delayed shipments. Higher is better here."),
        ("Risk AUC", f"{metrics['auc']:.2f}", "Ability to separate delayed vs safer shipments."),
    ]
    for column, card in zip(top, cards):
        with column:
            render_metric(*card)
    if any(key in metrics for key in ["cv_auc", "cv_f1", "cv_mae"]):
        bottom = st.columns(3)
        with bottom[0]:
            render_metric("CV AUC", f"{metrics.get('cv_auc', 0):.2f}", "Cross-validation AUC across folds.")
        with bottom[1]:
            render_metric("CV F1", f"{metrics.get('cv_f1', 0):.2f}", "Cross-validation F1 score across folds.")
        with bottom[2]:
            render_metric("CV MAE", f"{metrics.get('cv_mae', 0):.1f} days", "Cross-validation lead-time error across folds.")


def render_prediction_explainability(prediction: dict) -> None:
    interval_width = prediction.get("lead_time_upper", prediction["expected_lead_time"]) - prediction.get("lead_time_lower", prediction["expected_lead_time"])
    st.markdown("### Prediction Explainability")
    left, right = st.columns([0.58, 0.42], gap="large")
    with left:
        st.markdown(
            f"""
            <div class="panel-shell tight">
                <div class="panel-title">Lead-Time Confidence Interval</div>
                <div class="prediction-note">
                    Expected lead time is likely between <strong>{prediction.get('lead_time_lower', prediction['expected_lead_time']):.1f}</strong>
                    and <strong>{prediction.get('lead_time_upper', prediction['expected_lead_time']):.1f}</strong> days.
                    Interval width: <strong>{interval_width:.1f}</strong> days.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        shap_frame = prediction.get("shap_explanation", pd.DataFrame()).head(10).sort_values("shap_value")
        if not shap_frame.empty:
            fig_shap = px.bar(
                shap_frame,
                x="shap_value",
                y="feature",
                color="direction",
                orientation="h",
                hover_data=["feature_value"],
                title="Per-Prediction SHAP Drivers",
                labels={"shap_value": "Impact on Delay Risk", "feature": "Feature"},
                color_discrete_map={"Increase Risk": "#dc2626", "Reduce Risk": "#0f62fe"},
            )
            fig_shap.update_traces(marker_line_color="rgba(255,255,255,0.35)", marker_line_width=0.8)
            st.plotly_chart(base_layout(fig_shap, 420), use_container_width=True)

    with right:
        drift = prediction.get("drift_report", {})
        status = drift.get("status", "Low")
        drift_color = {"Low": "#0f62fe", "Medium": "#f59e0b", "High": "#dc2626"}.get(status, "#0f62fe")
        st.markdown(
            f"""
            <div class="panel-shell tight">
                <div class="panel-title">Drift Warning Panel</div>
                <div class="prediction-note">
                    Incoming shipment profile drift level:
                    <span class='risk-pill' style='background:{drift_color}'>{status}</span>
                    with drift score <strong>{drift.get('score', 0):.2f}</strong>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        alerts = pd.DataFrame(drift.get("numeric_alerts", []) + drift.get("categorical_alerts", []))
        if not alerts.empty:
            st.dataframe(alerts, use_container_width=True, hide_index=True)
        else:
            st.info("No strong drift signals were detected for the current shipment profile.")


def render_prediction_result(inputs: dict, prediction: dict, title: str = "Prediction Result") -> None:
    alert_title, alert_message = prediction_alert(prediction)
    st.markdown(f"### {title}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric("Delay Probability", f"{prediction['delay_probability']:.1%}", alert_title)
    with col2:
        render_metric("Risk Classification", prediction["risk"], "Low, Medium, or High delay risk.")
    with col3:
        render_metric(
            "Expected Lead Time",
            f"{prediction['expected_lead_time']:.1f} days",
            f"Estimated range: {prediction.get('lead_time_lower', prediction['expected_lead_time']):.1f} to {prediction.get('lead_time_upper', prediction['expected_lead_time']):.1f} days.",
        )
    with col4:
        render_metric("Confidence Score", f"{prediction['confidence']:.0f}%", "Based on matching historical sample size.")

    if prediction["risk"] == "High":
        st.error(alert_message)
    elif prediction["risk"] == "Medium":
        st.warning(alert_message)
    else:
        success_html = f'''<div style="background: linear-gradient(135deg, rgba(32, 122, 83, 0.95), rgba(20, 83, 45, 0.92)); color: #f6fff8; border: 1px solid rgba(134, 239, 172, 0.35); border-radius: 16px; padding: 1rem 1.2rem; font-weight: 700; box-shadow: 0 14px 26px rgba(0,0,0,0.22);">{alert_message}</div>'''
        st.markdown(success_html, unsafe_allow_html=True)

    left, right = st.columns([0.42, 0.58], gap="large")
    with left:
        st.markdown('<div class="panel-shell tight"><div class="panel-title">Risk Gauge</div><div class="prediction-note">Direct view of modeled delay probability for the current shipment inputs.</div></div>', unsafe_allow_html=True)
        render_risk_gauge(prediction)
    with right:
        importance = prediction["importance"].head(7)
        fig = px.bar(
            importance.sort_values("impact_percent"),
            x="impact_percent",
            y="feature",
            orientation="h",
            title="Feature Importance",
            labels={"impact_percent": "Impact (%)", "feature": "Feature"},
            color="impact_percent",
            color_continuous_scale=["#2563eb", "#d28a54", "#c4683c"],
        )
        fig.update_traces(marker_line_color="rgba(255,255,255,0.35)", marker_line_width=0.8, texttemplate="%{x:.1f}%", textposition="outside", cliponaxis=False)
        st.markdown('<div class="panel-shell tight"><div class="panel-title">Feature Importance</div><div class="prediction-note">Top factors driving the model output, ranked by relative impact.</div></div>', unsafe_allow_html=True)
        st.plotly_chart(base_layout(fig, 300), use_container_width=True)

    top_features = prediction["importance"].head(3)["feature"].tolist()
    explain_html = f'''<div style="background: linear-gradient(135deg, rgba(37, 99, 235, 0.92), rgba(30, 64, 175, 0.88)); color: #eff6ff; border: 1px solid rgba(147, 197, 253, 0.38); border-radius: 16px; padding: 1rem 1.2rem; font-weight: 650; line-height: 1.55; box-shadow: 0 14px 26px rgba(0,0,0,0.22);">Why this prediction happened: {prediction['risk']} risk is mainly influenced by {', '.join(top_features)}. Matched historical samples: route={prediction['samples']['route']}, state/mode={prediction['samples']['state_mode']}, state={prediction['samples']['state']}.</div>'''
    st.markdown(explain_html, unsafe_allow_html=True)

    recommendations = make_recommendations(prediction)
    st.markdown('<div class="panel-shell"><div class="panel-title">Recommendations Engine</div><div class="panel-copy">Suggested next actions based on historical route behavior, risk profile, shipment size, and service mode tradeoffs.</div></div>', unsafe_allow_html=True)
    rec_columns = st.columns(2)
    for index, recommendation in enumerate(recommendations, start=1):
        with rec_columns[(index - 1) % 2]:
            st.markdown(f'<div class="recommendation-card"><span>Recommendation {index}</span>{recommendation}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="cost-card"><div class="cost-label">Estimated Shipping Cost</div><div class="cost-value">${estimate_shipping_cost(prediction):.2f}</div></div>', unsafe_allow_html=True)
    render_prediction_explainability(prediction)


def scenario_inputs(
    label: str,
    options: dict,
    defaults: dict,
    key_prefix: str,
) -> dict:
    st.markdown(f"#### {label}")
    region = st.selectbox("Region", options["regions"], index=options["regions"].index(defaults["region"]) if defaults["region"] in options["regions"] else 0, key=f"{key_prefix}_region")
    state = st.selectbox("State", options["states"], index=options["states"].index(defaults["state"]) if defaults["state"] in options["states"] else 0, key=f"{key_prefix}_state")
    ship_mode = st.selectbox("Ship mode", options["ship_modes"], index=options["ship_modes"].index(defaults["ship_mode"]) if defaults["ship_mode"] in options["ship_modes"] else 0, key=f"{key_prefix}_mode")
    units = st.number_input("Units", min_value=1, max_value=1000, value=int(defaults["units"]), key=f"{key_prefix}_units")
    order_date = st.date_input("Order date", value=defaults["order_date"], key=f"{key_prefix}_date")
    priority = st.selectbox("Shipment priority", options["priorities"], index=options["priorities"].index(defaults["priority"]), key=f"{key_prefix}_priority")
    distance = st.number_input("Optional distance", min_value=0.0, value=float(defaults.get("distance", 0)), key=f"{key_prefix}_distance")
    factory = st.selectbox("Warehouse or factory", options["factories"], index=options["factories"].index(defaults["factory"]) if defaults["factory"] in options["factories"] else 0, key=f"{key_prefix}_factory")
    return {"region": region, "state": state, "ship_mode": ship_mode, "units": int(units), "order_date": order_date, "priority": priority, "distance": float(distance), "factory": factory}


def render_prediction_page(dataset_ref: dict) -> None:
    dataset_id = dataset_ref["dataset_id"]
    options = get_prediction_options(dataset_id)
    defaults = {"region": options["regions"][0], "state": options["states"][0], "ship_mode": options["ship_modes"][0], "units": int(max(1, options["defaults"]["median_units"])), "order_date": date.today(), "priority": "Standard", "distance": 0.0, "factory": options["factories"][0]}
    st.markdown("""<div class="hero-shell"><div class="eyebrow">Prediction Workspace</div><h1 class="hero-title">Delay Risk Prediction and Scenario Simulation</h1><div class="hero-copy">Predict delay probability, risk class, expected lead time, confidence, cost, recommendations, feature impact, similar historical cases, and batch shipment outcomes.</div></div>""", unsafe_allow_html=True)
    with st.expander("NLP Input", expanded=False):
        text_input = st.text_area("Type shipment details", placeholder="Example: 12 units to Ontario, Pacific region, Standard Class, urgent, from Sugar Shack, 900 miles")
        parsed = parse_nlp_shipment(text_input, options) if text_input else {}
        if parsed:
            st.success(f"Smart suggestions detected: {parsed}")
            defaults.update(parsed)
    st.markdown('<div class="panel-shell tight"><div class="panel-title">SLA Target</div><div class="prediction-note">Adjust the target lead time to test whether this shipment configuration is on track or needs escalation.</div></div>', unsafe_allow_html=True)
    sla_days = st.slider("Target SLA lead time", min_value=int(max(1, options["defaults"]["min_lead_time"])), max_value=int(options["defaults"]["max_lead_time"]), value=int(options["defaults"]["max_lead_time"] * 0.75), help="Used by the decision cockpit to classify SLA fit.")
    main_left, main_right = st.columns([0.48, 0.52], gap="large")
    with main_left:
        render_section_heading("05", "Input Controls", "Set shipment details, service priority, and destination context before running the prediction workspace.")
        input_col1, input_col2 = st.columns(2, gap="large")
        with input_col1:
            region = st.selectbox("Region", options["regions"], index=options["regions"].index(defaults["region"]) if defaults["region"] in options["regions"] else 0, key="primary_region")
            state = st.selectbox("State", options["states"], index=options["states"].index(defaults["state"]) if defaults["state"] in options["states"] else 0, key="primary_state")
            units = st.number_input("Units", min_value=1, max_value=1000, value=int(defaults["units"]), key="primary_units")
            order_date = st.date_input("Order date", value=defaults["order_date"], key="primary_date")
        with input_col2:
            ship_mode = st.selectbox("Ship mode", options["ship_modes"], index=options["ship_modes"].index(defaults["ship_mode"]) if defaults["ship_mode"] in options["ship_modes"] else 0, key="primary_mode")
            priority = st.selectbox("Shipment priority", options["priorities"], index=options["priorities"].index(defaults["priority"]), key="primary_priority")
            factory = st.selectbox("Warehouse or factory", options["factories"], index=options["factories"].index(defaults["factory"]) if defaults["factory"] in options["factories"] else 0, key="primary_factory")
            distance = st.number_input("Optional distance", min_value=0.0, value=float(defaults.get("distance", 0)), key="primary_distance")
        inputs = {"region": region, "state": state, "ship_mode": ship_mode, "units": int(units), "order_date": order_date, "priority": priority, "distance": float(distance), "factory": factory}
    prediction = predict_shipment(dataset_id, inputs)
    prediction["importance"] = pd.DataFrame(prediction.get("importance", []))
    prediction["shap_explanation"] = pd.DataFrame(prediction.get("shap_explanation", []))
    with main_right:
        st.markdown('<div class="section-shell"><div class="section-title">Overview</div><div class="section-copy">Immediate decision view of SLA fit, route standing, and modeled operational risk.</div></div>', unsafe_allow_html=True)
        render_decision_cockpit(inputs, prediction, sla_days, compact=True)
    render_model_performance(dataset_id)
    render_prediction_result(inputs, prediction)
    render_advanced_panels(dataset_id, options, inputs, prediction)
    render_section_heading("06", "Scenario Simulation", "Change ship mode or units and compare two shipment scenarios side by side.")
    scenario_a_defaults = inputs.copy()
    scenario_b_defaults = {**inputs, "ship_mode": options["ship_modes"][-1], "units": max(inputs["units"] + 5, 1)}
    sc1, sc2 = st.columns(2)
    with sc1:
        scenario_a = scenario_inputs("Scenario A", options, scenario_a_defaults, "scenario_a")
        prediction_a = predict_shipment(dataset_id, scenario_a, prediction["delay_threshold"])
    with sc2:
        scenario_b = scenario_inputs("Scenario B", options, scenario_b_defaults, "scenario_b")
        prediction_b = predict_shipment(dataset_id, scenario_b, prediction["delay_threshold"])
    comparison = pd.DataFrame([{"scenario": "A", "ship_mode": scenario_a["ship_mode"], "units": scenario_a["units"], "risk": prediction_a["risk"], "delay_probability": prediction_a["delay_probability"], "expected_lead_time": prediction_a["expected_lead_time"], "estimated_cost": prediction_a["estimated_cost"]}, {"scenario": "B", "ship_mode": scenario_b["ship_mode"], "units": scenario_b["units"], "risk": prediction_b["risk"], "delay_probability": prediction_b["delay_probability"], "expected_lead_time": prediction_b["expected_lead_time"], "estimated_cost": prediction_b["estimated_cost"]}])
    comparison["safe_probability"] = 1 - comparison["delay_probability"]
    comparison["label"] = "Scenario " + comparison["scenario"] + " | " + comparison["ship_mode"] + " | " + comparison["risk"] + " Risk"
    c1, c2 = st.columns([0.55, 0.45])
    with c1:
        fig_compare = go.Figure(data=[go.Pie(labels=comparison["label"], values=comparison["delay_probability"], hole=0.58, pull=[0.05, 0.05], sort=False, marker=dict(colors=[risk_color(comparison.iloc[0]["risk"]), risk_color(comparison.iloc[1]["risk"])], line=dict(color="rgba(255,255,255,0.45)", width=2)), customdata=comparison[["expected_lead_time", "estimated_cost", "safe_probability", "units"]], hovertemplate=("<b>%{label}</b><br>Risk share: %{percent}<br>Delay probability: %{value:.2%}<br>Safe probability: %{customdata[2]:.2%}<br>Expected lead time: %{customdata[0]:.1f} days<br>Estimated cost: $%{customdata[1]:.2f}<br>Units: %{customdata[3]}<extra></extra>"), texttemplate="%{label}<br>%{value:.1%}", textposition="outside")])
        fig_compare.add_annotation(text="<b>Scenario<br>Risk Mix</b>", x=0.5, y=0.5, showarrow=False, font=dict(size=18))
        fig_compare.update_layout(title="Advanced Scenario Risk Donut", showlegend=True)
        st.markdown('<div class="panel-shell tight"><div class="panel-title">Scenario Comparison Chart</div><div class="prediction-note">Compare two shipment plans by modeled delay share, cost, volume, and expected lead time.</div></div>', unsafe_allow_html=True)
        st.plotly_chart(base_layout(fig_compare, 420), use_container_width=True)
    with c2:
        st.markdown('<div class="panel-shell tight"><div class="panel-title">Scenario Comparison Table</div><div class="prediction-note">Side-by-side operational summary for both simulated shipment setups.</div></div>', unsafe_allow_html=True)
        st.dataframe(comparison[["scenario", "ship_mode", "units", "risk", "delay_probability", "expected_lead_time", "estimated_cost"]], use_container_width=True, hide_index=True)
    render_section_heading("07", "Optimization and Batch Prediction", "Compare mode alternatives, auto-rank options, and upload CSV batches for backend prediction.")
    mode_rows = []
    for mode in options["ship_modes"]:
        res = predict_shipment(dataset_id, {**inputs, "ship_mode": mode}, prediction["delay_threshold"])
        mode_rows.append({"ship_mode": mode, "risk": res["risk"], "delay_probability": res["delay_probability"], "expected_lead_time": res["expected_lead_time"], "estimated_cost": res["estimated_cost"]})
    mode_df = pd.DataFrame(mode_rows).sort_values(["delay_probability", "expected_lead_time", "estimated_cost"])
    st.dataframe(mode_df, use_container_width=True, hide_index=True)
    upload = st.file_uploader("Upload batch prediction CSV", type=["csv"], key="batch_prediction_upload")
    if upload is not None:
        batch = pd.read_csv(upload)
        results = []
        for _, row in batch.iterrows():
            row_inputs = {"region": row.get("region", inputs["region"]), "state": row.get("state", inputs["state"]), "ship_mode": row.get("ship_mode", inputs["ship_mode"]), "units": int(row.get("units", inputs["units"])), "order_date": row.get("order_date", str(inputs["order_date"])), "priority": row.get("priority", inputs["priority"]), "distance": float(row.get("distance", inputs["distance"])), "factory": row.get("factory", inputs["factory"])}
            res = predict_shipment(dataset_id, row_inputs, prediction["delay_threshold"])
            results.append({**row_inputs, "delay_probability": res["delay_probability"], "risk": res["risk"], "expected_lead_time": res["expected_lead_time"], "estimated_cost": res["estimated_cost"]})
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        st.download_button("Download Batch Results CSV", results_df.to_csv(index=False).encode("utf-8"), "batch_prediction_results.csv", "text/csv", use_container_width=True)
