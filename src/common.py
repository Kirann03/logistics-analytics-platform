from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.theme import apply_branding


def render_section_heading(number: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="section-shell">
            <div class="section-number">Section {number}</div>
            <div class="section-title">{title}</div>
            <div class="section-copy">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header(data_source: str, validation_messages: list[str] | tuple[str, ...], filtered_count: int) -> None:
    warning_block = ""
    if validation_messages:
        items = "".join(f"<li>{message}</li>" for message in list(validation_messages)[:3])
        warning_block = f"<div style='margin-top:0.85rem; color:var(--muted); font-size:0.96rem;'><strong style='color:var(--text);'>Data preparation highlights</strong><ul style='margin-top:0.45rem; padding-left:1.1rem;'>{items}</ul></div>"
    st.markdown(
        f"""
        <div class="hero-shell hero-shell-premium">
            <div class="eyebrow">Logistics Analytics Project</div>
            <div class="hero-badges">
                <span class="hero-badge">Data source: {data_source}</span>
                <span class="hero-badge">Cleaned shipments: {filtered_count:,}</span>
                <span class="hero-badge">FastAPI-backed workflow</span>
            </div>
            <h1 class="hero-title">Factory-to-Customer Shipping Route Efficiency Dashboard</h1>
            <div class="hero-copy">
                An executive-grade analytics workspace designed to surface route performance, geographic bottlenecks,
                service-mode tradeoffs, operational drill-downs, and predictive logistics insights with clarity and speed.
                {warning_block}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-shell">
            <div class="metric-kicker">{label}</div>
            <div class="metric-number">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_sparkline(label: str, value: str, note: str, spark_values: list[float], color: str = "#60a5fa") -> None:
    st.markdown(
        f"""
        <div class="sparkline-shell">
            <div class="sparkline-kicker">{label}</div>
            <div class="sparkline-number">{value}</div>
            <div class="sparkline-caption">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if spark_values:
        fig = go.Figure(
            go.Scatter(
                x=list(range(len(spark_values))),
                y=spark_values,
                mode="lines",
                line=dict(color=color, width=2.6),
                fill="tozeroy",
                fillcolor="rgba(96,165,250,0.12)",
            )
        )
        fig.update_layout(height=90, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def base_layout(fig: go.Figure, height: int = 430) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=52, b=10),
        font=dict(color="#94a3b8"),
        legend_title_text="",
    )
    fig.update_xaxes(
        color="#94a3b8",
        gridcolor="rgba(148,163,184,0.20)",
        zerolinecolor="rgba(148,163,184,0.28)",
    )
    fig.update_yaxes(
        color="#94a3b8",
        gridcolor="rgba(148,163,184,0.20)",
        zerolinecolor="rgba(148,163,184,0.28)",
    )
    return fig



def render_footer() -> None:
    st.markdown(
        """
        <div style="margin:2.5rem 0 0.75rem 0; text-align:center; opacity:0.86; font-size:0.95rem; letter-spacing:0.02em; line-height:1.8;">
            <div>Powered by Aishwarya &amp; Kiran</div>
            <div style="font-size:0.88rem; opacity:0.82;">&copy; 2026 Aishwarya &amp; Kiran. All rights reserved.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
