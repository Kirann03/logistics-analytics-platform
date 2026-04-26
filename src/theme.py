import streamlit as st


THEME_CSS = """
<style>
:root {
    --bg: #f5f7fb;
    --panel: #ffffff;
    --panel-soft: #f8fafc;
    --ink: #050505;
    --muted: #475569;
    --accent: #0f62fe;
    --accent-2: #38bdf8;
    --line: rgba(15, 98, 254, 0.16);
    --shadow: rgba(15, 23, 42, 0.08);
}

html[data-user-theme="dark"] {
    --bg: #05070b;
    --panel: #0b1220;
    --panel-soft: #111827;
    --ink: #f8fafc;
    --muted: #cbd5e1;
    --accent: #60a5fa;
    --accent-2: #38bdf8;
    --line: rgba(96, 165, 250, 0.20);
    --shadow: rgba(0, 0, 0, 0.36);
}

html[data-user-theme="light"] {
    --bg: #f5f7fb;
    --panel: #ffffff;
    --panel-soft: #f8fafc;
    --ink: #050505;
    --muted: #475569;
    --accent: #0f62fe;
    --accent-2: #38bdf8;
    --line: rgba(15, 98, 254, 0.16);
    --shadow: rgba(15, 23, 42, 0.08);
}

@media (prefers-color-scheme: dark) {
    html:not([data-user-theme="light"]) {
        --bg: #05070b;
        --panel: #0b1220;
        --panel-soft: #111827;
        --ink: #f8fafc;
        --muted: #cbd5e1;
        --accent: #60a5fa;
        --accent-2: #38bdf8;
        --line: rgba(96, 165, 250, 0.20);
        --shadow: rgba(0, 0, 0, 0.36);
    }
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(15, 98, 254, 0.12), transparent 28%),
        radial-gradient(circle at top right, rgba(56, 189, 248, 0.10), transparent 24%),
        linear-gradient(180deg, var(--panel-soft) 0%, var(--bg) 100%);
    color: var(--ink);
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1440px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050505 0%, #101827 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

[data-testid="stSidebar"] *,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}

[data-testid="stSidebar"] [data-baseweb="input"],
[data-testid="stSidebar"] [data-baseweb="select"],
[data-testid="stSidebar"] [data-baseweb="popover"],
[data-testid="stSidebar"] div[data-testid="stDateInput"],
[data-testid="stSidebar"] div[data-testid="stMultiSelect"] {
    background: rgba(255, 255, 255, 0.06) !important;
    border-radius: 14px !important;
}

[data-testid="stSidebar"] [data-baseweb="input"] > div,
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] div[data-testid="stDateInput"] > div,
[data-testid="stSidebar"] div[data-testid="stMultiSelect"] > div {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(96, 165, 250, 0.22) !important;
    border-radius: 14px !important;
    box-shadow: none !important;
}

[data-testid="stSidebar"] [data-baseweb="tag"] {
    background: linear-gradient(135deg, #0f62fe, #38bdf8) !important;
    border-radius: 10px !important;
    border: none !important;
    color: #ffffff !important;
}

.hero-shell,
.section-shell,
.metric-shell,
.panel-shell,
.decision-card,
.recommendation-card,
.cost-card,
.sparkline-shell {
    background: var(--panel);
    border: 1px solid var(--line);
    box-shadow: 0 18px 42px var(--shadow);
}

.hero-shell {
    border-radius: 30px;
    padding: 1.9rem;
    margin-bottom: 1rem;
    background:
        linear-gradient(135deg, rgba(15, 98, 254, 0.08), transparent 38%),
        var(--panel);
}

.eyebrow {
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.18em;
    font-size: 0.78rem;
    font-weight: 800;
    margin-bottom: 0.45rem;
}

.hero-title {
    color: var(--ink);
    font-family: Georgia, "Palatino Linotype", serif;
    font-size: 2.6rem;
    line-height: 1.05;
    margin: 0;
}

.hero-copy {
    color: var(--muted);
    font-size: 1rem;
    margin-top: 0.8rem;
    max-width: 58rem;
}

.section-shell,
.panel-shell {
    border-radius: 26px;
    padding: 1.2rem 1.2rem 1.1rem;
    margin-top: 1rem;
}

.panel-shell.tight {
    padding-top: 0.95rem;
}

.section-title {
    color: var(--ink);
    font-family: Georgia, "Palatino Linotype", serif;
    font-size: 1.6rem;
    margin: 0.15rem 0 0.2rem;
}

.section-copy,
.panel-copy,
.prediction-note,
.metric-note {
    color: var(--muted);
}

.metric-shell,
.sparkline-shell {
    border-radius: 22px;
    padding: 1rem 1.1rem;
    min-height: 132px;
}

.metric-kicker,
.sparkline-kicker,
.decision-label,
.cost-label {
    color: var(--muted);
    font-size: 0.84rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 800;
}

.metric-number,
.sparkline-number,
.decision-value,
.cost-value {
    color: var(--ink);
    font-size: 1.9rem;
    font-weight: 800;
    margin: 0.35rem 0;
}

.sparkline-shell {
    padding-bottom: 0.5rem;
}

.sparkline-caption {
    color: var(--muted);
    font-size: 0.82rem;
}

.decision-card {
    border-radius: 22px;
    padding: 1rem 1.1rem;
    min-height: 120px;
}

.risk-pill {
    display: inline-block;
    padding: 0.3rem 0.7rem;
    border-radius: 999px;
    color: #ffffff;
    font-weight: 800;
    letter-spacing: 0.04em;
}

.recommendation-card {
    border-top: 4px solid var(--accent);
    border-radius: 18px;
    padding: 1rem 1.05rem;
    min-height: 132px;
    color: var(--ink);
}

.recommendation-card span {
    color: var(--muted);
    font-size: 0.82rem;
    display: block;
    margin-bottom: 0.2rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 800;
}

.cost-card {
    border-radius: 20px;
    padding: 1rem 1.15rem;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(15, 98, 254, 0.08);
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 0.45rem 1rem;
    color: var(--ink);
}

.stDataFrame,
div[data-testid="stTable"] {
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid var(--line);
}

.stDownloadButton button,
.stButton button {
    background: linear-gradient(135deg, #0f62fe, #2563eb) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
}

.subtle-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--line), transparent);
    margin: 0.9rem 0 0.25rem;
}

.alert-badge-high,
.alert-badge-watch,
.alert-badge-ok {
    display: inline-block;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 800;
    color: white;
}

.alert-badge-high { background: #dc2626; }
.alert-badge-watch { background: #f59e0b; }
.alert-badge-ok { background: #0f62fe; }

@media (max-width: 1100px) {
    .hero-title {
        font-size: 2.15rem;
    }
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}

@media (max-width: 900px) {
    .hero-title {
        font-size: 1.85rem;
    }
    .section-title {
        font-size: 1.35rem;
    }
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
    }
}


.hero-shell-premium {
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(96, 165, 250, 0.22);
}

.hero-shell-premium::before {
    content: "";
    position: absolute;
    inset: -20% auto auto 68%;
    width: 240px;
    height: 240px;
    background: radial-gradient(circle, rgba(56, 189, 248, 0.18), transparent 68%);
    pointer-events: none;
}

.hero-shell-premium::after {
    content: "";
    position: absolute;
    inset: auto auto -35% -8%;
    width: 280px;
    height: 220px;
    background: radial-gradient(circle, rgba(15, 98, 254, 0.14), transparent 70%);
    pointer-events: none;
}

.hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin: 0.35rem 0 1rem;
}

.hero-badge,
.section-number {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.36rem 0.78rem;
    border-radius: 999px;
    background: rgba(15, 98, 254, 0.10);
    border: 1px solid rgba(96, 165, 250, 0.22);
    color: var(--accent);
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.section-shell {
    position: relative;
    background:
        linear-gradient(135deg, rgba(15, 98, 254, 0.06), transparent 32%),
        var(--panel);
}

.metric-shell,
.sparkline-shell,
.panel-shell,
.decision-card,
.recommendation-card,
.cost-card {
    backdrop-filter: blur(8px);
    transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
}

.metric-shell:hover,
.sparkline-shell:hover,
.panel-shell:hover,
.decision-card:hover,
.recommendation-card:hover,
.cost-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 24px 46px var(--shadow);
    border-color: rgba(96, 165, 250, 0.28);
}

.premium-band {
    display: grid;
    grid-template-columns: 1.2fr 0.8fr;
    gap: 1rem;
    margin: 1rem 0 1.25rem;
}

.premium-story,
.premium-highlight {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 24px;
    padding: 1.15rem 1.2rem;
    box-shadow: 0 16px 34px var(--shadow);
}

.premium-story-title,
.premium-highlight-title {
    color: var(--ink);
    font-size: 1rem;
    font-weight: 800;
    margin-bottom: 0.45rem;
}

.premium-story-copy,
.premium-highlight-copy {
    color: var(--muted);
    line-height: 1.6;
}

.premium-kpi-row {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.8rem;
    margin-top: 0.9rem;
}

.premium-mini {
    padding: 0.8rem 0.9rem;
    border-radius: 18px;
    background: rgba(15, 98, 254, 0.06);
    border: 1px solid rgba(96, 165, 250, 0.16);
}

.premium-mini-label {
    color: var(--muted);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 800;
}

.premium-mini-value {
    color: var(--ink);
    font-size: 1.15rem;
    font-weight: 800;
    margin-top: 0.25rem;
}

.plot-frame {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 24px;
    padding: 0.7rem 0.85rem 0.2rem;
    box-shadow: 0 14px 30px var(--shadow);
}

.table-frame {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: 0.85rem;
    box-shadow: 0 14px 30px var(--shadow);
}

@media (max-width: 980px) {
    .premium-band {
        grid-template-columns: 1fr;
    }
    .premium-kpi-row {
        grid-template-columns: 1fr;
    }
}
h1, h2, h3, h4, h5, h6, p, label, div, span {
    color: inherit;
}
</style>
"""


def apply_branding(theme_mode: str = "System") -> None:
    theme_value = {"System": "system", "Light": "light", "Dark": "dark"}.get(theme_mode, "system")
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
        <script>
        const root = window.parent.document.documentElement;
        if ({theme_value!r} === 'system') {{
            root.removeAttribute('data-user-theme');
        }} else {{
            root.setAttribute('data-user-theme', {theme_value!r});
        }}
        </script>
        """,
        unsafe_allow_html=True,
    )
