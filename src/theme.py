import streamlit as st


THEME_CSS = """
<style>
/* ?? Design tokens ??????????????????????????????????????????? */
:root {
    --bg:           #f4f6fb;
    --panel:        #ffffff;
    --panel-soft:   #f8fafc;
    --ink:          #0a0c10;
    --muted:        #4b5675;
    --accent:       #0f62fe;
    --accent-2:     #38bdf8;
    --line:         rgba(15,98,254,0.14);
    --shadow:       rgba(15,23,42,0.07);
    --radius-xl:    24px;
    --radius-lg:    18px;
    --radius-md:    12px;
    --radius-sm:    8px;
    --radius-pill:  999px;
}

@media (prefers-color-scheme: dark) {
    :root {
        --bg:         #080b12;
        --panel:      #0d1626;
        --panel-soft: #111c30;
        --ink:        #eef2f9;
        --muted:      #8fa3c8;
        --accent:     #60a5fa;
        --accent-2:   #38bdf8;
        --line:       rgba(96,165,250,0.18);
        --shadow:     rgba(0,0,0,0.40);
    }
}

/* ?? Global reset ???????????????????????????????????????????? */
*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: var(--bg);
    color: var(--ink);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
}

.block-container {
    padding: 1rem 1.5rem 3rem;
    max-width: 1440px;
    margin: 0 auto;
}

h1,h2,h3,h4,h5,h6,p,label,div,span { color: inherit; }

/* ?? Sidebar ????????????????????????????????????????????????? */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #06080f 0%, #0d1626 100%);
    border-right: 1px solid rgba(255,255,255,0.07);
}

[data-testid="stSidebar"] *,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f0f4ff !important;
}

[data-testid="stSidebar"] [data-baseweb="input"] > div,
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] div[data-testid="stDateInput"] > div,
[data-testid="stSidebar"] div[data-testid="stMultiSelect"] > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(96,165,250,0.20) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: none !important;
}

[data-testid="stSidebar"] [data-baseweb="tag"] {
    background: linear-gradient(135deg,#0f62fe,#38bdf8) !important;
    border-radius: var(--radius-sm) !important;
    border: none !important;
    color: #fff !important;
}

/* ?? Hero ???????????????????????????????????????????????????? */
.hero-shell {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius-xl);
    padding: 2rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 12px 36px var(--shadow);
    position: relative;
    overflow: hidden;
}

.hero-shell::before {
    content: "";
    position: absolute;
    inset: -30% 60% auto auto;
    width: 320px; height: 320px;
    background: radial-gradient(circle,
        rgba(56,189,248,0.14), transparent 65%);
    pointer-events: none;
}

.hero-shell-premium {
    border-color: rgba(96,165,250,0.22);
}

.eyebrow {
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-size: 0.72rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.hero-title {
    color: var(--ink);
    font-size: clamp(1.6rem, 3.5vw, 2.8rem);
    font-weight: 700;
    line-height: 1.1;
    margin: 0;
    letter-spacing: -0.02em;
}

.hero-copy {
    color: var(--muted);
    font-size: 0.97rem;
    line-height: 1.65;
    margin-top: 0.75rem;
    max-width: 56rem;
}

.hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0.5rem 0 1rem;
}

.hero-badge,
.section-number {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.3rem 0.75rem;
    border-radius: var(--radius-pill);
    background: rgba(15,98,254,0.08);
    border: 1px solid rgba(96,165,250,0.20);
    color: var(--accent);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ?? Section shell ??????????????????????????????????????????? */
.section-shell {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius-xl);
    padding: 1.25rem;
    margin-top: 1rem;
    box-shadow: 0 8px 24px var(--shadow);
}

.section-title {
    color: var(--ink);
    font-size: 1.35rem;
    font-weight: 700;
    margin: 0.1rem 0 0.2rem;
    letter-spacing: -0.01em;
}

.section-copy {
    color: var(--muted);
    font-size: 0.93rem;
    line-height: 1.6;
}

/* ?? Panel shell ????????????????????????????????????????????? */
.panel-shell {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius-xl);
    padding: 1.15rem 1.2rem;
    margin-top: 0.85rem;
    box-shadow: 0 6px 20px var(--shadow);
    transition: box-shadow 180ms ease, border-color 180ms ease;
}

.panel-shell:hover {
    box-shadow: 0 14px 32px var(--shadow);
    border-color: rgba(96,165,250,0.26);
}

.panel-shell.tight { padding-top: 0.9rem; }

.panel-title {
    color: var(--ink);
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}

.panel-copy,
.prediction-note {
    color: var(--muted);
    font-size: 0.88rem;
    line-height: 1.55;
}

/* ?? Metric cards ???????????????????????????????????????????? */
.metric-shell,
.sparkline-shell {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius-lg);
    padding: 1rem 1.1rem;
    min-height: 118px;
    box-shadow: 0 4px 16px var(--shadow);
    transition: box-shadow 180ms ease, transform 180ms ease,
                border-color 180ms ease;
}

.metric-shell:hover,
.sparkline-shell:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 28px var(--shadow);
    border-color: rgba(96,165,250,0.26);
}

.metric-kicker,
.sparkline-kicker,
.decision-label,
.cost-label {
    color: var(--muted);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-weight: 700;
}

.metric-number,
.sparkline-number,
.decision-value,
.cost-value {
    color: var(--ink);
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0.3rem 0;
    letter-spacing: -0.02em;
}

.metric-note,
.sparkline-caption {
    color: var(--muted);
    font-size: 0.8rem;
    line-height: 1.45;
}

/* ?? Decision cards ?????????????????????????????????????????? */
.decision-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius-lg);
    padding: 1rem 1.1rem;
    min-height: 110px;
    box-shadow: 0 4px 16px var(--shadow);
    transition: box-shadow 180ms ease, transform 180ms ease;
}

.decision-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 28px var(--shadow);
}

.risk-pill {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    border-radius: var(--radius-pill);
    color: #ffffff;
    font-weight: 700;
    font-size: 0.88rem;
    letter-spacing: 0.03em;
}

/* ?? Recommendation cards ???????????????????????????????????? */
.recommendation-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-top: 3px solid var(--accent);
    border-radius: var(--radius-lg);
    padding: 1rem 1.05rem;
    min-height: 120px;
    box-shadow: 0 4px 16px var(--shadow);
    color: var(--ink);
    transition: box-shadow 180ms ease, transform 180ms ease;
}

.recommendation-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 28px var(--shadow);
}

.recommendation-card span {
    color: var(--muted);
    font-size: 0.72rem;
    display: block;
    margin-bottom: 0.25rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-weight: 700;
}

/* ?? Cost card ??????????????????????????????????????????????? */
.cost-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius-lg);
    padding: 1rem 1.15rem;
    box-shadow: 0 4px 16px var(--shadow);
}

/* ?? Plot / table frames ????????????????????????????????????? */
.plot-frame,
.table-frame {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius-lg);
    padding: 0.85rem;
    box-shadow: 0 6px 20px var(--shadow);
}

/* ?? Premium band ???????????????????????????????????????????? */
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
    border-radius: var(--radius-xl);
    padding: 1.15rem 1.2rem;
    box-shadow: 0 8px 24px var(--shadow);
}

.premium-story-title,
.premium-highlight-title {
    color: var(--ink);
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
}

.premium-story-copy,
.premium-highlight-copy {
    color: var(--muted);
    font-size: 0.9rem;
    line-height: 1.6;
}

.premium-kpi-row {
    display: grid;
    grid-template-columns: repeat(3, minmax(0,1fr));
    gap: 0.75rem;
    margin-top: 0.85rem;
}

.premium-mini {
    padding: 0.75rem 0.9rem;
    border-radius: var(--radius-md);
    background: rgba(15,98,254,0.05);
    border: 1px solid rgba(96,165,250,0.14);
}

.premium-mini-label {
    color: var(--muted);
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-weight: 700;
}

.premium-mini-value {
    color: var(--ink);
    font-size: 1.1rem;
    font-weight: 700;
    margin-top: 0.2rem;
}

/* ?? Tabs ???????????????????????????????????????????????????? */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
    flex-wrap: wrap;
    background: transparent;
    border-bottom: 1px solid var(--line);
    padding-bottom: 0;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border: 1px solid transparent;
    border-radius: var(--radius-pill) var(--radius-pill) 0 0;
    padding: 0.45rem 1rem;
    color: var(--muted);
    font-size: 0.88rem;
    font-weight: 600;
    transition: color 150ms, background 150ms;
}

.stTabs [aria-selected="true"] {
    background: rgba(15,98,254,0.08) !important;
    border-color: var(--line) var(--line) transparent !important;
    color: var(--accent) !important;
}

/* ?? Data tables ????????????????????????????????????????????? */
.stDataFrame,
div[data-testid="stTable"] {
    border-radius: var(--radius-lg);
    overflow: hidden;
    border: 1px solid var(--line);
}

/* ?? Buttons ????????????????????????????????????????????????? */
.stDownloadButton button,
.stButton button {
    background: linear-gradient(135deg,#0f62fe,#1d4ed8) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 1.2rem !important;
    transition: opacity 160ms ease, transform 160ms ease !important;
}

.stDownloadButton button:hover,
.stButton button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

/* ?? Divider ????????????????????????????????????????????????? */
.subtle-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--line), transparent);
    margin: 1rem 0 0.25rem;
}

/* ?? Alert badges ???????????????????????????????????????????? */
.alert-badge-high,
.alert-badge-watch,
.alert-badge-ok {
    display: inline-block;
    padding: 0.3rem 0.7rem;
    border-radius: var(--radius-pill);
    font-size: 0.72rem;
    font-weight: 700;
    color: white;
}

.alert-badge-high  { background: #dc2626; }
.alert-badge-watch { background: #f59e0b; }
.alert-badge-ok    { background: #0f62fe; }

/* ?? Sliders, inputs ????????????????????????????????????????? */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
}

/* ?? Scrollbar ??????????????????????????????????????????????? */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(96,165,250,0.3);
                             border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(96,165,250,0.5); }

/* ?? Mobile: ? 768px ????????????????????????????????????????? */
@media (max-width: 768px) {
    .block-container {
        padding: 0.75rem 0.75rem 2rem;
    }

    .hero-shell {
        padding: 1.25rem;
        border-radius: var(--radius-lg);
    }

    .hero-title { font-size: 1.5rem; }

    .section-shell,
    .panel-shell,
    .plot-frame,
    .table-frame,
    .premium-story,
    .premium-highlight {
        border-radius: var(--radius-lg);
        padding: 1rem;
    }

    .metric-shell,
    .sparkline-shell,
    .decision-card,
    .recommendation-card,
    .cost-card {
        border-radius: var(--radius-md);
        min-height: auto;
        padding: 0.85rem;
    }

    .metric-number,
    .sparkline-number,
    .decision-value,
    .cost-value {
        font-size: 1.4rem;
    }

    .hero-badge,
    .section-number {
        font-size: 0.68rem;
        padding: 0.28rem 0.6rem;
    }

    .premium-band {
        grid-template-columns: 1fr;
    }

    .premium-kpi-row {
        grid-template-columns: repeat(2, minmax(0,1fr));
    }

    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: 0.75rem !important;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 0.82rem;
        padding: 0.38rem 0.75rem;
    }

    .stDownloadButton button,
    .stButton button {
        width: 100% !important;
        justify-content: center !important;
    }

    [data-testid="stSidebar"] {
        min-width: 17rem !important;
    }
}

/* ?? Small mobile: ? 480px ??????????????????????????????????? */
@media (max-width: 480px) {
    .block-container {
        padding: 0.5rem 0.5rem 1.5rem;
    }

    .hero-title { font-size: 1.3rem; }
    .section-title { font-size: 1.1rem; }

    .premium-kpi-row {
        grid-template-columns: 1fr;
    }

    .stTabs [data-baseweb="tab-list"] {
        flex-direction: column;
        border-bottom: none;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--line) !important;
        text-align: center;
        width: 100%;
    }

    .metric-shell:hover,
    .sparkline-shell:hover,
    .panel-shell:hover,
    .decision-card:hover,
    .recommendation-card:hover {
        transform: none;
    }
}

/* ?? Reduced motion ?????????????????????????????????????????? */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        transition: none !important;
        animation: none !important;
    }
}
</style>
"""




def apply_branding() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)
