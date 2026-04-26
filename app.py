from pathlib import Path

import streamlit as st

from src.api_client import ApiError, get_default_dataset, upload_dataset
from src.common import apply_branding, render_footer
from src.dashboard import render_dashboard_page
from src.prediction import render_prediction_page


st.set_page_config(
    page_title="Logistics Route Efficiency Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _resolve_dataset_reference(uploaded) -> dict:
    if uploaded is not None:
        upload_key = f"{uploaded.name}-{uploaded.size}"
        cached = st.session_state.get("uploaded_dataset_ref")
        if not cached or cached.get("upload_key") != upload_key:
            response = upload_dataset(uploaded.name, uploaded.getvalue())
            response["upload_key"] = upload_key
            st.session_state["uploaded_dataset_ref"] = response
        return st.session_state["uploaded_dataset_ref"]
    st.session_state.pop("uploaded_dataset_ref", None)
    return get_default_dataset()


def main() -> None:
    Path(__file__).resolve().parent

    with st.sidebar:
        st.markdown("## Sections")
        section = st.radio("Choose section", ["Dashboard", "Prediction"], label_visibility="collapsed")
        st.markdown("## Display")
        theme_mode = st.selectbox("Theme mode", ["System", "Light", "Dark"], index=0)
        st.markdown("## Dataset")
        uploaded = st.file_uploader(
            "Import CSV / Excel dataset",
            type=["csv", "xlsx", "xls"],
            help="Use uploaded files for prediction experiments and scenario testing. The Dashboard remains anchored to the curated project dataset for consistent portfolio analysis.",
        )

    apply_branding(theme_mode)

    try:
        default_dataset_ref = get_default_dataset()
        prediction_dataset_ref = _resolve_dataset_reference(uploaded)
    except ApiError as exc:
        st.error(f"Backend request failed: {exc}")
        st.info("Start the FastAPI server with `python -m uvicorn live_ingest_api:app --reload` and then refresh Streamlit.")
        return
    except Exception as exc:
        st.error(f"Dataset could not be processed: {exc}")
        return

    if section == "Prediction":
        render_prediction_page(prediction_dataset_ref)
    else:
        render_dashboard_page(default_dataset_ref)

    render_footer()


if __name__ == "__main__":
    main()
