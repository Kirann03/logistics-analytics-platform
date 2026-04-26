from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
import requests
import streamlit as st


class ApiError(RuntimeError):
    pass


def get_api_base_url() -> str:
    env_url = os.getenv("LOGISTICS_API_URL")
    if env_url:
        return env_url.rstrip("/")
    try:
        secrets_url = st.secrets.get("LOGISTICS_API_URL")
    except Exception:
        secrets_url = None
    if secrets_url:
        return str(secrets_url).rstrip("/")
    return "http://127.0.0.1:8000"


def _request(method: str, path: str, **kwargs) -> Any:
    url = f"{get_api_base_url().rstrip('/')}{path}"
    timeout = kwargs.pop("timeout", 120)
    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.RequestException as exc:
        raise ApiError(
            "The frontend could not connect to the backend API. "
            f"Check that LOGISTICS_API_URL is correct and the FastAPI service is running. Details: {exc}"
        ) from exc
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail")
        except Exception:
            detail = response.text
        raise ApiError(detail or f"API request failed: {response.status_code}")
    return response.json()


@st.cache_data(show_spinner=False)
def get_default_dataset() -> dict[str, Any]:
    return _request("GET", "/datasets/default")


@st.cache_data(show_spinner=False)
def upload_dataset(filename: str, file_bytes: bytes) -> dict[str, Any]:
    files = {"file": (filename, file_bytes)}
    return _request("POST", "/datasets/upload", files=files, timeout=300)


@st.cache_data(show_spinner=False)
def get_dataset_metadata(dataset_id: str) -> dict[str, Any]:
    return _request("GET", f"/datasets/{dataset_id}/metadata")


@st.cache_data(show_spinner=False)
def get_dashboard_overview(dataset_id: str, filters_json: str) -> dict[str, Any]:
    return _request(
        "POST",
        f"/datasets/{dataset_id}/dashboard/overview",
        json=json.loads(filters_json),
        timeout=300,
    )


@st.cache_data(show_spinner=False)
def get_prediction_options(dataset_id: str) -> dict[str, Any]:
    return _request("GET", f"/datasets/{dataset_id}/prediction/options")


@st.cache_data(show_spinner=False)
def get_prediction_performance(dataset_id: str) -> dict[str, Any]:
    return _request("GET", f"/datasets/{dataset_id}/prediction/performance", timeout=300)


def infer_prediction(dataset_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return _request("POST", f"/datasets/{dataset_id}/prediction/infer", json=payload, timeout=300)


def records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records or [])
    for column in frame.columns:
        if column.endswith("date") or column.endswith("_date") or column in {"order_month", "ds"}:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame
