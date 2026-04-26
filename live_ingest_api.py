from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.backend_service import (
    ROOT,
    dashboard_overview,
    dataset_metadata,
    get_default_bundle,
    predict_payload,
    prediction_options,
    prediction_performance,
    register_uploaded_dataset,
)


def _configured_allowed_origins() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "")
    if raw.strip():
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ]


app = FastAPI(
    title="Logistics Backend API",
    description="Backend API for logistics analytics, uploaded dataset handling, and ML inference.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_configured_allowed_origins(),
    allow_origin_regex=r"https://.*\.streamlit\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INGEST_PATH = ROOT / "incoming_shipments.jsonl"
CACHE_BUSTER = ROOT / "data_refresh.flag"


class ShipmentRow(BaseModel):
    order_id: str
    order_date: str
    ship_date: str
    ship_mode: str
    country_region: str
    city: str
    state: str
    region: str
    product_name: str
    sales: float
    units: int = Field(ge=1)
    gross_profit: float
    cost: float
    factory: str | None = None
    customer_id: str | None = None


class DashboardFilters(BaseModel):
    start_date: str | None = None
    end_date: str | None = None
    regions: list[str] = Field(default_factory=list)
    states: list[str] = Field(default_factory=list)
    modes: list[str] = Field(default_factory=list)
    delay_threshold: int = 7


class PredictionRequest(BaseModel):
    region: str
    state: str
    ship_mode: str
    units: int
    order_date: str
    priority: str = "Standard"
    distance: float = 0.0
    factory: str
    delay_threshold: int | None = None


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "logistics-backend",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/datasets/default")
def default_dataset() -> dict[str, Any]:
    get_default_bundle()
    return dataset_metadata("default")


@app.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        content = await file.read()
        return register_uploaded_dataset(content, file.filename)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/datasets/{dataset_id}/metadata")
def dataset_info(dataset_id: str) -> dict[str, Any]:
    try:
        return dataset_metadata(dataset_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/datasets/{dataset_id}/dashboard/overview")
def dashboard(dataset_id: str, filters: DashboardFilters) -> dict[str, Any]:
    try:
        return dashboard_overview(dataset_id, filters.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/datasets/{dataset_id}/prediction/options")
def prediction_option_payload(dataset_id: str) -> dict[str, Any]:
    try:
        return prediction_options(dataset_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/datasets/{dataset_id}/prediction/performance")
def prediction_performance_payload(dataset_id: str) -> dict[str, Any]:
    try:
        return prediction_performance(dataset_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/datasets/{dataset_id}/prediction/infer")
def infer_prediction(dataset_id: str, payload: PredictionRequest) -> dict[str, Any]:
    try:
        body = payload.model_dump()
        delay_threshold = body.pop("delay_threshold", None)
        return predict_payload(dataset_id, body, delay_threshold)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/shipments")
def ingest_shipment(row: ShipmentRow) -> dict[str, str]:
    try:
        payload = row.model_dump()
        payload["ingested_at"] = datetime.utcnow().isoformat()
        with INGEST_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        CACHE_BUSTER.write_text(datetime.utcnow().isoformat(), encoding="utf-8")
        return {
            "status": "accepted",
            "message": "Shipment row stored. Refresh the Streamlit app to include newly ingested rows in the next load cycle.",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
