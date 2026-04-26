# Deployment Guide

## 1. GitHub

Push the full `Logistics-Streamlit` project, including:
- `app.py`
- `live_ingest_api.py`
- `src/`
- `data.xlsx`
- `Factories Coordinates.xlsx`
- `Products and Factories Correlation.xlsx`
- `models/`
- `requirements.txt`
- `requirements-api.txt`
- `render.yaml`
- `runtime.txt`

Do not commit:
- `.venv/`
- `.streamlit/secrets.toml`
- `incoming_shipments.jsonl`
- `data_refresh.flag`

## 2. Render Backend

Use `render.yaml` or configure manually:
- Build command: `pip install -r requirements-api.txt`
- Start command: `uvicorn live_ingest_api:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`

Set environment variable:
- `ALLOWED_ORIGINS=https://your-streamlit-app.streamlit.app`

## 3. Streamlit Cloud Frontend

Deploy `app.py` from the same repo.

Set Streamlit secret:

```toml
LOGISTICS_API_URL = "https://your-render-backend.onrender.com"
```

## 4. Performance Notes

- The backend always uses the saved production model in `models/` for inference.
- Uploaded datasets do not trigger full retraining on the server.
- The dashboard always uses the curated repo dataset.
- Uploaded files are used only in prediction workflows.
- Heavy dashboard aggregations are computed in FastAPI and cached before Streamlit renders them.

## 5. Quick Validation

After deployment, verify:
- Render backend: `/health` and `/docs`
- Streamlit frontend loads without backend connection errors
- Prediction page returns results for a sample scenario
- Dashboard uses the repo dataset regardless of uploaded files
