# Factory-to-Customer Shipping Route Efficiency Dashboard

A production-ready logistics analytics platform with a `Streamlit` frontend and a `FastAPI` backend. The application supports executive dashboarding, operational drill-downs, ML-based delay prediction, scenario simulation, and flexible prediction-only dataset uploads.

## Architecture

- `Streamlit Cloud`: frontend UI and interactive experience
- `Render`: backend API, heavy data processing, analytics aggregation, uploaded-dataset normalization, and ML inference
- `GitHub`: single source repository for both services

This split keeps the frontend responsive while moving expensive computation off the Streamlit runtime.

## Project Structure

### Frontend
- `app.py`: Streamlit entry point
- `src/dashboard.py`: dashboard rendering
- `src/prediction.py`: prediction workspace rendering
- `src/common.py`: shared UI helpers
- `src/theme.py`: theme and styling
- `src/api_client.py`: frontend client for the backend API

### Backend
- `live_ingest_api.py`: FastAPI application
- `src/backend_service.py`: backend orchestration and payload generation
- `src/data.py`: dataset loading, validation, mapping, and normalization
- `src/analytics.py`: route, KPI, anomaly, and forecast analytics
- `src/ml.py`: model loading wrapper
- `src/ml_model.py`: CatBoost training and inference pipeline
- `src/alerts.py`: alert logic
- `train_model.py`: training script for model artifacts

## Data Behavior

- The `Dashboard` always analyzes the curated project dataset stored in the repository.
- Uploaded CSV or Excel files are used only inside the `Prediction` workspace.
- Uploaded datasets never overwrite or change the dashboard narrative.

## Frontend Requirements

`requirements.txt` is intentionally kept frontend-focused for Streamlit Cloud:

- `streamlit`
- `pandas`
- `numpy`
- `plotly`
- `openpyxl`
- `folium`
- `branca`
- `requests`

## Backend Requirements

`requirements-api.txt` is dedicated to Render and contains the heavier backend dependencies:

- FastAPI and request parsing
- analytics and ML dependencies
- CatBoost / Prophet / Optuna stack
- `python-multipart` for file upload endpoints

## Local Development

Create and activate the virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install frontend dependencies:

```powershell
python -m pip install -r requirements.txt
```

Install backend dependencies:

```powershell
python -m pip install -r requirements-api.txt
```

### Run backend

```powershell
python -m uvicorn live_ingest_api:app --reload
```

### Check in browser

Open:

```text
http://127.0.0.1:8000/docs
```

### Open new terminal

```powershell
python -m streamlit run app.py
```

### One-command launcher

```powershell
.
un_all.ps1
```

## GitHub Upload Checklist

Before pushing the repository:

- confirm `.venv/` is not committed
- confirm `.streamlit/secrets.toml` is not committed
- keep `data.xlsx`, `Factories Coordinates.xlsx`, and `Products and Factories Correlation.xlsx` in the repo if the dashboard must run on Render
- make sure `models/` is committed if you want inference without retraining on the server

## Render Deployment

Deploy the FastAPI backend from this same repository.

### Option 1: Use `render.yaml`

Render can read the included `render.yaml` and create the backend service with:

- build command: `pip install -r requirements-api.txt`
- start command: `uvicorn live_ingest_api:app --host 0.0.0.0 --port $PORT`
- health check: `/health`

### Required Render Environment Variables

- `ALLOWED_ORIGINS`
  - example: `https://your-frontend-name.streamlit.app`
  - use a comma-separated list if you want to allow multiple origins

### Backend Smoke Test

After deploy, verify:

- `https://your-render-service.onrender.com/health`
- `https://your-render-service.onrender.com/docs`

## Streamlit Cloud Deployment

Deploy the frontend from the same GitHub repository.

### Main file

- `app.py`

### Python version

The repo includes `runtime.txt` pinned to `python-3.11.9` for better deployment consistency.

### Streamlit Cloud Secret

Add this secret in Streamlit Cloud:

```toml
LOGISTICS_API_URL = "https://your-render-service.onrender.com"
```

The frontend will automatically use this backend URL through `src/api_client.py`.

## Important Deployment Notes

- CORS is enabled in the backend for local Streamlit development and Streamlit Cloud domains.
- If you use a custom Streamlit domain, add it to `ALLOWED_ORIGINS` on Render.
- Render filesystems are ephemeral, so runtime-generated files like `incoming_shipments.jsonl` are not permanent unless you add persistent storage.
- If you want uploaded datasets or ingestion logs to survive restarts, move them to object storage or a database.

## Model Training

To retrain and refresh saved artifacts:

```powershell
python train_model.py
```

Optional tuning:

```powershell
python train_model.py --tune --trials 20
```

## API Endpoints

Key endpoints:

- `/health`
- `/datasets/default`
- `/datasets/upload`
- `/datasets/{dataset_id}/metadata`
- `/datasets/{dataset_id}/dashboard/overview`
- `/datasets/{dataset_id}/prediction/options`
- `/datasets/{dataset_id}/prediction/performance`
- `/datasets/{dataset_id}/prediction/infer`
- `/shipments`

## Troubleshooting

### Frontend cannot connect to backend

If Streamlit shows a backend connection error:

- make sure the Render backend is running
- confirm `LOGISTICS_API_URL` is correct in Streamlit Cloud secrets
- confirm Render `ALLOWED_ORIGINS` includes your Streamlit Cloud URL

### `python-multipart` error on backend

Install it into the correct environment:

```powershell
python -m pip install -r requirements-api.txt
```

### Local `127.0.0.1:8000` connection refused

The frontend is up but the backend is not running yet. Start FastAPI first.
