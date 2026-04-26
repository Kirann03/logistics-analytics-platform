# Factory-to-Customer Shipping Route Efficiency Dashboard

A production-ready logistics analytics platform with a `Streamlit` frontend and a `FastAPI` backend. The application supports executive dashboarding, operational drill-downs, ML-based delay prediction, scenario simulation, and flexible prediction-only dataset uploads.

## Architecture

- `Streamlit Cloud`: frontend UI and interactive experience
- `Railway`: backend API, heavy data processing, analytics aggregation, uploaded-dataset normalization, and ML inference
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

`requirements-api.txt` is dedicated to Railway and contains the heavier backend dependencies:

- FastAPI and request parsing
- analytics and ML dependencies
- CatBoost / Prophet / Optuna stack
- `python-multipart` for file upload endpoints

## Local Development

Use Python `3.11.x` on both Windows and macOS for the most reliable experience.

### Create a virtual environment

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

Frontend dependencies:

```bash
python -m pip install -r requirements.txt
```

Backend dependencies:

```bash
python -m pip install -r requirements-api.txt
```

### Platform-specific fallback installers

If you want a single install command per operating system, use these fallback files:

Windows:

```powershell
python -m pip install -r requirements-windows.txt
```

macOS / Linux:

```bash
python -m pip install -r requirements-macos.txt
```

### Run backend manually

```bash
python -m uvicorn live_ingest_api:app --reload
```

### Check backend docs

Open:

```text
http://127.0.0.1:8000/docs
```

### Run frontend manually

In a second terminal:

```bash
python -m streamlit run app.py
```

### Cross-platform one-command launcher

This repo now includes a platform-neutral launcher:

```bash
python run_all.py
```

Windows-only PowerShell launcher is still available:

```powershell
.
un_all.ps1
```

macOS / Linux shell launcher is also included:

```bash
bash run_all.sh
```

## macOS Notes

- Use Python `3.11.x`, not `3.12`, for the most stable package support.
- Install Apple command-line tools if needed:

```bash
xcode-select --install
```

- On Apple Silicon (`M1` / `M2`), some ML packages can be more sensitive than on Windows. The app has been adjusted so runtime startup does not depend on training-only imports as aggressively.
- The backend and frontend must both be running locally.

## GitHub Upload Checklist

Before pushing the repository:

- confirm `.venv/` is not committed
- confirm `.streamlit/secrets.toml` is not committed
- keep `data.xlsx`, `Factories Coordinates.xlsx`, and `Products and Factories Correlation.xlsx` in the repo if the dashboard must run in production
- make sure `models/` is committed if you want inference without retraining on the server

## Railway Backend Deployment

Railway officially supports deploying FastAPI apps from a GitHub repository and using config-as-code with `railway.json`:
- FastAPI guide: https://docs.railway.com/guides/fastapi
- Config as code: https://docs.railway.com/config-as-code
- Variables: https://docs.railway.com/variables
- Start command: https://docs.railway.com/deployments/start-command

### Deploy steps

1. Open Railway.
2. Create a `New Project`.
3. Choose `Deploy from GitHub repo`.
4. Select this repository.
5. Railway will use the included `railway.json`.
6. After deployment, go to the service networking/settings area and `Generate Domain`.

### Railway variable

Set this environment variable in Railway:

```text
ALLOWED_ORIGINS=https://your-streamlit-app.streamlit.app
```

### Backend smoke test

After Railway gives you a domain, verify:
- `https://your-railway-domain/health`
- `https://your-railway-domain/docs`

## Streamlit Cloud Deployment

Deploy the frontend from the same GitHub repository.

### Main file

- `app.py`

### Python version

The repo includes `runtime.txt` pinned to `python-3.11.9` for better deployment consistency.

### Streamlit Cloud Secret

Add this secret in Streamlit Cloud:

```toml
LOGISTICS_API_URL = "https://your-railway-domain"
```

The frontend will automatically use this backend URL through `src/api_client.py`.

## Important Deployment Notes

- CORS is enabled in the backend for local Streamlit development and Streamlit Cloud domains.
- If you use a custom frontend domain, add it to `ALLOWED_ORIGINS` in Railway.
- Railway gives your service a public URL only after you generate a domain for it.
- Runtime-generated files like `incoming_shipments.jsonl` are not ideal for permanent storage. If you need persistence across restarts, move them to object storage or a database.

## Model Training

To retrain and refresh saved artifacts:

```bash
python train_model.py
```

Optional tuning:

```bash
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

- make sure the Railway backend is deployed and running
- make sure you generated a Railway domain
- confirm `LOGISTICS_API_URL` is correct in Streamlit Cloud secrets
- confirm `ALLOWED_ORIGINS` includes your Streamlit Cloud URL

### `python-multipart` error on backend

Install it into the correct environment:

```bash
python -m pip install -r requirements-api.txt
```

### Local `127.0.0.1:8000` connection refused

The frontend is up but the backend is not running yet. Start FastAPI first.
