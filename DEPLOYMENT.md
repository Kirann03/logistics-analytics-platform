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
- `railway.json`
- `runtime.txt`

Do not commit:
- `.venv/`
- `.streamlit/secrets.toml`
- `incoming_shipments.jsonl`
- `data_refresh.flag`

## 2. Railway Backend

Railway supports config-as-code through `railway.json`, and Railway's FastAPI guide recommends deploying directly from GitHub and then generating a public domain for the service. See the official docs:
- Config as code: https://docs.railway.com/config-as-code
- FastAPI deployment guide: https://docs.railway.com/guides/fastapi
- Variables: https://docs.railway.com/variables

### Deploy steps

1. Open Railway.
2. Create a new project.
3. Choose `Deploy from GitHub repo`.
4. Select `Kirann03/logistics-analytics-platform`.
5. Railway will read `railway.json` and use:
   - start command: `uvicorn live_ingest_api:app --host 0.0.0.0 --port $PORT`
   - health check path: `/health`
6. After the deployment finishes, open the service settings and use `Generate Domain`.

### Railway environment variables

Set this service variable in Railway:
- `ALLOWED_ORIGINS=https://your-streamlit-app.streamlit.app`

If you want to test before Streamlit Cloud is live, you can temporarily use a broader value such as your local origin and then tighten it later.

### Backend validation

After Railway gives you a domain, verify:
- `https://your-railway-domain/health`
- `https://your-railway-domain/docs`

## 3. Streamlit Cloud Frontend

Deploy `app.py` from the same repo on Streamlit Community Cloud.

### Streamlit secret

Add this secret:

```toml
LOGISTICS_API_URL = "https://your-railway-domain"
```

### Frontend validation

After deploy, verify:
- the dashboard loads from the repo dataset
- prediction works against the Railway backend
- uploaded datasets affect only the prediction workspace

## 4. Performance Notes

- The backend always uses the saved production model in `models/` for inference.
- Uploaded datasets do not trigger retraining on Railway.
- The dashboard always uses the curated repo dataset.
- Heavy aggregations are computed in FastAPI and cached before Streamlit renders them.
- Commit the `models/` folder so Railway does not need to train anything on boot.

## 5. If Railway deployment fails

Common checks:
- confirm `requirements-api.txt` is present in the repo
- confirm `models/` is committed
- confirm the dashboard Excel files are committed
- confirm you generated a public domain in Railway
- confirm `LOGISTICS_API_URL` in Streamlit Cloud exactly matches the Railway domain
