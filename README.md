# Rossmann Sales Forecasting - MLOps Pipeline

End-to-end MLOps pipeline for forecasting daily sales across 1,115 Rossmann drugstores in Germany.

## Stack

| Layer | Technology |
|-------|-----------|
| Model | XGBoost / LightGBM / CatBoost |
| Serving | FastAPI + Uvicorn |
| UI | Streamlit + Plotly |
| Experiment tracking | MLflow |
| Data versioning | DVC |
| CI/CD | GitHub Actions |
| Containerization | Docker + Docker Compose |

## Project Structure

```text
.
├── app/
│   ├── main.py               # FastAPI app
│   └── streamlit_app.py      # Streamlit UI
├── configs/
│   ├── config.yaml           # Runtime config
│   └── model_config.yaml     # Best model params
├── data/
│   ├── raw/                  # DVC-tracked raw CSVs
│   └── processed/            # Feature-engineered CSVs
├── scripts/
│   ├── run_pipeline.py       # Training pipeline entrypoint
│   ├── monitor.py            # Drift monitoring
│   └── retrain.py            # Auto-retrain
├── src/rossmann_mlops/       # Core library
├── tests/                    # Smoke tests
├── .github/workflows/        # CI/CD workflows
├── Dockerfile
└── docker-compose.yml
```

## Quick Start (Docker Compose)

```bash
# Copy environment variables
cp .env.example .env

# Build & run all services
docker compose up --build
```

- Streamlit UI: http://localhost:8501
- FastAPI:       http://localhost:8000
- API Docs:      http://localhost:8000/docs

## Local Setup

```bash
# Install dependencies
pip install -e .
pip install -e .[dev]

# Pull data via DVC
dvc pull data/raw/train.csv data/raw/store.csv data/raw/test.csv
```

## Run Training Pipeline

```bash
python scripts/run_pipeline.py --config configs/config.yaml
```

Windows:

```bat
scripts\run_pipeline.bat
```

Output artifacts:

- Model: `artifacts/models/rossmann_model.joblib`
- Metrics: `artifacts/metrics/metrics.json`

## Run API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Run Streamlit UI

```bash
streamlit run app/streamlit_app.py --server.port 8501
```

Health check:

```bash
curl http://localhost:8000/health
```

Prediction example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "records": [{
      "Store": 1,
      "DayOfWeek": 5,
      "Date": "2015-09-17",
      "Open": 1,
      "Promo": 1,
      "StateHoliday": "0",
      "SchoolHoliday": 0
    }]
  }'
```

## Monitoring

```bash
python scripts/monitor.py --current data/raw/test.csv
```

Reads reference from `data/raw/train.csv`, detects drift, writes logs to `logs/`.

## GitHub Actions Secrets

Set these in **Settings → Secrets & Variables → Actions**:

| Key | Type | Description |
|-----|------|-------------|
| `DOCKERHUB_TOKEN` | Secret | DockerHub access token |
| `DOCKERHUB_USERNAME` | Variable | DockerHub username |
| `AWS_ACCESS_KEY_ID` | Secret | S3/DVC access (optional) |
| `AWS_SECRET_ACCESS_KEY` | Secret | S3/DVC secret (optional) |
| `MLFLOW_TRACKING_URI` | Secret | MLflow server URI (optional) |


Run retraining:

```bash
python scripts/retrain.py --config configs/config.yaml
```

Output logs:

- `logs/monitoring_report.jsonl`
- `logs/performance.jsonl`
- `logs/alerts.jsonl`

## Reproducibility

This project is set up to support reproducible runs:

- `requirements.txt` lists the main dependencies for local setup.
- `configs/config.yaml` keeps the training seed and experiment log path in one place.
- `src/rossmann_mlops/train.py` fixes the global seed before training.
- Each training run writes an experiment record to `logs/experiments.jsonl`.
- The experiment record stores the model hash and DVC data hashes from `data/*.dvc`.

If you want to re-run the exact same setup, keep the same data version from DVC and the same seed value.

## Run Tests

```bash
pytest -q
```

## CI

GitHub Actions workflow file:

- `.github/workflows/ci.yml`

Pipeline actions:

- install dependencies
- run test suite

## Rubric Mapping (MLOps Pipeline)

- Pipeline train -> eval -> save model: `src/rossmann_mlops/train.py`
- API (FastAPI): `app/main.py`
- GitHub Actions CI: `.github/workflows/ci.yml`
- Automation script: `scripts/run_pipeline.py` and `scripts/run_pipeline.bat`
- Config file: `configs/config.yaml`
- Monitoring scripts: `scripts/monitor.py` and `scripts/retrain.py`
- Reproducibility: `requirements.txt`, `configs/config.yaml`, `logs/experiments.jsonl`
