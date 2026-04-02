# Fake News Detection - End-to-End AI/ML Project

A production-style machine learning project for fake news classification with:
- Data preprocessing pipeline
- TF-IDF + Logistic Regression training flow
- Model artifact management
- FastAPI inference service
- Streamlit analytics UI
- Batch prediction CLI
- Unit tests
- Docker support

## 1) Project Structure

```text
fake news/
├─ artifacts/
├─ data/
│  ├─ raw/
│  │  └─ news.csv
│  └─ processed/
├─ scripts/
│  ├─ batch_predict.py
│  ├─ run_api.py
│  └─ train.py
├─ src/
│  └─ fakenews/
│     ├─ api/
│     │  └─ main.py
│     ├─ app/
│     │  └─ streamlit_app.py
│     ├─ config/
│     │  └─ settings.py
│     ├─ data/
│     │  └─ preprocess.py
│     ├─ features/
│     │  └─ vectorizer.py
│     ├─ models/
│     │  ├─ evaluate.py
│     │  ├─ predict.py
│     │  └─ train.py
│     └─ utils/
│        ├─ io.py
│        └─ logger.py
├─ tests/
├─ Dockerfile
├─ docker-compose.yml
├─ pyproject.toml
└─ requirements.txt
```

## 2) Quick Start

### One-time easy setup (Windows PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

### Direct run (API + Dashboard)

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_local.ps1
```

Or double-click:
- `run_project.bat`

### Step A: Create environment and install dependencies

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step B: Train model

```bash
python scripts/train.py
```

Outputs:
- `artifacts/model_pipeline.joblib`
- `artifacts/metrics.json`
- `data/processed/train.csv`
- `data/processed/test.csv`

### Step C: Run API

```bash
python scripts/run_api.py
```

API docs:
- `http://localhost:8000/docs`

### Step D: Run Streamlit app

```bash
set PYTHONPATH=src
streamlit run src/fakenews/app/streamlit_app.py
```

## 3) API Endpoints

- `GET /health`
- `POST /predict`

Request body:

```json
{
  "text": "Breaking story text..."
}
```

- `POST /predict/batch`

Request body:

```json
{
  "texts": ["news item 1", "news item 2"]
}
```

## 4) Batch Prediction CLI

```bash
python scripts/batch_predict.py --input sample_input.csv --output sample_output.csv
```

Input CSV must contain a `text` column.

## 5) Testing

```bash
pytest
```

## 6) How to Scale This Project Further

- Replace sample dataset with larger benchmark datasets (LIAR, FakeNewsNet, ISOT)
- Add MLflow for experiment tracking
- Add model registry and CI/CD deployment
- Add data validation with Great Expectations
- Add drift monitoring and scheduled retraining
- Add transformer model benchmark (DistilBERT, RoBERTa)

## 7) Notes

- Label convention: `1 = FAKE`, `0 = REAL`
- Current baseline model: TF-IDF with Logistic Regression
- This setup is strong as a baseline and can be extended to deep learning models.

## 8) Push to GitHub

```powershell
git init
git add .
git commit -m "Initial fake news detection ML project"
git branch -M main
git remote add origin https://github.com/NAITIK74600/fake-news-.git
git push -u origin main
```
