from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fakenews.config.settings import FEATURE_IMPORTANCE_FILE, METRICS_FILE, MODEL_FILE
from fakenews.models.predict import FakeNewsPredictor
from fakenews.utils.io import load_json

predictor: FakeNewsPredictor | None = None


class PredictRequest(BaseModel):
    text: str = Field(min_length=5, description="News content to classify")


class PredictBatchRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=100)


class PredictResponse(BaseModel):
    label: int
    confidence: float
    verdict: str


class ExplainRequest(BaseModel):
    text: str = Field(min_length=5)
    top_n: int = Field(default=8, ge=1, le=25)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global predictor
    if MODEL_FILE.exists():
        predictor = FakeNewsPredictor(MODEL_FILE)
    yield


app = FastAPI(
    title="Fake News Detection API",
    version="1.0.0",
    description="Inference API for fake news classification",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
    }


@app.get("/model/info")
def model_info() -> dict:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train model first.")

    metrics = load_json(METRICS_FILE) if METRICS_FILE.exists() else {}
    features = load_json(FEATURE_IMPORTANCE_FILE) if FEATURE_IMPORTANCE_FILE.exists() else {}
    return {
        "selected_model": predictor.metadata.get("selected_model", "unknown"),
        "metrics": metrics,
        "feature_importance": features,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train model first.")

    result = predictor.predict_one(body.text)
    verdict = "FAKE" if result.label == 1 else "REAL"
    return PredictResponse(label=result.label, confidence=result.confidence, verdict=verdict)


@app.post("/predict/batch")
def predict_batch(body: PredictBatchRequest) -> dict:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train model first.")

    predictions = predictor.predict_batch(body.texts)
    results = [
        {
            "label": pred.label,
            "confidence": pred.confidence,
            "verdict": "FAKE" if pred.label == 1 else "REAL",
        }
        for pred in predictions
    ]
    return {"count": len(results), "results": results}


@app.post("/explain")
def explain(body: ExplainRequest) -> dict:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train model first.")
    return predictor.explain(body.text, top_n=body.top_n)
