from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.special import expit

from fakenews.config.settings import MODEL_FILE
from fakenews.models.explain import explain_text_prediction
from fakenews.utils.io import load_joblib


@dataclass
class Prediction:
    label: int
    confidence: float


class FakeNewsPredictor:
    def __init__(self, model_path=MODEL_FILE):
        loaded = load_joblib(model_path)
        self.pipeline: Any
        self.metadata: dict[str, Any]
        if isinstance(loaded, dict) and "pipeline" in loaded:
            self.pipeline = loaded["pipeline"]
            self.metadata = loaded.get("metadata", {})
        else:
            self.pipeline = loaded
            self.metadata = {"selected_model": "legacy_model"}

    def _get_confidence(self, texts: list[str]) -> np.ndarray:
        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(texts)
            return np.max(probs, axis=1)

        if hasattr(self.pipeline, "decision_function"):
            decision = self.pipeline.decision_function(texts)
            if np.ndim(decision) == 1:
                probs = expit(decision)
                return np.maximum(probs, 1 - probs)
            probs = expit(decision)
            return np.max(probs, axis=1)

        return np.ones(len(texts), dtype=float) * 0.5

    def predict_one(self, text: str) -> Prediction:
        labels = self.pipeline.predict([text])
        confidence_arr = self._get_confidence([text])
        label = int(labels[0])
        confidence = float(confidence_arr[0])
        return Prediction(label=label, confidence=confidence)

    def predict_batch(self, texts: Iterable[str]) -> list[Prediction]:
        texts = list(texts)
        labels = self.pipeline.predict(texts)
        confidence = self._get_confidence(texts)

        return [
            Prediction(label=int(lbl), confidence=float(conf))
            for lbl, conf in zip(labels, confidence, strict=True)
        ]

    def explain(self, text: str, top_n: int = 8) -> dict:
        return explain_text_prediction(self.pipeline, text=text, top_n=top_n)
