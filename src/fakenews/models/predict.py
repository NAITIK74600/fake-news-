from dataclasses import dataclass
from typing import Iterable

import numpy as np

from fakenews.config.settings import MODEL_FILE
from fakenews.utils.io import load_joblib


@dataclass
class Prediction:
    label: int
    confidence: float


class FakeNewsPredictor:
    def __init__(self, model_path=MODEL_FILE):
        self.pipeline = load_joblib(model_path)

    def predict_one(self, text: str) -> Prediction:
        labels = self.pipeline.predict([text])
        probabilities = self.pipeline.predict_proba([text])
        label = int(labels[0])
        confidence = float(np.max(probabilities[0]))
        return Prediction(label=label, confidence=confidence)

    def predict_batch(self, texts: Iterable[str]) -> list[Prediction]:
        texts = list(texts)
        labels = self.pipeline.predict(texts)
        probabilities = self.pipeline.predict_proba(texts)
        confidence = np.max(probabilities, axis=1)

        return [
            Prediction(label=int(lbl), confidence=float(conf))
            for lbl, conf in zip(labels, confidence, strict=True)
        ]
