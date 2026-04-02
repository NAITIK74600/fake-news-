from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fakenews.config.settings import METRICS_FILE, MODEL_FILE, RAW_DATA_FILE, TEST_DATA_FILE, TRAIN_DATA_FILE
from fakenews.data.preprocess import prepare_features, split_and_save
from fakenews.features.vectorizer import build_vectorizer
from fakenews.models.evaluate import evaluate_predictions
from fakenews.utils.io import load_json, save_joblib, save_json
from fakenews.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    model_path: str
    metrics_path: str
    train_rows: int
    test_rows: int


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", build_vectorizer()),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    max_iter=1500,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def train_model(raw_data_path=RAW_DATA_FILE) -> TrainingResult:
    logger.info("Loading raw dataset from %s", raw_data_path)
    raw_df = pd.read_csv(raw_data_path)

    logger.info("Preparing text features and labels")
    model_df = prepare_features(raw_df)

    split_stats = split_and_save(model_df)
    logger.info("Created train/test split: %s", split_stats)

    train_df = pd.read_csv(TRAIN_DATA_FILE)
    test_df = pd.read_csv(TEST_DATA_FILE)

    pipeline = build_pipeline()
    logger.info("Training model pipeline")
    pipeline.fit(train_df["combined_text"], train_df["label"])

    logger.info("Running evaluation")
    preds = pipeline.predict(test_df["combined_text"])
    metrics = evaluate_predictions(test_df["label"], preds)

    save_joblib(pipeline, MODEL_FILE)
    save_json(metrics, METRICS_FILE)

    logger.info("Model saved to %s", MODEL_FILE)
    logger.info("Metrics saved to %s", METRICS_FILE)

    return TrainingResult(
        model_path=str(MODEL_FILE),
        metrics_path=str(METRICS_FILE),
        train_rows=split_stats["train_rows"],
        test_rows=split_stats["test_rows"],
    )


def read_last_metrics() -> dict:
    return load_json(METRICS_FILE)
