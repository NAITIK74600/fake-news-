from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from fakenews.config.settings import (
    CV_FOLDS,
    FEATURE_IMPORTANCE_FILE,
    METRICS_FILE,
    MODEL_COMPARISON_FILE,
    MODEL_FILE,
    RANDOM_STATE,
    RAW_DATA_FILE,
    TEST_DATA_FILE,
    TRAIN_DATA_FILE,
    TRAINING_SUMMARY_FILE,
)
from fakenews.data.preprocess import prepare_features, split_and_save
from fakenews.features.vectorizer import build_vectorizer
from fakenews.models.explain import extract_global_feature_importance
from fakenews.models.evaluate import evaluate_predictions
from fakenews.models.registry import get_model_registry
from fakenews.utils.io import load_json, save_joblib, save_json
from fakenews.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    model_path: str
    metrics_path: str
    train_rows: int
    test_rows: int
    selected_model: str


def build_pipeline(estimator) -> Pipeline:
    return Pipeline(steps=[("tfidf", build_vectorizer()), ("clf", estimator)])


def compare_models(x_train: pd.Series, y_train: pd.Series) -> tuple[pd.DataFrame, str]:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "f1": "f1",
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
    }

    results: list[dict] = []
    for model_name, estimator in get_model_registry().items():
        pipeline = build_pipeline(estimator)
        scores = cross_validate(
            pipeline,
            x_train.to_numpy(),
            y_train.to_numpy(),
            scoring=scoring,
            cv=cv,
            n_jobs=1,
            return_train_score=False,
        )
        results.append(
            {
                "model": model_name,
                "cv_f1_mean": float(scores["test_f1"].mean()),
                "cv_f1_std": float(scores["test_f1"].std()),
                "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
                "cv_precision_mean": float(scores["test_precision"].mean()),
                "cv_recall_mean": float(scores["test_recall"].mean()),
            }
        )

    comparison_df = pd.DataFrame(results).sort_values("cv_f1_mean", ascending=False)
    best_model_name = str(comparison_df.iloc[0]["model"])
    return comparison_df, best_model_name


def train_model(raw_data_path=RAW_DATA_FILE) -> TrainingResult:
    logger.info("Loading raw dataset from %s", raw_data_path)
    raw_df = pd.read_csv(raw_data_path)

    logger.info("Preparing text features and labels")
    model_df = prepare_features(raw_df)

    split_stats = split_and_save(model_df)
    logger.info("Created train/test split: %s", split_stats)

    train_df = pd.read_csv(TRAIN_DATA_FILE)
    test_df = pd.read_csv(TEST_DATA_FILE)

    comparison_df, selected_model = compare_models(train_df["combined_text"], train_df["label"])
    MODEL_COMPARISON_FILE.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(MODEL_COMPARISON_FILE, index=False)
    logger.info("Best model selected from CV: %s", selected_model)

    estimator = get_model_registry()[selected_model]
    pipeline = build_pipeline(estimator)
    logger.info("Training selected model pipeline")
    pipeline.fit(train_df["combined_text"], train_df["label"])

    logger.info("Running evaluation")
    preds = pipeline.predict(test_df["combined_text"])
    metrics = evaluate_predictions(test_df["label"], preds)
    metrics["selected_model"] = selected_model
    metrics["cv_comparison_file"] = str(MODEL_COMPARISON_FILE)

    explanation = extract_global_feature_importance(pipeline)

    save_joblib({"pipeline": pipeline, "metadata": {"selected_model": selected_model}}, MODEL_FILE)
    save_json(metrics, METRICS_FILE)
    save_json(explanation, FEATURE_IMPORTANCE_FILE)
    save_json(
        {
            "selected_model": selected_model,
            "train_rows": split_stats["train_rows"],
            "test_rows": split_stats["test_rows"],
            "metrics_file": str(METRICS_FILE),
            "model_file": str(MODEL_FILE),
            "feature_importance_file": str(FEATURE_IMPORTANCE_FILE),
            "comparison_file": str(MODEL_COMPARISON_FILE),
        },
        TRAINING_SUMMARY_FILE,
    )

    logger.info("Model saved to %s", MODEL_FILE)
    logger.info("Metrics saved to %s", METRICS_FILE)

    return TrainingResult(
        model_path=str(MODEL_FILE),
        metrics_path=str(METRICS_FILE),
        train_rows=split_stats["train_rows"],
        test_rows=split_stats["test_rows"],
        selected_model=selected_model,
    )


def read_last_metrics() -> dict:
    return load_json(METRICS_FILE)
