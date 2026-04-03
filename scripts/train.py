from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fakenews.models.train import read_last_metrics, train_model
from fakenews.config.settings import FEATURE_IMPORTANCE_FILE, MODEL_COMPARISON_FILE, TRAINING_SUMMARY_FILE


if __name__ == "__main__":
    result = train_model()
    metrics = read_last_metrics()
    print("Training complete")
    print(f"Selected model: {result.selected_model}")
    print(f"Model: {result.model_path}")
    print(f"Metrics: {result.metrics_path}")
    print(f"Model comparison: {MODEL_COMPARISON_FILE}")
    print(f"Feature importance: {FEATURE_IMPORTANCE_FILE}")
    print(f"Training summary: {TRAINING_SUMMARY_FILE}")
    print(
        "Accuracy={:.4f}, Precision={:.4f}, Recall={:.4f}, F1={:.4f}".format(
            metrics.get("accuracy", 0),
            metrics.get("precision", 0),
            metrics.get("recall", 0),
            metrics.get("f1", 0),
        )
    )
