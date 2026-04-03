from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fakenews.config.settings import FEATURE_IMPORTANCE_FILE, METRICS_FILE, MODEL_COMPARISON_FILE
from fakenews.utils.io import load_json


if __name__ == "__main__":
    print("=== Model Metrics ===")
    if METRICS_FILE.exists():
        metrics = load_json(METRICS_FILE)
        print(
            "Selected={selected_model} | Accuracy={accuracy:.4f} | Precision={precision:.4f} | Recall={recall:.4f} | F1={f1:.4f}".format(
                selected_model=metrics.get("selected_model", "unknown"),
                accuracy=metrics.get("accuracy", 0),
                precision=metrics.get("precision", 0),
                recall=metrics.get("recall", 0),
                f1=metrics.get("f1", 0),
            )
        )
    else:
        print("Metrics file not found.")

    print("\n=== Cross Validation Comparison ===")
    if MODEL_COMPARISON_FILE.exists():
        comp = pd.read_csv(MODEL_COMPARISON_FILE)
        print(comp.to_string(index=False))
    else:
        print("Model comparison file not found.")

    print("\n=== Top Feature Importance ===")
    if FEATURE_IMPORTANCE_FILE.exists():
        features = load_json(FEATURE_IMPORTANCE_FILE)
        if features.get("supported"):
            print("Top fake tokens:")
            for item in features.get("top_fake_tokens", [])[:10]:
                print(f"  {item['token']}: {item['weight']:.4f}")
            print("Top real tokens:")
            for item in features.get("top_real_tokens", [])[:10]:
                print(f"  {item['token']}: {item['weight']:.4f}")
        else:
            print(features.get("reason", "Feature importance unavailable."))
    else:
        print("Feature importance file not found.")
