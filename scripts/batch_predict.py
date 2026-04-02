import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fakenews.config.settings import MODEL_FILE
from fakenews.models.predict import FakeNewsPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch inference for fake news detection")
    parser.add_argument("--input", required=True, help="Path to input CSV with `text` column")
    parser.add_argument("--output", required=True, help="Path to save output CSV")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    if "text" not in frame.columns:
        raise ValueError("Input CSV must have a `text` column")

    predictor = FakeNewsPredictor(MODEL_FILE)
    preds = predictor.predict_batch(frame["text"].fillna("").tolist())
    frame["label"] = [p.label for p in preds]
    frame["confidence"] = [p.confidence for p in preds]
    frame["verdict"] = frame["label"].map({1: "FAKE", 0: "REAL"})
    frame.to_csv(args.output, index=False)
    print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()
