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
    input_file = PROJECT_ROOT / "data" / "raw" / "testing_samples.csv"
    output_file = PROJECT_ROOT / "artifacts" / "testing_results.csv"

    if not input_file.exists():
        raise FileNotFoundError(f"Testing file not found: {input_file}")

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Model artifact not found. Train model first with: python scripts/train.py"
        )

    frame = pd.read_csv(input_file)
    if "text" not in frame.columns:
        raise ValueError("Testing CSV must contain a 'text' column")

    predictor = FakeNewsPredictor(MODEL_FILE)
    preds = predictor.predict_batch(frame["text"].fillna("").tolist())

    out = frame.copy()
    out["predicted_label"] = [p.label for p in preds]
    out["confidence"] = [p.confidence for p in preds]

    # Convert confidence to explicit REAL/FAKE percentages that always total 100.
    out["fake_percent"] = out.apply(
        lambda row: row["confidence"] * 100.0 if row["predicted_label"] == 1 else (1.0 - row["confidence"]) * 100.0,
        axis=1,
    )
    out["real_percent"] = 100.0 - out["fake_percent"]
    out["result_type"] = out["predicted_label"].map({1: "FAKE", 0: "REAL"})

    out["fake_percent"] = out["fake_percent"].round(2)
    out["real_percent"] = out["real_percent"].round(2)
    out["confidence"] = out["confidence"].round(4)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")
    print(out[["result_type", "fake_percent", "real_percent"]].to_string(index=False))


if __name__ == "__main__":
    main()
