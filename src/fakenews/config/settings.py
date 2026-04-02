from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RAW_DATA_FILE = RAW_DATA_DIR / "news.csv"
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test.csv"
MODEL_FILE = ARTIFACTS_DIR / "model_pipeline.joblib"
METRICS_FILE = ARTIFACTS_DIR / "metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
