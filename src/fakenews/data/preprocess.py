import re

import pandas as pd
from sklearn.model_selection import train_test_split

from fakenews.config.settings import RANDOM_STATE, TEST_SIZE, TEST_DATA_FILE, TRAIN_DATA_FILE
from fakenews.utils.io import save_json

TEXT_COLUMNS = ["title", "text"]
TARGET_COLUMN = "label"


def clean_text(text: str) -> str:
    """Normalize text by lowercasing and removing URLs/non-alpha chars."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\\S+|www\\.\\S+", " ", text)
    text = re.sub(r"[^a-z\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in [*TEXT_COLUMNS, TARGET_COLUMN] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["combined_text"] = (
        out["title"].fillna("").map(clean_text) + " " + out["text"].fillna("").map(clean_text)
    ).str.strip()
    out[TARGET_COLUMN] = out[TARGET_COLUMN].astype(int)
    return out[["combined_text", TARGET_COLUMN]]


def split_and_save(df: pd.DataFrame) -> dict[str, int]:
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )
    TRAIN_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_DATA_FILE, index=False)
    test_df.to_csv(TEST_DATA_FILE, index=False)

    return {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "total_rows": int(len(df)),
    }


def class_distribution(df: pd.DataFrame) -> dict[str, int]:
    counts = df[TARGET_COLUMN].value_counts().to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def save_data_profile(df: pd.DataFrame, path) -> None:
    profile = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "class_distribution": class_distribution(df),
    }
    save_json(profile, path)
