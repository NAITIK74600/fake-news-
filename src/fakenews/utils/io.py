import json
from pathlib import Path
from typing import Any

import joblib


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(data: dict[str, Any], path: Path) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_joblib(obj: Any, path: Path) -> None:
    ensure_parent(path)
    joblib.dump(obj, path)


def load_joblib(path: Path) -> Any:
    return joblib.load(path)
