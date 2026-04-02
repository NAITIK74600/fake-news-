from pathlib import Path
import sys

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


if __name__ == "__main__":
    uvicorn.run("fakenews.api.main:app", host="0.0.0.0", port=8000, reload=True)
