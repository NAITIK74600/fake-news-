import json

import pandas as pd
import streamlit as st

from fakenews.config.settings import METRICS_FILE, MODEL_FILE
from fakenews.models.predict import FakeNewsPredictor
from fakenews.utils.io import load_json

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.title("Fake News Detection Studio")
st.caption("Binary classifier powered by TF-IDF + Logistic Regression")

if not MODEL_FILE.exists():
    st.error("Model artifact not found. Run training first: `python scripts/train.py`")
    st.stop()

predictor = FakeNewsPredictor(MODEL_FILE)

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Single Prediction")
    user_text = st.text_area("Paste a headline or news article", height=250)
    if st.button("Analyze", use_container_width=True):
        if len(user_text.strip()) < 5:
            st.warning("Please provide at least 5 characters of text.")
        else:
            output = predictor.predict_one(user_text)
            verdict = "FAKE" if output.label == 1 else "REAL"
            st.metric("Prediction", verdict)
            st.progress(min(max(output.confidence, 0.0), 1.0), text=f"Confidence: {output.confidence:.2%}")

with col_right:
    st.subheader("Model Metrics")
    if METRICS_FILE.exists():
        metrics = load_json(METRICS_FILE)
        st.json(
            {
                "accuracy": round(metrics.get("accuracy", 0), 4),
                "precision": round(metrics.get("precision", 0), 4),
                "recall": round(metrics.get("recall", 0), 4),
                "f1": round(metrics.get("f1", 0), 4),
            }
        )
    else:
        st.info("No metrics file found yet.")

st.divider()
st.subheader("Batch Prediction")
upload = st.file_uploader("Upload CSV with a `text` column", type=["csv"])
if upload is not None:
    frame = pd.read_csv(upload)
    if "text" not in frame.columns:
        st.error("CSV must contain a `text` column.")
    else:
        outputs = predictor.predict_batch(frame["text"].fillna("").tolist())
        frame["label"] = [p.label for p in outputs]
        frame["confidence"] = [p.confidence for p in outputs]
        frame["verdict"] = frame["label"].map({1: "FAKE", 0: "REAL"})
        st.dataframe(frame.head(50), use_container_width=True)
        st.download_button(
            "Download predictions",
            frame.to_csv(index=False).encode("utf-8"),
            file_name="batch_predictions.csv",
            mime="text/csv",
        )
