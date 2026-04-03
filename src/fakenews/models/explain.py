from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline


def _get_clf(pipeline: Pipeline):
    return pipeline.named_steps.get("clf")


def model_supports_coefficients(pipeline: Pipeline) -> bool:
    clf = _get_clf(pipeline)
    return hasattr(clf, "coef_")


def extract_global_feature_importance(pipeline: Pipeline, top_n: int = 40) -> dict[str, Any]:
    """Extract token-level importances for linear models from TF-IDF coefficients."""
    tfidf = pipeline.named_steps["tfidf"]
    clf = _get_clf(pipeline)

    if clf is None or not hasattr(clf, "coef_"):
        return {
            "supported": False,
            "reason": "Selected model does not expose coefficients",
            "top_fake_tokens": [],
            "top_real_tokens": [],
        }

    features = np.array(tfidf.get_feature_names_out())
    coefs = clf.coef_[0]
    fake_idx = np.argsort(coefs)[-top_n:][::-1]
    real_idx = np.argsort(coefs)[:top_n]

    top_fake = [{"token": str(features[i]), "weight": float(coefs[i])} for i in fake_idx]
    top_real = [{"token": str(features[i]), "weight": float(coefs[i])} for i in real_idx]

    return {
        "supported": True,
        "top_fake_tokens": top_fake,
        "top_real_tokens": top_real,
    }


def explain_text_prediction(pipeline: Pipeline, text: str, top_n: int = 8) -> dict[str, Any]:
    """Explain a single prediction using TF-IDF weighted linear coefficients."""
    tfidf = pipeline.named_steps["tfidf"]
    clf = _get_clf(pipeline)

    if clf is None or not hasattr(clf, "coef_"):
        return {
            "supported": False,
            "reason": "Selected model does not expose coefficients",
            "tokens": [],
        }

    vec = tfidf.transform([text])
    dense = vec.toarray()[0]
    features = np.array(tfidf.get_feature_names_out())
    token_contrib = dense * clf.coef_[0]

    non_zero = np.where(dense > 0)[0]
    if len(non_zero) == 0:
        return {
            "supported": True,
            "tokens": [],
            "message": "No known vocabulary tokens found in input",
        }

    ranked = sorted(non_zero, key=lambda idx: abs(token_contrib[idx]), reverse=True)[:top_n]
    tokens = [
        {
            "token": str(features[idx]),
            "contribution": float(token_contrib[idx]),
            "direction": "FAKE" if token_contrib[idx] >= 0 else "REAL",
        }
        for idx in ranked
    ]

    return {
        "supported": True,
        "tokens": tokens,
    }
