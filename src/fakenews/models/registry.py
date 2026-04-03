from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from fakenews.config.settings import RANDOM_STATE


def get_model_registry() -> dict[str, object]:
    """Return candidate estimators used in model selection."""
    return {
        "logistic_regression": LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "sgd_logistic": SGDClassifier(
            loss="log_loss",
            max_iter=3000,
            alpha=1e-5,
            random_state=RANDOM_STATE,
        ),
        "complement_nb": ComplementNB(alpha=0.6),
        "linear_svc_calibrated": CalibratedClassifierCV(
            estimator=LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
            cv=3,
        ),
    }
