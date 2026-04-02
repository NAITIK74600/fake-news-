from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fakenews.models.predict import Prediction


def test_prediction_dataclass_fields():
    pred = Prediction(label=1, confidence=0.91)
    assert pred.label == 1
    assert pred.confidence == 0.91


def test_sklearn_pipeline_can_predict_probabilities():
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    x = ["this is fake news", "official statement confirms report"]
    y = [1, 0]
    pipeline.fit(x, y)

    probs = pipeline.predict_proba(["fake report with no source"])
    assert probs.shape == (1, 2)
