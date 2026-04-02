import pandas as pd

from fakenews.data.preprocess import clean_text, prepare_features


def test_clean_text_removes_urls_and_symbols():
    text = "Breaking!!! Visit https://example.com NOW"
    assert clean_text(text) == "breaking visit now"


def test_prepare_features_creates_combined_text_and_label():
    df = pd.DataFrame(
        {
            "title": ["Hello World"],
            "text": ["This is a Test"],
            "label": [1],
        }
    )
    output = prepare_features(df)

    assert "combined_text" in output.columns
    assert "label" in output.columns
    assert output.iloc[0]["label"] == 1
