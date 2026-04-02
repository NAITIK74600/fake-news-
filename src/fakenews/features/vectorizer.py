from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer() -> TfidfVectorizer:
    """Create a TF-IDF vectorizer tuned for short and medium news text."""
    return TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=60000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
