"""TF-IDF configurations used in the notebook and a fit-on-train helper.

Three distinct configurations exist in the original notebook:

``DAIR_TFIDF_CFG``              Part 1 (English dair-ai, 70/15/15).
``PANDORA_TFIDF_CFG``           Part 2 (PANDORA); ``lowercase=False`` because the
                                text is lower-cased by the cleaner.
``EMOTION_PIPELINE_TFIDF_CFG``  S9 (EN/FR logistic regression / SVM, 80/20).

The vectoriser is always **fitted on the training split only**.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer

DAIR_TFIDF_CFG: Dict = dict(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)

PANDORA_TFIDF_CFG: Dict = dict(
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True,
    max_features=150_000,
    lowercase=False,
)

EMOTION_PIPELINE_TFIDF_CFG: Dict = dict(
    max_features=50_000, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True
)


def fit_transform_tfidf(
    train_texts: Sequence[str], other_texts: Sequence[Sequence[str]] = (), **cfg
) -> Tuple[TfidfVectorizer, object, Tuple[object, ...]]:
    """Fit a TF-IDF vectoriser on ``train_texts``; transform train and the others.

    Returns ``(vectorizer, X_train, (X_other_1, X_other_2, ...))``.
    """
    vectorizer = TfidfVectorizer(**cfg)
    X_train = vectorizer.fit_transform(train_texts)
    others = tuple(vectorizer.transform(texts) for texts in other_texts)
    return vectorizer, X_train, others
