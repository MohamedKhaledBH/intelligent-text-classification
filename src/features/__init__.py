"""Feature extraction: TF-IDF, truncated SVD and sentence embeddings."""

from .tfidf import (
    DAIR_TFIDF_CFG,
    EMOTION_PIPELINE_TFIDF_CFG,
    PANDORA_TFIDF_CFG,
    fit_transform_tfidf,
)
from .dimensionality import fit_truncated_svd
from .embeddings import encode_with_sentence_transformer, load_sentence_encoder

__all__ = [
    "DAIR_TFIDF_CFG",
    "EMOTION_PIPELINE_TFIDF_CFG",
    "PANDORA_TFIDF_CFG",
    "fit_transform_tfidf",
    "fit_truncated_svd",
    "encode_with_sentence_transformer",
    "load_sentence_encoder",
]
