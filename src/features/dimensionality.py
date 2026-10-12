"""Truncated SVD of the PANDORA TF-IDF matrix (input of the stacking ensemble)."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD

SVD_DIM = 300


def fit_truncated_svd(
    X_train, other_matrices: Sequence = (), n_components: int = SVD_DIM, seed: int = 42
) -> Tuple[TruncatedSVD, np.ndarray, Tuple[np.ndarray, ...]]:
    """Fit ``TruncatedSVD(n_components, random_state=seed)`` on the training matrix.

    Outputs are cast to ``float32`` as in the notebook.
    """
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    train = svd.fit_transform(X_train).astype(np.float32)
    others = tuple(svd.transform(X).astype(np.float32) for X in other_matrices)
    return svd, train, others
