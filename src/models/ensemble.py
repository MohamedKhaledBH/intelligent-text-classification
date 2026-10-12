"""Stacked Ridge ensemble: ``[SVD(TF-IDF) | multi-task BERT predictions]`` → Ridge.

Protocol (notebook S7.2):

* The ridge of each trait is fitted on the **held-out** block of the training
  split (rows *not* used to train BERT), with continuous targets in [0, 1].
* The decision threshold of each trait is chosen on the **validation** split
  (grid 0.20–0.80, maximising macro-F1) and then applied unchanged to the test
  split.  The test split is never used for any selection.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from ..evaluation.metrics import evaluate_binary, find_best_threshold
from .classical import make_ridge


def fit_stacked_ridge_ensemble(
    X_train: np.ndarray,
    y_train_reg: np.ndarray,
    X_val: np.ndarray,
    y_val_bin: np.ndarray,
    X_test: np.ndarray,
    y_test_bin: np.ndarray,
    traits: Sequence[str],
    trait_names: Dict[str, str],
    alpha: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Return ``(val_results, test_results, best_thresholds)``."""
    val_rows, test_rows, best_thresholds = [], [], {}
    for i, c in enumerate(traits):
        ridge = make_ridge(alpha)
        ridge.fit(X_train, y_train_reg[:, i])
        pred_val = ridge.predict(X_val)
        pred_test = ridge.predict(X_test)

        best_t, _ = find_best_threshold(y_val_bin[:, i], pred_val)
        best_thresholds[c] = best_t

        m_val = evaluate_binary(y_val_bin[:, i], (pred_val >= best_t).astype(int))
        m_val.update(dict(Trait=c, Trait_Name=trait_names[c], Model="Ensemble"))
        val_rows.append(m_val)

        m_test = evaluate_binary(y_test_bin[:, i], (pred_test >= best_t).astype(int))
        m_test.update(dict(Trait=c, Trait_Name=trait_names[c], Model="Ensemble"))
        test_rows.append(m_test)
    return pd.DataFrame(val_rows), pd.DataFrame(test_rows), best_thresholds
