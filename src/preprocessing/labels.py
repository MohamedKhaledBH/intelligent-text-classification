"""PANDORA label preparation: [0, 1] regression targets and median binarisation.

The binary label of a trait is ``score >= median`` where the median is computed
**on the training split only** and re-used for validation and test (this is how
the notebook avoids look-ahead bias).  Predictions are binarised with
``median / 100`` because regression targets are ``score / 100``.
"""

from __future__ import annotations

from typing import Dict, Sequence

import pandas as pd


def compute_train_medians(df_train: pd.DataFrame, traits: Sequence[str]) -> Dict[str, float]:
    return {c: float(df_train[c].median()) for c in traits}


def scale_traits(df: pd.DataFrame, traits: Sequence[str]) -> pd.DataFrame:
    """Continuous targets in [0, 1] (``score / 100``)."""
    return pd.DataFrame({c: df[c] / 100.0 for c in traits})


def binarize_traits(
    df: pd.DataFrame, traits: Sequence[str], thresholds: Dict[str, float]
) -> pd.DataFrame:
    """Binary targets ``score >= threshold`` (thresholds in the 0–100 scale)."""
    return pd.DataFrame({c: (df[c] >= thresholds[c]).astype(int) for c in traits})
