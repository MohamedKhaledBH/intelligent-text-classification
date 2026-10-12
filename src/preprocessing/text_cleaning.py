"""Text cleaning functions, copied behaviour-for-behaviour from the notebook.

Two *different* cleaners exist in the original notebook and are intentionally
kept distinct:

``clean_text_dair``      (Part 1, English emotions) lower-case, replace every
                         character outside ``[a-z0-9'\\s]`` by a space, collapse
                         whitespace.
``clean_text_pandora``   (Part 2, PANDORA) additionally NaN → ``""``, removes
                         ``@mentions`` and HTML entities (``&amp;``) and
                         newlines/tabs *before* the same character filter.

Neither function strips accents or handles French specifically: the French
pipeline uses the pre-computed ``fr_text_cleaned`` column of the local CSV and
only trims whitespace (``prepare_emotion``).  Note that the ``[^a-z0-9']`` filter
would delete accented letters (``é``, ``à`` …), which is why it is *not* applied
to French text.
"""

from __future__ import annotations

import re
from typing import Tuple

import pandas as pd


def clean_text_dair(text) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_text_pandora(text) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"@\w+\b", " ", text)
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_pandora_splits(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add ``text_clean`` and drop rows whose cleaned text is empty (notebook S2.1)."""
    out = []
    for d in (df_train, df_val, df_test):
        d = d.copy()
        d["text_clean"] = d["text"].astype(str).map(clean_text_pandora)
        d = d[d["text_clean"].str.strip().ne("")].reset_index(drop=True)
        out.append(d)
    return tuple(out)  # type: ignore[return-value]


def prepare_emotion(
    df: pd.DataFrame, text_col: str, min_class_count_exclusive: int = 50
) -> pd.DataFrame:
    """Notebook S9.2 ``prepare_emotion``.

    Keeps ``[text_col, "emotion"]``, drops NaN, strips the text, drops empty
    texts and duplicate ``(text, emotion)`` pairs, then removes classes with
    ``<= min_class_count_exclusive`` samples (the original condition is
    ``count > 50``).  Returns columns ``text`` and ``emotion``.
    """
    d = df[[text_col, "emotion"]].dropna().copy()
    d.columns = ["text", "emotion"]
    d["text"] = d["text"].astype(str).str.strip()
    d = d[d["text"] != ""].drop_duplicates(["text", "emotion"]).reset_index(drop=True)
    counts = d["emotion"].value_counts()
    valid_classes = counts[counts > min_class_count_exclusive].index
    return d[d["emotion"].isin(valid_classes)].reset_index(drop=True)
