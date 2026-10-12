"""Metric definitions identical to the notebook's helper functions."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_classification_metrics(
    y_true, y_pred, model_name: str, rep_name: str, t_train: float, t_inf: float
) -> Dict:
    """Part 1 ``evaluate_model``: accuracy, macro P/R/F1, weighted F1 and timings.

    ``zero_division=0`` throughout, as in the notebook.
    """
    return {
        "Model": model_name,
        "Representation": rep_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Weighted F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Training Time": t_train,
        "Inference Time": t_inf,
    }


def evaluate_binary(y_true, y_pred) -> Dict:
    """PANDORA ``evaluate_binary``: accuracy and macro-averaged P/R/F1."""
    return dict(
        Accuracy=accuracy_score(y_true, y_pred),
        Macro_Precision=precision_score(y_true, y_pred, average="macro", zero_division=0),
        Macro_Recall=recall_score(y_true, y_pred, average="macro", zero_division=0),
        Macro_F1=f1_score(y_true, y_pred, average="macro", zero_division=0),
    )


def find_best_threshold(y_true_bin, y_score) -> Tuple[float, float]:
    """Threshold in ``arange(0.20, 0.81, 0.01)`` maximising macro-F1 (first maximum wins).

    **Must be called on validation data only.** Initial values ``(0.5, 0.0)``
    and the strict ``>`` comparison are those of the notebook.
    """
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.20, 0.81, 0.01):
        f1 = f1_score(y_true_bin, (y_score >= t).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def compute_emotion_pipeline_metrics(y_true, y_pred, model_name: str, lang: str) -> Dict:
    """S9 EN/FR pipeline row: accuracy, macro-F1, weighted-F1 (``zero_division=0``)."""
    return dict(
        lang=lang,
        Model=model_name,
        Accuracy=accuracy_score(y_true, y_pred),
        Macro_F1=f1_score(y_true, y_pred, average="macro", zero_division=0),
        Weighted_F1=f1_score(y_true, y_pred, average="weighted", zero_division=0),
    )


def french_benchmark_row(y_true, y_pred, model_name: str, t_train: float, t_inf: float) -> Dict:
    """French Part A / CamemBERT row (hyphenated ``Macro-F1`` / ``Weighted-F1`` names)."""
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro-F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Weighted-F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Training Time": t_train,
        "Inference Time": t_inf,
    }
