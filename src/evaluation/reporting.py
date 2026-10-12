"""Tables, classification reports and plots that reproduce the notebook's outputs."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

#: Column order of the English comparison table (notebook cell 12).
DAIR_RESULT_COLUMNS = [
    "Model",
    "Representation",
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
    "Weighted F1",
    "Training Time",
    "Inference Time",
]


def text_classification_report(y_true, y_pred, target_names: Sequence[str]) -> str:
    return classification_report(y_true, y_pred, target_names=list(target_names), zero_division=0)


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: Sequence[str],
    title: str,
    cmap: str = "Blues",
    figsize=(8, 6),
    xlabel: str = "Predicted Emotion",
    ylabel: str = "Actual Emotion",
):
    """Annotated confusion-matrix heat-map (counts). Returns the Matplotlib figure."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap, xticklabels=list(labels), yticklabels=list(labels), ax=ax
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    return fig


def misclassified_examples(
    texts: Sequence[str], y_true, y_pred, label_names: Sequence[str], n: int = 5
) -> List[dict]:
    """First ``n`` misclassified samples (notebook §8 "Sample Misclassifications")."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    idx = np.where(y_true != y_pred)[0]
    return [
        dict(
            index=int(i),
            text=texts[i],
            actual=label_names[y_true[i]],
            predicted=label_names[y_pred[i]],
        )
        for i in idx[:n]
    ]


def mean_macro_f1_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Mean macro-F1 over traits per model, sorted descending (PANDORA summaries)."""
    return (
        df.groupby("Model")["Macro_F1"].mean().sort_values(ascending=False).reset_index()
    )


def plot_trait_comparison(
    df: pd.DataFrame,
    title: str,
    palette: str = "Set2",
    ylim=(0.45, 0.65),
    figsize=(12, 6),
    x: str = "Trait_Name",
    chance_label: str = "Niveau du Hasard (0.5)",
):
    """Grouped bar plot of macro-F1 per trait and model with a chance-level line."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(data=df, x=x, y="Macro_F1", hue="Model", palette=palette, ax=ax)
    ax.axhline(0.5, color="red", ls="--", lw=1, alpha=0.7, label=chance_label)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Macro F1 Score")
    ax.set_ylim(*ylim)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def combine_french_benchmark(
    classical: pd.DataFrame, dl_transformer: pd.DataFrame
) -> pd.DataFrame:
    """Notebook Part C/D: stack classical and DL/Transformer rows on common columns.

    ``classical`` has ``Macro_F1`` / ``Weighted_F1``; ``dl_transformer`` has
    ``Macro-F1`` / ``Weighted-F1``.  **Caution:** in the original notebook the
    classical rows come from an 80/20 split and the DL/Transformer rows from a
    different 70/15/15 split, so the rows are *not* evaluated on the same test
    set (see ``docs/methodology.md``).
    """
    cls = classical.rename(columns={"Macro_F1": "Macro-F1", "Weighted_F1": "Weighted-F1"})
    cols = ["Model", "Accuracy", "Macro-F1", "Weighted-F1"]
    return pd.concat([cls[cols], dl_transformer[cols]], ignore_index=True)
