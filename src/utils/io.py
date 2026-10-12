"""Result-file naming convention and table / figure writers.

Convention (all lower-case, parts joined by ``__``)::

    {task}__{artifact}[__{model}][__{split}].{ext}

* ``task``     one of ``english_emotion``, ``french_emotion``, ``personality``,
               ``eda``, ``llm``
* ``artifact`` what the file contains (``model_comparison``, ``confusion_matrix``,
               ``classification_report``, ``trait_f1`` …)
* ``model``    optional model identifier (``logistic-regression``, ``gru``, ``ensemble`` …;
               spaces and punctuation become ``-``, underscores are kept)
* ``split``    optional evaluation split (``val`` or ``test``)

Files re-exported from the *original* notebook's recorded outputs carry an extra
``__src-cell<N>`` suffix so they stay traceable to the notebook cell they came
from (see ``results/README.md``).  Files written here never overwrite another
experiment's output unless the identical name is requested.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]+", "-", str(value).strip().lower())
    return value.strip("-")


def result_filename(
    task: str,
    artifact: str,
    ext: str,
    model: Optional[str] = None,
    split: Optional[str] = None,
) -> str:
    """Build a file name following the project's naming convention."""
    parts = [_slug(task), _slug(artifact)]
    if model:
        parts.append(_slug(model))
    if split:
        parts.append(_slug(split))
    return "__".join(parts) + "." + ext.lstrip(".")


def save_table(
    df: pd.DataFrame,
    tables_dir: Path,
    task: str,
    artifact: str,
    model: Optional[str] = None,
    split: Optional[str] = None,
    index: bool = False,
    overwrite: bool = True,
) -> Path:
    """Write ``df`` as CSV under ``tables_dir`` using the naming convention."""
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / result_filename(task, artifact, "csv", model, split)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    df.to_csv(path, index=index)
    return path


def save_figure(
    fig,
    figures_dir: Path,
    task: str,
    artifact: str,
    model: Optional[str] = None,
    split: Optional[str] = None,
    dpi: int = 150,
    overwrite: bool = True,
) -> Path:
    """Write a Matplotlib figure as PNG under ``figures_dir``."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / result_filename(task, artifact, "png", model, split)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
