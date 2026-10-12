"""LIME explanations (notebook §9 and S11).

``lime`` is an optional dependency.  The English notebook explains a logistic
regression pipeline (TF-IDF + classifier) for the predicted class of the first
misclassified test sample; the PANDORA notebook explains a small
50k-sample TF-IDF + logistic-regression pipeline for the *Extraversion* binary
label on one test text.  These are single-instance, qualitative illustrations,
not an evaluation of explanation quality.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple


def explain_instance(
    predict_proba,
    text: str,
    class_names: Sequence[str],
    num_features: int = 8,
    top_labels: int | None = 1,
):
    """Return ``(explanation, [(word, weight), ...], label_index)``.

    With ``top_labels=1`` the explained label is the model's top predicted class
    (English notebook).  With ``top_labels=None`` LIME's default (label 1) is
    used (PANDORA binary notebook).
    """
    from lime.lime_text import LimeTextExplainer  # lazy import

    explainer = LimeTextExplainer(class_names=list(class_names))
    kwargs = dict(num_features=num_features)
    if top_labels is not None:
        kwargs["top_labels"] = top_labels
    exp = explainer.explain_instance(text, predict_proba, **kwargs)
    label = exp.available_labels()[0]
    weights: List[Tuple[str, float]] = exp.as_list(label=label)
    return exp, weights, label
