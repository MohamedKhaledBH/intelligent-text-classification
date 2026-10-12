"""End-to-end smoke tests on tiny synthetic data (no downloads, no GPU).

They check that the pieces compose (cleaning → TF-IDF → models → metrics → tables
and plots).  They do **not** reproduce the paper-scale numbers.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from src.data import split_train_val_test_via_temp
from src.evaluation import (
    DAIR_RESULT_COLUMNS,
    combine_french_benchmark,
    compute_classification_metrics,
    mean_macro_f1_by_model,
    misclassified_examples,
    plot_confusion_matrix,
    plot_trait_comparison,
    text_classification_report,
)
from src.features import DAIR_TFIDF_CFG, fit_transform_tfidf, fit_truncated_svd
from src.models import dair_classical_models, fit_predict_timed
from src.preprocessing import clean_text_dair

LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
VOCAB = {
    0: "sad cry tears alone", 1: "happy joy glad smile", 2: "love adore care dear",
    3: "angry mad furious rage", 4: "afraid scared fear panic", 5: "wow surprised shock sudden",
}


def _synthetic_emotions(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 6, n)
    texts = [
        "I Feel " + " ".join(rng.choice(VOCAB[k].split(), 3)) + " " + rng.choice(["today!", "so much.", "again?"])
        for k in y
    ]
    return pd.DataFrame({"text": texts, "label": y})


def test_english_classical_pipeline_smoke():
    df = _synthetic_emotions()
    df["text_clean"] = df["text"].map(clean_text_dair)
    Xtr, Xva, Xte, ytr, yva, yte = split_train_val_test_via_temp(df["text_clean"], df["label"], seed=42)
    assert len(Xtr) + len(Xva) + len(Xte) == len(df)
    vec, Xtr_t, (Xva_t, Xte_t) = fit_transform_tfidf(Xtr, [Xva, Xte], **DAIR_TFIDF_CFG)
    rows = []
    for name, clf in dair_classical_models(42).items():
        pred, t_tr, t_inf = fit_predict_timed(clf, Xtr_t, ytr, Xte_t)
        rows.append(compute_classification_metrics(yte, pred, name.split(" + ")[1], "TF-IDF", t_tr, t_inf))
    table = pd.DataFrame(rows)[DAIR_RESULT_COLUMNS]
    assert list(table.columns) == DAIR_RESULT_COLUMNS and len(table) == 4
    assert table["Accuracy"].max() > 0.9  # separable synthetic data

    clf = dair_classical_models(42)["TF-IDF + Logistic Regression"].fit(Xtr_t, ytr)
    pred = clf.predict(Xte_t)
    assert "macro avg" in text_classification_report(yte, pred, LABELS)
    fig = plot_confusion_matrix(yte, pred, LABELS, "cm")
    assert fig is not None
    assert isinstance(misclassified_examples(list(Xte), yte.values, pred, LABELS), list)


def test_svd_shapes_and_dtype():
    from scipy.sparse import random as sprandom

    X = sprandom(80, 50, density=0.2, random_state=0, format="csr")
    svd, tr, (va,) = fit_truncated_svd(X, [X[:10]], n_components=5, seed=42)
    assert tr.shape == (80, 5) and va.shape == (10, 5) and tr.dtype == np.float32


def test_trait_reporting_helpers():
    rows = [
        dict(Trait=t, Trait_Name=t, Model=m, Macro_F1=f)
        for t in "OCEAN" for m, f in [("A", 0.58), ("B", 0.55)]
    ]
    df = pd.DataFrame(rows)
    summary = mean_macro_f1_by_model(df)
    assert summary.iloc[0]["Model"] == "A"
    assert plot_trait_comparison(df, "t") is not None


def test_french_benchmark_combination():
    cls = pd.DataFrame({"Model": ["LogReg"], "Accuracy": [0.7], "Macro_F1": [0.65], "Weighted_F1": [0.68], "lang": ["FR"]})
    dl = pd.DataFrame({"Model": ["GRU"], "Accuracy": [0.6], "Macro-F1": [0.5], "Weighted-F1": [0.55], "Training Time": [1.0]})
    out = combine_french_benchmark(cls, dl)
    assert list(out.columns) == ["Model", "Accuracy", "Macro-F1", "Weighted-F1"] and len(out) == 2


def test_lime_helper_on_tiny_pipeline():
    import pytest

    pytest.importorskip("lime")
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer

    from src.evaluation.explain import explain_instance

    df = _synthetic_emotions(300)
    pipe = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=500)).fit(df["text"], df["label"])
    exp, weights, label = explain_instance(pipe.predict_proba, df["text"].iloc[0], LABELS, num_features=4)
    assert 0 <= label < 6 and 1 <= len(weights) <= 4


def test_emotion_pipeline_and_french_rows():
    from sklearn.metrics import f1_score

    from src.evaluation import compute_emotion_pipeline_metrics, french_benchmark_row

    y, p = [0, 1, 2, 1, 0, 2], [0, 1, 1, 1, 0, 2]
    r = compute_emotion_pipeline_metrics(y, p, "LogReg", "FR")
    assert list(r) == ["lang", "Model", "Accuracy", "Macro_F1", "Weighted_F1"]
    assert r["Macro_F1"] == f1_score(y, p, average="macro")
    f = french_benchmark_row(y, p, "GRU", 1.0, 0.1)
    assert list(f) == ["Model", "Accuracy", "Macro Precision", "Macro Recall", "Macro-F1",
                       "Weighted-F1", "Training Time", "Inference Time"]
