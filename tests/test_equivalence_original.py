"""Refactored (torch-free) code vs. the original notebook's executable source."""

from __future__ import annotations

import ast
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from conftest import cell_source, exec_definitions, find_calls

# Cell indices in the original notebook (see REFACTORING_REPORT.md).
C_DAIR_PREP, C_DAIR_SPLIT, C_DAIR_CLASSICAL, C_DAIR_DL = 6, 7, 9, 10
C_PANDORA_CLEAN, C_PANDORA_HELPERS, C_PANDORA_TFIDF, C_PANDORA_ML = 28, 49, 51, 54
C_PANDORA_ENSEMBLE, C_EMOTION_PIPE, C_FR_DL_A = 72, 78, 80

TEXTS = [
    "I feel awful!! :( it's my job",
    "@john check &amp; this\tout\nnow",
    "Ünïcode café — déjà vu… 100% sure?",
    "   multiple    spaces   ",
    "",
    "!!!",
    "it's O'Neil's x_y 42",
    None,
    float("nan"),
]


# --------------------------------------------------------------------------- cleaning
def test_clean_text_dair_matches_original(original_nb):
    from src.preprocessing import clean_text_dair

    orig = exec_definitions(original_nb, C_DAIR_PREP, ["clean_text_dair"], {"re": re})["clean_text_dair"]
    for t in TEXTS:
        assert clean_text_dair(t) == orig(t), repr(t)


def test_clean_text_pandora_matches_original(original_nb):
    from src.preprocessing import clean_text_pandora

    orig = exec_definitions(original_nb, C_PANDORA_CLEAN, ["clean_text"], {"re": re, "pd": pd})["clean_text"]
    for t in TEXTS:
        assert clean_text_pandora(t) == orig(t), repr(t)


def test_clean_pandora_splits_drops_empty_and_matches_loop(original_nb):
    from src.preprocessing import clean_pandora_splits

    tr = pd.DataFrame({"text": ["Hello World", "!!!", "@a", "ok fine"], "O": [1, 2, 3, 4]})
    va = pd.DataFrame({"text": ["x y", ""], "O": [1, 2]})
    te = pd.DataFrame({"text": ["Zed"], "O": [1]})
    a, b, c = clean_pandora_splits(tr, va, te)
    assert list(a["text_clean"]) == ["hello world", "ok fine"]
    assert list(b["text_clean"]) == ["x y"]
    assert list(c["text_clean"]) == ["zed"]
    assert a.index.tolist() == [0, 1]  # reset_index


def test_prepare_emotion_matches_original(original_nb):
    from src.preprocessing import prepare_emotion

    orig = exec_definitions(original_nb, C_EMOTION_PIPE, ["prepare_emotion"], {"pd": pd})["prepare_emotion"]
    labels = ["joy"] * 120 + ["sadness"] * 80 + ["rare"] * 10 + ["edge"] * 51 + ["edge50"] * 50
    df = pd.DataFrame({"col": [f"text {i % 200}  " for i in range(len(labels))], "emotion": labels})
    df.loc[3, "col"] = None
    df.loc[5, "col"] = "   "
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    got = prepare_emotion(df, "col")
    want = orig(df, "col")
    pd.testing.assert_frame_equal(got, want)
    # boundary: strictly more than 50 samples are kept
    assert "edge50" not in set(got["emotion"]) or (want["emotion"] == "edge50").sum() > 50


# --------------------------------------------------------------------------- sequences
def test_vocab_and_encoding_match_original(original_nb):
    from src.preprocessing import build_vocab, encode_texts
    from src.preprocessing.sequences import embedding_vocab_size

    src = cell_source(original_nb, C_DAIR_DL)
    prefix = src.split("X_train_dl = encode_dl(X_train_raw)")[0]  # vocab + encode_dl, no torch use
    train_texts = pd.Series(["i feel good today", "good day good", "bad bad day i feel", "x"] * 5)
    ns = dict(np=np, pd=pd, X_train_raw=train_texts, Counter=__import__("collections").Counter)
    exec(compile(prefix, "<orig cell 10 prefix>", "exec"), ns)

    vocab = build_vocab(train_texts, ns["VOCAB_SIZE"])
    assert vocab == ns["vocab"]
    assert embedding_vocab_size(vocab) == len(ns["vocab"]) + 2

    probe = ["good feel zzz unknown", "i " * 100, "", "bad"]
    assert np.array_equal(encode_texts(probe, vocab, ns["MAX_LEN"]), ns["encode_dl"](probe))


# --------------------------------------------------------------------------- metrics
def _orig_metrics_ns(original_nb):
    ns = dict(
        np=np, f1_score=f1_score, accuracy_score=accuracy_score,
        precision_score=precision_score, recall_score=recall_score,
    )
    ns.update(exec_definitions(original_nb, C_PANDORA_HELPERS, ["evaluate_binary", "find_best_threshold"], ns))
    return ns


def test_evaluate_binary_and_threshold_match_original(original_nb):
    from src.evaluation import evaluate_binary, find_best_threshold

    ns = _orig_metrics_ns(original_nb)
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    score = np.clip(0.5 + 0.2 * (y - 0.5) + rng.normal(0, 0.2, 500), 0, 1)
    assert evaluate_binary(y, (score >= 0.5).astype(int)) == ns["evaluate_binary"](y, (score >= 0.5).astype(int))
    got, want = find_best_threshold(y, score), ns["find_best_threshold"](y, score)
    assert got[0] == want[0] and got[1] == want[1]
    # degenerate case: constant prediction keeps the notebook's default start value
    assert find_best_threshold(np.array([0, 0, 0]), np.array([0.1, 0.1, 0.1]))[0] == ns["find_best_threshold"](
        np.array([0, 0, 0]), np.array([0.1, 0.1, 0.1])
    )[0]


def test_threshold_grid_edges_match_original(original_nb):
    """Optimum located at the first / last grid value must be found identically."""
    from src.evaluation import find_best_threshold

    ns = _orig_metrics_ns(original_nb)
    grid = np.arange(0.20, 0.81, 0.01)
    # Floating-point quirk of the original expression: the grid has 62 points and its
    # last value is ~0.81, not 0.80 (documented in docs/methodology.md).
    assert len(grid) == 62 and abs(grid[-1] - 0.81) < 1e-9
    n = 20
    y = np.array([0] * n + [1] * n)
    upper = np.array([grid[-2]] * n + [grid[-1]] * n)  # perfect split only at the last grid value
    lower = np.array([grid[0] - 0.01] * n + [grid[0]] * n)  # perfect split already at the first value
    for score in (upper, lower):
        assert find_best_threshold(y, score) == ns["find_best_threshold"](y, score)
    assert find_best_threshold(y, upper)[0] == grid[-1]


def test_compute_classification_metrics_matches_original(original_nb):
    from src.evaluation import compute_classification_metrics

    ns = dict(
        accuracy_score=accuracy_score, precision_score=precision_score,
        recall_score=recall_score, f1_score=f1_score,
    )
    ns.update(exec_definitions(original_nb, C_DAIR_CLASSICAL, ["evaluate_model"], ns))
    y = np.array([0, 1, 2, 1, 0, 2, 2, 1, 0, 0])
    p = np.array([0, 2, 2, 1, 0, 1, 2, 1, 1, 0])
    assert compute_classification_metrics(y, p, "m", "r", 1.5, 0.5) == ns["evaluate_model"](y, p, "m", "r", 1.5, 0.5)


# --------------------------------------------------------------------------- splits
def test_english_split_matches_original(original_nb):
    from src.data import split_train_val_test_via_temp

    rng = np.random.default_rng(1)
    n = 1000
    df = pd.DataFrame(
        {"text_clean": [f"t{i}" for i in range(n)], "label": rng.choice(6, n, p=[.3, .3, .1, .15, .1, .05])}
    )
    ns = dict(df_dair=df, SEED=42, train_test_split=train_test_split)
    exec(compile(cell_source(original_nb, C_DAIR_SPLIT), "<orig cell 7>", "exec"), ns)
    Xtr, Xva, Xte, ytr, yva, yte = split_train_val_test_via_temp(df["text_clean"], df["label"], seed=42)
    for mine, theirs in [(Xtr, ns["X_train_raw"]), (Xva, ns["X_val_raw"]), (Xte, ns["X_test_raw"]),
                         (ytr, ns["y_train"]), (yva, ns["y_val"]), (yte, ns["y_test"])]:
        assert list(mine.index) == list(theirs.index)


def test_french_partA_split_and_subsample_match_original(original_nb):
    from src.data import split_holdout_test_then_val, subsample_stratified

    src = cell_source(original_nb, C_FR_DL_A)
    start = src.index("X_train_full, X_test, y_train_full, y_test")
    end = src.index("# Tokenization")
    snippet = src[start:end]
    rng = np.random.default_rng(2)
    n = 3000
    texts = [f"mot {i}" for i in range(n)]
    y = rng.choice(6, n)
    ns = dict(texts_fr=texts, y_fr=y, SEED=42, DL_TRAIN_SIZE=1500, train_test_split=train_test_split)
    exec(compile(snippet, "<orig cell 80 split>", "exec"), ns)
    Xtr, Xva, Xte, ytr, yva, yte = split_holdout_test_then_val(texts, y, 0.15, 0.15, seed=42)
    assert len(Xtr) > 1500
    Xtr_s, ytr_s = subsample_stratified(Xtr, ytr, 1500, seed=42)
    assert Xte == ns["X_test"] and Xva == ns["X_val"] and Xtr_s == ns["X_train"]
    assert np.array_equal(ytr_s, ns["y_train"]) and np.array_equal(yte, ns["y_test"])


def test_french_80_20_split_matches_original():
    from src.data import split_train_test

    rng = np.random.default_rng(3)
    texts = [f"mot {i}" for i in range(500)]
    y = rng.choice(6, 500)
    a = split_train_test(texts, y, 0.2, 42)
    b = train_test_split(texts, y, test_size=0.2, stratify=y, random_state=42)  # literal original call
    assert a[0] == b[0] and a[1] == b[1]


def test_bert_ensemble_indices_disjoint_and_seeded():
    from src.data import make_bert_ensemble_indices, sample_validation_indices

    a1, b1 = make_bert_ensemble_indices(10_000, 6_000, 1_000, seed=42)
    a2, b2 = make_bert_ensemble_indices(10_000, 6_000, 1_000, seed=42)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)
    assert not set(a1.tolist()) & set(b1.tolist())
    all_idx = np.random.default_rng(42).permutation(10_000)  # literal original expression
    assert np.array_equal(a1, all_idx[:6000]) and np.array_equal(b1, all_idx[6000:7000])
    v = sample_validation_indices(10_000, 100, seed=42)
    assert np.array_equal(v, np.random.default_rng(43).choice(10_000, 100, replace=False))


# --------------------------------------------------------------------------- features / models
def test_tfidf_configs_match_original(original_nb):
    from src.features import DAIR_TFIDF_CFG, EMOTION_PIPELINE_TFIDF_CFG, PANDORA_TFIDF_CFG

    assert find_calls(original_nb, C_DAIR_CLASSICAL, "TfidfVectorizer")[0] == {
        k: v for k, v in DAIR_TFIDF_CFG.items()
    }
    ns = {}
    src = cell_source(original_nb, C_PANDORA_TFIDF)
    exec(src.split("vec = TfidfVectorizer")[0], {"TfidfVectorizer": TfidfVectorizer, **ns}, ns)
    assert ns["TFIDF_CFG"] == PANDORA_TFIDF_CFG
    assert find_calls(original_nb, C_EMOTION_PIPE, "TfidfVectorizer")[0] == EMOTION_PIPELINE_TFIDF_CFG


def _models_from_dict_assignment(nb, idx, seed=42):
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC

    tree = ast.parse(cell_source(nb, idx))
    src = cell_source(nb, idx)
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "models":
            ns = dict(SEED=seed, LogisticRegression=LogisticRegression, SGDClassifier=SGDClassifier,
                      MultinomialNB=MultinomialNB, LinearSVC=LinearSVC)
            return eval(ast.get_source_segment(src, node.value), ns)
    raise AssertionError("models dict not found")


def _same_params(a, b):
    assert type(a) is type(b)
    assert a.get_params() == b.get_params()


def test_dair_classical_models_match_original(original_nb):
    from src.models import dair_classical_models

    orig, mine = _models_from_dict_assignment(original_nb, C_DAIR_CLASSICAL), dair_classical_models(42)
    assert list(orig) == list(mine)
    for k in orig:
        _same_params(orig[k], mine[k])


def test_pandora_binary_models_match_original(original_nb):
    from src.models import pandora_binary_models

    orig, mine = _models_from_dict_assignment(original_nb, C_PANDORA_ML), pandora_binary_models(42)
    assert list(orig) == list(mine)
    for k in orig:
        _same_params(orig[k], mine[k])


def test_emotion_pipeline_models_match_original(original_nb):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    from src.models import emotion_pipeline_models

    src = cell_source(original_nb, C_EMOTION_PIPE)
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_emotion_pipeline")
    lst = next(n for n in ast.walk(fn) if isinstance(n, ast.For) and isinstance(n.iter, ast.List)).iter
    orig = dict(eval(ast.get_source_segment(src, lst), {"LogisticRegression": LogisticRegression, "LinearSVC": LinearSVC}))
    mine = emotion_pipeline_models()
    assert list(orig) == list(mine)
    for k in orig:
        _same_params(orig[k], mine[k])


def test_ridge_factory_matches_original():
    from src.models import make_ridge

    assert make_ridge(1.0, 42).get_params() == Ridge(alpha=1.0, random_state=42).get_params()
    assert make_ridge(1.0).get_params() == Ridge(alpha=1.0).get_params()


def test_stacked_ensemble_matches_original_loop(original_nb):
    from src.models import fit_stacked_ridge_ensemble

    traits = ["O", "C", "E", "A", "N"]
    names = dict(O="Openness", C="Conscientiousness", E="Extraversion", A="Agreeableness", N="Neuroticism")
    rng = np.random.default_rng(5)
    Xtr, Xva, Xte = (rng.normal(size=(n, 12)) for n in (300, 200, 200))
    w = rng.normal(size=(12, 5))
    ytr = np.clip(0.5 + Xtr @ w * 0.05 + rng.normal(0, .1, (300, 5)), 0, 1).astype(np.float32)
    yva_reg = np.clip(0.5 + Xva @ w * 0.05 + rng.normal(0, .1, (200, 5)), 0, 1)
    yte_reg = np.clip(0.5 + Xte @ w * 0.05 + rng.normal(0, .1, (200, 5)), 0, 1)
    yva_bin, yte_bin = (yva_reg >= 0.5).astype(int), (yte_reg >= 0.5).astype(int)

    ns = _orig_metrics_ns(original_nb)
    src = cell_source(original_nb, C_PANDORA_ENSEMBLE)
    loop = src[src.index("ensemble_val_results = []"): src.index("ensemble_val_df = ")]
    ns.update(dict(Ridge=Ridge, TRAITS=traits, TRAIT_NAMES=names, X_ens_tr=Xtr, y_ens_tr=ytr,
                   X_ens_val=Xva, y_ens_val_bin=yva_bin, X_ens_test=Xte, y_ens_test_bin=yte_bin))
    exec(compile(loop, "<orig cell 72 loop>", "exec"), ns)

    val_df, test_df, thr = fit_stacked_ridge_ensemble(Xtr, ytr, Xva, yva_bin, Xte, yte_bin, traits, names)
    pd.testing.assert_frame_equal(val_df, pd.DataFrame(ns["ensemble_val_results"]))
    pd.testing.assert_frame_equal(test_df, pd.DataFrame(ns["ensemble_test_results"]))
    assert thr == ns["BEST_THRESHOLDS"]


# --------------------------------------------------------------------------- labels
def test_label_helpers():
    from src.preprocessing import binarize_traits, compute_train_medians, scale_traits

    tr = pd.DataFrame({"O": [10, 20, 30, 40], "C": [0, 50, 50, 100]})
    thr = compute_train_medians(tr, ["O", "C"])
    assert thr == {"O": 25.0, "C": 50.0}
    te = pd.DataFrame({"O": [24, 25, 26], "C": [49, 50, 51]})
    assert binarize_traits(te, ["O", "C"], thr).values.tolist() == [[0, 0], [1, 1], [1, 1]]  # ">= median"
    assert scale_traits(te, ["O"])["O"].tolist() == [0.24, 0.25, 0.26]
