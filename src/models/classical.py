"""Classical scikit-learn models with the notebook's exact hyper-parameters."""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def dair_classical_models(seed: int = 42) -> Dict[str, object]:
    """English dair-ai (Part 1). Class-balanced weights except Naive Bayes."""
    return {
        "TF-IDF + Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed
        ),
        "TF-IDF + Linear SVM": LinearSVC(class_weight="balanced", random_state=seed, max_iter=2000),
        "TF-IDF + Naive Bayes": MultinomialNB(),
        "TF-IDF + SGD Classifier": SGDClassifier(class_weight="balanced", random_state=seed),
    }


def pandora_binary_models(seed: int = 42) -> Dict[str, object]:
    """PANDORA binary classification baselines (one model per trait, trained on 300k rows)."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed, n_jobs=-1),
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(max_iter=2000, random_state=seed, dual=False),
    }


def emotion_pipeline_models() -> Dict[str, object]:
    """S9 EN/FR pipeline. NOTE: the notebook sets *no* random_state here."""
    return {
        "LogReg": LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1),
        "LinearSVM": LinearSVC(class_weight="balanced", max_iter=3000),
    }


def make_ridge(alpha: float = 1.0, seed: Optional[int] = None) -> Ridge:
    """Ridge regressor. The TF-IDF baseline passes ``random_state``; the ensemble does not."""
    return Ridge(alpha=alpha, random_state=seed) if seed is not None else Ridge(alpha=alpha)


def fit_predict_timed(clf, X_train, y_train, X_test) -> Tuple[np.ndarray, float, float]:
    """Fit then predict, returning ``(predictions, train_seconds, inference_seconds)``."""
    t0 = time.time()
    clf.fit(X_train, y_train)
    t_train = time.time() - t0
    t1 = time.time()
    pred = clf.predict(X_test)
    t_inf = time.time() - t1
    return pred, t_train, t_inf
