"""Behaviour tests for utilities, loaders' validation logic and result naming."""

import os

import pandas as pd
import pytest

from src.data import validate_dair_label_names, validate_pandora_frame, load_french_emotions
from src.data.emotion import french_text_column
from src.utils import get_paths, result_filename, save_figure, save_table, set_seed, save_json_config


def test_result_filename_convention():
    assert result_filename("english_emotion", "model_comparison", "csv") == "english_emotion__model_comparison.csv"
    assert (
        result_filename("personality", "trait_f1", ".csv", model="Ridge (TF-IDF)", split="val")
        == "personality__trait_f1__ridge-tf-idf__val.csv"
    )


def test_save_table_and_no_overwrite(tmp_path):
    df = pd.DataFrame({"a": [1, 2]})
    p = save_table(df, tmp_path, "eda", "x", split="train")
    assert p.name == "eda__x__train.csv" and p.exists()
    with pytest.raises(FileExistsError):
        save_table(df, tmp_path, "eda", "x", split="train", overwrite=False)


def test_save_figure(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1])
    assert save_figure(fig, tmp_path, "eda", "plot").exists()


def test_paths_are_env_configurable(tmp_path, monkeypatch):
    monkeypatch.setenv("ITC_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("ITC_RESULTS_DIR", str(tmp_path / "r"))
    p = get_paths(create=True)
    assert p.data_dir == (tmp_path / "d").resolve() and p.tables_dir == (tmp_path / "r" / "tables").resolve()
    assert p.tables_dir.is_dir()


def test_default_paths_do_not_contain_original_machine_paths():
    for var in ("ITC_DATA_DIR", "ITC_RESULTS_DIR", "ITC_CACHE_DIR", "ITC_CHECKPOINT_DIR"):
        os.environ.pop(var, None)
    p = get_paths()
    assert "/content" not in str(p.data_dir) and "PANDORA_Project" not in str(p.data_dir)


def test_set_seed_is_deterministic():
    import numpy as np

    set_seed(7)
    a = np.random.rand(3)
    set_seed(7)
    assert (a == np.random.rand(3)).all()


def test_dair_label_validation():
    validate_dair_label_names(["sadness", "joy", "love", "anger", "fear", "surprise"])
    with pytest.raises(ValueError):
        validate_dair_label_names(["sadness", "joy"])
    with pytest.raises(ValueError):
        validate_dair_label_names(["a", "b", "c", "d", "e", "f"])


def test_pandora_validation():
    good = pd.DataFrame({"text": ["a"], "O": [1.0], "C": [2.0], "E": [3.0], "A": [4.0], "N": [100.0]})
    validate_pandora_frame(good)
    with pytest.raises(KeyError):
        validate_pandora_frame(good.drop(columns=["N"]))
    with pytest.raises(ValueError):
        validate_pandora_frame(good.assign(O=101.0))


def test_french_loader_validation(tmp_path):
    with pytest.raises(FileNotFoundError, match="not distributed"):
        load_french_emotions(tmp_path / "missing.csv")
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1]}).to_csv(bad, index=False)
    with pytest.raises(KeyError):
        load_french_emotions(bad)
    ok = tmp_path / "ok.csv"
    pd.DataFrame({"fr_text_cleaned": ["salut"], "emotion": ["joy"]}).to_csv(ok, index=False)
    df = load_french_emotions(ok)
    assert french_text_column(df) == "fr_text_cleaned"


def test_save_json_config(tmp_path):
    from pathlib import Path

    out = save_json_config({"p": Path("/x"), "n": 1}, tmp_path / "c.json")
    assert out.read_text().count("/x") == 1
