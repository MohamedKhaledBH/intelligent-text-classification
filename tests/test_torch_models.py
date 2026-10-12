"""Torch / Transformers tests.

These are **skipped automatically** when PyTorch (or transformers) is not
installed.  They were written for the refactor but could not be executed in the
sandbox used to prepare it (no PyTorch wheel could be installed there); see
REFACTORING_REPORT.md → "Checks actually executed".  Run them locally with
``pytest tests/test_torch_models.py -v``.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from conftest import extract_nodes  # noqa: E402

C_DAIR_DL, C_PANDORA_DL, C_FR_DL_A, C_FR_DL_S4 = 10, 59, 80, 86
VOCAB = 50


def _exec_class(nb, idx, name, extra=None):
    ns = {"nn": nn, "torch": torch, "np": np, **(extra or {})}
    exec(compile(extract_nodes(nb, idx, kinds=(__import__("ast").ClassDef,), names={name}), f"<cell {idx}>", "exec"), ns)
    return ns[name]


def _params(m):
    return [p.detach().clone() for p in m.parameters()]


def _assert_same_init_and_forward(orig, mine, x):
    assert [tuple(p.shape) for p in orig.parameters()] == [tuple(p.shape) for p in mine.parameters()]
    for a, b in zip(_params(orig), _params(mine)):
        assert torch.equal(a, b)
    orig.eval(), mine.eval()
    assert torch.allclose(orig(x), mine(x))


@pytest.mark.parametrize("rnn_type", ["LSTM", "GRU", "BiLSTM"])
def test_rnn_matches_original_english(original_nb, rnn_type):
    from src.models.rnn import RNNClassifier

    Orig = _exec_class(original_nb, C_DAIR_DL, "RNNModel")
    torch.manual_seed(0); orig = Orig(rnn_type, VOCAB)
    torch.manual_seed(0); mine = RNNClassifier(rnn_type, VOCAB, num_outputs=6)
    _assert_same_init_and_forward(orig, mine, torch.randint(0, VOCAB, (4, 64)))


@pytest.mark.parametrize("rnn_type", ["LSTM", "GRU", "BiLSTM"])
def test_rnn_matches_original_pandora(original_nb, rnn_type):
    from src.models.rnn import RNNClassifier

    Orig = _exec_class(original_nb, C_PANDORA_DL, "MultiTraitRNN")
    torch.manual_seed(0); orig = Orig(VOCAB, rnn_type=rnn_type)
    torch.manual_seed(0); mine = RNNClassifier(rnn_type, VOCAB, num_outputs=5)
    _assert_same_init_and_forward(orig, mine, torch.randint(0, VOCAB, (4, 128)))


@pytest.mark.parametrize("rnn_type", ["LSTM", "GRU", "BiLSTM"])
def test_rnn_matches_original_french(original_nb, rnn_type):
    from src.models.rnn import RNNClassifier

    OrigA = _exec_class(original_nb, C_FR_DL_A, "FrenchRNN", dict(vocab={i: i for i in range(VOCAB - 2)}, EMBEDDING_DIM=128, HIDDEN_UNITS=64))
    torch.manual_seed(0); orig = OrigA(rnn_type)
    torch.manual_seed(0); mine = RNNClassifier(rnn_type, VOCAB, num_outputs=6)
    _assert_same_init_and_forward(orig, mine, torch.randint(0, VOCAB, (4, 100)))
    OrigB = _exec_class(original_nb, C_FR_DL_S4, "FrenchRNNModel")
    torch.manual_seed(0); orig_b = OrigB(rnn_type, VOCAB, num_classes=6)
    torch.manual_seed(0); mine_b = RNNClassifier(rnn_type, VOCAB, num_outputs=6)
    _assert_same_init_and_forward(orig_b, mine_b, torch.randint(0, VOCAB, (4, 64)))


def test_unsupported_rnn_type_rejected():
    from src.models.rnn import RNNClassifier

    with pytest.raises(ValueError):
        RNNClassifier("Transformer", VOCAB)


def _toy_loaders():
    from src.models.rnn import make_loader

    rng = np.random.default_rng(0)
    X = rng.integers(2, VOCAB, (128, 10))
    y = (X[:, 0] % 3).astype(int)
    return make_loader(X, y, 32, shuffle=True), make_loader(X, y, 64)


def test_fixed_epoch_training_reduces_loss_and_predicts():
    from src.models.rnn import RNNClassifier, predict_classes, train_fixed_epochs

    torch.manual_seed(0)
    train, evalu = _toy_loaders()
    model = RNNClassifier("GRU", VOCAB, num_outputs=3)
    seen = []
    hist, secs = train_fixed_epochs(
        model, train, nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), 1e-2), 8, "cpu",
        on_epoch_end=lambda e, l: seen.append(e),
    )
    assert len(hist) == 8 and seen == list(range(8)) and hist[-1] < hist[0]
    preds, _ = predict_classes(model, evalu, "cpu")
    assert len(preds) == 128 and set(preds) <= {0, 1, 2}


def test_early_stopping_restores_best_weights(tmp_path):
    from src.models.rnn import RNNClassifier, train_with_early_stopping

    torch.manual_seed(0)
    train, val = _toy_loaders()
    model = RNNClassifier("LSTM", VOCAB, num_outputs=3)
    opt = torch.optim.Adam(model.parameters(), 1e-2)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=1)
    ckpt = tmp_path / "best.pt"
    out = train_with_early_stopping(model, train, val, nn.CrossEntropyLoss(), opt, sched, 6, 2, "cpu", ckpt)
    assert ckpt.exists() and len(out["val_losses"]) <= 6
    assert out["best_val_loss"] == min(out["val_losses"])
    saved = torch.load(ckpt, weights_only=True)
    for k, v in model.state_dict().items():
        assert torch.equal(v, saved[k])


def test_predict_sigmoid_shape():
    from src.models.rnn import RNNClassifier, make_loader, predict_sigmoid

    X = np.random.default_rng(0).integers(2, VOCAB, (20, 8))
    loader = make_loader(X, np.zeros((20, 5)), 7, y_dtype=torch.float32)
    probs = predict_sigmoid(RNNClassifier("BiLSTM", VOCAB, num_outputs=5), loader, "cpu")
    assert probs.shape == (20, 5) and ((probs >= 0) & (probs <= 1)).all()


# ------------------------------------------------------------------ transformers
def test_multitrait_bert_with_tiny_encoder_and_metrics():
    pytest.importorskip("transformers")
    from transformers import DistilBertConfig, DistilBertModel

    from src.models.hf_finetuning import MultiTraitBERT, make_multitrait_compute_metrics

    cfg = DistilBertConfig(vocab_size=100, dim=32, hidden_dim=64, n_layers=4, n_heads=2, max_position_embeddings=32)
    model = MultiTraitBERT(freeze_layers=3, encoder=DistilBertModel(cfg))
    frozen = [all(not p.requires_grad for p in layer.parameters()) for layer in model.encoder.transformer.layer]
    assert frozen == [True, True, True, False]
    ids = torch.randint(0, 100, (3, 10))
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=torch.rand(3, 5))
    assert out["logits"].shape == (3, 5) and out["loss"].ndim == 0

    thr = dict(O=39.0, C=69.0, E=31.0, A=33.0, N=50.0)
    fn = make_multitrait_compute_metrics(thr, list("OCEAN"))
    labels = np.array([[0.5, 0.8, 0.4, 0.4, 0.6], [0.1, 0.1, 0.1, 0.1, 0.1]])
    m = fn((labels.copy(), labels))
    assert m["macro_f1_avg"] == 1.0 and set(m) == {"macro_f1_avg", *[f"f1_{c}" for c in "OCEAN"]}


def test_training_arguments_shim():
    pytest.importorskip("transformers")
    from src.models.hf_finetuning import build_training_arguments, multitrait_training_kwargs

    args = build_training_arguments(cpu=True, **multitrait_training_kwargs("tmp_out", cpu_mode=True))
    assert args.num_train_epochs == 1 and args.per_device_train_batch_size == 4
    assert args.learning_rate == 2e-5 and args.weight_decay == 0.1 and args.seed == 42
