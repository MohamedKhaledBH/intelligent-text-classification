"""Recurrent classifiers (LSTM / GRU / BiLSTM) and their training loops.

Requires PyTorch.

One class covers every recurrent model in the notebook.  The English
``RNNModel``, the PANDORA ``MultiTraitRNN`` (5 sigmoid outputs) and the two
French ``FrenchRNN`` / ``FrenchRNNModel`` classes all share the same
architecture ``Embedding → RNN → last time-step → Dropout(0.3) → Linear``
and differ only in ``num_outputs``.  Sub-module creation order (embedding, rnn,
dropout, fc) is the same as in the original so that, for a given seed, the
initial weights are identical.

.. warning::
   **Known methodological caveat, preserved on purpose.**  Sequences are
   *right*-padded with index 0 and the forward pass reads ``out[:, -1, :]``,
   i.e. the hidden state at the last *padded* position for every text shorter
   than ``max_len`` (for the BiLSTM the backward half has only seen the final
   padding token).  This probably explains the near-chance LSTM / BiLSTM scores
   recorded in the notebook (English: accuracy 0.349 / 0.415 vs 0.963 for the
   GRU; PANDORA: macro-F1 ≈ 0.37).  It is documented in ``docs/methodology.md``
   and **not** changed here because correcting it would alter the experiment.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

RNN_TYPES = ("LSTM", "GRU", "BiLSTM")

# --- Hyper-parameter presets, one per original experiment ---------------------
DAIR_RNN = dict(
    max_len=64, vocab_size=10_000, embed_dim=128, hidden_dim=64, dropout=0.3,
    lr=1e-3, epochs=5, batch_size=128, eval_batch_size=256, rnn_types=("LSTM", "GRU", "BiLSTM"),
)
PANDORA_RNN = dict(
    n_train=100_000, n_val=50_000, max_len=128, vocab_size=20_000, embed_dim=128, hidden_dim=64,
    dropout=0.3, lr=2e-3, epochs=3, batch_size=256, eval_batch_size=512,
    rnn_types=("LSTM", "BiLSTM", "GRU"),
)
FRENCH_RNN_PART_A = dict(
    train_size=20_000, vocab_size=20_000, max_len=100, embed_dim=128, hidden_dim=64, dropout=0.3,
    lr=1e-3, epochs=5, batch_size=64, early_stopping_patience=2, scheduler_patience=1,
    class_weighted_loss=True, rnn_types=("GRU", "LSTM", "BiLSTM"),
)
FRENCH_RNN_S4_LEGACY = dict(
    max_len=64, vocab_size=10_000, embed_dim=128, hidden_dim=64, dropout=0.3,
    lr=1e-3, epochs=5, batch_size=128, eval_batch_size=256, rnn_types=("LSTM", "GRU", "BiLSTM"),
)


class RNNClassifier(nn.Module):
    """Embedding → (Bi)LSTM/GRU → last time-step → dropout → linear."""

    def __init__(
        self,
        rnn_type: str,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 64,
        num_outputs: int = 6,
        dropout: float = 0.3,
    ):
        super().__init__()
        if rnn_type not in RNN_TYPES:
            raise ValueError(f"Unsupported rnn_type {rnn_type!r}; expected one of {RNN_TYPES}")
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn_type = rnn_type
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
            out_dim = hidden_dim
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
            out_dim = hidden_dim
        else:  # "BiLSTM"
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
            out_dim = hidden_dim * 2
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        out = out[:, -1, :]  # last time-step (see class warning about right padding)
        out = self.dropout(out)
        return self.fc(out)


def make_loader(
    X: np.ndarray, y, batch_size: int, shuffle: bool = False, y_dtype=torch.long
) -> DataLoader:
    """``DataLoader`` over ``(int sequences, labels)`` as built in the notebook."""
    dataset = TensorDataset(torch.tensor(X), torch.tensor(np.asarray(y), dtype=y_dtype))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_fixed_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: str,
    on_epoch_end: Optional[Callable[[int, float], None]] = None,
) -> Tuple[List[float], float]:
    """Plain training for a fixed number of epochs (English dair-ai, PANDORA, legacy French).

    ``on_epoch_end(epoch, mean_train_loss)`` is called after each epoch
    (the PANDORA notebook evaluates on the validation set there).
    Returns ``(per-epoch mean train loss, training seconds)``.
    """
    history: List[float] = []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item()
        mean_loss = total / len(train_loader)
        history.append(mean_loss)
        if on_epoch_end is not None:
            on_epoch_end(epoch, mean_loss)
    return history, time.time() - t0


def train_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epochs: int,
    patience: int,
    device: str,
    checkpoint_path: Optional[Path] = None,
) -> Dict:
    """French Part A loop: ReduceLROnPlateau on validation loss + early stopping.

    The best-validation-loss weights are restored at the end (the notebook wrote
    them to ``best_<type>.pt`` in the working directory and reloaded them; here
    they are kept in memory and optionally also written to ``checkpoint_path``).
    Validation loss is the mean of per-batch mean losses, as in the notebook.
    """
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    train_losses: List[float] = []
    val_losses: List[float] = []
    stopped_early = False

    t0 = time.time()
    for _epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_path is not None:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                stopped_early = True
                break
    train_time = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)
    return dict(
        train_losses=train_losses,
        val_losses=val_losses,
        best_val_loss=best_val_loss,
        stopped_early=stopped_early,
        train_time=train_time,
    )


def predict_classes(model: nn.Module, loader: DataLoader, device: str) -> Tuple[List[int], float]:
    """Argmax predictions over a loader. Returns ``(predictions, inference seconds)``."""
    model.eval()
    preds: List[int] = []
    t0 = time.time()
    with torch.no_grad():
        for xb, *_ in loader:
            preds.extend(model(xb.to(device)).argmax(dim=1).cpu().tolist())
    return preds, time.time() - t0


def predict_sigmoid(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    """Sigmoid probabilities for multi-output (PANDORA, 5 traits) models."""
    model.eval()
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for xb, *_ in loader:
            outputs.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    return np.vstack(outputs)
