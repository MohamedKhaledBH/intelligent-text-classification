"""Word-level vocabulary and padded integer sequences for the recurrent models.

Reproduces the notebook's hand-written tokenizer: whitespace split, vocabulary
= the ``vocab_size`` most frequent training words (indices start at 2),
``<PAD>`` = 0, ``<UNK>`` = 1, **right** padding, truncation to ``max_len``
words.

Note (kept as-is, documented in ``docs/methodology.md``): because sequences are
right-padded and the models read ``rnn_out[:, -1, :]``, the representation fed
to the classifier is the hidden state *after the padding positions* for any text
shorter than ``max_len``.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, Sequence

import numpy as np

PAD_INDEX = 0
UNK_INDEX = 1


def build_vocab(train_texts: Iterable[str], vocab_size: int) -> Dict[str, int]:
    words = " ".join(train_texts).split()
    counts = Counter(words)
    vocab = {w: i + 2 for i, (w, _c) in enumerate(counts.most_common(vocab_size))}
    vocab["<PAD>"] = PAD_INDEX
    vocab["<UNK>"] = UNK_INDEX
    return vocab


def encode_texts(texts: Sequence[str], vocab: Dict[str, int], max_len: int) -> np.ndarray:
    encoded = np.zeros((len(texts), max_len), dtype=int)
    for i, text in enumerate(texts):
        tokens = [vocab.get(w, UNK_INDEX) for w in str(text).split()[:max_len]]
        encoded[i, : len(tokens)] = tokens
    return encoded


def embedding_vocab_size(vocab: Dict[str, int]) -> int:
    """Embedding-table size used by the notebook: ``len(vocab) + 2``."""
    return len(vocab) + 2
