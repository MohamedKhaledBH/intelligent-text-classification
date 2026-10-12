"""Random seed initialisation.

The original notebook seeds Python's ``random``, NumPy and PyTorch (CPU and all
CUDA devices) with ``SEED = 42``.  It does **not** enable deterministic cuDNN
kernels, so GPU runs are not guaranteed to be bit-for-bit reproducible.  This
function reproduces exactly that behaviour and does not add anything on top.
"""

from __future__ import annotations

import random

import numpy as np

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> int:
    """Seed ``random``, ``numpy`` and (if installed) ``torch``; return the seed."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # torch is optional for the classical-ML parts
        return seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
