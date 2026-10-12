"""Sentence-embedding extraction (BGE-M3 in the notebook).

The notebook encodes 300,000 PANDORA training texts and the full validation
split with ``BAAI/bge-m3`` (``max_seq_length = 256``, ``batch_size = 128``,
normalised embeddings) and fits a multi-output Ridge on top.  BGE-M3 is large
and requires a GPU; the CPU fallback in the notebook (``all-MiniLM-L6-v2``) was
only used for a 4-sentence cross-lingual similarity demo.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

BGE_M3 = "BAAI/bge-m3"


def load_sentence_encoder(
    model_name: str = BGE_M3,
    device: str = "cuda",
    cache_dir: Optional[os.PathLike] = None,
    max_seq_length: Optional[int] = 256,
):
    """Load a SentenceTransformer exactly as in the notebook (safetensors weights)."""
    from sentence_transformers import SentenceTransformer  # lazy: optional dependency

    encoder = SentenceTransformer(
        model_name,
        device=device,
        cache_folder=str(cache_dir) if cache_dir is not None else None,
        model_kwargs={"use_safetensors": True},
    )
    if max_seq_length is not None:
        encoder.max_seq_length = max_seq_length
    return encoder


def encode_with_sentence_transformer(
    encoder,
    texts: Sequence[str],
    batch_size: int = 128,
    normalize: bool = True,
    show_progress_bar: bool = True,
) -> np.ndarray:
    return encoder.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=show_progress_bar,
    )
