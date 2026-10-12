"""Compute-device selection (CUDA if available, otherwise CPU)."""

from __future__ import annotations


def get_device() -> str:
    """Return ``"cuda"`` when a CUDA device is visible to PyTorch, else ``"cpu"``."""
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
