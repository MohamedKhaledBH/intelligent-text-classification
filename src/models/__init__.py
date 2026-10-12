"""Models.

``classical`` and ``ensemble`` only need scikit-learn and are imported here.
``rnn``, ``hf_finetuning`` and ``llm`` need PyTorch / Hugging Face and must be
imported explicitly (``from src.models.rnn import RNNClassifier``) so that the
package still imports in a torch-free environment.
"""

from .classical import (
    dair_classical_models,
    emotion_pipeline_models,
    fit_predict_timed,
    pandora_binary_models,
    make_ridge,
)
from .ensemble import fit_stacked_ridge_ensemble

__all__ = [
    "dair_classical_models",
    "emotion_pipeline_models",
    "fit_predict_timed",
    "pandora_binary_models",
    "make_ridge",
    "fit_stacked_ridge_ensemble",
]
