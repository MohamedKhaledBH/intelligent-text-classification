"""Intelligent Text Classification — reusable code extracted from the internship notebook.

Subpackages
-----------
data           dataset loading and validation
preprocessing  text cleaning, label preparation, vocabulary / sequence encoding
features       TF-IDF, truncated SVD, sentence-embedding extraction
models         classical, recurrent, transformer and ensemble models
evaluation     metrics, reports, plots, LIME helper
utils          seeding, device, paths / configuration, result file naming

Heavy optional dependencies (torch, transformers, datasets, sentence-transformers,
lime) are imported lazily inside the functions that need them, so that
``import src.<subpackage>`` works in a minimal environment.
"""

__version__ = "0.1.0"
