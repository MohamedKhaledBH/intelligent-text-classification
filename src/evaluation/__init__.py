"""Metrics, reports, plots and explanation helpers."""

from .metrics import (
    compute_classification_metrics,
    compute_emotion_pipeline_metrics,
    evaluate_binary,
    french_benchmark_row,
    find_best_threshold,
)
from .reporting import (
    DAIR_RESULT_COLUMNS,
    combine_french_benchmark,
    mean_macro_f1_by_model,
    misclassified_examples,
    plot_confusion_matrix,
    plot_trait_comparison,
    text_classification_report,
)

__all__ = [
    "compute_classification_metrics",
    "compute_emotion_pipeline_metrics",
    "french_benchmark_row",
    "evaluate_binary",
    "find_best_threshold",
    "DAIR_RESULT_COLUMNS",
    "combine_french_benchmark",
    "mean_macro_f1_by_model",
    "misclassified_examples",
    "plot_confusion_matrix",
    "plot_trait_comparison",
    "text_classification_report",
]
