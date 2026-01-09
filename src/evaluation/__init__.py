# Evaluation module
"""
Evaluation utilities for EEG-BCI project.

Modules:
- metrics: Classification metrics (accuracy, kappa, AUROC, etc.)
- visualization: Result plotting and analysis
"""

from .metrics import (
    calculate_metrics,
    balanced_accuracy,
    cohens_kappa,
    confusion_matrix_stats,
)

__all__ = [
    'calculate_metrics',
    'balanced_accuracy',
    'cohens_kappa',
    'confusion_matrix_stats',
]
