# Training module
"""
Training utilities for EEG-BCI project.

Modules:
- train_within_subject: Within-subject training for EEGNet and CBraMod models
  Supports binary/ternary/quaternary classification tasks with paper-aligned preprocessing.
"""

from .train_within_subject import WithinSubjectTrainer, train_subject_simple

__all__ = [
    'WithinSubjectTrainer',
    'train_subject_simple',
]
