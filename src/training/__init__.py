# Training module
"""
Training utilities for EEG-BCI project.

Modules:
- train_within_subject: Within-subject training for EEGNet and CBraMod models
  Supports binary/ternary/quaternary classification tasks with paper-aligned preprocessing.
- train_cross_subject: Cross-subject pretraining for transfer learning
- finetune: Individual subject finetuning with freeze strategies
"""

from .train_within_subject import WithinSubjectTrainer, train_subject_simple
from .train_cross_subject import train_cross_subject
from .finetune import finetune_subject, finetune_all_subjects

__all__ = [
    'WithinSubjectTrainer',
    'train_subject_simple',
    'train_cross_subject',
    'finetune_subject',
    'finetune_all_subjects',
]
