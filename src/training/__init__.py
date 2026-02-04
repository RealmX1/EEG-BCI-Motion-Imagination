# Training module
"""
Training utilities for EEG-BCI project.

Modules:
- train_within_subject: Within-subject training for EEGNet and CBraMod models
  Supports binary/ternary/quaternary classification tasks with paper-aligned preprocessing.
- train_cross_subject: Cross-subject pretraining for transfer learning
- finetune: Individual subject finetuning with freeze strategies
- schedulers: Custom LR schedulers (WSD, CosineDecayRestarts, CosineAnnealingWarmupDecay)
- evaluation: Evaluation functions (majority_vote_accuracy)
- trainer: WithinSubjectTrainer class

Usage:
    # Recommended: import from specific modules
    from src.training.train_within_subject import train_subject_simple
    from src.training.schedulers import CosineAnnealingWarmupDecay
    from src.training.evaluation import majority_vote_accuracy
    from src.training.trainer import WithinSubjectTrainer

    # Also available via train_within_subject for backward compatibility
    from src.training.train_within_subject import (
        WSDScheduler,
        majority_vote_accuracy,
        SCHEDULER_PRESETS,
    )
"""

from .train_within_subject import WithinSubjectTrainer, train_subject_simple
from .train_cross_subject import train_cross_subject
from .finetune import finetune_subject, finetune_all_subjects

# Re-exports from new modules for convenience
from .schedulers import (
    WSDScheduler,
    CosineDecayRestarts,
    CosineAnnealingWarmupDecay,
    visualize_lr_schedule,
)
from .evaluation import majority_vote_accuracy
from .trainer import WithinSubjectTrainer as _WithinSubjectTrainer  # Alias to avoid conflict

__all__ = [
    # Main training API
    'WithinSubjectTrainer',
    'train_subject_simple',
    'train_cross_subject',
    'finetune_subject',
    'finetune_all_subjects',
    # Schedulers
    'WSDScheduler',
    'CosineDecayRestarts',
    'CosineAnnealingWarmupDecay',
    'visualize_lr_schedule',
    # Evaluation
    'majority_vote_accuracy',
]
