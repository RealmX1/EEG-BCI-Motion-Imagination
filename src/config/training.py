"""
Training configuration for EEG-BCI project.

This module contains:
- SCHEDULER_PRESETS: Recommended epochs/patience for each scheduler type
- get_default_config(): Default training configurations for each model

Usage:
    from src.config.training import SCHEDULER_PRESETS, get_default_config

    # Get scheduler preset
    preset = SCHEDULER_PRESETS['cosine_annealing_warmup_decay']
    print(preset['epochs'])  # 100

    # Get default config for a model
    config = get_default_config('eegnet', 'binary')
    print(config['training']['learning_rate'])  # 1e-3
"""

from typing import Any, Dict

from .constants import TASKS


# ============================================================================
# Scheduler Presets
# ============================================================================

# Scheduler presets: recommended epochs/patience for each scheduler type
# These values can be overridden by user-specified config_overrides
SCHEDULER_PRESETS: Dict[str, Dict[str, Any]] = {
    'plateau': {
        # ReduceLROnPlateau - 靠 LR 衰减收敛
        'epochs': 30,
        'patience': 5,
        # Exploration phase (optional for traditional schedulers)
        'exploration_epochs': 5,
        'exploration_batch_size': 32,
    },
    'cosine': {
        # CosineAnnealingLR - 需要较多 epochs 到达 min LR
        'epochs': 30,
        'patience': 7,
        # Exploration phase
        'exploration_epochs': 5,
        'exploration_batch_size': 32,
    },
    'wsd': {
        # Warmup-Stable-Decay - 有明确的阶段
        'epochs': 50,
        'patience': 10,
        # WSD-specific parameters
        'warmup_ratio': 0.1,            # 10% = 5 epochs warmup
        'stable_ratio': 0.0,            # 0% = no stable phase
        'decay_ratio': 0.3,             # 30% = 15 epochs decay
        'eta_min': 1e-6,
        # Exploration phase
        'exploration_epochs': 5,        # 10% of epochs
        'exploration_batch_size': 32,
    },
    'cosine_decay': {
        # CosineDecayRestarts - 周期性重启
        'epochs': 50,
        'patience': 10,
        # CosineDecayRestarts-specific
        'decay_factor': 0.7,            # Peak reduces by 30% each cycle
        'num_cycles': 5,                # Number of restart cycles
        # Exploration phase
        'exploration_epochs': 6,
        'exploration_batch_size': 32,
    },
    'cosine_annealing_warmup_decay': {
        # 多阶段余弦，每阶段带 LR ramp-up + cosine decay
        'epochs': 100,
        'patience': 15,
        # CAWD-specific parameters
        'phase_epochs': 6,              # Epochs per cosine decay phase
        'phase_decay': 0.7,             # Peak LR decay between phases (100% → 70% → 49%...)
        'lr_ramp_ratio': 0.1,           # Fraction of each phase for LR ramp-up (10%)
        'eta_min': 1e-6,                # Minimum learning rate
        # Exploration phase (small batch for loss landscape exploration)
        'exploration_epochs': 6,        # Epochs with small batch size
        'exploration_batch_size': 32,   # Batch size during exploration
    },
}


# ============================================================================
# Default Model Configurations
# ============================================================================

def get_default_config(model_type: str, task: str) -> dict:
    """
    Get default configuration for a model type and task.

    This function provides the canonical training configurations used by all
    training scripts. These are the single source of truth for model hyperparameters.

    Used by:
    - train_subject_simple(): Simplified API for external callers
    - scripts/run_full_comparison.py: Batch training
    - scripts/run_single_model.py: Single model training

    Args:
        model_type: 'eegnet' or 'cbramod'
        task: 'binary', 'ternary', or 'quaternary'

    Returns:
        Configuration dict compatible with train_single_subject()

    Example:
        >>> config = get_default_config('eegnet', 'binary')
        >>> config['training']['epochs']
        30
    """
    tasks = TASKS

    if model_type == 'cbramod':
        default_scheduler = 'cosine_annealing_warmup_decay'
        config = {
            'model': {
                'name': 'CBraMod',
                'classifier_type': 'two_layer',
                'dropout_rate': 0.15,  # Slightly higher than v1 (0.1) for mild regularization
                'freeze_backbone': False,
            },
            'training': {
                'scheduler': default_scheduler,
                'epochs': SCHEDULER_PRESETS[default_scheduler]['epochs'],
                'patience': SCHEDULER_PRESETS[default_scheduler]['patience'],
                'batch_size': 128,
                'learning_rate': 1e-4,  # Restored to v1 value - crucial for hard subjects
                'backbone_lr': 1e-4,
                'classifier_lr': 3e-4,  # 3x backbone
                'weight_decay': 0.06,  # Slightly higher than v1 (0.05)
                'label_smoothing': 0.05,
            },
            'data': {},
            'tasks': tasks,
            'task': task,
        }
    else:  # eegnet
        default_scheduler = 'plateau'
        config = {
            'model': {
                'name': 'EEGNet-8,2',
                'F1': 8,
                'D': 2,
                'F2': 16,
                'kernel_length': 64,
                'dropout_rate': 0.5,
            },
            'training': {
                'scheduler': default_scheduler,
                'epochs': SCHEDULER_PRESETS[default_scheduler]['epochs'],
                'patience': SCHEDULER_PRESETS[default_scheduler]['patience'],
                'batch_size': 64,
                'learning_rate': 1e-3,
                'weight_decay': 0,
            },
            'data': {},
            'tasks': tasks,
            'task': task,
        }

    return config
