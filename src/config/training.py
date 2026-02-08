"""
Training configuration for EEG-BCI project.

This module contains:
- SCHEDULER_PRESETS: Recommended epochs/patience for each scheduler type
- get_default_config(): Default training configurations for each model (within-subject)
- get_cross_subject_config(): Cross-subject pretraining configurations
- CROSS_SUBJECT_SCHEDULER_OVERRIDES: Scheduler parameter overrides for cross-subject

Usage:
    from src.config.training import SCHEDULER_PRESETS, get_default_config

    # Get scheduler preset
    preset = SCHEDULER_PRESETS['cosine_annealing_warmup_decay']
    print(preset['epochs'])  # 100

    # Get default config for a model (within-subject)
    config = get_default_config('eegnet', 'binary')
    print(config['training']['learning_rate'])  # 1e-3

    # Get cross-subject config (stronger regularization)
    from src.config.training import get_cross_subject_config
    config = get_cross_subject_config('cbramod', 'binary')
    print(config['training']['weight_decay'])  # 0.12
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
    - scripts/run_within_subject_comparison.py: Batch training
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


# ============================================================================
# YAML Configuration Loader
# ============================================================================

def load_yaml_config(yaml_path: str) -> dict:
    """加载 YAML 配置文件，返回可作为 config_overrides 的 dict。

    YAML 文件中 'tasks', 'task', 'data' 等代码控制的 section 会被过滤。
    仅保留 'model', 'training', 'scheduler_config' 等可覆盖的配置。

    Args:
        yaml_path: YAML 配置文件路径

    Returns:
        可直接传入 apply_config_overrides() 的 dict

    Example:
        >>> overrides = load_yaml_config('configs/cbramod_cawd_old.yaml')
        >>> overrides['training']['scheduler']
        'cosine_annealing_warmup_decay'
    """
    import yaml
    from pathlib import Path

    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Config file is empty: {yaml_path}")
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must be a YAML mapping: {yaml_path}")

    # 过滤代码控制的 section
    return {k: v for k, v in raw.items() if k not in {'tasks', 'task', 'data'}}


# ============================================================================
# Cross-Subject Scheduler Overrides
# ============================================================================

# Cross-subject training uses different scheduler parameters to address
# the different optimization landscape (more data, higher overfitting risk)
CROSS_SUBJECT_SCHEDULER_OVERRIDES: Dict[str, Dict[str, Any]] = {
    'cosine_annealing_warmup_decay': {
        'phase_epochs': 10,             # 6→10, 更长周期避免频繁重启导致重新过拟合
        'phase_decay': 0.5,             # 0.7→0.5, 更激进的峰值衰减抑制过拟合
        'exploration_epochs': 6,        # 保持不变
        'exploration_batch_size': 64,   # 32→64, 多被试数据更稳定的梯度估计
    },
}


# ============================================================================
# Cross-Subject Model Configurations
# ============================================================================

def get_cross_subject_config(model_type: str, task: str) -> dict:
    """
    Get configuration for cross-subject pretraining.

    Builds on get_default_config() but applies cross-subject-specific
    hyperparameters optimized for multi-subject data pooling.

    Key differences from within-subject:
    - Stronger regularization (higher dropout, weight_decay, label_smoothing)
    - Larger batch sizes (more data available)
    - Lower learning rates (more diverse data distribution)
    - Different scheduler parameters (longer phases, more aggressive decay)

    Args:
        model_type: 'eegnet' or 'cbramod'
        task: 'binary', 'ternary', or 'quaternary'

    Returns:
        Configuration dict compatible with train_cross_subject()

    Example:
        >>> config = get_cross_subject_config('cbramod', 'binary')
        >>> config['training']['weight_decay']
        0.12
    """
    config = get_default_config(model_type, task)

    if model_type == 'cbramod':
        config['model']['dropout_rate'] = 0.35          # 0.15→0.35, classifier head 25600 维
        config['training'].update({
            'epochs': 100,                               # 与 within-subject 相同，靠 early stopping
            'patience': 15,                              # 与 within-subject 相同，靠 early stopping
            'batch_size': 256,                           # 2x within-subject
            'learning_rate': 5e-5,                       # 1e-4→5e-5
            'backbone_lr': 5e-5,                         # 1e-4→5e-5, 预训练权重温和调整
            'classifier_lr': 1.5e-4,                     # 3e-4→1.5e-4, 保持 3x 比例
            'weight_decay': 0.12,                        # 0.06→0.12, 4M 参数需更强正则
            'label_smoothing': 0.15,                     # 0.05→0.15, 跨被试标签噪声更大
            'gradient_clip': 0.5,                        # 1.0→0.5, 跨被试梯度方差更大
        })
    else:  # eegnet
        config['training'].update({
            'epochs': 50,                                # 30→50, 更多数据可以训练更久
            'patience': 10,                              # 5→10, 跨被试数据更多样
            'batch_size': 128,                           # 2x within-subject
            'learning_rate': 5e-4,                       # 1e-3→5e-4
            'weight_decay': 1e-4,                        # 0→1e-4, 轻微正则
        })

    return config
