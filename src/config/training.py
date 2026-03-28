"""
Training configuration for EEG-BCI project.

This module contains:
- SCHEDULER_PRESETS: max_epochs ceiling + scheduler-specific LR curve parameters
- get_default_config(): Default training configurations for each model (within-subject)
- get_cross_subject_config(): Cross-subject pretraining configurations
- CROSS_SUBJECT_SCHEDULER_OVERRIDES: Scheduler parameter overrides for cross-subject

Usage:
    from src.config.training import SCHEDULER_PRESETS, get_default_config

    # Get scheduler preset
    preset = SCHEDULER_PRESETS['cosine_annealing_warmup_decay']
    print(preset['max_epochs'])  # 500

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

# Scheduler presets: max_epochs is a safety ceiling; early stopping controls actual duration.
# Scheduler-specific LR curve parameters (cosine_T_max, warmup_epochs, cycle_epochs, etc.)
# are absolute values decoupled from max_epochs, so the LR schedule is unaffected by the ceiling.
# These values can be overridden by user-specified config_overrides.
# Early stopping patience is computed automatically in WithinSubjectTrainer.train():
#   CAWD: 1 * phase_epochs | others: 10
SCHEDULER_PRESETS: Dict[str, Dict[str, Any]] = {
    'plateau': {
        # ReduceLROnPlateau - 靠 LR 衰减收敛 (metric-driven, no epoch dependency)
        'max_epochs': 500,
        # Exploration phase (optional for traditional schedulers)
        'exploration_epochs': 5,
        'exploration_batch_size': 32,
    },
    'cosine': {
        # CosineAnnealingLR - 固定半周期，不随 max_epochs 变化
        'max_epochs': 500,
        'cosine_T_max': 30,             # Fixed cosine half-period (decoupled from max_epochs)
        # Exploration phase
        'exploration_epochs': 5,
        'exploration_batch_size': 32,
    },
    'wsd': {
        # Warmup-Stable-Decay - 使用绝对 epoch 数定义各阶段
        'max_epochs': 500,
        # WSD-specific parameters (absolute epochs, decoupled from max_epochs)
        'warmup_epochs': 5,             # Absolute warmup duration (was warmup_ratio=0.1)
        'stable_epochs': 0,             # Absolute stable duration (was stable_ratio=0.0)
        'decay_epochs': 15,             # Absolute decay duration (was decay_ratio=0.3)
        'eta_min': 1e-6,
        # Exploration phase
        'exploration_epochs': 5,
        'exploration_batch_size': 32,
    },
    'cosine_decay': {
        # CosineDecayRestarts - 固定周期长度
        'max_epochs': 500,
        # CosineDecayRestarts-specific (absolute cycle length, decoupled from max_epochs)
        'decay_factor': 0.7,            # Peak reduces by 30% each cycle
        'cycle_epochs': 10,             # Fixed cycle length in epochs (was total_steps // num_cycles)
        # Exploration phase
        'exploration_epochs': 6,
        'exploration_batch_size': 32,
    },
    'cosine_annealing_warmup_decay': {
        # 多阶段余弦，每阶段带 LR ramp-up + cosine decay (naturally compatible with high max_epochs)
        'max_epochs': 500,
        # CAWD-specific parameters (HPO-optimized for within-subject)
        'phase_epochs': 8,              # 6→8 (HPO within)
        'phase_decay': 0.47,            # 0.7→0.47 (HPO within: 0.468)
        'lr_ramp_ratio': 0.1,           # Fraction of each phase for LR ramp-up (10%)
        'eta_min': 1e-6,                # Minimum learning rate
        # Exploration phase (small batch for loss landscape exploration)
        'exploration_epochs': 4,        # 6→4 (HPO within)
        'exploration_batch_size': 64,   # 32→64 (HPO within)
    },
}


# ============================================================================
# Muon Optimizer Presets
# ============================================================================

MUON_PRESETS: Dict[str, Dict[str, Any]] = {
    'default': {
        'muon_lr': 0.02,
        'muon_momentum': 0.95,
        'muon_ns_steps': 5,
        'adamw_backbone_lr': 1e-4,
        'adamw_classifier_lr': 3e-4,
        'weight_decay': 0.06,
    },
    'cross_subject': {
        'muon_lr': 0.02,
        'muon_momentum': 0.95,
        'muon_ns_steps': 5,
        'adamw_backbone_lr': 5e-5,
        'adamw_classifier_lr': 1.5e-4,
        'weight_decay': 0.12,
    },
    'conservative': {
        # 使用 match_rms_adamw 自动缩放 LR，可复用 AdamW 调参经验
        'muon_lr': 1e-4,
        'muon_momentum': 0.95,
        'muon_ns_steps': 5,
        'adamw_backbone_lr': 1e-4,
        'adamw_classifier_lr': 3e-4,
        'weight_decay': 0.06,
    },
}


def get_muon_config(preset: str = 'default') -> dict:
    """获取 Muon 优化器超参数预设."""
    if preset not in MUON_PRESETS:
        raise ValueError(
            f"Unknown Muon preset: {preset}. "
            f"Available: {list(MUON_PRESETS.keys())}"
        )
    return MUON_PRESETS[preset].copy()


# ============================================================================
# Default Model Configurations
# ============================================================================

def get_default_config(model_type: str, task: str, n_channels: int = None) -> dict:
    """
    Get default configuration for a model type and task.

    This function provides the canonical training configurations used by all
    training scripts. These are the single source of truth for model hyperparameters.

    Used by:
    - train_subject_simple(): Simplified API for external callers
    - scripts/run_within_subject_comparison.py: Batch training
    - scripts/run_within_subject.py: Within-subject single model training

    Args:
        model_type: 'eegnet' or 'cbramod'
        task: 'binary', 'ternary', or 'quaternary'
        n_channels: Number of EEG channels (8, 32, or 128). If 8 or 32, applies
            reduced-channel-specific presets for CBraMod. Default None = no override.

    Returns:
        Configuration dict compatible with train_single_subject()

    Example:
        >>> config = get_default_config('eegnet', 'binary')
        >>> config['training']['epochs']
        500
    """
    tasks = TASKS

    if model_type == 'cbramod':
        default_scheduler = 'cosine_annealing_warmup_decay'
        config = {
            'model': {
                'name': 'CBraMod',
                'classifier_type': 'two_layer',
                'dropout_rate': 0.10,  # HPO within: 0.098, rounded to 0.10
                'freeze_backbone': False,
            },
            'training': {
                'scheduler': default_scheduler,
                'epochs': SCHEDULER_PRESETS[default_scheduler]['max_epochs'],
                'batch_size': 256,       # 128→256 (HPO within)
                'learning_rate': 2.9e-4, # 1e-4→2.9e-4 (HPO within: 2.87e-4)
                'backbone_lr': 2.9e-4,   # = learning_rate (HPO within)
                'classifier_lr': 1.2e-3, # backbone×4.03=1.16e-3, rounded (HPO within)
                'weight_decay': 0.026,   # 0.06→0.026 (HPO within: 0.0264)
                'label_smoothing': 0.05, # Keep original (HPO suggested 0.09)
                'gradient_clip': 0.73,   # New (HPO within: 0.729)
            },
            'data': {},
            'tasks': tasks,
            'task': task,
        }
    else:  # eegnet
        default_scheduler = 'plateau'
        config = {
            'model': {
                'name': 'EEGNet-16,4',  # HPO within: F1=16, D=4
                'F1': 16,               # 8→16 (HPO within)
                'D': 4,                 # 2→4 (HPO within)
                'F2': 64,               # F1×D=64 (HPO within)
                'kernel_length': 64,
                'dropout_rate': 0.27,   # 0.5→0.27 (HPO within: 0.271)
            },
            'training': {
                'scheduler': default_scheduler,
                'epochs': SCHEDULER_PRESETS[default_scheduler]['max_epochs'],
                'batch_size': 64,
                'learning_rate': 4e-3,  # 1e-3→4e-3 (HPO within: 3.98e-3)
                'weight_decay': 1e-5,   # 0→1e-5 (HPO within: 1.09e-5)
            },
            'data': {},
            'tasks': tasks,
            'task': task,
        }

    # Apply reduced-channel presets for CBraMod (before user overrides)
    if model_type == 'cbramod' and n_channels == 8:
        for section, overrides in EIGHT_CHANNEL_WITHIN_SUBJECT_OVERRIDES.items():
            if section in config and isinstance(overrides, dict):
                config[section].update(overrides)
    elif model_type == 'cbramod' and n_channels == 32:
        for section, overrides in THIRTYTWO_CHANNEL_WITHIN_SUBJECT_OVERRIDES.items():
            if section in config and isinstance(overrides, dict):
                config[section].update(overrides)
    elif model_type == 'cbramod' and n_channels == 61:
        for section, overrides in SIXTYONE_CHANNEL_WITHIN_SUBJECT_OVERRIDES.items():
            if section in config and isinstance(overrides, dict):
                config[section].update(overrides)

    return config


# ============================================================================
# 8-Channel Presets (applied only when n_channels=8)
# ============================================================================

CBRAMOD_CLASSIFIER_TYPES = ('two_layer', 'three_layer', 'one_layer', 'attention_pool')

EIGHT_CHANNEL_WITHIN_SUBJECT_OVERRIDES = {
    'model': {
        'dropout_rate': 0.30,
    },
    'training': {
        'batch_size': 64,
        'backbone_lr': 1e-4,
        'classifier_lr': 2.5e-4,
        'weight_decay': 0.10,
        'label_smoothing': 0.10,
        'gradient_clip': 0.5,
    },
}

EIGHT_CHANNEL_CROSS_SUBJECT_OVERRIDES = {
    'model': {
        'dropout_rate': 0.45,
    },
    'training': {
        'batch_size': 128,
        'backbone_lr': 3e-5,
        'classifier_lr': 1.5e-4,
        'weight_decay': 0.18,
        'label_smoothing': 0.20,
        'gradient_clip': 0.3,
    },
}

EIGHT_CHANNEL_FINETUNE_OVERRIDES = {
    'epochs': 500,               # Safety ceiling; early stopping controls actual duration
    'learning_rate': 5e-5,
}


# ============================================================================
# 32-Channel Presets (applied only when n_channels=32)
# Values interpolated between 8ch and 128ch defaults
# ============================================================================

THIRTYTWO_CHANNEL_WITHIN_SUBJECT_OVERRIDES = {
    'model': {
        'dropout_rate': 0.20,
    },
    'training': {
        'weight_decay': 0.08,
    },
}

THIRTYTWO_CHANNEL_CROSS_SUBJECT_OVERRIDES = {
    'model': {
        'dropout_rate': 0.40,
    },
    'training': {
        'weight_decay': 0.15,
    },
}

THIRTYTWO_CHANNEL_FINETUNE_OVERRIDES = {
    'epochs': 500,               # Safety ceiling; early stopping controls actual duration
    'learning_rate': 8e-5,
}


# ============================================================================
# 61-Channel Presets (applied only when n_channels=61)
# Values closer to 128ch defaults since 61ch provides good spatial coverage
# ============================================================================

SIXTYONE_CHANNEL_WITHIN_SUBJECT_OVERRIDES = {
    'model': {
        'dropout_rate': 0.18,    # Between 32ch (0.20) and 128ch (0.15)
    },
    'training': {
        'weight_decay': 0.07,    # Between 32ch (0.08) and 128ch (0.06)
    },
}

SIXTYONE_CHANNEL_CROSS_SUBJECT_OVERRIDES = {
    'model': {
        'dropout_rate': 0.38,    # Between 32ch (0.40) and 128ch (0.35)
    },
    'training': {
        'weight_decay': 0.13,    # Between 32ch (0.15) and 128ch (0.12)
    },
}

SIXTYONE_CHANNEL_FINETUNE_OVERRIDES = {
    'epochs': 500,               # Safety ceiling; early stopping controls actual duration
    'learning_rate': 9e-5,       # Between 32ch (8e-5) and 128ch default (1e-4)
}


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
    # 'data' section 保留以支持 YAML 覆盖 (channels, channel_config 等)
    # 'tasks'/'task' 由 CLI 控制，不允许 YAML 覆盖
    return {k: v for k, v in raw.items() if k not in {'tasks', 'task'}}


# ============================================================================
# Cross-Subject Scheduler Overrides
# ============================================================================

# Cross-subject training uses different scheduler parameters to address
# the different optimization landscape (more data, higher overfitting risk)
CROSS_SUBJECT_SCHEDULER_OVERRIDES: Dict[str, Dict[str, Any]] = {
    'cosine_annealing_warmup_decay': {
        'phase_epochs': 10,             # 6→10 (HPO cross)
        'phase_decay': 0.50,            # 0.50 (HPO cross: 0.499, essentially unchanged)
        'exploration_epochs': 3,        # 6→3 (HPO cross)
        'exploration_batch_size': 128,  # 64→128 (HPO cross)
    },
}


# ============================================================================
# Cross-Subject Model Configurations
# ============================================================================

def get_cross_subject_config(model_type: str, task: str, n_channels: int = None) -> dict:
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
        n_channels: Number of EEG channels (8, 32, or 128). If 8 or 32, applies
            reduced-channel-specific presets for CBraMod. Default None = no override.

    Returns:
        Configuration dict compatible with train_cross_subject()

    Example:
        >>> config = get_cross_subject_config('cbramod', 'binary')
        >>> config['training']['weight_decay']
        0.12
    """
    config = get_default_config(model_type, task)

    if model_type == 'cbramod':
        config['model']['dropout_rate'] = 0.37          # HPO cross: 0.369, rounded
        config['training'].update({
            'epochs': 500,                               # Safety ceiling; early stopping controls actual duration
            'batch_size': 256,                           # (unchanged)
            'learning_rate': 1.3e-4,                     # 5e-5→1.3e-4 (HPO cross: 1.335e-4)
            'backbone_lr': 1.3e-4,                       # = learning_rate (HPO cross)
            'classifier_lr': 2.2e-4,                     # backbone×1.62=2.17e-4 (HPO cross)
            'weight_decay': 0.13,                        # 0.12→0.13 (HPO cross: 0.130)
            'label_smoothing': 0.05,                     # Keep conservative (HPO suggested 0.28)
            'gradient_clip': 1.4,                        # 0.5→1.4 (HPO cross: 1.363)
        })
    else:  # eegnet
        config['model'].update({
            'F1': 16,                                    # HPO within: capacity is biggest lever
            'D': 4,                                      # HPO within
            'F2': 64,                                    # F1×D
            'dropout_rate': 0.35,                        # Lower than old 0.5, higher than within 0.27
        })
        config['model']['name'] = 'EEGNet-16,4'         # Update name to match architecture
        config['training'].update({
            'epochs': 500,                               # Safety ceiling; early stopping controls actual duration
            'batch_size': 128,                           # (unchanged)
            'learning_rate': 1e-3,                       # Between within HPO (4e-3) and old (5e-4)
            'weight_decay': 1e-4,                        # (unchanged)
        })

    # Apply reduced-channel cross-subject presets for CBraMod
    if model_type == 'cbramod' and n_channels == 8:
        for section, overrides in EIGHT_CHANNEL_CROSS_SUBJECT_OVERRIDES.items():
            if section in config and isinstance(overrides, dict):
                config[section].update(overrides)
    elif model_type == 'cbramod' and n_channels == 32:
        for section, overrides in THIRTYTWO_CHANNEL_CROSS_SUBJECT_OVERRIDES.items():
            if section in config and isinstance(overrides, dict):
                config[section].update(overrides)
    elif model_type == 'cbramod' and n_channels == 61:
        for section, overrides in SIXTYONE_CHANNEL_CROSS_SUBJECT_OVERRIDES.items():
            if section in config and isinstance(overrides, dict):
                config[section].update(overrides)

    return config
