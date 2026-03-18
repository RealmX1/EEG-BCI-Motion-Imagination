"""
HPO 搜索空间定义与参数映射。

定义 6 种模型×范式组合的搜索空间，并将 Optuna trial 参数
转换为训练管线接受的 config_overrides 格式。
"""

import optuna


def sample_search_space(
    trial: optuna.Trial,
    model_type: str,
    paradigm: str,
) -> dict:
    """
    根据模型类型和训练范式采样超参数。

    Args:
        trial: Optuna trial 对象
        model_type: 'cbramod' 或 'eegnet'
        paradigm: 'within_subject', 'cross_subject', 或 'transfer'

    Returns:
        采样的超参数字典
    """
    key = (model_type.lower(), paradigm.lower())
    dispatch = {
        ('cbramod', 'within_subject'): _sample_cbramod_within,
        ('cbramod', 'cross_subject'): _sample_cbramod_cross,
        ('cbramod', 'transfer'): _sample_cbramod_transfer,
        ('eegnet', 'within_subject'): _sample_eegnet_within,
        ('eegnet', 'cross_subject'): _sample_eegnet_cross,
        ('eegnet', 'transfer'): _sample_eegnet_transfer,
    }

    sampler = dispatch.get(key)
    if sampler is None:
        raise ValueError(
            f"Unsupported model_type={model_type}, paradigm={paradigm}. "
            f"Valid combinations: {list(dispatch.keys())}"
        )
    return sampler(trial)


def params_to_config_overrides(
    params: dict,
    model_type: str,
    paradigm: str,
) -> dict:
    """
    将 Optuna trial 采样的参数映射为 config_overrides 格式。

    - within/cross: 返回 {'model': {...}, 'training': {...}, 'scheduler_config': {...}}
    - transfer: 返回原始 params dict（直接传给 finetune_subject 的显式参数）

    Args:
        params: sample_search_space() 返回的参数字典
        model_type: 'cbramod' 或 'eegnet'
        paradigm: 'within_subject', 'cross_subject', 或 'transfer'

    Returns:
        config_overrides dict 或 transfer 的原始参数 dict
    """
    if paradigm == 'transfer':
        # transfer 直接返回原始参数，由 objective 传给 finetune_subject
        return params

    if model_type == 'cbramod':
        return _cbramod_to_overrides(params, paradigm)
    else:
        return _eegnet_to_overrides(params)


# ============================================================
# CBraMod search spaces
# ============================================================

def _sample_cbramod_within(trial: optuna.Trial) -> dict:
    """CBraMod Within-Subject: 11 parameters (CAWD scheduler)."""
    return {
        'backbone_lr': trial.suggest_float('backbone_lr', 1e-5, 1e-3, log=True),
        'classifier_lr_ratio': trial.suggest_float('classifier_lr_ratio', 1.0, 5.0),
        'weight_decay': trial.suggest_float('weight_decay', 0.01, 0.3, log=True),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.05, 0.45),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
        'label_smoothing': trial.suggest_float('label_smoothing', 0.0, 0.2),
        'gradient_clip': trial.suggest_float('gradient_clip', 0.3, 2.0),
        # CAWD scheduler params
        'phase_decay': trial.suggest_float('phase_decay', 0.3, 0.9),
        'phase_epochs': trial.suggest_int('phase_epochs', 4, 10),
        'exploration_epochs': trial.suggest_int('exploration_epochs', 3, 9),
        'exploration_batch_size': trial.suggest_categorical(
            'exploration_batch_size', [16, 32, 64]
        ),
    }


def _sample_cbramod_cross(trial: optuna.Trial) -> dict:
    """CBraMod Cross-Subject: 11 parameters (CAWD scheduler, tighter ranges)."""
    return {
        'backbone_lr': trial.suggest_float('backbone_lr', 1e-5, 5e-4, log=True),
        'classifier_lr_ratio': trial.suggest_float('classifier_lr_ratio', 1.0, 3.0),
        'weight_decay': trial.suggest_float('weight_decay', 0.03, 0.5, log=True),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.15, 0.55),
        'batch_size': trial.suggest_categorical('batch_size', [128, 256]),
        'label_smoothing': trial.suggest_float('label_smoothing', 0.05, 0.3),
        'gradient_clip': trial.suggest_float('gradient_clip', 0.2, 1.5),
        # CAWD scheduler params
        'phase_decay': trial.suggest_float('phase_decay', 0.2, 0.7),
        'phase_epochs': trial.suggest_int('phase_epochs', 4, 10),
        'exploration_epochs': trial.suggest_int('exploration_epochs', 3, 9),
        'exploration_batch_size': trial.suggest_categorical(
            'exploration_batch_size', [32, 64, 128]
        ),
    }


def _sample_cbramod_transfer(trial: optuna.Trial) -> dict:
    """
    CBraMod Transfer: 3 parameters only.

    finetune_subject() 只接受 learning_rate, batch_size, epochs 作为可调参数。
    freeze_strategy 固定为 'none'，不搜索。
    """
    return {
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 5e-4, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'finetune_epochs': trial.suggest_int('finetune_epochs', 10, 40),
    }


# ============================================================
# EEGNet search spaces
# ============================================================

def _sample_eegnet_within(trial: optuna.Trial) -> dict:
    """EEGNet Within-Subject: 7 parameters."""
    F1 = trial.suggest_categorical('F1', [4, 8, 16])
    D = trial.suggest_categorical('D', [1, 2, 4])
    return {
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 0.1, log=True),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.7),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'F1': F1,
        'D': D,
        'F2': F1 * D,  # derived
        'kernel_length': trial.suggest_categorical('kernel_length', [32, 64, 128]),
    }


def _sample_eegnet_cross(trial: optuna.Trial) -> dict:
    """EEGNet Cross-Subject: 7 parameters (wider batch, lower lr)."""
    F1 = trial.suggest_categorical('F1', [4, 8, 16])
    D = trial.suggest_categorical('D', [1, 2, 4])
    return {
        'learning_rate': trial.suggest_float('learning_rate', 5e-5, 5e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 0.2, log=True),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.3, 0.7),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
        'F1': F1,
        'D': D,
        'F2': F1 * D,  # derived
        'kernel_length': trial.suggest_categorical('kernel_length', [32, 64, 128]),
    }


def _sample_eegnet_transfer(trial: optuna.Trial) -> dict:
    """EEGNet Transfer: v1 不支持。"""
    raise ValueError("EEGNet transfer HPO is not supported in v1.")


# ============================================================
# Parameter → config_overrides mapping
# ============================================================

def _cbramod_to_overrides(params: dict, paradigm: str) -> dict:
    """CBraMod 参数映射为 config_overrides 格式。"""
    backbone_lr = params['backbone_lr']
    classifier_lr = backbone_lr * params['classifier_lr_ratio']

    overrides = {
        'model': {
            'dropout_rate': params['dropout_rate'],
        },
        'training': {
            'backbone_lr': backbone_lr,
            'classifier_lr': classifier_lr,
            'learning_rate': backbone_lr,
            'weight_decay': params['weight_decay'],
            'label_smoothing': params['label_smoothing'],
            'gradient_clip': params['gradient_clip'],
            'batch_size': params['batch_size'],
        },
        'scheduler_config': {
            'phase_decay': params.get('phase_decay'),
            'phase_epochs': params.get('phase_epochs'),
            'exploration_epochs': params.get('exploration_epochs'),
            'exploration_batch_size': params.get('exploration_batch_size'),
        },
    }

    # 过滤 None 值
    overrides['scheduler_config'] = {
        k: v for k, v in overrides['scheduler_config'].items() if v is not None
    }

    return overrides


def _eegnet_to_overrides(params: dict) -> dict:
    """EEGNet 参数映射为 config_overrides 格式。"""
    return {
        'model': {
            'F1': params.get('F1', 8),
            'D': params.get('D', 2),
            'F2': params['F2'],
            'kernel_length': params.get('kernel_length', 64),
            'dropout_rate': params['dropout_rate'],
        },
        'training': {
            'learning_rate': params['learning_rate'],
            'weight_decay': params['weight_decay'],
            'batch_size': params['batch_size'],
        },
    }
