"""
HPO Objective 函数：within-subject, cross-subject, transfer。

每个 objective 函数：
1. 从 trial 采样超参数
2. 调用对应的训练函数
3. report 中间结果给 Optuna（用于剪枝）
4. 返回最终指标（用于优化）
"""

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import optuna
import torch

from .search_spaces import params_to_config_overrides, sample_search_space

log = logging.getLogger(__name__)


def within_subject_objective(
    trial: optuna.Trial,
    model_type: str,
    task: str,
    paradigm: str,
    subjects: List[str],
    *,
    eeg_paradigm: str = 'imagery',
    data_root: str = 'data',
    save_dir: str = 'checkpoints/hpo',
    cache_only: bool = False,
    cache_index_path: str = '.cache_index.json',
    n_channels: int = 128,
) -> float:
    """
    Within-subject objective: 逐被试训练独立模型，report 累积均值。

    Args:
        trial: Optuna trial
        model_type: 'eegnet' 或 'cbramod'
        task: 'binary', 'ternary', 或 'quaternary'
        paradigm: 'within_subject'
        subjects: 被试 ID 列表
        eeg_paradigm: 'imagery' 或 'movement'
        data_root: 数据根目录
        save_dir: 检查点保存目录
        cache_only: 是否仅从缓存加载数据
        cache_index_path: 缓存索引路径
        n_channels: 通道数

    Returns:
        所有被试的平均 test_accuracy_majority
    """
    from src.training.train_within_subject import train_subject_simple

    params = sample_search_space(trial, model_type, 'within_subject')
    config_overrides = params_to_config_overrides(params, model_type, 'within_subject')

    # 注入通道数
    if n_channels != 128:
        config_overrides.setdefault('data', {})['channels'] = n_channels

    run_tag = f"hpo_t{trial.number}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    trial_save_dir = str(Path(save_dir) / f'within_trial_{trial.number}')

    accs = []
    for i, subject_id in enumerate(subjects):
        try:
            result = train_subject_simple(
                subject_id=subject_id,
                model_type=model_type,
                task=task,
                paradigm=eeg_paradigm,
                data_root=data_root,
                save_dir=trial_save_dir,
                run_tag=run_tag,
                config_overrides=config_overrides,
                cache_only=cache_only,
                cache_index_path=cache_index_path,
                cbramod_channels=n_channels,
                no_wandb=True,
                verbose=0,
            )
            acc = result['test_accuracy_majority']
        except torch.cuda.OutOfMemoryError:
            log.warning(f"Trial {trial.number}, {subject_id}: CUDA OOM")
            torch.cuda.empty_cache()
            _cleanup_dir(trial_save_dir)
            raise optuna.TrialPruned(f"CUDA OOM at {subject_id}")
        except Exception as e:
            log.warning(f"Trial {trial.number}, {subject_id}: failed ({e})")
            acc = 0.0

        accs.append(acc)
        trial.report(np.mean(accs), step=i)

        if trial.should_prune():
            _cleanup_dir(trial_save_dir)
            raise optuna.TrialPruned()

    _cleanup_dir(trial_save_dir)
    return float(np.mean(accs))


def cross_subject_objective(
    trial: optuna.Trial,
    model_type: str,
    task: str,
    paradigm: str,
    subjects: List[str],
    *,
    eeg_paradigm: str = 'imagery',
    data_root: str = 'data',
    save_dir: str = 'checkpoints/hpo',
    cache_only: bool = False,
    cache_index_path: str = '.cache_index.json',
    n_channels: int = 128,
) -> float:
    """
    Cross-subject objective: 单模型训练，依赖内置 early stopping。

    v1 不做 epoch 级剪枝。

    Returns:
        mean_test_acc (跨被试平均测试准确率)
    """
    from src.training.train_cross_subject import train_cross_subject

    params = sample_search_space(trial, model_type, 'cross_subject')
    config_overrides = params_to_config_overrides(params, model_type, 'cross_subject')

    if n_channels != 128:
        config_overrides.setdefault('data', {})['channels'] = n_channels

    run_tag = f"hpo_t{trial.number}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    trial_save_dir = str(Path(save_dir) / f'cross_trial_{trial.number}')

    try:
        result = train_cross_subject(
            subjects=subjects,
            model_type=model_type,
            task=task,
            paradigm=eeg_paradigm,
            save_dir=trial_save_dir,
            data_root=data_root,
            run_tag=run_tag,
            config_overrides=config_overrides,
            cache_only=cache_only,
            cache_index_path=cache_index_path,
            wandb_enabled=False,
            verbose=0,
        )
        mean_acc = result['mean_test_acc']
    except torch.cuda.OutOfMemoryError:
        log.warning(f"Trial {trial.number}: CUDA OOM")
        torch.cuda.empty_cache()
        _cleanup_dir(trial_save_dir)
        raise optuna.TrialPruned("CUDA OOM")
    except Exception as e:
        log.error(f"Trial {trial.number}: failed ({e})")
        _cleanup_dir(trial_save_dir)
        raise

    _cleanup_dir(trial_save_dir)
    return float(mean_acc)


def transfer_objective(
    trial: optuna.Trial,
    model_type: str,
    task: str,
    paradigm: str,
    subjects: List[str],
    pretrained_path: str,
    *,
    eeg_paradigm: str = 'imagery',
    data_root: str = 'data',
    save_dir: str = 'checkpoints/hpo',
    cache_only: bool = False,
    cache_index_path: str = '.cache_index.json',
    n_channels: int = 128,
) -> float:
    """
    Transfer objective: 逐被试微调预训练模型，report 累积均值。

    仅支持 CBraMod (v1)。搜索空间: learning_rate, batch_size, finetune_epochs。
    freeze_strategy 固定为 'none'。

    Returns:
        所有被试的平均 test_acc
    """
    from src.training.finetune import finetune_subject

    if model_type != 'cbramod':
        raise ValueError(f"Transfer HPO only supports cbramod in v1, got {model_type}")

    params = sample_search_space(trial, model_type, 'transfer')
    # transfer 的 params 直接使用，不走 config_overrides

    run_tag = f"hpo_t{trial.number}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    trial_save_dir = str(Path(save_dir) / f'transfer_trial_{trial.number}')

    accs = []
    for i, subject_id in enumerate(subjects):
        try:
            result = finetune_subject(
                pretrained_path=pretrained_path,
                subject_id=subject_id,
                freeze_strategy='none',
                run_tag=run_tag,
                epochs=params['finetune_epochs'],
                learning_rate=params['learning_rate'],
                batch_size=params['batch_size'],
                save_dir=trial_save_dir,
                data_root=data_root,
                paradigm=eeg_paradigm,
                task=task,
                channels=n_channels if n_channels != 128 else None,
                cache_only=cache_only,
                cache_index_path=cache_index_path,
                no_wandb=True,
                verbose=0,
            )
            acc = result['test_acc']
        except torch.cuda.OutOfMemoryError:
            log.warning(f"Trial {trial.number}, {subject_id}: CUDA OOM")
            torch.cuda.empty_cache()
            _cleanup_dir(trial_save_dir)
            raise optuna.TrialPruned(f"CUDA OOM at {subject_id}")
        except Exception as e:
            log.warning(f"Trial {trial.number}, {subject_id}: failed ({e})")
            acc = 0.0

        accs.append(acc)
        trial.report(np.mean(accs), step=i)

        if trial.should_prune():
            _cleanup_dir(trial_save_dir)
            raise optuna.TrialPruned()

    _cleanup_dir(trial_save_dir)
    return float(np.mean(accs))


def _cleanup_dir(path: str) -> None:
    """清理 trial 检查点目录（静默失败）。"""
    try:
        p = Path(path)
        if p.exists():
            shutil.rmtree(p)
    except Exception:
        pass
