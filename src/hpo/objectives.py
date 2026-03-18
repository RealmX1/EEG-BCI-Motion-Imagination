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
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import optuna
import torch

from .search_spaces import params_to_config_overrides, sample_search_space

log = logging.getLogger(__name__)

# Lazy import to avoid circular; cached after first call
_colors_loaded = False
_colored = None
_Colors = None


def _ensure_colors():
    global _colors_loaded, _colored, _Colors
    if not _colors_loaded:
        from src.utils.timing import Colors, colored
        _Colors = Colors
        _colored = colored
        _colors_loaded = True


def _hpo_subject_log(trial_number: int, subject_id: str, idx: int,
                     total: int, acc: float, cum_mean: float):
    """Print colored per-subject HPO progress line."""
    _ensure_colors()
    tag = _colored("[HPO]", _Colors.BRIGHT_YELLOW, bold=True)

    # acc 着色：>0.7 绿, >0.5 黄, 否则红
    if acc > 0.7:
        acc_str = _colored(f"{acc:.4f}", _Colors.BRIGHT_GREEN)
    elif acc > 0.5:
        acc_str = _colored(f"{acc:.4f}", _Colors.YELLOW)
    else:
        acc_str = _colored(f"{acc:.4f}", _Colors.RED)

    mean_str = _colored(f"{cum_mean:.4f}", _Colors.BRIGHT_CYAN)
    subj_str = _colored(f"{subject_id}", _Colors.WHITE, bold=True)

    print(f"  {tag} T{trial_number} | {subj_str} ({idx+1}/{total}) | "
          f"acc={acc_str} | mean={mean_str}")


def _trial_start_banner(trial_number: int, params: dict):
    """Print trial start banner with sampled hyperparameters."""
    _ensure_colors()
    sep = _colored("=" * 70, _Colors.BRIGHT_BLUE)
    tag = _colored(f" [HPO] Trial {trial_number} START", _Colors.BRIGHT_BLUE, bold=True)

    # Format params compactly
    parts = []
    for k, v in params.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4g}")
        else:
            parts.append(f"{k}={v}")
    params_line = _colored(" | ".join(parts), _Colors.WHITE)

    print(f"\n{sep}")
    print(f"{tag}")
    print(f"  {params_line}")
    print(sep)


def _trial_end_banner(trial_number: int, final_value: Optional[float],
                      elapsed_secs: float, status: str):
    """Print trial end banner with result and elapsed time."""
    _ensure_colors()

    mins, secs = divmod(int(elapsed_secs), 60)
    time_str = f"{mins}m{secs:02d}s" if mins > 0 else f"{secs}s"

    if status == 'COMPLETE':
        status_str = _colored(status, _Colors.BRIGHT_GREEN, bold=True)
    elif status == 'PRUNED':
        status_str = _colored(status, _Colors.BRIGHT_RED, bold=True)
    else:
        status_str = _colored(status, _Colors.RED, bold=True)

    if final_value is not None:
        val_str = _colored(f"{final_value:.4f}", _Colors.BRIGHT_CYAN)
    else:
        val_str = _colored("N/A", _Colors.DIM)

    sep = _colored("-" * 70, _Colors.BRIGHT_BLUE)
    tag = _colored(f" [HPO] Trial {trial_number}", _Colors.BRIGHT_BLUE, bold=True)

    print(f"{sep}")
    print(f"{tag} {status_str} | mean_acc={val_str} | "
          f"time={_colored(time_str, _Colors.WHITE)}")
    print(f"{sep}\n")


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

    _trial_start_banner(trial.number, params)
    t0 = time.perf_counter()
    _status = 'COMPLETE'
    _final_value = None

    try:
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
                acc = float('nan')

            accs.append(acc)
            valid_accs = [a for a in accs if not np.isnan(a)]
            if not valid_accs:
                # 所有被试都失败了，无有效结果
                if i == len(subjects) - 1:
                    raise optuna.TrialPruned("all subjects failed")
                continue  # 跳过 report，等待后续有效结果
            cum_mean = float(np.mean(valid_accs))
            _hpo_subject_log(trial.number, subject_id, i, len(subjects), acc, cum_mean)
            trial.report(cum_mean, step=len(valid_accs) - 1)

            if trial.should_prune():
                _cleanup_dir(trial_save_dir)
                raise optuna.TrialPruned()

        valid_accs = [a for a in accs if not np.isnan(a)]
        if not valid_accs:
            raise optuna.TrialPruned("all subjects failed")
        _cleanup_dir(trial_save_dir)
        _final_value = float(np.mean(valid_accs))
        return _final_value

    except optuna.TrialPruned:
        _status = 'PRUNED'
        valid_accs = [a for a in accs if not np.isnan(a)]
        _final_value = float(np.mean(valid_accs)) if valid_accs else None
        raise
    except Exception:
        _status = 'FAILED'
        raise
    finally:
        _trial_end_banner(trial.number, _final_value, time.perf_counter() - t0, _status)


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

    # Reject batch_size=512 (causes VRAM overflow on 12GB GPUs)
    if params.get('batch_size', 0) > 256:
        raise optuna.TrialPruned(f"batch_size={params['batch_size']} excluded (VRAM limit)")

    config_overrides = params_to_config_overrides(params, model_type, 'cross_subject')

    if n_channels != 128:
        config_overrides.setdefault('data', {})['channels'] = n_channels

    run_tag = f"hpo_t{trial.number}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    trial_save_dir = str(Path(save_dir) / f'cross_trial_{trial.number}')

    _trial_start_banner(trial.number, params)
    t0 = time.perf_counter()
    _status = 'COMPLETE'
    _final_value = None

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
        _final_value = float(result['mean_test_acc'])
    except torch.cuda.OutOfMemoryError:
        log.warning(f"Trial {trial.number}: CUDA OOM")
        torch.cuda.empty_cache()
        _cleanup_dir(trial_save_dir)
        _status = 'PRUNED'
        raise optuna.TrialPruned("CUDA OOM")
    except Exception as e:
        log.error(f"Trial {trial.number}: failed ({e})")
        _cleanup_dir(trial_save_dir)
        _status = 'FAILED'
        raise
    else:
        _cleanup_dir(trial_save_dir)
        return _final_value
    finally:
        _trial_end_banner(trial.number, _final_value, time.perf_counter() - t0, _status)


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

    _trial_start_banner(trial.number, params)
    t0 = time.perf_counter()
    _status = 'COMPLETE'
    _final_value = None

    try:
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
                acc = float('nan')

            accs.append(acc)
            valid_accs = [a for a in accs if not np.isnan(a)]
            if not valid_accs:
                # 所有被试都失败了，无有效结果
                if i == len(subjects) - 1:
                    raise optuna.TrialPruned("all subjects failed")
                continue  # 跳过 report，等待后续有效结果
            cum_mean = float(np.mean(valid_accs))
            _hpo_subject_log(trial.number, subject_id, i, len(subjects), acc, cum_mean)
            trial.report(cum_mean, step=len(valid_accs) - 1)

            if trial.should_prune():
                _cleanup_dir(trial_save_dir)
                raise optuna.TrialPruned()

        valid_accs = [a for a in accs if not np.isnan(a)]
        if not valid_accs:
            raise optuna.TrialPruned("all subjects failed")
        _cleanup_dir(trial_save_dir)
        _final_value = float(np.mean(valid_accs))
        return _final_value

    except optuna.TrialPruned:
        _status = 'PRUNED'
        valid_accs = [a for a in accs if not np.isnan(a)]
        _final_value = float(np.mean(valid_accs)) if valid_accs else None
        raise
    except Exception:
        _status = 'FAILED'
        raise
    finally:
        _trial_end_banner(trial.number, _final_value, time.perf_counter() - t0, _status)


def _cleanup_dir(path: str) -> None:
    """清理 trial 检查点目录（静默失败）。"""
    try:
        p = Path(path)
        if p.exists():
            shutil.rmtree(p)
    except Exception:
        pass
