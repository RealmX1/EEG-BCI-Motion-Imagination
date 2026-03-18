#!/usr/bin/env python
"""
Optuna HPO 入口脚本。

用法:
  uv run python scripts/run_hpo.py \\
      --paradigm within_subject \\
      --model cbramod \\
      --task binary \\
      --n-trials 50 \\
      --pruner probabilistic \\
      --prune-threshold 0.05

  # Transfer 模式
  uv run python scripts/run_hpo.py \\
      --paradigm transfer \\
      --model cbramod \\
      --pretrained-path checkpoints/cross_subject/model.pt \\
      --n-trials 20

  # 禁用剪枝，指定被试子集
  uv run python scripts/run_hpo.py \\
      --paradigm within_subject \\
      --model eegnet \\
      --n-trials 2 \\
      --subjects S01 S02 \\
      --pruner none
"""

import argparse
import json
import logging
import sys
from functools import partial
from pathlib import Path

# 添加项目根目录到 sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import optuna
import torch

from src.hpo.objectives import (
    cross_subject_objective,
    transfer_objective,
    within_subject_objective,
)
from src.hpo.pruner import ProbabilisticSubjectPruner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
log = logging.getLogger(__name__)

# 降低 optuna 的冗余输出
optuna.logging.set_verbosity(optuna.logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Optuna HPO for EEG-BCI models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 必需参数
    parser.add_argument(
        '--paradigm', required=True,
        choices=['within_subject', 'cross_subject', 'transfer'],
        help='训练范式',
    )
    parser.add_argument(
        '--model', required=True,
        choices=['cbramod', 'eegnet'],
        help='模型类型',
    )

    # 任务配置
    parser.add_argument('--task', default='binary',
                        choices=['binary', 'ternary', 'quaternary'])
    parser.add_argument('--eeg-paradigm', default='imagery',
                        choices=['imagery', 'movement'],
                        help='EEG 范式 (default: imagery)')
    parser.add_argument('--n-channels', type=int, default=128,
                        help='通道数 (default: 128)')

    # HPO 配置
    parser.add_argument('--n-trials', type=int, default=50,
                        help='搜索次数 (default: 50)')
    parser.add_argument('--study-name', type=str, default=None,
                        help='Study 名称 (default: auto-generate)')
    parser.add_argument('--storage', type=str, default=None,
                        help='Optuna storage URL (default: sqlite:///results/hpo/hpo.db)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (default: 42)')

    # 剪枝配置
    parser.add_argument('--pruner', default='probabilistic',
                        choices=['probabilistic', 'median', 'none'],
                        help='剪枝策略 (default: probabilistic)')
    parser.add_argument('--prune-threshold', type=float, default=0.05,
                        help='概率剪枝阈值 (default: 0.05)')

    # Transfer 专用
    parser.add_argument('--pretrained-path', type=str, default=None,
                        help='预训练模型路径 (transfer 模式必须)')

    # 被试选择
    parser.add_argument('--subjects', nargs='+', default=None,
                        help='被试子集 (default: 自动发现所有被试)')

    # 数据路径
    parser.add_argument('--data-root', default='data', help='数据根目录')
    parser.add_argument('--save-dir', default='checkpoints/hpo',
                        help='检查点保存目录')
    parser.add_argument('--cache-only', action='store_true',
                        help='仅从缓存加载数据')
    parser.add_argument('--cache-index-path', default='.cache_index.json')

    return parser.parse_args()


def create_pruner(args, n_subjects: int) -> optuna.pruners.BasePruner:
    """根据参数创建剪枝器。"""
    if args.pruner == 'probabilistic':
        return ProbabilisticSubjectPruner(
            n_total_steps=n_subjects,
            threshold=args.prune_threshold,
        )
    elif args.pruner == 'median':
        return optuna.pruners.MedianPruner(n_startup_trials=5, n_min_trials=3)
    else:
        return optuna.pruners.NopPruner()


def _reenqueue_interrupted_trials(study: optuna.Study) -> None:
    """
    检测上次中断的 RUNNING trials，重新排队其参数。

    当进程被 kill 时，正在运行的 trial 会留在 SQLite 中处于 RUNNING 状态。
    Optuna 不会自动重试这些 trial。此函数使用 enqueue_trial() 将它们的
    参数重新排入队列，确保这些超参数组合不被浪费。
    """
    running_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.RUNNING
    ]
    if not running_trials:
        return

    # Current valid categorical values (used to filter stale params)
    _VALID_CATEGORICALS = {
        'batch_size': {32, 64, 128, 256, 512},
        'exploration_batch_size': {16, 32, 64, 128},
    }

    for t in running_trials:
        if not t.params:
            log.warning(f"Interrupted trial #{t.number} has no params, skipping")
            continue

        # Skip if any categorical param is no longer in the search space
        skip = False
        for param, valid_values in _VALID_CATEGORICALS.items():
            if param in t.params and t.params[param] not in valid_values:
                log.info(
                    f"Skipping interrupted trial #{t.number}: "
                    f"{param}={t.params[param]} no longer in search space"
                )
                skip = True
                break
        if skip:
            continue

        study.enqueue_trial(t.params, skip_if_exists=True)
        log.info(
            f"Re-enqueued interrupted trial #{t.number} params: "
            f"{{{', '.join(f'{k}={v}' for k, v in t.params.items())}}}"
        )


def main():
    args = parse_args()

    # 验证 transfer 参数
    if args.paradigm == 'transfer':
        if args.pretrained_path is None:
            print("Error: --pretrained-path is required for transfer paradigm")
            sys.exit(1)
        if args.model != 'cbramod':
            print("Error: transfer HPO only supports cbramod in v1")
            sys.exit(1)
        if not Path(args.pretrained_path).exists():
            print(f"Error: pretrained model not found: {args.pretrained_path}")
            sys.exit(1)

    # 发现被试
    if args.subjects:
        subjects = args.subjects
    else:
        from scripts._training_utils import discover_subjects
        subjects = discover_subjects(
            data_root=args.data_root,
            paradigm=args.eeg_paradigm,
            task=args.task,
            cache_only=args.cache_only,
            cache_index_path=args.cache_index_path,
        )
    log.info(f"Subjects ({len(subjects)}): {subjects}")

    # Study 名称
    study_name = args.study_name or f"{args.model}_{args.paradigm}_{args.task}"

    # Storage
    storage_url = args.storage
    if storage_url is None:
        hpo_dir = Path('results/hpo')
        hpo_dir.mkdir(parents=True, exist_ok=True)
        storage_url = f"sqlite:///{hpo_dir / 'hpo.db'}"

    # Pruner (cross_subject 不用被试级剪枝)
    if args.paradigm == 'cross_subject':
        pruner = optuna.pruners.NopPruner()
        log.info("Cross-subject: pruning disabled (relies on early stopping)")
    else:
        pruner = create_pruner(args, n_subjects=len(subjects))

    # 创建/恢复 Study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction='maximize',
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        load_if_exists=True,
    )

    existing_trials = len(study.trials)
    if existing_trials > 0:
        log.info(f"Resuming study '{study_name}' with {existing_trials} existing trials")
        _reenqueue_interrupted_trials(study)

    # 构建 objective
    common_kwargs = dict(
        model_type=args.model,
        task=args.task,
        paradigm=args.paradigm,
        subjects=subjects,
        eeg_paradigm=args.eeg_paradigm,
        data_root=args.data_root,
        save_dir=args.save_dir,
        cache_only=args.cache_only,
        cache_index_path=args.cache_index_path,
        n_channels=args.n_channels,
    )

    if args.paradigm == 'within_subject':
        objective = partial(within_subject_objective, **common_kwargs)
    elif args.paradigm == 'cross_subject':
        objective = partial(cross_subject_objective, **common_kwargs)
    else:  # transfer
        objective = partial(
            transfer_objective,
            pretrained_path=args.pretrained_path,
            **common_kwargs,
        )

    # 运行优化
    log.info(f"Starting HPO: {study_name}, {args.n_trials} trials, pruner={args.pruner}")
    study.optimize(
        objective, n_trials=args.n_trials,
        catch=(RuntimeError, torch.cuda.OutOfMemoryError, ValueError, OSError),
    )

    # 输出结果
    print_results(study, args)


def print_results(study: optuna.Study, args):
    """输出 HPO 结果摘要。"""
    print("\n" + "=" * 70)
    print(f" HPO Results: {study.study_name}")
    print("=" * 70)

    # 统计
    n_complete = len([t for t in study.trials
                      if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned = len([t for t in study.trials
                    if t.state == optuna.trial.TrialState.PRUNED])
    n_failed = len([t for t in study.trials
                    if t.state == optuna.trial.TrialState.FAIL])

    print(f"  Total trials: {len(study.trials)}")
    print(f"  Completed: {n_complete}, Pruned: {n_pruned}, Failed: {n_failed}")

    if n_complete == 0:
        print("\n  No completed trials. Cannot determine best parameters.")
        return

    # 最优结果
    best = study.best_trial
    print(f"\n  Best trial: #{best.number}")
    print(f"  Best value: {best.value:.4f}")
    print(f"\n  Best parameters:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # 导出最优参数 JSON
    hpo_dir = Path('results/hpo')
    hpo_dir.mkdir(parents=True, exist_ok=True)
    output_path = hpo_dir / f"{study.study_name}_best_params.json"
    with open(output_path, 'w') as f:
        json.dump({
            'study_name': study.study_name,
            'best_trial_number': best.number,
            'best_value': best.value,
            'best_params': best.params,
            'n_trials': len(study.trials),
            'n_complete': n_complete,
            'n_pruned': n_pruned,
        }, f, indent=2)
    print(f"\n  Best params saved to: {output_path}")

    # 参数重要性
    try:
        importances = optuna.importance.get_param_importances(study)
        if importances:
            print("\n  Parameter importance:")
            for param, imp in importances.items():
                bar = '█' * int(imp * 30)
                print(f"    {param:30s} {imp:.4f} {bar}")
    except Exception as e:
        log.debug(f"Could not compute param importances: {e}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
