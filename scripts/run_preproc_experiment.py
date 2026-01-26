#!/usr/bin/env python
"""
CBraMod Data Preprocessing ML Engineering Experiment Script.

This script systematically evaluates different preprocessing parameters
on CBraMod performance for single-finger motor decoding tasks.

Experiment Groups:
- A (Filtering): Bandpass and notch filter variations
- C (Normalization): Extra normalization after divide by 100
- D (Window): Sliding window step size variations
- F (Quality): Amplitude rejection threshold variations

Note: A1 = C1 = D1 = F1 is the baseline configuration (only run once).

Usage:
    # Run a specific experiment
    uv run python scripts/run_preproc_experiment.py --exp A2

    # Run all experiments in a group
    uv run python scripts/run_preproc_experiment.py --group A

    # Run all unique experiments
    uv run python scripts/run_preproc_experiment.py --all

    # Run with specific subjects
    uv run python scripts/run_preproc_experiment.py --exp A2 --subjects S01 S02 S03

    # Prototype validation (first 3 subjects)
    uv run python scripts/run_preproc_experiment.py --prototype

    # Run EEGNet baseline for comparison
    uv run python scripts/run_preproc_experiment.py --eegnet-baseline
"""

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.device import set_seed, check_cuda_available, get_device
from src.utils.logging import SectionLogger, setup_logging
from src.preprocessing.experiment_config import (
    ExperimentPreprocessConfig,
    ALL_EXPERIMENTS,
    EXPERIMENT_GROUPS,
    UNIQUE_EXPERIMENT_IDS,
    get_experiment_config,
    get_experiments_in_group,
    get_unique_experiments,
    is_baseline,
    print_experiment_summary,
)
from src.preprocessing.data_loader import PreprocessConfig, discover_available_subjects

# Import training utilities
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

from _training_utils import (
    TrainingResult,
    result_to_dict,
    dict_to_result,
    compute_model_statistics,
    train_and_get_result,
)


setup_logging('preproc_experiment')
logger = logging.getLogger(__name__)
log_main = SectionLogger(logger, 'main')
log_exp = SectionLogger(logger, 'exp')
log_io = SectionLogger(logger, 'io')


# ============================================================================
# Constants
# ============================================================================

EXPERIMENT_CACHE_PREFIX = 'data_preproc_ml_eng'
RESULTS_FILENAME = 'preproc_ml_eng_results.json'


# ============================================================================
# Experiment Result Data Class
# ============================================================================

class ExperimentResult:
    """Result from a single experiment configuration."""

    def __init__(
        self,
        experiment_id: str,
        experiment_group: str,
        task: str,
        results: List[TrainingResult],
        statistics: Dict,
    ):
        self.experiment_id = experiment_id
        self.experiment_group = experiment_group
        self.task = task
        self.results = results
        self.statistics = statistics

    def to_dict(self) -> Dict:
        """Convert to serializable dict."""
        return {
            'experiment_id': self.experiment_id,
            'experiment_group': self.experiment_group,
            'task': self.task,
            'subjects': [result_to_dict(r) for r in self.results],
            'statistics': self.statistics,
        }


# ============================================================================
# Cache Management for Experiments
# ============================================================================

def get_experiment_cache_path(output_dir: str, paradigm: str) -> Path:
    """Get path to experiment results cache."""
    return Path(output_dir) / f'{EXPERIMENT_CACHE_PREFIX}_cache_{paradigm}.json'


def load_experiment_cache(output_dir: str, paradigm: str) -> Dict:
    """Load experiment results cache."""
    cache_path = get_experiment_cache_path(output_dir, paradigm)
    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            log_io.warning(f"加载实验缓存失败: {e}")
    return {'experiments': {}, 'baselines': {}}


def save_experiment_cache(output_dir: str, paradigm: str, cache: Dict):
    """Save experiment results cache."""
    cache_path = get_experiment_cache_path(output_dir, paradigm)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache['last_updated'] = datetime.now().isoformat()
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)


# ============================================================================
# Training Function with Experiment Config
# ============================================================================

def train_experiment(
    experiment_config: ExperimentPreprocessConfig,
    subject_id: str,
    task: str,
    paradigm: str,
    data_root: str,
    save_dir: str,
    no_wandb: bool = False,
    upload_model: bool = False,
    wandb_group: Optional[str] = None,
) -> TrainingResult:
    """
    Train CBraMod with a specific experiment configuration.

    This function creates a custom PreprocessConfig from the experiment
    configuration and delegates to train_and_get_result, which uses the
    standard training pipeline with TableEpochLogger, Timer, etc.
    """
    # Create PreprocessConfig from experiment config
    preprocess_config = PreprocessConfig.from_experiment(experiment_config)

    # Use the shared training function
    return train_and_get_result(
        subject_id=subject_id,
        model_type='cbramod',
        task=task,
        paradigm=paradigm,
        data_root=data_root,
        save_dir=save_dir,
        no_wandb=no_wandb,
        upload_model=upload_model,
        wandb_group=wandb_group,
        preprocess_config=preprocess_config,
    )


def train_eegnet_baseline(
    subject_id: str,
    task: str,
    paradigm: str,
    data_root: str,
    save_dir: str,
    no_wandb: bool = False,
    upload_model: bool = False,
    wandb_group: Optional[str] = None,
) -> TrainingResult:
    """Train EEGNet baseline for comparison."""
    from _training_utils import train_and_get_result

    return train_and_get_result(
        subject_id=subject_id,
        model_type='eegnet',
        task=task,
        paradigm=paradigm,
        data_root=data_root,
        save_dir=save_dir,
        no_wandb=no_wandb,
        upload_model=upload_model,
        wandb_group=wandb_group,
    )


# ============================================================================
# Main Experiment Runner
# ============================================================================

def run_experiment(
    experiment_config: ExperimentPreprocessConfig,
    subjects: List[str],
    task: str,
    paradigm: str,
    data_root: str,
    output_dir: str,
    cache: Dict,
    force_retrain: bool = False,
    no_wandb: bool = False,
    upload_model: bool = False,
) -> ExperimentResult:
    """
    Run a single experiment configuration across all subjects.

    Args:
        experiment_config: Experiment configuration
        subjects: List of subject IDs
        task: Task type ('binary' or 'ternary')
        paradigm: Paradigm ('imagery' or 'movement')
        data_root: Data directory
        output_dir: Output directory
        cache: Experiment cache dict
        force_retrain: Force retraining even if cached
        no_wandb: Disable WandB
        upload_model: Upload model to WandB

    Returns:
        ExperimentResult with all subject results
    """
    exp_id = experiment_config.experiment_id
    cache_key = f"{exp_id}_{task}"

    # Generate WandB group once for all subjects in this experiment
    wandb_group = f"exp_{exp_id}_{task}_{paradigm}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    log_exp.info(f"实验 {exp_id}: {experiment_config.description}")

    # Check cache for this experiment
    exp_cache = cache.get('experiments', {}).get(cache_key, {})

    results: List[TrainingResult] = []

    for idx, subject_id in enumerate(subjects, 1):
        progress = f"[{idx}/{len(subjects)}]"

        # Check cache
        if subject_id in exp_cache and not force_retrain:
            log_exp.info(f"{progress} {subject_id} ({exp_id}): 已缓存")
            cached_result = dict_to_result(exp_cache[subject_id])
            results.append(cached_result)
            continue

        # Train
        log_exp.info(f"{progress} {subject_id} ({exp_id}): 训练中...")

        try:
            set_seed(42)

            result = train_experiment(
                experiment_config=experiment_config,
                subject_id=subject_id,
                task=task,
                paradigm=paradigm,
                data_root=data_root,
                save_dir=output_dir,
                no_wandb=no_wandb,
                upload_model=upload_model,
                wandb_group=wandb_group,
            )

            results.append(result)

            # Update cache
            if 'experiments' not in cache:
                cache['experiments'] = {}
            if cache_key not in cache['experiments']:
                cache['experiments'][cache_key] = {}
            cache['experiments'][cache_key][subject_id] = result_to_dict(result)
            save_experiment_cache(output_dir, paradigm, cache)

            log_exp.info(f"{progress} {subject_id} ({exp_id}): {result.test_acc_majority:.1%}")

        except Exception as e:
            log_exp.error(f"{progress} {subject_id} ({exp_id}): 失败 - {e}")
            traceback.print_exc()
            continue

    # Compute statistics
    stats = compute_model_statistics(results)

    return ExperimentResult(
        experiment_id=exp_id,
        experiment_group=experiment_config.experiment_group,
        task=task,
        results=results,
        statistics=stats,
    )


def run_all_experiments(
    experiment_ids: List[str],
    subjects: List[str],
    tasks: List[str],
    paradigm: str,
    data_root: str,
    output_dir: str,
    force_retrain: bool = False,
    no_wandb: bool = False,
    upload_model: bool = False,
) -> Dict[str, ExperimentResult]:
    """
    Run multiple experiments across all subjects and tasks.

    Returns:
        Dict mapping "{exp_id}_{task}" to ExperimentResult
    """
    # Load cache
    cache = load_experiment_cache(output_dir, paradigm)

    all_results = {}

    # Track baseline to avoid running duplicates
    baseline_run_for_task = {}

    total_configs = len(experiment_ids) * len(tasks)
    current = 0

    for exp_id in experiment_ids:
        for task in tasks:
            current += 1
            config_key = f"{exp_id}_{task}"

            # Check if this is a baseline alias
            if is_baseline(exp_id) and task in baseline_run_for_task:
                # Reuse baseline results
                baseline_key = f"A1_{task}"
                if baseline_key in all_results:
                    log_exp.info(f"[{current}/{total_configs}] {exp_id} ({task}): = A1 基线 (复用)")
                    all_results[config_key] = all_results[baseline_key]
                    continue

            log_exp.info(f"\n[{current}/{total_configs}] 运行 {exp_id} ({task})")

            experiment_config = get_experiment_config(exp_id)

            result = run_experiment(
                experiment_config=experiment_config,
                subjects=subjects,
                task=task,
                paradigm=paradigm,
                data_root=data_root,
                output_dir=output_dir,
                cache=cache,
                force_retrain=force_retrain,
                no_wandb=no_wandb,
                upload_model=upload_model,
            )

            all_results[config_key] = result

            # Mark baseline as run
            if is_baseline(exp_id):
                baseline_run_for_task[task] = exp_id

            # Log summary
            if result.statistics['n_subjects'] > 0:
                log_exp.info(
                    f"{exp_id} ({task}): {result.statistics['mean']:.1%} +/- {result.statistics['std']:.1%} "
                    f"(n={result.statistics['n_subjects']})"
                )

    return all_results


def run_eegnet_baseline(
    subjects: List[str],
    tasks: List[str],
    paradigm: str,
    data_root: str,
    output_dir: str,
    cache: Dict,
    force_retrain: bool = False,
    no_wandb: bool = False,
    upload_model: bool = False,
) -> Dict[str, ExperimentResult]:
    """Run EEGNet baseline for comparison."""
    results = {}
    wandb_group = f"baseline_eegnet_{paradigm}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    for task in tasks:
        cache_key = f"baseline_eegnet_{task}"
        task_cache = cache.get('baselines', {}).get(cache_key, {})

        task_results = []

        for idx, subject_id in enumerate(subjects, 1):
            progress = f"[{idx}/{len(subjects)}]"

            if subject_id in task_cache and not force_retrain:
                log_exp.info(f"{progress} {subject_id} (EEGNet {task}): 已缓存")
                task_results.append(dict_to_result(task_cache[subject_id]))
                continue

            log_exp.info(f"{progress} {subject_id} (EEGNet {task}): 训练中...")

            try:
                set_seed(42)
                result = train_eegnet_baseline(
                    subject_id=subject_id,
                    task=task,
                    paradigm=paradigm,
                    data_root=data_root,
                    save_dir=output_dir,
                    no_wandb=no_wandb,
                    upload_model=upload_model,
                    wandb_group=wandb_group,
                )
                task_results.append(result)

                # Update cache
                if 'baselines' not in cache:
                    cache['baselines'] = {}
                if cache_key not in cache['baselines']:
                    cache['baselines'][cache_key] = {}
                cache['baselines'][cache_key][subject_id] = result_to_dict(result)
                save_experiment_cache(output_dir, paradigm, cache)

            except Exception as e:
                log_exp.error(f"{progress} {subject_id} (EEGNet {task}): 失败 - {e}")
                traceback.print_exc()

        stats = compute_model_statistics(task_results)
        results[cache_key] = ExperimentResult(
            experiment_id='baseline_eegnet',
            experiment_group='baseline',
            task=task,
            results=task_results,
            statistics=stats,
        )

        if stats['n_subjects'] > 0:
            log_exp.info(
                f"EEGNet ({task}): {stats['mean']:.1%} +/- {stats['std']:.1%} "
                f"(n={stats['n_subjects']})"
            )

    return results


# ============================================================================
# Results Saving
# ============================================================================

def save_experiment_results(
    all_results: Dict[str, ExperimentResult],
    baseline_results: Optional[Dict[str, ExperimentResult]],
    paradigm: str,
    output_dir: str,
) -> Path:
    """Save all experiment results to JSON."""
    output = {
        'metadata': {
            'experiment_type': 'data_preproc_ml_eng',
            'paradigm': paradigm,
            'timestamp': datetime.now().isoformat(),
            'n_experiments': len(all_results),
        },
        'experiments': {k: v.to_dict() for k, v in all_results.items()},
    }

    if baseline_results:
        output['baselines'] = {k: v.to_dict() for k, v in baseline_results.items()}

    filename = f"{datetime.now().strftime('%Y%m%d_%H%M')}_{RESULTS_FILENAME}"
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    log_io.info(f"结果已保存: {output_path}")
    return output_path


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CBraMod Data Preprocessing ML Engineering Experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run a specific experiment
  uv run python scripts/run_preproc_experiment.py --exp A2

  # Run all experiments in Group A (filtering)
  uv run python scripts/run_preproc_experiment.py --group A

  # Run all unique experiments
  uv run python scripts/run_preproc_experiment.py --all

  # Prototype validation (first 3 subjects)
  uv run python scripts/run_preproc_experiment.py --prototype

  # Run EEGNet baseline
  uv run python scripts/run_preproc_experiment.py --eegnet-baseline

  # List all available experiments
  uv run python scripts/run_preproc_experiment.py --list
'''
    )

    # Experiment selection
    exp_group = parser.add_mutually_exclusive_group()
    exp_group.add_argument(
        '--exp', type=str, nargs='+',
        help='Specific experiment ID(s) to run (e.g., A2 C2)'
    )
    exp_group.add_argument(
        '--group', type=str, choices=['A', 'C', 'D', 'F'],
        help='Run all experiments in a group'
    )
    exp_group.add_argument(
        '--all', action='store_true',
        help='Run all unique experiments'
    )
    exp_group.add_argument(
        '--prototype', action='store_true',
        help='Prototype validation: run representative configs on S01-S03'
    )
    exp_group.add_argument(
        '--list', action='store_true',
        help='List all available experiments'
    )

    # Baseline
    parser.add_argument(
        '--eegnet-baseline', action='store_true',
        help='Also run EEGNet baseline for comparison'
    )
    parser.add_argument(
        '--baseline-only', action='store_true',
        help='Only run baselines (CBraMod A1 + EEGNet)'
    )

    # Data parameters
    parser.add_argument(
        '--data-root', type=str, default='data',
        help='Path to data directory (default: data)'
    )
    parser.add_argument(
        '--subjects', nargs='+', default=None,
        help='Specific subjects to run (default: S01-S07)'
    )
    parser.add_argument(
        '--paradigm', type=str, default='imagery',
        choices=['imagery', 'movement'],
        help='Experiment paradigm (default: imagery)'
    )
    parser.add_argument(
        '--tasks', nargs='+', default=['binary', 'ternary'],
        choices=['binary', 'ternary'],
        help='Classification tasks (default: binary ternary)'
    )

    # Output
    parser.add_argument(
        '--output-dir', type=str, default='results',
        help='Directory to save results (default: results)'
    )
    parser.add_argument(
        '--force-retrain', action='store_true',
        help='Force retraining, ignore cache'
    )
    parser.add_argument(
        '--no-wandb', action='store_true',
        help='Disable WandB logging'
    )
    parser.add_argument(
        '--upload-model', action='store_true',
        help='Upload model artifacts to WandB'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )

    args = parser.parse_args()

    # Handle --list
    if args.list:
        print_experiment_summary()
        return 0

    # Check GPU
    check_cuda_available(required=True)
    device = get_device()
    log_main.info(f"设备: {device}")

    # Set seed
    set_seed(args.seed)

    # Discover subjects
    if args.subjects:
        subjects = args.subjects
    elif args.prototype:
        subjects = ['S01', 'S02', 'S03']
    else:
        subjects = discover_available_subjects(args.data_root, args.paradigm, 'binary')

    if not subjects:
        log_main.error(f"在 {args.data_root} 中未找到被试")
        return 1

    log_main.info(f"被试: {subjects}")
    log_main.info(f"任务: {args.tasks}")
    log_main.info(f"范式: {args.paradigm}")

    # Determine which experiments to run
    if args.baseline_only:
        experiment_ids = ['A1']  # Just the baseline
    elif args.prototype:
        # Prototype: representative configs from each group
        experiment_ids = ['A1', 'A6', 'C2', 'D2', 'F2']
        log_main.info("原型模式: 运行代表性配置")
    elif args.exp:
        experiment_ids = args.exp
    elif args.group:
        experiment_ids = EXPERIMENT_GROUPS[args.group]
    elif args.all:
        experiment_ids = UNIQUE_EXPERIMENT_IDS
    else:
        # Default: just baseline
        experiment_ids = ['A1']

    # Validate experiment IDs
    for exp_id in experiment_ids:
        if exp_id not in ALL_EXPERIMENTS:
            log_main.error(f"未知实验 ID: {exp_id}")
            return 1

    log_main.info(f"实验: {experiment_ids}")

    # Start timer
    start_time = time.time()

    # Load cache
    cache = load_experiment_cache(args.output_dir, args.paradigm)

    # Run EEGNet baseline if requested
    baseline_results = None
    if args.eegnet_baseline or args.baseline_only:
        log_main.info("\n" + "=" * 60)
        log_main.info("运行 EEGNet 基线")
        log_main.info("=" * 60)
        baseline_results = run_eegnet_baseline(
            subjects=subjects,
            tasks=args.tasks,
            paradigm=args.paradigm,
            data_root=args.data_root,
            output_dir=args.output_dir,
            cache=cache,
            force_retrain=args.force_retrain,
            no_wandb=args.no_wandb,
            upload_model=args.upload_model,
        )

    # Run experiments
    log_main.info("\n" + "=" * 60)
    log_main.info("运行 CBraMod 实验")
    log_main.info("=" * 60)

    all_results = run_all_experiments(
        experiment_ids=experiment_ids,
        subjects=subjects,
        tasks=args.tasks,
        paradigm=args.paradigm,
        data_root=args.data_root,
        output_dir=args.output_dir,
        force_retrain=args.force_retrain,
        no_wandb=args.no_wandb,
        upload_model=args.upload_model,
    )

    # Save results
    output_path = save_experiment_results(
        all_results=all_results,
        baseline_results=baseline_results,
        paradigm=args.paradigm,
        output_dir=args.output_dir,
    )

    # Print summary
    log_main.info("\n" + "=" * 70)
    log_main.info("实验总结")
    log_main.info("=" * 70)

    for config_key, result in sorted(all_results.items()):
        if result.statistics['n_subjects'] > 0:
            print(f"  {config_key}: {result.statistics['mean']:.1%} +/- {result.statistics['std']:.1%} "
                  f"(n={result.statistics['n_subjects']})")

    if baseline_results:
        print("\n基线:")
        for config_key, result in baseline_results.items():
            if result.statistics['n_subjects'] > 0:
                print(f"  {config_key}: {result.statistics['mean']:.1%} +/- {result.statistics['std']:.1%}")

    # Total time
    total_time = time.time() - start_time
    if total_time >= 3600:
        log_main.info(f"\n总耗时: {total_time/3600:.1f}h")
    elif total_time >= 60:
        log_main.info(f"\n总耗时: {total_time/60:.1f}m")
    else:
        log_main.info(f"\n总耗时: {total_time:.1f}s")

    log_main.info(f"结果已保存: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
