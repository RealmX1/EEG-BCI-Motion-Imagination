"""
Shared utilities for EEG-BCI training scripts.

DEPRECATED: This module is maintained for backward compatibility.
New code should import directly from:
- src.config: MODEL_COLORS, PARADIGM_CONFIG
- src.results: TrainingResult, PlotDataSource, load_cache, save_cache, etc.
- src.visualization: generate_combined_plot

This module re-exports all symbols from the new locations.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import SectionLogger, setup_logging

# ============================================================================
# Re-exports from src.config
# ============================================================================
from src.config.constants import (
    MODEL_COLORS,
    PARADIGM_CONFIG,
)

# ============================================================================
# Re-exports from src.results
# ============================================================================
from src.results.dataclasses import TrainingResult, PlotDataSource
from src.results.serialization import (
    result_to_dict,
    dict_to_result,
    generate_result_filename,
)
from src.results.cache import (
    get_cache_path,
    find_latest_cache,
    find_cache_by_tag,
    load_cache,
    save_cache,
    find_compatible_historical_results,
    find_compatible_cross_subject_results,
    build_data_sources_from_historical,
    prepare_combined_plot_data,
    SelectionStrategy,
)
from src.results.experiment_db import ExperimentDB
from src.results.statistics import (
    compute_model_statistics,
    print_model_summary,
)

# ============================================================================
# Re-exports from src.visualization
# ============================================================================
from src.visualization.comparison import generate_combined_plot

# ============================================================================
# Additional imports needed for local functions
# ============================================================================
from src.utils.device import set_seed
from src.preprocessing.data_loader import discover_available_subjects, PreprocessConfig
from src.training.train_within_subject import train_subject_simple


# Setup logging
setup_logging('training')
logger = logging.getLogger(__name__)
log_cache = SectionLogger(logger, 'cache')
log_train = SectionLogger(logger, 'train')
log_io = SectionLogger(logger, 'io')


# ============================================================================
# Local Functions (thin wrappers, kept for backward compatibility)
# ============================================================================

def discover_subjects(
    data_root: str,
    paradigm: str = 'imagery',
    task: str = 'binary',
    cache_only: bool = False,
) -> List[str]:
    """
    Discover all available subjects.

    Args:
        data_root: Root directory containing subject folders
        paradigm: 'imagery' or 'movement'
        task: 'binary', 'ternary', or 'quaternary'
        cache_only: If True, discover from cache index instead of filesystem

    Returns:
        List of subject IDs (e.g., ['S01', 'S02', ...])
    """
    if cache_only:
        from src.preprocessing.data_loader import discover_subjects_from_cache_index
        return discover_subjects_from_cache_index(paradigm, task)
    else:
        return discover_available_subjects(data_root, paradigm, task)


def print_subject_result(subject_id: str, model_type: str, result: TrainingResult):
    """Print formatted result for a single subject."""
    print("\n" + "=" * 60)
    print(f" {model_type.upper()} - {subject_id} COMPLETE")
    print("=" * 60)
    print(f"  Validation Accuracy:  {result.best_val_acc:.2%}")
    print(f"  Test Accuracy:        {result.test_acc_majority:.2%} (majority voting, Sess2 Finetune)")
    print(f"  Epochs Trained:       {result.epochs_trained}")
    print(f"  Training Time:        {result.training_time:.1f}s")
    print("=" * 60 + "\n")


def train_and_get_result(
    subject_id: str,
    model_type: str,
    task: str,
    paradigm: str,
    data_root: str,
    save_dir: str,
    run_tag: Optional[str] = None,
    no_wandb: bool = False,
    upload_model: bool = False,
    wandb_group: Optional[str] = None,
    wandb_project: str = 'eeg-bci',
    wandb_entity: Optional[str] = None,
    preprocess_config: Optional[PreprocessConfig] = None,
    cache_only: bool = False,
    config_overrides: Optional[Dict] = None,
    verbose: int = 2,
    # Transfer learning (optional)
    pretrained_path: Optional[str] = None,
    freeze_strategy: Optional[str] = None,
    # Session override (for extra sessions experiment)
    session_folders_override: Optional[Dict] = None,
    # Precomputed data (for fixed test set strategies)
    precomputed_data: Optional[Dict] = None,
) -> TrainingResult:
    """
    Train a model for a single subject and return TrainingResult.

    This is a thin wrapper around train_subject_simple from train_within_subject.py.

    Args:
        subject_id: Subject ID (e.g., 'S01')
        model_type: 'eegnet' or 'cbramod'
        task: 'binary', 'ternary', or 'quaternary'
        paradigm: 'imagery' or 'movement'
        data_root: Path to data directory
        save_dir: Path to save checkpoints
        no_wandb: Disable wandb logging
        upload_model: Upload model to WandB
        wandb_group: WandB run group
        wandb_project: WandB project name (default: eeg-bci)
        wandb_entity: WandB entity (team/username)
        preprocess_config: Optional custom PreprocessConfig for ML engineering experiments
        cache_only: If True, load data exclusively from cache index
        config_overrides: Config overrides dict (from YAML + CLI merge). Passed to train_subject_simple.
        verbose: Logging verbosity level (0=silent, 1=minimal, 2=full). Default: 2.
        pretrained_path: Path to a pretrained checkpoint for transfer learning.
        freeze_strategy: Freeze strategy for transfer learning ('none', 'backbone', 'partial').
    """
    result_dict = train_subject_simple(
        subject_id=subject_id,
        model_type=model_type,
        task=task,
        paradigm=paradigm,
        data_root=data_root,
        save_dir=save_dir,
        run_tag=run_tag,
        no_wandb=no_wandb,
        upload_model=upload_model,
        wandb_group=wandb_group,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        preprocess_config=preprocess_config,
        cache_only=cache_only,
        config_overrides=config_overrides,
        verbose=verbose,
        pretrained_path=pretrained_path,
        freeze_strategy=freeze_strategy,
        session_folders_override=session_folders_override,
        precomputed_data=precomputed_data,
    )

    if not result_dict:
        raise ValueError(f"Training failed for {subject_id}")

    # Extract subtask_results for unified mode (contains per-subtask accuracy breakdown)
    subtask_results = result_dict.get('subtask_results')
    # Strip heavy fields (detailed_results with per-trial predictions) for serialization
    if subtask_results is not None:
        subtask_results = {
            k: ({'accuracy': v['accuracy'], 'n_trials': v.get('n_trials', 0)}
                if isinstance(v, dict) else v)
            for k, v in subtask_results.items()
            if k in ('binary', 'ternary', 'quaternary', 'mean_accuracy')
        }

    return TrainingResult(
        subject_id=subject_id,
        task_type=task,
        model_type=model_type,
        best_val_acc=result_dict.get('best_val_acc', result_dict.get('val_accuracy', 0.0)),
        test_acc=result_dict.get('test_accuracy', 0.0),
        test_acc_majority=result_dict.get('test_accuracy_majority', result_dict.get('test_accuracy', 0.0)),
        epochs_trained=result_dict.get('epochs_trained', result_dict.get('best_epoch', 0)),
        training_time=result_dict.get('training_time', 0.0),
        subtask_results=subtask_results,
    )


def add_wandb_args(parser) -> None:
    """添加标准化 WandB CLI 参数到 argparse parser。

    所有实验脚本共用，确保一致的 WandB 参数接口。
    """
    group = parser.add_argument_group('WandB')
    group.add_argument(
        '--no-wandb', action='store_true',
        help='Disable WandB logging'
    )
    group.add_argument(
        '--upload-model', action='store_true',
        help='Upload model artifacts (.pt) to WandB'
    )
    group.add_argument(
        '--wandb-project', type=str, default='eeg-bci',
        help='WandB project name (default: eeg-bci)'
    )
    group.add_argument(
        '--wandb-entity', type=str, default=None,
        help='WandB entity (team/username)'
    )


# ============================================================================
# Shared Argparse Builders
# ============================================================================

def add_common_args(parser):
    """Add shared arguments: --data-root, --paradigm, --task, --seed, --output-dir, --no-plot, --subjects."""
    parser.add_argument('--data-root', type=str, default='data', help='Path to data directory (default: data)')
    parser.add_argument('--subjects', nargs='+', default=None, help='Specific subjects to run (default: all available)')
    parser.add_argument('--paradigm', type=str, default='imagery', choices=['imagery', 'movement'], help='Experiment paradigm (default: imagery)')
    parser.add_argument('--task', type=str, default='binary', choices=['binary', 'ternary', 'quaternary', 'unified'], help='Classification task (default: binary)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory to save results (default: results)')
    parser.add_argument('--no-plot', action='store_true', help='Suppress plot generation')


def add_cache_resume_args(parser):
    """Add shared cache/resume arguments."""
    parser.add_argument('--resume', nargs='?', const='', default=None, metavar='TAG',
                        help='Resume a previous run. Without TAG: most recent. With TAG: matching run')
    parser.add_argument('--force-retrain', action='store_true', help='Force retraining, ignore cache')
    parser.add_argument('--skip-training', action='store_true', help='Skip training, load existing results')
    parser.add_argument('--cache-only', action='store_true', help='Load data from cache index only (no filesystem scan)')


def add_channel_args(parser):
    """Add shared channel selection arguments."""
    from src.config.constants import FULL_N_CHANNELS, SUPPORTED_CHANNEL_COUNTS
    parser.add_argument('--channels', type=int, default=FULL_N_CHANNELS, choices=SUPPORTED_CHANNEL_COUNTS,
                        help=f'Number of EEG channels (default: {FULL_N_CHANNELS})')
    parser.add_argument('--channel-config', type=str, default='motor_cortex', help='Channel configuration name (default: motor_cortex)')


def add_training_config_args(parser):
    """Add shared training config arguments."""
    parser.add_argument('--config', type=str, default=None, metavar='YAML_PATH', help='YAML config file path')
    parser.add_argument('--scheduler', type=str, default=None,
                        choices=['plateau', 'cosine', 'wsd', 'cosine_decay', 'cosine_annealing_warmup_decay'],
                        help='Learning rate scheduler (default: model-specific)')
    parser.add_argument('--classifier-type', type=str, default=None,
                        choices=['two_layer', 'three_layer', 'one_layer', 'attention_pool'],
                        help='Override CBraMod classifier head type')
    parser.add_argument('--no-pretrained', action='store_true', help='Train CBraMod from scratch (no pretrained weights)')


def add_transfer_args(parser):
    """Add transfer learning arguments."""
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained checkpoint for transfer learning')
    parser.add_argument('--freeze-strategy', type=str, default='none', choices=['none', 'backbone', 'partial'],
                        help='Freeze strategy for fine-tuning (default: none)')


# ============================================================================
# Output / Run Tag / DB Lifecycle Helpers
# ============================================================================

def resolve_output_dir(args) -> str:
    """Auto-redirect to results/{n}_channel/{config}/ for reduced channel mode."""
    from src.config.constants import FULL_N_CHANNELS
    output_dir = getattr(args, 'output_dir', None) or getattr(args, 'results_dir', 'results')
    channels = getattr(args, 'channels', FULL_N_CHANNELS)
    channel_config = getattr(args, 'channel_config', 'motor_cortex')
    if channels != FULL_N_CHANNELS and output_dir == 'results':
        return f'results/{channels}_channel/{channel_config}'
    return output_dir


def resolve_run_tag(args, paradigm, task, output_dir, cache_type=None) -> str:
    """Handle --resume logic: find existing tag or generate new one."""
    import sys
    from datetime import datetime

    if getattr(args, 'resume', None) is not None:
        tag_hint = args.resume if args.resume != '' else None
        found = find_cache_by_tag(output_dir, paradigm, task, tag_substring=tag_hint, cache_type=cache_type)
        if found:
            _, run_tag = found
            log_cache.info(f"Resuming run: {run_tag}")
            return run_tag
        else:
            log_cache.error("No previous run found to resume")
            sys.exit(1)
    else:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M")
        log_cache.info(f"Starting new run: {run_tag}")
        return run_tag


def init_db_run(run_tag, experiment_type, paradigm, task, args):
    """Create or resume ExperimentDB run. Returns (db, db_run_id)."""
    import shlex
    import sqlite3
    import sys
    from src.config.constants import FULL_N_CHANNELS

    db = ExperimentDB()
    db_run_id = None
    channels = getattr(args, 'channels', FULL_N_CHANNELS)
    channel_config = getattr(args, 'channel_config', 'motor_cortex')
    is_baseline = getattr(args, 'baseline', False)

    try:
        db_run_id = db.create_run(
            run_tag=run_tag,
            experiment_type=experiment_type,
            paradigm=paradigm,
            task=task,
            n_channels=channels,
            channel_config=channel_config if channels != FULL_N_CHANNELS else None,
            command=" ".join(shlex.quote(a) for a in sys.argv),
            is_baseline=is_baseline,
        )
        log_train.info(f"DB run created: {db_run_id}")
    except sqlite3.IntegrityError:
        existing = db.find_run_by_tag(run_tag, paradigm, task, experiment_type=experiment_type)
        if existing:
            db_run_id = existing['run_id']
            log_train.info(f"DB run resumed: {db_run_id}")
            if is_baseline and not existing.get('is_baseline'):
                try:
                    db.set_baseline(db_run_id)
                except Exception:
                    pass
        else:
            log_train.warning(f"DB run creation failed: duplicate but tag not found")
    except Exception as e:
        log_train.warning(f"DB run creation failed: {e}")

    return db, db_run_id


def finalize_db_run(db, db_run_id, comparison, n_subjects, **extra):
    """Save comparison, mark complete, close DB."""
    if db_run_id:
        try:
            if comparison:
                db.save_comparison(db_run_id, comparison)
            db.update_n_subjects(db_run_id, n_subjects)

            transfer_config = extra.get('transfer_config')
            if transfer_config:
                db.save_transfer_config(db_run_id, **transfer_config)

            db.mark_complete(db_run_id)
        except Exception as e:
            log_train.warning(f"DB finalize failed: {e}")
    db.close()


def build_config_overrides(args) -> Optional[Dict]:
    """Build merged config_overrides dict from YAML + CLI args."""
    from src.config.training import load_yaml_config
    from src.config.constants import FULL_N_CHANNELS

    config_overrides = load_yaml_config(args.config) if getattr(args, 'config', None) else {}

    if getattr(args, 'scheduler', None):
        config_overrides.setdefault('training', {})['scheduler'] = args.scheduler
    if getattr(args, 'no_pretrained', False):
        config_overrides.setdefault('model', {})['no_pretrained'] = True

    channels = getattr(args, 'channels', FULL_N_CHANNELS)
    if channels != FULL_N_CHANNELS:
        config_overrides.setdefault('data', {})['channels'] = channels
        config_overrides.setdefault('data', {})['channel_config'] = getattr(args, 'channel_config', 'motor_cortex')

    if getattr(args, 'classifier_type', None):
        config_overrides.setdefault('model', {})['classifier_type'] = args.classifier_type

    return config_overrides or None


# ============================================================================
# Checkpoint Discovery & Transfer Learning Validation
# ============================================================================

def find_best_checkpoint_path(model_type, paradigm, task, subjects, results_dir='results', n_channels=None):
    """Auto-discover best cross-subject pretrained checkpoint."""
    import json
    import torch

    cross_result = find_compatible_cross_subject_results(
        output_dir=results_dir, paradigm=paradigm, task=task,
        subjects=subjects, model_type=model_type, n_channels=n_channels,
    )
    if not cross_result:
        return None

    source_file = cross_result['source_file']
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        model_path = data.get('training_info', {}).get('model_path', '')
        if model_path and Path(model_path).exists():
            log_io.info(f"Found {model_type} checkpoint: {model_path}")
            return model_path
    except (json.JSONDecodeError, OSError):
        pass

    checkpoint_dir = Path('checkpoints/cross_subject')
    if checkpoint_dir.exists():
        for subdir in sorted(checkpoint_dir.iterdir(), reverse=True):
            if subdir.is_dir() and model_type in subdir.name and paradigm in subdir.name and task in subdir.name:
                best_pt = subdir / 'best.pt'
                if best_pt.exists():
                    if n_channels is not None:
                        try:
                            ckpt = torch.load(best_pt, map_location='cpu', weights_only=False)
                            ckpt_channels = ckpt.get('model_config', {}).get('n_channels')
                            if ckpt_channels is not None and ckpt_channels != n_channels:
                                continue
                        except Exception:
                            continue
                    log_io.info(f"Found {model_type} checkpoint (fallback): {best_pt}")
                    return str(best_pt)
    return None


def validate_checkpoint_compatibility(pretrained_paths, task):
    """Validate n_classes matches and extract classifier_types."""
    import sys
    import torch
    from src.config.constants import TASKS

    classifier_types = {}
    expected_n_classes = TASKS[task]['n_classes']

    for model_type, path in pretrained_paths.items():
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            ct = ckpt.get('model_config', {}).get('classifier_type', 'two_layer')
            classifier_types[model_type] = ct
            ckpt_n_classes = ckpt.get('model_config', {}).get('n_classes')
            ckpt_task = ckpt.get('training_config', {}).get('task', 'unknown')
            if ckpt_n_classes is not None and ckpt_n_classes != expected_n_classes:
                log_train.error(
                    f"Checkpoint/task mismatch for {model_type.upper()}: "
                    f"pretrained n_classes={ckpt_n_classes} (task='{ckpt_task}'), "
                    f"but current task '{task}' expects n_classes={expected_n_classes}. "
                    f"Checkpoint: {path}"
                )
                sys.exit(1)
        except Exception:
            classifier_types[model_type] = 'unknown'
    return classifier_types


# ============================================================================
# Replot Helpers
# ============================================================================

def add_replot_arg(parser):
    """Add --replot argument to comparison scripts."""
    parser.add_argument(
        '--replot', type=str, default=None, metavar='RUN_TAG',
        help='Re-generate plots for a completed run (no training, no DB writes). '
             'Requires a run tag (e.g., 20260322_1116).'
    )


def load_replot_context(
    run_tag: str,
    experiment_type: str,
    results_dir_override: Optional[str] = None,
) -> Dict:
    """
    查找已完成的实验 run 并加载其结果用于 replot.

    从 ExperimentDB 读取 run 元数据和 per-subject 结果，
    不创建任何新的 DB 条目。

    Args:
        run_tag: 实验运行标识 (e.g., '20260322_1116')
        experiment_type: 'within_subject', 'cross_subject', 'transfer'
        results_dir_override: 可选的输出目录覆盖

    Returns:
        dict with keys: run_tag, run_id, paradigm, task, n_channels,
        channel_config, models, subjects, results_by_model, results_dir, db

    Raises:
        SystemExit: 找不到 run 或结果为空时退出
    """
    from src.config.constants import FULL_N_CHANNELS
    from src.results import ExperimentDB

    logger = logging.getLogger(__name__)
    db = ExperimentDB()

    # 查找 run
    run = db.find_run_by_tag(run_tag, experiment_type=experiment_type)
    if run is None:
        logger.error(
            f"Run '{run_tag}' not found in ExperimentDB "
            f"(experiment_type={experiment_type})"
        )
        db.close()
        sys.exit(1)

    run_id = run['run_id']
    if not run['is_complete']:
        logger.warning(f"Run '{run_tag}' is not marked complete — replotting anyway")

    # 加载 per-subject 结果
    results_by_model = db.get_results_by_model(run_id)
    if not results_by_model:
        logger.error(f"No subject results found for run '{run_tag}' (run_id={run_id})")
        db.close()
        sys.exit(1)

    models = sorted(results_by_model.keys())
    subjects = sorted({
        r.subject_id
        for rs in results_by_model.values()
        for r in rs
    })

    # 计算 results_dir
    n_channels = run.get('n_channels', FULL_N_CHANNELS)
    channel_config = run.get('channel_config')

    if results_dir_override:
        results_dir = results_dir_override
    elif n_channels != FULL_N_CHANNELS and channel_config:
        results_dir = f'results/{n_channels}_channel/{channel_config}'
    else:
        results_dir = 'results'

    logger.info(
        f"Replot context: run_tag={run_tag}, paradigm={run['paradigm']}, "
        f"task={run['task']}, models={models}, {len(subjects)} subjects, "
        f"results_dir={results_dir}"
    )

    return {
        'run_tag': run_tag,
        'run_id': run_id,
        'paradigm': run['paradigm'],
        'task': run['task'],
        'n_channels': n_channels,
        'channel_config': channel_config,
        'models': models,
        'subjects': subjects,
        'results_by_model': results_by_model,
        'results_dir': results_dir,
        'db': db,
    }
