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
    CACHE_FILENAME,
    CACHE_FILENAME_WITH_TAG,
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
    build_data_sources_from_historical,
    prepare_combined_plot_data,
    SelectionStrategy,
)
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


# ============================================================================
# Local Functions (thin wrappers, kept for backward compatibility)
# ============================================================================

def discover_subjects(
    data_root: str,
    paradigm: str = 'imagery',
    task: str = 'binary',
    cache_only: bool = False,
    cache_index_path: str = ".cache_index.json"
) -> List[str]:
    """
    Discover all available subjects.

    Args:
        data_root: Root directory containing subject folders
        paradigm: 'imagery' or 'movement'
        task: 'binary', 'ternary', or 'quaternary'
        cache_only: If True, discover from cache index instead of filesystem
        cache_index_path: Path to cache index file (default: .cache_index.json)

    Returns:
        List of subject IDs (e.g., ['S01', 'S02', ...])
    """
    if cache_only:
        from src.preprocessing.data_loader import discover_subjects_from_cache_index
        return discover_subjects_from_cache_index(cache_index_path, paradigm, task)
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
    cache_index_path: str = ".cache_index.json",
    config_overrides: Optional[Dict] = None,
    verbose: int = 2,
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
        cache_index_path: Path to cache index file for cache_only mode
        config_overrides: Config overrides dict (from YAML + CLI merge). Passed to train_subject_simple.
        verbose: Logging verbosity level (0=silent, 1=minimal, 2=full). Default: 2.
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
        cache_index_path=cache_index_path,
        config_overrides=config_overrides,
        verbose=verbose,
    )

    if not result_dict:
        raise ValueError(f"Training failed for {subject_id}")

    return TrainingResult(
        subject_id=subject_id,
        task_type=task,
        model_type=model_type,
        best_val_acc=result_dict.get('best_val_acc', result_dict.get('val_accuracy', 0.0)),
        test_acc=result_dict.get('test_accuracy', 0.0),
        test_acc_majority=result_dict.get('test_accuracy_majority', result_dict.get('test_accuracy', 0.0)),
        epochs_trained=result_dict.get('epochs_trained', result_dict.get('best_epoch', 0)),
        training_time=result_dict.get('training_time', 0.0),
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
