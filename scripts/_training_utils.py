"""
Shared utilities for EEG-BCI training scripts.

This module contains common functions used by both run_single_model.py
and run_full_comparison.py, including:
- Cache management (load/save)
- Subject discovery
- Training result dataclass
- Training wrapper functions
"""

import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.device import set_seed
from src.utils.logging import SectionLogger, setup_logging
from src.preprocessing.data_loader import discover_available_subjects, PreprocessConfig
from src.training.train_within_subject import train_subject_simple


# Setup logging
setup_logging('training')
logger = logging.getLogger(__name__)
log_cache = SectionLogger(logger, 'cache')
log_train = SectionLogger(logger, 'train')


# ============================================================================
# Constants
# ============================================================================

CACHE_FILENAME = 'comparison_cache_{paradigm}_{task}.json'
CACHE_FILENAME_WITH_TAG = '{tag}_comparison_cache_{paradigm}_{task}.json'


def generate_result_filename(
    prefix: str,
    paradigm: str,
    task: str,
    ext: str = 'json',
    run_tag: Optional[str] = None
) -> str:
    """
    生成统一格式的结果文件名: {timestamp}_{prefix}_{paradigm}_{task}.{ext}

    Args:
        prefix: 文件名前缀 (如 'comparison', 'eegnet', 'finetune_backbone')
        paradigm: 范式 ('imagery' 或 'movement')
        task: 任务类型 ('binary', 'ternary', 'quaternary')
        ext: 文件扩展名 (默认 'json')
        run_tag: 可选的运行标签，如果提供则替代自动生成的时间戳

    Returns:
        格式化的文件名，如 '20260203_151711_comparison_imagery_binary.json'
    """
    if run_tag:
        timestamp = run_tag
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parts = [timestamp, prefix, paradigm, task]
    return "_".join(parts) + f".{ext}"

PARADIGM_CONFIG = {
    'imagery': {
        'description': 'Motor Imagery (MI)',
    },
    'movement': {
        'description': 'Motor Execution (ME)',
    },
}

MODEL_COLORS = {
    'eegnet': '#2E86AB',   # Blue
    'cbramod': '#E94F37',  # Red/Coral
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TrainingResult:
    """Result from a single training run."""
    subject_id: str
    task_type: str
    model_type: str
    best_val_acc: float
    test_acc: float
    test_acc_majority: float
    epochs_trained: int
    training_time: float


# ============================================================================
# Subject Discovery
# ============================================================================

def discover_subjects(
    data_root: str,
    paradigm: str = 'imagery',
    task: str = 'binary',
    use_cache_index: bool = False,
    cache_index_path: str = ".cache_index.json"
) -> List[str]:
    """
    Discover all available subjects.

    Args:
        data_root: Root directory containing subject folders
        paradigm: 'imagery' or 'movement'
        task: 'binary', 'ternary', or 'quaternary'
        use_cache_index: If True, discover from cache index instead of filesystem
        cache_index_path: Path to cache index file (default: .cache_index.json)

    Returns:
        List of subject IDs (e.g., ['S01', 'S02', ...])
    """
    if use_cache_index:
        from src.preprocessing.data_loader import discover_subjects_from_cache_index
        return discover_subjects_from_cache_index(cache_index_path, paradigm, task)
    else:
        return discover_available_subjects(data_root, paradigm, task)


# ============================================================================
# Cache Management
# ============================================================================

def get_cache_path(output_dir: str, paradigm: str, task: str, run_tag: Optional[str] = None) -> Path:
    """Get path to cache file."""
    if run_tag:
        filename = CACHE_FILENAME_WITH_TAG.format(tag=run_tag, paradigm=paradigm, task=task)
    else:
        filename = CACHE_FILENAME.format(paradigm=paradigm, task=task)
    return Path(output_dir) / filename


def find_latest_cache(output_dir: str, paradigm: str, task: str) -> Optional[Path]:
    """Find the latest cache file (tagged or untagged) for the given paradigm and task."""
    results_dir = Path(output_dir)
    if not results_dir.exists():
        return None

    # Pattern matches both tagged and untagged cache files
    pattern = f'*comparison_cache_{paradigm}_{task}.json'
    cache_files = list(results_dir.glob(pattern))

    if not cache_files:
        # Fallback: try old format without paradigm
        old_pattern = f'*comparison_cache_{task}.json'
        cache_files = list(results_dir.glob(old_pattern))

    if not cache_files:
        return None

    # Sort by modification time, newest first
    def safe_mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    cache_files.sort(key=safe_mtime, reverse=True)
    return cache_files[0]


def load_cache(
    output_dir: str,
    paradigm: str,
    task: str,
    run_tag: Optional[str] = None,
    find_latest: bool = False
) -> Tuple[Dict[str, Dict[str, dict]], Dict]:
    """Load cached results with backward compatibility for old cache format.

    Args:
        output_dir: Directory containing cache files
        paradigm: 'imagery' or 'movement'
        task: 'binary', 'ternary', or 'quaternary'
        run_tag: Optional tag for specific run (e.g., '20260110_2317')
        find_latest: If True and run_tag is None, find the latest cache file

    Returns:
        Tuple of (results_dict, metadata_dict) where metadata contains:
        - run_tag: Optional[str] - the run tag from cache
        - wandb_groups: Dict[str, str] - model_type -> wandb_group mapping
    """
    empty_metadata = {'run_tag': None, 'wandb_groups': {}}

    def extract_metadata(data: dict) -> Dict:
        """Extract metadata from cache data with backward compatibility."""
        return {
            'run_tag': data.get('run_tag'),
            'wandb_groups': data.get('wandb_groups', {}),
        }

    if find_latest and not run_tag:
        latest_cache = find_latest_cache(output_dir, paradigm, task)
        if latest_cache:
            try:
                with open(latest_cache, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                log_cache.info(f"Loaded latest cache: {latest_cache.name}")
                return data.get('results', {}), extract_metadata(data)
            except Exception as e:
                log_cache.warning(f"Failed to load latest cache: {e}")
        return {}, empty_metadata

    cache_path = get_cache_path(output_dir, paradigm, task, run_tag)
    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            log_cache.info(f"Loaded from {cache_path.name}")
            return data.get('results', {}), extract_metadata(data)
        except Exception as e:
            log_cache.warning(f"Failed to load: {e}")

    # Fallback: try old format without paradigm (backward compatibility)
    if not run_tag:
        old_format_path = Path(output_dir) / f'comparison_cache_{task}.json'
        if old_format_path.exists():
            try:
                with open(old_format_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                log_cache.info(f"Loaded legacy {old_format_path} -> new: {cache_path.name}")
                return data.get('results', {}), extract_metadata(data)
            except Exception as e:
                log_cache.warning(f"Failed to load legacy: {e}")

    return {}, empty_metadata


def save_cache(
    output_dir: str,
    paradigm: str,
    task: str,
    results: Dict[str, Dict[str, dict]],
    run_tag: Optional[str] = None,
    wandb_groups: Optional[Dict[str, str]] = None,
):
    """Save results to cache using atomic write to prevent corruption.

    Args:
        output_dir: Directory to save cache files
        paradigm: 'imagery' or 'movement'
        task: 'binary', 'ternary', or 'quaternary'
        results: Training results dict
        run_tag: Optional tag for specific run
        wandb_groups: Optional dict mapping model_type to wandb_group name
    """
    cache_path = get_cache_path(output_dir, paradigm, task, run_tag)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'paradigm': paradigm,
        'task': task,
        'run_tag': run_tag,
        'wandb_groups': wandb_groups or {},
        'last_updated': datetime.now().isoformat(),
        'results': results,
    }

    # Atomic write: write to temp file, then rename
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=cache_path.parent,
            suffix='.tmp',
            delete=False
        ) as f:
            json.dump(data, f, indent=2)
            temp_path = f.name
        os.replace(temp_path, cache_path)
    except Exception as e:
        # Fallback to direct write if atomic fails
        log_cache.warning(f"Atomic write failed, fallback: {e}")
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)


# ============================================================================
# Result Serialization
# ============================================================================

def result_to_dict(result: TrainingResult) -> dict:
    """Convert TrainingResult to serializable dict."""
    return {
        'subject_id': result.subject_id,
        'task_type': result.task_type,
        'model_type': result.model_type,
        'best_val_acc': result.best_val_acc,
        'test_acc': result.test_acc,
        'test_acc_majority': result.test_acc_majority,
        'epochs_trained': result.epochs_trained,
        'training_time': result.training_time,
    }


def dict_to_result(d: dict) -> TrainingResult:
    """Convert dict back to TrainingResult."""
    return TrainingResult(
        subject_id=d['subject_id'],
        task_type=d.get('task_type', 'binary'),
        model_type=d['model_type'],
        best_val_acc=d['best_val_acc'],
        test_acc=d['test_acc'],
        test_acc_majority=d['test_acc_majority'],
        epochs_trained=d['epochs_trained'],
        training_time=d['training_time'],
    )


# ============================================================================
# Training Functions
# ============================================================================

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
    no_wandb: bool = False,
    upload_model: bool = False,
    wandb_group: Optional[str] = None,
    preprocess_config: Optional[PreprocessConfig] = None,
    wandb_interactive: bool = False,
    wandb_metadata: Optional[Dict] = None,
    cache_only: bool = False,
    cache_index_path: str = ".cache_index.json",
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
        preprocess_config: Optional custom PreprocessConfig for ML engineering experiments
        wandb_interactive: Prompt for run details interactively
        wandb_metadata: Pre-collected metadata (goal, hypothesis, notes) for batch training
        cache_only: If True, load data exclusively from cache index
        cache_index_path: Path to cache index file for cache_only mode
    """
    result_dict = train_subject_simple(
        subject_id=subject_id,
        model_type=model_type,
        task=task,
        paradigm=paradigm,
        data_root=data_root,
        save_dir=save_dir,
        no_wandb=no_wandb,
        upload_model=upload_model,
        wandb_group=wandb_group,
        preprocess_config=preprocess_config,
        wandb_interactive=wandb_interactive,
        wandb_metadata=wandb_metadata,
        cache_only=cache_only,
        cache_index_path=cache_index_path,
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


# ============================================================================
# Statistics Functions
# ============================================================================

def compute_model_statistics(results: List[TrainingResult]) -> Dict:
    """Compute summary statistics for a single model's results."""
    if not results:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0,
            'n_subjects': 0,
        }

    accs = [r.test_acc_majority for r in results]
    return {
        'mean': float(np.mean(accs)),
        'std': float(np.std(accs)),
        'min': float(np.min(accs)),
        'max': float(np.max(accs)),
        'median': float(np.median(accs)),
        'n_subjects': len(accs),
    }


def print_model_summary(model_type: str, stats: Dict, results: List[TrainingResult]):
    """Print formatted summary for a single model."""
    print("\n" + "=" * 70)
    print(f" {model_type.upper()} SUMMARY")
    print("=" * 70)

    # Per-subject table
    print(f"\n{'Subject':<10} {'Val Acc':<12} {'Test Acc':<12} {'Epochs':<10} {'Time (s)':<10}")
    print("-" * 54)
    for r in sorted(results, key=lambda x: x.subject_id):
        print(f"{r.subject_id:<10} {r.best_val_acc:<12.2%} {r.test_acc_majority:<12.2%} "
              f"{r.epochs_trained:<10} {r.training_time:<10.1f}")

    # Statistics
    print("\n" + "-" * 54)
    print(f"{'Statistic':<15} {'Value':<15}")
    print("-" * 30)
    print(f"{'N Subjects':<15} {stats['n_subjects']:<15}")
    print(f"{'Mean':<15} {stats['mean']:.2%}")
    print(f"{'Median':<15} {stats['median']:.2%}")
    print(f"{'Std':<15} {stats['std']:.2%}")
    print(f"{'Min':<15} {stats['min']:.2%}")
    print(f"{'Max':<15} {stats['max']:.2%}")
    print("=" * 70 + "\n")
