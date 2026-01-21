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
from typing import Dict, List, Optional

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.device import set_seed
from src.utils.logging import SectionLogger, setup_logging
from src.preprocessing.data_loader import discover_available_subjects
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

def discover_subjects(data_root: str, paradigm: str = 'imagery', task: str = 'binary') -> List[str]:
    """
    Discover all available subjects in data directory.

    Uses the discover_available_subjects function which checks for
    required test data (Session 2 Finetune).
    """
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
) -> Dict[str, Dict[str, dict]]:
    """Load cached results with backward compatibility for old cache format.

    Args:
        output_dir: Directory containing cache files
        paradigm: 'imagery' or 'movement'
        task: 'binary', 'ternary', or 'quaternary'
        run_tag: Optional tag for specific run (e.g., '20260110_2317')
        find_latest: If True and run_tag is None, find the latest cache file
    """
    if find_latest and not run_tag:
        latest_cache = find_latest_cache(output_dir, paradigm, task)
        if latest_cache:
            try:
                with open(latest_cache, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                log_cache.info(f"Loaded latest cache: {latest_cache.name}")
                return data.get('results', {})
            except Exception as e:
                log_cache.warning(f"Failed to load latest cache: {e}")
        return {}

    cache_path = get_cache_path(output_dir, paradigm, task, run_tag)
    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            log_cache.info(f"Loaded from {cache_path.name}")
            return data.get('results', {})
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
                return data.get('results', {})
            except Exception as e:
                log_cache.warning(f"Failed to load legacy: {e}")

    return {}


def save_cache(
    output_dir: str,
    paradigm: str,
    task: str,
    results: Dict[str, Dict[str, dict]],
    run_tag: Optional[str] = None
):
    """Save results to cache using atomic write to prevent corruption."""
    cache_path = get_cache_path(output_dir, paradigm, task, run_tag)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'paradigm': paradigm,
        'task': task,
        'run_tag': run_tag,
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
    wandb_group: Optional[str] = None,
) -> TrainingResult:
    """
    Train a model for a single subject and return TrainingResult.

    This is a thin wrapper around train_subject_simple from train_within_subject.py.
    """
    result_dict = train_subject_simple(
        subject_id=subject_id,
        model_type=model_type,
        task=task,
        paradigm=paradigm,
        data_root=data_root,
        save_dir=save_dir,
        no_wandb=no_wandb,
        wandb_group=wandb_group,
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
