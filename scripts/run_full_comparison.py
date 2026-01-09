#!/usr/bin/env python
"""
Full Model Comparison Script for EEG-BCI Project.

This script runs training and evaluation of both EEGNet and CBraMod
on all available subjects, then compares their performance.

Data Split (follows paper protocol):
- Training: Offline + Session 1 (Base + Finetune) + Session 2 Base
- Validation: Last 20% of training data (temporal split)
- Test: Session 2 Finetune (completely held out)

Features:
- Uses FingerEEGDataset with HDF5 preprocessing cache (20-40x speedup)
- Supports both Motor Imagery (MI) and Motor Execution (ME) paradigms
- Incremental result caching: Results saved after each subject completes
- Resume capability: Automatically skips subjects with cached results
- New run mode: Use --new-run to start fresh experiments without overwriting old results

Usage:
    # Run on Motor Imagery (default paradigm, plots generated automatically)
    uv run python scripts/run_full_comparison.py

    # Run on Motor Execution
    uv run python scripts/run_full_comparison.py --paradigm movement

    # Start a NEW experiment (preserves old results/plots with datetime tag)
    uv run python scripts/run_full_comparison.py --new-run

    # Force retrain (overwrites existing cache)
    uv run python scripts/run_full_comparison.py --force-retrain

    # Run on specific subjects
    uv run python scripts/run_full_comparison.py --subjects S01 S02 S03

    # Run only EEGNet
    uv run python scripts/run_full_comparison.py --models eegnet

    # Load existing results only (no training)
    uv run python scripts/run_full_comparison.py --skip-training

    # Suppress plot generation
    uv run python scripts/run_full_comparison.py --no-plot

    # Combine options: ME paradigm, new run
    uv run python scripts/run_full_comparison.py --paradigm movement --new-run
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.device import set_seed, check_cuda_available, get_device
from src.utils.logging import YellowFormatter, SectionLogger, setup_logging
from src.preprocessing.data_loader import discover_available_subjects
from src.training.train_within_subject import train_subject_simple


setup_logging('compare')
logger = logging.getLogger(__name__)

# Section-specific loggers
log_main = SectionLogger(logger, 'main')
log_cache = SectionLogger(logger, 'cache')
log_train = SectionLogger(logger, 'train')
log_stats = SectionLogger(logger, 'stats')
log_io = SectionLogger(logger, 'io')


# Cache file for incremental results
CACHE_FILENAME = 'comparison_cache_{paradigm}_{task}.json'
CACHE_FILENAME_WITH_TAG = '{tag}_comparison_cache_{paradigm}_{task}.json'

# Paradigm configurations (MI vs ME)
PARADIGM_CONFIG = {
    'imagery': {
        'description': 'Motor Imagery (MI)',
    },
    'movement': {
        'description': 'Motor Execution (ME)',
    },
}


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


@dataclass
class ComparisonResult:
    """Result of model comparison."""
    n_subjects: int
    eegnet_mean: float
    eegnet_std: float
    cbramod_mean: float
    cbramod_std: float
    difference_mean: float
    difference_std: float
    paired_ttest_t: float
    paired_ttest_p: float
    wilcoxon_stat: Optional[float]
    wilcoxon_p: Optional[float]
    better_model: str
    significant: bool


def discover_subjects(data_root: str, paradigm: str = 'imagery', task: str = 'binary') -> List[str]:
    """
    Discover all available subjects in data directory.

    Uses the new discover_available_subjects function which checks for
    required test data (Session 2 Finetune).
    """
    return discover_available_subjects(data_root, paradigm, task)


def get_cache_path(output_dir: str, paradigm: str, task: str, run_tag: Optional[str] = None) -> Path:
    """Get path to cache file."""
    if run_tag:
        filename = CACHE_FILENAME_WITH_TAG.format(tag=run_tag, paradigm=paradigm, task=task)
    else:
        filename = CACHE_FILENAME.format(paradigm=paradigm, task=task)
    return Path(output_dir) / filename


def load_cache(output_dir: str, paradigm: str, task: str, run_tag: Optional[str] = None) -> Dict[str, Dict[str, dict]]:
    """Load cached results with backward compatibility for old cache format."""
    cache_path = get_cache_path(output_dir, paradigm, task, run_tag)
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            log_cache.info(f"Loaded from {cache_path}")
            return data.get('results', {})
        except Exception as e:
            log_cache.warning(f"Failed to load: {e}")

    # Fallback: try old format without paradigm (backward compatibility)
    if not run_tag:
        old_format_path = Path(output_dir) / f'comparison_cache_{task}.json'
        if old_format_path.exists():
            try:
                with open(old_format_path, 'r') as f:
                    data = json.load(f)
                log_cache.info(f"Loaded legacy {old_format_path} → new: {cache_path.name}")
                return data.get('results', {})
            except Exception as e:
                log_cache.warning(f"Failed to load legacy: {e}")

    return {}


def save_cache(output_dir: str, paradigm: str, task: str, results: Dict[str, Dict[str, dict]], run_tag: Optional[str] = None):
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
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)


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
) -> TrainingResult:
    """
    Train a model for a single subject and return TrainingResult.

    This is a thin wrapper around train_subject_simple from train_within_subject.py.
    It handles the conversion from the dict result to TrainingResult dataclass.

    Args:
        subject_id: Subject ID (e.g., 'S01')
        model_type: 'eegnet' or 'cbramod'
        task: 'binary', 'ternary', or 'quaternary'
        paradigm: 'imagery' or 'movement'
        data_root: Path to data directory
        save_dir: Path to save checkpoints

    Returns:
        TrainingResult with training metrics
    """
    # Call the unified training function
    result_dict = train_subject_simple(
        subject_id=subject_id,
        model_type=model_type,
        task=task,
        paradigm=paradigm,
        data_root=data_root,
        save_dir=save_dir,
    )

    # Handle empty result (training failed)
    if not result_dict:
        raise ValueError(f"Training failed for {subject_id}")

    # Convert to TrainingResult dataclass
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


def run_with_cache(
    data_root: str,
    subject_ids: List[str],
    task: str,
    paradigm: str,
    model_types: List[str],
    output_dir: str,
    force_retrain: bool = False,
    run_tag: Optional[str] = None,
) -> Dict[str, List[TrainingResult]]:
    """
    Run experiments with incremental caching.

    Uses FingerEEGDataset for preprocessing cache.
    Results are saved after each subject completes.

    Args:
        run_tag: Optional datetime tag for new runs (e.g., "20260103_1430")
    """
    device = get_device()

    paradigm_config = PARADIGM_CONFIG[paradigm]
    log_train.info(f"Paradigm: {paradigm_config['description']}")

    # Load existing result cache
    if force_retrain:
        cache = {}
        log_train.info("Force retrain - ignoring cache")
    elif run_tag:
        # New run with tag - load from its own cache (supports resume)
        cache = load_cache(output_dir, paradigm, task, run_tag)
        if cache:
            log_train.info(f"Resuming '{run_tag}' - {sum(len(v) for v in cache.values())} cached")
        else:
            log_train.info(f"New run '{run_tag}' - fresh start")
    else:
        cache = load_cache(output_dir, paradigm, task)

    results: Dict[str, List[TrainingResult]] = {m: [] for m in model_types}

    total_tasks = len(model_types) * len(subject_ids)
    completed = 0

    for model_type in model_types:
        log_train.info(f"{'='*50} {model_type.upper()} {'='*50}")

        if model_type not in cache:
            cache[model_type] = {}

        for subject_id in subject_ids:
            completed += 1
            progress = f"[{completed}/{total_tasks}]"

            # Check result cache
            if subject_id in cache[model_type] and not force_retrain:
                log_train.info(f"{progress} {subject_id}: cached")
                cached_result = dict_to_result(cache[model_type][subject_id])
                results[model_type].append(cached_result)
                print_subject_result(subject_id, model_type, cached_result)
                continue

            # Train
            log_train.info(f"{progress} {subject_id}: training {model_type}...")

            try:
                # IMPORTANT: Reset seed before each training to ensure reproducibility
                # Without this, the random state drifts after each subject/model,
                # causing different results compared to running subjects individually
                set_seed(42)

                result = train_and_get_result(
                    subject_id=subject_id,
                    model_type=model_type,
                    task=task,
                    paradigm=paradigm,
                    data_root=data_root,
                    save_dir=output_dir,
                )

                results[model_type].append(result)

                # Save to result cache immediately
                cache[model_type][subject_id] = result_to_dict(result)
                save_cache(output_dir, paradigm, task, cache, run_tag)

                print_subject_result(subject_id, model_type, result)

            except Exception as e:
                log_train.error(f"{progress} {subject_id}: FAILED - {e}")
                traceback.print_exc()
                continue

        # Model summary
        if results[model_type]:
            accs = [r.test_acc_majority for r in results[model_type]]
            log_train.info(f"{model_type.upper()} done: {np.mean(accs):.1%}±{np.std(accs):.1%} (n={len(accs)}, best={np.max(accs):.1%})")

    return results


def compare_models(
    eegnet_results: List[TrainingResult],
    cbramod_results: List[TrainingResult],
) -> ComparisonResult:
    """Perform statistical comparison between EEGNet and CBraMod."""
    eegnet_by_subject = {r.subject_id: r for r in eegnet_results}
    cbramod_by_subject = {r.subject_id: r for r in cbramod_results}

    common_subjects = set(eegnet_by_subject.keys()) & set(cbramod_by_subject.keys())

    if len(common_subjects) < 2:
        raise ValueError("Need at least 2 subjects for comparison")

    # Warn about small sample size
    if len(common_subjects) < 5:
        log_stats.warning(f"Small sample (n={len(common_subjects)}): stats may be unreliable")

    eegnet_accs = []
    cbramod_accs = []

    for subject_id in sorted(common_subjects):
        eegnet_accs.append(eegnet_by_subject[subject_id].test_acc_majority)
        cbramod_accs.append(cbramod_by_subject[subject_id].test_acc_majority)

    eegnet_accs = np.array(eegnet_accs)
    cbramod_accs = np.array(cbramod_accs)
    differences = cbramod_accs - eegnet_accs

    # Paired t-test
    t_stat, t_pvalue = stats.ttest_rel(cbramod_accs, eegnet_accs)

    # Wilcoxon signed-rank test (may fail with very small n or all-zero differences)
    try:
        w_stat, w_pvalue = stats.wilcoxon(cbramod_accs, eegnet_accs)
    except ValueError:
        # Wilcoxon requires n >= 10 for reliable results, may fail with smaller n
        w_stat, w_pvalue = None, None

    # Determine which model has higher mean accuracy
    # NOTE: This does NOT imply statistical significance - check 'significant' field
    if np.mean(cbramod_accs) > np.mean(eegnet_accs):
        higher_mean_model = 'cbramod'
    elif np.mean(eegnet_accs) > np.mean(cbramod_accs):
        higher_mean_model = 'eegnet'
    else:
        higher_mean_model = 'tie'

    return ComparisonResult(
        n_subjects=len(common_subjects),
        eegnet_mean=float(np.mean(eegnet_accs)),
        eegnet_std=float(np.std(eegnet_accs)),
        cbramod_mean=float(np.mean(cbramod_accs)),
        cbramod_std=float(np.std(cbramod_accs)),
        difference_mean=float(np.mean(differences)),
        difference_std=float(np.std(differences)),
        paired_ttest_t=float(t_stat),
        paired_ttest_p=float(t_pvalue),
        wilcoxon_stat=float(w_stat) if w_stat is not None else None,
        wilcoxon_p=float(w_pvalue) if w_pvalue is not None else None,
        better_model=higher_mean_model,  # Kept as 'better_model' for API compatibility
        significant=bool(t_pvalue < 0.05),
    )


def print_comparison_report(
    results: Dict[str, List[TrainingResult]],
    comparison: Optional[ComparisonResult],
    task_type: str,
    paradigm: str = 'imagery',
    run_tag: Optional[str] = None,
):
    """Print a detailed comparison report."""
    paradigm_desc = PARADIGM_CONFIG[paradigm]['description']
    print("\n" + "=" * 70)
    print(" EEG-BCI MODEL COMPARISON REPORT")
    print("=" * 70)
    print(f"\nParadigm: {paradigm_desc}")
    print(f"Task: {task_type.upper()}")
    if run_tag:
        print(f"Run Tag: {run_tag}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n" + "-" * 70)
    print(" PER-SUBJECT RESULTS (Test on Session 2 Finetune, Majority Voting)")
    print("-" * 70)
    print(f"{'Subject':<10} {'EEGNet':<15} {'CBraMod':<15} {'Difference':<15}")
    print("-" * 70)

    eegnet_by_subject = {r.subject_id: r for r in results.get('eegnet', [])}
    cbramod_by_subject = {r.subject_id: r for r in results.get('cbramod', [])}

    all_subjects = sorted(set(eegnet_by_subject.keys()) | set(cbramod_by_subject.keys()))

    for subject_id in all_subjects:
        eegnet_acc = eegnet_by_subject.get(subject_id)
        cbramod_acc = cbramod_by_subject.get(subject_id)

        e_str = f"{eegnet_acc.test_acc_majority:.2%}" if eegnet_acc else "N/A"
        c_str = f"{cbramod_acc.test_acc_majority:.2%}" if cbramod_acc else "N/A"

        if eegnet_acc and cbramod_acc:
            diff = cbramod_acc.test_acc_majority - eegnet_acc.test_acc_majority
            diff_str = f"{diff:+.2%}"
            if diff > 0:
                diff_str += " (CBraMod)"
            elif diff < 0:
                diff_str += " (EEGNet)"
        else:
            diff_str = "N/A"

        print(f"{subject_id:<10} {e_str:<15} {c_str:<15} {diff_str:<15}")

    print("\n" + "-" * 70)
    print(" SUMMARY STATISTICS")
    print("-" * 70)
    print(f"{'Model':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 70)

    for model_type in ['eegnet', 'cbramod']:
        model_results = results.get(model_type, [])
        if model_results:
            accs = [r.test_acc_majority for r in model_results]
            print(f"{model_type.upper():<15} {np.mean(accs):.2%}      {np.std(accs):.2%}      "
                  f"{np.min(accs):.2%}      {np.max(accs):.2%}")

    if comparison:
        print("\n" + "-" * 70)
        print(" STATISTICAL COMPARISON")
        print("-" * 70)
        print(f"Subjects compared: {comparison.n_subjects}")
        if comparison.n_subjects < 5:
            print(f"  WARNING: Small sample size - results may be unreliable")
        print(f"Difference (CBraMod - EEGNet): {comparison.difference_mean:+.2%} +/- {comparison.difference_std:.2%}")
        print(f"\nPaired t-test:")
        print(f"  t-statistic: {comparison.paired_ttest_t:.4f}")
        print(f"  p-value: {comparison.paired_ttest_p:.4f}")

        if comparison.wilcoxon_stat is not None:
            print(f"\nWilcoxon signed-rank test:")
            print(f"  W-statistic: {comparison.wilcoxon_stat:.4f}")
            print(f"  p-value: {comparison.wilcoxon_p:.4f}")
        else:
            print(f"\nWilcoxon signed-rank test: N/A (requires larger sample or non-zero differences)")

        print(f"\nConclusion:")
        if comparison.significant:
            print(f"  {comparison.better_model.upper()} has significantly higher mean accuracy (p < 0.05)")
        else:
            print(f"  No significant difference between models (p = {comparison.paired_ttest_p:.4f})")

        # Add caveat for small sample sizes
        if comparison.n_subjects < 10:
            print(f"  Note: With n={comparison.n_subjects}, statistical power is limited.")

    print("\n" + "=" * 70)


def generate_plot(
    results: Dict[str, List[TrainingResult]],
    comparison: ComparisonResult,
    output_path: str,
    task_type: str = 'binary',
):
    """Generate comparison plots.

    Args:
        results: Training results by model type.
        comparison: Statistical comparison result.
        output_path: Path to save the plot.
        task_type: 'binary', 'ternary', or 'quaternary' - determines chance level.
    """
    # Calculate chance level based on task type
    chance_levels = {'binary': 0.5, 'ternary': 1/3, 'quaternary': 0.25}
    chance_level = chance_levels.get(task_type, 0.5)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log_io.warning("matplotlib not installed, skipping plots")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    eegnet_results = results.get('eegnet', [])
    cbramod_results = results.get('cbramod', [])

    # Match subjects
    eegnet_by_subj = {r.subject_id: r for r in eegnet_results}
    cbramod_by_subj = {r.subject_id: r for r in cbramod_results}
    common = sorted(set(eegnet_by_subj.keys()) & set(cbramod_by_subj.keys()))

    # Handle empty data
    if not common:
        log_io.warning("No common subjects for plotting")
        return

    eegnet_accs = [eegnet_by_subj[s].test_acc_majority for s in common]
    cbramod_accs = [cbramod_by_subj[s].test_acc_majority for s in common]

    # Plot 1: Bar chart
    ax1 = axes[0]
    x = np.arange(len(common))
    width = 0.35

    ax1.bar(x - width/2, eegnet_accs, width, label='EEGNet', color='steelblue')
    ax1.bar(x + width/2, cbramod_accs, width, label='CBraMod', color='coral')
    ax1.set_xlabel('Subject')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Per-Subject Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(common, rotation=45)
    ax1.legend()
    ax1.set_ylim([0, 1])
    ax1.axhline(y=chance_level, color='gray', linestyle='--', alpha=0.5,
                label=f'Chance ({chance_level*100:.1f}%)')

    # Plot 2: Box plot
    ax2 = axes[1]
    ax2.boxplot([eegnet_accs, cbramod_accs], labels=['EEGNet', 'CBraMod'])
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Accuracy Distribution')
    ax2.axhline(y=chance_level, color='gray', linestyle='--', alpha=0.5)

    # Plot 3: Scatter plot
    ax3 = axes[2]
    ax3.scatter(eegnet_accs, cbramod_accs, s=100, alpha=0.7)
    for i, subj in enumerate(common):
        ax3.annotate(subj, (eegnet_accs[i], cbramod_accs[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

    lims = [min(min(eegnet_accs), min(cbramod_accs)) - 0.05,
            max(max(eegnet_accs), max(cbramod_accs)) + 0.05]
    ax3.plot(lims, lims, 'k--', alpha=0.5, label='Equal')
    ax3.set_xlabel('EEGNet Accuracy')
    ax3.set_ylabel('CBraMod Accuracy')
    ax3.set_title('Paired Comparison')
    ax3.set_xlim(lims)
    ax3.set_ylim(lims)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    log_io.info(f"Plot saved: {output_path}")
    plt.close()


def save_full_results(
    results: Dict[str, List[TrainingResult]],
    comparison: Optional[ComparisonResult],
    task_type: str,
    paradigm: str,
    output_dir: str,
    run_tag: Optional[str] = None,
):
    """Save comprehensive results to JSON."""
    output = {
        'metadata': {
            'paradigm': paradigm,
            'paradigm_description': PARADIGM_CONFIG[paradigm]['description'],
            'task_type': task_type,
            'run_tag': run_tag,
            'timestamp': datetime.now().isoformat(),
            'n_subjects': len(set(
                r.subject_id for model_results in results.values()
                for r in model_results
            )),
        },
        'models': {},
        'comparison': None,
    }

    for model_type, model_results in results.items():
        accs = [r.test_acc_majority for r in model_results]
        output['models'][model_type] = {
            'subjects': [result_to_dict(r) for r in model_results],
            'summary': {
                'mean': float(np.mean(accs)) if accs else 0,
                'std': float(np.std(accs)) if accs else 0,
                'min': float(np.min(accs)) if accs else 0,
                'max': float(np.max(accs)) if accs else 0,
            }
        }

    if comparison:
        output['comparison'] = asdict(comparison)

    # Use run_tag if provided, otherwise generate timestamp
    if run_tag:
        filename = f'{run_tag}_comparison_{paradigm}_{task_type}.json'
    else:
        filename = f'comparison_{paradigm}_{task_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    log_io.info(f"Results saved: {output_path}")
    return output_path


def load_existing_results(results_file: str) -> Dict[str, List[TrainingResult]]:
    """Load results from a previous run."""
    with open(results_file, 'r') as f:
        data = json.load(f)

    results = {}
    for model_type, model_data in data.get('models', data).items():
        subjects_data = model_data.get('subjects', model_data)
        if isinstance(subjects_data, dict):
            subjects_data = subjects_data.get('subjects', [])

        results[model_type] = [dict_to_result(s) for s in subjects_data]

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run full model comparison on all subjects',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run on Motor Imagery (default, plots generated automatically)
  uv run python scripts/run_full_comparison.py

  # Run on Motor Execution
  uv run python scripts/run_full_comparison.py --paradigm movement

  # Start a NEW experiment (preserves old results/plots with datetime tag)
  uv run python scripts/run_full_comparison.py --new-run

  # Force retrain (overwrites existing cache for this paradigm/task)
  uv run python scripts/run_full_comparison.py --force-retrain

  # Suppress plot generation
  uv run python scripts/run_full_comparison.py --no-plot
'''
    )

    parser.add_argument(
        '--data-root', type=str, default='data',
        help='Path to data directory (default: data)'
    )
    parser.add_argument(
        '--subjects', nargs='+', default=None,
        help='Specific subjects to run (default: all available)'
    )
    parser.add_argument(
        '--models', nargs='+', default=['eegnet', 'cbramod'],
        choices=['eegnet', 'cbramod'],
        help='Models to train (default: both)'
    )
    parser.add_argument(
        '--paradigm', type=str, default='imagery',
        choices=['imagery', 'movement'],
        help='Experiment paradigm: imagery (MI) or movement (ME) (default: imagery)'
    )
    parser.add_argument(
        '--task', type=str, default='binary',
        choices=['binary', 'ternary', 'quaternary'],
        help='Classification task (default: binary)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='results',
        help='Directory to save results (default: results)'
    )
    parser.add_argument(
        '--new-run', action='store_true',
        help='Start a NEW experiment run with datetime tag (preserves old results)'
    )
    parser.add_argument(
        '--force-retrain', action='store_true',
        help='Force retraining, overwriting cached results for this paradigm/task'
    )
    parser.add_argument(
        '--skip-training', action='store_true',
        help='Skip training entirely and load existing results file'
    )
    parser.add_argument(
        '--results-file', type=str, default=None,
        help='Path to existing results file (used with --skip-training)'
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help='Suppress plot generation (plots are generated by default)'
    )

    args = parser.parse_args()

    # Check GPU
    check_cuda_available(required=True)
    device = get_device()
    log_main.info(f"Device: {device}")

    # Set seed
    set_seed(args.seed)
    log_main.info(f"Seed: {args.seed}")

    # Generate run tag if --new-run is specified
    run_tag = None
    if args.new_run:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M")
        log_main.info(f"New run: {run_tag}")

    paradigm_desc = PARADIGM_CONFIG[args.paradigm]['description']
    log_main.info(f"Paradigm: {paradigm_desc}")

    if args.skip_training:
        if args.results_file is None:
            results_dir = Path(args.output_dir)
            # Find results matching current paradigm and task
            pattern = f'*comparison_{args.paradigm}_{args.task}*.json'
            result_files = sorted(results_dir.glob(pattern), reverse=True)
            if not result_files:
                # Fallback to old format without paradigm
                result_files = sorted(results_dir.glob(f'comparison_{args.task}_*.json'), reverse=True)
            if not result_files:
                log_main.error("No results files found. Run training first.")
                sys.exit(1)
            args.results_file = str(result_files[0])
            log_io.info(f"Using: {args.results_file}")

        results = load_existing_results(args.results_file)
        log_io.info(f"Loaded: {args.results_file}")
    else:
        if args.subjects:
            subjects = args.subjects
        else:
            subjects = discover_subjects(args.data_root, args.paradigm, args.task)

        if not subjects:
            log_main.error(f"No subjects in {args.data_root}")
            sys.exit(1)

        log_main.info(f"Subjects: {subjects} | Models: {args.models} | Task: {args.task}")

        if not args.force_retrain and not args.new_run:
            log_main.info("Using cache (--new-run for fresh, --force-retrain to overwrite)")

        results = run_with_cache(
            data_root=args.data_root,
            subject_ids=subjects,
            task=args.task,
            paradigm=args.paradigm,
            model_types=args.models,
            output_dir=args.output_dir,
            force_retrain=args.force_retrain,
            run_tag=run_tag,
        )

    # Compare models
    comparison = None
    if 'eegnet' in results and 'cbramod' in results:
        if len(results['eegnet']) >= 2 and len(results['cbramod']) >= 2:
            try:
                comparison = compare_models(results['eegnet'], results['cbramod'])
            except ValueError as e:
                log_stats.warning(f"Cannot compare: {e}")

    print_comparison_report(results, comparison, args.task, args.paradigm, run_tag)

    output_path = save_full_results(
        results, comparison, args.task, args.paradigm, args.output_dir, run_tag
    )

    # Generate plots by default (unless --no-plot is specified)
    if not args.no_plot and comparison:
        if run_tag:
            plot_filename = f'{run_tag}_comparison_{args.paradigm}_{args.task}.png'
        else:
            plot_filename = f'comparison_{args.paradigm}_{args.task}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plot_path = Path(args.output_dir) / plot_filename
        generate_plot(results, comparison, str(plot_path), task_type=args.task)

    return 0


if __name__ == '__main__':
    sys.exit(main())
