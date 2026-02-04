"""
Statistical analysis functions for training results.

This module provides functions for computing summary statistics,
performing statistical comparisons, and printing formatted reports.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

from ..config.constants import PARADIGM_CONFIG
from ..utils.logging import SectionLogger
from .dataclasses import ComparisonResult, TrainingResult

logger = logging.getLogger(__name__)
log_stats = SectionLogger(logger, 'stats')


def compute_model_statistics(results: List[TrainingResult]) -> Dict:
    """Compute summary statistics for a single model's results.

    Args:
        results: List of TrainingResult objects

    Returns:
        Dict with keys: mean, std, min, max, median, n_subjects
    """
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
    """Print formatted summary for a single model.

    Args:
        model_type: Model name ('eegnet' or 'cbramod')
        stats: Statistics dict from compute_model_statistics()
        results: List of TrainingResult objects
    """
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


def compare_models(
    eegnet_results: List[TrainingResult],
    cbramod_results: List[TrainingResult],
) -> ComparisonResult:
    """Perform statistical comparison between EEGNet and CBraMod.

    Args:
        eegnet_results: List of TrainingResult for EEGNet
        cbramod_results: List of TrainingResult for CBraMod

    Returns:
        ComparisonResult with statistical analysis

    Raises:
        ValueError: If fewer than 2 common subjects
    """
    eegnet_by_subject = {r.subject_id: r for r in eegnet_results}
    cbramod_by_subject = {r.subject_id: r for r in cbramod_results}

    common_subjects = set(eegnet_by_subject.keys()) & set(cbramod_by_subject.keys())

    if len(common_subjects) < 2:
        raise ValueError("Need at least 2 subjects for comparison")

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

    # Wilcoxon signed-rank test
    try:
        w_stat, w_pvalue = stats.wilcoxon(cbramod_accs, eegnet_accs)
    except ValueError:
        w_stat, w_pvalue = None, None

    # Determine which model has higher mean accuracy
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
        eegnet_median=float(np.median(eegnet_accs)),
        cbramod_mean=float(np.mean(cbramod_accs)),
        cbramod_std=float(np.std(cbramod_accs)),
        cbramod_median=float(np.median(cbramod_accs)),
        difference_mean=float(np.mean(differences)),
        difference_std=float(np.std(differences)),
        paired_ttest_t=float(t_stat),
        paired_ttest_p=float(t_pvalue),
        wilcoxon_stat=float(w_stat) if w_stat is not None else None,
        wilcoxon_p=float(w_pvalue) if w_pvalue is not None else None,
        better_model=higher_mean_model,
        significant=bool(t_pvalue < 0.05),
    )


def print_comparison_report(
    results: Dict[str, List[TrainingResult]],
    comparison: Optional[ComparisonResult],
    task_type: str,
    paradigm: str = 'imagery',
    run_tag: Optional[str] = None,
):
    """Print a detailed comparison report.

    Args:
        results: Dict mapping model_type to list of TrainingResult
        comparison: ComparisonResult or None
        task_type: 'binary', 'ternary', or 'quaternary'
        paradigm: 'imagery' or 'movement'
        run_tag: Optional run identifier
    """
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
    print(f"{'Model':<15} {'Mean':<12} {'Median':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 70)

    for model_type in ['eegnet', 'cbramod']:
        model_results = results.get(model_type, [])
        if model_results:
            accs = [r.test_acc_majority for r in model_results]
            print(f"{model_type.upper():<15} {np.mean(accs):.2%}      {np.median(accs):.2%}      {np.std(accs):.2%}      "
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

        if comparison.n_subjects < 10:
            print(f"  Note: With n={comparison.n_subjects}, statistical power is limited.")

    print("\n" + "=" * 70)
