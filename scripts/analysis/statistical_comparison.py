#!/usr/bin/env python
"""
Statistical Comparison Analysis for EEGNet vs CBraMod.

This script performs comprehensive statistical analysis comparing
the performance of EEGNet and CBraMod models on EEG-BCI tasks.

Usage:
    uv run python scripts/analysis/statistical_comparison.py \
        --results "results/comparison_imagery_binary_20260203_151711 21-subject per-subject.json"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.results import (
    load_comparison_results,
    compare_models,
    compute_model_statistics,
    print_comparison_report,
)


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size for paired samples."""
    diff = group2 - group1
    return float(np.mean(diff) / np.std(diff, ddof=1))


def compute_confidence_interval(
    data: np.ndarray, confidence: float = 0.95
) -> tuple[float, float]:
    """Compute confidence interval for the mean."""
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return float(mean - h), float(mean + h)


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def print_extended_report(
    results: dict,
    comparison,
    task_type: str,
    paradigm: str,
):
    """Print extended statistical report with additional metrics."""

    # First print the standard report
    print_comparison_report(results, comparison, task_type, paradigm)

    if not comparison:
        return

    # Extract accuracies for additional analysis
    eegnet_by_subject = {r.subject_id: r for r in results.get('eegnet', [])}
    cbramod_by_subject = {r.subject_id: r for r in results.get('cbramod', [])}

    common_subjects = sorted(
        set(eegnet_by_subject.keys()) & set(cbramod_by_subject.keys())
    )

    if len(common_subjects) < 2:
        return

    eegnet_accs = np.array([eegnet_by_subject[s].test_acc_majority for s in common_subjects])
    cbramod_accs = np.array([cbramod_by_subject[s].test_acc_majority for s in common_subjects])
    differences = cbramod_accs - eegnet_accs

    # Additional statistics
    print("\n" + "=" * 70)
    print(" EXTENDED STATISTICAL ANALYSIS")
    print("=" * 70)

    # Cohen's d
    cohens_d = compute_cohens_d(eegnet_accs, cbramod_accs)
    effect_interpretation = interpret_effect_size(cohens_d)
    print(f"\nEffect Size (Cohen's d): {cohens_d:.4f} ({effect_interpretation})")

    # Confidence intervals
    diff_ci = compute_confidence_interval(differences)
    print(f"95% CI for difference: [{diff_ci[0]:.2%}, {diff_ci[1]:.2%}]")

    eegnet_ci = compute_confidence_interval(eegnet_accs)
    cbramod_ci = compute_confidence_interval(cbramod_accs)
    print(f"95% CI for EEGNet mean: [{eegnet_ci[0]:.2%}, {eegnet_ci[1]:.2%}]")
    print(f"95% CI for CBraMod mean: [{cbramod_ci[0]:.2%}, {cbramod_ci[1]:.2%}]")

    # IQR and quartiles
    print("\n" + "-" * 70)
    print(" DISTRIBUTION STATISTICS")
    print("-" * 70)

    for model_name, accs in [("EEGNet", eegnet_accs), ("CBraMod", cbramod_accs)]:
        q1, q2, q3 = np.percentile(accs, [25, 50, 75])
        iqr = q3 - q1
        print(f"\n{model_name}:")
        print(f"  Q1 (25%): {q1:.2%}")
        print(f"  Q2 (50%): {q2:.2%}")
        print(f"  Q3 (75%): {q3:.2%}")
        print(f"  IQR:      {iqr:.2%}")

    # Per-subject analysis
    print("\n" + "-" * 70)
    print(" PER-SUBJECT DIFFERENCE ANALYSIS")
    print("-" * 70)

    # Sort by difference
    sorted_indices = np.argsort(differences)[::-1]  # Descending

    print(f"\n{'Subject':<10} {'EEGNet':<12} {'CBraMod':<12} {'Diff':<12} {'Winner':<10}")
    print("-" * 56)

    cbramod_wins = 0
    eegnet_wins = 0
    ties = 0

    for idx in sorted_indices:
        subject = common_subjects[idx]
        e_acc = eegnet_accs[idx]
        c_acc = cbramod_accs[idx]
        diff = differences[idx]

        if diff > 0.001:
            winner = "CBraMod"
            cbramod_wins += 1
        elif diff < -0.001:
            winner = "EEGNet"
            eegnet_wins += 1
        else:
            winner = "Tie"
            ties += 1

        print(f"{subject:<10} {e_acc:<12.2%} {c_acc:<12.2%} {diff:+.2%}      {winner:<10}")

    print("\n" + "-" * 56)
    print(f"CBraMod wins: {cbramod_wins}/{len(common_subjects)} ({cbramod_wins/len(common_subjects):.1%})")
    print(f"EEGNet wins:  {eegnet_wins}/{len(common_subjects)} ({eegnet_wins/len(common_subjects):.1%})")
    print(f"Ties:         {ties}/{len(common_subjects)}")

    # Sign test
    n_positive = np.sum(differences > 0)
    n_negative = np.sum(differences < 0)
    n_nonzero = n_positive + n_negative

    if n_nonzero > 0:
        # Binomial test (two-sided)
        sign_test_p = stats.binomtest(n_positive, n_nonzero, 0.5, alternative='two-sided').pvalue
        print(f"\nSign test: {n_positive} positive, {n_negative} negative, p = {sign_test_p:.4f}")

    # Normality test for differences
    if len(differences) >= 8:
        shapiro_stat, shapiro_p = stats.shapiro(differences)
        print(f"\nShapiro-Wilk normality test on differences: W = {shapiro_stat:.4f}, p = {shapiro_p:.4f}")
        if shapiro_p < 0.05:
            print("  Note: Differences may not be normally distributed; Wilcoxon test preferred")

    # Summary interpretation
    print("\n" + "=" * 70)
    print(" SUMMARY INTERPRETATION")
    print("=" * 70)

    diff_mean = np.mean(differences)

    if comparison.significant:
        if comparison.better_model == 'cbramod':
            print(f"\n[+] CBraMod significantly outperforms EEGNet (p < 0.05)")
        else:
            print(f"\n[+] EEGNet significantly outperforms CBraMod (p < 0.05)")
    else:
        print(f"\n[-] No statistically significant difference between models (p = {comparison.paired_ttest_p:.4f})")
        if comparison.paired_ttest_p < 0.1:
            print(f"  However, there is a trend toward CBraMod advantage (p < 0.10)")

    print(f"\nMean difference: {diff_mean:+.2%} ({comparison.better_model.upper()} higher)")
    print(f"Effect size: {effect_interpretation} (d = {cohens_d:.3f})")

    if abs(diff_ci[0]) < 0.001 or abs(diff_ci[1]) < 0.001:
        print("Note: 95% CI includes zero, suggesting no reliable difference")
    elif diff_ci[0] > 0:
        print("Note: 95% CI is entirely positive, suggesting CBraMod is consistently better")
    elif diff_ci[1] < 0:
        print("Note: 95% CI is entirely negative, suggesting EEGNet is consistently better")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Perform statistical comparison between EEGNet and CBraMod'
    )
    parser.add_argument(
        '--results', type=str, required=True,
        help='Path to comparison results JSON file'
    )
    parser.add_argument(
        '--paradigm', type=str, default='imagery',
        choices=['imagery', 'movement'],
        help='Experiment paradigm (default: imagery)'
    )
    parser.add_argument(
        '--task', type=str, default='binary',
        choices=['binary', 'ternary', 'quaternary'],
        help='Classification task (default: binary)'
    )

    args = parser.parse_args()

    # Load results
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)

    print(f"Loading results from: {results_path}")
    results = load_comparison_results(str(results_path))

    # Check data
    eegnet_count = len(results.get('eegnet', []))
    cbramod_count = len(results.get('cbramod', []))
    print(f"Loaded: EEGNet ({eegnet_count} subjects), CBraMod ({cbramod_count} subjects)")

    # Perform comparison
    comparison = None
    if eegnet_count >= 2 and cbramod_count >= 2:
        try:
            comparison = compare_models(results['eegnet'], results['cbramod'])
        except ValueError as e:
            print(f"Warning: Cannot compare models: {e}")

    # Print extended report
    print_extended_report(results, comparison, args.task, args.paradigm)

    return 0


if __name__ == '__main__':
    sys.exit(main())
