#!/usr/bin/env python
"""
Visualize training time analysis for preprocessing ML engineering experiments.

Usage:
    uv run python scripts/visualize_preproc_timing.py
    uv run python scripts/visualize_preproc_timing.py --json results/20260126_0934_preproc_ml_eng_results.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


# Experiment descriptions
EXPERIMENT_DESCRIPTIONS = {
    'A1': 'Baseline (0.3-75Hz, 60Hz notch)',
    'A6': 'No notch filter',
    'C2': 'Extra z-score normalization',
    'D2': '250ms step (75% overlap)',
    'D3': '500ms step (50% overlap)',
    'F2': 'No amplitude rejection',
}


def load_results(json_path: Path) -> Dict:
    """Load experiment results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def analyze_timing(data: Dict) -> Tuple[Dict, Dict]:
    """
    Analyze training times from experiment data.

    Returns:
        experiment_times: Dict mapping exp_key to list of (subject, time, epochs)
        summary: Dict with aggregated statistics
    """
    experiments = data.get('experiments', {})

    experiment_times = defaultdict(list)

    for exp_key, exp_data in experiments.items():
        # Handle both formats: list of subjects or dict with 'subjects' key
        if isinstance(exp_data, dict) and 'subjects' in exp_data:
            subjects = exp_data['subjects']
        elif isinstance(exp_data, dict):
            # Cache format: dict of subject_id -> result
            subjects = list(exp_data.values())
        else:
            continue

        for subj in subjects:
            if isinstance(subj, dict) and 'training_time' in subj:
                experiment_times[exp_key].append({
                    'subject': subj.get('subject_id', 'unknown'),
                    'time': subj['training_time'],
                    'epochs': subj.get('epochs_trained', 0),
                })

    # Compute summary statistics
    summary = {}
    for exp_key, times in experiment_times.items():
        if times:
            time_values = [t['time'] for t in times]
            epoch_values = [t['epochs'] for t in times]
            summary[exp_key] = {
                'total_time': sum(time_values),
                'mean_time': np.mean(time_values),
                'std_time': np.std(time_values),
                'min_time': min(time_values),
                'max_time': max(time_values),
                'n_subjects': len(times),
                'mean_epochs': np.mean(epoch_values),
                'total_epochs': sum(epoch_values),
            }

    return dict(experiment_times), summary


def format_time(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def print_timing_table(summary: Dict) -> None:
    """Print timing summary as a formatted table."""
    print("\n" + "=" * 90)
    print("Training Time Analysis - Preprocessing ML Engineering Experiments")
    print("=" * 90)

    # Group by experiment ID (A1, A6, C2, etc.)
    exp_groups = defaultdict(dict)
    for exp_key, stats in summary.items():
        parts = exp_key.rsplit('_', 1)
        if len(parts) == 2:
            exp_id, task = parts
            exp_groups[exp_id][task] = stats

    # Sort by experiment ID
    sorted_exps = sorted(exp_groups.keys(), key=lambda x: (x[0], int(x[1:]) if x[1:].isdigit() else 0))

    # Print header
    print(f"\n{'Experiment':<12} {'Description':<30} {'Task':<8} {'Total':<10} {'Mean±Std':<16} {'N':>3} {'Epochs':>6}")
    print("-" * 90)

    grand_total = 0

    for exp_id in sorted_exps:
        desc = EXPERIMENT_DESCRIPTIONS.get(exp_id, '')[:28]
        tasks = exp_groups[exp_id]

        for i, (task, stats) in enumerate(sorted(tasks.items())):
            exp_label = exp_id if i == 0 else ''
            desc_label = desc if i == 0 else ''

            total = format_time(stats['total_time'])
            mean_std = f"{stats['mean_time']:.0f}s ± {stats['std_time']:.0f}s"
            n = stats['n_subjects']
            epochs = f"{stats['mean_epochs']:.1f}"

            print(f"{exp_label:<12} {desc_label:<30} {task:<8} {total:<10} {mean_std:<16} {n:>3} {epochs:>6}")
            grand_total += stats['total_time']

    print("-" * 90)
    print(f"{'TOTAL':<12} {'':<30} {'':<8} {format_time(grand_total):<10}")
    print("=" * 90)

    # Per-configuration summary
    print("\n" + "=" * 70)
    print("Per-Configuration Summary (Binary + Ternary combined)")
    print("=" * 70)
    print(f"\n{'Experiment':<12} {'Description':<35} {'Total Time':<12} {'Speedup':<10}")
    print("-" * 70)

    config_totals = {}
    for exp_id in sorted_exps:
        tasks = exp_groups[exp_id]
        total = sum(stats['total_time'] for stats in tasks.values())
        config_totals[exp_id] = total

    baseline_time = config_totals.get('A1', 1)

    for exp_id in sorted_exps:
        desc = EXPERIMENT_DESCRIPTIONS.get(exp_id, '')[:33]
        total = config_totals[exp_id]
        speedup = baseline_time / total if total > 0 else 0
        speedup_str = f"{speedup:.2f}x" if exp_id != 'A1' else '(baseline)'
        print(f"{exp_id:<12} {desc:<35} {format_time(total):<12} {speedup_str:<10}")

    print("=" * 70)


def plot_timing(summary: Dict, output_path: Path = None) -> None:
    """Create visualization of training times."""
    # Prepare data
    exp_groups = defaultdict(dict)
    for exp_key, stats in summary.items():
        parts = exp_key.rsplit('_', 1)
        if len(parts) == 2:
            exp_id, task = parts
            exp_groups[exp_id][task] = stats

    sorted_exps = sorted(exp_groups.keys(), key=lambda x: (x[0], int(x[1:]) if x[1:].isdigit() else 0))

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Colors
    colors = {'binary': '#2ecc71', 'ternary': '#3498db'}

    # Plot 1: Total time per experiment
    ax1 = axes[0]
    x = np.arange(len(sorted_exps))
    width = 0.35

    binary_times = []
    ternary_times = []

    for exp_id in sorted_exps:
        tasks = exp_groups[exp_id]
        binary_times.append(tasks.get('binary', {}).get('total_time', 0) / 60)  # Convert to minutes
        ternary_times.append(tasks.get('ternary', {}).get('total_time', 0) / 60)

    bars1 = ax1.bar(x - width/2, binary_times, width, label='Binary', color=colors['binary'], alpha=0.8)
    bars2 = ax1.bar(x + width/2, ternary_times, width, label='Ternary', color=colors['ternary'], alpha=0.8)

    ax1.set_xlabel('Experiment Configuration')
    ax1.set_ylabel('Total Training Time (minutes)')
    ax1.set_title('Total Training Time per Experiment')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sorted_exps)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.annotate(f'{height:.0f}m',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

    # Plot 2: Mean time per subject with error bars
    ax2 = axes[1]

    binary_means = []
    binary_stds = []
    ternary_means = []
    ternary_stds = []

    for exp_id in sorted_exps:
        tasks = exp_groups[exp_id]
        binary_means.append(tasks.get('binary', {}).get('mean_time', 0))
        binary_stds.append(tasks.get('binary', {}).get('std_time', 0))
        ternary_means.append(tasks.get('ternary', {}).get('mean_time', 0))
        ternary_stds.append(tasks.get('ternary', {}).get('std_time', 0))

    ax2.bar(x - width/2, binary_means, width, yerr=binary_stds,
            label='Binary', color=colors['binary'], alpha=0.8, capsize=3)
    ax2.bar(x + width/2, ternary_means, width, yerr=ternary_stds,
            label='Ternary', color=colors['ternary'], alpha=0.8, capsize=3)

    ax2.set_xlabel('Experiment Configuration')
    ax2.set_ylabel('Mean Training Time per Subject (seconds)')
    ax2.set_title('Mean Training Time per Subject (with std)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sorted_exps)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")

    plt.show()


def plot_speedup(summary: Dict, output_path: Path = None) -> None:
    """Create speedup comparison plot."""
    # Calculate per-config totals
    exp_groups = defaultdict(float)
    for exp_key, stats in summary.items():
        parts = exp_key.rsplit('_', 1)
        if len(parts) == 2:
            exp_id, task = parts
            exp_groups[exp_id] += stats['total_time']

    sorted_exps = sorted(exp_groups.keys(), key=lambda x: (x[0], int(x[1:]) if x[1:].isdigit() else 0))

    baseline_time = exp_groups.get('A1', 1)

    # Calculate speedups
    speedups = [baseline_time / exp_groups[exp] for exp in sorted_exps]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ['#e74c3c' if s < 1 else '#2ecc71' if s > 1.1 else '#95a5a6' for s in speedups]

    bars = ax.barh(sorted_exps, speedups, color=colors, alpha=0.8)
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1, label='Baseline (A1)')

    ax.set_xlabel('Speedup vs Baseline (A1)')
    ax.set_ylabel('Experiment Configuration')
    ax.set_title('Training Time Speedup Comparison')
    ax.grid(axis='x', alpha=0.3)

    # Add labels with descriptions
    for i, (exp, speedup) in enumerate(zip(sorted_exps, speedups)):
        desc = EXPERIMENT_DESCRIPTIONS.get(exp, '')
        time_total = exp_groups[exp]
        label = f'{speedup:.2f}x ({format_time(time_total)})'
        ax.annotate(label, xy=(speedup, i), xytext=(5, 0),
                   textcoords="offset points", ha='left', va='center', fontsize=9)

    # Add descriptions on the left
    ax.set_yticklabels([f"{exp}: {EXPERIMENT_DESCRIPTIONS.get(exp, '')[:25]}" for exp in sorted_exps])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Speedup plot saved to: {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze preprocessing experiment training times')
    parser.add_argument('--json', type=str, help='Path to results JSON file')
    parser.add_argument('--cache', type=str, help='Path to cache JSON file')
    parser.add_argument('--output', type=str, help='Output directory for plots')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    args = parser.parse_args()

    # Find input file
    results_dir = Path(__file__).parent.parent / 'results'

    if args.json:
        json_path = Path(args.json)
    elif args.cache:
        json_path = Path(args.cache)
    else:
        # Find most recent results file
        pattern = '*preproc_ml_eng*.json'
        json_files = sorted(results_dir.glob(pattern), reverse=True)
        if not json_files:
            print(f"No results files found matching {pattern} in {results_dir}")
            return
        json_path = json_files[0]

    print(f"Loading: {json_path}")
    data = load_results(json_path)

    # Analyze timing
    experiment_times, summary = analyze_timing(data)

    if not summary:
        print("No timing data found in the results file.")
        return

    # Print table
    print_timing_table(summary)

    # Create plots
    if not args.no_plot:
        output_dir = Path(args.output) if args.output else results_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_timing(summary, output_dir / 'preproc_timing_analysis.png')
        plot_speedup(summary, output_dir / 'preproc_timing_speedup.png')


if __name__ == '__main__':
    main()
