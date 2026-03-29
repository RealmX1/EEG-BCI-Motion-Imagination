#!/usr/bin/env python
"""
Analyze timing_breakdown.csv files to assess subject-level pipeline feasibility.

Computes data_loading_ratio across all within-subject runs, grouped by model
and task, and produces visualizations to inform the pipeline optimization
decision (subject-level data prefetch while GPU trains current subject).

Usage:
    uv run python scripts/analysis/analyze_timing_breakdown.py
    uv run python scripts/analysis/analyze_timing_breakdown.py --latest-only
    uv run python scripts/analysis/analyze_timing_breakdown.py --output results/timing_analysis
    uv run python scripts/analysis/analyze_timing_breakdown.py --no-plot
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Phases recorded by Timer in train_single_subject()
PHASE_COLUMNS = [
    'train_data_loading', 'test_data_loading', 'data_splitting',
    'dataloader_creation', 'model_creation', 'trainer_setup',
    'training', 'val_evaluation', 'test_evaluation',
]

# Visually prominent phases for the stacked bar chart (in stack order, bottom to top)
DISPLAY_PHASES = [
    'training', 'train_data_loading', 'test_data_loading',
    'val_evaluation', 'test_evaluation', 'other',
]

PHASE_COLORS = {
    'training':            '#95a5a6',  # gray (dominant, recessive)
    'train_data_loading':  '#E94F37',  # red (key metric, prominent)
    'test_data_loading':   '#F6AE2D',  # amber
    'val_evaluation':      '#2E86AB',  # blue
    'test_evaluation':     '#3498db',  # light blue
    'other':               '#dfe6e9',  # very light gray
}

PHASE_LABELS = {
    'training':            'Training',
    'train_data_loading':  'Train Data Loading',
    'test_data_loading':   'Test Data Loading',
    'val_evaluation':      'Val Evaluation',
    'test_evaluation':     'Test Evaluation',
    'other':               'Other (setup)',
}


# ── Data collection ─────────────────────────────────────────────────────────

def discover_timing_csvs(search_dirs: List[Path]) -> List[Path]:
    """Recursively find all timing_breakdown.csv files."""
    paths = []
    for d in search_dirs:
        if d.exists():
            paths.extend(d.rglob('timing_breakdown.csv'))
    return sorted(set(paths), key=lambda p: p.stat().st_mtime, reverse=True)


def load_all_timing_data(csv_paths: List[Path]) -> pd.DataFrame:
    """Load and concatenate all timing CSVs into a single DataFrame."""
    frames = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            if df.empty:
                continue
            # Provenance columns
            df['task_dir'] = path.parent.name       # e.g. "binary"
            df['source_dir'] = path.parent.parent.name  # e.g. "20260329_0431_cbramod_within_subject"
            df['source_path'] = str(path)
            frames.append(df)
        except Exception as e:
            warnings.warn(f"Skipping {path}: {e}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Ensure all phase columns exist and are numeric
    for col in PHASE_COLUMNS:
        if col not in combined.columns:
            combined[col] = 0.0
        combined[col] = pd.to_numeric(combined[col], errors='coerce').fillna(0.0)

    combined['total_time'] = pd.to_numeric(combined['total_time'], errors='coerce')

    return combined


# ── Filtering ───────────────────────────────────────────────────────────────

def filter_runs(
    df: pd.DataFrame,
    latest_only: bool = False,
    min_subjects: int = 5,
) -> pd.DataFrame:
    """Filter runs by subject count; optionally keep only largest per model/task."""
    # Count unique subjects per run
    run_counts = (
        df.groupby(['source_dir', 'model_type', 'task'])['subject_id']
        .nunique()
        .reset_index(name='n_subjects')
    )

    # Drop runs below min_subjects
    valid_runs = run_counts[run_counts['n_subjects'] >= min_subjects]

    if latest_only:
        # Keep only the run with most subjects per (model_type, task)
        idx = valid_runs.groupby(['model_type', 'task'])['n_subjects'].idxmax()
        valid_runs = valid_runs.loc[idx]

    keys = valid_runs[['source_dir', 'model_type', 'task']]
    filtered = df.merge(keys, on=['source_dir', 'model_type', 'task'], how='inner')

    # Deduplicate: keep first row per (source_dir, subject_id) to handle HPO multi-trial
    filtered = filtered.drop_duplicates(subset=['source_dir', 'subject_id'], keep='first')

    return filtered


# ── Computation ─────────────────────────────────────────────────────────────

def compute_phase_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add percentage columns for each phase and a combined loading ratio."""
    df = df.copy()
    for col in PHASE_COLUMNS:
        df[f'{col}_pct'] = df[col] / df['total_time'] * 100

    df['data_loading_ratio'] = (
        (df['train_data_loading'] + df['test_data_loading']) / df['total_time'] * 100
    )

    # "Other" = everything not in the main display phases
    main_phases = ['training', 'train_data_loading', 'test_data_loading',
                   'val_evaluation', 'test_evaluation']
    df['other_pct'] = 100.0 - sum(df[f'{p}_pct'] for p in main_phases)
    df['other_pct'] = df['other_pct'].clip(lower=0)

    return df


def compute_statistics(df: pd.DataFrame) -> Dict:
    """Compute grouped statistics on phase percentages."""
    ratio_cols = [f'{c}_pct' for c in PHASE_COLUMNS] + ['data_loading_ratio']

    def _agg(sub: pd.DataFrame) -> Dict:
        result = {}
        for col in ratio_cols:
            if col in sub.columns:
                vals = sub[col].dropna()
                result[col] = {
                    'mean': vals.mean(),
                    'std': vals.std(),
                    'min': vals.min(),
                    'max': vals.max(),
                    'n': len(vals),
                }
        return result

    stats = {
        'overall': _agg(df),
        'by_model': {},
        'by_task': {},
        'by_model_task': {},
    }

    for model, grp in df.groupby('model_type'):
        stats['by_model'][model] = _agg(grp)

    for task, grp in df.groupby('task'):
        stats['by_task'][task] = _agg(grp)

    for (model, task), grp in df.groupby(['model_type', 'task']):
        stats['by_model_task'][(model, task)] = _agg(grp)

    return stats


# ── Console output ──────────────────────────────────────────────────────────

def print_statistics_report(stats: Dict, df: pd.DataFrame) -> None:
    """Print formatted statistics to console."""
    overall = stats['overall']
    dlr = overall.get('data_loading_ratio', {})

    print()
    print("=" * 70)
    print("  Timing Breakdown Analysis -- Pipeline Feasibility Assessment")
    print("=" * 70)

    print(f"\n  Data sources: {df['source_dir'].nunique()} run(s), "
          f"{len(df)} subject-run records")

    # List runs
    for (model, task), grp in df.groupby(['model_type', 'task']):
        sources = grp['source_dir'].unique()
        for src in sources:
            n = grp[grp['source_dir'] == src]['subject_id'].nunique()
            print(f"    {model} / {task}: {src} ({n} subjects)")

    print(f"\n  Overall data_loading_ratio:")
    print(f"    Mean:  {dlr.get('mean', 0):.2f}% +/- {dlr.get('std', 0):.2f}%")
    print(f"    Range: [{dlr.get('min', 0):.2f}%, {dlr.get('max', 0):.2f}%]")
    print(f"    N:     {dlr.get('n', 0)} subject-runs")

    # By model
    print(f"\n  By Model:")
    for model, mstats in sorted(stats['by_model'].items()):
        mdlr = mstats.get('data_loading_ratio', {})
        mtrain = mstats.get('train_data_loading_pct', {})
        print(f"    {model:>8s}:  train_load={mtrain.get('mean', 0):.1f}% +/- {mtrain.get('std', 0):.1f}%"
              f"  total_load={mdlr.get('mean', 0):.1f}% +/- {mdlr.get('std', 0):.1f}%"
              f"  (n={mdlr.get('n', 0)})")

    # By task
    print(f"\n  By Task:")
    for task, tstats in sorted(stats['by_task'].items()):
        tdlr = tstats.get('data_loading_ratio', {})
        print(f"    {task:>8s}:  total_load={tdlr.get('mean', 0):.1f}% +/- {tdlr.get('std', 0):.1f}%"
              f"  (n={tdlr.get('n', 0)})")

    # Full phase breakdown table
    print()
    print("-" * 70)
    print("  Phase Breakdown (mean % of total_time)")
    print("-" * 70)

    models = sorted(stats['by_model'].keys())
    header = f"  {'Phase':<25s} {'Overall':>8s}"
    for m in models:
        header += f"  {m:>8s}"
    print(header)
    print("  " + "-" * (25 + 10 + 10 * len(models)))

    for phase in PHASE_COLUMNS:
        col = f'{phase}_pct'
        overall_val = stats['overall'].get(col, {}).get('mean', 0)
        row = f"  {phase:<25s} {overall_val:>7.2f}%"
        for m in models:
            val = stats['by_model'][m].get(col, {}).get('mean', 0)
            row += f"  {val:>7.2f}%"
        print(row)

    print("=" * 70)


def generate_pipeline_recommendation(stats: Dict) -> Tuple[str, str]:
    """Produce recommendation based on 5%/10% thresholds."""
    overall_mean = stats['overall'].get('data_loading_ratio', {}).get('mean', 0)

    if overall_mean > 10:
        recommendation = "RECOMMENDED"
        detail = (
            f"Data loading accounts for {overall_mean:.1f}% of total time on average. "
            f"Subject-level pipelining (prefetching next subject data during current "
            f"subject GPU training) would yield measurable speedup."
        )
    elif overall_mean > 5:
        recommendation = "CONDITIONALLY RECOMMENDED"
        detail = (
            f"Data loading accounts for {overall_mean:.1f}% of total time on average. "
            f"Pipelining benefit is moderate."
        )
        eegnet_mean = (
            stats['by_model'].get('eegnet', {})
            .get('data_loading_ratio', {})
            .get('mean', 0)
        )
        if eegnet_mean > 10:
            detail += (
                f"\n    Note: EEGNet loading ratio is {eegnet_mean:.1f}%, "
                f"exceeding the 10% threshold individually."
            )
    else:
        recommendation = "NOT RECOMMENDED"
        detail = (
            f"Data loading accounts for only {overall_mean:.1f}% of total time. "
            f"Pipeline optimization ROI is low. Focus on training speed instead."
        )

    return recommendation, detail


# ── Visualization ───────────────────────────────────────────────────────────

def plot_stacked_bar(df: pd.DataFrame, output_path: Path) -> None:
    """Stacked bar chart of phase proportions grouped by model/task."""
    groups = df.groupby(['model_type', 'task'])

    labels = []
    phase_means = {p: [] for p in DISPLAY_PHASES}

    for (model, task), grp in sorted(groups):
        labels.append(f"{model}\n{task}")
        for phase in DISPLAY_PHASES:
            if phase == 'other':
                col = 'other_pct'
            else:
                col = f'{phase}_pct'
            phase_means[phase].append(grp[col].mean())

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 6))

    bottom = np.zeros(len(labels))
    bars_dict = {}
    for phase in DISPLAY_PHASES:
        vals = np.array(phase_means[phase])
        bars = ax.bar(
            x, vals, bottom=bottom,
            label=PHASE_LABELS[phase],
            color=PHASE_COLORS[phase],
            edgecolor='white', linewidth=0.5,
        )
        bars_dict[phase] = bars
        bottom += vals

    # Annotate train_data_loading percentage
    train_load_bars = bars_dict['train_data_loading']
    training_vals = np.array(phase_means['training'])
    for i, (bar, val) in enumerate(zip(train_load_bars, phase_means['train_data_loading'])):
        if val > 1:  # Only annotate if visible
            y = training_vals[i] + val / 2
            ax.text(bar.get_x() + bar.get_width() / 2, y,
                    f'{val:.1f}%', ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white')

    ax.set_ylabel('Percentage of Total Time (%)')
    ax.set_title('Training Phase Breakdown by Model / Task')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Stacked bar saved to: {output_path}")
    plt.close(fig)


def plot_loading_ratio_boxplot(df: pd.DataFrame, output_path: Path) -> None:
    """Boxplot of data_loading_ratio distribution by model."""
    models = sorted(df['model_type'].unique())
    model_colors = {'eegnet': '#2E86AB', 'cbramod': '#E94F37'}

    fig, ax = plt.subplots(figsize=(8, 5))

    data_groups = [df[df['model_type'] == m]['data_loading_ratio'].values for m in models]
    bp = ax.boxplot(
        data_groups,
        tick_labels=[m.upper() for m in models],
        patch_artist=True,
        widths=0.5,
    )

    for patch, model in zip(bp['boxes'], models):
        patch.set_facecolor(model_colors.get(model, '#95a5a6'))
        patch.set_alpha(0.7)

    # Overlay scatter with jitter
    for i, (model, vals) in enumerate(zip(models, data_groups)):
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            np.full(len(vals), i + 1) + jitter, vals,
            color=model_colors.get(model, '#95a5a6'),
            alpha=0.5, s=20, zorder=3,
        )

    # Threshold lines
    ax.axhline(y=5, color='green', linestyle='--', alpha=0.7,
               label='5% (low ROI)')
    ax.axhline(y=10, color='red', linestyle='--', alpha=0.7,
               label='10% (recommended)')

    ax.set_ylabel('Data Loading Ratio (%)')
    ax.set_title('Data Loading as % of Total Training Time')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Boxplot saved to: {output_path}")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Analyze timing breakdowns to assess subject-level pipeline feasibility'
    )
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for plots (default: results/)')
    parser.add_argument('--latest-only', action='store_true',
                        help='Keep only the best (most subjects) run per model/task')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plot generation')
    parser.add_argument('--min-subjects', type=int, default=5,
                        help='Minimum subjects per run to include (default: 5)')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    search_dirs = [project_root / 'results', project_root / 'checkpoints']

    # 1. Discover
    csv_paths = discover_timing_csvs(search_dirs)
    print(f"Found {len(csv_paths)} timing_breakdown.csv file(s)")

    if not csv_paths:
        print("No timing data found. Run a within-subject experiment first.")
        return

    # 2. Load
    df = load_all_timing_data(csv_paths)
    print(f"Loaded {len(df)} subject-run records from {df['source_dir'].nunique()} runs")

    # 3. Filter
    df = filter_runs(df, latest_only=args.latest_only, min_subjects=args.min_subjects)
    if df.empty:
        print(f"No runs with >= {args.min_subjects} subjects found.")
        return
    print(f"After filtering: {len(df)} records from {df['source_dir'].nunique()} run(s)")

    # 4. Compute
    df = compute_phase_ratios(df)
    stats = compute_statistics(df)

    # 5. Report
    print_statistics_report(stats, df)

    # 6. Recommendation
    recommendation, detail = generate_pipeline_recommendation(stats)
    print(f"\n  Pipeline Recommendation: {recommendation}")
    print(f"  {detail}")
    print()

    # 7. Plots
    if not args.no_plot:
        output_dir = Path(args.output) if args.output else project_root / 'results'
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_stacked_bar(df, output_dir / 'timing_phase_breakdown.png')
        plot_loading_ratio_boxplot(df, output_dir / 'timing_loading_ratio_boxplot.png')


if __name__ == '__main__':
    main()
