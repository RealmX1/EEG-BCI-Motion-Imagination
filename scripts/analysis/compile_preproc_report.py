#!/usr/bin/env python
"""
Compile CBraMod Preprocessing ML Engineering Experiment Report.

This script reads experiment results and generates:
1. JSON summary with statistical analysis
2. Markdown report with tables and figures
3. Visualization figures (bar charts, heatmaps, box plots)

Statistical Analysis:
- Paired t-test vs baseline
- Wilcoxon signed-rank test (non-parametric)
- Cohen's d effect size
- Bonferroni correction for multiple comparisons

Usage:
    # Generate report from latest results
    uv run python scripts/compile_preproc_report.py

    # Specify results file
    uv run python scripts/compile_preproc_report.py --results results/20260125_1430_preproc_ml_eng_results.json

    # Generate PDF (requires pandoc)
    uv run python scripts/compile_preproc_report.py --pdf
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import SectionLogger, setup_logging
from src.config.experiment_config import (
    ALL_EXPERIMENTS,
    EXPERIMENT_GROUPS,
    is_baseline,
    get_experiment_config,
)


setup_logging('report')
logger = logging.getLogger(__name__)
log_main = SectionLogger(logger, 'main')
log_stats = SectionLogger(logger, 'stats')
log_io = SectionLogger(logger, 'io')


# ============================================================================
# Statistical Analysis Functions
# ============================================================================

def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size.

    Cohen's d = (M1 - M2) / pooled_std

    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def effect_size_interpretation(d: float) -> str:
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


def run_statistical_tests(
    experiment_accs: np.ndarray,
    baseline_accs: np.ndarray,
    alpha: float = 0.05,
) -> Dict:
    """
    Run statistical tests comparing experiment to baseline.

    Args:
        experiment_accs: Accuracies from experiment
        baseline_accs: Accuracies from baseline (same subjects)
        alpha: Significance level

    Returns:
        Dict with test results
    """
    from scipy import stats

    results = {
        'n_subjects': len(experiment_accs),
        'exp_mean': float(np.mean(experiment_accs)),
        'exp_std': float(np.std(experiment_accs)),
        'baseline_mean': float(np.mean(baseline_accs)),
        'baseline_std': float(np.std(baseline_accs)),
        'difference_mean': float(np.mean(experiment_accs) - np.mean(baseline_accs)),
    }

    # Paired t-test
    if len(experiment_accs) >= 2:
        t_stat, t_p = stats.ttest_rel(experiment_accs, baseline_accs)
        results['t_test_t'] = float(t_stat)
        results['t_test_p'] = float(t_p)
    else:
        results['t_test_t'] = None
        results['t_test_p'] = None

    # Wilcoxon signed-rank test (non-parametric)
    if len(experiment_accs) >= 5:  # Wilcoxon needs at least 5 samples
        try:
            w_stat, w_p = stats.wilcoxon(experiment_accs, baseline_accs)
            results['wilcoxon_stat'] = float(w_stat)
            results['wilcoxon_p'] = float(w_p)
        except ValueError:
            # All differences are zero
            results['wilcoxon_stat'] = None
            results['wilcoxon_p'] = 1.0
    else:
        results['wilcoxon_stat'] = None
        results['wilcoxon_p'] = None

    # Cohen's d effect size
    results['cohens_d'] = compute_cohens_d(experiment_accs, baseline_accs)
    results['effect_size'] = effect_size_interpretation(results['cohens_d'])

    # Significance (with Bonferroni option)
    results['significant_t'] = results['t_test_p'] < alpha if results['t_test_p'] else False
    results['significant_w'] = results['wilcoxon_p'] < alpha if results['wilcoxon_p'] else False

    return results


def apply_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[List[bool], float]:
    """
    Apply Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of p-values
        alpha: Original significance level

    Returns:
        Tuple of (list of significant flags, corrected alpha)
    """
    n_comparisons = len(p_values)
    corrected_alpha = alpha / n_comparisons

    significant = [p < corrected_alpha if p is not None else False for p in p_values]

    return significant, corrected_alpha


# ============================================================================
# Data Loading
# ============================================================================

def find_latest_results(output_dir: str) -> Optional[Path]:
    """Find the latest experiment results file."""
    results_dir = Path(output_dir)
    if not results_dir.exists():
        return None

    pattern = '*preproc_ml_eng_results.json'
    files = list(results_dir.glob(pattern))

    if not files:
        return None

    # Sort by modification time
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def load_results(results_path: Path) -> Dict:
    """Load experiment results from JSON."""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_accuracies(results: Dict, config_key: str) -> Tuple[List[str], np.ndarray]:
    """
    Extract subject IDs and accuracies from results.

    Args:
        results: Results dict
        config_key: Key like "A2_binary"

    Returns:
        Tuple of (subject_ids, accuracies)
    """
    exp_results = results.get('experiments', {}).get(config_key, {})
    subjects = exp_results.get('subjects', [])

    subject_ids = [s['subject_id'] for s in subjects]
    accs = np.array([s['test_acc_majority'] for s in subjects])

    return subject_ids, accs


# ============================================================================
# Report Generation
# ============================================================================

def generate_summary_table(
    results: Dict,
    tasks: List[str] = ['binary', 'ternary'],
) -> str:
    """Generate Markdown summary table."""
    lines = []
    lines.append("## Experiment Summary")
    lines.append("")

    for task in tasks:
        lines.append(f"### {task.capitalize()} Classification")
        lines.append("")

        # Table header
        lines.append("| Experiment | Group | Description | Mean Acc | Std | vs Baseline | p-value | Effect |")
        lines.append("|------------|-------|-------------|----------|-----|-------------|---------|--------|")

        # Get baseline
        baseline_key = f"A1_{task}"
        baseline_data = results.get('experiments', {}).get(baseline_key, {})
        baseline_stats = baseline_data.get('statistics', {})

        if baseline_stats:
            lines.append(
                f"| A1 | A | Baseline | {baseline_stats['mean']:.1%} | {baseline_stats['std']:.1%} | - | - | - |"
            )

        # Other experiments
        for exp_id in ['A2', 'A3', 'A4', 'A5', 'A6', 'C2', 'C3', 'C4', 'D2', 'D3', 'F2']:
            config_key = f"{exp_id}_{task}"
            exp_data = results.get('experiments', {}).get(config_key, {})

            if not exp_data:
                continue

            stats = exp_data.get('statistics', {})
            exp_config = get_experiment_config(exp_id)

            mean_acc = stats.get('mean', 0)
            std_acc = stats.get('std', 0)

            # Calculate difference from baseline
            diff = mean_acc - baseline_stats.get('mean', 0) if baseline_stats else 0
            diff_str = f"{diff:+.1%}" if diff != 0 else "-"

            # Get statistical test results if available
            stat_results = exp_data.get('stat_tests', {})
            p_value = stat_results.get('t_test_p')
            p_str = f"{p_value:.3f}" if p_value is not None else "-"
            effect = stat_results.get('effect_size', '-')

            lines.append(
                f"| {exp_id} | {exp_config.experiment_group} | "
                f"{exp_config.description[:40]}... | {mean_acc:.1%} | {std_acc:.1%} | "
                f"{diff_str} | {p_str} | {effect} |"
            )

        lines.append("")

    return "\n".join(lines)


def generate_recommendations(results: Dict, task: str = 'binary') -> str:
    """Generate recommendations based on results."""
    lines = []
    lines.append("## Recommendations")
    lines.append("")

    # Find best performing configuration
    experiments = results.get('experiments', {})
    best_exp = None
    best_acc = 0

    for config_key, exp_data in experiments.items():
        if not config_key.endswith(f"_{task}"):
            continue

        stats = exp_data.get('statistics', {})
        mean_acc = stats.get('mean', 0)

        if mean_acc > best_acc:
            best_acc = mean_acc
            best_exp = config_key.replace(f"_{task}", "")

    if best_exp:
        exp_config = get_experiment_config(best_exp)
        lines.append(f"### Best Configuration: {best_exp}")
        lines.append(f"- **Description**: {exp_config.description}")
        lines.append(f"- **Mean Accuracy**: {best_acc:.1%}")
        lines.append("")

        # Configuration details
        lines.append("**Preprocessing Parameters:**")
        lines.append(f"- Bandpass: {exp_config.bandpass_low}-{exp_config.bandpass_high} Hz")
        lines.append(f"- Notch: {exp_config.notch_freq} Hz" if exp_config.notch_freq else "- Notch: None")
        lines.append(f"- Extra normalization: {exp_config.extra_normalize or 'None'}")
        lines.append(f"- Sliding step: {exp_config.segment_step_ms} ms")
        lines.append(f"- Amplitude threshold: {exp_config.amplitude_threshold} uV" if exp_config.amplitude_threshold else "- Amplitude threshold: None")

    return "\n".join(lines)


def generate_markdown_report(
    results: Dict,
    output_path: Path,
    tasks: List[str] = ['binary', 'ternary'],
) -> None:
    """Generate full Markdown report."""
    lines = []

    # Header
    lines.append("# CBraMod Data Preprocessing ML Engineering Report")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # Metadata
    metadata = results.get('metadata', {})
    lines.append("## Experiment Metadata")
    lines.append("")
    lines.append(f"- **Paradigm**: {metadata.get('paradigm', 'unknown')}")
    lines.append(f"- **N Experiments**: {metadata.get('n_experiments', 0)}")
    lines.append(f"- **Timestamp**: {metadata.get('timestamp', 'unknown')}")
    lines.append("")

    # Summary table
    lines.append(generate_summary_table(results, tasks))

    # Per-group analysis
    lines.append("## Group Analysis")
    lines.append("")

    for group, exp_ids in EXPERIMENT_GROUPS.items():
        group_names = {
            'A': 'Filtering Parameters',
            'C': 'Normalization Strategies',
            'D': 'Window Sliding Strategy',
            'F': 'Data Quality Control',
        }

        lines.append(f"### Group {group}: {group_names.get(group, '')}")
        lines.append("")

        for task in tasks:
            lines.append(f"**{task.capitalize()}:**")

            for exp_id in exp_ids:
                config_key = f"{exp_id}_{task}"
                exp_data = results.get('experiments', {}).get(config_key, {})

                if not exp_data:
                    continue

                stats = exp_data.get('statistics', {})
                exp_config = get_experiment_config(exp_id)

                lines.append(f"- {exp_id}: {stats.get('mean', 0):.1%} +/- {stats.get('std', 0):.1%} ({exp_config.description})")

            lines.append("")

    # Recommendations
    lines.append(generate_recommendations(results))

    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    log_io.info(f"报告已保存: {output_path}")


# ============================================================================
# Visualization
# ============================================================================

def generate_comparison_plot(
    results: Dict,
    output_path: Path,
    task: str = 'binary',
) -> None:
    """Generate comparison bar chart."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        log_io.warning("未安装 matplotlib，跳过绑图")
        return

    experiments = results.get('experiments', {})

    # Collect data
    exp_ids = []
    means = []
    stds = []
    groups = []

    group_colors = {
        'A': '#2E86AB',  # Blue
        'C': '#E94F37',  # Red
        'D': '#45B69C',  # Green
        'F': '#9B59B6',  # Purple
    }

    for exp_id in ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'C2', 'C3', 'C4', 'D2', 'D3', 'F2']:
        config_key = f"{exp_id}_{task}"
        exp_data = experiments.get(config_key, {})

        if not exp_data:
            continue

        stats = exp_data.get('statistics', {})
        exp_ids.append(exp_id)
        means.append(stats.get('mean', 0) * 100)
        stds.append(stats.get('std', 0) * 100)
        groups.append(exp_id[0])

    if not exp_ids:
        log_io.warning("无实验数据可绑图")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(exp_ids))
    colors = [group_colors.get(g, '#666666') for g in groups]

    bars = ax.bar(x, means, yerr=stds, color=colors, edgecolor='black',
                  linewidth=0.5, alpha=0.8, capsize=3)

    # Baseline line
    baseline_mean = means[0] if means else 0
    ax.axhline(y=baseline_mean, color='red', linestyle='--', linewidth=2,
               label=f'Baseline ({baseline_mean:.1f}%)')

    # Chance level
    chance = 50 if task == 'binary' else 33.3
    ax.axhline(y=chance, color='gray', linestyle=':', alpha=0.5,
               label=f'Chance ({chance:.1f}%)')

    # Labels
    ax.set_xlabel('Experiment Configuration')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title(f'CBraMod Preprocessing Experiments - {task.capitalize()} Classification')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_ids, rotation=45, ha='right')

    # Legend
    legend_patches = [
        mpatches.Patch(color=group_colors['A'], label='A: Filtering'),
        mpatches.Patch(color=group_colors['C'], label='C: Normalization'),
        mpatches.Patch(color=group_colors['D'], label='D: Window'),
        mpatches.Patch(color=group_colors['F'], label='F: Quality'),
    ]
    ax.legend(handles=legend_patches, loc='lower right')

    ax.set_ylim([0, 105])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    log_io.info(f"图表已保存: {output_path}")


def generate_heatmap(
    results: Dict,
    output_path: Path,
    task: str = 'binary',
) -> None:
    """Generate experiment x subject accuracy heatmap."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        log_io.warning("未安装 matplotlib/seaborn，跳过热力图")
        return

    experiments = results.get('experiments', {})

    # Collect all subjects
    all_subjects = set()
    for config_key, exp_data in experiments.items():
        if not config_key.endswith(f"_{task}"):
            continue
        for subject in exp_data.get('subjects', []):
            all_subjects.add(subject['subject_id'])

    subjects = sorted(all_subjects)
    exp_ids = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'C2', 'C3', 'C4', 'D2', 'D3', 'F2']

    # Build matrix
    matrix = np.zeros((len(exp_ids), len(subjects)))

    for i, exp_id in enumerate(exp_ids):
        config_key = f"{exp_id}_{task}"
        exp_data = experiments.get(config_key, {})

        subject_accs = {s['subject_id']: s['test_acc_majority'] for s in exp_data.get('subjects', [])}

        for j, subj in enumerate(subjects):
            matrix[i, j] = subject_accs.get(subj, np.nan) * 100

    if np.all(np.isnan(matrix)):
        log_io.warning("无数据可生成热力图")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        matrix, ax=ax,
        xticklabels=subjects,
        yticklabels=exp_ids,
        annot=True, fmt='.1f',
        cmap='RdYlGn',
        vmin=40, vmax=100,
        cbar_kws={'label': 'Accuracy (%)'},
    )

    ax.set_xlabel('Subject')
    ax.set_ylabel('Experiment')
    ax.set_title(f'Experiment x Subject Accuracy - {task.capitalize()}')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    log_io.info(f"热力图已保存: {output_path}")


def generate_boxplot(
    results: Dict,
    output_path: Path,
    task: str = 'binary',
) -> None:
    """Generate box plot comparing groups."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log_io.warning("未安装 matplotlib，跳过箱线图")
        return

    experiments = results.get('experiments', {})

    # Collect data by group
    group_data = {}

    for config_key, exp_data in experiments.items():
        if not config_key.endswith(f"_{task}"):
            continue

        exp_id = config_key.replace(f"_{task}", "")
        group = exp_id[0]

        if group not in group_data:
            group_data[group] = []

        accs = [s['test_acc_majority'] * 100 for s in exp_data.get('subjects', [])]
        group_data[group].extend(accs)

    if not group_data:
        log_io.warning("无数据可生成箱线图")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    groups = ['A', 'C', 'D', 'F']
    data = [group_data.get(g, []) for g in groups]
    group_labels = [
        'A: Filtering',
        'C: Normalization',
        'D: Window',
        'F: Quality',
    ]

    group_colors = ['#2E86AB', '#E94F37', '#45B69C', '#9B59B6']

    bp = ax.boxplot(data, tick_labels=group_labels, patch_artist=True,
                    showmeans=True, meanline=True)

    for patch, color in zip(bp['boxes'], group_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Chance level
    chance = 50 if task == 'binary' else 33.3
    ax.axhline(y=chance, color='gray', linestyle=':', alpha=0.5,
               label=f'Chance ({chance:.1f}%)')

    ax.set_xlabel('Experiment Group')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title(f'Accuracy Distribution by Experiment Group - {task.capitalize()}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    log_io.info(f"箱线图已保存: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compile CBraMod Preprocessing ML Engineering Report',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--results', type=str, default=None,
        help='Path to results JSON file (default: find latest)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='results',
        help='Output directory for reports (default: results)'
    )
    parser.add_argument(
        '--tasks', nargs='+', default=['binary', 'ternary'],
        choices=['binary', 'ternary'],
        help='Tasks to include in report (default: binary ternary)'
    )
    parser.add_argument(
        '--no-plots', action='store_true',
        help='Skip plot generation'
    )
    parser.add_argument(
        '--pdf', action='store_true',
        help='Generate PDF report (requires pandoc)'
    )

    args = parser.parse_args()

    # Find results file
    if args.results:
        results_path = Path(args.results)
    else:
        results_path = find_latest_results(args.output_dir)

    if not results_path or not results_path.exists():
        log_main.error("未找到结果文件")
        return 1

    log_main.info(f"从以下位置加载结果: {results_path}")

    # Load results
    results = load_results(results_path)

    # Add statistical tests
    log_stats.info("正在运行统计检验...")

    # Get baseline accuracies for comparison
    experiments = results.get('experiments', {})

    for task in args.tasks:
        baseline_key = f"A1_{task}"
        baseline_data = experiments.get(baseline_key, {})
        baseline_subjects = baseline_data.get('subjects', [])

        if not baseline_subjects:
            continue

        baseline_accs = np.array([s['test_acc_majority'] for s in baseline_subjects])
        baseline_subject_map = {s['subject_id']: s['test_acc_majority'] for s in baseline_subjects}

        # Compare each experiment to baseline
        for config_key, exp_data in experiments.items():
            if not config_key.endswith(f"_{task}"):
                continue

            if config_key == baseline_key:
                continue

            exp_subjects = exp_data.get('subjects', [])
            if not exp_subjects:
                continue

            # Match subjects
            matched_exp = []
            matched_base = []

            for s in exp_subjects:
                subj_id = s['subject_id']
                if subj_id in baseline_subject_map:
                    matched_exp.append(s['test_acc_majority'])
                    matched_base.append(baseline_subject_map[subj_id])

            if len(matched_exp) >= 2:
                stat_results = run_statistical_tests(
                    np.array(matched_exp),
                    np.array(matched_base),
                )
                exp_data['stat_tests'] = stat_results

    # Create output directory for figures
    figures_dir = Path(args.output_dir) / 'figures' / 'preproc_ml_eng'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Generate report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = Path(args.output_dir) / f'{timestamp}_preproc_ml_eng_report.md'

    generate_markdown_report(results, report_path, args.tasks)

    # Generate plots
    if not args.no_plots:
        for task in args.tasks:
            # Comparison bar chart
            plot_path = figures_dir / f'{timestamp}_comparison_{task}.png'
            generate_comparison_plot(results, plot_path, task)

            # Heatmap
            heatmap_path = figures_dir / f'{timestamp}_heatmap_{task}.png'
            generate_heatmap(results, heatmap_path, task)

            # Boxplot
            boxplot_path = figures_dir / f'{timestamp}_boxplot_{task}.png'
            generate_boxplot(results, boxplot_path, task)

    # Save updated results with stats
    updated_results_path = Path(args.output_dir) / f'{timestamp}_preproc_ml_eng_results_with_stats.json'
    with open(updated_results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    log_io.info(f"更新后的结果已保存: {updated_results_path}")

    # Generate PDF if requested
    if args.pdf:
        try:
            import subprocess
            pdf_path = report_path.with_suffix('.pdf')
            subprocess.run([
                'pandoc', str(report_path),
                '-o', str(pdf_path),
                '--pdf-engine=xelatex',
            ], check=True)
            log_io.info(f"PDF 报告已保存: {pdf_path}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            log_io.warning(f"PDF 生成失败 (需要 pandoc): {e}")

    log_main.info("报告生成完成!")
    log_main.info(f"  Markdown: {report_path}")
    log_main.info(f"  图表: {figures_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
