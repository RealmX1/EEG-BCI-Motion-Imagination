"""
Single model visualization.

This module provides functions for generating visualizations
for a single model's training results.
"""

import logging
from typing import Dict, List

import numpy as np

from ..config.constants import MODEL_COLORS
from ..results.dataclasses import TrainingResult
from ..utils.logging import SectionLogger
from .plots import (
    CHANCE_LEVELS, annotate_bars_with_leaders, accuracy_ylim,
    separate_paired_labels, draw_label_with_leader,
)

logger = logging.getLogger(__name__)
log_plot = SectionLogger(logger, 'plot')


def generate_single_model_plot(
    model_type: str,
    results: List[TrainingResult],
    statistics: Dict,
    output_path: str,
    task_type: str = 'binary',
):
    """
    Generate 2-panel visualization for a single model.

    Panel 1: Per-subject bar chart
    Panel 2: Distribution box plot with individual points

    Args:
        model_type: 'eegnet' or 'cbramod'
        results: List of TrainingResult
        statistics: Statistics dict from compute_model_statistics()
        output_path: Path to save the plot
        task_type: 'binary', 'ternary', or 'quaternary'
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        log_plot.warning("matplotlib not installed, skipping plots")
        return

    if not results:
        log_plot.warning("No results to plot")
        return

    # Calculate chance level (in percentage)
    chance_levels_pct = {'binary': 50.0, 'ternary': 33.3, 'quaternary': 25.0}
    chance_level = chance_levels_pct.get(task_type, 50.0)

    model_color = MODEL_COLORS.get(model_type, '#666666')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Sort results by subject ID
    sorted_results = sorted(results, key=lambda x: x.subject_id)
    subjects = [r.subject_id for r in sorted_results]
    accs = [r.test_acc_majority * 100 for r in sorted_results]

    # =========================================================================
    # Panel 1: Per-subject bar chart
    # =========================================================================
    ax1 = axes[0]

    bars = ax1.bar(subjects, accs, color=model_color, edgecolor='black', linewidth=0.5, alpha=0.8)

    # Add value labels with leader lines
    annotate_bars_with_leaders(
        ax1, [(bars, accs, True)],
        base_margin=3.0, step_height=1.5, scale=1, fontsize=8,
    )

    # Chance level line
    ax1.axhline(y=chance_level, color='gray', linestyle='--', alpha=0.5,
                label=f'Chance ({chance_level:.1f}%)')

    # Mean line
    mean_acc = statistics['mean'] * 100
    ax1.axhline(y=mean_acc, color='red', linestyle='-', linewidth=2,
                label=f'Mean ({mean_acc:.1f}%)')

    ax1.set_xlabel('Subject')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title(f'{model_type.upper()} Per-Subject Accuracy')
    ax1.legend(loc='lower right')
    ax1.set_ylim(accuracy_ylim(task_type, is_pct=True))

    # Rotate x-labels if many subjects
    if len(subjects) > 10:
        ax1.set_xticklabels(subjects, rotation=45, ha='right')

    # =========================================================================
    # Panel 2: Box plot with individual points
    # =========================================================================
    ax2 = axes[1]

    bp = ax2.boxplot([accs], tick_labels=[model_type.upper()],
                     patch_artist=True, widths=0.5,
                     showmeans=True, meanline=True,
                     meanprops={'color': 'red', 'linewidth': 2, 'linestyle': (0, (3, 2))},
                     medianprops={'color': 'black', 'linewidth': 2})

    bp['boxes'][0].set_facecolor(model_color)
    bp['boxes'][0].set_alpha(0.7)

    # Add individual points (jittered)
    np.random.seed(42)  # For reproducible jitter
    x_jitter = np.random.normal(1, 0.04, len(accs))
    ax2.scatter(x_jitter, accs, color=model_color, edgecolor='black',
               linewidth=0.5, s=80, zorder=3, alpha=0.8)

    # Add value annotations
    median_acc = statistics['median'] * 100
    box_right = max(v[0] for v in bp['boxes'][0].get_path().vertices)
    adj_mean, adj_med = separate_paired_labels(mean_acc, median_acc, min_gap=2.0)
    draw_label_with_leader(
        ax2, mean_acc, adj_mean, box_right,
        f'Mean: {mean_acc:.1f}%', color='red', fontsize=9, fontweight='bold')
    draw_label_with_leader(
        ax2, median_acc, adj_med, box_right,
        f'Median: {median_acc:.1f}%', color='black', fontsize=9, fontweight='bold')

    # Chance level
    ax2.axhline(y=chance_level, color='gray', linestyle='--', alpha=0.5)

    # Statistics text box
    stats_text = (f"n={statistics['n_subjects']}\n"
                  f"Std: {statistics['std']*100:.1f}%\n"
                  f"Range: [{statistics['min']*100:.1f}%, {statistics['max']*100:.1f}%]")
    ax2.text(0.95, 0.05, stats_text, transform=ax2.transAxes,
             fontsize=9, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title(f'{model_type.upper()} Accuracy Distribution')
    ax2.set_xlim([0.5, 1.8])
    ax2.set_ylim(accuracy_ylim(task_type, is_pct=True, top_pad=0.08))

    # Legend for mean/median lines
    legend_elements = [
        Line2D([0], [0], color='black', linewidth=2, linestyle='-', label='Median'),
        Line2D([0], [0], color='red', linewidth=2, linestyle=(0, (3, 2)), label='Mean')
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    log_plot.info(f"Single model plot saved: {output_path}")
    plt.close()
