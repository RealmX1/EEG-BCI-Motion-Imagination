"""
FDR vs Commercial 32ch 全管线对比图.

从 ExperimentDB 查询两种配置在所有实验类型 (cross-subject, transfer) 下的
CBraMod / EEGNet 表现，生成分组柱状图 (mean ± std).

Usage:
    uv run python scripts/analysis/plot_fdr_vs_commercial.py
    uv run python scripts/analysis/plot_fdr_vs_commercial.py --task ternary
    uv run python scripts/analysis/plot_fdr_vs_commercial.py --output results/custom.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.results.experiment_db import ExperimentDB
from src.config.constants import MODEL_COLORS
from src.visualization.plots import CHANCE_LEVELS


# ============================================================================
# Data collection
# ============================================================================

def collect_config_pipeline_data(
    db: ExperimentDB,
    config: str,
    task: str,
    paradigm: str = 'imagery',
    n_channels: int = 32,
) -> Dict[str, Dict[str, List[float]]]:
    """Collect per-subject accuracies for a config across all experiment types.

    Returns:
        {experiment_type: {model_type: [acc_per_subject]}}
    """
    data: Dict[str, Dict[str, List[float]]] = {}

    for exp_type in ['cross_subject', 'transfer']:
        runs = db.find_runs(
            paradigm=paradigm,
            task=task,
            n_channels=n_channels,
            channel_config=config,
            experiment_type=exp_type,
            is_complete=True,
        )
        if not runs:
            continue

        # Take the most recent run
        run = runs[0]
        results = db.get_results(run['run_id'])

        model_accs: Dict[str, List[float]] = {}
        for r in results:
            model_accs.setdefault(r.model_type, []).append(r.test_acc_majority)

        if model_accs:
            data[exp_type] = model_accs

    return data


# ============================================================================
# Plot generation
# ============================================================================

EXPERIMENT_LABELS = {
    'cross_subject': 'Cross-Subject',
    'transfer': 'Transfer',
}

CONFIG_COLORS = {
    'fdr': '#2E86AB',         # Blue
    'commercial': '#E94F37',  # Red/Coral
}

CONFIG_HATCHES = {
    'fdr': '',
    'commercial': '///',
}


def generate_fdr_vs_commercial_plot(
    fdr_data: Dict[str, Dict[str, List[float]]],
    comm_data: Dict[str, Dict[str, List[float]]],
    output_path: str,
    task: str = 'binary',
    paradigm: str = 'imagery',
    baseline_128ch: Optional[Dict[str, float]] = None,
) -> None:
    """Generate grouped bar chart: FDR vs Commercial across experiment types.

    X-axis groups: experiment types (Cross-Subject, Transfer)
    Within each group: 4 bars (FDR-EEGNet, FDR-CBraMod, Comm-EEGNet, Comm-CBraMod)
    Error bars: ±1 std across subjects.
    """
    chance_level = CHANCE_LEVELS.get(task, 0.5)

    # Determine which experiment types have data from EITHER config
    exp_types = []
    for et in ['cross_subject', 'transfer']:
        if et in fdr_data or et in comm_data:
            exp_types.append(et)

    if not exp_types:
        print("No data available for plotting.")
        return

    # Build bar groups: each group = 1 experiment type, up to 4 bars
    # Order within group: FDR-CBraMod, FDR-EEGNet, Comm-CBraMod, Comm-EEGNet
    bar_defs = [
        ('fdr', 'cbramod', 'FDR CBraMod'),
        ('fdr', 'eegnet', 'FDR EEGNet'),
        ('commercial', 'cbramod', 'Comm CBraMod'),
        ('commercial', 'eegnet', 'Comm EEGNet'),
    ]

    config_model_colors = {
        ('fdr', 'cbramod'): '#E94F37',
        ('fdr', 'eegnet'): '#2E86AB',
        ('commercial', 'cbramod'): '#F4A261',
        ('commercial', 'eegnet'): '#6BAED6',
    }

    config_model_hatches = {
        ('fdr', 'cbramod'): '',
        ('fdr', 'eegnet'): '',
        ('commercial', 'cbramod'): '///',
        ('commercial', 'eegnet'): '///',
    }

    n_groups = len(exp_types)
    n_bars = len(bar_defs)
    bar_width = 0.18
    group_width = n_bars * bar_width + 0.08

    fig, (ax_bar, ax_box) = plt.subplots(
        1, 2, figsize=(max(10, n_groups * 5), 6),
        gridspec_kw={'width_ratios': [1.6, 1], 'wspace': 0.25},
    )

    # =========================================================================
    # Panel 1: Grouped bar chart
    # =========================================================================
    x_groups = np.arange(n_groups)

    for bar_idx, (config, model, label) in enumerate(bar_defs):
        data_source = fdr_data if config == 'fdr' else comm_data
        means = []
        stds = []
        for et in exp_types:
            accs = data_source.get(et, {}).get(model, [])
            if accs:
                means.append(np.mean(accs))
                stds.append(np.std(accs))
            else:
                means.append(0)
                stds.append(0)

        offset = (bar_idx - (n_bars - 1) / 2) * bar_width
        color = config_model_colors[(config, model)]
        hatch = config_model_hatches[(config, model)]

        bars = ax_bar.bar(
            x_groups + offset, means, bar_width,
            label=label,
            color=color, alpha=0.85,
            edgecolor='black', linewidth=1.0,
            hatch=hatch,
            yerr=stds, capsize=3, error_kw={'linewidth': 1.0},
        )

        # Value labels (inside bars to avoid overlap)
        for bar, val, std in zip(bars, means, stds):
            if val > 0:
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() - 0.03,
                    f'{val * 100:.1f}',
                    ha='center', va='top', fontsize=7.5,
                    fontweight='bold', color='white',
                )

    # 128ch baseline
    if baseline_128ch:
        for model, lstyle in [('cbramod', '--'), ('eegnet', ':')]:
            if model in baseline_128ch:
                ax_bar.axhline(
                    y=baseline_128ch[model],
                    color=MODEL_COLORS[model], linestyle=lstyle,
                    linewidth=1.5, alpha=0.6,
                    label=f'128ch {model.upper()} ({baseline_128ch[model]*100:.1f}%)',
                )

    ax_bar.axhline(y=chance_level, color='gray', linestyle=':', alpha=0.5,
                   label=f'Chance ({chance_level * 100:.0f}%)')

    ax_bar.set_xticks(x_groups)
    ax_bar.set_xticklabels([EXPERIMENT_LABELS.get(et, et) for et in exp_types], fontsize=11)
    ax_bar.set_ylabel('Mean Test Accuracy', fontsize=11)
    ax_bar.set_ylim([0, min(1.08, ax_bar.get_ylim()[1] + 0.05)])
    ax_bar.set_title(
        f'32ch FDR vs Commercial — Full Pipeline\n'
        f'{paradigm.title()} {task.title()} (21 subjects)',
        fontsize=12,
    )
    ax_bar.legend(loc='upper left', fontsize=7.5, ncol=2)

    # =========================================================================
    # Panel 2: Box plot (CBraMod only, FDR vs Commercial per experiment type)
    # =========================================================================
    box_data = []
    box_labels = []
    box_colors = []

    for et in exp_types:
        et_label = EXPERIMENT_LABELS.get(et, et)
        for config, color in [('fdr', '#E94F37'), ('commercial', '#F4A261')]:
            data_source = fdr_data if config == 'fdr' else comm_data
            accs = data_source.get(et, {}).get('cbramod', [])
            if accs:
                box_data.append(accs)
                box_labels.append(f'{et_label}\n{config.upper()}')
                box_colors.append(color)

    if box_data:
        bp = ax_box.boxplot(
            box_data, tick_labels=box_labels, patch_artist=True,
            showmeans=True, meanline=True,
            meanprops={'color': '#E63946', 'linewidth': 2, 'linestyle': (0, (3, 2))},
        )
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)

        # Annotate mean values
        for i, accs in enumerate(box_data):
            mean_val = np.mean(accs)
            ax_box.text(
                i + 1.3, mean_val, f'{mean_val * 100:.1f}%',
                ha='left', va='center', fontsize=7, color='#E63946',
            )

        ax_box.axhline(y=chance_level, color='gray', linestyle=':', alpha=0.5)
        if baseline_128ch and 'cbramod' in baseline_128ch:
            ax_box.axhline(
                y=baseline_128ch['cbramod'], color=MODEL_COLORS['cbramod'],
                linestyle='--', linewidth=1.2, alpha=0.6,
                label=f'128ch ({baseline_128ch["cbramod"]*100:.1f}%)',
            )
            ax_box.legend(loc='upper right', fontsize=7)

        ax_box.set_ylabel('Test Accuracy (CBraMod)', fontsize=10)
        ax_box.set_title('CBraMod Distribution', fontsize=11)
        ax_box.set_ylim([0, 1.05])
        ax_box.tick_params(axis='x', labelsize=8)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {output_path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='FDR vs Commercial 32ch full-pipeline comparison')
    parser.add_argument('--task', default='binary', choices=['binary', 'ternary'])
    parser.add_argument('--paradigm', default='imagery')
    parser.add_argument('--output', default=None, help='Output path (auto-generated if omitted)')
    args = parser.parse_args()

    db = ExperimentDB()

    fdr_data = collect_config_pipeline_data(db, 'fdr', args.task, args.paradigm)
    comm_data = collect_config_pipeline_data(db, 'commercial', args.task, args.paradigm)

    print(f"FDR data: {', '.join(f'{k}: {list(v.keys())}' for k, v in fdr_data.items())}")
    print(f"Comm data: {', '.join(f'{k}: {list(v.keys())}' for k, v in comm_data.items())}")

    # 128ch baseline from DB — pick run with highest mean accuracy per model
    baseline_128ch: Dict[str, float] = {}
    for model in ['eegnet', 'cbramod']:
        cs_runs = db.find_runs(
            paradigm=args.paradigm, task=args.task,
            n_channels=128, experiment_type='cross_subject', is_complete=True,
        )
        best_mean = 0.0
        for run in cs_runs:
            results = db.get_results(run['run_id'], model)
            if results:
                run_mean = np.mean([r.test_acc_majority for r in results])
                if run_mean > best_mean:
                    best_mean = run_mean
        if best_mean > 0:
            baseline_128ch[model] = best_mean

    output = args.output or f'results/32ch_fdr_vs_commercial_{args.paradigm}_{args.task}.png'
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    generate_fdr_vs_commercial_plot(
        fdr_data=fdr_data,
        comm_data=comm_data,
        output_path=output,
        task=args.task,
        paradigm=args.paradigm,
        baseline_128ch=baseline_128ch or None,
    )

    db.close()


if __name__ == '__main__':
    main()
