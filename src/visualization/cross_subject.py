"""
Cross-subject training visualization.

This module provides functions for generating plots specific to
cross-subject pretraining experiments, including comparisons with
within-subject historical data.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from ..config.constants import MODEL_COLORS
from ..results.dataclasses import ComparisonResult, PlotDataSource, TrainingResult
from ..utils.logging import SectionLogger
from .plots import CHANCE_LEVELS

logger = logging.getLogger(__name__)
log_plot = SectionLogger(logger, 'plot')

# 配置显示名称（用于图表标签）
CONFIG_DISPLAY_NAMES: Dict[str, str] = {
    'motor_cortex': 'Motor Cortex',
    'commercial': 'Commercial',
    'fdr': 'FDR',
    'csp': 'CSP',
    'attention': 'Attention',
    'band_power': 'Band Power',
}


def generate_cross_subject_single_plot(
    result: Dict,
    model_type: str,
    output_path: str,
    task_type: str = 'binary',
    paradigm: str = 'imagery',
    historical_within_subject: Optional[Dict] = None,
) -> None:
    """
    生成单模型跨被试结果图.

    布局：2 子图
    - 左：每被试准确率柱状图（可选：叠加 within-subject 历史）
    - 右：箱线图

    Args:
        result: train_cross_subject() 返回的字典
        model_type: 'eegnet' 或 'cbramod'
        output_path: 输出文件路径
        task_type: 任务类型
        paradigm: 范式
        historical_within_subject: 可选的 within-subject 历史数据
            格式: {'eegnet': {...}, 'cbramod': {...}, 'timestamp': str}
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        log_plot.warning("matplotlib not installed, skipping plot generation")
        print("[WARNING] matplotlib 未安装，无法生成图表。请运行: pip install matplotlib")
        return

    chance_level = CHANCE_LEVELS.get(task_type, 0.5)
    colors = MODEL_COLORS

    # 提取当前运行数据
    per_subject_acc = (
        result.get('per_subject_test_acc')
        if result.get('per_subject_test_acc') is not None
        else result.get('results', {}).get('per_subject_test_acc', {})
    )

    if not per_subject_acc:
        log_plot.warning("No per-subject accuracy data for plotting")
        return

    subjects = sorted(per_subject_acc.keys())
    current_accs = [per_subject_acc[s] for s in subjects]

    # 检查是否有历史数据
    has_historical = False
    hist_accs = []
    hist_timestamp = None

    if historical_within_subject:
        model_hist = historical_within_subject.get(model_type, {})
        hist_subjects_data = model_hist.get('subjects', [])
        if hist_subjects_data:
            hist_by_subj = {
                s.get('subject_id'): s.get('test_acc_majority', s.get('test_acc', 0))
                for s in hist_subjects_data
            }
            hist_accs = [hist_by_subj.get(s, 0) for s in subjects]
            has_historical = any(a > 0 for a in hist_accs)
            hist_timestamp = historical_within_subject.get('timestamp', 'unknown')

    # 创建 2 列布局
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_bar, ax_box = axes

    # =========================================================================
    # Panel 1: 每被试准确率柱状图
    # =========================================================================
    n_subjects = len(subjects)
    x = np.arange(n_subjects)

    if has_historical:
        # 两组柱子：历史 + 当前
        bar_width = 0.35

        # 历史数据（半透明，斜线填充）
        ax_bar.bar(
            x - bar_width/2, hist_accs, bar_width,
            label=f'{model_type.upper()} (within-subj)',
            color=colors[model_type],
            alpha=0.4,
            edgecolor='gray',
            linewidth=0.5,
            hatch='///',
        )

        # 当前运行（实心）
        bars = ax_bar.bar(
            x + bar_width/2, current_accs, bar_width,
            label=f'{model_type.upper()} (cross-subj)',
            color=colors[model_type],
            alpha=1.0,
            edgecolor='black',
            linewidth=1.5,
        )

        # 添加数值标签
        for bar, val in zip(bars, current_accs):
            ax_bar.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{val*100:.1f}',
                ha='center', va='bottom', fontsize=7
            )
    else:
        # 单组柱子
        bars = ax_bar.bar(
            x, current_accs, 0.6,
            label=f'{model_type.upper()} (cross-subj)',
            color=colors[model_type],
            alpha=1.0,
            edgecolor='black',
            linewidth=1.5,
        )

        for bar, val in zip(bars, current_accs):
            ax_bar.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{val*100:.1f}',
                ha='center', va='bottom', fontsize=7
            )

    # 添加均值线
    mean_acc = np.mean(current_accs)
    ax_bar.axhline(y=mean_acc, color=colors[model_type], linestyle='-', alpha=0.7,
                   linewidth=2, label=f'Mean: {mean_acc*100:.1f}%')

    ax_bar.set_xlabel('Subject')
    ax_bar.set_ylabel('Test Accuracy')
    title = f'{model_type.upper()} Cross-Subject Results ({paradigm.title()} {task_type.title()})'
    if hist_timestamp:
        title += f'\n(vs within-subject from: {hist_timestamp[:10]})'
    ax_bar.set_title(title)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(subjects, rotation=45, ha='right')
    ax_bar.axhline(y=chance_level, color='gray', linestyle='--', alpha=0.5,
                   label=f'Chance ({chance_level*100:.1f}%)')
    ax_bar.set_ylim([0, 1.05])
    ax_bar.legend(loc='upper right', fontsize=8)

    # =========================================================================
    # Panel 2: 箱线图
    # =========================================================================
    median_color = 'black'
    mean_color = '#E63946'

    box_data = [current_accs]
    box_labels = [f'{model_type.upper()}\n(cross-subj)']

    if has_historical:
        box_data.insert(0, hist_accs)
        box_labels.insert(0, f'{model_type.upper()}\n(within-subj)')

    bp = ax_box.boxplot(
        box_data, labels=box_labels, patch_artist=True,
        showmeans=True, meanline=True,
        meanprops={'color': mean_color, 'linewidth': 2, 'linestyle': (0, (3, 2))}
    )

    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[model_type])
        # 第一个是历史（如果有），alpha=0.4；最后一个是当前，alpha=1.0
        if has_historical and i == 0:
            patch.set_alpha(0.4)
        else:
            patch.set_alpha(1.0)

    for median in bp['medians']:
        median.set_color(median_color)
        median.set_linewidth(2)

    # 添加统计标注
    for i, accs_list in enumerate(box_data):
        mean_val = np.mean(accs_list)
        median_val = np.median(accs_list)
        x_offset = 0.35

        ax_box.text(i + 1 + x_offset, mean_val, f'{mean_val*100:.1f}',
                    ha='left', va='center', fontsize=7, color=mean_color)
        ax_box.text(i + 1 + x_offset, median_val, f'{median_val*100:.1f}',
                    ha='left', va='center', fontsize=7, color=median_color)

    legend_elements = [
        Line2D([0], [0], color=median_color, linewidth=2, linestyle='-', label='Median'),
        Line2D([0], [0], color=mean_color, linewidth=2, linestyle=(0, (3, 2)), label='Mean')
    ]
    ax_box.legend(handles=legend_elements, loc='upper right', fontsize=7)

    ax_box.set_ylabel('Test Accuracy')
    ax_box.set_title('Accuracy Distribution')
    ax_box.axhline(y=chance_level, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    log_plot.info(f"Cross-subject single plot saved: {output_path}")
    plt.close()


def generate_config_comparison_plot(
    config_results: Dict[str, Dict[str, Dict[str, float]]],
    output_path: str,
    task_type: str = 'binary',
    paradigm: str = 'imagery',
    n_channels: int = 32,
    baseline_accs: Optional[Dict[str, float]] = None,
) -> None:
    """
    生成多通道配置综合对比图（N 个配置 × 2 模型）.

    布局（2 行）:
    +------------------------------------------+
    |  均值准确率对比柱状图（全宽）               |
    |  x 轴: 各配置, 每组 2 根柱子 (EEGNet/CBraMod) |
    +--------------------+---------------------+
    |  CBraMod 分布箱线图  |  EEGNet 分布箱线图  |
    |  (N 个配置各一箱)   |  (N 个配置各一箱)   |
    +--------------------+---------------------+

    Args:
        config_results: {config_name: {model_type: {subject_id: accuracy}}}
        output_path: 输出文件路径
        task_type: 任务类型 ('binary', 'ternary', 'quaternary')
        paradigm: 范式 ('imagery', 'movement')
        n_channels: 通道数（用于标题）
        baseline_accs: 全通道基线均值 {model_type: mean_acc}（绘制参考横线）
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.lines import Line2D
        import matplotlib.patches as mpatches
    except ImportError:
        log_plot.warning("matplotlib not installed, skipping plot generation")
        return

    if not config_results:
        log_plot.warning("No config results for plotting")
        return

    chance_level = CHANCE_LEVELS.get(task_type, 0.5)
    colors = MODEL_COLORS
    median_color = 'black'
    mean_color = '#E63946'

    configs = list(config_results.keys())
    n_configs = len(configs)
    config_labels = [CONFIG_DISPLAY_NAMES.get(c, c) for c in configs]

    # 收集各配置各模型的准确率列表
    model_types = ['eegnet', 'cbramod']
    data: Dict[str, Dict[str, List[float]]] = {}
    for cfg in configs:
        data[cfg] = {}
        for mt in model_types:
            subj_accs = config_results[cfg].get(mt, {})
            data[cfg][mt] = list(subj_accs.values()) if subj_accs else []

    # =========================================================================
    # 图形布局
    # =========================================================================
    fig = plt.figure(figsize=(max(14, n_configs * 2.5), 11))
    gs = GridSpec(2, 2, height_ratios=[1.1, 1.0], hspace=0.35, wspace=0.28)
    ax_bar = fig.add_subplot(gs[0, :])       # 顶部全宽柱状图
    ax_cbramod = fig.add_subplot(gs[1, 0])   # 左下 CBraMod 箱线图
    ax_eegnet = fig.add_subplot(gs[1, 1])    # 右下 EEGNet 箱线图

    # =========================================================================
    # Panel 1: 均值准确率柱状图
    # =========================================================================
    x = np.arange(n_configs)
    bar_width = 0.35

    eegnet_means = [np.mean(data[c]['eegnet']) if data[c]['eegnet'] else 0 for c in configs]
    eegnet_stds = [np.std(data[c]['eegnet']) if data[c]['eegnet'] else 0 for c in configs]
    cbramod_means = [np.mean(data[c]['cbramod']) if data[c]['cbramod'] else 0 for c in configs]
    cbramod_stds = [np.std(data[c]['cbramod']) if data[c]['cbramod'] else 0 for c in configs]

    bars_eeg = ax_bar.bar(
        x - bar_width / 2, eegnet_means, bar_width,
        label='EEGNet', color=colors['eegnet'], alpha=0.85,
        edgecolor='black', linewidth=1.2,
        yerr=eegnet_stds, capsize=4, error_kw={'linewidth': 1.2},
    )
    bars_cbr = ax_bar.bar(
        x + bar_width / 2, cbramod_means, bar_width,
        label='CBraMod', color=colors['cbramod'], alpha=0.85,
        edgecolor='black', linewidth=1.2,
        yerr=cbramod_stds, capsize=4, error_kw={'linewidth': 1.2},
    )

    # 柱顶均值标签
    for bar, val in zip(bars_eeg, eegnet_means):
        if val > 0:
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(eegnet_stds) + 0.005,
                f'{val * 100:.1f}%', ha='center', va='bottom', fontsize=7.5,
                color=colors['eegnet'], fontweight='bold',
            )
    for bar, val in zip(bars_cbr, cbramod_means):
        if val > 0:
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(cbramod_stds) + 0.005,
                f'{val * 100:.1f}%', ha='center', va='bottom', fontsize=7.5,
                color=colors['cbramod'], fontweight='bold',
            )

    # 基线参考线（128ch 全通道）
    if baseline_accs:
        if 'eegnet' in baseline_accs:
            ax_bar.axhline(
                y=baseline_accs['eegnet'], color=colors['eegnet'],
                linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'EEGNet 128ch baseline ({baseline_accs["eegnet"]*100:.1f}%)',
            )
        if 'cbramod' in baseline_accs:
            ax_bar.axhline(
                y=baseline_accs['cbramod'], color=colors['cbramod'],
                linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'CBraMod 128ch baseline ({baseline_accs["cbramod"]*100:.1f}%)',
            )

    ax_bar.axhline(y=chance_level, color='gray', linestyle=':', alpha=0.6,
                   label=f'Chance ({chance_level * 100:.0f}%)')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(config_labels, fontsize=10)
    ax_bar.set_ylabel('Mean Test Accuracy', fontsize=11)
    ax_bar.set_ylim([0, min(1.05, max(max(eegnet_means), max(cbramod_means)) + 0.15)])
    ax_bar.set_title(
        f'{n_channels}-Channel Configuration Comparison — {paradigm.title()} {task_type.title()}\n'
        f'(bars show mean ± std across {len(list(config_results[configs[0]].get("cbramod", {}).keys()))} subjects)',
        fontsize=12,
    )
    ax_bar.legend(loc='upper right', fontsize=8, ncol=2)

    # =========================================================================
    # Panel 2 & 3: 箱线图（CBraMod / EEGNet）
    # =========================================================================
    for ax, model_type, model_label in [
        (ax_cbramod, 'cbramod', 'CBraMod'),
        (ax_eegnet, 'eegnet', 'EEGNet'),
    ]:
        box_data = [data[c][model_type] for c in configs]
        valid_mask = [bool(d) for d in box_data]
        valid_data = [d for d in box_data if d]
        valid_labels = [config_labels[i] for i, v in enumerate(valid_mask) if v]

        if not valid_data:
            ax.text(0.5, 0.5, f'No {model_label} data', ha='center', va='center',
                    transform=ax.transAxes)
            continue

        bp = ax.boxplot(
            valid_data, labels=valid_labels, patch_artist=True,
            showmeans=True, meanline=True,
            meanprops={'color': mean_color, 'linewidth': 2, 'linestyle': (0, (3, 2))},
        )

        model_color = colors[model_type]
        for patch in bp['boxes']:
            patch.set_facecolor(model_color)
            patch.set_alpha(0.75)
        for median in bp['medians']:
            median.set_color(median_color)
            median.set_linewidth(2)

        # 均值 / 中位数标注
        x_offset = 0.32
        for i, accs in enumerate(valid_data):
            mean_val = np.mean(accs)
            median_val = np.median(accs)
            ax.text(i + 1 + x_offset, mean_val, f'{mean_val * 100:.1f}',
                    ha='left', va='center', fontsize=6.5, color=mean_color)
            ax.text(i + 1 + x_offset, median_val, f'{median_val * 100:.1f}',
                    ha='left', va='center', fontsize=6.5, color=median_color)

        ax.axhline(y=chance_level, color='gray', linestyle=':', alpha=0.6)
        if baseline_accs and model_type in baseline_accs:
            ax.axhline(
                y=baseline_accs[model_type], color=model_color,
                linestyle='--', linewidth=1.2, alpha=0.6,
                label=f'128ch: {baseline_accs[model_type]*100:.1f}%',
            )
            ax.legend(fontsize=7, loc='upper right')

        ax.set_ylabel('Test Accuracy', fontsize=10)
        ax.set_title(f'{model_label} — Distribution by Config', fontsize=10)
        ax.set_ylim([0, 1.05])
        ax.tick_params(axis='x', labelsize=8.5)

        legend_elements = [
            Line2D([0], [0], color=median_color, linewidth=2, linestyle='-', label='Median'),
            Line2D([0], [0], color=mean_color, linewidth=2, linestyle=(0, (3, 2)), label='Mean'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=7)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    log_plot.info(f"Config comparison plot saved: {output_path}")
    plt.close()
