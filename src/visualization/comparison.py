"""
Model comparison visualization.

This module provides functions for generating comparison plots
between different models (EEGNet vs CBraMod).
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


def generate_combined_plot(
    data_sources: List[PlotDataSource],
    output_path: str,
    task_type: str = 'binary',
    paradigm: str = 'imagery',
    historical_timestamp: Optional[str] = None,
):
    """
    生成组合对比图（支持混合新旧数据）.

    布局（2 行，第一行跨两列）:
    +------------------------------------------+
    |          条形图 (2x 宽度)                   |
    |   每被试 3 条: 历史数据(半透明) + 当前运行      |
    +--------------------+---------------------+
    |    箱线图(3 蜡烛)     |    配对对比图        |
    +--------------------+---------------------+

    视觉效果:
    - 历史数据: alpha=0.4, 斜线填充
    - 当前运行: alpha=1.0, 无填充, 粗边框

    Args:
        data_sources: 数据源列表（2-3 个 PlotDataSource）
        output_path: 输出文件路径
        task_type: 任务类型
        paradigm: 范式
        historical_timestamp: 历史数据时间戳（用于标题标注）
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.lines import Line2D
    except ImportError:
        log_plot.warning("matplotlib not installed, skipping plot")
        return

    chance_level = CHANCE_LEVELS.get(task_type, 0.5)
    colors = MODEL_COLORS

    # 收集所有被试
    all_subjects = set()
    for source in data_sources:
        for r in source.results:
            all_subjects.add(r.subject_id)
    subjects = sorted(all_subjects)

    if not subjects:
        log_plot.warning("No subjects for plotting")
        return

    # 创建 2 行布局，第一行跨两列
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, height_ratios=[1.2, 1], hspace=0.3, wspace=0.25)

    ax_bar = fig.add_subplot(gs[0, :])      # 顶部条形图（跨两列）
    ax_box = fig.add_subplot(gs[1, 0])      # 左下箱线图
    ax_scatter = fig.add_subplot(gs[1, 1])  # 右下配对散点图

    # =========================================================================
    # Panel 1: 条形图
    # =========================================================================
    n_subjects = len(subjects)
    n_sources = len(data_sources)
    bar_width = 0.8 / n_sources
    x_base = np.arange(n_subjects)

    for i, source in enumerate(data_sources):
        x_positions = x_base + (i - (n_sources - 1) / 2) * bar_width

        # 按被试排序获取准确率
        result_by_subj = {r.subject_id: r.test_acc_majority for r in source.results}
        accs = [result_by_subj.get(s, 0) for s in subjects]

        alpha = 1.0 if source.is_current_run else 0.4
        edgecolor = 'black' if source.is_current_run else 'gray'
        linewidth = 1.5 if source.is_current_run else 0.5
        hatch = '' if source.is_current_run else '///'

        bars = ax_bar.bar(
            x_positions, accs, bar_width,
            label=source.label,
            color=colors[source.model_type],
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=linewidth,
            hatch=hatch,
        )

        # 仅为当前运行添加数值标签
        if source.is_current_run:
            for bar, val in zip(bars, accs):
                if val > 0:
                    ax_bar.text(
                        bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.01,
                        f'{val*100:.1f}',
                        ha='center', va='bottom', fontsize=7
                    )

    ax_bar.set_xlabel('Subject')
    ax_bar.set_ylabel('Test Accuracy')
    title = f'Per-Subject Accuracy Comparison ({paradigm.title()} {task_type.title()})'
    if historical_timestamp:
        title += f'\n(Historical data from: {historical_timestamp[:10]})'
    ax_bar.set_title(title)
    ax_bar.set_xticks(x_base)
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

    box_data = []
    box_labels = []
    box_colors = []
    box_alphas = []

    for source in data_sources:
        accs = [r.test_acc_majority for r in source.results]
        if accs:
            box_data.append(accs)
            box_labels.append(source.label)
            box_colors.append(colors[source.model_type])
            box_alphas.append(1.0 if source.is_current_run else 0.4)

    if box_data:
        bp = ax_box.boxplot(
            box_data, labels=box_labels, patch_artist=True,
            showmeans=True, meanline=True,
            meanprops={'color': mean_color, 'linewidth': 2, 'linestyle': (0, (3, 2))}
        )

        for patch, color, alpha in zip(bp['boxes'], box_colors, box_alphas):
            patch.set_facecolor(color)
            patch.set_alpha(alpha)

        for median in bp['medians']:
            median.set_color(median_color)
            median.set_linewidth(2)

        # 添加统计标注
        for i, (source, accs_list) in enumerate(zip(data_sources, box_data)):
            mean_val = np.mean(accs_list)
            median_val = np.median(accs_list)
            x_offset = 0.35
            fontweight = 'bold' if source.is_current_run else 'normal'

            ax_box.text(i + 1 + x_offset, mean_val, f'{mean_val*100:.1f}',
                        ha='left', va='center', fontsize=7,
                        color=mean_color, fontweight=fontweight)
            ax_box.text(i + 1 + x_offset, median_val, f'{median_val*100:.1f}',
                        ha='left', va='center', fontsize=7,
                        color=median_color, fontweight=fontweight)

        legend_elements = [
            Line2D([0], [0], color=median_color, linewidth=2, linestyle='-', label='Median'),
            Line2D([0], [0], color=mean_color, linewidth=2, linestyle=(0, (3, 2)), label='Mean')
        ]
        ax_box.legend(handles=legend_elements, loc='upper right', fontsize=7)

    ax_box.set_ylabel('Test Accuracy')
    ax_box.set_title('Accuracy Distribution')
    ax_box.axhline(y=chance_level, color='gray', linestyle='--', alpha=0.5)

    # =========================================================================
    # Panel 3: 配对对比散点图（支持双配对：当前 vs 历史）
    # =========================================================================
    eegnet_sources = [s for s in data_sources if s.model_type == 'eegnet']
    cbramod_sources = [s for s in data_sources if s.model_type == 'cbramod']

    # 分离当前和历史数据源
    eegnet_current = next((s for s in eegnet_sources if s.is_current_run), None)
    eegnet_hist = next((s for s in eegnet_sources if not s.is_current_run), None)
    cbramod_current = next((s for s in cbramod_sources if s.is_current_run), None)
    cbramod_hist = next((s for s in cbramod_sources if not s.is_current_run), None)

    # 使用 EEGNet 作为 X 轴基准（优先使用当前数据）
    eegnet_baseline = eegnet_current or eegnet_hist

    # 配对颜色配置
    pair_colors = {
        'current': '#E94F37',   # 当前 CBraMod: 红色
        'historical': '#E94F37',  # 历史 CBraMod: 红色 (alpha=0.5 半透明)
    }

    all_accs = []  # 用于计算坐标轴范围
    has_any_pair = False

    if eegnet_baseline:
        eegnet_by_subj = {r.subject_id: r.test_acc_majority for r in eegnet_baseline.results}

        # 绘制当前配对：当前 CBraMod vs EEGNet
        if cbramod_current:
            cbramod_by_subj = {r.subject_id: r.test_acc_majority for r in cbramod_current.results}
            common = sorted(set(eegnet_by_subj.keys()) & set(cbramod_by_subj.keys()))

            if common:
                eegnet_accs = [eegnet_by_subj[s] for s in common]
                cbramod_accs = [cbramod_by_subj[s] for s in common]
                all_accs.extend(eegnet_accs + cbramod_accs)

                ax_scatter.scatter(eegnet_accs, cbramod_accs, s=100, alpha=0.9,
                                   c=pair_colors['current'], label='CBraMod (current)',
                                   edgecolors='black', linewidths=1)

                # 为当前运行添加被试标签
                for i, subj in enumerate(common):
                    ax_scatter.annotate(subj, (eegnet_accs[i], cbramod_accs[i]),
                                        xytext=(5, 5), textcoords='offset points', fontsize=7)
                has_any_pair = True

        # 绘制历史配对：历史 CBraMod vs EEGNet
        if cbramod_hist:
            cbramod_hist_by_subj = {r.subject_id: r.test_acc_majority for r in cbramod_hist.results}
            common_hist = sorted(set(eegnet_by_subj.keys()) & set(cbramod_hist_by_subj.keys()))

            if common_hist:
                eegnet_accs_hist = [eegnet_by_subj[s] for s in common_hist]
                cbramod_accs_hist = [cbramod_hist_by_subj[s] for s in common_hist]
                all_accs.extend(eegnet_accs_hist + cbramod_accs_hist)

                ax_scatter.scatter(eegnet_accs_hist, cbramod_accs_hist, s=80, alpha=0.5,
                                   c=pair_colors['historical'], label='CBraMod (hist)',
                                   edgecolors='gray', linewidths=0.5, marker='s')
                has_any_pair = True

        if has_any_pair and all_accs:
            lims = [min(all_accs) - 0.05, max(all_accs) + 0.05]
            ax_scatter.plot(lims, lims, 'k--', alpha=0.5, label='Equal')
            ax_scatter.set_xlim(lims)
            ax_scatter.set_ylim(lims)
            ax_scatter.set_xlabel(f'{eegnet_baseline.label} Accuracy')
            ax_scatter.set_ylabel('CBraMod Accuracy')
            ax_scatter.legend(loc='upper left', fontsize=7)
        else:
            ax_scatter.text(0.5, 0.5, 'No common subjects\nfor paired comparison',
                            ha='center', va='center', transform=ax_scatter.transAxes)
    else:
        ax_scatter.text(0.5, 0.5, 'Insufficient data\nfor paired comparison',
                        ha='center', va='center', transform=ax_scatter.transAxes)

    ax_scatter.set_title('CBraMod vs EEGNet (Paired Comparison)')

    # 保存
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    log_plot.info(f"Combined plot saved: {output_path}")
    plt.close()


def generate_comparison_plot(
    results: Dict[str, List[TrainingResult]],
    comparison: ComparisonResult,
    output_path: str,
    task_type: str = 'binary',
):
    """
    Generate standard 3-panel comparison plot.

    Panel 1: Per-subject bar chart
    Panel 2: Box plot with distribution
    Panel 3: Scatter plot (paired comparison)

    Args:
        results: Dict mapping model_type to list of TrainingResult
        comparison: ComparisonResult with statistics
        output_path: Path to save the plot
        task_type: 'binary', 'ternary', or 'quaternary'
    """
    chance_level = CHANCE_LEVELS.get(task_type, 0.5)

    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        log_plot.warning("matplotlib not installed, skipping plots")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    eegnet_results = results.get('eegnet', [])
    cbramod_results = results.get('cbramod', [])

    eegnet_by_subj = {r.subject_id: r for r in eegnet_results}
    cbramod_by_subj = {r.subject_id: r for r in cbramod_results}
    common = sorted(set(eegnet_by_subj.keys()) & set(cbramod_by_subj.keys()))

    if not common:
        log_plot.warning("No common subjects for plotting")
        return

    eegnet_accs = [eegnet_by_subj[s].test_acc_majority for s in common]
    cbramod_accs = [cbramod_by_subj[s].test_acc_majority for s in common]

    # =========================================================================
    # Panel 1: Bar chart
    # =========================================================================
    ax1 = axes[0]
    x = np.arange(len(common))
    width = 0.35

    bars1 = ax1.bar(x - width/2, eegnet_accs, width, label='EEGNet', color='steelblue')
    bars2 = ax1.bar(x + width/2, cbramod_accs, width, label='CBraMod', color='coral')
    ax1.set_xlabel('Subject')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Per-Subject Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(common, rotation=45)
    ax1.legend()
    ax1.set_ylim([0, 1])
    ax1.axhline(y=chance_level, color='gray', linestyle='--', alpha=0.5,
                label=f'Chance ({chance_level*100:.1f}%)')

    for bar, val in zip(bars1, eegnet_accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val*100:.1f}', ha='center', va='bottom', fontsize=7)
    for bar, val in zip(bars2, cbramod_accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val*100:.1f}', ha='center', va='bottom', fontsize=7)

    # =========================================================================
    # Panel 2: Box plot
    # =========================================================================
    ax2 = axes[1]
    median_color = 'black'
    mean_color = '#E63946'

    bp = ax2.boxplot([eegnet_accs, cbramod_accs], tick_labels=['EEGNet', 'CBraMod'],
                     patch_artist=True,
                     showmeans=True, meanline=True,
                     meanprops={'color': mean_color, 'linewidth': 2,
                               'linestyle': (0, (3, 2))})
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('coral')
    bp['boxes'][1].set_alpha(0.7)
    for median in bp['medians']:
        median.set_color(median_color)
        median.set_linewidth(2)
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Accuracy Distribution')
    ax2.axhline(y=chance_level, color='gray', linestyle='--', alpha=0.5)

    eegnet_mean = np.mean(eegnet_accs)
    eegnet_median = np.median(eegnet_accs)
    cbramod_mean = np.mean(cbramod_accs)
    cbramod_median = np.median(cbramod_accs)

    x_offset = 0.35
    ax2.text(1 + x_offset, eegnet_mean, f'{eegnet_mean*100:.1f}',
             ha='left', va='center', fontsize=7, color=mean_color, fontweight='bold')
    ax2.text(1 + x_offset, eegnet_median, f'{eegnet_median*100:.1f}',
             ha='left', va='center', fontsize=7, color=median_color, fontweight='bold')
    ax2.text(2 + x_offset, cbramod_mean, f'{cbramod_mean*100:.1f}',
             ha='left', va='center', fontsize=7, color=mean_color, fontweight='bold')
    ax2.text(2 + x_offset, cbramod_median, f'{cbramod_median*100:.1f}',
             ha='left', va='center', fontsize=7, color=median_color, fontweight='bold')

    legend_elements = [
        Line2D([0], [0], color=median_color, linewidth=2, linestyle='-', label='Median'),
        Line2D([0], [0], color=mean_color, linewidth=2, linestyle=(0, (3, 2)), label='Mean')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=7)

    # =========================================================================
    # Panel 3: Scatter plot (paired comparison)
    # =========================================================================
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
    log_plot.info(f"Comparison plot saved: {output_path}")
    plt.close()
