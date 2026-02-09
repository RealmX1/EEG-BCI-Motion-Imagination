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
