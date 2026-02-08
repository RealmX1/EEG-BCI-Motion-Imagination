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


def cross_subject_result_to_plot_data(
    result: Dict,
    model_type: str,
    is_current_run: bool = True,
    training_type: str = 'cross-subject',
) -> PlotDataSource:
    """
    将 train_cross_subject 返回值或 cross-subject JSON 转换为 PlotDataSource.

    Args:
        result: train_cross_subject() 返回的字典或 JSON 加载的数据
        model_type: 'eegnet' 或 'cbramod'
        is_current_run: 是否为当前运行（影响图表样式）
        training_type: 'cross-subject' 或 'within-subject'（用于标签）

    Returns:
        PlotDataSource 对象，可用于绘图
    """
    # 从 per_subject_test_acc 创建 TrainingResult 列表
    training_results = []

    # 支持多种数据格式
    per_subject = (
        result.get('per_subject_test_acc') or
        result.get('results', {}).get('per_subject_test_acc', {})
    )

    task = result.get('task', result.get('metadata', {}).get('task', 'binary'))
    val_acc = (
        result.get('val_acc') or
        result.get('val_majority_acc') or
        result.get('results', {}).get('best_val_acc', 0)
    )
    best_epoch = (
        result.get('best_epoch') or
        result.get('results', {}).get('best_epoch', 0)
    )
    total_time = (
        result.get('training_time') or
        result.get('training_info', {}).get('training_time', 0)
    )
    n_subjects = len(per_subject) if per_subject else 1

    for subject_id, test_acc in per_subject.items():
        training_results.append(TrainingResult(
            subject_id=subject_id,
            task_type=task,
            model_type=model_type,
            best_val_acc=val_acc,
            test_acc=test_acc,
            test_acc_majority=test_acc,  # 跨被试训练的测试准确率已是 majority vote
            epochs_trained=best_epoch,
            training_time=total_time / max(n_subjects, 1),
        ))

    label_suffix = "Cross" if training_type == 'cross-subject' else "Within"
    return PlotDataSource(
        model_type=model_type,
        results=training_results,
        is_current_run=is_current_run,
        label=f'{model_type.upper()} ({label_suffix})',
    )


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
        log_plot.warning("matplotlib not installed, skipping plot")
        return

    chance_level = CHANCE_LEVELS.get(task_type, 0.5)
    colors = MODEL_COLORS

    # 提取当前运行数据
    per_subject_acc = (
        result.get('per_subject_test_acc') or
        result.get('results', {}).get('per_subject_test_acc', {})
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


def generate_cross_subject_comparison_plot(
    current_results: Dict[str, Dict],
    within_subject_historical: Optional[Dict],
    cross_subject_historical: Optional[Dict],
    comparison: Optional[ComparisonResult],
    output_path: str,
    task_type: str = 'binary',
    paradigm: str = 'imagery',
) -> None:
    """
    生成跨被试训练对比图.

    数据源（最多 5 个）：
    1. EEGNet Within-Subject (历史, 半透明)
    2. CBraMod Within-Subject (历史, 半透明)
    3. CBraMod Cross-Subject (历史, 半透明+点线, 可选)
    4. EEGNet Cross-Subject (当前, 实心)
    5. CBraMod Cross-Subject (当前, 实心)

    布局：3 子图
    - 左：柱状图（每被试准确率，分组显示）
    - 中：箱线图（分布对比）
    - 右：统计摘要表

    Args:
        current_results: 当前跨被试训练结果 {model_type: train_cross_subject 返回值}
        within_subject_historical: within-subject 历史数据
            格式: {'eegnet': {...}, 'cbramod': {...}, 'timestamp': str}
        cross_subject_historical: cross-subject 历史数据（仅 CBraMod）
            格式: {'per_subject_test_acc': {...}, 'timestamp': str, 'model_type': str}
        comparison: 可选的统计比较结果
        output_path: 输出文件路径
        task_type: 任务类型
        paradigm: 范式
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
    for model_type, result in current_results.items():
        per_subj = (
            result.get('per_subject_test_acc') or
            result.get('results', {}).get('per_subject_test_acc', {})
        )
        all_subjects.update(per_subj.keys())
    subjects = sorted(all_subjects)

    if not subjects:
        log_plot.warning("No subjects for plotting")
        return

    # 构建数据源列表
    data_sources = []

    # 1. EEGNet Within-Subject (历史)
    if within_subject_historical:
        eegnet_hist = within_subject_historical.get('eegnet', {})
        if eegnet_hist.get('subjects'):
            data_sources.append({
                'name': 'EEGNet (within)',
                'model_type': 'eegnet',
                'training_type': 'within-subject',
                'is_current': False,
                'is_cross_historical': False,
                'data': {s['subject_id']: s.get('test_acc_majority', s.get('test_acc', 0))
                         for s in eegnet_hist['subjects']},
            })

    # 2. CBraMod Within-Subject (历史)
    if within_subject_historical:
        cbramod_hist = within_subject_historical.get('cbramod', {})
        if cbramod_hist.get('subjects'):
            data_sources.append({
                'name': 'CBraMod (within)',
                'model_type': 'cbramod',
                'training_type': 'within-subject',
                'is_current': False,
                'is_cross_historical': False,
                'data': {s['subject_id']: s.get('test_acc_majority', s.get('test_acc', 0))
                         for s in cbramod_hist['subjects']},
            })

    # 3. CBraMod Cross-Subject (历史) - 如果有
    if cross_subject_historical:
        per_subj_hist = cross_subject_historical.get('per_subject_test_acc', {})
        if per_subj_hist:
            data_sources.append({
                'name': 'CBraMod (cross-hist)',
                'model_type': 'cbramod',
                'training_type': 'cross-subject',
                'is_current': False,
                'is_cross_historical': True,
                'data': per_subj_hist,
            })

    # 4. EEGNet Cross-Subject (当前)
    if 'eegnet' in current_results:
        per_subj = (
            current_results['eegnet'].get('per_subject_test_acc') or
            current_results['eegnet'].get('results', {}).get('per_subject_test_acc', {})
        )
        if per_subj:
            data_sources.append({
                'name': 'EEGNet (cross)',
                'model_type': 'eegnet',
                'training_type': 'cross-subject',
                'is_current': True,
                'is_cross_historical': False,
                'data': per_subj,
            })

    # 5. CBraMod Cross-Subject (当前)
    if 'cbramod' in current_results:
        per_subj = (
            current_results['cbramod'].get('per_subject_test_acc') or
            current_results['cbramod'].get('results', {}).get('per_subject_test_acc', {})
        )
        if per_subj:
            data_sources.append({
                'name': 'CBraMod (cross)',
                'model_type': 'cbramod',
                'training_type': 'cross-subject',
                'is_current': True,
                'is_cross_historical': False,
                'data': per_subj,
            })

    if not data_sources:
        log_plot.warning("No data sources for comparison plot")
        return

    # 创建 3 列布局
    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(1, 3, width_ratios=[2, 1, 1], wspace=0.25)

    ax_bar = fig.add_subplot(gs[0])
    ax_box = fig.add_subplot(gs[1])
    ax_stats = fig.add_subplot(gs[2])

    # =========================================================================
    # Panel 1: 柱状图
    # =========================================================================
    n_subjects = len(subjects)
    n_sources = len(data_sources)
    bar_width = 0.8 / n_sources
    x_base = np.arange(n_subjects)

    for i, source in enumerate(data_sources):
        x_positions = x_base + (i - (n_sources - 1) / 2) * bar_width

        accs = [source['data'].get(s, 0) for s in subjects]

        # 样式配置
        if source['is_current']:
            alpha = 1.0
            edgecolor = 'black'
            linewidth = 1.5
            hatch = ''
        elif source['is_cross_historical']:
            alpha = 0.4
            edgecolor = 'gray'
            linewidth = 0.5
            hatch = '...'  # 点状填充
        else:
            alpha = 0.4
            edgecolor = 'gray'
            linewidth = 0.5
            hatch = '///'  # 斜线填充

        bars = ax_bar.bar(
            x_positions, accs, bar_width,
            label=source['name'],
            color=colors[source['model_type']],
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=linewidth,
            hatch=hatch,
        )

        # 仅为当前运行添加数值标签
        if source['is_current']:
            for bar, val in zip(bars, accs):
                if val > 0:
                    ax_bar.text(
                        bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.01,
                        f'{val*100:.1f}',
                        ha='center', va='bottom', fontsize=6
                    )

    ax_bar.set_xlabel('Subject')
    ax_bar.set_ylabel('Test Accuracy')
    ax_bar.set_title(f'Cross-Subject Training Comparison\n({paradigm.title()} {task_type.title()})')
    ax_bar.set_xticks(x_base)
    ax_bar.set_xticklabels(subjects, rotation=45, ha='right')
    ax_bar.axhline(y=chance_level, color='gray', linestyle='--', alpha=0.5,
                   label=f'Chance ({chance_level*100:.1f}%)')
    ax_bar.set_ylim([0, 1.05])
    ax_bar.legend(loc='upper right', fontsize=7, ncol=2)

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
        accs = [source['data'].get(s, 0) for s in subjects if source['data'].get(s, 0) > 0]
        if accs:
            box_data.append(accs)
            # 缩短标签
            short_name = source['name'].replace('-subject', '').replace('(', '\n(')
            box_labels.append(short_name)
            box_colors.append(colors[source['model_type']])
            box_alphas.append(1.0 if source['is_current'] else 0.4)

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

        legend_elements = [
            Line2D([0], [0], color=median_color, linewidth=2, linestyle='-', label='Median'),
            Line2D([0], [0], color=mean_color, linewidth=2, linestyle=(0, (3, 2)), label='Mean')
        ]
        ax_box.legend(handles=legend_elements, loc='upper right', fontsize=7)

    ax_box.set_ylabel('Test Accuracy')
    ax_box.set_title('Distribution')
    ax_box.axhline(y=chance_level, color='gray', linestyle='--', alpha=0.5)
    ax_box.tick_params(axis='x', labelsize=7)

    # =========================================================================
    # Panel 3: 统计摘要表
    # =========================================================================
    ax_stats.axis('off')

    # 构建统计表数据
    table_data = []
    table_headers = ['Model', 'Type', 'Mean', 'Std', 'Med']

    for source in data_sources:
        accs = [source['data'].get(s, 0) for s in subjects if source['data'].get(s, 0) > 0]
        if accs:
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            median_acc = np.median(accs)

            model_name = source['model_type'].upper()
            type_name = 'Cross' if source['training_type'] == 'cross-subject' else 'Within'
            if not source['is_current'] and source['is_cross_historical']:
                type_name = 'Cross(H)'
            elif not source['is_current']:
                type_name = f'{type_name}(H)'

            table_data.append([
                model_name,
                type_name,
                f'{mean_acc*100:.1f}%',
                f'{std_acc*100:.1f}%',
                f'{median_acc*100:.1f}%',
            ])

    if table_data:
        table = ax_stats.table(
            cellText=table_data,
            colLabels=table_headers,
            loc='center',
            cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        # 设置表头样式
        for j in range(len(table_headers)):
            table[(0, j)].set_facecolor('#f0f0f0')
            table[(0, j)].set_text_props(fontweight='bold')

    ax_stats.set_title('Statistics Summary', pad=20)

    # 添加比较结果（如果有）
    if comparison:
        stats_text = f"\nStatistical Comparison:\n"
        stats_text += f"t-test p-value: {comparison.t_test_p:.4f}\n"
        stats_text += f"Wilcoxon p-value: {comparison.wilcoxon_p:.4f}\n"
        stats_text += f"Mean diff: {comparison.mean_diff*100:.2f}%"

        ax_stats.text(0.5, 0.05, stats_text, transform=ax_stats.transAxes,
                      ha='center', va='bottom', fontsize=8,
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    log_plot.info(f"Cross-subject comparison plot saved: {output_path}")
    plt.close()
