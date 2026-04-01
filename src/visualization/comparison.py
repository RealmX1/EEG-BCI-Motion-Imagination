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
from .plots import (
    CHANCE_LEVELS, annotate_bars_with_leaders, accuracy_ylim,
    separate_paired_labels, draw_label_with_leader,
)

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

    # 创建 2 行布局，第一行跨两列; 底部行加高以容纳正方形子图
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(2, 2, height_ratios=[1.0, 1.2], hspace=0.30, wspace=0.25)

    ax_bar = fig.add_subplot(gs[0, :])      # 顶部条形图（跨两列）
    ax_box = fig.add_subplot(gs[1, 0])      # 左下箱线图
    ax_scatter = fig.add_subplot(gs[1, 1])  # 右下配对散点图
    ax_box.set_box_aspect(1)
    ax_scatter.set_box_aspect(1)

    # =========================================================================
    # Panel 1: 条形图
    # =========================================================================
    n_subjects = len(subjects)
    n_sources = len(data_sources)
    bar_width = 0.8 / n_sources
    x_base = np.arange(n_subjects)

    bar_entries = []
    for i, source in enumerate(data_sources):
        x_positions = x_base + (i - (n_sources - 1) / 2) * bar_width

        # 按被试排序获取准确率
        result_by_subj = {r.subject_id: r.test_acc_majority for r in source.results}
        accs = [result_by_subj.get(s, 0) for s in subjects]

        alpha = 1.0 if source.is_current_run else 0.4
        edgecolor = 'black' if source.is_current_run else 'gray'
        linewidth = 1.5 if source.is_current_run else 0.5
        if source.hatch is not None:
            hatch = source.hatch
        else:
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
        bar_entries.append((bars, accs, source.is_current_run))

    annotate_bars_with_leaders(ax_bar, bar_entries)

    # data-driven y lower bound
    _all_accs = [a for _, accs, _ in bar_entries for a in accs if a > 0]
    _data_min = min(_all_accs) if _all_accs else None

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
    ax_bar.set_ylim(accuracy_ylim(task_type, data_min=_data_min))
    ax_bar.legend(loc='lower right', fontsize=8)

    # =========================================================================
    # Panel 2: 箱线图
    # =========================================================================
    median_color = 'black'
    mean_color = '#E63946'

    box_data = []
    box_labels = []
    box_colors = []
    box_alphas = []
    box_hatches = []

    for source in data_sources:
        accs = [r.test_acc_majority for r in source.results]
        if accs:
            box_data.append(accs)
            box_labels.append(source.label)
            box_colors.append(colors[source.model_type])
            box_alphas.append(1.0 if source.is_current_run else 0.4)
            if source.hatch is not None:
                box_hatches.append(source.hatch)
            else:
                box_hatches.append('' if source.is_current_run else '///')

    if box_data:
        # 交错标签：位置 2,4,... 下移一行，避免重叠
        for i in range(len(box_labels)):
            if i % 2 == 1:
                box_labels[i] = '\n' + box_labels[i]

        bp = ax_box.boxplot(
            box_data, labels=box_labels, patch_artist=True,
            showmeans=True, meanline=True,
            meanprops={'color': mean_color, 'linewidth': 2, 'linestyle': (0, (3, 2))}
        )

        for patch, color, alpha, hatch in zip(bp['boxes'], box_colors, box_alphas, box_hatches):
            patch.set_facecolor(color)
            patch.set_alpha(alpha)
            patch.set_hatch(hatch)

        for median in bp['medians']:
            median.set_color(median_color)
            median.set_linewidth(2)

        # 添加统计标注 (leader starts at box right edge)
        for i, (source, accs_list) in enumerate(zip(data_sources, box_data)):
            mean_val = np.mean(accs_list)
            median_val = np.median(accs_list)
            box_right = max(v[0] for v in bp['boxes'][i].get_path().vertices)
            fontweight = 'bold' if source.is_current_run else 'normal'
            adj_mean, adj_med = separate_paired_labels(mean_val, median_val, min_gap=0.02)
            draw_label_with_leader(
                ax_box, mean_val, adj_mean, box_right,
                f'{mean_val*100:.1f}', color=mean_color, fontsize=7,
                fontweight=fontweight)
            draw_label_with_leader(
                ax_box, median_val, adj_med, box_right,
                f'{median_val*100:.1f}', color=median_color, fontsize=7,
                fontweight=fontweight)

        legend_elements = [
            Line2D([0], [0], color=median_color, linewidth=2, linestyle='-', label='Median'),
            Line2D([0], [0], color=mean_color, linewidth=2, linestyle=(0, (3, 2)), label='Mean')
        ]
        ax_box.legend(handles=legend_elements, loc='lower right', fontsize=7)

    ax_box.set_ylabel('Test Accuracy')
    ax_box.set_title('Accuracy Distribution')
    ax_box.axhline(y=chance_level, color='gray', linestyle='--', alpha=0.5)
    ax_box.set_ylim(accuracy_ylim(task_type, data_min=_data_min, top_pad=0.08))

    # =========================================================================
    # Panel 3: 配对对比散点图（支持双配对：当前 vs 历史）
    # =========================================================================
    eegnet_sources = [s for s in data_sources if s.model_type == 'eegnet']
    cbramod_sources = [s for s in data_sources if s.model_type == 'cbramod']

    # 分离当前和历史数据源
    eegnet_current = next((s for s in eegnet_sources if s.is_current_run), None)
    eegnet_hist = next((s for s in eegnet_sources if not s.is_current_run), None)
    cbramod_current = next((s for s in cbramod_sources if s.is_current_run), None)
    cbramod_hist_sources = [s for s in cbramod_sources if not s.is_current_run]

    # 使用 EEGNet 作为 X 轴基准（优先使用当前数据）
    eegnet_baseline = eegnet_current or eegnet_hist

    # 历史数据源 marker 样式（按顺序: 被试内=方块, 跨被试=菱形, ...）
    hist_markers = ['s', 'D', '^', 'v']

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

                current_label = cbramod_current.label.replace('CBRAMOD', 'CBraMod')
                ax_scatter.scatter(eegnet_accs, cbramod_accs, s=100, alpha=0.9,
                                   c='#E94F37', label=current_label,
                                   edgecolors='black', linewidths=1)

                # 为当前运行添加被试标签
                for i, subj in enumerate(common):
                    ax_scatter.annotate(subj, (eegnet_accs[i], cbramod_accs[i]),
                                        xytext=(5, 5), textcoords='offset points', fontsize=7)
                has_any_pair = True

        # 绘制所有历史配对：各历史 CBraMod vs EEGNet
        for idx, hist_source in enumerate(cbramod_hist_sources):
            hist_by_subj = {r.subject_id: r.test_acc_majority for r in hist_source.results}
            common_hist = sorted(set(eegnet_by_subj.keys()) & set(hist_by_subj.keys()))

            if common_hist:
                eegnet_accs_hist = [eegnet_by_subj[s] for s in common_hist]
                cbramod_accs_hist = [hist_by_subj[s] for s in common_hist]
                all_accs.extend(eegnet_accs_hist + cbramod_accs_hist)

                marker = hist_markers[idx % len(hist_markers)]
                hist_label = hist_source.label.replace('CBRAMOD', 'CBraMod')
                sc = ax_scatter.scatter(eegnet_accs_hist, cbramod_accs_hist, s=80, alpha=0.5,
                                        c='#E94F37', label=hist_label,
                                        edgecolors='gray', linewidths=0.5, marker=marker)
                hist_hatch = hist_source.hatch if hist_source.hatch is not None else '///'
                sc.set_hatch(hist_hatch)
                has_any_pair = True

        if has_any_pair and all_accs:
            lims = [min(all_accs) - 0.05, max(all_accs) + 0.05]
            ax_scatter.plot(lims, lims, 'k--', alpha=0.5, label='Equal')
            ax_scatter.set_xlim(lims)
            ax_scatter.set_ylim(lims)
            ax_scatter.set_xlabel(f'{eegnet_baseline.label} Accuracy')
            ax_scatter.set_ylabel('CBraMod Accuracy')
            ax_scatter.legend(loc='lower right', fontsize=7)
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
    _data_min = min(eegnet_accs + cbramod_accs) if (eegnet_accs or cbramod_accs) else None

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
    ax1.set_ylim(accuracy_ylim(task_type, data_min=_data_min))
    ax1.axhline(y=chance_level, color='gray', linestyle='--', alpha=0.5,
                label=f'Chance ({chance_level*100:.1f}%)')

    annotate_bars_with_leaders(
        ax1,
        [(bars1, eegnet_accs, True), (bars2, cbramod_accs, True)],
    )

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
    ax2.set_ylim(accuracy_ylim(task_type, data_min=_data_min, top_pad=0.08))

    eegnet_mean = np.mean(eegnet_accs)
    eegnet_median = np.median(eegnet_accs)
    cbramod_mean = np.mean(cbramod_accs)
    cbramod_median = np.median(cbramod_accs)

    for box_i, (bm, bmed) in enumerate([(eegnet_mean, eegnet_median), (cbramod_mean, cbramod_median)]):
        box_right = max(v[0] for v in bp['boxes'][box_i].get_path().vertices)
        adj_mean, adj_med = separate_paired_labels(bm, bmed, min_gap=0.02)
        draw_label_with_leader(
            ax2, bm, adj_mean, box_right,
            f'{bm*100:.1f}', color=mean_color, fontsize=7, fontweight='bold')
        draw_label_with_leader(
            ax2, bmed, adj_med, box_right,
            f'{bmed*100:.1f}', color=median_color, fontsize=7, fontweight='bold')

    legend_elements = [
        Line2D([0], [0], color=median_color, linewidth=2, linestyle='-', label='Median'),
        Line2D([0], [0], color=mean_color, linewidth=2, linestyle=(0, (3, 2)), label='Mean')
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=7)

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


def plot_unified_comparison(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
    title: str = "Unified Model Comparison",
) -> 'plt.Figure':
    """
    生成统一多任务模型对比图（3+1+1x3 布局，共 8 个子图）.

    布局:
    +--------------------------------------------------+
    | Row 1: Binary 每被试准确率对比（全宽）               |
    +--------------------------------------------------+
    | Row 2: Ternary 每被试准确率对比（全宽）              |
    +--------------------------------------------------+
    | Row 3: Quaternary 每被试准确率对比（全宽）           |
    +--------------------------------------------------+
    | Row 4: 分组柱状图 — 三任务均值对比（全宽）           |
    +--------------------------------------------------+
    | Row 5: [Binary 配对] [Ternary 配对] [Quaternary 配对] |
    +--------------------------------------------------+

    Args:
        results: 模型名 -> 统一结果字典，结构:
            {
                'cbramod': {
                    'subtask_results': {
                        'binary': {'accuracy': 0.85, ...},
                        'ternary': {'accuracy': 0.70, ...},
                        'quaternary': {'accuracy': 0.55, ...},
                        'mean_accuracy': 0.70,
                    },
                    'per_subject': {
                        'S01': {
                            'binary': {'accuracy': 0.80},
                            'ternary': {'accuracy': 0.65},
                            'quaternary': {'accuracy': 0.50},
                        },
                        ...
                    }
                },
                'eegnet': { ... }
            }
        save_path: 输出文件路径（可选）
        title: 图表标题

    Returns:
        matplotlib Figure 对象
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        log_plot.warning("matplotlib not installed, skipping unified comparison plot")
        return None

    colors = MODEL_COLORS
    subtasks = ['binary', 'ternary', 'quaternary']
    subtask_chance = {t: CHANCE_LEVELS.get(t, 0.5) for t in subtasks}

    model_names = list(results.keys())
    n_models = len(model_names)

    # 检查是否有 per_subject 数据
    has_per_subject = any(
        'per_subject' in results[m] and results[m]['per_subject']
        for m in model_names
    )

    # =========================================================================
    # 根据数据完备性决定布局
    # =========================================================================
    if has_per_subject:
        # 完整 5 行布局: 3 per-subject + 1 grouped bar + 1x3 pairwise
        fig = plt.figure(figsize=(18, 28))
        gs = GridSpec(
            5, 3,
            height_ratios=[1.0, 1.0, 1.0, 0.9, 0.9],
            hspace=0.35, wspace=0.30,
        )
    else:
        # 仅 Row 4: 只有均值柱状图
        fig = plt.figure(figsize=(12, 5))
        gs = GridSpec(1, 1)

    # =========================================================================
    # Row 1-3: 每被试准确率对比（每个 subtask 一行）
    # =========================================================================
    if has_per_subject:
        for row_idx, subtask in enumerate(subtasks):
            ax = fig.add_subplot(gs[row_idx, :])
            chance = subtask_chance[subtask]

            # 收集该 subtask 下所有被试（跨模型取并集，排除 0 trial 被试）
            all_subjects = set()
            for m in model_names:
                per_subj = results[m].get('per_subject', {})
                for sid, task_data in per_subj.items():
                    if subtask in task_data:
                        acc_val = task_data[subtask].get('accuracy')
                        n_trials = task_data[subtask].get('n_trials')
                        # 排除 quaternary 中 0 trial 的被试
                        if n_trials is not None and n_trials == 0:
                            continue
                        if acc_val is not None:
                            all_subjects.add(sid)

            subjects = sorted(all_subjects)
            if not subjects:
                ax.text(
                    0.5, 0.5,
                    f'No per-subject data for {subtask}',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12,
                )
                ax.set_title(f"{subtask.capitalize()} — Per-Subject Test Accuracy")
                continue

            n_subjects = len(subjects)
            bar_width = 0.8 / max(n_models, 1)
            x_base = np.arange(n_subjects)

            model_accs_list = []  # 用于后面计算 mean±std
            bar_entries = []

            for i, m in enumerate(model_names):
                per_subj = results[m].get('per_subject', {})
                x_positions = x_base + (i - (n_models - 1) / 2) * bar_width

                accs = []
                for sid in subjects:
                    task_data = per_subj.get(sid, {}).get(subtask, {})
                    acc_val = task_data.get('accuracy', 0)
                    accs.append(acc_val if acc_val is not None else 0)

                model_accs_list.append(accs)

                # 确定颜色：优先用 MODEL_COLORS，否则自动分配
                color = colors.get(m, f'C{i}')

                bars = ax.bar(
                    x_positions, accs, bar_width,
                    label=m.upper() if m in colors else m,
                    color=color, alpha=0.85,
                    edgecolor='black', linewidth=0.8,
                )
                bar_entries.append((bars, accs, True))

            annotate_bars_with_leaders(ax, bar_entries, fontsize=6)

            # Chance level 参考线
            ax.axhline(
                y=chance, color='gray', linestyle='--', alpha=0.5,
                label=f'Chance ({chance * 100:.1f}%)',
            )

            # Mean±std 标注
            for i, m in enumerate(model_names):
                accs = model_accs_list[i]
                valid_accs = [a for a in accs if a > 0]
                if valid_accs:
                    mean_val = np.mean(valid_accs)
                    std_val = np.std(valid_accs)
                    color = colors.get(m, f'C{i}')
                    ax.text(
                        0.98, 0.95 - i * 0.07,
                        f'{m.upper()}: {mean_val * 100:.1f} ± {std_val * 100:.1f}%',
                        transform=ax.transAxes,
                        ha='right', va='top', fontsize=8,
                        color=color, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                    )

            _sub_accs = [a for _, accs, _ in bar_entries for a in accs if a > 0]
            _sub_min = min(_sub_accs) if _sub_accs else None

            ax.set_xlabel('Subject')
            ax.set_ylabel('Test Accuracy')
            ax.set_title(f"{subtask.capitalize()} — Per-Subject Test Accuracy", fontsize=11)
            ax.set_xticks(x_base)
            ax.set_xticklabels(subjects, rotation=45, ha='right', fontsize=7)
            ax.set_ylim(accuracy_ylim(subtask, data_min=_sub_min))
            ax.legend(loc='lower right', fontsize=8)

    # =========================================================================
    # Row 4: 分组柱状图 — 三任务均值对比
    # =========================================================================
    if has_per_subject:
        ax_bar = fig.add_subplot(gs[3, :])
    else:
        ax_bar = fig.add_subplot(gs[0, 0])

    # 背景色带（pastel task group shading）
    task_bg_colors = {
        'binary': '#D6EAF8',      # 淡蓝
        'ternary': '#D5F5E3',     # 淡绿
        'quaternary': '#FDEBD0',  # 淡橙
    }

    n_tasks = len(subtasks)
    x_tasks = np.arange(n_tasks)
    group_width = 0.8
    bar_width_grouped = group_width / max(n_models, 1)

    # 绘制背景色带
    for j, subtask in enumerate(subtasks):
        ax_bar.axvspan(
            j - 0.45, j + 0.45,
            facecolor=task_bg_colors.get(subtask, '#F0F0F0'),
            alpha=0.4, zorder=0,
        )

    row4_bar_entries = []
    row4_yerr_entries = []
    for i, m in enumerate(model_names):
        subtask_results = results[m].get('subtask_results', {})
        x_positions = x_tasks + (i - (n_models - 1) / 2) * bar_width_grouped

        means = []
        stds = []
        for subtask in subtasks:
            task_res = subtask_results.get(subtask, {})
            acc = task_res.get('accuracy', 0)
            means.append(acc if acc is not None else 0)
            # 如果有 per_subject 数据，从中计算 std
            per_subj = results[m].get('per_subject', {})
            task_accs = []
            for sid_data in per_subj.values():
                if subtask in sid_data:
                    a = sid_data[subtask].get('accuracy')
                    if a is not None:
                        task_accs.append(a)
            stds.append(np.std(task_accs) if task_accs else 0)

        color = colors.get(m, f'C{i}')

        bars = ax_bar.bar(
            x_positions, means, bar_width_grouped,
            label=m.upper() if m in colors else m,
            color=color, alpha=0.85,
            edgecolor='black', linewidth=1.2,
            yerr=stds, capsize=5, error_kw={'linewidth': 1.2},
        )
        row4_bar_entries.append((bars, means, True))
        row4_yerr_entries.append(stds)

    annotate_bars_with_leaders(
        ax_bar, row4_bar_entries,
        fmt='{:.1f}%', fontsize=9,
        yerr_entries=row4_yerr_entries,
    )

    # 各任务的 chance level 参考线
    for j, subtask in enumerate(subtasks):
        chance = subtask_chance[subtask]
        ax_bar.plot(
            [j - 0.45, j + 0.45], [chance, chance],
            color='gray', linestyle=':', linewidth=1.2, alpha=0.7,
        )
        # 仅在第一个任务标注 label
        if j == 0:
            ax_bar.text(
                j + 0.47, chance, 'chance', fontsize=7,
                color='gray', va='center',
            )

    ax_bar.set_xticks(x_tasks)
    ax_bar.set_xticklabels(
        [t.capitalize() for t in subtasks], fontsize=11, fontweight='bold',
    )
    _row4_accs = [a for _, means, _ in row4_bar_entries for a in means if a > 0]
    _row4_min = min(_row4_accs) if _row4_accs else None
    ax_bar.set_ylabel('Mean Test Accuracy', fontsize=11)
    ax_bar.set_title('Overall Accuracy by Task', fontsize=12)
    ax_bar.set_ylim(accuracy_ylim('quaternary', data_min=_row4_min))
    ax_bar.legend(loc='lower right', fontsize=9)

    # =========================================================================
    # Row 5: 配对对比散点图（1x3，各 subtask 一个）
    # =========================================================================
    if has_per_subject and n_models >= 2:
        # 默认取前两个模型做配对
        m1, m2 = model_names[0], model_names[1]

        for col_idx, subtask in enumerate(subtasks):
            ax_sc = fig.add_subplot(gs[4, col_idx])
            chance = subtask_chance[subtask]

            per_subj_m1 = results[m1].get('per_subject', {})
            per_subj_m2 = results[m2].get('per_subject', {})

            # 找共同被试（且该 subtask 都有数据、非 0 trial）
            common_subjects = []
            for sid in sorted(set(per_subj_m1.keys()) & set(per_subj_m2.keys())):
                d1 = per_subj_m1[sid].get(subtask, {})
                d2 = per_subj_m2[sid].get(subtask, {})
                a1 = d1.get('accuracy')
                a2 = d2.get('accuracy')
                n1 = d1.get('n_trials')
                n2 = d2.get('n_trials')
                # 跳过 0 trial 被试
                if (n1 is not None and n1 == 0) or (n2 is not None and n2 == 0):
                    continue
                if a1 is not None and a2 is not None:
                    common_subjects.append(sid)

            if not common_subjects:
                ax_sc.text(
                    0.5, 0.5,
                    'No common subjects\nfor paired comparison',
                    ha='center', va='center', transform=ax_sc.transAxes,
                    fontsize=10,
                )
                ax_sc.set_title(f"{subtask.capitalize()} Pairwise", fontsize=10)
                continue

            accs_m1 = [
                per_subj_m1[s][subtask]['accuracy'] for s in common_subjects
            ]
            accs_m2 = [
                per_subj_m2[s][subtask]['accuracy'] for s in common_subjects
            ]

            # 散点
            color_m1 = colors.get(m1, 'C0')
            color_m2 = colors.get(m2, 'C1')
            ax_sc.scatter(
                accs_m1, accs_m2, s=80, alpha=0.8,
                c=colors.get(m2, '#E94F37'),
                edgecolors='black', linewidths=0.8,
            )

            # 被试标签
            for k, sid in enumerate(common_subjects):
                ax_sc.annotate(
                    sid, (accs_m1[k], accs_m2[k]),
                    xytext=(4, 4), textcoords='offset points', fontsize=6,
                )

            # 对角线 y=x
            all_accs = accs_m1 + accs_m2
            lims = [
                max(0, min(all_accs) - 0.05),
                min(1.0, max(all_accs) + 0.05),
            ]
            ax_sc.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
            ax_sc.set_xlim(lims)
            ax_sc.set_ylim(lims)

            # 统计胜/平/负
            wins = sum(1 for a, b in zip(accs_m2, accs_m1) if a > b)
            ties = sum(1 for a, b in zip(accs_m2, accs_m1) if abs(a - b) < 1e-6)
            losses = sum(1 for a, b in zip(accs_m2, accs_m1) if a < b)

            m2_label = m2.upper() if m2 in colors else m2
            m1_label = m1.upper() if m1 in colors else m1
            ax_sc.text(
                0.05, 0.95,
                f'{m2_label} wins: {wins}\nTies: {ties}\n{m1_label} wins: {losses}',
                transform=ax_sc.transAxes,
                ha='left', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
            )

            ax_sc.set_xlabel(f'{m1_label} Accuracy', fontsize=9)
            ax_sc.set_ylabel(f'{m2_label} Accuracy', fontsize=9)
            ax_sc.set_title(f"{subtask.capitalize()} Pairwise", fontsize=10)
            ax_sc.set_aspect('equal', adjustable='box')

    elif has_per_subject and n_models < 2:
        # 只有一个模型，无法做配对对比，跳过 Row 5
        for col_idx in range(3):
            ax_sc = fig.add_subplot(gs[4, col_idx])
            ax_sc.text(
                0.5, 0.5,
                'Single model\n(no pairwise comparison)',
                ha='center', va='center', transform=ax_sc.transAxes,
                fontsize=10,
            )
            ax_sc.set_title(f"{subtasks[col_idx].capitalize()} Pairwise", fontsize=10)

    # =========================================================================
    # 全局标题与保存
    # =========================================================================
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        log_plot.info(f"Unified comparison plot saved: {save_path}")

    return fig
