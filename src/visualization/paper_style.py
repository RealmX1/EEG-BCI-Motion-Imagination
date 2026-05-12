"""
论文图表样式中央注册表 (paper figure style registry).

本模块是所有论文图表 (paper figures) 颜色、字号与样式辅助函数的**单一事实
来源** (single source of truth)。新写图表脚本时应**仅从本模块**导入颜色与字号，
避免在脚本中散落硬编码 (hard-coded) 的色值与字号。

视觉语言对齐 Ding et al. 2025 *Nat Commun* "EEG-based brain-computer interface
enables real-time robotic hand control at individual finger level"。详见
`paper/style_guide_ding2025.md` 与 `C:/Users/zhang/.claude/plans/
analyze-the-plotting-style-shiny-toast.md` 中提取的 §1.1–§1.6 规范。

用法示例 (usage example):

    from src.visualization.paper_style import (
        PAPER_COLORS,
        FONT_SIZES,
        apply_paper_style,
        chance_line,
        add_panel_label,
        add_stat_bracket,
        violin_with_subjects,
    )

    fig, ax = plt.subplots(figsize=paper_figsize(rows=1, cols=1))
    violin_with_subjects(ax, {'Base': base_acc, 'Fine-tuned': ft_acc},
                         x_labels=['Base', 'Fine-tuned'],
                         colors=[PAPER_COLORS['baseline_blue'],
                                 PAPER_COLORS['finetuned_orange']])
    chance_line(ax, level=0.5)
    add_panel_label(ax, 'A')
    apply_paper_style(fig=fig)         # 收尾统一字号/去除顶右轴线
    fig.savefig('example.png', dpi=300)

注意 (notes):
    - 模型色 (`cbramod`, `eegnet`) 复用 `src/config/constants.py` 的
      `MODEL_COLORS`，请不要在此重复定义。
    - 论文配色键 (`baseline_blue`, `finetuned_orange`, `chance_red` 等) 是
      Ding 2025 的精确镜像，新写图应优先使用。
    - 不在此模块管理 provenance footer（用户明确排除）。
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import matplotlib.figure
import matplotlib.axes
import matplotlib.patches as mpatches

from src.config.constants import MODEL_COLORS

# ============================================================================
# 颜色注册表 (Color Registry)
# ============================================================================

PAPER_COLORS: Dict[str, str] = {
    # ---- 模型色 (复用 MODEL_COLORS，不重复定义) ----
    **MODEL_COLORS,  # cbramod, eegnet

    # ---- 论文配色 (Ding et al. 2025 精确镜像) ----
    # 用法对应:
    #   baseline_blue / finetuned_orange  → Fig 1A/B / 3A/B 配对小提琴
    #   session1_light / session2_deep    → Fig 1C / 3C 双 session 条形
    #   fbcsp_green                       → Fig 6A/B 与 EEGNet 对照的 boxplot
    #   chance_red                        → 随机基线水平线 (论文每张准确率图)
    #   median_gray                       → 各条件中位数参考虚线
    #   subject_dot                       → 叠加在汇总统计上的被试散点
    'baseline_blue':    '#9DBED7',  # 柔和蓝 (soft pastel blue) — 基线
    'finetuned_orange': '#F2B670',  # 暖橙   (warm orange)      — 微调
    'session1_light':   '#F4C68C',  # 浅橙 — Session 1
    'session2_deep':    '#D67A30',  # 深橙 — Session 2
    'fbcsp_green':      '#A2C49A',  # 鼠尾草绿 — FBCSP boxplot
    'chance_red':       '#D9534F',  # 红虚线 — 随机基线
    'median_gray':      '#888888',  # 中位数参考虚线
    'subject_dot':      '#222222',  # 被试散点近黑色

    # ---- 通道选择方法色 (method colors) — 既有方案保留 ----
    'fdr':              '#d62728',  # 红
    'band_power':       '#2ca02c',  # 绿
    'csp':              '#ff7f0e',  # 橙
    'attention':        '#1f77b4',  # 蓝
    'negative_control': '#7f7f7f',  # 中性灰 — 控制项，区分于 method 色但保留 visibility

    # ---- 范式色 (paradigm colors) ----
    'imagery':   '#5e35b1',   # 深紫 — Motor Imagery
    'motor':     '#00897b',   # 蓝绿 — Motor Execution
    'movement':  '#00897b',   # alias
    'execution': '#fb8c00',   # alt label

    # ---- 工具色 (utility colors) ----
    # 注意: 'chance_level' 改为论文红 (#D9534F)，原 'gray' 已废弃。
    # 仍保留 'chance_level' 键以保 backward-compat。
    'chance_level':   '#D9534F',
    'mean_marker':    'black',
    'secondary_blue': '#1976D2',
    'delta_neg':      '#c0392b',
    'delta_pos':      '#27ae60',
}


# ============================================================================
# 字号注册表 (Font Size Registry)
# ============================================================================
# 当前 codebase 多采用 (12-15) × (5-12) 的 figsize；为了让现有图表不缩字，
# 默认字号保留为论文 1.2× 的尺寸。当模块切换到 paper_figsize() (~7 in 宽) 时，
# 同时切换到 FONT_SIZES_TIGHT。

FONT_SIZES: Dict[str, int] = {
    'title':       12,
    'axis_label':  11,
    'tick':        10,
    'annotation':  9,
    'legend':      9,
    'footer':      7,
    'panel_label': 14,  # 'A.', 'B.', 'C.' panel-letter (右上角粗体)
}

# 论文紧凑模式字号 (用于 ~7 in 宽的两栏论文 figure；与 FONT_SIZES 等价但下调 1pt)
FONT_SIZES_TIGHT: Dict[str, int] = {
    'title':       11,
    'axis_label':  10,
    'tick':        9,
    'annotation':  9,
    'legend':      8,
    'footer':      7,
    'panel_label': 14,
}

TITLE_WEIGHT: str = 'bold'
LABEL_WEIGHT: str = 'normal'


# ============================================================================
# Figsize 辅助 (target: Nature/Nat Commun two-column ≈ 7.0 in 宽)
# ============================================================================

def paper_figsize(
    rows: int = 1,
    cols: int = 1,
    *,
    width_in: float = 7.0,
    row_height_in: float = 3.5,
) -> Tuple[float, float]:
    """返回针对论文双栏宽度 (~7.0 in) 的 (width, height)。

    Args:
        rows: 图表行数 (subplot grid rows)。
        cols: 图表列数；不影响宽度，仅影响高度估计。
        width_in: 总宽度，默认 7.0 in (Nat Commun 双栏)。
        row_height_in: 每行高度，默认 3.5 in。

    Returns:
        (width, height) 浮点二元组，可直接传给 ``plt.subplots(figsize=...)``。
    """
    return (width_in, max(2.5, rows * row_height_in))


# ============================================================================
# 样式应用函数 (Style Application)
# ============================================================================

def _apply_to_ax(
    ax: matplotlib.axes.Axes,
    *,
    sizes: Dict[str, int] = FONT_SIZES,
    despine: bool = True,
) -> None:
    """对单个 ax 应用论文字号 + 移除顶/右轴线 (helper, idempotent)."""
    # 标题：仅在标题文本非空时设置
    title_obj = ax.title
    if title_obj is not None:
        title_text = title_obj.get_text() if hasattr(title_obj, 'get_text') else ''
        if title_text:
            try:
                title_obj.set_fontsize(sizes['title'])
                title_obj.set_fontweight(TITLE_WEIGHT)
            except Exception:
                pass

    try:
        xlabel_obj = ax.xaxis.label
        if xlabel_obj is not None and xlabel_obj.get_text():
            xlabel_obj.set_fontsize(sizes['axis_label'])
            xlabel_obj.set_fontweight(LABEL_WEIGHT)
    except Exception:
        pass

    try:
        ylabel_obj = ax.yaxis.label
        if ylabel_obj is not None and ylabel_obj.get_text():
            ylabel_obj.set_fontsize(sizes['axis_label'])
            ylabel_obj.set_fontweight(LABEL_WEIGHT)
    except Exception:
        pass

    try:
        ax.tick_params(axis='both', which='major', labelsize=sizes['tick'], width=0.8)
    except Exception:
        pass

    try:
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(sizes['legend'])
    except Exception:
        pass

    # 论文风格: 去除顶/右轴线，留下左/下轴线
    if despine:
        try:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.8)
            ax.spines['bottom'].set_linewidth(0.8)
        except Exception:
            pass


def apply_paper_style(
    ax: Optional[matplotlib.axes.Axes] = None,
    fig: Optional[matplotlib.figure.Figure] = None,
    *,
    tight: bool = False,
    despine: bool = True,
) -> None:
    """对图表应用论文统一字号、字重与轴线样式 (idempotent).

    Args:
        ax: 单个 Axes；若提供，仅对该 ax 应用。
        fig: Figure；若提供，遍历 ``fig.axes`` 对每个 ax 应用。
        tight: True 则使用 :data:`FONT_SIZES_TIGHT` (论文 ~7 in 宽优化)。
        despine: True 则隐藏 top / right 轴线 (论文风格)。

    若 ``ax`` 与 ``fig`` 同时提供，两者都会被处理；若都为 None，函数静默返回。
    """
    sizes = FONT_SIZES_TIGHT if tight else FONT_SIZES
    if fig is not None:
        for sub_ax in fig.axes:
            _apply_to_ax(sub_ax, sizes=sizes, despine=despine)
    if ax is not None:
        _apply_to_ax(ax, sizes=sizes, despine=despine)


# ============================================================================
# 参考线辅助 (chance / median reference lines)
# ============================================================================

def chance_line(
    ax: matplotlib.axes.Axes,
    level: float,
    *,
    label: Optional[str] = None,
    linewidth: float = 1.0,
    alpha: float = 0.85,
) -> None:
    """在 ax 上画论文风格的红色虚线随机基线 (Ding 2025 §1.6 标准样式)."""
    if label is None:
        label = f'Chance ({level * 100:.1f}%)'
    ax.axhline(
        y=level,
        color=PAPER_COLORS['chance_red'],
        linestyle='--',
        linewidth=linewidth,
        alpha=alpha,
        label=label,
        zorder=1,
    )


def median_reference_line(
    ax: matplotlib.axes.Axes,
    value: float,
    *,
    label: Optional[str] = None,
    linewidth: float = 0.8,
    alpha: float = 0.7,
) -> None:
    """画灰色虚线表示 per-condition 中位数参考 (Ding 2025 Fig 1A/B / 3A/B 风格)."""
    ax.axhline(
        y=value,
        color=PAPER_COLORS['median_gray'],
        linestyle='--',
        linewidth=linewidth,
        alpha=alpha,
        label=label,
        zorder=1,
    )


# ============================================================================
# Panel-letter 辅助 ('A.', 'B.', 'C.' …)
# ============================================================================

def add_panel_label(
    ax: matplotlib.axes.Axes,
    letter: str,
    *,
    x: float = -0.12,
    y: float = 1.05,
    fontsize: Optional[int] = None,
) -> None:
    """在 ax 左上方 (轴外) 添加论文风格的粗体 panel 标签。

    Args:
        ax: 目标 Axes。
        letter: 字母 ('A', 'B', 'C' …) 或带句点的形式 ('A.')。
        x, y: Axes 坐标系下的位置 (default (-0.12, 1.05) → 轴框左上外)。
        fontsize: 覆盖默认 panel-label 字号。
    """
    text = letter if letter.endswith('.') else f'{letter}.'
    ax.text(
        x, y, text,
        transform=ax.transAxes,
        fontsize=fontsize or FONT_SIZES['panel_label'],
        fontweight='bold',
        va='top', ha='left',
    )


# ============================================================================
# 统计括号 (significance brackets)
# ============================================================================

def _p_to_symbol(p_value: float, *, hash_marks: bool = False) -> str:
    """根据 p 值返回 ``'***'``/`'**'`/`'*'`/`'n.s.'` 或对应 hash marks."""
    sym = '#' if hash_marks else '*'
    if p_value < 0.001:
        return sym * 3
    if p_value < 0.01:
        return sym * 2
    if p_value < 0.05:
        return sym
    return 'n.s.'


def add_stat_bracket(
    ax: matplotlib.axes.Axes,
    x1: float,
    x2: float,
    y: float,
    p_value: float,
    *,
    hash_marks: bool = False,
    height: float = 0.015,
    linewidth: float = 1.0,
    fontsize: Optional[int] = None,
    color: str = 'black',
) -> None:
    """在 ax 上画一段论文风格的细线显著性括号 + 顶部符号。

    Args:
        ax: 目标 Axes。
        x1, x2: 比较的两个 x 位置 (data coords)。
        y: 括号底部高度 (data coords)。
        p_value: 用于挑选符号 (***/.../n.s.) 的 p 值。
        hash_marks: True 则用 ``###`` 表示 ANOVA 主效应；False 用 ``***`` 表示
            post-hoc / Wilcoxon (Ding 2025 §1.5 约定)。
        height: 括号竖向短线高度 (data coords units of y)。
        linewidth: 线宽。
        fontsize: 覆盖默认 annotation 字号。
        color: 括号颜色，默认 black。
    """
    sym = _p_to_symbol(p_value, hash_marks=hash_marks)
    bar_y_top = y + height
    ax.plot([x1, x1, x2, x2], [y, bar_y_top, bar_y_top, y],
            color=color, linewidth=linewidth, clip_on=False)
    ax.text((x1 + x2) / 2, bar_y_top, sym,
            ha='center', va='bottom',
            fontsize=fontsize or FONT_SIZES['annotation'],
            color=color)


# ============================================================================
# Boxplot outlier 钻石 (paper convention: diamond markers for outliers)
# ============================================================================

def style_boxplot_outliers(
    box_dict: dict,
    *,
    marker: str = 'D',
    markerfacecolor: str = 'black',
    markersize: float = 4.0,
    markeredgecolor: str = 'black',
    markeredgewidth: float = 0.5,
) -> None:
    """对 ``ax.boxplot()`` 返回字典的 ``fliers`` 应用钻石标记 (Ding 2025 §1.3)."""
    for flier in box_dict.get('fliers', []):
        flier.set_marker(marker)
        flier.set_markerfacecolor(markerfacecolor)
        flier.set_markersize(markersize)
        flier.set_markeredgecolor(markeredgecolor)
        flier.set_markeredgewidth(markeredgewidth)


# ============================================================================
# Violin + 被试散点 (paired violin with overlaid per-subject dots)
# ============================================================================

def violin_with_subjects(
    ax: matplotlib.axes.Axes,
    data: Dict[str, np.ndarray],
    *,
    x_labels: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
    jitter: float = 0.08,
    dot_size: float = 12.0,
    dot_alpha: float = 0.7,
    show_median: bool = True,
    paired: bool = False,
) -> None:
    """画论文风格的 violin + 被试散点 (Ding 2025 Fig 1A/B / 3A/B 标准面板).

    Args:
        ax: 目标 Axes。
        data: ``{label: 1D array of subject values}``。
        x_labels: x 轴 tick 标签，默认 ``data.keys()``。
        colors: 每个 violin 的颜色；默认从 paper palette 循环
            ``[baseline_blue, finetuned_orange, fbcsp_green, ...]``。
        jitter: 被试散点的水平抖动幅度 (axes coords units of x)。
        dot_size: 被试散点尺寸 (matplotlib scatter ``s``)。
        dot_alpha: 散点透明度。
        show_median: 在 violin 内部画黑色中位数横线。
        paired: True 时, 同一被试在不同 violin 间用细线相连 (paired plot)。
    """
    keys = list(data.keys())
    if x_labels is None:
        x_labels = keys
    if colors is None:
        default_palette = [
            PAPER_COLORS['baseline_blue'],
            PAPER_COLORS['finetuned_orange'],
            PAPER_COLORS['fbcsp_green'],
            PAPER_COLORS['session2_deep'],
        ]
        colors = [default_palette[i % len(default_palette)] for i in range(len(keys))]

    positions = np.arange(len(keys), dtype=float)
    arrays = [np.asarray(data[k], dtype=float) for k in keys]
    arrays = [a[~np.isnan(a)] for a in arrays]

    # Violin bodies
    parts = ax.violinplot(
        arrays,
        positions=positions,
        widths=0.7,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, color in zip(parts['bodies'], colors):
        body.set_facecolor(color)
        body.set_edgecolor('black')
        body.set_linewidth(0.8)
        body.set_alpha(0.85)

    # Per-subject scatter (overlaid, near-black dots)
    rng = np.random.default_rng(seed=0)
    for pos, arr in zip(positions, arrays):
        if len(arr) == 0:
            continue
        offsets = rng.uniform(-jitter, jitter, size=len(arr))
        ax.scatter(
            pos + offsets, arr,
            s=dot_size,
            color=PAPER_COLORS['subject_dot'],
            alpha=dot_alpha,
            edgecolor='none',
            zorder=4,
        )

    # Median bar inside violin
    if show_median:
        for pos, arr in zip(positions, arrays):
            if len(arr) == 0:
                continue
            med = np.median(arr)
            ax.plot(
                [pos - 0.18, pos + 0.18],
                [med, med],
                color='black',
                linewidth=1.4,
                solid_capstyle='butt',
                zorder=5,
            )

    # Paired-subject connecting lines (optional)
    if paired and len(arrays) >= 2:
        n = min(len(a) for a in arrays)
        for i in range(n):
            ys = [a[i] for a in arrays]
            ax.plot(positions, ys, color='gray', alpha=0.25, linewidth=0.6, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels)
