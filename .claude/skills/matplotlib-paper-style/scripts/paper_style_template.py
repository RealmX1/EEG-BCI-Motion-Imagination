"""Paper-figure style registry — drop-in template.

Drop this file into your project (e.g. as `src/visualization/paper_style.py`
or `<project>/style.py`) and fill in `PAPER_COLORS` with the hex codes that
matter for your work. The `FONT_SIZES` defaults below target a two-column
journal paper figure; bump every value by ~2pt for posters/slides.

Usage:

    from <project>.paper_style import (
        PAPER_COLORS,
        FONT_SIZES,
        apply_paper_style,
        chance_line,
        add_panel_label,
        add_stat_bracket,
        violin_with_subjects,
        paper_figsize,
    )

    fig, ax = plt.subplots(figsize=paper_figsize(rows=1, cols=1))
    ax.plot(x, y, color=PAPER_COLORS['model_a'], label='Model A')
    chance_line(ax, level=0.5)              # paper-red dashed
    add_panel_label(ax, 'A')                # bold 'A.' upper-left, outside box
    apply_paper_style(fig=fig)              # normalize fonts, despine
    fig.savefig('example.png', dpi=300)

The HELPERS section at the bottom (chance_line, add_panel_label,
add_stat_bracket, style_boxplot_outliers, median_reference_line,
violin_with_subjects, paper_figsize) is OPTIONAL but battle-tested.
Delete any you don't need; keep the ones that match your figure language.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import matplotlib.figure
import matplotlib.axes


# ============================================================================
# Color Registry — fill in with your project-specific hex codes
# ============================================================================
#
# Convention:
# - "Domain colors" are the semantically meaningful ones (one per modeled
#   entity: a model name, a method name, a paradigm, etc.).
# - "Utility colors" are reusable accents (chance level, mean marker, etc.).
#
# When you replace a hardcoded `'#......'` somewhere in your codebase, add an
# entry HERE first, then reference it via `PAPER_COLORS['<key>']`. This makes
# every future palette swap a one-line edit instead of a global find-and-replace.

PAPER_COLORS: Dict[str, str] = {
    # ---- Domain colors (replace with your project's entities) ----
    # Examples (delete and replace with your own):
    # 'model_a':    '#E94F37',
    # 'model_b':    '#2E86AB',
    # 'method_x':   '#d62728',
    # 'method_y':   '#2ca02c',

    # ---- Paper-figure utility palette (Ding et al. 2025 Nat Commun mirror) ----
    # These are sensible defaults if you're targeting a Nature-family aesthetic.
    # For paired baseline-vs-proposed comparisons, use baseline_blue +
    # finetuned_orange instead of saturated red/blue — pastel tones read better
    # at small print sizes and are colorblind-friendlier.
    'baseline_blue':    '#9DBED7',   # soft pastel blue — baseline / "before"
    'finetuned_orange': '#F2B670',   # warm orange      — proposed / "after"
    'session1_light':   '#F4C68C',   # pale orange      — session 1
    'session2_deep':    '#D67A30',   # deep orange      — session 2
    'fbcsp_green':      '#A2C49A',   # sage green       — third-condition contrast
    'subject_dot':      '#222222',   # near-black       — per-subject scatter

    # ---- Utility colors (sensible defaults; adjust as needed) ----
    'chance_level':   '#D9534F',   # red dashed — paper convention for chance
    'chance_red':     '#D9534F',   # alias for clarity
    'median_gray':    '#888888',   # gray dashed — reference median lines
    'mean_marker':    'black',
    'delta_neg':      '#c0392b',   # paired delta, negative direction
    'delta_pos':      '#27ae60',   # paired delta, positive direction
    'secondary_blue': '#1976D2',   # secondary accent color
}


# ============================================================================
# Font Size Registry
# ============================================================================
#
# Defaults targeting a 2-column journal figure at 200-300 DPI. For posters,
# slides, or large-format figures, multiply everything by ~1.3-1.5 (use a
# helper or just edit the dict for that one render).
#
# FONT_SIZES_TIGHT is the variant for ~7-inch-wide paper figures (use with
# paper_figsize() below). FONT_SIZES is the default for larger ad-hoc figures.

FONT_SIZES: Dict[str, int] = {
    'title':       12,
    'axis_label':  11,
    'tick':        10,
    'annotation':  9,
    'legend':      9,
    'footer':      7,
    'panel_label': 14,   # 'A.', 'B.', 'C.' panel-letter labels
}

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
# Figsize helper (target: Nature/Nat-Commun two-column ~7.0 in wide)
# ============================================================================

def paper_figsize(
    rows: int = 1,
    cols: int = 1,
    *,
    width_in: float = 7.0,
    row_height_in: float = 3.5,
) -> Tuple[float, float]:
    """Return (width, height) targeting a two-column paper figure (~7.0 in)."""
    return (width_in, max(2.5, rows * row_height_in))


# ============================================================================
# Style Application
# ============================================================================

def _apply_to_ax(
    ax: matplotlib.axes.Axes,
    *,
    sizes: Dict[str, int] = FONT_SIZES,
    despine: bool = True,
) -> None:
    """Apply paper font sizes/weights + despine to a single Axes (idempotent).

    Safely no-ops when title/labels/legend are absent. We wrap each block in
    try/except so a single missing attribute doesn't crash the whole call —
    this matters because plot pipelines often produce figures with optional
    titles or legends, and the sweep should handle both cases.
    """
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

    # Paper convention: hide top/right spines, slim left/bottom spines
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
    """Apply unified paper-style fonts/weights/spines to a figure or axis.

    Idempotent: safe to call multiple times. Pass `fig=fig` to sweep every
    axis (handles multi-panel figures, twinx, etc.). Pass `ax=ax` for a
    single axis. Passing both is OK. Passing neither silently no-ops.

    Args:
        ax: single Axes; if provided, only this ax is styled.
        fig: Figure; if provided, every fig.axes entry is styled.
        tight: True selects FONT_SIZES_TIGHT (use with paper_figsize()).
        despine: True hides top/right spines (paper convention).

    Why this exists: rather than scattering `fontsize=12, fontweight='bold'`
    across every `set_title` call in your codebase, set them once via the
    `FONT_SIZES` registry above and call this function before `savefig` to
    apply uniformly. Changes to the registry now reach every figure with
    no per-function edits.
    """
    sizes = FONT_SIZES_TIGHT if tight else FONT_SIZES
    if fig is not None:
        for sub_ax in fig.axes:
            _apply_to_ax(sub_ax, sizes=sizes, despine=despine)
    if ax is not None:
        _apply_to_ax(ax, sizes=sizes, despine=despine)


# ============================================================================
# Reference-line helpers (chance, median)
# ============================================================================

def chance_line(
    ax: matplotlib.axes.Axes,
    level: float,
    *,
    label: Optional[str] = None,
    linewidth: float = 1.0,
    alpha: float = 0.85,
) -> None:
    """Draw paper-style red dashed chance-level reference line."""
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
    """Draw paper-style gray dashed per-condition median reference line."""
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
# Panel-letter labels ('A.', 'B.', 'C.' …)
# ============================================================================

def add_panel_label(
    ax: matplotlib.axes.Axes,
    letter: str,
    *,
    x: float = -0.12,
    y: float = 1.05,
    fontsize: Optional[int] = None,
) -> None:
    """Add a bold panel-letter label (e.g. 'A.') outside the upper-left of ax.

    Args:
        ax: target Axes.
        letter: a letter ('A', 'B', ...) or already-suffixed form ('A.').
        x, y: position in axes-fraction coords (default top-left, outside).
        fontsize: override default panel-label font size.
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
# Statistical-significance brackets
# ============================================================================

def _p_to_symbol(p_value: float, *, hash_marks: bool = False) -> str:
    """Map p-value to '***'/`'**'`/`'*'`/`'n.s.'` (or hash variant for ANOVA)."""
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
    """Draw a thin significance bracket between x1 and x2 with a symbol on top.

    Args:
        ax: target Axes.
        x1, x2: x positions of the two compared groups (data coords).
        y: bracket-bottom height (data coords).
        p_value: drives the symbol selection (***/.../n.s.).
        hash_marks: True selects '###' (ANOVA main effect convention) instead
            of '***' (Wilcoxon/post-hoc convention).
        height: vertical extent of the bracket's short legs.
        linewidth: bracket line width.
        fontsize: override default annotation font size.
        color: bracket + symbol color, default black.
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
# Boxplot outlier styling (paper convention: diamond markers)
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
    """Patch the fliers of a matplotlib boxplot dict to use diamond markers."""
    for flier in box_dict.get('fliers', []):
        flier.set_marker(marker)
        flier.set_markerfacecolor(markerfacecolor)
        flier.set_markersize(markersize)
        flier.set_markeredgecolor(markeredgecolor)
        flier.set_markeredgewidth(markeredgewidth)


# ============================================================================
# Violin + per-subject overlay (paired-comparison panel)
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
    """Paper-style violin + per-subject scatter overlay.

    Mirrors the figure language of Ding et al. 2025 Fig 1A/B / 3A/B: pastel
    violin bodies, near-black per-subject dots overlaid with light jitter,
    visible median bar, optional paired-subject connector lines.

    Args:
        ax: target Axes.
        data: ``{label: 1D array of per-subject values}``.
        x_labels: x-tick labels (default: data.keys()).
        colors: one color per violin (default: cycles paper palette).
        jitter: horizontal scatter jitter (data coords units of x).
        dot_size: matplotlib scatter `s` value.
        dot_alpha: scatter alpha.
        show_median: draw a black median bar inside each violin.
        paired: connect same-subject dots with thin lines across violins.
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

    if paired and len(arrays) >= 2:
        n = min(len(a) for a in arrays)
        for i in range(n):
            ys = [a[i] for a in arrays]
            ax.plot(positions, ys, color='gray', alpha=0.25, linewidth=0.6, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels)
