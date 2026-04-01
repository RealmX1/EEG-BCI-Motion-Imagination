"""
Base plotting utilities for EEG-BCI project.

This module provides shared constants and helper functions
for visualization across the project.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Re-export from config for convenience
from ..config.constants import MODEL_COLORS

# Chance levels for different task types
CHANCE_LEVELS: Dict[str, float] = {
    'binary': 0.5,
    'ternary': 1/3,
    'quaternary': 0.25,
    'unified': (0.5 + 1/3 + 0.25) / 3,  # Mean of subtask chance levels
}


def get_chance_level(task_type: str) -> float:
    """Get chance level for a task type.

    Args:
        task_type: 'binary', 'ternary', or 'quaternary'

    Returns:
        Chance level (0.5, 0.33, or 0.25)
    """
    return CHANCE_LEVELS.get(task_type, 0.5)


# ---------------------------------------------------------------------------
# Leader-line bar annotations
# ---------------------------------------------------------------------------

def annotate_bars_with_leaders(
    ax,
    bar_entries: List[Tuple],
    *,
    base_margin: float = 0.03,
    step_height: float = 0.015,
    fontsize: int = 7,
    fmt: str = '{:.1f}',
    scale: float = 100,
    leader_lw: float = 0.8,
    yerr_entries: Optional[List] = None,
    highlight_best: bool = True,
):
    """Add staggered leader-line annotations to a grouped bar chart.

    For each group (e.g. each subject), draws a vertical leader line from each
    bar top to a common base height (just above the tallest bar), with heights
    staggered so leftmost is tallest.  Rotated text (45 degrees) is placed
    directly at the leader top.  Colors match the originating bars.

    The best-performing bar in each group is highlighted with bold text and a
    slightly larger font.

    Args:
        ax: matplotlib Axes.
        bar_entries: ``[(BarContainer, values, annotate_bool), ...]``.
            One tuple per data source / model.  *values* has one element per
            grouped item (subject / config).  *annotate_bool* selects whether
            text labels are drawn for that source.
        base_margin: gap above tallest bar in each group (y data units).
        step_height: height increment between successive leaders (y data units).
        fontsize: base text size.
        fmt: format string applied to ``val * scale``.
        scale: multiplier before formatting (100 → show percentages from 0-1).
        leader_lw: line width for leader lines.
        yerr_entries: list parallel to *bar_entries* of error-bar arrays.
            Effective bar top = height + |yerr| when computing base height.
        highlight_best: bold + enlarge the best value in each group.
    """
    if not bar_entries:
        return

    n_items = len(bar_entries[0][1])

    for idx in range(n_items):
        # -- gather every bar that belongs to this group -----------------------
        group = []
        for src_i, (bars, vals, annotate) in enumerate(bar_entries):
            if idx >= len(vals):
                continue
            bar = bars[idx]
            val = vals[idx]
            yerr = 0.0
            if (yerr_entries
                    and src_i < len(yerr_entries)
                    and yerr_entries[src_i] is not None):
                yerr = abs(yerr_entries[src_i][idx])
            effective_top = bar.get_height() + yerr
            # full-opacity version of bar colour
            fc = list(bar.get_facecolor())
            fc[3] = 1.0
            group.append((bar, val, annotate, effective_top, tuple(fc)))

        positive = [g for g in group if g[1] > 0]
        if not positive:
            continue

        max_top = max(et for _, _, _, et, _ in positive)
        base_h = max_top + base_margin

        # annotatable bars, ordered left → right
        to_annotate = sorted(
            [(b, v, c) for b, v, a, _, c in group if a and v > 0],
            key=lambda t: t[0].get_x(),
        )
        n_ann = len(to_annotate)
        if n_ann == 0:
            continue

        best_val = max(v for _, v, _ in to_annotate)

        for rank, (bar, val, color) in enumerate(to_annotate):
            x = bar.get_x() + bar.get_width() / 2
            bar_top = bar.get_height()

            # stagger: leftmost highest, decreasing rightward
            extra = (n_ann - 1 - rank) * step_height
            leader_top = base_h + extra

            is_best = highlight_best and val == best_val and n_ann > 1
            fw = 'bold' if is_best else 'normal'
            fs = fontsize + 1 if is_best else fontsize

            # vertical leader line (bar colour)
            ax.plot(
                [x, x], [bar_top, leader_top],
                color=color, linewidth=leader_lw,
                clip_on=False, zorder=5,
            )

            # rotated text directly at leader top (no extra stub line)
            ax.text(
                x, leader_top,
                fmt.format(val * scale),
                fontsize=fs, fontweight=fw,
                rotation=45, rotation_mode='anchor',
                ha='left', va='bottom',
                color=color,
                clip_on=False, zorder=6,
            )


def accuracy_ylim(
    task_type: str,
    *,
    data_min: Optional[float] = None,
    is_pct: bool = False,
    top_pad: float = 0.18,
) -> Tuple[float, float]:
    """Compute y-axis limits that truncate below chance and pad above 1.0.

    When *data_min* is supplied the lower bound is expanded so that every
    data point remains visible (with a 5 pp margin).

    Args:
        task_type: 'binary', 'ternary', 'quaternary', or 'unified'.
        data_min: lowest accuracy value in the data (0-1 scale, or 0-100
            when *is_pct* is ``True``).  Pass ``None`` to keep the
            default chance-based lower bound.
        is_pct: ``True`` when the axis uses 0–100 scale.
        top_pad: extra space above 1.0 (in 0-1 units) for leader lines.

    Returns:
        ``(bottom, top)`` for ``ax.set_ylim()``.
    """
    chance = CHANCE_LEVELS.get(task_type, 0.5)
    bottom = chance - 0.05
    if data_min is not None:
        # Normalise to 0-1 scale for comparison
        dm = data_min / 100 if is_pct else data_min
        bottom = min(bottom, dm - 0.05)
    top = 1.0 + top_pad
    if is_pct:
        return (bottom * 100, top * 100)
    return (bottom, top)


# ---------------------------------------------------------------------------
# Paired-value label separation
# ---------------------------------------------------------------------------

def separate_paired_labels(
    y1: float,
    y2: float,
    min_gap: float,
) -> Tuple[float, float]:
    """Push two y-values apart symmetrically if closer than *min_gap*.

    Returns ``(adj_y1, adj_y2)`` preserving which is higher/lower.
    If already far enough apart, returns the originals unchanged.
    """
    gap = abs(y1 - y2)
    if gap >= min_gap:
        return y1, y2
    nudge = (min_gap - gap) / 2
    if y1 >= y2:
        return y1 + nudge, y2 - nudge
    return y1 - nudge, y2 + nudge


def draw_label_with_leader(
    ax,
    actual_y: float,
    label_y: float,
    x_start: float,
    text: str,
    *,
    color: str = 'black',
    fontsize: float = 7,
    fontweight: str = 'normal',
    leader_style: str = ':',
    leader_lw: float = 0.8,
    x_pad: float = 0.05,
):
    """Draw a value label, adding a dotted leader line if displaced.

    If *label_y* differs from *actual_y* beyond a tiny threshold, a dotted
    line is drawn from ``(x_start, actual_y)`` to ``(x_start + x_pad, label_y)``
    before placing the text.  Otherwise the text is placed directly.
    """
    displaced = abs(label_y - actual_y) > 1e-6
    text_x = x_start + x_pad if displaced else x_start
    if displaced:
        ax.plot(
            [x_start, text_x], [actual_y, label_y],
            linestyle=leader_style, color=color, linewidth=leader_lw,
            clip_on=False, zorder=4,
        )
    ax.text(
        text_x, label_y, text,
        ha='left', va='center', fontsize=fontsize,
        fontweight=fontweight, color=color,
    )


# ---------------------------------------------------------------------------
# Force-directed label layout for scatter plots
# ---------------------------------------------------------------------------

def force_directed_label_layout(
    points: np.ndarray,
    ax: plt.Axes,
    *,
    w_point: float = 0.0005,
    w_label: float = 0.0005,
    w_diagonal: float = 0.0005,
    w_spring: float = 50.0,
    w_edge: float = 0.0005,
    iterations: int = 100,
    initial_offset: float = 0.03,
) -> np.ndarray:
    """Compute force-directed label positions for scatter points.

    All forces are computed in **normalised axes coordinates** (0-1) so the
    algorithm is scale-independent.  The caller supplies data-coordinate
    *points* and an ``Axes`` for the coordinate transform; the returned
    positions are in data coordinates.

    Parameters
    ----------
    points : (N, 2) array
        Data coordinates of scatter points.
    ax : matplotlib Axes
        Used for data <-> axes coordinate conversion.
    w_point : float
        Repulsion weight from *all* data points.
    w_label : float
        Label-label repulsion weight.
    w_diagonal : float
        Repulsion from the y=x diagonal.
    w_spring : float
        Spring attraction back to the label's own data point.
    w_edge : float
        Repulsion from the four axes edges.
    iterations : int
        Number of simulation steps.
    initial_offset : float
        Starting offset (axes coords) added to each label position.

    Returns
    -------
    positions : (N, 2) array
        Final label positions in data coordinates.
    """
    n = len(points)
    if n == 0:
        return np.empty((0, 2))

    # --- convert data coords -> axes coords --------------------------------
    display_pts = ax.transData.transform(points)
    inv_axes = ax.transAxes.inverted()
    ax_pts = inv_axes.transform(display_pts)  # (N, 2) in [0,1]

    # initialise label positions with a small diagonal offset
    positions = ax_pts.copy() + initial_offset

    dt = 0.05  # step size (decays over iterations)

    for step in range(iterations):
        t = 1.0 - step / max(iterations, 1)  # linear decay 1 -> 0
        current_dt = dt * (0.3 + 0.7 * t)    # annealing

        forces = np.zeros_like(positions)

        # --- 1. Point repulsion: labels repelled from ALL data points ------
        if w_point > 0:
            for i in range(n):
                for j in range(n):
                    diff = positions[i] - ax_pts[j]
                    dist = np.linalg.norm(diff) + 1e-8
                    if dist < 0.25:
                        forces[i] += w_point * diff / (dist ** 3)

        # --- 2. Label-label repulsion -------------------------------------
        if w_label > 0:
            for i in range(n):
                for j in range(i + 1, n):
                    diff = positions[i] - positions[j]
                    dist = np.linalg.norm(diff) + 1e-8
                    if dist < 0.3:
                        f = w_label * diff / (dist ** 3)
                        forces[i] += f
                        forces[j] -= f

        # --- 3. Diagonal repulsion (y = x line in axes coords) ------------
        if w_diagonal > 0:
            for i in range(n):
                px, py = positions[i]
                # closest point on y=x: midpoint projection
                mid = (px + py) / 2
                closest = np.array([mid, mid])
                diff = positions[i] - closest
                dist = np.linalg.norm(diff) + 1e-8
                if dist < 0.2:
                    forces[i] += w_diagonal * diff / (dist ** 3)

        # --- 4. Spring back to own data point -----------------------------
        if w_spring > 0:
            for i in range(n):
                diff = ax_pts[i] - positions[i]
                forces[i] += w_spring * diff

        # --- 5. Edge repulsion (stay away from 0 and 1 boundaries) --------
        if w_edge > 0:
            margin = 0.05
            for i in range(n):
                px, py = positions[i]
                # left edge
                if px < margin:
                    forces[i, 0] += w_edge / ((px + 1e-8) ** 2)
                # right edge
                if px > 1.0 - margin:
                    forces[i, 0] -= w_edge / ((1.0 - px + 1e-8) ** 2)
                # bottom edge
                if py < margin:
                    forces[i, 1] += w_edge / ((py + 1e-8) ** 2)
                # top edge
                if py > 1.0 - margin:
                    forces[i, 1] -= w_edge / ((1.0 - py + 1e-8) ** 2)

        # --- apply forces --------------------------------------------------
        # clamp force magnitude to avoid explosions
        mag = np.linalg.norm(forces, axis=1, keepdims=True)
        max_force = 0.5
        scale = np.where(mag > max_force, max_force / (mag + 1e-8), 1.0)
        forces *= scale

        positions += forces * current_dt

        # soft-clamp to [0.02, 0.98]
        positions = np.clip(positions, 0.02, 0.98)

    # --- convert axes coords -> data coords --------------------------------
    display_label = ax.transAxes.transform(positions)
    data_label = ax.transData.inverted().transform(display_label)
    return data_label
