"""Standalone force-directed label placement for matplotlib scatter plots.

Drop this file into your project's plotting helpers (e.g. as
`<project>/visualization/labels.py`) and import the function from your figure
scripts. No project dependencies — just numpy + matplotlib.

Use this whenever you have N scatter points and want to label each one
**without** the labels overlapping each other, the points themselves, or
critical reference lines (like a y=x diagonal). Fixed-offset annotations
(`xytext=(5, 5)`) fail when points cluster; this algorithm uses a small
force-directed simulation to push labels into available whitespace while
keeping each one anchored near its point.

Example:

    import numpy as np
    from <project>.labels import force_directed_label_layout

    points = np.array(zip(x_values, y_values))
    labels = ['S01', 'S02', ...]

    # Compute final label positions (data coordinates)
    fig.canvas.draw()                    # required if axes use log/twin scales
    label_positions = force_directed_label_layout(points, ax)

    # Draw leader lines + text yourself (caller controls styling)
    for (xa, ya), (xt, yt), txt in zip(points, label_positions, labels):
        ax.plot([xa, xt], [ya, yt], color='gray', lw=0.5, alpha=0.6, zorder=4)
        ax.text(xt, yt, txt,
                fontsize=8, ha='center', va='center', zorder=7,
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='white', edgecolor='none', alpha=0.85))

Why two-step (compute positions, then draw): so the caller controls the text
styling, leader lines, and z-ordering. The algorithm is purely numerical.

Tuning the force weights:

The defaults work for typical 10-30 point paired comparisons. If labels
still overlap, increase `w_label` (label-label repulsion). If labels drift
too far from their data point, increase `w_spring`. If labels jam against
the axes, increase `w_edge`. If labels cluster on the y=x diagonal,
increase `w_diagonal`. If the algorithm is slow on large N (>50), reduce
`iterations` to 50 — quality vs speed.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


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
    """Compute non-overlapping label positions for scatter points.

    All forces are computed in **normalised axes coordinates** (0-1) so the
    algorithm is scale-independent. The caller supplies data-coordinate
    `points` and the `ax`; the returned positions are in data coordinates
    (ready to pass to `ax.text` or `ax.annotate`).

    Parameters
    ----------
    points : (N, 2) array
        Data coordinates of scatter points to label.
    ax : matplotlib Axes
        Used for data <-> axes coordinate conversion. The axes' limits and
        scales (e.g. log) must be finalized before calling — invoke
        ``fig.canvas.draw()`` first if you've just set up the axes.
    w_point : float
        Repulsion weight from *all* data points (not just the label's own).
        Increase to push labels further from dense clusters.
    w_label : float
        Label-label repulsion weight. Increase if labels overlap each other.
    w_diagonal : float
        Repulsion from the y=x diagonal (axes coords). Useful for paired
        comparison plots where the diagonal is a reference. Set to 0 if
        there's no meaningful diagonal in your plot.
    w_spring : float
        Spring attraction back to each label's own data point. Increase to
        keep labels closer to their anchor; decrease to let them roam.
    w_edge : float
        Repulsion from the four axes edges. Increase if labels jam against
        the left/right/top/bottom of the plot frame.
    iterations : int
        Number of simulation steps. 100 is a good default; reduce to 50 for
        speed on large point sets, increase to 200 for crowded plots.
    initial_offset : float
        Starting offset (axes coords) added diagonally to each label
        position. Avoids the degenerate case where label = point exactly.

    Returns
    -------
    positions : (N, 2) array
        Final label positions in data coordinates.

    Notes
    -----
    - Returns an empty (0, 2) array if `points` is empty.
    - Uses linear annealing: forces shrink over time so positions stabilize.
    - Forces are magnitude-clamped per-step to avoid explosive divergence.
    - Positions are soft-clamped to [0.02, 0.98] in axes coords every step.
    """
    n = len(points)
    if n == 0:
        return np.empty((0, 2))

    # --- Convert data coords -> normalised axes coords ---------------------
    display_pts = ax.transData.transform(points)
    inv_axes = ax.transAxes.inverted()
    ax_pts = inv_axes.transform(display_pts)  # (N, 2) in [0, 1]

    # Initialise label positions with a small diagonal offset from their
    # anchor — pure overlap with the data point is the worst starting state.
    positions = ax_pts.copy() + initial_offset

    dt = 0.05  # base step size (decays over iterations)

    for step in range(iterations):
        # Linear annealing — force magnitudes shrink as we approach equilibrium
        t = 1.0 - step / max(iterations, 1)
        current_dt = dt * (0.3 + 0.7 * t)

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

        # --- 5. Edge repulsion (stay away from 0 and 1 axes-coord borders) -
        if w_edge > 0:
            margin = 0.05
            for i in range(n):
                px, py = positions[i]
                if px < margin:
                    forces[i, 0] += w_edge / ((px + 1e-8) ** 2)
                if px > 1.0 - margin:
                    forces[i, 0] -= w_edge / ((1.0 - px + 1e-8) ** 2)
                if py < margin:
                    forces[i, 1] += w_edge / ((py + 1e-8) ** 2)
                if py > 1.0 - margin:
                    forces[i, 1] -= w_edge / ((1.0 - py + 1e-8) ** 2)

        # --- Apply forces --------------------------------------------------
        # Clamp force magnitude per-step to prevent runaway label motion
        mag = np.linalg.norm(forces, axis=1, keepdims=True)
        max_force = 0.5
        scale = np.where(mag > max_force, max_force / (mag + 1e-8), 1.0)
        forces *= scale

        positions += forces * current_dt

        # Soft-clamp to keep labels visible within the plot
        positions = np.clip(positions, 0.02, 0.98)

    # --- Convert normalised axes coords -> data coords ---------------------
    display_label = ax.transAxes.transform(positions)
    data_label = ax.transData.inverted().transform(display_label)
    return data_label
