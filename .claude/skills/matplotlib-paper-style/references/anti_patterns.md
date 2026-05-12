# Common matplotlib anti-patterns and their fixes

Each section names the bad pattern, explains *why* it's bad in a paper-figure context, and gives a copy-paste fix.

---

## 1. Fixed-offset annotations on a clustered scatter

**Bad:**
```python
for i, subj in enumerate(subject_ids):
    ax.annotate(subj, (x[i], y[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=8)
```

**Why it's bad:** Every label gets the same `(+5, +5)` offset, so when points cluster (e.g. several models accuracy ~85% on the same subject), labels stack on top of each other and become unreadable.

**Fix:** Use a force-directed algorithm (see `scripts/force_directed_label_layout.py`):

```python
from <project>.labels import force_directed_label_layout

# Make sure axes are finalized first (log/twin scales depend on this)
fig.canvas.draw()

label_positions = force_directed_label_layout(
    np.array(list(zip(x, y))), ax,
    w_diagonal=0.0005,  # repel from y=x reference line (set 0 if no diagonal)
)

for (xa, ya), (xt, yt), txt in zip(zip(x, y), label_positions, subject_ids):
    ax.plot([xa, xt], [ya, yt], color='gray', lw=0.5, alpha=0.6, zorder=4)
    ax.text(xt, yt, txt,
            fontsize=FONT_SIZES['annotation'], ha='center', va='center', zorder=7,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='none', alpha=0.85))
```

The caller draws leader lines + text so styling stays under your control.

---

## 2. Bold mean line covering per-subject scatter

**Bad:**
```python
ax.scatter(x, y_subjects, s=80)
ax.plot(x, y_mean, color='black', linewidth=3, marker='o', markersize=8)
```

**Why it's bad:** No `zorder` set, so matplotlib draws in call order — the mean line ends up on top of scatter points. With `linewidth=3` + `markersize=8` filled markers, the mean line *replaces* the data visually instead of summarizing it.

**Fix:** Give scatter the higher zorder, soften the mean line, hollow its markers:
```python
ax.scatter(x, y_subjects, s=80, alpha=0.95,
           edgecolor='white', linewidth=0.8, zorder=3)
ax.plot(x, y_mean,
        linewidth=2, marker='o', markersize=6,
        color=PAPER_COLORS['mean_marker'],
        alpha=0.85, zorder=2,
        markerfacecolor='white', markeredgewidth=1.8)
```

Three combined moves do the work: (1) scatter `zorder=3` puts it above the mean line, (2) `alpha=0.85` makes the mean line semi-transparent so points underneath show through, (3) `markerfacecolor='white'` hollows the mean markers so they ring rather than blob the data.

---

## 3. Inline #rank + value labels at each data point

**Bad:**
```python
for level_idx, level in enumerate(levels):
    ranking = sorted(method_results[level], reverse=True)
    for rank, (method, value) in enumerate(ranking, start=1):
        ax.annotate(f'#{rank}', (level_idx, value), xytext=(8, 4), textcoords='offset points')
        ax.annotate(f'{value:.1f}%', (level_idx, value), xytext=(8, -12), textcoords='offset points')
```

**Why it's bad:** Two labels per point × 4 points per group means 8 labels in a narrow x-range. When method values cluster (e.g. at the level where method differences are smallest), all 8 labels overlap into an illegible blob.

**Fix:** Stack the ranking at the top of each level position instead:
```python
y_top = ax.get_ylim()[1]
for level_idx, level in enumerate(levels):
    ranking = sorted(
        [(m, method_results[m][level_idx]) for m in methods if not np.isnan(method_results[m][level_idx])],
        key=lambda x: x[1], reverse=True,
    )
    stack_text = '\n'.join(
        f'#{r} {m:<11} {v:.1f}%' for r, (m, v) in enumerate(ranking, start=1)
    )
    ax.text(level_idx, y_top * 0.995, stack_text,
            ha='center', va='top',
            fontsize=FONT_SIZES['annotation'] - 1,
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='white', edgecolor='lightgray', alpha=0.9))
```

Now each level group has ONE legible monospace block at the top of the plot, the data lines stay clean, and the reader's eye can scan the ranking without decoding overlapping annotations. May need to raise `ax.set_ylim` by ~10% to make room for the stack.

---

## 4. Inline annotations on a dual-y-axis plot pushing against the axes

**Bad:**
```python
axL.plot(x, left_values, color='red')
axR = axL.twinx()
axR.plot(x, right_values, color='blue')
for c, v in zip(x, left_values):
    axL.annotate(f'{v:.2f}', (c, v), xytext=(8, 8), textcoords='offset points')
for c, v in zip(x, right_values):
    axR.annotate(f'{v:.2f}', (c, v), xytext=(-30, -28), textcoords='offset points')
```

**Why it's bad:** Inline annotations at fixed offsets push against the left and right y-axes when points fall near them. With twinx, both axes exist in the same physical space — labels often end up overlapping y-axis tick text.

**Fix:** Use force-directed on each axis separately with bumped `w_edge` for axis avoidance:
```python
fig.canvas.draw()  # finalize transData on the twin axes before transforming

pts_L = np.array(list(zip(x, left_values)))
labels_L = [f'{v:.2f}' for v in left_values]
adjusted_L = force_directed_label_layout(
    pts_L, axL, w_diagonal=0.0,
    w_edge=0.002,   # bumped to keep labels away from y-axes
)
for (xa, ya), (xt, yt), txt in zip(pts_L, adjusted_L, labels_L):
    axL.plot([xa, xt], [ya, yt], color='gray', lw=0.5, alpha=0.6)
    axL.text(xt, yt, txt, fontsize=FONT_SIZES['annotation'],
             color=PAPER_COLORS['fdr'], fontweight='bold',
             ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                       edgecolor='none', alpha=0.85))
# Repeat for axR with its own data
```

`w_diagonal=0.0` because there's no meaningful y=x in a dual-y-axis plot. `w_edge=0.002` is bumped from the default `0.0005` to repel from the axes.

---

## 5. Hardcoded hex codes scattered across every figure function

**Bad:**
```python
# generate_figure_a()
ax.plot(x, y_model_a, color='#E94F37')

# generate_figure_b()
ax.bar(x, vals, color='#E94F37')

# generate_figure_c()
ax.scatter(x, y, color='#E94F37', edgecolor='#2E86AB')
```

**Why it's bad:** When you decide model_a should be `#C41E3A` instead, you have to find and replace across N files, hoping you got them all. If the same color appears under different semantic names in different files, automated find-replace is dangerous.

**Fix:** Import everything from a central registry:
```python
from <project>.paper_style import PAPER_COLORS

# generate_figure_a()
ax.plot(x, y_model_a, color=PAPER_COLORS['model_a'])

# generate_figure_b()
ax.bar(x, vals, color=PAPER_COLORS['model_a'])

# generate_figure_c()
ax.scatter(x, y, color=PAPER_COLORS['model_a'], edgecolor=PAPER_COLORS['model_b'])
```

Now changing model_a's color is a one-line edit in `paper_style.py`.

---

## 6. Figure-level title that duplicates the paper caption

**Bad:**
```python
fig.suptitle('Within-Subject 128ch Binary Comparison', fontsize=14, fontweight='bold')
# ... or ...
ax.set_title('Per-Subject Accuracy Comparison (Imagery Binary)')
```

**Why it's bad:** In a published figure, the *caption* below the figure is authoritative. Putting the same text inside the PNG means:
- Reader sees the same words twice (visual noise)
- Title takes layout space that could be data
- When the caption is edited, the in-image title goes stale
- Inconsistent capitalization / phrasing between caption and in-image title undermines polish

**Fix:** Delete `fig.suptitle(...)` calls and any single-axes `ax.set_title(...)` that describes the whole figure. **Keep** panel-level titles in multi-panel figures (`'A. Distribution'`, `'B. Comparison'`, `'C. Trajectory'`) — those describe individual axes, not the figure as a whole, and the caption typically doesn't repeat them.

```python
# Drop this:
fig.suptitle('Within-Subject 128ch Binary Comparison')

# Keep these (panel-level):
ax_a.set_title('A. Accuracy Distribution')
ax_b.set_title('B. Per-Subject Comparison')
ax_c.set_title('C. Paired Scatter')
```

---

## 7. Per-bar accuracy text labels on a grouped bar chart that overlap

**Bad:**
```python
bars = ax.bar(x + offsets, values, width)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{val:.1f}', ha='center', fontsize=9)
```

**Why it's bad:** With grouped bars, adjacent text labels can overlap horizontally even when bars don't. With error bars, the text often collides with the error-bar caps.

**Fix:** Skip per-bar text labels entirely; if the precise numbers matter, put a small table beside or below the figure. If the chart visually compares relative magnitudes, the numbers add little. If the user insists on inline values, switch the chart to horizontal bars with values at the bar tips — there's much more horizontal room than vertical room.

For a "ranking" effect (which 3 bars are highest?), put a sorted stack at the top of each x-group instead — see anti-pattern #3.

---

## When to skip the force-directed approach

Force-directed labeling is overkill for:
- Plots with ≤ 5 well-separated points (a fixed offset works fine)
- Plots where leader lines visually clutter the data (some bar charts)
- Plots where each label is a numeric value on a regular x-grid (a small inline text suffices)

Use the algorithm when:
- ≥ 8 points
- Points cluster in some region of the plot
- Labels are short identifiers (subject IDs, condition names) not full sentences
- The plot has a critical reference line (y=x, chance level) that labels shouldn't cross
