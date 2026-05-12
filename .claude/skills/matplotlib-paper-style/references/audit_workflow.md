# Audit workflow — step-by-step refactor walkthrough

This guide walks through applying `matplotlib-paper-style` to an existing codebase with > 5 plot-generating functions. For small codebases (1-3 figures), skip the inventory step and just apply the fixes directly.

---

## Step 0 — Take a snapshot first

Before any edit, snapshot the current state of generated images so you can visually diff after the refactor. If the companion skill `figure-snapshot-diff` is available, invoke it:

```bash
python <skill-path>/figure-snapshot-diff/scripts/snapshot_figures.py \
    --source <path-to-figure-output-dir> \
    --tag pre_style_centralization
```

If not available, `cp -r` the figure directory to a sibling with a dated name. The point is to have *something* to compare against at the end.

---

## Step 1 — Inventory

Find every plot-emitting function and grep for the anti-patterns:

```bash
# Find figure-generating function definitions
grep -rn "^def generate_\|fig.savefig\|plt.savefig" <plot-dir>

# Find figure-level titles to drop
grep -rn "fig.suptitle\|set_title(" <plot-dir>

# Find fixed-offset annotations (paired-scatter label overlap)
grep -rn "xytext=(\|textcoords='offset points'" <plot-dir>

# Find hardcoded hex codes
grep -rn "color=\s*['\"]#[0-9a-fA-F]\{6\}" <plot-dir>
grep -rn "color\s*=\s*['\"]#[0-9a-fA-F]\{3,6\}['\"]" <plot-dir>

# Find hardcoded fontsize literals
grep -rn "fontsize=[0-9]" <plot-dir>
```

For each grep, save counts per file. This data drives the next step.

**Why this matters:** before committing to a refactor, you need to know the scope. "13 occurrences of #E94F37 across 8 files" is a manageable swap; "237 occurrences across 80 files" is a different conversation about whether to refactor incrementally or bite the bullet.

---

## Step 2 — Define the registry

Talk to the user (or decide yourself if context is clear) about:

**Domain colors** — list every meaningful entity in the project's figures:
- Model names (one color each)
- Method names (one color each)
- Paradigm names (one color each)
- Subject groupings (if relevant)

**Utility colors** — universally useful accents:
- `chance_level` — typically `'gray'`
- `mean_marker` — typically `'black'`
- `delta_neg` / `delta_pos` — colored deltas for paired comparisons
- Secondary accents — extra colors for dual-axis or contrast purposes

**Font sizes** — pick from these defaults unless you have specific journal requirements:
- title: 12, axis_label: 11, tick: 10, annotation: 9, legend: 9, footer: 7

For posters/slides, multiply everything by ~1.4.

---

## Step 3 — Drop in the template

Copy `scripts/paper_style_template.py` from this skill into the user's project. Suggested locations:
- Existing visualization module: `<project>/visualization/paper_style.py`
- Otherwise: `<project>/style.py`

Fill in `PAPER_COLORS` with the registry from step 2. Leave `FONT_SIZES` at the defaults unless step 2 chose different sizes.

Smoke test the import:
```bash
python -c "from <project>.paper_style import PAPER_COLORS, FONT_SIZES, apply_paper_style; print(list(PAPER_COLORS.keys()))"
```

Should print the color keys without error.

---

## Step 4 — Sweep the figure functions (parallel subagent pattern)

For a codebase with many figure functions, **parallelize the sweep** by dispatching one subagent per non-overlapping region of the file (or per file). Each subagent makes a uniform set of mechanical replacements:

**Subagent prompt template:**

```
In <file>, find the function `generate_<name>()` between lines <start> and <end>.

Apply these mechanical replacements:

1. Add to file top imports (if not already present):
   from <project>.paper_style import PAPER_COLORS, FONT_SIZES, apply_paper_style

2. Replace hex codes with PAPER_COLORS keys:
   - '#E94F37' -> PAPER_COLORS['model_a']
   - '#2E86AB' -> PAPER_COLORS['model_b']
   - [...one line per color in the registry...]

3. Replace fontsize literals with FONT_SIZES keys:
   - fontsize=12 (in set_title) -> fontsize=FONT_SIZES['title']
   - fontsize=11 (in set_xlabel / set_ylabel) -> FONT_SIZES['axis_label']
   - fontsize=10 (in tick_params) -> FONT_SIZES['tick']
   - fontsize=9 (in legend / annotate / text) -> FONT_SIZES['annotation']
     or FONT_SIZES['legend']
   - fontsize=7 (in footer) -> FONT_SIZES['footer']

4. Delete fig.suptitle(...) calls entirely. Preserve panel titles
   (ax.set_title('A. ...'), 'B. ...', etc.)

5. Find any single-axes ax.set_title(...) that describes the WHOLE figure
   (not a panel). Delete it. The paper caption is authoritative.

6. Just before fig.savefig(...), insert:
       apply_paper_style(fig=fig)

Report: per-replacement counts, any ambiguous cases you skipped.
```

**Conflict guidance:** if multiple functions live in the same file, dispatch ONE subagent per file (not per function) to avoid file-level edit conflicts. Within the prompt, give the subagent the list of functions in scope.

---

## Step 5 — Handle the special patterns

The mechanical sweep doesn't catch these — they need dedicated edits. Open `references/anti_patterns.md` for the full fix per case:

- **Paired-scatter with subject labels** (anti-pattern #1) — replace `xytext=(5, 5)` with `force_directed_label_layout()`. Drop `scripts/force_directed_label_layout.py` into the project's plotting helpers and call it.
- **Mean line covering scatter** (anti-pattern #2) — change `linewidth=3` → `2`, add `alpha=0.85`, `zorder=2`, `markerfacecolor='white'`. Add `zorder=3` and `edgecolor='white'` to the scatter call.
- **Inline #rank + value labels** (anti-pattern #3) — replace the per-point annotate loop with a single sorted stack at the top of each x-group position.
- **Dual-y-axis with inline annotations** (anti-pattern #4) — use `force_directed_label_layout` on each axis with bumped `w_edge` for axis avoidance.

For each special pattern, dispatch ONE focused subagent with the specific function and line range, the anti-pattern reference, and the desired fix.

---

## Step 6 — Verify visually

Regenerate every figure:
```bash
<your-figure-pipeline-regenerate-command>
# E.g. for a paper-figure script with --figure all:
python scripts/paper/generate_paper_figures.py --figure all
```

Then visually diff. If `figure-snapshot-diff` is available:
```bash
python <skill-path>/figure-snapshot-diff/scripts/build_compare_page.py \
    --backup-dir <step-0-snapshot> \
    --current-dir <figure-output-dir> \
    --output figures_compare.html
```

Open the HTML in a browser, drag-wipe through every figure. Expected differences (these are desired outcomes):
- Figure-level titles disappear (smaller PNG file size)
- Subject labels in paired scatters no longer overlap (visible leader lines)
- Mean lines are thinner and semi-transparent
- Colors are consistent across figures (no off-shade duplicates)

Unexpected differences (these need investigation):
- Whole figure rendered blank or with traceback in logs — likely an import or `apply_paper_style` call that fired on an unexpected axes structure
- Colors swapped between two entities — likely a wrong `PAPER_COLORS` key in one place

---

## Common pitfalls

1. **The sweep agent skips a hex code it doesn't recognize.** This is intentional — only replace hex codes that have a registry entry. Random one-off hex codes (e.g., for a single highlighted bar) can stay as literals if they're not paradigm/model/method colors.

2. **`apply_paper_style(fig=fig)` runs but fonts don't change.** Likely the calling site is using `ax.set_title(..., fontsize=15)` AFTER `apply_paper_style` — the explicit fontsize wins. Move `apply_paper_style` to be the LAST call before `savefig`.

3. **Multi-panel figure with nested GridSpec — some axes don't restyle.** `apply_paper_style(fig=fig)` iterates `fig.axes` which catches all top-level axes but might miss inset axes. For inset axes, call `apply_paper_style(ax=inset)` explicitly.

4. **Force-directed labels go to the wrong spot when using log axes.** Call `fig.canvas.draw()` BEFORE invoking `force_directed_label_layout`. The transform stays unset until first draw.

5. **The 'movement' vs 'motor' alias issue.** If your codebase uses one paradigm name in some places and a synonym in others (e.g., 'motor' in some functions, 'movement' in others), add both keys to `PAPER_COLORS` pointing at the same hex value. Don't try to rename in code — too risky for a style refactor.

---

## Estimated time budget

Rough numbers from a 19-figure / 16-function codebase (one real example):
- Step 0 (snapshot): 1 minute
- Step 1 (inventory): 5 minutes
- Step 2 (registry): 5 minutes
- Step 3 (template drop-in): 5 minutes
- Step 4 (parallel sweep): 15-30 minutes (the big one — depends on subagent parallelism)
- Step 5 (special patterns): 10-20 minutes per pattern (do them sequentially since they touch specific function bodies)
- Step 6 (visual diff): 10 minutes of human review

Total: ~1-2 hours for a project this size, mostly subagent dispatching + waiting.
