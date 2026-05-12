---
name: matplotlib-paper-style
description: Refactor matplotlib plotting code in a paper / report / dashboard codebase so colors, fonts, and label placement follow a centralized publication-quality style. Use this skill whenever the user mentions any of: standardizing or unifying plot/figure style across files, "my plot labels are overlapping," "centralize the colors," "drop the in-image figure title because the caption already says it," "mean line covers the per-subject scatter," "make my matplotlib figures consistent," "force-directed labels," "paired scatter labels collide," "audit my figure code for hardcoded colors/fontsize," or any signal that they're cleaning up a directory of plot-generation scripts for publication. Also use proactively when the user is about to restyle figures for a paper submission, journal revision, or thesis chapter, or after dramatic color/font changes that need to propagate everywhere. Trigger even when the user doesn't say "skill" — phrases like "every figure should look the same" or "my reviewer complained the labels overlap" should fire this. The skill bundles a drop-in `paper_style.py` template (centralized PAPER_COLORS + FONT_SIZES + apply_paper_style helper), a standalone `force_directed_label_layout` algorithm for non-overlapping scatter labels, an anti-patterns reference, and a step-by-step audit workflow.
---

# matplotlib-paper-style

## Why this exists

Plotting code in research codebases rots in a specific way: each figure script starts as a self-contained one-off, accumulates project-specific hex codes (`'#E94F37'`, `'#2E86AB'`, ...), uses ad-hoc fontsize literals, and embeds figure-level titles inside the PNG ("Within-Subject 128ch Binary Comparison") that *also* appear in the paper's caption — meaning the title now says the same thing twice, and any change has to be made in two places. Paired comparison scatters acquire fixed-offset subject labels (`xytext=(5, 5)`) that overlap when points cluster. Mean lines drawn on top of per-subject scatter cover the data points. None of these defects are individually fatal, but together they make figures harder to read and the codebase harder to change.

This skill captures the **refactor pattern** that fixes all of the above at once: centralize colors + fontsize into a single Python module, swap hardcoded values everywhere, drop in-image figure titles, replace fixed-offset annotations with a force-directed algorithm, and standardize layering (zorder) for overlay statistics on top of data.

## When to invoke

Use this skill the moment the user signals one of these intentions:

- "Standardize / unify / centralize the style across my plots"
- "My plot labels are overlapping" (especially in paired scatters)
- "Mean line covers the scatter points"
- "Drop the in-image titles — the paper caption is authoritative"
- "Make all my matplotlib figures look consistent"
- "Audit my figure code for hardcoded values"
- After they've made dramatic style decisions (e.g., chose a new palette) and need to propagate

Also invoke proactively *before* making any restyle change in a project with > 5 figure-generating functions — the audit catches anti-patterns that a one-off color swap would miss.

## The refactor workflow

This skill applies in five steps. Steps 1-2 are non-destructive (no source code changes); steps 3-5 modify the user's code.

### Step 1 — Inventory the existing code

Find every place that emits a paper figure. Typical patterns:

- A `scripts/paper/` or `scripts/figures/` directory with `generate_*` functions
- A `src/visualization/` module with shared plotting helpers
- One-off `*_comparison.py` or experiment-script files that call `fig.savefig(...)`

For each function, take note of:
- The PNG output path
- Any `fig.suptitle(...)` or top-level `ax.set_title(...)` calls (figure-level titles, candidates for removal)
- Any `xytext=(N, N)` fixed-offset annotations (candidates for force-directed replacement)
- Any hex codes (`'#......'`) or `fontsize=N` literals

A quick `grep -rn "fig.suptitle\|xytext=\|#[0-9a-fA-F]\{6\}\|fontsize=" <plot-dir>` surfaces most of these.

**Why this step matters:** the user often underestimates how many places hardcode the same color. Showing them the grep counts ("13 functions use `'#E94F37'`, 9 use `'#d62728'`") makes the case for centralization.

### Step 2 — Decide the color/font registry

Talk to the user about:
- **Color palette:** what does each entity get? (Models, methods, paradigms, etc.) Use the existing hex codes from inventory — don't invent new ones unless the user asks. Distinguish:
  - **Domain colors** — the meaningful ones (CBraMod = red, EEGNet = blue)
  - **Utility colors** — `chance_level`, `mean_marker`, secondary accents
- **Font sizes:** title 12 / axis_label 11 / tick 10 / annotation 9 / legend 9 / footer 7 is a good default for two-column paper figures. Adjust for poster / slide contexts (bump everything by ~2pt).
- **In-image titles:** drop if the paper caption is authoritative. Keep panel-level subtitles (`'A.'`, `'B.'`, `'C.'`).

This step is fast — the registry is small (one Python file) and once it exists, the rest is mechanical.

### Step 3 — Create the central style module

Use the bundled `scripts/paper_style_template.py` as the starting point. Copy it into the user's project (typically `src/visualization/paper_style.py` or `<project>/style.py`). Fill in the project-specific colors based on the registry decided in step 2.

The template exports three core names:
- `PAPER_COLORS: dict[str, str]` — single source of truth for hex values (includes Ding-2025-aligned defaults: `baseline_blue`, `finetuned_orange`, `chance_red`, `median_gray`, `subject_dot`, `session1_light`, `session2_deep`, `fbcsp_green`)
- `FONT_SIZES: dict[str, int]` + `FONT_SIZES_TIGHT` — two registries (defaults for ad-hoc figures, tight for ~7-in paper figures)
- `apply_paper_style(ax=None, fig=None, *, tight=False, despine=True)` — idempotent function that normalizes title/label/tick/legend fontsize and removes top/right spines

Plus seven optional but battle-tested helpers (delete any you don't need):
- `paper_figsize(rows, cols)` — returns Nature/Nat-Commun two-column-width tuples (~7 in)
- `chance_line(ax, level)` — paper-red dashed chance reference line
- `median_reference_line(ax, value)` — gray dashed condition-median reference
- `add_panel_label(ax, 'A')` — bold panel-letter outside upper-left of axis
- `add_stat_bracket(ax, x1, x2, y, p_value, hash_marks=False)` — thin bracket + ***/n.s. (or `###` for ANOVA main effects)
- `style_boxplot_outliers(box_dict)` — patches matplotlib boxplot fliers to diamond markers (Ding 2025 convention)
- `violin_with_subjects(ax, data, paired=True)` — pastel-violin + per-subject scatter overlay (Fig 1A/B / 3A/B Ding 2025 standard panel)

### Step 4 — Sweep the figure functions

For each in-scope figure function:

1. Import the new module: `from <project>.style import PAPER_COLORS, FONT_SIZES, apply_paper_style`
2. Replace each literal hex (`'#......'`) that maps to a registry entry → `PAPER_COLORS['<key>']`
3. Replace each `fontsize=<N>` that matches a registry size → `FONT_SIZES['<role>']`
4. Delete `fig.suptitle(...)` calls (the paper caption now carries this). Preserve panel titles (`'A. ...'`, etc.)
5. Find any single-axes `ax.set_title('<whole figure description>')` — delete; the caption does this job
6. Insert `apply_paper_style(fig=fig)` immediately before the `fig.savefig(...)` call

For **paired comparison scatter panels** (any plot with a y=x diagonal where each point is labeled), replace fixed-offset annotations with the bundled force-directed algorithm. See `scripts/force_directed_label_layout.py` — it's a single function you can drop into the project's plotting helpers.

For specific anti-patterns (mean line covers data, inline rank labels stack on top of each other, dual y-axis annotations push against the axes), see `references/anti_patterns.md` — it has copy-paste fixes per case.

### Step 5 — Verify visually

After the sweep, regenerate every figure and visually diff against backups. The companion skill `figure-snapshot-diff` (separate skill, same author) builds an interactive HTML page that lets the user drag-wipe between before/after images for each figure. If it's available, suggest using it — beats opening 20 PNGs side-by-side.

If `figure-snapshot-diff` isn't available, fall back to: `git stash` the changes, run the figure pipeline to capture "before" PNGs, `git stash pop`, regenerate, then ask the user to compare in their image viewer.

## How to think about scope

This skill is for **refactoring existing plotting code**, not for designing new plots. If the user is asking "what colors should I use for this brand-new dashboard?", redirect to a design conversation — this skill doesn't decide palettes, it just centralizes them.

The skill is also paradigm-agnostic across matplotlib subdomains: it works for seaborn (which sits on matplotlib), pandas `.plot()` (same), and pure-matplotlib code. It does NOT work for plotly, bokeh, altair, or other plotting libraries — those need their own style mechanisms.

## What this skill is NOT

- Not a linter — it won't automatically detect every anti-pattern. Use `grep` to find candidates, then apply changes manually or with a small subagent task.
- Not a color theory tool — palette choice is a human decision; the skill just enforces consistency once chosen.
- Not a publication template generator — it doesn't produce `\figure{}` LaTeX or DOCX figure inserts.

## Reference docs

- `references/anti_patterns.md` — common bad-vs-good plotting patterns with copy-paste fixes (label overlap, mean line covers data, inline rank stacks, dual y-axis annotations, hardcoded hex)
- `references/audit_workflow.md` — step-by-step refactor walkthrough with example grep commands and subagent dispatch templates for parallel sweep tasks

## Bundled scripts

- `scripts/paper_style_template.py` — drop-in `paper_style.py` template (PAPER_COLORS + FONT_SIZES + apply_paper_style)
- `scripts/force_directed_label_layout.py` — standalone force-directed label placement function (no project dependencies; just numpy + matplotlib)
