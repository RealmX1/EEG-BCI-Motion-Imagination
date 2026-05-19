---
name: figure-snapshot-diff
description: Track and visually compare versions of generated figures (paper plots, dashboards, charts, screenshots). Two modes — (1) **History mode** (recommended for paper figures): per-figure version chain stored under `paper/figures/_history/<fig_id>/` with trunk + staging + rejected branches, browsed via a local HTTP server + Web UI (`history_server.py`). Any two versions can be slider-compared; rejection is soft-delete (kept for compare/learning); new proposals appear in real time and the user accepts/rejects them. The history mode persists free-form user comments (`comment-add`) keyed by SHA-256 of the image content so feedback survives accept/reject and version renames. Comment body can contain both critique of the current pair AND requests for future updates (no feedback/request kind split). (2) **Legacy pair mode**: static before/after HTML built from two image directories (`snapshot_figures.py` + `build_compare_page.py`). Triggers strongly on phrases like "compare before/after of these images", "show me the version history of figure 4b", "approve this plot change", "diff these two image directories visually", or any workflow where the user has multiple states of rendered images and wants to inspect or accept/reject them. **Also auto-triggers on conversation-mode plot change requests** ("make Fig 4b's y-axis bigger", "change the colors of Fig 7 to match", "add a legend") — these MUST be captured immediately via `comment-at-tip` (alias `request-add`) before implementing, so the request is recorded against the figure's trunk tip and surfaced to any future implementer.
---

# figure-snapshot-diff

## Two modes

This skill has two delivery modes that share the goal of visually comparing rendered images:

### History mode (recommended for paper / long-lived figures)

A persistent, append-only **version chain** per figure with a **staging** area for proposed changes, **comments** attached to (before, after) pairs, and an agent-driven workflow to address open comments. Backed by:

- `scripts/history_cli.py` — propose / accept / reject (soft-delete) / list / import-snapshots / comment-add / comment-at-tip / comment-status / comments-open / context-bundle
- `scripts/history_server.py` — local HTTP server (stdlib http.server, port 8765, bind 127.0.0.1) that serves a vanilla-JS web UI
- `web/{index.html, app.js, style.css}` — single-page UI: 3-column layout (sidebar / info+staging+comments+trunk / slider) with auto-advance after staging is cleared
- `references/history_manifest_format.md` — manifest schema (trunk + staging)
- `references/staging_workflow.md` — propose → accept/reject lifecycle
- `references/comment_workflow.md` — comment system: SHA-keyed (survives accept/reject), researcher feedback → agent ideation → plotting-skill subagent dispatch → resolution

When to use this mode:
- Paper figures evolving over many revisions
- Multiple agents proposing changes; user is the gate
- Want "show me how figure 4b evolved from v0 to today" or "is the latest agent proposal an improvement"
- Researcher leaves persistent feedback ("y-axis is too small", "legend covers data point", etc.) that should drive future updates

**How to start:**

```bash
# (one-time) populate _history/ from existing snapshots + paper/figures/
uv run python paper/figures/_audit_corpus/2026-05-13/_import_history.py

# Start the UI server
uv run python .claude/skills/figure-snapshot-diff/scripts/history_server.py --port 8765
# → open http://127.0.0.1:8765/
```

Agents propose new versions via:

```bash
python .claude/skills/figure-snapshot-diff/scripts/history_cli.py propose \
    fig4b /path/to/new.png \
    --tag <slug> --source-cmd "<full command>" --proposed-by "<identifier>"
```

User accepts/rejects via the UI buttons (or `history_cli.py accept fig4b s1`).

### Legacy pair mode

Static HTML built once from two image dirs. Two scripts: `snapshot_figures.py` (copies a dir with a tag) and `build_compare_page.py` (emits one HTML with N×2 slider pairs). No staging, no live updates. Useful for one-shot reviews of an ad-hoc refactor in a project that doesn't have a history-mode setup. The legacy mode docs are below.

> **DEPRECATED for the EEG-BCI paper figures (2026-05-20, Phase 6).** `scripts/build_compare_page.py` here, the project-level `scripts/paper/build_figures_compare_page.py`, and the root `paper/figures_compare*.html` artifacts it generated are all superseded by History mode. They are kept only as a fallback for projects without a history-mode setup. For this repo's paper figures use the History mode server + `generate_paper_figures.py --stage-history`; see the project `CLAUDE.md` section "## 论文图表生成与版本管理".

---

## Why this exists

Rendering changes are hard to review from a diff alone — a 1-line CSS change can cascade into 30 unrelated charts looking subtly off, and a 100-line plot refactor often produces output that's visually indistinguishable from before. The only reliable check is to *look* at the images.

This skill captures the workflow:

1. **Snapshot** the current image directory before any change.
2. **Modify** the code (the user's actual work).
3. **Build** an HTML page that overlays each before/after pair with a draggable comparison slider.

The HTML page is the deliverable: it loads in any browser, requires no install, and lets the user (or a reviewer) drag a vertical handle across each image to wipe between the two versions.

## When to invoke

Use this skill **before** the user starts modifying any code path that produces images. Specifically when:

- The user mentions intent to restyle, recolor, refactor, or refine any plot/figure code
- The user asks to "standardize" or "unify" visual elements
- The user is about to migrate a plotting library (matplotlib v1 → v2, seaborn → plotly, etc.)
- The user has *already* regenerated some images and wants to compare against what was there before (in this case skip step 1, jump to step 3 with whatever snapshot exists)

If no snapshot exists when the user asks for a comparison, *ask* whether they have a pre-change backup somewhere — without one, the comparison page can only show the current state vs. itself.

## Workflow

### Step 1 — Snapshot

Run the bundled snapshot script with a descriptive tag indicating what's about to change:

```bash
python <skill-path>/scripts/snapshot_figures.py \
  --source <path-to-image-dir> \
  --tag <change-description>
```

The script copies all images under `<source>` (PNG, JPG, JPEG, GIF, WEBP, SVG, PDF) into a sibling directory named `<source-parent>/<source-name>_snapshot_<tag>_<YYYYMMDD_HHMM>/`. Subdirectory structure is preserved.

**Tag examples:** `pre_color_unify`, `pre_v2_migration`, `before_legend_fix`.

**Why this naming convention:** the snapshot directory sits next to the source, so a glance at `ls` shows what state was preserved and when. The tag is part of the directory name so multiple snapshots taken on the same day don't collide.

Confirm the snapshot succeeded by checking the printed file count + the directory size before letting the user proceed to step 2.

### Step 2 — User modifies code

This is the work the user actually wanted to do — restyling, refactoring, regenerating figures. The skill stays out of the way during this phase.

When the user reports done, regenerate the affected images (whatever pipeline they use — running their training script, calling `--figure all`, re-executing notebook cells, etc.).

### Step 3 — Build the compare HTML

```bash
python <skill-path>/scripts/build_compare_page.py \
  --backup-dir <snapshot-dir> \
  --current-dir <source-dir> \
  --output <output.html>
```

Optional:
- `--descriptions <descriptions.json>` for per-image labels and change descriptions (schema in `references/descriptions_format.md`)
- `--title "..."` for the page `<title>` and `<h1>`
- `--extra-pairs <extra.json>` for image pairs that live outside the main `--current-dir` (e.g., paper figures referenced by historical timestamp paths in a results/ directory)

The HTML output is fully self-contained: it pulls the `<img-comparison-slider>` web component from a CDN at runtime, but everything else (CSS, image paths, page structure) lives in the file itself. Open it in any browser to use.

**Visual features the page includes:**

- A **sticky top nav** with anchor links to each pair (uses paper-figure labels from descriptions JSON if provided, otherwise filenames)
- Per-pair section with title, change description, before/after file-size delta (color-coded: green if smaller, red if larger)
- A **draggable slider** with:
  - Orange (`#ff6b35`) 4px divider line
  - Orange circle handle with white left/right arrows (custom SVG, replaces the component's near-invisible default)
  - White "BEFORE" badge top-left, orange "AFTER" badge top-right (left = before convention)
  - Dynamic `aspect-ratio` per image (computed via PIL) so wide and tall images both render at their natural shape — no letterboxing
- **Byte-identity check**: if the before and after PNG hash to the same SHA-256, the slider is suppressed and a yellow warning block appears instead saying "Not regenerated this session — only the source code was patched". This makes the "I forgot to regenerate this one" failure mode visible instead of silent.

## Sidecar descriptions JSON (optional)

When the user provides a JSON file via `--descriptions`, each section heading and nav link gets richer text. Useful for paper figures with canonical numbering ("图 4b") or when describing what changed per image. See `references/descriptions_format.md` for the schema.

If no JSON is provided, sections fall back to the filename and the change description is omitted.

## Output expectations

- A single HTML file at the path given by `--output`
- Optional companion: tell the user to refresh their browser if they re-run the builder
- The output references images by relative path. Don't move the HTML elsewhere without also moving the snapshot + current image dirs, or the slider images break.

## What this skill is NOT

- Not a generic image diff tool (no pixel-level diff, no SSIM, no perceptual hashing — those answer "are these different?", this answers "show me how they differ")
- Not a build system — it doesn't trigger regeneration of images, it just compares whatever's on disk
- Not for comparing two arbitrary single images — use a regular image viewer for that; this skill's value is the batch workflow over a directory

## Reference docs

- `references/history_manifest_format.md` — manifest + global index schema (History mode)
- `references/staging_workflow.md` — propose / accept / reject lifecycle (History mode)
- `references/descriptions_format.md` — the sidecar JSON schema for legacy pair mode

## Bundled scripts

History mode (recommended):
- `scripts/history_cli.py` — propose / accept / reject / list / import-snapshots
- `scripts/history_server.py` — local HTTP server + Web UI launcher

Legacy pair mode:
- `scripts/snapshot_figures.py` — Step 1, atomic copy with tag
- `scripts/build_compare_page.py` — Step 3, the static HTML generator
