---
name: figure-snapshot-diff
description: Generate a static HTML page that lets the user visually drag-wipe between before and after versions of a directory of images (paper figures, dashboard plots, generated charts, screenshots, anything rendered). Use this skill whenever the user is about to modify plot/figure-generation code, regenerate a batch of figures with new params, restyle plots, refactor visualization helpers, or wants to see "what changed visually" after any visualization update. Triggers strongly on phrases like "compare before/after of these images", "I just regenerated the figures, show me what changed", "snapshot these plots before I touch them", "diff these two image directories visually", "build me a slider comparison page", "did my style change break anything visually", or any workflow where the user has a `pre-change` and `post-change` state of rendered images and wants to compare them side-by-side. The skill provides a snapshot → modify → compare workflow with two bundled scripts (`snapshot_figures.py` and `build_compare_page.py`) that handle the backup copy and the HTML generation respectively. Also use proactively before agreeing to modify any code that generates images, so a snapshot exists for later comparison.
---

# figure-snapshot-diff

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

- `references/descriptions_format.md` — the sidecar JSON schema, with worked example

## Bundled scripts

- `scripts/snapshot_figures.py` — Step 1, atomic copy with tag
- `scripts/build_compare_page.py` — Step 3, the HTML generator with all the visual features above
