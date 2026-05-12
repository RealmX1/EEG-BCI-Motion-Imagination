# Descriptions sidecar JSON — schema and examples

The `build_compare_page.py` script accepts an optional `--descriptions <file.json>` argument that lets you attach per-image metadata (labels, change descriptions, sort order). This file is the only way to put nice human-readable text into the section headings and nav links — without it, every entry falls back to its filename.

## Top-level schema

```json
{
  "title": "Optional page <h1> and <title>. Falls back to the --title CLI flag if absent.",
  "entries": {
    "<rel-path-or-basename>": {
      "label": "Short label shown in nav and as the section heading prefix",
      "description": "Longer description of what changed for this image",
      "order": [4, 1]
    }
  }
}
```

## Key lookup

The `entries` dict is looked up first by **relative path** (e.g. `"subdir/figure_3.png"`), then by **basename** (e.g. `"figure_3.png"`). Use relative paths when basename collisions are possible across subdirs; use basenames otherwise — it's less typing.

## Field details

### `label` (string, optional)

Shown in:
- The sticky nav as the clickable link text
- The first half of each section's `<h2>` heading (before the change description)

Use for canonical figure numbers ("Figure 4b"), short codes ("OnboardingV2"), or any short identifier that's more meaningful than the filename. If absent, nav and heading fall back to the filename.

### `description` (string, optional)

Shown in the section heading, after the label. One sentence describing what changed for this image — e.g. "removed inline annotations, added top-of-axis sorted stack" or "switched from matplotlib to plotly". Distinct from the filename or label: this is *what* changed, not *what the image is*.

### `order` (number or array, optional)

Controls sort order in the output page. Two forms:

- **Number** — single sort key. Higher numbers sort later.
- **Array** — nested sort key. `[4, 1]` sorts before `[4, 2]` which sorts before `[5, 0]`. Use this when your images have hierarchical labels like "Figure 4 / 4b / 4c / 5".

Entries without an `order` field sort alphabetically *after* all ordered entries.

## Worked example

Suppose your figure dir has:

```
paper/figures/
├── overview.png
├── method_diagram.png
├── results_binary.png
└── results_ternary.png
```

and you want them ordered "overview first, then method diagram, then the two results side by side":

```json
{
  "title": "Paper revision 2026-05 — figure changes",
  "entries": {
    "overview.png": {
      "label": "Fig 1",
      "description": "redrew with new color palette",
      "order": 1
    },
    "method_diagram.png": {
      "label": "Fig 2",
      "description": "added the v2 pipeline branch",
      "order": 2
    },
    "results_binary.png": {
      "label": "Fig 3a — Binary",
      "description": "force-directed labels in panel 3",
      "order": [3, 0]
    },
    "results_ternary.png": {
      "label": "Fig 3b — Ternary",
      "description": "force-directed labels in panel 3",
      "order": [3, 1]
    }
  }
}
```

The resulting HTML nav reads "Fig 1 / Fig 2 / Fig 3a — Binary / Fig 3b — Ternary" and each section heading reads (e.g.) "Fig 3a — Binary — force-directed labels in panel 3".

## When to skip the sidecar entirely

If your image set is small (< 5 files) and the filenames are already meaningful, the sidecar is overhead. Just run the builder without `--descriptions` and live with `<h2>filename.png (under <current-dir-name>/)</h2>`.

## Tips

- Keep `description` to one short sentence. The heading runs across the page; long descriptions wrap awkwardly.
- Reserve `label` for cases where the filename is opaque (timestamps, hashes, version codes). When the filename is already human-readable, leave `label` out.
- If you have many images and don't care about ordering, omit `order` everywhere — alphabetical is usually fine.
