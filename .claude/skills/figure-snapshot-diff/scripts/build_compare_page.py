# DEPRECATED (2026-05-20, Phase 6): legacy pair-only mode. The recommended
# workflow is now History mode — a persistent per-figure version chain with a
# local history server (trunk/staging/rejected + content-addressed comments):
#   uv run python .claude/skills/figure-snapshot-diff/scripts/history_server.py --port 8765
# This script is kept as a fallback for one-off snapshot-vs-current diffs only.
# See SKILL.md "History mode (recommended)".
"""Step 3 of the figure-snapshot-diff workflow. [DEPRECATED — see banner above]

Read a "backup" directory (the snapshot taken by `snapshot_figures.py`) and a
"current" directory (the post-change images), pair files by relative path, and
emit a self-contained HTML page where each pair has a draggable before/after
slider. The page pulls the `<img-comparison-slider>` web component from a CDN at
runtime but is otherwise standalone — drop it on disk and open in any browser.

Usage (minimal):
    python build_compare_page.py \
        --backup-dir paper/figures_snapshot_pre_color_unify_20260512_1246 \
        --current-dir paper/figures \
        --output paper/figures_compare.html

Usage (with descriptions sidecar JSON; see references/descriptions_format.md):
    python build_compare_page.py \
        --backup-dir <snapshot> --current-dir <current> --output <out.html> \
        --descriptions descriptions.json \
        --title "My Project Figure Comparison"

Usage (with extra pairs whose paths live outside --current-dir):
    --extra-pairs extra_pairs.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from html import escape
from pathlib import Path

try:
    from PIL import Image  # type: ignore
except ImportError:  # pragma: no cover
    Image = None  # Fallback: every slider gets 4:3 aspect ratio.


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}


# ---------------------------------------------------------------------------
# Caching helpers — recomputing SHA-256 and PIL.Image.open() per pair is slow
# enough on large image sets to be worth caching across the run.
# ---------------------------------------------------------------------------

_HASH_CACHE: dict[str, str] = {}
_SIZE_CACHE: dict[str, tuple[int, int]] = {}


def _hash_file(path: Path) -> str:
    key = str(path.resolve())
    if key in _HASH_CACHE:
        return _HASH_CACHE[key]
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    digest = h.hexdigest()
    _HASH_CACHE[key] = digest
    return digest


def _files_identical(p1: Path, p2: Path) -> bool:
    """True iff both files have identical byte size AND identical SHA-256."""
    try:
        if p1.stat().st_size != p2.stat().st_size:
            return False
        return _hash_file(p1) == _hash_file(p2)
    except OSError:
        return False


def _image_size(path: Path) -> tuple[int, int]:
    """(width, height) of an image; (4, 3) fallback if Pillow missing or file unreadable."""
    key = str(path.resolve())
    if key in _SIZE_CACHE:
        return _SIZE_CACHE[key]
    if Image is None:
        size = (4, 3)
    else:
        try:
            with Image.open(path) as im:
                size = im.size
        except Exception:
            size = (4, 3)
    _SIZE_CACHE[key] = size
    return size


def fmt_size(n_bytes: int) -> str:
    return f"{n_bytes / 1024:.1f} KB"


def fmt_delta(before: int, after: int) -> tuple[str, str]:
    """(text, css_class) for a Δ size annotation. Empty class = neutral."""
    if before == 0:
        return ("n/a", "")
    pct = (after - before) / before * 100
    text = f"{pct:+.1f}%"
    if abs(pct) < 1.0:
        return (text, "")
    return (text, "delta-neg" if pct < 0 else "delta-pos")


def relpath_from(html_path: Path, target: Path) -> str:
    """Forward-slash relative path from the directory holding the HTML to `target`."""
    html_dir = html_path.resolve().parent
    target_abs = target.resolve()
    try:
        rel = target_abs.relative_to(html_dir)
        return rel.as_posix()
    except ValueError:
        # target is not under html_dir — walk up with ../
        import os
        return Path(os.path.relpath(target_abs, html_dir)).as_posix()


# ---------------------------------------------------------------------------
# Description sidecar
# ---------------------------------------------------------------------------


def load_descriptions(path: Path | None) -> dict:
    """Load the sidecar JSON. Returns an empty mapping if not provided."""
    if path is None:
        return {"title": None, "entries": {}}
    if not path.exists():
        sys.exit(f"[build_compare_page] descriptions file does not exist: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        sys.exit(f"[build_compare_page] descriptions file is not valid JSON: {e}")
    return {
        "title": data.get("title"),
        "entries": data.get("entries", {}),
    }


def _lookup_entry(rel_path: str, name: str, descriptions: dict) -> dict:
    """Try to find a per-image description, first by relative path then basename."""
    entries = descriptions.get("entries", {})
    return entries.get(rel_path) or entries.get(name) or {}


def _format_heading(name: str, rel_path: str, descriptions: dict, source_label: str) -> tuple[str, str]:
    """Return (section_heading, meta_text) for an entry.

    Heading prefers description-supplied label + description over filename;
    meta_text shows the path as a small monospaced reference.
    """
    e = _lookup_entry(rel_path, name, descriptions)
    label = e.get("label")
    desc = e.get("description")
    suffix = f" ({source_label})" if source_label else ""

    if label and desc:
        return (f"{label} — {desc}{suffix}", rel_path)
    if label:
        return (f"{label}{suffix}", rel_path)
    if desc:
        return (f"{name} — {desc}{suffix}", rel_path)
    return (f"{name}{suffix}", rel_path)


def _nav_text(name: str, rel_path: str, descriptions: dict) -> str:
    e = _lookup_entry(rel_path, name, descriptions)
    return e.get("label") or name.replace(".png", "").replace(".jpg", "")


def _sort_key(entry: dict, descriptions: dict) -> tuple:
    """Sort by description order field if present; alphabetical fallback."""
    rel = entry["rel_path"]
    name = entry["name"]
    e = _lookup_entry(rel, name, descriptions)
    order = e.get("order")
    if order is None:
        return (1, rel)  # un-ordered entries after ordered ones
    if isinstance(order, (int, float)):
        return (0, (order,), rel)
    # list/tuple — nested ordering (e.g. [4, 1] for "fig 4b")
    return (0, tuple(order), rel)


# ---------------------------------------------------------------------------
# HTML emission
# ---------------------------------------------------------------------------

HANDLE_SVG = (
    '<svg slot="handle" xmlns="http://www.w3.org/2000/svg" '
    'width="56" height="56" viewBox="-15 -15 30 30">'
    '<circle r="13" fill="#ff6b35" stroke="white" stroke-width="2"/>'
    '<path d="M-7 -4 L-2 0 L-7 4 Z M7 -4 L2 0 L7 4 Z" fill="white"/>'
    "</svg>"
)


def _slider_block(before_src: str, after_src: str, before_path: Path) -> str:
    w, h = _image_size(before_path)
    return (
        f'      <div class="slider-wrap" style="aspect-ratio: {w}/{h};">\n'
        f'        <span class="badge badge-before">BEFORE</span>\n'
        f'        <span class="badge badge-after">AFTER</span>\n'
        f'        <img-comparison-slider>\n'
        f'          <img slot="first"  src="{escape(before_src)}" alt="before"/>\n'
        f'          <img slot="second" src="{escape(after_src)}" alt="after"/>\n'
        f'          {HANDLE_SVG}\n'
        f'        </img-comparison-slider>\n'
        f'      </div>'
    )


# ---------------------------------------------------------------------------
# Pair discovery
# ---------------------------------------------------------------------------


def collect_pairs(backup_dir: Path, current_dir: Path, extra_pairs: list[dict]) -> list[dict]:
    """Walk backup_dir recursively, pair every image with current_dir/<rel-path>.

    Then append explicit extra_pairs (whose `before` / `after` paths can be
    absolute or relative to the CWD).
    """
    entries: list[dict] = []
    backup_dir = backup_dir.resolve()
    current_dir = current_dir.resolve()

    for backup_path in backup_dir.rglob("*"):
        if not backup_path.is_file():
            continue
        if backup_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = backup_path.relative_to(backup_dir)
        rel_str = rel.as_posix()
        current_path = current_dir / rel
        if not current_path.exists():
            # No matching after image — skip (user can supply via extra_pairs if needed).
            continue
        entries.append({
            "id": _safe_id(rel_str),
            "name": backup_path.name,
            "rel_path": rel_str,
            "before_path": backup_path,
            "after_path": current_path,
            "source_label": f"under {current_dir.name}/",
        })

    for spec in extra_pairs:
        before = Path(spec["before"]).resolve()
        after = Path(spec["after"]).resolve()
        if not (before.exists() and after.exists()):
            print(f"[build_compare_page] skipping extra pair — missing file(s): {spec}")
            continue
        name = spec.get("name") or before.name
        rel_str = spec.get("rel_path") or name
        entries.append({
            "id": _safe_id(rel_str),
            "name": name,
            "rel_path": rel_str,
            "before_path": before,
            "after_path": after,
            "source_label": spec.get("source_label", "extra"),
        })

    return entries


def _safe_id(rel_path: str) -> str:
    """Make an HTML-id-safe string from a relative path."""
    return "".join(c if (c.isalnum() or c in "_-") else "_" for c in rel_path).strip("_") or "entry"


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------


def build_html(entries: list[dict], descriptions: dict, page_title: str, output_path: Path) -> str:
    """Render the full HTML page as a string."""

    def nav_for(e: dict) -> str:
        return _nav_text(e["name"], e["rel_path"], descriptions)

    nav_links = "\n      ".join(
        f'<a href="#{escape(e["id"])}">{escape(nav_for(e))}</a>' for e in entries
    )

    sections: list[str] = []
    for e in entries:
        before_path: Path = e["before_path"]
        after_path: Path = e["after_path"]
        before_size = before_path.stat().st_size
        after_size = after_path.stat().st_size
        delta_text, delta_class = fmt_delta(before_size, after_size)
        delta_span = (
            f'<span class="{delta_class}">Δ: {escape(delta_text)}</span>'
            if delta_class else f"<span>Δ: {escape(delta_text)}</span>"
        )
        before_src = relpath_from(output_path, before_path)
        after_src = relpath_from(output_path, after_path)

        identical = _files_identical(before_path, after_path)

        section_heading, meta_text = _format_heading(
            e["name"], e["rel_path"], descriptions, e.get("source_label", "")
        )

        if identical:
            body = (
                '      <div class="notice notice-warn">\n'
                '        <strong>⚠ Not regenerated this session</strong> — '
                'before and after are byte-identical (matching SHA-256). '
                'Either the change did not affect this image, or you forgot to '
                'rerun the pipeline that produces it.\n'
                '      </div>'
            )
        else:
            body = _slider_block(before_src, after_src, before_path)

        sections.append(
            f'\n    <section id="{escape(e["id"])}">\n'
            f'      <h2>{escape(section_heading)}</h2>\n'
            f'      <p class="meta"><code>{escape(meta_text)}</code></p>\n'
            f'      <p class="stats">\n'
            f'        <span>Before: {escape(fmt_size(before_size))}</span>\n'
            f'        <span>After: {escape(fmt_size(after_size))}</span>\n'
            f'        {delta_span}\n'
            f'      </p>\n'
            f'{body}\n'
            f'    </section>'
        )

    sections_html = "\n".join(sections)
    title_html = escape(page_title)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{title_html}</title>
  <script type="module" src="https://cdn.jsdelivr.net/npm/img-comparison-slider@8/dist/index.js"></script>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/img-comparison-slider@8/dist/styles.css" />
  <style>
    body {{
      font-family: -apple-system, system-ui, sans-serif;
      max-width: 1400px;
      margin: 2rem auto;
      padding: 0 1rem;
      font-size: 17px;
      line-height: 1.6;
      color: #222;
    }}
    h1 {{ font-size: 1.8rem; font-weight: 700; }}
    h2 {{ font-size: 1.35rem; font-weight: 600; margin-top: 2rem; border-top: 1px solid #ddd; padding-top: 1rem; }}
    .meta {{ color: #555; font-size: 1.05rem; margin: 0.25rem 0 0.75rem; }}

    .slider-wrap {{
      position: relative;
      width: 100%;
      max-width: 1400px;
      margin: 0 auto;
    }}
    img-comparison-slider {{
      display: block;
      width: 100%;
      height: 100%;
      --divider-color: #ff6b35;
      --divider-width: 4px;
      --default-handle-color: #ff6b35;
      --default-handle-opacity: 1.0;
      --default-handle-width: 56px;
    }}
    img-comparison-slider img {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #fafafa;
    }}

    .badge {{
      position: absolute;
      top: 10px;
      padding: 6px 14px;
      font-weight: 700;
      font-size: 0.95rem;
      border-radius: 4px;
      z-index: 5;
      pointer-events: none;
      box-shadow: 0 1px 4px rgba(0,0,0,0.15);
      letter-spacing: 0.05em;
    }}
    .badge-before {{ left: 10px; background: white; color: #333; border: 1px solid #ccc; }}
    .badge-after  {{ right: 10px; background: #ff6b35; color: white; }}

    .stats {{ display: flex; gap: 1rem; font-size: 1.0rem; color: #444; font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }}
    .stats span.delta-neg {{ color: #2e7d32; font-weight: 600; }}
    .stats span.delta-pos {{ color: #c62828; font-weight: 600; }}

    nav {{
      position: sticky;
      top: 0;
      background: rgba(255, 255, 255, 0.97);
      backdrop-filter: blur(6px);
      padding: 0.75rem 0;
      border-bottom: 1px solid #ddd;
      font-size: 1.0rem;
      font-weight: 500;
      z-index: 10;
      box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }}
    nav a {{
      display: inline-block;
      padding: 0.5em 0.75em;
      color: #1565c0;
      text-decoration: none;
      white-space: nowrap;
    }}
    nav a:hover {{ text-decoration: underline; background: #f0f6fc; border-radius: 4px; }}

    code {{ background: #f3f3f3; padding: 0.1rem 0.3rem; border-radius: 3px; font-size: 0.95em; font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }}

    .notice {{
      padding: 1rem 1.25rem;
      border-radius: 6px;
      margin: 0.75rem 0;
      font-size: 1.0rem;
      line-height: 1.55;
    }}
    .notice-warn {{
      background: #fff8e1;
      border: 1px solid #f9a825;
      border-left: 6px solid #f9a825;
      color: #5d4037;
    }}
  </style>
</head>
<body>
  <h1>{title_html}</h1>
  <p class="meta">Drag the orange handle on each slider to wipe between BEFORE (left) and AFTER (right). Sections where before and after are byte-identical show a yellow warning instead of a slider.</p>
  <nav>
      {nav_links}
  </nav>
  {sections_html}
</body>
</html>
'''


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a before/after image comparison HTML page from two image directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--backup-dir", required=True, type=Path, help="Directory containing the pre-change snapshot (output of snapshot_figures.py).")
    parser.add_argument("--current-dir", required=True, type=Path, help="Directory containing the post-change images.")
    parser.add_argument("--output", required=True, type=Path, help="Destination HTML file.")
    parser.add_argument("--descriptions", type=Path, default=None, help="Optional sidecar JSON with per-image labels/descriptions/order. See references/descriptions_format.md.")
    parser.add_argument("--extra-pairs", type=Path, default=None, help="Optional JSON file listing additional before/after pairs whose paths live outside --current-dir. Schema: [{\"before\": \"path\", \"after\": \"path\", \"name\": \"id\", \"rel_path\": \"foo.png\", \"source_label\": \"results/\"}]")
    parser.add_argument("--title", default=None, help="Page title and <h1>. Defaults to 'Figure Before/After Comparison' or the title in the descriptions JSON if provided.")
    args = parser.parse_args(argv)

    backup_dir = args.backup_dir.expanduser().resolve()
    current_dir = args.current_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not backup_dir.is_dir():
        sys.exit(f"[build_compare_page] --backup-dir not a directory: {backup_dir}")
    if not current_dir.is_dir():
        sys.exit(f"[build_compare_page] --current-dir not a directory: {current_dir}")

    descriptions = load_descriptions(args.descriptions)
    extra_pairs: list[dict] = []
    if args.extra_pairs is not None:
        if not args.extra_pairs.exists():
            sys.exit(f"[build_compare_page] --extra-pairs file does not exist: {args.extra_pairs}")
        extra_pairs = json.loads(args.extra_pairs.read_text(encoding="utf-8"))

    entries = collect_pairs(backup_dir, current_dir, extra_pairs)
    if not entries:
        sys.exit(f"[build_compare_page] no image pairs found. Check --backup-dir vs --current-dir contents.")

    entries.sort(key=lambda e: _sort_key(e, descriptions))

    page_title = (
        args.title
        or descriptions.get("title")
        or "Figure Before/After Comparison"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = build_html(entries, descriptions, page_title, output_path)
    output_path.write_text(html, encoding="utf-8")
    print(f"[build_compare_page] wrote {output_path} ({len(entries)} pairs, {output_path.stat().st_size} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
