#!/usr/bin/env python
"""Registry-driven consistency checker/fixer for paper-draft figure paths.

Phase 6 (2026-05-20). For every Markdown image ref ``![label](path)`` in a
paper draft, this tool:

  1. Parses the figure label from the alt text (e.g. ``图 4b. 通道...`` → ``图 4b``).
  2. Looks up the matching ``FigureSpec`` in ``scripts/paper/figure_registry.py``
     by ``paper_label``.
  3. Computes the EXPECTED relative path = ``canonical_output_path`` rendered
     relative to the draft's directory.
  4. Compares against the path actually written in the draft.

Why a checker, not a blind ``results/ → figures/`` rewriter (the original plan
text): the as-built system (registry, the 5 ``_history/<fig>/manifest.json``,
the Phase-5 dispatch, and the draft) is already uniformly keyed on each
figure's true ``canonical_output_path`` — for fig1/2/3c/6/6b that is
``results/<timestamp>...`` BY DESIGN (timestamped data-provenance figures).
Those refs are already correct; this tool just *proves* that and catches any
figure whose draft ref genuinely drifted from its registry canonical path.

Refs whose label is not in the registry (电极放置图 3a / S3–S6 etc.) are
reported ``NOT_IN_REGISTRY`` and left untouched — out of scope by design.

Usage:
    # report only (default; exit 1 if any MISMATCH — CI-friendly)
    uv run python scripts/paper/update_draft_image_paths.py
    uv run python scripts/paper/update_draft_image_paths.py --draft paper/drafts/paper_draft_v3.1.md
    # rewrite MISMATCH paths in place (writes a timestamped .bak first)
    uv run python scripts/paper/update_draft_image_paths.py --apply
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/paper/

import figure_registry  # noqa: E402  (scripts/paper is not a package)

DEFAULT_DRAFT = "paper/drafts/paper_draft_v3.1.md"

# Markdown image:  ![alt text](relative/path.png "optional title")
_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)\s]+)(?:\s+\"[^\"]*\")?\)")

# Label is everything before the first sentence terminator in the alt text:
#   "图 4b. 通道选择方法排序翻转"  -> "图 4b"
#   "Figure S2. 21-被试 × 8-条件"  -> "Figure S2"
#   "图 3a. 5 配置电极空间分布"     -> "图 3a"  (not in registry -> skipped)
_LABEL_RE = re.compile(r"^\s*(?P<label>.+?)\s*[.．。]")


def _label_to_spec() -> dict:
    """paper_label (normalized) -> FigureSpec."""
    return {s.paper_label.strip(): s for s in figure_registry.all_figures()}


def _expected_relpath(canonical_output_path: str, draft_dir: Path) -> str:
    """canonical_output_path (repo-relative) as a POSIX relpath from draft_dir."""
    abs_target = (PROJECT_ROOT / canonical_output_path).resolve()
    rel = os.path.relpath(abs_target, start=draft_dir.resolve())
    return Path(rel).as_posix()


def _parse_label(alt: str) -> str | None:
    m = _LABEL_RE.match(alt)
    return m.group("label").strip() if m else None


def check_draft(draft_path: Path):
    """Return (rows, text). rows: list of dicts with status per image ref."""
    text = draft_path.read_text(encoding="utf-8")
    draft_dir = draft_path.parent
    label_map = _label_to_spec()
    rows = []
    for m in _IMG_RE.finditer(text):
        alt = m.group("alt")
        cur = m.group("path")
        label = _parse_label(alt)
        spec = label_map.get(label) if label else None
        if spec is None:
            rows.append({
                "label": label or "(no label)",
                "alt": alt, "current": cur, "expected": None,
                "status": "NOT_IN_REGISTRY", "span": m.span(),
            })
            continue
        expected = _expected_relpath(spec.canonical_output_path, draft_dir)
        status = "OK" if cur == expected else "MISMATCH"
        rows.append({
            "label": label, "fig_id": spec.fig_id,
            "alt": alt, "current": cur, "expected": expected,
            "status": status, "span": m.span(),
            "match_start": m.start("path"), "match_end": m.end("path"),
        })
    return rows, text


def apply_fixes(draft_path: Path, rows, text: str) -> int:
    """Rewrite MISMATCH paths in place. Returns number of edits."""
    edits = [r for r in rows if r["status"] == "MISMATCH"]
    if not edits:
        return 0
    # Apply right-to-left so earlier spans stay valid.
    new_text = text
    for r in sorted(edits, key=lambda r: r["match_start"], reverse=True):
        s, e = r["match_start"], r["match_end"]
        new_text = new_text[:s] + r["expected"] + new_text[e:]
    backup = draft_path.with_suffix(
        draft_path.suffix + f".bak_{datetime.now():%Y%m%d_%H%M%S}"
    )
    shutil.copy2(draft_path, backup)
    draft_path.write_text(new_text, encoding="utf-8")
    print(f"[apply] backup written: {backup.relative_to(PROJECT_ROOT)}")
    print(f"[apply] {len(edits)} path(s) rewritten in "
          f"{draft_path.relative_to(PROJECT_ROOT)}")
    return len(edits)


def _print_report(rows) -> None:
    by = {"OK": 0, "MISMATCH": 0, "NOT_IN_REGISTRY": 0}
    for r in rows:
        by[r["status"]] += 1
    print(f"\nScanned {len(rows)} image ref(s): "
          f"OK={by['OK']}  MISMATCH={by['MISMATCH']}  "
          f"NOT_IN_REGISTRY={by['NOT_IN_REGISTRY']}\n")
    for r in rows:
        if r["status"] == "OK":
            print(f"  OK        {r['label']:<12} {r['current']}")
    for r in rows:
        if r["status"] == "MISMATCH":
            print(f"  MISMATCH  {r['label']:<12} ({r.get('fig_id','?')})")
            print(f"            current : {r['current']}")
            print(f"            expected: {r['expected']}")
    skipped = [r for r in rows if r["status"] == "NOT_IN_REGISTRY"]
    if skipped:
        print(f"\n  {len(skipped)} ref(s) NOT_IN_REGISTRY (left untouched — "
              f"out of scope: electrode placement / 3a / S3–S6 etc.):")
        for r in skipped:
            print(f"    - {r['label']:<14} {r['current']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Registry-driven draft figure-path checker")
    ap.add_argument("--draft", default=DEFAULT_DRAFT,
                    help=f"draft markdown path (default: {DEFAULT_DRAFT})")
    ap.add_argument("--apply", action="store_true",
                    help="rewrite MISMATCH paths in place (writes .bak first)")
    args = ap.parse_args(argv)

    draft_path = (PROJECT_ROOT / args.draft).resolve()
    if not draft_path.exists():
        print(f"ERROR: draft not found: {args.draft}", file=sys.stderr)
        return 2

    rows, text = check_draft(draft_path)
    _print_report(rows)

    n_mismatch = sum(1 for r in rows if r["status"] == "MISMATCH")
    if args.apply:
        apply_fixes(draft_path, rows, text)
        return 0
    if n_mismatch:
        print(f"\n{n_mismatch} mismatch(es). Re-run with --apply to fix.")
        return 1
    print("\nAll registry-mapped figure refs are canonical-correct.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
