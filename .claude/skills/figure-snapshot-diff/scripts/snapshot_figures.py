"""Step 1 of the figure-snapshot-diff workflow.

Copy a directory of images (PNG, JPG, etc.) to a sibling backup directory tagged
with a descriptive name + timestamp. Run BEFORE modifying any code that affects
the images so the resulting `<source-name>_snapshot_<tag>_<YYYYMMDD_HHMM>/`
folder can be diffed against the post-change state by `build_compare_page.py`.

Usage:
    python snapshot_figures.py --source paper/figures --tag pre_color_unify

Result (example, assuming today is 2026-05-12 12:46):
    paper/figures_snapshot_pre_color_unify_20260512_1246/
        ├── figure_1.png
        ├── figure_2.png
        └── subdir/figure_3.png   # subdirectory structure preserved
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
from pathlib import Path

# Image extensions we treat as "figures" worth snapshotting. Add more here if you
# regularly use SVG/PDF as the canonical figure output and want them included.
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".pdf"}


def _resolve_source(source: Path) -> Path:
    src = source.expanduser().resolve()
    if not src.exists():
        sys.exit(f"[snapshot] source directory does not exist: {src}")
    if not src.is_dir():
        sys.exit(f"[snapshot] source must be a directory, got file: {src}")
    return src


def _compute_dest(source: Path, tag: str, dest_parent: Path | None) -> Path:
    parent = dest_parent.expanduser().resolve() if dest_parent else source.parent
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    safe_tag = "".join(c if (c.isalnum() or c in "._-") else "_" for c in tag).strip("_") or "snapshot"
    name = f"{source.name}_snapshot_{safe_tag}_{stamp}"
    return parent / name


def _copy_images(source: Path, dest: Path) -> tuple[int, int]:
    """Copy every image file (by extension) preserving the relative path. Returns (n_files, total_bytes)."""
    n_files = 0
    total_bytes = 0
    for src_path in source.rglob("*"):
        if not src_path.is_file():
            continue
        if src_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = src_path.relative_to(source)
        dst_path = dest / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        n_files += 1
        total_bytes += src_path.stat().st_size
    return n_files, total_bytes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Snapshot a directory of figures/images before modifying their generation code.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", required=True, type=Path, help="Directory containing the images to snapshot.")
    parser.add_argument("--tag", required=True, help="Short label describing what change is about to happen (used in the snapshot directory name).")
    parser.add_argument(
        "--dest-parent",
        type=Path,
        default=None,
        help="Parent directory for the snapshot. Defaults to the parent of --source so the backup sits next to the original.",
    )
    args = parser.parse_args(argv)

    source = _resolve_source(args.source)
    dest = _compute_dest(source, args.tag, args.dest_parent)

    if dest.exists():
        sys.exit(f"[snapshot] destination already exists, refusing to overwrite: {dest}")

    dest.mkdir(parents=True)
    n_files, total_bytes = _copy_images(source, dest)

    if n_files == 0:
        # Clean up the empty dir so a noisy snapshot doesn't linger.
        dest.rmdir()
        sys.exit(f"[snapshot] no image files (extensions: {sorted(IMAGE_EXTENSIONS)}) found under {source}")

    size_mb = total_bytes / (1024 * 1024)
    print(f"[snapshot] copied {n_files} image(s), {size_mb:.2f} MB")
    print(f"[snapshot] destination: {dest}")
    print(f"[snapshot] next step: modify your figure-generation code, regenerate the images, then run:")
    print(f"           python build_compare_page.py --backup-dir \"{dest}\" --current-dir \"{source}\" --output figures_compare.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
