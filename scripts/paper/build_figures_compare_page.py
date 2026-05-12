"""Generate paper/figures_compare.html — side-by-side before/after slider for every figure
modified by the 2026-05-12 standardization pass.

用法:
    uv run python scripts/paper/build_figures_compare_page.py

输出:
    paper/figures_compare.html  (浏览器打开即可拖动对比)

页面通过 img-comparison-slider Web Component (CDN) 渲染叠加滑动对比。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from html import escape

try:
    from PIL import Image  # type: ignore
except ImportError:  # pragma: no cover
    Image = None  # 退化：未安装 Pillow 时 fallback 为静态 4:3

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = REPO_ROOT / "paper"
BACKUP_DIR = PAPER_DIR / "figures_backup_20260512_pre_standardization"
FIGURES_DIR = PAPER_DIR / "figures"
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_HTML = PAPER_DIR / "figures_compare.html"

# 描述提示 (filename basename -> 1-line description)
DESCRIPTIONS: dict[str, str] = {
    "channel_method_ranking_flip.png": "图 4b: 删除每节点 #rank+value 内联标签，改为顶端 sorted stack",
    "sensitivity_scaling.png": "图 4c: 内联 annotate 替换为 force-directed label 布局",
    "extra_sessions_binary.png": "图 7: mean line 减细加透明 (zorder=2)，scatter 加白边 (zorder=3)",
    "extra_sessions_ternary.png": "图 8: mean line 减细加透明 (zorder=2)，scatter 加白边 (zorder=3)",
    "32ch_comparison.png": "图 3b: 删图级 title + 字号/颜色统一",
    "dapt_v1_v5_smallmultiples.png": "图 10a: small-multiples 6-panel (rows=task, cols=paradigm)",
    "exploratory_ablation_overview.png": "图 12: 仅样式刷扫；待用户重画规格",
    "channel_scaling_curve.png": "图 4: 字号/颜色统一 + 删图级 title",
    "cross_subject_pooling_forest.png": "图 2b: 字号/颜色统一",
    "subject_heatmap.png": "图 S2: 删图级 title + 字号统一",
    "inference_latency.png": "图 11: 字号/颜色统一",
    "further_pretraining.png": "图 10b: 字号/颜色统一 + 删图级 title",
    "extra_sessions_paradigm_binary.png": "Extra sessions paradigm: 字号/颜色统一",
    "extra_sessions_strategy_comparison.png": "Extra sessions strategy: 删图级 title + 字号统一",
    "fig5_4ch_optimal_vs_neg_control.png": "图 5: 字号/颜色统一",
    "20260323_2237_combined_imagery_binary.png": "图 1: comparison.py 第三栏改 force-directed labels + 删图级 title",
    "20260330_0709_cross-subject_combined_imagery_binary.png": "图 2: comparison.py 第三栏改 force-directed labels",
    "20260330_0836_cross-subject_combined_imagery_binary.png": "图 3c: comparison.py 第三栏改 force-directed labels",
    "20260329_0507_transfer_combined_imagery_binary.png": "图 6: comparison.py 第三栏改 force-directed labels + 删图级 title（通过 --replot 20260329_0507 重生成）",
    "20260329_0448_transfer_combined_imagery_ternary.png": "图 6b: 同上（通过 --replot 20260329_0448 重生成）",
}

# 论文图编号排序权重 (sort key)
PAPER_ORDER: dict[str, tuple[int, int]] = {
    "20260323_2237_combined_imagery_binary.png": (1, 0),  # 图 1
    "20260330_0709_cross-subject_combined_imagery_binary.png": (2, 0),  # 图 2
    "cross_subject_pooling_forest.png": (2, 1),  # 图 2b
    "20260330_0836_cross-subject_combined_imagery_binary.png": (3, 2),  # 图 3c
    "32ch_comparison.png": (3, 1),  # 图 3b
    "channel_scaling_curve.png": (4, 0),  # 图 4
    "channel_method_ranking_flip.png": (4, 1),  # 图 4b
    "sensitivity_scaling.png": (4, 2),  # 图 4c
    "fig5_4ch_optimal_vs_neg_control.png": (5, 0),  # 图 5
    "20260329_0507_transfer_combined_imagery_binary.png": (6, 0),  # 图 6
    "20260329_0448_transfer_combined_imagery_ternary.png": (6, 1),  # 图 6b
    "extra_sessions_binary.png": (7, 0),  # 图 7
    "extra_sessions_ternary.png": (8, 0),  # 图 8
    "dapt_v1_v5_smallmultiples.png": (10, 0),  # 图 10a
    "further_pretraining.png": (10, 1),  # 图 10b
    "inference_latency.png": (11, 0),  # 图 11
    "exploratory_ablation_overview.png": (12, 0),  # 图 12
    "subject_heatmap.png": (99, 2),  # 图 S2 (supplementary — sort after main figures)
}

# --- helpers ---------------------------------------------------------------

_HASH_CACHE: dict[str, str] = {}
_SIZE_CACHE: dict[str, tuple[int, int]] = {}


def _parse_label_and_desc(name: str) -> tuple[str | None, str]:
    """Parse the DESCRIPTIONS entry into (paper_label, change_description).

    Examples:
        "图 4b: 删除每节点 #rank+value..." → ("图 4b", "删除每节点 #rank+value...")
        "图 S2: 删图级 title..." → ("图 S2", "删图级 title...")
        "Extra sessions paradigm: 字号/颜色统一" → (None, "Extra sessions paradigm: 字号/颜色统一")

    Only treats the prefix as a paper label when it starts with "图". This avoids
    accidentally consuming the prefix of non-paper-numbered descriptions.
    """
    description = DESCRIPTIONS.get(name, "")
    if ":" in description:
        prefix, rest = description.split(":", 1)
        prefix = prefix.strip()
        if prefix.startswith("图"):
            return prefix, rest.strip()
    return None, description


def _hash_file(path: Path) -> str:
    """SHA-256 hash of file contents, cached per resolved path."""
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
    """True iff both files have identical byte size and identical SHA-256."""
    try:
        if p1.stat().st_size != p2.stat().st_size:
            return False
        return _hash_file(p1) == _hash_file(p2)
    except OSError:
        return False


def _image_size(path: Path) -> tuple[int, int]:
    """Return (width, height) of a PNG; fallback to (4, 3) if Pillow missing."""
    key = str(path.resolve())
    if key in _SIZE_CACHE:
        return _SIZE_CACHE[key]
    if Image is None:
        size = (4, 3)
    else:
        try:
            with Image.open(path) as im:
                size = im.size  # (w, h)
        except Exception:
            size = (4, 3)
    _SIZE_CACHE[key] = size
    return size


def fmt_size(n_bytes: int) -> str:
    return f"{n_bytes / 1024:.1f} KB"


def fmt_delta(before: int, after: int) -> tuple[str, str]:
    """Return (text, css_class)."""
    if before == 0:
        return ("n/a", "")
    pct = (after - before) / before * 100
    text = f"{pct:+.1f}%"
    if abs(pct) < 1.0:
        return (text, "")
    return (text, "delta-neg" if pct < 0 else "delta-pos")


def relpath_from_html(path: Path) -> str:
    """Compute a forward-slash relative path from paper/figures_compare.html to path."""
    rel = path.resolve().relative_to(PAPER_DIR.resolve()) if path.resolve().is_relative_to(PAPER_DIR.resolve()) else None
    if rel is not None:
        return rel.as_posix()
    # results/ paths — go up one from paper/
    rel = path.resolve().relative_to(REPO_ROOT.resolve())
    return ("../" + rel.as_posix())


# --- per-pair HTML emission ------------------------------------------------

HANDLE_SVG = (
    '<svg slot="handle" xmlns="http://www.w3.org/2000/svg" '
    'width="56" height="56" viewBox="-15 -15 30 30">'
    '<circle r="13" fill="#ff6b35" stroke="white" stroke-width="2"/>'
    '<path d="M-7 -4 L-2 0 L-7 4 Z M7 -4 L2 0 L7 4 Z" fill="white"/>'
    "</svg>"
)


def _slider_block(
    before_src: str,
    after_src: str,
    before_path: Path,
    after_path: Path,
    before_alt: str = "before",
    after_alt: str = "after",
) -> str:
    """Emit one slider wrapped with aspect-ratio + BEFORE/AFTER badges + custom handle.

    The wrapper aspect-ratio is set from whichever image has the *smaller* w/h
    (i.e. the taller image). Combined with ``object-fit: contain`` on the inner
    <img> elements, this guarantees both BEFORE and AFTER render in full —
    the shorter-relative image letterboxes inside the container instead of
    being cropped.
    """
    wb, hb = _image_size(before_path)
    wa, ha = _image_size(after_path)
    # Pick the dimensions of whichever image has the smaller w/h ratio (taller).
    if wb * ha <= wa * hb:  # wb/hb <= wa/ha
        aspect_w, aspect_h = wb, hb
    else:
        aspect_w, aspect_h = wa, ha
    return (
        f'      <div class="slider-wrap" style="aspect-ratio: {aspect_w}/{aspect_h};">\n'
        f'        <span class="badge badge-before">BEFORE</span>\n'
        f'        <span class="badge badge-after">AFTER</span>\n'
        f'        <img-comparison-slider>\n'
        f'          <img slot="first"  src="{escape(before_src)}" alt="{escape(before_alt)}"/>\n'
        f'          <img slot="second" src="{escape(after_src)}" alt="{escape(after_alt)}"/>\n'
        f'          {HANDLE_SVG}\n'
        f'        </img-comparison-slider>\n'
        f'      </div>'
    )


def collect_entries() -> list[dict]:
    entries: list[dict] = []

    # 1) paper/figures/ pairs
    for backup_png in sorted(BACKUP_DIR.glob("*.png")):
        name = backup_png.name
        if "_copy" in name:
            continue
        current = FIGURES_DIR / name
        if not current.exists():
            continue
        entries.append({
            "id": name.replace(".png", ""),
            "title": name,
            "before_path": backup_png,
            "after_path": current,
            "after_label": "After (figures/)",
            "description": DESCRIPTIONS.get(name, "样式标准化（颜色/字号统一 + 删图级 title）"),
            "extra_afters": [],
        })

    # 2) results/ paper-referenced PNGs
    results_specs = [
        ("20260323_2237_combined_imagery_binary.png", "20260323_2237_combined_imagery_binary.png", None),
        ("20260330_0709_cross-subject_combined_imagery_binary.png",
         "20260330_0709_cross-subject_combined_imagery_binary.png", None),
        ("20260329_0507_transfer_combined_imagery_binary.png",
         "20260329_0507_transfer_combined_imagery_binary.png", None),
        ("20260329_0448_transfer_combined_imagery_ternary.png",
         "20260329_0448_transfer_combined_imagery_ternary.png", None),
        ("32_channel/fdr/20260330_0836_cross-subject_combined_imagery_binary.png",
         "32_channel/fdr/20260330_0836_cross-subject_combined_imagery_binary.png", None),
    ]
    for backup_rel, after_rel, new_after_rel in results_specs:
        backup_path = BACKUP_DIR / "results" / backup_rel
        after_path = RESULTS_DIR / after_rel
        if not backup_path.exists():
            continue
        if not after_path.exists():
            continue
        name = Path(backup_rel).name
        extra_afters = []
        if new_after_rel is not None:
            new_after_path = RESULTS_DIR / new_after_rel
            if new_after_path.exists():
                extra_afters.append({
                    "label": f"After-new (results/{new_after_rel})",
                    "path": new_after_path,
                })
        entries.append({
            "id": "results-" + name.replace(".png", ""),
            "title": name + "  (results/)",
            "before_path": backup_path,
            "after_path": after_path,
            "after_label": f"After (results/{after_rel})",
            "description": DESCRIPTIONS.get(name, "样式标准化（颜色/字号统一 + 删图级 title）"),
            "extra_afters": extra_afters,
        })

    # Sort: paper-referenced first (by figure number), then others
    def sort_key(entry: dict):
        name = Path(entry["title"].split("  ")[0]).name
        order = PAPER_ORDER.get(name)
        if order is not None:
            return (0, order[0], order[1], name)
        return (1, 999, 0, name)

    entries.sort(key=sort_key)
    return entries


def build_html(entries: list[dict]) -> str:
    def _nav_text(entry: dict) -> str:
        name = Path(entry["title"].split("  ")[0]).name
        label, _ = _parse_label_and_desc(name)
        return label if label else name.replace(".png", "")

    nav_links = "\n      ".join(
        f'<a href="#{escape(e["id"])}">{escape(_nav_text(e))}</a>'
        for e in entries
    )

    sections = []
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
        before_src = relpath_from_html(before_path)
        after_src = relpath_from_html(after_path)

        # Detect byte-identical before/after (Fix 5) — suppress the main slider.
        identical_main = _files_identical(before_path, after_path)

        extra_blocks = []
        for extra in e["extra_afters"]:
            extra_size = extra["path"].stat().st_size
            extra_delta_text, extra_delta_class = fmt_delta(before_size, extra_size)
            extra_delta_span = (
                f'<span class="{extra_delta_class}">Δ: {escape(extra_delta_text)}</span>'
                if extra_delta_class else f"<span>Δ: {escape(extra_delta_text)}</span>"
            )
            extra_src = relpath_from_html(extra["path"])
            heading_prefix = "Primary comparison" if identical_main else "vs"
            extra_blocks.append(
                f'\n      <h3 class="extra-heading">{escape(heading_prefix)} {escape(extra["label"])}</h3>\n'
                f'      <p class="stats">\n'
                f'        <span>Before: {escape(fmt_size(before_size))}</span>\n'
                f'        <span>After-new: {escape(fmt_size(extra_size))}</span>\n'
                f'        {extra_delta_span}\n'
                f'      </p>\n'
                f'{_slider_block(before_src, extra_src, before_path, extra["path"], before_alt="before", after_alt="after-new")}'
            )

        extra_html = "\n".join(extra_blocks)

        title_base = Path(e["title"].split("  ")[0]).name
        source_kind = " (results/)" if "results/" in e["title"] else " (paper/figures/)"
        paper_lbl, desc_no_prefix = _parse_label_and_desc(title_base)
        # 新标题格式: "图 N — 改动描述 (源目录)"; 无 paper label 时回退 PNG basename
        if paper_lbl:
            section_heading = f"{paper_lbl} — {desc_no_prefix}{source_kind}"
            # description 已并入标题，meta 行展示原文件名作技术参考
            meta_text = title_base
        else:
            section_heading = f"{title_base}{source_kind}"
            meta_text = e.get("description", "")

        # Build main body: either main slider OR yellow notice.
        if identical_main:
            has_extra = bool(e["extra_afters"])
            secondary_note = (
                "见下方 After-new 对比。" if has_extra
                else "本次会话未重新生成此 PNG。"
            )
            main_body = (
                f'      <div class="notice notice-warn">\n'
                f'        <strong>⚠ Not regenerated this session</strong> — '
                f'before 与 after 字节级一致（SHA-256 相同），仅源码 <code>comparison.py</code> 已修补。{escape(secondary_note)}\n'
                f'      </div>'
            )
        else:
            main_body = _slider_block(before_src, after_src, before_path, after_path)

        sections.append(
            f'\n    <section id="{escape(e["id"])}">\n'
            f'      <h2>{escape(section_heading)}</h2>\n'
            f'      <p class="meta"><code>{escape(meta_text)}</code></p>\n'
            f'      <p class="stats">\n'
            f'        <span>Before: {escape(fmt_size(before_size))}</span>\n'
            f'        <span>After: {escape(fmt_size(after_size))}</span>\n'
            f'        {delta_span}\n'
            f'      </p>\n'
            f'{main_body}{extra_html}\n'
            f'    </section>'
        )

    sections_html = "\n".join(sections)

    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <title>论文图表 — Before/After 对比</title>
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
    .extra-heading {{ font-size: 1.1rem; font-weight: 600; margin-top: 1.5rem; color: #555; }}
    .meta {{ color: #555; font-size: 1.05rem; margin: 0.25rem 0 0.75rem; }}

    /* Slider wrapper now drives aspect ratio per image. */
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
      width: 100% !important;
      height: 100% !important;
      object-fit: contain !important;
      background: #fafafa;
    }}

    /* BEFORE / AFTER badges overlaid on slider corners. */
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
  <h1>论文图表 Before/After 对比</h1>
  <p class="meta">Generated 2026-05-12. Before = pre-standardization 快照 (<code>paper/figures_backup_20260512_pre_standardization/</code>)。After = 当前状态 (<code>paper/figures/</code> 或 <code>results/</code>)。拖动橙色手柄左右擦除即可对比。</p>
  <nav>
    <strong>Jump to:</strong>
    {nav_links}
  </nav>
{sections_html}
</body>
</html>
'''


def main() -> None:
    entries = collect_entries()
    if not entries:
        raise SystemExit("No figure pairs found — abort.")
    html = build_html(entries)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT_HTML} with {len(entries)} figure pairs.")


if __name__ == "__main__":
    main()
