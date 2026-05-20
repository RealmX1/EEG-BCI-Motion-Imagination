#!/usr/bin/env python
"""C#4 — Programmatic channel-scaling retention table generator.

Replaces the deleted fig4d (channel_retention_faceted) with a single source-of-
truth Markdown table that maps `[config × channel count] → (pct of 128ch baseline,
pct reduction from 2N tier)`. The table is consumed by the paper draft via the
inlined Appendix section and can be regenerated any time data updates.

Output: `paper/appendix/channel_scaling_retention_table.md`

Per-cell content (each task gets its own table):
    "{pct_of_128:.1f}% ({pct_reduction_from_2N:.1f}%)"

Where:
    pct_of_128            = 100 * mean_acc(N, method, task) / mean_acc(128_baseline, task)
    pct_reduction_from_2N = pct_of_128(N) - pct_of_128(2N)  (drop from 2N → N)

For N=64 there is no same-method 2N=128 counterpart (128 is the baseline itself
across all methods), so the second value is rendered as "—".

Channel counts: [4, 8, 16, 32, 64, 128]
Methods: FDR, Band Power, Attention, CSP, Negative Control

Optional 61ch standard 10-10 row is appended below the matrix as a single
reference line per task (it does not slot into the 2N halving lattice).

Optional 4ch FDR ∩ Attention overlap is appended as a binary-only reference
line (no ternary data).

Run:
    uv run python scripts/paper/generate_channel_scaling_table.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))  # scripts/paper/ for siblings

# NOTE: we intentionally do NOT import from `generate_paper_figures` — that
# module transitively pulls in `src.utils → torch`, which is a multi-GB optional
# dependency this read-only table generator does not need. The two helpers
# (load_json_cache, extract_model_accs) below are inlined and kept BIT-FOR-BIT
# in sync with the originals; if you change one, change the other.
from src.paper.run_registry import (  # noqa: E402
    get_run_entry,
    get_run_path,
    resolve_project_path,
)


def load_json_cache(path: str) -> dict:
    """Load a JSON result cache file. (Mirror of generate_paper_figures.load_json_cache.)"""
    with open(resolve_project_path(path), encoding="utf-8") as f:
        return json.load(f)


def _extract_accs_from_subject_mapping(subject_mapping: Dict[str, dict]) -> List[float]:
    accs = []
    for subject_id, subject_data in sorted(subject_mapping.items()):
        if subject_id in {"metadata", "comparison", "summary", "statistics"}:
            continue
        if not isinstance(subject_data, dict):
            continue
        acc = subject_data.get("test_acc_majority", subject_data.get("test_acc"))
        if acc is not None:
            accs.append(acc * 100)
    return accs


def _extract_accs_from_subject_list(subjects: List[dict]) -> List[float]:
    accs = []
    for subject_data in sorted(
        subjects,
        key=lambda item: item.get("subject_id", "") if isinstance(item, dict) else "",
    ):
        if not isinstance(subject_data, dict):
            continue
        acc = subject_data.get("test_acc_majority", subject_data.get("test_acc"))
        if acc is not None:
            accs.append(acc * 100)
    return accs


def extract_model_accs(cache: dict, model: str) -> List[float]:
    """Inlined mirror of generate_paper_figures.extract_model_accs (torch-free)."""
    results = cache.get("results", {})
    model_results = results.get(model, {}) if isinstance(results, dict) else {}
    if isinstance(model_results, dict):
        per_subj = model_results.get("per_subject_test_acc", {})
        if per_subj:
            return [acc * 100 for _, acc in sorted(per_subj.items())]
        accs = _extract_accs_from_subject_mapping(model_results)
        if accs:
            return accs
    if isinstance(results, dict):
        per_subj = results.get("per_subject_test_acc", {})
        metadata_model = cache.get("metadata", {}).get("model_type")
        if per_subj and metadata_model in (None, model):
            return [acc * 100 for _, acc in sorted(per_subj.items())]
    subjects = cache.get("subjects", [])
    if isinstance(subjects, list):
        metadata_model = cache.get("metadata", {}).get("model_type")
        if metadata_model in (None, model):
            accs = _extract_accs_from_subject_list(subjects)
            if accs:
                return accs
    model_data = cache.get(model, {})
    if isinstance(model_data, dict):
        return _extract_accs_from_subject_mapping(model_data)
    return []


# Layout knobs --------------------------------------------------------------
CHANNEL_TIERS = [4, 8, 16, 32, 64]  # 128ch handled separately (baseline = 100%)
ALL_COLUMNS = CHANNEL_TIERS + [128]

METHODS: List[Tuple[str, str]] = [
    # (display_label, registry_key_suffix)
    ("FDR",              "fdr"),
    ("Band Power",       "band_power"),
    ("Attention",        "attention"),
    ("CSP",              "csp"),
    ("Neg. control",     "negative_control"),
]

TASKS = ["binary", "ternary"]

OUTPUT_PATH = PROJECT_ROOT / "paper" / "appendix" / "channel_scaling_retention_table.md"

AUTOGEN_HEADER = (
    "<!-- AUTOGENERATED by scripts/paper/generate_channel_scaling_table.py;"
    " do not hand-edit -->\n"
    "<!-- Regenerate via: uv run python scripts/paper/generate_channel_scaling_table.py -->\n"
)


# ---------------------------------------------------------------------------

def _safe_get_path(alias: str) -> Optional[str]:
    try:
        get_run_entry(alias)
    except KeyError:
        return None
    return get_run_path(alias)


def _load_mean(alias: str) -> Optional[float]:
    """Return mean cross-subject accuracy (%) for an alias, or None if missing."""
    path = _safe_get_path(alias)
    if path is None:
        return None
    if not resolve_project_path(path).exists():
        return None
    cache = load_json_cache(path)
    accs = extract_model_accs(cache, "cbramod")
    if not accs:
        return None
    return float(np.mean(accs))


def _load_baseline_128(task: str) -> Optional[float]:
    """128ch CBraMod cross-subject baseline (no method dimension)."""
    return _load_mean(f"cross_cbramod_{task}")


def _load_method_at_n(n_ch: int, method_key: str, task: str) -> Optional[float]:
    return _load_mean(f"reduced_{n_ch}_{method_key}_{task}")


def _format_cell(pct_of_128: Optional[float],
                 pct_reduction_from_2N: Optional[float],
                 is_64ch: bool) -> str:
    """Render one matrix cell.

    is_64ch: at N=64 we always emit "—" for the second value (no same-method
    2N=128 row exists; 128 is the baseline across all methods).
    """
    if pct_of_128 is None:
        return "—"
    first = f"{pct_of_128:.1f}%"
    if is_64ch:
        second = "—"
    elif pct_reduction_from_2N is None:
        second = "n/a"
    else:
        second = f"{pct_reduction_from_2N:+.1f}pp"
    return f"{first} ({second})"


def _build_task_table(task: str) -> str:
    """Build Markdown for one task. Returns the table block (no trailing blank)."""
    baseline = _load_baseline_128(task)
    if baseline is None:
        return (f"### {task.capitalize()}: 128ch baseline missing — table cannot be built. "
                f"(Check `cross_cbramod_{task}` in `paper/run_registry.yaml`.)\n")

    # method -> n_ch -> pct_of_128
    pct_table: Dict[str, Dict[int, Optional[float]]] = {}
    for label, key in METHODS:
        pct_table[label] = {}
        for n_ch in CHANNEL_TIERS:
            mean = _load_method_at_n(n_ch, key, task)
            pct_table[label][n_ch] = (100.0 * mean / baseline) if mean is not None else None
        pct_table[label][128] = 100.0  # baseline-per-method assumed = 128ch CBraMod

    # Build header
    header = "| 方法 (method) | " + " | ".join(f"{c}ch" for c in ALL_COLUMNS) + " |"
    align = "|" + "|".join(["---"] * (1 + len(ALL_COLUMNS))) + "|"

    rows = []
    for label, _ in METHODS:
        cells = [f"**{label}**"]
        for n_ch in ALL_COLUMNS:
            if n_ch == 128:
                cells.append(f"{pct_table[label][n_ch]:.1f}% (baseline)")
                continue
            pct = pct_table[label][n_ch]
            two_n = 2 * n_ch
            if two_n in pct_table[label]:
                pct_2n = pct_table[label][two_n]
                drop = (pct - pct_2n) if (pct is not None and pct_2n is not None) else None
            else:
                drop = None
            cells.append(_format_cell(pct, drop, is_64ch=(n_ch == 64)))
        rows.append("| " + " | ".join(cells) + " |")

    # Optional reference rows: 61ch standard 10-10, 4ch FDR∩Att overlap
    extra_lines: List[str] = []
    std_61 = _load_mean(f"standard_1010_61_cross_{task}")
    if std_61 is not None:
        extra_lines.append(
            f"- **61ch standard 10-10** (reference, not in 2× halving lattice): "
            f"{(100.0 * std_61 / baseline):.1f}% of 128ch baseline "
            f"(absolute {std_61:.2f}%)."
        )
    if task == "binary":
        overlap = _load_mean("reduced_4_fdr_attention_overlap_binary")
        if overlap is not None:
            extra_lines.append(
                f"- **4ch FDR ∩ Attention overlap** (favorable outlier, binary only): "
                f"{(100.0 * overlap / baseline):.1f}% of 128ch baseline "
                f"(absolute {overlap:.2f}%)."
            )

    parts = [
        f"### {task.capitalize()} task (CBraMod cross-subject, N = 21)",
        "",
        f"> 128ch baseline = **{baseline:.2f}%** absolute accuracy. "
        f"Each cell: `pct of 128ch baseline (pct-point change from the 2× tier)`. "
        f"`—` in the 64ch column means no same-method 2N=128 counterpart "
        f"(128ch baseline is shared across all methods).",
        "",
        header,
        align,
        *rows,
    ]
    if extra_lines:
        parts.extend(["", *extra_lines])
    return "\n".join(parts) + "\n"


def build_full_document() -> str:
    blocks = [
        AUTOGEN_HEADER,
        "## Appendix A. Per-Config Channel Scaling Retention\n",
        "本程序化生成表给出 [配置 × 通道数] 网格上的两项保留率指标 "
        "(C#4, 替代 2026-05-20 删除的 fig4d 手绘版本):\n",
        "1. **`pct of 128ch`** — 该 (方法, 通道数) cell 的绝对准确率 / 128ch CBraMod "
        "跨被试 baseline 准确率 × 100 (按 task 分别归一化)。\n",
        "2. **`pct change from 2N`** — 同方法在 2× 通道数档位上的 `pct of 128ch` 减去当前 "
        "档位的 `pct of 128ch`, 单位 pp (percentage points)。负值表示通道减半带来的额外 "
        "retention 损失 (符号约定: 数据驱动方法通常为负, 量级越大说明通道减半的边际代价越高)。\n",
        "3. **64ch 列**没有同方法的 2N=128 counterpart (128 baseline 跨所有方法共享), "
        "故 `pct change from 2N` 一律渲染为 `—`。\n",
        "4. **128ch 列**所有方法都等于 baseline (100.0%), 仅作锚点。\n",
        "数据来源: 全部经 `paper/run_registry.yaml` alias 加载, 与 §3.5.2 图 4 / 表 9 "
        "/ 表 10 共用 cache; 因此本表与正文中数字逐位一致。重新生成命令: "
        "`uv run python scripts/paper/generate_channel_scaling_table.py`。\n",
    ]
    for task in TASKS:
        blocks.append(_build_task_table(task))
        blocks.append("")
    return "\n".join(blocks)


def main():
    doc = build_full_document()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(doc, encoding="utf-8")
    print(f"[ok] wrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"     {len(doc.splitlines())} lines, "
          f"{len(doc.encode('utf-8'))} bytes")
    # tiny per-task sanity summary
    for task in TASKS:
        baseline = _load_baseline_128(task)
        if baseline is None:
            print(f"     [warn] {task}: baseline missing")
            continue
        n_filled = 0
        n_total = 0
        for _, key in METHODS:
            for n_ch in CHANNEL_TIERS:
                n_total += 1
                if _load_method_at_n(n_ch, key, task) is not None:
                    n_filled += 1
        print(f"     [{task}] baseline = {baseline:.2f}%; "
              f"matrix cells filled: {n_filled}/{n_total}")


if __name__ == "__main__":
    main()
