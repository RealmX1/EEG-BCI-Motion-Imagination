"""Stat recompute for V1/V2/V3 transfer 6 new DAPT cells.

Mirrors `scripts/internal/recompute_v4v5_within_transfer.py` methodology:
- scipy.stats.ttest_rel (two-sided, paired)
- Cohen's dz = mean_diff / sd_diff
- 95% CI via t-distribution: mean_diff ± t_crit * SE
- BH-FDR @ 0.05 within full 30-cell DAPT family (existing 24 + new 6)
- Stouffer Z combine via signed inverse-normal
- Arithmetic mean Δ over each paradigm-level aggregate (5 V cells per paradigm × task)

Output:
- paper/reviews/stage4_step1d_v1v2v3_transfer.md (audit doc)
- prints 6 new tuples for DAPT_V_RESULTS literal in scripts/paper/generate_paper_figures.py
- prints 2 updated Stouffer dict entries (transfer_binary, transfer_ternary)
  + 4 preserved (cross_binary, cross_ternary, within_binary, within_ternary)
"""
from __future__ import annotations
import json
from pathlib import Path
import scipy.stats as stats
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT = REPO / "paper" / "reviews" / "stage4_step1d_v1v2v3_transfer.md"

# 6 new V1/V2/V3 transfer JSONs + 2 baseline JSONs (shared with V4/V5 transfer)
NEW_CELLS = [
    # (V, paradigm, task, treatment_json, baseline_json)
    ("V1", "transfer", "binary",  "dapt_v1/20260510_2357_transfer_cache_imagery_binary.json",  "20260329_0507_transfer_cache_imagery_binary.json"),
    ("V1", "transfer", "ternary", "dapt_v1/20260511_0012_transfer_cache_imagery_ternary.json", "20260329_0521_transfer_cache_imagery_ternary.json"),
    ("V2", "transfer", "binary",  "dapt_v2/20260511_0031_transfer_cache_imagery_binary.json",  "20260329_0507_transfer_cache_imagery_binary.json"),
    ("V2", "transfer", "ternary", "dapt_v2/20260511_0042_transfer_cache_imagery_ternary.json", "20260329_0521_transfer_cache_imagery_ternary.json"),
    ("V3", "transfer", "binary",  "dapt_v3/20260511_0058_transfer_cache_imagery_binary.json",  "20260329_0507_transfer_cache_imagery_binary.json"),
    ("V3", "transfer", "ternary", "dapt_v3/20260511_0109_transfer_cache_imagery_ternary.json", "20260329_0521_transfer_cache_imagery_ternary.json"),
]

# Existing 24-cell Step 1c (V1-V5 within+cross + V4/V5 transfer) Δ, p values for joint BH-FDR + Stouffer.
# Direction: positive = DAPT > Baseline. Values from paper §3.6 table 16 (lines 748-771) and
# stage4_step1c_v4v5_within_transfer.md (8 V4/V5 within+transfer cells).
EXISTING_24 = [
    ("V1", "within",   "binary",  -1.25, 0.115),
    ("V1", "within",   "ternary", -0.30, 0.656),
    ("V1", "cross",    "binary",  -1.85, 0.009),
    ("V1", "cross",    "ternary", +0.79, 0.353),
    ("V2", "within",   "binary",  -2.86, 0.002),
    ("V2", "within",   "ternary", -1.47, 0.093),
    ("V2", "cross",    "binary",  -1.25, 0.025),
    ("V2", "cross",    "ternary", +0.44, 0.462),
    ("V3", "within",   "binary",  -1.34, 0.112),
    ("V3", "within",   "ternary", -0.24, 0.729),
    ("V3", "cross",    "binary",  -1.46, 0.051),
    ("V3", "cross",    "ternary", +0.62, 0.384),
    ("V4", "within",   "binary",  -1.10, 0.194),
    ("V4", "within",   "ternary", -0.56, 0.553),
    ("V4", "cross",    "binary",  -1.61, 0.008),
    ("V4", "cross",    "ternary", +0.22, 0.808),
    ("V4", "transfer", "binary",  -1.67, 0.026),
    ("V4", "transfer", "ternary", -0.32, 0.709),
    ("V5", "within",   "binary",  -2.92, 0.020),
    ("V5", "within",   "ternary", -2.02, 0.078),
    ("V5", "cross",    "binary",  -2.77, 0.014),
    ("V5", "cross",    "ternary", -1.17, 0.137),
    ("V5", "transfer", "binary",  -1.22, 0.086),
    ("V5", "transfer", "ternary", -1.47, 0.059),
]


def load_per_subject(path: Path, model: str = "cbramod") -> dict[str, float]:
    """Return {subject_id: test_acc} from cache JSON nested-dict schema."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    res = data["results"]
    if model not in res:
        raise KeyError(f"Model '{model}' not found in {path}; available: {list(res.keys())}")
    model_block = res[model]
    out: dict[str, float] = {}
    for sid, entry in model_block.items():
        acc = entry.get("test_acc")
        if isinstance(acc, (int, float)) and not (isinstance(acc, float) and np.isnan(acc)):
            out[sid] = float(acc)
    return out


def paired_ttest_summary(treat: dict[str, float], base: dict[str, float]) -> dict:
    common = sorted(set(treat) & set(base))
    if len(common) < 5:
        raise RuntimeError(f"Too few paired subjects: n={len(common)}")
    t_arr = np.array([treat[s] for s in common])
    b_arr = np.array([base[s] for s in common])
    diff = t_arr - b_arr
    n = len(diff)
    mean_diff = float(diff.mean())
    sd_diff = float(diff.std(ddof=1))
    t_stat, p = stats.ttest_rel(t_arr, b_arr)
    dz = mean_diff / sd_diff if sd_diff > 0 else 0.0
    se = sd_diff / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_lo = mean_diff - t_crit * se
    ci_hi = mean_diff + t_crit * se
    return {
        "n": n,
        "mean_treat_pct": float(t_arr.mean()) * 100,
        "mean_base_pct": float(b_arr.mean()) * 100,
        "mean_diff_pct": mean_diff * 100,
        "sd_diff_pct": sd_diff * 100,
        "t": float(t_stat),
        "p": float(p),
        "dz": float(dz),
        "ci_lo_pct": ci_lo * 100,
        "ci_hi_pct": ci_hi * 100,
    }


def bh_fdr(p_values: list[float], alpha: float = 0.05):
    p = np.array(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranks = np.arange(1, n + 1)
    q_sorted = p[order] * n / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(q_sorted, 0, 1)
    sig = q < alpha
    return q.tolist(), sig.tolist()


def stouffer(p_values: list[float], directions: list[float]):
    """Signed Stouffer combination.
    p_values: two-sided raw p; directions: ± mean_diff signs.
    Returns (Z_signed, two_sided_p).
    """
    p_arr = np.array(p_values, dtype=float)
    sign = np.array([1.0 if d >= 0 else -1.0 for d in directions])
    one_sided = np.where(sign > 0, p_arr / 2, 1 - p_arr / 2)
    z_arr = stats.norm.ppf(1 - one_sided)
    z_combined = float(z_arr.sum() / np.sqrt(len(z_arr)))
    p_combined = 2 * (1 - stats.norm.cdf(abs(z_combined)))
    return z_combined, float(p_combined)


def main() -> None:
    print("=" * 78)
    print("Loading 6 new V1/V2/V3 transfer + 2 baseline JSONs")
    print("=" * 78)

    new_results = []
    for V, paradigm, task, treat_rel, base_rel in NEW_CELLS:
        treat_path = RESULTS / treat_rel
        base_path = RESULTS / base_rel
        if not treat_path.exists():
            raise FileNotFoundError(f"Missing treatment cache: {treat_path}")
        if not base_path.exists():
            raise FileNotFoundError(f"Missing baseline cache: {base_path}")
        treat = load_per_subject(treat_path)
        base = load_per_subject(base_path)
        s = paired_ttest_summary(treat, base)
        new_results.append((V, paradigm, task, s, str(treat_rel), str(base_rel)))
        print(
            f"  {V} {paradigm:<8} {task:<7}  n={s['n']:2d}  "
            f"treat={s['mean_treat_pct']:6.2f}%  base={s['mean_base_pct']:6.2f}%  "
            f"Δ={s['mean_diff_pct']:+.2f} pp  t={s['t']:+.3f}  p={s['p']:.4f}  "
            f"dz={s['dz']:+.3f}  CI=[{s['ci_lo_pct']:+.2f}, {s['ci_hi_pct']:+.2f}]"
        )

    # ---- 30-cell joint BH-FDR ----
    all_cells = []  # (label, mean_pp, p)
    for V, paradigm, task, mean_pp, p in EXISTING_24:
        all_cells.append((f"{V}_{paradigm}_{task}", mean_pp, p))
    for V, paradigm, task, s, *_ in new_results:
        all_cells.append((f"{V}_{paradigm}_{task}", s["mean_diff_pct"], s["p"]))
    p_all = [c[2] for c in all_cells]
    q_all, sig_all = bh_fdr(p_all, alpha=0.05)
    print()
    print(f"BH-FDR @ 0.05 over 30-cell DAPT family: {sum(sig_all)} survivors")
    print()
    print("All 30 cells (sorted by p ascending):")
    sort_idx = np.argsort(p_all)
    for i in sort_idx:
        label, mean_pp, p = all_cells[i]
        q = q_all[i]
        marker = "  **Y**" if sig_all[i] else ""
        print(f"  {label:<28}  Δ={mean_pp:+6.2f} pp  p={p:.4f}  q={q:.4f}{marker}")

    # ---- 6 paradigm-level Stouffer aggregates (5V each) ----
    print()
    print("=" * 78)
    print("6 paradigm-level Stouffer aggregates over the full 5V × 3-paradigm matrix")
    print("=" * 78)

    aggregates = {}
    for paradigm in ("within", "cross", "transfer"):
        for task in ("binary", "ternary"):
            sub = [c for c in EXISTING_24 if c[1] == paradigm and c[2] == task]
            sub_new = [(s, V) for V, p_, t_, s, *_ in new_results if p_ == paradigm and t_ == task]
            means = [c[3] for c in sub] + [s[0]["mean_diff_pct"] for s in sub_new]
            pvals = [c[4] for c in sub] + [s[0]["p"] for s in sub_new]
            n_cells = len(means)
            mean_delta = float(np.mean(means))
            z, p_comb = stouffer(pvals, means)
            aggregates[f"{paradigm}_{task}"] = {
                "n_cells": n_cells,
                "mean_delta_pp": mean_delta,
                "Z": z,
                "p": p_comb,
            }
            print(
                f"  {paradigm:<8} {task:<7}  n_cells={n_cells}  "
                f"mean Δ={mean_delta:+.3f} pp  Z={z:+.3f}  p={p_comb:.4f}"
            )

    # ---- Tuples for generate_paper_figures.py ----
    print()
    print("=" * 78)
    print("New tuples for DAPT_V_RESULTS literal (paste below V5 transfer_ternary line)")
    print("=" * 78)
    new_q_slice = q_all[24:]
    new_sig_slice = sig_all[24:]
    for i, (V, paradigm, task, s, *_) in enumerate(new_results):
        q = new_q_slice[i]
        sig = new_sig_slice[i]
        print(
            f"    ('{V}', '{paradigm:<8}', '{task:<7}', "
            f"{s['mean_diff_pct']:+.2f}, "
            f"{s['ci_lo_pct']:+.2f}, {s['ci_hi_pct']:+.2f}, "
            f"{q:.3f}, {str(sig):>5}, {s['p']:.3f}),"
        )

    write_audit_doc(new_results, q_all, sig_all, all_cells, aggregates)
    print()
    print(f"Audit doc written: {OUT}")


def write_audit_doc(new_results, q_all, sig_all, all_cells, aggregates) -> None:
    """Write paper/reviews/stage4_step1d_v1v2v3_transfer.md (parallel to step1c)."""
    lines: list[str] = []
    lines.append("# Stage 4 Step 1d — V1/V2/V3 transfer Statistical Recompute")
    lines.append("")
    lines.append("**Date**: 2026-05-12")
    lines.append("**Supersedes (extends)**: `stage4_step1c_v4v5_within_transfer.md` (24 cells → 30 cells)")
    lines.append("**Scope**: 6 new V1/V2/V3 transfer cells; full DAPT family BH-FDR re-applied at 30 cells; 6 paradigm-level Stouffer aggregates fully populated to 5V each.")
    lines.append("**Method**: Identical to Step 1c — `scipy.stats.ttest_rel` (two-sided paired), Cohen's dz = mean_diff/sd_diff, 95% CI via t-distribution (df=n−1), BH-FDR @ α=0.05 within new 30-cell DAPT family, Stouffer combination via signed inverse-normal.")
    lines.append("")
    lines.append("> All `mean_diff` and 95% CI shown in **percentage points (pp)**. Effect direction = treatment − baseline (positive → DAPT > Baseline).")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Section 1: data sources
    lines.append("## 1. Data Sources (6 new V1/V2/V3 transfer cells)")
    lines.append("")
    lines.append("All 6 caches in commit `90b9fc4` (handoff `docs/handoffs/2026-05-10_dapt_v4_v5.md` 追加 (2026-05-11) section).")
    lines.append("Baselines shared with V4/V5 transfer (Step 1c).")
    lines.append("")
    lines.append("| Cell | Treatment cache | Baseline cache |")
    lines.append("|---|---|---|")
    for V, paradigm, task, _s, treat_rel, base_rel in new_results:
        lines.append(f"| {V} {paradigm} {task} | `results/{treat_rel}` | `results/{base_rel}` |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Section 2: recomputed 6 cells
    lines.append("## 2. Recomputed 6 new cells (paired-t, dz, 95% CI, BH-q within new 30-cell family)")
    lines.append("")
    lines.append("| V | Paradigm | Task | n | mean_treat (%) | mean_base (%) | mean_diff (pp) | SD_diff | t | p (raw) | dz | 95% CI (pp) | q (BH, 30-family) | BH sig? |")
    lines.append("|---|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|:-:|---:|:-:|")
    for i, (V, paradigm, task, s, *_) in enumerate(new_results):
        q = q_all[24 + i]
        sig = sig_all[24 + i]
        bh_mark = "**Y**" if sig else "n"
        lines.append(
            f"| {V} | {paradigm} | {task} | {s['n']} | "
            f"{s['mean_treat_pct']:.2f} | {s['mean_base_pct']:.2f} | "
            f"{s['mean_diff_pct']:+.2f} | {s['sd_diff_pct']:.2f} | "
            f"{s['t']:+.3f} | {s['p']:.3f} | {s['dz']:+.3f} | "
            f"[{s['ci_lo_pct']:+.2f}, {s['ci_hi_pct']:+.2f}] | {q:.3f} | {bh_mark} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")

    # Section 3: 30-cell BH
    lines.append("## 3. BH-FDR Re-application (30-cell DAPT family)")
    lines.append("")
    n_sig = sum(sig_all)
    lines.append(f"**Survivors at q < 0.05 within 30-cell DAPT family: {n_sig}**.")
    lines.append("")
    step1c_q = {
        "V2_within_binary": 0.048,
        "V1_cross_binary":  0.072,
        "V4_cross_binary":  0.072,
    }
    lines.append("Most-significant 5 cells (q ascending) and Step 1c (24-family) comparison:")
    lines.append("")
    lines.append("| Cell | mean_diff (pp) | p (raw) | q (24-family, Step 1c) | q (30-family, Step 1d) | shift |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    sort_idx = np.argsort([c[2] for c in all_cells])
    for i in sort_idx[:5]:
        label, mean_pp, p = all_cells[i]
        q = q_all[i]
        old = step1c_q.get(label)
        old_str = f"{old:.3f}" if old is not None else "—"
        shift = f"{q-old:+.3f}" if old is not None else "(was n.s.)"
        lines.append(f"| `{label}` | {mean_pp:+.2f} | {p:.4f} | {old_str} | {q:.3f} | {shift} |")
    lines.append("")
    if n_sig == 0:
        lines.append("**`V2_within_binary`** — the lone Step 1c survivor (q=0.048 in 24-family) — exits BH significance at q=0.060 in the 30-family. Family-size correction is the sole cause; the cell's raw p=0.002 is unchanged. Read paradigm-level Stouffer aggregates (Section 4) for collective evidence rather than single-cell BH.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Section 4: 6 paradigm-level Stouffer
    lines.append("## 4. Six Paradigm-Level Stouffer Aggregates (5V each)")
    lines.append("")
    lines.append("With V1/V2/V3 transfer added, the `transfer_binary` and `transfer_ternary` aggregates are upgraded from n_cells=2 (V4/V5 only, Step 1c) to n_cells=5 (V1–V5). The other four aggregates are unchanged from Step 1c.")
    lines.append("")
    lines.append("| Aggregate | n_cells | mean Δ (pp) | Stouffer Z (signed) | Combined p (two-sided) | Step 1c → 1d change |")
    lines.append("|---|:-:|---:|---:|---:|---|")
    step1c_aggs = {
        "cross_binary":     ("5", "−5.32",  "<0.001"),
        "cross_ternary":    ("5", "+0.58",  "0.564"),
        "within_binary":    ("5", "−4.42",  "<0.0001"),
        "within_ternary":   ("5", "−2.16",  "0.031"),
        "transfer_binary":  ("2", "−2.79",  "0.005"),
        "transfer_ternary": ("2", "−1.60",  "0.110"),
    }
    for key, agg in aggregates.items():
        p_str = f"{agg['p']:.4f}" if agg["p"] >= 0.0001 else "<0.0001"
        old = step1c_aggs.get(key)
        if old is None or old[0] == str(agg["n_cells"]):
            change = "unchanged from Step 1c"
        else:
            change = f"**n={old[0]}→{agg['n_cells']}**, Z {old[1]}→{agg['Z']:+.3f}, p {old[2]}→{p_str}"
        lines.append(
            f"| {key.replace('_', '-')} | {agg['n_cells']} | "
            f"{agg['mean_delta_pp']:+.3f} | {agg['Z']:+.3f} | {p_str} | {change} |"
        )
    lines.append("")
    lines.append("**Key reversal (transfer-ternary)**: V4/V5-only 2-cell aggregate was directionally negative (Z=−1.60, p=0.110); adding V1 (+0.65), V2 (+0.18), V3 (+1.09) — all directionally positive — flips the 5V aggregate to weakly positive (Z=+0.18, p=0.86). The v3.1 narrative \"transfer-ternary 整体负向\" no longer holds; ternary task shows paradigm-dependent direction inconsistency.")
    lines.append("")
    lines.append("**Strengthened (transfer-binary)**: V4/V5-only 2-cell Z=−2.79 (p=0.005); adding V1 (Δ=−1.10), V2 (Δ=−0.74), V3 (Δ=−1.01) — all directionally negative — strengthens the aggregate to Z=−3.39 (p=0.0007). 15/15 binary cells across cross/within/transfer paradigms now all directionally negative.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Section 5: Implications for §3.6 narrative
    lines.append("## 5. Implications for §3.6 Narrative")
    lines.append("")
    lines.append("- **DAPT evaluation matrix closes**: 30/30 cells evaluated, 0 missing. Caveat #6 (\"is DAPT failure cross-subject-specific?\") closes definitively on binary task (15/15 cells directionally negative across 3 paradigms; all 3 paradigm-level Stouffer p<0.001).")
    lines.append("- **Ternary direction is paradigm-dependent**: cross-ternary mean Δ=+0.18 (Z=+0.58, n.s.); within-ternary mean Δ=−0.92 (Z=−2.16, p=0.031); transfer-ternary mean Δ=+0.026 (Z=+0.18, n.s.). \"Ternary uniform negative\" claim not supported.")
    lines.append("- **V3 transfer-ternary +1.09 pp** is the global-max positive Δ across the entire 30-cell matrix (all 15 binary Δ are negative).")
    lines.append("- **Transfer rescue gradient (binary)**: V1/V2/V3 attenuate strongly (Δ magnitude reduced 31–41%, all transfer p≥0.171 vs cross p≤0.051); V5 attenuates partially (Δ −2.77→−1.22, magnitude reduced 56%, sig→n.s.); **V4 is the unique exception** (Δ −1.61→−1.67, both cross and transfer BH-edge). V4's specific surgical fix (3-set domain alignment + strict filter) imprints the most rigid wrong prior; V5's channel-geometry mismatch is partially correctable by per-subject fine-tune.")
    lines.append("- **BH at 30-family**: 0/30 survive at q<0.05. Family-size correction alone — every single raw p unchanged from Step 1c. Read Stouffer aggregates for collective evidence.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Section 6: reproducibility
    lines.append("## 6. Reproducibility")
    lines.append("")
    lines.append("```powershell")
    lines.append("uv run python scripts/internal/recompute_v1v2v3_transfer.py")
    lines.append("```")
    lines.append("")
    lines.append("Deterministic recompute from per-subject `test_acc` values in 6 + 2 cache JSONs. Output identical across runs (no RNG, no resampling).")
    lines.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
