# Stage 4 Step 1d — V1/V2/V3 transfer Statistical Recompute

**Date**: 2026-05-12
**Supersedes (extends)**: `stage4_step1c_v4v5_within_transfer.md` (24 cells → 30 cells)
**Scope**: 6 new V1/V2/V3 transfer cells; full DAPT family BH-FDR re-applied at 30 cells; 6 paradigm-level Stouffer aggregates fully populated to 5V each.
**Method**: Identical to Step 1c — `scipy.stats.ttest_rel` (two-sided paired), Cohen's dz = mean_diff/sd_diff, 95% CI via t-distribution (df=n−1), BH-FDR @ α=0.05 within new 30-cell DAPT family, Stouffer combination via signed inverse-normal.

> All `mean_diff` and 95% CI shown in **percentage points (pp)**. Effect direction = treatment − baseline (positive → DAPT > Baseline).

---

## 1. Data Sources (6 new V1/V2/V3 transfer cells)

All 6 caches in commit `90b9fc4` (handoff `docs/handoffs/2026-05-10_dapt_v4_v5.md` 追加 (2026-05-11) section).
Baselines shared with V4/V5 transfer (Step 1c).

| Cell | Treatment cache | Baseline cache |
|---|---|---|
| V1 transfer binary | `results/dapt_v1/20260510_2357_transfer_cache_imagery_binary.json` | `results/20260329_0507_transfer_cache_imagery_binary.json` |
| V1 transfer ternary | `results/dapt_v1/20260511_0012_transfer_cache_imagery_ternary.json` | `results/20260329_0521_transfer_cache_imagery_ternary.json` |
| V2 transfer binary | `results/dapt_v2/20260511_0031_transfer_cache_imagery_binary.json` | `results/20260329_0507_transfer_cache_imagery_binary.json` |
| V2 transfer ternary | `results/dapt_v2/20260511_0042_transfer_cache_imagery_ternary.json` | `results/20260329_0521_transfer_cache_imagery_ternary.json` |
| V3 transfer binary | `results/dapt_v3/20260511_0058_transfer_cache_imagery_binary.json` | `results/20260329_0507_transfer_cache_imagery_binary.json` |
| V3 transfer ternary | `results/dapt_v3/20260511_0109_transfer_cache_imagery_ternary.json` | `results/20260329_0521_transfer_cache_imagery_ternary.json` |

---

## 2. Recomputed 6 new cells (paired-t, dz, 95% CI, BH-q within new 30-cell family)

| V | Paradigm | Task | n | mean_treat (%) | mean_base (%) | mean_diff (pp) | SD_diff | t | p (raw) | dz | 95% CI (pp) | q (BH, 30-family) | BH sig? |
|---|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|:-:|---:|:-:|
| V1 | transfer | binary | 21 | 89.02 | 90.12 | -1.10 | 3.55 | -1.421 | 0.171 | -0.310 | [-2.72, +0.52] | 0.301 | n |
| V1 | transfer | ternary | 21 | 75.69 | 75.04 | +0.65 | 3.81 | +0.787 | 0.441 | +0.172 | [-1.08, +2.39] | 0.575 | n |
| V2 | transfer | binary | 21 | 89.38 | 90.12 | -0.74 | 2.91 | -1.172 | 0.255 | -0.256 | [-2.07, +0.58] | 0.386 | n |
| V2 | transfer | ternary | 21 | 75.22 | 75.04 | +0.18 | 2.76 | +0.297 | 0.770 | +0.065 | [-1.08, +1.43] | 0.796 | n |
| V3 | transfer | binary | 21 | 89.11 | 90.12 | -1.01 | 3.98 | -1.165 | 0.258 | -0.254 | [-2.82, +0.80] | 0.386 | n |
| V3 | transfer | ternary | 21 | 76.13 | 75.04 | +1.09 | 3.00 | +1.665 | 0.111 | +0.363 | [-0.28, +2.46] | 0.230 | n |

---

## 3. BH-FDR Re-application (30-cell DAPT family)

**Survivors at q < 0.05 within 30-cell DAPT family: 0**.

Most-significant 5 cells (q ascending) and Step 1c (24-family) comparison:

| Cell | mean_diff (pp) | p (raw) | q (24-family, Step 1c) | q (30-family, Step 1d) | shift |
|---|---:|---:|---:|---:|---:|
| `V2_within_binary` | -2.86 | 0.0020 | 0.048 | 0.060 | +0.012 |
| `V4_cross_binary` | -1.61 | 0.0080 | 0.072 | 0.090 | +0.018 |
| `V1_cross_binary` | -1.85 | 0.0090 | 0.072 | 0.090 | +0.018 |
| `V5_cross_binary` | -2.77 | 0.0140 | — | 0.105 | (was n.s.) |
| `V5_within_binary` | -2.92 | 0.0200 | — | 0.111 | (was n.s.) |

**`V2_within_binary`** — the lone Step 1c survivor (q=0.048 in 24-family) — exits BH significance at q=0.060 in the 30-family. Family-size correction is the sole cause; the cell's raw p=0.002 is unchanged. Read paradigm-level Stouffer aggregates (Section 4) for collective evidence rather than single-cell BH.

---

## 4. Six Paradigm-Level Stouffer Aggregates (5V each)

With V1/V2/V3 transfer added, the `transfer_binary` and `transfer_ternary` aggregates are upgraded from n_cells=2 (V4/V5 only, Step 1c) to n_cells=5 (V1–V5). The other four aggregates are unchanged from Step 1c.

| Aggregate | n_cells | mean Δ (pp) | Stouffer Z (signed) | Combined p (two-sided) | Step 1c → 1d change |
|---|:-:|---:|---:|---:|---|
| within-binary | 5 | -1.894 | -4.419 | <0.0001 | unchanged from Step 1c |
| within-ternary | 5 | -0.918 | -2.159 | 0.0309 | unchanged from Step 1c |
| cross-binary | 5 | -1.788 | -5.328 | <0.0001 | unchanged from Step 1c |
| cross-ternary | 5 | +0.180 | +0.577 | 0.5637 | unchanged from Step 1c |
| transfer-binary | 5 | -1.149 | -3.391 | 0.0007 | **n=2→5**, Z −2.79→-3.391, p 0.005→0.0007 |
| transfer-ternary | 5 | +0.027 | +0.176 | 0.8600 | **n=2→5**, Z −1.60→+0.176, p 0.110→0.8600 |

**Key reversal (transfer-ternary)**: V4/V5-only 2-cell aggregate was directionally negative (Z=−1.60, p=0.110); adding V1 (+0.65), V2 (+0.18), V3 (+1.09) — all directionally positive — flips the 5V aggregate to weakly positive (Z=+0.18, p=0.86). The v3.1 narrative "transfer-ternary 整体负向" no longer holds; ternary task shows paradigm-dependent direction inconsistency.

**Strengthened (transfer-binary)**: V4/V5-only 2-cell Z=−2.79 (p=0.005); adding V1 (Δ=−1.10), V2 (Δ=−0.74), V3 (Δ=−1.01) — all directionally negative — strengthens the aggregate to Z=−3.39 (p=0.0007). 15/15 binary cells across cross/within/transfer paradigms now all directionally negative.

---

## 5. Implications for §3.6 Narrative

- **DAPT evaluation matrix closes**: 30/30 cells evaluated, 0 missing. Caveat #6 ("is DAPT failure cross-subject-specific?") closes definitively on binary task (15/15 cells directionally negative across 3 paradigms; all 3 paradigm-level Stouffer p<0.001).
- **Ternary direction is paradigm-dependent**: cross-ternary mean Δ=+0.18 (Z=+0.58, n.s.); within-ternary mean Δ=−0.92 (Z=−2.16, p=0.031); transfer-ternary mean Δ=+0.026 (Z=+0.18, n.s.). "Ternary uniform negative" claim not supported.
- **V3 transfer-ternary +1.09 pp** is the global-max positive Δ across the entire 30-cell matrix (all 15 binary Δ are negative).
- **Transfer rescue gradient (binary)**: V1/V2/V3 attenuate strongly (Δ magnitude reduced 31–41%, all transfer p≥0.171 vs cross p≤0.051); V5 attenuates partially (Δ −2.77→−1.22, magnitude reduced 56%, sig→n.s.); **V4 is the unique exception** (Δ −1.61→−1.67, both cross and transfer BH-edge). V4's specific surgical fix (3-set domain alignment + strict filter) imprints the most rigid wrong prior; V5's channel-geometry mismatch is partially correctable by per-subject fine-tune.
- **BH at 30-family**: 0/30 survive at q<0.05. Family-size correction alone — every single raw p unchanged from Step 1c. Read Stouffer aggregates for collective evidence.

---

## 6. Reproducibility

```powershell
uv run python scripts/internal/recompute_v1v2v3_transfer.py
```

Deterministic recompute from per-subject `test_acc` values in 6 + 2 cache JSONs. Output identical across runs (no RNG, no resampling).
