# Stage 4 Step 1c — V4/V5 within+transfer Statistical Recompute

**Date**: 2026-05-10
**Supersedes (extends)**: `stage4_step1b_stat_recompute_v4v5.md` (16 cells → 24 cells)
**Scope**: 8 new V4/V5 cells × {within, transfer} × {binary, ternary}; full DAPT family BH-FDR re-applied at 24 cells; 4 new paradigm-level Stouffer aggregates added.
**Method**: Identical to Step 1b — scipy.stats.ttest_rel (two-sided paired), Cohen's dz = mean_diff/sd_diff, 95% CI via t-distribution (df=n−1), BH-FDR @ α=0.05 within new 24-cell DAPT family, Stouffer combination via signed inverse-normal.

> All `mean_diff` and 95% CI shown in **percentage points (pp)**. Effect direction = treatment − baseline (positive → DAPT > Baseline).

---

## 1. Data Sources (8 new V4/V5 within+transfer cells)

All 8 caches verified to exist + ExperimentDB-registered as of 2026-05-10 22:29 (handoff `docs/handoffs/2026-05-10_dapt_v4_v5.md`).

| Cell | Treatment cache | Baseline cache |
|---|---|---|
| V4 within binary | `results/dapt_v4/20260510_1950_within_subject_cache_imagery_binary.json` | `results/20260323_2237_comparison_cache_imagery_binary.json` |
| V4 within ternary | `results/dapt_v4/20260510_2010_within_subject_cache_imagery_ternary.json` | `results/20260323_2320_comparison_cache_imagery_ternary.json` |
| V4 transfer binary | `results/dapt_v4/20260510_2038_transfer_cache_imagery_binary.json` | `results/20260329_0507_transfer_cache_imagery_binary.json` |
| V4 transfer ternary | `results/dapt_v4/20260510_2053_transfer_cache_imagery_ternary.json` | `results/20260329_0521_transfer_cache_imagery_ternary.json` |
| V5 within binary | `results/dapt_v5/20260510_2113_within_subject_cache_imagery_binary.json` | `results/20260323_2237_comparison_cache_imagery_binary.json` |
| V5 within ternary | `results/dapt_v5/20260510_2131_within_subject_cache_imagery_ternary.json` | `results/20260323_2320_comparison_cache_imagery_ternary.json` |
| V5 transfer binary | `results/dapt_v5/20260510_2157_transfer_cache_imagery_binary.json` | `results/20260329_0507_transfer_cache_imagery_binary.json` |
| V5 transfer ternary | `results/dapt_v5/20260510_2210_transfer_cache_imagery_ternary.json` | `results/20260329_0521_transfer_cache_imagery_ternary.json` |

---

## 2. Recomputed 8 new cells (paired-t, dz, 95% CI, BH-q within new 24-cell family)

| V | Paradigm | Task | n | mean_treat (%) | mean_base (%) | mean_diff (pp) | SD_diff | t | p (raw) | dz | 95% CI (pp) | q (BH, 24-family) | BH sig? |
|---|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|:-:|---:|:-:|
| V4 | within | binary | 21 | 84.05 | 85.15 | -1.10 | 3.75 | -1.344 | 0.194 | -0.293 | [-2.81, +0.61] | 0.291 | n |
| V4 | within | ternary | 21 | 68.89 | 69.44 | -0.56 | 4.22 | -0.603 | 0.553 | -0.132 | [-2.48, +1.36] | 0.664 | n |
| V4 | transfer | binary | 21 | 88.45 | 90.12 | -1.67 | 3.18 | -2.404 | 0.026 | -0.525 | [-3.11, -0.22] | 0.089 | n |
| V4 | transfer | ternary | 21 | 74.72 | 75.04 | -0.32 | 3.84 | -0.379 | 0.709 | -0.083 | [-2.07, +1.43] | 0.761 | n |
| V5 | within | binary | 21 | 82.23 | 85.15 | -2.92 | 5.27 | -2.537 | 0.020 | -0.554 | [-5.31, -0.52] | 0.089 | n |
| V5 | within | ternary | 21 | 67.42 | 69.44 | -2.02 | 5.00 | -1.856 | 0.078 | -0.405 | [-4.30, +0.25] | 0.186 | n |
| V5 | transfer | binary | 21 | 88.90 | 90.12 | -1.22 | 3.10 | -1.806 | 0.086 | -0.394 | [-2.63, +0.19] | 0.186 | n |
| V5 | transfer | ternary | 21 | 73.57 | 75.04 | -1.47 | 3.37 | -1.999 | 0.059 | -0.436 | [-3.00, +0.06] | 0.158 | n |

---

## 3. BH-FDR Re-application (24-cell DAPT family)

With 8 new cells joining the 16 existing Step 1b cells, the BH-FDR threshold shifts. Survivors at q < 0.05 within the new 24-cell family:

| Cell | mean_diff (pp) | p (raw) | q (24-family) | Δ vs Step 1b q |
|---|---:|---:|---:|---:|
| `V2_within_binary` | -2.86 | 0.0020 | 0.048 | from 0.033 (shift +0.015) |

**New-only BH survivors among 8 new cells**: 0

---

## 4. Four New Paradigm-Level Stouffer Aggregates

Per user direction (Stage 4' visualization plan): existing `cross_binary`, `cross_ternary`, `full_dapt` aggregates **preserved unchanged** to maintain continuity with the v3.1 published statistic. The four new aggregates below are *additive* paradigm-level summaries.

| Aggregate | n cells | Stouffer Z (signed) | Combined p (two-sided) | Direction |
|---|:-:|---:|---:|---|
| within-binary | 5 | -4.422 | <0.0001 | negative (DAPT < Baseline) — all 5 V negative; aggregate corroborates cross-binary finding in within paradigm |
| within-ternary | 5 | -2.158 | 0.0309 | negative (DAPT < Baseline) — mixed signs; no aggregate finding |
| transfer-binary | 2 | -2.788 | 0.0053 | negative (DAPT < Baseline) — V4, V5 both negative; small n=2 |
| transfer-ternary | 2 | -1.597 | 0.1103 | negative (DAPT < Baseline) — V4, V5 both negative; small n=2 |

Interpretation: 4/4 new paradigm-level aggregates are directionally negative; within-binary (n=5) is the most robust signal among the new aggregates, supporting the §3.6.1 task-asymmetric narrative reproduction across paradigms (binary worse than ternary in within / cross / transfer).

---

## 5. Implications for §3.6 Narrative

- **Caveat #6 closure**: V4/V5 within+transfer evaluation completes the 12-cell V4/V5 matrix. 0/12 V4/V5 cells positive significant; 0/12 V4/V5 cells positive even directionally (full negative or near-zero) — confirms DAPT failure is not cross-subject-specific.
- **Task asymmetry reproduction**: V4 binary average Δ −1.46 pp / ternary −0.22 pp (gap 1.24 pp); V5 binary −2.30 pp / ternary −1.55 pp (gap 0.75 pp, asymmetry shrinks). Confirms binary suffers more under DAPT than ternary across paradigms; V5 single-source 60ch geometry blurs the gap.
- **V5 systematic worsening**: V5 worse than V4 in 5/6 cells by 1.15–1.82 pp (only transfer-binary reverses, both n.s.) — channel diversity is a protective factor in DAPT, not a confound.
- **Mechanism narrowing**: Stieger-dominance ruled out (V3 + V4); channel-heterogeneity-as-confound reverse-falsified (V5); MI-granularity-mismatch surviving hypothesis stands.

---

## 6. Reproducibility

```powershell
uv run python scripts/internal/recompute_v4v5_within_transfer.py
```

Generates this audit doc deterministically from per-subject `test_acc` values in the 8 + 4 cache JSONs. Output identical across runs.
