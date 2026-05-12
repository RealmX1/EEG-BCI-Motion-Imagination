# Stage 4 Step 1 — Statistical Recompute Report

**Date**: 2026-05-10  
**Scope**: All paired tests across main tables (T6, T7, T11, T12a/b, T13a/b, T15, T15b, T16, T18) + per-subject paired-t for §3.6 DAPT V1/V2/V3 vs Baseline.  
**Method**: scipy.stats.ttest_rel (two-sided) on per-subject test_acc; Cohen's dz = mean_diff / SD_diff; 95% CI of mean_diff = mean_diff ± t_{0.975, df=n−1} × SE; BH FDR @ α=0.05 across all tests globally.

> All `mean_diff` and 95% CI shown in **percentage points (pp)**. Effect direction = treatment − baseline (positive → treatment > baseline).

---

## 1. Paired Test Inventory

| ID | Group | Description | n | mean_treat (pp) | mean_base (pp) | Status |
|----|-------|-------------|:-:|---------------:|---------------:|--------|
| `T16_V1_within_bin` | T16 | FT-V1 vs Baseline (within, binary) | 21 | 83.84 | 85.09 | ok |
| `T16_V1_cross_bin` | T16 | FT-V1 vs Baseline (cross, binary) | 21 | 88.84 | 90.54 | ok |
| `T16_V1_within_ter` | T16 | FT-V1 vs Baseline (within, ternary) | 21 | 69.25 | 69.54 | ok |
| `T16_V1_cross_ter` | T16 | FT-V1 vs Baseline (cross, ternary) | 21 | 75.67 | 75.42 | ok |
| `T16_V2_within_bin` | T16 | FT-V2 vs Baseline (within, binary) | 21 | 82.23 | 85.09 | ok |
| `T16_V2_cross_bin` | T16 | FT-V2 vs Baseline (cross, binary) | 21 | 89.43 | 90.54 | ok |
| `T16_V2_within_ter` | T16 | FT-V2 vs Baseline (within, ternary) | 21 | 68.08 | 69.54 | ok |
| `T16_V2_cross_ter` | T16 | FT-V2 vs Baseline (cross, ternary) | 21 | 75.32 | 75.42 | ok |
| `T16_V3_within_bin` | T16 | FT-V3 vs Baseline (within, binary) | 21 | 83.75 | 85.09 | ok |
| `T16_V3_cross_bin` | T16 | FT-V3 vs Baseline (cross, binary) | 21 | 89.23 | 90.54 | ok |
| `T16_V3_within_ter` | T16 | FT-V3 vs Baseline (within, ternary) | 21 | 69.31 | 69.54 | ok |
| `T16_V3_cross_ter` | T16 | FT-V3 vs Baseline (cross, ternary) | 21 | 75.50 | 75.42 | ok |
| `T6_within_bin` | T6 | CBraMod vs EEGNet (within, binary) | 21 | 85.15 | 78.10 | ok |
| `T6_within_ter` | T6 | CBraMod vs EEGNet (within, ternary) | 21 | 69.44 | 66.81 | ok |
| `T7_cross_bin` | T7 | CBraMod vs EEGNet (cross, binary) | 21 | 90.68 | 76.67 | ok |
| `T7_cross_ter` | T7 | CBraMod vs EEGNet (cross, ternary) | 21 | 74.88 | 61.23 | ok |
| `T11_cbramod_xsift_vs_cross_bin` | T11 | CBraMod XSI-FT vs cross (binary) | 21 | 90.12 | 90.68 | ok |
| `T11_cbramod_xsift_vs_cross_ter` | T11 | CBraMod XSI-FT vs cross (ternary) | 21 | 75.08 | 74.88 | ok |
| `T11_eegnet_xsift_vs_cross_bin` | T11 | EEGNet XSI-FT vs cross (binary) | 21 | 80.77 | 76.67 | ok |
| `T11_eegnet_xsift_vs_cross_ter` | T11 | EEGNet XSI-FT vs cross (ternary) | 21 | 66.23 | 61.23 | ok |
| `T7_cbramod_cross_vs_within_bin` | T7 | CBraMod cross vs within (binary, pooling gain) | 21 | 90.68 | 85.15 | ok |
| `T7_eegnet_cross_vs_within_bin` | T7 | EEGNet cross vs within (binary, pooling gain) | 21 | 76.67 | 78.10 | ok |
| `T12a_cbramod_within_bin_s05` | T12a | CBraMod within +Sess05 vs Baseline (binary) | 16 | 93.36 | 87.23 | ok |
| `T12b_eegnet_within_bin_s05` | T12b | EEGNet within +Sess05 vs Baseline (binary) | 16 | 87.85 | 80.51 | ok |
| `T13a_cbramod_within_ter_s05` | T13a | CBraMod within +Sess05 vs Baseline (ternary) | 16 | 83.06 | 74.51 | ok |
| `T13b_eegnet_within_ter_s05` | T13b | EEGNet within +Sess05 vs Baseline (ternary) | 16 | 76.08 | 71.48 | ok |
| `T15_cbramod_within_bin_s05` | T15 | Within +Sess05 vs Baseline (already in T12a; repeated for context) | 16 | 93.36 | 87.23 | ok |
| `T15_cbramod_cross_bin_s05` | T15 | Cross +Sess05 vs Baseline (binary) | 16 | 93.24 | 92.38 | ok |
| `T15_cbramod_xsift_bin_s05` | T15 | XSI-FT +Sess05 vs Baseline (binary) | 16 | 92.93 | 87.23 | ok |
| `T15b_eegnet_cross_bin_s05` | T15b | EEGNet Cross +Sess05 vs Baseline (binary) | 16 | 81.33 | 81.45 | ok |
| `T15b_cbramod_cross_ter_s05` | T15b | CBraMod Cross +Sess05 vs Baseline (ternary) | 16 | 83.78 | 80.05 | ok |
| `T18_within_bin` | T18 | Original-weights CBraMod vs random-init (within, binary) | 21 | 85.15 | 62.05 | ok |
| `T18_within_ter` | T18 | Original-weights CBraMod vs random-init (within, ternary) | 21 | 69.44 | 38.65 | ok |
| `T18_cross_bin` | T18 | Original-weights CBraMod vs random-init (cross, binary) | 21 | 90.68 | 86.34 | ok |
| `T18_cross_ter` | T18 | Original-weights CBraMod vs random-init (cross, ternary) | 21 | 74.88 | 73.06 | ok |
| `T18_xsift_bin` | T18 | Original-weights CBraMod vs random-init (XSI-FT, binary) | 21 | 90.12 | 86.22 | ok |
| `T18_xsift_ter` | T18 | Original-weights CBraMod vs random-init (XSI-FT, ternary) | 21 | 75.08 | 73.43 | ok |

---

## 2. Recomputed Statistics

### 2.1 T16 — §3.6 DAPT V1/V2/V3 vs TUEG Baseline (CBraMod) — 12 paired-t cells

| Test | n | mean_diff (pp) | SD_diff (pp) | t | p (raw) | dz | 95% CI (pp) | q (BH) | BH sig? |
|------|:-:|---------------:|-------------:|--:|--------:|---:|:-----------:|-------:|:-------:|
| `T16_V1_within_bin` | 21 | -1.25 | 3.48 | -1.646 | 0.115 | -0.359 | [-2.83, +0.33] | 0.171 | n |
| `T16_V1_cross_bin` | 21 | -1.70 | 3.11 | -2.502 | 0.021 | -0.546 | [-3.11, -0.28] | 0.046 | **Y** |
| `T16_V1_within_ter` | 21 | -0.30 | 3.02 | -0.452 | 0.656 | -0.099 | [-1.67, +1.08] | 0.765 | n |
| `T16_V1_cross_ter` | 21 | +0.26 | 3.49 | +0.339 | 0.738 | +0.074 | [-1.33, +1.85] | 0.804 | n |
| `T16_V2_within_bin` | 21 | -2.86 | 3.71 | -3.533 | 0.002 | -0.771 | [-4.54, -1.17] | 0.008 | **Y** |
| `T16_V2_cross_bin` | 21 | -1.10 | 2.40 | -2.099 | 0.049 | -0.458 | [-2.20, -0.01] | 0.090 | n |
| `T16_V2_within_ter` | 21 | -1.47 | 3.81 | -1.767 | 0.093 | -0.385 | [-3.20, +0.27] | 0.156 | n |
| `T16_V2_cross_ter` | 21 | -0.10 | 2.69 | -0.169 | 0.868 | -0.037 | [-1.33, +1.13] | 0.917 | n |
| `T16_V3_within_bin` | 21 | -1.34 | 3.69 | -1.662 | 0.112 | -0.363 | [-3.02, +0.34] | 0.171 | n |
| `T16_V3_cross_bin` | 21 | -1.31 | 3.48 | -1.722 | 0.101 | -0.376 | [-2.90, +0.28] | 0.162 | n |
| `T16_V3_within_ter` | 21 | -0.24 | 3.11 | -0.351 | 0.729 | -0.077 | [-1.65, +1.18] | 0.804 | n |
| `T16_V3_cross_ter` | 21 | +0.08 | 2.98 | +0.122 | 0.904 | +0.027 | [-1.28, +1.44] | 0.929 | n |

### 2.2 T6 — §3.1 Within-subject 128ch CBraMod vs EEGNet

| Test | n | mean_diff (pp) | SD_diff (pp) | t | p (raw) | dz | 95% CI (pp) | q (BH) | BH sig? |
|------|:-:|---------------:|-------------:|--:|--------:|---:|:-----------:|-------:|:-------:|
| `T6_within_bin` | 21 | +7.05 | 7.97 | +4.056 | <0.001 | +0.885 | [+3.43, +10.68] | 0.003 | **Y** |
| `T6_within_ter` | 21 | +2.64 | 11.24 | +1.076 | 0.295 | +0.235 | [-2.48, +7.76] | 0.376 | n |

### 2.3 T7 — §3.2 Cross-subject 128ch CBraMod vs EEGNet (+ pooling-gain paired-t)

| Test | n | mean_diff (pp) | SD_diff (pp) | t | p (raw) | dz | 95% CI (pp) | q (BH) | BH sig? |
|------|:-:|---------------:|-------------:|--:|--------:|---:|:-----------:|-------:|:-------:|
| `T7_cross_bin` | 21 | +14.02 | 8.68 | +7.397 | <0.001 | +1.614 | [+10.06, +17.97] | <0.001 | **Y** |
| `T7_cross_ter` | 21 | +13.65 | 9.38 | +6.671 | <0.001 | +1.456 | [+9.38, +17.92] | <0.001 | **Y** |
| `T7_cbramod_cross_vs_within_bin` | 21 | +5.54 | 5.21 | +4.868 | <0.001 | +1.062 | [+3.16, +7.91] | <0.001 | **Y** |
| `T7_eegnet_cross_vs_within_bin` | 21 | -1.43 | 8.62 | -0.760 | 0.456 | -0.166 | [-5.35, +2.49] | 0.563 | n |

### 2.4 T11 — §3.3 XSI-FT vs cross-subject baselines (CBraMod & EEGNet)

| Test | n | mean_diff (pp) | SD_diff (pp) | t | p (raw) | dz | 95% CI (pp) | q (BH) | BH sig? |
|------|:-:|---------------:|-------------:|--:|--------:|---:|:-----------:|-------:|:-------:|
| `T11_cbramod_xsift_vs_cross_bin` | 21 | -0.57 | 1.91 | -1.360 | 0.189 | -0.297 | [-1.43, +0.30] | 0.259 | n |
| `T11_cbramod_xsift_vs_cross_ter` | 21 | +0.20 | 0.79 | +1.156 | 0.261 | +0.252 | [-0.16, +0.56] | 0.345 | n |
| `T11_eegnet_xsift_vs_cross_bin` | 21 | +4.11 | 6.87 | +2.742 | 0.013 | +0.598 | [+0.98, +7.23] | 0.031 | **Y** |
| `T11_eegnet_xsift_vs_cross_ter` | 21 | +5.00 | 5.10 | +4.490 | <0.001 | +0.980 | [+2.68, +7.32] | 0.001 | **Y** |

### 2.5 T12a — §3.4.1 CBraMod within +Sess05 vs Baseline (binary, N=16)

| Test | n | mean_diff (pp) | SD_diff (pp) | t | p (raw) | dz | 95% CI (pp) | q (BH) | BH sig? |
|------|:-:|---------------:|-------------:|--:|--------:|---:|:-----------:|-------:|:-------:|
| `T12a_cbramod_within_bin_s05` | 16 | +6.13 | 7.83 | +3.134 | 0.007 | +0.784 | [+1.96, +10.30] | 0.021 | **Y** |

### 2.6 T12b — §3.4.1 EEGNet within +Sess05 vs Baseline (binary, N=16)

| Test | n | mean_diff (pp) | SD_diff (pp) | t | p (raw) | dz | 95% CI (pp) | q (BH) | BH sig? |
|------|:-:|---------------:|-------------:|--:|--------:|---:|:-----------:|-------:|:-------:|
| `T12b_eegnet_within_bin_s05` | 16 | +7.34 | 9.83 | +2.989 | 0.009 | +0.747 | [+2.11, +12.58] | 0.026 | **Y** |

### 2.7 T13a — §3.4.2 CBraMod within +Sess05 vs Baseline (ternary, N=16)

| Test | n | mean_diff (pp) | SD_diff (pp) | t | p (raw) | dz | 95% CI (pp) | q (BH) | BH sig? |
|------|:-:|---------------:|-------------:|--:|--------:|---:|:-----------:|-------:|:-------:|
| `T13a_cbramod_within_ter_s05` | 16 | +8.55 | 11.95 | +2.861 | 0.012 | +0.715 | [+2.18, +14.92] | 0.031 | **Y** |

### 2.8 T13b — §3.4.2 EEGNet within +Sess05 vs Baseline (ternary, N=16)

| Test | n | mean_diff (pp) | SD_diff (pp) | t | p (raw) | dz | 95% CI (pp) | q (BH) | BH sig? |
|------|:-:|---------------:|-------------:|--:|--------:|---:|:-----------:|-------:|:-------:|
| `T13b_eegnet_within_ter_s05` | 16 | +4.60 | 12.64 | +1.455 | 0.166 | +0.364 | [-2.14, +11.33] | 0.236 | n |

### 2.9 T15 — §3.4.4 CBraMod three-paradigm extra sessions binary +Sess05 vs Baseline (N=16)

| Test | n | mean_diff (pp) | SD_diff (pp) | t | p (raw) | dz | 95% CI (pp) | q (BH) | BH sig? |
|------|:-:|---------------:|-------------:|--:|--------:|---:|:-----------:|-------:|:-------:|
| `T15_cbramod_within_bin_s05` | 16 | +6.13 | 7.83 | +3.134 | 0.007 | +0.784 | [+1.96, +10.30] | 0.021 | **Y** |
| `T15_cbramod_cross_bin_s05` | 16 | +0.86 | 7.70 | +0.447 | 0.662 | +0.112 | [-3.24, +4.96] | 0.765 | n |
| `T15_cbramod_xsift_bin_s05` | 16 | +5.70 | 8.32 | +2.743 | 0.015 | +0.686 | [+1.27, +10.13] | 0.035 | **Y** |

### 2.10 T15b — §3.4.5 Cross-subject extra sessions edge cases (N=16)

| Test | n | mean_diff (pp) | SD_diff (pp) | t | p (raw) | dz | 95% CI (pp) | q (BH) | BH sig? |
|------|:-:|---------------:|-------------:|--:|--------:|---:|:-----------:|-------:|:-------:|
| `T15b_eegnet_cross_bin_s05` | 16 | -0.12 | 7.29 | -0.064 | 0.950 | -0.016 | [-4.00, +3.77] | 0.950 | n |
| `T15b_cbramod_cross_ter_s05` | 16 | +3.73 | 8.24 | +1.811 | 0.090 | +0.453 | [-0.66, +8.12] | 0.156 | n |

### 2.11 T18 — §3.7.2 Original-weights CBraMod vs random-init (3 paradigms × 2 tasks)

| Test | n | mean_diff (pp) | SD_diff (pp) | t | p (raw) | dz | 95% CI (pp) | q (BH) | BH sig? |
|------|:-:|---------------:|-------------:|--:|--------:|---:|:-----------:|-------:|:-------:|
| `T18_within_bin` | 21 | +23.10 | 17.09 | +6.193 | <0.001 | +1.351 | [+15.32, +30.87] | <0.001 | **Y** |
| `T18_within_ter` | 21 | +30.79 | 16.68 | +8.460 | <0.001 | +1.846 | [+23.20, +38.39] | <0.001 | **Y** |
| `T18_cross_bin` | 21 | +4.35 | 3.15 | +6.316 | <0.001 | +1.378 | [+2.91, +5.78] | <0.001 | **Y** |
| `T18_cross_ter` | 21 | +1.83 | 3.38 | +2.476 | 0.022 | +0.540 | [+0.29, +3.36] | 0.046 | **Y** |
| `T18_xsift_bin` | 21 | +3.90 | 3.65 | +4.889 | <0.001 | +1.067 | [+2.24, +5.56] | <0.001 | **Y** |
| `T18_xsift_ter` | 21 | +1.65 | 3.24 | +2.330 | 0.030 | +0.508 | [+0.17, +3.12] | 0.059 | n |

---

## 3. Aggregate BH Correction (α = 0.05, global across all groups)

- Total tests (status=ok): **37**
- Nominal p < 0.05: **20**
- BH-significant (q < 0.05): **18**
- Lost to BH correction: **2**

**Tests becoming non-significant after BH:**
- `T16_V2_cross_bin` (FT-V2 vs Baseline (cross, binary)): p=0.0487, q=0.0900
- `T18_xsift_ter` (Original-weights CBraMod vs random-init (XSI-FT, ternary)): p=0.0304, q=0.0592

## 4. SURPRISES Summary

### 4.1 §3.6 DAPT V1/V2/V3 vs Baseline — Critical

- Total cells computed: **12** (out of designed 12)
- BH-significant **negative** (V < Baseline): **2**
- BH-significant **positive** (V > Baseline, sign reversal): **0**
- **Not significant after BH (q ≥ 0.05)**: **10**  ← these undermine the 'consistent negative transfer' claim

**T16 cells failing BH significance:**
  - `T16_V1_within_bin` — mean_diff=-1.25 pp, p=0.1154, q=0.1707, dz=-0.359
  - `T16_V1_within_ter` — mean_diff=-0.30 pp, p=0.6564, q=0.7650, dz=-0.099
  - `T16_V1_cross_ter` — mean_diff=+0.26 pp, p=0.7384, q=0.8035, dz=+0.074
  - `T16_V2_cross_bin` — mean_diff=-1.10 pp, p=0.0487, q=0.0900, dz=-0.458
  - `T16_V2_within_ter` — mean_diff=-1.47 pp, p=0.0926, q=0.1557, dz=-0.385
  - `T16_V2_cross_ter` — mean_diff=-0.10 pp, p=0.8677, q=0.9172, dz=-0.037
  - `T16_V3_within_bin` — mean_diff=-1.34 pp, p=0.1122, q=0.1707, dz=-0.363
  - `T16_V3_cross_bin` — mean_diff=-1.31 pp, p=0.1005, q=0.1617, dz=-0.376
  - `T16_V3_within_ter` — mean_diff=-0.24 pp, p=0.7290, q=0.8035, dz=-0.077
  - `T16_V3_cross_ter` — mean_diff=+0.08 pp, p=0.9041, q=0.9292, dz=+0.027

**T16 cells with directional POSITIVE mean_diff (V > Baseline) — sign opposes 'negative transfer' narrative:**
  - `T16_V1_cross_ter` — mean_diff=+0.26 pp, p=0.7384, q=0.8035, BH sig: N
  - `T16_V3_cross_ter` — mean_diff=+0.08 pp, p=0.9041, q=0.9292, BH sig: N

### 4.2 Tests becoming non-significant after BH (across all groups)

- `T16_V2_cross_bin` (FT-V2 vs Baseline (cross, binary)): p=0.0487 → q=0.0900
- `T18_xsift_ter` (Original-weights CBraMod vs random-init (XSI-FT, ternary)): p=0.0304 → q=0.0592

### 4.3 Sign reversals vs paper claims

- `T16_V1_cross_ter`: positive direction (DAPT > Baseline) — paper §3.6 claims uniformly negative
- `T16_V3_cross_ter`: positive direction (DAPT > Baseline) — paper §3.6 claims uniformly negative

## 5. Recommended §3.6 / §3.7 / Abstract Narrative Adjustments

**§3.6 DAPT survival rate**: 2/12 of V_version × paradigm × task pairs survive BH q<0.05 as significantly negative.

⇒ Survival rate **17% < 50%** — paper's '一致负迁移 / consistent negative transfer' phrasing should be softened to **'directional negative observation; only a minority of (V × paradigm × task) cells are statistically significant after multiple-comparison correction'**.

**Top 3 narrative adjustments**:

1. **§3.6 lead**: replace 'consistent negative transfer' with a survival-rate-based claim (e.g., '2/12 V × paradigm × task cells reach BH q<0.05; the rest are directionally negative but not individually significant'). The group-mean negative deltas (V1 −0.75 / V2 −1.38 / V3 −0.70 pp) are smaller than within-subject SD (~10 pp) — that is *the* reason most cells fail to reach significance.
2. **Abstract / §1 / §7 Finding 4**: same softening — 'three independent DAPT configurations all show **directional** negative transfer' is correct; 'consistent' / 'uniformly' significant is not.
3. **§3.7 / Table 18**: with original-weights vs random-init paired tests, sign and significance match paper claims; no narrative change needed for §3.7.

## 6. Implementation Notes

### Code

```python
from scipy import stats
# paired t (two-sided) on shared subjects
t, p = stats.ttest_rel(arr_treat, arr_base)
diffs = arr_treat - arr_base
dz = diffs.mean() / diffs.std(ddof=1)
se = diffs.std(ddof=1) / sqrt(n)
tcrit = stats.t.ppf(0.975, df=n-1)
ci = (diffs.mean() - tcrit*se, diffs.mean() + tcrit*se)
# BH FDR (Benjamini-Hochberg step-up at α=0.05)
# implemented inline in this file; equivalent to statsmodels.multipletests(method='fdr_bh')
```

### Data sources

- **ExperimentDB** (canonical): `results/experiments.db`, table `subject_results` joined on `runs.run_tag`.
- **JSON caches**: per-subject `test_acc` extracted via 4-schema loader (within-subject dict, cross-subject `per_subject_test_acc` dict, legacy `subjects` flat list, extra-sessions `<step>` keyed dict).
- **Run-tag → file resolution**: `paper/run_registry.yaml` (used only as cross-check; actual file paths hard-coded in this script).

### Numeric precision

- mean_diff, SD_diff, 95% CI: 2 decimal places in pp
- Cohen's dz: 3 decimal places
- p, q: 3 decimal places (or `<0.001` when p<1e-3)

### Reproducibility

- Run from repo root: `python paper/reviews/stat_recompute_runner.py`
- Outputs this Markdown file deterministically; relies only on existing JSONs + DB (no GPU, no re-training).
- All 21 V1/V2/V3 vs Baseline pairs are over the same 21-subject cohort (S01..S21), so paired-t is fully balanced; no missing-subject drops occurred.
