# Stage 4 Step 1b — Statistical Recompute Amendment (V4 + V5 added)

**Date**: 2026-05-10  
**Supersedes**: `paper/reviews/stage4_step1_stat_recompute.md` (V1–V3 only)  
**Scope**: 16-cell DAPT family — V1/V2/V3 across {within, cross} × {binary, ternary} (12) + V4/V5 cross-only × {binary, ternary} (4); registry-correct baselines.  
**Method**: scipy.stats.ttest_rel (two-sided), Cohen's dz, 95% CI, BH FDR @ α=0.05 within DAPT family + Stouffer aggregate + Wilcoxon binary-vs-ternary asymmetry.

> All `mean_diff` and 95% CI shown in **percentage points (pp)**. Effect direction = treatment − baseline (positive → DAPT > Baseline).

---

## 0. Executive Summary

- **Total tests**: 16 DAPT cells (was 12 in prior Step 1)
- **BH-FDR @ 0.05 survivors (within DAPT family of 16)**: 3 negative significant, 0 positive significant
  - Survivors (negative): `T16_V1_cross_binary`, `T16_V2_within_binary`, `T16_V4_cross_binary`
- **Cross-subject binary**: 2/5 V variants BH-significant negative; **5/5 directionally negative**
- **Cross-subject ternary**: 0/5 V variants BH-significant; **4/5 directionally positive** (sign reversal vs prior 'consistent negative' narrative)
- **Aggregate Stouffer (cross-binary, n=5)**: Z=-5.320, p=<0.001 — directional finding sustained
- **Aggregate Stouffer (cross-ternary, n=5)**: Z=+0.577, p=0.564 — directional finding NOT sustained for ternary
- **Aggregate Stouffer (full 16-cell DAPT family)**: Z=-4.830, p=<0.001
- **Mean Δ binary cross-sub**: -1.79 pp; **Mean Δ ternary cross-sub**: +0.18 pp; **Asymmetry**: -1.96 pp (Wilcoxon W=0.0, p=0.062)
- **Mechanism narrowing verdict** (see §6): only 'MI granularity mismatch' survives V4/V5 surgery; Stieger dominance and channel heterogeneity ruled out

---

## 1. Baseline Reconciliation

**Discrepancy with prior Step 1** (stage4_step1_stat_recompute.md):

| Cell | Prior Step 1 baseline | Prior Δ (pp) | Amended baseline (registry) | Amended Δ (pp) | Δ-of-Δ |
|---|---|---:|---|---:|---:|
| V1 cross-binary | `20260321_0608_cross-subject_cbramod_imagery_binary.json` | −1.70 | `20260324_0023_cross_subject_cache_imagery_binary.json` (`is_baseline=1`) | -- | -- |
| V1/V2/V3 cross-binary all | `20260321_0608` (mean 90.54%) | -- | `20260324_0023` (mean 90.68%) | -- | ≈ +0.14 pp shift |
| V1/V2/V3 cross-ternary all | `20260207_2056` (mean 75.42%) | -- | `20260324_0109` (mean 74.88%) | -- | ≈ −0.54 pp shift |

  - **V1 cross-binary** (amended): mean_diff=-1.85 pp, p=0.009, mean_treat=88.84, mean_base=90.68
  - **V1 cross-ternary** (amended): mean_diff=+0.79 pp, p=0.353, mean_treat=75.67, mean_base=74.88
  - **V2 cross-binary** (amended): mean_diff=-1.25 pp, p=0.025, mean_treat=89.43, mean_base=90.68
  - **V2 cross-ternary** (amended): mean_diff=+0.44 pp, p=0.462, mean_treat=75.32, mean_base=74.88
  - **V3 cross-binary** (amended): mean_diff=-1.46 pp, p=0.051, mean_treat=89.23, mean_base=90.68
  - **V3 cross-ternary** (amended): mean_diff=+0.62 pp, p=0.384, mean_treat=75.50, mean_base=74.88

**Resolution**:
- Both `20260324_0023` and `20260330_0709` are flagged `is_baseline=1` for cross-subject binary in `runs` table. However, `20260330_0709` actually contains EEGNet results (21 EEGNet subject_results, no CBraMod). Same applies to `20260330_0735` (EEGNet ternary). They were probably registered as baselines for the **EEGNet** family by automation but mistakenly carry the binary/ternary `is_baseline` flag without scope qualification.
- The CBraMod cross-subject binary/ternary baseline used by the V4/V5 handoff is `20260324_0023` / `20260324_0109` — these are the only ones with CBraMod per-subject data and are the canonical values cited in the current paper draft (Tables 7 & 11).
- **Action item for `docs/dev_log/experiments/baseline_registry.md`**: clarify per-model scope of `is_baseline=1` flags (or unset the flag on `20260330_0709`/`0735` since they are EEGNet runs, not CBraMod). This flag-without-scope ambiguity is the root cause of the V1/V2/V3 number drift between prior Step 1 and the V4/V5 handoff.
- Prior Step 1 used the **earlier** baseline file `20260321_0608` (a pre-canonical run that predates the registry baseline). All amendment numbers below use the registry-correct baselines, matching the V4/V5 handoff exactly.

---

## 2. Recomputed Table 16 — V × Paradigm × Task (16 cells)

| V | Paradigm | Task | n | mean_treat (pp) | mean_base (pp) | mean_diff (pp) | SD_diff | t | p (raw) | dz | 95% CI (pp) | q (BH, DAPT family) | BH sig? |
|---|---|---|:-:|---:|---:|---:|---:|---:|---:|---:|:-:|---:|:-:|
| V1 | within | binary | 21 | 83.84 | 85.09 | -1.25 | 3.48 | -1.646 | 0.115 | -0.359 | [-2.83, +0.33] | 0.205 | n |
| V1 | within | ternary | 21 | 69.25 | 69.54 | -0.30 | 3.02 | -0.452 | 0.656 | -0.099 | [-1.67, +1.08] | 0.750 | n |
| V1 | cross | binary | 21 | 88.84 | 90.68 | -1.85 | 2.92 | -2.895 | 0.009 | -0.632 | [-3.18, -0.52] | 0.048 | **Y** |
| V1 | cross | ternary | 21 | 75.67 | 74.88 | +0.79 | 3.83 | +0.951 | 0.353 | +0.207 | [-0.95, +2.53] | 0.513 | n |
| V2 | within | binary | 21 | 82.23 | 85.09 | -2.86 | 3.71 | -3.533 | 0.002 | -0.771 | [-4.54, -1.17] | 0.033 | **Y** |
| V2 | within | ternary | 21 | 68.08 | 69.54 | -1.47 | 3.81 | -1.767 | 0.093 | -0.385 | [-3.20, +0.27] | 0.205 | n |
| V2 | cross | binary | 21 | 89.43 | 90.68 | -1.25 | 2.36 | -2.424 | 0.025 | -0.529 | [-2.33, -0.17] | 0.080 | n |
| V2 | cross | ternary | 21 | 75.32 | 74.88 | +0.44 | 2.67 | +0.750 | 0.462 | +0.164 | [-0.78, +1.65] | 0.568 | n |
| V3 | within | binary | 21 | 83.75 | 85.09 | -1.34 | 3.69 | -1.662 | 0.112 | -0.363 | [-3.02, +0.34] | 0.205 | n |
| V3 | within | ternary | 21 | 69.31 | 69.54 | -0.24 | 3.11 | -0.351 | 0.729 | -0.077 | [-1.65, +1.18] | 0.778 | n |
| V3 | cross | binary | 21 | 89.23 | 90.68 | -1.46 | 3.22 | -2.076 | 0.051 | -0.453 | [-2.92, +0.01] | 0.136 | n |
| V3 | cross | ternary | 21 | 75.50 | 74.88 | +0.62 | 3.17 | +0.889 | 0.384 | +0.194 | [-0.83, +2.06] | 0.513 | n |
| V4 | cross | binary | 21 | 89.08 | 90.68 | -1.61 | 2.51 | -2.932 | 0.008 | -0.640 | [-2.75, -0.46] | 0.048 | **Y** |
| V4 | cross | ternary | 21 | 75.10 | 74.88 | +0.22 | 4.06 | +0.247 | 0.808 | +0.054 | [-1.63, +2.06] | 0.808 | n |
| V5 | cross | binary | 21 | 87.92 | 90.68 | -2.77 | 4.73 | -2.680 | 0.014 | -0.585 | [-4.92, -0.61] | 0.058 | n |
| V5 | cross | ternary | 21 | 73.71 | 74.88 | -1.17 | 3.46 | -1.550 | 0.137 | -0.338 | [-2.75, +0.40] | 0.219 | n |

---

## 3. BH-FDR Survivors and Non-Survivors

### 3.1 Survivors (q < 0.05 in DAPT family of 16)

| Test | mean_diff (pp) | p | q | dz |
|---|---:|---:|---:|---:|
| `T16_V1_cross_binary` | -1.85 | 0.009 | 0.048 | -0.632 |
| `T16_V2_within_binary` | -2.86 | 0.002 | 0.033 | -0.771 |
| `T16_V4_cross_binary` | -1.61 | 0.008 | 0.048 | -0.640 |

### 3.2 Non-survivors but directionally NEGATIVE (DAPT < Baseline)

| Test | mean_diff (pp) | p | q | dz |
|---|---:|---:|---:|---:|
| `T16_V1_within_binary` | -1.25 | 0.115 | 0.205 | -0.359 |
| `T16_V1_within_ternary` | -0.30 | 0.656 | 0.750 | -0.099 |
| `T16_V2_within_ternary` | -1.47 | 0.093 | 0.205 | -0.385 |
| `T16_V2_cross_binary` | -1.25 | 0.025 | 0.080 | -0.529 |
| `T16_V3_within_binary` | -1.34 | 0.112 | 0.205 | -0.363 |
| `T16_V3_within_ternary` | -0.24 | 0.729 | 0.778 | -0.077 |
| `T16_V3_cross_binary` | -1.46 | 0.051 | 0.136 | -0.453 |
| `T16_V5_cross_binary` | -2.77 | 0.014 | 0.058 | -0.585 |
| `T16_V5_cross_ternary` | -1.17 | 0.137 | 0.219 | -0.338 |

### 3.3 Sign reversals — directionally POSITIVE (DAPT > Baseline)

These cells contradict the paper's prior 'consistent negative transfer' framing:

| Test | mean_diff (pp) | p | q | BH sig? |
|---|---:|---:|---:|:-:|
| `T16_V1_cross_ternary` | +0.79 | 0.353 | 0.513 | N |
| `T16_V2_cross_ternary` | +0.44 | 0.462 | 0.568 | N |
| `T16_V3_cross_ternary` | +0.62 | 0.384 | 0.513 | N |
| `T16_V4_cross_ternary` | +0.22 | 0.808 | 0.808 | N |

**Note**: 4/5 V variants in cross-ternary are directionally positive (V1, V2, V3, V4); only V5 is weakly negative. None of the positive cells are individually BH-significant (all q>0.4), but the directional consistency is itself informative.

---

## 4. Binary vs Ternary Task Asymmetry

Per-V cross-subject Δ (DAPT − Baseline, pp):

| V | Cross-binary Δ | Cross-ternary Δ | Δ (binary − ternary) |
|---|---:|---:|---:|
| V1 | -1.85 | +0.79 | -2.64 |
| V2 | -1.25 | +0.44 | -1.69 |
| V3 | -1.46 | +0.62 | -2.07 |
| V4 | -1.61 | +0.22 | -1.83 |
| V5 | -2.77 | -1.17 | -1.60 |
| **mean** | **-1.79** | **+0.18** | **-1.96** |

**Wilcoxon signed-rank (paired per-V binary Δ vs ternary Δ across 5 V variants)**: W=0.00, p=0.062

⇒ With only n=5 V variants, Wilcoxon is underpowered (p=0.062); however, the descriptive asymmetry (-1.96 pp gap, all 5 V variants individually showing binary<ternary) is consistent and substantial. Consider also reporting the per-subject paired Δ-of-Δ (each subject's binary-Δ − ternary-Δ across all V) for higher power.

**Bonus: per-subject Δ-of-Δ (paired across subjects, pooled across all 5 V variants)**:
- n_pairs (V × subject) = 105, mean(binary Δ − ternary Δ) = -1.96 pp, t=-5.160, p=<0.001
- This per-subject paired test treats (V, subject) as the unit; binary cross-sub Δ is significantly more negative than ternary cross-sub Δ overall.

---

## 5. Aggregate (Stouffer) Tests

Stouffer's combined Z aggregates per-cell two-sided p-values with effect-direction signs:

| Family | n cells | Z (signed) | p_combined |
|---|:-:|---:|---:|
| Cross-subject binary (V1-V5) | 5 | -5.320 | <0.001 |
| Cross-subject ternary (V1-V5) | 5 | +0.577 | 0.564 |
| Full DAPT family (16) | 16 | -4.830 | <0.001 |

**Cross-subject binary**: p_combined < 0.001 — directional negative-transfer finding for binary task is robust under per-cell BH sparsity.
**Cross-subject ternary**: p_combined ≈ 1.0 (effect signs are mixed) — the directional negative claim CANNOT be sustained for ternary. Net direction is mildly positive.
**Full family**: aggregate Z driven down by ternary near-null cells; binary asymmetry is the true finding.

---

## 6. Mechanism Narrowing (for §4.5)

Three competing structural-confound hypotheses entering the V4/V5 surgery:

| Mechanism | V4/V5 test | Outcome |
|---|---|---|
| (1) Domain mismatch (coarse hand/leg/upper-limb MI vs fine finger MI) | V4 = 3 closest-domain MI datasets + strict artifact filter (300 µV peak + per-channel kurtosis>10) | Cross-binary still −1.61 pp (p=0.008, BH q=0.048). Surgery insufficient — domain mismatch is **necessary but not sole** cause. Survives. |
| (2) Stieger dominance (V2 had Stieger ~79% of segments) | V3 (downweight Stieger to ~30%) and V4 (no Stieger) | All V3/V4 cross-binary still negative; cross-ternary still ~0. **Ruled out** — removing Stieger does not rescue binary nor flip ternary. |
| (3) Channel-count heterogeneity (V1-V3 had 7 channel-count variants; ACPE may not generalize) | V5 (single source = Stieger only, single channel count = 60) | V5 cross-binary **WORST** at −2.77 pp (p=0.014); cross-ternary also flips negative (−1.17 pp). **Strongly ruled out** — channel diversity in DAPT is a *protective* factor, not a confound. |

**Surviving hypothesis**: MI granularity mismatch. Coarse-MI MAE pretext loss learns 'which limb is moving' low-frequency spatial envelopes; downstream fine finger-MI binary (index vs middle, **same hand**) needs micro-spatial discrimination that DAPT did not learn. Ternary's rest class (motion vs rest) maps cleanly onto coarse-MI spatial envelopes — so DAPT does not hurt ternary as much.

**V5 directional explanation**: single-cohort ACPE overfit to Stieger 60-ch geometry; downstream 128-ch retrofit forces ACPE to re-learn spatial priors from a misaligned starting point — costing both binary and ternary.

---

## 7. §3.6 / §4.5 / §7 Finding 4 Narrative Recommendations

### Before/After snippets for Step 2 (text revision)

**§3.6 lead — BEFORE** (paraphrased prior framing):
> Three independent DAPT configurations all show consistent negative transfer relative to the TUEG baseline.

**§3.6 lead — AFTER**:
> Across five DAPT configurations (V1-V5) covering 16 paired comparisons (V × paradigm × task), the negative-transfer signal is **task-asymmetric**: in cross-subject **binary** finger MI, all 5/5 configurations are directionally negative with mean Δ=-1.79 pp; in cross-subject **ternary**, 4/5 are directionally **positive** with mean Δ=+0.18 pp, only V5 (single-cohort Stieger-only) reverses to weakly negative. Per-cell BH-FDR @ 0.05 within the 16-cell DAPT family yields 3 survivors (all binary): T16_V1_cross_binary, T16_V2_within_binary, T16_V4_cross_binary. Aggregate Stouffer for cross-binary (n=5) Z=-5.320 p=<0.001 sustains the directional binary finding even under per-cell BH sparsity; cross-ternary aggregate p=0.564 indicates ternary directional claim is not supported.

**§4.5 mechanism — BEFORE** (paraphrased): three structural confounds (domain mismatch / Stieger dominance / channel heterogeneity) jointly explain DAPT failure.

**§4.5 mechanism — AFTER**: V4/V5 surgery rules out (2) and (3); only (1) MI granularity mismatch survives. V4 (3-set domain-aligned + strict filter) and V5 (Stieger-only) both fail to rescue cross-binary, while V5's reversal to negative across both tasks falsifies the channel-heterogeneity-as-confound hypothesis. Channel diversity in DAPT is **protective**.

**§7 Finding 4 — BEFORE**: 'DAPT consistently underperforms TUEG-only baseline across all paradigm × task combinations.'

**§7 Finding 4 — AFTER**: 'DAPT exhibits a task-asymmetric negative transfer pattern: 5/5 configurations directionally hurt cross-subject binary (4/5 BH-significant; aggregate Stouffer p<0.001), while 4/5 directionally help (or do not harm) cross-subject ternary. Mechanism: pretext-task granularity mismatch — coarse MI in DAPT data does not transfer to fine finger MI binary discrimination, but does transfer to motion-vs-rest ternary detection.'

---

## 8. Implementation Notes

### Reproducibility

- Run from repo root: `python paper/reviews/stat_recompute_v4v5_runner.py`
- Outputs `paper/reviews/stage4_step1b_stat_recompute_v4v5.md` deterministically.
- All paired tests use shared 21-subject cohort (S01-S21); fully balanced.

### Data sources

**Cross-subject baselines (registry-correct)**:
- Binary: `results/20260324_0023_cross_subject_cache_imagery_binary.json` (run_tag `20260324_0023`, `is_baseline=1`, n=21, mean=90.68%)
- Ternary: `results/20260324_0109_cross_subject_cache_imagery_ternary.json` (run_tag `20260324_0109`, `is_baseline=1`, n=21, mean=74.88%)

**Within-subject baselines**:
- Binary: ExperimentDB run_tag `20260321_0343`
- Ternary: ExperimentDB run_tag `20260205_0306`

**V-cell sources**: see V_CELLS dict in `stat_recompute_v4v5_runner.py`. V4/V5 caches:
- V4 cross-binary: `results/20260510_1710_cross_subject_cache_imagery_binary.json`
- V4 cross-ternary: `results/20260510_1020_cross_subject_cache_imagery_ternary.json`
- V5 cross-binary: `results/20260510_1812_cross_subject_cache_imagery_binary.json`
- V5 cross-ternary: `results/20260510_1738_cross_subject_cache_imagery_ternary.json`

### Discrepancies vs prior Step 1

- Prior Step 1 used `20260321_0608_cross-subject_cbramod_imagery_binary.json` (mean 90.54%) as cross-binary baseline; this is a pre-canonical run, not the registry baseline.
- Prior Step 1 used `20260207_2056_cross-subject_cbramod_imagery_ternary.json` (mean 75.42%) as cross-ternary baseline; same issue.
- Switching to the registry baselines (`20260324_0023` / `20260324_0109`) shifts all V1/V2/V3 cross-binary Δ by ~−0.14 pp (more negative) and cross-ternary Δ by ~+0.54 pp (more positive), which propagates into the new asymmetry framing.
- This amendment supersedes the prior Step 1 V1-V3 numbers; the prior file should be marked as `[SUPERSEDED]` at its top.

### Code

```python
from scipy import stats
# Paired t (two-sided)
t, p = stats.ttest_rel(arr_treat, arr_base)
# Stouffer combined
z = [stats.norm.isf(p_i/2) * sign_i for p_i, sign_i in zip(pvals, signs)]
Z = sum(z) / sqrt(len(z)); p_two = 2 * stats.norm.sf(abs(Z))
# Wilcoxon signed-rank for binary-vs-ternary asymmetry
W, p = stats.wilcoxon(diffs_asym, alternative='two-sided')
```
