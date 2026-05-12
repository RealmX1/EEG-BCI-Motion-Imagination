# Stage 4 Step 3 — Figure Generation Report

**Date**: 2026-05-10
**Operator**: Figure Generation Specialist (sub-agent for ARS Stage 4 Step 3)
**Inputs**: paper_draft_v3.1.md (READ-ONLY for image refs); stage4_step1b_stat_recompute_v4v5.md (statistical foundation); paper/run_registry.yaml + V4/V5 cache paths from `docs/handoffs/2026-05-10_dapt_v4_v5.md`.

---

## 1. Summary

- **10 figures targeted** (NEW-A, NEW-B, T3.2, T3.4, T3.5, T3.6, T3.7×2 binary/ternary, T3.8, T3.9, T3.10).
- **Successfully generated: 10 / 10**. All PNG files present in `paper/figures/` with size > 100 KB.
- **Skipped / blocked: 0**. No GPU experiments triggered; all data sourced from existing JSON caches and ExperimentDB.
- **paper_draft_v3.1.md image-reference updates: 5** (NEW-A, NEW-B, T3.2, T3.5, T3.10). The other 5 figures (T3.4, T3.6, T3.7×2, T3.8, T3.9) replace existing PNGs at the same path, requiring no markdown edit.
- **paper_draft_v3.0.1.md** byte-identical, untouched.

---

## 2. Per-figure status

| ID | Title | Output Path | Status | Script (--figure flag) | Provenance footer |
|----|-------|-------------|--------|------------------------|-------------------|
| NEW-A | §3.7 Exploratory Ablation Overview | `paper/figures/exploratory_ablation_overview.png` | OK (300 KB) | `exploratory_ablation_overview` | yes |
| NEW-B | §3.6 V1-V5 DAPT 16-cell Forest Plot | `paper/figures/dapt_v1_v5_forest.png` | OK (260 KB) | `dapt_v1_v5_forest` | yes |
| T3.2 | Sensitivity Scaling (dual y-axis 32/8/4ch spread vs best acc) | `paper/figures/sensitivity_scaling.png` | OK (179 KB) | `sensitivity_scaling` | yes |
| T3.4 | Extra Sessions 3-paradigm + per-subject + Δ swarm (replaces existing) | `paper/figures/extra_sessions_paradigm_binary.png` | OK (362 KB) | `extra_sessions_paradigm` | yes |
| T3.5 | 21-subject × 8-condition heatmap (Sup Fig S2) | `paper/figures/subject_heatmap.png` | OK (255 KB) | `subject_heatmap` | yes |
| T3.6 | Channel Scaling — split into 3 panels (replaces existing) | `paper/figures/channel_scaling_curve.png` | OK (309 KB) | `channel_scaling` | yes |
| T3.7-bin | Extra Sessions Binary baseline-colored scatter | `paper/figures/extra_sessions_binary.png` | OK (325 KB) | `extra_sessions_binary_v2` | yes |
| T3.7-ter | Extra Sessions Ternary baseline-colored scatter | `paper/figures/extra_sessions_ternary.png` | OK (325 KB) | `extra_sessions_ternary_v2` | yes |
| T3.8 | Further-pretraining V1-V5 + reverse-gradient panel (replaces existing) | `paper/figures/further_pretraining.png` | OK (204 KB) | `further_pretraining` | yes |
| T3.9 | Inference latency + throughput panel | `paper/figures/inference_latency.png` | OK (128 KB) | `inference_latency` | yes |
| T3.10 | Fig 5a / Fig 5b merged into single Figure 5 | `paper/figures/fig5_4ch_optimal_vs_neg_control.png` | OK (149 KB) | `fig5_merged` | yes |

(11 PNG entries because T3.7 expands into binary + ternary; 10 figure work items per spec.)

---

## 3. v3.1.md image reference updates

| Item | Section | Action | Notes |
|------|---------|--------|-------|
| NEW-A | §3.7 (after caveat block, before §3.7.1) | INSERT new image ref `../figures/exploratory_ablation_overview.png` | Inserted as 图 12 with Chinese caption emphasizing "在受限 HPO 预算下观察" |
| NEW-B | §3.6 (after Table 16, replacing single 图 10) | INSERT new image ref `../figures/dapt_v1_v5_forest.png` (图 10a) | Existing 图 10 split into 10a (forest) + 10b (matrix + reverse-gradient). 10b path unchanged: `../../paper/figures/further_pretraining.png` |
| T3.2 | §3.5.3 末 (before §3.5.4) | INSERT new image ref `../figures/sensitivity_scaling.png` | Inserted as 图 4c with caption tying to Table 9 / §3.5.3 |
| T3.5 | Supplementary (Figure S2 inserted before existing Figure S1) | INSERT new image ref `../figures/subject_heatmap.png` | Section 5 supplementary; data-source line included |
| T3.10 | §3.5.3 (Fig 5 area) | DELETE 2 image refs (5a + 5b on lines ~676 / ~678) → INSERT single merged ref `../figures/fig5_4ch_optimal_vs_neg_control.png` | Caption updated to describe single-figure side-by-side panel |
| T3.4 | §3.4.4 (line ~540 in v3.1) | NO markdown edit (path unchanged: `../../paper/figures/extra_sessions_paradigm_binary.png`) | New PNG replaces old at same path |
| T3.6 | §3.5.2 (line ~618) | NO markdown edit (path unchanged: `../../paper/figures/channel_scaling_curve.png`) | New PNG replaces old at same path |
| T3.7-bin | §3.4.1 (line ~450) | NO markdown edit (path unchanged) | New PNG replaces old at same path |
| T3.7-ter | §3.4.2 (line ~497) | NO markdown edit (path unchanged) | New PNG replaces old at same path |
| T3.8 | §3.6 (line ~752) | NO markdown edit (path unchanged: `../../paper/figures/further_pretraining.png`) | New PNG replaces old at same path; replacement framed as 图 10b in caption |
| T3.9 | §3.8 (line ~908) | NO markdown edit (path unchanged: `../../paper/figures/inference_latency.png`) | New PNG replaces old at same path |

---

## 4. Reproducibility

All 10 figure generators are entry points in `scripts/paper/generate_paper_figures.py`. The dispatcher dictionary `FIGURE_GENERATORS` was extended with 7 new keys (existing 5 keys had their target functions swapped to the v2/v3 implementations; same `--figure` flag, new function body):

```bash
# New work items (Stage 4 Step 3 additions)
uv run python scripts/paper/generate_paper_figures.py --figure dapt_v1_v5_forest
uv run python scripts/paper/generate_paper_figures.py --figure exploratory_ablation_overview
uv run python scripts/paper/generate_paper_figures.py --figure sensitivity_scaling
uv run python scripts/paper/generate_paper_figures.py --figure subject_heatmap
uv run python scripts/paper/generate_paper_figures.py --figure extra_sessions_binary_v2
uv run python scripts/paper/generate_paper_figures.py --figure extra_sessions_ternary_v2
uv run python scripts/paper/generate_paper_figures.py --figure fig5_merged

# Replacements at existing flag names (now point to the new v2/v3 functions)
uv run python scripts/paper/generate_paper_figures.py --figure further_pretraining
uv run python scripts/paper/generate_paper_figures.py --figure extra_sessions_paradigm
uv run python scripts/paper/generate_paper_figures.py --figure channel_scaling
uv run python scripts/paper/generate_paper_figures.py --figure inference_latency

# Convenience: regenerate all 10 in one shot
uv run python scripts/paper/generate_paper_figures.py --figure all
```

A single source-of-truth constant `DAPT_V_RESULTS_STEP1B` (at the top of the additions block) holds all 16 Step-1b verified Δ / CI / q values; both `dapt_v1_v5_forest` and `further_pretraining_v3` consume the same constant, ensuring fidelity to Step 1b numbers (no manual transcription).

A second constant `STOUFFER_AGG` holds the aggregate Z and p values (cross-binary Z=−5.32, p<0.001; cross-ternary Z=+0.58, p=0.564; full DAPT family Z=−4.83, p<0.001) for both figures.

`_data_quality_label()` helper maps subject ID → quality bin per §2.9 Table 5; used by the heatmap left annotation strip.

---

## 5. Quality notes

**Publication-level polish DEFERRED to Stage 5** per work item spec; the figures are working drafts with the following known minor issues:

- **`extra_sessions_paradigm_binary.png`** (T3.4): matplotlib emitted `UserWarning: figure includes Axes that are not compatible with tight_layout` due to colorbar in panel C; layout is acceptable but a `constrained_layout` migration would be cleaner.
- **`channel_scaling_curve.png`** (T3.6): same tight_layout warning. Panel A's red-envelope annotations partly overlap the dotted-line per-method markers at 32ch — readable but not ideal at small print size; Stage 5 polish should pull annotation labels further out.
- **`further_pretraining.png`** (T3.8 panel B): the OLS regression line is pulled by the cross-paradigm / log-N x-axis design; the regression slope tells a directional story but is not a real statistical test (per-cell paired tests already done in §3.6 Table 16). Consider relabelling "OLS slope" → "descriptive slope" in Stage 5.
- **`subject_heatmap.png`** (T3.5): some accuracy cells are missing for EEGNet cross-binary at S03/S05 (per registry: `20260330_0709` returns NaN for those subjects). The figure shows blank cells — this is faithful to the data, not a bug.
- **`dapt_v1_v5_forest.png`** (NEW-B): three Stouffer-aggregate diamonds are placed below the cell rows; their x-position is the mean Δ for cross-bin (-1.79) / cross-ter (+0.18); the "full 16-cell" diamond is placed at x=0 because the aggregate is multi-condition (no scalar mean). Annotation text gives the Z and p values.
- **`exploratory_ablation_overview.png`** (NEW-A): three decomposition arrows (capacity ladder / cross-arch / TUEG-pretraining) plus a caveat box in the bottom-right corner; the box wording aligns word-for-word with the §3.7 chapter intro caveat.

**No data inconsistencies detected**:
- V4/V5 caches all loaded successfully (`20260510_1710 / _1020 / _1812 / _1738`); per-subject test_acc fields parsed exactly as expected.
- `extra_sessions_cross_binary` cache schema (`per_subject_test_acc` nested under `step.per_subject_test_acc`) handled by the existing `extract_cross_subject_extra_session_step_accs` helper.
- Step 1b mean Δ for cross-binary computed across V1-V5 reproduces -1.79 pp; Stouffer Z and p values match the report.

---

## 6. Verification

PNG files in `paper/figures/` (run `ls -lh paper/figures/*.png` for live state):

| File | Size | Modified |
|------|-----:|----------|
| `32ch_comparison.png` | 109 KB | 2026-03-31 (unchanged, not in scope) |
| `channel_method_ranking_flip.png` | 198 KB | 2026-05-05 (unchanged, not in scope) |
| `channel_scaling_curve.png` | 309 KB | 2026-05-10 (T3.6, NEW) |
| `cross_subject_pooling_forest.png` | 156 KB | 2026-05-05 (unchanged) |
| `dapt_v1_v5_forest.png` | 260 KB | 2026-05-10 (NEW-B) |
| `exploratory_ablation_overview.png` | 300 KB | 2026-05-10 (NEW-A) |
| `extra_sessions_binary.png` | 325 KB | 2026-05-10 (T3.7-bin, NEW) |
| `extra_sessions_paradigm_binary.png` | 362 KB | 2026-05-10 (T3.4, NEW) |
| `extra_sessions_strategy_comparison.png` | 316 KB | 2026-04-02 (unchanged) |
| `extra_sessions_ternary.png` | 325 KB | 2026-05-10 (T3.7-ter, NEW) |
| `fig5_4ch_optimal_vs_neg_control.png` | 149 KB | 2026-05-10 (T3.10, NEW) |
| `further_pretraining.png` | 204 KB | 2026-05-10 (T3.8, NEW) |
| `inference_latency.png` | 128 KB | 2026-05-10 (T3.9, NEW) |
| `sensitivity_scaling.png` | 179 KB | 2026-05-10 (T3.2, NEW) |
| `subject_heatmap.png` | 255 KB | 2026-05-10 (T3.5, NEW) |

11 of 15 PNGs in `paper/figures/` have either been newly generated or updated under this Stage 4 Step 3 cycle; the remaining 4 are out-of-scope helper figures from earlier stages.

---

## 7. Notes for downstream stages

- **Step 4 (R&R Letter)**: figures are ready to cite. The forest plot (NEW-B) and reverse-gradient panel (T3.8-B) jointly support the §3.6 task-asymmetric narrative; the exploratory overview (NEW-A) gives a single visual hook for the §3.7 caveat-heavy narrative; the merged Fig 5 (T3.10) frees one image slot in the §3.5.3 control-experiment block.
- **Step 5 (formatting / final polish)**: defer publication-level typography (font sizes, axis label uniformity, color palette CMYK conversion) to Stage 5. The three minor layout warnings noted in §5 above should be addressed.
- **Reviewers' likely follow-up**: per-cell n=21 vs Stouffer n=5 disparity in NEW-B; the diamond annotations make this explicit but a reviewer may still ask for a verbal note in the caption — the v3.1.md caption already states "5 V × paradigm × task = 16 cells".
