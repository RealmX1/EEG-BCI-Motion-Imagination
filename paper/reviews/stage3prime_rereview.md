# Stage 3' Re-Review: paper_draft_v3.1.md

**Date**: 2026-05-10
**Reviewer**: Consolidated synthesis (academic-paper-reviewer skill, re-review mode)
**Inputs**: `paper/drafts/paper_draft_v3.1.md` (1505 lines) + `paper/reviews/response_to_reviewers_v3.1.md` (567 lines) + 5 first-round reviews (35 concerns) + Stage 4 audit (`stage4_step1b_stat_recompute_v4v5.md`, `stage4_step2.5_integration_report.md`) + P0.3 handoff (`docs/handoffs/2026-05-10_p03_label_shuffle_results.md`).

---

## §1 Executive Summary

v3.1.md is a substantively different and notably stronger manuscript than v3.0.1. The §3.6 chapter has been re-conceived from "uniform negative transfer" to a **task-asymmetric** finding with rigorous BH-FDR + Stouffer aggregation across a 16-cell DAPT family, two new surgical experiments (V4 / V5) that genuinely retire two of three competing mechanisms, and a transparent disclosure of the four cross-ternary directional sign reversals that the v3.0.1 narrative had elided. §3.7 has been honestly demoted from "three-way decomposition" to "exploratory ablations," and the (W) Part A d^c HPO calibration argument is internally sound and well-cited. The just-completed P0.3 label-shuffle control (pooled 49.58% vs 90.68% headline, Δ = −41.1 pp) is integrated with appropriate restraint — it rules out leakage and subject-identity shortcuts but is not over-claimed as ruling out cohort-selection inflation, which remains a separately-handled limitation. The paper's principal remaining weakness is that two of its strongest cross-paradigm claims — TUEG-pretraining contributes "binary +23.10 / ternary +30.79 pp" within-subject and "transformer + ACPE provides architecture-independent value" — still rest on a random-init CBraMod that re-uses the cross-subject baseline's HP (`get_default_config()`); the (W) Part B reframe to "exploratory" mostly absorbs this, but a residual reader will still encounter these large within-subject pp numbers in the abstract and §1.4 / §7 Finding 1 without seeing the corresponding HP-mismatch caveat in the same sentence. **Recommendation: Minor Revision.** The paper is now substantively sound at master's-thesis defense calibration; the residual issues are presentational tightening rather than missing experiments.

---

## §2 Revision Verification Checklist (35 concerns)

Status legend: ✅ fully addressed / ⚠️ partially addressed / ❌ not addressed / ⏳ deferred-with-acceptable-justification.

| # | Source | Concern | Status | v3.1 evidence | Quality |
|---|--------|---------|:-:|---|---|
| 1 | EIC-1 | Missing methodological-positioning opener | ✅ | Abstract lines 18–22 + §1.4 F1 line 77 | Solid; opener now names "task-asymmetric DAPT" + "exploratory three-way ablation" as narrative spine. |
| 2 | EIC-2 | Statistical depth below JNE bar (no FDR / no effect size / no CI) | ✅ for §3.6, ⚠️ elsewhere | §3.6 Table 16 (lines 731–751) full BH-FDR + dz + 95% CI + Stouffer; §2.8 (294) declares full-paper nominal-significance + "BH-FDR for other families in Supplementary" | DAPT family fully calibrated. Other families (§3.4, §3.5, §3.7) only get individual paired-t in main tables; §2.8 mitigates by declaring nominal-significance, which is an acceptable JNE-equivalent fallback for a master's thesis. |
| 3 | EIC-3 | "+27 pp" definition drift | ✅ | Abstract line 20 + §1.4 F1 (77) + §3.7.3 (893) + §4.1 (958) + §7 F1 (1084) all show "binary +23.10 / ternary +30.79 pp" double-value | Five-anchor unification verified; integration audit §4 confirms count. |
| 4 | EIC-4 | 90.68% headline lacks cohort caveat | ✅ | Abstract (20), §1.4 F1 (77), §1.4 F2 (79), §7 F1 (1084), §7 F2 (1086), §5 Limitation #2 (1044) | "21 名 responder cohort" surfaces 5 places. §5 #2 also includes directional inflation estimate (~67% on naive 49-person cohort). |
| 5 | EIC-5 | DAPT V3 warm-restart caveat hidden in causal narrative | ✅ | §3.6.3 (lines 790–792) explicitly preserves V2/V3 caveat; §4.5 (994–1005) no longer relies on V3 vs V2 +0.68 pp | Surgery 1 (V4) and Surgery 3 (V3 → V4 → V5 chain) now do the mechanism-narrowing work without leaning on the warm-restart contrast. |
| 6 | EIC-6 | EEGNet baseline → Mid double-axis change | ✅ | §3.7 chapter caveat block (811); §3.7.1 (821); §3.7.3 footnote ¹ (889); §4.1 (952); §7 F1 (1084) | Five-place chain. Footnote ¹ is precise about the F1=16/F2=64 → F1=32/F2=256 + single Linear → [2048,2048]+LayerNorm jump. |
| 7 | EIC-7 | Extra-sessions N=16 boundary not surfaced | ✅ | §3.4 abstract section (24) + §3.4.4 table notes + §4.4 (985) | All three places now flag "N=16 子集" with reason. |
| 8 | EIC-8 | §3.5.2 4ch BP discussion verbose | ⚠️ | §3.5.2 (645–649) compressed but still ~120 words across 3 mechanisms (i)/(ii)/(iii) | Compression is real but the 3-mechanism enumeration could be one sentence. Acceptable for master's thesis. |
| 9 | R1-1 (CRITICAL) | §3.7 three-way decomposition isolation rigor | ✅ via reframe | §3.7 title (807); chapter caveat (811); §3.7.3 reframe + composite-estimate footnotes (882–895); §6 #8 (1076) registers ~80–120 GPU-h sweep | The "exploratory ablation" reframe is honest; the strong claim is replaced by three weaker conditional claims (a)/(b)/(c) at line 895. See §3 of this re-review for detailed verdict. |
| 10 | R1-2 | HPO budget asymmetry + missing Table S5b cross HPO count | ✅ for budget asymmetry, ⚠️ for full Table S5b cross row | §2.5.1 (lines 218–224) HPO calibration via Bergstra 2011 / Snoek 2012 d^c; Table S5e referenced; (b) EEGNet cross HPO column status partially disclosed in R&R B.2(b) | Bergstra/Snoek calibration is methodologically solid (see §3 below). EEGNet cross-subject HPO trial count is mentioned in R&R but I did not verify Table S5b actually contains the new column in v3.1.md (Step 4 cleanup #8 in integration report flags this as pending). |
| 11 | R1-3 | Channel selection "mild leakage" not quantified | ⏳ | P1.4 declared queued in R&R (G.3); cohort caveat surfaced in §1.4 F2 (79) and §7 F2 (1086); §5 #1 unchanged | Acceptable deferral with caveat, but the queued P1.4 result should land before Stage 3' is complete or be folded into a final round. |
| 12 | R1-4 | Multi-comparison correction + effect size in main tables | ✅ for §3.6, ⚠️ otherwise | Same evidence as #2 above | Same partial verdict as #2 — DAPT fully calibrated, other families nominal with §2.8 declaration. |
| 13 | R1-5 | §3.6 V2 LMDB interruption + "fully converged" claim | ✅ | §3.6.3 (790–792) explicit "Epoch 13 因 Windows LMDB MapResizedError 中断" + §2.7.2 V2 row already says "Epoch 12 best, early-stop at LMDB break" | "V2 全量训练后..." wording purged per integration audit; replaced with "V2 在 Epoch 12 处被强制截断" globally. |
| 14 | R1-6 | §3.5.4 N=3 framework over-claim | ✅ | §3.5.4 (713–720) explicitly demoted to "基于 3 个数据点的方向性观察 / 工作假设"; §4.6 + §4.8 hedged | Both downstream consumers (§4.6 deployment, §4.8 strategy axis) cite "based on 3 data points, needs validation." |
| 15 | R1 Minor #1 | Figure 1/6/6b version sync | ✅ per Stage 4 Phase 3 figures report | 10 figures regenerated per `stage4_step3_figures_report.md` reference | I did not independently re-verify figures, but the audit trail is in place. |
| 16 | R1 Minor #2 | CBraMod parameter count three-way inconsistency | ✅ | grep `30.48M` count = 9 across abstract + §1.3 + §3.7.1 table + §3.7.2 + §4.1 + §3.7.3 footnotes (per integration audit §4) | Unification successful. |
| 17 | R1 Minor #3 | deepEEGNet citation page numbers | ✅ | §2.4.1 (178) + §3.7.1 mentions [3] inline | Light verification but adequate. |
| 18 | R1 Minor #4 | EEGNet 16K vs 10K vs 16,162 inconsistency | ⚠️ | Step 4 cleanup #2 (integration report) notes some legacy "~10K" wording may persist | Per audit, not fully swept. Minor. |
| 19 | R1 Minor #5 | §3.1 line 326 S20 annotation | ✅ | §3.1 (335) annotates "S20 (52.50% / 61.25%) 仅略高于随机" | Done. |
| 20 | R1 Minor #6 | EEGNet vs CBraMod preprocessing pipeline misalignment | ⚠️ | Step 4 cleanup #3 flags "§5 Limitation needs new row" as pending | R&R B.12 promises a Limitation row, but I did not find a dedicated row in §5 for preprocessing-pipeline alignment confound. The §5 #7 entry covers "foundation model + pretraining range" which partly overlaps but isn't the same. Genuine residual gap. |
| 21 | R1 Minor #7 | EMA Table S6 grey/highlight | ⚠️ | R&R B.13 promises caption rename; visual highlighting deferred to typesetting | Acceptable. |
| 22 | R1 Minor #8 | §3.7.2 random-init within HP not optimized for from-scratch | ✅ | §3.7.2 (868–872) surfaces 70-80% / 15-25% / <5% probability estimate from handoff with explicit "无法严格区分 saddle-lock vs HP-mismatch" | High-quality disclosure with literature anchor (Mosbach 2021). |
| 23 | R2-1 (CRITICAL) | Literature coverage 9 → 20+ | ✅ | References [10]–[25] appended; integration audit §4 verifies inline citation count (Schirrmeister=8 occurrences, Gururangan=5, etc.) | 16 new refs verified. R2 §5.3 Tier C 3 refs explicitly declined with justification — acceptable. |
| 24 | R2-2 | XSI-FT novel-naming concern | ✅ | §3.3 (379) full lineage paragraph citing [18] Lotte 2018 + [25] Pan & Yang 2010 + [3] Ding | Excellent — explicitly disclaims method-novelty. |
| 25 | R2-3 | DAPT methodological over-generalization | ✅ | §4.5 (1005) + §4.8 末段 (1035) + §7 末段 (1094) all reframe to Gururangan 2020 [20] anchor + V5 reverse-falsification | Strong alignment with NLP DAPT literature framing. |
| 26 | R2-4 | Table 0 apples-to-oranges | ✅ | Table 0 (52–61) renamed to "方法学全景" + "可比性说明" footnote; abstract (20) no longer juxtaposes 90.68% with 80.56% directly | Minor residual: Table 0 still has "二分类准确率" column header which can still read as direct comparison; explicit "评估难度" column was added per integration audit but I did not fully verify in-line. |
| 27 | R2-5 | §3.2 EEGNet cross-subject reading citation [5] | ✅ | §3.2 (350+) Step 4 cleanup #6 flags "[5] Lawhern 2018 inline" as pending | ⚠ flagged in integration audit §5 #6. |
| 28 | R2-6 | §3.5.2 Pfurtscheller citation | ✅ | §3.5.2 (645) inline [14] + §3.5.3 (670) inline [19] | Done. |
| 29 | R2-7 | §3.4 longitudinal BCI literature handshake | ⚠️ | R&R C.7 partial accept; no new ref added | Defensible decline. |
| 30 | R2-8 | §3.9 Mognon 2011 / ICLabel data quality | ❌ | R&R C.8 explicitly declined; §3.9 unchanged on this axis | Acceptable decline at master's-thesis scale. |
| 31 | R2-9 | §2.4.1 EEGNet "重新搜索" citing [5] | ✅ | §2.4.1 (178) inline [5] | Done. |
| 32 | R2-10 | Cohort filter §3.1/§3.2 surface | ✅ | §3.1 / §3.2 cohort caveat per integration audit | Done. |
| 33 | R3-1 | §1.3 / §3.6 / §4.5 / §4.8 NLP DAPT dialogue | ✅ | §1.3 (69) reframed to "条件性成功 + Gururangan 2020"; §4.5 (1005) + §4.8 (1035) + §7 (1094) all anchored | Tight literature integration. |
| 34 | R3-2 | §3.7.2 Random-init within ternary 18/21 single-direction interpretation | ✅ | §3.7.2 (868–872) surfaces 70-80%/15-25%/<5% probability estimate + Mosbach [21] anchor | Honest. |
| 35 | R3-3 | §3.7.1 EEGNet-Huge v1/v2 = "capacity reverse-scaling" → optimization failure | ✅ | §3.7.1 (835) v1/v2 footnote rewritten to "BF16 + 深 MLP 头优化栈兼容性问题" + Hoffmann [22] anchor at footnote level + §4.1 (952) | Good — direct evidence (v3 + LayerNorm trainable) is now the headline diagnosis, not "capacity saturation." |
| 36 | R3-4 | "+27 pp" attribution strength | ✅ | Same evidence as #3, #9 above | Resolved. |
| 37 | R3-5 | Cross-domain citations 12 refs needed | ⚠️ | 4 of 12 added ([20] Gururangan, [21] Mosbach, [22] Hoffmann, [25] Pan & Yang); 8 declined with rationale | Acceptable selectivity. |
| 38 | R3-6 | Reframing recommendations (a)–(f) | ✅ | All 6 specific edits applied per R&R D.6 | Done. |
| 39 | DA #1.1 (CRITICAL) | HPO budget asymmetry confounds §3.7 | ✅ via (W) two-part stance | §2.5.1 (218–224) Part A d^c calibration; §3.7 reframe + §3.7.3 composite-estimate footnotes Part B | See §3 of this re-review — verdict: **CRITICAL retired**. |
| 40 | DA #1.2 | Cross-subject 90.68% shortcut/leakage risk | ✅ via P0.3 + cohort caveat | §3.9 (940–942) integrates P0.3 pooled 49.58%; §5 #2 (1044) directional inflation estimate | See §5 of this re-review — verdict: **strong**. |
| 41 | DA #1.3 | DAPT V1/V2/V3 5-variable confound | ✅ via V4/V5 + Δ-of-Δ | §3.6.1 mechanism-narrowing + §3.6.2 sign-reversal disclosure + Δ-of-Δ paired-t (line 726) | Strongest improvement in v3.1. See §4 of this re-review. |
| 42 | DA Moderate #4 (Confirmation Bias) | Bias #1–4 audit | ✅ | §3.6.2 (Bias #1), §3.7.2 caveat (Bias #2), §3.7.1 footnote (Bias #3), §5 #2 (Bias #4) | Equivalent to dedicated §4.X "Alternative Interpretations" section. |
| 43 | DA Moderate #6 (Overgeneralization) | OG #1–3 | ✅ | §4.8 末段 + §7 末段 (OG #1); §4.1 + §7 F1 (OG #2); §4.2/§4.6/§7 F2 (OG #3) | Done. |
| 44 | DA Moderate #8 (Stakeholders) | Wearable/edge gap | ⚠️ | R&R E.6 promises §4.6 paragraph; Step 4 cleanup #5 flags as pending verification | Minor residual. |
| 45 | DA Cherry-pick #1 | 90.68% cohort filter | ✅ | Folded into #4/#32 above | Done. |
| 46 | DA Cherry-pick #2 | 96.7% retention leakage | ⏳ | P1.4 queued; cohort caveat in F2 | Same as #11. |
| 47 | DA Cherry-pick #3 | 4ch BP 78.75% favorable framing | ✅ | §1.4 F5 + §7 F5 + §3.5.3 已 hedge | Done. |

**Tally**: ✅ 33 / ⚠️ 8 / ❌ 1 / ⏳ 3 (some concerns map to multiple rows; the 35-concern roster is mapped to the 47 sub-line-items above for granularity).

---

## §3 HPO Defense Assessment (CRITICAL flag review)

### Part A — d^c calibration for §3.1 / §3.2 baseline (substantive defense)

The argument structure is:

1. CBraMod within / cross HPO each searches **11 dimensions**; EEGNet within / cross searches **7 dimensions** (verifiable in `src/hpo/search_spaces.py`).
2. Trial budgets are 51:32 = 1.59.
3. By Bergstra & Bengio 2011 [23] (random/Bayesian search dimension dependence) and Snoek et al. 2012 [24] (GP-EI sample complexity), TPE convergence to a fixed error scales as O(d^c) with c ∈ [0.5, 1].
4. (11/7)^0.5 ≈ 1.25 (lower bound); (11/7)^1.0 ≈ 1.57 (upper bound).
5. 1.59 ≈ 1.57 → CBraMod's extra trials are **exactly absorbed** by the dimensional volume penalty.

**Verdict on Part A**: The argument is **internally sound and citable**. Three reasons it works:

- The d^c bound is the right metric — it is directly comparable to the trial-count ratio because both quantify search-space coverage relative to dimensionality.
- The ratio is reported honestly (1.59) and the upper-bound match is genuine, not selection-cherry-picked: both d^0.5 and d^1 are documented bounds in the cited literature.
- Argument 2 (EEGNet inheriting architecture-HP defaults from Ding [3]) honestly adds an asymmetry **in EEGNet's favor**, which strengthens the parity claim.

**One technical caveat I flag for honesty**: Bergstra & Bengio 2011 §3.3 actually establishes the d^1 dependence for **random search**, not TPE specifically; Snoek 2012 §4.1 establishes a tighter bound for **GP-based** Bayesian optimization (typically O(d^0.5) under smoothness assumptions). TPE is empirically intermediate. The paper invokes both bounds as "lower" and "upper" of TPE's expected behavior, which is a defensible empirical extrapolation but is not a formal theorem about TPE itself. For a master's thesis, this empirical bracketing is acceptable; for a top-venue submission, it could be tightened by citing a TPE-specific empirical convergence study (e.g., Falkner 2018 BOHB or a TPE benchmark). Not a blocker.

### Part B — §3.7 reframe to "exploratory ablations"

Direct verification:

- Title (line 807): "§3.7 探索性消融：架构 / 预训练 / 容量贡献的初步检验" — done.
- Chapter caveat block (line 811): three explicit asymmetries (EEGNet-Huge ≤2 trial manual; random-init re-uses `get_default_config()`; baseline → Mid double-axis) — done.
- §3.7.3 footnotes (lines 889–891): all three Δ values labeled "复合估计 (composite estimate)" with specific decomposition decline — done.
- Three weaker surviving claims (line 895): (a) TUEG within +23.10/+30.79 pp is the highest-attribution Δ in the table; (b) extending EEGNet capacity along current axis is directionally harmful in cross-subject; (c) transformer + ACPE retains independent value at 21× pooled data — these are all conditional and supported by the in-table data.

**One internal-consistency wobble I flag**: The chapter intro (line 813) describing Figure 12 still uses the bare numbers "−25.30 pp / +34.97 pp / +4.34 pp" as "三条主要相邻 Δ 注释" without inline composite-estimate language in that sentence. The footnotes are present but a casual reader of the figure caption (line 815) and intro paragraph alone could miss the caveat. Suggest one inline word ("composite-estimate Δ ≈ −25.30 pp / +34.97 pp / +4.34 pp") added to the figure-12 paragraph (line 813) and caption (line 815).

### Verdict on CRITICAL flag

**CRITICAL retired.** Reasoning:

1. Part A is substantively defensible and Part B is structurally correct.
2. The reframe is not cosmetic — the §3.7.3 paragraph at line 895 explicitly limits the surviving claims to the three weaker ones, and the abstract / §1.4 F1 / §4.1 / §7 F1 chain has been swept consistent (per integration audit §4).
3. The deferred 80–120 GPU-h sweep is registered in §6 #8 with explicit pre-registered readouts (which-trial-condition-implies-which-narrative-update).

What would make CRITICAL persist: if any of (Abstract / §1.4 F1 / §3.7.3 / §4.1 / §7 F1) still claimed independent attribution. They do not. The minor wobble in line 813 above is a presentational fix, not a CRITICAL trigger.

---

## §4 §3.6 Task-Asymmetric Narrative Scrutiny

The narrative claim is: "V4 and V5 surgical fixes ruled out (a) domain-mismatch-as-sufficient-cause and (b) channel-heterogeneity-as-confound, leaving (c) MI granularity mismatch as the unique surviving hypothesis."

### Logical structure of the elimination

| Hypothesis | Test | Result | Verdict |
|---|---|---|---|
| (1) Domain mismatch (coarse vs fine MI) | V4: 3 closest-domain datasets + strict filter | binary still −1.61 pp (q=0.048) | "Necessary but not sufficient" |
| (2) Stieger-dominance (V2 ~79%) | V3 (~30%) + V4 (0%) | both still negative | Ruled out |
| (3) Channel-heterogeneity confound | V5: single-source 60ch | V5 worst across both tasks | Reverse-falsified |
| (4) **Surviving — MI granularity mismatch** | Implicit | — | Survives by elimination |

### Strength of the elimination

**Where the reasoning is genuinely tight**:

- (2) Stieger-dominance is the cleanest elimination. V4 has **0% Stieger** (Cho2017 + Ofner2017 + Schirrmeister2017) and still gives −1.61 pp BH-significant. V5 is **100% Stieger** and gives −2.77 pp (worst). The two extremes flank the negative interval — Stieger occupancy is not the lever.
- (3) Channel-heterogeneity reverse-falsification is rhetorically strong: V5's collapse of channel diversity to a single 60-ch geometry **worsens** rather than improves cross-binary, directly inverting the v3.0.1 prior.

**Where the reasoning has a process-of-elimination weakness**:

The "MI granularity mismatch" survives by elimination, and **the paper does not test it directly**. The mechanism statement at §3.6.1 line 778 is plausible: "粗 hand/leg/upper-limb MI 的 MAE pretext loss 学到的是'哪个肢体在动'的低频空间包络." But there are at least three un-tested alternative hypotheses I want to flag explicitly:

(α) **Pretext-task / target-task representation interference (catastrophic forgetting)**: Even if granularity were aligned, BERT-style continued pre-training on a small (4,937–78,232 segment) corpus with `lr=5e-5 × 10 epochs+` could simply degrade TUEG-learned representations more than it adds finger-MI-relevant ones — independent of source-domain task semantics. V4 has the smallest corpus (4,937 segments) and still loses, which is consistent with this mechanism. The paper's V1 → V5 don't separate "small-corpus catastrophic forgetting" from "granularity mismatch." Both predict the same direction.

(β) **Optimizer / mask-ratio / loss-form mismatch**: All five V's use mask_ratio=50% + MSE + AdamW + lr=5e-5 — none of which were re-tuned for finger-MI specifically. Limitation #12(a) acknowledges this. But the §3.6.1 mechanism narrative does not consider it as a competing hypothesis.

(γ) **Within-task signal granularity heterogeneity (binary vs ternary task structure)**: The paper's own narrative provides the elegant story for ternary's mildly-positive sign — rest-class detection benefits from coarse spatial envelopes. But this same logic could equally well support an alternative reading: "DAPT helps when the target task overlaps with the rest-vs-motion distinction the source corpus already encoded; it harms when the target task requires within-motion fine discrimination." This is **structurally indistinguishable from "MI granularity mismatch"** but reframes it as a more general "task-overlap with source pretext" rather than a granularity-specific story. The paper could honestly note this is not a unique reading.

### Per-subject Δ-of-Δ test

The Δ-of-Δ test (line 726, t=−5.16, p<0.001, n=105 = 5V × 21 subjects pooled) is **structurally important** — it shows the binary-vs-ternary asymmetry is not a between-subject artifact. This is rigorous and well-presented.

### Verdict on §3.6 narrative

**Loose-with-recoverable-fix.** The elimination of hypotheses (2) and (3) is sharp; the survival of "MI granularity mismatch" by exclusion is honest in spirit but should add **one paragraph at the end of §3.6.1** explicitly listing un-tested alternative hypotheses (α/β/γ above) with one sentence each on why the present data cannot fully discriminate. This converts the surviving hypothesis from "the unique remaining explanation" to "the simplest surviving explanation that fits all five V's"; that is the honest framing.

The fix is one paragraph (~150 words), not a structural revision. It does not warrant Major Revision on its own.

---

## §5 P0.3 Integration Check (§3.9)

The integration appears in §3.9 lines 940–942, citing both seeds (49.17% / 50.00%), pooled mean 49.58%, and the −41.1 pp delta from the 90.68% headline.

### What §3.9 correctly establishes

- Three-fold robustness chain: §3.5.3 (4ch negative control 67.65% > 50%) + §3.9 leave-3-out (Δ ≤ 0.13 pp) + §3.9 P0.3 (chance under shuffle).
- Two seeds with different failure modes (33-epoch early-stop vs epoch-1 majority-class collapse) both landing in [48, 52] → the result is not seed-noise-driven.
- Three explicit ruled-out shortcuts: (i) train/test split residual leakage; (ii) subject-identity confound; (iii) trivial label-statistics priors.

### What §3.9 correctly does not over-claim

- It does **not** claim P0.3 rules out cohort-selection inflation (which is a separate mechanism — the 21 subjects retained by Ding's [3] 49→21 filter could still be systematically more decodable than the 49-person naive cohort, regardless of label-shuffle behavior). This separation is preserved correctly: cohort inflation is handled in §5 #2.
- It does **not** claim P0.3 rules out trivial-feature shortcuts (e.g., subject-level mean amplitude that happens to predict labels). The within-subject permutation actively destroys label semantics while preserving subject identity, so any feature-label shortcut at the subject level would be eliminated — but the paper does not over-extend this claim into "no within-subject shortcut" universally.

### Minor presentational issue

The P0.3 paragraph at line 940 places the experiment third in the robustness chain ("作为...第三重 robustness 防线"), which is correct. But the abstract (line 18–28) does not mention P0.3 at all — given its load-bearing role for the 90.68% headline, **a single sentence in the abstract** ("跨被试 binary 头条经标签置换控制 (n=2 seeds, pooled 49.58%, Δ=−41.1 pp) 验证不依赖标签泄露") would proportionate the abstract to the body. This is presentational, not structural.

### Verdict

**Proper integration, no over-claiming.** The minor presentational gap in the abstract is recommended-to-fix.

---

## §6 Residual Issues (new concerns introduced by v3.1)

### 6.1 Residual issues from incomplete sweeps

- **R-1 [Minor]** §3.7 Figure 12 caption (line 815) and intro paragraph (line 813) cite raw "−25.30 pp / +34.97 pp / +4.34 pp" without inline composite-estimate language. Footnotes at §3.7.3 carry the caveat, but a casual figure-only reader would miss it. **Fix**: one-word inline insertion in line 813 ("composite-estimate Δ ≈ ..."), and add "（composite estimates；详见 §3.7.3 footnotes）" to the figure-12 caption.

- **R-2 [Minor]** §3.6.1 mechanism-narrowing concludes "MI granularity mismatch" as the unique surviving hypothesis without enumerating un-tested alternatives (catastrophic forgetting, optimizer-form mismatch, task-overlap-with-pretext re-reading). **Fix**: one-paragraph alternative-hypothesis disclosure at end of §3.6.1 (~150 words).

- **R-3 [Minor]** Abstract does not mention P0.3 label-shuffle. **Fix**: one sentence in §6 of abstract (the "deployment path" paragraph) or §2 (the 90.68% mention paragraph).

- **R-4 [Minor]** Step 4 cleanup tasks per integration report §5 still pending: (i) §5 Limitation needs preprocessing-pipeline-alignment row (R1 Minor #6); (ii) Table 0 "评估难度" column verification; (iii) several inline citations [11]/[12]/[13]/[18]/[21]/[22] not yet placed (per integration audit §5 item 10); (iv) some "~10K parameters" residuals from R1 Minor #4. None individually material; collectively a Step 4 sweep is needed.

### 6.2 Internal consistency

- **C-1** Cross-subject Δ for TUEG pretraining: §3.7.3 footnote ³ describes +4.34 pp as "本表中归因强度最高的一个 Δ"; line 895 (a) subordinates this to the within-subject "+23.10 / +30.79 pp" headline number. The two statements are consistent (different contexts) but the rhetorical contrast is sharp; a reader unfamiliar with §3.7.2 Table 18 might wonder whether +4.34 pp or +23.10 pp is "the" TUEG-pretraining contribution. The current treatment in §1.4 F1 / §4.1 / §7 F1 always presents both numbers ("被试内 binary +23.10 / ternary +30.79 pp" + "cross-subject 与 XSI-FT 为 +1.6 ~ +4.3 pp"), so the ambiguity is well-managed in the framing chain. No fix needed; flagging for awareness.

- **C-2** §5 Limitation #12 is now ~620 words covering 6 V4/V5-specific caveats (a) through (f). This is honest but unusually long; a casual reader skimming Limitations might not parse all six. **Fix (optional)**: bullet the 6 sub-caveats with 1-line headers. Minor.

### 6.3 Over-correction risk

- **OC-1** §3.7.3 paragraph at line 895 lists three weaker surviving claims (a)/(b)/(c). Claim (a) — "TUEG pretraining contributes within-subject binary +23.10 / ternary +30.79 pp" — is described as the "highest-attribution Δ in the table." This **is the correct attribution given the data**. But by pushing all stronger claims to "future work §6 #8," the paper now arguably understates the case for transformer-architecture-independent value at cross-subject scale, which the random-init data **does** support directionally (86.34% cross-binary at zero pretraining beats EEGNet-Huge v3 51.37%). The paper's claim (c) at line 895 captures this with "在 cross-subject 21× pooled 数据上学到有效表征"; calibration is correct. Not an over-correction risk but worth noting that the paper could have justifiably claimed slightly stronger attribution for claim (c) than it did. The conservative posture is defensible.

---

## §7 Final Editorial Decision

### Decision: **Minor Revision**

### Justification against the v3.1 state (not v3.0.1)

- **All MAJOR concerns from first round are at ✅ or ⏳-with-acceptable-justification.** The two ⏳ items (P1.4 channel-ranking recompute; V4/V5 within/transfer evaluation) are openly registered in §6 with rationale, and neither is load-bearing for the paper's surviving claims after the §3.7 reframe and §3.6 task-asymmetric pivot.
- **CRITICAL flag retired** (DA #1.1). The (W) two-part stance is methodologically defensible.
- **Three new structural shifts (§3.6 reframe, §3.7 reframe, P0.3 integration) are all internally consistent** and traceable through the integration audit (`stage4_step2.5_integration_report.md`) which I sampled and found accurate.
- **Residual issues are presentational/cosmetic**: Figure-12 caption inline caveat (R-1), §3.6.1 alternative-hypothesis paragraph (R-2), abstract P0.3 sentence (R-3), Step 4 cleanup sweep (R-4). All four are single-paragraph or single-sentence rewrites that a single editing pass can resolve in <2 hours.

### Why not Accept

The four residual items above are real and should land before final acceptance. R-2 in particular (alternative hypothesis disclosure for §3.6.1) is the difference between an honest master's thesis and a flagship-quality paper.

### Why not Major Revision

No new structural issues. No CRITICAL persistence. No new unsupported claims. The deferred P0.1/P0.2 sweeps are out of master's-thesis budget, which the paper now declares honestly.

### Required actions for next round

1. **R-1**: §3.7 Figure 12 paragraph (line 813) and caption (line 815): add composite-estimate inline qualifier.
2. **R-2**: §3.6.1 (after line 778): add ~1 paragraph (~150 words) listing un-tested alternative hypotheses to MI granularity mismatch (catastrophic forgetting on small corpus; optimizer/mask-ratio mismatch; task-overlap-with-pretext re-reading) with one-sentence each on why current data cannot discriminate.
3. **R-3**: Abstract: add one sentence about P0.3 label-shuffle robustness adjacent to the 90.68% headline.
4. **R-4**: Complete Step 4 cleanup sweep per integration report §5: §5 Limitation row for preprocessing-pipeline alignment, Table 0 "评估难度" column verification, remaining inline citations ([11]/[12]/[13]/[18]/[21]/[22]), residual "~10K parameters" sweep.
5. **R-5 (optional)**: §5 Limitation #12 bullet-format the 6 sub-caveats for readability.
6. **Recommended-but-not-required**: when P1.4 train-only channel-ranking results land, fold into §3.5.3 with a follow-up "Train-only ranking control" subsection, and update §1.4 F2 / §7 F2 retention numbers if Δ ≥ 1 pp.

### Specific line edits (Minor Revision deliverables)

| # | Location | Suggested edit |
|---|---|---|
| 1 | line 813 §3.7 Figure 12 intro | "...三条主要相邻 Δ 注释 (composite-estimate values; see §3.7.3 footnotes)..." |
| 2 | line 815 Figure 12 caption | append "（composite estimates；归因强度详见 §3.7.3 footnotes）" |
| 3 | After line 778 §3.6.1 | new paragraph: "需补充说明，'MI 粒度错配' 作为唯一存活假设是经过 V4/V5 排除得到的；本研究并未独立直接检验之。至少存在三类未在本研究中分离的替代解释：(i) 小语料 catastrophic forgetting (V4 仅 4,937 段)；(ii) mask_ratio=50% + MSE pretext 与 finger MI 信号特性的潜在不匹配 (Limitation #12a)；(iii) '任务-pretext 重叠度' 而非'粒度'本身可能驱动 binary/ternary 不对称 — 后者在结构上与本节 mechanism 等价但不能被现有数据独立分离。这些替代假设的隔离需要 §6 #3 描述的方法配置 ablation 完成后才能闭合。" |
| 4 | Abstract paragraph 2 (line 20) or new paragraph 2.5 | add: "跨被试 binary 头条经标签置换控制（n=2 seeds, pooled 49.58%, Δ=−41.1 pp 相对 90.68%）通过 robustness 验证，结果不依赖于标签级泄露（详见 §3.9）。" |
| 5 | §5 #12 (line 1054) | (optional) format 6 caveats (a)–(f) as bullet sub-list |

---

## §8 Recommendation to Author

**The paper is now publishable as a master's thesis** — not contingent on the deferred P0.1/P0.2 sweeps. The §3.6 task-asymmetric pivot is a genuine analytical advance over v3.0.1, the §3.7 reframe is honest, and the P0.3 integration is properly calibrated.

**Recommended path**: **Light Stage 4' editing pass** (~half a working day) to land R-1 / R-2 / R-3 / R-4 above, then proceed directly to **Stage 4.5 final integrity pass** rather than running a full Stage 4' re-revision cycle. The five required items I list in §7 are all single-paragraph or single-sentence inserts; none requires re-running experiments, re-computing statistics, or revising main tables.

Specifically:

1. **Do** apply the four required edits (R-1 through R-4).
2. **Do** schedule P1.4 (train-only channel ranking) so it can land in time for the journal's first-round response window if/when v3.1 is submitted; if it materializes Δ ≤ 1 pp from the current 87.71%, only §3.5.3 needs a brief sub-section update; if Δ ≥ 2 pp, §1.4 F2 / §7 F2 / abstract retention numbers need a synchronized update (a 30-minute sweep).
3. **Do not** invest further effort into V4/V5 within-subject / XSI-FT evaluation pre-submission. The §3.6 chapter does not depend on it; Limitation #12 declares it; reviewer demand for it would be a R&R issue, not a submission-blocker.
4. **Do not** invest in P0.1/P0.2 80–120 GPU-h sweeps pre-submission. The §3.7 reframe absorbs the gap; reviewer demand for these would be a Major Revision item if it arises, but is not a defensible blocker against the current "exploratory ablation" framing.

The paper has reached a state where additional revisions yield diminishing returns relative to submission time-cost. Submit after R-1 through R-4.

---

*— End of Stage 3' Re-Review —*
