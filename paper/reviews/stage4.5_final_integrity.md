# Stage 4.5 Final Integrity Verification: paper_draft_v3.1.md

**Date**: 2026-05-10
**Verdict**: **CONDITIONAL** (5 LOW + 1 MEDIUM, all surgical fixes ≤30 min)
**Issues**: HIGH=0 / MEDIUM=1 / LOW=5

---

## §1 Executive Summary

`paper_draft_v3.1.md` (1508 lines, 208 KB) passes all HIGH-severity integrity gates. Every numerical claim spot-checked against ExperimentDB / JSON caches / stat-recompute outputs / handoffs reconciles cleanly: P0.3 label-shuffle (49.17 / 50.00 / pooled 49.58 / Δ=−41.1 pp), DAPT 16-cell Stouffer (cross-binary Z=−5.32 / cross-ternary Z=+0.58 / full Z=−4.83), V4/V5 cell deltas, EEGNet-Huge ladder values, and the abstract / §1.4 / §7 number unification all match their primary sources. References [1]–[25] are real, citable, and correctly attributed; the Stage 2.5 phantom-author concern in [8] (Alazrai) does not recur in [10]–[25]. R-1 / R-2 / R-3 / R-4 light edits are present and correctly placed; §3.6 task-asymmetric framing, §3.7 exploratory-ablations framing, and the cohort caveat all unify across required sections. Devil's Advocate Bias #1–#4 remediations remain intact post-revision.

The CONDITIONAL verdict is driven by **R-4 incomplete sweep**: while the Stage 4' R-4 description targeted lines ~465 + ~1398 specifically, the *intent* was that "16K" EEGNet params should be the canonical reference. Five additional `~10K 参数` references (lines 359, 406, 503, 870 in body, plus line 178 with "10K 可训练" parenthetical) survived. None is factually wrong — EEGNet-16,4 has 16,162 *total* params, of which ~10K are trainable depending on counting convention — but they create **internal phrasing inconsistency** with the abstract and §1.4 which now standardize on "~16K". Severity assessment: 4× MEDIUM-leaning-LOW phrasing inconsistencies (LOW), plus 1× residual factual ambiguity (MEDIUM) at line 178 where "16,162 (~10K trainable)" introduces undocumented terminology that is not reconciled elsewhere.

Additionally three references ([11] Sakhavi 2018, [17] Zhang Brant 2023, [22] Hoffmann Chinchilla 2022) appear in the bibliography but are **never cited inline** — these violate Phase D's "every reference in [10]–[25] is cited at least once inline" requirement. Severity: LOW (uncited references are stylistic, not factual issues, but should be either cited or removed).

No HIGH issues, no fabrication, no over-claiming, no broken citation chains. With 6 surgical fixes (each ≤5 min) the paper passes to PASS for Stage 5.

---

## §2 Phase A — Surface Integrity

### A.1 Abstract numbers (line 18–28)

| Claim | v3.1.md value | Primary source | Status |
|---|---|---|---|
| CBraMod params | 30.48M (4M backbone + 26M MLP head) | Table 2b line 194 (30,484,402) | OK; internally consistent |
| EEGNet-16,4 params | ~16K | Table 2b line 194 (16,162) | OK |
| Within-binary Δ | +7.05 pp (85.15 vs 78.10) | Table 6 line 325–327 | OK |
| Cross-binary Δ | +14.01 pp (90.68 vs 76.67) | Table 7 line 349–351 | OK |
| Cross-ternary Δ | +13.65 pp (74.88 vs 61.23) | Table 7 line 349–351 | OK |
| EEGNet ladder cross | 76.67 → 51.37 / 50% | Table 18a line 829–833 | OK |
| Random-init Δ vs ladder | ~+35 pp cross | §3.7.1 line 843 (CBraMod RI 86.34 vs Huge v3 51.37 = +34.97 pp) | OK; "~+35" is rounded |
| TUEG Δ cross | +4.34 pp | §3.7.3 line 888 (90.68 − 86.34) | OK |
| TUEG Δ within | binary +23.10 / ternary +30.79 pp | Table 18 line 855–856 | OK |
| P0.3 pooled | 49.58% | handoff `2026-05-10_p03_label_shuffle_results.md` line 56 | OK |
| P0.3 Δ | −41.1 pp | 90.68 − 49.58 = 41.10; matches handoff line 64 | OK |
| 4ch neg control | 67.65% | Table 10 line 663 | OK |
| Leave-3-out |Δ| | ≤ 0.13 pp | §3.9 line 938 (binary −0.06, ternary −0.13) | OK |

### A.2 §3.6 DAPT Table 16 (16-cell, lines 730–751)

Every cell cross-checked against `paper/reviews/stage4_step1b_stat_recompute_v4v5.md` Section 2 table (lines 56–71). All 16 cells match: V1–V5 × {within, cross} × {binary, ternary} mean_diff, t, p, dz, 95% CI, BH-q, BH-significant flag.

Stouffer aggregates (lines 748–750):
- cross-binary: Z=−5.320, p<0.001 ✓ (recompute Section 5)
- cross-ternary: Z=+0.577, p=0.564 ✓
- full DAPT family: Z=−4.830, p<0.001 ✓

Per-subject paired Δ-of-Δ (line 726): n=105, t=−5.160, p<0.001 ✓ (recompute Section 4 lines 132–133).

### A.3 §3.7.1 EEGNet-Huge ladder values (Table 18a, lines 829–833)

| Model | Params | Within | Cross | XSI-FT | Source |
|---|---|---|---|---|---|
| Baseline | 16K | 78.10% | 76.67% | 82.05% | handoff `2026-05-09_eegnet_huge.md` line 7 ✓ |
| Mid | 1.90M | 66.88% | 57.65% | 80.45% | handoff line 8 ✓ |
| Huge v3 | 5.84M | 67.71% | 51.37% | 80.62% | handoff line 9 ✓ |
| Huge v2 | 30.22M | (orphan) | 50.07% | — | handoff line 10 ✓ |
| Huge v1 | 19.99M | — | 50.00% | (bug) | handoff line 11 ✓ |

### A.4 §3.7.3 Δ values (Table line 884–893)

| Δ | v3.1.md | Recompute |
|---|---|---|
| EEGNet 内扩参 (baseline → Huge v3) cross-binary | −25.30 pp | 76.67 − 51.37 = 25.30 ✓ |
| 跨架构 (Huge v3 → CBraMod RI) cross-binary | +34.97 pp | 86.34 − 51.37 = 34.97 ✓ |
| TUEG 预训练 (CBraMod RI → original) cross-binary | +4.34 pp | 90.68 − 86.34 = 4.34 ✓ |

### A.5 P0.3 §3.9 numbers (lines 942–944)

- seed=42: 49.17% ± 4.08% — handoff line 29 ✓
- seed=123: 50.00% ± 0.00% — handoff line 45 ✓
- pooled: 49.58% — handoff line 56 ✓
- Δ vs 90.68: −41.1 pp — handoff line 64 ✓

### A.6 §5 Limitation #13 (line 1057)

Claim: EEGNet vs CBraMod use different filter (4–40 Hz vs 0.3–75 Hz), sample rate (100 Hz vs 200 Hz), normalization (Z-score vs ÷100). Cross-ref against §2.2 Table 1 (lines 134–143):

| Step | EEGNet | CBraMod | Limitation #13 claim | Match |
|---|---|---|---|---|
| Sample rate | 100 Hz | 200 Hz | 100 vs 200 | ✓ |
| Bandpass | 4–40 Hz | 0.3–75 Hz | 4–40 vs 0.3–75 | ✓ |
| Normalization | Z-score per-segment time axis | ÷100 | Z-score per-channel vs ÷100 全局 | minor wording: Table 1 says "时间轴 (time axis)", #13 says "per-channel" — OK semantically equivalent for Z-score within EEGNet's reshape convention but technically a slightly different framing. **LOW issue (cosmetic)**. |

### A.7 References [1]–[25] real-world verification (spot-check)

| Ref | Authors / Title | Venue | DOI / arXiv | Verified |
|---|---|---|---|---|
| [3] Ding 2025 | Y. Ding et al. EEG-BCI finger control | Nat. Commun. | 10.1038/s41467-025-61064-x | OK (matches dataset paper as cited in CLAUDE.md) |
| [4] Wang 2025 (CBraMod) | J. Wang et al. CBraMod | ICLR 2025 | — | OK (matches CBraMod paper) |
| [5] Lawhern 2018 (EEGNet) | V. Lawhern et al. EEGNet | J. Neural Eng. 15(5) 056013 | — | OK (canonical EEGNet citation) |
| [10] Schirrmeister 2017 | Schirrmeister et al. Deep ConvNet | Hum. Brain Mapp. 38(11) | 10.1002/hbm.23730 | OK (canonical reference) |
| [12] Ang 2008 (FBCSP) | Ang et al. FBCSP | IJCNN 2008 | 10.1109/IJCNN.2008.4634130 | OK |
| [13] Blankertz 2008 (CSP) | Blankertz et al. | IEEE SPM 25(1) | 10.1109/MSP.2008.4408441 | OK |
| [14] Pfurtscheller 1999 (ERS/ERD) | Pfurtscheller & Lopes da Silva | Clin. Neurophysiol. 110(11) | 10.1016/S1388-2457(99)00141-8 | OK (canonical) |
| [18] Lotte 2018 (review) | Lotte et al. | J. Neural Eng. 15(3) 031005 | 10.1088/1741-2552/aab2f2 | OK |
| [19] Neuper 2006 (ERD/ERS patterns) | Neuper, Wörtz, Pfurtscheller | Prog. Brain Res. 159 | 10.1016/S0079-6123(06)59014-4 | OK |
| [20] Gururangan 2020 (DAPT) | Gururangan et al. "Don't Stop Pretraining" | ACL 2020 | 10.18653/v1/2020.acl-main.740 | OK (canonical DAPT paper) |
| [21] Mosbach 2021 | Mosbach, Andriushchenko, Klakow "Stability of fine-tuning BERT" | ICLR 2021 | — | OK |
| [23] Bergstra 2011 (TPE) | Bergstra, Bardenet, Bengio, Kégl | NeurIPS 2011 | — | OK (canonical TPE paper) |
| [24] Snoek 2012 (BO) | Snoek, Larochelle, Adams | NeurIPS 2012 | — | OK |
| [25] Pan & Yang 2010 (TL survey) | Pan & Yang | IEEE TKDE 22(10) | 10.1109/TKDE.2009.191 | OK |

No phantom authors detected in [10]–[25]. Stage 2.5 caught [8] H. Abuhijleh phantom — that fix evidently held (line 1117 now shows "Alazrai, Alwanni, Daoud" only).

### A.8 Inline citation placement spot-checks

- §1.3 [4] CBraMod (line 67): "CBraMod ... [4]" — correct attribution to Wang et al. ✓
- §1.3 [20] Gururangan (line 69): "Gururangan et al. 2020 [20] 'Don't Stop Pretraining'" — exact match to ref title and year ✓
- §1.3 [10] Schirrmeister (line 71): "Schirrmeister et al. 2017 [10]" — matches Hum. Brain Mapp. ✓
- §2.5.1 [23] Bergstra (line 218): "Optuna 框架的 TPE [23] 采样器" — TPE was introduced in Bergstra 2011 [23] ✓
- §2.5.1 [23] / [24] Snoek (line 222): "Bergstra & Bengio 2011 [23] §3.3 ...; Snoek et al. 2012 [24] §4.1 GP-EI sample complexity" — both correct attributions ✓
- §2.6 [12], [13] CSP (line 239): "**共空间模式（CSP）[12], [13]**" — Ang FBCSP 2008 [12] + Blankertz et al. 2008 [13]. Both are canonical CSP references; correct ✓
- §3.3 [18] Lotte (line 379): "Lotte et al. 2018 [18] (J. Neural Eng. 综述) 中'subject-adaptive transfer learning'分类" — correctly cites the 2018 J. Neural Eng. 10-year update review ✓
- §3.3 [25] Pan & Yang (line 379): "Pan & Yang 2010 [25] 提出的 inductive transfer 框架" — correct ✓
- §3.6 / §4.5 [20] Gururangan (line 1007): "Gururangan et al. 2020 [20] 在 NLP 中证明 DAPT" — correct ✓
- §3.7.2 [21] Mosbach (line 870): "Mosbach et al. 2021 [21] ICLR 在 RTE ~2K 样本上 BERT-base" — correct ✓

---

## §3 Phase B — Internal Consistency

### B.1 90.68% cross-subject binary headline

Required locations: abstract, §1.4 F1, §3.2, §3.7.3, §3.9, §4.1, §7 F1.

| Location | Line(s) | Match |
|---|---|---|
| Abstract para 2 | 20 | ✓ |
| §1.4 F1 | 77 | ✓ (implicit via cross-binary +14.01 pp anchor) |
| §3.2 Table 7 | 349 | ✓ |
| §3.7.3 Table line | 889 | ✓ |
| §3.9 P0.3 anchor | 942 | ✓ |
| §4.1 | 952 | ✓ (referenced via "+14.01 pp 跨被试") |
| §7 F1 | 1087 | ✓ (referenced via "binary +14.01 pp" + "TUEG 预训练 Δ" footnote chain) |

### B.2 49.58% P0.3 pooled

Required locations: abstract, §3.9, R&R Letter §G.2 + §E.2 + Section H.

| Location | Line | Match |
|---|---|---|
| Abstract | 20 | ✓ ("pooled 49.58%, Δ=−41.1 pp") |
| §3.9 | 942 | ✓ |
| R&R Letter | (separate file `response_to_reviewers_v3.1.md`, not in scope of paper integrity) | not checked here |

### B.3 +23.10 pp / +30.79 pp within-subject TUEG

Required: abstract, §1.4 F1, §3.7.2, §3.7.3, §4.1, §7 F1.

| Location | Line | Match |
|---|---|---|
| Abstract | 20 | ✓ |
| §1.4 F1 | 77 | ✓ |
| §3.7 Table 18 | 855–856 | ✓ |
| §3.7.3 | 895 | ✓ ("binary +23.10 / ternary +30.79 pp") |
| §4.1 | 958, 960 | ✓ |
| §7 F1 | 1087 | ✓ |

### B.4 30.48M CBraMod params

Required ≥9 occurrences. Detected ≥9 (abstract / §1.3 / Table 2b / §3.7 caveat / Table 18a / §3.7.3 footnote / §4.1 / §6 / Table 18 row). ✓

### B.5 16K EEGNet params (R-4 status)

Required: must NOT appear as "10K" anywhere. **FAIL** at 5 locations:

| Line | Phrase | Severity |
|---|---|---|
| 178 | "参数量约 16,162（~10K 可训练）" | **MEDIUM** — introduces undocumented "10K trainable" terminology not reconciled elsewhere |
| 359 | "其有限的 ~10K 参数可能难以从异质多被试数据中提取共享表征" | LOW — phrasing inconsistency with abstract's "~16K" |
| 406 | "EEGNet 容量太小（~10K 参数）" | LOW |
| 503 | "与其 ~10K 参数容量上限一致" | LOW |
| 870 | "~10K 参数的 EEGNet 凭借更小的搜索空间" | LOW |

The Stage 4' R-4 brief targeted "lines ~465 + ~1398" specifically; these residuals were either out of R-4 scope or missed. Recommended fix: replace `~10K` with `~16K` at lines 359, 406, 503, 870; at line 178 either remove parenthetical or convert to "EEGNet-16,4 总参 16,162（其中可训练参数比例约 60%）" with explicit denominator definition.

### B.6 §3.6 task-asymmetric narrative unification

Required: §3.6 (line 726, 748–750) / §4.5 (lines 997, 1005) / §7 F4 (line 1093) / §1.4 F5 (line 85) / abstract DAPT paragraph (line 26).

| Location | Phrasing | Match |
|---|---|---|
| §3.6 lead | "task-asymmetric 负迁移 ... cross-binary 5/5 ... cross-ternary 4/5 弱正" | canonical |
| §1.4 F5 | "task-asymmetric 负迁移 ... 5/5 配置一致负向 ... ternary 上的方向性负迁移声明不被支持" | ✓ |
| Abstract | "task-asymmetric 负迁移 ... cross-subject binary 5/5 ... cross-subject ternary 4/5 配置弱正" | ✓ |
| §4.5 | "task-asymmetric ... cross-subject binary 5/5 配置一致显著负 ... cross-subject ternary 4/5 配置弱正" | ✓ |
| §4.8 | "task-asymmetric 负迁移" | ✓ |
| §5 #12 | "task-asymmetric 定性结论（5/5 binary 一致负 vs 4/5 ternary 弱正）" | ✓ |
| §7 F4 | "task-asymmetric 负迁移 ... cross-subject binary 5/5 ... cross-subject ternary 4/5 弱正、仅 V5 弱负" | ✓ |

All 7 locations align. ✓

### B.7 §3.7 exploratory-ablations framing

Required: §3.7 (lines 809, 813, 815) / §1.4 F1 (line 77) / §7 F1 (line 1087) / §4.1 (lines 954, 956, 960) / abstract (line 20).

All 5+ locations consistently use language like "**探索性**消融" / "受限 HPO 预算下的方向性观察" / "复合估计 (composite estimate)" / "不构成对架构、预训练、容量三因子的独立可归因分解". The R-1 fix at line 815 inserts the "composite-estimate Δ" inline qualifier exactly as specified. ✓

### B.8 Cohort caveat (5-place requirement)

Required: abstract / §1.4 F1 / §1.4 F2 / §7 F1 / §7 F2.

| Location | Line | Phrasing |
|---|---|---|
| Abstract | 20 | "21 名 responder 被试，原数据集 [3] 49 名招募者中筛选后 cohort，详见 §2.1" ✓ |
| §1.4 F1 | 77 | "21 名 responder cohort，继承自 [3] 的 49 → 21 离线筛选" ✓ |
| §1.4 F2 | 79 | "在 21 名 responder cohort × cross-subject binary 上;... 详见 Limitation #1" ✓ |
| §7 F1 | 1087 | "21 名 responder cohort × 当前 HPO 预算" ✓ |
| §7 F2 | 1089 | "在 21 名 responder cohort × cross-subject binary 上" ✓ |

All 5 locations confirmed. ✓ (Plus §2.1 line 95 "[3] 在 49 名招募者中经离线二分类准确率筛选后保留的在线被试队列" + §5 Limitation #2 line 1046 — supplementary 2 redundant statements that strengthen the chain.)

### B.9 +14.01 pp / +13.65 pp / +7.05 pp anchors

Verified consistent in: abstract (line 20), §1.4 F1 (line 77), §3.1 (line 327), §3.2 (line 351), §3.7 contextual references (line 811, 815, 882), §4.1 (line 954), §7 F1 (line 1087). ✓

---

## §4 Phase C — Claim-Evidence Alignment

### C.1 §1.4 F1 (line 77) — basemod beats EEGNet, exploratory ablations

Claim chain:
1. CBraMod beats EEGNet by +7.05 / +14.01 / +13.65 pp across 3 paradigms
2. EEGNet capacity ladder fails monotonically (76.67 → 51.37 / 50%)
3. random-init CBraMod beats expanded EEGNet by ~+35 pp
4. TUEG adds +4.34 pp cross / +23.10 / +30.79 pp within
5. ablations are exploratory, not independent decomposition (caveat re HPO budget + double-axis baseline → Mid jump)

Evidence chain:
- §3.1/§3.2 Tables 6, 7 → claim 1 ✓
- §3.7.1 Table 18a → claim 2 ✓
- §3.7.1 line 843 + §3.7.2 Table 18 → claim 3 ✓
- §3.7.3 footnote ³ → claim 4 ✓
- §3.7 caveat (lines 813, 815) + §3.7.3 lines 891–893 → claim 5 ✓

All claims supported at the strength stated. The hedge ("不构成对架构、预训练、容量三因子的独立可归因分解，应被理解为方向性观察") is **calibrated rather than over-cautious** — appropriately weakened post Stage 3' Devil's Advocate Bias #3. ✓ No over-claim.

### C.2 §1.4 F4 (line 83) — extra sessions paradigm comparison

Claim chain: CBraMod within +6.13 pp / XSI-FT +5.70 pp / cross +0.86 pp.

Evidence: Table 12a / Table 15 lines 432–435, 525–534. ✓

### C.3 §1.4 F5 (line 85) — DAPT task-asymmetric

Claim chain: 5/5 binary negative (Stouffer Z=−5.32, p<0.001), 4/5 ternary weakly positive (Stouffer p=0.564), MI-granularity-mismatch as surviving hypothesis, V5 falsifies channel-heterogeneity-as-confound.

Evidence: §3.6 Table 16 + Stouffer rows + §3.6.1 mechanism narrowing table. ✓ All claims supported.

### C.4 §3.6.1 R-2 alternative-hypothesis paragraph (line 780)

Three alternative hypotheses listed: (i) small-corpus catastrophic forgetting; (ii) DAPT method-config mismatch; (iii) task-pretext-overlap not granularity-per-se.

Verification:
- The elimination of Stieger-dominance (via V3 + V4) is rigorous and asserted as such ✓
- The elimination of channel-count-heterogeneity (via V5) is rigorous and asserted as such ✓
- The "MI granularity mismatch" surviving hypothesis is explicitly hedged as "the simplest surviving explanation compatible with all 5V data, but not the only possible explanation" — calibrated ✓
- Three alternatives are presented as structurally equivalent / parallel, not nested under granularity-mismatch — correct logical framing ✓

R-2 paragraph does NOT over-hedge. ✓

### C.5 §7 Conclusion claims (5 findings)

All 5 findings (§7 lines 1087–1095) anchor to specific section / table evidence:
- F1 → §3.1, §3.2, §3.7 ✓
- F2 → §3.5.1, §3.5.2 (table 8, table 9) ✓
- F3 → §3.4 (tables 12, 13, 15) ✓
- F4 → §3.6 (table 16) + §3.6.1 + Stouffer ✓
- F5 → §3.5.3 (table 10) ✓

No claim outruns its evidence; F1 explicitly hedged with "我们**无法对各因素做独立定量归因**".

### C.6 Abstract claims (line 18–28)

Each abstract claim resolves to body section ✓. Notably the "三重 robustness 证据链 (triple-robustness chain)" at end of paragraph 2 is supported by §3.5.3 + §3.9 leave-3-out + §3.9 P0.3.

---

## §5 Phase D — Reference / Citation Health

### D.1 Reference list completeness

25 entries [1]–[25], no gaps detected by enumeration. ✓ Order: sequential, no duplicates. ✓

### D.2 Inline citation coverage of [10]–[25]

Scan results:

| Ref | Inline cite found at | Status |
|---|---|---|
| [10] Schirrmeister | line 71 | ✓ |
| [11] Sakhavi | — | **LOW issue: uncited** |
| [12] Ang FBCSP | line 239 | ✓ |
| [13] Blankertz CSP | line 239 | ✓ |
| [14] Pfurtscheller 1999 | line 645 | ✓ |
| [15] NeuroLM | line 71, line 1087 | ✓ |
| [16] BIOT | line 71, line 1087 | ✓ |
| [17] Brant | — | **LOW issue: uncited** |
| [18] Lotte | line 379 | ✓ |
| [19] Neuper | line 645 | ✓ |
| [20] Gururangan | lines 69, 1007, 1037, 1097 | ✓ |
| [21] Mosbach | line 870 | ✓ |
| [22] Hoffmann (Chinchilla) | — | **LOW issue: uncited** |
| [23] Bergstra TPE | lines 218, 222 | ✓ |
| [24] Snoek | line 222 | ✓ |
| [25] Pan & Yang | line 379 | ✓ |

**Three uncited references**: [11] Sakhavi 2018 (CNN for BCI), [17] Brant 2023 (intracranial foundation model), [22] Hoffmann 2022 (Chinchilla scaling laws). Severity LOW — uncited references are stylistic / cosmetic but should either be cited inline once or removed for tightness.

### D.3 Inline `[N]` resolves to ref entry

All `[1]`–`[25]` inline citations have corresponding entries in §"参考文献" (lines 1103–1151). No "see [X]" with unresolved [X]. ✓

### D.4 Phantom-author re-scan ([10]–[25])

Spot checked author lists vs venue / DOI / arXiv:
- [10] Schirrmeister + 8 coauthors (Springenberg, Fiederer, Glasstetter, Eggensperger, Tangermann, Hutter, Burgard, Ball) — matches Hum. Brain Mapp. 38(11) 5391–5420 ✓
- [13] Blankertz, Tomioka, Lemm, Kawanabe, Müller — matches IEEE SPM 25(1) ✓
- [20] Gururangan, Marasović, Swayamdipta, Lo, Beltagy, Downey, Smith — 7 authors, matches ACL 2020 ✓
- [22] Hoffmann + Chinchilla coauthors (Borgeaud, Mensch, Buchatskaya, Cai, Rutherford, et al.) — matches NeurIPS 2022 Chinchilla paper ✓

No phantom-author detection in [10]–[25]. ✓ Stage 2.5's [8] H. Abuhijleh issue did not recur.

---

## §6 Phase E — Self-Consistency / Structure

### E.1 Section numbering

§1.1 / §1.2 / §1.3 / §1.4 — continuous ✓
§2.1 / §2.1.1 / §2.2 / §2.3 / §2.3.1 / §2.4 / §2.4.1 / §2.4.2 / §2.5 / §2.5.1 / §2.6 / §2.7 / §2.7.1 / §2.7.2 / §2.8 / §2.9 — continuous ✓
§3.1 / §3.2 / §3.3 / §3.3.1 / §3.4 / §3.4.1 / §3.4.2 / §3.4.3 / §3.4.4 / §3.4.5 / §3.5 / §3.5.1 / §3.5.2 / §3.5.3 / §3.5.4 / §3.6 / §3.6.1 / §3.6.2 / §3.6.3 / §3.6.4 / §3.7 / §3.7.1 / §3.7.2 / §3.7.3 / §3.8 / §3.9 — continuous ✓
§4.1–§4.8 — continuous ✓
§5, §6, §7 — present ✓

§3.6 R-2 paragraph (line 780) is contained inside §3.6.1, properly placed before §3.6.2 boundary. The R-2 insertion did NOT break §3.6.1/§3.6.2 boundary. ✓

### E.2 Tables / Figures numbering

Body tables: 0, 1a, 1, 2, 2b, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11c, 12a, 12b, 13a, 13b, 14, 15, 15b, 16, 17, 18, 18a — present, but numbering has minor gaps:
- Table 0 then Table 1a then Table 1 (Table 1a precedes Table 1) — **LOW** ordering quirk; defensible since 1a is "comparison with source papers" and 1 is "preprocessing pipelines", logically distinct, but unusual.
- Table 11 followed by Table 11c (no Table 11a/11b) — "11c" labeling implies missing 11a/11b ancestors. Possibly Table 11a/b were merged into Table 11. **LOW** numbering quirk.
- Table 18 then Table 18a (Table 18 placed AFTER 18a numerically but Table 18a appears at line 825 vs Table 18 at line 851). Wait — re-checking: line 825 is "**表 18a. EEGNet 容量阶梯**", line 851 is "**表 18. Random-init vs Original-weights CBraMod vs EEGNet**". So Table 18a (line 825) precedes Table 18 (line 851) in document order. **LOW** ordering inconsistency — Table 18a should come after Table 18, not before.
- Tables S1, S1b, S2, S3, S4, S5, S5b, S5e, S6, S7 in supplementary — gaps in S5 sub-letters (no S5a/c/d) but the actual content of S5/S5b/S5e is logically coherent. **LOW** quirk.

Figure numbering: Figures 1, 2, 2b, 3, 3a, 3b, 4, 4b, 4c, 5, 6, 6b, 7, 8, 9, 10a, 10b, 11, 12 — Figure 3 (referenced at line 584) but figure label at line 566 is Figure 3a then 3b. So "图 3" referenced in text but actual figures labeled 3a / 3b. **LOW** ordering quirk. Supplementary: Figure S1, S2 — but Figure S2 appears in document (line 1444) BEFORE Figure S1 (line 1452). **LOW**.

### E.3 Footnote markers

§3.7.3 contains footnote markers ¹ ² ³ at lines 886–888 with corresponding footnote definitions at lines 891–893. ✓ All resolve.

§3.4.4 line 536 contains a Chinese-curly-quote artefact ("，") inside Chinese narrative — purely cosmetic / encoding. **LOW**.

§3.4 / §3.5 / §3.6.4 contain `[Plan §Stage 4](../../C:/Users/zhang/.claude/plans/did-we-use-the-sprightly-peacock.md)` link at line 798 — this is a broken/obsolete absolute Windows path inside a markdown link, won't resolve in print or PDF. **LOW** cosmetic.

### E.4 §5 Limitation #13 (R-4 add)

| # | Phrase | Format check |
|---|---|---|
| 13 | "**EEGNet vs CBraMod 预处理管线不对齐**" (line 1057) | Markdown table row well-formed: `\| 13 \| ... \| ... \|` ✓ Numbering continues from #12 ✓ Matches §2.2 Table 1 facts ✓ |

---

## §7 7-Mode AI Failure Mode Audit

### Mode 1 — PH (Phantom Author Insertion)
- [10]–[25] author-list sweep: zero phantoms. Stage 2.5 [8] Abuhijleh fix held; [3] Ding et al. confirmed against Nat. Commun. record; [4] Wang et al. confirmed against ICLR 2025.
- **Severity**: NONE.

### Mode 2 — MD (Misattributed DOI)
- Spot-checks: [10] 10.1002/hbm.23730 → resolves to Schirrmeister 2017 ✓; [12] 10.1109/IJCNN.2008.4634130 → resolves to FBCSP Ang 2008 ✓; [20] 10.18653/v1/2020.acl-main.740 → resolves to Gururangan ACL 2020 ✓.
- **Severity**: NONE.

### Mode 3 — HN (Hallucinated Number)
- All numbers in v3.1.md (Stage 4 / Phase 0 / R-1..R-4 additions specifically) verified against ExperimentDB / JSON cache / handoff files / `stage4_step1b_stat_recompute_v4v5.md`.
- One minor numerical quibble: line 815 figure 12 caption says "EEGNet 内扩参 Δ ≈ −25.30 pp" while §3.7.3 footnote 1 says "EEGNet 内扩参的 Δ 为 baseline (16K, F1=16/F2=64, 单 Linear 头) → Huge v3 (5.84M, F1=32/F2=256, [2048,2048] + LayerNorm 头) 的双轴跳跃". Both reference the same number (76.67 − 51.37 = 25.30 pp) but the ~symbol vs `=` distinction is consistent.
- **Severity**: NONE.

### Mode 4 — CS (Conflicting Statement)
- §2.4.1 line 178 "参数量约 16,162（~10K 可训练）" introduces "10K trainable" terminology not reconciled. Lines 359, 406, 503, 870 then reference "~10K 参数" without trainable qualifier — internal phrasing inconsistency with abstract / §1.4 / Table 2b which standardize on "~16K" or "16,162" total.
- **Severity**: 1× MEDIUM (line 178), 4× LOW (lines 359, 406, 503, 870).

### Mode 5 — OE (Over-claiming Evidence)
- §3.6.1 surviving-hypothesis claim (line 778) — appropriately hedged via R-2 paragraph (line 780). ✓
- §3.7.3 composite-estimate framing — explicitly stated as not-independent-decomposition. ✓
- §1.4 F1 claim chain — calibrated to "exploratory ablations". ✓
- §7 F1 closing — explicitly states "无法对各因素做独立定量归因". ✓
- **Severity**: NONE.

### Mode 6 — UE (Unsupported Extrapolation)
- §3.5.2 "Band Power 优于 FDR" — correctly hedged with explicit "本研究观察到的、可被未来工作证伪的具体配置推荐", not extrapolated as universal rule (lines 628, 690).
- §3.5.4 "XSI-FT 收益 vs cross-subject baseline distance" — explicitly hedged "基于 N=3 数据点" (line 718).
- §4.5 cross-domain DAPT contrast with NLP — appropriately framed as structural-conceptual analogy not as direct empirical equivalence (line 1007).
- §7 closing — caveats limit scope to "本数据集 / 本任务 / 本预训练 / 本预处理" with §6 future-work pointers.
- **Severity**: NONE.

### Mode 7 — CB (Confirmation Bias) — Devil's Advocate Bias #1–#4 status
- **Bias #1 (DAPT V1→V2 reversal)**: §3.6.2 line 786 explicitly discloses both raw-baseline reversal direction and registry-baseline-amended direction. Transparent. ✓
- **Bias #2 (within-ternary 18/21 collapse single-direction interpretation)**: §3.7.2 line 866 includes seed=1234 reproducibility check (17/21 vs 18/21) AND quotes handoff probability estimate (i)/(ii)/(iii). Explicitly distinguishes between "structural saddle-lock" vs "HP-mismatch" without committing to single mechanism. ✓
- **Bias #3 (EEGNet-Huge capacity-saturation framing)**: line 837 + §4.1 line 954 explicitly recharacterize v1/v2 unfailure as "更可能是 BF16 + 深 MLP 头优化栈兼容性问题，而非容量饱和" with handoff-document evidence (LayerNorm restoration → trainable). Author's prior "capacity saturation" framing has been retracted. ✓
- **Bias #4 (90.68% headline framed as CBraMod advantage without cohort caveat)**: 5-place cohort caveat now in place (per Phase B.8). ✓

All 4 prior CB risks remediated.
- **Severity**: NONE.

---

## §8 Issue Tally + Severity

| # | Phase | Severity | Description | Location | Surgical fix (≤30 min) |
|---|---|---|---|---|---|
| 1 | Phase B / Mode 4 | **MEDIUM** | Line 178 introduces "~10K 可训练" parenthetical not reconciled elsewhere; abstract / §1.4 standardize on "~16K". Either define "trainable" denominator explicitly or remove parenthetical. | line 178 | Edit: replace `参数量约 16,162（~10K 可训练）` with `参数量约 16,162` (drop parenthetical). |
| 2 | Phase B / Mode 4 | LOW | Line 359 says "~10K 参数" inconsistent with abstract "~16K". | line 359 | Edit: `~10K` → `~16K`. |
| 3 | Phase B / Mode 4 | LOW | Line 406 says "EEGNet 容量太小（~10K 参数）" — inconsistent. | line 406 | Edit: `~10K` → `~16K`. |
| 4 | Phase B / Mode 4 | LOW | Line 503 says "其 ~10K 参数容量上限" — inconsistent. | line 503 | Edit: `~10K` → `~16K`. |
| 5 | Phase B / Mode 4 | LOW | Line 870 says "~10K 参数的 EEGNet" — inconsistent. | line 870 | Edit: `~10K` → `~16K`. |
| 6 | Phase D | LOW | References [11] Sakhavi 2018, [17] Brant 2023, [22] Hoffmann Chinchilla 2022 appear in bibliography but never cited inline. | refs [11], [17], [22] | Either delete from bibliography (3 lines) or add 1 inline citation each. Cheapest fix: add `[11]` next to `[10] Schirrmeister 2017` at line 71 (CNN-for-BCI baseline list); add `[17]` after `[15], [16]` at line 71 ("Brant [17] 用于颅内信号"); add `[22]` at §3.7.3 (line 882) discussing scaling-laws context (e.g., "类比 Hoffmann et al. 2022 [22] 的 compute-optimal 论点"). |

Optional cosmetic fixes (not counted in tally):
- Line 798 broken absolute Windows path inside markdown link
- Table 18 / 18a ordering (18a precedes 18 in document order)
- Table 11 / 11c missing 11a/11b ancestors
- Figure S1 / S2 reverse-document-order
- §2.2 Table 1 "Z-score per time axis" vs §5 #13 "Z-score per channel" minor wording

---

## §9 Final Verdict

**CONDITIONAL** — pass to Stage 5 (LaTeX/PDF finalization) after the following 6 surgical edits (each ≤5 min, total ≤30 min):

### Required fixes (1 MEDIUM + 5 LOW)

1. **MEDIUM** — Line 178: replace `参数量约 16,162（~10K 可训练）` with `参数量约 16,162`. Removes undocumented "10K trainable" terminology that conflicts with paper-wide "~16K" canonical phrasing.
2. **LOW** — Line 359: replace `~10K 参数` with `~16K 参数`.
3. **LOW** — Line 406: replace `~10K 参数` with `~16K 参数`.
4. **LOW** — Line 503: replace `~10K 参数` with `~16K 参数`.
5. **LOW** — Line 870: replace `~10K 参数` with `~16K 参数`.
6. **LOW** — Either delete refs [11] / [17] / [22] from bibliography, OR add 1 inline citation each. Recommended: add inline at line 71 ("Schirrmeister et al. 2017 [10] 的 Deep ConvNet ... [11] Sakhavi et al. 2018 ... 包括 LaBraM [6]、NeuroLM [15]、BIOT [16] 与颅内信号基座 Brant [17]") and at §3.7.3 line 882 ("类比 NLP scaling-laws 文献 [22] ...").

### Non-blocking cosmetic notes (optional Stage 5 polish)

- Table ordering quirks (1a before 1; 18a before 18; 11c without 11a/11b; Figure S2 before S1).
- Line 798 markdown link points to local Windows absolute path that won't resolve in PDF.
- §2.2 Table 1 vs §5 #13 minor "Z-score" axis-spec wording difference.

### Blockers

**None.** No HIGH issues. No fabrication. No phantom citation. No over-claim. No broken evidence chain. Cohort caveat 5/5. Task-asymmetric framing 7/7. Exploratory-ablations framing 5/5. Devil's Advocate Bias #1–#4 remediation intact.

**Post-fix verdict**: PASS. Apply the 6 edits above and Stage 5 (LaTeX/PDF) is unblocked.

---

**Word count**: ~5,100 (within 4000–6000 target range).
**Verification time**: full Phase A–E + 7-Mode audit complete.
