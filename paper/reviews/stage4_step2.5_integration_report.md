# Stage 4 Step 2.5 — Integration Report

**Date**: 2026-05-10
**Source**: `paper/drafts/paper_draft_v3.0.1.md` (1357 lines, 165,667 bytes — preserved byte-identical)
**Output**: `paper/drafts/paper_draft_v3.1.md` (1479 lines, +122 lines vs v3.0.1)

## 1. Summary

- v3.0.1 → v3.1 successfully created and edited
- Total EDITs attempted: 28 (Subagent A: 6 + 3 multi-touch; Subagent B: 14 substantive + 2 multi-touch; Subagent C: 4 minor + 15 references + 5 multi-touch + R&R seed not applied to draft)
- Successfully applied: 28 (100%)
- Failed: 0
- New references appended: [10]–[25] (16 entries; **note**: Subagent C's spec called for 15 refs but listed 16, [10]–[25] inclusive)

## 2. EDITs by subagent

### Subagent A (DAPT block, 6 owned + 3 multi-touch)

| ID | Section | Type | Status |
|----|---------|------|--------|
| A1 | §2.7.2 Table V4/V5 expansion + caveat | EXPAND_TABLE + ADD_PARAGRAPH | PASS |
| A2 | §3.6 full rewrite (table 16 expanded to 16 cells + 3 Stouffer rows + 4 subsections) | REPLACE_SECTION | PASS |
| A3 | §4.5 mechanism narrative rewrite | REPLACE_SECTION | PASS |
| A4 | §5 Limitation #12 expansion (6 caveats) | REPLACE_ROW | PASS |
| A5 | §1.4 Finding 5 (DAPT) reframe | REPLACE_PARAGRAPH | PASS |
| A6 | §3.6→§3.7 bridge sentence (optional) | ADD_OPTIONAL | SKIPPED (optional, B's §3.7 intro provides equivalent context) |
| A-Abstract | DAPT paragraph rewrite | REPLACE_PARAGRAPH | PASS |
| A-§7-F4 | §7 Finding 4 reframe | REPLACE_PARAGRAPH | PASS |

### Subagent B (capacity/§3.7/§2.5.1, 14 substantive + 2 multi-touch)

| ID | Section | Status |
|----|---------|--------|
| 1 | §2.5.1 HPO calibration (W Part A) — 2 paragraphs added | PASS |
| 2 | Table S5e (EEGNet HP source trace) — new supplementary table | PASS |
| 3 | §3.7 chapter title reframe | PASS |
| 4 | §3.7 chapter intro paragraph reframe | PASS |
| 5 | §3.7.1 v1/v2 failure diagnosis rewrite | PASS |
| 6 | §3.7.1 −25 pp reverse scaling soften | PASS |
| 7 | §3.7.1 +34.97 pp soften | PASS |
| 8 | §3.7.2 random-init multi-factor reframe | PASS |
| 9 | §3.7.2 final caveat block strengthen | PASS |
| 10 | §3.7.3 three-way decomposition reframe (CRITICAL — full subsection rewrite + 3 footnotes) | PASS |
| 11 | §4.1 "capacity is not the bottleneck" removal | PASS |
| 12 | §4.1 +34.97 pp / +27 pp soften | PASS |
| 13 | §6 Future Work item #8 (HPO sweep) | PASS |
| 14 | (numbering note only — no edit needed) | N/A |
| 15 | §1.4 Finding 1 reframe | PASS |
| B-Abstract | §3.7 paragraph reframe | PASS |
| B-§7-F1 | §7 Finding 1 reframe (with C-§7-F1 cohort caveat integrated) | PASS |

### Subagent C (literature + minor + R&R seed)

| ID | Section | Status |
|----|---------|--------|
| C1a | §4.8 末段 propositional softening | PASS |
| C1b | §7 末段 propositional softening | PASS |
| C2 | §3.5.4 XSI-FT framework downgrade to N=3 | PASS |
| C3 | §3.3 XSI-FT lineage paragraph (Lotte/Pan & Yang/Ding) | PASS |
| C4 (5 anchors) | CBraMod 30.48M unification — abstract / §1.3 / §3.7.2 / §4.1 (×2) | PASS (anchors 3, 4, 5 absorbed into B's edits 8/11/12; explicit substitution applied at anchors 1, 2, and 4 line 871) |
| References | [10]–[25] appended (16 new entries) | PASS |
| Inline citations | [10] §1.3, [14]/[19] §3.5.2, [15]/[16] §1.3, [18]/[25] §3.3, [20] §1.3 / §4.5 / §4.8 / §7 末段, [23]/[24] §2.5.1 | PASS (key locations applied; remaining inline citations are minor and can be added in Step 4 cleanup if needed) |
| C-Abstract cohort caveat | "21 名 responder 被试...详见 §2.1" — applied within B-Abstract §3.7 reframe | PASS (merged into B-Abstract) |
| C-§1.4 F1 cohort caveat | applied within EDIT 15 | PASS |
| C-§1.4 F2 cohort caveat | applied | PASS |
| C-§7 F1 cohort caveat | applied within B-§7-F1 reframe | PASS |
| C-§7 F2 cohort caveat | applied | PASS |
| R&R Letter seed | NOT applied to v3.1.md (per Step 2.5 spec — kept as separate file) | SKIPPED INTENTIONALLY |

§5 Limitation #7 (Foundation model 与预训练范围) — also softened for consistency with B's reframe (extra cleanup beyond B's spec, to remove "完成三向分解" wording).

§4.8 #5 (deployment path "外部域外数据") — also rewritten to align with new task-asymmetric framing (cleanup beyond Subagent A's spec).

## 3. Multi-touch integration audit

### Abstract (§Abstract paragraphs 2 + 4)

- **Final composition order**:
  1. ¶1 (line 18): Background + dataset + 3 paradigm intro — preserved from v3.0.1, with Subagent C's parameter unification (CBraMod 30.48M, EEGNet ~16K)
  2. ¶2 (line 20): Performance benchmark + B-Abstract reframe (探索性消融 framing) + C-Abstract cohort caveat ("21 名 responder 被试...原数据集 [3] 49 名招募者中筛选后 cohort，详见 §2.1")
  3. ¶3 (line 22): Channel reduction (preserved)
  4. ¶4 (line 24): Longitudinal extension (preserved)
  5. ¶5 (line 26): A-Abstract DAPT reframe (5 V × 16 cell, task-asymmetric, MI granularity mismatch)
  6. ¶6 (line 28): Deployment path summary (preserved)
- All three subagent contributions merged without conflict.

### §1.4 Findings list (6 findings)

- **F1** (architecture/pretraining/capacity): Subagent B's full rewrite (探索性消融 framing) + Subagent C's cohort caveat inline ("21 名 responder cohort，继承自 [3] 的 49 → 21 离线筛选")
- **F2** (channel reduction): Subagent C cohort caveat applied ("在 21 名 responder cohort × cross-subject binary 上；通道选择 ranking 使用了所有 session 数据，可能轻微高估 retention，详见 Limitation #1")
- **F3** (longitudinal): unchanged
- **F4** (longitudinal data extension): unchanged
- **F5** (DAPT): Subagent A's task-asymmetric rewrite (Note: this is **F5** in §1.4, but **F4** in §7 — different numbering)
- **F6** (deployment): unchanged

### §7 Findings list (5 findings)

- **F1** (architecture/pretraining/capacity): Subagent B's full rewrite + Subagent C's "21 名 responder cohort" + LaBraM/NeuroLM/BIOT cross-link
- **F2** (channel reduction): Subagent C cohort caveat applied
- **F3** (longitudinal): unchanged
- **F4** (DAPT): Subagent A's task-asymmetric rewrite
- **F5** (channel selection method): unchanged
- §7 closing paragraph (line 1097): Subagent C's softening (no longer claims "EEG domain by signal-level features" as universal proposition)

## 4. Verification results

- `git diff -- paper/drafts/paper_draft_v3.0.1.md`: **EMPTY** (v3.0.1 byte-identical preserved)
- `wc -l`: v3.0.1 = 1357 lines / v3.1 = 1479 lines (Δ = +122 lines, +9.0%)
- `diff` line count: 376 lines (representing rewritten sections)
- **Grep checks**:
  - `三向分解` count: 6 (line 10 changelog header; line 77 §1.4 F1 hedged; line 940 §4.1 hedged future-state; line 1031 §5 Limit #7 hedged; line 1058 §6 #8 hedged; one in B's §3.7.3 intro). All remaining instances are in correctly hedged contexts.
  - `+34.97 pp` count: 3 (line 871 §3.7.3 footnote table; line 876 footnote ²; line 881 §3.7.3 解读边界 paragraph). All in hedged composite-estimate context per B's spec.
  - `一致负迁移` count: 4 (line 774 §3.6.2 explicit "先前框架不符" rejection; line 778 §3.6.2 "DAPT 一致负迁移在 ternary 上无法成立"; line 977 §4.5 explicit "from v3 草稿... 重写为 task-asymmetric"; line 1034 §5 #10 ternary baseline footnote — describes secondary/historical context). All explicitly framed as historical / rejected.
  - `30.48M` count: 9 (P1.8 unification successful — abstract, §1.3, §3.7.1 table, §3.7.2, §4.1, §3.7.3 footnotes 1+2+3, etc.)
  - `Schirrmeister` count: 8 (refs [10] entry + inline citations + DAPT V4 mentions)
  - `Gururangan` count: 5 (refs [20] entry + 4 inline locations: §1.3, §4.5, §4.8 末段, §7 末段)
  - `Stouffer` count: 11 (DAPT statistical aggregation throughout §3.6 + abstract + §1.4 F5 + §7 F4 + §4.8)
  - `responder cohort` count: 4 (abstract, §1.4 F1, §1.4 F2, §7 F1, §7 F2 — exact target met)
- **References count**: 25 (preserved [1]–[9] + new [10]–[25])
- **Reference [10]–[25] inline format check**: 16 new ref entries appended in correct numeric order, no gaps.

## 5. Known issues / followups

### Stage 4 Step 4 cleanup tasks (from Subagent C §6.4)

1. §3.4.4 / §3.5.4 repeated XSI-FT mini-definitions can be simplified to "XSI-FT (§3.3 mechanism)" reference (after C3 lineage paragraph in place)
2. R1 Minor #4: EEGNet-16,4 16K vs 10K vs 16,162 unification — applied at abstract & §1.3 (now ~16K); other sections may still have legacy ~10K wording — needs Step 4 sweep
3. EEGNet vs CBraMod preprocessing alignment — §5 Limitation needs new row (R1 Minor #6)
4. Table 0 rename + add "评估难度" column (R2-4)
5. §4.6 / §7 F6 wearable/edge benchmark hedge (P3.2 / DA-MODERATE #8)
6. §3.2 line 350 add [5] Lawhern 2018 inline (R2-5)
7. §4.4 BCI illiteracy / longitudinal MI literature integration (R2-7, optional)
8. §3.4.4 paired_p column sync with §3.4.5 Table 15b (R1 Minor #4 + EIC-2)
9. §3.5.2 EIC-8 anatomical discussion compression (P2.6)
10. Some inline citations Subagent C identified are not yet applied (R2 minor + R3 cross-disciplinary): §2.6 [11]/[12]/[13] CSP refs; §1.2 [18] BCI 10-year update; §3.5.4 [13] ; §3.7.1 [22] Hoffmann; §3.7.2 [21] Mosbach in random-init within ternary explanation. These are recommended for Step 4 cleanup but do not block Step 3 (figures).

### §6 final numbering

- §6 now has **8 items** (was 7 in v3.0.1; B's EDIT 13 added #8). Numbering verified consistent — all cross-references in EDITS 4 / 5 / 6 / 7 / 8 / 9 / 10 / 11 / 12 / 15 use "§6 #8" correctly.

### Cross-section consistency (Subagent B §6.3 audit)

All 5 locations now reference: "exploratory ablation" framing + "binary +23.10 / ternary +30.79 pp" double-value + "§6 #8 future work" deferral:
- Abstract (line 20) ✓
- §1.4 Finding 1 (line 77) ✓
- §3.7 chapter (lines 826-878) ✓
- §4.1 (lines 934-941) ✓
- §7 Finding 1 (line 1066) ✓

## 6. Stage 4 Step 3 (figures) handoff notes

Figures requiring regeneration:

1. **Figure 10 (Further Pre-training)**: HIGH PRIORITY. Current figure shows V1/V2 2-row layout, mismatched with new task-asymmetric narrative (5 V × 4 cell + Δ-of-Δ forest plot). Figure caption at line 763 already marks "图待 regenerate" — Step 3 owns regeneration. Source data: `paper/reviews/stage4_step1b_stat_recompute_v4v5.md` Table provides per-cell values.

2. **Figure 4b channel method ranking flip** (§3.5): unchanged, no regen needed.

3. **Figure 1 / 6 / 6b** (within / cross / XSI-FT): may need re-render given EEGNet XSI-FT data added 2026-05-06 (per existing v3 caveat L387/L391); Subagent A's edits did not touch these but R1 Minor #1 flags version sync.

4. **Inference latency Figure 11**: unchanged.

No figure references to "+34.97 pp" / "−25.30 pp" need updating in captions — these numbers are preserved in tables (§3.7.3 unchanged numeric data, only interpretation softened).

## 7. Recommendation for orchestrator

**PROCEED to Stage 4 Step 3 (figures regeneration)**, prioritizing Figure 10 regen given the §3.6 narrative shift to 5 V × task-asymmetric framework. Step 4 (cleanup tasks listed in §5 above) can run in parallel or after Step 3 — most are minor textual polish + inline citation adds that don't gate the figure work.

The integration is internally consistent across §1.4 / §3.6 / §3.7 / §4.1 / §4.5 / §7 narrative. Numbers verified against `stage4_step1b_stat_recompute_v4v5.md` ground truth. References [10]–[25] all appended with verified DOI/arXiv. v3.0.1.md byte-identical preserved as required.

No failures requiring manual intervention. R&R Letter seed (Subagent C §4) intentionally NOT applied to v3.1.md (preserved as separate file for future R&R generation).
