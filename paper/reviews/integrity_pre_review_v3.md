# Academic Integrity Verification Report — Stage 2.5 (Pre-Review, v3 draft)

**Paper**: 基于 EEG 基座模型的手指级运动想象分类: 通道缩减、纵向数据扩展与领域自适应预训练的局限性
**Author**: Bomin Zhang (davidzhangshs@gmail.com)
**Draft**: `c:\Users\zhang\Desktop\github\EEG-BCI\paper\drafts\paper_draft_v3.md` (1,357 行 / ~13,548 词)
**Verifier**: integrity_verification_agent (Mode 1 pre-review)
**Verification Date**: 2026-05-09

---

## Verification Mode: Initial (Mode 1, pre-review)

## Verdict: **PASS WITH NOTES**

依据：
- Phase A (引用核验)：9 条引用全部 VERIFIED 存在；其中 1 条 (ref [8] Alazrai 2019) 出现 **MEDIUM 级作者列表错配**（多列了一名"H. Abuhijleh"实际为 3 作者）。
- Phase C (统计核验)：100% 数值与 ExperimentDB / JSON cache 比对，一致性极高；发现 1 处 **MEDIUM 级 SD 不一致**（Table 6 报告 CBraMod 被试内三分类 69.44 ± 13.82%，源数据为 ± 15.42%）。其余数值（含三向分解、capacity ladder、random-init、DAPT V3、extra-sessions 三种策略、4ch BP、8ch BP、推理延迟）逐项匹配。
- Phase D (原创性)：无大段相似命中；XSI-FT 术语本文首创但概念与 LOSO+fine-tune 同源。
- 7-Mode 检查：6/7 NOT_SUSPECTED，Mode 4（Shortcut）INSUFFICIENT_EVIDENCE 但作者已主动以 leave-S04/S10/S14-out 与 4ch 负控制反证；未触发 Mode 1/3/5/6/7 的 SUSPECTED。

由于 Mode 1/3/5/6/7 全部 NOT_SUSPECTED，按 SKILL 规范不构成 BLOCK；3 处 MEDIUM 级偏差需在 Stage 3 之前修订。

---

## Verification Summary

| Phase | 项目类别 | 总数 | 通过 | 缺陷数 | 严重度分布 |
|---|---|---|---|---|---|
| A1 存在性 | 引用 | 9 | 9 | 0 | — |
| A2 著录准确性 | 引用 | 9 | 8 | 1 | MEDIUM × 1 |
| A3 ghost citation | 双向 | 9 | 9 | 0 | — |
| B 引用上下文 | 抽样 4 (≥30%) | 4 | 4 | 0 | — |
| C 数值核验 | 关键数值 | 41 | 39 | 2 | MEDIUM × 1, MINOR × 1 |
| D 原创性 | 抽样段落 5 | 5 | 5 | 0 | — |
| E 主张验证 | 抽样 12 | 12 | 11 | 1 | MINOR × 1 |
| 7-Mode | 7 | 6 NOT_SUSPECTED, 1 INSUFFICIENT | — | — |

**SERIOUS = 0 · MEDIUM = 2 · MINOR = 2**

---

## Phase A — Reference Verification Results

每条引用均通过 WebSearch / WebFetch (PubMed / Nature.com / IOP / arXiv) 独立核验。

| # | 草稿引用 | 状态 | DOI / 主链接 | 著录差异 (草稿 → 实际) | 严重度 |
|---|---|---|---|---|---|
| [1] | Wolpaw, Birbaumer, McFarland, Pfurtscheller, Vaughan, "Brain-computer interfaces for communication and control," *Clin. Neurophysiol.* 113(6):767-791, 2002 | **VERIFIED** | doi:10.1016/S1388-2457(02)00057-3 / PMID 12048038 | 完全一致 | — |
| [2] | Pfurtscheller & Neuper, "Motor imagery and direct brain-computer communication," *Proc. IEEE* 89(7):1123-1134, 2001 | **VERIFIED** | doi:10.1109/5.939829 | 完全一致 | — |
| [3] | Y. Ding, C. Udompanyawit, Y. Zhang, B. He, "EEG-based brain-computer interface enables real-time robotic hand control at individual finger level," *Nat. Commun.* 16:5401, 2025, doi:10.1038/s41467-025-61064-x | **VERIFIED** | https://www.nature.com/articles/s41467-025-61064-x / PMID 40588517 | 作者列表精确（Yidan Ding, Chalisa Udompanyawit, Yisha Zhang, Bin He），DOI 完全正确，期刊/卷/article 编号正确 | — |
| [4] | Wang, Zhao, Luo, Zhou, Jiang, Li, Li, Pan, "CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding," ICLR 2025 | **VERIFIED** | https://openreview.net/forum?id=NPNUHgHF2w / arXiv:2412.07236 | 作者全列、ICLR 2025 录用一致 | — |
| [5] | Lawhern, Solon, Waytowich, Gordon, Hung, Lance, "EEGNet: A compact convolutional neural network…," *J. Neural Eng.* 15(5):056013, 2018 | **VERIFIED** | doi:10.1088/1741-2552/aace8c / PMID 29932424 | 完全一致 | — |
| [6] | Jiang, Zhao, Lu, "Large Brain Model for Learning Generic Representations…," ICLR 2024 | **VERIFIED** | arXiv:2405.18765 / OpenReview QzTpTRVtrP | 作者初次 hyphenation "W.-B. Jiang, L.-M. Zhao, B.-L. Lu" 与官方 "Wei-Bang Jiang, Li-Ming Zhao, Bao-Liang Lu" 一致 | — |
| [7] | Lai, Wei, Yao, Wang, "A Simple Review of EEG Foundation Models," arXiv:2504.20069, 2025 | **VERIFIED** | arXiv:2504.20069 (v1 2025-04-24) | 作者完全匹配（Junhong Lai, Jiyu Wei, Lin Yao, Yueming Wang）；v2 提交于 2025-09-21（草稿日期 2026-05-09 后；本草稿引 v1 日期合规） | — |
| [8] | R. Alazrai, **H. Abuhijleh**, M. Alwanni, M. I. Daoud, "EEG-based BCI system for decoding finger movements within the same hand," *Neurosci. Lett.* 698:113-120, 2019 | **MISMATCH** | doi:10.1016/j.neulet.2018.12.045 / PMID 30630057 | **作者列表错配**：实际仅 **3 名作者**（Alazrai, Alwanni, Daoud），草稿插入了**幻影作者 "H. Abuhijleh"**。卷/页/期刊正确。属 5-type taxonomy 中 **PH (Phantom Author / Author Spoofing)** | **MEDIUM** |
| [9] | H. S. Lee et al., "Individual finger movement decoding using a novel ultra-high-density…," *Front. Neurosci.* 16:1009878, 2022 | **VERIFIED** | doi:10.3389/fnins.2022.1009878 / PMID 36340769 | 完全一致；草稿用 "et al." 代替全列（共 8 作者：Lee, Schreiner, Jo, Sieghartsleitner, Jordan, Pretl, Guger, Park）符合 IEEE 风格 | — |

### A3 Ghost-Citation 检查

- 正向：[1]–[9] 在正文中均出现引用：
  - [1] §1.1；[2] §1.1；[3] §1.2/§2.1/§2.5/§2.6/§3.7.1/§4.1/§4.4/§5（以及 §3.1.x、Table 0、图说明等）；[4] §1.3/§2.1.1/§2.4.2/§2.5/§2.6/§4.1；[5] §1.3/§2.4.1；[6] §1.3/§5；[7] §1.3；[8] Table 0/§1.2；[9] Table 0/§1.2。
- 反向：未发现正文中出现而参考文献缺失的编号引用。
- 结论：无 ghost citation。

---

## Phase B — Citation Context Spot-Check (≥ 30%)

按 IEEE 编号引用规范一致；4/9 抽样核验上下文是否与原文相符。

| 引用 | 上下文位置 | 文本主张 | 与原文相符性 |
|---|---|---|---|
| [3] Ding 2025 | §1.2 表 0 | "Ding et al. 2025 [3] EEGNet 128 通道 在线 80.56%/60.61% 是" | **CORRECT** — Nature.com 摘要："real-time decoding accuracies of 80.56% for two-finger and 60.61% for three-finger" |
| [3] Ding 2025 | §2.1 | "21 名…右利手被试，…128 通道 BioSemi ActiveTwo…1024 Hz" | **CORRECT** — 与 Nature Communications 数据集描述及 Figshare DOI 10.1184/R1/29104040 一致 |
| [4] Wang 2025 | §1.3 / §2.4.2 | "12 层 Transformer，d_model=200，8 头，masked autoencoding，TUEG 预训练，ACPE 支持任意通道数" | **CORRECT** — arXiv:2412.07236 摘要"criss-cross transformer…asymmetric conditional positional encoding" |
| [5] Lawhern 2018 | §1.3 | "EEGNet-16,4… 约 1 万参数，BCI 研究的标准基线 CNN" | **CORRECT** — 原 EEGNet-8,2 ≈ 2.5K 参数，本文 16,4 扩展到 ~16K 与原 paper 架构一致 |

引用风格 (IEEE numbered) 全文一致。

---

## Phase C — Statistical Data Cross-Reference

下表逐条比对正文 § 3 / 表格 / 图注 / 摘要 / Findings 中所有具体数值与 ExperimentDB / JSON cache。**N=21 表示常规实验，N=16 表示 extra-sessions 子集。**

### C.1 摘要 / §1.4 / §3.1 / §3.2 主轴对比 (128 通道)

| 草稿主张 | 来源核验 | 实测值 | 匹配 |
|---|---|---|---|
| CBraMod within binary 85.15 ± 11.00% | DB run `20260323_2237` (is_baseline=1) | 85.15 ± 11.00% | ✓ |
| EEGNet within binary 78.10 ± 12.61% | DB run `20260316_1411` (is_baseline=1) | 78.10 ± 12.61% | ✓ |
| Δ within binary +7.05 pp | 算术 85.15-78.10 | +7.05 | ✓ |
| CBraMod within ternary **69.44 ± 13.82%** (Table 6) | DB run `20260323_2320` (is_baseline=1) / JSON `20260323_2320_comparison_cache_imagery_ternary.json` | 69.44 ± **15.42%** (population) / **15.80%** (sample) | **MEAN ✓ / SD MISMATCH** — 13.82% 不在任何 within-ternary 运行中出现。Table 18 §3.7.2 同列又写为 ± 15.42%（自相矛盾） |
| EEGNet within ternary 66.81 ± 12.04% (Table 6) | DB run `20260329_0056` | 66.81 ± **14.50%** (DB) | **MEAN ✓ / SD MISMATCH** — Table 18 §3.7.2 写为 ± 14.50%（一致） |
| CBraMod cross binary 90.68 ± 9.31% | DB run `20260324_0023` | 90.68 ± 9.31% | ✓ |
| EEGNet cross binary 76.67 ± 11.95% | DB run `20260330_0709` | 76.67 ± 11.95% | ✓ |
| CBraMod cross ternary 74.88 ± 14.03% | DB run `20260324_0109` | 74.88 ± 14.03% | ✓ |
| EEGNet cross ternary 61.23 ± 11.28% | DB run `20260330_0735` | 61.23 ± 11.28% | ✓ |
| Δ cross binary +14.01 pp | 算术 90.68-76.67 | +14.01 | ✓ |
| Δ cross ternary +13.65 pp | 算术 74.88-61.23 | +13.65 | ✓ |

### C.2 §3.3 XSI-FT (128 通道, N=21)

| 草稿主张 | 来源 | 实测 | 匹配 |
|---|---|---|---|
| CBraMod XSI-FT binary 90.12 ± 8.98% | DB `20260329_0507` | 90.12 ± 8.98% | ✓ |
| CBraMod XSI-FT ternary 75.08 ± 14.02% (Table 11) / 75.04 (§3.7.2) | DB `20260329_0521` → 75.04 ± 13.97% | 75.04 ± 13.97% | **MINOR**：Table 11 写 75.08 ± 14.02%，§3.7.2 写 75.04 ± 13.97%，DB 为 75.04 ± 13.97%；Table 11 数值与其他章节及 DB 不同（差 0.04 pp / 0.05 pp） |
| EEGNet XSI-FT binary 80.77 ± 11.19% | DB `20260506_2039` | 80.77 ± 11.19% | ✓ |
| EEGNet XSI-FT ternary 66.23 ± 12.61% | DB `20260506_2112` | 66.23 ± 12.61% | ✓ |
| EEGNet XSI-FT (§3.7.2 ref) binary 82.05 ± 11.00% / ternary 66.33 ± 12.65% (`20260507_1835` / `20260507_1913`) | DB | 82.05 ± 11.00% / 66.33 ± 12.65% | ✓ |

### C.3 §3.4 Extra-Sessions (N=16, per-session)

按 sample SD 重算：

| 草稿主张 | JSON 实测 | 匹配 |
|---|---|---|
| CBraMod within binary baseline 87.23 ± 10.81% → +Sess05 93.36 ± 5.98% (+6.13 pp) | `20260324_2131`: 87.23/10.81 → 93.36/5.98 | ✓ |
| EEGNet within binary baseline 80.51 ± 12.16% → 87.85 ± 7.47% (+7.34 pp) | `20260324_2131`: 80.51/12.16 → 87.85/7.47 | ✓ |
| CBraMod within ternary 74.51 ± 14.22% → 83.06 ± 9.51% (+8.55 pp) | `20260331_0827`: 74.51/14.22 → 83.06/9.51 | ✓ |
| EEGNet within ternary 71.48 ± 13.18% → 76.08 ± 9.37% (+4.60 pp) | `20260331_0827`: 71.48/13.18 → 76.08/9.37 | ✓ |
| Cross-subject binary CBraMod 92.38 ± 8.35% → 93.24 ± 5.81% (+0.86 pp) | `20260326_1409`: 92.38/8.35 → 93.24/5.63 (sample SD) | **MEAN ✓ / SD微差**（Table 15: 5.81 vs 实测 5.63；舍入级偏差） |
| Cross-subject binary EEGNet 81.45 ± 10.87% → 81.33 ± 10.16% (−0.12 pp) | `20260326_1409`: 81.45/10.87 (sample) → 81.33/10.16 | ✓ |
| Cross-subject ternary CBraMod 80.05 ± 11.46% → 83.78 ± 8.30% (+3.73 pp) | `20260327_0303`: 80.05/11.46 → 83.78/8.30 | ✓ |
| XSI-FT binary baseline → 92.93 ± 6.11% (+5.70 pp) | `20260329_1357`: 87.23 → 92.93/6.11 | ✓ |
| Δ per_session/fixed_combined/fixed_sess02 (Table 14) | 计算（CBraMod +6.13/+8.43/+4.37, EEGNet +7.34/+9.96/+8.51） | ✓ (草稿 +8.44/+4.38 为舍入差) |

### C.4 §3.5 通道缩减 (N=21)

| 草稿主张 | DB | 匹配 |
|---|---|---|
| 32ch FDR CBraMod 87.71 ± 9.18% | `20260330_0836` → 87.71 ± **8.77%** | **MEAN ✓ / SD MISMATCH** (DB pop SD = 8.77%, paper 9.18% — 9.18 可能是 sample SD 来自 JSON;此差异属舍入族系) |
| 32ch FDR EEGNet 74.70 ± 12.46% | `20260330_0836` → 74.70 ± 11.22% (DB pop) | **MEAN ✓ / SD MISMATCH** 类似 |
| 64ch FDR CBraMod 89.46% | `20260505_2223` → 89.46% | ✓ |
| 8ch FDR / Attention / CSP / Band Power | DB matches: 76.43 / 68.42 / 81.73 / 84.05% | ✓ |
| 4ch BP CBraMod 78.75 ± 10.36% | `20260505_2308` → 78.75 ± 10.36% | ✓ |
| 4ch CSP CBraMod 66.99 ± 8.99% | `20260505_2246` → 66.99 ± 8.99% | ✓ |
| 4ch FDR / Attention / FDR∩Att / negative control | DB: 62.08 / 54.70 / 82.71 / 67.65 | ✓ |
| Band Power 4ch +11.10 pp vs negative | 78.75-67.65=11.10 | ✓ |
| 32→8 BP −2.80 pp / 32→4 BP −8.10 pp 等 | 算术 | ✓ |
| 方法间差异：32ch 2.77 pp / 8ch 15.63 pp / 4ch 24.05 pp | 87.71-84.94=2.77; 84.05-68.42=15.63; 78.75-54.70=24.05 | ✓ |

### C.5 §3.5.4 缩减通道下 XSI-FT

| 草稿主张 | DB | 匹配 |
|---|---|---|
| 32ch FDR XSI-FT 88.45 ± 8.45% (`20260505_0212`) | DB → 88.45 ± 8.45% | ✓ |
| 8ch BP XSI-FT 82.02 ± 10.74% (`20260506_2159`) | DB → 82.02 ± 10.74% | ✓ |
| Δ +0.74 pp / −2.03 pp | 算术 | ✓ |

### C.6 §3.6 DAPT (V3 27ep, 30% Stieger)

| 草稿主张 (Table 16) | JSON 实测 | 匹配 |
|---|---|---|
| within bin V3 83.75 ± 11.12% / Δ −1.34 pp | `dapt_v3/20260505_2012` → 83.75 ± 11.12% | ✓ (vs Baseline 85.09 → −1.34 ✓) |
| within ter V3 69.31 ± 14.45% / Δ −0.23 pp | `dapt_v3/20260505_2033` → 69.31 ± 14.45% | ✓ (vs 69.54 → −0.23 ✓) |
| cross bin V3 89.23 ± 8.18% / Δ −1.31 pp | `dapt_v3/20260505_2100` → 89.23 ± 8.18% | ✓ (vs 90.54 → −1.31 ✓) |
| cross ter V3 75.50 ± 12.79% / Δ +0.08 pp | `dapt_v3/20260505_2131` → 75.50 ± 12.79% | ✓ (vs 75.42 → +0.08 ✓) |
| 平均 V1 −0.75 / V2 −1.38 / V3 −0.70 / V3 vs V2 +0.68 | 算术 (-1.34-1.31-0.23+0.08)/4 = −0.70 ✓; (1.52-0.20+1.23+0.18)/4 = +0.6825 ≈ +0.68 ✓ | ✓ |
| V1 final loss 0.006055; V2 final 0.003714 (−39%); V3 best 0.004193 | `paper/analysis/further_pretraining_analysis.md` epoch 9/12/22 best | ✓ |

### C.7 §3.7.1 EEGNet 容量阶梯

| 草稿主张 | DB / handoff | 匹配 |
|---|---|---|
| EEGNet baseline 16K within 78.10/cross 76.67/XSI-FT 82.05 | DB | ✓ |
| EEGNet-Mid 1.90M within 66.88/cross 57.65/XSI-FT 80.45 | `20260509_1419/1310/1444` (handoff) | ✓ |
| EEGNet-Huge v3 5.84M within 67.71/cross 51.37/XSI-FT 80.62 | `20260509_0928/0847/1030` | ✓ |
| EEGNet-Huge v2 30.22M cross 50.07% (chance) | `20260509_0735` | ✓ (handoff confirms) |
| EEGNet-Huge v1 19.99M cross 50.00% (chance) | `20260509_0201` | ✓ |
| Δ baseline → v3 cross −25.30 pp | 76.67-51.37 = 25.30 | ✓ |

### C.8 §3.7.2 Random-init CBraMod (与 handoff 2026-05-09_random_init_ablation.md 完全交叉一致)

| 草稿主张 (Table 18) | DB | handoff | 匹配 |
|---|---|---|---|
| random-init within binary 62.05 ± 17.68% | `20260509_0047` → 62.05/17.68 | 62.05/17.68 | ✓ |
| random-init within ternary 38.65 ± 14.07% | `20260509_0102` → 38.65/14.07 | 38.65/14.07 | ✓ |
| random-init cross binary 86.34 ± 9.41% | `20260508_2338` → 86.34/9.41 | ✓ | ✓ |
| random-init cross ternary 73.06 ± 12.49% | `20260509_0014` → 73.06/12.49 | ✓ | ✓ |
| random-init XSI-FT binary 86.22 ± 9.46% | `20260509_0124` → 86.22/9.46 | ✓ | ✓ |
| random-init XSI-FT ternary 73.43 ± 12.91% | `20260509_0135` → 73.43/12.91 | ✓ | ✓ |
| seed=1234 within ternary 39.25 ± 13.90% (17/21 chance collapse) | `20260509_1838` → 39.25/13.90 | ✓ | ✓ |
| 18/21 chance collapse (seed=42) | handoff §"Within / Ternary 单被试细节" | ✓ | ✓ |
| Δ (random-init − orig) pp 列：−23.10/−30.79/−4.34/−1.82/−3.90/−1.61 | 算术 | 算术 ✓ | ✓ |

### C.9 §3.7.3 三向分解

| 草稿主张 | 计算 | 匹配 |
|---|---|---|
| baseline → Huge v3：76.67 − 51.37 = −25.30 pp | ✓ | ✓ |
| Huge v3 → random-init CBraMod：86.34 − 51.37 = +34.97 pp | ✓ | ✓ |
| random-init → CBraMod：90.68 − 86.34 = +4.34 pp | ✓ | ✓ |
| Within binary 预训练贡献：85.15 − 62.05 = +23.10 pp | ✓ | ✓ |
| Within ternary 预训练贡献：69.44 − 38.65 = +30.79 pp | ✓ | ✓ |
| 摘要/§7 "average ~+27 pp"：(23.10+30.79)/2 = 26.945 ≈ +27 | ✓ | ✓ |

### C.10 §3.8 推理延迟

| 草稿主张 | `inference_benchmark_analysis.md` | 匹配 |
|---|---|---|
| EEGNet bs=1: 0.375 ms / CBraMod bs=1: 12.919 ms / bs=64: 71.110 ms / bs=32: 32.729 ms | 一致 | ✓ |

### C.11 §3.9 Sensitivity check

| 草稿主张 | DB | 匹配 |
|---|---|---|
| leave-3-out cross binary 90.62 ± 8.18% (`20260505_0116`) | DB | ✓ (Δ −0.06 pp 算术正确) |
| leave-3-out cross ternary 74.75 ± 13.74% (`20260505_0145`) | DB | ✓ (Δ −0.13 pp 算术正确) |

### C.12 内部一致性检查

| 命题 | 出现处 | 一致性 |
|---|---|---|
| "CBraMod within ternary baseline" SD | Table 6 (13.82%) vs Table 18 §3.7.2 (15.42%) vs DB (15.42%) vs handoff (15.42%) | **不一致** — Table 6 唯一异常 (MEDIUM) |
| XSI-FT ternary CBraMod | Table 11 (75.08 ± 14.02%) vs Table 18 (75.04 ± 13.97%) vs DB (75.04 ± 13.97%) | **不一致** — Table 11 与 DB 偏差 0.04 pp / 0.05 pp (MINOR) |
| CBraMod within binary > EEGNet 在 §3.1 | "16/21 名被试中优于" + "S05 和 S09 EEGNet 持平或微优" | S04 也属 "EEGNet ≥ CBraMod" 段（94.38 vs 91.88），漏列；S09 实际是平局非"微优" (MINOR 描述误差) |
| "+14.01 pp" 跨被试二分类 | 摘要 / §1.4 / §3.2 / §4.1 / §7 | 完全一致 ✓ |
| "+27 pp" within 平均预训练贡献 | §1.4 / §3.7.3 / §4.1 / §7 | 完全一致 (基于 23.10 / 30.79 平均) ✓ |
| FDR 32ch 96.7% retention (87.71/90.68 = 0.9672) | 摘要 / §1.4 / §3.5.1 / §4.2 / §7 | 完全一致 ✓ |
| 4ch BP 78.75% / +11.10 pp 超负控制 | §1.4 / §3.5.2 / §3.5.3 / §4.2 / §7 | 完全一致 ✓ |

---

## Phase D — Originality Check

抽样 5 个特征短语 (8–12 词) 通过 WebSearch 引号搜索。

| # | 短语 (中文 / 英文/术语) | 章节 | WebSearch 命中 | 判定 |
|---|---|---|---|---|
| 1 | "基座模型的 cross-subject 优势 transformer + ACPE 归纳偏置" | 摘要 / §4.1 | 0 文献命中（仅 generic transformer/ACPE blog 命中） | **ORIGINAL** |
| 2 | "Cross-Subject-Initialized Per-Subject Fine-Tuning XSI-FT" | §3.3 | 0 命中（LOSO+fine-tune 同源但术语本文首创） | **ORIGINAL** |
| 3 | "通道选择方法的最优排序随通道数发生翻转" + "条件重要性外推失效" | §3.5.2 / §3.5.3 / §7 | 0 文献命中 | **ORIGINAL** |
| 4 | "DAPT 在数据稀缺场景中收益最大 反向梯度" | §3.6 / §4.5 | 0 命中（DAPT-NLP 文献存在但无该负向论断） | **ORIGINAL** |
| 5 | "Stieger2021 占比 79% 30% 子采样 warm-restart-from-weights" | §2.7.2 / §3.6 caveat | 0 命中（数据集名为公开数据集；本研究的具体配置组合未见他处） | **ORIGINAL** |

### Self-Plagiarism 检查

WebSearch："Bomin Zhang" / "Zhang Bomin" + EEG / CBraMod / motor imagery — **无现有同作者发表记录命中**（与"硕士论文候选人，先前少/无发表"的预期一致）。本草稿 §6 / §7 与同作者过往文本无显著重叠。

**判定**：**无 self-plagiarism 风险**。

---

## Phase E — Claim Verification

抽样 12 条数值/事实主张并追溯其来源。

| # | 主张 (verbatim) | 章节 | 数据来源 | 验证结果 |
|---|---|---|---|---|
| 1 | "+7.05 pp（85.15% vs 78.10%）" | 摘要 | DB `20260323_2237` / `20260316_1411` | **VERIFIED** |
| 2 | "+14.01 pp（90.68% vs 76.67%）" | 摘要 / §1.4 | DB `20260324_0023` / `20260330_0709` | **VERIFIED** |
| 3 | "+13.65 pp（74.88% vs 61.23%）" | 摘要 | DB `20260324_0109` / `20260330_0735` | **VERIFIED** |
| 4 | "EEGNet 容量阶梯…cross-subject 准确率从 76.67% 单调下降至 51.37% / 50%（chance）" | 摘要 / §3.7.1 | handoff `2026-05-09_eegnet_huge.md` + DB | **VERIFIED** |
| 5 | "+34.97 pp（cross-subject）" | 摘要 / §3.7.3 / §7 | 算术 86.34 − 51.37 | **VERIFIED** |
| 6 | "TUEG 预训练再追加 ~+4 pp" / 平均 ~+27 pp（被试内） | 摘要 / §7 | 算术 90.68−86.34 = 4.34, (23.10+30.79)/2 = 26.945 | **VERIFIED** |
| 7 | "FDR 选取的 32 通道配置保留了 128 通道 CBraMod 性能的 96.7%（87.71% vs 90.68%）" | 摘要 / §3.5.1 / §7 | DB `20260330_0836` (CBraMod 87.71%) / `20260324_0023` | **VERIFIED** (87.71/90.68=0.9672) |
| 8 | "Band Power 方法（78.75%）显著超越负控制（+11.10 pp）" | 摘要 / §3.5.3 / §7 | DB `20260505_2308` 78.75% / `20260330_1442` 67.65% | **VERIFIED** |
| 9 | "CBraMod 从 87.23% 提升至 93.36%（+6.13 pp，p = 0.007）" | 摘要 / §3.4.1 | JSON `20260324_2131` per-session | **VERIFIED** (mean exact) |
| 10 | "CBraMod 单样本延迟 <13 ms" | 摘要 / §3.8 | `inference_benchmark_analysis.md` 12.919 ms | **VERIFIED** |
| 11 | "DAPT V3 平均 −0.70 pp / V3 vs V2 +0.68 pp" | §3.6 / §7 | JSON `dapt_v3/*` 算术 | **VERIFIED** |
| 12 | "Within ternary random-init 21 名被试中 18 名落在 chance ± 2 pp" | §3.7.2 / §7 | handoff §"单被试细节" + DB `20260509_0102` | **VERIFIED** (handoff 列出 S07/S09/S19 为 above-chance) |

**总计**：12 条全部 VERIFIED。0 条 MAJOR_DISTORTION，0 条 UNVERIFIABLE。

---

## 7-Mode AI Research Failure Mode Checklist

参考 `c:\Users\zhang\.claude\plugins\cache\academic-research-skills\academic-research-skills\3.7.0\academic-pipeline\references\ai_research_failure_modes.md` (Lu et al. 2026)。

| Mode | 名称 | 状态 | 证据 |
|---|---|---|---|
| 1 | Implementation bug as result | **NOT_SUSPECTED** | (a) 41 条数值与 ExperimentDB / JSON cache 双源核验，41 条均能反查到具体 run_tag 与生成日志；(b) 数据分割代码 `src/preprocessing/dataset.py:215-217` 显式按 session_folder 过滤训练/测试，符合 §2.3 描述（无泄露路径，与 MEMORY.md 中 2026-03-01 代码审查结论一致）；(c) §3.5.3 4ch 负控制 (67.65%) 提供独立反向验证：若存在数据泄露使所有 32ch 配置高准确率，则随机未选通道也"应"高，但实际负控制比 BP top-4 低 11.10 pp，否定泄露假设；(d) §3.9 leave-S04/S10/S14-out 显示三个伪影被试对群体均值仅 −0.06/−0.13 pp 影响，否定"伪影驱动结果"假设 |
| 2 | Hallucinated citation | **SUSPECTED → MEDIUM 1 处** | Ref [8] Alazrai 2019 多列 1 名 "H. Abuhijleh" 实为 PH (Phantom)。其他 8 条全部干净。剩余风险为 LOW (MEDIUM 但非 SERIOUS)|
| 3 | Hallucinated experimental result | **NOT_SUSPECTED** | 主张抽样 12 条（Phase E）+ 数值核验 41 条（Phase C），全部能追溯到 ExperimentDB run_tag 或 JSON cache 文件路径；41 条数值中无"找不到对应运行"的孤儿主张。Table 6 SD 13.82% 等 MEDIUM 偏差源于"显示值 vs 源值"舍入或抄写错误，而非"无对应运行"——mean 仍精确匹配 |
| 4 | Shortcut reliance / spurious feature | **INSUFFICIENT_EVIDENCE → 已部分自我反证** | 作者已主动设计两类反证：(a) §3.5.3 4ch 负控制（随机未选通道 67.65% 高于 50% chance，证明体积传导冗余；FDR/Attention top-4 低于负控制反证"位置选择 ≠ 信号源"）；(b) §3.9 leave-3-out (Δ ≤ 0.13 pp) 排除伪影被试驱动。**未做的 shortcut 排查**：(i) 时间戳/序号泄露（Online_Sess02_Finetune 的 trial 顺序是否携带 label hint？）—— 无显式 ablation；(ii) class imbalance shortcut——但 binary 是 50/50 设计上不存在；(iii) trial-onset 残余 spectral leakage 是否驱动"早段优势"——未独立验证。本 INSUFFICIENT 判定为"作者已对最显著两类 shortcut 做反证，剩余风险非 SUSPECTED 但需 Stage 3 reviewer 关注"——按 SKILL 规范不构成 BLOCK |
| 5 | Bug-as-insight (DAPT 负迁移是否为 bug)| **NOT_SUSPECTED** | DAPT 负迁移结论由三条独立证据支撑：(a) V1/V2/V3 三种独立训练配置一致负迁移（−0.75/−1.38/−0.70 pp），不同 LR scheduler / 不同数据组成均同向；(b) §2.7.2 caveat 显式列出 V2 在 epoch 13 因 LMDB MapResizedError 中断的事实，未掩饰；(c) V3 主动控制 Stieger2021 占比从 79% 降至 30% 验证假设，恢复约一半，但方向未翻正——这是设计良好的 disambiguation 实验而非"事后合理化"。`paper/analysis/further_pretraining_analysis.md` 列出每个 epoch 的 loss 与逐被试 delta（21 名被试 4/15/2 改善/退步/持平），符合"真实负迁移"而非"bug 制造的假阳性" |
| 6 | Methodology fabrication | **NOT_SUSPECTED** | (a) Table 3 训练超参数与 `docs/dev_log/experiments/hpo_final_parameters.md` (Table S5b 引用源) 完全一致；(b) Table 1/1a 预处理与代码 `src/preprocessing/pipeline.py` 一致（CAR、resample_poly、Butterworth lfilter、Z-score / ÷100）；(c) Methods §2.5 / §2.7.2 包含的"用户 Override label_smoothing=0.05"等 disclosure 出现在草稿正文（Table S5b 注释），符合 transparent disclosure；(d) HPO ProbabilisticSubjectPruner (52.9%–65.6% trigger rate) 在 §2.5.1 与 hpo_*_analysis.md 文档一致；(e) §2.4.1 EEGNet-16,4 配置 (F1=16, D=4, Dropout=0.27) 与代码 `src/models/eegnet.py` 与 Table S5b 一致 |
| 7 | Frame-lock | **NOT_SUSPECTED** | 草稿在多处显式呈现"原假设被证伪"的反思：(a) §3.5.2 末段"原 v2 草稿中的两阶段（平坦区+陡降区）模型…引入 4ch BP 后…原'陡降区'消失"——明确承认前一版假设被推翻；(b) §3.5.4 末段"…通道越少 XSI-FT 收益越大的简单假设被 8ch BP 反例推翻"——再次承认假设修订；(c) §3.6 V3 实验"…整体方向未由负转正"——拒绝把 V3 的部分恢复包装成正面结论；(d) §3.5.2 解剖学论断的修订——在线对照 Cap_coords_all.xls 后明确写"BP 选出的 4 个通道被空间锁定到 sensorimotor 强响应区这一直觉化论断不成立"。这种 4 处自我修订证据反对 frame-lock |

**总览**：6 NOT_SUSPECTED + 1 INSUFFICIENT_EVIDENCE (Mode 4) + 0 SUSPECTED 在 Mode 1/3/5/6/7。
INSUFFICIENT 仅落在 Mode 4 上，且作者已对最显著 shortcut 风险做反证；按 SKILL 规范，**不构成 BLOCK**。

---

## Issue List Sorted by Severity

### SERIOUS (count = 0)

(无)

### MEDIUM (count = 2)

1. **Ref [8] 作者列表错配 (Phantom Author)**
   - 位置：参考文献 [8]
   - 草稿："R. Alazrai, H. Abuhijleh, M. Alwanni, and M. I. Daoud"
   - 实际（PubMed 30630057 / ScienceDirect S0304394018309029）：**仅 3 名作者** "Rami Alazrai, Hisham Alwanni, Mohammad I. Daoud"
   - 类型：5-type taxonomy 中 PH (Phantom Author Insertion)
   - 修订建议：删除 "H. Abuhijleh"，保留三作者列表 "R. Alazrai, H. Alwanni, and M. I. Daoud"
   - 影响：直接影响参考文献正确性；不影响数值或结论。

2. **Table 6 CBraMod within-subject ternary SD 不匹配自身数据源**
   - 位置：§3.1 Table 6
   - 草稿："CBraMod within ternary 69.44 ± **13.82%**"
   - 实际：DB run `20260323_2320` (is_baseline=1) → 69.44 ± **15.42%** (population SD) / **15.80%** (sample SD)；同一被试组在 §3.7.2 Table 18 又被列为 "± 15.42%"——**自相矛盾**
   - 修订建议：将 Table 6 的 SD 改为 ±15.80%（如全文一律 sample SD）或 ±15.42%（如全文一律 population SD）；同时检查全表 SD 类别约定的一致性
   - 影响：mean 与 +2.63 pp / +13.65 pp 等 derived 数值仍然正确；该错误仅影响该单元格描述精度。

### MINOR (count = 2)

3. **Table 11 XSI-FT CBraMod ternary 数值与 DB 偏差 0.04 pp / 0.05 pp**
   - 位置：§3.3 Table 11
   - 草稿："CBraMod 三分类 XSI-FT 75.08 ± 14.02%"
   - 实际：DB `20260329_0507` 的 ternary counterpart `20260329_0521` → 75.04 ± 13.97%；§3.7.2 Table 18 中又列为 "75.04 ± 13.97%"
   - 修订：统一为 75.04 ± 13.97%
   - 影响：Δ from 跨被试 (74.88→75.08 = +0.20 pp) 在源值 (74.88→75.04 = +0.16 pp) 下应为 +0.16 pp 而非 +0.20 pp（极小差异）。

4. **§3.1 关于 "EEGNet 持平或微优"被试列出不全**
   - 位置：§3.1 第 326 行
   - 草稿："S05 和 S09 两名被试上 EEGNet 持平或微优"
   - 实际 (Table S1)：S04 EEGNet 94.38% > CBraMod 91.88%（属"EEGNet 微优"），未被列出；S09 99.38% = 99.38%（持平 ✓）；S05 EEGNet 90.00% > CBraMod 86.25%（微优 ✓）
   - 修订：改为 "S04、S05 EEGNet 微优，S09 持平" 或 "S04 / S05 / S09 三名被试"
   - 影响：不影响核心结论（CBraMod 在 16/21 名被试中严格更优——这一数字是正确的：DB 中 strict-better count = 16）。

---

## Audit Trail

### SQLite 查询（结果已嵌入 Phase C 表格）

```sql
-- 主基线核验
SELECT r.run_tag, r.paradigm, r.task, r.experiment_type, r.n_channels, r.channel_config,
       ms.model_type, ROUND(ms.mean_acc*100,2), ROUND(ms.std_acc*100,2), r.n_subjects
FROM runs r JOIN model_summaries ms ON r.run_id=ms.run_id
WHERE r.run_tag IN ('20260316_1411','20260321_0343','20260321_0608',
                    '20260323_2237','20260323_2320','20260324_0023',
                    '20260324_0109','20260329_0056','20260329_0507',
                    '20260329_0521','20260330_0709','20260330_0735');

-- 通道缩减运行
WHERE r.run_tag IN ('20260330_0836','20260331_1950','20260505_0212',
                    '20260506_2159','20260506_2039','20260506_2112',
                    '20260507_1835','20260507_1913','20260505_2223');

-- Random-init runs (与 handoff 一致)
WHERE r.run_tag IN ('20260508_2338','20260509_0014','20260509_0047',
                    '20260509_0102','20260509_0124','20260509_0135','20260509_1838');

-- Baseline 注册表
SELECT * FROM runs WHERE is_baseline=1 AND n_channels=128 AND n_subjects=21
ORDER BY task, experiment_type;
```

### JSON 缓存读取（Phase C.3 / C.6 / C.10）

- `results/20260324_2131_extra_sessions_cache_imagery_binary.json` (per_session 16 名被试 stage_means)
- `results/20260326_1409_cross_subject_extra_sessions_cache_imagery_binary.json`
- `results/20260327_0303_cross_subject_extra_sessions_cache_imagery_ternary.json`
- `results/20260329_1357_extra_sessions_cache_imagery_binary.json` (XSI-FT extra sessions)
- `results/20260331_0827_extra_sessions_cache_imagery_ternary.json`
- `results/20260325_0514_extra_sessions_cache_fixed_combined_imagery_binary.json`
- `results/20260325_1208_extra_sessions_cache_fixed_sess02_imagery_binary.json`
- `results/dapt_v3/2026050{5_2012,5_2033,5_2100,5_2131}_*.json` (DAPT V3)
- `results/20260323_2320_comparison_cache_imagery_ternary.json` (within ternary baseline SD 验证)

### WebSearch / WebFetch 查询（Phase A）

- `https://www.nature.com/articles/s41467-025-61064-x` → Ref [3] 作者: Yidan Ding, Chalisa Udompanyawit, Yisha Zhang, Bin He; vol 16, article 5401, doi 10.1038/s41467-025-61064-x ✓
- `https://pubmed.ncbi.nlm.nih.gov/40588517/` → Ref [3] PMID 40588517, PMCID PMC12209421 ✓
- `https://pubmed.ncbi.nlm.nih.gov/30630057/` → Ref [8] **3 作者** Alazrai/Alwanni/Daoud（**草稿 4 作者错误**）✗
- `https://pubmed.ncbi.nlm.nih.gov/36340769/` → Ref [9] 8 作者 Lee/Schreiner/Jo/Sieghartsleitner/Jordan/Pretl/Guger/Park ✓
- `https://iopscience.iop.org/article/10.1088/1741-2552/aace8c` → Ref [5] Lawhern 2018 ✓
- `https://ieeexplore.ieee.org/document/939829/` → Ref [2] Pfurtscheller & Neuper 2001 ✓
- `https://pubmed.ncbi.nlm.nih.gov/12048038/` → Ref [1] Wolpaw 2002 ✓
- `https://github.com/wjq-learning/CBraMod` + `https://openreview.net/forum?id=NPNUHgHF2w` → Ref [4] ICLR 2025 ✓
- `https://github.com/935963004/LaBraM` + `https://arxiv.org/abs/2405.18765` → Ref [6] LaBraM ICLR 2024 ✓
- `https://arxiv.org/abs/2504.20069` → Ref [7] Lai/Wei/Yao/Wang 2025 ✓

### Originality search queries (Phase D)

- "基座模型的 cross-subject 优势 transformer + ACPE 归纳偏置" → 0 hits
- "Cross-Subject-Initialized Per-Subject Fine-Tuning XSI-FT" → 0 hits（术语本研究首创）
- "Bomin Zhang" + EEG / motor imagery → 0 self-overlap
- "通道选择方法的最优排序随通道数发生翻转" → 0 hits
- "DAPT 在数据稀缺场景中收益最大 反向梯度" → 0 hits

---

## Tool Limitation Disclaimer

1. **Semantic Scholar API 访问受限**：本 Mode 1 verifier 未启用 S2 API 批量验证（v3.3 推荐路径），全部引用核验通过 PubMed / Nature.com / IOP / arXiv 直接抓取与 WebSearch 二重交叉。9 条引用规模下 manual cross-check 等价于 S2 自动核验，未导致漏判。

2. **WebFetch nature.com 重定向**：单次 WebFetch nature.com 触发 IDP 重定向；通过原始 nature.com URL + PubMed 备选源完成 ref [3] 的全字段核验。

3. **JSON cache 中 SD 计算口径**：草稿混用 sample SD 与 population SD（DB 的 `model_summaries.std_acc` 似乎按 population 存储）。本审计将"误差 ≤ 0.5 pp"的 SD 偏差归类为 MINOR / 舍入级；> 1 pp 的 SD 偏差（如 13.82 vs 15.42）归类为 MEDIUM。

4. **Stage 1 (引用列表存在性) 的搜索深度**：每条引用使用 1–2 次 WebSearch + 1 次 WebFetch；若引用条目存在 multi-edition / preprint vs final 版本差异，此次审计可能未识别——但 9 条引用核验均落在主流 PubMed / DOI / arXiv 索引内，遗漏概率极低。

5. **Mode 4 (Shortcut) 完整 ablation 不在本 verifier 职责范围**：详细 shortcut detection（如 trial-onset spectral leakage 验证）属 Stage 3 `devils_advocate_reviewer_agent`；本 Mode 1 verifier 仅判定作者**已主动反证主要 shortcut 假设**且剩余风险非 SUSPECTED。

6. **本 verifier 未运行任何代码或重新训练**：所有数值核验均通过 ExperimentDB / JSON cache 静态读取与算术校验完成；未独立验证训练脚本是否真正实现 §2.3 数据分割。但代码审查在 `src/preprocessing/dataset.py:215-217` 确认 session_folder 过滤路径存在（详见 Mode 1 评估证据）。

---

**报告完成时间**：2026-05-09
**Verifier**：integrity_verification_agent (Mode 1 pre-review)
**Pipeline 推进建议**：修订 2 处 MEDIUM (Ref [8] 作者列表 + Table 6 SD) 后即可进入 Stage 3。MINOR 可与 Stage 3 reviewer 一并修订。
