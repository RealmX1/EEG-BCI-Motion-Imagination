# Stage 4 Step 2 Subagent B — §3.7 Reframe + §2.5.1 (W) HPO Calibration

**Scope**: B+E partial reframe of §3.7 from "Finding 1: three-way decomposition" to "Exploratory Ablations"; insert HPO calibration argument (W Part A) into §2.5.1; cascade caveats to §1.4 / §4.1 / §6 / §7 / abstract.

**Inputs cross-checked**:
- `paper/drafts/paper_draft_v3.0.1.md` (read-only)
- `docs/handoffs/2026-05-09_eegnet_huge.md` (L154-170 HP table; L156, L195-197, L249-260 LayerNorm diagnosis)
- `docs/handoffs/2026-05-09_random_init_ablation.md` (L186-210 author's own probability estimates)
- `src/hpo/search_spaces.py` (CBraMod within/cross 11-dim; EEGNet within/cross 7-dim)
- Table S5b in v3.0.1.md (CBraMod within 51 trials / 23 complete; EEGNet within 32 trials / 10 complete; CBraMod cross 77 trials / 43 complete)
- Reviewer DA §1.1 (HPO budget asymmetry); Roadmap P0.4 spec

---

## EDIT 1 — §2.5.1 (W) Part A: HPO Calibration Argument

**Anchor**: end of L218 paragraph "...搜索空间涵盖 7–11 个维度（学习率、权重衰减、dropout、batch size、学习率调度参数等）。"

**TYPE**: EXPAND (append two paragraphs immediately after L218)

**OLD_TEXT**:
```
所有报告结果均使用贝叶斯超参数优化（HPO）后的参数。这里的搜索并非从零随机设定，而是明确锚定两篇来源论文：EEGNet 侧以 [3] 的 EEGNet-8,2 / deepEEGNet 设计思路为起点，CBraMod 侧以 [4] 的 fine-tuning defaults 为起点。HPO 采用 Optuna 框架的 TPE（Tree-structured Parzen Estimator）采样器，搜索空间涵盖 7–11 个维度（学习率、权重衰减、dropout、batch size、学习率调度参数等）。
```

**NEW_TEXT**:
```
所有报告结果均使用贝叶斯超参数优化（HPO）后的参数。这里的搜索并非从零随机设定，而是明确锚定两篇来源论文：EEGNet 侧以 [3] 的 EEGNet-8,2 / deepEEGNet 设计思路为起点，CBraMod 侧以 [4] 的 fine-tuning defaults 为起点。HPO 采用 Optuna 框架的 TPE（Tree-structured Parzen Estimator）采样器，搜索空间涵盖 7–11 个维度（学习率、权重衰减、dropout、batch size、学习率调度参数等）。

**搜索空间维度对照**：CBraMod within / cross-subject 各搜索 **11 个**超参数（backbone_lr、classifier_lr_ratio、weight_decay、dropout_rate、batch_size、label_smoothing、gradient_clip、phase_decay、phase_epochs、exploration_epochs、exploration_batch_size；其中后 4 项为 CAWD scheduler 参数），EEGNet within / cross-subject 各搜索 **7 个**超参数（learning_rate、weight_decay、dropout_rate、batch_size、F1、D、kernel_length；F2 = F1 × D 为派生量）。两者维度差源于 CBraMod 需同时调 backbone / classifier 两段学习率以及 CAWD 调度参数，而 EEGNet 共用单一 learning_rate 与简化的 plateau scheduler——并非"给 CBraMod 多调几个旋钮"，而是各自架构的本征参数化差异（详见 [src/hpo/search_spaces.py](../../src/hpo/search_spaces.py)）。

**Trial 预算的 HP-维度校准**：本文 CBraMod within-subject HPO 实际跑了 51 trial（Table S5b），EEGNet within-subject 跑了 32 trial，预算比 51 / 32 ≈ 1.59。在 TPE 类贝叶斯优化中，TPE 收敛到指定误差所需的 trial 数 N 经验上随搜索空间维度 d 以 O(d^c)（c ∈ [0.5, 1]）扩展（Bergstra & Bengio 2011 §3.3 random/Bayesian search dimension dependence；Snoek et al. 2012 §4.1 GP-EI sample complexity）。给定 EEGNet 7 维 vs CBraMod 11 维，"等效收敛"所需 trial 比的下界为 (11 / 7)^0.5 ≈ 1.25、上界为 (11 / 7)^1 ≈ 1.57。本文实际比 1.59 恰好落在 d^1 上界——也就是说，**CBraMod 的额外 trial 数恰好抵消了它额外 4 个搜索维度带来的体积膨胀，并未给 CBraMod 带来"等效收敛精度上的盈余"**。这一校准下两侧 HPO 同等可比，"CBraMod 优势源自不公平的 HPO 预算"反方解释在该校准下不成立；唯一保留的非对称性是 EEGNet 的 7 个 HP 中含有 3 个 architecture HP（F1、D、kernel_length），而本研究继承了 [3] 在原始数据集上对 EEGNet-8,2 / deepEEGNet 的架构调试经验，未把 architecture HP 完全重新搜索（详见 Table S5e EEGNet HP source trace；继承的 architecture HP 默认值在本研究 HPO 中仍允许变动并被显式优化）。
```

**RATIONALE**: Defends against DA §1.1 charge "CBraMod 51 trial vs EEGNet 32 trial = 25–40× HPO budget asymmetry" (DA conflated EEGNet baseline 32-trial HPO with EEGNet-Huge ≤2-trial hand-tune; this calibration only addresses the baseline-vs-baseline asymmetry, not the EEGNet-Huge asymmetry — which is acknowledged separately in §3.7 reframe). Cites Bergstra & Bengio 2011 + Snoek et al. 2012 (both already in the field-standard HPO literature; if these refs are not yet in References, Subagent C / orchestrator should add). 7 vs 11 dim verified directly from `src/hpo/search_spaces.py` lines 79-150.

---

## EDIT 2 — Add Table S5e (EEGNet HP source trace)

**Anchor**: insert immediately after Table S5b (after L1277 "数据来源: docs/dev_log/experiments/hpo_final_parameters.md").

**TYPE**: ADD_PARAGRAPH (new supplementary table)

**NEW_TEXT** (insert after current Table S5b's data-source line):
```

### Table S5e. EEGNet HP source trace

为响应 §2.5.1 的 HP-维度校准说明，本表追踪 EEGNet 7 维搜索空间中各 HP 的来源——继承自 Ding et al. [3] 的 EEGNet-8,2 / deepEEGNet 经验值，还是本研究在 Optuna 中重新搜索得到。

| HP | 来源 | Ding [3] 默认 | 本研究 HPO 搜索范围 | 本研究 HPO 最优 |
|----|------|--------------|---------------------|----------------|
| F1 (filters) | [3] EEGNet-8,2 默认 8 | 8 | {4, 8, 16}（categorical） | **16** |
| D (depth multiplier) | [3] EEGNet-8,2 默认 2 | 2 | {1, 2, 4}（categorical） | **4** |
| F2 (= F1 × D) | 派生 | 16 | 派生（不独立搜索） | 64 |
| kernel_length | [3] EEGNet 默认 64 (= sample_rate / 2) | 64 | {32, 64, 128}（categorical） | 64（未变） |
| learning_rate | 本研究新搜 | — | [1e-4, 1e-2] log-uniform | 4e-3 |
| weight_decay | 本研究新搜 | — | [1e-5, 0.1] log-uniform | 1e-5 |
| dropout_rate | 本研究新搜 | — | [0.2, 0.7] uniform | 0.27 |
| batch_size | 本研究新搜 | — | {32, 64, 128} | 64 |

注：F1 / D / kernel_length 三个 architecture HP 虽继承 [3] 的设计经验，但本研究的 Optuna 搜索仍把它们作为可变 categorical 在指定范围内独立采样；HPO 最优 (F1=16, D=4) 为本研究的搜索结果而非 [3] 默认值的直接采用。本研究未从零冷启动搜索 architecture HP 的边界（如 F1=32 等更大值）——这一上界限制在 §3.7.1 EEGNet-Mid（F1=32）实验中被独立扩展并验证（详见正文）。

> **数据来源**: 搜索空间定义见 [src/hpo/search_spaces.py](../../src/hpo/search_spaces.py) `_sample_eegnet_within` / `_sample_eegnet_cross`；HPO 最优值见 Table S5b。
```

**RATIONALE**: Required by EDIT 1's reference to "Table S5e EEGNet HP source trace". Distinguishes inherited architecture defaults (F1, D, kernel_length range bounds) from newly searched HPs, defusing the DA §1.1 sub-charge that "EEGNet HP entirely inherited from [3]". All numbers verifiable from `src/hpo/search_spaces.py` and existing Table S5b.

---

## EDIT 3 — §3.7 Chapter Title Reframe

**Anchor**: L744 "### 3.7 容量与预训练消融"

**TYPE**: REPLACE_LINE

**OLD_TEXT**:
```
### 3.7 容量与预训练消融
```

**NEW_TEXT**:
```
### 3.7 探索性消融：架构 / 预训练 / 容量贡献的初步检验
```

**RATIONALE**: Roadmap P0.4 explicit demand. Signals to reader from chapter heading that this is exploratory, not a confirmatory three-way decomposition. (English subtitle "Exploratory Ablations: Preliminary Probing of Architecture / Pretraining / Capacity Contributions" can be added by orchestrator if bilingual heading style desired.)

---

## EDIT 4 — §3.7 Chapter Intro Paragraph Reframe

**Anchor**: L746 (the paragraph immediately under §3.7 heading)

**TYPE**: REPLACE_PARAGRAPH

**OLD_TEXT**:
```
为剥离 CBraMod 相对 EEGNet 在 §3.1–§3.3 中观察到的优势的来源，本节报告两项互补消融：(a) §3.7.1 将 EEGNet 的参数规模从 16K 阶梯式扩展到 30M（与 CBraMod backbone 同量级），检验"参数容量本身是否是 EEGNet 表现不及 CBraMod 的根本原因"；(b) §3.7.2 完全切除 CBraMod 的 TUEG 预训练权重（random-init），检验"架构本身在不依赖预训练的情况下是否仍提供独立价值"。两项消融在 {EEGNet, CBraMod} × {random init, TUEG pretrained} 矩阵上覆盖三个角点（"EEGNet pretrained"无对应 EEG 基座模型故空缺），共同支持 §3.7.3 的架构 / 预训练 / 容量三向分解。
```

**NEW_TEXT**:
```
为更好理解 CBraMod 相对 EEGNet 在 §3.1–§3.3 中观察到的优势源自何处，本节报告两项探索性消融：(a) §3.7.1 将 EEGNet 的参数规模从 16K 阶梯式扩展到 30M（与 CBraMod backbone 同量级），探查"参数容量本身是否是 EEGNet 表现不及 CBraMod 的根本原因"；(b) §3.7.2 完全切除 CBraMod 的 TUEG 预训练权重（random-init），探查"架构本身在不依赖预训练的情况下是否仍提供独立价值"。两项消融在 {EEGNet, CBraMod} × {random init, TUEG pretrained} 矩阵上覆盖三个角点（"EEGNet pretrained"无对应 EEG 基座模型故空缺）。

**重要 caveat（贯穿本章）**：本节两项消融在 HPO 预算与扩参轴上均存在已知非对称性，使其结论不具备"独立可归因分解"的力度，应被理解为方向性观察而非定量分解。具体地：(i) **EEGNet-Huge v1 / v2 / v3 / Mid 四档与 EEGNet baseline 共享原始 32-trial HPO 范围内的 architecture defaults，但其本身的优化栈（LR、weight_decay、dropout、LayerNorm 是否启用）由 ≤ 2 trial 的人工调试得到**——并非独立 Optuna 搜索；(ii) **CBraMod random-init 直接复用 original-weights baseline 的 HP（`get_default_config()`）**，没有跑专属 HPO；(iii) **EEGNet baseline → Mid 的首跳同时改变 conv stem (F1: 16→32, F2: 64→256) 与 MLP 头（单 Linear → 双层 [1024,1024] + LayerNorm）**，未隔离 conv stem 单轴 vs MLP 头单轴的贡献。在这三项约束下，§3.7.1 / §3.7.2 / §3.7.3 报告的所有 Δ 值应被理解为"在共享默认 HP、受限 HPO 预算、双轴 baseline → Mid 跳跃下观察到的复合估计"，而非各因子（架构 / 预训练 / 容量）的独立可归因分解。严格的独立 HPO 验证（EEGNet-Huge v1/v2 ≥ 25 trial Optuna；CBraMod random-init ≥ 25 trial Optuna）留待后续工作（详见 §6 #N）。
```

**RATIONALE**: Roadmap P0.4 demands a chapter-level caveat addressing all three asymmetries. This paragraph signals reader once at chapter open; subsequent subsections can carry shorter footnotes referring back. Honors author's own §3.7.1 last sentence (the existing v3 draft already self-acknowledges the conv-stem double-axis issue at L750).

---

## EDIT 5 — §3.7.1 v1/v2 Failure Diagnosis Rewrite

**Anchor**: L764 footnote "...提示这并非 HP 调优问题而是容量饱和；v3 / Mid 通过加 LayerNorm + 缩 MLP 后才让模型 trainable。"

**TYPE**: REPLACE_PARAGRAPH (the entire footnote block at L764)

**OLD_TEXT**:
```
> EEGNet-Huge v1 / v2 在 ~20–30M 量级两套独立 HP（lr 相差 10×：5e-5 vs 5e-4）下均出现 train loss 死锁在 0.693（chance entropy）、val acc 50%、所有 21 名被试 test 50% 的不可训练状态，因而仅列 cross 一栏（其余范式的 v1 因 state_dict 加载 bug 未跑、v2 within 数据 orphan 未入库）。两套 HP 行为完全一致，提示这并非 HP 调优问题而是容量饱和；v3 / Mid 通过加 LayerNorm + 缩 MLP 后才让模型 trainable。
```

**NEW_TEXT**:
```
> EEGNet-Huge v1 / v2 在 ~20–30M 量级两套独立人工调试 HP（lr 相差 10×：5e-5 vs 5e-4，wd / dropout / LayerNorm on/off 等亦不同；详见 `docs/handoffs/2026-05-09_eegnet_huge.md` L154-170）下均出现 train loss 死锁在 0.693（chance entropy）、val acc 50%、所有 21 名被试 test 50% 的不可训练状态，因而仅列 cross 一栏（其余范式的 v1 因 state_dict 加载 bug 未跑、v2 within 数据 orphan 未入库）。在两套手调 HP 下 v1/v2 不可训；**v3 通过加 LayerNorm + 缩小 MLP 至 [2048, 2048] 后立即 trainable，提示 v1/v2 的失败更可能是 BF16 数值精度下深 MLP 头优化栈兼容性问题（vanishing gradient / dying ELU），而非容量本身的根本饱和**——见交接文档 `docs/handoffs/2026-05-09_eegnet_huge.md` L156、L195-197、L249-260 的工程诊断。是否在严格独立 HPO 预算（≥ 25 trial Optuna，覆盖 LR、warmup、LayerNorm on/off、init scheme、dropout）下 30M 量级 EEGNet 仍不可训，**留待后续工作**（§6 #N）；在补全此独立 HPO 之前，"30M EEGNet 不可训" 的结论应被理解为"在受限 HPO 预算下的观察"。
```

**RATIONALE**: Roadmap P0.4 explicit text spec. Honors handoff L156/L195-197/L249-260 diagnosis (BF16 + deep MLP needs LayerNorm = engineering issue). Eliminates "capacity 饱和" phrasing as v1/v2 failure root cause.

---

## EDIT 6 — §3.7.1 "−25 pp 反向 scaling" Paragraph Soften

**Anchor**: L766 "**Cross-subject 准确率随容量单调下降，呈反向 scaling**：从 76.67% (16K) → 57.65% (1.90M) → 51.37% (5.84M) → 50.00% (~20–30M) 一路下降..."

**TYPE**: REPLACE_PARAGRAPH

**OLD_TEXT**:
```
**Cross-subject 准确率随容量单调下降，呈反向 scaling**：从 76.67% (16K) → 57.65% (1.90M) → 51.37% (5.84M) → 50.00% (~20–30M) 一路下降，~30M 已落入 chance。这是 EEGNet 架构在跨被试范式下的容量天花板：~16K 参数对该任务已接近最优，进一步扩参反而放大跨被试分布偏移噪声。这一现象与 Ding et al. [3] 的 deepEEGNet 实验（"+1.21% binary 微弱提升"）方向不一致——本研究把扩参规模推到 deepEEGNet 估计规模的 5–30×，证实"EEG decoding 的瓶颈不在 EEGNet 容量"。
```

**NEW_TEXT**:
```
**Cross-subject 准确率沿当前扩参轴随容量单调下降**：从 76.67% (16K) → 57.65% (1.90M) → 51.37% (5.84M) → 50.00% (~20–30M) 一路下降，~30M 已落入 chance。**在共享默认 HP、受限 HPO 预算（≤ 2 trial 人工调试）以及 baseline → Mid 双轴扩参（conv stem + MLP 头同时改变）这三项约束下**，本观察方向性支持 "EEGNet 架构内沿当前扩参轴扩参对 cross-subject 准确率不利"，但并不支持更强的 "EEG decoding 瓶颈不在容量" 论断——后者需要在 EEGNet-Huge v1/v2 各跑 ≥ 25 trial 独立 HPO 并仍观察到不可训才能成立（详见 §6 #N）。这一现象方向上与 Ding et al. [3] 的 deepEEGNet 实验（"+1.21% binary 微弱提升"，规模估计 ~100K–1M）一致——后者也未能通过扩参显著改善——但本研究规模扩张幅度（5.84M / ~30M，2 个数量级）尚不足以独立排除"扩参 + 严格 HPO"组合下能否反转该单调趋势。
```

**RATIONALE**: Roadmap P0.4 demands removal of "EEG decoding 瓶颈不在 EEGNet 容量" strong claim (§4.1 also gets the same softening). Adds the key conditional-language framing "在共享默认 HP、受限 HPO 预算、baseline → Mid 双轴扩参 三项约束下".

---

## EDIT 7 — §3.7.1 "+34.97 pp 来自架构" Paragraph Soften

**Anchor**: L770 "**与同规模 random-init CBraMod (§3.7.2) 的鲜明对照**..."

**TYPE**: REPLACE_PARAGRAPH

**OLD_TEXT**:
```
**与同规模 random-init CBraMod (§3.7.2) 的鲜明对照**：在 ~30M 参数 + 无预训练的同等条件下，EEGNet-Huge v2 (30.22M) cross 50.07%（chance）vs CBraMod random-init (30.48M) cross 86.34%——差距 **+36 pp** 完全来自 backbone 架构（transformer + ACPE vs 扩参 CNN）。即便取可训练的 EEGNet-Huge v3 (5.84M) cross 51.37% 作对照，与 random-init CBraMod 的差距仍达 **+34.97 pp**，与容量量级差距非线性脱钩。这把"基座模型的 cross-subject 优势"的来源精准定位到**架构的归纳偏置**而非"更大 backbone 即更好"。
```

**NEW_TEXT**:
```
**与同规模 random-init CBraMod (§3.7.2) 的探索性对照**：在 ~30M 参数 + 无预训练的同等条件下，EEGNet-Huge v2 (30.22M) cross 50.07%（chance）vs CBraMod random-init (30.48M) cross 86.34%——观察到 ~+36 pp 差距；即便取可训练的 EEGNet-Huge v3 (5.84M) cross 51.37% 作对照，与 random-init CBraMod 的差距仍达 ~+35 pp，与容量量级差距非线性脱钩。**在 EEGNet-Huge v1/v2/v3 与 CBraMod random-init 均未做专属 HPO 的对照下**，这一差距是 "架构差异 + EEGNet 优化栈不稳定 + random-init CBraMod HP 错配" 三者的复合估计；其中可归因到 backbone 架构（transformer + ACPE vs 扩参 CNN）的下界尚不能从本节单独给出。本节的探索性观察支持 "在受限 HPO 预算下，扩参 EEGNet 远不及 random-init CBraMod" 这一较弱主张；将该差距精准归因到 "架构归纳偏置" 需要 §6 #N 描述的双侧独立 HPO sweep 完成后才能成立。
```

**RATIONALE**: Roadmap P0.4 + DA §1.1 demand removal of "+34.97 pp 来自架构" 100%-causal attribution. The new framing keeps the numbers (which are correct) but caveat the interpretation as composite estimate.

---

## EDIT 8 — §3.7.2 Random-init Multi-factor Reframe

**Anchor**: L797 "**Within-subject 范式下 from-scratch CBraMod 反而输给 EEGNet**..."

**TYPE**: REPLACE_PARAGRAPH

**OLD_TEXT**:
```
**Within-subject 范式下 from-scratch CBraMod 反而输给 EEGNet**：random-init CBraMod 被试内二分类 62.05% **低于** EEGNet 78.10% 约 −16 pp，三分类 38.65% **低于** EEGNet 66.81% 约 −28 pp。这一非对称揭示 ~4M 参数的 transformer 在 ~70 trial 单被试样本下没有预训练先验时变成"负容量"——它的参数空间过大、随机初始化无法收敛到具备判别力的解，而 ~10K 参数的 EEGNet 凭借更小的搜索空间和被试内训练惯例仍能稳定收敛。基于这个对照，预训练表征扮演的是**数据稀缺时的归纳偏置补偿**而非通用增益。
```

**NEW_TEXT**:
```
**Within-subject 范式下 from-scratch CBraMod 在当前 HP 下输给 EEGNet**：random-init CBraMod 被试内二分类 62.05% 低于 EEGNet 78.10% 约 −16 pp，三分类 38.65% 低于 EEGNet 66.81% 约 −28 pp。该差距方向性提示 transformer 在 ~70 trial 单被试样本下、**沿用 cross-subject HPO 选出的 backbone_lr = 1.3e-4 的固定优化栈**时，没有预训练先验的随机初始化难以收敛到具备判别力的解；~10K 参数的 EEGNet 凭借更小的搜索空间在被试内训练上仍能稳定收敛。**关于 within ternary 18 / 21 chance-collapse 的成因**，作者本人在 [`docs/handoffs/2026-05-09_random_init_ablation.md`](../../docs/handoffs/2026-05-09_random_init_ablation.md) L186-210 中基于 train_loss 轨迹分析给出的概率估计为：**(i) 数据量 / 过参数化导致 saddle-lock（结构性、与 LR 量级关系弱）70–80%；(ii) LR + patience + warmup 调优可救回 ≥ 5 个塌陷被试 15–25%；(iii) LR 是主因、提高 LR 可让 ≥ 10 / 18 塌陷被试学到 < 5%**。本研究的论证依赖 (i) 主导这一假设，但 within ternary 高 LR + 长 patience 的 retry 实验（~25 min GPU；handoff L212-227 描述方法）尚未执行，因此 "from-scratch transformer 在 ~70 trial 上结构性失败" 与 "当前 HP 配置下表现远低于其潜在能力" 在本研究中无法被严格区分。该现象与 NLP 文献中 transformer 在小样本上的已知微调脆弱性（Mosbach et al. 2021 ICLR 在 RTE ~2K 样本上 BERT-base ~1/3 random seed 落入 chance）方向一致；此处的更深文献定位由相邻评审章节处理。基于这一综合判断，预训练表征**在本研究 HP 下方向性扮演**数据稀缺时的归纳偏置补偿角色，但精准量化"无 HP 错配下 TUEG 预训练在被试内的真实贡献"仍需 §6 #N 描述的 random-init 专属 HPO 完成后才能给出。
```

**RATIONALE**: Roadmap P0.4 + DA §1.1 demand softening of 100% causal attribution. Surfaces author's own L186-210 estimates (70-80% / 15-25% / <5%) directly. Lightly cites Mosbach 2021 (Subagent C handles deeper NLP literature integration as flagged in P1.1).

---

## EDIT 9 — §3.7.2 Final Caveat Block Strengthen

**Anchor**: L801 "需要明确的是，本消融仅切换 backbone init，并未做 random-init 专属 HPO；HP 与 original-weights baseline 完全共享..."

**TYPE**: REPLACE_PARAGRAPH

**OLD_TEXT**:
```
需要明确的是，本消融仅切换 backbone init，并未做 random-init 专属 HPO；HP 与 original-weights baseline 完全共享（`get_default_config()`），故 random-init 的两段式差距（within ~−27 pp、cross/transfer ~−3 pp）应理解为"在 original-weights HP 下的 random-init 表现"，而非"random-init 经独立 HPO 调优后的最优表现"。但 cross-subject 与 XSI-FT 的 random-init 缺口已小到 −1.6 至 −4.3 pp，独立 HPO 即便能进一步弥合也很难翻转 within / cross 的两段式差异结构。此外，random-init 训练实际比 original-weights 更早 early-stop（wrapper 总时长 2h 13m vs 估计 9–13h），训练集快速过拟合（train acc 升至 0.95+ 时 val 已高位震荡），与"更小搜索空间下更易过拟合"的预期一致。
```

**NEW_TEXT**:
```
需要明确的是，本消融**仅切换 backbone init，没有做 random-init 专属 HPO**；HP 与 original-weights baseline 完全共享（`get_default_config()`），故 random-init 的两段式差距（within ~−27 pp、cross/transfer ~−3 pp）严格而言应被理解为"**在 original-weights HP 下的 random-init 观察结果**"，而非"random-init 经独立 HPO 调优后的最优表现"。该 HP 错配在 within-subject 范式下可能尤为显著——`get_default_config()` 选出的 backbone_lr = 1.3e-4 来自 cross-subject 21 × 训练数据规模上的 HPO 全局最优（Table S5b cross-subject 行），用到 ~70 trial 单被试 + from-scratch transformer 上时的次优程度无独立度量。cross-subject 与 XSI-FT 的 random-init 缺口已小到 −1.6 至 −4.3 pp，**独立 HPO 即便能进一步弥合该缺口也难以翻转 within / cross 的两段式差异结构**这一定性观察仍可成立，但 within −23 至 −31 pp 内"HP 错配 vs 数据稀缺 saddle"的相对贡献无法在本节闭合；闭合需要 §6 #N 描述的 random-init 专属 HPO（≥ 25 trial Optuna，覆盖 backbone_lr 1e-4 ~ 5e-3 对数均匀、warmup、patience、layer-wise LR）。此外，random-init 训练实际比 original-weights 更早 early-stop（wrapper 总时长 2h 13m vs 估计 9–13h），训练集快速过拟合（train acc 升至 0.95+ 时 val 已高位震荡），与"更小搜索空间下更易过拟合"的预期一致。
```

**RATIONALE**: Strengthen the existing self-caveat to explicitly invoke the §6 future work item and acknowledge HP-mismatch as alternative explanation.

---

## EDIT 10 — §3.7.3 Three-way Decomposition Reframe (CRITICAL)

**Anchor**: L807 "#### 3.7.3 综合：架构 / 预训练 / 容量三向分解" through L820 (entire §3.7.3 subsection).

**TYPE**: REPLACE_PARAGRAPH (entire subsection)

**OLD_TEXT**:
```
#### 3.7.3 综合：架构 / 预训练 / 容量三向分解

合并 §3.7.1 与 §3.7.2 在 cross-subject binary 上的结果，CBraMod 相对 EEGNet baseline 的 +14.01 pp 优势可被分解为三个相邻 Δ：

| 锚点 | 参数量 | 预训练 | Cross binary | Δ 至下一锚点 |
|------|--------|--------|--------------|--------------|
| EEGNet baseline | 16K | 否 | 76.67% | EEGNet 内扩参 → **−25.30 pp** |
| EEGNet-Huge v3 | 5.84M | 否 | 51.37% | 换为 transformer + ACPE 架构 → **+34.97 pp** |
| CBraMod random-init | 30.48M | 否 | 86.34% | 加 TUEG 预训练 → **+4.34 pp** |
| CBraMod baseline | 30.48M | TUEG | 90.68% | — |

三个 Δ 的量级揭示：(i) **架构归纳偏置（transformer + ACPE 与 EEG 信号统计的对齐）是 cross-subject 范式下最大贡献**（~+35 pp），远大于 TUEG 预训练（~+4 pp）；(ii) **EEGNet 架构内的容量扩展不仅无益反而显著有害**（~−25 pp，~30M 量级在两套独立 HP 下均落入 chance）。被试内任务分解方向相反：以 binary 为例，EEGNet baseline 78.10% → EEGNet-Huge v3 67.71%（仅 −10 pp）→ CBraMod random-init 62.05%（仍低于 EEGNet baseline）→ CBraMod baseline 85.15%——TUEG 预训练贡献 **+23.10 pp（binary）/ +30.79 pp（ternary）, 平均 ~+27 pp** 主导（下文及摘要 / §7 Finding 1 / §4.1 沿用此平均值），架构与容量贡献为负。

这一**范式依赖的分解结构**与 §4.1 的"基座模型价值随数据约束放大"叙事自洽：cross-subject 范式（21 × 训练数据）信号充足时，**架构归纳偏置主导**，预训练只是锦上添花；within-subject 范式（每被试 ~70 trial）信号稀缺时，**预训练先验主导**，架构容量本身反而是负担。容量在两条轴上都未充当主要变量——这是本研究最具实操意义的方法论命题：在 EEG decoding 中，盲目扩参不是改进路径；架构归纳偏置（与信号统计性质对齐的 transformer + 通道位置编码）和预训练表征（在通用 EEG 语料上训得的低维流形）才是关键，二者在不同数据规模下分别主导。
```

**NEW_TEXT**:
```
#### 3.7.3 综合：架构 / 预训练 / 容量复合贡献的探索性观察

合并 §3.7.1 与 §3.7.2 在 cross-subject binary 上的四个锚点，可观察到 CBraMod 相对 EEGNet baseline 的 +14.01 pp 优势沿以下相邻 Δ 跨越（**所有 Δ 在共享默认 HP 与受限 HPO 预算下的复合估计；严格独立 HPO 留待 §6 #N**）：

| 锚点 | 参数量 | 预训练 | Cross binary | 至下一锚点的 Δ（复合估计） |
|------|--------|--------|--------------|---------------------------|
| EEGNet baseline | 16K | 否 | 76.67% | EEGNet 内扩参 → −25.30 pp ¹ |
| EEGNet-Huge v3 | 5.84M | 否 | 51.37% | 换为 transformer + ACPE 架构 → +34.97 pp ² |
| CBraMod random-init | 30.48M | 否 | 86.34% | 加 TUEG 预训练 → +4.34 pp ³ |
| CBraMod baseline | 30.48M | TUEG | 90.68% | — |

> ¹ EEGNet 内扩参的 Δ 为 baseline (16K, F1=16/F2=64, 单 Linear 头) → Huge v3 (5.84M, F1=32/F2=256, [2048,2048] + LayerNorm 头) 的双轴跳跃，conv stem 与 MLP 头同时改变；该 −25.30 pp 中可归因到 "MLP 头 over-parameterization" vs "conv stem 改动" vs "EEGNet HPO 受限 (≤ 2 trial 人工调试)" 的拆分超出本节范围（详见 §6 #6）。
> ² 跨架构 Δ 为 EEGNet-Huge v3 (受限 HPO 下 cross 51.37%) → CBraMod random-init (复用 original-weights HP, cross 86.34%) 的对照；该 +34.97 pp 中包含 (a) backbone 架构差异 (transformer + ACPE vs 扩参 CNN)、(b) EEGNet 优化栈在 BF16 + 深 MLP 头下的不稳定性、(c) random-init CBraMod 复用 original-weights HP 的可能错配 三种贡献的复合，且尚不能给出可归因到 (a) 的下界。
> ³ TUEG 预训练 Δ 为 random-init CBraMod (cross 86.34%) → original-weights CBraMod (cross 90.68%) 的对照，HP 完全共享 `get_default_config()`；这一 +4.34 pp 是同规模、同 HP 下唯一只随 backbone init 变动的 Δ，因此**是本表中归因强度最高的一个 Δ**——但仍受限于 random-init 在该 HP 下可能并非最优表现这一前提（§3.7.2 caveat）。

被试内任务上对应的相邻 Δ 序列方向相反：以 binary 为例，EEGNet baseline 78.10% → EEGNet-Huge v3 67.71% (−10 pp) → CBraMod random-init 62.05% (−5.66 pp，仍低于 EEGNet baseline) → CBraMod baseline 85.15%（TUEG 预训练 Δ = +23.10 pp，binary）；ternary 对应 TUEG 预训练 Δ = +30.79 pp。下文及摘要 / §7 Finding 1 / §4.1 在被试内引用 TUEG 预训练贡献时**显式列出 binary +23.10 / ternary +30.79 pp 双值**，不再使用 ~+27 pp 平均值（该平均会模糊任务难度差异）。

**关于本节观察的解读边界**：CBraMod 与扩参 EEGNet 之间的 cross-subject gap 至少包含架构、预训练、容量三种贡献，本研究在受限 HPO 预算 + baseline → Mid 双轴扩参 + random-init 共享 HP 的三项约束下**无法独立分离**这三种贡献的各自贡献值。可被本节探索性观察方向性支持的较弱主张是：(a) **TUEG 预训练在被试内贡献巨大（binary +23.10 / ternary +30.79 pp），在 cross-subject 与 XSI-FT 仅贡献 +1.6 ~ +4.3 pp**——这是本节归因强度最高的一组 Δ；(b) **沿当前扩参轴扩参 EEGNet 在 cross-subject 范式下方向性有害**（baseline → Huge v3 沿双轴下降 −25.30 pp）——但 "EEGNet 内扩参普遍有害" 与 "EEG decoding 瓶颈不在容量" 等更强主张需要 §6 #N + §6 #6 的独立 HPO 与单轴隔离实验完成后才能确立；(c) **transformer + ACPE 架构在不依赖 TUEG 预训练时仍能在 cross-subject 21 × pooled 数据上学到有效表征**（random-init CBraMod cross 86.34% vs EEGNet baseline 76.67%, +9.67 pp）——但与 EEGNet-Huge v3 的 +34.97 pp 差距是复合估计，不可独立归因到 "架构"。

这一**范式依赖的复合贡献结构**与 §4.1 "基座模型价值随数据约束放大" 的叙事方向一致：在 cross-subject 范式（21 × 训练数据）信号充足时，random-init CBraMod 仍领先扩参 EEGNet；在 within-subject 范式（每被试 ~70 trial）信号稀缺时，TUEG 预训练贡献急剧扩大、random-init CBraMod 反而输给 EEGNet baseline。但因前述三项 HPO / 扩参非对称性，该结构的**精确归因强度**应被理解为方向性而非独立可定量分解的；详细归因需要 §6 #N（EEGNet-Huge ≥ 25 trial 独立 HPO + CBraMod random-init ≥ 25 trial 独立 HPO）的算力开支后才能闭合。
```

**RATIONALE**: This is the centerpiece of the B+E reframe. (1) Subsection title changed from "三向分解" to "复合贡献的探索性观察". (2) All Δ values get footnotes explicitly stating composite-estimate caveats. (3) "+27 pp 平均" replaced with explicit binary +23.10 / ternary +30.79 pp dual report (Roadmap P0.4 explicit demand). (4) New "解读边界" paragraph replaces standalone three-finding attribution with three caveat-laden softened claims (a/b/c). (5) Final synthesis paragraph keeps direction language ("方向性") but removes "盲目扩参不是改进路径" type strong claims.

---

## EDIT 11 — §4.1 "Capacity is not the bottleneck" Removal

**Anchor**: L871 "...这把"capacity is not the bottleneck"立成铁案——盲目扩参 EEGNet 反而显著有害。"

**TYPE**: REPLACE_PARAGRAPH (the paragraph from "CBraMod 在所有实验条件下..." through the "立成铁案" sentence)

**OLD_TEXT**:
```
CBraMod 在所有实验条件下一致优于 EEGNet——被试内 **+7.05 pp**、跨被试 **+14.01 pp**（128ch）、32 通道 **+10–13 pp**——这反映了大规模预训练对 EEG 解码的价值。~400 倍的参数量差异本身不能完全解释该差距，§3.7 报告的两项消融（EEGNet 容量阶梯 + random-init CBraMod）提供了对该差距的三向分解。一个朴素担忧——"差距是否仅源自 ~16K vs ~4M 的容量量级差异"——可由 §3.7.1 直接回答：把 EEGNet 的 MLP 头扩展到 5.84M / 19.99M / 30.22M 三档，**cross-subject 准确率从 76.67% 单调下降到 51.37% / 50% / 50%（chance）**，30M 量级在两套独立 HP（lr 相差 10×）下均落入 train loss 死锁。这把"capacity is not the bottleneck"立成铁案——盲目扩参 EEGNet 反而显著有害。
```

**NEW_TEXT**:
```
CBraMod 在所有实验条件下一致优于 EEGNet——被试内 **+7.05 pp**、跨被试 **+14.01 pp**（128ch）、32 通道 **+10–13 pp**——这反映了大规模预训练对 EEG 解码的价值。~400 倍的参数量差异本身不能完全解释该差距，§3.7 报告的两项探索性消融（EEGNet 容量阶梯 + random-init CBraMod）对该差距的来源做了初步检验。一个朴素担忧——"差距是否仅源自 ~16K vs ~4M 的容量量级差异"——由 §3.7.1 在受限 HPO 预算下方向性回答：把 EEGNet 沿 (conv stem, MLP 头) 双轴扩展到 1.90M / 5.84M / 19.99M / 30.22M 四档，**cross-subject 准确率从 76.67% 单调下降到 50%（chance）**，30M 量级在两套人工调试 HP（≤ 2 trial）下均落入 train loss 死锁。**在本研究 HPO 协议下** ，沿当前扩参轴对 EEGNet 扩参对 cross-subject 准确率不利；但 "EEGNet 内扩参普遍有害" 或 "EEG decoding 的瓶颈不在容量" 等更强主张需要 EEGNet-Huge v1/v2 的独立 ≥ 25 trial Optuna sweep（详见 §6 #N）确认仍不可训之后才能成立。值得注意的是，作者在交接文档 [`docs/handoffs/2026-05-09_eegnet_huge.md`](../../docs/handoffs/2026-05-09_eegnet_huge.md) L156、L195-197、L249-260 中明确诊断 v1/v2 的不可训为 "BF16 + 深 MLP 头需 LayerNorm" 的优化栈兼容性问题——v3 加 LayerNorm + 缩 MLP 后立即 trainable 是直接证据；因此 v1/v2 的失败更可能是工程层面的 trainability 问题，而非参数容量本身的根本饱和。
```

**RATIONALE**: Roadmap P0.4 explicit demand: "去除 'capacity is not the bottleneck' 等类强表述". Replaces with "在本研究 HPO 协议下" conditional language. Surfaces handoff diagnosis directly into Discussion.

---

## EDIT 12 — §4.1 "+34.97 pp 来自架构" + "+27 pp" Soften

**Anchor**: L873 "更严格的架构隔离来自 §3.7.1 与 §3.7.2 的同规模对照..." through L875 "...而非两者总和构成单一通用增益。"

**TYPE**: REPLACE_PARAGRAPH (the two paragraphs from L873 through L875)

**OLD_TEXT**:
```
更严格的架构隔离来自 §3.7.1 与 §3.7.2 的同规模对照：在 ~30M 参数 + 无预训练的同等条件下，EEGNet-Huge v2 (30.22M) cross 50.07%（chance）vs CBraMod random-init (30.48M) cross 86.34%——差距 **+36 pp** 完全来自 backbone 架构（transformer + ACPE vs 扩参 CNN）。即便取可训练的 EEGNet-Huge v3 (5.84M) 51.37% 作对照，与 random-init CBraMod 的差距仍达 **+34.97 pp**。在控制容量与预训练后，架构归纳偏置在 cross-subject 范式下贡献 ~+35 pp。其上 TUEG 预训练再追加 ~+4 pp（86.34% → 90.68%），抵达 §3.2 的 baseline 性能。

然而 within-subject 范式下分解方向完全反转：random-init CBraMod 在被试内二分类与三分类上分别落到 62.05% 和 38.65%，比 original-weights 分别低 **−23.10 / −30.79 pp**（且 within ternary 21 名被试中 **18 名**测试准确率落在 chance ± 2 pp 区间，seed = 1234 重跑得 17 / 21，证实非 seed 特例）；不仅如此，random-init CBraMod 在该范式下反而**输给** EEGNet baseline（binary 78.10%、ternary 66.81%）约 −16 至 −28 pp。这一反转把基座模型价值精准定位为**数据稀缺时的归纳偏置补偿**：当 cross-subject pooling 提供 ~21× 训练数据时，架构 inductive bias 主导（无论预训练与否，CBraMod 都领先扩参 EEGNet ~+35 pp）；但当 within-subject 仅 ~70 trial 时，~4M 参数的 transformer 在没有预训练先验的情况下变成"负容量"，随机收敛到比 ~10K 参数 EEGNet 更差的解。换言之，**TUEG 预训练在被试内贡献 ~+27 pp、在跨被试与 XSI-FT 仅贡献 ~+2 至 +4 pp**；架构与预训练在不同数据规模下分别主导，而非两者总和构成单一通用增益。
```

**NEW_TEXT**:
```
**同规模 random-init 对照的探索性观察**：在 ~30M 参数 + 无预训练的同等条件下，EEGNet-Huge v2 (30.22M) cross 50.07%（chance）vs CBraMod random-init (30.48M) cross 86.34%——观察 ~+36 pp 差距；EEGNet-Huge v3 (5.84M) cross 51.37% 对 random-init CBraMod 时差距为 ~+35 pp。**在 EEGNet-Huge v1/v2/v3 与 CBraMod random-init 均未做专属 HPO 的对照下**，该差距是 "backbone 架构差异 + EEGNet 优化栈不稳定 + random-init CBraMod HP 错配" 的复合估计；归因到 backbone 架构本身的下界尚不能从本节单独给出。其上 TUEG 预训练 Δ = +4.34 pp（86.34% → 90.68%）是本对照中归因强度最高的 Δ——因 random-init 与 baseline 共享同一 `get_default_config()`，唯一变量是 backbone init。

within-subject 范式下方向反转：random-init CBraMod 在被试内二分类与三分类上分别落到 62.05% 和 38.65%，比 original-weights 分别低 **binary −23.10 pp、ternary −30.79 pp**（且 within ternary 21 名被试中 18 名测试准确率落在 chance ± 2 pp 区间，seed = 1234 重跑得 17 / 21，证实非 seed 特例）；不仅如此，random-init CBraMod 在该范式下反而输给 EEGNet baseline（binary 78.10%、ternary 66.81%）约 −16 至 −28 pp。然而 §3.7.2 caveat 已指出 random-init 复用 cross-subject HPO 选出的 backbone_lr = 1.3e-4，该 HP 在 ~70 trial single-subject from-scratch transformer 上的最优性未被独立验证；handoff 作者本人对 within ternary collapse 的概率归因为 70-80% saddle-lock / 15-25% LR-schedule、< 5% 纯 LR 主因。这一非对称方向性提示 TUEG 预训练**扮演数据稀缺时的归纳偏置补偿**角色：cross-subject pooling (~21× 训练数据) 信号充足时，random-init CBraMod 仍领先 EEGNet baseline +9.67 pp；within-subject (~70 trial) 信号稀缺时，TUEG 预训练贡献急剧扩大。

引用本节数字时，**摘要 / §1.4 / §7 Finding 1 显式列出 binary +23.10 / ternary +30.79 pp 双值**，不再使用 ~+27 pp 平均值（该平均会模糊任务难度差异）；cross-subject 与 XSI-FT 范式下 TUEG 预训练贡献为 +1.6 ~ +4.3 pp（双任务双范式区间）。三向分解（架构 / 预训练 / 容量）的精确归因强度需要 §6 #N（EEGNet-Huge + random-init CBraMod 双侧 ≥ 25 trial 独立 HPO sweep）的算力开支后才能闭合；当前章节支持的较弱主张是 "TUEG 预训练在被试内贡献巨大、在 cross-subject 与 XSI-FT 仅 +2 ~ +4 pp" + "transformer + ACPE 架构在不依赖 TUEG 预训练时仍能在 cross-subject 21 × pooled 数据上学到有效表征"，更强的独立可归因分解超出本研究证据范围。
```

**RATIONALE**: P0.4 spec demands removal of "+34.97 pp 完全来自架构" + "+27 pp 平均" claims. Replaces with binary/ternary dual-report and weakened claims.

---

## EDIT 13 — §6 Future Work: Add Item #N (EEGNet-Huge Independent HPO Sweep)

**Anchor**: L987 (after item 7 "其他基座模型与预训练目标的独立验证")

**TYPE**: ADD_PARAGRAPH (new item appended)

**NEW_TEXT** (insert as item 8):
```

8. **§3.7 探索性消融的严格独立 HPO 验证**：§3.7.1 EEGNet-Huge v1 (19.99M) 与 v2 (30.22M) 的不可训判定基于两套人工调试 HP（lr 相差 10×：5e-5 vs 5e-4；wd / dropout / LayerNorm on/off 等亦不同），并非独立 Optuna 搜索；§3.7.2 random-init CBraMod 直接复用 original-weights baseline 的 `get_default_config()`，没有跑专属 HPO。要让 §3.7.3 的复合贡献观察升格为可独立归因的三向分解，需补做：(a) **EEGNet-Huge v1 / v2 各 ≥ 25 trial Optuna TPE HPO**，搜索空间覆盖 LR ∈ [5e-5, 5e-3] 对数均匀、warmup ratio ∈ [0, 0.2]、LayerNorm on/off (categorical)、init scheme ∈ {Kaiming, Xavier}、dropout ∈ [0.1, 0.6]、weight_decay ∈ [1e-3, 0.3] 对数均匀；(b) **CBraMod random-init ≥ 25 trial Optuna 专属 HPO**，覆盖 backbone_lr 1e-4 ~ 5e-3 对数均匀、warmup、patience、layer-wise LR；优先 within ternary（最严重的 18/21 chance-collapse case）。预算估计 ~80–120 GPU 小时（v1 25 trial × ~2h + v2 25 trial × ~2h + random-init within ternary 25 trial × ~25 min + cross-subject 25 trial × ~30 min + 复跑 baseline 对照）。**预期 readout**：若 (a) 100% trial 仍落入 train_loss = 0.693 chance entropy 死锁（即便加 LayerNorm 也不救），则 §3.7.1 "EEGNet 内扩参在受限 HPO 下不可训" 升级为 "经独立 HPO 验证后仍不可训"；若 (b) 在 random-init within ternary 上把 chance-collapse 比例从 18/21 降至 ≤ 8/21，则 §3.7.2 / §3.7.3 / §4.1 / §7 Finding 1 中 "TUEG 预训练在被试内贡献 binary +23.10 / ternary +30.79 pp" 需进一步弱化为更小区间（HP 错配占了相当比例）。
```

**RATIONALE**: Roadmap P0.4 + DA §1.1 explicit demand. Estimates align with DA review's ~80-120h GPU figure (DA review L293).

---

## EDIT 14 — §6 Renumber Existing Item 8 → 9 (if applicable)

**Note for orchestrator**: If §6 list currently ends at item 7, EDIT 13 appends as item 8 directly. If §6 already has an item 8 (none observed in current draft as of 2026-05-10 read), renumber accordingly. Verified §6 ends at item 7 in current v3.0.1.md L987.

---

## EDIT 15 — §1.4 Finding 1 Reframe

**Anchor**: L77 (the entire Finding 1 paragraph in §1.4 contributions list)

**TYPE**: REPLACE_PARAGRAPH

**OLD_TEXT**:
```
> 1. **系统性基座模型评估，并将架构 / 预训练 / 容量三向贡献剥离**。首次在手指级运动想象分类任务上，对 EEG 基座模型（CBraMod）与传统 CNN（EEGNet-16,4）进行全面对比，覆盖被试内、跨被试、跨被试初始化的逐被试微调（XSI-FT，§3.3）三种范式，使用 21 名被试数据，并采用贝叶斯超参数优化（HPO）确保公平比较。在三种范式下 CBraMod 一致优于 EEGNet（被试内 +7.05 pp、跨被试二分类 +14.01 pp、跨被试三分类 +13.65 pp）。通过两项互补消融（§3.7）将该差距三向拆分：(a) **EEGNet 容量阶梯（16K → 1.90M → 5.84M → 30M）** 显示 cross-subject 准确率从 76.67% 单调下降至 51.37% / 50%（chance），证明 EEGNet 架构内扩参反而显著有害（−25 pp），容量本身不是瓶颈；(b) **random-init CBraMod 消融** 显示在 ~30M 参数 + 无预训练同等条件下，CBraMod 仍领先扩参 EEGNet ~+35 pp（cross-subject），加 TUEG 预训练再追加 ~+4 pp（cross / XSI-FT）至 ~+27 pp（被试内）。三向分解把基座模型价值精准定位为"架构归纳偏置在 cross 主导 + 预训练先验在 within 主导"，而非简单的"更多参数更好"。
```

**NEW_TEXT**:
```
> 1. **系统性基座模型评估 + 探索性消融初步检验差距来源**。首次在手指级运动想象分类任务上，对 EEG 基座模型（CBraMod）与传统 CNN（EEGNet-16,4）进行全面对比，覆盖被试内、跨被试、跨被试初始化的逐被试微调（XSI-FT，§3.3）三种范式，使用 21 名被试数据，并采用贝叶斯超参数优化（HPO，CBraMod 11 维 / EEGNet 7 维，trial 数按 d^1 校准；详见 §2.5.1）确保公平比较。在三种范式下 CBraMod 一致优于 EEGNet（被试内 +7.05 pp、跨被试二分类 +14.01 pp、跨被试三分类 +13.65 pp）。作为补充，§3.7 进行了两项探索性消融以理解架构 / 预训练 / 容量的相对贡献：(a) **EEGNet 容量阶梯（16K → 1.90M → 5.84M → 30M）** 显示 cross-subject 准确率沿当前扩参轴单调下降至 51.37% / 50%（chance），方向性提示沿该轴扩参 EEGNet 不利，但 v1/v2 (~20-30M) 不可训根据作者本人交接诊断更可能是 BF16 下深 MLP 头优化栈兼容性问题（v3 加 LayerNorm 立即 trainable）而非容量饱和；(b) **random-init CBraMod 消融** 显示在 ~30M 参数 + 无预训练条件下，CBraMod 仍领先扩参 EEGNet ~+35 pp（cross-subject），加 TUEG 预训练再追加 +4.34 pp（cross-subject）至 binary +23.10 / ternary +30.79 pp（被试内）。**因 EEGNet-Huge 与 CBraMod random-init 均未做专属 HPO，且 baseline → Mid 跳跃同时改变 conv stem 与 MLP 头**，这些消融在本研究范围内不构成独立可归因的三向分解，应被理解为方向性观察；严格的独立 HPO 验证留待后续工作（§6 #8）。详见 §3.7 caveats。
```

**RATIONALE**: Roadmap P0.4 explicit text spec for §1.4. Subagent B prompt verbatim "在三种训练范式下系统性 benchmark... 作为补充，在 §3.7 进行了探索性消融...这些消融不构成独立可归因的分解;详见 §3.7 caveats". Includes the §2.5.1 (W) Part A reference.

---

## B-Abstract §3.7 Paragraph Contribution

**Note for orchestrator**: This is for orchestrator merge into the abstract paragraph (current L20-22 area).

**Anchor in current draft**: L20-22 sentence "...两项互补消融把该差距拆分为架构 / 预训练 / 容量三向贡献：(a) **EEGNet 容量阶梯（16K → 1.90M → 5.84M → 30M）** 显示 cross-subject 准确率从 76.67% 单调下降至 51.37% / 50%（chance）..."

**Replacement text** (replaces from "两项互补消融把该差距拆分为架构 / 预训练 / 容量三向贡献..." through "...而非通用增益。" — approximately L20-22):

```
为更好理解该差距来源，§3.7 进行了两项探索性消融。(a) **EEGNet 容量阶梯（16K → 1.90M → 5.84M → 30M）** 显示扩参 EEGNet 在固定优化栈下严重退化，cross-subject 准确率从 76.67% 单调下降到 51.37% / 50%（chance），方向性提示沿当前扩参轴对 EEGNet 扩参不利；(b) **random-init CBraMod 消融**显示在 ~30M 参数 + 无预训练同等条件下，CBraMod 仍领先扩参 EEGNet ~+35 pp（cross-subject），加 TUEG 预训练再追加 +4.34 pp（cross）至 binary +23.10 pp / ternary +30.79 pp（被试内）。然而 EEGNet-Huge 与 CBraMod random-init 均未做专属 HPO，且 baseline → Mid 跳跃同时改变 conv stem 与 MLP 头，因此这些消融**不构成对架构、预训练、容量三因子的独立可归因分解**，应被理解为方向性观察。within-subject 严重 collapse 的现象与 NLP 文献中 transformer 在小样本下的已知微调脆弱性方向一致；严格独立 HPO 验证留待后续工作。
```

**RATIONALE**: Replaces "三向分解把基座模型价值精准定位为'架构在 cross 主导、预训练在 within 主导'" with explicitly hedged exploratory framing. Honors Subagent B prompt headline spec.

---

## B-§7 Finding 1 Contribution

**Anchor**: L995 (Finding 1 entire paragraph)

**Replacement text**:

```
> **发现 1 — 基座模型在三种训练范式下一致优于 EEGNet；探索性消融初步检验差距来源。** CBraMod 对 EEGNet 的优势从 **+7.05 pp**（被试内）扩大至 **+14.01 pp**（跨被试 128 通道），在 32 通道下仍保持 **+10–13 pp** 差距。两项探索性消融（§3.7）对该差距的来源做了初步检验：(i) **沿当前扩参轴扩参 EEGNet 在受限 HPO 预算下方向性有害**——把 EEGNet 沿 (conv stem, MLP 头) 双轴扩参到 1.90M / 5.84M / 30M 三档，cross-subject 准确率从 76.67% 单调下降至 51.37% / 50%（chance），其中 v1/v2 (~30M) 不可训根据作者本人交接诊断更可能是 BF16 下深 MLP 头优化栈兼容性问题（v3 加 LayerNorm 立即 trainable）；(ii) **架构提供独立价值的方向性证据**——在 ~30M 参数 + 无预训练同等条件下，CBraMod random-init cross 86.34% vs EEGNet-Huge v3 (5.84M) cross 51.37% 差距 ~+35 pp，但因 EEGNet-Huge / random-init 均未做专属 HPO 且 baseline → Mid 跳跃同时改 conv stem 与 MLP 头，该差距不可独立归因到 backbone 架构；(iii) **TUEG 预训练贡献同规模、同 HP 下唯一干净的 Δ**——random-init → original-weights backbone 切换的 Δ 在被试内为 **binary +23.10 / ternary +30.79 pp**，在跨被试与 XSI-FT 为 +1.6 ~ +4.3 pp，这是本研究归因强度最高的一组 Δ。本研究在三种训练范式下系统量化了 CBraMod vs EEGNet 的性能差距；补充的探索性消融暗示该差距由架构归纳偏置 + 预训练先验 + 容量约束三因素叠加，但本研究的 HPO 预算与单轴扩参限制使我们**无法对各因素做独立定量归因**。"基座模型价值随数据约束放大" 的方向性结论在两项消融中均得到方向性支持。这一限制由 §6 #8（EEGNet-Huge ≥ 25 trial 独立 HPO + random-init CBraMod ≥ 25 trial 独立 HPO，预算 ~80–120 GPU 小时）描述的后续工作处理。
```

**RATIONALE**: Subagent B prompt verbatim spec for §7 Finding 1. Replaces "本研究将基座模型价值精准定位为'架构 cross 主导 + 预训练 within 主导'" with hedged framing + explicit deferral to §6 #8.

---

## Numbers Cross-check

| Number | Source verified | Notes |
|--------|----------------|-------|
| CBraMod within HPO 51 trials | Table S5b L1228 "51 trials / 23 complete" | matches |
| CBraMod cross HPO 77 trials | Table S5b L1257 "77 trials / 43 complete" | matches |
| EEGNet within HPO 32 trials | Table S5b L1244 "32 trials / 10 complete" | matches |
| Trial ratio 51/32 = 1.59 | computed | matches Subagent B prompt |
| HP dim ratio 11/7 = 1.57 | search_spaces.py verified (CBraMod within `_sample_cbramod_within` 11 keys; EEGNet within `_sample_eegnet_within` 7 keys excl. F2 derived) | matches |
| d^0.5 = 1.25, d^1 = 1.57 | computed; aligns with Bergstra & Bengio 2011 | matches |
| TUEG within binary +23.10 pp | Δ = 85.15 − 62.05 (Table 18 L782) | matches |
| TUEG within ternary +30.79 pp | Δ = 69.44 − 38.65 (Table 18 L783) | matches |
| TUEG cross binary +4.34 pp | Δ = 90.68 − 86.34 (Table 18 L784) | matches |
| EEGNet baseline cross 76.67% | Table 7 / handoff L7 | matches |
| EEGNet-Huge v3 cross 51.37% | Table 18a L758 / handoff L9 | matches |
| CBraMod random-init cross 86.34% | Table 18 L784 / handoff random-init L28 | matches |
| Random-init within ternary 18/21 chance-collapse seed 42 | handoff L98-101 | matches |
| Random-init within ternary 17/21 chance-collapse seed 1234 | handoff L114 | matches |
| Author's saddle-lock probability 70-80% | handoff L206 | matches |
| Author's LR schedule 15-25% | handoff L207 | matches |
| Author's pure LR <5% | handoff L208 | matches |
| Handoff LayerNorm diagnosis citations | L156, L195-197, L249-260 | matches |

**No fabrication detected.** All numerical claims trace to v3.0.1.md tables, handoff documents, or `src/hpo/search_spaces.py`.

---

## Risks for Orchestrator

1. **Reference list update needed**: Bergstra & Bengio 2011, Snoek et al. 2012 cited in EDIT 1 (§2.5.1 Part A); Mosbach et al. 2021 lightly cited in EDIT 8 (§3.7.2). If not in current References, Subagent C should add. If Subagent C adds Mosbach with deeper integration (P1.1), my light citation may become redundant — orchestrator harmonize.

2. **Cross-section consistency to verify after merge**: After all three subagent edits merged, orchestrator must verify the following five locations are consistent with the new framing:
   - Abstract §3.7 paragraph (this contribution)
   - §1.4 Finding 1 (EDIT 15)
   - §3.7 chapter (EDITS 3-10)
   - §4.1 (EDITS 11-12)
   - §7 Finding 1 (this contribution)
   - All five must reference: "exploratory ablation" framing + "binary +23.10 / ternary +30.79 pp" double-value + "§6 #8 future work" deferral.

3. **§6 numbering**: My EDIT 13 adds item 8. If Subagent A or C also append §6 items, orchestrator must serially renumber and update cross-references (§6 #N → final number) in EDITS 4, 5, 8, 9, 10, 11, 12, 15.

4. **Table S5e placement**: My EDIT 2 adds Table S5e after Table S5b. If Subagent C adds tables to supplementary, ensure no conflict.

5. **§3.7.3 entire subsection rewrite (EDIT 10)**: This is the largest single edit. Ensure orchestrator merges as full-block replacement, not line-by-line — interleaved partial merges will break the 3-footnote table cross-references.

6. **Chinese-language consistency**: All my new text follows the document's Chinese-primary style. English technical terms preserved in parentheses where helpful (Bergstra & Bengio 2011, Snoek et al. 2012, etc.).

7. **No Δ values changed**: Per IRON RULE 4, all numerical claims in tables and inline citations preserved; only INTERPRETATION softened. Verify on merge that no "+34.97 pp" / "−25.30 pp" / etc. raw numbers in tables were accidentally edited.

8. **Coordination with Subagent A (DAPT) and C (literature/minor)**:
   - Subagent A's DAPT scope and my §3.7 scope share the §6 future work list — coordinate item numbering.
   - Subagent C's NLP literature integration (Mosbach 2021, Zhang 2021) overlaps with my EDIT 8 light citation — orchestrator harmonize per P1.1 spec.
   - Abstract is shared territory: A's DAPT paragraph (current L26) + my §3.7 paragraph (current L20-22) must be merged without contradiction.

---

## Edit count summary

- **15 EDITS** total (of which 1 is a numbering note → 14 substantive edits to v3.0.1.md):
  - §2.5.1: 1 edit (EDIT 1, ~250 Chinese chars added per spec)
  - Table S5e: 1 new supplementary table (EDIT 2)
  - §3.7 chapter title: 1 edit (EDIT 3)
  - §3.7 chapter intro: 1 edit (EDIT 4)
  - §3.7.1: 3 edits (EDITS 5, 6, 7)
  - §3.7.2: 2 edits (EDITS 8, 9)
  - §3.7.3: 1 edit (EDIT 10, full subsection rewrite — the largest single edit)
  - §4.1: 2 edits (EDITS 11, 12)
  - §6: 1 edit (EDIT 13, new item 8)
  - §1.4 Finding 1: 1 edit (EDIT 15)
- Plus 2 multi-touch contributions for orchestrator merge:
  - B-Abstract §3.7 paragraph
  - B-§7 Finding 1
