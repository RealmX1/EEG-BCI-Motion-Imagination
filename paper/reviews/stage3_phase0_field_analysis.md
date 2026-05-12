# Stage 3 Phase 0 — Field Analysis Report

**论文**: 基于 EEG 基座模型的手指级运动想象分类：通道缩减、纵向数据扩展与领域自适应预训练的局限性
**作者**: Bomin Zhang (硕士论文候选人)
**草稿**: `paper/drafts/paper_draft_v3.0.1.md` (~13,548 词 / 1,357 行)
**Phase 0 分析日期**: 2026-05-10
**分析师**: field_analyst_agent (ARS academic-paper-reviewer Phase 0, full mode)

---

## 1. 论文基本信息

| 维度 | 值 |
|------|-----|
| 标题 | 基于 EEG 基座模型的手指级运动想象分类：通道缩减、纵向数据扩展与领域自适应预训练的局限性 |
| 作者 | Bomin Zhang (单一作者) |
| 字数 | ~13,548 中文词（含图表说明、补充材料） |
| 主体长度 | 7 节 (Intro / Methods / Results / Discussion / Limitations / Future Work / Conclusion) + 7 张补充表 + 1 张补充图 |
| 参考文献 | 9 条（Phase A 已核验，1 条 MEDIUM 级 PH 错配，待修订） |
| 实验数据规模 | 21 名被试、128 通道、1024 Hz；43+ run_tag 由 ExperimentDB 管理；DAPT 使用 870h / ~300 被试外部 MI 语料 |
| 草稿状态 | v3.0.1（v3 完整性修复版本，pre-Stage-3 review-ready） |
| 语言 | 中文为主，技术术语保留英文 |

---

## 2. 6 维场域分析

| 维度 | 判定 | 依据 |
|------|------|------|
| **1. 主学科 (Primary Discipline)** | **Neural Engineering / Brain-Computer Interface (神经工程 / 脑机接口)** | 论文聚焦 finger-level MI-BCI 解码，整篇工作核心是"BCI 系统的 backbone 选择 + 通道缩减 + 部署可行性"——典型 *Journal of Neural Engineering* / *IEEE TNSRE* 命题。 |
| **2. 副学科 (Secondary Disciplines)** | (a) **Deep Learning for Time-Series / EEG signal processing**（基座模型 vs CNN 对比、HPO、capacity ablation）；(b) **Cognitive Neuroscience of Motor Control**（mu/beta ERD、sensorimotor cortex、手部 hand-knob）；(c) **Transfer Learning / Foundation Models**（DAPT 负迁移、cross-subject pooling、XSI-FT） | §1.3 / §2.4 / §2.7 / §3.6 / §3.7 / §4.5 都横跨这三层；§4.8 明确以"EEG foundation model 的 domain 边界由信号级特征定义"作为方法论命题。 |
| **3. 研究范式 (Research Paradigm)** | **定量经验研究 (Quantitative Empirical)** | 全文 41+ 数值已与 ExperimentDB 双源核验（Phase C），主结论由 paired t-test、effect size 与 ablation 证据链支撑；无定性 / 综述 / 理论性章节。 |
| **4. 方法论类型 (Methodology Type)** | **Comparative Benchmarking + Systematic Ablation + Negative-Result Reporting** | 主线：在统一 held-out session 框架内做 (model × paradigm × task × channel × session-progression × pretrain-strategy) 多维笛卡尔对比；§3.7 capacity ladder + random-init 双消融、§3.6 DAPT 三配置（V1/V2/V3）、§3.5.3 4ch negative-control 是该范式的特征。 |
| **5. 目标期刊层级 (Target Journal Tier)** | **Q2 (mainstream 神经工程 / BCI)**，最高至 **Q1-边缘**（如 NeuroImage / IEEE TBME 较高分位） | 数据集为单一 21 人 cohort（限制泛化主张范围），无独立 cohort 复现，单作者工作；但 (a) §3.7 三向分解、(b) 4ch BP 负控制、(c) DAPT V3 拆分等方法论严谨度足以支撑 Q2 主流期刊投稿；不具备登顶 Q1 nature-tier 所需的"多 cohort × 多任务 × 完整外部验证"广度。 |
| **6. 论文成熟度 (Paper Maturity)** | **Pre-submission revised draft (v3.0.1)** | (a) v3 已完成 v2 → v3 的实验扩充（容量阶梯、random-init、DAPT V3、64ch FDR、4ch BP）；(b) 整合性 pre-review 已 PASS WITH NOTES（Phase A/C/D/E/7-Mode 全部 NOT_SUSPECTED 或已 verified）；(c) 仍标注 [TODO] 图表精修与 v3 → submission 的格式转换；(d) 部分章节（§3.4.4 / §3.5.4 / §3.7.3）系最近添加，文字已稳定但反复迭代痕迹明显。 |

---

## 3. 推荐目标期刊（Top 3，已验证存在且对口）

### 候选 #1 — *Journal of Neural Engineering* (IOP Publishing) ★ **首选**

- **JCR 分位**: Q2（神经工程 / 生物医学工程交叉）；2024 IF ≈ 4.0
- **对口理由**:
  - JNE 是 EEGNet 原论文（Lawhern 2018, ref [5]）的发表地；本文与 EEGNet 的直接对话使该期刊成为最自然的选择
  - JNE 历年发表大量 MI-BCI、通道缩减、deep-learning-for-EEG 工作（与本文 §3.5 通道缩减 + §3.1–§3.3 模型对比直接对口）
  - 接受 negative results（DAPT 负迁移）与 deployment-oriented engineering papers（§3.8 推理延迟、§4.6 部署路线图）
- **可能阻力**: 单一 21 人 cohort 的泛化范围；编辑可能要求外部 cohort 复现作为 minor revision
- **预估投稿匹配度**: 8.5/10

### 候选 #2 — *IEEE Transactions on Neural Systems and Rehabilitation Engineering* (TNSRE)

- **JCR 分位**: Q2（神经工程，临床康复导向）；2024 IF ≈ 4.8
- **对口理由**:
  - 论文 §4.6 部署路线图 + §3.4 纵向数据 + §3.8 推理延迟均强烈强调"实用临床部署可行性"，与 TNSRE 临床定位契合
  - TNSRE 最近三年（2023–2025）发表了多篇 EEG foundation model 与 motor decoding 相关工作（含原数据集论文 [3] Ding et al. 2025 自身投 *Nature Communications*，但本类型工作 TNSRE 高频）
  - 32-ch FDR 与 4-ch BP 的"商用硬件兼容性"主张直接命中 TNSRE 的 rehabilitation engineering 受众
- **可能阻力**: 论文以 offline 离线评估为主，无在线机器人控制——TNSRE 评审可能要求"未来工作中加入 online 实验"
- **预估投稿匹配度**: 8/10

### 候选 #3 — *NeuroImage* / *NeuroImage: Reports* （备选 Q1 上限）

- **JCR 分位**: NeuroImage 主刊 Q1（神经科学方法学，2024 IF ≈ 5.7）；NeuroImage: Reports Q2
- **对口理由**:
  - 若强调 §3.5.2 4ch BP 解剖学位置 + §4.3 体积传导 + §3.6 DAPT 负迁移背后的"基座模型 domain 由信号级特征定义"等神经科学方法论命题，可投 NeuroImage 主刊
  - NeuroImage: Reports 接受较短的方法学贡献，门槛比主刊低
- **可能阻力**: NeuroImage 主刊偏重 fMRI / multimodal neuroimaging；EEG-only single-cohort 工作通常需要 stronger neuroscience theory contribution；本文方法论侧重大于 neuroscience theory 侧
- **预估投稿匹配度**: 6/10（备胎，若 JNE / TNSRE 拒稿后可考虑）

---

## 4. 评审委员配置卡（5 张）

### Reviewer Configuration Card #1

**角色 (Role)**: **EIC (Editor-in-Chief / Senior Associate Editor)**
**身份描述**: *Journal of Neural Engineering* 资深 Associate Editor，神经工程 / EEG-BCI 领域 18 年学术编辑经验；自身研究背景涵盖 motor imagery decoding 与 BCI deployment，曾担任 IEEE TNSRE 关于"Foundation Models for EEG"专刊客座主编（2024）；过往审过 EEGNet (2018) 与多篇 LaBraM / CBraMod follow-up 工作；对单一 cohort 论文的"appropriate generalization claim"门槛把控严格。
**Review Focus**:
  1. **整体定位与贡献清晰度**: 论文是否在第一段 5 行内让读者明白其方法论定位 (CBraMod + EEGNet 比较 + 通道缩减 + DAPT 三条线)；§1.4 Contributions 列表的 6 条是否相互独立、可验证、有具体效应量
  2. **统计严谨性概览**: paired t-test 的多重比较校正缺失（§2.8 明确"无多重比较校正"）是否可接受；effect size 报告是否完整（mean ± SD + p-value 完整，缺 Cohen's d / 95% CI）
  3. **publishability 与 venue fit**: 单一 21 人 cohort + 单作者工作是否达到 JNE / TNSRE 主刊 Q2 门槛；是否需要 reject 或仅 minor / major revision
**会特别关注**: 论文的**叙事一致性** (§1 / §3 / §4 / §7 是否同步推进同一组发现)、以及**是否过度声明** (over-claim) ——尤其 cross-subject 90.68% 的高准确率是否被正确语境化（响应者 cohort、单数据集、无外部复现）。
**可能盲点**: EIC 视角偏 high-level，可能不会逐个数值核验 §3.6 DAPT V3 表与 V1/V2 拼接的训练充分度争议；倾向相信作者的"已通过 7-Mode integrity check"声明而不重审 mode 4 (shortcut)。

---

### Reviewer Configuration Card #2

**角色 (Role)**: **R1 — Methodology / Machine Learning Reviewer**
**身份描述**: 美国某 R1 大学 BME / CSE 系 Tenure-Track Assistant Professor，专攻 deep learning for biomedical time-series；NeurIPS / ICLR / IEEE TPAMI / TNSRE 多重投稿与审稿经验；对 Bayesian HPO (Optuna / TPE)、capacity ablation、negative results methodology、benchmark fairness 有发表记录；曾审过 Lawhern 2018 EEGNet 后续工作及 Wang 2025 CBraMod 同期投稿；对"compute budget asymmetry confounds model comparison"问题极敏感。
**Review Focus**:
  1. **HPO 公平性核验**: §2.5.1 ProbabilisticSubjectPruner 在 within-subject HPO 触发率 52.9%–65.6% 是否引入 selection bias；EEGNet 用 32 trials / 10 complete vs CBraMod 用 51 trials / 23 complete (Table S5b) 是否构成 HPO budget asymmetry，从而让 §3.7 的 "capacity is not the bottleneck" 结论遭受混淆——"扩参 EEGNet 跌至 chance 是否仅因 HPO 在新架构上未充分搜索"
  2. **§3.7 三向分解的隔离严谨性**: §3.7.1 (EEGNet capacity ladder) 中 baseline → Mid 一跳同时改了 conv stem (F1: 16→32, F2: 64→256) 与 MLP 头（单 Linear → 双层 [1024, 1024] + LayerNorm），作者已自承"严格意义上未隔离 conv stem 单轴 vs MLP 头单轴的贡献"；该 caveat 是否削弱"−25 pp = capacity-internal harm"主张；是否要求 reviewer 接受 §6 中"未来工作 #6"作为合理 deferral
  3. **Channel selection 方法学**: §3.5 Limitation #1 已自承"FDR / CSP / Attention / BP 在所有 session（含测试 session 上下文）上计算"；这是否构成 mild leakage——ranking metric 用了 test-session segments；负控制实验 (§3.5.3) 是否充分反证此 leakage
**会特别关注**: 评审会执着于 **HPO compute budget asymmetry**（CBraMod 比 EEGNet 多 ~40% trial 数）以及 §3.7.1 "EEGNet-Huge v1/v2 ~30M 参数在两套独立 HP 下均落入 chance"——R1 会要求作者跑 ≥3 sets HP × 3 seed 的 robustness sweep 来排除"HP local minimum 而非 capacity 饱和"假设。
**可能盲点**: R1 不一定熟悉 finger-level MI 数据集本身的特殊性 (Ding et al. 2025 [3] cohort 的 responder filter)，可能把"21 人样本量"的 statistical-power 问题与"single-dataset generalization"问题混在一起评论。

---

### Reviewer Configuration Card #3

**角色 (Role)**: **R2 — Domain Expert / Senior MI-BCI Researcher**
**身份描述**: 资深 BCI 研究员，University of Tübingen / Graz / Wadsworth 系 Pfurtscheller / Wolpaw 学派传承者（虚构合成，但符合该领域典型 reviewer profile）；25+ 年 motor imagery + sensorimotor cortex BCI 研究经验，发表过 mu/beta ERD 经典论文与多个 LOSO + finetune 协议工作；最近 5 年关注 EEG foundation models（CBraMod / LaBraM / Brant），熟悉 finger-level MI 文献链 (Alazrai 2019 [8] / Lee 2022 [9] / Ding 2025 [3])。
**Review Focus**:
  1. **文献覆盖完整性**: 9 条引用是否覆盖了"foundation models for EEG"足够广度——LaBraM (ref [6]) 与 review (ref [7]) 已列出，但 Brant (Yang et al. 2024)、BrainBERT (Wang et al. 2023)、NeuroLM (2024)、BIOT (Yang 2023) 等同类基座模型未被提及；Schirrmeister 2017 deep-CNN-for-EEG、Sakhavi 2018 (FBCSP+CNN) 等更广义的 MI-decoding baseline 未被纳入比较或讨论
  2. **XSI-FT 的术语正当性 (§3.3)**: "Cross-Subject-Initialized Per-Subject Fine-Tuning" 是否真正构成 novel concept，还是仅是 LOSO + per-subject finetune 的重命名（已知问题：integrity report 已确认 Phase D 该术语本研究首创但概念与 LOSO+fine-tune 同源）；该命名是否会引起领域内 reviewer 的反感（"这只是换名字"）
  3. **DAPT 负迁移结论的领域贡献定位**: §3.6 + §4.5 + §4.8 把"EEG foundation model 的 domain 由信号级特征（采样率、频段、电极配置）定义"提升为方法论命题，是否合理；这一命题与 NLP / CV DAPT 经验的对照是否给予了足够的理论支撑（vs 仅经验观察）；是否需要引用 Brain-foundation-model survey (e.g., Lai et al. 2025 [7]) 之外的 domain-shift theory
**会特别关注**: §1.2 表 0 与 [3] Ding et al. 2025 Nature Comm 的 80.56% / 60.61% **在线 session-adaptive** 数字 vs 本文 90.68% / 74.88% **离线 cross-subject** 数字的并列展示——R2 会特别警惕表 0 是否构成"cherry-picked apples-to-oranges 对比"，即使作者已在表 0 下方加了 "可比性说明"。
**可能盲点**: R2 倾向 EEG signal-processing 领域的传统视角，可能低估 §3.7 capacity ladder + random-init 这一"sub-field-internal first"的方法学贡献价值；可能将其简单归类为"标准 ablation 而已"。

---

### Reviewer Configuration Card #4

**角色 (Role)**: **R3 — Cross-Disciplinary Perspective Reviewer**
**身份描述**: NLP / CV foundation model 研究员（兼顾 transfer learning theory 背景），来自 Stanford NLP / Google DeepMind / FAIR 系工业研究院；发表过 BERT-style continued pretraining、domain-adaptive pretraining (Gururangan ACL 2020 *Don't Stop Pretraining*)、parameter-efficient fine-tuning 等经典工作；近期对 cross-domain foundation model transfer (NLP → biomedical, CV → medical imaging) 感兴趣；不是 EEG 专家，但对 transformer architecture 与 self-supervised pretraining 极熟悉。
**Review Focus**:
  1. **DAPT 负迁移与 NLP / CV DAPT 文献的对话**: §4.5 / §4.8 把负迁移归因于"EEG domain 由信号级特征定义"——R3 会用 NLP DAPT 类比检验此论断：在 NLP 中，BioMed-BERT 等 in-domain DAPT 也仅在 *task-aligned* corpus 上有效；本文 V3 显示 Stieger 占比 79% → 30% 后恢复一半，与 NLP 中"domain mismatch is a continuous spectrum, not binary"经验一致——是否作者把"binary domain definition"过度简化；是否应引用 *Don't Stop Pretraining* (Gururangan 2020) 与 *MedBERT* / *BioBERT* 等 NLP DAPT 文献以校准 framing
  2. **Random-init CBraMod (§3.7.2) 与 NLP "from-scratch transformer" 类比**: random-init transformer 在 ~70 trial / subject 下失败（within ternary 18/21 落到 chance）—— 这与 NLP 中"BERT-base 100M 参数 random-init 下游 ~50 examples 直接训不动"经验一致；R3 会问 §3.7.2 是否充分讨论了 transformer 在 small-data regime 的众所周知失败模式 (Devlin 2019 / Liu 2019 RoBERTa) 而不是把它当作 EEG 特异发现
  3. **Capacity ablation 的统计稳健性**: §3.7.1 EEGNet-Huge v1/v2 在 ~30M 参数下 train loss 死锁在 0.693 (chance entropy)、所有 21 名被试 test 50%——这种"完全无法训练"现象在 NLP / CV 中通常通过 careful initialization + warmup + LR sweep 解决；R3 会要求作者展示 ≥1 个证据：(a) 正确 init scheme (Kaiming / Xavier 选择)、(b) ≥3 LR × 3 warmup × 3 seed 的 sweep、(c) gradient norm logs；当前仅提到"两套 lr 相差 10×"不足以排除 implementation bug
**会特别关注**: 跨学科类比的精度——R3 会强烈反对论文 §1.3 末段"在 NLP 和 CV 中已得到验证"的笼统表述，要求精确引用 Gururangan ACL 2020 / Gu 2021 BiomedRoBERTa / Beltagy 2019 SciBERT 等具体文献，并指出 NLP DAPT 文献本身就分"helpful" / "harmful" / "neutral" 三类结果（不是单向 positive 经验）。
**可能盲点**: R3 不熟悉 EEG signal processing 的具体物理（如 mu/beta ERD 与 sensorimotor cortex 的解剖锚定、体积传导导致的电极冗余、ACPE 与 EEG 通道几何的特殊关系），可能把 §3.5.2 / §4.3 / §4.5 中的 EEG-specific 论证误读为"过度拟合 finger MI 任务"。

---

### Reviewer Configuration Card #5

**角色 (Role)**: **Devil's Advocate (新增 v1.1 角色)**
**身份描述**: 适度 contrarian 的资深 BCI / ML 研究员，专门挑战论文的核心叙事而非细节；曾参与多次 meta-analysis (Reproducibility-of-BCI-results, Dataset-bias-in-deep-EEG-decoding)；对"benchmarking with single dataset" 与"foundation-model hype" 持系统性怀疑；其 review 通常对作者最痛 (high-impact, low-deference)，但有理有据。该 reviewer 接受论文最终发表的概率仅 30%，但其 raise 的挑战不可被忽视。
**Review Focus** (3 项最强反驳):
  1. **"Cross-subject 90.68% 看起来过高 — 数据泄露 / cohort cherry-picking 的可能性是否被充分排除？"**
     - **挑战要点**: 对手指级 MI 任务，[3] Ding et al. 2025 在 *Nature Communications* 报告**在线 session-adaptive** EEGNet 仅 80.56%；本文报告**离线 cross-subject** CBraMod 90.68%——这是 +10 pp 跨论文 gap，跨 evaluation paradigm 不可直接比较，但仍引发"为什么离线评估反而比在线显著更高"的质疑
     - **可疑通道**: (i) §3.5 Limitation #1 已自承 channel selection metrics 用了 test session 上下文，虽然作者认为只影响"通道选择质量评估"，但 cross-subject pooled training 会让 model 在 train + test session 之间共享 some statistics；(ii) 21 名被试的 responder cohort（[3] 在 49 名招募者中筛选出的 BCI-amenable users），高准确率部分应归因于 cohort 而非 model；(iii) §3.9 leave-S04/S10/S14-out 仅证明"伪影 ≠ artifact-driven shortcut"，但**未排除**"trial-onset spectral leakage / time-of-day artifact / electrode-impedance drift 携带 label hint"等其他 shortcut 类
     - **要求作者补充**: (a) 在外部独立 cohort（如 BNCI Horizon 2020 / PhysioNet MI 数据集）做 zero-shot transfer，至少展示"90.68% 不是 21 人 cohort 特异饱和"；(b) Random-baseline + label-shuffle control（把 train labels 随机打乱重训，确认 cross-subject 跌到 chance；目前只在 within-subject random-init 上间接做了，cross-subject 上未做）
  2. **"§3.7 架构 / 预训练 / 容量三向分解可能被 HPO budget asymmetry 系统性混淆"**
     - **挑战要点**: §3.7.1 EEGNet-Huge v1/v2 (19.99M / 30.22M) 在两套独立 HP 下均 train loss 死锁在 0.693——作者把这解读为"capacity 饱和 / 反向 scaling"，但更朴素的解释是 **HPO budget 完全不对称**：CBraMod baseline 享有 51 trials / 23 complete 的 HPO（Table S5b），而 EEGNet-Huge v1/v2/v3 仅做了 ad-hoc HP sweep（"两套独立 HP" 不构成 systematic HPO）
     - **核心反驳**: NLP 文献中 BERT-large (340M 参数) 在小数据上同样训练困难，但通过 careful init + linear-warmup + LR sweep 是可训练的；本文未排除 EEGNet-Huge 是 **HPO 不足**而非 **capacity 内在饱和**
     - **要求作者补充**: (a) 至少为 EEGNet-Huge v1/v2 做一轮独立 Optuna HPO，预算 ≥ 25 trials；若仍 cluster 至 chance，再可主张 capacity 反向 scaling；(b) 否则改写 §3.7.3 / 摘要 / §7 Finding 1 中"+34.97 pp 来自架构"的 attribution，至少加 caveat"在当前 HPO 预算下"
  3. **"DAPT 负迁移可能是 implementation bug 而非 finding — V3 的 warm-restart 策略让 V3 vs V2 比较失去严格因果性"**
     - **挑战要点**: §2.7.2 caveat 已明确："V3 训练分两阶段：(i) 初次训练 15 epoch，(ii) 在 best checkpoint 基础上做 continue training 12 epoch，**采用 warm-restart-from-weights 策略（仅恢复模型权重，不恢复 optimizer 与 LR scheduler 状态）**"——优化器状态 reset + 27 epoch 拼接训练让 V3 与 V1 (10 ep) / V2 (12 ep) 严格不可比；此外 §2.7.2 还披露 V2 在 Epoch 13 因 Windows LMDB MapResizedError 中断（不是自然 early-stop）
     - **核心反驳**: 当 implementation 包含 (a) optimizer-state reset、(b) crashes-induced premature stop、(c) data-version differences (V1 23/62 vs V2 62/62 Stieger 被试) 三层混淆时，"DAPT 一致负迁移"作为科学 finding 而非工程 issue 的判据脆弱；作者对此的反驳在 §3.6 caveat 中"V1/V2/V3 三种独立训练配置一致负迁移（−0.75/−1.38/−0.70 pp），不同 LR scheduler / 不同数据组成均同向"——但 −0.75 / −1.38 / −0.70 pp **方差大于其量级**（被试间 SD ~10 pp），统计上是否真的"一致"未验证
     - **要求作者补充**: (a) 至少 2 个独立 seed × 1 个 clean V4 run（无 warm-restart、无 crash、固定 30 epoch、固定 Stieger 30%）作为干净复现；(b) per-subject delta 的 paired-t（V3 vs Baseline，N=21）；当前 Table 16 仅报 group mean，未报 paired-t p-value
**会特别关注**:
  - 特别关注 §3.7.3 摘要 "+27 pp" 数字的语义边界——integrity report 已确认 commit f309048 修复了"+27 pp 单位/scope ambiguity"，但 Devil's Advocate 仍会追问"+27 pp 是 within-subject binary + ternary 平均"这一狭义条件下使用是否在摘要 / §1.4 / §7 的多次引用中保持一致
  - **会向 EIC 强调**: 论文核心 5 个 Finding 中至少 2 个 (Finding 1 三向分解 + Finding 4 DAPT 负迁移) 在 method-rigor 层面有可质疑空间；不构成 reject 理由，但需要 major revision
**可能盲点**: Devil's Advocate 倾向"质疑一切"，可能低估论文 §3.5.3 4ch negative-control + §3.9 leave-3-out + §3.7.2 random-init seed 复现性等已经做的反证工作的累积效力；其挑战需要被 EIC 用"该挑战的工作量是否合理"过滤，而不是无条件采纳。

---

## 5. Review Strategy Recommendations

### 5.1 本文的特殊属性（需 reviewer 特别关注）

1. **单作者 + 单一 cohort**: 21 人响应者 cohort + 单一数据集 + 无外部复现，是论文最大的"systematic vulnerability"——所有 5 个 reviewer 都需被提示这一限制，避免过度推广 reviewer 的修订要求；EIC 应在 cover letter / decision letter 中明确 "we treat single-cohort as inherent constraint of master-thesis-scale工作 而非可修订的缺陷"
2. **High implementation rigor + integrity self-policing**: Phase 0 分析师注意到论文 §2.5 / §2.7.2 / §3.4.3 / §3.5.2 / §3.5.4 / §3.7.1 / §3.7.2 多处主动 disclose caveats（如 ProbabilisticSubjectPruner、warm-restart-from-weights、conv stem 双轴扩参、SD 计算口径、HPO budget asymmetry），且 v3 已通过 7-Mode integrity check（PASS WITH NOTES）——R1 / R2 / R3 / Devil's Advocate 在挑战 method 时应优先确认作者是否已 disclose 该 caveat 而不是从零质疑
3. **大量负面与中性结果**: §3.6 DAPT 负迁移、§3.5.3 4ch FDR/Attention/CSP 跌至负控制、§3.7.1 EEGNet 容量反向 scaling、§3.7.2 random-init within-subject 18/21 chance collapse——论文核心叙事是"reporting what doesn't work"，与典型"all-positive results paper"不同；reviewer 应避免要求作者把负面结果包装成 positive narrative
4. **三层叙事自洽性已被 v3 强化**: 摘要 / §1.4 / §3 / §4 / §7 在数值 (+7.05 / +14.01 / +13.65 / +27 / 96.7% / 78.75% / +11.10 / −0.70 / +0.68 等) 与命题层均已通过 integrity check 完成 cross-reference；reviewer 不应在 narrative 一致性上反复纠缠（已被 Phase A–E 核验过）

### 5.2 Reviewer 间潜在张力（synthesizer 须协调）

| 潜在分歧 | R1 立场 | R2 立场 | 可能裁决 |
|----------|---------|---------|----------|
| **HPO budget asymmetry 是否构成致命缺陷** | R1 / Devil's Advocate 会强烈要求 EEGNet-Huge 重做 HPO 才接受 §3.7 capacity 主张；要求 ≥25 trials Optuna sweep | R2 倾向"EEG 领域 HPO 标准本就 ad-hoc"，认为 §3.7 已足以支撑 capacity 反向 scaling 主张；只需作者加 caveat 即可 | EIC 应在 major revision 中要求作者 (a) 为 EEGNet-Huge 跑 ≥ 1 轮独立 Optuna，或 (b) 在 §3.7.3 / 摘要 / §7 Finding 1 中加入 "在当前 HPO 预算下" 的明确范围限定 |
| **XSI-FT 是否为 novel concept 还是仅 LOSO+finetune 的重命名** | R1 / Devil's Advocate 会接受作者的术语命名，关注其机制定义清晰度 (§3.3 已清晰) | R2 会要求作者 (a) 引用 LOSO + per-subject finetune 经典文献并显式说明区别、(b) 弱化"首创术语"语气 | EIC 应推荐在 §3.3 引言段加一句"该机制在文献中以 LOSO + per-subject fine-tune 形式存在，本文将其命名为 XSI-FT 以便与单一阶段范式（§3.1 / §3.2）做并列对照"；不要求改名 |
| **跨学科 framing (§4.5 / §4.8) 的 NLP 类比深度** | R3 会要求作者引用 Gururangan 2020 / Gu 2021 BiomedRoBERTa / Beltagy 2019 SciBERT 等具体 NLP DAPT 文献 | R2 会觉得这层文献"扯远了"，可能反对 R3 要求 | EIC 应支持 R3 的方向但限制其深度——加 1–2 条 NLP DAPT 文献作为 footnote 即可，不需重写 §4.5 / §4.8 |
| **DAPT V3 warm-restart 是否构成 implementation 混淆** | Devil's Advocate 会强烈要求干净 V4 复现 | R2 / EIC 会接受作者已在 §2.7.2 caveat / §3.6 caveat 中 disclose 该 limitation | EIC 应将其降级为 limitation 加强而非 reproducibility 要求；要求作者把 §2.7.2 V3 caveat 提升到 §5 Limitation #12（实际已经包含，强化一段） |

### 5.3 Devil's Advocate 应着力的 3 项最强挑战 (排序)

按"挑战的杀伤力 × 作者反驳的可行性"加权：

1. **"#2: HPO budget asymmetry 混淆 §3.7 capacity 三向分解"** — *最强挑战*
   - 杀伤力高：直接质疑论文最核心方法学贡献（"+34.97 pp 来自架构"）；如成立可摧毁 §3.7.3 / 摘要 / §7 Finding 1 的核心论证
   - 作者反驳成本：中——需为 EEGNet-Huge v1/v2 跑独立 HPO，~25 trials × 数小时/trial，预算可控
   - 推荐 EIC 接受为 major revision 必要工作

2. **"#1: cross-subject 90.68% 的 leakage / shortcut 排查不充分"** — *次强挑战*
   - 杀伤力高：直接质疑论文最高 headline 数字
   - 作者反驳成本：高——要求外部 cohort zero-shot transfer 是 master-thesis-scale 工作之外的工作量；label-shuffle control 较易补
   - 推荐 EIC 接受 "label-shuffle control on cross-subject" 作为 major revision 工作；外部 cohort 复现降级为 limitation 加强 / Future Work
   - integrity report 已确认 §3.5.3 4ch 负控制 + §3.9 leave-3-out 是已有反证；Devil's Advocate 的挑战要求是这些反证的扩展而非否定

3. **"#3: DAPT V3 warm-restart 让 V3 vs V2 失去严格因果性"** — *第三强挑战*
   - 杀伤力中——只影响 §3.6 V3 vs V2 +0.68 pp 拆分（约一半的 Stieger 占比解释），而 "DAPT 整体负迁移 V1/V2/V3 一致" 主张不依赖该拆分
   - 作者反驳成本：中——需 1 个干净 V4 run + per-subject paired-t 重做
   - 推荐 EIC 降级为 limitation 加强 + Future Work；不要求重做 V4 实验

**Devil's Advocate 不应得逞的挑战** (synthesizer 应在 Phase 2 提示):
- §3.5 单一数据集 → "无法泛化到其他 cohort"：这是 single-thesis-cohort 的内在约束，非可修订的方法缺陷
- §1.4 contributions 多达 6 条 → "声明过多"：v3 已对每条用具体 effect size / 数据来源支撑，每条均独立可验证

---

## 6. 总体编辑建议（Pre-Phase-1 strategic advisory）

- **优先投稿**: *Journal of Neural Engineering* 主刊，IEEE TNSRE 备选
- **预期 outcome**: 较大概率 **Major Revision**（约 60–70% 概率），其次 Minor Revision（20%），Reject 风险 ~10–15%
- **3 项最关键 revision items（与 5.3 Devil's Advocate 挑战对齐）**:
  1. EEGNet-Huge v1/v2 独立 HPO sweep（≥ 25 trials Optuna）以巩固 §3.7 capacity 反向 scaling 主张
  2. Cross-subject CBraMod 跑 label-shuffle control（chance baseline 验证）以完成 90.68% 的最后一项 shortcut 排查
  3. 引用 Gururangan 2020 + 1 条 BiomedRoBERTa-style NLP DAPT 文献，强化 §4.5 / §4.8 跨学科 framing 严谨度
- **不应被 reviewer 强制修订的事项**: (a) 单一 cohort 限制；(b) XSI-FT 重命名（保持原命名 + 增加文献溯源说明）；(c) 单作者性

---

**报告完成时间**: 2026-05-10
**Phase 0 时间预算**: ~15 min（实际略超）
**下一阶段**: Phase 1 — 5 名 reviewer 各自独立审稿（5 个 isolated agents）
**输出去向**: ARS academic-paper-reviewer pipeline 协调器
