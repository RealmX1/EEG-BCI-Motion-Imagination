# EIC Review Report — Stage 3 Phase 1

**Reviewer Role**: Editor-in-Chief
**Identity**: Senior Associate Editor, *Journal of Neural Engineering* (IOPP)
**Manuscript**: `paper/drafts/paper_draft_v3.0.1.md`（题目：基于 EEG 基座模型的手指级运动想象分类——通道缩减、纵向数据扩展与领域自适应预训练的局限性）
**Date**: 2026-05-10
**Recommendation**: **Major Revision**

---

## 1. 论文整体框架（EIC 视角的一段式概述）

本研究在一个 21 名被试、128 通道、单一手指级 MI 数据集（Ding et al. 2025 [3]）上，对 CBraMod（~30M 参数 EEG 基座模型）与 EEGNet-16,4（~16K 参数 CNN）进行三种范式（被试内 / 跨被试 / XSI-FT）下的离线对比，并沿三条互补轴展开消融：(i) **通道缩减**（128→64→32→8→4，五种选择方法 × 负控制），(ii) **纵向数据扩展**（16 名被试的 3–5 个额外 session），(iii) **领域自适应预训练（DAPT）** 在 870 h 外部 MI 数据上的负迁移。在论文最后阶段（§3.7）作者补做了 EEGNet 容量阶梯（16K→1.90M→5.84M→30M）与 random-init CBraMod 双消融，给出**架构 / 预训练 / 容量三向贡献分解**。整体叙事拉到一条主线："基座模型价值随数据约束放大，架构归纳偏置在 cross 主导、TUEG 预训练在 within 主导、容量本身不是瓶颈"，并以 CBraMod + FDR 32 通道为部署推荐。论文是一篇 well-organized、自我意识较强、报告大量负面/反直觉结果的工作——其分析深度超过典型 master's thesis 水平，但**单 cohort、单数据集、无外部复现**这一根本边界明显限制了发表平台的上限。

---

## 2. Strengths（按重要性排序）

1. **三向消融分解（§3.7）是本稿最大的方法论贡献**。EEGNet 容量阶梯（cross-subject 76.67% → 51.37% → 50% chance，反向 scaling）与 random-init CBraMod (cross 86.34%, within ternary 18/21 chance collapse) 的同规模对照，把 +14 pp 的 CBraMod−EEGNet 差距精准拆解为 "+35 pp 架构 + +4 pp 预训练 + −25 pp 扩参反向"。这种把 backbone 优势分解到三轴的设计在 CBraMod / LaBraM 后续工作中很少见，**是 JNE 读者会感兴趣的核心方法学命题**。

2. **负面结果报告的克制与诚实**。DAPT 三个版本（V1/V2/V3）均为负迁移，作者在 §3.6 / §4.5 / §7 finding 4 的论述都明确承认负面方向，并通过 V3 的 Stieger 占比削减实验把"V2 阶段加重负迁移"拆分为"主导数据集占比 + 整体粗运动 MI 域错配"两层归因——这种"recovered half but not all"的克制表达比常见的"我们发现意外结果！"叙事更可信，符合 JNE 编辑对负面结果稿件的期望。

3. **数据来源标注规范程度高**。每个数据点都附 run_tag + JSON 路径，便于 reproducibility check；ExperimentDB + JSON 双写架构（CLAUDE.md 项目惯例）给审稿人提供了少见的 traceability。

4. **方法学谦抑的"作用域"声明**。多处（§3.5.2 末尾、§3.5.3 末尾、Finding 5）作者主动声明"不外推到其他 cohort / 任务 / backbone"，并在 §6 列出 7 项后续工作。这种边界意识在硕士论文级稿件中相对少见。

5. **Sensitivity check 完成度**（§3.9 leave-3-out）。三名重度伪影被试的去除对 cross-subject 群体均值仅 −0.06 / −0.13 pp 影响，这一 robustness 验证主动回答了审稿人最常见的"主结果是否被异常被试驱动"质疑。

---

## 3. Concerns（按重要性排序，含位置 / 严重度）

### Concern 1（**Major**，§摘要 / §1.4 / §7）：核心方法论定位声明缺失开篇 5 行

**位置**：§摘要前 6 行；§1.1（背景与动机）；§1.4（贡献列表）。

**问题**：摘要第一段直接进入数字（"+7.05 pp / +14.01 pp / +13.65 pp"），但**没有一句话讲清楚本研究在三条独立技术轴之外的统一定位**。这三条轴（CBraMod vs EEGNet 比较、通道缩减、DAPT）在文献中是相对独立的子领域，单看任一条都不构成 JNE 级贡献。论文的真正贡献在于**把三轴绑到同一个 cohort 上做联合系统评估，并通过 §3.7 的三向消融把 backbone 优势机制化**——这才是有 publishability 的核心命题。但开篇没有 1–2 句把这一统一定位说清楚，读者（包括审稿人）需要读完 §3.7.3 才能 reconstructive 出来论文真正在做什么。

**建议**：摘要前 3 行重写为 "本研究在同一 21 人 finger-MI cohort 上，沿（基座模型 vs 紧凑 CNN、通道密度、纵向数据、外部数据 DAPT）四条独立轴系统评估 EEG decoding 的工作点，并通过架构/预训练/容量三向消融将 backbone 优势机制化"——直接亮出"三向分解"作为 narrative spine。当前版本把 §3.7 当成补充消融来 frame，是**严重低估了它对论文的承重作用**。

**严重度**：Major（影响 desk-review 与首轮审稿印象）。

### Concern 2（**Major**，§2.8 / 全文）：统计报告深度低于 JNE 通常预期

**位置**：§2.8（评估协议）；§3 全部表格。

**问题**：作者 explicit 声明"无多重比较校正"，并仅报告 mean ± SD + 配对 t-test p value。在 §3.4 / §3.5 中作者并行检验了 3 种 paradigm × 2 种 task × 多档 channel × 多个方法的二维 / 三维矩阵，**潜在多重比较数量 >50**。仅披露一条 "no correction" 声明在 JNE 当前编辑标准下偏弱。具体：
- §3.5.4 三档 XSI-FT 比较（128 / 32ch FDR / 8ch BP）报告了 +0.74 / −2.03 pp 方向反转，但未给配对 t p-value（仅 §3.3 有 p）；这恰好是 §4.8 部署路线图的 load-bearing 数据点之一。
- §3.6 V3 vs V2 +0.68 pp 也无 p value；用作 "Stieger 主导是 V1→V2 主因之一" 的关键归因证据。
- 全文几乎无 **Cohen's d / Hedges' g**，也无 95% CI。在 N=21 / N=16 小样本下，单纯 p value 容易让审稿人怀疑 family-wise error。

**EIC-level 容忍度**：JNE 不强制要求 Bonferroni / Holm，但通常预期在以下任一 fallback 中至少满足一个：(a) FDR-BH 校正后给出 q value，(b) 用 mixed-effects model 显式建模 model × task × paradigm 三向，(c) bootstrap 95% CI。当前稿件三者均缺失。**§3.4.2 末段已明确写道 "当前样本量不足以支持正式交互检验"——这个自我意识是好的，但同时把责任推给后续工作，对审稿人是不够的**。

**严重度**：Major（不致命，但首轮审稿一定会被 R1 打到）。

### Concern 3（**Moderate-to-Major**，§3 / §4 / §7）："+27 pp" 的定义在三处略有漂移

**位置**：摘要第 2 段（"加 TUEG 预训练再追加 ~+4 pp（cross / XSI-FT）至 ~+27 pp（被试内）"）；§1.4 Finding 1（"加 TUEG 预训练再追加 ~+4 pp（cross / XSI-FT）至 ~+27 pp（被试内）"）；§3.7.3（"TUEG 预训练贡献 +23.10 pp（binary）/ +30.79 pp（ternary）, 平均 ~+27 pp"）；§7 Finding 1（"+27 pp（被试内）"）。

**问题**：摘要 / §1.4 的 "+27 pp" 在表层语法上是 "preceded by '至 ~+27 pp（被试内）'"，会让读者误读为 "TUEG 预训练在被试内贡献 +27 pp"——但 §3.7.3 的实际定义是 "binary +23.10 + ternary +30.79 平均"。两个 task 的 binary / ternary 数值差异巨大（差 7.7 pp），把它们做无加权平均得到 27 pp 后再回写为 "predicted within 范式 +27 pp"，存在**算术意义上的边界滑动**。

**进一步**：§4.1 第 4 段反复写 "TUEG 预训练在被试内贡献 ~+27 pp、在跨被试与 XSI-FT 仅贡献 ~+2 至 +4 pp"——但 §3.7.3 表格的 "加 TUEG 预训练 → +4.34 pp" 仅对 cross binary 成立；ternary cross +1.82 pp、XSI-FT binary +3.90 pp、ternary +1.61 pp。"+2 至 +4 pp" 跨度大但 lower bound 是 ternary cross 1.82——叙述与数据基本对得上，但 +27 pp 这边的简化是真的偏不严谨。

**建议**：在 §3.7.3、§4.1、§7 中把 "+27 pp" 改为 "binary +23.10 / ternary +30.79 pp" 双数列出，或者用 "+23–31 pp 范围"，避免无加权平均后再单点引用。这是 R1 / Devil's Advocate 一定会抓的点。

**严重度**：Moderate-to-Major（数值本身没有错误，但叙述简化带来的误读概率高，在严格审稿下会被 question）。

### Concern 4（**Moderate**，§摘要 / §1.4 / §7）：90.68% 的 cross-subject 头条数字脱离 cohort caveat

**位置**：摘要表 0；§1.2 表 0（与文献横向对比）；§3.2；§7。

**问题**：表 0 把 "本文 CBraMod 128ch cross-subject 90.68%" 与 Ding et al. [3] EEGNet 80.56%（在线 session 自适应）、Lee 2022 70%、Alazrai 2019 65% 并列。作者已在表下加了"评估范式不可直接比较"的可比性说明，**这个 caveat 写得相对充分**——但摘要正文（"跨被试二分类 +14.01 pp（90.68% vs 76.67%）"）和 §7 conclusion 都把 90.68% 当作 standalone headline number 引用，缺乏 cohort 边界的提醒。

**关键边界**：§5 Limitation #2 已明示 "responder cohort 继承自 [3]，仅保留离线二分类 ≥70% 阈值的被试" → **90.68% 是 BCI-amenable 子群准确率，对未筛选总体可能高估** 。但摘要 / §1.4 / §7 没有把这一关键 caveat 主动 surface 到读者眼前。JNE 审稿对 "responder cohort" 数字 vs "naive cohort" 数字的处理标准近年趋严（2023-2025 多篇撤稿都涉及 cohort 选择泛化问题）。

**建议**：摘要中在 "90.68%" 第一次出现处加一句 "（21 名 responder 被试，原数据集筛选后 cohort，详见 §2.1）"。

**严重度**：Moderate（不致命；但单数据集 + responder cohort + 高头条数字的组合容易被 reviewer 抓"过度泛化"。）

### Concern 5（**Moderate**，§3.6 / §4.5 / §7 finding 4）：DAPT 负迁移的因果归因仍偏强

**位置**：§3.6 末段；§4.5 第 3 段；§7 Finding 4。

**问题**：作者用 V3 实验（Stieger ~30%）证明 V2 阶段（Stieger ~79%）的负迁移加剧，"约恢复了一半"。但论文同时承认 V3 是"warm-restart-from-weights"两阶段训练（§2.7.2 caveat），优化器状态在阶段 ii 重置——**严格而言 V3 vs V2 的差异是 (Stieger 占比 + warm-restart 续训) 的混合效应**。作者在 §2.7.2 已 disclose 这一点，但在 §4.5 / §7 finding 4 的归因叙述中忽略了 warm-restart 这一干扰项，直接把 "+0.68 pp" 归到 Stieger 占比变化。

**第二项问题**：§4.5 末段引入的"通道几何错位"（10 个外部数据集分布在 7 个不同电极配置；仅 5.4% 样本与下游 128 通道对齐）是一个有价值的 structural caveat，但被放在很靠后的位置，没有在 §3.6 实验设计或 §7 结论中显式 surface。这是论文中**最具 generalizable methodological insight 的观察之一**——"EEG foundation model 的 domain 边界由信号级特征定义"，但被埋藏在 §4.5 末段。

**建议**：(a) 把 V3 的 warm-restart 限制在 §3.6 / §7 finding 4 的归因叙述中也提一遍；(b) 把"通道几何错位"提升到 §3.6 章节末段或 §1.4 Finding 5。

**严重度**：Moderate（不影响主结论方向，但归因叙述不够严谨）。

### Concern 6（**Minor-to-Moderate**，§3.7.1 / §4.1）：EEGNet 容量阶梯的 baseline → Mid 一跳混淆 conv stem 与 MLP 头两轴

**位置**：§3.7.1 第 1 段；§6 后续工作 #6（作者已自承）。

**问题**：作者已主动 disclose（§3.7.1 第 1 段："严格意义上未隔离 conv stem 单轴 vs MLP 头单轴的贡献"）并把隔离实验列为 §6 后续工作 #6。这种 self-disclosure 是好的，但 §4.1 / §7 finding 1 的语气仍然偏强（"EEGNet 内扩参 → −25.30 pp"）——读者会自然地把 "−25 pp" 解读为容量纯效应，而实际是 (conv stem F1: 16→32 + F2: 64→256 + 单 Linear → 双层 MLP) 的混合。**严格地说，−25 pp 中 0–19 pp 可能源自 MLP 头双层化的过拟合而非容量本身**。

**建议**：把 "EEGNet 架构内扩参 → −25 pp" 在 §4.1 / §7 finding 1 中改写为 "EEGNet 在 (F1=32 conv stem + 双层 MLP 头) 架构变体下扩参 → −25 pp"，并明确这一变体到 ~30M 后仍 chance 的事实。

**严重度**：Minor-to-Moderate（自我披露已存在，但叙述简化在重要位置仍偏强）。

### Concern 7（**Minor**，§摘要 / §3.4.4 / §4.4）：Extra sessions 的样本量边界不够 surface

**位置**：§3.4.4（N=16）；§4.4；摘要第 4 段。

**问题**：纵向 extra sessions 的全部分析基于 N=16 的子集，作者已在每个表注明 N=16，但 §4.4 部分叙述（"标准差从 10.81% 压缩至 5.98%"）没有 surface "**N=16 而非 21**" 这一边界。+6.13 pp / +5.70 pp 的 paired-t p value 是用 N=16 计算的，自由度为 15——审稿人很可能会问 "为什么不在 21 名被试上做？" 答案是 "其他 5 名没有 extra sessions"——这个答案是合理的，但应该更显式。

**严重度**：Minor。

### Concern 8（**Minor**，§3.5.2）：4ch Band Power "意外强劲" 的解读结构被作者已大幅修订，叙述仍可进一步紧凑

**位置**：§3.5.2 中段；§4.3。

**问题**：§3.5.2 关于 BP top-4 通道的解剖学论断已经修订得很谨慎（4 个通道中只有 D27 真正落在 hand knob 区，作者明确 retract 了原"空间锁定 sensorimotor"论断）。但解释机制部分（i / ii / iii 三种 hypothesis）行文偏冗长，读者会迷失。**建议把这一段压缩到 1/3 长度，把空间留给 §3.7 三向分解**——后者承重远超 4ch BP 的解剖学解读。

**严重度**：Minor。

---

## 4. Statistical & Reporting Standards (EIC-Level)

| 维度 | 当前状态 | EIC 评估 | 建议 |
|------|---------|---------|------|
| 单数据点 p-value | 在 §3.4 / §3.6 主表中报告 | OK | 无需变更 |
| Multiple comparison correction | 无（明示） | **不达 JNE 主流标准** | 至少补 FDR-BH q-value 或 mixed-effects model |
| Effect size (Cohen's d / Hedges' g) | 全无 | **不达 JNE 主流标准** | 主表至少补 paired Cohen's d |
| 95% CI | 全无（仅 SD） | 弱 | 主表 mean ± SD 旁加 95% CI |
| Sample size justification | §5 Limitation #2 / #6 提及；无 a priori power analysis | OK for retrospective analysis | 无需变更 |
| Subject-level disclosure | Tables S1–S4 完整 | **优于平均** | 维持现状 |
| Run tag / source path 可追溯性 | 全文规范 | **优于平均** | 维持现状 |

总评：**stats 报告深度大约处于 JNE 当前可接受的下沿**。提交前必须补 effect size 与至少一种 multiple comparison 的处理，否则 R1 一定会要求 major revision。这不是"提升论文档次"的额外工作，而是"达到 JNE 收稿基线"的最低补丁。

---

## 5. Venue Fit & Publishability Assessment

### 5.1 是否适合 JNE？

**Yes, with caveats**。本稿在以下维度与 JNE 收稿画像吻合：
- (a) 神经工程方法学贡献（基座模型 × MI BCI）；
- (b) 负面结果与方法论 caveat 的诚实报告；
- (c) 部署导向的实用结论（FDR 32 通道、推理延迟基准）。

**Mismatch 维度**：
- 单 cohort（21 人 responder）在 JNE 通常需要至少一项外部 sanity check（如 BCI Competition IV-2a 上的 cross-validation 或 LaBraM 上的方法重现）；当前稿件完全在单数据集上闭环。
- 单作者硕士论文 framing 在 JNE 是相对少见的（不致命，但不寻常）；JNE 接收单作者稿件比例 < 5%。
- 13,548 字 + 7 supplementary tables + 1 figure 的结构偏臃肿，更接近 *Frontiers in Neuroscience* / *Sensors* 风格而非 JNE 的 typical brief / methods paper。JNE 较偏好 6,000–9,000 字稿件。

### 5.2 替代刊物建议

| 刊物 | 适合度 | 主要顾虑 |
|------|-------|---------|
| **Journal of Neural Engineering (JNE)** | **首选**（在 major revision 后） | 单 cohort、单作者、长度 |
| IEEE TNSRE | 适合 | 与 JNE 类似；可能更偏好临床闭环验证 |
| Frontiers in Neuroscience (BCI section) | 备选 | 接收门槛低；论文质量高于该刊典型水准会浪费 |
| eLife | 不建议 | 缺乏跨数据集 / 跨 cohort 复现 |
| Neural Networks (Elsevier) | 备选 | 偏方法学；DAPT 负迁移分析是亮点 |

**建议**：**首选 JNE，备选 IEEE TNSRE**。两者审稿周期与编辑标准接近。

### 5.3 读者兴趣评估

**JNE 读者会感兴趣的点**：
1. §3.7 三向分解（架构 vs 预训练 vs 容量）——这是 EEG foundation model 子社区目前最缺的对比实验之一；
2. DAPT 负迁移 + 通道几何错位假设——挑战 NLP/CV 的 DAPT 范式直接迁移；
3. 32ch FDR 部署可行性（96.7% retention）——临床转化导向；
4. 4ch BP 反例（vs FDR/Attention/CSP）——通道选择方法论小贡献。

**JNE 读者可能感到疲劳的点**：
1. 大量 self-rephrasing 的 caveat 段落（§3.5.2 末、§3.5.3 末、§5 全部、§6 全部）会让读者觉得论文在反复给自己设防御工事；
2. 7 个 supplementary tables 的逐被试数据列得过细，Tables S1 / S1b / S2 可合并为一个 wide table。

### 5.4 单 cohort + 单作者的适宜性

JNE 历史上有发表单作者方法学稿件的先例，**前提是方法论贡献本身足够有 weight**。本稿的 §3.7 三向分解 + DAPT 负迁移确实达到了这一门槛，但 cohort 规模（21 responder）是审稿一定会 push back 的点。**最低线建议**：在 BCI Competition IV-2a 上把 §3.1（CBraMod vs EEGNet 单数据点 cross-subject）跑一次外部 sanity check，作为 rebuttal 弹药。如不愿额外做实验，则需要在 §5 Limitation 中显式补一段"为何外部复现在本研究边界外"。

---

## 6. Narrative & Over-claim Audit

### 6.1 内部一致性

总体良好。Abstract / §1.4 / §3 / §4 / §7 五处对核心数字的引用基本对齐：
- "+7.05 / +14.01 / +13.65 pp" 三个核心 delta 在五处叙述一致；
- "96.7% retention" 在三处一致；
- "+27 pp" 在三处出现，但**与 §3.7.3 表的 binary +23.10 / ternary +30.79 是无加权平均**——见 Concern 3。

**轻微不一致**：
- §1.4 Finding 1 写 "加 TUEG 预训练再追加 ~+4 pp（cross / XSI-FT）至 ~+27 pp（被试内）"——这里 "至" 字会让读者误解为递增叠加；建议改为 "在 cross / XSI-FT 范式下 TUEG 预训练贡献 ~+2 至 +4 pp，在被试内范式下贡献 ~+23 至 +31 pp"。
- §4.6 部署路线图建议 "起步方案：CBraMod + FDR 32 通道（87.71% 基线）"，但 §1.4 / §7 头条仍以 "90.68%（128ch cross）" 引用。读者会困惑：到底是部署 32ch 还是 128ch？建议 §7 conclusion 一段把"研究上限 90.68%（128ch cross-subject）"与"部署推荐 87.71%（32ch FDR）"两条线分开陈述。

### 6.2 可能过度声明的具体语句

1. **§7 末段**："**EEG 基座模型的 transfer 路径与 NLP/CV 的 domain-adaptive pre-training 范式不同**——其 'domain' 边界由信号级特征（采样率、频段、电极配置）而非任务级语义定义。" → 这是基于**单一 backbone（CBraMod）+ 单一 DAPT 数据池（粗运动 MI）+ 单一下游任务（finger MI）** 的观察。把"domain 边界由信号级特征定义"作为 generalizable 方法论命题，跨度过大。建议改为 "提示" 而非 "证明"；或者明确加 "在本研究 (CBraMod backbone, 粗运动 MI 数据池, finger MI 下游) 组合下"。

2. **§7 finding 1**："**架构归纳偏置（transformer + ACPE）是 cross-subject 范式下最大贡献**" → 这一结论在 §3.7.1 / §3.7.2 中确实有 +35 pp 的同规模对照支撑，但仍是**单 backbone 的 N=1 观察**——LaBraM、BIOT、EEGPT 等其他 transformer-based 基座是否表现一致是未验证的。作者在 §5 Limitation #7 已承认这一点，但 §7 finding 1 的 framing 仍偏强。建议在 finding 1 末尾加"该结论限于 CBraMod backbone，其他 EEG transformer 是否复现需独立验证"。

3. **§4.8 末段**："**EEG 基座模型的 'domain' 边界由信号级特征（采样率、频段、电极配置）而非任务级语义定义**——这区别于 NLP/CV 的 domain-adaptive pre-training 经验" → 同 1 的问题。

4. **§3.5.2 / Finding 5**：作者已经多次自我限制（"不外推到其他数据集 / 任务"），这部分**没有过度声明**——是一个良好示范。

### 6.3 Required Contextualization

| 头条数字 | 当前 framing | 建议 contextualization |
|---------|-------------|------------------------|
| 90.68% (128ch cross binary) | 摘要 / §1.4 / §7 standalone 引用 | 加 "21 名 responder 被试（原数据集筛选后 cohort）" |
| +27 pp (TUEG 预训练 within 贡献) | 三处出现 | 改为 "binary +23.10 / ternary +30.79 pp" 或 "+23 至 +31 pp" |
| 96.7% retention (32ch FDR) | §1.4 Finding 2 | 加 "在本数据集 cross-subject binary 上" |
| 4ch BP 78.75% | §1.4 Finding 5 | 已加充分 caveat（无需变更） |
| +35 pp 架构贡献 | §7 Finding 1 | 加 "限于 CBraMod backbone × 本数据集" |

---

## 7. Editorial Decision Rationale

### 7.1 推荐 Major Revision 的理由

**Pro acceptance**:
- §3.7 三向分解是 substantive methodological 贡献；
- DAPT 负迁移 + V3 部分归因是 publishable 负面结果；
- Reproducibility 标注规范（run_tag + JSON）超平均水准；
- Limitation 章节诚实度高。

**Against acceptance（需 revision 解决）**:
- Statistical reporting 深度低于 JNE 主流（无 effect size、无 multi-comparison correction）→ Concern 2；
- 核心数字 framing 简化（+27 pp 无加权平均）→ Concern 3；
- 90.68% 头条脱离 cohort caveat → Concern 4；
- 部分声明（§7 末段）过度泛化 → Concern 5/6.2；
- 单 cohort 无外部 sanity check → §5.4。

**为何不是 Minor**：上述 5 项中 Concerns 2 / 3 / 4 都需要改写正文 + 重做部分统计，单纯 typo 级修订无法覆盖。

**为何不是 Reject**：核心方法论贡献（§3.7）有真实价值；负面结果（§3.6）可发表；问题集中在 framing + statistical depth，可在 8–12 周 revision 内解决。

### 7.2 解锁 acceptance 的最小补丁集

按优先级排序：
1. （Must）补 effect size（paired Cohen's d）+ FDR-BH q value 或 95% CI 到 §3 主表；
2. （Must）摘要前 3 行重写为统一定位声明；将 §3.7 三向分解从"补充消融"提升为 narrative spine；
3. （Must）把 "+27 pp" 三处叙述改为 "binary +23 / ternary +31 pp" 或区间表示；
4. （Should）在摘要 + §7 头条数字旁加 "responder cohort" caveat；
5. （Should）§7 finding 1 + 末段加 "限于 CBraMod backbone × 本数据集" 限定；
6. （Should）DAPT V3 warm-restart 干扰项在 §4.5 / §7 finding 4 显式 surface；
7. （Nice-to-have）BCI Competition IV-2a 等外部数据集上跑 §3.1 sanity check，或在 §5 显式说明为何外部验证在研究边界外；
8. （Nice-to-have）§3.5.2 4ch BP 解剖学讨论压缩到 1/3 长度。

完成 1–6 后可重投；7–8 视审稿人口味调整。

---

## 8. Confidence in Review

**Confidence**: **4/5**

**Self-assessment**:
- 对 EIC 视角的 publishability / venue fit / over-claim 判断：confident（5/5）；
- 对 §3.7 三向分解的方法学价值判断：confident（5/5）；
- 对统计学校正深度判断：confident（4/5），存在 JNE 编辑部具体标准近年漂移的不确定性；
- 对单 cohort / 单作者适宜性判断：confident（4/5），偏向保守；
- **Defer to specialist reviewers**:
  - R1 Methodology: HPO 预算（§2.5.1 ProbabilisticSubjectPruner）的搜索质量评估、effect size 具体数值、DAPT 三向 ablation 的统计显著性边界——这些是 R1 本职；
  - R2 Domain: EEG foundation model 文献覆盖（LaBraM / BIOT / EEGPT / Brant 等）；finger-MI 解码的最新 SOTA 锚点；ACPE / criss-cross attention 与 EEG 信号统计的对齐机制；
  - Devil's Advocate: §3.7 三向分解是否真的成立？是否存在 "30M EEGNet 调不出来 ≠ EEGNet 容量限制" 的替代解释？4ch BP 的可复现性是否被 cohort 偏置驱动？
  - Integrity verification 已经 PASS WITH NOTES，本 EIC review 假设其 NOTES 已被作者吸收。

---

*— End of EIC Review —*
