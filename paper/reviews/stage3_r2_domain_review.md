# R2 Domain Review Report — Stage 3 Phase 1

**Reviewer Role**: R2 (Domain Expert / Senior MI-BCI Researcher, Pfurtscheller / Wolpaw lineage)
**Paper**: `paper/drafts/paper_draft_v3.0.1.md` ("基于 EEG 基座模型的手指级运动想象分类：通道缩减、纵向数据扩展与领域自适应预训练的局限性")
**Recommendation**: **Major Revision**

> 提示：以下评议从 MI-BCI 领域专家的文献、理论与贡献定位视角出发，刻意回避方法学统计细节、算力公平性与跨学科类比（这三块分别由 R1/R3/EIC 处理）。本报告独立完成，未参考其他 reviewer 的产出。

---

## 1. Summary of the Paper's Domain Contribution

本文将 Ding et al. 2025 [3] 的 finger-level MI 数据集（21 名 cohort，128ch BioSemi）置于一个统一的 held-out-session 离线评估框架下，系统对比了 EEGNet-16,4 与 CBraMod（ICLR 2025 [4]）这两类极端代表性架构。在三个轴向上推进了贡献：

1. **架构 / 预训练 / 容量三向消融**（§3.7）：通过 EEGNet 容量阶梯（16K → 30M）和 random-init CBraMod 同规模对照，对 cross-subject 范式的 +14 pp 优势做了较为干净的归因——transformer + ACPE 架构本身贡献 ~+35 pp、TUEG 预训练再追加 ~+4 pp（cross）/ ~+27 pp（within）。这是本研究最有方法学价值的一段。
2. **通道缩减谱系**（§3.5）：在 {128, 64, 61, 32, 8, 4} 六档 × 五种选择方法上系统刻画了"方法间差异随通道数减小而扩大"的非线性现象，并识别出 4ch Band Power（78.75%）作为低密度部署的可行候选。
3. **DAPT 负面结果**（§3.6）：在 870 小时外部 MI 数据上做 masked-autoencoding further pre-training，观察到 V1/V2/V3 三种配置一致负迁移，并通过 Stieger2021 占比消融（V3）做了部分归因。

整体上，论文对 finger-MI 文献链（[8] Alazrai 2019 → [9] Lee 2022 → [3] Ding 2025）的延续是清晰的，三向消融的设计也走在领域前沿。但作为一篇以基座模型 benchmarking 为主题的论文，**9 条参考文献的覆盖密度严重偏低**，且对若干领域核心概念（XSI-FT、"DAPT 域边界由信号特征定义"）的文献定位不足，这是后续讨论的主要焦点。

---

## 2. Strengths in Domain Contribution

1. **三向消融设计在 BCI 领域具有方法学示范价值**。§3.7.2 random-init CBraMod 配合 §3.7.1 EEGNet 容量阶梯，这一设计在 EEG benchmarking 文献中并不常见——大多数 EEG foundation model 论文（包括 [4] CBraMod 自身、[6] LaBraM、BIOT/Brant 等）只做"pretrained vs from-scratch"的二元对照，而本文通过把 EEGNet 扩参到与 CBraMod 同量级（30M），剥离了"参数容量"这一通常被混淆进"基座模型 vs 小模型"叙事的变量。结论"在 EEGNet 架构内扩参反而 −25 pp"对 BCI 社区是一个有用的反直觉信号。
2. **通道缩减谱系在 finger-MI 文献中是迄今最系统的**。Ding et al. [3] 的原始通道缩减分析仅覆盖 64/32/21 通道手工布局；本文推进到 8/4 通道极端低密度档位，并引入 4 种数据驱动方法 + 1 种商用布局的横向对照。4ch BP (78.75%) 显著超越负控制（+11.10 pp）这一具体观察对临床/消费级 BCI 部署有直接价值。
3. **DAPT 负面结果的方向性归因**。§3.6 报告 V1/V2/V3 一致负迁移是有 BCI 文献价值的——TUEG 域差距在领域内长期是默认假设但很少被严格量化测试。V3 通过 Stieger2021 占比消融把"单数据集主导效应"与"整体粗运动 MI 域错配"做了拆分，方向清晰。
4. **Trial-level 时序分割 + held-out session 测试**协议（§2.3）严格防泄露，这一点在 finger-MI 文献中并非默认（部分早期工作存在 segment-level 分割导致的隐性泄露），值得肯定。
5. **数据来源标注规范严格**。每个数值都附有 `results/` 路径或 ExperimentDB run_tag，方法学可追溯性达到了远超绝大多数 MI-BCI benchmarking 论文的水准——这一规范应成为该方向的标准做法。

---

## 3. Major Concerns

### 3.1 Literature coverage gaps

**这是本评议最严重的问题**。一篇以"系统对比 EEG 基座模型与传统 CNN"为主题、覆盖 finger-MI / 通道缩减 / 纵向 session / DAPT 四个研究维度的 benchmarking 论文，**只有 9 条参考文献是不可接受的**——即使作为硕士论文也明显偏低。具体覆盖空缺如下。

#### 3.1.1 EEG foundation model 文献链严重残缺

论文 §1.3 + §1.4 + §3.7 反复引用 CBraMod [4] 与 LaBraM [6]，并把"基座模型 vs CNN"作为一条核心叙事线，但完全未提及该领域的其他核心 baseline：

- **BIOT (Yang et al., NeurIPS 2023)**：跨数据集 biosignal transformer，处理 mismatched channels / 变长输入的另一种方案；与 CBraMod 的 ACPE 对应一个直接的方法学对照。
  - 引用建议：C. Yang, M. B. Westover, J. Sun, "BIOT: Biosignal Transformer for Cross-data Learning in the Wild," *NeurIPS*, 2023.
- **Brant (Zhang et al., NeurIPS 2023)**：500M 参数 SEEG/EEG foundation model；与 CBraMod 的 4M 参数形成 scaling 对比。
  - 引用建议：D. Zhang, Z. Yuan, Y. Yang, J. Chen, J. Wang, Y. Li, "Brant: Foundation Model for Intracranial Neural Signal," *NeurIPS*, 2023.
- **NeuroLM (Jiang et al., ICLR 2025)**：1.7B 参数、首个 EEG-LLM 多任务基座模型；与 CBraMod (ICLR 2025) 同一会议的并行工作，**不引用是难以原谅的疏漏**。
  - 引用建议：W.-B. Jiang et al., "NeuroLM: A Universal Multi-task Foundation Model for Bridging the Gap between Language and EEG Signals," *ICLR*, 2025.

§4.1 论文反复声称"基座模型价值随数据约束放大"是 EEG 领域一项普遍命题；要支持该命题，至少需要一段段落（不必是完整复现）讨论这些 backbone 在哪些方向上验证或反驳了此说法。当前版本下，结论实际上是基于 **CBraMod 单一 backbone**，§5 局限性 #7 也明确承认了这一点——但局限性的承认不能替代相关文献的讨论。

#### 3.1.2 经典 MI-BCI 解码 baseline 缺失

§2.4.1 EEGNet 作为 baseline 模型出现，但论文从未引用：

- **Schirrmeister et al. 2017** "Deep learning with convolutional neural networks for EEG decoding and visualization" (*Human Brain Mapping*)：deep ConvNet / shallow ConvNet 的奠基工作，是与 EEGNet 并列的另一个 BCI CNN baseline。Ding et al. [3] 自己测试过 deepEEGNet（论文 §2.4.1 提及但未独立引用 Schirrmeister 原作）。
  - 引用建议：R. T. Schirrmeister et al., "Deep learning with convolutional neural networks for EEG decoding and visualization," *Human Brain Mapping*, vol. 38, no. 11, pp. 5391–5420, 2017.
- **FBCSP (Ang et al. 2008)**：MI 解码的 pre-deep-learning 标杆方法，至今仍是 BCI Competition IV 的强 baseline；论文 §2.6 提到 CSP 通道选择但只引用了 MNE-Python 的 Ledoit-Wolf 实现细节，未引用 FBCSP 原作。
  - 引用建议：K. K. Ang, Z. Y. Chin, H. Zhang, C. Guan, "Filter Bank Common Spatial Pattern (FBCSP) in Brain-Computer Interface," *IJCNN*, 2008.
- **Sakhavi et al. 2018** "Learning Temporal Information for Brain-Computer Interface Using Convolutional Neural Networks" (*IEEE TNNLS*)：FBCSP + CNN 混合方法，是 EEGNet 之外的另一条主流 deep MI 解码 baseline。
  - 引用建议：S. Sakhavi, C. Guan, S. Yan, "Learning Temporal Information for Brain-Computer Interface Using Convolutional Neural Networks," *IEEE TNNLS*, vol. 29, no. 11, pp. 5619–5629, 2018.

§5 局限性条目只声明"未与传统 CSP/FBCSP baseline 对比"，但即便不复现这些 baseline，**至少应在 §1.2 表 0 中加入它们作为文献锚点**——一篇论述 finger-MI 解码的论文，如果只对比了 [3] / [8] / [9] 三条文献而忽略 FBCSP / Schirrmeister，会被领域内审稿人立刻识别为文献覆盖不足。

#### 3.1.3 通道选择方法的原始文献完全未引

§2.6 列举了 4 种通道选择方法（FDR / CSP / Attention / Band Power），但**没有任何一项的方法学原始文献被引用**。这在 BCI 领域是不可接受的：

- **Fisher Discriminant Ratio (FDR)** for EEG channel selection：可参 Lal et al. 2004 "Support Vector Channel Selection in BCI" (*IEEE TBME*) 或 Schröder et al. 2005。
- **Common Spatial Pattern (CSP)**：Koles 1991（原作）+ Blankertz et al. 2008 "Optimizing Spatial Filters for Robust EEG Single-Trial Analysis" (*IEEE Signal Processing Magazine*) 是领域标准引用。
- **Gradient Attention**（基于模型梯度的特征/通道重要性）：至少应引用 Simonyan et al. 2014 (saliency maps) 或在 EEG 领域 Lawhern et al. 2018 [5] 已展示的 input gradient 解释方法。
- **Band Power + ANOVA F**：可追溯到 Pfurtscheller 经典 mu/beta ERD 文献以及 Blankertz et al. 2008 "The Berlin Brain–Computer Interface" (*IEEE TBME*) 的 SPoC / band power features 链。

至少补充 4–6 条上述方法学引用，否则 §2.6 读起来像"我们用了若干通道选择方法但不告诉你它们从哪儿来"。

#### 3.1.4 DAPT 概念的 BCI 文献基础未给出

§3.6 + §4.5 把"DAPT 在 EEG 基座模型上一致负迁移"作为一个领域级方法学命题提出。但：

- 论文未引用 EEG / 神经信号方向已有的 DAPT / cross-corpus pretraining 文献。即便讨论 NLP/CV 类比留给 R3，BCI 领域内部已有相关讨论（如 Demir et al. 2022 EEG-GPT 类工作、Wagh & Varatharajah 2020 cross-corpus EEG transfer 等），论文应至少引用 1–2 条。
- §4.5 / §4.8 / §7 的"EEG foundation model 'domain' 边界由信号级特征定义"命题（详见 3.3 节本评议）作为一个方法学论断被强力推出，但缺乏现有 EEG transfer 文献支撑——既不是建立在已有分类（如电极配置 vs 任务类型 vs 受试群体三因子分解）之上，也没有引用任何提出过该问题的先期工作。这个命题需要要么有领域文献支撑、要么明确标记为本文首次提出的假设。

#### 3.1.5 Pfurtscheller / Neuper 的 mu/beta ERD 经典工作引用单薄

§3.5.2 解读 4ch Band Power 的解剖学位置时，明确依赖 "Pfurtscheller & Neuper 经典手部 mu/beta ERD 强响应带（C3/C4 hand knob 区域）" 这一概念，但**全文未引用 Pfurtscheller & Neuper 任何工作**。论文 §4.4 / §7 等多处依赖 mu/beta ERD 物理解释，但参考文献列表 [2] 仅列了一篇 2001 年综述（"Motor imagery and direct brain-computer communication"）——这显然不够。建议至少补充：

- G. Pfurtscheller, F. H. Lopes da Silva, "Event-related EEG/MEG synchronization and desynchronization: basic principles," *Clinical Neurophysiology*, vol. 110, no. 11, pp. 1842–1857, 1999.
- C. Neuper, M. Wörtz, G. Pfurtscheller, "ERD/ERS patterns reflecting sensorimotor activation and deactivation," *Progress in Brain Research*, vol. 159, pp. 211–222, 2006.

### 3.2 XSI-FT 术语：novel naming vs 已知机制

XSI-FT (Cross-Subject-Initialized Per-Subject Fine-Tuning) 在 §3.3 被作为本文方法学贡献正式引入。然而**该机制本身（"先 LOSO/cross-subject pooled pretrain，再 per-subject fine-tune"）在 BCI 领域至少十年是已知的、约定俗成的训练协议**：

- Lotte et al. 2018 review "A review of classification algorithms for EEG-based brain-computer interfaces: a 10 year update" (*J. Neural Engineering*) 即把该范式分类为 "subject-adaptive transfer learning"。
- Jayaram & Barachant 2018 "MOABB: Trustworthy algorithm benchmarking for BCIs" 中 cross-session / cross-subject + per-subject finetune 是默认评估协议之一。
- Ding et al. [3] 自身的在线 MI 控制就是 cross-session pretrain + same-day finetune，机制上与 XSI-FT 同构，仅 finetune 时机不同。
- 在 EEG foundation model 文献中（CBraMod [4] / LaBraM [6] / BIOT），fine-tuning 阶段几乎都是 "pooled pretrain → per-subject head adaptation"，本质等价。

**问题**：论文将该机制以正式名称 (XSI-FT) 引入，并在摘要 / §1.4 / §3.3 / §4.6 / §4.8 / §7 反复使用，给人的印象是这是本文方法学新颖性。即便 Phase D 完整性检查认定该缩写本身（"XSI-FT"作为字符串）此前未被使用过，"提出一个新缩写"和"提出一个新方法"是两件事——后者要求该机制的某个**实质性方面**是新的。

**领域审稿人的预期反应**：会立刻识别出这是 well-known LOSO + per-subject finetune 的换名，并质疑作者文献功底——在 BCI 圈这是典型的"造词陷阱"，会显著减分。

**建议**（按优先级）：

1. **首选**：保留 XSI-FT 名称作为本文实验记号便利，但在 §3.3 第一次定义时加一段"文献溯源"说明：明确指出该机制对应 Lotte et al. 2018 的 "subject-adaptive" 类别 + Pan & Yang 2010 的 inductive transfer 框架在 EEG 上的 instance + Ding et al. [3] same-day finetune 的离线版本，本研究的贡献限于"在 finger-MI 数据 + EEG foundation model 设置下系统量化它的边际收益与饱和条件"。
2. **次选**：完全废弃 XSI-FT 缩写，改用领域标准术语 "pretrain-then-finetune (cross-subject pretrain → per-subject finetune)" 或 "subject-adaptive transfer"。这对论文叙事改动较大但学术上最稳妥。
3. **不建议**：保留现状。这会让论文在领域审稿环节直接遭遇可信度损失。

附带：论文 §3.3 与 §3.5.4 / §3.4.4 间反复重复 XSI-FT 定义（§3.3 / §3.4.4 / §3.5.4 三处展开），如果保留缩写应统一到 §3.3 一处定义、其他章节只引用，避免冗余。

### 3.3 DAPT 方法学论断的过度推广

§4.5 / §4.8 / §7 共同构造了一个高阶方法学命题：

> "**EEG 基座模型的 'domain' 边界由信号级特征（采样率、频段、电极配置）而非任务级语义定义**——粗运动 MI 数据不能作为 finger MI 任务的 domain-adaptive 来源，即使两者都属 'MI' 语义类别。"

作为 BCI 领域审稿人，我对这一命题持**强保留意见**，理由如下：

1. **单 source × 单 target 的样本不足以支撑普适命题**。本研究的 source pool 由 10 个外部 MI 数据集（其中 Stieger2021 占 79%，6.4× 于 target）组成，target 是单一 finger-MI 数据集。即使 V3 把 Stieger 占比降至 30%，也仅仅是在同一 source 池内部重新加权——并没有真正测试"如果换一个 source pool（如全部 finger-level 或全部 hand-grasp MI）会怎样"。论文 §4.5 自己也承认了这一点（"只在存在类型更接近的 source MI 数据可用时再考虑 DAPT"），但 §4.8 / §7 的总结句仍然把命题升级到了"信号级 domain 定义"的普适层面。
2. **"信号级特征 vs 任务级语义"二分本身在 EEG 文献中并无既有分类支撑**。文献中通常使用更细分的因子：(i) 电极配置（10-20 vs 高密 BioSemi）、(ii) 采样率/滤波带、(iii) 任务类型（粗运动 / 精细运动 / 静息态 / 病理）、(iv) 受试群体（健康 / 患者 / 年龄段）、(v) 录制条件（实验室 / 移动）。本文把 (i)+(ii) 归为"信号级"、把 (iii) 归为"任务级"——这一二分既非领域共识也未在论文里被定义清楚。建议至少明确化为"通道几何 + 信号采样配置错位 vs 任务表征语义对齐"，并承认本研究只测试了前者。
3. **§4.5 已识别"通道数极度异质"作为独立 caveat**，这实际上削弱了主命题——如果通道几何错位（Stieger 60ch + 其他低密度数据集占总样本 95%）独立驱动负迁移，那么"信号级 domain"就退化为"通道几何 domain"，与"采样率"和"频段"无关，原命题需要重新表述。
4. **跨 EEG foundation model 普遍性未验证**。即使该命题在 CBraMod 上成立，其他 backbone（LaBraM / BIOT / NeuroLM）的 DAPT 行为可能完全不同——LaBraM 的 codebook 离散化、BIOT 的多数据集联合预训练、NeuroLM 的 LLM 对接都改变了"什么算 domain"。§5 局限性 #7 已部分承认这一点。

**建议修订路径**：

- §4.5 / §4.8 / §7 把强命题"由信号级特征定义"改为弱化版本：**"在 CBraMod backbone + 当前 source-target 配置下，通道几何错位（target 128ch vs source 95% 低密度）与训练超参数（mask ratio / lr 调度）对 DAPT 结果的影响至少与任务粒度（粗运动 vs finger MI）相当；下游 BCI 实践应优先匹配通道几何与信号尺度，再考虑任务语义对齐。"**
- 在 §6 未来工作中明确加入一项："在 source pool 通道几何与 target 严格对齐（如全部 128ch finger-MI source）的条件下重做 DAPT，验证负迁移是否消失。"

不做这一修订的话，命题会被领域审稿人作为 over-claim 攻击点，且容易对后续读者形成误导（"看起来 CBraMod 不能 DAPT 进 MI 任务"——但严格说本研究只证明了"用 95% 低密度 source 对 128ch target 不能 DAPT"）。

### 3.4 Table 0（§1.2）的 apples-to-oranges 风险

§1.2 表 0 同时列出 Alazrai 2019 [8] (~65%, 64ch, 离线被试内)、Lee 2022 [9] (~70%, 256ch, 离线被试内)、Ding 2025 [3] (80.56% online, 128ch)、本文 90.68% (offline cross-subject) 等数值。即便论文在表 0 后加了"可比性说明" footnote（"评估范式不同，准确率数值不可直接比较"），**这一表格仍存在被读者过度解读的实质风险**。

具体问题：

1. **Ding et al. [3] 的 80.56% 是 online same-day finetune 性能**，本文 90.68% 是 offline cross-subject 离线评估——这两者 *测试集都不一样*。Ding [3] 的在线 session_finetune 设置中模型已见过 same-day base 数据并做了 fine-tune；本文 cross-subject 完全留出 Sess02_Finetune。在 BCI 领域共识下，**离线 cross-subject 的"难度"低于真实在线评估**（在线包含运动伪影 / 操作员变量 / 时间压力等噪声）。把本文的 offline 90.68% 与 [3] 的 online 80.56% 并列，会让不严谨的读者误认为"本文方法 +10 pp 优于在线 SOTA"。
2. 即便 footnote 已经免责，**表头"二分类准确率"列把不同评估范式的数字放在同一列就构成了视觉性的等价比较**。footnote 弱化了文字层面的 over-claim，但 figure 视觉冲击仍然在。
3. §1.4 贡献声明列表第 1 条"首次在手指级运动想象分类任务上对 EEG 基座模型与传统 CNN 进行全面对比"使用了"首次"措辞——这是合理的（[3] 自己只用 EEGNet，没做 CBraMod 对比），但与表 0 的视觉对比叠加会强化"本文 SOTA"的暗示。

**建议**：

- 把表 0 重命名并加副标题："**表 0. 已有 finger-level MI EEG 分类研究的方法学全景**（注：评估范式不同，数值不可作为性能优劣比较）"，并把 footnote 提升到表上方紧贴标题处。
- 在表中专门加一列 **"评估难度"**（offline-within / offline-cross / online-finetune / online-cross），明确标记每条记录的 difficulty regime。
- §1.4 第 1 条贡献删掉与 [3] 的隐性 SOTA 暗示，明确改为"首次将 EEG 基座模型 vs 紧凑 CNN 的对比置于 finger-MI 数据上的统一离线评估框架"。
- 摘要 / §7 结论中**不要**用"90.68%"这类数字与 [3] 的"80.56%"直接对话；改为说"在本文统一离线评估框架下，CBraMod 达到 90.68% cross-subject"。

不修订的话，领域审稿人会把这一表格作为"作者刻意营造 SOTA 印象"的指控点，影响整体论文的可信度。

---

## 4. Minor Concerns

1. **§3.2 EEGNet cross-subject 解读偏弱**（"未观察到显著收益，−1.43 pp, p = 0.456"）。BCI 文献中 EEGNet ~10K 参数对 21 名被试 pooled 数据"学不动"是已知现象（Lawhern 2018 [5] 自己也讨论过 small CNN 的容量上限）；论文应引用 [5] 的相关讨论。
2. **§3.5.2 4ch BP 的解剖学位置解释**。论文已经在 §3.5.2 表中明确地修正了 v2 草稿"4ch 落在 sensorimotor"的过度解读，这一谨慎态度值得肯定。但解读段落依赖"Pfurtscheller 经典手部 mu/beta ERD 强响应带（C3/C4 hand knob 区域）"——既然依赖这一概念，就应正式引用 Pfurtscheller 1999 / Neuper 2006（见本评议 3.1.5）。
3. **§3.4 extra sessions 的 longitudinal framing 与 BCI 文献的对照**。论文呈现 within-subject vs cross-subject 在纵向 session 上的差异化获益模式，但未与 BCI 领域已有的 longitudinal 研究（如 Wolpaw lab 的 trans-tympanic 长程 BCI、Schalk 2017 longitudinal MI 研究、Jeunet 2016 mental imagery learning）对照。即便不是核心贡献，至少应在 §4.4 加一段，把本文观察到的"低基线被试获益最多 (+18.75 pp)"现象与已有的"BCI illiteracy / responder effect" 文献链对接。
4. **§3.9 数据质量分类**。三名重度伪影被试 S04/S10/S14 的处理是合理的（leave-out 验证），但分类标准（"振幅 > 38K µV"等）来自内部 `data_quality_report.md`，未引用任何外部 EEG 伪影检测文献（如 Mognon 2011 ADJUST、Pion-Tonachini 2019 ICLabel）。即便不复现这些方法，也应至少把分类规则与领域共识做一下对照。
5. **EEGNet-16,4 vs EEGNet-8,2 的"重新搜索"叙事**（§2.4.1 + §2.5.1）。论文说 HPO 找到 EEGNet-16,4 比 [3] 的 EEGNet-8,2 / deepEEGNet 都更优——但 Lawhern 2018 [5] 原作就明确说明 EEGNet-8,2 是 within-subject 配置，cross-subject 时需要更大 F1/D。本文 HPO 的"发现"实际是在重新验证 [5] 已有结论，这一点应在 §2.4.1 / §2.5.1 引用 [5] 加以说明，避免给读者"我们独立发现"的错误印象。
6. **§7 结论"基座模型 transfer 路径与 NLP/CV 不同"**：这一论断的展开本属 R3 评议范围，但从纯 BCI 视角看它过于笼统。建议至少加一句说明本文的论断仅基于"CBraMod backbone × MOABB-style external MI source"这一具体配置。
7. **Ding et al. [3] cohort 筛选的影响**。§2.1 + §5 局限性 #2 已承认 21 名被试是 [3] 在 49 名招募者中筛选后的 responder cohort，对总体泛化的影响被高估。但论文正文（§3.1 / §3.2 / 摘要）的"21 名被试"叙事并未在每次出现时强化这一限制。建议在摘要至少加一句"在 [3] 筛选后的 21 名 responder cohort 上"。
8. **CBraMod 输入约定与 [4] 的对齐**（§2.2）。论文把 ÷100 描述为"以 100 µV 为尺度的数值归一化"——这是 [4] 的 fine-tuning convention 而非严格意义的 normalization；表述应更精确（如"沿用 [4] CBraMod 输入 scale convention"）。
9. **§2.3.1 渐进式分割与 [3] 原作一致性**：§2.3.1 表第二行写 "+Sess03 训练 = 标准 + Sess02_Finetune + Sess03_Base"——但 §2.3 标准训练已经包含 Sess02_Base，而 Sess02_Finetune 在 §2.3 是测试集。这意味着 +Sess03 step 把原测试集也并入训练集了。这种"逐 session 滚动"在 [3] 原作中是默认设置，但本文应明确说明"+Sess03 step 的测试集变更为 Sess03_Finetune，原 Sess02_Finetune 进入训练集"——否则读者可能困惑于测试集为什么变。

---

## 5. Required Literature Additions

按优先级与必要性排序：

### 5.1 必须添加（不加会导致领域审稿人直接拒稿）

1. **R. T. Schirrmeister et al., "Deep learning with convolutional neural networks for EEG decoding and visualization,"** *Human Brain Mapping*, 2017.
   理由：与 EEGNet [5] 并列的 BCI deep learning baseline；论文在 §3.7.1 讨论 EEGNet 容量阶梯时已经引用了 [3] 的 deepEEGNet，但 deepEEGNet 本身是 Schirrmeister 设计的衍生。该引用是 BCI deep learning 文献基础。
2. **S. Sakhavi, C. Guan, S. Yan, "Learning Temporal Information for Brain-Computer Interface Using Convolutional Neural Networks,"** *IEEE TNNLS*, 2018.
   理由：FBCSP+CNN 混合方法，是 EEGNet 之外的另一条主流 deep MI 解码 baseline，与本文的 §2.6 CSP-based channel selection 直接相关。
3. **K. K. Ang, Z. Y. Chin, H. Zhang, C. Guan, "Filter Bank Common Spatial Pattern (FBCSP) in Brain-Computer Interface,"** *IJCNN*, 2008.
   理由：MI 解码与 CSP 通道选择的领域基础引用；§2.6 描述 CSP 通道选择必须引用。
4. **B. Blankertz, R. Tomioka, S. Lemm, M. Kawanabe, K.-R. Müller, "Optimizing Spatial Filters for Robust EEG Single-Trial Analysis,"** *IEEE Signal Processing Magazine*, 2008.
   理由：CSP 现代综述与 robust 实现，配合 [4] 的 ACPE 讨论形成方法学背景。
5. **G. Pfurtscheller, F. H. Lopes da Silva, "Event-related EEG/MEG synchronization and desynchronization: basic principles,"** *Clinical Neurophysiology*, 1999.
   理由：§3.5.2 + §4.4 + §7 反复依赖 mu/beta ERD 概念；只引用 [2]（2001 综述）不够。
6. **W.-B. Jiang et al., "NeuroLM: A Universal Multi-task Foundation Model for Bridging the Gap between Language and EEG Signals,"** *ICLR*, 2025.
   理由：与 CBraMod [4] 同一会议同年发表的 EEG foundation model；§1.3 / §3.7 / §4.1 讨论 EEG 基座模型不引用 NeuroLM 是文献覆盖严重缺陷。

### 5.2 强烈建议添加

7. **C. Yang, M. B. Westover, J. Sun, "BIOT: Biosignal Transformer for Cross-data Learning in the Wild,"** *NeurIPS*, 2023.
   理由：与 CBraMod 的 ACPE 通道灵活性形成方法学对照（BIOT 的"biosignal sentence" tokenization vs ACPE）；§2.4.2 讨论 ACPE 时应作为对比引用。
8. **D. Zhang et al., "Brant: Foundation Model for Intracranial Neural Signal,"** *NeurIPS*, 2023.
   理由：500M 参数的另一类 EEG/SEEG foundation model；与本文 §3.7.3 "盲目扩参"的命题形成对照（Brant 显示 scaling 在 SEEG domain 是有效的）。
9. **F. Lotte et al., "A review of classification algorithms for EEG-based brain-computer interfaces: a 10 year update,"** *J. Neural Engineering*, 2018.
   理由：BCI 算法综述领域标准引用；§3.3 XSI-FT 文献溯源应引用其 "subject-adaptive transfer learning" 分类。
10. **C. Neuper, M. Wörtz, G. Pfurtscheller, "ERD/ERS patterns reflecting sensorimotor activation and deactivation,"** *Progress in Brain Research*, 2006.
    理由：§3.5.2 解剖学解读直接依赖 ERD/ERS 概念框架。

### 5.3 锦上添花

11. **V. Jayaram, A. Barachant, "MOABB: Trustworthy algorithm benchmarking for BCIs,"** *J. Neural Engineering*, 2018.
    理由：§2.7.1 提到"通过 MOABB 框架"收集 10 个外部 MI 数据集，但未引用 MOABB 原文。
12. **Z. J. Koles, "The quantitative extraction and topographic mapping of the abnormal components in the clinical EEG,"** *Electroencephalography and Clinical Neurophysiology*, 1991.
    理由：CSP 算法原始引用，配合 Blankertz 2008 形成 CSP 完整文献链。
13. **M. Ahn, S. C. Jun, "Performance variation in motor imagery brain–computer interface: a brief review,"** *J. Neuroscience Methods*, 2015.
    理由：§4.4 / §3.4 关于"低基线被试获益最大"现象与"BCI illiteracy"文献的对接。

总计建议添加 6 条必须引用 + 4 条强烈建议 + 3 条锦上添花 = **10–13 条**。从当前 9 条扩展到 ~20 条，对一篇 benchmarking + 三向消融的论文是合理覆盖密度（Ding [3] 自己列了 ~50 条参考；CBraMod [4] 列了 ~60 条；这些都是同类工作的合理水准）。

---

## 6. Theoretical Framework Assessment

**总体评估**：§4 讨论部分的整体框架是 *合理的*，但若干处把"CBraMod-specific 观察"过度推广到"EEG foundation model 通用属性"。具体如下：

### 6.1 §4.1（基座模型优势：何时与为何）

**优点**：三向分解（架构 / 预训练 / 容量）的叙事清晰，结合 §3.7 实验数据有力。"架构在 cross 主导、预训练在 within 主导"这一两段式结构是本文最有原创性的贡献，逻辑链条完整。

**问题**：

- 命题"基座模型价值随数据约束放大"在论文中按 within (~70 trial) > cross (~1.5K trial) 的方向描述——但严格说"数据约束"在 within 范式下指的是**单被试样本量小**，在 cross 范式下指的是**跨被试异质性高**，两者不是同一维度的"约束"。建议在 §4.1 第二段明确区分这两种"约束"。
- "EEGNet 内扩参反而 −25 pp"作为 §3.7.1 的核心观察，但论文未充分讨论为何扩参 EEGNet 会落入 chance（仅简单归因为"分布偏移噪声放大"）。一个合理的领域解释是：cross-subject pooling 下不同被试的 spatial filter 模式差异极大，~10K 参数 EEGNet 学到的是"群体平均 spatial filter"；扩参后模型有能力学到 subject-specific filter 但 21 个被试的 ground truth filter 互相冲突，导致优化方向退化为 chance。这个机制性解释在 §4.1 缺失，建议补充。

### 6.2 §4.4（纵向数据扩展）

**优点**：cross-subject pooled 模型在 +Sess05 上仅 +0.86 pp 的发现是有领域价值的——它直接挑战了 BCI 文献中"更多 cross-subject 数据 → 更好的 cross-subject 模型"的隐性假设。论文用"两层异质性叠加"（cross-session + cross-subject）解释，方向合理。

**问题**：未与 BCI longitudinal 文献对接（见 4.3）。

### 6.3 §4.5（DAPT 局限）

**优点**：V3 实验把"Stieger 主导"与"整体粗运动 MI 域错配"做了拆分，方向清晰；§4.5 第三段加入"通道几何异质"作为独立 caveat 是审慎的，符合实际数据情况。

**问题**：见本评议 3.3。强命题（信号级 vs 任务级 domain）需要弱化或撤回。

### 6.4 §4.8（综合：数据稀缺梯度下的策略选择）

**优点**：把 5 个数据可得性档位串联起来，配以具体方法推荐（CBraMod cross / FDR 32ch / XSI-FT 等），实操性强。这是论文工程价值最高的一段，对 BCI 部署研究人员有直接指导意义。

**问题**：

- "32ch FDR 距上限远（XSI-FT 有空间），8ch BP 接近上限（XSI-FT 反而引入过拟合）"——这一解释框架基于 §3.5.4 的三个数据点（128ch / 32ch FDR / 8ch BP），样本量过小。即便 §6 #2 已承认需要更全的 (channel × method) 矩阵，§4.8 / §7 仍把它包装成确定性结论。建议加 hedge："基于三个数据点的初步框架，需更多 (channel, method) 组合验证。"
- "EEG 基座模型 'domain' 边界由信号级特征定义"作为综合贯穿命题——见本评议 3.3。

### 6.5 §7 结论

发现 1–5 整体表述清晰，但发现 4 / 发现 5 都把"特定配置下的观察"上升为"方法学命题"。最后一段"EEG 基座模型的 transfer 路径与 NLP/CV 的 domain-adaptive pre-training 范式不同"是过强的总结句——本研究并未直接对照 NLP/CV 的 DAPT 设置。建议把这句话弱化或移除（详细 NLP/CV 类比留给 R3 评议）。

---

## 7. Confidence in Review

**Confidence: 4 / 5**

依据：
- 我对 finger-level MI 文献链 ([8] / [9] / [3])、EEG foundation model 主线 ([4] / [6] / BIOT / Brant / NeuroLM)、Pfurtscheller / Wolpaw 经典 BCI 理论框架（mu/beta ERD、subject-adaptive transfer、BCI illiteracy）均有直接研究经验，对本评议的文献覆盖建议、XSI-FT 术语判断与 DAPT 命题约束均有把握。
- 唯一减一分的考虑：本评议对 §3.7 三向消融的方法学评估（架构 vs 预训练 vs 容量分解）部分依赖论文报告的数据正确性；如果 R1 方法论评议发现 §3.7 实验设置存在统计或 HPO 公平性问题，本评议对该部分的"strength"判断需要相应下调。这是与 R1 评议视角的天然耦合，无法在独立评议中完全消除。

---

**报告产出于**：2026-05-10
**报告签署**：R2 (Senior MI-BCI Researcher)
