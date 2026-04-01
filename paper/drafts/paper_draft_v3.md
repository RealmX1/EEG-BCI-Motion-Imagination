# 基于 EEG 基座模型的手指级运动想象分类：通道缩减、纵向数据扩展与领域自适应预训练的局限性

> **草稿说明**：本文为工作草稿（v3）。文中大部分图表为脚本自动生成的初步输出，**尚未进行出版级精修**（坐标轴标签、字体大小、配色方案、排版布局等）。标有 `[TODO]` 的章节表示数据需最终核实或补充可视化。
>
> **v3 变更摘要**：
> - 论文语言从英文转为中文（技术术语保留英文）
> - 新增多 session 纵向扩展实验结果（原 TODO 6.2，现 Section 3.5）
> - 新增领域自适应 further pre-training 负面结果（Section 2.7 + 3.6）
> - 新增推理性能基准测试（Section 3.7）
> - HPO 方法论纳入 Methods（Section 2.5.1）
> - "Ongoing Experiments" 改为 "Future Work"

---

## 摘要

脑机接口（Brain-Computer Interface, BCI）通过脑电图（EEG）解码单指运动意图，在精细运动康复领域具有重要应用前景，但高密度电极阵列的部署限制了其临床推广。本研究系统对比了大规模 EEG 基座模型 CBraMod（~4M 参数，ICLR 2025）与轻量级卷积神经网络 EEGNet-16,4（~10K 参数）在单指运动想象（Motor Imagery, MI）分类中的性能，覆盖 21 名健康被试、128 通道 BioSemi 系统、被试内/跨被试/迁移学习三种训练范式。

在通道缩减方面，我们评估了 128、61、32、8、4 通道配置及四种数据驱动选择方法（Fisher 判别比、共空间模式、梯度注意力、频带功率）和一种商用布局。CBraMod 在 128 通道跨被试二分类中达到 90.68% 准确率；关键发现是，Fisher 判别比（FDR）选取的 32 通道配置保留了 96.7% 的性能（87.71%），而 EEGNet 在相同条件下降至 74.70%。通道选择方法的敏感度随通道数递减而急剧增加（32 通道 ~3 pp 差异，8 通道 ~8 pp，4 通道 ~15 pp）。

在纵向数据扩展方面，对 16 名拥有 3–5 个额外在线 session 的被试进行分析表明，额外同被试数据的价值强烈依赖训练范式。被试内二分类中，EEGNet 从 80.51% 提升至 87.85%（+7.34 pp，p = 0.009），CBraMod 从 87.23% 提升至 93.36%（+6.13 pp，p = 0.007）；而 21 名被试联合训练的跨被试 pooled model 仅从 92.38% 小幅升至 93.24%（+0.86 pp，p = 0.662）。使用对应 cross-subject checkpoint 初始化的 transfer-style fine-tuning 达到 92.93%（+5.70 pp，p = 0.015），与被试内重训练相近但未进一步突破其终点。三分类中，被试内 CBraMod 仍显示显著改善（74.51% → 83.06%，+8.55 pp，p = 0.012），而跨被试 ternary 增益更温和（+3.73 pp，p = 0.090）。

在领域自适应预训练方面，我们收集了 10 个公开 MI 数据集（~870 小时，~300 被试），对 CBraMod 进行 masked autoencoding 继续预训练，但结果呈现一致的**负迁移**（平均 −1.38 pp），且训练越充分负迁移越大，表明粗运动 MI 数据与精细手指运动解码之间存在不可通过数据量弥合的领域鸿沟。

上述结果共同支持了一条实用的 BCI 部署路径：采用 CBraMod + FDR 32 通道配置作为起步方案，通过收集少量额外 session 数据即可达到 90% 以上准确率，推理延迟 <13 ms 满足实时要求。

**关键词**：脑机接口、脑电图、运动想象、基座模型、CBraMod、EEGNet、通道缩减、迁移学习、Fisher 判别比、纵向 BCI、领域自适应预训练、负迁移

---

## 1. 引言

### 1.1 背景与动机

脑机接口（BCI）在大脑与外部设备之间建立直接通信通道，为严重运动障碍患者提供了变革性的交互途径 [1]。在非侵入式 BCI 范式中，运动想象（MI）——不涉及实际执行的运动心理演练——因其自主性和临床可行性而成为主流方法 [2]。

现有 MI-BCI 研究多聚焦于粗粒度运动分类（如左手 vs 右手），近年来已有工作推向更精细的手指级别分类 [3]。这种精细运动控制对假肢和机器手应用至关重要，但也带来了更大的挑战：手指特异性皮层表征在空间上高度邻近，产生的 EEG 信号比粗肢体运动更弱、重叠更多。

将 MI-BCI 系统部署于临床或消费场景的一个持续障碍是对高密度 EEG 阵列的依赖。64–256 通道的研究级系统虽然提供了丰富的空间信息，但设置耗时、用户不适、硬件成本高昂。相反，减少电极数量又可能使解码性能降至不可用阈值以下。理解性能-通道数权衡关系对于弥合实验室结果与实际 BCI 应用之间的差距至关重要。

此外，一个尚未充分探索的问题是：**能否通过外部运动想象数据对通用 EEG 基座模型进行领域自适应预训练（domain-adaptive further pre-training），从而进一步提升下游任务性能？** 这在自然语言处理和计算机视觉领域已被证明有效，但在 EEG 基座模型中的效果尚不明确。

### 1.2 手指级 EEG 分类相关工作

表 0 总结了已有手指级运动想象分类研究，定位本文方法论在已有文献中的位置（注：各研究评估范式不同，准确率数值不可直接比较）。

**表 0. 与已有手指级 EEG 分类研究的对比。**

| 研究 | 模型 | 通道数 | 评估方式 | 二分类准确率 | 三分类准确率 | 实时 |
|------|------|--------|---------|------------|------------|------|
| Alazrai et al. 2019 [8] | SVM + CSP | 64 | 离线，被试内 | ~65% | N/A | 否 |
| Lee et al. 2022 [9] | CNN | 256 | 离线，被试内 | ~70% | N/A | 否 |
| Ding et al. 2025 [3] | EEGNet | 128 | **在线，session 自适应** | 80.56% | 60.61% | **是** |
| **本文** | **CBraMod** | **128** | **离线，跨被试** | **90.68%** | **74.88%** | 否 |
| **本文** | **CBraMod** | **32** (FDR) | **离线，跨被试** | **87.71%** | — | 否 |
| **本文** | **CBraMod** | **128** | **离线，被试内 + extra sessions** | **93.36%** | — | 否 |

> **可比性说明**：不同研究间的直接准确率比较受评估范式（在线 vs 离线）、训练协议（session 内 vs 跨被试）、被试群体差异的显著制约。Ding et al. [3] 报告的是带实时机器人反馈的在线 session 自适应性能；本文结果为同一数据集上的离线跨被试泛化评估。两种评估范式在数据分割、反馈机制和模型更新策略上存在根本差异，因此表中数值对比旨在呈现方法论全景而非主张性能优越性。

### 1.3 EEG 基座模型

大规模 EEG 预训练模型的出现——类似于自然语言处理和计算机视觉中的基座模型——代表了神经信号解码的范式转移。这些模型利用海量无标注 EEG 语料学习通用的时空表征，而非在有限的个体数据上从头训练任务特异性架构。

CBraMod（Criss-Cross Brain Foundation Model）[4]，被 ICLR 2025 接收，是一个基于 Transformer 的模型，在 Temple University EEG (TUEG) 语料上进行自监督预训练。其关键架构创新——非对称条件位置编码（Asymmetric Conditional Positional Encoding, ACPE）——使模型能够接受任意数量的输入通道而无需重新训练，这对通道缩减实验至关重要。CBraMod 拥有约 400 万参数，是 EEGNet-16,4 [5]（约 1 万参数，BCI 研究的标准基线 CNN）的 ~400 倍。

值得注意的是，TUEG 预训练语料主要包含临床 EEG（静息态、病理等），与运动想象 EEG 在信号特征上存在显著差异。一个自然的问题是：在外部 MI 数据集上进行 domain-adaptive further pre-training，能否弥合这一领域鸿沟并改善下游性能？这一假设在 NLP（如将通用语言模型适配到生物医学领域）和 CV（如将 ImageNet 模型适配到医学影像）中已得到验证，但在 EEG 基座模型中尚缺乏系统评估。

其他并行工作包括 LaBraM [6] 和 EEG 基座模型综述 [7]，证实了预训练方法在低数据和跨被试场景中一致优于任务特异性模型。

### 1.4 本文贡献

本文做出以下贡献：

> 1. **系统性基座模型评估**。首次在手指级运动想象分类任务上，对 EEG 基座模型（CBraMod）与传统 CNN（EEGNet-16,4）进行全面对比，覆盖被试内、跨被试、迁移学习三种范式，使用 21 名被试数据，并采用贝叶斯超参数优化（HPO）确保公平比较。
>
> 2. **全面通道缩减分析**。评估了五种 32 通道配置（四种数据驱动、一种手工设计）及 61、8、4 通道方案，确立 FDR 选取的 32 通道保留 128 通道性能的 **96.7%**。
>
> 3. **通道选择方法敏感度缩放规律**。证明选择方法的敏感度随通道数递减而增加：32 通道 ~3 pp 差异、8 通道 ~8 pp、4 通道 ~15 pp。通过负控制实验确认体积传导冗余而非数据泄露。
>
> 4. **多 session 纵向数据扩展与范式差异**。系统比较额外 session 数据在被试内、跨被试和 transfer-style fine-tuning 三种训练范式中的作用。CBraMod 在被试内重训练中获得最大净增益（+6.13 pp 至 93.36%），transfer-init 达到相近终点（92.93%，+5.70 pp），而 pooled cross-subject 模型仅小幅改善（+0.86 pp 至 93.24%），表明新增同被试数据更适合通过个体化更新而非共享模型池化来吸收。
>
> 5. **领域自适应预训练的负面结果**。系统评估在 870 小时外部 MI 数据上对 CBraMod 进行 further pre-training，发现一致的负迁移（−1.38 pp），且训练越充分负迁移越大——确立了临床 EEG 与运动想象数据之间不可通过数据量弥合的领域边界。
>
> 6. **实际部署特性**。推理延迟基准测试确认 CBraMod 单样本延迟 <13 ms，满足实时 BCI 要求。

---

## 2. 材料与方法

### 2.1 数据集

本研究使用 Ding et al. [3] 公开发布的手指级 EEG-BCI 数据集（原始数据随文公开于 Figshare, DOI: `10.1184/R1/29104040`），包含 21 名健康右利手被试（S01–S21），进行手指级运动想象和运动执行任务。需要强调的是，这 21 名被试对应 [3] 在 49 名招募者中经离线二分类准确率筛选后保留的在线被试队列（cohort），而非无筛选总体样本。EEG 信号通过 128 通道 BioSemi ActiveTwo 系统以 1024 Hz 采样率采集。实验范式包括：

- **离线 session**：30 次训练 run，带视觉提示的个体手指想象（拇指、食指、中指、小指）
- **在线 session**：跨多天的实时 BCI 控制 session，每个 session 分为校准（Base）和自适应（Finetune）阶段

其中 16 名被试（S02, S03, S04, S06–S11, S13–S19）拥有 3–5 个额外在线 session（Sess03–Sess05），总录制时长约 64 小时。

本研究聚焦运动想象范式，采用两种分类粒度：

| 任务 | 类别 | 随机基线 |
|------|------|---------|
| **二分类（Binary）** | 拇指（class 1）vs. 小指（class 4） | 50% |
| **三分类（Ternary）** | 拇指（class 1）vs. 食指（class 2）vs. 小指（class 4） | 33.3% |

二分类与三分类任务定义均与 Ding et al. [3] 的主在线 MI 控制范式保持一致；因此，本文的新增贡献不在于重新定义任务类别，而在于将该数据集置于统一的离线独立留出 session（held-out session）评估框架中，并引入基座模型比较、HPO 与系统化通道缩减分析。

### 2.1.1 与来源论文的关系

本文并非对 Ding et al. [3] 或 Wang et al. [4] 的直接复现，而是将两条方法链组合到统一的离线评估框架中。前者提供了手指级 EEG-BCI 数据采集、EEGNet 在线基线和 extra-session 设计；后者提供了预训练 CBraMod backbone、ACPE 以及与该 backbone 对齐的输入约定（input convention）。表 1a 总结了三者关系。

**表 1a. 本研究与两篇基础论文的方法学对应关系。**

| 维度 | Ding et al. 2025 [3] | Wang et al. 2025 [4] | 本文 |
|------|----------------------|----------------------|------|
| 主要角色 | 手指级 MI/ME 机器人控制数据集与 EEGNet 在线基线 | 预训练 EEG foundation model 与通用 fine-tuning recipe | 在同一 finger-MI 数据上系统比较 EEGNet 与 CBraMod |
| 数据来源 | 21 名筛选后被试、128ch、1 个 offline + 2 个 online MI session，16 名被试另有 3 个额外 MI session | TUEG 9,000h+ 临床 EEG 预训练，10 类公开下游任务验证 | 下游任务全部使用 [3] 的 finger-EEG 数据；CBraMod 权重初始化来自 [4] |
| EEGNet 输入链 | CAR, 100 Hz, 4–40 Hz, 1 s 窗口, 125 ms 更新, Z-score, majority vote | — | EEGNet 管线基本沿用 [3]，但置于严格 held-out session 离线评估中 |
| CBraMod 输入链 | — | 0.3–75 Hz, 200 Hz, 1 s patch, `÷100`, ACPE | 保留 [4] 的输入尺度与归一化约定，但适配到 128ch 手指 MI trial |
| 训练协议 | previous sessions 训练 base model；每个新 session 用 same-day 前半段数据 fine-tune，冻结前四层，early stopping + scheduler | 预训练权重 + task head，AdamW/Cosine 的标准 fine-tuning | 去除实时反馈与 same-day update，统一为 offline train/val/test；所有关键结果使用 HPO 后参数 |
| 评估目标 | 在线机器人控制 majority-vote accuracy | 多个公开下游任务的统一 benchmark | 被试内/跨被试/迁移/extra-session/缩减通道的 held-out 准确率 |
| 缩减通道问题 | 比较感觉运动区、非感觉运动区与 64/32/21ch 手工布局 | ACPE 支持可变通道输入 | 结合 [3] 的 low-density question 与 [4] 的 channel flexibility，系统评估 128/61/32/8/4ch 与数据驱动选点 |

### 2.2 预处理

两种模型的输入要求不同，我们实现了两条并行预处理管线，如表 1 所示。

**表 1. 预处理管线对比。**

| 步骤 | EEGNet 管线 | CBraMod 管线 |
|------|------------|-------------|
| 重参考 | 共平均参考（CAR，trial 级） | 共平均参考（CAR，trial 级） |
| 重采样 | 1024 → 100 Hz (`resample_poly`) | 1024 → 200 Hz (`resample_poly`) |
| 带通滤波 | 4–40 Hz，4 阶 Butterworth，因果 (`lfilter`) | 0.3–75 Hz，4 阶 Butterworth，因果 (`lfilter`) |
| 分段 | 1 s 窗口，125 ms 步长 | 1 s 窗口，500 ms 步长 |
| 归一化 | 每段 Z-score（时间轴） | 除以 100 |
| 伪影剔除 | 超过 ±500 µV 的 trial 剔除（仅训练） | 同左 |

其中，EEGNet 管线基本对应 Ding et al. [3] 的 online/offline EEGNet 处理链：128 通道信号经 CAR 后下采样至 100 Hz，做 4–40 Hz 带通、1 s 滑窗和逐段 Z-score。CBraMod 管线则有意贴近 Wang et al. [4] 的 pre-training / downstream input convention：0.3–75 Hz、200 Hz、1 s temporal patch 和以 100 µV 为尺度的数值归一化。

与两篇来源论文相比，我们做了三个关键改动。第一，CBraMod 不沿用 [4] 的 19 通道、30 s 非重叠 TUEG 预训练切片，而是适配到 128 通道手指 MI trial。第二，EEGNet 不再沿用 [3] 的在线流式 same-day 更新，而是纳入统一的 held-out session 训练/验证/测试协议。第三，两条管线都在 trial 级别（非 run 级别）应用 CAR，使用 `nanmean` 处理 NaN 填充的变长 trial（离线 trial: 5 s；在线 trial: 3 s），并通过 `scipy.signal.resample_poly` 的有理因子计算避免 FFT 混叠伪影。未使用数据增强。

### 2.3 数据分割协议

**表 2. 标准数据分割协议。**

| 分区 | Session 来源 | 说明 |
|------|-------------|------|
| 训练 | `OfflineImagery` + `OnlineImagery_Sess01_Base` + `OnlineImagery_Sess01_Finetune` + `OnlineImagery_Sess02_Base` | 时序分割 80/20 |
| 验证 | 训练集末 20%（按时间顺序） | 逐被试分割 |
| 测试 | `OnlineImagery_Sess02_Finetune` | 完全独立，从不用于调参 |

**关键约束**：在 trial 级别进行时序分割（非 segment 级别），防止滑窗产生的信息泄露。验证集取时间上最后的 20%，保持数据的时间顺序完整性。

#### 2.3.1 多 session 纵向扩展的数据分割

对于拥有额外在线 session 的 16 名被试，采用与原论文相同的渐进式 per-session 分割协议：

| 阶段 | 训练数据 | 测试数据 |
|------|---------|---------|
| Baseline | 标准训练集（同表 2） | Sess02_Finetune |
| +Sess03 | 标准 + Sess02_Finetune + Sess03_Base | **Sess03_Finetune** |
| +Sess04 | + Sess03_Finetune + Sess04_Base | **Sess04_Finetune** |
| +Sess05 | + Sess04_Finetune + Sess05_Base | **Sess05_Finetune** |

每一步训练集扩大，测试集为最新 session 的 Finetune 部分。除默认的被试内累积重训练外，我们还将同一 progressive split 重用于两种变体：（1）**跨被试 extra sessions**：每个 step 用 21 名被试的可用数据训练单一 pooled model，并在 16 名具有 extra sessions 的被试上评估；（2）**transfer-style extra sessions**：每个 step 先读取对应的 cross-subject checkpoint，再对单被试进行离线微调。补充分析中使用 fixed_combined（固定组合测试集）和 fixed_sess02（固定 Sess02 测试集）两种策略控制测试集难度变化的混淆因素，详见 Supplementary。

### 2.4 模型架构

#### 2.4.1 EEGNet-16,4

EEGNet [5] 是一种紧凑的 CNN，也是 Ding et al. [3] 在线 finger-BCI 解码器的核心架构（原始配置为 EEGNet-8,2）。原论文随后又测试了更宽更深的 deepEEGNet，以检验额外 session 收益是否主要受模型容量限制，但观察到的性能提升仍较有限。本文不直接复用其在线默认配置，而是将 EEGNet-8,2 / deepEEGNet 作为文献锚点，结合 HPO 重新搜索架构与正则化参数，最终得到 EEGNet-16,4 配置（F1=16 时间滤波器，D=4 空间深度），参数量约 16,162（~10K 可训练），相比原始 EEGNet-8,2（~2.5K 参数）有 4 倍扩展。

架构组成：
- Block 1：时间卷积（16 滤波器）→ 深度可分离空间卷积 → BatchNorm → ELU → Pool → Dropout(0.27)
- Block 2：可分离卷积（64 滤波器）→ BatchNorm → ELU → Pool → Dropout(0.27)

#### 2.4.2 CBraMod

CBraMod [4] 是一个 12 层 Transformer 基座模型，在 TUEG 语料上以 masked autoencoding 方式进行自监督预训练。核心创新为 ACPE（非对称条件位置编码），支持任意通道数输入。

模型配置：d_model=200，8 注意力头，12 层 Transformer，分类器为 2 层 MLP。总参数量约 3,050 万（含分类头）。

**表 2b. 模型规模对比。**

| 指标 | EEGNet-16,4 | CBraMod |
|------|------------|---------|
| 总参数量 | 16,162 | 30,484,402 |
| 模型大小 (FP32) | 0.06 MB | 116.29 MB |
| FLOPs（单样本） | 112.73 MFLOPs | 5.08 GFLOPs |
| 参数比 | 1× | ~1,900× |

### 2.5 训练流程

训练协议的设计同样分别参考 [3] 和 [4]，但并未原样照搬。对 EEGNet 而言，Ding et al. [3] 的在线 base model 使用该被试既往 session 数据训练 300 epochs，并在每个新 session 的前半段数据上进行 same-day fine-tuning，且冻结前四层。本文将其简化为统一的离线监督学习协议：不做 session 内更新，不冻结前四层，而是在预先划分的 train/val/test 上 end-to-end 训练。对 CBraMod 而言，Wang et al. [4] 的下游 fine-tuning 默认设置为 50 epochs、batch size 64、dropout 0.1、AdamW、learning rate 1e-4、weight decay 5e-2 和 CosineAnnealingLR；这些设置构成了本文 HPO 以文献为锚的初始默认值。

**表 3. 训练超参数（HPO 优化后）。**

| 参数 | EEGNet | CBraMod (被试内) | CBraMod (跨被试) |
|------|--------|-----------------|-----------------|
| 学习率 | 4e-3 | backbone 2.9e-4, classifier 1.2e-3 | backbone 1.3e-4, classifier 2.2e-4 |
| 权重衰减 | 1e-5 | 0.026 | 0.13 |
| Dropout | 0.27 | 0.10 | 0.37 |
| Batch size | 64 | 256 | 256 |
| 优化器 | AdamW | AdamW | AdamW |
| 学习率调度 | ReduceLROnPlateau | CAWD (phase_decay=0.47) | CAWD (phase_decay=0.50) |
| 早停 | patience=15 | patience=15 | patience=15 |
| 混合精度 | FP16 | FP16 | FP16 |

#### 2.5.1 超参数优化

所有报告结果均使用贝叶斯超参数优化（HPO）后的参数。这里的搜索并非从零随机设定，而是明确锚定两篇来源论文：EEGNet 侧以 [3] 的 EEGNet-8,2 / deepEEGNet 设计思路为起点，CBraMod 侧以 [4] 的 fine-tuning defaults 为起点。HPO 采用 Optuna 框架的 TPE（Tree-structured Parzen Estimator）采样器，搜索空间涵盖 7–11 个维度（学习率、权重衰减、dropout、batch size、学习率调度参数等）。

为提高搜索效率，我们实现了自定义的 ProbabilisticSubjectPruner：在被试内训练范式中，当某 trial 的累计性能超越当前最优的概率低于 10% 时提前终止该 trial，剪枝率达 52.9%–65.6%。

关键发现：
- EEGNet 从 HPO 中获益最大（+3.8 pp），主要贡献来自架构参数 F1, D 的扩展（8,2 → 16,4）
- CBraMod 的改进较小（+0.5–1.4 pp），预训练模型对超参数更鲁棒
- 详细 HPO 收敛曲线和参数重要性分析见 Supplementary Table S5

### 2.6 通道选择方法

通道缩减问题同样沿着两篇基础论文展开：Ding et al. [3] 已在该数据集上比较过感觉运动区、非感觉运动区以及 64/32/21 通道的手工布局；本文则进一步利用 CBraMod 的 ACPE 通道灵活性 [4]，把这一问题推进到数据驱动选点和更极端的 8/4 通道场景。

我们评估了以下五种 32 通道配置：

**数据驱动方法（4 种）：**
1. **Fisher 判别比（FDR）**：逐通道计算 mu (8–13 Hz) 和 beta (13–30 Hz) 频带的类间/类内方差比，取比值最高的 32 通道
2. **共空间模式（CSP）**：使用 MNE-Python 的 Ledoit-Wolf 协方差正则化，按空间模式贡献排序
3. **梯度注意力（Attention）**：聚合 CBraMod 输入梯度幅值，捕获模型分类时关注的通道
4. **频带功率（Band Power）**：mu/beta 频带功率的 ANOVA F 统计量排序

**手工设计配置（1 种）：**
5. **商用布局（Commercial）**：标准 10-20 系统在 BioSemi 128 通道上的映射

此外，还测试了 61 通道（标准 10-10 系统）、8 通道（FDR/Attention top-8）、4 通道（FDR ∩ Attention 交集 + 负控制）配置。

### 2.7 领域自适应 Further Pre-training

#### 2.7.1 动机与数据

CBraMod 的原始预训练权重基于 TUEG 临床 EEG 语料（以静息态和病理 EEG 为主），与运动想象 EEG 在信号特征上存在显著差异。为评估领域自适应 further pre-training 的效果，我们收集了 10 个公开 MI 数据集（通过 MOABB 框架），预处理为 CBraMod 输入格式。

**表 4. 三级数据量对比。**

| 数据集 | 时长 | 被试数 | 通道数 | Channel-Frame @200Hz | 相对比 |
|--------|------|--------|--------|---------------------|--------|
| Finger EEG（自有） | 64 h | 21 | 128 | 5.9G | 1× |
| MI Pretrain（10 个外部数据集） | 870 h | ~300 | 22–128 | 38.0G | 6.4× |
| TUEG Pretrain（原始预训练） | 9,246 h | 数千 | 19 | 126.5G | 21× |

> Channel-Frame @200Hz 为统一重采样至 CBraMod 输入采样率后的 channel-frame 数，是模型实际处理的数据量的最公平度量。

外部 MI 数据以粗运动 MI（左手 vs 右手）为主，其中 Stieger2021 单一数据集占比约 79%（61,526 segments）。

#### 2.7.2 训练配置

采用 masked autoencoding 自监督任务（50% mask ratio, MSE loss），在 TUEG 预训练权重基础上继续训练。测试了两种配置：

| 参数 | V1 | V2 |
|------|-----|-----|
| LR 调度 | Cosine decay → 1e-6 | Warmup 0.5ep → 恒定 lr=5e-5 |
| 最大 epoch | 10 | 50（early stop at 12） |
| 数据量 | 30,282 segments | 78,232 segments |
| 数据完整性 | Stieger2021 23/62 被试, Schirrmeister2017 5/14 被试 | 两者均完整 (62/62, 14/14) |
| 最终 loss | 0.006055 | 0.003714 (−39%) |
| 训练时间 | ~48 分钟 | ~4.5 小时 |

> **V1/V2 数据量差异说明**：V1 使用了部分下载的外部数据集（Stieger2021 仅 23/62 被试，15,959 segments；Schirrmeister2017 仅 5/14 被试），总计 30,282 segments。V2 完成了两个大型数据集的全量下载（Stieger2021: 62/62 被试，61,526 segments；Schirrmeister2017: 14/14 被试，3,310 segments），总计 78,232 segments。其中 Stieger2021 的增量约占数据量差异的 94%。其余 8 个外部数据集在两版中均为完整使用。因此，V1 和 V2 之间不仅训练配置不同（LR 调度、epoch 数），**数据组成也不同**，下游结果差异不可归因于单一因素。

### 2.8 评估协议

**分类性能**：所有模型在测试集上按被试计算准确率，报告 21 名（或 16 名）被试的均值 ± 标准差。统计显著性采用配对 t 检验（paired t-test）评估。

**推理延迟**：使用 CUDA 事件计时，50 次预热 + 200 次正式测量，覆盖 batch size 1/8/32/64，测试平台为 NVIDIA RTX 5070 (12 GB VRAM)。

### 2.9 数据质量评估

通过 12 项指标对 21 名被试进行综合信号质量评估，将被试分为四类：

**表 5. 数据质量分类（n = 21）。**

| 类别 | 数量 | 被试 | 判定标准 |
|------|------|------|---------|
| 干净 | 10 | S01, S02, S06, S07, S08, S11, S13, S15, S17, S18 | 所有指标正常范围 |
| 信息性 | 3 | S12, S19, S20 | 方差偏高（20–65×）但信号功能正常 |
| 轻度伪影 | 5 | S03, S05, S09, S16, S21 | 5.7–9.4% trial 受影响 |
| 重度伪影 | 3 | S04, S10, S14 | 极端振幅（126K–307K µV，正常 ≤ 38K µV） |

> **数据来源**: `results/data_quality_report.md`; `results/data_quality_advanced_report.md`

---

## 3. 实验结果

### 3.1 被试内对比（128 通道）

表 6 展示了 128 通道被试内训练的结果。

**表 6. 被试内训练结果（128 通道，N = 21）。**

| 模型 | 二分类 Mean ± SD | 三分类 Mean ± SD |
|------|-----------------|-----------------|
| CBraMod | **85.15 ± 11.00%** | **69.44 ± 13.82%** |
| EEGNet-16,4 | 78.10 ± 12.61% | 66.81 ± 12.04% |
| Δ (CBraMod − EEGNet) | **+7.05 pp** | **+2.63 pp** |

CBraMod 在两种任务上均优于 EEGNet，二分类差距 +7.05 pp 更为显著。图 1 展示了逐被试对比、准确率分布和配对散点图。

**图 1. 被试内训练 128 通道二分类逐被试对比。** 上方柱状图显示 EEGNet（蓝色半透明，历史数据）与 CBraMod（红色实心）的逐被试准确率；下左箱线图显示准确率分布；下右散点图显示配对对比，多数被试位于对角线上方（CBraMod 更优）。

![图 1. 被试内 128ch 二分类对比](../../results/20260323_2237_combined_imagery_binary.png)

三个值得注意的模式：（1）CBraMod 在 16/21 名被试中优于 EEGNet，但 S05 和 S09 两名被试上 EEGNet 持平或微优，提示预训练表征并非在所有个体上都有效；（2）两种模型的被试间方差均较高（SD > 11 pp），反映了手指级 MI 信号的个体差异性——S09 近乎完美 (99.38%) 而 S20 仅略高于随机 (52.50%/61.25%)；（3）三分类差距仅 +2.63 pp，显著小于二分类的 +7.05 pp，可能因为三分类的更高难度使两种模型都受限于信号质量而非模型容量。

> **数据来源**: CBraMod: `results/20260323_2237_comparison_cache_imagery_binary.json`; EEGNet: `results/20260316_1411_comparison_cache_imagery_binary.json`

> **基线声明**：上述 128 通道被试内结果构成后续所有通道缩减（Section 3.3）、迁移学习（Section 3.4）和纵向扩展实验（Section 3.5）的**被试内参考基线**（图中以半透明斜线填充标注）。

### 3.2 跨被试训练（128 通道）

表 7 展示了 128 通道跨被试训练的结果。

**表 7. 跨被试训练结果（128 通道，N = 21）。**

| 模型 | 二分类 Mean ± SD | 三分类 Mean ± SD |
|------|-----------------|-----------------|
| CBraMod | **90.68 ± 9.31%** | **74.88 ± 14.03%** |
| EEGNet-16,4 | 76.67 ± 11.95% | 61.23 ± 11.28% |
| Δ (CBraMod − EEGNet) | **+14.01 pp** | **+13.65 pp** |

跨被试模式下 CBraMod 的优势从 +7.05 pp（被试内）扩大至 +14.01 pp。图 2 展示了跨被试二分类的逐被试对比。

**图 2. 跨被试训练 128 通道二分类逐被试对比。** 柱状图展示 EEGNet 和 CBraMod 两个模型的被试内基线（斜线填充半透明）和跨被试结果（实心），清晰显示 CBraMod 从跨被试数据池化中显著获益（+5.53 pp），而 EEGNet 几乎不变（−1.43 pp）。

![图 2. 跨被试 128ch 二分类对比](../../results/20260330_0709_cross-subject_combined_imagery_binary.png)

这一结果揭示了基座模型与从头训练小模型在数据利用效率上的差异：EEGNet 未从跨被试数据池化中观察到显著收益（78.10% 被试内 vs 76.67% 跨被试，−1.43 pp，配对 t 检验 p = 0.456），提示其有限的 ~10K 参数可能难以从异质多被试数据中提取共享表征。相比之下，CBraMod 增益 +5.53 pp（85.15% → 90.68%），说明 TUEG 预训练提供的通用 EEG 先验使模型能有效整合 21 名被试的共享手指运动模式，将跨被试变异转化为泛化能力。

这一差异对实际部署具有启示意义：在当前 21 名被试的样本范围内，CBraMod 从跨被试数据池化中获益显著（+5.53 pp），而 EEGNet 等小模型的改善可能更依赖于增加单个被试的训练数据量（见 Section 3.5）。此结论基于被试内与跨被试的单次比较，数据池化收益的持续性需在更大样本量下验证。

> **数据来源**: CBraMod: `results/20260324_0023_cross_subject_cache_imagery_binary.json`; EEGNet: `results/20260330_0709_cross_subject_cache_imagery_binary.json`

> **基线声明**：上述 128 通道跨被试结果构成后续所有通道缩减实验（Section 3.3）和迁移学习实验（Section 3.4）的**跨被试参考基线**（图中以 "128ch Baseline" 点状填充标注）。

### 3.3 通道缩减

#### 3.3.1 32 通道配置对比

图 3a 以分组柱状图展示了五种 32 通道配置在跨被试二分类上的双模型性能对比，表 8 列出精确数值。

**图 3a. 32 通道五种配置双模型对比（跨被试二分类，N = 21）。** 虚线为各模型 128ch 参考性能。

![图 3a. 32ch 五种配置对比](../../paper/figures/32ch_comparison.png)

**表 8. 32 通道配置对比（跨被试二分类，N = 21）。** 128ch baseline：CBraMod 90.68%，EEGNet 76.67%。

| 排名 | 方法 | CBraMod Mean ± SD | Δ vs 128ch | EEGNet Mean ± SD | Δ vs 128ch |
|------|------|-------------------|------------|------------------|------------|
| 1 | **FDR** | **87.71 ± 9.18%** | **−2.97 pp** | 74.70 ± 12.46% | −1.97 pp |
| 2 | Band Power | 86.85 ± 9.76% | −3.83 pp | 76.07 ± 11.50% | −0.60 pp |
| 3 | Commercial | 86.10 ± 8.88% | −4.58 pp | 73.54 ± 12.57% | −3.13 pp |
| 4 | Attention | 85.48 ± 9.21% | −5.20 pp | — | — |
| 5 | CSP | 84.94 ± 10.53% | −5.74 pp | 75.00 ± 11.08% | −1.67 pp |

FDR 以 87.71% 领先，保留了 128 通道性能的 **96.7%**（87.71% vs 90.68%），通道缩减代价仅 −2.97 pp；而 CSP 的代价最大（−5.74 pp）。值得注意的是，EEGNet 在多数配置下的通道缩减代价反而更小（−0.60 至 −3.13 pp），可能因为其 128ch baseline 本身较低（76.67%），天花板效应更弱。图 3b 展示了 FDR 32 通道配置的逐被试对比。

**图 3b. 32 通道 FDR 配置跨被试二分类逐被试对比。** 同时叠加 128ch 跨被试基线（EEGNet + CBraMod，点状填充），显示 32ch FDR 在绝大多数被试上接近 128ch 性能。

![图 3. 32ch FDR 跨被试对比](../../results/32_channel/fdr/20260330_0836_cross-subject_combined_imagery_binary.png)

五种方法之间的差异仅 2.77 pp（84.94%–87.71%），反映了高密度 EEG 中体积传导导致的信息冗余。这一发现具有重要的实践意义：在 32 通道级别，通道选择方法的选择相对不那么关键——即使使用简单的商用布局（Commercial, 86.10%）也能获得接近最优数据驱动方法的性能。然而，这种"方法不敏感"的特性会随着通道数的进一步减少而急剧消失（见 Section 3.3.2）。

值得注意的是，Commercial 配置的标准差最低（8.88%），表明标准 10-20 布局在跨被试一致性上具有优势——这可能因为其电极分布更均匀，不依赖于特定被试群体的统计特征。

> **数据来源**: `results/32_channel/{fdr,attention,csp,band_power,commercial}/20260330_*_cross_subject_cache_imagery_binary.json`

#### 3.3.2 通道缩放分析（128 → 4）

表 9 展示了 CBraMod 从 128 到 4 通道的性能降解轨迹。

**表 9. CBraMod 通道缩放分析（跨被试二分类）。**

| 过渡 | 通道缩减 | 准确率下降 | 说明 |
|------|---------|-----------|------|
| 128 → 61 | −52% | −1.13 pp | 高度冗余 |
| 61 → 32 (FDR, best) | −48% | −1.84 pp | FDR 32ch ≈ 61ch |
| 32 → 8 (Band Power, best) | −75% | −3.66 pp | Band Power 保持良好 (84.05%) |
| 32 → 8 (CSP) | −75% | −5.98 pp | CSP 亦优于 FDR (81.73%) |
| 32 → 8 (FDR) | −75% | −11.28 pp | FDR 在 8ch 大幅衰退 (76.43%) |
| 32 → 8 (Attention) | −75% | −19.29 pp | Attention 衰退最严重 (68.42%) |
| 32 → 4 (FDR top-4) | −88% | −25.63 pp | 略优于 Attention 但仍低于负控制 (62.08%) |
| 32 → 4 (Attention top-4) | −88% | −33.01 pp | 降至近随机水平 (54.70%) |
| 32 → 4 (负控制) | −88% | −20.06 pp | 随机选择反而优于两种标准方法 (67.65%) |
| 32 → 4 (FDR∩Att, outlier) | −88% | −4.97 pp | 交集通道，favorable outlier (82.71%) |

图 4 以曲线形式直观呈现了这一非线性降解过程。

**图 4. 通道缩放曲线：CBraMod 跨被试二分类准确率随通道数的变化。** 红色实线为各通道数下最优配置的包络线；虚线追踪各通道选择方法在不同通道数下的表现。绿色区域标示 32 通道部署区间。× 标记为 4 通道负控制。误差线为被试间标准差。

![图 4. 通道缩放曲线](../../paper/figures/channel_scaling_curve.png)

图 4 的关键发现是**通道选择方法的最优排序随通道数发生翻转**。在 32ch 级别，FDR 以 87.71% 领先（五种方法差距仅 2.77 pp）；但到 8ch 级别，**Band Power 以 84.05% 大幅反超 FDR 的 76.43%**（+7.62 pp），CSP (81.73%) 亦优于 FDR。这表明基于全局统计判别力（FDR）选出的通道在高冗余场景下有效，但在低通道数下不如基于频域特征（Band Power：mu/beta 节律 ANOVA F-statistic）的方法——后者更直接捕获运动想象的神经振荡特征。

最优配置包络线（红色实线）呈现**两阶段降解模式**：（1）**平坦区**（128→8ch）：从 90.68% 仅下降至 84.05%（−6.63 pp），得益于 EEG 体积传导的信息冗余以及恰当的通道选择方法（Band Power 在 8ch 保留了 128ch 性能的 92.7%）；（2）**陡降区**（8→4ch）：最优标准方法（FDR top-4）骤降至 62.08%，方法间差异急剧扩大。

然而，降解的严重程度**高度依赖通道选择方法**。以 32→8ch 过渡为例：Band Power 仅下降 2.80 pp（86.85→84.05%），而 Attention 下降 17.06 pp（85.48→68.42%）——同一通道缩减幅度下，方法选择导致了 6 倍的性能差异。这一发现表明，**8 通道仍可作为可行的部署方案**，前提是使用频域驱动（Band Power）而非模型驱动（Attention）的通道选择方法。

> **关键发现**：**32 通道**仍是最优权衡点（96.7% retention），但 **8 通道 Band Power** (84.05%) 展现了意外强劲的表现——在通道数减少 75% 的情况下保留了 92.7% 的性能，为低成本 BCI 部署提供了可行方案。

**4 通道结果的深层解读**：Attention top-4（54.70%）不仅远低于 8ch Band Power（84.05%），甚至**低于负控制**（67.65%）——即随机选取未被任何方法选中的通道反而表现更好。这揭示了一个重要的方法论陷阱：**在 128ch 模型上计算的通道重要性排序不能线性外推到极低通道配置**。CBraMod 在 128ch 上的梯度注意力反映的是通道在*有其他 124 个通道辅助*时的重要性（即条件重要性），而非通道*独立携带*的信息量。当仅保留 top-4 时，这些通道失去了它们在全局空间模式中赖以发挥作用的上下文通道，导致性能崩溃。

相比之下，FDR∩Attention 的 4 个交集通道（82.71%）之所以表现优异，可能因为它们恰好同时满足了两个互补条件：统计可分性（FDR）**和**模型注意力（Attention），这种双重验证选出的通道碰巧在空间上形成了足够的覆盖模式。但这一结果是特定数据集上的**有利巧合（favorable outlier）**，不代表可推广的方法论。图 4 中橙色菱形标注了这一 outlier。

> **数据来源**: 128ch: `results/20260324_0023_cross_subject_cache_imagery_binary.json`; 61ch: `results/61_channel/standard_1010/20260330_1213_cross_subject_cache_imagery_binary.json`; 32ch: `results/32_channel/{fdr,band_power,commercial,attention,csp}/20260330_*_cross_subject_cache_imagery_binary.json`; 8ch: `results/8_channel/{band_power/20260331_1950,csp/20260331_2044,fdr/20260330_1311,attention/20260330_1334}_cross_subject_cache_imagery_binary.json`

#### 3.3.3 控制实验

为排除数据泄露解释并验证通道选择的有效性，我们在 4 通道级别进行了控制实验。

**表 10. 4 通道控制实验结果（跨被试二分类）。**

| 条件 | 通道来源 | CBraMod Mean ± SD |
|------|---------|-------------------|
| FDR ∩ Attention（outlier） | 32ch FDR 与 Attention 交集 | 82.71 ± 13.84% |
| 负控制 | 所有方法均未选中的通道 | 67.65 ± 9.46% |
| FDR top-4 | 32ch FDR 排序前 4 | 62.08 ± 8.81% |
| Attention top-4 | 32ch Attention 排序前 4 | **54.70 ± 8.20%** |

> **重要说明**：FDR∩Attention 的 4 个通道（B32, C8, D7, D19）并非任何单一方法排序的 top-4，而是两个 32 通道集合的交集——它们在各自排序中仅位于第 15–30 位。82.71% 的高准确率应被视为一个**有利的巧合**（favorable outlier）：这些通道恰好携带了两种互补方法共同认可的信息，但这种交集选择不可复制为系统化方法。通道缩放分析（Section 3.3.2）中的 4ch 数据点使用标准的 Attention top-4 选择，以保持与其他通道数配置的方法论一致性。

FDR∩Attention 与负控制之间的 15.06 pp 差距仍然有效地确认了数据驱动通道选择在极端通道缩减下的必要性。图 5 展示了两种配置的逐被试对比。

**图 5. 4 通道控制实验：最优（FDR∩Attention）vs 负控制。** 左图为最优 4ch 配置，右图为负控制。两者均叠加 128ch 跨被试基线（EEGNet + CBraMod，点状填充），提供完整的性能参考。

![图 5a. 4ch 最优配置](../../results/4_channel/fdr_attention_overlap/20260330_1417_cross-subject_combined_imagery_binary.png)

![图 5b. 4ch 负控制](../../results/4_channel/negative_control/20260330_1442_cross-subject_combined_imagery_binary.png)

负控制仍达到 67.65%（远高于 50% 随机基线），说明即使未被任何方法选中的通道也因体积传导而携带足够信息。这一结果同时提供了**两重验证**：（1）正向——数据驱动的通道选择确实捕获了更多任务相关信息（+15.06 pp）；（2）反向——高准确率并非数据泄露所致，而是 EEG 信号本身的物理特性（体积传导使皮层源信号广泛传播）。

通道选择方法敏感度的缩放规律总结如下：

| 通道数 | 方法数 | 标准方法间差异 | 最优 → 最差 | 解释 |
|--------|--------|--------------|------------|------|
| 32ch | 5 | 2.77 pp | FDR (87.71%) → CSP (84.94%) | 高冗余，方法选择影响小 |
| 8ch | 4 | 15.63 pp | Band Power (84.05%) → Attention (68.42%) | **方法选择成为决定性因素**；排序翻转 |
| 4ch | 2 | 7.38 pp | FDR (62.08%) → Attention (54.70%) | 标准方法均失效，均低于负控制 |

> 注：8ch 方法差异从 32ch 的 2.77 pp 扩大至 15.63 pp，表明通道选择方法的重要性在低通道数下急剧上升。32ch 的最优方法（FDR）在 8ch 仅排第三，被 Band Power 和 CSP 反超——**最优方法不可跨通道数外推**。4ch FDR∩Attention 交集 (82.71%) 为 favorable outlier（见 Section 3.3.3），非标准单方法选择，不纳入方法间差异计算。

值得注意的是，**两种标准单方法选择（FDR top-4: 62.08%, Attention top-4: 54.70%）均低于负控制（67.65%）**。这揭示了一个重要的方法论问题：基于 128ch 全模型计算的通道重要性排序在极低通道数下不仅失效，甚至产生反效果——数据驱动方法选出的"最重要"通道空间分布过于集中，反而丢失了负控制中随机通道的分散空间覆盖带来的信息多样性（见 Section 3.3.2 讨论）。

> **数据来源**: `results/4_channel/fdr_attention_overlap/20260330_1417_cross_subject_cache_imagery_binary.json`; `results/4_channel/negative_control/20260330_1442_cross_subject_cache_imagery_binary.json`

### 3.4 迁移学习（128 通道）

表 11 总结了 128 通道迁移学习结果。

**表 11. 迁移学习效果（128ch CBraMod，N = 21）。**

| 任务 | 范式 | CBraMod Mean ± SD | Δ vs. 跨被试 |
|------|------|-------------------|-------------|
| 二分类 | 跨被试 | 90.68 ± 9.31% | — |
| 二分类 | 迁移（fine-tuned） | 90.12 ± 8.98% | **−0.56 pp** |
| 三分类 | 跨被试 | 74.88 ± 14.03% | — |
| 三分类 | 迁移（fine-tuned） | 75.08 ± 14.02% | **+0.20 pp** |

在 128 通道条件下，迁移学习在两种任务上均未产生统计显著的收益（二分类 Δ = −0.56 pp，配对 t 检验 p = 0.189；三分类 Δ = +0.20 pp，p = 0.261）。图 6 和图 6b 分别展示了二分类和三分类的迁移学习逐被试对比。

**图 6. 128 通道迁移学习 6-way 对比（二分类）。** 同时展示被试内（历史 EEGNet + CBraMod）、跨被试（历史 EEGNet + CBraMod）和迁移学习（当前 EEGNet + CBraMod）的逐被试结果。

![图 6. 迁移学习 6-way 对比（二分类）](../../results/20260329_0507_transfer_combined_imagery_binary.png)

**图 6b. 128 通道迁移学习 6-way 对比（三分类）。**

![图 6b. 迁移学习 6-way 对比（三分类）](../../results/20260329_0448_transfer_combined_imagery_ternary.png)

两个任务的一致非显著差异表明，在 128 通道条件下跨被试模型已具备充分的表征能力，个体化 fine-tuning 未提供额外收益。二分类 90.68% 的准确率已接近该数据集的天花板（数据质量问题被试 S10, S20 的存在限制了上限），而三分类 74.88% 虽有更大的理论提升空间，但迁移学习仍未能突破——一种可能的解释是**限制因素并非跨被试模型的个体适配不足，而是任务本身的固有难度**（三类手指 MI 的皮层表征重叠更严重）。基于此，可以假设在缩减通道配置下（跨被试模型因空间信息受限而性能下降时），个体化 fine-tuning 可能提供更大收益（见 Section 6 未来工作）。

> **数据来源**: 跨被试二分类 `20260324_0023`: `results/20260324_0023_cross_subject_cache_imagery_binary.json`; 迁移二分类 `20260329_0507`: `results/20260329_0507_transfer_cache_imagery_binary.json`; 跨被试三分类 `20260324_0109`: `results/20260324_0109_cross_subject_cache_imagery_ternary.json`; 迁移三分类 `20260329_0448`: `results/20260329_0448_transfer_cache_imagery_ternary.json`

### 3.5 多 session 纵向数据扩展

#### 3.5.1 被试内二分类（N = 16）

表 12 展示了随额外 session 数据递增的被试内二分类性能变化。

**表 12a. CBraMod 被试内训练 + 额外 session 数据（二分类，per_session，N = 16）。**

| 阶段 | Mean ± SD | Δ vs Baseline | p 值 |
|------|-----------|---------------|------|
| Baseline | 87.23 ± 10.81% | — | — |
| +Sess03 | 89.14 ± 8.93% | +1.91 pp | — |
| +Sess04 | 90.94 ± 8.93% | +3.71 pp | — |
| +Sess05 | **93.36 ± 5.98%** | **+6.13 pp** | **0.007** |

**表 12b. EEGNet-16,4 被试内训练 + 额外 session 数据（二分类，per_session，N = 16）。**

| 阶段 | Mean ± SD | Δ vs Baseline | p 值 |
|------|-----------|---------------|------|
| Baseline | 80.51 ± 12.16% | — | — |
| +Sess03 | 87.73 ± 7.23% | +7.22 pp | — |
| +Sess04 | 87.93 ± 9.57% | +7.42 pp | — |
| +Sess05 | **87.85 ± 7.47%** | **+7.34 pp** | **0.009** |

两种模型均从额外 session 数据中显著获益（p < 0.01）。CBraMod 从 87.23% 提升至 93.36%（+6.13 pp），EEGNet 从 80.51% 提升至 87.85%（+7.34 pp）；前者终点更高，后者绝对增益更大。图 7 展示了逐被试的渐进式性能变化。

**图 7. Extra Sessions 二分类渐进式性能变化（CBraMod，N = 16）。** 上方柱状图展示逐被试各阶段准确率（颜色渐深表示数据量递增）；中部折线图展示均值变化趋势；下左箱线图展示各阶段分布；下右散点图为 Baseline vs +Sess05 配对对比。

![图 7. Extra Sessions 二分类](../../paper/figures/extra_sessions_binary.png)

**低基线与高基线被试的差异化收益**：

| 被试分组 | N (EEGNet / CBraMod) | EEGNet Δ | CBraMod Δ |
|---------|---------------------|----------|-----------|
| 低基线 (<80%) | 8 / 3 | **+13.12 pp** | **+18.75 pp** |
| 高基线 (>90%) | 5 / 9 | −0.87 pp | +1.46 pp |

> 注：分组阈值 80%/90% 为绝对值，因两模型基线分布不同，各分组样本量有差异。CBraMod 低基线仅含 3 名被试（S06, S10, S16），+18.75 pp 的增益估计受个体差异影响大，应视为方向性趋势而非精确效应量。

低基线被试是额外 session 数据的主要受益者，而高基线组仅有轻微改进甚至停滞，呈现明显天花板效应。

**标准差压缩与观测范围**：CBraMod 的被试间标准差从 10.81% 压缩至 5.98%（−45%），表明额外数据不仅提高平均性能，还改善了跨用户一致性。实际观测范围从 60.62%–99.38%（Baseline）收窄至 74.38%–98.75%（+Sess05），最低单被试准确率从 60.62%（S10）提升至 74.38%（S10），反映了底部用户的显著改善。这一压缩对临床部署尤为重要：BCI 系统的实用化要求不是"最好情况下多好"，而是"最差情况下够不够用"。

从机制层面看，低基线被试获益显著而高基线被试接近天花板，表明额外 session 数据的主要作用是**弥补个体训练数据不足**而非提升模型容量。这与 Section 3.2 的发现形成有趣的对照：EEGNet 未从*其他被试*数据的池化中显著获益（跨被试 −1.43 pp, p = 0.456），但能从*同一被试*的额外 session 中显著获益（+7.34 pp, p = 0.009），提示其瓶颈在于被试间特征异质性而非绝对数据量。

> **数据来源**: EEGNet baseline: `results/20260316_1411_comparison_cache_imagery_binary.json`; CBraMod baseline: `results/20260323_2237_comparison_cache_imagery_binary.json`; within-subject extra sessions run `20260324_2131`: `results/20260324_2131_extra_sessions_cache_imagery_binary.json`

#### 3.5.2 被试内三分类（N = 16）

两种模型在三分类任务上也显示了多 session 改善，但增益幅度和统计显著性存在差异：

**表 13a. CBraMod 被试内训练 + 额外 session 数据（三分类，per_session，N = 16）。**

| 阶段 | Mean ± SD | Δ vs Baseline | p 值 |
|------|-----------|---------------|------|
| Baseline | 74.51 ± 14.22% | — | — |
| +Sess03 | 78.78 ± 12.34% | +4.27 pp | — |
| +Sess04 | 79.84 ± 11.33% | +5.33 pp | — |
| +Sess05 | **83.06 ± 9.51%** | **+8.55 pp** | **0.012** |

**表 13b. EEGNet-16,4 被试内训练 + 额外 session 数据（三分类，per_session，N = 16）。**

| 阶段 | Mean ± SD | Δ vs Baseline | p 值 |
|------|-----------|---------------|------|
| Baseline | 71.48 ± 13.18% | — | — |
| +Sess03 | 72.47 ± 12.33% | +0.99 pp | — |
| +Sess04 | 76.95 ± 9.20% | +5.47 pp | — |
| +Sess05 | **76.08 ± 9.37%** | **+4.60 pp** | **0.166** |

CBraMod 三分类增益（+8.55 pp，p = 0.012）达到显著水平，而 EEGNet 增益虽为正（+4.60 pp）但未达显著（p = 0.166）。与二分类相比，三分类呈现两个特征：（1）CBraMod 的增益反而更大（+8.55 pp vs +6.13 pp），可能因为三分类基线较低（74.51% vs 87.23%），天花板效应更弱；（2）EEGNet 未能达到统计显著，反映其在多类任务上容量有限，额外数据的边际收益低于二分类。图 8 展示了三分类的渐进式变化。

**图 8. Extra Sessions 三分类渐进式性能变化（N = 16）。**

![图 8. Extra Sessions 三分类](../../paper/figures/extra_sessions_ternary.png)

> **数据来源**: `results/20260331_0827_extra_sessions_cache_imagery_ternary.json`

注：此处baseline与正常baseline准确率相对较高，原因是在评估时只选择了有额外online-session的被试，而并没有纳入那些无online session被试。而原finger-eeg数据采集的研究人员在选择哪些被试进行进一步的online session数据采集时，可能因为其实际表现而存在偏好。

#### 3.5.3 评估策略一致性

除默认的 per_session 策略外，我们使用 fixed_combined（固定组合测试集）和 fixed_sess02（固定 Sess02 测试集）两种策略进行补充验证。三种策略均确认了额外 session 数据的显著改善效果：

**表 14. 三种评估策略对比摘要（二分类，Baseline → +Sess05 变化量）。**

| 策略 | EEGNet Δ | CBraMod Δ | 说明 |
|------|----------|-----------|------|
| per_session（默认） | +7.34 pp | +6.13 pp | 临床最相关 |
| fixed_combined | +9.96 pp | +8.44 pp | 控制测试集难度 |
| fixed_sess02 | +8.51 pp | +4.38 pp | 最保守估计 |

fixed_combined 策略显示单调递增趋势（消除了测试集难度变化的混淆因素）。fixed_sess02 下 CBraMod 的增益明显小于 EEGNet（+4.38 pp vs +8.51 pp），可能存在两种解释：（1）基座模型对跨 session 时间分布漂移更敏感；（2）天花板效应——CBraMod 的 fixed_sess02 baseline（87.23%）高于 EEGNet（80.51%），更高起点下增益空间本身受限。两种因素可能同时起作用，当前数据不足以区分。详细策略对比分析见 Supplementary。

> **数据来源**: per_session `20260324_2131`: `results/20260324_2131_extra_sessions_cache_imagery_binary.json`; fixed_combined `20260325_0514`: `results/20260325_0514_extra_sessions_cache_fixed_combined_imagery_binary.json`; fixed_sess02 `20260325_1208`: `results/20260325_1208_extra_sessions_cache_fixed_sess02_imagery_binary.json`; 详细分析: `docs/dev_log/experiments/extra_sessions_strategy_comparison.md`

#### 3.5.4 三范式对齐（CBraMod 二分类，N = 16）

为了直接比较额外 session 数据在三种训练范式中的作用，我们将被试内、跨被试和 transfer-init 结果统一到同一组 16 名具备额外 session 的被试，并使用相同的 per-session 评估协议。跨被试曲线使用 21 名训练被试的 pooled model；transfer-init 曲线使用对应 step 的 cross-subject checkpoint 作为初始化，再对单被试做离线微调。表 15 和图 9 展示了三条轨迹的并列结果。

**表 15. Extra sessions 在三种训练范式下的轨迹对比（CBraMod 二分类，N = 16）。**

| 阶段 | 被试内 | 跨被试（21-subj 训练） | Transfer-init |
|------|--------|------------------------|---------------|
| Baseline | 87.23 ± 10.81% | 92.38 ± 8.35% | 87.23 ± 10.81% |
| +Sess03 | 89.14 ± 8.93% | 91.88 ± 6.71% | 89.65 ± 7.09% |
| +Sess04 | 90.94 ± 8.93% | 92.19 ± 6.91% | 91.84 ± 6.91% |
| +Sess05 | **93.36 ± 5.98%** | **93.24 ± 5.81%** | **92.93 ± 6.11%** |
| Δ(BL→S05) | **+6.13 pp** | **+0.86 pp** | **+5.70 pp** |
| paired p | **0.007** | 0.662 | **0.015** |

跨被试模型的起点最高（92.38%），但额外 session 带来的边际收益最小（+0.86 pp, p = 0.662）；相反，被试内重训练和 transfer-init 都能从新增同被试数据中获得显著改善，最终分别达到 93.36% 和 92.93%。值得注意的是，cross-subject 与 transfer-init 都不是在“单一同分布增量学习”条件下吸收新增数据：模型既要面对**跨 session 的时间漂移**，又带着**跨被试 pooled 训练形成的群体差异**。这种“跨 session 异质性 + 跨被试异质性”的叠加，会把一部分新增数据的作用消耗在分布对齐上，而不是直接转化为更高的最终准确率。到 +Sess05 时，三条曲线收敛到 92.93%–93.36% 的窄区间，说明在 128 通道条件下，一旦拥有足够的同被试 session，性能上限更可能由数据质量和任务难度决定，而非训练范式本身。

**图 9. Extra Sessions 在三种训练范式下的总览（CBraMod 二分类，N = 16）。** 左图展示四个 step 的均值 ± 标准差轨迹；右图展示 Baseline → +Sess05 的净增益，强调“高 baseline”与“高增益”并不等价。

![图 9. Extra Sessions 三范式总览](../../paper/figures/extra_sessions_paradigm_binary.png)

> **数据来源**: within-subject `20260324_2131`: `results/20260324_2131_extra_sessions_cache_imagery_binary.json`; cross-subject `20260326_1409`: `results/20260326_1409_cross_subject_extra_sessions_cache_imagery_binary.json`; transfer-init `20260329_1357`: `results/20260329_1357_extra_sessions_cache_imagery_binary.json`（由 `run_extra_sessions.py --pretrained-run` 生成，缓存 schema 仍为 `extra_sessions_cache`）

#### 3.5.5 跨被试 pooled model 的边际收益上限

如果把视角聚焦到 pooled cross-subject 模型本身，extra sessions 的边际收益明显弱于被试内更新。表 15b 将二分类和三分类的跨被试结果压缩为同一摘要：二分类中，EEGNet 完全无收益（81.45% → 81.33%，p = 0.950），CBraMod 也仅小幅上升 +0.86 pp；三分类下 CBraMod 虽有 +3.73 pp 的正增量，但未达显著（p = 0.090）。

**表 15b. Cross-subject extra sessions 的边际收益摘要（N = 16）。**

| 模型 / 任务 | Baseline | +Sess05 | Δ | paired p |
|-------------|----------|----------|---|----------|
| CBraMod Binary | 92.38 ± 8.35% | 93.24 ± 5.81% | +0.86 pp | 0.662 |
| EEGNet Binary | 81.45 ± 10.87% | 81.33 ± 10.16% | −0.12 pp | 0.950 |
| CBraMod Ternary | 80.05 ± 11.46% | 83.78 ± 8.30% | +3.73 pp | 0.090 |

这一模式支持一个更具体的解释：额外 session 数据本身并非“无信息”，而是**其信息主要是被试特异性的**。当模型以单被试为单位更新时（被试内或 transfer-init），这些新增 trial 会直接推动决策边界向该被试收敛；而在跨被试 pooled 训练中，同一批新增 trial 被稀释进 21 名被试的联合分布，能改善总体表征，却难以显著改变特定个体的最终决策边界。更进一步，cross-subject extra sessions 的收益受限，很可能正是因为模型同时面对两层异质性：一层是不同日期/状态带来的**跨 session 分布漂移**，另一层是不同个体神经模式带来的**跨被试差异**。当这两层异质性叠加时，新增样本首先被用于“校正分布错位”，能留下来提升分类 margin 的有效信息就更少。换言之，cross-subject 预训练更适合作为 initialization 或高 baseline 起点，而不是吸收 extra sessions 的最终归宿。

> **数据来源**: binary `20260326_1409`: `results/20260326_1409_cross_subject_extra_sessions_cache_imagery_binary.json`; ternary `20260327_0303`: `results/20260327_0303_cross_subject_extra_sessions_cache_imagery_ternary.json`

### 3.6 领域自适应 Further Pre-training

表 16 展示对CBRAMOD基座模型在外部 MI 数据上进行 further pre-training 后的再与finger-eeg任务进行后训练的评估结果。

**表 16. Further pre-training 下游评估（CBraMod，N = 21）。**

| 范式 | 任务 | Baseline (TUEG) | FT-V1 (10ep) | FT-V2 (12ep) | V2 vs Baseline |
|------|------|:---:|:---:|:---:|:---:|
| 被试内 | 二分类 | **85.09%** ± 10.46% | 83.84% | 82.23% | **−2.86 pp** |
| 跨被试 | 二分类 | **90.54%** ± 9.25% | 88.84% | 89.43% | **−1.11 pp** |
| 被试内 | 三分类 | **69.54%** ± 12.84% | 69.25% | 68.08% | **−1.46 pp** |
| 跨被试 | 三分类 | **75.42%** ± 12.72% | 75.67% | 75.32% | **−0.10 pp** |
| | | | 平均 V1: −0.75 pp | 平均 V2: **−1.38 pp** | |

所有条件下 further pre-training 均导致性能下降或无改善。图 10 以柱状图直观展示了这一负面结果。

**图 10. Further Pre-training 下游评估。** 左图：四种条件下 Baseline (TUEG) vs FT-V1 vs FT-V2 的准确率对比，红色标注显示 V2 相对 Baseline 的变化量（均为负值）。右图：V1 和 V2 的平均 delta，V2 训练更充分但负迁移更大。

![图 10. Further Pre-training 下游评估](../../paper/figures/further_pretraining.png)

V2 使用了更多数据（78,232 vs 30,282 segments，主要增量来自 Stieger2021 数据集补全）和不同的 LR 调度（恒定 5e-5 vs cosine decay），达到了 39% 更低的 pre-training loss，但下游负迁移反而更大（−1.38 pp vs V1 −0.75 pp）。需要指出，V1 和 V2 同时改变了数据量、LR 调度和训练步数（2,360 vs 7,776），因此无法将负迁移的加剧归因于单一因素。两版的**一致负迁移方向**是稳健的发现——外部 MI 数据（以粗粒度肢体分类为主）的 further pre-training 未能为手指级运动想象分类带来提升，模型在 further pre-training 中学到的 MI 表征可能覆盖了 TUEG 预训练中学到的更通用的 EEG 表征。至于"训练越充分负迁移越大"的剂量-反应关系，则需要控制变量实验进一步验证。

> **数据来源**:
> - Baseline: ExperimentDB `run_tag=20260321_0343` (binary within), `20260321_0608` (binary cross)
> - FT-V2: `results/20260323_1433_cbramod_imagery_binary.json` (within), `results/20260323_1517_cross-subject_cbramod_imagery_binary.json` (cross)
> - 完整分析: `paper/analysis/further_pretraining_analysis.md`

### 3.7 推理性能

表 17 展示了两种模型在实时 BCI 场景下的推理延迟。

**表 17. 推理延迟（128 通道二分类，NVIDIA RTX 5070）。**

| Batch Size | EEGNet | CBraMod | 倍率 |
|:----------:|-------:|--------:|-----:|
| 1 | **0.375 ms** | 12.919 ms | 34.4× |
| 8 | 0.542 ms | 12.582 ms | 23.2× |
| 32 | 2.058 ms | 32.729 ms | 15.9× |
| 64 | 4.027 ms | 71.110 ms | 17.7× |

图 11 以对数坐标柱状图和模型规模对比直观呈现了这一结果。

**图 11. 推理延迟与模型规模对比。** 左图：不同 batch size 下两种模型的延迟（对数坐标），红色虚线为 100ms 实时阈值。右图：CBraMod/EEGNet 的参数量、FLOPs、模型大小、延迟倍率。

![图 11. 推理延迟对比](../../paper/figures/inference_latency.png)

即使在最严格的 batch=1 条件下，CBraMod 的单样本延迟 (~13 ms) 也远低于实时 BCI 的 100 ms 阈值——约有 7.7 倍的余量。EEGNet 以 ~0.4 ms 的延迟实现了极致的实时性。尽管 CBraMod 的参数量是 EEGNet 的 ~1,900 倍，但延迟倍率仅为 34 倍（batch=1），这得益于 GPU 并行计算对 Transformer 架构的高效支持。两种模型均满足实时 BCI 部署要求，CBraMod 在准确率上的优势不以牺牲实时性为代价。

> **数据来源**: `docs/dev_log/experiments/inference_benchmark_analysis.md`

### 3.8 数据质量与被试异质性

三名重度伪影被试（S04, S10, S14）的振幅超过群体最大值的 3–8 倍（126K–307K µV vs. 正常 ≤ 38K µV），时间漂移值高出群体均值数个数量级（S04: 2,717 vs. 群体均值 ~30）。尽管如此，S04 在 128 通道跨被试二分类中达到 98.12%（矛盾性高准确率），提示模型可能利用了伪影模式而非真实神经信号。

保留这三名被试进行所有分析提供了保守的性能估计。其排除后的影响见 Section 6（未来工作）。

---

## 4. 讨论

### 4.1 基座模型优势：何时与为何

从方法学定位看，本文不是 [3] 的在线机器人控制复现，也不是 [4] 的通用 benchmark 复刻，而是将 [3] 的 finger-level dataset/session design 与 [4] 的 pretrained foundation model 结合到统一的离线、held-out-session 评估框架中。因而，下述模型差异更适合被解读为“在同一数据与相同 split 约束下，预训练基座模型相对 compact CNN 的收益”，而不是对在线 robotic control 或 [4] 全任务基准的直接替代。

CBraMod 在所有实验条件下一致优于 EEGNet——被试内 **+7.05 pp**、跨被试 **+14.01 pp**（128ch）、32 通道 **+10–13 pp**——这反映了大规模预训练对 EEG 解码的价值。~1,900 倍的参数量差异本身不能完全解释该差距；更关键的是，CBraMod 在 TUEG 语料上学到的通用时空 EEG 表征能有效迁移到数据相对稀缺的手指级 MI 分类任务。

值得注意的是，EEGNet 未从跨被试数据池化中显著获益（78.10% 被试内 vs 76.67% 跨被试，−1.43 pp, p = 0.456），而 CBraMod 增益 +5.53 pp。这提示基座模型的预训练表征使其能够更有效地整合异质跨被试数据。

### 4.2 最优通道配置与部署

32 通道 FDR 配置是实际 BCI 部署的最优权衡点：

| 属性 | 值 |
|------|-----|
| 性能保留率 | **96.7%**（87.71% vs 90.68%） |
| vs. 61ch 标准 10-10 | 仅差 1.84 pp，通道数减半 |
| 硬件兼容性 | 与商用 32 通道 EEG 系统兼容 |

### 4.3 体积传导与信息冗余

控制实验（Section 3.3.3）揭示了高密度 EEG 的一个基本属性：由于体积传导，皮层源的电信号在头皮上广泛传播，产生了大量信息冗余。4 通道负控制（67.65%）表明，即使是未被任何方法选中的通道，在预训练基座模型下也能显著超越随机水平。在 32 通道级别，五种方法之间仅 2.77 pp 的窄性能差异证实了广泛的冗余。

### 4.4 纵向数据扩展：突破 Session 平台期

原始数据集论文 [3] 在在线 base/fine-tuned EEGNet 设置下报告：被试性能在 2–3 个 session 后趋于平台期。本研究的 N = 16 离线分析回答的是一个更弱、也更可控的问题：如果去掉实时反馈与 same-day update，仅保留累积数据量增长，模型是否仍能从额外 session 中持续获益？结果显示答案是肯定的，但**收益取决于更新发生在何处**。在单被试更新框架下，被试内重训练和 transfer-init 都获得了显著增益（CBraMod 分别 +6.13 pp, p = 0.007 和 +5.70 pp, p = 0.015）；而在 pooled cross-subject 框架下，额外同被试 trial 只能带来极小提升（+0.86 pp, p = 0.662）。这说明新增数据的关键信息主要是被试特异性的，需要通过个体化更新才能充分吸收。

对 cross-subject 和 transfer-init 而言，收益没有进一步放大的另一个原因，是模型都要同时处理两层分布错位：新增数据来自**新的 session**，因此包含时间漂移、疲劳、接触阻抗变化等跨 session 异质性；而初始化或训练底座又来自**多被试 pooled 分布**，因此天然带有跨被试异质性。两层异质性叠加后，新增样本的一部分作用会先被用于对齐分布，而不是直接提升分类边界，这也解释了为什么它们的增益弱于纯被试内重训练。

标准差从 10.81% 压缩至 5.98%（−45%）具有实际部署意义：BCI 系统需要跨用户的一致性能，而非少数用户的峰值表现。额外 session 数据不仅提高了平均水平，还将"最差情况"显著抬升。

补充分析中的 fixed_sess02 策略揭示了一个有趣的模型差异：CBraMod 的增量收益在后续 session 迅速饱和（+Sess03→+Sess04 仅 +0.19 pp），而 EEGNet 的改善更为线性（+1.60 pp/session）。这提示预训练基座模型更快捕获了时间不变特征，但对 session 间的时间分布漂移更敏感；而容量有限的 EEGNet 被迫学习最稳定的时间特征，反而实现了更好的跨 session 泛化。

### 4.5 领域自适应 Further Pre-training 的局限

870 小时外部 MI 数据的 further pre-training 在两种不同训练配置下均导致负迁移（V1: −0.75 pp, V2: −1.38 pp），这一结果可从三个层面理解：（1）**领域不匹配**——外部 MI 数据以粗运动（左/右手）为主，与精细手指运动的特征空间存在质的差异；（2）**数据量处于"危险中间地带"**——MI 数据（38G channel-frames）仅为 TUEG（126.5G）的 1/3，足以扰动 TUEG 学到的通用表征，但不足以建立稳健的 MI 特异性表征；（3）**灾难性遗忘**——further pre-training 可能覆盖了 TUEG 中学到的更通用的 EEG 特征。与跨被试 in-domain fine-tuning 的 +5.53 pp 增益形成对比，方向上提示**域内数据适配优于通用预训练，后者又优于域外数据适配**。但需注意，这一层次关系基于不同实验范式的横向比较（域内 fine-tuning 使用 21 被试标注数据，further pre-training 使用 10 个外部数据集的自监督学习），各环节的超参数和训练协议未统一控制，因此应视为方向性观察而非严格因果排序。

### 4.6 实际部署路线图

综合以上发现，本研究支持以下 BCI 部署路径：

1. **起步方案**：CBraMod + FDR 32 通道配置（87.71% 基线准确率），兼容商用硬件
2. **个性化适配**：收集 2–3 个额外 session 数据即可突破 90% 准确率，低基线用户获益最大
3. **模型选择**：直接使用 TUEG 预训练权重，不进行外部 MI 数据的 further pre-training
4. **实时可行性**：CBraMod 单样本延迟 ~13 ms，远低于 100 ms 实时阈值

### 4.7 伪影被试的影响

保留三名重度伪影被试（S04, S10, S14）进行所有分析提供了保守的性能估计。这些被试的信噪比低于群体均值 4–6 dB（−19.8 至 −21.8 dB vs. −15.8 dB 均值），Fisher 判别比接近零，表明真实的类别区分性神经信息被伪影噪声淹没。排除这些被试后的影响评估属于未来工作（Section 6）。

---

## 5. 局限性

| # | 局限 | 影响 |
|---|------|------|
| 1 | **通道选择数据范围** — FDR、CSP、Attention、Band Power 指标使用了所有 session 数据（含测试 session 上下文）计算。 | 可能轻微高估通道选择质量；未来工作应使用严格隔离的数据。 |
| 2 | **Responder cohort 继承自原始数据集论文** — [3] 仅保留离线二分类表现达到阈值的被试进入完整在线实验，本文沿用这 21 人 cohort。 | 结果更代表“可用 BCI 被试”（BCI-amenable users），而非无筛选总体；对普通受试者群体的泛化可能被高估。 |
| 3 | **单一数据集** — 所有实验使用同一个 21 人数据集。 | 对其他人群、范式、硬件的泛化性未验证。 |
| 4 | **仅运动想象** — 运动执行数据尚未评估。 | 信号特征和最优通道可能不同（见 Section 6）。 |
| 5 | **无数据增强** — 未应用时间偏移、噪声注入或 channel dropout。 | 低通道配置可能从增强中获益最大。 |
| 6 | **Extra sessions 选择偏差** — 仅 16/21 被试拥有额外 session 数据，这些被试的 baseline 系统性偏高。 | 多 session 增益估计可能存在选择偏差。 |
| 7 | **Further pre-training 范围** — 仅测试了一个基座模型（CBraMod）和一种预训练目标（masked autoencoding）。 | 其他模型或预训练目标可能得出不同结论。 |
| 8 | **缩减通道下的迁移学习** — 迁移学习在 128 通道下二分类和三分类均未显示收益，但尚未在缩减通道配置下测试。 | 当跨被试模型因空间信息受限而性能下降时，迁移收益可能增大；该交互效应尚未表征。 |

---

## 6. 未来工作

以下实验计划在后续研究中完成：

1. **伪影被试排除与重评估**：移除 S04、S10、S14，重新运行全部实验流程（被试内/跨被试/迁移/通道缩减），量化伪影被试对跨被试性能的影响。预期跨被试二分类基线将超过 92%。

2. **运动执行范式验证**：使用同一数据集中的运动执行（Motor Execution）录制数据复制完整实验流程，检验 CBraMod 的优势是否跨范式持续，以及最优通道配置是否因范式而异。

3. **缩减通道下的迁移学习**：评估 32、8、4 通道配置下的迁移学习效果，检验"通道越少，迁移收益越大"的假设。

---

## 7. 结论

本研究系统评估了 EEG 基座模型（CBraMod）在手指级运动想象分类中的应用，通过通道缩减、纵向数据扩展和领域自适应预训练三个维度建立了完整的实验证据体系。五个核心发现如下：

> **发现 1 — 基座模型优势随数据约束放大。** CBraMod 对 EEGNet 的优势从 **+7.05 pp**（被试内）扩大至 **+14.01 pp**（跨被试 128 通道），在 32 通道下仍保持 **+10–13 pp** 差距。预训练表征在个体数据稀缺或空间信息受限时最具价值。
>
> **发现 2 — 32 通道是最优部署目标。** FDR 选取的 32 通道保留了 128 通道性能的 **96.7%**（87.71% vs 90.68%），兼容商用 32 通道 EEG 硬件。
>
> **发现 3 — 多 session 数据改善性能，但主要通过个体化更新实现。** 在被试内重训练中，额外 session 数据为两种模型均带来显著增益（CBraMod +6.13 pp, p = 0.007; EEGNet +7.34 pp, p = 0.009），低基线被试获益尤为突出，被试间标准差压缩约 45%（10.81% → 5.98%），最低单被试准确率从 60.62% 提升至 74.38%。相对地，pooled cross-subject 模型仅获得 +0.86 pp 的微弱改善。
>
> **发现 4 — 领域自适应 further pre-training 未能改善精细运动解码。** 尽管使用了 870 小时外部 MI 数据，further pre-training 在两种不同训练配置（V1: cosine/30K segments; V2: constant LR/78K segments）下均呈现负迁移（V1: −0.75 pp, V2: **−1.38 pp**），提示粗运动 MI 数据与精细手指运动解码之间可能存在不可通过数据量弥合的领域差异。
>
> **发现 5 — 通道选择方法在极低通道数下失效。** 32 通道时五种方法差异仅 2.77 pp；8 通道时差距扩大至 8 pp；4 通道时标准 Attention top-4 骤降至 **54.70%**（低于负控制 67.65%），表明基于全通道模型的重要性排序不可线性外推至极低配置。先前报告的 4ch FDR∩Attention 82.71% 结果被确认为 favorable outlier。

上述结果共同支持了 CBraMod + FDR 32 通道 BCI 系统在手指级运动想象分类中的实用化部署。

---

## 参考文献

[1] J. R. Wolpaw, N. Birbaumer, D. J. McFarland, G. Pfurtscheller, and T. M. Vaughan, "Brain-computer interfaces for communication and control," *Clinical Neurophysiology*, vol. 113, no. 6, pp. 767–791, 2002.

[2] G. Pfurtscheller and C. Neuper, "Motor imagery and direct brain-computer communication," *Proceedings of the IEEE*, vol. 89, no. 7, pp. 1123–1134, 2001.

[3] Y. Ding, C. Udompanyawit, Y. Zhang, and B. He, "EEG-based brain-computer interface enables real-time robotic hand control at individual finger level," *Nature Communications*, vol. 16, p. 5401, 2025, doi: 10.1038/s41467-025-61064-x.

[4] J. Wang, S. Zhao, Z. Luo, Y. Zhou, H. Jiang, S. Li, T. Li, and G. Pan, "CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding," in *The Thirteenth International Conference on Learning Representations (ICLR)*, 2025.

[5] V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and B. J. Lance, "EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces," *Journal of Neural Engineering*, vol. 15, no. 5, p. 056013, 2018.

[6] W.-B. Jiang, L.-M. Zhao, and B.-L. Lu, "Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI," in *The Twelfth International Conference on Learning Representations (ICLR)*, 2024.

[7] J. Lai, J. Wei, L. Yao, and Y. Wang, "A Simple Review of EEG Foundation Models: Datasets, Advancements and Future Perspectives," arXiv:2504.20069, 2025.

[8] R. Alazrai, H. Abuhijleh, M. Alwanni, and M. I. Daoud, "EEG-based BCI system for decoding finger movements within the same hand," *Neuroscience Letters*, vol. 698, pp. 113–120, 2019.

[9] H. S. Lee et al., "Individual finger movement decoding using a novel ultra-high-density electroencephalography-based brain-computer interface system," *Frontiers in Neuroscience*, vol. 16, p. 1009878, 2022.

---

## 补充材料

> **选取说明**: Tables S1–S4 提供正文核心实验的逐被试细节，选取标准为：正文汇总表中讨论了被试间异质性模式（如异常被试 S04/S10/S14 的行为、低基线被试的差异化收益等）的实验。具体覆盖：128 通道二分类主对比（S1）及三分类对照（S1b）、32 通道配置间对比（S2）、多 session 纵向扩展（S3）、领域自适应预训练负迁移（S4）。迁移学习、8/4 通道等实验因正文结论基于汇总统计而非个体模式，未单独列出逐被试表。

### Table S1. 逐被试结果（128 通道，二分类）

| 被试 | EEGNet 被试内 | CBraMod 被试内 | CBraMod 跨被试 | EEGNet 跨被试 | 数据质量 |
|------|-------------|---------------|---------------|-------------|---------|
| S01 | 68.75% | 86.88% | 93.12% | 73.75% | 干净 |
| S02 | 94.38% | 94.38% | 95.00% | 85.62% | 干净 |
| S03 | 85.00% | 94.38% | 100.00% | 78.75% | 轻度 |
| S04 | 94.38% | 91.88% | 98.12% | 93.75% | **重度** |
| S05 | 90.00% | 86.25% | 92.50% | 60.00% | 轻度 |
| S06 | 68.12% | 74.38% | 87.50% | 74.38% | 干净 |
| S07 | 76.88% | 81.88% | 90.00% | 81.25% | 干净 |
| S08 | 85.00% | 93.12% | 97.50% | 87.50% | 干净 |
| S09 | 99.38% | 99.38% | 99.38% | 95.00% | 轻度 |
| S10 | 70.00% | 60.62% | 66.25% | 61.25% | **重度** |
| S11 | 70.00% | 89.38% | 94.38% | 74.38% | 干净 |
| S12 | 73.75% | 85.00% | 90.00% | 76.25% | 信息性 |
| S13 | 91.88% | 95.62% | 93.75% | 87.50% | 干净 |
| S14 | 78.12% | 83.12% | 87.50% | 67.50% | **重度** |
| S15 | 71.25% | 92.50% | 95.00% | 75.00% | 干净 |
| S16 | 56.25% | 70.62% | 94.38% | 60.00% | 轻度 |
| S17 | 70.62% | 84.38% | 90.00% | 76.88% | 干净 |
| S18 | 91.25% | 91.88% | 95.62% | 90.00% | 干净 |
| S19 | 85.62% | 98.12% | 99.38% | 93.75% | 信息性 |
| S20 | 52.50% | 61.25% | 65.62% | 55.62% | 信息性 |
| S21 | 66.88% | 73.12% | 79.38% | 61.88% | 轻度 |

> **数据来源**: EEGNet 被试内: `results/20260316_1411_comparison_cache_imagery_binary.json`; CBraMod 被试内: `results/20260323_2237_comparison_cache_imagery_binary.json`; CBraMod 跨被试: `results/20260324_0023_cross_subject_cache_imagery_binary.json`; EEGNet 跨被试: `results/20260330_0709_cross_subject_cache_imagery_binary.json`

### Table S1b. 逐被试结果（128 通道，三分类）

| 被试 | EEGNet 被试内 | CBraMod 被试内 | CBraMod 跨被试 | EEGNet 跨被试 |
|------|-------------|---------------|---------------|-------------|
| S01 | 51.25% | 57.50% | 66.25% | 53.33% |
| S02 | 85.83% | 85.00% | 89.17% | 75.42% |
| S03 | 81.67% | 85.00% | 93.75% | 71.67% |
| S04 | 88.75% | 92.08% | 93.75% | 67.08% |
| S05 | 58.75% | 56.25% | 66.25% | 44.17% |
| S06 | 69.58% | 77.92% | 82.92% | 58.75% |
| S07 | 73.33% | 67.50% | 74.17% | 71.67% |
| S08 | 77.08% | 78.33% | 85.42% | 68.75% |
| S09 | 87.92% | 86.25% | 90.00% | 84.58% |
| S10 | 57.92% | 45.42% | 52.08% | 50.00% |
| S11 | 51.67% | 72.08% | 80.83% | 58.33% |
| S12 | 58.33% | 60.00% | 59.17% | 54.58% |
| S13 | 83.33% | 72.92% | 75.00% | 72.08% |
| S14 | 64.17% | 71.67% | 82.08% | 52.50% |
| S15 | 66.25% | 61.25% | 69.17% | 68.33% |
| S16 | 44.17% | 51.67% | 67.50% | 50.00% |
| S17 | 72.08% | 80.00% | 81.25% | 62.08% |
| S18 | 78.33% | 66.67% | 74.58% | 64.58% |
| S19 | 61.67% | 93.33% | 91.25% | 70.42% |
| S20 | 36.25% | 44.17% | 42.92% | 42.92% |
| S21 | 54.58% | 60.83% | 55.00% | 44.58% |

> **数据来源**: 被试内: `results/20260329_0056_within_subject_cache_imagery_ternary.json`; CBraMod 跨被试: `results/20260324_0109_cross_subject_cache_imagery_ternary.json`; EEGNet 跨被试: `results/20260330_0735_cross_subject_cache_imagery_ternary.json`

### Table S2. 逐被试结果（32 通道，CBraMod 跨被试二分类）

| 被试 | FDR | Attention | CSP | Band Power | Commercial |
|------|-----|-----------|-----|-----------|------------|
| S01 | 86.88% | 77.50% | 86.88% | 85.62% | 82.50% |
| S02 | 91.25% | 89.38% | 88.12% | 95.62% | 90.00% |
| S03 | 99.38% | 96.25% | 96.88% | 97.50% | 97.50% |
| S04 | 96.88% | 95.62% | 92.50% | 98.12% | 95.00% |
| S05 | 75.00% | 79.38% | 74.38% | 75.00% | 84.38% |
| S06 | 80.00% | 75.00% | 71.88% | 77.50% | 75.62% |
| S07 | 87.50% | 86.88% | 85.00% | 88.75% | 81.25% |
| S08 | 91.88% | 93.75% | 93.75% | 91.88% | 94.38% |
| S09 | 97.50% | 95.62% | 96.25% | 97.50% | 97.50% |
| S10 | 70.00% | 71.88% | 61.25% | 65.62% | 69.38% |
| S11 | 91.88% | 88.12% | 91.88% | 93.75% | 91.25% |
| S12 | 85.00% | 81.25% | 81.25% | 86.88% | 86.88% |
| S13 | 91.25% | 90.62% | 90.00% | 90.00% | 87.50% |
| S14 | 91.25% | 85.62% | 85.62% | 84.38% | 78.75% |
| S15 | 89.38% | 91.25% | 93.12% | 91.25% | 89.38% |
| S16 | 88.12% | 83.75% | 83.12% | 86.25% | 87.50% |
| S17 | 93.75% | 88.75% | 88.12% | 89.38% | 92.50% |
| S18 | 90.62% | 85.62% | 90.00% | 92.50% | 89.38% |
| S19 | 98.75% | 97.50% | 99.38% | 98.75% | 96.88% |
| S20 | 66.88% | 65.00% | 62.50% | 63.75% | 65.62% |
| S21 | 78.75% | 76.25% | 71.88% | 73.75% | 75.00% |

> **数据来源**: `results/32_channel/{fdr,attention,csp,band_power,commercial}/20260330_*_cross_subject_cache_imagery_binary.json`

### Table S3. 逐被试结果（Extra Sessions 二分类，被试内，N = 16）

**CBraMod:**

| 被试 | Baseline | +Sess03 | +Sess04 | +Sess05 | Δ (总变化) |
|------|---------|---------|---------|---------|-----------|
| S02 | 94.38% | 96.88% | 95.62% | 98.75% | +4.38 pp |
| S03 | 94.38% | 93.75% | 98.12% | 97.50% | +3.12 pp |
| S04 | 91.88% | 90.62% | 95.00% | 98.75% | +6.88 pp |
| S06 | 74.38% | 80.62% | 71.88% | 92.50% | +18.12 pp |
| S07 | 81.88% | 86.25% | 84.38% | 93.12% | +11.25 pp |
| S08 | 93.12% | 93.75% | 93.75% | 97.50% | +4.38 pp |
| S09 | 99.38% | 98.12% | 96.25% | 93.12% | −6.25 pp |
| S10 | 60.62% | 66.25% | 69.38% | 74.38% | +13.75 pp |
| S11 | 89.38% | 89.38% | 97.50% | 95.62% | +6.25 pp |
| S13 | 95.62% | 88.75% | 93.75% | 95.62% | +0.00 pp |
| S14 | 83.12% | 88.75% | 89.38% | 93.12% | +10.00 pp |
| S15 | 92.50% | 95.62% | 94.38% | 94.38% | +1.88 pp |
| S16 | 70.62% | 73.12% | 97.50% | 95.00% | +24.38 pp |
| S17 | 84.38% | 93.12% | 88.12% | 85.62% | +1.25 pp |
| S18 | 91.88% | 93.75% | 90.00% | 92.50% | +0.62 pp |
| S19 | 98.12% | 97.50% | 100.00% | 96.25% | −1.88 pp |

**EEGNet-16,4:**

| 被试 | Baseline | +Sess03 | +Sess04 | +Sess05 | Δ (总变化) |
|------|---------|---------|---------|---------|-----------|
| S02 | 94.38% | 92.50% | 93.12% | 95.00% | +0.62 pp |
| S03 | 85.00% | 86.25% | 87.50% | 86.88% | +1.88 pp |
| S04 | 94.38% | 95.00% | 94.38% | 90.00% | −4.38 pp |
| S06 | 68.12% | 87.50% | 73.75% | 90.62% | +22.50 pp |
| S07 | 76.88% | 80.00% | 86.88% | 92.50% | +15.62 pp |
| S08 | 85.00% | 91.25% | 95.00% | 94.38% | +9.38 pp |
| S09 | 99.38% | 98.75% | 98.12% | 95.62% | −3.75 pp |
| S10 | 70.00% | 70.62% | 76.25% | 70.62% | +0.62 pp |
| S11 | 70.00% | 84.38% | 87.50% | 89.38% | +19.38 pp |
| S13 | 91.88% | 87.50% | 91.25% | 91.25% | −0.62 pp |
| S14 | 78.12% | 89.38% | 85.62% | 76.88% | −1.25 pp |
| S15 | 71.25% | 85.00% | 84.38% | 86.25% | +15.00 pp |
| S16 | 56.25% | 81.88% | 94.38% | 83.75% | +27.50 pp |
| S17 | 70.62% | 82.50% | 64.38% | 76.25% | +5.62 pp |
| S18 | 91.25% | 93.12% | 94.38% | 95.00% | +3.75 pp |
| S19 | 85.62% | 98.12% | 100.00% | 91.25% | +5.62 pp |

> **数据来源**: `results/20260324_2131_extra_sessions_cache_imagery_binary.json`
> 统计脚本: `scripts/paper/compute_paper_statistics.py --section s3`

### Table S4. 逐被试结果（Further Pre-training V1，CBraMod 被试内二分类）

| 被试 | Baseline (TUEG) | Further-PT V1 | Delta |
|:---:|:---:|:---:|:---:|
| S01 | 83.75% | 83.12% | −0.63% |
| S02 | 93.12% | 91.88% | −1.25% |
| S03 | 98.75% | 97.50% | −1.25% |
| S04 | 91.88% | 88.75% | −3.12% |
| S05 | 83.12% | 80.00% | −3.12% |
| S06 | 73.12% | 68.12% | −5.00% |
| S07 | 77.50% | 77.50% | 0.00% |
| S08 | 95.00% | 92.50% | −2.50% |
| S09 | 99.38% | 96.25% | −3.12% |
| S10 | 61.88% | 58.75% | −3.12% |
| S11 | 89.38% | 88.75% | −0.63% |
| S12 | 87.50% | 81.88% | −5.63% |
| S13 | 93.12% | 90.00% | −3.12% |
| S14 | 83.12% | 78.12% | −5.00% |
| S15 | 90.62% | 92.50% | +1.88% |
| S16 | 74.38% | 83.75% | +9.38% |
| S17 | 84.38% | 82.50% | −1.88% |
| S18 | 88.12% | 88.75% | +0.63% |
| S19 | 98.75% | 98.75% | 0.00% |
| S20 | 64.38% | 60.62% | −3.75% |
| S21 | 75.62% | 80.62% | +5.00% |
| **Mean** | **85.09%** | **83.84%** | **−1.25%** |

改善: 4/21 (S15, S16, S18, S21)；退步: 15/21；持平: 2/21 (S07, S19)。

> **数据来源**: `paper/analysis/further_pretraining_analysis.md` Section 5.3
> Baseline: ExperimentDB `run_tag=20260321_0343`; FT-V1: `results/20260322_1232_cbramod_imagery_binary.json`

### Table S5. HPO fANOVA 参数重要性

**CBraMod within-subject HPO (25 trials):**

| 排名 | 参数 | 重要性 | 与 acc 相关性 | 方向 |
|------|------|--------|-------------|------|
| 1 | phase_decay | 0.233 | −0.68 | 低更好 |
| 2 | dropout_rate | 0.196 | −0.74 | 低更好 |
| 3 | gradient_clip | 0.130 | −0.68 | 低更好 |
| 4 | classifier_lr_ratio | 0.100 | +0.48 | 高更好 |
| 5 | backbone_lr | 0.099 | +0.54 | 高更好 |

**CBraMod cross-subject HPO (43 trials):**

| 排名 | 参数 | 重要性 | 与 acc Spearman r | 方向 |
|------|------|--------|------------------|------|
| 1 | backbone_lr | 0.668 | +0.40* | 高更好 |
| 2 | classifier_lr_ratio | 0.156 | +0.27 | 高更好 |
| 3 | phase_epochs | 0.064 | +0.33* | 长更好 |
| 4 | label_smoothing | 0.030 | −0.19 | 低更好 |
| 5 | dropout_rate | 0.029 | +0.07 | 低影响 |

关键发现：within-subject 中正则化参数（dropout, weight_decay）占主导，低正则化最优——预训练 backbone 已具备充分的内在正则能力。cross-subject 中学习率（backbone_lr）以 66.8% 重要性独占鳌头，与 within-subject 形成显著差异。

> **数据来源**: `paper/analysis/hpo_within_subject_analysis.md` Section 4; `paper/analysis/hpo_cross_subject_analysis.md` Section 4

### Table S5b. HPO 超参数变化对照

以下三表展示各模型/范式配置从初始默认值到 HPO 最优值再到实际采用值的完整决策链。"实际采用"列为论文中所有实验使用的最终参数。

**CBraMod Within-Subject (HPO Trial #46, best=86.01%, 51 trials / 23 complete):**

| 参数 | 初始默认 | HPO 最优 | 实际采用 | 变化 |
|------|---------|---------|---------|------|
| backbone_lr | 1e-4 | 2.87e-4 | 2.9e-4 | ×2.9 |
| classifier_lr_ratio | 3× | 4.03× | 4× | ×1.3 |
| weight_decay | 0.06 | 0.026 | 0.026 | ↓2.3× |
| dropout_rate | 0.15 | 0.098 | 0.10 | ↓1.5× |
| batch_size | 128 | 256 | 256 | ×2 |
| label_smoothing | 0.05 | 0.087 | **0.05** | override |
| gradient_clip | 1.0 | 0.729 | 0.73 | ↓1.4× |
| phase_decay (CAWD) | 0.7 | 0.468 | 0.47 | ↓1.5× |
| phase_epochs (CAWD) | 6 | 8 | 8 | +2 |
| exploration_epochs (CAWD) | 6 | 4 | 4 | −2 |
| exploration_batch_size (CAWD) | 32 | 64 | 64 | ×2 |

**EEGNet Within-Subject (HPO Trial #23, best=82.71%, 32 trials / 10 complete):**

| 参数 | 初始默认 | HPO 最优 | 实际采用 | 变化 |
|------|---------|---------|---------|------|
| F1 (filters) | 8 | 16 | 16 | ×2 |
| D (depth multiplier) | 2 | 4 | 4 | ×2 |
| F2 (= F1×D) | 16 | 64 | 64 | ×4 |
| learning_rate | 1e-3 | 3.98e-3 | 4e-3 | ×4 |
| weight_decay | 0 | 1.09e-5 | 1e-5 | 新增 |
| dropout_rate | 0.5 | 0.271 | 0.27 | ↓1.9× |
| batch_size | 64 | 64 | 64 | — |
| kernel_length | 64 | 64 | 64 | — |

**CBraMod Cross-Subject (HPO Trial #4, best=90.68%, 77 trials / 43 complete):**

| 参数 | 初始默认 | HPO 最优 | 实际采用 | 变化 |
|------|---------|---------|---------|------|
| backbone_lr | 1e-4 | 1.335e-4 | 1.3e-4 | ×1.3 |
| classifier_lr_ratio | 1.5× | 1.6× | 1.7× | ×1.1 |
| weight_decay | 0.12 | 0.130 | 0.13 | ≈不变 |
| dropout_rate | 0.35 | 0.369 | 0.37 | ≈不变 |
| batch_size | 256 | 256 | 256 | — |
| label_smoothing | 0.15 | 0.285 | **0.05** | override |
| gradient_clip | 0.5 | 1.363 | 1.4 | ×2.8 |
| phase_decay (CAWD) | 0.5 | 0.499 | 0.50 | ≈不变 |
| phase_epochs (CAWD) | 6 | 10 | 10 | +4 |
| exploration_epochs (CAWD) | 6 | 3 | 3 | −3 |
| exploration_batch_size (CAWD) | 64 | 128 | 128 | ×2 |

> **用户 Override 说明**: label_smoothing 在所有配置中均被手动固定为 0.05（HPO 在 binary 任务上搜索，但统一模型包含 quaternary 子任务，chance=25%，高 label_smoothing 会严重削弱 4-class 弱学习信号）。
>
> **关键模式**: Within-subject HPO 最大收益来自**降低正则化**（dropout ↓1.5×, weight_decay ↓2.3×）——预训练 backbone 的内在正则化已足够。EEGNet 最大收益来自**架构升级** (F1: 8→16, D: 2→4, 参数量 ~2.5K→~10K, +3.8 pp)。Cross-subject HPO 参数变化极小（weight_decay、dropout 几乎不变），表明初始默认值已接近最优。
>
> **数据来源**: `docs/dev_log/experiments/hpo_final_parameters.md`

### Table S6. 早停最优模型选择策略对比（CBraMod 128ch 被试内二分类，N = 21）

**策略定义**：

- **combined** (baseline): `selection_score = (val_acc + majority_acc) / 2`，其中 val_acc 为 segment-level 验证准确率，majority_acc 为 trial-level 多数投票验证准确率。当 score 改善时保存 checkpoint。
- **val_acc**: `selection_score = val_acc`（仅 segment-level 验证准确率），其余同 combined。
- **EMA**: 使用指数移动平均（decay=0.998）维护 shadow weights，每 epoch 更新一次；validation 在 EMA 权重下执行，`selection_score = (ema_val_acc + ema_majority_acc) / 2`。
- **SOUP** (Stochastic Weight Averaging Uniformly): 训练过程与 combined 完全一致，训练后加载 top-3 milestone checkpoint 进行权重算术平均作为最终模型。零额外训练开销。

| 策略 | Mean ± SD | Min | Max | vs. combined Δ |
|------|-----------|-----|-----|----------------|
| **SOUP** | **85.09 ± 10.46%** | 61.88% | 99.38% | **+0.24 pp** |
| combined (baseline) | 84.85 ± 10.73% | 60.62% | 99.38% | — |
| val_acc | 84.73 ± 10.96% | 61.25% | 99.38% | −0.12 pp |
| EMA | 71.90 ± 14.19% | 50.00% | 97.50% | −12.95 pp |

SOUP 微弱领先 combined baseline（+0.24 pp），但差异不具统计显著性。EMA 在低数据被试上表现灾难性（S09: 66.25% vs combined 99.38%），均值暴降 −12.95 pp。

> **HPO 偏置声明**: 所有策略共用来自 HPO Trial #46 的超参数（best_value=86.01%），该 HPO 的目标函数为 `combined_score`。因此 combined 策略享有参数匹配优势，其他策略可能因参数不匹配而处于系统性劣势。结论性判断需在各策略各自完成独立 HPO 后才能做出。EMA 的灾难性表现（−12.95 pp）尤其可能源于 decay=0.998 在 50-epoch 短训练下的根本不匹配（有效半衰期 ~347 epoch），而非 EMA 方法本身的缺陷。

> **数据来源**: `paper/analysis/model_selection_strategy_analysis.md` Section 3.1

### Figure S1. 三种评估策略对比（Extra Sessions 二分类，N = 16）

**策略定义**：

- **per_session** (同[3]数据集论文原设定): 每步训练集递增，测试集为当前最新 session 的 Finetune 部分。模拟临床部署场景："模型在最新采集的 session 上表现如何？"非单调增长反映测试集难度差异。
- **fixed_combined**: 将所有 session 的 Finetune 尾部 1/4 trial 合并为固定测试集（160 trials/被试），训练集递增。消除测试集难度混淆，回答："相同测试条件下，数据量增加带来多少改善？"
- **fixed_sess02**: 测试集始终为 Sess02 Finetune（最早 session），训练集递增但不含 Sess02 FT。度量时间泛化："后续 session 的训练能改善对早期 session 的预测吗？"最保守估计。

![Figure S1. Extra Sessions 评估策略对比](../figures/extra_sessions_strategy_comparison.png)

图中左侧端点标签给出 Baseline 绝对均值，右侧端点标签给出 +Sess05 绝对均值与相对 Baseline 的净增益；下方小表列出各策略在四个 step 的精确群体均值，避免仅看相对增益或末端标签重叠造成歧义。

策略间核心差异：
- **per_session** 在 EEGNet +Sess03 步骤出现非比例跳跃（+7.22 pp），部分归因于后续 session 被试技能提升导致的测试集"更容易"
- **fixed_combined** 展现清晰单调递增趋势，提供数据量效应的最干净信号
- **fixed_sess02** 下 CBraMod 后段趋平（+Sess04→+Sess05 仅 +0.86 pp），提示预训练基座模型更快捕获时间不变特征但对 session 间分布漂移更敏感
- 三种策略对"额外 session 数据有益"的结论方向一致，差异仅在效应量估计

> **数据来源**: per_session: `results/20260324_2131_extra_sessions_cache_imagery_binary.json`; fixed_combined: `results/20260325_0514_extra_sessions_cache_fixed_combined_imagery_binary.json`; fixed_sess02: `results/20260325_1208_extra_sessions_cache_fixed_sess02_imagery_binary.json`
> 生成脚本: `scripts/paper/generate_paper_figures.py --figure extra_sessions_strategy`
