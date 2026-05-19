# 基于 EEG 基座模型的手指级运动想象分类：通道缩减、纵向数据扩展与领域自适应预训练的局限性

> **草稿说明**：本文为工作草稿（v3）。文中大部分图表为脚本自动生成的初步输出，**尚未进行出版级精修**（坐标轴标签、字体大小、配色方案、排版布局等）。标有 `[TODO]` 的章节表示数据需最终核实或补充可视化。
>
> **v3 变更摘要**：
> - 论文语言从英文转为中文（技术术语保留英文）
> - 新增多 session 纵向扩展实验结果（原 TODO 6.2，现 Section 3.4）
> - 新增领域自适应 further pre-training 负面结果（Section 2.7 + 3.6）
> - 新增推理性能基准测试（Section 3.8）
> - 新增容量与预训练消融（Section 3.7）：EEGNet 容量阶梯（16K → 30M，§3.7.1）+ random-init CBraMod（§3.7.2）+ 架构 / 预训练 / 容量三向分解（§3.7.3）
> - HPO 方法论纳入 Methods（Section 2.5.1）
> - "Ongoing Experiments" 改为 "Future Work"

---

## 摘要

脑机接口（Brain-Computer Interface, BCI）通过脑电图（EEG）解码单指运动意图，在精细运动康复领域具有重要应用前景，但高密度电极阵列的部署限制了其临床推广。本研究系统对比了大规模 EEG 基座模型 CBraMod（30.48M 参数含分类头；~4M backbone + ~26M MLP 头，ICLR 2025）与轻量级卷积神经网络 EEGNet-16,4（~16K 参数）在单指运动想象（Motor Imagery, MI）分类中的性能，覆盖 21 名健康被试、128 通道 BioSemi 系统、被试内/跨被试/XSI-FT（Cross-Subject-Initialized Per-Subject Fine-Tuning）三种训练范式。

在三种训练范式下（128 通道），CBraMod 一致优于 EEGNet——被试内二分类 **+7.05 pp**（85.15% vs 78.10%）、跨被试二分类 **+14.01 pp**（90.68% vs 76.67%；21 名 responder 被试，原数据集 [3] 49 名招募者中筛选后 cohort，详见 §2.1）、跨被试三分类 **+13.65 pp**（74.88% vs 61.23%）——其中跨被试范式下双位数 pp 的差距是本研究最稳健的 backbone 改进。为更好理解该差距来源，§3.7 进行了两项探索性消融。(a) **EEGNet 容量阶梯（16K → 1.90M → 5.84M → 30M）** 显示扩参 EEGNet 在固定优化栈下严重退化，cross-subject 准确率从 76.67% 单调下降到 51.37% / 50%（chance），方向性提示沿当前扩参轴对 EEGNet 扩参不利；(b) **random-init CBraMod 消融**显示在 ~30M 参数 + 无预训练同等条件下，CBraMod 仍领先扩参 EEGNet ~+35 pp（cross-subject），加 TUEG 预训练再追加 +4.34 pp（cross）至 binary +23.10 pp / ternary +30.79 pp（被试内）。然而 EEGNet-Huge 与 CBraMod random-init 均未做专属 HPO，且 baseline → Mid 跳跃同时改变 conv stem 与 MLP 头，因此这些消融**不构成对架构、预训练、容量三因子的独立可归因分解**，应被理解为方向性观察。within-subject 严重 collapse 的现象与 NLP 文献中 transformer 在小样本下的已知微调脆弱性方向一致；严格独立 HPO 验证留待后续工作。**头条 robustness 验证**：跨被试 binary 90.68% 头条经标签置换控制（n=2 seeds, pooled 49.58%, Δ=−41.1 pp 相对真实标签）通过验证，结果不依赖于标签级泄露或被试身份混淆——与 §3.5.3 的 4ch 负控制（67.65% 远高于 chance）+ §3.9 的 leave-3-out（重度伪影被试去除 |Δ| ≤ 0.13 pp）共同构成三重 robustness 证据链（详见 §3.9）。

在通道缩减方面，我们评估了 128、64、61、32、8、4 通道配置及四种数据驱动选择方法（Fisher 判别比、共空间模式、梯度注意力、频带功率）和一种商用布局，构建覆盖 {4, 8, 32, 64}ch × 5 method × {binary, ternary} = 40 cell 的完整 cross-subject 矩阵。Fisher 判别比（FDR）选取的 32 通道配置保留了 128 通道 CBraMod 性能的 96.7%（87.71% vs 90.68%），64 通道 FDR 进一步达到 98.7%（89.46%），而 EEGNet 在 32 通道相同条件下降至 74.70%。通道选择方法间差异随通道数递减而扩大，且在 binary / ternary 两 task 上同向复现：**binary** 64/32/8/4 通道分别 3.24 / 2.77 / 15.63 / 24.05 pp，**ternary** 1.77 / 2.08 / 6.83 / 19.12 pp（4 数据驱动方法 max−min）。在 32ch+ 档位（含新增的 64ch 全 5 method 行），数据驱动方法之间以及与负控制之间的差异均在 ±0.32 pp 内，表明此区间方法选择对性能影响在 run-to-run noise 量级；在 4 通道极端约束下，mu/beta 频带 Band Power 方法在 binary (78.75%) 与 ternary (60.67%) 上均显著超越负控制（+11.10 / +7.30 pp）并稳居所有数据驱动方法之首，而 FDR/Attention/CSP 在两 task 上全部跌至负控制水平或以下——本研究将其解读为"基于全模型的条件重要性排序在极低通道数下因失去上下文而崩溃"的具体表现，而非"频域方法在通道选择中具有普适优势"的方法论断。

在纵向数据扩展方面，对 16 名拥有 3–5 个额外在线 session 的被试进行分析表明，额外同被试数据的价值强烈依赖训练范式。被试内二分类中，EEGNet 从 80.51% 提升至 87.85%（+7.34 pp，p = 0.009），CBraMod 从 87.23% 提升至 93.36%（+6.13 pp，p = 0.007）；而 21 名被试联合训练的跨被试 pooled model 仅从 92.38% 小幅升至 93.24%（+0.86 pp，p = 0.662）。使用对应 cross-subject checkpoint 作为初始权重再做单被试 fine-tune 的 XSI-FT（Cross-Subject-Initialized Per-Subject Fine-Tuning，详见 §3.3）达到 92.93%（+5.70 pp，p = 0.015），与被试内重训练相近但未进一步突破其终点。三分类中，被试内 CBraMod 仍显示显著改善（74.51% → 83.06%，+8.55 pp，p = 0.012），而跨被试 ternary 增益更温和（+3.73 pp，p = 0.090）。

在领域自适应预训练方面，我们评估了 5 个独立训练配置（V1–V3：10-dataset 系列；V4：3-set 域对齐 + strict filter；V5：Stieger 单源 60ch），共 24 个 paired comparison cell（V1–V3 × within+cross × bin+ter = 12；V4–V5 × within+cross+transfer × bin+ter = 12；V4/V5 within+transfer 8 cell 于 2026-05-10 补完）。结果呈现 **task-asymmetric 负迁移且跨范式复现**：cross-subject **binary** 5/5 一致负向（平均 Δ=**−1.79 pp**，Stouffer Z=−5.32, p<0.001）、within-subject **binary** 5/5 一致负向（Stouffer Z=−4.42, p<0.0001）、transfer **binary** V4/V5 双双负向（Stouffer Z=−2.79, p=0.005）——**binary 任务上 DAPT 失败不是 cross-subject 特有现象，而是跨三种 paradigm 的稳健模式**。Ternary 任务相对温和：cross 4/5 弱正（mean Δ=+0.18 pp，Stouffer p=0.564）、within 5/5 负但弱（mean Δ=−0.92 pp，Stouffer Z=−2.16, p=0.031）、transfer V4/V5 均弱负（mean Δ=−0.90 pp，p=0.110）。V4/V5 12-cell 全矩阵下**0/12 正向显著**且 V5 在 5/6 cell 上比 V4 更差（−1.15 至 −1.82 pp 量级），把候选机制收紧到唯一存活假设——**MI 粒度错配（pretext-task granularity mismatch）**：粗 hand/leg/upper-limb MI 学到的是低频空间包络，下游 finger-level binary 需要 DAPT 未学到的细粒度区别；ternary 的 rest 类可用粗粒度识别，因此不那么受损。V5 单源 60ch 反方向证伪了"通道数异质性是混淆"假设——通道多样性在 DAPT 中是**保护因子**而非 bug。BH-FDR 在新 24-cell DAPT family 内重做后，仅 V2_within_binary (q=0.048) 单一显著存活（v3.1 16-cell family 下原 3 个 survivors 在更严苛的多重比较惩罚下退到 q ≈ 0.07–0.09，但 Stouffer 聚合的 paradigm-level 集体证据全部仍稳健）。

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

CBraMod（Criss-Cross Brain Foundation Model）[4]，被 ICLR 2025 接收，是一个基于 Transformer 的模型，在 Temple University EEG (TUEG) 语料上进行自监督预训练。其关键架构创新——非对称条件位置编码（Asymmetric Conditional Positional Encoding, ACPE）——使模型能够接受任意数量的输入通道而无需重新训练，这对通道缩减实验至关重要。CBraMod 含分类头共 30.48M 参数（其中 backbone ~4M + MLP 分类头 ~26M），是 EEGNet-16,4 [5]（~16K 参数，BCI 研究的标准基线 CNN）的约 1,900 倍（如表 2b）。

值得注意的是，TUEG 预训练语料主要包含临床 EEG（静息态、病理等），与运动想象 EEG 在信号特征上存在显著差异。一个自然的问题是：在外部 MI 数据集上进行 domain-adaptive further pre-training，能否弥合这一领域鸿沟并改善下游性能？这一假设在 NLP（Gururangan et al. 2020 [20] "Don't Stop Pretraining"）与 CV（如将 ImageNet 模型适配到医学影像）中已被广泛研究，但在 EEG 基座模型中尚缺乏系统评估。

其他并行工作包括 LaBraM [6]、NeuroLM [15]、BIOT [16] 与颅内信号基座模型 Brant [17]，以及 Schirrmeister et al. 2017 [10] 的 Deep ConvNet 与 Sakhavi et al. 2018 [11] 的 temporal CNN（均为与 EEGNet 并列的 BCI deep learning baseline）以及 EEG 基座模型综述 [7]，共同证实了预训练方法在低数据和跨被试场景中一致优于任务特异性模型。

### 1.4 本文贡献

本文做出以下贡献：

> 1. **系统性基座模型评估 + 探索性消融初步检验差距来源**。首次在手指级运动想象分类任务上，对 EEG 基座模型（CBraMod）与传统 CNN（EEGNet-16,4）进行全面对比，覆盖被试内、跨被试、跨被试初始化的逐被试微调（XSI-FT，§3.3）三种范式，使用 21 名被试数据（21 名 responder cohort，继承自 [3] 的 49 → 21 离线筛选），并采用贝叶斯超参数优化（HPO，CBraMod 11 维 / EEGNet 7 维，trial 数按 d^1 校准；详见 §2.5.1）确保公平比较。在三种范式下 CBraMod 一致优于 EEGNet（被试内 +7.05 pp、跨被试二分类 +14.01 pp、跨被试三分类 +13.65 pp）。作为补充，§3.7 进行了两项探索性消融以理解架构 / 预训练 / 容量的相对贡献：(a) **EEGNet 容量阶梯（16K → 1.90M → 5.84M → 30M）** 显示 cross-subject 准确率沿当前扩参轴单调下降至 51.37% / 50%（chance），方向性提示沿该轴扩参 EEGNet 不利，但 v1/v2 (~20-30M) 不可训根据作者本人交接诊断更可能是 BF16 下深 MLP 头优化栈兼容性问题（v3 加 LayerNorm 立即 trainable）而非容量饱和；(b) **random-init CBraMod 消融** 显示在 ~30M 参数 + 无预训练条件下，CBraMod 仍领先扩参 EEGNet ~+35 pp（cross-subject），加 TUEG 预训练再追加 +4.34 pp（cross-subject）至 binary +23.10 / ternary +30.79 pp（被试内）。**因 EEGNet-Huge 与 CBraMod random-init 均未做专属 HPO，且 baseline → Mid 跳跃同时改变 conv stem 与 MLP 头**，这些消融在本研究范围内不构成独立可归因的三向分解，应被理解为方向性观察；严格的独立 HPO 验证留待后续工作（§6 #8）。详见 §3.7 caveats。
>
> 2. **全面通道缩减分析与完整 {channel × method × task} 矩阵**。建立 {4, 8, 32, 64}ch × {FDR, Attention, Band Power, CSP, negative_control} × {binary, ternary} = 40 cell 的 cross-subject CBraMod 矩阵（含 5 种 32 通道配置中的 4 数据驱动 + 1 商用布局，以及 61 / 8 / 4 通道方案）。FDR 选取的 32 通道保留 128 通道性能的 **96.7%**（在 21 名 responder cohort × cross-subject binary 上；通道选择 ranking 使用了所有 session 数据，可能轻微高估 retention，详见 Limitation #1）。
>
> 3. **通道选择方法间差异随通道数减少而扩大；binary / ternary 双 task 同向复现**。在本数据集上，4 数据驱动方法（FDR / Attention / Band Power / CSP）的 max−min spread 随通道数递减而单调扩张：binary 在 64 / 32 / 8 / 4 通道分别为 3.24 / 2.77 / 15.63 / 24.05 pp，ternary 分别为 1.77 / 2.08 / 6.83 / 19.12 pp（数据来源：[docs/handoffs/2026-05-11_reduced_channel_matrix_closure.md](../../docs/handoffs/2026-05-11_reduced_channel_matrix_closure.md) §40-cell 矩阵）。在 32ch+ 档位，数据驱动方法之间以及与负控制之间在 ternary 上的差异均在 ±0.32 pp 内（即被试间 std ≈ 13 pp 的 noise 量级），支持"32ch 起方法选择对性能影响在统计上不可区分"的论断。通过负控制实验确认体积传导冗余而非数据泄露；并在 4 通道下识别 mu/beta Band Power 在双 task 上均保持判别力（binary 78.75% +11.10 pp、ternary 60.67% +7.30 pp 超过负控制）——其评分机制不依赖全模型上下文，因而免疫"条件重要性外推失效"陷阱（本研究观察的具体机制，未声称跨数据集普适）。
>
> 4. **多 session 纵向数据扩展与范式差异**。系统比较额外 session 数据在被试内、跨被试 pooling、以及 **XSI-FT**（**Cross-Subject-Initialized Per-Subject Fine-Tuning**，跨被试初始化的逐被试微调；机制：以 cross-subject checkpoint 作为单被试 fine-tune 的初始权重；正式定义见 §3.3）三种训练范式中的作用。CBraMod 在被试内重训练中获得最大净增益（+6.13 pp 至 93.36%），XSI-FT 达到相近终点（92.93%，+5.70 pp），而 pooled cross-subject 模型仅小幅改善（+0.86 pp 至 93.24%）——这一对照表明，随同被试数据的累积，cross-subject 训练所带来的额外优势随之减弱。
>
> 5. **领域自适应预训练的 task-asymmetric 负面结果与机制收紧；跨 paradigm 复现稳健**。系统评估在外部 MI 数据上对 CBraMod 进行 further pre-training，覆盖 5 个独立训练配置（V1/V2/V3：10-dataset 系列；V4：3-set 域对齐 + strict filter；V5：Stieger 单源 60ch）共 24 个 paired comparison cell（V1–V3 within+cross + V4/V5 within+cross+transfer，于 2026-05-10 补完 V4/V5 within+transfer 8 cell）。结果呈 **task-asymmetric 负迁移且跨 paradigm 复现**：binary 任务上三种 paradigm 全部一致负向——cross-subject 5/5（mean Δ=−1.79 pp，Stouffer Z=−5.32, p<0.001），within-subject 5/5（Stouffer Z=−4.42, p<0.0001），transfer V4/V5（Stouffer Z=−2.79, p=0.005）；ternary 任务相对温和——cross 4/5 弱正、mean Δ=+0.18 pp、Stouffer p=0.564；within 5/5 弱负但 mean Δ 仅−0.92 pp（Stouffer Z=−2.16, p=0.031）；transfer V4/V5 均弱负 mean Δ=−0.90 pp（p=0.110）。**ternary 上的方向性负迁移声明不被支持**，但 binary 失败的稳健性现已得到三种 paradigm 的独立复现——DAPT 失败不是 cross-subject 特有现象。V4/V5 12-cell 全矩阵下 **0/12 正向显著**，V5 在 5/6 cell 上比 V4 更差 1.15–1.82 pp。BH-FDR 在新 24-cell DAPT family 下重做后仅 V2_within_binary (q=0.048) 单一显著存活（v3.1 16-cell family 下原 V1_cross_binary / V4_cross_binary 在更严苛的多重比较惩罚下退到 q≈0.07–0.09，但 paradigm-level Stouffer 集体证据全部仍稳健）。V4/V5 两次 surgical fix 把"域错配 / Stieger 占主导 / 通道数异质"三个候选机制收紧到唯一存活假设——**MI 粒度错配（pretext-task granularity mismatch）**：粗 hand/leg/upper-limb MI 的 MAE pretext loss 学到的是"哪个肢体在动"的低频空间包络，下游 finger-level binary 需要 DAPT 没学到的细粒度区别；ternary 的 rest 类则可用粗粒度空间包络识别。V5 单源 60-ch 反方向证伪"通道数异质是混淆"——通道多样性在 DAPT 中是**保护因子**，不是 bug。
>
> 6. **实际部署特性**。推理延迟基准测试确认 CBraMod 单样本延迟 <13 ms，满足实时 BCI 要求。

---

## 2. 材料与方法

### 2.1 数据集

本研究使用 Ding et al. [3] 公开发布的手指级 EEG-BCI 数据集（原始数据随文公开于 Figshare, DOI: `10.1184/R1/29104040`），包含 21 名健康右利手被试（S01–S21），进行手指级运动想象和运动执行任务。需要强调的是，这 21 名被试对应 [3] 在 49 名招募者中经离线二分类准确率筛选后保留的在线被试队列（cohort），而非无筛选总体样本。EEG 信号通过 128 通道 BioSemi ActiveTwo 系统以 1024 Hz 采样率采集。实验范式包括：

- **离线 session**：30 次训练 run，带视觉提示的个体手指想象（拇指、食指、中指、小指）
- **在线 session**：跨多天的实时 BCI 控制 session，每个 session 分为校准（Base）和自适应（Finetune）阶段

其中 16 名被试（S02, S03, S04, S06–S11, S13–S19）拥有 3–5 个额外在线 session（Sess03–Sess05），总录制时长约 64 小时。

本研究聚焦运动想象范式，采用两种主要分类粒度：

| 任务 | 类别 | 随机基线 |
|------|------|---------|
| **二分类（Binary）** | 拇指（class 1）vs. 小指（class 4） | 50% |
| **三分类（Ternary）** | 拇指（class 1）vs. 食指（class 2）vs. 小指（class 4） | 33.3% |
| **四分类（Quaternary）** | 拇指/食指/中指/小指 | 25% |

由于 finger-EEG 数据中 quaternary 四分类数据仅有 offline session 且采集于每名被试 session 序列的最早段（详见 §3.3.1 与 Sup Table S7），本文与数据集原文一样不将其作为主要研究范式：§3 主线只报告 binary 与 ternary，quaternary 全范式（被试内 / 跨被试 / XSI-FT）辅助结果整体放在补充材料 Table S7。Quaternary 同时被用于 unified-task HPO 的 label_smoothing 校准（Sup Table S5/S5b）。

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
| 评估目标 | 在线机器人控制 majority-vote accuracy | 多个公开下游任务的统一 benchmark | 被试内/跨被试/XSI-FT/extra-session/缩减通道的 held-out 准确率 |
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

与两篇来源论文相比，我们在两个方面对预处理协议做了显式偏离。第一，EEGNet 不再沿用 [3] 的在线流式 same-day 更新，而是纳入统一的 held-out session 训练/验证/测试协议。第二，两条管线都在 trial 级别（非 run 级别）应用 CAR，使用 `nanmean` 处理 NaN 填充的变长 trial（离线 trial: 5 s；在线 trial: 3 s），并通过 `scipy.signal.resample_poly` 的有理因子计算避免 FFT 混叠伪影。需要说明的是，CBraMod 在 128 通道手指 MI trial 上运行并非偏离 [4]——ACPE 本身即为支持任意通道数的输入而设计，是该模型的预期使用方式。未使用数据增强。

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

每一步训练集扩大，测试集为最新 session 的 Finetune 部分。除默认的被试内累积重训练外，我们还将同一 progressive split 重用于两种变体：（1）**跨被试 extra sessions**：每个 step 用 21 名被试的可用数据训练单一 pooled model，并在 16 名具有 extra sessions 的被试上评估；（2）**XSI-FT extra sessions**：每个 step 先读取对应的 cross-subject checkpoint 作为初始权重（即 §3.3 定义的 XSI-FT 机制），再对单被试进行离线微调。补充分析中使用 fixed_combined（固定组合测试集）和 fixed_sess02（固定 Sess02 测试集）两种策略控制测试集难度变化的混淆因素，详见 Supplementary。

### 2.4 模型架构

#### 2.4.1 EEGNet-16,4

EEGNet [5] 是一种紧凑的 CNN，也是 Ding et al. [3] 在线 finger-BCI 解码器的核心架构（原始配置为 EEGNet-8,2）。原论文随后又测试了更宽更深的 deepEEGNet，以检验额外 session 收益是否主要受模型容量限制，但观察到的性能提升仍较有限。本文不直接复用其在线默认配置，而是将 EEGNet-8,2 / deepEEGNet 作为文献锚点，结合 HPO 重新搜索架构与正则化参数，最终得到 EEGNet-16,4 配置（F1=16 时间滤波器，D=4 空间深度），参数量约 16,162，相比原始 EEGNet-8,2（~2.5K 参数）有 4 倍扩展。

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

所有报告结果均使用贝叶斯超参数优化（HPO）后的参数。这里的搜索并非从零随机设定，而是明确锚定两篇来源论文：EEGNet 侧以 [3] 的 EEGNet-8,2 / deepEEGNet 设计思路为起点，CBraMod 侧以 [4] 的 fine-tuning defaults 为起点。HPO 采用 Optuna 框架的 TPE（Tree-structured Parzen Estimator）[23] 采样器，搜索空间涵盖 7–11 个维度（学习率、权重衰减、dropout、batch size、学习率调度参数等）。

**搜索空间维度对照**：CBraMod within / cross-subject 各搜索 **11 个**超参数（backbone_lr、classifier_lr_ratio、weight_decay、dropout_rate、batch_size、label_smoothing、gradient_clip、phase_decay、phase_epochs、exploration_epochs、exploration_batch_size；其中后 4 项为 CAWD scheduler 参数），EEGNet within / cross-subject 各搜索 **7 个**超参数（learning_rate、weight_decay、dropout_rate、batch_size、F1、D、kernel_length；F2 = F1 × D 为派生量）。两者维度差源于 CBraMod 需同时调 backbone / classifier 两段学习率以及 CAWD 调度参数，而 EEGNet 共用单一 learning_rate 与简化的 plateau scheduler——并非"给 CBraMod 多调几个旋钮"，而是各自架构的本征参数化差异（详见 [src/hpo/search_spaces.py](../../src/hpo/search_spaces.py)）。

**Trial 预算的 HP-维度校准**：本文 CBraMod within-subject HPO 实际跑了 51 trial（Table S5b），EEGNet within-subject 跑了 32 trial，预算比 51 / 32 ≈ 1.59。在 TPE 类贝叶斯优化中，TPE 收敛到指定误差所需的 trial 数 N 经验上随搜索空间维度 d 以 O(d^c)（c ∈ [0.5, 1]）扩展（Bergstra & Bengio 2011 [23] §3.3 random/Bayesian search dimension dependence；Snoek et al. 2012 [24] §4.1 GP-EI sample complexity）。给定 EEGNet 7 维 vs CBraMod 11 维，"等效收敛"所需 trial 比的下界为 (11 / 7)^0.5 ≈ 1.25、上界为 (11 / 7)^1 ≈ 1.57。本文实际比 1.59 恰好落在 d^1 上界——也就是说，**CBraMod 的额外 trial 数恰好抵消了它额外 4 个搜索维度带来的体积膨胀，并未给 CBraMod 带来"等效收敛精度上的盈余"**。这一校准下两侧 HPO 同等可比，"CBraMod 优势源自不公平的 HPO 预算"反方解释在该校准下不成立；唯一保留的非对称性是 EEGNet 的 7 个 HP 中含有 3 个 architecture HP（F1、D、kernel_length），而本研究继承了 [3] 在原始数据集上对 EEGNet-8,2 / deepEEGNet 的架构调试经验，未把 architecture HP 完全重新搜索（详见 Table S5e EEGNet HP source trace；继承的 architecture HP 默认值在本研究 HPO 中仍允许变动并被显式优化）。

由于被试内 HPO 需要在每次 trial 中遍历全部 21 名被试、对每个个体训练独立模型，单次 trial 的 GPU 时间相当昂贵，朴素 random/TPE 搜索在我们的算力预算下几乎无法跑完一轮覆盖性扫描。为此我们实现了自定义的 ProbabilisticSubjectPruner：在被试内 trial 推进过程中，对当前已完成被试上的累计准确率与同期最优 trial 在相同被试集合上的累计准确率做比较，当一个 trial 的"累计性能超越当前最优"的后验概率低于 10% 时即提前终止剩余被试。该剪枝在被试内 HPO 中触发率为 52.9%–65.6%——也就是说，剪枝消除了大约一半到三分之二的"显然落后"被试评估，从而把单次 HPO 总训练 epoch 数与等效计算成本降低到原来的 ~35%–47% 区间，使我们能在有限 GPU 预算下完成更宽的搜索空间扫描。重要的是，这一加速并未明显牺牲搜索质量：被剪掉的 trial 集中在已经低于当前 best 的早期分支，最终被采纳的 best trial 仍由完整跑完所有被试的运行决定（详细参数见 Supplementary Table S5/S5b），而这些 best trial 的 best_value 与早期未启用剪枝的小规模对照搜索处于同一量级。

关键发现：
- EEGNet 从 HPO 中获益最大（+3.8 pp），主要贡献来自架构参数 F1, D 的扩展（8,2 → 16,4）
- CBraMod 的改进较小（+0.5–1.4 pp），预训练模型对超参数更鲁棒
- 详细 HPO 收敛曲线和参数重要性分析见 Supplementary Table S5

### 2.6 通道选择方法

通道缩减问题同样沿着两篇基础论文展开：Ding et al. [3] 已在该数据集上比较过感觉运动区、非感觉运动区以及 64/32/21 通道的手工布局；本文则进一步利用 CBraMod 的 ACPE 通道灵活性 [4]，把这一问题推进到数据驱动选点和更极端的 8/4 通道场景。

我们评估了以下五种 32 通道配置：

**数据驱动方法（4 种）：**
1. **Fisher 判别比（FDR）**：逐通道计算 mu (8–13 Hz) 和 beta (13–30 Hz) 频带的类间/类内方差比，取比值最高的 32 通道
2. **共空间模式（CSP）[12], [13]**：使用 MNE-Python 的 Ledoit-Wolf 协方差正则化，按空间模式贡献排序
3. **梯度注意力（Attention）**：聚合 CBraMod 输入梯度幅值，捕获模型分类时关注的通道
4. **频带功率（Band Power）**：mu/beta 频带功率的 ANOVA F 统计量排序

**手工设计配置（1 种）：**
5. **商用布局（Commercial）**：标准 10-20 系统在 BioSemi 128 通道上的映射

此外，还测试了 64、61（标准 10-10 系统）、16、8（FDR/Attention top-K）、4（FDR ∩ Attention 交集 + 负控制）通道配置——完整 64–4ch × 5 method × {binary, ternary} 缩放矩阵见 §3.5.2。5 种方法的具体电极空间布局以及 §3.5.3 控制实验所用的 32 通道负控制电极位置，详见 Figure S3（2D 详细图）与 Figure S4（3D 多视角）。

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

采用 masked autoencoding 自监督任务（50% mask ratio, MSE loss），在 TUEG 预训练权重基础上继续训练。测试了三种配置：

| 参数 | V1 | V2 | V3 (continued) | V4 | V5 |
|------|-----|-----|----------------|-----|-----|
| Base LR | 5e-5 | 5e-5 | 5e-5 | 5e-5 | 5e-5 |
| LR 调度 | Cosine decay → 1e-6 | Warmup 0.5ep → 恒定 lr=5e-5 | 恒定 lr=5e-5 | Warmup 0.5ep → 恒定 lr=5e-5 | Warmup 0.5ep → 恒定 lr=5e-5 |
| 最大 epoch | 10 | 50（early stop at 12） | 50（best at 22；continue training 共 27 epoch） | 50（best at 40；patience=5 早停） | 50（best at 21；patience=5 早停 @26） |
| 数据集组成 | 10 datasets | 10 datasets (rebalanced) | 10 datasets (Stieger 减权) | **3 datasets**：Cho2017 + Ofner2017 + Schirrmeister2017（去 Stieger） | **1 dataset**：Stieger2021 only |
| 通道数分布 | 22/30/60/61/62/64/128（7 种） | same as V1 | same as V1 | **3 种**（61 / 64 / 128） | **1 种**（60） |
| 伪影过滤 | basic 500 µV mean-abs | basic | basic | **strict**（300 µV peak + per-channel kurtosis>10） | basic |
| Stieger2021 占比 | ~52% (23/62 被试) | ~79% (62/62 被试) | ~30%（按 segment 子采样） | 0% | 100% |
| 总数据量 | 30,282 segments | 78,232 segments | ~46K segments | **4,937 segments**（Cho 1,135 + Ofner 492 + Schirr 3,310；strict filter 后） | 67,068 segments |
| 最终 loss | 0.006055 | 0.003714 (−39%) | 0.004193 | **0.001914**（最低） | 0.003108 |
| 数值精度 | FP16 AMP | FP16 AMP | FP16 AMP | FP16 AMP | FP16 AMP |
| 训练时间 | ~48 分钟 | ~4.5 小时 | ~2.2h + 2.2h | ~1.7 小时 | 6 小时 18 分钟 |

> **V4 / V5 设计动机与 caveat**：V4 与 V5 是针对 §3.6 中"DAPT 负迁移"三个候选机制（域错配 / Stieger 占主导 / 通道数异质）的两次 surgical fix。V4 同时改"数据组成"与"过滤强度"——选取与下游 finger MI 域最接近的 3 个公开数据集（Cho2017 = 双手抓握 MI；Ofner2017 = 手腕 MI；Schirrmeister2017 = 上肢 MI），并应用 strict filter（300 µV peak + per-channel kurtosis>10 双阈 AND）替代 basic 500 µV mean-abs；V5 仅改通道几何——单源 Stieger 60-ch，其余配置与 V2/V3 一致。V4 的 strict filter 实现入口为 [scripts/pretraining/preprocess_mi_datasets.py:filter_segments_strict()](../../scripts/pretraining/preprocess_mi_datasets.py)；保留率 Cho 47% / Ofner 33% / Schirr 100%。**已知 caveat**：(i) V4 同时改了数据组成与过滤强度，二者效应未隔离（V6 = V2 数据组成 + strict filter 未运行）；(ii) V5 的 Stieger 仅过 basic filter（重处理 ~25 h wall-clock 妥协），与 V4 三数据集 strict filter 形成 scope 不一致；(iii) V4 的 Schirrmeister 占采样权重 67%（4,937 段中 3,310 段），意味着"3-set 域对齐"实质上偏向 Schirrmeister 主导（128ch 通道匹配下游，但属 motor execution 而非纯 imagery）。

> **超参数与原始 CBraMod 预训练的差异说明**：原始 CBraMod 在 TUEG 上的预训练使用 base LR=5e-4、纯 CosineAnnealingLR、40 epoch（论文 §3.1 + Appendix B）；本研究 DAPT 三个版本均采用 base LR=5e-5（**较原始预训练降低 10×**），这是 BERT-style 继续预训练的标准实践——TUEG 权重已位于稳定盆地，过大学习率会瞬间破坏已学表征。其余共享配置（AdamW、weight_decay=0.05、effective batch size=128、mask ratio=50%、MSE 重建损失、grad clip norm=1.0、模型架构 d_model=200/n_layer=12/nhead=8/ff=800、30s @ 200Hz 输入、1s patch）与原始预训练保持一致。AMP/FP16 为本研究的工程加速选择，不影响优化方向。

> **V1/V2 数据量差异说明**：V1 使用了部分下载的外部数据集（Stieger2021 仅 23/62 被试，15,959 segments；Schirrmeister2017 仅 5/14 被试），总计 30,282 segments。V2 完成了两个大型数据集的全量下载（Stieger2021: 62/62 被试，61,526 segments；Schirrmeister2017: 14/14 被试，3,310 segments），总计 78,232 segments。其中 Stieger2021 的增量约占数据量差异的 94%。其余 8 个外部数据集在两版中均为完整使用。因此，V1 和 V2 之间不仅训练配置不同（LR 调度、epoch 数），**数据组成也不同**，下游结果差异不可归因于单一因素。

> **V3 设计动机与 caveat**：V3 旨在分离 V2 中"Stieger2021 占比 ~79%"这一最可能驱动负迁移加剧的因子——保持 V2 的全部其他设置不变，仅将 Stieger2021 子集按 segment 维度子采样到 ~30%（其余 9 个外部数据集全量保留），目标占比与 V2 形成显著反差。V3 训练分两阶段：(i) 初次训练 15 epoch（best at epoch 15, loss 0.005037），(ii) 在该 best checkpoint 基础上做 continue training 12 epoch，**采用 warm-restart-from-weights 策略**（仅恢复模型权重，不恢复 optimizer 与 LR scheduler 状态；初始 LR 重置为 5e-5），共计 27 epoch，best at epoch 22, loss 0.004193。两阶段拼接后的 loss 单调下降无明显反弹，但因优化器状态在阶段 ii 重置，V3 与 V1/V2（单阶段训练）的"训练充分度"不严格可比；V3 vs V2 的下游差异应被视为"Stieger 占比降低 + warm-restart 后续训练"的混合效应，而非单纯的 Stieger 占比变量结果。

### 2.8 评估协议

**分类性能**：所有模型在测试集上按被试计算准确率，报告 21 名（或 16 名）被试的均值 ± 标准差。统计显著性采用配对 t 检验（paired t-test）评估：每个被试按其对应于两条件下的 trial-level majority-vote 准确率构成一对，使用 `scipy.stats.ttest_rel` 在被试 ID 对齐后的两条件准确率向量上做单次双尾检验，无多重比较校正（结果章节中各表的 p value 均为该独立检验的原始值）。当对照两组的被试集合不完全相同（例如 §3.2 vs §3.4 跨 N=16/N=21 的对照），按被试 ID 取交集后再做配对。计算实现位于 [scripts/paper/compute_paper_statistics.py:117-139](../../scripts/paper/compute_paper_statistics.py#L117-L139) 的 `paired_ttest()`。

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
| CBraMod | **85.15 ± 11.00%** | **69.44 ± 15.42%** |
| EEGNet-16,4 | 78.10 ± 12.61% | 66.81 ± 12.04% |
| Δ (CBraMod − EEGNet) | **+7.05 pp** | **+2.63 pp** |

CBraMod 在两种任务上均优于 EEGNet，二分类差距 +7.05 pp 更为显著。图 1 展示了逐被试对比、准确率分布和配对散点图。

**图 1. 被试内训练 128 通道二分类逐被试对比。** 上方柱状图显示 EEGNet（蓝色半透明，历史数据）与 CBraMod（红色实心）的逐被试准确率；下左箱线图显示准确率分布；下右散点图显示配对对比，多数被试位于对角线上方（CBraMod 更优）。

![图 1. 被试内 128ch 二分类对比](../../results/20260323_2237_combined_imagery_binary.png)

三个值得注意的模式：（1）CBraMod 在 16/21 名被试中优于 EEGNet，但 S04、S05 和 S09 三名被试上 EEGNet 持平或微优（其中 S04、S05 微优，S09 持平），提示预训练表征并非在所有个体上都有效；（2）两种模型的被试间方差均较高（SD > 11 pp），反映了手指级 MI 信号的个体差异性——S09 近乎完美 (99.38%) 而 S20 仅略高于随机 (52.50%/61.25%)；（3）三分类差距仅 +2.63 pp，显著小于二分类的 +7.05 pp，可能因为三分类的更高难度使两种模型都受限于信号质量而非模型容量。

> **数据来源**: CBraMod: `results/20260323_2237_comparison_cache_imagery_binary.json`; EEGNet: `results/20260316_1411_comparison_cache_imagery_binary.json`
> 生成命令: 图 1 由 `uv run python scripts/experiments/run_within_subject_comparison.py --replot 20260323_2237` 重绘（CBraMod 主图；EEGNet 半透明叠加来自 `--replot 20260316_1411`）

> **基线声明**：上述 128 通道被试内结果构成后续所有 XSI-FT（Section 3.3）、纵向扩展实验（Section 3.4）和通道缩减（Section 3.5）的**被试内参考基线**（图中以半透明斜线填充标注）。

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

这一结果揭示了基座模型与从头训练小模型在数据利用效率上的差异：EEGNet 未从跨被试数据池化中观察到显著收益（78.10% 被试内 vs 76.67% 跨被试，−1.43 pp，配对 t 检验 p = 0.456），提示其有限的 ~16K 参数可能难以从异质多被试数据中提取共享表征。相比之下，CBraMod 增益 +5.53 pp（85.15% → 90.68%），说明 TUEG 预训练提供的通用 EEG 先验使模型能有效整合 21 名被试的共享手指运动模式，将跨被试变异转化为泛化能力。

这一差异对实际部署具有启示意义：在当前 21 名被试的样本范围内，CBraMod 从跨被试数据池化中获益显著（+5.53 pp），而 EEGNet 等小模型的改善可能更依赖于增加单个被试的训练数据量（见 Section 3.4）。此结论基于被试内与跨被试的单次比较，数据池化收益的持续性需在更大样本量下验证。图 2b 在一张图上汇总两个模型在 within-subject 与 cross-subject 两种范式下的 mean ± SD、被试个体散点与 paired-t Δ，让该非对称获益直接可视化。

**图 2b. 跨被试 vs 被试内 pooling 增益 forest plot。** 4 个单元格 × 21 名被试散点，最右侧标注 Δ(cross − within) 与 paired-t p value——CBraMod cross-subject 显著高于 within-subject (Δ ≈ +5.53 pp, p < 0.05)，EEGNet 方向反转且不显著 (Δ ≈ −1.43 pp, p = 0.456)。

![图 2b. 跨被试 pooling 增益 forest plot](../figures/cross_subject_pooling_forest.png)

> **数据来源**: CBraMod: `results/20260324_0023_cross_subject_cache_imagery_binary.json`; EEGNet: `results/20260330_0709_cross_subject_cache_imagery_binary.json`
> 生成命令: 图 2 由 `uv run python scripts/experiments/run_cross_subject_comparison.py --replot 20260330_0709` 重绘；图 2b 由 `uv run python scripts/paper/generate_paper_figures.py --figure cross_subject_pooling_forest` 生成

> **基线声明**：上述 128 通道跨被试结果构成后续所有 XSI-FT 实验（Section 3.3）和通道缩减实验（Section 3.5）的**跨被试参考基线**（图中以 "128ch Baseline" 点状填充标注）。

### 3.3 跨被试初始化的逐被试微调（XSI-FT，128 通道）

为定量评估"先用群体数据训得初始化、再针对每位被试微调"这一两阶段策略相对于单一阶段范式（被试内训练 §3.1、跨被试 pooling §3.2）的边际增益，我们在此正式引入 **Cross-Subject-Initialized Per-Subject Fine-Tuning（跨被试初始化的逐被试微调，下文统一简称 XSI-FT）**：

1. 沿用 §3.2 的 cross-subject 训练流程在 21 名被试上得到 pooled checkpoint；
2. 对每名被试，**以该 cross-subject checkpoint 作为初始权重**，在该被试的 §2.3 标准 train/val split 上做 fine-tune（HPO 后超参数；不冻结层）；
3. 在该被试的 held-out test session 上评估，得到逐被试准确率，群体上 21 人聚合。

> **该机制在 BCI 文献中已知，并非本研究方法学新颖性**。XSI-FT 对应 Lotte et al. 2018 [18] (J. Neural Eng. 综述) 中"subject-adaptive transfer learning"分类的离线版本；同时也是 Pan & Yang 2010 [25] 提出的 inductive transfer 框架在 EEG 上的具体 instance；机制层面与 Ding et al. [3] 的 same-day finetune 同构（仅 finetune 时机不同——[3] 为在线 same-day 增量更新，本研究为离线 held-out session 评估）。本研究将"cross-subject pretrain → per-subject finetune"命名为 XSI-FT 仅作为本论文实验记号便利；**本研究的方法学贡献限于在 finger-MI 数据 + EEG foundation model (CBraMod) 设置下系统量化它的边际收益与饱和条件**（§3.3 标准 split / §3.4.4 extra sessions / §3.5.4 缩减通道下三种维度，均在本节及对应章节展开）。

XSI-FT 与 §3.2 cross-subject 的区别在于（3）每名被试拿到独立模型而非共享单一 pooled 模型；与 §3.1 within-subject 的区别在于（1）初始权重来自群体而非随机初始化。这一两阶段定义与 §3.4.4 的"XSI-FT extra sessions"（§2.3.1 同名）机制相同，只是适用数据范围不同（§3.3 限于标准 train split，§3.4.4 允许逐 session 累积）。表 11 总结了 128 通道 XSI-FT 结果。

**表 11. XSI-FT 效果（128 通道，N = 21）。**

| 模型 | 任务 | 范式 | Mean ± SD | Δ vs. 跨被试 |
|------|------|------|-----------|-------------|
| CBraMod | 二分类 | 跨被试 | 90.68 ± 9.31% | — |
| CBraMod | 二分类 | XSI-FT | 90.12 ± 8.98% | **−0.56 pp** |
| CBraMod | 三分类 | 跨被试 | 74.88 ± 14.03% | — |
| CBraMod | 三分类 | XSI-FT | 75.04 ± 13.97% | **+0.16 pp** |
| EEGNet | 二分类 | 跨被试 | 76.67 ± 11.95% | — |
| EEGNet | 二分类 | XSI-FT | **82.05 ± 11.28%** | **+5.38 pp** |
| EEGNet | 三分类 | 跨被试 | 61.23 ± 11.28% | — |
| EEGNet | 三分类 | XSI-FT | **66.33 ± 12.96%** | **+5.10 pp** |

在 128 通道条件下，CBraMod 的 XSI-FT 在两种任务上均未产生统计显著的收益（二分类 Δ = −0.56 pp，配对 t 检验 p = 0.189；三分类 Δ = +0.20 pp，p = 0.261）。EEGNet 的反应方向相反：XSI-FT 在二分类与三分类上分别提供 **+5.38 pp**（配对 t 检验 p = 0.004）和 **+5.10 pp**（p = 0.001）的统计显著正增益。两种模型在同一 XSI-FT 协议下方向不同（CBraMod 非显著、EEGNet 双 task 均 p < 0.01），是一个值得专门讨论的非对称（见下方解读）。图 6 和图 6b 分别展示了二分类和三分类的 XSI-FT 逐被试对比。

**图 6. 128 通道 XSI-FT 对比（二分类，6-way）。** 同时展示被试内（EEGNet + CBraMod）、跨被试（EEGNet + CBraMod）和 XSI-FT（EEGNet + CBraMod）的逐被试结果，覆盖表 11 全部 6 行。

![图 6. XSI-FT 对比（二分类）](../../results/20260329_0507_transfer_combined_imagery_binary.png)

**图 6b. 128 通道 XSI-FT 对比（三分类，6-way）。** 同时展示被试内（EEGNet + CBraMod）、跨被试（EEGNet + CBraMod）和 XSI-FT（EEGNet + CBraMod）的逐被试结果，覆盖表 11 全部 6 行。

![图 6b. XSI-FT 对比（三分类）](../../results/20260329_0448_transfer_combined_imagery_ternary.png)

CBraMod 在 128 通道条件下 XSI-FT 两个任务上均无统计显著收益，表明其跨被试模型已在表征层面饱和。然而 EEGNet 的方向相反——它在跨被试 pooling 下方向性受损（§3.2 二分类 −1.43 pp），但在 XSI-FT 下反而获得 +5.38/+5.10 pp 的统计显著正增益（两 task 均 p < 0.01）。这种非对称指向一个具体机制：EEGNet 容量太小（~16K 参数）无法吸收 21 名被试的异质 cross-subject 分布，被迫学习"被试均值附近"的折衷表征；当 XSI-FT 阶段把模型暴露给单被试数据后，少数已有的 weights 被重新校准到该被试，反而恢复了被试-特异性 spatial filter。CBraMod 的 ~30M 参数则在 cross-subject 阶段已成功容纳了多被试变异，单被试 fine-tune 没有进一步信息可学。换言之，**XSI-FT 是不是必要，由 cross-subject 是否对该模型容量"过载"决定，而不是由模型大小本身决定**。

需要指出的是，EEGNet XSI-FT 的效应量（+5.38/+5.10 pp，两 task 均 p < 0.01）虽统计显著，但仍小于其被试内训练（§3.1，binary 78.10%）相对于 cross-subject pooling 的差距（~+1.4 pp 内），且 EEGNet 二分类被试内 78.10% 仍未追上 EEGNet XSI-FT 的 82.05%——XSI-FT 提供的"全群体先验"对 EEGNet 仍是有用的初始化，但 cross-subject pooling 本身对 EEGNet 是次优策略。

CBraMod 的非显著结果还指向一个更宽的假设：XSI-FT 在缩减通道配置下（跨被试模型因空间信息受限而性能下降时）可能提供更大收益。§3.5.4 报告了一项 32ch FDR 对照实验给出方向性支持，但**8ch Band Power 档位下方向反转**（详见 §3.5.4），表明该假设并非简单的"通道越少收益越大"，需要 cross-subject baseline 饱和度的额外条件。

为验证 128ch CBraMod XSI-FT ceiling 不是 TUEG 预训练 backbone 的副作用，§3.7 random-init CBraMod 消融在两种任务上均显示同方向 ceiling：random-init cross→XSI-FT 二分类 Δ = −0.12 pp（86.34% → 86.22%）、三分类 Δ = +0.37 pp（73.06% → 73.43%），与本节 −0.56 / +0.20 pp 模式一致。两条独立证据（pretrained vs from-scratch）共同表明，128ch 下 CBraMod 的 XSI-FT ceiling 由（任务 × cohort × 通道密度）共同决定，而非 TUEG backbone 的过度正则化。

> **数据来源**: 跨被试二分类 `20260324_0023`: `results/20260324_0023_cross_subject_cache_imagery_binary.json`; XSI-FT 二分类 CBraMod `20260329_0507`: `results/20260329_0507_transfer_cache_imagery_binary.json`; 跨被试三分类 `20260324_0109`: `results/20260324_0109_cross_subject_cache_imagery_ternary.json`; XSI-FT 三分类 CBraMod `20260329_0448`: `results/20260329_0448_transfer_cache_imagery_ternary.json`; EEGNet 跨被试三分类 `20260330_0735`: `results/20260330_0735_cross_subject_cache_imagery_ternary.json`; EEGNet XSI-FT 二分类 `20260507_1835`: `results/20260507_1835_transfer_cache_imagery_binary.json`（与 `20260506_2039` 同 recipe 的 N=21 replication，`db.find_baseline_run()` 默认返回的 baseline 候选，详见 §3.7.2 footnote）; EEGNet XSI-FT 三分类 `20260507_1913`: `results/20260507_1913_transfer_cache_imagery_ternary.json`（同上）
> 生成命令: 图 6 / 图 6b 由 `uv run python scripts/paper/generate_paper_figures.py --figure fig6` / `--figure fig6b` 重绘（内部走 `run_transfer_comparison.py --replot 20260329_0507 --merge-cache 20260507_1835 --cache-only` 和 `--replot 20260329_0448 --merge-cache 20260507_1913 --cache-only`，把单模型 cache 合并为 6-way 对比）

#### 3.3.1 Quaternary（仅在补充材料中报告）

Quaternary 四分类（拇指 / 食指 / 中指 / 小指）作为辅助粒度报告，但其数据条件与定位与 binary / ternary 主线**不可等量齐观**——这一区别源自数据集原始实验设计本身而非本研究的额外约束。Ding et al. [3] 的实时机器人控制主线**只**包含 2-class 与 3-class 两种 paradigm，并通过多 session 在线实验配合 same-day fine-tuning 评估性能；4-class 结果仅出现在该论文的 offline decoding 分析中（原文在 offline 阶段同时采集了 ME 与 MI 的四指数据，本研究聚焦运动想象，因此本文 quaternary 数据来源仅为**单个无反馈 offline MI session**，没有对应的 Online_Sess01/02 增量数据）。需要澄清的是，[3] 中参与者的筛选门槛是 **offline ME 与 MI 的 binary 准确率均 ≥ 70%**，而非 quaternary 准确率——offline MI session 在 [3] 中并不充当"quaternary 筛选/校准"角色，本文不应把该 session 描述为"quaternary 校准 session"。

在上述实验设计前提下，本文报告 quaternary 时面临三项实质性数据限制：(i) 单被试可用 trial 数远低于 binary / ternary 主线（无 online 增量数据）；(ii) 该 offline MI session 在 MI 子序列内部位于最早段，被试尚未通过实时反馈适应任务，信号质量相对较差；(iii) 训练 / 验证 / 测试三段都来自同一 session 的时序切片（参 §2.3 quaternary 协议），分布漂移结构与 binary / ternary 的"跨 session 留出"完全不同。综合这三点，本研究**不把 quaternary 结果纳入 §3 主线结论**，全套结果（cross-subject / within-subject / XSI-FT × EEGNet / CBraMod 共 6 个运行）报告于补充材料 Table S7（参 §"补充材料"）；正文以下小节继续以 binary / ternary 为主分析范围。

### 3.4 多 session 纵向数据扩展

#### 3.4.1 被试内二分类（N = 16）

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

**图 7. Extra Sessions 二分类被试内 EEGNet vs CBraMod 对比（N = 16）。** Panel A：16 名被试在四个数据阶段（Baseline → +Sess03 → +Sess04 → +Sess05）的逐人 trajectory（淡线）+ 两模型 mean ± SE 粗实线（蓝 EEGNet / 红 CBraMod），CBraMod 全程在 EEGNet 上方且两模型均呈正向收益。Panel B：每被试 Δ (Sess05 − Baseline) 配对 boxplot，灰线连接同一被试在两模型间的 Δ；EEGNet Δ = +7.34 pp、CBraMod Δ = +6.13 pp（与 Tables 12a / 12b 一致）。生成命令：`uv run python scripts/paper/generate_extra_sessions_comparison.py --task binary`。

![图 7. Extra Sessions 二分类对比](../figures/extra_sessions_binary.png)

**低基线与高基线被试的差异化收益**：

| 被试分组 | N (EEGNet / CBraMod) | EEGNet Δ | CBraMod Δ |
|---------|---------------------|----------|-----------|
| 低基线 (<80%) | 8 / 3 | **+13.12 pp** | **+18.75 pp** |
| 高基线 (>90%) | 5 / 9 | −0.87 pp | +1.46 pp |

> 注：分组阈值 80%/90% 为绝对值，因两模型基线分布不同，各分组样本量有差异。CBraMod 低基线仅含 3 名被试（S06, S10, S16），+18.75 pp 的增益估计受个体差异影响大，应视为方向性趋势而非精确效应量。

低基线被试是额外 session 数据的主要受益者，而高基线组仅有轻微改进甚至停滞，呈现明显天花板效应。

**标准差压缩与观测范围**：CBraMod 的被试间标准差从 10.81% 压缩至 5.98%（−45%），表明额外数据不仅提高平均性能，还改善了跨用户一致性。实际观测范围从 60.62%–99.38%（Baseline）收窄至 74.38%–98.75%（+Sess05），最低单被试准确率从 60.62%（S10）提升至 74.38%（S10），反映了底部用户的显著改善。这一压缩对临床部署尤为重要：BCI 系统的实用化要求不是"最好情况下多好"，而是"最差情况下够不够用"。

> 注：EEGNet 在 S04/S09/S13 等高基线被试上呈现微弱负 Δ（详见 Table S3）；最可能的两种解释是 (i) 首 session 数据偶然较干净造成的 baseline 偏高、(ii) EEGNet ~16K 参数容量难以从更多 session 中提取额外特征。两者均未构成系统性 finding，本文不将其单列为命名 pattern。

从机制层面看，低基线被试获益显著而高基线被试接近天花板，表明额外 session 数据的主要作用是**弥补个体训练数据不足**而非提升模型容量。这与 Section 3.2 的发现形成有趣的对照：EEGNet 未从*其他被试*数据的池化中显著获益（跨被试 −1.43 pp, p = 0.456），但能从*同一被试*的额外 session 中显著获益（+7.34 pp, p = 0.009），提示其瓶颈在于被试间特征异质性而非绝对数据量。

> **数据来源**: EEGNet baseline: `results/20260316_1411_comparison_cache_imagery_binary.json`; CBraMod baseline: `results/20260323_2237_comparison_cache_imagery_binary.json`; within-subject extra sessions run `20260324_2131`: `results/20260324_2131_extra_sessions_cache_imagery_binary.json`

#### 3.4.2 被试内三分类（N = 16）

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

**图 8. Extra Sessions 三分类被试内 EEGNet vs CBraMod 对比（N = 16；S07 缺 sess03 故 trajectory 单线为 N=15）。** Layout 与图 7 相同。Panel A 的 mean ± SE 折线使用 per-stage 全量数据（baseline/sess04/sess05 各 N=16，sess03 N=15）；Panel B 的 paired Δ 使用同时有 baseline + sess05 的 16 名被试。EEGNet Δ = +4.60 pp、CBraMod Δ = +8.55 pp（与 Tables 13a / 13b 一致）。生成命令：`uv run python scripts/paper/generate_extra_sessions_comparison.py --task ternary`。

![图 8. Extra Sessions 三分类对比](../figures/extra_sessions_ternary.png)

> **数据来源**: `results/20260331_0827_extra_sessions_cache_imagery_ternary.json`

注：此处baseline与正常baseline准确率相对较高，原因是在评估时只选择了有额外online-session的被试，而并没有纳入那些无online session被试。而原finger-eeg数据采集的研究人员在选择哪些被试进行进一步的online session数据采集时，可能因为其实际表现而存在偏好。

**Task × paradigm 跨单元格观察**：汇总 §3.1/§3.2/§3.4 的 (model × task × paradigm) 8 个单元格可见两个模式：(i) 在 within-subject 范式下，CBraMod 二分类领先 EEGNet +7.05 pp 但三分类仅 +2.63 pp，提示三分类受任务难度而非模型容量限制；(ii) 在 extra sessions per_session 协议下方向反转——CBraMod 三分类增益 +8.55 pp 反而大于二分类 +6.13 pp，符合 binary 接近天花板的预期。EEGNet 在 ternary 任务的 extra sessions 增益不显著（+4.60 pp, p = 0.166），与其 ~16K 参数容量上限一致。这些方向性观察建议后续在更大 N 下进行 mixed-effects 模型显式拟合 model × task × paradigm 三向交互项；当前样本量 (N = 16/21) 不足以支持正式交互检验。

#### 3.4.3 评估策略一致性

除默认的 per_session 策略外，我们使用 fixed_combined（固定组合测试集）和 fixed_sess02（固定 Sess02 测试集）两种策略进行补充验证。三种策略均确认了额外 session 数据的显著改善效果：

**表 14. 三种评估策略对比摘要（二分类，Baseline → +Sess05 变化量）。**

| 策略 | EEGNet Δ | CBraMod Δ | 说明 |
|------|----------|-----------|------|
| per_session（默认） | +7.34 pp | +6.13 pp | 临床最相关 |
| fixed_combined | +9.96 pp | +8.44 pp | 控制测试集难度 |
| fixed_sess02 | +8.51 pp | +4.38 pp | 最保守估计 |

fixed_combined 策略显示单调递增趋势（消除了测试集难度变化的混淆因素）。fixed_sess02 下 CBraMod 的增益明显小于 EEGNet（+4.38 pp vs +8.51 pp），可能存在两种解释：（1）基座模型对跨 session 时间分布漂移更敏感；（2）天花板效应——CBraMod 的 fixed_sess02 baseline（87.23%）高于 EEGNet（80.51%），更高起点下增益空间本身受限。两种因素可能同时起作用，当前数据不足以区分。详细策略对比分析见 Supplementary。

> **数据来源**: per_session `20260324_2131`: `results/20260324_2131_extra_sessions_cache_imagery_binary.json`; fixed_combined `20260325_0514`: `results/20260325_0514_extra_sessions_cache_fixed_combined_imagery_binary.json`; fixed_sess02 `20260325_1208`: `results/20260325_1208_extra_sessions_cache_fixed_sess02_imagery_binary.json`; 详细分析: `docs/dev_log/experiments/extra_sessions_strategy_comparison.md`

#### 3.4.4 三范式对齐（CBraMod 二分类，N = 16）

为了直接比较额外 session 数据在三种训练范式中的作用，我们将被试内、跨被试和 XSI-FT 结果统一到同一组 16 名具备额外 session 的被试，并使用相同的 per-session 评估协议。跨被试曲线使用 21 名训练被试的 pooled model；XSI-FT 曲线（与 §3.3 相同机制，但每 step 重新读取该 step 的 cross-subject checkpoint 作为初始权重）按 §2.3.1 协议在每 step 对单被试做离线微调。表 15 和图 9 展示了三条轨迹的并列结果。

**表 15. Extra sessions 在三种训练范式下的轨迹对比（CBraMod 二分类，N = 16）。**

| 阶段 | 被试内 | 跨被试（21-subj 训练） | XSI-FT |
|------|--------|------------------------|--------|
| Baseline | 87.23 ± 10.81% | 92.38 ± 8.35% | 87.23 ± 10.81% |
| +Sess03 | 89.14 ± 8.93% | 91.88 ± 6.71% | 89.65 ± 7.09% |
| +Sess04 | 90.94 ± 8.93% | 92.19 ± 6.91% | 91.84 ± 6.91% |
| +Sess05 | **93.36 ± 5.98%** | **93.24 ± 5.81%** | **92.93 ± 6.11%** |
| Δ(BL→S05) | **+6.13 pp** | **+0.86 pp** | **+5.70 pp** |
| paired p | **0.007** | 0.662 | **0.015** |

跨被试模型的起点最高（92.38%），但额外 session 带来的边际收益最小（+0.86 pp, p = 0.662）；相反，被试内重训练和 XSI-FT 都能从新增同被试数据中获得显著改善，最终分别达到 93.36% 和 92.93%。值得注意的是，cross-subject 与 XSI-FT 都不是在“单一同分布增量学习”条件下吸收新增数据：模型既要面对**跨 session 的时间漂移**，又带着**跨被试 pooled 训练形成的群体差异**。这种“跨 session 异质性 + 跨被试异质性”的叠加，会把一部分新增数据的作用消耗在分布对齐上，而不是直接转化为更高的最终准确率。到 +Sess05 时，三条曲线收敛到 92.93%–93.36% 的窄区间，说明在 128 通道条件下，一旦拥有足够的同被试 session，性能上限更可能由数据质量和任务难度决定，而非训练范式本身。

**图 9. Extra Sessions 在三种训练范式下的总览（CBraMod 二分类，N = 16）。** 左图展示四个 step 的均值 ± 标准差轨迹；右图展示 Baseline → +Sess05 的净增益，强调“高 baseline”与“高增益”并不等价。

![图 9. Extra Sessions 三范式总览](../figures/extra_sessions_paradigm_binary.png)

> **数据来源**: within-subject `20260324_2131`: `results/20260324_2131_extra_sessions_cache_imagery_binary.json`; cross-subject `20260326_1409`: `results/20260326_1409_cross_subject_extra_sessions_cache_imagery_binary.json`; XSI-FT `20260329_1357`: `results/20260329_1357_extra_sessions_cache_imagery_binary.json`（由 `run_extra_sessions.py --pretrained-run` 生成，缓存 schema 仍为 `extra_sessions_cache`）
> 生成命令: 图 9 由 `uv run python scripts/paper/generate_paper_figures.py --figure extra_sessions_paradigm` 生成

#### 3.4.5 跨被试 pooled model 的边际收益上限

如果把视角聚焦到 pooled cross-subject 模型本身，extra sessions 的边际收益明显弱于被试内更新。表 15b 将二分类和三分类的跨被试结果压缩为同一摘要：二分类中，EEGNet 完全无收益（81.45% → 81.33%，p = 0.950），CBraMod 也仅小幅上升 +0.86 pp；三分类下 CBraMod 虽有 +3.73 pp 的正增量，但未达显著（p = 0.090）。

**表 15b. Cross-subject extra sessions 的边际收益摘要（N = 16）。**

| 模型 / 任务 | Baseline | +Sess05 | Δ | paired p |
|-------------|----------|----------|---|----------|
| CBraMod Binary | 92.38 ± 8.35% | 93.24 ± 5.81% | +0.86 pp | 0.662 |
| EEGNet Binary | 81.45 ± 10.87% | 81.33 ± 10.16% | −0.12 pp | 0.950 |
| CBraMod Ternary | 80.05 ± 11.46% | 83.78 ± 8.30% | +3.73 pp | 0.090 |

这一模式支持一个更具体的解释：额外 session 数据本身并非“无信息”，而是**其信息主要是被试特异性的**。当模型以单被试为单位更新时（被试内或 XSI-FT），这些新增 trial 会直接推动决策边界向该被试收敛；而在跨被试 pooled 训练中，同一批新增 trial 被稀释进 21 名被试的联合分布，能改善总体表征，却难以显著改变特定个体的最终决策边界。更进一步，cross-subject extra sessions 的收益受限，很可能正是因为模型同时面对两层异质性：一层是不同日期/状态带来的**跨 session 分布漂移**，另一层是不同个体神经模式带来的**跨被试差异**。当这两层异质性叠加时，新增样本首先被用于“校正分布错位”，能留下来提升分类 margin 的有效信息就更少。换言之，cross-subject 预训练更适合作为 XSI-FT 的初始化或高 baseline 起点，而不是吸收 extra sessions 的最终归宿。

> **数据来源**: binary `20260326_1409`: `results/20260326_1409_cross_subject_extra_sessions_cache_imagery_binary.json`; ternary `20260327_0303`: `results/20260327_0303_cross_subject_extra_sessions_cache_imagery_ternary.json`

### 3.5 通道缩减

#### 3.5.1 32 通道配置对比

图 3a 在 BioSemi 128 头皮坐标上对比了五种 32 通道配置的空间分布。可见这五种方法的覆盖模式高度异质：Commercial 配置呈标准 10-20 系统的均匀分布（作为参考布局）；FDR 与 Band Power 形成跨越前-中-外侧的广泛覆盖，但具体选点不同；Attention 与 CSP 则呈现明显的左侧偏侧化（半数以上通道集中在头皮左侧），且更多落在外缘环（lateral / temporal / occipital 边缘）。脑区层面的精确占比详见 Figure S5b。配置之间的成对 Jaccard 系数仅 0.12–0.23（详见 §4.3 与 Figure S6），意味着这些方法在 96 通道补集中选出了大体不重叠的子集——这一空间证据将在 §4.3 中作为"高密 EEG 的体积传导冗余"的关键支撑。

**图 3a. 5 种 32 通道配置的电极空间分布对比。** 在 BioSemi 128 头皮 2D 投影上分别标注 FDR / Band Power / CSP / Attention / Commercial 5 种配置选中的 32 个通道（红点），灰色为未选中通道。Commercial 模拟标准 10-20 布局；其余 4 种为数据驱动方法。

![图 3a. 5 配置电极空间分布](../../results/32_channel/electrode_placements_5configs/grid_all_configs_2d.png)

> **数据来源**: 通道索引 `results/32_channel/channel_selections.json`
> 生成命令: `uv run python scripts/analysis/visualize_electrode_placements.py --configs attention band_power commercial csp fdr --output-dir results/32_channel/electrode_placements_5configs`

图 3b 以分组柱状图展示了五种 32 通道配置在跨被试二分类上的双模型性能对比，表 8 列出精确数值。

**图 3b. 32 通道五种配置双模型对比（跨被试二分类，N = 21）。** 虚线为各模型 128ch 参考性能。

![图 3b. 32ch 五种配置对比](../figures/32ch_comparison.png)

**表 8. 32 通道配置对比（跨被试二分类，N = 21）。** 128ch baseline：CBraMod 90.68%，EEGNet 76.67%。

| 排名 | 方法 | CBraMod Mean ± SD | Δ vs 128ch | EEGNet Mean ± SD | Δ vs 128ch |
|------|------|-------------------|------------|------------------|------------|
| 1 | **FDR** | **87.71 ± 9.18%** | **−2.97 pp** | 74.70 ± 12.46% | −1.97 pp |
| 2 | Band Power | 86.85 ± 9.76% | −3.83 pp | 76.07 ± 11.50% | −0.60 pp |
| 3 | Commercial | 86.10 ± 8.88% | −4.58 pp | 73.54 ± 12.57% | −3.13 pp |
| 4 | Attention | 85.48 ± 9.21% | −5.20 pp | — | — |
| 5 | CSP | 84.94 ± 10.53% | −5.74 pp | 75.00 ± 11.08% | −1.67 pp |

FDR 以 87.71% 领先，保留了 128 通道性能的 **96.7%**（87.71% vs 90.68%），通道缩减代价仅 −2.97 pp；而 CSP 的代价最大（−5.74 pp）。值得注意的是，EEGNet 在多数配置下的通道缩减代价反而更小（−0.60 至 −3.13 pp），可能因为其 128ch baseline 本身较低（76.67%），天花板效应更弱。图 3c 展示了 FDR 32 通道配置的逐被试对比。

**图 3c. 32 通道 FDR 配置跨被试二分类逐被试对比。** 同时叠加 128ch 跨被试基线（EEGNet + CBraMod，点状填充），显示 32ch FDR 在绝大多数被试上接近 128ch 性能。

![图 3c. 32ch FDR 跨被试对比](../../results/32_channel/fdr/20260330_0836_cross-subject_combined_imagery_binary.png)

五种方法之间的差异仅 2.77 pp（84.94%–87.71%），反映了高密度 EEG 中体积传导导致的信息冗余。Figure S5b 进一步以脑区分布证实 5 种方法虽数值接近但空间分布迥异。这一发现具有重要的实践意义：在 32 通道级别，通道选择方法的选择相对不那么关键——即使使用简单的商用布局（Commercial, 86.10%）也能获得接近最优数据驱动方法的性能。然而，这种"方法不敏感"的特性会随着通道数的进一步减少而急剧消失（见 Section 3.5.2）。

值得注意的是，Commercial 配置的标准差最低（8.88%），表明标准 10-20 布局在跨被试一致性上具有优势——这可能因为其电极分布更均匀，不依赖于特定被试群体的统计特征。

> **数据来源**: `results/32_channel/{fdr,attention,csp,band_power,commercial}/20260330_*_cross_subject_cache_imagery_binary.json`
> 生成命令: 图 3b 由 `uv run python scripts/paper/generate_paper_figures.py --figure 32ch_comparison` 生成；图 3c 由 `uv run python scripts/experiments/run_cross_subject_comparison.py --replot 20260330_0836` 重绘

#### 3.5.2 通道缩放分析（128 → 4）

表 9 展示了 CBraMod 从 128 到 4 通道的性能降解轨迹（binary 与 ternary 平行）。Ternary baseline 取 128ch CBraMod cross-subject `20260324_0109`，mean = **74.88%**（run-to-run range 跨 6 个 21 名被试完整运行为 73.06–75.50%，详见 §3.2）。

**表 9. CBraMod 通道缩放分析（跨被试 binary / ternary 双 task）。** **Δ 列基准依"过渡"列首端而定（双基准约定）**：`128 → X` 行的 Δ 相对 **128ch 单点 baseline**（binary 90.68% / ternary 74.88%，即表头括注值）；`X → Y` 行（如 `32 → 8`、`32 → 4`、`16 → 8`）的 Δ 相对 **源端 X 通道档同一方法的该 task 准确率**——`32 → ` 行相对 32ch best（binary 87.71% / ternary 72.20%），`16 → ` 行相对该方法的 16ch 值（如 `16 → 8 (FDR)` 相对 16ch FDR 84.26%）。因此表头括注 90.68% / 74.88% 仅对 `128 →` 行成立，`X → Y` 行的 Δ 不可直接与 90.68% / 74.88% 对照（其 (绝对准确率) 括注值才是跨行可比的统一锚点）。64ch 行新增 attention / band_power / csp / negative_control 4 method（数据来源：2026-05-11 21-cell 矩阵闭合，详见 [docs/handoffs/2026-05-11_reduced_channel_matrix_closure.md](../../docs/handoffs/2026-05-11_reduced_channel_matrix_closure.md)）。**16ch 行（2026-05-13 transition-point sweep）**填补了 32→8ch 之间的中间档位，揭示 method-agnostic 区间在 32ch 以下开始崩溃（详见正文）。

| 过渡 | 通道缩减 | Δ binary (vs 90.68%) | Δ ternary (vs 74.88%) | 说明 |
|------|---------|---:|---:|------|
| 128 → 64 (FDR) | −50% | **−1.22 pp** (89.46%) | **+0.24 pp** (75.12%) | binary 中间档位；ternary 在 128ch run-to-run noise 内 |
| 128 → 64 (Band Power) | −50% | −2.79 pp (87.89%) | +0.14 pp (75.02%) | 5 method 在 64ch 上的 spread 仅 3.24 / 2.09 pp |
| 128 → 64 (Attention) | −50% | −3.15 pp (87.53%) | −1.07 pp (73.81%) | |
| 128 → 64 (CSP) | −50% | −4.46 pp (86.22%) | −1.53 pp (73.35%) | |
| 128 → 64 (negative_control) | −50% | −2.11 pp (88.57%) | +0.56 pp (75.44%) | 与 64ch 最优数据驱动方法在 ±0.32 pp 内（见 §3.5.3 caveat） |
| 128 → 61 (standard_1010) | −52% | −1.13 pp (89.55%) | **−1.83 pp** (76.71%) ★ ternary 新增 | 标准 10-10 (Yazıcı et al. 2025 [26] 文献参考点)；binary 来自 `20260330_1213`，ternary 新增 `20260513_1938` |
| 61 → 32 (FDR, best) | −48% | −1.84 pp | — | FDR 32ch ≈ 61ch |
| 128 → 16 (Band Power) | −88% | **−5.44 pp** (85.24%) | −7.28 pp (67.60%) | **16ch binary 最优**；2.80 pp 优于 16ch neg_ctrl，5-entry spread 8.69 pp |
| 128 → 16 (FDR) | −88% | −6.42 pp (84.26%) | **−5.57 pp** (69.31%) | **16ch ternary 最优**；与 BP top-2 binary 差 0.98 pp |
| 128 → 16 (CSP) | −88% | −7.32 pp (83.36%) | −12.04 pp (62.84%) | binary 第三；ternary 跌至 neg_ctrl 之下 |
| 128 → 16 (negative_control) | −88% | −9.07 pp (81.61%) | −10.51 pp (64.37%) | 仍超随机；与 16ch BP 在 binary 上差 3.63 pp |
| 128 → 16 (Attention) | −88% | −14.13 pp (76.55%) | −13.21 pp (61.67%) | **16ch 双 task 最差**；Attention top-K 外推失效已在 16ch 显现 |
| 16 → 8 (Band Power) | −50% | −1.19 pp (84.05%) | −1.27 pp (66.33%) | 16→8 BP 几乎平滑过渡 |
| 16 → 8 (FDR) | −50% | −7.83 pp (76.43%) | −4.43 pp (64.88%) | FDR 在 16→8 衰退显著（binary）|
| 32 → 8 (Band Power, best) | −75% | −3.66 pp (84.05%) | −5.87 pp (66.33%) | Band Power 仍为 8ch ternary 最优 |
| 32 → 8 (CSP) | −75% | −5.98 pp (81.73%) | −8.35 pp (61.77%) | binary 第二；ternary 第三 |
| 32 → 8 (FDR) | −75% | −11.28 pp (76.43%) | −5.91 pp (64.88%) | binary 大幅衰退；ternary 反而是第二 |
| 32 → 8 (Attention) | −75% | −19.29 pp (68.42%) | −10.38 pp (59.50%) | 双 task 衰退最严重 |
| 32 → 8 (negative_control) | −75% | −11.74 pp (76.34%) | −10.83 pp (59.05%) | 8ch ternary 最低（低于 attention 0.45 pp） |
| 32 → 4 (Band Power top-4) | −88% | **−8.96 pp** (78.75%) | **−12.20 pp** (60.67%) | **双 task 均最强**：保留 86.8% binary / 81.0% ternary |
| 32 → 4 (FDR top-4) | −88% | −25.63 pp (62.08%) | −24.66 pp (46.05%) | binary 低于负控制；ternary 亦低 |
| 32 → 4 (CSP top-4) | −88% | −20.72 pp (66.99%) | −22.59 pp (47.62%) | ≈ binary 负控制水平 |
| 32 → 4 (Attention top-4) | −88% | −33.01 pp (54.70%) | −30.66 pp (41.55%) | 双 task 均近随机 |
| 32 → 4 (负控制) | −88% | −20.06 pp (67.65%) | −18.85 pp (53.37%) | 双 task 上仍 ≈ FDR/CSP top-4 |
| 32 → 4 (FDR∩Att, outlier) | −88% | −4.97 pp (82.71%) | — | 交集通道，favorable outlier（binary 仅） |

图 3d 以分组柱状的 2×5 grid panel 形式直观呈现 reduced-channel × method × task 矩阵，让 method × channel × task 三向交互一次可视。

**图 3d. Reduced-channel × method × task 矩阵全景（cross-subject CBraMod, N = 21）。** Row：binary（panel A-E）/ ternary（panel F-J）；col：4 / 8 / 32 / 61 / 64 ch（每 task 5 列，其中 61ch 为 standard 10-10 单配置列）。每 panel 内 5 method 分组柱：FDR（红）/ Attention（蓝）/ Band Power（绿）/ CSP（橙）/ negative_control（灰）；4ch 列额外含 FDR∩Att 交集柱；柱高为 mean cross-subject accuracy，error bar 为 subject 间 std。横虚线为 128ch CBraMod cross-subject baseline（binary 90.68% / ternary 74.88%）。**注：本图当前为 4/8/32/61/64ch 版本，尚未纳入 2026-05-13 新增的 16ch 列（5×5×2 = 50 cell 中的 16ch × 5 method × 2 task = 10 cell）；16ch 数据以表 9 为权威来源，图 3d 的 16ch 列待图表重生成后补齐。** **核心可视化论断**：(i) panel E / J（64ch）柱高接近 128ch baseline 虚线、5 method 之间几乎齐平 — "32ch+ method-agnostic" 视觉证据；(ii) panel A / F（4ch）BP 绿柱孤立突出于其他 4 个 method — "低通道下 BP 单独保持判别力"；(iii) panel C / H（32ch）灰柱（neg_ctrl）与 4 数据驱动 method 柱高在 ±0.32 pp 内不可区分 — 体积传导冗余的强证据（见 §3.5.3 / §4.3）。Panel 注释数字 = mean accuracy 1 位小数。

![图 3d. reduced-channel × method × task 矩阵（4/8/32/61/64ch，16ch 列待补）](../figures/reduced_channel_40cell_grid.png)

> **数据来源**：alias `reduced_{N}_{method}_{task}`（N ∈ {4,8,32,64}, method ∈ 5, task ∈ {binary, ternary}）加 `standard_1010_61_cross_{task}` 注册在 [paper/run_registry.yaml](../../paper/run_registry.yaml)；完整 run_tag → mean_acc 映射见 [docs/handoffs/2026-05-11_reduced_channel_matrix_closure.md §40-cell 矩阵](../../docs/handoffs/2026-05-11_reduced_channel_matrix_closure.md#40-cell-矩阵-cross-subject--cbramod--n21)。**当前 PNG 文件名沿用 `reduced_channel_40cell_grid.png`，但实际渲染为 4/8/32/61/64ch 列；16ch 列（表 9 已含）尚未进入本图，以表 9 为权威。**
> 生成命令：`uv run python scripts/paper/generate_paper_figures.py --figure reduced_channel_40cell_grid`

图 4 以曲线形式直观呈现了这一非线性降解过程。

**图 4. 通道缩放曲线：CBraMod 跨被试二分类准确率随通道数的变化。** 红色实线为各通道数下最优配置的包络线；虚线追踪各通道选择方法在不同通道数下的表现。绿色区域标示 32 通道部署区间。× 标记为 4 通道负控制。误差线为被试间标准差。

![图 4. 通道缩放曲线](../figures/channel_scaling_curve.png)

图 4 的关键观察是**通道选择方法的最优排序随通道数发生翻转，且该翻转在 binary / ternary 上的具体形式不同**。Binary：32ch 级别 FDR 以 87.71% 领先（五种方法差距仅 2.77 pp）；但到 8ch 级别，**Band Power 以 84.05% 大幅反超 FDR 的 76.43%**（+7.62 pp），CSP (81.73%) 亦优于 FDR；推进到 4ch 时翻转进一步极化——Band Power 仍保持 78.75%（远高于负控制 67.65%），而 FDR/CSP/Attention 均跌至负控制水平或以下。Ternary：32ch / 64ch 上 leading method 不再是 FDR——32ch ternary 上 Band Power 72.20% 居 4 数据驱动方法之首（与 negative_control 72.38% 在 ±0.18 pp 内 indistinguishable），64ch ternary 上 FDR 75.12% 居首（与 negative_control 75.44% 在 ±0.32 pp 内 indistinguishable）；8ch / 4ch ternary 上 Band Power 重新成为唯一 method（8ch BP 66.33% > FDR 64.88% > CSP 61.77%；4ch BP 60.67% 单独超越 negative_control 53.37% 达 +7.30 pp）。**Band Power 在 4 个通道档 × 2 task = 8 cell 上从不是 4 数据驱动方法的最差者**——这是本数据集下最稳健的横向观察。图 4b 以 slope chart 形式直观呈现 binary 翻转。本研究在四个通道档（64 / 32 / 8 / 4ch）× 双 task 的同向观察支撑了"低通道下方法选择敏感度上升"这一现象，但我们刻意避免把它升级为"FDR 与 BP 的相对排序应外推到其他数据集 / 其他任务"这种跨数据集方法论命题——单一 21 人 cohort、单一 MI 任务粒度的样本不足以支持这一升级。

**图 4b. 32ch → 8ch → 4ch 通道选择方法排序翻转。** Slope chart：4 种数据驱动方法（FDR/Band Power/CSP/Attention）在三档通道数下的 cross-subject CBraMod 准确率，每档位标注当前 ranking。32ch 第一名 FDR 在 8ch 跌至第三、Band Power 反超；4ch 时 Band Power 单独保持在负控制（虚线）之上，FDR/CSP/Attention 均跌至或低于负控制。4ch BP top-4 (`20260505_2308`, 78.75%) 与 CSP top-4 (`20260505_2246`, 66.99%) 已纳入；Slope chart 数值与表 9、§3.5.3 表 10 一致。

![图 4b. 通道选择方法排序翻转](../figures/channel_method_ranking_flip.png)

最优配置包络线（红色实线，按"每档最优方法"取）呈现**渐进降解模式**而非原假设的两阶段陡降。**Binary 包络线**：90.68% (128ch) → 89.46% (64ch FDR) → 87.71% (32ch FDR) → 84.05% (8ch BP) → 78.75% (4ch BP)，每减半通道损失约 1.5–5 pp，且 4ch 不再像之前认为的那样进入"全部失效"区间——**前提是选用 Band Power 方法**。**Ternary 包络线**：74.88% (128ch) → 75.12% (64ch FDR，与 128ch baseline 在 run-to-run noise 内一致) → 72.20% (32ch BP) → 66.33% (8ch BP) → 60.67% (4ch BP)；ternary 上 32ch 与 128ch 的差距比 binary 更大（2.68 pp vs 2.97 pp），但 32ch+ 段的方法 leadership 已从 FDR 切换到 BP。原 v2 草稿中的"两阶段（平坦区 + 陡降区）"模型基于 4ch FDR/Attention top-4 数据点；引入 4ch BP 后，包络线整体向上平移，原"陡降区"消失，**这一观察现已通过 ternary 维度独立复现**。

降解的严重程度**高度依赖通道选择方法**。以 32→8ch 过渡为例（binary）：Band Power 仅下降 2.80 pp（86.85→84.05%），而 Attention 下降 17.06 pp（85.48→68.42%）——同一通道缩减幅度下，方法选择导致了 6 倍的性能差异。在 32→4ch 过渡上方法依赖更极端：Band Power 仅下降 8.10 pp（86.85→78.75%），而 Attention 下降 30.78 pp（85.48→54.70%）——~4 倍的方法依赖差异。Ternary 上同向但量级更温和：32→8ch (BP) 下降 5.87 pp、32→8ch (Attention) 下降 12.03 pp；32→4ch (BP) 下降 11.53 pp、32→4ch (Attention) 下降 30.00 pp。在本数据集与本任务范围内，**8 通道乃至 4 通道仍可作为可行的部署方案**——但仅当选用 Band Power 这一具体方法时；其他方法（FDR / CSP / Attention）在这两档上均显著退化。Band Power 在 64 / 32 / 8 / 4ch × binary / ternary 共 8 个 cell 上从未是 4 数据驱动方法的最差者，是本数据集中最稳健的跨档位 / 跨 task 横向观察；但我们仍不把"Band Power 优于其他方法"延伸为通用规则，仅作为本研究观察到的、可被未来工作证伪的具体配置推荐。

64ch FDR (89.46%) 落在 32ch FDR (87.71%) 与 128ch (90.68%) 之间且接近 61ch (89.55%)，进一步弱化了"32ch 是 sweet spot"的强主张：从 32ch 起每翻一倍通道，binary 性能增益依次为 +1.75 pp (32→64ch)、+0.09 pp (64→61ch，几乎重合)、+1.13 pp (61→128ch)。也就是说 64ch 相对 32ch 仍有 ~1.7 pp 的边际 binary 收益，但相对 128ch 已经只差 ~1.2 pp——**32→64ch 之间存在 ~一半的"剩余增益"**，与"边际增益减弱"框架一致但反对"32ch 已饱和"的强表述。Ternary 维度上 32→64ch (FDR) 增益更大 (+4.33 pp，从 70.79% 升至 75.12%)，但 64ch 75.12% 与 128ch baseline 74.88% 的差值 (+0.24 pp) 落在 128ch run-to-run noise 范围内（128ch ternary 6 个完整 21 人 run 跨度 73.06–75.50%，详见 §3.2）——故 ternary 上"32→64ch 的剩余增益"主要由 32ch ternary baseline 偏低驱动，而非"64ch 超越 128ch"。**16ch 档位（2026-05-13 sweep）现已纳入**——binary 最优 BP 85.24%（vs 32ch FDR 87.71% 差 −2.47 pp，vs 8ch BP 84.05% 差 +1.19 pp），ternary 最优 FDR 69.31%（vs 32ch BP 72.20% 差 −2.89 pp，vs 8ch BP 66.33% 差 +2.98 pp）；**16→8 BP 几乎平滑 (−1.19 / −1.27 pp)，而 32→16 FDR 衰退 −3.45 pp binary / −1.48 pp ternary**——说明 32→16ch 之间已有 sizable 衰退（binary 上 1/2 翻倍仅多损失 1 pp 已不成立），16→8 反而是相对平滑过渡。整体看 binary 包络线 90.68% → 89.46% (64ch FDR) → 89.55% (61ch) → 87.71% (32ch FDR) → 85.24% (16ch BP) → 84.05% (8ch BP) → 78.75% (4ch BP) 呈现 **6 档渐进降解，每减半通道约损失 1.5–5 pp**。本研究仍未评估 96ch 档位，因此"电极数量 scaling 在 64ch 以上完全饱和"仍属未验证假设。

**64ch 横向方法对比**——本批新增的 5 种 64 通道配置（FDR / Band Power / Attention / CSP / negative_control）在 binary 上范围 86.22–89.46%（5-entry spread 3.24 pp）、ternary 上 73.35–75.44%（5-entry spread 2.09 pp），与 32ch 上 5 entry 之间的 binary 2.77 pp / ternary 2.08 pp 量级一致：**32ch 起方法选择对性能的影响已落到 ≤ 3.24 pp 量级，并在 64ch 上保持**。注意 §3.5.3 末段敏感度表使用的是 4 数据驱动方法（排除 negative_control 控制项）的 max−min spread（64ch ternary 为 1.77 pp、32ch ternary 为 2.08 pp），与本段 5-entry spread 在 ternary 上的 ≤ 0.32 pp 差值来自 negative_control 是否计入 —— 两个 framing 都正确，分别支持"4 method 之间不可区分"和"含控制项在内的 5 entries 全部不可区分"两个等价强度的论断。在 32ch 与 64ch 上的 ternary，5 method 与负控制之间的差异 ≤ 0.32 pp（well within 21 名被试 std ≈ 13 pp），即统计上不可区分；该 task / 通道档组合下"用数据驱动方法选择最优 32 / 64 通道"与"选择未被任何方法选中的 32 / 64 通道"性能在 paired 比较下无显著差异（详见 §3.5.3 末段）。

**16ch 是 method-agnostic 区间的崩溃点**——2026-05-13 新增的 5 种 16 通道配置（FDR / Band Power / Attention / CSP / negative_control）在 binary 上范围 76.55–85.24%（5-entry spread **8.69 pp**）、ternary 上 61.67–69.31%（5-entry spread **7.64 pp**），相比 32ch 的 2.77 / 2.08 pp **跳升约 3–4 倍**，相比 64ch 的 3.24 / 2.09 pp 跳升约 2.7–3.7 倍。同时仍未达到 4ch 上 24.05 / 19.12 pp 的"高度方法依赖"量级，因此 16ch 表现为 **method-sensitive 区间的明确入口**：32ch+ 上"方法选择 ≤ 3 pp" 的 method-agnostic 论断**到 16ch 已经不再成立**。具体 16ch 排序——binary：BP (85.24%) > FDR (84.26%) > CSP (83.36%) > neg_ctrl (81.61%) > Attention (76.55%)；ternary：FDR (69.31%) > BP (67.60%) > neg_ctrl (64.37%) > CSP (62.84%) > Attention (61.67%)。两个方向上的一致信号：(i) Attention 在 16ch 双 task 上均为最差（与 8ch / 4ch 上 Attention 大幅退化的趋势同向，把"Attention top-K 外推失效"现象的临界点从 8ch 提前到 16ch）；(ii) BP 在 binary 上仍然居首（与 8ch / 4ch BP-dominant 一致），但 ternary 上让位于 FDR（与 32ch ternary 上 FDR/BP 几乎并列的局面一致）；(iii) 16ch negative_control binary 81.61% 比 16ch FDR/BP 仅低 2.65–3.63 pp、ternary 64.37% 比 16ch BP 低仅 3.23 pp——**体积传导冗余在 16ch 仍能让"未被任何方法选中"的随机通道接近最优方法**，但与 32ch / 64ch ternary 上 ≤ 0.32 pp 的不可区分相比，差距已经从"统计噪声内"放大到"3–5 pp 量级"。这把"低度 method-overlap 配置 ≈ 数据驱动配置"的等价性论断**严格限制在 ≥ 32ch 通道档**。Band Power 在 5 个通道档 × 2 task = 10 cell 上从不是 4 数据驱动方法的最差者（之前 8 cell 现扩展到 10 cell），是本数据集下跨档位 / 跨 task 横向最稳健的方法学观察。

**4 通道结果的深层解读**：Attention top-4（54.70%）不仅远低于 8ch Band Power（84.05%），甚至**低于负控制**（67.65%）；FDR top-4 (62.08%) 与 CSP top-4 (66.99%) 同样跌至负控制附近——即随机选取未被任何方法选中的通道反而与这些方法持平或略优。这揭示了一个重要的方法论陷阱：**在 128ch 全模型上计算的通道重要性排序不能线性外推到极低通道配置**。CBraMod 在 128ch 上的梯度注意力反映的是通道在*有其他 124 个通道辅助*时的重要性（即条件重要性），而非通道*独立携带*的信息量。当仅保留 top-4 时，这些通道失去了它们在全局空间模式中赖以发挥作用的上下文通道，导致性能崩溃。

唯一的例外是 Band Power top-4 (78.75%)：它依赖的不是"在全模型中的重要性排序"而是"在 mu/beta 频带上独立计算的 ANOVA F 统计量"，本质上是一个**模型无关的频域指标**——其选点机制不需要"还有哪些通道在场"作为上下文，因此天然免疫上述外推失效陷阱。

需要谨慎对待的是这一指标与解剖学位置的关系。Band Power top-4 选出的 4 个通道在 BioSemi 128 layout 中的位置经在线对照官方 `Cap_coords_all.xls` 后整理如下：

| 通道 | 坐标 (x, y, z) mm | 10-10 近似定位 | 与手部 mu/beta ERD 区的关系 |
|------|------------------|-----------------|------------------------------|
| **B28** (idx 59) | (+82, +27, +14) | 介于 FT8 与 FC6 之间，偏 FT8 | 右侧前颞-下额，**不在**经典手部 ERD 强响应带 |
| **C23** (idx 86) | (0, +34, +81) | **FCz**（官方标注） | 中线辅助运动区 / 运动前区，不在 C3/C4 hand knob |
| **D11** (idx 106) | (-68, +28, +47) | 介于 FC5 与 FC3 之间 | 左侧前运动 / SM1 边缘，**部分**重叠手部表征前缘 |
| **D27** (idx 122) | (-68, -28, +47) | 介于 CP5 与 CP3 之间 | 左侧体感后皮层，**最接近**经典右手 MI 对侧 ERD 区 |

> **解剖学论断的修订**：4 个通道中只有 **D27** 真正落在 Pfurtscheller & Lopes da Silva [14] / Neuper et al. [19] 经典手部 mu/beta ERD 强响应带（C3/C4 hand knob 区域）；D11 处于其前运动边缘；C23 位于 SMA / FCz 中线运动前区；B28 (≈FT8) 完全偏离 sensorimotor cortex。**因此"BP 选出的 4 个通道被空间锁定到 sensorimotor 强响应区"这一直觉化论断不成立**——更精确的描述是：BP top-4 在 sensorimotor + premotor + SMA + 一个右前颞外点 之间形成左偏侧化（3/4 在左半球）的分布式覆盖，而非聚焦于 hand knob。

这一分布的具体机制本研究不能确定。一些可能的解释（互不排斥、均未在本数据上验证）包括：(i) 手指级 (finger-level) MI 任务在皮层上相邻手指间距毫米级，C3 单点 ERD 的类别可分性可能不如前运动区 + SMA + 顶后的分布式联合编码——已有 finger-MI 高密 EEG 文献指出该方向；(ii) ANOVA F (rest vs MI) 衡量的是任务态间的功率差异，与"原始 mu/beta 节律最强处"并非同一指标，可能让非 sensorimotor 区域因稳定的 mu 同步或 beta 偶联而进入排序前列；(iii) 跨被试 F 统计量在被试间 ERD 焦点位置波动较大时会向"群体平均后仍稳健的位点"漂移，这未必是 C3/C4。我们不在本研究中尝试区分这三类机制，留作后续工作（详见 §6）。

FDR∩Attention 的 4 个交集通道（82.71%）的高准确率应被视为一个**有利的巧合**（favorable outlier）。其抽样机制如下：32ch FDR 与 32ch Attention 是两个独立选出的 32 通道集合（覆盖 128 中各占 25%）；它们在**128 通道全空间中的随机交集期望大小**为 32×32 / 128 = **8 个通道**，而本研究观察到的实际交集为 **4 个通道**（B32, C8, D7, D19）——比期望值还少一半。换言之，这 4 个通道并非任一方法排序的 top-4，也不是被两个方法"双重共识"的 top 元素，而是在 32+32 集合的相对小交集中 *碰巧落在的* 4 个位置；它们在 FDR 单独排序中位列第 15–30 位、在 Attention 单独排序中亦位列第 15–30 位（远低于各自 top-4）。82.71% 的高准确率因此不源于"两种方法都认为它们最重要"，而源于交集随机性 + 体积传导冗余 + 这 4 个通道在 cohort 上恰好捕获了部分有效空间模式——属于本数据集的偶然性配置，**不可作为系统化方法复制**。Band Power top-4 与 FDR∩Attention 的差距（82.71% − 78.75% = 3.96 pp）说明 outlier 仍略胜一筹，但 Band Power 提供了一个**可复现、单一方法 top-4、不依赖交集运气**的替代路径。图 4 中橙色菱形标注了 FDR∩Attention 这一 outlier。

> **数据来源 — Binary**: 128ch: `results/20260324_0023_cross_subject_cache_imagery_binary.json`; 64ch FDR `20260505_2223`: `results/64_channel/fdr/20260505_2223_cross_subject_cache_imagery_binary.json`; 64ch attention `20260511_1038` / band_power `20260511_1050` / csp `20260511_1111` / negative_control `20260511_1131`: `results/64_channel/{attention,band_power,csp,negative_control}/20260511_*_cross_subject_cache_imagery_binary.json`; 61ch: `results/61_channel/standard_1010/20260330_1213_cross_subject_cache_imagery_binary.json`; 32ch: `results/32_channel/{fdr,band_power,commercial,attention,csp}/20260330_*_cross_subject_cache_imagery_binary.json`; **16ch（2026-05-13 sweep）fdr `20260513_1959` / csp `20260513_2027` / attention `20260513_2048` / band_power `20260513_2108` / negative_control `20260513_2132`: `results/16_channel/{fdr,csp,attention,band_power,negative_control}/20260513_*_cross_subject_cache_imagery_binary.json`**; 8ch: `results/8_channel/{band_power/20260331_1950,csp/20260331_2044,fdr/20260330_1311,attention/20260330_1334}_cross_subject_cache_imagery_binary.json`; 8ch negative_control `20260511_1425`: `results/8_channel/negative_control/20260511_1425_cross_subject_cache_imagery_binary.json`; 4ch BP `20260505_2308`: `results/4_channel/band_power/20260505_2308_cross_subject_cache_imagery_binary.json`; 4ch CSP `20260505_2246`: `results/4_channel/csp/20260505_2246_cross_subject_cache_imagery_binary.json`
>
> **数据来源 — Ternary（2026-05-11 21-cell 矩阵闭合 + 2026-05-13 16ch / 61ch 补齐）**: 128ch baseline `20260324_0109`; **61ch standard_1010 `20260513_1938`: `results/61_channel/standard_1010/20260513_1938_cross_subject_cache_imagery_ternary.json`**; 64ch fdr `20260511_1148` / attention `20260511_1217` / band_power `20260511_1237` / csp `20260511_1256` / negative_control `20260511_1314`; 32ch fdr `20260221_0332` / attention `20260228_2247` / band_power `20260511_1348` / csp `20260511_1404` / negative_control `20260511_1757`; **16ch fdr `20260513_2146` / csp `20260513_2227` / attention `20260513_2241` / band_power `20260513_2319` / negative_control `20260513_2343`: `results/16_channel/{fdr,csp,attention,band_power,negative_control}/20260513_*_cross_subject_cache_imagery_ternary.json`**; 8ch fdr `20260511_1439` / attention `20260302_2140` / band_power `20260511_1508` / csp `20260511_1539` / negative_control `20260511_1600`; 4ch fdr `20260511_1618` / attention `20260511_1642` / band_power `20260511_1655` / csp `20260511_1731` / negative_control `20260310_0054`; 完整 run_tag → mean_acc 映射见 [docs/handoffs/2026-05-11_reduced_channel_matrix_closure.md §40-cell 矩阵](../../docs/handoffs/2026-05-11_reduced_channel_matrix_closure.md#40-cell-矩阵-cross-subject--cbramod--n21) + [docs/handoffs/2026-05-14_16ch_transition_point.md](../../docs/handoffs/2026-05-14_16ch_transition_point.md)（16ch / 61ch ternary 补齐记录）。
> 生成命令: 图 4 由 `uv run python scripts/paper/generate_paper_figures.py --figure channel_scaling` 生成；图 4b 由 `uv run python scripts/paper/generate_paper_figures.py --figure channel_ranking_flip` 生成；ternary 维度的对应可视化见 §3.5.2 图 3d (`--figure reduced_channel_40cell_grid`)，当前渲染 4/8/32/61/64ch 矩阵（16ch 列待补，以表 9 为权威）。

#### 3.5.3 控制实验

为排除数据泄露解释并验证通道选择的有效性，我们在 4 通道级别进行了控制实验。

**表 10. 4 通道控制实验结果（跨被试二分类，N = 21）。**

| 条件 | 通道来源 | CBraMod Mean ± SD | vs 负控制 (67.65%) |
|------|---------|-------------------|---------------------|
| FDR ∩ Attention（outlier） | 32ch FDR 与 Attention 交集 | 82.71 ± 13.84% | **+15.06 pp** |
| **Band Power top-4** | mu/beta ANOVA F 统计量前 4 | **78.75 ± 10.36%** | **+11.10 pp** |
| 负控制 | 所有方法均未选中的通道 | 67.65 ± 9.46% | — |
| CSP top-4 | 32ch CSP 排序前 4 | 66.99 ± 8.99% | −0.66 pp |
| FDR top-4 | 32ch FDR 排序前 4 | 62.08 ± 8.81% | −5.57 pp |
| Attention top-4 | 32ch Attention 排序前 4 | **54.70 ± 8.20%** | −12.95 pp |

> **重要说明**：FDR∩Attention 的 4 个通道（B32, C8, D7, D19；空间位置见 Figure S6b）并非任何单一方法排序的 top-4，而是两个 32 通道集合的相对小交集（128 通道全空间中 32×32/128 = 8 的随机交集期望，本研究实际交集为 4）——它们在各自单方法排序中仅位于第 15–30 位。82.71% 的高准确率应被视为一个**有利的巧合**（favorable outlier）：这是从相对小的交集中"碰巧"落到的 4 个位置，并非"被两种方法共同认定为最重要"的强一致选择，因而**不能作为系统化方法复制**——详细抽样机制讨论见 §3.5.2 末尾段。**Band Power top-4** 与 outlier 不同，它是单一方法 top-4 的标准化输出（mu/beta 频带 ANOVA F 统计量前 4 通道：**B28, C23, D11, D27**，详见 `results/4_channel/channel_selections.json`；解剖位置详见 §3.5.2），是可复现的系统化选取——下方"4 通道是否可用"的部署判断主要依赖 Band Power 这一可复现路径，而非 FDR∩Attention 这一 outlier。

修订后的"标准方法在 4ch 是否失效"图景较此前更细致：（1）**模型驱动方法（Attention top-4）和全局判别方法（FDR top-4）确实失效**——均显著低于负控制；（2）**空间滤波方法（CSP top-4）几乎与负控制持平**（−0.66 pp）——意味着 CSP 选出的"最重要"通道与"未被任何方法选中"的通道在 4ch 极端约束下信息量等价；（3）**频域物理动机方法（Band Power top-4）显著超越负控制**（+11.10 pp）——保留了显著的判别能力。这与原"4ch 标准方法均失效"的笼统结论不同：4ch 失效的是 conditional importance 类方法（在全模型上下文中重要 ≠ 独立携带信息），但物理动机直接锚定的频域方法（mu/beta ERD 是手指 MI 的标志）仍然有效。

图 5 展示了 4ch 四种关键配置的逐被试对比：FDR∩Attention outlier、负控制、Band Power top-4 (`20260505_2308`)、CSP top-4 (`20260505_2246`)。

**图 5. 4 通道控制实验：四种配置逐被试对比。** 四个子图分别为 FDR∩Att outlier、负控制、Band Power top-4、CSP top-4；共享 y 轴，叠加 128ch 跨被试 baseline 横虚线（EEGNet + CBraMod），并显示各配置的逐被试柱与组均值横线。BP/CSP top-4 仅有 CBraMod 跑次，相应子图无 EEGNet 柱与均值线。

![图 5. 4ch 最优配置 vs 负控制](../figures/fig5_4ch_optimal_vs_neg_control.png)

负控制仍达到 67.65%（远高于 50% 随机基线），说明即使未被任何方法选中的通道也因体积传导而携带足够信息。这一结果同时提供了**两重验证**：（1）正向——数据驱动的通道选择确实捕获了更多任务相关信息（+15.06 pp）；（2）反向——高准确率并非数据泄露所致，而是 EEG 信号本身的物理特性（体积传导使皮层源信号广泛传播）。

**表 10b. 4 通道控制实验结果（跨被试三分类，N = 21；2026-05-11 矩阵闭合新增）。**

| 条件 | 通道来源 | CBraMod Mean ± SD | vs 负控制 (53.37%) |
|------|---------|-------------------|---------------------|
| **Band Power top-4** | mu/beta ANOVA F 统计量前 4 | **60.67%** | **+7.30 pp** |
| 负控制 | 所有方法均未选中的通道 | 53.37% | — |
| CSP top-4 | 32ch CSP 排序前 4 | 47.62% | −5.75 pp |
| FDR top-4 | 32ch FDR 排序前 4 | 46.05% | −7.32 pp |
| Attention top-4 | 32ch Attention 排序前 4 | **41.55%** | −11.82 pp |

Ternary 维度的 4ch 控制实验**与 binary 同向复现**：Band Power top-4 唯一显著超越负控制 (+7.30 pp，binary 为 +11.10 pp)；FDR / CSP / Attention top-4 全部跌至或低于负控制水平（Δ ∈ [−5.75, −11.82] pp）；排序 BP > neg_ctrl > CSP > FDR > Attention 与 binary 完全一致。这把 §3.5.3 的"条件重要性外推失效在 4ch 崩溃"论断从单 task 升级为**双 task 独立复现的现象**。Ternary 上 BP 对 neg_ctrl 的优势 (+7.30 pp) 略小于 binary 上的 +11.10 pp，与 ternary 任务整体难度更高、绝对天花板更低（128ch 74.88% vs binary 90.68%）一致。

通道选择方法敏感度的缩放规律（双 task）总结如下：

| 通道数 | binary spread (4 method, max−min) | ternary spread (4 method) | binary 最优 → 最差 | ternary 最优 → 最差 |
|--------|---:|---:|---|---|
| 64ch | **3.24 pp** | **1.77 pp** | FDR 89.46 → CSP 86.22 | FDR 75.12 → CSP 73.35 |
| 32ch | 2.77 pp | 2.08 pp | FDR 87.71 → CSP 84.94 | BP 72.20 → CSP 70.12 |
| 8ch | 15.63 pp | 6.83 pp | BP 84.05 → Attention 68.42 | BP 66.33 → Attention 59.50 |
| 4ch | **24.05 pp** | **19.12 pp** | BP 78.75 → Attention 54.70 | BP 60.67 → Attention 41.55 |

> 注：方法间 spread 在 binary / ternary 上**双 task 同向单调扩张**——在本数据集与本任务的四档观察上，通道选择方法间的差异随通道数减少而扩大；32ch+ 上 spread ≤ 3.24 pp（落在被试间 std ≈ 13 pp 的 noise 量级），低通道档 spread 急剧扩至 4ch binary 24 pp。32ch 的 binary 最优方法（FDR）在 8ch / 4ch / 32ch ternary / 64ch ternary 上均不再领先——具体的 leading method 在不同 (通道, task) 组合上不同（binary 64/32ch: FDR；binary 8/4ch: BP；ternary 32/8/4ch: BP；ternary 64ch: FDR 但与 neg_ctrl indistinguishable）。我们不把"最优方法在通道数变化下重排序"概括为通用方法论命题，仅作为本研究中的具体观察。**Band Power 在 8 个 cell 上从未是 4 数据驱动方法的最差者**，是本数据集下最稳健的横向观察。4ch FDR∩Attention 交集 (82.71%) 为 favorable outlier，非标准单方法选择，不纳入方法间差异计算；ternary 上 FDR∩Attention 未跑（仅 binary `20260330_1417` 存在）。

**Caveat — 32ch / 64ch negative_control 不是纯互补**：32ch `negative_control` 注册表（`results/32_channel/channel_selections.json`）由 31 个 pure-complement 通道（不在 fdr / attention / csp / band_power / commercial 任一 32ch 选集中）加上 1 个 seed=42 随机 pad（来自 4 method union 的 A30 / 索引 29）组成，以满足 `len(indices) == n_channels` 的注册表约束。64ch `negative_control` 因 4 method 在 64ch 各选 64，union 已覆盖 124 通道，pure complement 仅剩 4 个——故 64ch neg_ctrl 是 **4 pure-complement + 60 seed=42 pad from method-union**。这意味着 64ch neg_ctrl 实质是"低度 method-overlap 配置"而非"纯互补配置"，**§4.3 体积传导论证中 32ch neg_ctrl 仍可作为弱互补证据使用，但 64ch neg_ctrl 不能作为"纯互补通道"被论证**。32ch neg_ctrl ternary 72.38%（vs 31-index 历史 binary 84.08%）的 +0.18 pp 反超 BP 处于 ±SE 内，定性结论不受 pad 影响。

这一结果揭示了一个具体的方法论提醒（见 §3.5.2 讨论）：基于 128ch 全模型计算的通道重要性排序在极低通道数下不仅失效，甚至产生反效果——FDR/Attention/CSP 选出的"最重要"通道空间分布过于集中，反而丢失了负控制中随机通道的分散空间覆盖带来的信息多样性。Band Power 在 4ch / 8ch 档保持判别力的事实与这一观察兼容（其评分机制不依赖全模型上下文，因此天然免疫"条件重要性外推"问题），但本研究**不主张** Band Power 与其他方法之间存在普适性的优劣排序——以下任意一项条件改变都可能让该排序翻转：被试群体（更大 cohort、不同年龄段）、任务粒度（粗运动 MI、四分类、ME）、模型 backbone（非 CBraMod 基座）、预处理流水线（不同滤波带、采样率）。本研究的结论限于"在该 (cohort, 任务, 模型, 预处理) 组合下，4ch / 8ch 部署应至少考虑 Band Power 作为候选方法"这一具体配置层级。

> **数据来源**: FDR∩Attention `20260330_1417`: `results/4_channel/fdr_attention_overlap/20260330_1417_cross_subject_cache_imagery_binary.json`; 负控制 `20260330_1442`: `results/4_channel/negative_control/20260330_1442_cross_subject_cache_imagery_binary.json`; FDR top-4 / Attention top-4: 见 §3.5.2 数据来源行；Band Power top-4 `20260505_2308`: `results/4_channel/band_power/20260505_2308_cross_subject_cache_imagery_binary.json`; CSP top-4 `20260505_2246`: `results/4_channel/csp/20260505_2246_cross_subject_cache_imagery_binary.json`
> 生成命令: 图 5 由 `uv run python scripts/paper/generate_paper_figures.py --figure fig5_merged` 生成；图 4c 由 `uv run python scripts/paper/generate_paper_figures.py --figure sensitivity_scaling` 生成

为了把上文表格中"通道数减少 → 方法选择敏感度上升"的趋势直观化，图 4c 把 32ch / 8ch / 4ch 三档的方法间 spread (max−min, pp) 与最优方法的绝对准确率合并到一张双轴图：左轴红色折线为方法间 spread（2.77 / 15.63 / 24.05 pp），右轴蓝色折线为最优方法的绝对准确率（FDR 32ch 87.71% → BP 8ch 84.05% → BP 4ch 78.75%）。两条曲线方向相反——通道越少时方法选择越关键，但最优方法的绝对天花板降幅有限（87.71% → 78.75%, ~9 pp）。

**图 4c. 通道选择方法敏感度随通道数缩放。** 左 y 轴（红色）：四种方法在该档位的 max−min 准确率 spread；右 y 轴（蓝色）：每档位最优方法的绝对 cross-subject 准确率（按 §3.5 表 9 / §3.5.3 表 10 的最高值）。

![图 4c. Sensitivity Scaling](../figures/sensitivity_scaling.png)

#### 3.5.4 缩减通道下的 XSI-FT

§3.3 已显示在 128 通道条件下 CBraMod XSI-FT 未提供统计显著收益（Δ = −0.56 pp / +0.20 pp）。一个开放问题是：当跨被试模型因空间信息受限而性能下降时，XSI-FT 的单被试 fine-tune 阶段是否会重新显现增益？我们在 32 通道 FDR 与 8 通道 Band Power 两档配置下做了对照实验。

**表 11c. 缩减通道下 XSI-FT 对比（CBraMod 二分类，N = 21）。**

| 通道配置 | 跨被试 | XSI-FT | Δ (XSI-FT − xsubj) | 数据来源 run_tag |
|----------|--------|--------|---------------------|----------------|
| 128ch | 90.68 ± 9.31% | 90.12 ± 8.98% | −0.56 pp | xsubj `20260324_0023` / XSI-FT `20260329_0507` |
| 32ch FDR | 87.71 ± 9.18% | **88.45 ± 8.45%** | **+0.74 pp** | xsubj `20260330_0836` / XSI-FT `20260505_0212` |
| 8ch Band Power | 84.05 ± 9.21% | **82.02 ± 10.74%** | **−2.03 pp** | xsubj `20260331_1950` / XSI-FT `20260506_2159` |

32ch FDR 配置下，XSI-FT 提供 +0.74 pp 的方向性正增益（从 87.71% 升至 88.45%）；但 **8ch Band Power 配置下方向反转**——XSI-FT 反而损失 −2.03 pp（84.05 → 82.02%）。这两个数据点联合起来推翻了原假设的简单形式（"通道越少 XSI-FT 收益越大"），并提出一个更细致的图景：

1. **128ch CBraMod 已饱和**：cross-subject 表征足够丰富，XSI-FT 的单被试 fine-tune 阶段没有增益空间（Δ = −0.56 pp，p = 0.189）。
2. **32ch FDR：被试-特异性 spatial adaptation 的窗口**：FDR 选出的 32 通道在 cross-subject 已经接近 96.7% retention，但被试间 spatial topography 仍存在个体差异；XSI-FT 在这一档恰好释放了 +0.74 pp 的边际收益。
3. **8ch Band Power：cross-subject 已锚定物理签名**：BP 选出的 8 个通道由 mu/beta ERD 物理动机决定，**其空间位置在被试间的差异远小于全 128ch 下的统计 / 注意力 ranking 差异**——cross-subject 已经把"BP 选定的 sensorimotor 通道在群体上的最优响应"学到，XSI-FT 的单被试 fine-tune 阶段提供的边际信号反而被该阶段引入的过拟合风险所抵消。8ch BP cross 84.05% 接近这一通道集合的容量上限，剩余信号容量不足以分摊 fine-tune 的方差代价。

换言之，**基于 3 个数据点（128ch / 32ch FDR / 8ch BP）的方向性观察提示一个工作假设**：XSI-FT 收益可能不是通道数量的单调函数，而是与"cross-subject baseline 离该 (channel, method) 组合的容量上限的距离"相关——32ch FDR 距离上限较远（XSI-FT 有空间），8ch BP 已接近上限（XSI-FT 反而有害），128ch CBraMod 在表征层面对该任务已经饱和。**该工作假设基于 N=3 数据点，强烈受样本量限制；要把它升级为可推广方法论命题，至少需要在 8ch FDR、32ch BP、4ch BP 等额外 (channel, method) 组合上独立验证（§6 后续工作 #2）。** 在该 caveat 下，该方向性观察为 §4.6 部署路线图的"低密度 + XSI-FT"组合添加了一个有待证伪的约束（详见 §4.6 / §4.8）。

需要明确的是，本节仅评估了两个低密度档位 (32ch FDR + 8ch BP)；要把"XSI-FT 收益取决于 baseline 距容量上限"框架升级为可推广的方法论结论，需要在更密集的 (channel, method) 组合上独立观察（如 8ch FDR、4ch BP、64ch FDR 各自的 XSI-FT 等），见 §6 后续工作。

> **数据来源**: 32ch FDR XSI-FT `20260505_0212`: `results/32_channel/fdr/20260505_0212_transfer_cache_imagery_binary.json`; 8ch BP XSI-FT `20260506_2159`: `results/8_channel/band_power/20260506_2159_transfer_cache_imagery_binary.json`（cross-subject baselines: 32ch FDR `20260330_0836_cbramod_imagery_binary`; 8ch BP `20260331_1950`）

### 3.6 领域自适应 Further Pre-training

表 16 在五种独立的 DAPT 训练配置（V1–V5）× 三种下游范式（被试内、跨被试、迁移 XSI-FT）× 两种任务（二分类、三分类）共 **30 个 evaluated cell**（V1–V5 × within + cross + transfer × bin + ter；DAPT 评估全闭合，0 cell 未跑；V1/V2/V3 transfer 6 cell 于 2026-05-11 补完，详见 §3.6.4）上系统评估在外部 MI 数据上进一步预训练 CBraMod 后的下游表现。**核心发现是 task-asymmetric 负迁移在三种 paradigm 上稳健复现于 binary 任务**：cross-subject 5/5、within-subject 5/5、transfer 5/5 一致方向性负向（cross mean Δ=−1.788 pp / Stouffer Z=−5.328, p<0.001；within mean Δ=−1.894 pp / Z=−4.419, p<0.0001；**transfer mean Δ=−1.149 pp / Z=−3.391, p=0.0007**——由 V1/V2/V3 transfer 数据扩展，原 V4/V5-only Z=−2.79 加强）——**总 15/15 binary cell 方向性一致负向**。

**Ternary 任务的方向性是 paradigm-依赖的**：cross 4/5 方向性弱正（V1 +0.79 / V2 +0.44 / V3 +0.62 / V4 +0.22 pp，仅 V5 −1.17）mean Δ=+0.180 pp / Z=+0.577, p=0.564；within 5/5 弱负 mean Δ=−0.918 pp / Z=−2.159, p=0.031；**transfer 3/5 方向性正**（V1 +0.65 / V2 +0.18 / V3 +1.09，V4 −0.32 / V5 −1.47）mean Δ=+0.027 pp / Z=+0.176, p=0.860——**transfer-ternary 5V 聚合相对 v3.1 草稿中 V4/V5-only Z=−1.60, p=0.110 的结论发生方向翻转**，"ternary 一致负迁移"不被支持。BH-FDR @ 0.05 在新 30-cell DAPT family 下重做后 **0/30 cell q<0.05 存活**（v3.1 24-family 下唯一 BH-显著的 V2_within_binary q=0.048 因 family 扩张退到 q=0.060；原始 p 全部不变，q 推高纯属多重比较惩罚）。**优先以 paradigm-level Stouffer 集体证据阅读**——单元格 BH 已对 family 扩张做了 conservative 惩罚。Per-subject paired Δ-of-Δ（每被试的 binary Δ − ternary Δ，以 (V, subject) 为单元 pooled across 5 V cross-subject cells，n=105）：mean=−1.96 pp, t=−5.160, p<0.001——binary cross-sub 退化显著大于 ternary cross-sub 在被试层面成立。V4/V5 跨 paradigm 复现的更细 task-asymmetric gap 见 [paper/reviews/stage4_step1c_v4v5_within_transfer.md](../reviews/stage4_step1c_v4v5_within_transfer.md) §5；V1/V2/V3 transfer 补完的完整 30-cell 重算见 [paper/reviews/stage4_step1d_v1v2v3_transfer.md](../reviews/stage4_step1d_v1v2v3_transfer.md)。

**表 16. Further pre-training 下游评估（CBraMod，N = 21；30 cell，DAPT 评估全闭合）。**

| V | 范式 | 任务 | mean_treat (%) | mean_base (%) | Δ (pp) | t | p (raw) | dz | 95% CI (pp) | q (BH, 30-family) | BH 显著 |
|---|------|------|---:|---:|---:|---:|---:|---:|:-:|---:|:-:|
| V1 | 被试内 | 二分类 | 83.84 | 85.09 | −1.25 | −1.65 | 0.115 | −0.359 | [−2.83, +0.33] | 0.230 | n |
| V1 | 被试内 | 三分类 | 69.25 | 69.54 | −0.30 | −0.45 | 0.656 | −0.099 | [−1.67, +1.08] | 0.757 | n |
| V1 | 跨被试 | 二分类 | 88.84 | 90.68 | **−1.85** | −2.90 | 0.009 | −0.632 | [−3.18, −0.52] | 0.090 | n † |
| V1 | 跨被试 | 三分类 | 75.67 | 74.88 | +0.79 | +0.95 | 0.353 | +0.207 | [−0.95, +2.53] | 0.504 | n |
| V1 | 迁移 | 二分类 | 89.02 | 90.12 | −1.10 | −1.42 | 0.171 | −0.310 | [−2.72, +0.52] | 0.301 | n |
| V1 | 迁移 | 三分类 | 75.69 | 75.04 | +0.65 | +0.79 | 0.441 | +0.172 | [−1.08, +2.39] | 0.575 | n |
| V2 | 被试内 | 二分类 | 82.23 | 85.09 | **−2.86** | −3.53 | 0.002 | −0.771 | [−4.54, −1.17] | 0.060 | n ★ |
| V2 | 被试内 | 三分类 | 68.08 | 69.54 | −1.47 | −1.77 | 0.093 | −0.385 | [−3.20, +0.27] | 0.230 | n |
| V2 | 跨被试 | 二分类 | 89.43 | 90.68 | −1.25 | −2.42 | 0.025 | −0.529 | [−2.33, −0.17] | 0.111 | n |
| V2 | 跨被试 | 三分类 | 75.32 | 74.88 | +0.44 | +0.75 | 0.462 | +0.164 | [−0.78, +1.65] | 0.578 | n |
| V2 | 迁移 | 二分类 | 89.38 | 90.12 | −0.74 | −1.17 | 0.255 | −0.256 | [−2.07, +0.58] | 0.387 | n |
| V2 | 迁移 | 三分类 | 75.22 | 75.04 | +0.18 | +0.30 | 0.770 | +0.065 | [−1.08, +1.43] | 0.796 | n |
| V3 | 被试内 | 二分类 | 83.75 | 85.09 | −1.34 | −1.66 | 0.112 | −0.363 | [−3.02, +0.34] | 0.230 | n |
| V3 | 被试内 | 三分类 | 69.31 | 69.54 | −0.24 | −0.35 | 0.729 | −0.077 | [−1.65, +1.18] | 0.781 | n |
| V3 | 跨被试 | 二分类 | 89.23 | 90.68 | −1.46 | −2.08 | 0.051 | −0.453 | [−2.92, +0.01] | 0.191 | n |
| V3 | 跨被试 | 三分类 | 75.50 | 74.88 | +0.62 | +0.89 | 0.384 | +0.194 | [−0.83, +2.06] | 0.524 | n |
| V3 | 迁移 | 二分类 | 89.11 | 90.12 | −1.01 | −1.17 | 0.258 | −0.254 | [−2.82, +0.80] | 0.387 | n |
| V3 | 迁移 | 三分类 | 76.13 | 75.04 | **+1.09** | +1.67 | 0.111 | +0.363 | [−0.28, +2.46] | 0.230 | n ‡ |
| V4 | 被试内 | 二分类 | 84.05 | 85.15 | −1.10 | −1.34 | 0.194 | −0.293 | [−2.81, +0.61] | 0.323 | n |
| V4 | 被试内 | 三分类 | 68.89 | 69.44 | −0.56 | −0.60 | 0.553 | −0.132 | [−2.48, +1.36] | 0.664 | n |
| V4 | 跨被试 | 二分类 | 89.08 | 90.68 | **−1.61** | −2.93 | 0.008 | −0.640 | [−2.75, −0.46] | 0.090 | n † |
| V4 | 跨被试 | 三分类 | 75.10 | 74.88 | +0.22 | +0.25 | 0.808 | +0.054 | [−1.63, +2.06] | 0.808 | n |
| V4 | 迁移 | 二分类 | 88.45 | 90.12 | **−1.67** | −2.40 | 0.026 | −0.525 | [−3.11, −0.22] | 0.111 | n |
| V4 | 迁移 | 三分类 | 74.72 | 75.04 | −0.32 | −0.38 | 0.709 | −0.083 | [−2.07, +1.43] | 0.781 | n |
| V5 | 被试内 | 二分类 | 82.23 | 85.15 | **−2.92** | −2.54 | 0.020 | −0.554 | [−5.31, −0.52] | 0.111 | n |
| V5 | 被试内 | 三分类 | 67.42 | 69.44 | −2.02 | −1.86 | 0.078 | −0.405 | [−4.30, +0.25] | 0.230 | n |
| V5 | 跨被试 | 二分类 | 87.92 | 90.68 | −2.77 | −2.68 | 0.014 | −0.585 | [−4.92, −0.61] | 0.105 | n |
| V5 | 跨被试 | 三分类 | 73.71 | 74.88 | −1.17 | −1.55 | 0.137 | −0.338 | [−2.75, +0.40] | 0.257 | n |
| V5 | 迁移 | 二分类 | 88.90 | 90.12 | −1.22 | −1.81 | 0.086 | −0.394 | [−2.63, +0.19] | 0.230 | n |
| V5 | 迁移 | 三分类 | 73.57 | 75.04 | −1.47 | −2.00 | 0.059 | −0.436 | [−3.00, +0.06] | 0.197 | n |
| **Stouffer 聚合 — cross-binary（V1–V5, n=5 cell）** ||| | | **−1.788 (mean)** | | **Z=−5.328, p<0.001** |  |  |  | n.a. |
| **Stouffer 聚合 — cross-ternary（V1–V5, n=5 cell）** ||| | | **+0.180 (mean)** | | **Z=+0.577, p=0.564** |  |  |  | n.a. |
| **Stouffer 聚合 — within-binary（V1–V5, n=5 cell）** ||| | | **−1.894 (mean)** | | **Z=−4.419, p<0.0001** |  |  |  | n.a. |
| **Stouffer 聚合 — within-ternary（V1–V5, n=5 cell）** ||| | | **−0.918 (mean)** | | **Z=−2.159, p=0.031** |  |  |  | n.a. |
| **Stouffer 聚合 — transfer-binary（V1–V5, n=5 cell）★ Step 1d 升级** ||| | | **−1.149 (mean)** | | **Z=−3.391, p=0.0007** |  |  |  | n.a. |
| **Stouffer 聚合 — transfer-ternary（V1–V5, n=5 cell）★ Step 1d 升级（方向翻转 Z=−1.60→+0.18）** ||| | | **+0.027 (mean)** | | **Z=+0.176, p=0.860** |  |  |  | n.a. |
| **Stouffer 聚合 — full DAPT family v3.1（16 cell）保留 legacy** ||| | | | | **Z=−4.830, p<0.001** |  |  |  | n.a. |

> † 标记的两单元（V1_cross_binary q=0.090, V4_cross_binary q=0.090）在 v3.1 16-cell BH-FDR family 下原 q=0.048 BH 显著，但在 24-cell 与新 30-cell family 下因更严苛多重比较惩罚不再 survive（q 阈值随 cell 数增加而收紧）；定性方向不变。★ 标记的 V2_within_binary 是 24-cell family 下唯一 BH-显著的 cell（q=0.048）；在 30-cell family 下退到 q=0.060，与 V1_cross_binary / V4_cross_binary 同样退出 BH 显著。‡ 标记的 V3_transfer_ternary +1.09 pp 是整个 30-cell DAPT 矩阵的**全局最大正向 Δ**（任意 task / paradigm；15 个 binary cell 全部方向负）。这是 family 边界扩张的统计学必然代价；paradigm-level Stouffer 集体证据反而稳健加强（transfer-bin 从 Z=−2.79 升至 −3.39, transfer-ter 从 Z=−1.60 翻转至 +0.18）。

> **方法学独立性 caveat**：Stouffer 聚合假设 cell 间 p 值独立；5 个 V 在共享 21 被试 + 共享 baseline 上的 p 值存在弱相关，paradigm-level Z 可能轻微高估方向证据（这是 v3.1 既有的方法学选择，本研究未做改变）。Paradigm-level 跨范式复现一致性（binary 在 cross/within/transfer 三 paradigm 均 Z 负、p<0.001）才是更稳健的方向证据。

> 所有 paired t 检验为双尾，n=21（每被试一对 trial-level majority-vote 准确率）；BH-FDR 在新 30-cell DAPT family 内重做（v3.1 24-cell BH 结果详见 `paper/reviews/stage4_step1c_v4v5_within_transfer.md`，单元格 q 值随 family 扩展而变；6 个 paradigm-level Stouffer 聚合详见 `paper/reviews/stage4_step1d_v1v2v3_transfer.md` §4）。完整 reproducibility 入口：`scripts/internal/recompute_v4v5_within_transfer.py`（V4/V5 within+transfer 8 cells）+ `scripts/internal/recompute_v1v2v3_transfer.py`（V1/V2/V3 transfer 6 cells + 30-cell BH 重做 + 6 个 5V Stouffer），二者命令均为 `uv run python <script>`，确定性输出（无 RNG）。

**图 10a. DAPT V1-V5 30-cell paradigm × task 小矩阵图（含 V1/V2/V3 transfer 2026-05-11 补完）。** 6 个 panel 排成 2 行 × 3 列：**列 = paradigm**（A/D within-subject、B/E cross-subject、C/F transfer）、**行 = task**（A-C binary 上、D-F ternary 下）。每 panel 内 5 根柱表示 V1-V5 的 mean Δ (pp)，黑色 error bar 为 95% CI；柱色按 CI 是否过 0 三分：**红 = CI 全负**（cell-level 方向证据强）、**灰 = CI 跨 0**（方向不定）、绿 = CI 全正（30 cell 中无）。Row 1 三 panel（Within/Cross/Transfer · Binary）一致红色 + Stouffer Z ***：DAPT 在 binary 任务上跨 3 paradigm 系统性失败；Row 2 三 panel（Within/Cross/Transfer · Ternary）以灰色为主：任务不对称（cross/transfer-ternary 弱正 Z=+0.58/+0.18，within-ternary 弱负 Z=−2.16*）。每 panel 右下角嵌 paradigm-level Stouffer Z 与 p 值（cross-bin Z=−5.33 / cross-ter Z=+0.58 / within-bin Z=−4.42 / within-ter Z=−2.16 / **transfer-bin Z=−3.39**（V1-V5 5-cell，从 V4/V5 only Z=−2.79 加强）/ **transfer-ter Z=+0.18**（V1-V5 5-cell，**从 V4/V5 only Z=−1.60 方向翻转**））。Panel F 右下 **V3 transfer-ternary 金色边框 bar + ★（Δ=+1.09 pp, p=0.111）= 全矩阵最正 Δ**。BH-FDR 在新 30-cell DAPT family 内重算后 0/30 cell survive q<0.05；V2 within-bin 24-family q=0.048 → 30-family q=0.060 退出。

![图 10a. DAPT V1-V5 paradigm × task small-multiples](../figures/dapt_v1_v5_smallmultiples.png)

**图 10b. Further Pre-training 下游评估（V1-V5 + reverse-gradient，post Step 1d 30-cell expansion）。** 左图：5 V × 6 paradigm-task 柱状（transfer 列升级为完整 5V；thick border = BH-FDR 显著，新 30-cell family 下 0/30 cell survive）；右图：(effective sample size, Δ) 反向梯度散点 — 全 30 cell 含 transfer 标记（★/♦ 表示 transfer-bin/ternary，x-jittered 至 ~94/142 trials 与 within markers 区分）；每被试 ~80 trial 的 within / transfer 范式上 Δ 更深负向（binary）或方向翻转（ternary 3/5 正），cross 21× pooled 上 4/5 ternary 弱正、5/5 binary 一致负——task asymmetry 在 (sample size, Δ) 反向梯度图上视觉化为 binary 全负 / ternary 在 transfer 与 cross 两个 paradigm 上接近 0 或弱正。**Panel 标注约定**（原图内文字已移至此 caption）：左图 (Panel A) 柱色按 V1–V5 配色（见图例 V1–V5），**粗黑边框 = BH-FDR q<0.05（在新 30-cell DAPT family 内重算，0/30 cell survive；V1/V2/V3 transfer 6 cell 于 2026-05-11 补完）**；右图 (Panel B) marker 编码 **circle = within、▲ = cross、★ = transfer-binary、♦ = transfer-ternary，粗边框 = BH-FDR 显著，transfer markers 沿 x 轴轻微 jitter（×1.18，至 ~94/142 trials）以与 within (per-subject) markers 区分**。**Stouffer 聚合（Step 1d，每条 5V；与原图内文字框逐字一致）**：

```
Stouffer aggregates (Step 1d, 5V each):
cross-bin (n=5): Z=−5.33, p<0.001
cross-ter (n=5): Z=+0.58, p=0.564
within-bin (n=5): Z=−4.42, p<0.0001
within-ter (n=5): Z=−2.16, p=0.031
transfer-bin (n=5) ★1d: Z=−3.39, p=0.0007
transfer-ter (n=5) ★1d: Z=+0.18, p=0.860
— legacy (v3.1) —
full DAPT family (n=16): Z=−4.83, p<0.001
```

![图 10b. Further Pre-training 下游评估](../figures/further_pretraining.png)

#### 3.6.1 V4 / V5 surgical fix 与机制收紧

V1–V3 同时改变了数据量、LR 调度与训练步数，留下三个未隔离的混淆假设：(1) **域错配**（粗 hand/leg/upper-limb MI vs 细 finger MI）、(2) **Stieger 占主导**（V2 中 ~79%）、(3) **通道数异质**（7 种通道数 22/30/60/61/62/64/128，可能让 ACPE 难以为下游 128ch 网格校准）。V4 与 V5 是针对这三个假设的两次 surgical fix。

**V4（3-set 域对齐 + strict filter）**：选取与下游 finger MI 域最接近的 3 个数据集（Cho2017, Ofner2017, Schirrmeister2017），并应用 strict filter（300 µV peak + per-channel kurtosis>10）替代 basic 500 µV mean-abs，达到全 5 V 中**最低的 pre-train loss 0.001914**（−48% vs V2 的 0.003714）。结果：cross-binary Δ=−1.61 pp（p=0.008, q=0.048, BH 显著），cross-ternary Δ=+0.22 pp（n.s.）——**域对齐 + 数据净化双管齐下仍未救援 binary**。这说明 (1) 域错配是必要但非充分原因；strict filter 本身没有把 binary 拉回正向。

**V5（Stieger 单源 60ch）**：单源 + 单一通道几何，直接消除假设 (3)。结果：cross-binary Δ=**−2.77 pp**（5 V 中**最差**, p=0.014），cross-ternary Δ=**−1.17 pp**（5 V 中唯一弱负）——V5 在 binary / ternary 上**双向恶化**，**反方向证伪**了"通道数异质是混淆"的假设。机制解释：单源 ACPE 在 Stieger 60ch 几何上过拟合空间先验，下游 128ch fine-tune 必须从错位起点重新校准 ACPE；V1–V3 的 7 种通道数反而强迫 backbone 学 channel-agnostic 表示——**通道多样性在 DAPT 中是保护因子，不是 bug**。

**机制收紧表**：

| 候选机制 | V4/V5 检验 | 检验结果 |
|----------|-------------|----------|
| (1) 域错配（粗 MI vs 细 finger MI） | V4: 3-set + strict filter | binary 仍 −1.61 pp（q=0.048）→ **必要但 surgery 不足以救援** |
| (2) Stieger 占主导（V2 ~79%） | V3 (~30%) + V4 (0%) | 全部仍负向 → **基本排除** |
| (3) 通道数异质（7 种 → 1 种） | V5 单源 60ch | V5 双向最差 → **强烈反方向证伪**（通道多样性是保护因子） |

**唯一存活假设——MI 粒度错配**：粗 hand/leg/upper-limb MI 的 MAE pretext loss 学到的是"哪个肢体在动"的低频空间包络；下游 finger-level binary（食指 vs 中指，**同手**）需要的是 DAPT 没学到的细粒度区别。Ternary 的 rest 类（不动 vs 运动）正好能用 DAPT 学到的粗粒度空间包络识别——所以 ternary 没那么糟，部分配置（V1/V2/V3/V4）甚至轻微正向。这一机制同时解释了 task asymmetry（binary 需细判别 / ternary 受益于粗判别）与 V5 的反向恶化（单源 ACPE 几何过拟合加重粒度错配的下游代价）。

**Transfer rescue gradient（binary 任务，新增于 Step 1d）**：V1/V2/V3 transfer 补完后揭示了一个意外的细粒度模式——**V1/V2/V3 在 binary 上呈强 cross→transfer 衰减**（Δ magnitude 衰减 31–41%，显著性全部消失：V1 cross p=0.009 → transfer p=0.171；V2 p=0.025 → 0.255；V3 p=0.051 → 0.258）；**V5 呈部分衰减**（Δ −2.77 → −1.22，magnitude 衰减 56%，cross 显著 → transfer p=0.086 边缘 n.s.）；**V4 是唯一例外**（Δ −1.61 → −1.67，cross/transfer 都 BH-边缘 q=0.090）。V4 的独特性——strict filter + 3-set 域对齐——**印下了 surgical fix 最具体也最顽固的错误先验**，per-subject fine-tune 无法清洗；V5 单源 60ch 几何错位虽顽固，但每被试 fine-tune 仍能部分校正 ACPE；V1/V2/V3 的弥散扰动近似"小随机噪声"，最易被下游清洗。这一三档差异同时提供机制证据：**fine-tune 清洗的难度与 surgical fix 的"具体性"正相关**——V4 最具体 → 最难清洗；V5 中间；V1/V2/V3 最弥散 → 最易清洗。这与 §3.6.1 的 MI 粒度错配机制兼容（清洗顺序与"DAPT 印迹的精度"而非"DAPT 印迹的方向"耦合），并为 V4 的存活假设提供了独立的方向性证据。

**未独立检验的替代假设（透明披露）**：需补充说明，"MI 粒度错配"作为唯一存活假设是经过 V4/V5 对"Stieger 占主导"与"通道数异质混淆"两个候选的排除得到的，本研究并未独立直接检验"粒度错配"机制本身。至少存在三类未在本研究中分离的结构等价/平行替代假设：(i) **小语料 catastrophic forgetting** — V4 仅 4,937 段、V1–V3 在数千至数万段量级，BERT-style continued pretraining 在小语料上可能损耗 TUEG-学到的表征多于其新增 finger-MI 相关部分，与"哪种 MI 粒度被学到"无关；(ii) **DAPT 方法配置不匹配** — mask_ratio=50% + MSE pretext + AdamW + lr=5e-5 全部沿用 [4] 在 TUEG 下游 fine-tuning 的默认值，未针对 MI 数据特性系统调参（详见 Limitation #12a）；(iii) **"任务-pretext 重叠度"而非"粒度"本身驱动 task asymmetry** — 若 DAPT 学到的"是否运动 vs 静息"边界与 ternary 的 rest 类天然契合而与 binary 的"食指 vs 中指（同手 motor execution intent）"无关，结构上与本节"粒度错配"机制等价但表述更宽，且不能被现有 V1–V5 数据独立分离。这三类替代假设的隔离需要 §6 #3 描述的方法配置 ablation（mask ratio / loss / epoch / warmup 扫描）+ 单数据集 leave-one-out 完成后才能闭合。当前结论应被理解为"在 V1–V5 五个配置下，'MI 粒度错配'是同时与所有 5 V 数据兼容的最简存活解释，但非唯一可能解释"。

#### 3.6.2 透明披露：方向反转

诚实披露三类与"一致负迁移"先前框架不符的反转：

1. **V1→V2 cross-binary 反向恢复**（[paper/analysis/further_pretraining_analysis.md §6.3](../analysis/further_pretraining_analysis.md)）：在原始 baseline 下 V2 (89.43%) 高于 V1 (88.84%) 约 +0.59 pp，是 V1 vs V2 四个条件中唯一 V2 优于 V1 的组合。在 Step 1b 修订的 registry baseline (90.68%) 下，V1 cross-binary Δ=−1.85，V2 cross-binary Δ=−1.25，方向不变但 V2 比 V1 弱 0.60 pp。仍可看作 cross-subject 训练数据规模本身的正则化效应部分稀释了被破坏的 backbone 初始化的影响，与 §3.6.1 的 task-asymmetric 机制兼容。

2. **Cross-ternary 4 个 V 反向（弱正）**：V1 +0.79 / V2 +0.44 / V3 +0.62 / V4 +0.22 pp，单元格层面均 BH 不显著（q>0.4），但**方向性一致**。这驱动了 cross-ternary Stouffer Z=+0.577, p=0.564 的 mildly-positive 聚合方向，**令"DAPT 一致负迁移"在 ternary 任务上无法成立**。

3. **V5 cross-ternary 单点反向（弱负）**：V5 −1.17 pp，5 V 中唯一与其他 V 方向相反的 ternary cell；其余 4 V 均弱正。如 §3.6.1 所述，V5 的双向恶化由其单源 ACPE 几何过拟合机制独立解释，与"通道多样性保护"的反方向证据自洽。

4. **V1/V2/V3 transfer-ternary 三个方向性正（Step 1d 补完）**：V1 +0.65 / V2 +0.18 / V3 +1.09 pp，其中 **V3 transfer-ternary +1.09 pp 是整个 30-cell DAPT 矩阵的全局最大正向 Δ**（任意 task / paradigm；15 个 binary cell 全部方向负，所以最正 Δ 必在 ternary 侧；n.s., p=0.111；dz=+0.363 中等效应）。这驱动了 transfer-ternary 5V Stouffer 从 V4/V5-only 的 Z=−1.60 (p=0.110) **翻转**为 Z=+0.18 (p=0.860)，意味着"transfer-ternary 整体负向"的 v3.1 草稿结论**不再成立**——transfer 在 ternary 上的 DAPT 效应近零、个别 V 甚至方向性正。与 §3.6.1 的 MI 粒度错配解释**相容**：ternary 的 rest 类既受益于 DAPT 学到的粗粒度空间包络，又在 per-subject fine-tune 后获得局部细节，两次叠加产生中性到弱正效应。

#### 3.6.3 V2 训练 caveat（保留 v3）

**V2 训练在 Epoch 13 因 Windows LMDB MapResizedError 中断**，使用 Epoch 12 checkpoint 作为 best model，未触发由 patience=5 决定的 early stopping。**V3 采用 warm-restart-from-weights**（先训 15 ep + continue 训 12 ep，optimizer 与 LR scheduler 状态在阶段 ii 重置）；V4/V5 均为单阶段训练。这些训练组态差异不改变 §3.6.1 的 task-asymmetric 定性结论，但意味着"V2/V3 是否在更长连续训练后达到不同结论"严格意义上不可证。

#### 3.6.4 评估覆盖范围

V1/V2/V3 已评估被试内、跨被试两种范式各两 task（共 12 cell）；V4/V5 经 2026-05-10 补充评估后覆盖被试内、跨被试、迁移（XSI-FT）三种范式各两 task（共 12 cell，即 V4 6 + V5 6）；**V1/V2/V3 transfer 6 cell 于 2026-05-11 补完**（runner: [scripts/experiments/run_dapt_v1_v2_v3_transfer.sh](../../scripts/experiments/run_dapt_v1_v2_v3_transfer.sh)；统计重算: [paper/reviews/stage4_step1d_v1v2v3_transfer.md](../reviews/stage4_step1d_v1v2v3_transfer.md)；reproducibility: `uv run python scripts/internal/recompute_v1v2v3_transfer.py`）。**整体评估覆盖 30/30 cell，DAPT 矩阵全闭合**。

**Caveat #6 ("DAPT 是否仅在 cross-subject 范式失败")**：在 **binary 任务**上以 15/15 cell 方向负向、三个 paradigm Stouffer 均 p<0.001（cross Z=−5.33, within Z=−4.42, transfer Z=−3.39）的强证据**支持关闭**；在 **ternary 任务**上则呈现 paradigm-依赖的方向不一致（cross 4/5 弱正 / within 5/5 弱负 / transfer 3/5 弱正），**不支持** "ternary 一致负迁移" 的更弱结论。30-cell V1–V5 全矩阵 0/30 cell BH 显著、0/15 binary cell 方向性正（一致负向）、6/15 ternary cell 方向性正（其中 V3_transfer_ternary +1.09 pp 为全 30-cell 矩阵的全局最大正向 Δ）——DAPT 失败不是 cross-subject 范式特有现象，而是 task-asymmetric × paradigm-dependent 的方向性结果。原 [Plan §Stage 4](../../C:/Users/zhang/.claude/plans/did-we-use-the-sprightly-peacock.md) 的 gating 规则按字面 4/4 cross cell fail 不解锁——但作为 reviewer-defense 仍执行了 V4/V5 8-cell within+transfer 流水线 + V1/V2/V3 transfer 6-cell 补完，结果与 cross 在 binary 上完全方向一致、在 ternary 上揭示了 transfer 与 cross 类似的 paradigm-依赖方向性。这一更广 paradigm 矩阵的实证补全详见 §5 limitation #12 (b)、[paper/reviews/stage4_step1c_v4v5_within_transfer.md](../reviews/stage4_step1c_v4v5_within_transfer.md)、以及 [paper/reviews/stage4_step1d_v1v2v3_transfer.md](../reviews/stage4_step1d_v1v2v3_transfer.md)。

> **数据来源**:
> - Baseline (registry-correct, n=21): cross-binary `results/20260324_0023_cross_subject_cache_imagery_binary.json` (run_tag `20260324_0023`, `is_baseline=1`, mean=90.68%); cross-ternary `results/20260324_0109_cross_subject_cache_imagery_ternary.json` (run_tag `20260324_0109`, `is_baseline=1`, mean=74.88%); within-binary ExperimentDB run_tag `20260321_0343`; within-ternary `20260205_0306`.
> - V1: pretrain checkpoint `checkpoints/cbramod/further_pretrain_20260322_0042/best_model.pth`（Epoch 9, loss=0.006055；legacy, V1 评估缓存见 paper/analysis/further_pretraining_analysis.md §9）。
> - V2: pretrain checkpoint `checkpoints/cbramod/further_pretrain_20260323_0609/best_model.pth`（Epoch 12, loss=0.003714，因 LMDB MapResizedError 中断）；下游缓存 `results/20260323_1433_cbramod_imagery_binary.json` (within bin), `results/20260323_1517_cross-subject_cbramod_imagery_binary.json` (cross bin), `results/20260323_1615_cbramod_imagery_ternary.json` (within ter), `results/20260323_1709_cross-subject_cbramod_imagery_ternary.json` (cross ter)。
> - V3: pretrain checkpoint `checkpoints/cbramod/further_pretrain_v3_continued_20260505_1800/best_model.pth` (epoch 22)；下游缓存 `results/dapt_v3/20260505_2012_within_subject_cache_imagery_binary.json`, `..._2033_within_subject_cache_imagery_ternary.json`, `..._2100_cross_subject_cache_imagery_binary.json`, `..._2131_cross_subject_cache_imagery_ternary.json`。
> - V4: pretrain checkpoint `checkpoints/cbramod/further_pretrain_v4_20260509_2345/best_model.pth`（Epoch 40, loss=0.001914；3-set + strict filter, 4,937 segments）；下游缓存 `results/20260510_1710_cross_subject_cache_imagery_binary.json` (cross bin), `results/20260510_1020_cross_subject_cache_imagery_ternary.json` (cross ter)。
> - V5: pretrain checkpoint `checkpoints/cbramod/further_pretrain_v5_20260510_1049/best_model.pth`（Epoch 21, loss=0.003108；Stieger-only 60 ch, 67,068 segments）；下游缓存 `results/20260510_1812_cross_subject_cache_imagery_binary.json` (cross bin), `results/20260510_1738_cross_subject_cache_imagery_ternary.json` (cross ter)。
> - V1 transfer (Step 1d): treatment `results/dapt_v1/20260510_2357_transfer_cache_imagery_binary.json` (bin), `results/dapt_v1/20260511_0012_transfer_cache_imagery_ternary.json` (ter)；init weights = V1 cross-subject checkpoint `checkpoints/cross_subject/20260322_1116_cbramod_imagery_binary/best.pt` (bin) / `20260322_1543_cbramod_imagery_ternary/best.pt` (ter)。
> - V2 transfer (Step 1d): treatment `results/dapt_v2/20260511_0031_transfer_cache_imagery_binary.json` (bin), `results/dapt_v2/20260511_0042_transfer_cache_imagery_ternary.json` (ter)；init weights = V2 cross-subject checkpoint `checkpoints/cross_subject/20260323_1517_cbramod_imagery_binary/best.pt` (bin) / `20260323_1709_cbramod_imagery_ternary/best.pt` (ter)。
> - V3 transfer (Step 1d): treatment `results/dapt_v3/20260511_0058_transfer_cache_imagery_binary.json` (bin), `results/dapt_v3/20260511_0109_transfer_cache_imagery_ternary.json` (ter)；init weights = V3 cross-subject checkpoint `checkpoints/cross_subject/20260505_2100_cbramod_imagery_binary/best.pt` (bin) / `20260505_2131_cbramod_imagery_ternary/best.pt` (ter)。
> - Transfer baseline (TUEG-original cross_subject init, n=21): `results/20260329_0507_transfer_cache_imagery_binary.json` (bin, mean=90.12%) / `results/20260329_0521_transfer_cache_imagery_ternary.json` (ter, mean=75.04%)。
> - 完整统计与 Reproducibility: `paper/reviews/stage4_step1b_stat_recompute_v4v5.md` (V1-V3 + V4/V5 cross, 16 cells) → `stage4_step1c_v4v5_within_transfer.md` (+8 V4/V5 within+transfer = 24 cells) → `stage4_step1d_v1v2v3_transfer.md` (+6 V1/V2/V3 transfer = 30 cells, BH 重做 + 6 个 5V Stouffer)；历史背景与 V1/V2 详细比较：`paper/analysis/further_pretraining_analysis.md`；V4/V5 实验交接：`docs/handoffs/2026-05-10_dapt_v4_v5.md` 含 "追加 (2026-05-11)" 段记录 V1/V2/V3 transfer。
> 生成命令: 图 10a 由 `uv run python scripts/paper/generate_paper_figures.py --figure dapt_v1_v5_smallmultiples` 生成；图 10b 由 `uv run python scripts/paper/generate_paper_figures.py --figure further_pretraining` 生成；30-cell 统计重算由 `uv run python scripts/internal/recompute_v1v2v3_transfer.py` 生成。

### 3.7 探索性消融：架构 / 预训练 / 容量贡献的初步检验

为更好理解 CBraMod 相对 EEGNet 在 §3.1–§3.3 中观察到的优势源自何处，本节报告两项探索性消融：(a) §3.7.1 将 EEGNet 的参数规模从 16K 阶梯式扩展到 30M（与 CBraMod backbone 同量级），探查"参数容量本身是否是 EEGNet 表现不及 CBraMod 的根本原因"；(b) §3.7.2 完全切除 CBraMod 的 TUEG 预训练权重（random-init），探查"架构本身在不依赖预训练的情况下是否仍提供独立价值"。两项消融在 {EEGNet, CBraMod} × {random init, TUEG pretrained} 矩阵上覆盖三个角点（"EEGNet pretrained"无对应 EEG 基座模型故空缺）。

**重要 caveat（贯穿本章）**：本节两项消融在 HPO 预算与扩参轴上均存在已知非对称性，使其结论不具备"独立可归因分解"的力度，应被理解为方向性观察而非定量分解。具体地：(i) **EEGNet-Huge v1 / v2 / v3 / Mid 四档与 EEGNet baseline 共享原始 32-trial HPO 范围内的 architecture defaults，但其本身的优化栈（LR、weight_decay、dropout、LayerNorm 是否启用）由 ≤ 2 trial 的人工调试得到**——并非独立 Optuna 搜索；(ii) **CBraMod random-init 直接复用 original-weights baseline 的 HP（`get_default_config()`）**，没有跑专属 HPO；(iii) **EEGNet baseline → Mid 的首跳同时改变 conv stem (F1: 16→32, F2: 64→256) 与 MLP 头（单 Linear → 双层 [1024,1024] + LayerNorm）**，未隔离 conv stem 单轴 vs MLP 头单轴的贡献。在这三项约束下，§3.7.1 / §3.7.2 / §3.7.3 报告的所有 Δ 值应被理解为"在共享默认 HP、受限 HPO 预算、双轴 baseline → Mid 跳跃下观察到的复合估计"，而非各因子（架构 / 预训练 / 容量）的独立可归因分解。严格的独立 HPO 验证（EEGNet-Huge v1/v2 ≥ 25 trial Optuna；CBraMod random-init ≥ 25 trial Optuna）留待后续工作（详见 §6 #8）。

图 12 把本章的探索性观察压缩到一张"参数量 × cross-binary 准确率"双轴图上，便于读者一眼看到 EEGNet 容量阶梯（蓝色实线，16K → 30M）的下行趋势、random-init CBraMod (橙色菱形, 30.48M) 与 TUEG-pretrained CBraMod (红色五角星, 30.48M) 两个单点的相对位置，以及三条主要相邻 **composite-estimate Δ 注释（复合估计；详见 §3.7.3 footnotes）**：capacity ladder ~−25.30 pp / 跨架构 ~+34.97 pp / TUEG 预训练 ~+4.34 pp。**所有数值在共享默认 HP + 受限 HPO 预算 + baseline → Mid 双轴跳跃三项约束下观察**，因此各 Δ 不可被解读为架构、预训练、容量任一因子的独立可归因贡献，建议与 §3.7.1–§3.7.3 的细分章节及 §3.7.3 三个脚注一并阅读。

**图 12. §3.7 探索性消融总览（cross-subject binary, N=21, 128ch）。** 蓝色方块连线为 EEGNet 容量阶梯（baseline 16K → Mid 1.90M → Huge v3 5.84M → Huge v1/v2 ~20–30M）；橙色菱形为 random-init CBraMod (~30.5M, 86.34%)；红色五角星为 TUEG-pretrained CBraMod (~30.5M, 90.68%)。三条注释箭头分别标注 EEGNet 内扩参 Δ ≈ −25.30 pp、跨架构 Δ ≈ +34.97 pp、TUEG 预训练 Δ = +4.34 pp 的复合估计，详见 §3.7.3 表与脚注。

![图 12. §3.7 探索性消融总览](../figures/exploratory_ablation_overview.png)

> 生成命令: 图 12 由 `uv run python scripts/paper/generate_paper_figures.py --figure exploratory_ablation_overview` 生成（数据合并自 §3.7.1 EEGNet ladder runs + §3.7.2 random-init CBraMod runs，原始 run_tag 见后续 §3.7.1 / §3.7.2 数据来源行）。

#### 3.7.1 EEGNet 容量阶梯（16K → 30M，128 通道）

为检验 EEGNet 相对 CBraMod 的差距是否仅源自参数容量限制（~16K vs ~30M，~1900× 差距），我们沿 (conv stem, MLP 头) 双轴扩展 EEGNet，构建四档容量阶梯：EEGNet baseline (16K, **F1=16, D=4, F2=64**, 单 Linear 头，沿用 §3.1–§3.3 的 EEGNet-16,4 配置)、EEGNet-Mid (1.90M, **F1=32, D=4, F2=256**, [1024, 1024] + LayerNorm + ELU)、EEGNet-Huge v3 (5.84M, [2048, 2048] + LayerNorm)、以及两个 ~20–30M 量级版本 EEGNet-Huge v1 (19.99M, [4096, 4096], 无 LN) / v2 (30.22M, [5120, 5120], 无 LN)。**Mid / Huge 系列均共享扩展后的 conv stem (F1=32, D=4, F2=256, kernel_length=64)，与 baseline 的 F1=16/F2=64 不同**——因此 baseline → Mid 的首跳同时改变 conv stem (F1: 16→32, F2: 64→256) 与 MLP 头（单 Linear → 双层 [1024,1024]），严格意义上未隔离 conv stem 单轴 vs MLP 头单轴的贡献；Mid → v3 → v1/v2 三档则沿 MLP 头单轴扩参（conv stem 完全相同）。HP 在两阶段调试中找到稳定配置（v3 / Mid 共用 lr = 8e-4 至 1.5e-3、wd = 0.03–0.05、CAWD scheduler；详见 `docs/handoffs/2026-05-09_eegnet_huge.md`）。

**表 18a. EEGNet 容量阶梯准确率（N = 21，128 通道二分类）。**

| 模型 | 参数量 | 被试内 | 跨被试 | XSI-FT |
|------|--------|--------|--------|--------|
| EEGNet baseline | 16K | **78.10%** | **76.67%** | **82.05%** |
| EEGNet-Mid | 1.90M | 66.88% | 57.65% | 80.45% |
| EEGNet-Huge v3 | 5.84M | 67.71% | 51.37% | 80.62% |
| EEGNet-Huge v2 | 30.22M | (orphan) | 50.07% (chance) | — |
| EEGNet-Huge v1 | 19.99M | — | 50.00% (chance) | (state_dict bug) |
| CBraMod random-init | 30.48M | 62.05% | 86.34% | 86.22% |
| CBraMod baseline | 30.48M | **85.15%** | **90.68%** | **90.12%** |

> EEGNet-Huge v1 / v2 在 ~20–30M 量级两套独立人工调试 HP（lr 相差 10×：5e-5 vs 5e-4，wd / dropout / LayerNorm on/off 等亦不同；详见 `docs/handoffs/2026-05-09_eegnet_huge.md` L154-170）下均出现 train loss 死锁在 0.693（chance entropy）、val acc 50%、所有 21 名被试 test 50% 的不可训练状态，因而仅列 cross 一栏（其余范式的 v1 因 state_dict 加载 bug 未跑、v2 within 数据 orphan 未入库）。在两套手调 HP 下 v1/v2 不可训；**v3 通过加 LayerNorm + 缩小 MLP 至 [2048, 2048] 后立即 trainable，提示 v1/v2 的失败更可能是 BF16 数值精度下深 MLP 头优化栈兼容性问题（vanishing gradient / dying ELU），而非容量本身的根本饱和**——见交接文档 `docs/handoffs/2026-05-09_eegnet_huge.md` L156、L195-197、L249-260 的工程诊断。是否在严格独立 HPO 预算（≥ 25 trial Optuna，覆盖 LR、warmup、LayerNorm on/off、init scheme、dropout）下 30M 量级 EEGNet 仍不可训，**留待后续工作**（§6 #8）；在补全此独立 HPO 之前，"30M EEGNet 不可训" 的结论应被理解为"在受限 HPO 预算下的观察"。

**Cross-subject 准确率沿当前扩参轴随容量单调下降**：从 76.67% (16K) → 57.65% (1.90M) → 51.37% (5.84M) → 50.00% (~20–30M) 一路下降，~30M 已落入 chance。**在共享默认 HP、受限 HPO 预算（≤ 2 trial 人工调试）以及 baseline → Mid 双轴扩参（conv stem + MLP 头同时改变）这三项约束下**，本观察方向性支持 "EEGNet 架构内沿当前扩参轴扩参对 cross-subject 准确率不利"，但并不支持更强的 "EEG decoding 瓶颈不在容量" 论断——后者需要在 EEGNet-Huge v1/v2 各跑 ≥ 25 trial 独立 HPO 并仍观察到不可训才能成立（详见 §6 #8）。这一现象方向上与 Ding et al. [3] 的 deepEEGNet 实验（"+1.21% binary 微弱提升"，规模估计 ~100K–1M）一致——后者也未能通过扩参显著改善——但本研究规模扩张幅度（5.84M / ~30M，2 个数量级）尚不足以独立排除"扩参 + 严格 HPO"组合下能否反转该单调趋势。

**Within / XSI-FT 范式下容量损失更温和**：被试内从 78.10% 降至 ~67%（~−11 pp），但 v3 与 Mid 之间已饱和；XSI-FT 仅从 82.05% 降至 80.45–80.62%（~−1.5 pp），对容量基本不敏感。XSI-FT 对扩参 EEGNet 的鲁棒性与 §3.3 的 EEGNet XSI-FT 增益（+5.38 / +5.10 pp，两 task 均 p < 0.01）一致——单被试 fine-tune 阶段把过参数化的分类头校准回单被试分布。

**与同规模 random-init CBraMod (§3.7.2) 的探索性对照**：在 ~30M 参数 + 无预训练的同等条件下，EEGNet-Huge v2 (30.22M) cross 50.07%（chance）vs CBraMod random-init (30.48M) cross 86.34%——观察到 ~+36 pp 差距；即便取可训练的 EEGNet-Huge v3 (5.84M) cross 51.37% 作对照，与 random-init CBraMod 的差距仍达 ~+35 pp，与容量量级差距非线性脱钩。**在 EEGNet-Huge v1/v2/v3 与 CBraMod random-init 均未做专属 HPO 的对照下**，这一差距是 "架构差异 + EEGNet 优化栈不稳定 + random-init CBraMod HP 错配" 三者的复合估计；其中可归因到 backbone 架构（transformer + ACPE vs 扩参 CNN）的下界尚不能从本节单独给出。本节的探索性观察支持 "在受限 HPO 预算下，扩参 EEGNet 远不及 random-init CBraMod" 这一较弱主张；将该差距精准归因到 "架构归纳偏置" 需要 §6 #8 描述的双侧独立 HPO sweep 完成后才能成立。

> **数据来源**: EEGNet-Mid runs `20260509_1419` (within), `20260509_1310` (cross), `20260509_1444` (XSI-FT)；EEGNet-Huge v3 runs `20260509_0928` (within), `20260509_0847` (cross), `20260509_1030` (XSI-FT)；EEGNet-Huge v1 / v2 runs `20260509_0201` (cross v1), `20260509_0735` (cross v2)。完整 HP / 架构规格 / 失败 HP 调试细节见 `docs/handoffs/2026-05-09_eegnet_huge.md`。EEGNet baseline 与 CBraMod baseline 来源见 §3.1–§3.3 / §3.7.2。

#### 3.7.2 Random-init CBraMod 消融（128 通道）

为剥离 CBraMod 在三种范式下相对 EEGNet 的优势中、来自 TUEG 预训练的部分与来自架构本身的部分，我们以完全随机初始化的 CBraMod 重跑 §3.1–§3.3 的 6 个核心 condition（被试内 / 跨被试 / XSI-FT × 二分类 / 三分类，N = 21，128 通道）。除 backbone 初始化以外（`--no-pretrained`），所有超参数与 §3.1–§3.3 baseline 完全相同（沿用 `get_default_config()`，含 cross-subject HPO 后默认）；XSI-FT 阶段使用本节产出的 random-init cross-subject checkpoint 作为初始权重，确保整条 transfer 链是 end-to-end from-scratch（不混入原始 TUEG 权重）。本消融与 §3.6 further pre-training 形成对极方向——后者朝"更多预训练"扰动，本节朝"零预训练"扰动，两端共同界定原始 CBraMod recipe 在本任务上的位置。

**表 18. Random-init vs Original-weights CBraMod vs EEGNet 三方对比（N = 21，128 通道）。**

| 范式 | 任务 | random-init CBraMod | original-weights CBraMod | EEGNet | Δ (random − orig) |
|------|------|---------------------|--------------------------|--------|-------------------|
| 被试内 | 二分类 | 62.05 ± 17.68% | **85.15 ± 11.00%** | 78.10 ± 12.61% | **−23.10 pp** |
| 被试内 | 三分类 | 38.65 ± 14.07% | **69.44 ± 15.42%** | 66.81 ± 14.50% | **−30.79 pp** |
| 跨被试 | 二分类 | 86.34 ± 9.41% | **90.68 ± 9.31%** | 76.67 ± 11.95% | −4.34 pp |
| 跨被试 | 三分类 | 73.06 ± 12.49% | **74.88 ± 14.03%** | 61.23 ± 11.28% | −1.82 pp |
| XSI-FT | 二分类 | 86.22 ± 9.46% | **90.12 ± 8.98%** | 82.05 ± 11.00% † | −3.90 pp |
| XSI-FT | 三分类 | 73.43 ± 12.91% | **75.04 ± 13.97%** | 66.33 ± 12.65% † | −1.61 pp |

> † EEGNet XSI-FT 无 `is_baseline=1` 标记，引用最近的 N = 21 XSI-FT 运行 `20260507_1835`（二分类）和 `20260507_1913`（三分类）作为参考值。

**预训练贡献按数据规模呈两段式分布**：被试内（每名被试 ~70 trial，单被试训练）下随机初始化下降 −23 至 −31 pp；跨被试与 XSI-FT（~21× 训练数据，1.5K+ trial）下仅下降 −1.6 至 −4.3 pp，前者跨度约为后者的 7×。具体到 within ternary 极端例：random-init 下 21 名被试中 **18 名**测试准确率落在 chance ± 2 pp 区间（≈ 33.33%）——三分类 from-scratch CBraMod 在被试内基本无法学到任何信号；唯三例外为 S07（61.67%）、S09（59.58%）、S19（90.42%）。

**Seed 复现性检查**：为排除 18 / 21 collapse 是 seed = 42 的运气特例，重跑 within / ternary random-init 仅替换为 seed = 1234（其余 HP 与 `20260509_0102` 完全一致），得到 39.25% ± 13.90%（vs seed = 42 的 38.65% ± 14.07%）和 17 / 21 chance-collapse（vs 18 / 21）。两次 above-chance 被试交集为 {S09, S19}（两个 seed 下都逃出 chance），仅 S07（seed 42 only）和 {S13, S14}（seed 1234 only）为 seed 特异。两次 mean 差 0.6 pp、collapse 计数差 1 名——within ternary 的 chance collapse 是 from-scratch CBraMod 在该范式下的稳健行为而非 seed 噪声（seed = 1234 cache: `results/20260509_1838_within_subject_cache_imagery_ternary.json`）。

**架构本身的独立价值在 cross-subject 与 XSI-FT 下被验证**：random-init CBraMod 即便完全切除 TUEG 预训练，在跨被试二分类仍达 86.34%（vs EEGNet 76.67%，**+9.67 pp**）、跨被试三分类 73.06%（vs 61.23%，**+11.83 pp**）；XSI-FT 二分类 86.22%（vs EEGNet 82.05%，+4.17 pp）、三分类 73.43%（vs 66.33%，+7.10 pp）。这说明 transformer + ACPE 架构在 21× pooled 数据下具备独立学习能力，与"差距全部来自 TUEG 预训练"的最弱归因相反。

**Within-subject 范式下 from-scratch CBraMod 在当前 HP 下输给 EEGNet**：random-init CBraMod 被试内二分类 62.05% 低于 EEGNet 78.10% 约 −16 pp，三分类 38.65% 低于 EEGNet 66.81% 约 −28 pp。该差距方向性提示 transformer 在 ~70 trial 单被试样本下、**沿用 cross-subject HPO 选出的 backbone_lr = 1.3e-4 的固定优化栈**时，没有预训练先验的随机初始化难以收敛到具备判别力的解；~16K 参数的 EEGNet 凭借更小的搜索空间在被试内训练上仍能稳定收敛。**关于 within ternary 18 / 21 chance-collapse 的成因**，作者本人在 [`docs/handoffs/2026-05-09_random_init_ablation.md`](../../docs/handoffs/2026-05-09_random_init_ablation.md) L186-210 中基于 train_loss 轨迹分析给出的概率估计为：**(i) 数据量 / 过参数化导致 saddle-lock（结构性、与 LR 量级关系弱）70–80%；(ii) LR + patience + warmup 调优可救回 ≥ 5 个塌陷被试 15–25%；(iii) LR 是主因、提高 LR 可让 ≥ 10 / 18 塌陷被试学到 < 5%**。本研究的论证依赖 (i) 主导这一假设，但 within ternary 高 LR + 长 patience 的 retry 实验（~25 min GPU；handoff L212-227 描述方法）尚未执行，因此 "from-scratch transformer 在 ~70 trial 上结构性失败" 与 "当前 HP 配置下表现远低于其潜在能力" 在本研究中无法被严格区分。该现象与 NLP 文献中 transformer 在小样本上的已知微调脆弱性（Mosbach et al. 2021 [21] ICLR 在 RTE ~2K 样本上 BERT-base ~1/3 random seed 落入 chance）方向一致；此处的更深文献定位由相邻评审章节处理。基于这一综合判断，预训练表征**在本研究 HP 下方向性扮演**数据稀缺时的归纳偏置补偿角色，但精准量化"无 HP 错配下 TUEG 预训练在被试内的真实贡献"仍需 §6 #8 描述的 random-init 专属 HPO 完成后才能给出。

**XSI-FT ceiling 在两种 init 下独立成立**：random-init cross→XSI-FT 的 Δ 为二分类 −0.12 pp（86.34% → 86.22%）、三分类 +0.37 pp（73.06% → 73.43%），与 §3.3 原始 weights 条件下的 −0.56 / +0.20 pp 模式一致——两条独立路径（pretrained vs from-scratch）均未能让 XSI-FT 超越对应的 cross-subject baseline。这一双重独立证据支持 §3.3 的 ceiling 解释（任务 × cohort × 通道密度共同决定上限），并排除"ceiling 是 TUEG 预训练 backbone 过度正则化的副作用"这一替代假设。

需要明确的是，本消融**仅切换 backbone init，没有做 random-init 专属 HPO**；HP 与 original-weights baseline 完全共享（`get_default_config()`），故 random-init 的两段式差距（within ~−27 pp、cross/transfer ~−3 pp）严格而言应被理解为"**在 original-weights HP 下的 random-init 观察结果**"，而非"random-init 经独立 HPO 调优后的最优表现"。该 HP 错配在 within-subject 范式下可能尤为显著——`get_default_config()` 选出的 backbone_lr = 1.3e-4 来自 cross-subject 21 × 训练数据规模上的 HPO 全局最优（Table S5b cross-subject 行），用到 ~70 trial 单被试 + from-scratch transformer 上时的次优程度无独立度量。cross-subject 与 XSI-FT 的 random-init 缺口已小到 −1.6 至 −4.3 pp，**独立 HPO 即便能进一步弥合该缺口也难以翻转 within / cross 的两段式差异结构**这一定性观察仍可成立，但 within −23 至 −31 pp 内"HP 错配 vs 数据稀缺 saddle"的相对贡献无法在本节闭合；闭合需要 §6 #8 描述的 random-init 专属 HPO（≥ 25 trial Optuna，覆盖 backbone_lr 1e-4 ~ 5e-3 对数均匀、warmup、patience、layer-wise LR）。此外，random-init 训练实际比 original-weights 更早 early-stop（wrapper 总时长 2h 13m vs 估计 9–13h），训练集快速过拟合（train acc 升至 0.95+ 时 val 已高位震荡），与"更小搜索空间下更易过拟合"的预期一致。

> **数据来源**: random-init runs `20260508_2338` (cross binary), `20260509_0014` (cross ternary), `20260509_0047` (within binary), `20260509_0102` (within ternary), `20260509_0124` (XSI-FT binary), `20260509_0135` (XSI-FT ternary)；JSON cache 路径与单被试明细见 `docs/handoffs/2026-05-09_random_init_ablation.md`。
> Original-weights baseline: ExperimentDB run_tag `20260323_2237` (within binary), `20260323_2320` (within ternary), `20260324_0023` (cross binary), `20260324_0109` (cross ternary), `20260329_0507` (XSI-FT binary), `20260329_0521` (XSI-FT ternary)。
> EEGNet baseline: ExperimentDB run_tag `20260316_1411` (within binary), `20260329_0056` (within ternary), `20260330_0709` (cross binary), `20260330_0735` (cross ternary), `20260507_1835` (XSI-FT binary, 无 baseline 标记), `20260507_1913` (XSI-FT ternary, 无 baseline 标记)。

#### 3.7.3 综合：架构 / 预训练 / 容量复合贡献的探索性观察

合并 §3.7.1 与 §3.7.2 在 cross-subject binary 上的四个锚点，可观察到 CBraMod 相对 EEGNet baseline 的 +14.01 pp 优势沿以下相邻 Δ 跨越（**所有 Δ 在共享默认 HP 与受限 HPO 预算下的复合估计；严格独立 HPO 留待 §6 #8**；本节观察到的 EEGNet 内扩参单调退化方向上与 LLM 领域 Hoffmann et al. 2022 [22] compute-optimal 假说提示的 "over-parameterization 在固定算力 / 数据预算下退化训练效率" 一致，但 EEG decoding 尺度下的 scaling laws 在本研究规模 (≤ 30M, ≤ 21 被试) 之外尚未系统建立）：

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

**关于本节观察的解读边界**：CBraMod 与扩参 EEGNet 之间的 cross-subject gap 至少包含架构、预训练、容量三种贡献，本研究在受限 HPO 预算 + baseline → Mid 双轴扩参 + random-init 共享 HP 的三项约束下**无法独立分离**这三种贡献的各自贡献值。可被本节探索性观察方向性支持的较弱主张是：(a) **TUEG 预训练在被试内贡献巨大（binary +23.10 / ternary +30.79 pp），在 cross-subject 与 XSI-FT 仅贡献 +1.6 ~ +4.3 pp**——这是本节归因强度最高的一组 Δ；(b) **沿当前扩参轴扩参 EEGNet 在 cross-subject 范式下方向性有害**（baseline → Huge v3 沿双轴下降 −25.30 pp）——但 "EEGNet 内扩参普遍有害" 与 "EEG decoding 瓶颈不在容量" 等更强主张需要 §6 #8 + §6 #6 的独立 HPO 与单轴隔离实验完成后才能确立；(c) **transformer + ACPE 架构在不依赖 TUEG 预训练时仍能在 cross-subject 21 × pooled 数据上学到有效表征**（random-init CBraMod cross 86.34% vs EEGNet baseline 76.67%, +9.67 pp）——但与 EEGNet-Huge v3 的 +34.97 pp 差距是复合估计，不可独立归因到 "架构"。

这一**范式依赖的复合贡献结构**与 §4.1 "基座模型价值随数据约束放大" 的叙事方向一致：在 cross-subject 范式（21 × 训练数据）信号充足时，random-init CBraMod 仍领先扩参 EEGNet；在 within-subject 范式（每被试 ~70 trial）信号稀缺时，TUEG 预训练贡献急剧扩大、random-init CBraMod 反而输给 EEGNet baseline。但因前述三项 HPO / 扩参非对称性，该结构的**精确归因强度**应被理解为方向性而非独立可定量分解的；详细归因需要 §6 #8（EEGNet-Huge ≥ 25 trial 独立 HPO + CBraMod random-init ≥ 25 trial 独立 HPO）的算力开支后才能闭合。

### 3.8 推理性能

实时 BCI 部署存在两种典型场景：(i) **单用户场景**（个人 BCI 设备）下用户独占模型，相关指标是 batch=1 端到端延迟；(ii) **多用户共享服务场景**（服务器侧 BCI 推理服务）下一个 GPU 通过 batching 同时服务 N 个用户，相关指标是 batch=N 的端到端延迟（即每个用户从发请求到拿结果的最坏延迟，受同 batch 内其他用户拖累）以及每用户的平均 GPU 占用时间。表 17 同时报告这两个视角。需要强调的是，多用户共享服务**只对 CBraMod 适用**——§3.2 已显示 EEGNet 在跨被试 pooling 下方向性受损（−1.43 pp, p = 0.456），不存在合理的"21 名被试共享一个 EEGNet 服务"用例；EEGNet 列在表中仅作单用户延迟对照。

**表 17. 推理延迟与吞吐量（128 通道二分类，NVIDIA RTX 5070）。**

| Batch Size | 模型 | Mean Latency (ms) | Throughput (samples/s) | Per-User GPU Time (ms) |
|:----------:|:-----:|------------------:|-----------------------:|-----------------------:|
| 1 | EEGNet | **0.375** | 2,665 | 0.375 |
| 1 | CBraMod | 12.919 | 77 | 12.919 |
| 8 | EEGNet | 0.542 | 14,756 | 0.068 |
| 8 | CBraMod | 12.582 | 636 | 1.573 |
| 32 | EEGNet | 2.058 | 15,547 | 0.064 |
| 32 | **CBraMod** | 32.729 | **978** | **1.023** |
| 64 | EEGNet | 4.027 | 15,894 | 0.063 |
| 64 | CBraMod | 71.110 | 900 | 1.111 |

> Per-User GPU Time = Mean Latency / Batch Size，等价于 1/Throughput 的毫秒形式；含义是"摊到每个用户身上的 GPU 占用时间"。

图 11 以对数坐标柱状图和模型规模对比直观呈现了这一结果。

**图 11. 推理延迟与模型规模对比。** 左图：不同 batch size 下两种模型的延迟（对数坐标），红色虚线为 100ms 实时阈值。右图：CBraMod/EEGNet 的参数量、FLOPs、模型大小、延迟倍率。

![图 11. 推理延迟对比](../figures/inference_latency.png)

**单用户场景**：batch=1 下 CBraMod 单样本延迟 12.9 ms，远低于实时 BCI 的 100 ms 阈值（~7.7× 余量）；EEGNet 以 0.375 ms 实现极致实时性。两种模型均满足单用户实时部署。

**多用户共享服务（仅 CBraMod 适用）**：在 batch=64 下 CBraMod 端到端延迟为 71.1 ms——仍低于 100 ms 实时阈值——意味着**单张 RTX 5070 可同时服务 64 名用户而每名用户仍获得 <100 ms 响应**；峰值显存仅 537 MB（4.5% of 12 GB），余量充足。每用户 GPU 时间从 batch=1 的 12.92 ms 降至 batch=32 的 1.02 ms（**~12.6× compute 缩减**），与 EEGNet batch=32 的 0.064 ms 仍有 16× gap，但与 batch=1 下 34× gap 相比已大幅收窄；该缩减来自 GPU 并行计算的更高利用率与 kernel launch 开销摊薄，与 batch 内的 transformer attention 矩阵乘法天然适配 GPU 张量核。

总结：单用户视角下 CBraMod 已满足实时性；多用户共享服务视角下 CBraMod 同样在 batch=64 内保持 <100 ms 延迟，并且每用户 compute 开销随 batching 显著降低，使大规模 BCI 云服务部署在硬件成本上变得可行。

> **数据来源**: `docs/dev_log/experiments/inference_benchmark_analysis.md`（数据采集 2026-03-23）
> 生成命令: 图 11 由 `uv run python scripts/paper/generate_paper_figures.py --figure inference_latency` 生成

### 3.9 数据质量与被试异质性

三名重度伪影被试（S04, S10, S14）的振幅超过群体最大值的 3–8 倍（126K–307K µV vs. 正常 ≤ 38K µV），时间漂移值高出群体均值数个数量级（S04: 2,717 vs. 群体均值 ~30）。这三人在跨被试 binary 任务上的表现差异极大：S04=98.12%、S14=87.50%、S10=66.25%——同样"重度伪影"标签下相差 32 pp，提示"重度伪影"并非单一类别。S04 的 1024 Hz 原始振幅显示为 episodic 大幅 spike（疑似 EMG 串扰）+ 低频漂移，trial 级 ±500 µV 阈值剔除后剩余信号可能仍承载有效手指 MI 模式；S10 主导的是持续性高方差噪声，全程被噪声淹没。机制层面，跨被试模型从 18 名干净/轻度被试身上学到的群体表征仍能 generalize 到 S04 的偶发性高质量片段，但对 S10 的持续性噪声无能为力。

**Sensitivity check（leave-S04/S10/S14-out, N = 18）**：去除三名重度伪影被试后重新训练 cross-subject CBraMod，binary 测得 90.62% ± 8.18%（vs N=21 90.68%，Δ = −0.06 pp），ternary 测得 74.75% ± 13.74%（vs N=21 74.88%，Δ = −0.13 pp）——两者 |Δ| 均远小于 1 pp。结论：三名重度伪影被试在跨被试群体均值上的影响处于统计噪声范围内，**主要 finding 不依赖于其包含与否**。这一结果与 §3.8 关于"S04 高准确率主要源自偶发性高质量片段而非伪影模式"的解释一致——若模型真的在系统性利用伪影，去除三人本应让群体均值显著下降。

> **数据来源**: leave-3-out binary `20260505_0116`: `results/sensitivity_leave3out/20260505_0116_cross_subject_cache_imagery_binary.json`; leave-3-out ternary `20260505_0145`: `results/sensitivity_leave3out/20260505_0145_cross_subject_cache_imagery_ternary.json`

**Label-shuffle control (P0.3)**：作为 cross-subject 90.68% headline 数字的第三重 robustness 防线，我们对 21 名被试 cross-subject CBraMod binary 做 within-subject trial-level 随机重排 label（保持 input EEG 不变、保留每被试类别平衡）重新训练 n=2 seeds。**结果**：seed=42 测得 49.17% ± 4.08%（best epoch=23, 33 epoch 早停），seed=123 测得 50.00% ± 0.00%（majority-class collapse, best epoch=1, patience 耗尽即停）；pooled 均值 **49.58%** 落在 Scenario A 接受带 [48%, 52%] 中央。相对真实 label 的 90.68% headline，**Δ = −41.1 pp**——远超 ±5 pp Scenario A 接受阈，强证据排除三类潜在 shortcut leakage：(i) train/test split 残留泄露（任何残留泄露应在 permutation 后存活）；(ii) subject-identity 混淆（within-subject shuffle 保留被试身份但销毁 label 语义）；(iii) trivial label 统计 prior 蒙混。两个 seed 通过不同失败模式（seed=42 训练 33 epoch 后 patience 耗尽、seed=123 epoch 1 即 majority-class collapse）独立落到 chance level，进一步证实 shuffled labels 不存在可被泛化的信号。本控制与 §3.5.3（4ch 负控制 67.65% vs chance 50%，方向性反证通道选择独立 leakage）+ §3.9 leave-3-out（重度伪影被试去除对群体均值仅 −0.06 / −0.13 pp 影响）共同构成 cross-subject headline 的三重 robustness 证据链。

> **数据来源**: seed=42 `results/20260510_1847_labelshuffle_seed42_cross_subject_cache_imagery_binary.json`（ExperimentDB run_tag `20260510_1847_labelshuffle_seed42`）；seed=123 `results/20260510_1914_labelshuffle_seed123_cross_subject_cache_imagery_binary.json`（ExperimentDB run_tag `20260510_1914_labelshuffle_seed123`）；handoff [`docs/handoffs/2026-05-10_p03_label_shuffle_results.md`](../../docs/handoffs/2026-05-10_p03_label_shuffle_results.md)

---

## 4. 讨论

### 4.1 基座模型优势：何时与为何

从方法学定位看，本文不是 [3] 的在线机器人控制复现，也不是 [4] 的通用 benchmark 复刻，而是将 [3] 的 finger-level dataset/session design 与 [4] 的 pretrained foundation model 结合到统一的离线、held-out-session 评估框架中。因而，下述模型差异更适合被解读为“在同一数据与相同 split 约束下，预训练基座模型相对 compact CNN 的收益”，而不是对在线 robotic control 或 [4] 全任务基准的直接替代。

CBraMod 在所有实验条件下一致优于 EEGNet——被试内 **+7.05 pp**、跨被试 **+14.01 pp**（128ch）、32 通道 **+10–13 pp**——这反映了大规模预训练对 EEG 解码的价值。~400 倍的参数量差异本身不能完全解释该差距，§3.7 报告的两项探索性消融（EEGNet 容量阶梯 + random-init CBraMod）对该差距的来源做了初步检验。一个朴素担忧——"差距是否仅源自 ~16K vs 30.48M 的容量量级差异"——由 §3.7.1 在受限 HPO 预算下方向性回答：把 EEGNet 沿 (conv stem, MLP 头) 双轴扩展到 1.90M / 5.84M / 19.99M / 30.22M 四档，**cross-subject 准确率从 76.67% 单调下降到 50%（chance）**，30M 量级在两套人工调试 HP（≤ 2 trial）下均落入 train loss 死锁。**在本研究 HPO 协议下** ，沿当前扩参轴对 EEGNet 扩参对 cross-subject 准确率不利；但 "EEGNet 内扩参普遍有害" 或 "EEG decoding 的瓶颈不在容量" 等更强主张需要 EEGNet-Huge v1/v2 的独立 ≥ 25 trial Optuna sweep（详见 §6 #8）确认仍不可训之后才能成立。值得注意的是，作者在交接文档 [`docs/handoffs/2026-05-09_eegnet_huge.md`](../../docs/handoffs/2026-05-09_eegnet_huge.md) L156、L195-197、L249-260 中明确诊断 v1/v2 的不可训为 "BF16 + 深 MLP 头需 LayerNorm" 的优化栈兼容性问题——v3 加 LayerNorm + 缩 MLP 后立即 trainable 是直接证据；因此 v1/v2 的失败更可能是工程层面的 trainability 问题，而非参数容量本身的根本饱和。

**同规模 random-init 对照的探索性观察**：在 ~30M 参数 + 无预训练的同等条件下，EEGNet-Huge v2 (30.22M) cross 50.07%（chance）vs CBraMod random-init (30.48M) cross 86.34%——观察 ~+36 pp 差距；EEGNet-Huge v3 (5.84M) cross 51.37% 对 random-init CBraMod 时差距为 ~+35 pp。**在 EEGNet-Huge v1/v2/v3 与 CBraMod random-init 均未做专属 HPO 的对照下**，该差距是 "backbone 架构差异 + EEGNet 优化栈不稳定 + random-init CBraMod HP 错配" 的复合估计；归因到 backbone 架构本身的下界尚不能从本节单独给出。其上 TUEG 预训练 Δ = +4.34 pp（86.34% → 90.68%）是本对照中归因强度最高的 Δ——因 random-init 与 baseline 共享同一 `get_default_config()`，唯一变量是 backbone init。

within-subject 范式下方向反转：random-init CBraMod 在被试内二分类与三分类上分别落到 62.05% 和 38.65%，比 original-weights 分别低 **binary −23.10 pp、ternary −30.79 pp**（且 within ternary 21 名被试中 18 名测试准确率落在 chance ± 2 pp 区间，seed = 1234 重跑得 17 / 21，证实非 seed 特例）；不仅如此，random-init CBraMod 在该范式下反而输给 EEGNet baseline（binary 78.10%、ternary 66.81%）约 −16 至 −28 pp。然而 §3.7.2 caveat 已指出 random-init 复用 cross-subject HPO 选出的 backbone_lr = 1.3e-4，该 HP 在 ~70 trial single-subject from-scratch transformer 上的最优性未被独立验证；handoff 作者本人对 within ternary collapse 的概率归因为 70-80% saddle-lock / 15-25% LR-schedule、< 5% 纯 LR 主因。这一非对称方向性提示 TUEG 预训练**扮演数据稀缺时的归纳偏置补偿**角色：cross-subject pooling (~21× 训练数据) 信号充足时，random-init CBraMod 仍领先 EEGNet baseline +9.67 pp；within-subject (~70 trial) 信号稀缺时，TUEG 预训练贡献急剧扩大。

引用本节数字时，**摘要 / §1.4 / §7 Finding 1 显式列出 binary +23.10 / ternary +30.79 pp 双值**，不再使用 ~+27 pp 平均值（该平均会模糊任务难度差异）；cross-subject 与 XSI-FT 范式下 TUEG 预训练贡献为 +1.6 ~ +4.3 pp（双任务双范式区间）。三向分解（架构 / 预训练 / 容量）的精确归因强度需要 §6 #8（EEGNet-Huge + random-init CBraMod 双侧 ≥ 25 trial 独立 HPO sweep）的算力开支后才能闭合；当前章节支持的较弱主张是 "TUEG 预训练在被试内贡献巨大、在 cross-subject 与 XSI-FT 仅 +2 ~ +4 pp" + "transformer + ACPE 架构在不依赖 TUEG 预训练时仍能在 cross-subject 21 × pooled 数据上学到有效表征"，更强的独立可归因分解超出本研究证据范围。

值得注意的是，EEGNet 未从跨被试数据池化中显著获益（78.10% 被试内 vs 76.67% 跨被试，−1.43 pp, p = 0.456），而 CBraMod 增益 +5.53 pp。这提示基座模型的预训练表征使其能够更有效地整合异质跨被试数据。EEGNet 反而是从 XSI-FT 中获益的那一方（128ch XSI-FT +5.38/+5.10 pp, p < 0.01, §3.3）：cross-subject pooling 的 21 名被试异质分布让 EEGNet 学不动，但 XSI-FT 的单被试 fine-tune 阶段给它一个具体的目标分布去对齐。这一非对称（CBraMod 偏好 cross-subject、EEGNet 偏好 XSI-FT）从模型容量角度可统一解释——大容量基座能直接吸收异质群体分布，小容量 CNN 必须在 cross-subject 阶段先抽出"群体共享 spatial filter"作为初始化、再在 fine-tune 阶段重新校准到单被试。

### 4.2 最优通道配置与部署

32 通道 FDR 配置是本研究评估范围内最稳健的精度-硬件权衡点：

| 属性 | 值 |
|------|-----|
| 性能保留率 | **96.7%**（87.71% vs 90.68%） |
| vs. 61ch 标准 10-10 | 仅差 1.84 pp，通道数减半 |
| vs. 64ch FDR (89.46%) | 仅差 1.75 pp，通道数减半 |
| 硬件兼容性 | 与商用 32 通道 EEG 系统兼容 |

64ch 全 5 method 数据点（2026-05-11 矩阵闭合）填补了 32→128ch 之间的中间档位空白：从 32ch FDR 到 64ch FDR 仍有 +1.75 pp 边际 binary 增益，从 64ch 到 128ch 再 +1.22 pp；ternary 上 32→64ch FDR +4.33 pp 但 64ch 75.12% 与 128ch baseline 74.88% 在 run-to-run noise 内一致（详见 §3.5.2）。"32ch 已饱和"的强表述在 binary 上不成立——32→64ch 之间仍可恢复 ~一半的剩余性能空间，但每翻一倍通道的边际增益已落到 1–2 pp 区间，硬件成本与设置时间的边际成本通常超过这一性能收益。综合来看，32ch FDR 仍是部署最优选择，64ch FDR 适合追求 ~89% binary / ~75% ternary 而硬件预算更宽松的场景。

**64ch 上方法不敏感性已验证**：64ch 5 method（FDR / Band Power / Attention / CSP / negative_control）binary 范围 86.22–89.46%（spread 3.24 pp），ternary 范围 73.35–75.44%（spread 2.09 pp）——量级与 32ch（2.77 / 2.08 pp）一致。即原"32ch 起方法选择对性能影响相对不敏感"的论断**维持到 64ch**。在 32ch / 64ch ternary 上，数据驱动方法与 negative_control 的差值 ≤ 0.32 pp（well within 21 名被试 std ≈ 13 pp），统计上不可区分；在该 (通道, task) 组合下"用数据驱动方法选择最优 32 / 64 通道"与"选择未被任何方法选中的 32 / 64 通道"性能等价（详见 §3.5.3 / §4.3）。**16ch 档位（2026-05-13 sweep）现已纳入并定位方法依赖临界 boundary**：5-entry spread 跃升到 binary 8.69 pp / ternary 7.64 pp，相比 32ch 的 2.77 / 2.08 pp 跳升约 3–4 倍，**method-agnostic 区间在 32→16ch 之间崩溃**（详见 §3.5.2）；同档 negative_control 与最优方法的差距亦从 32ch 上的 ≤ 0.32 pp 放大到 16ch binary 上的 3.63 pp / ternary 上的 4.94 pp——即"低 method-overlap 配置 ≈ 数据驱动配置"的等价性**严格限制在 ≥ 32ch 通道档**。本研究仍未评估 96ch 档位，因此"电极数量 scaling 在 64ch 以上完全饱和"仍属未验证细节。

低密度区间（≤8ch）的部署门槛同样被本批 4ch BP 实验放宽：原 v2 草稿建议的"部署阈值 8ch"基于 4ch 标准方法均失效；引入 4ch BP (78.75%) 后，**在本数据集与本任务范围内 4 通道 Band Power 是可行的极简部署候选**（保留 86.8% 的 128ch 性能）。这把可部署谱系在本研究的具体配置下从 {128, 64, 32, 8} 扩展到 {128, 64, 32, 8, 4}——仍以 32ch FDR 为推荐起点；极端低成本场景下 4ch BP 是一个值得在新部署 cohort 上独立验证的候选，而非已确立的通用兜底方案。

### 4.3 体积传导与信息冗余

控制实验（Section 3.5.3）揭示了高密度 EEG 的一个基本属性：由于体积传导，皮层源的电信号在头皮上广泛传播，产生了大量信息冗余。4 通道负控制（binary 67.65% / ternary 53.37%）表明，即使是未被任何方法选中的通道，在预训练基座模型下也能显著超越随机水平。在 32 通道级别，五种方法之间仅 2.77 pp 的窄 binary 性能差异、2.08 pp 的 ternary 性能差异证实了广泛的冗余——这一性能层面的"方法不敏感"在空间布局上对应着"大体不重叠的通道集合"：5 配置两两 Jaccard ∈ [0.12, 0.23] 的量化证据见 Figure S6a。

新增的 32ch / 64ch ternary 矩阵为体积传导冗余提供了更强的实证证据：**在 32ch+ ternary 上，数据驱动方法与"未被任何方法选中"的随机通道在性能上统计不可区分**——32ch ternary negative_control 72.38% vs 该档最优数据驱动方法 BP 72.20%（Δ=+0.18 pp），64ch ternary negative_control 75.44% vs FDR 75.12%（Δ=+0.32 pp）。两个 Δ 均远小于 21 名被试间 std ≈ 13 pp，paired 比较下无显著差异。换言之，**"选择被某个数据驱动方法识别为最重要的 32 / 64 通道"与"选择不被任何方法看中的 32 / 64 通道"在 ternary 任务上提供的判别力等价**。这把 §3.5.3 的"4 通道负控制超越随机"弱论断升级为"32ch+ ternary 任务下方法识别行为对最终性能没有信号增益"的强形式。需要 disclose 的是 32ch / 64ch negative_control 注册表实质是"31 pure-complement + 1 pad" / "4 pure-complement + 60 pad"（详见 §3.5.3 末段 Caveat），所以这一论断的严格形式是"低度 method-overlap 配置 ≈ 数据驱动配置"，而非"纯互补 ≈ 数据驱动"。64ch neg_ctrl 因 pure complement 仅 4 通道、60 通道来自 method union pad，**不能单独作为"纯互补通道仍然有效"的论据**；32ch neg_ctrl 仅 1 通道 pad，弱化程度有限，仍可作为体积传导论证的辅助证据。Binary 任务上 32ch+ neg_ctrl 也与最优方法接近（32ch neg_ctrl 84.08% vs FDR 87.71% 差 3.63 pp，64ch neg_ctrl 88.57% vs FDR 89.46% 差 0.89 pp），但 binary 差距比 ternary 略大——可能反映 binary 任务对空间精度的要求高于 ternary（后者多一个 rest 类，rest 检测对通道选择鲁棒性更高）。

### 4.4 纵向数据扩展：突破 Session 平台期

原始数据集论文 [3] 在在线 base/fine-tuned EEGNet 设置下报告：被试性能在 2–3 个 session 后趋于平台期。本研究的 N = 16 离线分析回答的是一个更弱、也更可控的问题：如果去掉实时反馈与 same-day update，仅保留累积数据量增长，模型是否仍能从额外 session 中持续获益？结果显示答案是肯定的，但**收益取决于更新发生在何处**。在单被试更新框架下，被试内重训练和 XSI-FT 都获得了显著增益（CBraMod 分别 +6.13 pp, p = 0.007 和 +5.70 pp, p = 0.015）；而在 pooled cross-subject 框架下，额外同被试 trial 只能带来极小提升（+0.86 pp, p = 0.662）。这说明新增数据的关键信息主要是被试特异性的，需要通过个体化更新才能充分吸收。

对 cross-subject 和 XSI-FT 而言，收益没有进一步放大的另一个原因，是模型都要同时处理两层分布错位：新增数据来自**新的 session**，因此包含时间漂移、疲劳、接触阻抗变化等跨 session 异质性；而初始化或训练底座又来自**多被试 pooled 分布**，因此天然带有跨被试异质性。两层异质性叠加后，新增样本的一部分作用会先被用于对齐分布，而不是直接提升分类边界，这也解释了为什么它们的增益弱于纯被试内重训练。

标准差从 10.81% 压缩至 5.98%（−45%）具有实际部署意义：BCI 系统需要跨用户的一致性能，而非少数用户的峰值表现。额外 session 数据不仅提高了平均水平，还将"最差情况"显著抬升。

补充分析中的 fixed_sess02 策略下，EEGNet (+8.51 pp) 增益约为 CBraMod (+4.38 pp) 的两倍，但如 §3.4.3 已指出，这一差距同时兼容两种解释——天花板效应（CBraMod baseline 87.23%，增益空间本身受限）与时间分布敏感度差异——本研究的样本量 (N = 16) 不足以区分。一个可能的进一步验证是在更长 session 跨度下做线性外推：若 CBraMod 即使从更低 baseline (e.g., 缩减通道下 ~78%) 开始仍快速饱和，则可归因于漂移敏感度；若与 EEGNet 同步提升，则前者主因为天花板。当前数据无法做出该判断。

### 4.5 领域自适应 Further Pre-training 的局限

§3.6 把 DAPT 的下游表现从 v3 草稿的"三种配置一致负迁移"重写为 **task-asymmetric × paradigm-dependent**：DAPT 评估全闭合至 30 cell 后，**binary 任务**在 cross / within / transfer 三个 paradigm 上均 5/5 一致显著负向（cross Z=−5.33 / within Z=−4.42 / **transfer Z=−3.39**，全部 p<0.001；15/15 binary cell 方向负），而 **ternary 任务**呈现 paradigm-依赖的方向不一致（cross 4/5 弱正，within 5/5 弱负，**transfer 3/5 弱正**——其中 V3 transfer-ternary +1.09 pp 是 30-cell 矩阵的全局最大正向 Δ）。本节以"机制收紧"的视角解释这一分裂：V4 与 V5 两次 surgical fix 把 v3 草稿提出的三个候选混淆假设（域错配 / Stieger 占主导 / 通道数异质）逐一筛除，唯一存活的解释是 **MI 粒度错配（pretext-task granularity mismatch）**。

**Surgery 1 — V4 把"域错配 + 数据净化"双管齐下**：选取与下游 finger MI 域最接近的 3 个数据集（Cho2017 / Ofner2017 / Schirrmeister2017，去除 Stieger），并应用 strict filter（300 µV peak + per-channel kurtosis>10）替代 basic 500 µV mean-abs，达到全 5 V 中最低的 pre-train loss 0.001914。结果：cross-binary Δ=−1.61 pp（p=0.008, q=0.048, BH 显著），cross-ternary Δ=+0.22 pp（n.s.）——**域对齐 + 数据净化双管齐下仍未救援 binary**。说明 (1) 域错配是必要但非充分原因；strict filter 本身没有把 binary 拉回正向。

**Surgery 2 — V5 把通道几何降到单一 60ch**：单源 Stieger 60-ch，直接消除"通道数异质混淆"假设。结果：V5 cross-binary Δ=−2.77 pp（5 V 中**最差**），cross-ternary Δ=−1.17 pp（5 V 中**唯一弱负**）——V5 在 binary / ternary 上**双向恶化**，**反方向证伪**了"通道数异质是混淆"假设。机制解释：单源 ACPE 在 Stieger 60-ch 几何上过拟合空间先验，下游 128ch fine-tune 必须从错位起点重新校准 ACPE；V1–V3 的 7 种通道数反而强迫 backbone 学 channel-agnostic 表示——**通道多样性在 DAPT 中是保护因子，不是 bug**。这与 v3 草稿原先把通道异质性作为"第三项结构性 caveat"的方向相反，需明确撤回。

**Surgery 3 — V3 已部分排除 Stieger 占主导**：V3 将 Stieger 占比从 ~79% 削减到 ~30%（其余 9 个外部数据集与训练超参数保持不变），cross-binary Δ 从 V2 的 −1.25 弱化到 V3 的 −1.46，cross-ternary 几乎不变（+0.44 → +0.62）——Stieger 主导**不是** binary 显著负向的主因。叠加 V4（完全去 Stieger）仍 −1.61 pp，可基本排除假设 (2)。

**收紧后的唯一存活假设——MI 粒度错配**：粗 hand/leg/upper-limb MI 的 MAE pretext loss 学到的是"哪个肢体在动"的低频空间包络；下游 finger-level binary（食指 vs 中指，**同手**）需要的是 DAPT 没学到的细粒度区别。Ternary 的 rest 类（不动 vs 运动）正好可以用 DAPT 学到的粗粒度空间包络识别——所以 ternary 没那么糟，部分配置（V1/V2/V3/V4）甚至轻微正向（mean Δ=+0.18 pp）。这一机制以一致的方式解释了 (a) task asymmetry（binary 需细判别，被错配伤害；ternary 可受益于粗判别，被错配影响小）、(b) V5 双向恶化（单源 ACPE 几何过拟合加重粒度错配的下游代价）、(c) per-subject Δ-of-Δ 显著（每被试的 binary Δ − ternary Δ pooled across 5 V，n=105，t=−5.16, p<0.001）。

这与 NLP 领域的 domain-adaptive pre-training（DAPT）经验形成有意义的对照。Gururangan et al. 2020 [20] 在 NLP 中证明 DAPT 在 source 与 target domain 语义临近时一致受益；本研究的负面结果不挑战该结论，而是把"language" 与 "EEG" 的 transfer 边界条件区分开——在 EEG 中 'domain' 由信号级特征（采样率、频段、电极配置、**任务粒度**）而非任务语义类别（"都是 MI"）定义。粗 MI 与 finger MI 共享语义但不共享信号粒度，DAPT 在 NLP 类的 "task-language 都对齐" 假设上不再成立。这与 §4.8 的"EEG foundation model 的 'domain' 边界由信号级特征定义"命题自洽，并把该命题从直觉上升为 **5 个 surgically-distinct DAPT 变体共同支撑的实证结论**。

**评估范围更新**：原 v3.1 草稿中标记的 V1/V2/V3 transfer 6 cell 已于 2026-05-11 补完（详见 [paper/reviews/stage4_step1d_v1v2v3_transfer.md](../reviews/stage4_step1d_v1v2v3_transfer.md)），整体 DAPT 评估覆盖 30/30 cell。三个新数据点贡献了 §3.6.2 的反转 #4——**V3 transfer-ternary +1.09 pp 是整个 30-cell 矩阵的全局最大正向 Δ**（任意 task / paradigm），driving transfer-ternary 5V Stouffer 从 V4/V5-only 的 Z=−1.60 (p=0.110) **翻转**为 Z=+0.18 (p=0.860)，意味着 ternary 任务的 "DAPT 一致负迁移" 命题**不成立**。Binary 任务上 transfer Stouffer 反而从 Z=−2.79 加强到 Z=−3.39 (p<0.001)，binary 15/15 cell 全负的统一性证据进一步收紧 MI-粒度-错配的唯一存活假设。值得记录的另一观察是 **'transfer rescue gradient'**（详见 §3.6.1）：V1/V2/V3 在 binary 上呈强 cross→transfer 衰减（Δ magnitude 31–41% 衰减，显著性消失）；V5 部分衰减（衰减 56%，显著→边缘 n.s.）；**V4 是唯一未被 fine-tune 部分清洗的配置**（strict filter + 3-set 域对齐印下了最具体的错误先验）——这与 "fine-tune 清洗的难度与 surgical fix 的具体性正相关" 机制解释自洽，为 V4 的存活假设提供独立的方向性证据。综合而言：原 "DAPT 仅在 cross 失败" 的 Caveat #6 通过 30/30 cell 全闭合证据**在 binary 任务上支持关闭**（15/15 cell 方向负、3 个 paradigm Stouffer 全 p<0.001），在 ternary 任务上则**不支持 "一致负迁移"** 的更弱结论。BH-FDR 在 30-cell family 下 0/30 cell 存活——单元格 BH 已被 family 扩张惩罚到 conservative，**优先以 paradigm-level Stouffer 集体证据阅读**。

### 4.6 实际部署路线图

综合以上发现，本研究支持以下 BCI 部署路径：

1. **起步方案（推荐）**：CBraMod + FDR 32 通道配置（87.71% 基线准确率），兼容商用硬件；中端预算可上 64ch FDR (89.46%)；极简成本场景下 4ch Band Power (78.75%, +11.10 pp vs 负控制) 是一个值得独立 cohort 验证的候选，而非已确立的通用兜底
2. **个性化适配**：收集 2–3 个额外 session 数据即可突破 90% 准确率，低基线用户获益最大；XSI-FT 在 32ch FDR 配置下提供 +0.74 pp 增益（§3.5.4），在 8ch BP 配置下反而损失 −2.03 pp——即**XSI-FT 不应作为低密度部署的默认推荐**，需先确认所选 (channel, method) 组合是否仍有余量空间
3. **模型选择（按数据量）**：CBraMod 在 cross-subject 与高密度通道（≥32ch）下最优；EEGNet 在低预算/边缘场景且必须用 XSI-FT 时可作备选——但 EEGNet 在所有任务上均落后 CBraMod，不应作为首选
4. **领域自适应预训练**：直接使用 TUEG 预训练权重；只在存在类型更接近的 source MI 数据（手指级、手部精细动作 MI）可用时再考虑 DAPT，本研究使用的粗运动 MI 数据池不推荐
5. **实时可行性**：单用户场景下 CBraMod batch=1 延迟 ~13 ms，远低于 100 ms 实时阈值；多用户共享服务场景下 batch=64 延迟仍为 ~71 ms（满足 100 ms 阈值），每用户 GPU 时间降至 ~1 ms，使一张 GPU 可并发服务数十用户

### 4.7 伪影被试的影响

§3.9 已通过 leave-S04/S10/S14-out sensitivity check 证实三名重度伪影被试对群体均值的影响处于统计噪声范围内（binary Δ = −0.06 pp、ternary Δ = −0.13 pp）。从机制层面看，三人呈现两种性质（episodic spike vs 持续高方差噪声），提示 BCI pipeline 中"伪影类型"而非"伪影量级"决定模型行为；在临床部署中，应增加伪影类型识别（而非单纯阈值剔除）作为前置 module。

### 4.8 综合：数据稀缺梯度下的策略选择

将通道缩减、纵向数据扩展、DAPT 三条线整合，可以在"数据可得性"坐标轴上勾勒一条决策路径：

1. **零额外数据 + 高密度通道**：CBraMod 跨被试 pooled 模型（binary 90.68%）是最佳起点；EEGNet 在该范式下边际收益为负（−1.43 pp, p = 0.456），不适合 cohort-pooled 训练；但 EEGNet 在 XSI-FT 范式下反而获得 +5.38/+5.10 pp（§3.3，p < 0.01），构成"小模型必须借助 XSI-FT 才能从群体数据中获益"的证据。
2. **零额外数据 + 低密度通道（<32ch）**：32ch FDR 保留 96.7%、64ch FDR 保留 98.7%、8ch Band Power 保留 92.7%、**4ch Band Power 仍保留 86.8%**——可部署谱系比初版 v2 草稿（4ch 全失效）显著放宽。32ch FDR 配置下 XSI-FT 提供 +0.74 pp 方向性增益；但 8ch BP 配置下 XSI-FT 反而损失 −2.03 pp——这说明 **XSI-FT 收益取决于 cross-subject baseline 离 (channel, method) 容量上限的距离，而非通道数本身**：32ch FDR 距上限远（XSI-FT 有空间），8ch BP 接近上限（XSI-FT 反而引入过拟合）。低密度部署应先评估 cross-subject baseline 是否已饱和，再决定是否使用 XSI-FT。
3. **少量同被试新数据 (~1 session)**：XSI-FT (+5.70 pp) 与被试内重训练 (+6.13 pp) 终点接近，但 XSI-FT 起点更高，更适合冷启动用户；EEGNet 在 XSI-FT 下也获得 +5.38/+5.10 pp（p < 0.01，小但统计显著），适合极低算力部署。
4. **多 session 同被试 (3-5 sessions)**：被试内重训练达到 93.36% 全文最高终点；标准差从 10.81% 压缩至 5.98%——临床部署的"最差用户"承诺。
5. **外部域外数据 (~870h, 以 grasp/wrist MI 为主)**：DAPT 5 个独立配置（V1–V5）评估呈 **task-asymmetric 负迁移**——cross-subject binary 5/5 一致负向（mean Δ=−1.79 pp，Stouffer Z=−5.32, p<0.001），cross-subject ternary 4/5 弱正、仅 V5 弱负（mean Δ=+0.18 pp，Stouffer p=0.564）。V4 (3-set 域对齐 + strict filter) 与 V5 (Stieger 单源 60ch) 两次 surgical fix 把候选机制收紧到 **MI 粒度错配** 唯一存活假设，并反方向证伪"通道数异质"假设。本研究的负面结果不构成对 DAPT 范式本身的否定，但提示在 finger MI 任务上**source domain 的信号粒度对齐比任务语义类别更关键**——只在存在类型更接近的 source MI 数据集（如手指级、手部精细动作 MI）时才值得再尝试 DAPT；以粗运动 MI 为主的当前外部数据池在 CBraMod backbone 设置下不推荐（详见 §4.5 / §3.6）。

此外，Sup Table S5 的 fANOVA 显示 within-subject HPO 主导参数为正则化（phase_decay 23.3% / dropout 19.6%）、cross-subject 主导为 backbone_lr (66.8%)——两种范式下 CBraMod 的瓶颈本质不同：前者受过拟合限制，后者受 backbone 适配限制；这从优化角度印证了"范式选择即策略选择"。

贯穿这条路径的方向性观察是：本研究观察的负迁移与 NLP DAPT 文献中"低 task-corpus 对齐 + source corpus 不足"失败案例（Gururangan et al. 2020 [20] §5.2 reviews 域结果）在结构上一致；在 CBraMod backbone × masked-AE 预训练目标 × 粗运动 MI source pool × finger MI target 的具体配置下，通道几何错位（target 128ch vs source 95% 低密度）与训练超参数对 DAPT 结果的影响至少与任务粒度相当。下游 BCI 实践应优先匹配通道几何与信号尺度，再考虑任务语义对齐。判断 EEG 基座模型是否需要不同于 NLP/CV 的 transfer 设计原则，需要在多 backbone × 多 source corpus × 多预训练目标的矩阵下验证。

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
| 7 | **Foundation model 与预训练范围** — 主结果基于 CBraMod（masked autoencoding 预训练）。§3.7 通过 EEGNet 容量阶梯（16K → 30M）+ random-init CBraMod 双消融对架构 / 预训练 / 容量贡献做了**探索性初步检验**（在共享默认 HP、受限 HPO 预算、baseline → Mid 双轴扩参 三项约束下的方向性观察，不构成独立可归因的三向分解）。但其他基座模型架构（LaBraM 等）以及其他预训练目标（contrastive、predictive 等）尚未测试。 | "CBraMod + TUEG masked autoencoding"特定组合是否泛化到其他 backbone × objective 组合仍属开放问题；本研究的"基座模型价值随数据约束放大"方向性结论是否在其他 backbone 上重现需独立验证（§6 #7 / #8）。 |
| 8 | **Further pre-training 评估覆盖（部分闭合）** — V4/V5 已于 2026-05-10 补完 within + transfer 评估（共 8 cell），与原 V1/V2/V3 within+cross 12 cell 合并共 24 cell；剩余 6 cell 未跑全部为 V1/V2/V3 × XSI-FT × {bin, ter}。V4/V5 跨三种 paradigm 全部方向负、Stouffer 集体证据稳健，先验上不期望 V1–V3 transfer 反转方向。V2 训练亦在 Epoch 13 因 Windows LMDB MapResizedError 中断而非自然 early-stop。 | V4/V5 三-paradigm 全矩阵负向的事实部分回答了"DAPT 是否仅在 cross-subject 失败"——失败是跨范式稳健现象（详见 §3.6.4 Caveat #6 闭合）；V1–V3 transfer 严格意义上未被回答；V2 是否在更长训练后达到不同结论缺乏直接证据。 |
| 9 | **Stieger2021 主导效应通过 V3 实验部分验证，未做完整逐数据集消融** — V3（Stieger 占比 ~30%）相对 V2（~79%）平均改善 +0.68 pp，约恢复 V1→V2 阶段加剧负迁移的一半，但整体仍呈方向性负迁移（vs Baseline −0.70 pp 平均）。完整的 leave-one-out 数据集消融（逐数据集排除）尚未完成。 | 已能判断 Stieger 主导是 V2 阶段加剧负迁移的主因之一，但其余 9 个数据集的独立贡献仍未隔离；当前结论支持"两层归因"（Stieger 主导 + 整体粗运动 MI 域错配）。 |
| 10 | **Ternary 任务 baseline 时间不齐** — 三分类 baseline 来自 pre-HPO 运行（2026-02），与 binary post-HPO baseline（2026-03）不在同一管线版本下，引入 confound。 | Ternary delta 的精度估计弱于 Binary，但定性方向（一致负迁移）不受影响。 |
| 11 | **缩减通道下 XSI-FT 的全 (channel, method) 矩阵未完成** — §3.5.4 现覆盖 128ch / 32ch FDR / 8ch BP 三档：32ch FDR 下 XSI-FT +0.74 pp、8ch BP 下 XSI-FT −2.03 pp、128ch 下 −0.56 pp。Cross-subject baseline 在本论文 2026-05-11 矩阵闭合后已覆盖 {4, 8, 32, 64}ch × 5 method × {binary, ternary} = 40 cell（§3.5.2 表 9 / §3.5.3 表 10b），但 XSI-FT 仍只覆盖这 3 个 cell。三档样本不足以系统验证"XSI-FT 收益取决于 baseline 距容量上限"的解释框架；同档位下不同方法（如 8ch FDR、32ch BP）以及 4ch 档位下任何 method 的 XSI-FT 行为、以及全部 ternary 的 XSI-FT 行为均尚未测试。 | 解释框架基于三个数据点的归纳，可证伪但尚未充分检验；§4.8 决策路径在 4ch / 8ch 多方法组合下的精确度受限。Cross-subject baseline 矩阵闭合是该框架进一步检验的必要前置条件，现已就绪。 |
| 12 | **DAPT 训练配置的单次性 + V1–V3 transfer 评估缺失** — (a) V1–V5 均为单次 pre-training 尝试；V3 采用"先训 15 ep + warm-restart-from-weights 续训 12 ep"的两阶段策略（详见 §2.7.2 caveat），optimizer 与 LR scheduler 状态在阶段 ii 重置，与 V1/V2/V4/V5 的单阶段训练严格意义上不可同等比较。训练超参数（mask_ratio=50%、AdamW、warmup 0.5 epoch、恒定/cosine lr=5e-5）沿用 [4] 在 TUEG 上的下游 fine-tuning 默认值，未针对 MI 数据特性系统调参。(b) **V4 / V5 已覆盖三种 paradigm**（within + cross + transfer × bin + ter，2026-05-10 补完）；**V1 / V2 / V3 仍仅覆盖 within + cross 两种 paradigm，未运行 XSI-FT (transfer)**。即 5 V × 3 paradigm × 2 task = 30 cell 中实际评估 24 cell（V1–V3: 12 within+cross；V4/V5: 12 within+cross+transfer），剩余 6 cell 全部为 V1–V3 × XSI-FT × {bin, ter} 未跑；V4/V5 三-paradigm 全部方向负向支持先验"V1–V3 在 transfer 上不会反转"，但严格意义上 V1/V2/V3 transfer 仍属未回答。(c) **V4 同时变更"数据组成"与"过滤强度"**（3-set + strict filter），未运行 V6=V2 数据组成 + strict filter 以隔离过滤效应——当前结论"strict filter + 域对齐均未救回 binary"不可严格归因到单一变量。(d) **Stieger filter scope 不一致**：V4 三数据集均过 strict filter，V5 的 Stieger 仅过 basic filter（重处理 ~25h wall-clock 妥协）。V5 binary 显著恶化（−2.77 pp）的极小一部分可能受此 filter 不一致影响，但 V1/V2/V3 共用 basic filter 上 binary 也均负向，故这不是 V5 binary 恶化的主因。(e) **V1/V2 cross-subject 不在 ExperimentDB**：V1/V2 时期评估走 ad-hoc JSON cache 路径无双写 DB，本论文表 16 中的 V1/V2 t-test 是用 paper/analysis/further_pretraining_analysis.md 中记录的 per-subject acc + 当前 baseline 重算的，与 V3/V4/V5 走 DB 路径不完全对称。(f) **V4 small-data 警告**：V4 仅 4,937 段（Cho 1,135 + Schirr 3,310 + Ofner 492），Schirrmeister 占 67% 采样权重——"3-set 域对齐"实质偏向 Schirrmeister 主导（128ch 通道匹配下游，但属 motor execution 而非纯 imagery）。strict filter 让 Cho/Ofner 大幅减重的副作用，V4 binary 负向可能部分受此偏倚影响。(g) **24-cell BH-FDR 重做后survivor 退化**：在新 24-cell DAPT family 下重做 BH-FDR @ 0.05 后，原 v3.1 16-cell family 下的 3 个 survivors 仅 V2_within_binary (q=0.048) 仍存活；V1_cross_binary (q=0.072) 与 V4_cross_binary (q=0.072) 在更严苛的多重比较惩罚下不再 BH 显著。但 paradigm-level Stouffer 集体证据全部仍稳健（cross-bin Z=−5.32 / within-bin Z=−4.42 / transfer-bin Z=−2.79 p≤0.005，full v3.1 family Z=−4.83）——多重比较的功效损失不改变 task-asymmetric 定性结论。 | (a) 观测到的负迁移可能部分源于 DAPT 方法配置（mask ratio、loss 公式）与 MI 数据不匹配，而非纯粹反映外部 MI 数据的领域差异；分离两类成因需扫 mask ratio / loss / epoch 数等系统 ablation。(b) V1–V3 在 XSI-FT 范式上严格意义上未被回答；考虑 V4/V5 三-paradigm 全部方向负、within / cross / transfer 上 task-asymmetric 模式均成立，先验难以期望 V1–V3 transfer 反转方向，但补全属后续工作。(c) V6 缺失留待未来；(d) (e) (f) 三项 caveat 不影响 task-asymmetric 定性结论（5/5 binary 一致负 vs 4/5 ternary 弱正在 cross Stouffer 聚合下分别 p<0.001 / p=0.564；within / transfer 上 binary 同向负、Stouffer Z<-2.7），但弱化"V4 = pure 3-set domain alignment"与"V5 = pure single-cohort"作为干净因果隔离的强主张。(g) 单元格层面的 BH 退化提示读者优先关注集体证据（Stouffer）而非任一 single-cell 显著性。 |
| 13 | **EEGNet vs CBraMod 预处理管线不对齐** — 两模型使用不同的滤波带通（4–40 Hz vs 0.3–75 Hz）、采样率（100 Hz vs 200 Hz）和归一化（Z-score per-channel vs ÷100 全局缩放）；这是为各自模型架构和原训练管线分别选取的"近最优"配置（EEGNet 沿用 [3]/[5] 的标准 mu/beta 频带配置；CBraMod 沿用 [4] TUEG 预训练阶段的滤波/采样率约束）。 | 严格意义上 §3.1–§3.3 报告的"模型差异"是"模型架构 + 预训练 + 预处理管线"三因子复合估计，无法将 backbone 优势完全归因到"模型架构 / 预训练"——预处理管线本身可能贡献 1–2 pp 的独立效应。隔离方案需要交叉对调实验（EEGNet 用 200 Hz / 0.3–75 Hz / ÷100 vs CBraMod 用 100 Hz / 4–40 Hz / Z-score），属 §6 后续工作；当前结论的方向性不变（CBraMod 优势 ≥ 7 pp），但定量分解需此交叉实验闭合。 |
| 14 | **32ch / 64ch negative_control 不是纯互补配置** — 32ch `negative_control` 注册表为 31 pure-complement + 1 seed=42 pad（pad 来自 4 method union 的索引 29 / A30），以满足 `len(indices)==n_channels` 校验；64ch `negative_control` 为 4 pure-complement + 60 seed=42 pad（4 method 在 64ch 各选 64 后 union 已覆盖 124 通道，pure complement 仅余 4）。详见 §3.5.3 末段。 | 32ch neg_ctrl 1-channel pad 不影响其作为"低 method-overlap 配置"的论证价值（仍 31/32 = 96.9% 纯互补）；64ch neg_ctrl 60/64 = 93.8% 来自 method-union，实质是"低度 method-overlap 配置"而非"纯互补"。§4.3 体积传导论证中 64ch neg_ctrl 不可单独作为"纯互补通道仍有效"的论据，但 32ch neg_ctrl 与 4ch / 8ch neg_ctrl 仍可作为体积传导论证的主要证据。 |
| 15 | **缩减通道矩阵未做 per-channel-count HPO** — 2026-05-11 完成的 4×5×2 = 40 cell 矩阵全部共用 `configs/cbramod_v3_cross.yaml` 一套超参数（V3 pretrained checkpoint + cosine_annealing_warmup_decay + batch=256 + backbone_lr=1.3e-4），该超参由 128ch baseline 的 HPO 搜索得出（详见 §2.5.1 + Sup Table S5）。不同 channel-count（4 / 8 / 32 / 64）及不同 task（binary / ternary）的最优 LR / batch / dropout 可能不同——例如 4ch 可能受益于更小 batch、64ch 可能受益于更大 LR、ternary 可能受益于更长 patience。 | 矩阵内 cell-vs-cell 的对比（同 task 下的 method 间 spread、同 channel 下的 binary vs ternary 比较）仍然有效（共用 HP 提供公平的横向对比 baseline）；但 cell 的绝对准确率可能在 per-cell HPO 下进一步提升 1–3 pp 量级。§3.5.2 / §3.5.3 / §4.2 / §4.3 的方向性论断（rank flip、方法不敏感、BP 鲁棒、体积传导等）不依赖于绝对天花板，故 HP 共用不影响这些定性结论；但 §4.6 部署路线图中"4ch BP 78.75% / 60.67%"等绝对数字可能在专属 HPO 下小幅波动。 |

---

## 6. 未来工作

以下实验计划在后续研究中完成：

1. **运动执行范式验证**：使用同一数据集中的运动执行（Motor Execution）录制数据复制完整实验流程，检验 CBraMod 的优势是否跨范式持续，以及最优通道配置是否因范式而异。

2. **缩减通道下 XSI-FT 的全 (channel, method) 矩阵**：§3.5.4 现已覆盖 128ch / 32ch FDR / 8ch BP 三档，发现"通道越少 XSI-FT 收益越大"的简单假设被 8ch BP 反例推翻，并提出"XSI-FT 收益取决于 cross-subject baseline 距 (channel, method) 容量上限的距离"的修订框架。验证该框架需要补全 (channel, method) 矩阵：8ch FDR、32ch BP、4ch BP、64ch FDR 各自的 XSI-FT 是优先候选。

3. **DAPT 配置 ablation**：本研究的 DAPT 负迁移结论已通过 V3 实验完成"Stieger 主导效应 vs 整体域错配"的初步拆分（§3.6 / Limitation #9），剩余的"方法配置不匹配"成因（mask ratio、loss 公式、epoch 数、warmup schedule）尚需扫描 ablation 才能与"域内分布偏移"分离；同时单数据集 leave-one-out 消融（逐一排除 10 个外部数据集）可进一步定量化各数据集对负迁移的边际贡献。

4. **>64ch 中间档位与显著性检验**：本研究 2026-05-11 矩阵闭合后 {4, 8, 32, 64}ch × 5 method × {binary, ternary} = 40 cell 已全部跑通；2026-05-13 新增 16ch × 5 method × 2 task = 10 cell 将矩阵扩展到 **5×5×2 = 50 cell** 并定位 method-agnostic 区间的崩溃边界——16ch 5-entry spread binary 8.69 / ternary 7.64 pp，相比 32ch 的 2.77 / 2.08 pp 跳升 3–4 倍（详见 §3.5.2 / §4.2 / 图 3d）。"16ch 是否处于方法依赖临界 boundary 两侧"现已答（**16ch 即崩溃入口**）。剩余开放问题为 (a) **96ch 中间档位** —— >64ch 边际增益是否完全饱和仍未评估；(b) **method × channel × task ANOVA** —— paired ANOVA on 5 method × 21 subject for each (channel, task) cell 可定量验证"32ch+ 上 method 无显著效应"以及"16ch / 8ch / 4ch 上 method 显著"两端论断，本论文未跑显著性检验（图 3d 视觉证据已传达定性结论但缺 p 值硬声明）。

5. **4ch Band Power 的可复现性与跨范式稳健性**：4ch BP (78.75%) 是本批最大反例，但仅在 cross-subject binary 上观察到；其在三分类、XSI-FT、被试内、运动执行范式下是否同样保持优势需要独立验证。

6. **EEGNet 容量扩展沿 conv stem 单轴的隔离**：§3.7.1 沿 (conv stem, MLP 头) 双轴扩展了 EEGNet——baseline → Mid 的首跳同时改了 conv stem (F1: 16→32, F2: 64→256) 与 MLP 头（单 Linear → 双层 [1024,1024]）；Mid → v3 → v1/v2 三档则沿 MLP 头单轴扩参（conv stem 不变）。三档 MLP 单轴的 cross-subject 单调下降（57.65% → 51.37% → 50%）已可成立"在 F1=32 conv stem 之上 MLP 扩参无益"，但 baseline → Mid 的 −19 pp 一跳无法归因到单一轴。一项最小隔离实验是固定 MLP 头为单 Linear，扫 F1 ∈ {16, 32, 64, 128}（D=4 不变），观察 cross-subject 是否仍呈反向 scaling；若是，则可把"容量在 EEGNet 架构内一律有害"从 (F1=32, MLP 头) 单轴扩展为 (F1, MLP 头) 二维容量平面的全域结论；若 conv stem 扩参反而有益，则现有 baseline → Mid 的 −19 pp 主要来自 MLP 头（双层取代单层）的过拟合而非容量本身。预算 ~6 hr GPU。

7. **其他基座模型与预训练目标的独立验证**：§3.7 random-init ablation 已就 CBraMod 特定情境下"架构 vs TUEG masked autoencoding 预训练"的贡献完成初步剥离，但本研究的"基座模型价值随数据约束放大"结论是否在其他 backbone（LaBraM、LaBraM-base 等）和其他预训练目标（contrastive、predictive 等）上重现仍属开放问题。一项最小验证可在同一 finger MI 数据集上跑 LaBraM × {original-weights, random-init} 同样 6 个 condition 的对照，看 within / cross 两段式差距结构是否再现；若再现，则该机制可被升格为"EEG 基座模型的通用属性"而非"CBraMod 特异属性"。

8. **§3.7 探索性消融的严格独立 HPO 验证**：§3.7.1 EEGNet-Huge v1 (19.99M) 与 v2 (30.22M) 的不可训判定基于两套人工调试 HP（lr 相差 10×：5e-5 vs 5e-4；wd / dropout / LayerNorm on/off 等亦不同），并非独立 Optuna 搜索；§3.7.2 random-init CBraMod 直接复用 original-weights baseline 的 `get_default_config()`，没有跑专属 HPO。要让 §3.7.3 的复合贡献观察升格为可独立归因的三向分解，需补做：(a) **EEGNet-Huge v1 / v2 各 ≥ 25 trial Optuna TPE HPO**，搜索空间覆盖 LR ∈ [5e-5, 5e-3] 对数均匀、warmup ratio ∈ [0, 0.2]、LayerNorm on/off (categorical)、init scheme ∈ {Kaiming, Xavier}、dropout ∈ [0.1, 0.6]、weight_decay ∈ [1e-3, 0.3] 对数均匀；(b) **CBraMod random-init ≥ 25 trial Optuna 专属 HPO**，覆盖 backbone_lr 1e-4 ~ 5e-3 对数均匀、warmup、patience、layer-wise LR；优先 within ternary（最严重的 18/21 chance-collapse case）。预算估计 ~80–120 GPU 小时（v1 25 trial × ~2h + v2 25 trial × ~2h + random-init within ternary 25 trial × ~25 min + cross-subject 25 trial × ~30 min + 复跑 baseline 对照）。**预期 readout**：若 (a) 100% trial 仍落入 train_loss = 0.693 chance entropy 死锁（即便加 LayerNorm 也不救），则 §3.7.1 "EEGNet 内扩参在受限 HPO 下不可训" 升级为 "经独立 HPO 验证后仍不可训"；若 (b) 在 random-init within ternary 上把 chance-collapse 比例从 18/21 降至 ≤ 8/21，则 §3.7.2 / §3.7.3 / §4.1 / §7 Finding 1 中 "TUEG 预训练在被试内贡献 binary +23.10 / ternary +30.79 pp" 需进一步弱化为更小区间（HP 错配占了相当比例）。

---

## 7. 结论

本研究系统评估了 EEG 基座模型（CBraMod）在手指级运动想象分类中的应用，通过通道缩减、纵向数据扩展和领域自适应预训练三个维度建立了完整的实验证据体系。五个核心发现如下：

> **发现 1 — 基座模型在三种训练范式下一致优于 EEGNet；探索性消融初步检验差距来源。** CBraMod 对 EEGNet 的优势从 **+7.05 pp**（被试内）扩大至 **+14.01 pp**（跨被试 128 通道），在 32 通道下仍保持 **+10–13 pp** 差距。两项探索性消融（§3.7）对该差距的来源做了初步检验：(i) **沿当前扩参轴扩参 EEGNet 在受限 HPO 预算下方向性有害**——把 EEGNet 沿 (conv stem, MLP 头) 双轴扩参到 1.90M / 5.84M / 30M 三档，cross-subject 准确率从 76.67% 单调下降至 51.37% / 50%（chance），其中 v1/v2 (~30M) 不可训根据作者本人交接诊断更可能是 BF16 下深 MLP 头优化栈兼容性问题（v3 加 LayerNorm 立即 trainable）；(ii) **架构提供独立价值的方向性证据**——在 ~30M 参数 + 无预训练同等条件下，CBraMod random-init cross 86.34% vs EEGNet-Huge v3 (5.84M) cross 51.37% 差距 ~+35 pp，但因 EEGNet-Huge / random-init 均未做专属 HPO 且 baseline → Mid 跳跃同时改 conv stem 与 MLP 头，该差距不可独立归因到 backbone 架构；(iii) **TUEG 预训练贡献同规模、同 HP 下唯一干净的 Δ**——random-init → original-weights backbone 切换的 Δ 在被试内为 **binary +23.10 / ternary +30.79 pp**，在跨被试与 XSI-FT 为 +1.6 ~ +4.3 pp，这是本研究归因强度最高的一组 Δ。本研究在三种训练范式下系统量化了 CBraMod vs EEGNet 的性能差距；补充的探索性消融暗示该差距由架构归纳偏置 + 预训练先验 + 容量约束三因素叠加，但本研究的 HPO 预算与单轴扩参限制使我们**无法对各因素做独立定量归因**。"基座模型价值随数据约束放大" 的方向性结论在两项消融中均得到方向性支持。**该结论限于 CBraMod backbone × 本数据集（21 名 responder cohort）× 当前 HPO 预算；其他 EEG transformer backbone (LaBraM [6], NeuroLM [15], BIOT [16]) 是否复现该三向分解需独立验证（§6 #7）。** 这一限制由 §6 #8（EEGNet-Huge ≥ 25 trial 独立 HPO + random-init CBraMod ≥ 25 trial 独立 HPO，预算 ~80–120 GPU 小时）描述的后续工作处理。
>
> **发现 2 — 32 通道是最优部署目标，64 通道追加 +1.75 pp，且 32ch+ 上方法选择 statistically indistinguishable。** FDR 选取的 32 通道保留了 128 通道性能的 **96.7%**（87.71% vs 90.68%；在 21 名 responder cohort × cross-subject binary 上；通道选择 ranking 包含全 session 信息，可能轻微夸大 retention，详见 Limitation #1），兼容商用 32 通道 EEG 硬件；中端预算可上 64ch FDR (89.46% binary / 75.12% ternary，binary 98.7% retention，ternary 与 128ch baseline 在 run-to-run noise 内一致)；极简成本场景下 4ch Band Power 在双 task 上均可作为兜底方案（binary 78.75% / 86.8% retention；ternary 60.67% / 81.0% retention）——这把可部署谱系从初版草稿的 {128, 32, 8} 扩展到 {128, 64, 32, 8, 4}。2026-05-11 矩阵闭合后 {4, 8, 32, 64}ch × 5 method × {binary, ternary} = 40 cell 完整数据表明：**32ch 起，5 method 之间以及与 negative_control 之间的差异均在 ±3.24 pp 内（ternary ≤ 2.09 pp，落在被试间 std ≈ 13 pp 的 noise 量级内），统计上不可区分**；该不敏感性维持到 64ch（binary spread 3.24 pp、ternary 2.09 pp，与 32ch 同量级）。换言之，32ch+ 部署只需选**任一**数据驱动方法（含商用 10-20 布局或随机 method-complement 通道），性能等价；硬件选择应以舒适度、成本、可用性为主，方法选择影响在该档位被体积传导冗余主导。
>
> **发现 3 — 同被试数据增加显著改善性能；同时 cross-subject 训练所带来的额外优势随之减弱。** 在被试内重训练中，额外 session 数据为两种模型均带来显著增益（CBraMod 二分类 +6.13 pp / 三分类 +8.55 pp, p ≤ 0.012；EEGNet 二分类 +7.34 pp, p = 0.009）；XSI-FT（§3.3 定义；以 cross-subject checkpoint 作为单被试 fine-tune 初始权重）在二分类上达到 +5.70 pp (p = 0.015) 至 92.93% 的相近终点，与被试内重训练接近但未进一步突破。低基线被试获益尤为突出，被试间标准差压缩约 45%（10.81% → 5.98%），最低单被试准确率从 60.62% 提升至 74.38%。相对地，沿用相同 21 名被试 cross-subject 训练并随 session 累积训练数据的 CBraMod 模型仅获得 +0.86 pp 的微弱改善（p = 0.662）——这一对照说明：当个体已经有足够的同被试数据时，cross-subject 训练所带来的额外优势随之减弱，新增同被试 trial 不再依赖跨被试群体信息即可推动决策边界收敛。
>
> **发现 4 — 领域自适应 further pre-training 在以粗运动 MI 为主的外部数据上呈 task-asymmetric 负迁移、跨 paradigm 复现稳健；机制收紧到 MI 粒度错配。** 5 个独立 DAPT 配置（V1–V3 = 10-dataset 系列、V4 = 3-set 域对齐 + strict filter、V5 = Stieger 单源 60ch）共 24 paired-cell 评估（V1–V3 within+cross + V4/V5 within+cross+transfer，2026-05-10 补完 V4/V5 within+transfer 8 cell）显示：**binary 任务上三种 paradigm 全部一致负向**——cross 5/5（mean Δ=−1.79 pp，Stouffer Z=−5.32, p<0.001）、within 5/5（Stouffer Z=−4.42, p<0.0001）、transfer V4/V5（Stouffer Z=−2.79, p=0.005）；**DAPT 失败不是 cross-subject 范式特有现象**。Ternary 任务相对温和：cross 4/5 弱正（mean Δ=+0.18 pp，Stouffer p=0.564，**ternary 方向性负迁移不被支持**），within 5/5 弱负（mean Δ=−0.92 pp，Stouffer Z=−2.16, p=0.031），transfer V4/V5 均弱负（mean Δ=−0.90 pp，p=0.110）。V4/V5 12-cell 全矩阵下 **0/12 正向显著**，且 V5 在 5/6 cell 上比 V4 更差 1.15–1.82 pp——通道多样性消除（V5 单源 60ch）反而系统性恶化 DAPT 转移。BH-FDR 在新 24-cell DAPT family 下重做后仅 V2_within_binary (q=0.048) 单一显著存活（v3.1 16-cell family 下原 V1_cross_binary / V4_cross_binary 在更严苛多重比较惩罚下退到 q ≈ 0.07–0.09，但 paradigm-level Stouffer 集体证据全部仍稳健）。V4 (3-set 域对齐 + strict filter, 最低 pre-train loss 0.001914) 与 V5 (Stieger 单源 60ch) 两次 surgical fix 把候选机制收紧到唯一存活假设——**MI 粒度错配**：粗 hand/leg/upper-limb MI 学到的是"哪个肢体在动"的低频空间包络，下游 finger-level binary（食指 vs 中指，**同手**）需要 DAPT 未学到的细粒度判别；ternary 的 rest 类（不动 vs 运动）可用粗粒度空间包络识别，因此不那么糟。V5 单源 60ch 反方向证伪"通道数异质性是混淆"假设——通道多样性在 DAPT 中是**保护因子**而非 bug。该结论限于粗运动 MI 数据池；DAPT 能否改善 finger MI 解码取决于 source domain 的**信号粒度对齐**而非任务语义类别（"都是 MI"）。
>
> **发现 5 — 通道选择方法间差异在低密度档位放大，binary / ternary 双 task 同向复现；基于全模型的"条件重要性"排序在 4 通道下崩溃。** 4 数据驱动方法（FDR / Attention / Band Power / CSP）的 max−min spread 在 64 / 32 / 8 / 4 通道上单调扩张：**binary** 3.24 / 2.77 / 15.63 / 24.05 pp，**ternary** 1.77 / 2.08 / 6.83 / 19.12 pp（详见 §3.5.3 敏感度表）。基于全通道模型的条件重要性排序（FDR / Attention / CSP）在 4 通道极端约束下均跌至或低于负控制（binary 67.65% / ternary 53.37%）——这是因为 128ch 上算出的"该通道在有其他 124 个通道辅助时的重要性"在仅保留 top-4 时失去了上下文支撑。**mu/beta 频带 Band Power top-4 在双 task 上均保持判别力（binary 78.75% +11.10 pp、ternary 60.67% +7.30 pp 超过负控制）**，且在 8 个 (channel, task) cell 上从未是 4 数据驱动方法的最差者；其评分机制不依赖全模型上下文，因而免疫上述外推失效。这把"4ch 标准方法均失效"的初版结论修订为"条件重要性方法失效，频域指标在该 cohort/任务上保留判别力，且该结论在 binary / ternary 上独立复现"。我们不把"BP 优于其他方法"概括为通用规则——以下任意条件改变（cohort 规模、任务粒度、模型 backbone、预处理）都可能让该排序翻转——结论限于本研究 (cohort, 任务, 模型, 预处理) 组合。先前报告的 4ch FDR∩Attention 82.71% (binary) 仍为 favorable outlier（其本质是从 32+32 集合的相对小交集中"碰巧"落到的 4 个位置，详见 §3.5.2 / §3.5.3），不可作为系统化方法复制；ternary 维度下 FDR∩Attention 未跑。

上述发现共同支持了 CBraMod + FDR 32 通道 BCI 系统在手指级运动想象分类中的实用化部署。本研究观察的 DAPT 负迁移与 NLP DAPT 文献中"低 task-corpus 对齐 + source corpus 不足"失败案例（Gururangan et al. 2020 [20]）在结构上一致；在 CBraMod backbone × masked autoencoding × 粗运动 MI source pool × finger MI target 的具体配置下，通道几何错位（target 128ch vs source 95% 低密度）与任务粒度差异均独立驱动负迁移。本研究**不主张**"EEG foundation model 的 transfer 路径与 NLP/CV 范式级不同"——单 backbone × 单 source pool × 单下游任务的样本不足以支持该普适命题；下游 BCI 实践应优先匹配通道几何与信号尺度，并在存在更高 task-corpus 对齐度（如手指级、手部精细动作 MI source）时再考虑 DAPT。判断 EEG 基座模型是否需要独立于 NLP/CV 的 transfer 设计原则，需要在多 backbone × 多 source corpus × 多预训练目标的矩阵下系统验证（§6 后续工作 #3, #7）。

---

## 参考文献

[1] J. R. Wolpaw, N. Birbaumer, D. J. McFarland, G. Pfurtscheller, and T. M. Vaughan, "Brain-computer interfaces for communication and control," *Clinical Neurophysiology*, vol. 113, no. 6, pp. 767–791, 2002.

[2] G. Pfurtscheller and C. Neuper, "Motor imagery and direct brain-computer communication," *Proceedings of the IEEE*, vol. 89, no. 7, pp. 1123–1134, 2001.

[3] Y. Ding, C. Udompanyawit, Y. Zhang, and B. He, "EEG-based brain-computer interface enables real-time robotic hand control at individual finger level," *Nature Communications*, vol. 16, p. 5401, 2025, doi: 10.1038/s41467-025-61064-x.

[4] J. Wang, S. Zhao, Z. Luo, Y. Zhou, H. Jiang, S. Li, T. Li, and G. Pan, "CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding," in *The Thirteenth International Conference on Learning Representations (ICLR)*, 2025.

[5] V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and B. J. Lance, "EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces," *Journal of Neural Engineering*, vol. 15, no. 5, p. 056013, 2018.

[6] W.-B. Jiang, L.-M. Zhao, and B.-L. Lu, "Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI," in *The Twelfth International Conference on Learning Representations (ICLR)*, 2024.

[7] J. Lai, J. Wei, L. Yao, and Y. Wang, "A Simple Review of EEG Foundation Models: Datasets, Advancements and Future Perspectives," arXiv:2504.20069, 2025.

[8] R. Alazrai, H. Alwanni, and M. I. Daoud, "EEG-based BCI system for decoding finger movements within the same hand," *Neuroscience Letters*, vol. 698, pp. 113–120, 2019.

[9] H. S. Lee et al., "Individual finger movement decoding using a novel ultra-high-density electroencephalography-based brain-computer interface system," *Frontiers in Neuroscience*, vol. 16, p. 1009878, 2022.

[10] R. T. Schirrmeister, J. T. Springenberg, L. D. J. Fiederer, M. Glasstetter, K. Eggensperger, M. Tangermann, F. Hutter, W. Burgard, and T. Ball, "Deep learning with convolutional neural networks for EEG decoding and visualization," *Human Brain Mapping*, vol. 38, no. 11, pp. 5391–5420, 2017, doi: 10.1002/hbm.23730.

[11] S. Sakhavi, C. Guan, and S. Yan, "Learning Temporal Information for Brain-Computer Interface Using Convolutional Neural Networks," *IEEE Transactions on Neural Networks and Learning Systems*, vol. 29, no. 11, pp. 5619–5629, 2018, doi: 10.1109/TNNLS.2018.2789927.

[12] K. K. Ang, Z. Y. Chin, H. Zhang, and C. Guan, "Filter Bank Common Spatial Pattern (FBCSP) in Brain-Computer Interface," in *Proc. 2008 IEEE Int. Joint Conf. Neural Networks (IJCNN)*, Hong Kong, 2008, pp. 2390–2397, doi: 10.1109/IJCNN.2008.4634130.

[13] B. Blankertz, R. Tomioka, S. Lemm, M. Kawanabe, and K.-R. Müller, "Optimizing Spatial Filters for Robust EEG Single-Trial Analysis," *IEEE Signal Processing Magazine*, vol. 25, no. 1, pp. 41–56, 2008, doi: 10.1109/MSP.2008.4408441.

[14] G. Pfurtscheller and F. H. Lopes da Silva, "Event-related EEG/MEG synchronization and desynchronization: basic principles," *Clinical Neurophysiology*, vol. 110, no. 11, pp. 1842–1857, 1999, doi: 10.1016/S1388-2457(99)00141-8.

[15] W.-B. Jiang, Y. Wang, B.-L. Lu, and D. Li, "NeuroLM: A Universal Multi-task Foundation Model for Bridging the Gap between Language and EEG Signals," in *The Thirteenth International Conference on Learning Representations (ICLR)*, 2025.

[16] C. Yang, M. B. Westover, and J. Sun, "BIOT: Biosignal Transformer for Cross-data Learning in the Wild," in *Advances in Neural Information Processing Systems 36 (NeurIPS)*, 2023.

[17] D. Zhang, Z. Yuan, Y. Yang, J. Chen, J. Wang, and Y. Li, "Brant: Foundation Model for Intracranial Neural Signal," in *Advances in Neural Information Processing Systems 36 (NeurIPS)*, 2023.

[18] F. Lotte, L. Bougrain, A. Cichocki, M. Clerc, M. Congedo, A. Rakotomamonjy, and F. Yger, "A review of classification algorithms for EEG-based brain–computer interfaces: a 10 year update," *Journal of Neural Engineering*, vol. 15, no. 3, p. 031005, 2018, doi: 10.1088/1741-2552/aab2f2.

[19] C. Neuper, M. Wörtz, and G. Pfurtscheller, "ERD/ERS patterns reflecting sensorimotor activation and deactivation," in *Progress in Brain Research*, vol. 159, pp. 211–222, 2006, doi: 10.1016/S0079-6123(06)59014-4.

[20] S. Gururangan, A. Marasović, S. Swayamdipta, K. Lo, I. Beltagy, D. Downey, and N. A. Smith, "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks," in *Proc. 58th Annual Meeting of the Association for Computational Linguistics (ACL)*, 2020, pp. 8342–8360, doi: 10.18653/v1/2020.acl-main.740.

[21] M. Mosbach, M. Andriushchenko, and D. Klakow, "On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines," in *International Conference on Learning Representations (ICLR)*, 2021.

[22] J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, et al., "Training Compute-Optimal Large Language Models," in *Advances in Neural Information Processing Systems 35 (NeurIPS)*, 2022.

[23] J. Bergstra, R. Bardenet, Y. Bengio, and B. Kégl, "Algorithms for Hyper-Parameter Optimization," in *Advances in Neural Information Processing Systems 24 (NeurIPS)*, 2011, pp. 2546–2554.

[24] J. Snoek, H. Larochelle, and R. P. Adams, "Practical Bayesian Optimization of Machine Learning Algorithms," in *Advances in Neural Information Processing Systems 25 (NeurIPS)*, 2012.

[25] S. J. Pan and Q. Yang, "A Survey on Transfer Learning," *IEEE Transactions on Knowledge and Data Engineering*, vol. 22, no. 10, pp. 1345–1359, 2010, doi: 10.1109/TKDE.2009.191.

[26] G. Yazıcı, A. Ulutaş, and M. Okuyan, "Effect of EEG Electrode Numbers on Source Estimation in Motor Imagery," *Brain Sciences*, vol. 15, no. 7, p. 685, 2025, doi: 10.3390/brainsci15070685.

---

## 补充材料

> **选取说明**: Tables S1–S4 提供正文核心实验的逐被试细节，选取标准为：正文汇总表中讨论了被试间异质性模式（如异常被试 S04/S10/S14 的行为、低基线被试的差异化收益等）的实验。具体覆盖：128 通道二分类主对比（S1）及三分类对照（S1b）、32 通道配置间对比（S2）、多 session 纵向扩展（S3）、领域自适应预训练负迁移（S4）。XSI-FT、8/4 通道等实验因正文结论基于汇总统计而非个体模式，未单独列出逐被试表。

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
> **关键模式**: Within-subject HPO 最大收益来自**降低正则化**（dropout ↓1.5×, weight_decay ↓2.3×）——预训练 backbone 的内在正则化已足够。EEGNet 最大收益来自**架构升级** (F1: 8→16, D: 2→4, 参数量 ~2.5K→~16K, +3.8 pp)。Cross-subject HPO 参数变化极小（weight_decay、dropout 几乎不变），表明初始默认值已接近最优。
>
> **数据来源**: `docs/dev_log/experiments/hpo_final_parameters.md`

### Table S5e. EEGNet HP source trace

为响应 §2.5.1 的 HP-维度校准说明，本表追踪 EEGNet 7 维搜索空间中各 HP 的来源——继承自 Ding et al. [3] 的 EEGNet-8,2 经验值，还是本研究在 Optuna 中重新搜索得到。

| HP | 来源 | Ding [3] 实际值 | 本研究 HPO 搜索范围 | 本研究 HPO 最优 |
|----|------|----------------|---------------------|----------------|
| F1 (filters) | [3] EEGNet-8,2 默认 8 | 8 | {4, 8, 16}（categorical） | **16** |
| D (depth multiplier) | [3] EEGNet-8,2 默认 2 | 2 | {1, 2, 4}（categorical） | **4** |
| F2 (= F1 × D) | 派生 | 16 | 派生（不独立搜索） | 64 |
| kernel_length | [3] Ding 实际显式设为 32（EEGNet 库形参默认 64 面向 128 Hz；Ding 100 Hz / 4-40 Hz 带通沿用 EEGNet 原作者对 SMR 高通数据的 32 建议） | 32 | {32, 64, 128}（categorical） | 64（≠ Ding 的 32，HPO 选择放大 2×） |
| learning_rate | 本研究新搜 | 1e-3 (Orig) / 1e-4 (Finetune) | [1e-4, 1e-2] log-uniform | 4e-3 |
| weight_decay | 本研究新搜 | — | [1e-5, 0.1] log-uniform | 1e-5 |
| dropout_rate | 本研究新搜 | 0.5 (Orig) / 0.65 (Finetune) | [0.2, 0.7] uniform | 0.27 |
| batch_size | 本研究新搜 | 16 | {32, 64, 128} | 64 |

注：F1 / D 两个 architecture HP 虽继承 [3] 的 EEGNet-8,2 设计经验，但本研究的 Optuna 搜索仍把它们作为可变 categorical 在指定范围内独立采样；HPO 最优 (F1=16, D=4) 为本研究的搜索结果而非 [3] 默认值的直接采用。kernel_length 在本研究的搜索空间 {32, 64, 128} 中以 Ding 实际值 32 为下界、库形参默认 64 为中点；HPO 选择 64，相对 Ding 的 32 增加 2×。本研究未从零冷启动搜索 architecture HP 的边界（如 F1=32 等更大值）——这一上界限制在 §3.7.1 EEGNet-Mid（F1=32）实验中被独立扩展并验证（详见正文）。

> **数据来源**: 搜索空间定义见 [src/hpo/search_spaces.py](../../src/hpo/search_spaces.py) `_sample_eegnet_within` / `_sample_eegnet_cross`；HPO 最优值见 Table S5b。

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

### Figure S2. 21-被试 × 8-条件 准确率热图

被试按 CBraMod 跨被试二分类 baseline 降序排列；左侧色条标注 §2.9 数据质量分类（绿=干净 / 蓝=信息性高方差 / 黄=轻度伪影 / 红=重度伪影）；八列覆盖 model × paradigm × task 主对比 + 32ch FDR 缩减通道 + 多 session +Sess05 内三个扩展实验。色调（viridis）映射准确率，数字为单元格的精确百分比。

![Figure S2. 21-被试 × 8-条件 准确率热图](../figures/subject_heatmap.png)

> **数据来源**：Within EEGNet `20260316_1411`、Within CBraMod `20260323_2237`、Cross EEGNet `20260330_0709`、Cross CBraMod `20260324_0023`、Cross CBraMod ternary `20260324_0109`、XSI-FT `20260329_0507`、32ch FDR `20260330_0836`、Extra Sessions +Sess05 `20260324_2131`。生成脚本：`scripts/paper/generate_paper_figures.py --figure subject_heatmap`。

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

### Figure S3. 5 种 32 通道配置 + 负控制的单配置 2D 电极布局（含通道标签）

每张子图独立呈现一种配置的 32 个电极在 BioSemi 128 头皮 2D 投影上的位置，并标注每个电极的 BioSemi 通道编号（A1–D32），便于复现。Panels a–e 为主文比较的 5 种方法；panel f 为 §3.5.3 控制实验所用的负控制电极集（未被任何数据驱动方法选中的 32 通道），仅展示其空间位置，不纳入聚合统计图（Figure S5 / Figure S6）。

**(a) FDR**（Fisher Discriminant Ratio top-32）

![Figure S3a. FDR](../../results/32_channel/electrode_placements_5configs/single_fdr_2d.png)

**(b) Band Power**（mu/beta ANOVA F top-32）

![Figure S3b. Band Power](../../results/32_channel/electrode_placements_5configs/single_band_power_2d.png)

**(c) CSP**（Common Spatial Pattern weight top-32）

![Figure S3c. CSP](../../results/32_channel/electrode_placements_5configs/single_csp_2d.png)

**(d) Attention**（CBraMod 输入梯度幅值 top-32）

![Figure S3d. Attention](../../results/32_channel/electrode_placements_5configs/single_attention_2d.png)

**(e) Commercial**（标准 10-20 布局 32 通道）

![Figure S3e. Commercial](../../results/32_channel/electrode_placements_5configs/single_commercial_2d.png)

**(f) Negative Control**（未被 4 种数据驱动方法选中的 32 通道；§3.5.3 控制实验的电极对照基线）

![Figure S3f. Negative Control](../../results/32_channel/electrode_placements_5configs/single_negative_control_2d.png)

> **数据来源**: 通道索引 `results/32_channel/channel_selections.json`
> 生成命令: 主文 5 配置 `uv run python scripts/analysis/visualize_electrode_placements.py --configs attention band_power commercial csp fdr --output-dir results/32_channel/electrode_placements_5configs`；负控制单图 `uv run python scripts/analysis/visualize_electrode_placements.py --configs negative_control --output-dir <临时目录>` 后仅保留 `single_negative_control_2d.png` / `multiview_negative_control_3d.png`

### Figure S4. 5 种 32 通道配置 + 负控制的 3D 多视角呈现

每张子图为该配置在 3D 头部模型上的 4 视角组合（前 / 后 / 左 / 右），用于补充 2D 投影丢失的深度信息。Panels a–e 为主文比较的 5 种方法；panel f 为 §3.5.3 控制实验所用的负控制电极集，仅展示其空间位置。

**(a) FDR**

![Figure S4a. FDR](../../results/32_channel/electrode_placements_5configs/multiview_fdr_3d.png)

**(b) Band Power**

![Figure S4b. Band Power](../../results/32_channel/electrode_placements_5configs/multiview_band_power_3d.png)

**(c) CSP**

![Figure S4c. CSP](../../results/32_channel/electrode_placements_5configs/multiview_csp_3d.png)

**(d) Attention**

![Figure S4d. Attention](../../results/32_channel/electrode_placements_5configs/multiview_attention_3d.png)

**(e) Commercial**

![Figure S4e. Commercial](../../results/32_channel/electrode_placements_5configs/multiview_commercial_3d.png)

**(f) Negative Control**

![Figure S4f. Negative Control](../../results/32_channel/electrode_placements_5configs/multiview_negative_control_3d.png)

> **数据来源**: 同 Figure S3。

### Figure S5. 5 配置的总览：3D 叠加与脑区分布

**(a) 3D 叠加视图**：5 种配置共同绘制在单个 3D 头部模型上，每种配置以独立颜色标记选中通道，目视可见各方法的空间集中区域差异。

![Figure S5a. 5 configs 3D overlay](../../results/32_channel/electrode_placements_5configs/all_configs_3d.png)

**(b) 脑区分布柱状图**：将 BioSemi 128 通道按解剖学分区（Frontal / Central / Parietal / Temporal / Occipital）归类，统计每种配置在 5 个区域中的通道数。两点观察值得注意：(i) **Temporal 区在所有配置下均为最大占比**（10–15 / 32 通道）——这是 BioSemi 128 layout 几何特性的反映（外缘环在该分区方案中被归入 Temporal），并非 5 种方法都聚焦于颞叶皮层信号；(ii) 在 Frontal 与 Occipital 上各方法呈现明显差异：FDR / Band Power / Commercial 的 Frontal 占比较高（9–11），而 Attention / CSP 的 Occipital 占比较高（9）——前一组偏前-中分布，后一组偏后-外侧分布。这一脑区层面的异质性与 Figure S6a 量化的"几乎不重叠"互为补充：5 种方法不仅覆盖几乎不同的电极，也覆盖不同的脑区结构。

![Figure S5b. Region distribution](../../results/32_channel/electrode_placements_5configs/region_distribution.png)

> **数据来源**: 同 Figure S3。

### Figure S6. 5 配置间通道重叠分析（支撑 §4.3 的体积传导冗余论证）

**(a) Jaccard 相似度热图**：5 种 32 通道配置两两比较的 Jaccard 系数（共享通道数 / 并集通道数）。所有配置对的 Jaccard 系数 ∈ [0.12, 0.23]，最小值出现在 FDR vs CSP（0.12，仅 7 个共有通道），最大值出现在多组（0.23，12 个共有通道，含 FDR vs Attention / FDR vs Commercial / CSP vs Attention / CSP vs Commercial）——任意两种方法的 32 通道集合**至少有 77% 的电极不重叠**。这是高密 EEG 信息冗余的直接证据，也是 §4.3 "方法不敏感性"叙事的空间侧支撑。

![Figure S6a. Jaccard heatmap](../../results/32_channel/electrode_placements_5configs/overlap_analysis.png)

**(b) FDR vs Attention 成对重叠（128 通道全空间）**：FDR 与 Attention 的 32 通道集合在 BioSemi 128 上的可视化（红=FDR 独有，蓝=Attention 独有，紫=两者共有）。仅 4 个共有通道（B32, C8, D7, D19），随机期望为 32×32/128 = 8。这 4 个交集通道在 §3.5.3 的控制实验中被作为 4ch FDR∩Attention "favorable outlier" (82.71%) 的电极来源。

![Figure S6b. FDR vs Attention](../../results/32_channel/electrode_placements_5configs/overlap_fdr_vs_attention_2d.png)

**(c) FDR vs Band Power 成对重叠**：两个表现最佳的数据驱动方法（FDR 87.71% 与 Band Power 86.85%）之间的电极重叠。

![Figure S6c. FDR vs Band Power](../../results/32_channel/electrode_placements_5configs/overlap_fdr_vs_band_power_2d.png)

**(d) FDR 8ch vs FDR 32ch 嵌套关系**：可视化 FDR 8 通道是否构成 FDR 32 通道的真子集（128 通道空间）。作为 §3.5.2 通道缩放叙事中"最优排序在不同通道档间是否保持嵌套"的视觉补充。

![Figure S6d. FDR 8ch vs 32ch](../../results/32_channel/electrode_placements_5configs/overlap_fdr8_vs_fdr32_2d.png)

> **数据来源**: 同 Figure S3。

### Table S7. Quaternary 全范式辅助结果（128 通道，N = 21）

Quaternary 不进入 §3 主线结论的理由参 §3.3.1。本表汇总三种范式 × 两种模型共 6 个运行；所有运行使用 OfflineImagery 单 session 时序 70/15/15 切分（train/val/test），与 binary / ternary 主线的 "OfflineImagery + Online_Sess01/02_Base 训练 → Online_Sess02_Finetune 测试"协议**不可直接对照**。

**Table S7a. Quaternary 各范式 × 各模型测试准确率。**

| 范式 | 模型 | Test Acc Mean ± SD | Val Acc | val→test gap | Δ vs chance (25%) |
|------|------|---------------------|---------|--------------|--------------------|
| 跨被试 | EEGNet | 43.65 ± 7.62% | 37.17% | +6.48 pp | +18.65 pp |
| 跨被试 | **CBraMod** | **46.30 ± 8.86%** | 38.29% | +8.01 pp | +21.30 pp |
| 被试内 | CBraMod | 40.69 ± 11.42% | 35.90% | +4.80 pp | +15.69 pp |
| 被试内 | **EEGNet** | **47.81 ± 11.35%** | 40.80% | +7.02 pp | +22.81 pp |
| XSI-FT | **EEGNet** | **47.57 ± 10.60%** | 41.49% | +6.08 pp | +22.57 pp |
| XSI-FT | CBraMod | 45.29 ± 12.15% | 38.99% | +6.30 pp | +20.29 pp |

**Table S7b. 模型间相对差异的范式依赖。**

| 范式 | Δ (CBraMod − EEGNet) | 说明 |
|------|---------------------|------|
| 跨被试 | +2.65 pp | CBraMod 微胜 |
| 被试内 | **−7.12 pp** | EEGNet 反超 |
| XSI-FT | −2.28 pp | EEGNet 微胜 |

**Table S7c. 跨任务难度对照（CBraMod cross-subject）。**

| 任务 | chance | Test Acc | Acc / Chance |
|------|--------|----------|--------------|
| 二分类 | 50.0% | 90.68% | 1.81× |
| 三分类 | 33.3% | 74.88% | 2.25× |
| 四分类 | 25.0% | 46.30% | 1.85× |

quaternary 的 Acc/Chance 比落回 binary 量级（1.85× vs 1.81×），低于 ternary（2.25×）——在该数据集上四类任务的"边际可分性"反而比三分类窄，这与"四类细粒度间在感觉运动皮层上的相邻表征更难区分"的方向一致。

**Quaternary 上 CBraMod 优势的反转**：CBraMod 在 binary / ternary cross-subject 范式下相对 EEGNet 的优势（+14.01 / +13.65 pp）在 quaternary 上**显著收窄至 +2.65 pp**，且在 within-subject / XSI-FT 范式下方向反转（EEGNet 微胜 2.28~7.12 pp）。这一反转是本数据集上首次观察到的"EEGNet 持平或微胜 CBraMod"的范式，但鉴于 §3.3.1 列出的三项数据限制（offline-only、采集于 session 序列最早、单 session 时序切分），不应作为"基座模型在细粒度任务上失效"的主张性结论；解释 quaternary 反转需要在更大、更平衡的细粒度 MI 数据池上独立验证。

> **数据来源**: cross-subject `20260508_1221`（同 run 含 EEGNet + CBraMod 双 model_summaries）: `results/20260508_1221_cross_subject_cache_imagery_quaternary.json`；within-subject CBraMod `20260508_1518`: `results/20260508_1518_within_subject_cache_imagery_quaternary.json`；within-subject EEGNet `20260508_1538`: `results/20260508_1538_within_subject_cache_imagery_quaternary.json`；XSI-FT `20260508_1611`（同 run 含 EEGNet + CBraMod）: `results/20260508_1611_transfer_cache_imagery_quaternary.json`
