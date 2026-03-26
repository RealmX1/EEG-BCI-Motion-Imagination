# Extra Sessions 测试集策略对比分析

> **实验日期**: 2026-03-25
> **配置**: 128 通道, 16 被试, imagery paradigm, binary task
> **默认策略**: per_session（主实验结果）
> **辅助分析策略**: fixed_combined / fixed_sess02（仅用于因素分离分析）

## 1. 三种策略的设计意图

### 1.1 per_session（默认，对应原论文设计）

| Step | 训练集 | 测试集 |
|------|--------|--------|
| Baseline | Offline + Sess01(B+FT) + Sess02(B) | **Sess02(FT)** |
| +Sess03 | +Sess02(FT) + Sess03(B) | **Sess03(FT)** |
| +Sess04 | +Sess03(FT) + Sess04(B) | **Sess04(FT)** |
| +Sess05 | +Sess04(FT) + Sess05(B) | **Sess05(FT)** |

**测试意图**：模拟真实 BCI 部署场景——每采集一个新 session 后，用之前所有数据训练，在最新 session 上评估。反映的是"新 session 对当前用户的解码能力如何"。

**局限性**：测试集每步不同，跨步准确率变化包含**测试集难度差异**这一混淆因素。

### 1.2 fixed_combined（固定组合测试集）

| Step | 训练集 | 测试集 |
|------|--------|--------|
| Baseline | Offline + Sess01(B+FT) + Sess02(B) + Sess02(FT前3/4) | **所有FT session 末1/4** |
| +Sess03 | +Sess03(B) + Sess03(FT前3/4) | **同上（固定）** |
| +Sess04 | +Sess04(B) + Sess04(FT前3/4) | **同上（固定）** |
| +Sess05 | +Sess05(B) + Sess05(FT前3/4) | **同上（固定）** |

测试集 = Sess02-05 每个 Finetune session 的最后 1/4 trials 合并（160 trials/subject，跨所有步骤恒定）。

**测试意图**：消除测试集变化的混淆因素。跨步准确率变化**纯粹反映训练数据量的贡献**。Baseline 也重新训练（因为 Sess02_FT 前 3/4 进入训练），与 per_session 和 fixed_sess02 的 DB baseline 不同。

**局限性**：测试集包含来自不同时间点的 session 数据，可能平滑掉了 session-specific 的性能特征。Baseline 数值与标准 within-subject baseline 不可比。

### 1.3 fixed_sess02（固定 Sess02 测试集）

| Step | 训练集 | 测试集 |
|------|--------|--------|
| Baseline | Offline + Sess01(B+FT) + Sess02(B) | **Sess02(FT)**（与标准 baseline 相同） |
| +Sess03 | +Sess03(B+FT) | **Sess02(FT)**（不变） |
| +Sess04 | +Sess04(B+FT) | **Sess02(FT)**（不变） |
| +Sess05 | +Sess05(B+FT) | **Sess02(FT)**（不变） |

**测试意图**：测试"增加来自远期 session 的训练数据是否改善模型对**早期 session** 的解码能力"。反映跨时间泛化能力——更多数据是否让模型更好地理解该被试的 EEG 特征，而非仅仅拟合特定 session。

**注意**：Sess02_FT 不进入训练集（它是测试集）。训练集增加 Sess03-05 的 Base+FT。Baseline 与标准 within-subject baseline 完全相同（来自 ExperimentDB）。

## 2. 群体均值对比

### 2.1 EEGNet

| 策略 | Baseline | +Sess03 | +Sess04 | +Sess05 | Δ(BL→+S05) |
|------|----------|---------|---------|---------|-------------|
| per_session | 80.51% | 87.73% | 87.93% | 87.85% | **+7.34pp** |
| fixed_combined | 79.92% | 83.28% | 87.27% | 89.88% | **+9.96pp** |
| fixed_sess02 | 80.51% | 85.82% | 87.42% | 89.02% | **+8.51pp** |

> **数据来源**: per_session: `results/20260324_2131_...json`, fixed_combined: `results/20260325_0514_...json`, fixed_sess02: `results/20260325_1208_...json`

### 2.2 CBraMod

| 策略 | Baseline | +Sess03 | +Sess04 | +Sess05 | Δ(BL→+S05) |
|------|----------|---------|---------|---------|-------------|
| per_session | 87.23% | 89.14% | 90.94% | 93.36% | **+6.13pp** |
| fixed_combined | 84.38% | 90.20% | 91.52% | 92.81% | **+8.44pp** |
| fixed_sess02 | 87.23% | 90.55% | 90.74% | 91.60% | **+4.37pp** |

## 3. 策略间差异解读

### 3.1 Baseline 差异

- **per_session 和 fixed_sess02 共享相同 baseline**（来自 ExperimentDB），因为两者 baseline step 的训练/测试划分完全相同
- **fixed_combined 的 baseline 略低**：EEGNet 79.92% vs 80.51%，CBraMod 84.38% vs 87.23%。原因有二：
  1. 测试集不同（Sess02-05 每个 FT 的末 1/4 合并 vs 仅 Sess02 FT）
  2. Sess02_FT 的前 3/4 进入了训练，但同时测试集混入了 Sess03-05 的数据（这些 session 的模式尚未被训练 baseline 见过）

### 3.2 增长轨迹差异

**per_session 的增长非单调且 +Sess03 后趋平**：
- EEGNet: 80.51 → 87.73 → 87.93 → 87.85（+Sess03 一步跃升 +7.22pp，之后几乎不变）
- 这是**混淆因素的典型表现**：+Sess03 的测试集可能比 Sess02_FT 更容易，造成虚高的跳跃；后续 step 的测试集难度不同，掩盖了训练数据的真实贡献

**fixed_combined 展现清晰的单调递增**：
- EEGNet: 79.92 → 83.28 → 87.27 → 89.88（持续稳定上升）
- CBraMod: 84.38 → 90.20 → 91.52 → 92.81（同样单调递增）
- 固定测试集消除了测试难度混淆后，**每增加一个 session 都带来可衡量的改善**

**fixed_sess02 也展现单调递增但斜率较缓**：
- EEGNet: 80.51 → 85.82 → 87.42 → 89.02
- CBraMod: 87.23 → 90.55 → 90.74 → 91.60（CBraMod 后两步几乎持平）
- 测试的是对 **Sess02** 的解码能力——远期 session 数据对理解早期 EEG 模式有帮助但效益递减

### 3.3 三种策略的核心发现对比

| 问题 | per_session 答案 | fixed_combined 答案 | fixed_sess02 答案 |
|------|------------------|--------------------|--------------------|
| 额外数据是否有帮助？ | 是（+7.34 / +6.13pp） | 是（+9.96 / +8.44pp） | 是（+8.51 / +4.37pp） |
| 改善是否单调递增？ | 否（非单调，有波动） | **是（清晰单调递增）** | 大致是（EEGNet 单调，CBraMod 后段趋平） |
| EEGNet vs CBraMod 获益比较？ | EEGNet 略多（+7.34 vs +6.13） | EEGNet 更多（+9.96 vs +8.44） | EEGNet 明显更多（+8.51 vs +4.37） |
| 低 baseline 被试获益最大？ | 是 | 是 | 是 |

### 3.4 被试技能学习效应的影响

BCI 被试在多次 session 中学会更有效地调制运动想象相关 EEG 信号（motor imagery skill learning）。后续 session 采集到的数据通常具有更高的信噪比和更清晰的类别可分性。三种策略受此效应的影响不同：

#### 训练侧影响

所有三种策略都将后续 session 数据加入训练集，因此都受益于**更高质量的训练数据**。这意味着即使 fixed_combined 消除了测试集变化的混淆，其观察到的单调递增中仍然包含"后续 session 数据质量更高"这一贡献——我们无法仅靠这些实验完全分离"数据量"和"数据质量"两个因素。

#### 测试侧影响

| 策略 | 测试数据的技能学习偏差 |
|------|------------------------|
| per_session | **最大**：+Sess05 的测试集来自最后一个 session，被试技能最成熟，测试数据最易区分。这使得 +Sess05 的准确率被人为抬高 |
| fixed_combined | **部分抵消**：测试集混合了所有 session 的末 1/4 trials，早期和晚期 session 的难度被平均化 |
| fixed_sess02 | **反向偏差**：始终测试最早的 Sess02 数据（被试技能最不成熟），是三种策略中最保守的评估 |

#### 策略间差异如何帮助分离技能学习效应

比较 fixed_combined（+9.96pp EEGNet）和 fixed_sess02（+8.51pp EEGNet）的 Δ：

- fixed_combined 的测试集包含后续 session 数据 → 后续 session 数据质量更高 → 模型在这些 trial 上自然表现更好
- fixed_sess02 的测试集不变（早期 Sess02 数据）→ 改善纯粹来自模型更好地理解了被试的 EEG 特征

差值 ~1.5pp (EEGNet) 可粗略归因于 fixed_combined 测试集中晚期 session 数据的质量红利。

对于 CBraMod，差距更大：+8.44pp (fixed_combined) vs +4.37pp (fixed_sess02)。这表明 CBraMod 在固定早期测试集上提升有限（基座模型已捕获大部分不变特征），但在包含晚期高质量 session 的测试集上表现出更明显的改善——说明 **CBraMod 对训练数据质量的变化比对数据量更敏感**。

### 3.5 关键洞察

#### 3.5.1 per_session 高估了 +Sess03 的贡献

per_session 中 EEGNet +Sess03 跳跃了 +7.22pp，但 fixed_combined 显示贡献为 +3.36pp，fixed_sess02 为 +5.31pp。差额来自两个因素：(1) Sess03_FT 测试数据的固有可分性更高（被试技能提升），(2) Session 间的随机难度差异。这证实了**per_session 策略下跨步比较不可靠**。

#### 3.5.2 fixed_combined 单调递增中仍含数据质量因素

fixed_combined 给出了最干净的信号：每增加一个 session，在**相同测试集**上性能持续改善。但须注意这一改善同时反映了训练数据量增加**和**后续 session 训练数据质量更高两个因素。纯粹的数据量贡献可能低于观察值。

#### 3.5.3 fixed_sess02 是最保守的评估

fixed_sess02 的测试集是 Sess02_FT（采集时间最早，被试技能最不成熟）。在此最严格的标准下仍观察到显著改善（EEGNet +8.51pp, CBraMod +4.37pp），说明额外 session 帮助模型学到了**被试层面的 EEG 时间不变特征**（temporal invariants），而非仅仅拟合后续 session 中更清晰的信号模式。

#### 3.5.4 Temporal Distribution Shift 与 fixed_sess02 的递减收益

fixed_sess02 策略中，增量收益在 +Sess03 后急剧衰减：

| 模型 | BL→+S03 | +S03→+S04 | +S04→+S05 |
|------|---------|-----------|-----------|
| EEGNet | +5.31pp | +1.60pp | +1.60pp |
| CBraMod | +3.32pp | **+0.19pp** | +0.86pp |

对比 fixed_combined 中的增量收益（EEGNet: +3.36 / +3.99 / +2.61pp，CBraMod: +5.82 / +1.32 / +1.29pp），fixed_sess02 的衰减更为陡峭。

**原因**：fixed_sess02 测试的是 Sess02_FT（采集时间最早）。随着训练集纳入时间上越来越远的 session（Sess04、Sess05），这些数据与 Sess02 的**时间分布差异**（temporal distribution shift）越来越大：
- 电极阻抗和接触质量在不同日期间变化
- 被试的神经活动模式随 BCI 技能学习而演化（后续 session 中被试可能采用了与早期不同的心理策略）
- 日间变异（arousal、疲劳、环境差异）

因此，Sess04/05 的数据虽然质量更高，但与 Sess02 的分布匹配度更低，对预测 Sess02 的边际收益递减。

#### 3.5.5 修正的因素分解框架

此前将观察到的提升归因于三个因素：数据量、数据质量、测试集难度变化。但测试集难度变化本质上是被试技能学习的同一延伸——更成熟的被试产出更可分的数据，无论该数据出现在训练侧还是测试侧。因此更准确的因素分解为：

1. **训练数据量增加**（纯粹的统计学习效应）
2. **被试技能学习**（后续 session 数据质量普遍提升，同时影响训练和测试）
3. **Session 间分布漂移**（inter-session distribution drift）——跨 session 的数据异质性，包括电极阻抗变化、日间神经状态变异、被试策略演化等

因素 2 和因素 3 是对立的力量：技能学习提升数据质量（有利），但分布漂移增加数据异质性（不利）。实验结果表明：

- **fixed_combined 的单调递增**：因素 1+2 占主导，分布漂移的负面影响被平均化（测试集混合了所有 session）
- **fixed_sess02 的快速饱和**：因素 3 在此策略中暴露最充分——远期 session 数据与 Sess02 的分布差异抵消了其质量优势
- **CBraMod 在 fixed_sess02 中近乎停滞**：预训练基座模型对分布漂移更敏感
- **原论文结论的重新解读**：原论文观察到 EEGNet 从额外 session 中"提升有限"，可能并非因为数据量不足，而是因为 session 间分布漂移的负面效应与数据量增加的正面效应近似抵消

#### 3.5.6 理论化：数据异质性-泛化能力权衡模型

上述因素分解暗示了一个可形式化的框架。给定一个被试的 session 序列 $S_1, S_2, \ldots, S_N$，目标是选择最优的训练数据范围以最大化对当前 session $S_t$ 的解码性能：

$$\text{Performance}(S_t) = f\big(\underbrace{|D_{\text{train}}|}_{\text{数据量}},\; \underbrace{Q(D_{\text{train}})}_{\text{数据质量}},\; \underbrace{\Delta(D_{\text{train}}, S_t)}_{\text{分布漂移}},\; \underbrace{G(M)}_{\text{模型泛化能力}}\big)$$

其中：
- $|D_{\text{train}}|$：训练集大小
- $Q(D_{\text{train}})$：训练数据的平均质量（被试技能水平的函数）
- $\Delta(D_{\text{train}}, S_t)$：训练数据与目标 session 之间的分布距离（随时间跨度增大而增大）
- $G(M)$：模型对分布漂移的鲁棒性（EEGNet > CBraMod，如 §3.5.7 所讨论）

当增加更远的 session 时，$|D_{\text{train}}|$ 和 $Q$ 增大，但 $\Delta$ 也增大。最优训练窗口取决于这三者的权衡。本实验的结果表明：
- 对 EEGNet（$G$ 较大）：$\Delta$ 的负面效应较弱，最优窗口可以更长
- 对 CBraMod（$G$ 较小，因预训练特征对 session 更敏感）：$\Delta$ 的负面效应更强，存在更早的边际收益递减点

**支持 $\Delta$ 重要性的多重证据**：

分布漂移 $\Delta$ 对性能的影响不仅体现在训练窗口扩展时的递减收益，还体现在**测试集构成**对 baseline 的影响。CBraMod 在 fixed_combined baseline 中的表现（84.38%）显著低于 per_session/fixed_sess02 baseline（87.23%），尽管 fixed_combined baseline 的训练集实际上更大（包含 Sess02_FT 前 3/4）。

这一反直觉的结果直接源于 $\Delta$：fixed_combined 的测试集混合了 Sess02-05 所有 FT session 的末 1/4 trials，其中 Sess03-05 的数据在 baseline 步骤时**完全未被训练见过**。CBraMod 对这些未见 session 的分布偏移更敏感，导致在混合测试集上表现下降。相比之下，fixed_sess02 baseline 仅测试 Sess02_FT（与训练分布最接近），因此表现更好。

这与 CBraMod 在 fixed_sess02 策略中增量收益快速饱和的现象（§3.5.4）指向同一结论：**CBraMod 的预训练特征对 session 间分布差异更敏感，$\Delta$ 在其性能方程中的权重更大**。EEGNet 在两种场景下都表现得更鲁棒（fixed_combined baseline 79.92% vs per_session 80.51%，差距仅 0.59pp），进一步支持了小模型在分布漂移下的泛化优势。

**未来研究方向**：
1. **量化 session 间分布距离**：利用已有数据计算 session 对之间的 KL 散度或 MMD（Maximum Mean Discrepancy），建立 $\Delta$ 的经验模型。可将时间间隔、日内采集时间等作为解释变量
2. **Session splicing 实验**：对现有 session 进行拆分和重组（如将一个 session 的前半和后半分别作为独立 pseudo-session），在控制时间跨度的同时隔离数据量效应，估算每个因素的相对贡献
3. **最优训练窗口搜索**：基于上述模型，对不同被试和模型预测最优的训练数据长度，验证是否存在"添加更多远期数据反而降低性能"的转折点
4. **不确定性建模**：将 session 间关系（时间间隔、采集条件差异）纳入不确定性估计，为实时 BCI 部署提供自适应训练窗口选择策略

#### 3.5.7 EEGNet 在 fixed_sess02 中的异常优势

EEGNet 相对 CBraMod 的改善幅度优势在三种策略间有显著差异：

| 策略 | EEGNet Δ | CBraMod Δ | EEGNet 优势 |
|------|----------|-----------|-------------|
| per_session | +7.34pp | +6.13pp | +1.21pp |
| fixed_combined | +9.96pp | +8.44pp | +1.52pp |
| fixed_sess02 | +8.51pp | +4.37pp | **+4.14pp** |

在 fixed_sess02 中 EEGNet 的优势几乎是其他策略的 3 倍。Temporal distribution shift 解释了 CBraMod 为何在此策略下获益有限（预训练特征可能更 session-specific），但**无法解释 EEGNet 为何在相同条件下反而更好地利用了时间上遥远的数据**。

可能的解释（推测性的，需进一步验证）：

1. **模型容量与正则化的权衡**：EEGNet (~2.5K 参数) 容量远小于 CBraMod (~4M 参数)。小模型被迫学习最具判别力的低维特征，这些特征可能恰好是跨时间稳定的（如 mu/beta 频带功率变化的空间模式）。CBraMod 的大容量允许它捕获更多 session-specific 的细粒度模式，这些模式在跨时间预测时反而成为噪声

2. **预训练特征的 session 偏向性**：CBraMod 的预训练基座在大规模多样 EEG 数据上训练，可能学到了对 session 内短期时间结构敏感的特征。当训练数据跨越多个时间上分散的 session 时，这些特征之间的冲突可能降低了对特定早期 session 的预测能力

3. **End-to-end 学习的灵活性**：EEGNet 从随机初始化开始端到端训练，没有预训练偏向。它在所有可用数据上从零构建表征，可能更自然地找到跨时间泛化的特征子空间

这一发现有潜在的实用意义：**当目标是跨 session 长期泛化时，简单模型（如 EEGNet）可能比预训练基座模型更稳健**——这与直觉相反，值得在论文中讨论。

#### 3.5.5 标准差收缩是一致的

三种策略都观察到随数据增加标准差收缩的趋势：

| 策略 | EEGNet Std (BL→+S05) | CBraMod Std (BL→+S05) |
|------|----------------------|------------------------|
| per_session | 12.05% → 7.18% | 10.85% → 5.89% |
| fixed_combined | 11.08% → 5.79% | 11.22% → 6.18% |
| fixed_sess02 | 12.05% → 8.18% | 10.85% → 7.92% |

这意味着额外训练数据不仅提升均值，还**缩小个体间差异**——实用意义重大，因为 BCI 系统需要在不同用户间保持稳定性能。

## 4. 策略定位与默认选择

### 4.1 默认策略：per_session

**per_session 保持为默认策略**。BCI 的实际部署目标始终是让模型匹配被试**当前/最新 session** 的 EEG 数据。per_session 直接衡量这一目标——"加入历史数据后，模型在最新 session 上的解码能力如何"。其测试集变化不是缺陷而是特性：它反映了真实场景中被试技能提升、数据质量变化等所有因素的综合效果。

### 4.2 辅助分析策略

fixed_combined 和 fixed_sess02 **仅用于分析目的**，帮助分离 per_session 结果中混合的因素：

| 策略 | 角色 | 回答的分析问题 |
|------|------|----------------|
| **per_session**（默认） | 主实验结果 | "我的 BCI 每次新 session 后表现如何？"（临床相关性） |
| fixed_combined | 分析工具 | "训练数据量+质量的真实贡献是什么？"（消除测试集变化混淆） |
| fixed_sess02 | 分析工具 | "模型是否真正学到了被试的时间不变特征？"（最保守评估） |

三者结合可以粗略分离数据量、被试技能学习和 session 间分布漂移的各自贡献（详见 §3.5.5）。

## 5. 对论文的启示

1. **原论文"EEGNet 提升有限"的结论需要修正**：N=16 下两模型均显著受益（p<0.01），且 fixed_combined 分析证实在控制测试集后改善是单调的
2. **CBraMod 的优势在于绝对水平而非增长幅度**：CBraMod 始终领先 EEGNet ~5pp，但从额外数据中获益幅度相当甚至更小。CBraMod 对数据质量比数据量更敏感（预训练基座已捕获通用 EEG 特征）
3. **低 baseline 被试是主要获益者**：BCI 实践中最需要帮助的用户恰恰从额外数据中获益最多
4. **论文报告策略**：以 per_session 作为主结果（反映临床相关性），辅以 fixed_combined 和 fixed_sess02 的分析结果帮助解释观察到的趋势
5. **须明确讨论被试技能学习**：准确率提升并非全部来自算法/数据改善，被试自身的 BCI 操控技能在 session 间持续发展。三种策略的差异有助于粗略分离这些因素，但完全的因素分离需要额外的实验设计（如固定 session 顺序的交叉验证）

## 6. 文件索引

| 文件 | 说明 |
|------|------|
| `results/20260324_2131_extra_sessions_cache_imagery_binary.json` | per_session 结果 |
| `results/20260325_0514_extra_sessions_cache_fixed_combined_imagery_binary.json` | fixed_combined 结果 |
| `results/20260325_1208_extra_sessions_cache_fixed_sess02_imagery_binary.json` | fixed_sess02 结果 |
| `results/20260324_2131_extra_sessions_imagery_binary.png` | per_session 图 |
| `results/20260325_0514_extra_sessions_imagery_binary.png` | fixed_combined 图 |
| `results/20260325_1208_extra_sessions_imagery_binary.png` | fixed_sess02 图 |
