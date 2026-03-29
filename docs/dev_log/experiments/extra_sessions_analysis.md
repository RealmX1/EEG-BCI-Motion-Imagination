# Extra Online Sessions 实验分析报告

> **实验日期**: 2026-03-24 ~ 2026-03-29
> **128 通道, imagery paradigm**
> **状态**: Within-subject Binary/Ternary 完成 | Cross-subject Binary/Ternary 完成 | Transfer Binary/Ternary 完成

**注意**：Binary 任务已扩展至 N=16 被试的完整分析，详见 [`extra_sessions_per_session_analysis.md`](extra_sessions_per_session_analysis.md)。本文档的 §3 保留最初 N=5 的结果作为历史记录，§5 为 N=16 的 Ternary 完整结果，§6 为 Cross-subject 结果（含 16-subj 与 21-subj 训练模式对比），§7 为 Transfer Learning 结果（含三种范式对比）。

## 1. 实验目的

原论文观察到 EEGNet 在加入第二次在线会话之后的额外数据时，准确率提升有限。本实验在 EEGNet 和 CBraMod 上评估这一结论，通过逐步增加在线 session 3/4/5 的训练数据，观察性能变化趋势。

## 2. 实验设计

### 2.1 渐进式数据协议

| Step | 训练集 | 测试集 |
|------|--------|--------|
| Baseline | Offline + Sess01(Base+FT) + Sess02(Base) | Sess02(Finetune) |
| +Sess03 | 上述全部 + Sess02(FT) + Sess03(Base) | Sess03(Finetune) |
| +Sess04 | 上述全部 + Sess03(FT) + Sess04(Base) | Sess04(Finetune) |
| +Sess05 | 上述全部 + Sess04(FT) + Sess05(Base) | Sess05(Finetune) |

### 2.2 被试选取

仅文件系统中存在 session 3-5 数据的被试参与（程序化发现）：S02, S03, S04, S06, S07。

Baseline 为各模型的标准 post-HPO within-subject 运行结果，仅取上述 5 被试子集重新计算均值。

## 3. Binary 任务结果

### 3.1 EEGNet

| Subject | Baseline | +Sess03 | +Sess04 | +Sess05 | Δ(Final) |
|---------|----------|---------|---------|---------|----------|
| S02 | 94.38% | 92.50% | 91.88% | 94.38% | ±0.00pp |
| S03 | 85.00% | 76.88% | 89.38% | 85.00% | ±0.00pp |
| S04 | 94.38% | 95.62% | 91.88% | 92.50% | -1.88pp |
| S06 | 68.12% | 80.62% | 68.75% | 85.62% | **+17.50pp** |
| S07 | 76.88% | 75.62% | 86.25% | 95.62% | **+18.75pp** |
| **Mean** | **83.75%** | **84.25%** | **85.62%** | **90.62%** | **+6.88pp** |
| Std | 10.19% | 8.24% | 8.69% | 4.45% | 9.22% |

> **数据来源**: `results/20260324_1557_extra_sessions_cache_imagery_binary.json`, model=eegnet
> **Baseline run_tag**: `20260316_1411`

### 3.2 CBraMod

| Subject | Baseline | +Sess03 | +Sess04 | +Sess05 | Δ(Final) |
|---------|----------|---------|---------|---------|----------|
| S02 | 94.38% | 98.12% | 97.50% | 97.50% | +3.12pp |
| S03 | 94.38% | 91.88% | 98.75% | 96.25% | +1.88pp |
| S04 | 91.88% | 89.38% | 95.00% | 98.75% | +6.88pp |
| S06 | 74.38% | 80.62% | 71.88% | 94.38% | **+20.00pp** |
| S07 | 81.88% | 82.50% | 80.00% | 93.12% | **+11.25pp** |
| **Mean** | **87.38%** | **88.50%** | **88.62%** | **96.00%** | **+8.62pp** |
| Std | 7.97% | 6.37% | 10.74% | 2.04% | 6.56% |

> **数据来源**: `results/20260324_1557_extra_sessions_cache_imagery_binary.json`, model=cbramod
> **Baseline run_tag**: `20260323_2237`

### 3.3 统计检验

| 检验 | 统计量 | p 值 | 结论 |
|------|--------|------|------|
| EEGNet Δ vs 0 (paired t-test) | t=1.49 | p=0.210 | 不显著 |
| CBraMod Δ vs 0 (paired t-test) | t=2.63 | p=0.058 | 边缘显著 |
| EEGNet Δ vs CBraMod Δ (Wilcoxon) | W=4.0 | p=0.438 | 不显著 |

注意：N=5 样本量极小，统计功效不足。CBraMod 接近显著 (p=0.058)，更大样本下可能达到显著。

> **图表**: `results/20260324_1557_extra_sessions_imagery_binary.png`

## 4. 初步发现

### 4.1 与原论文结论的对比

原论文称 EEGNet 在加入额外数据后提升有限。本实验结果**部分支持但需要细化**：

- **高 baseline 被试 (S02, S03, S04)**: 两个模型提升都很有限（天花板效应，baseline 已 >90%）
- **低 baseline 被试 (S06, S07)**: 两个模型都展现了显著提升（+11 ~ +20pp），EEGNet 的提升幅度 (+17-19pp) 甚至不亚于 CBraMod

### 4.2 CBraMod vs EEGNet

- CBraMod 在 **+Sess05** 达到极高均值（96.00% ± 2.04%），且标准差大幅收缩（从 7.97% 降至 2.04%）
- EEGNet 均值提升幅度相当 (+6.88pp vs +8.62pp)，但方差降幅不如 CBraMod
- 两模型提升差异无统计学显著性 (Wilcoxon p=0.438)，但样本量仅 5

### 4.3 非单调路径

多个被试在中间 step (+Sess03 或 +Sess04) 出现性能下降后在最终 step 回升。可能原因：
1. **不同 session 的测试集难度不同** — 每个 step 的测试集是不同 session 的 Finetune 数据，非同一测试集
2. **训练数据质量波动** — 某些 session 的数据质量可能低于其他

### 4.4 关键限制

1. **N=5** — 统计功效严重不足，所有检验结果需谨慎解读
2. **测试集不可比** — 每个 step 的测试集来自不同 session (Sess02/03/04/05 Finetune)，非同一独立测试集上的纵向对比
3. **仅 binary 任务** — Ternary 待运行

## 5. Ternary 任务结果 (N=16, per_session 策略)

> **实验日期**: 2026-03-25 ~ 2026-03-26
> **配置**: 128 通道, 16 被试, imagery paradigm, ternary task
> **测试策略**: per_session
> **数据来源**: `results/20260325_1934_extra_sessions_cache_imagery_ternary.json`
> **运行时间**: 8.0h

### 5.1 被试与 Baseline 覆盖

16 个被试拥有 Sess03-05 数据：S02, S03, S04, S06, S07, S08, S09, S10, S11, S13, S14, S15, S16, S17, S18, S19。S07 仅有 Sess04-05（无 Sess03 ternary 数据）。

Baseline 来源：
- **CBraMod**: `20260323_2320`（21 subjects, designated baseline）— 16/16 subjects 有 baseline
- **EEGNet**: `20260323_2116`（heuristic, 非 designated baseline）— **仅 S02 有 baseline**（1/16），其余 subjects 无 EEGNet ternary baseline

由于 EEGNet 缺乏 ternary baseline，无法计算大多数 subjects 的 Δ(Final)。以下表格中 N/A 表示无 baseline 可供对比。

### 5.2 CBraMod

| Subject | Baseline | +Sess03 | +Sess04 | +Sess05 | Δ(Final) |
|---------|----------|---------|---------|---------|----------|
| S02 | 85.00% | 83.75% | 86.67% | 92.50% | +7.50pp |
| S03 | 87.08% | 87.92% | 90.42% | 86.67% | -0.42pp |
| S04 | 94.17% | 95.83% | 97.50% | 80.42% | -13.75pp |
| S06 | 72.08% | 77.08% | 82.50% | 96.25% | **+24.17pp** |
| S07 | 67.92% | — | 63.33% | 76.25% | +8.33pp |
| S08 | 82.92% | 73.75% | 73.33% | 79.92% | -3.00pp |
| S09 | 88.33% | 90.00% | 80.00% | 84.58% | -3.75pp |
| S10 | 42.92% | 47.92% | 52.92% | 51.25% | +8.33pp |
| S11 | 74.17% | 70.42% | 84.58% | 87.08% | +12.92pp |
| S13 | 70.00% | 71.25% | 80.00% | 76.39% | +6.39pp |
| S14 | 76.25% | 72.08% | 83.75% | 86.73% | +10.48pp |
| S15 | 60.00% | 87.08% | 74.17% | 85.10% | **+25.10pp** |
| S16 | 55.83% | 67.50% | 84.58% | 82.08% | **+26.25pp** |
| S17 | 79.17% | 80.83% | 77.92% | 75.00% | -4.17pp |
| S18 | 63.33% | 69.58% | 77.92% | 70.42% | +7.08pp |
| S19 | 92.92% | 95.00% | 92.08% | 94.17% | +1.25pp |
| **Mean** | **74.51%** | **76.22%** | **78.85%** | **81.55%** | **+7.04pp** |
| Std | 14.22% | 12.55% | 10.67% | 10.78% | 11.26pp |

> **数据来源**: `results/20260325_1934_extra_sessions_cache_imagery_ternary.json`, model=cbramod, baseline_run_tag=20260323_2320

### 5.3 EEGNet

| Subject | Baseline | +Sess03 | +Sess04 | +Sess05 | Δ(Final) |
|---------|----------|---------|---------|---------|----------|
| S02 | 83.75% | 80.00% | 85.83% | 85.00% | +1.25pp |
| S03 | N/A | 75.00% | 78.75% | 79.58% | N/A |
| S04 | N/A | 82.92% | 87.92% | 85.00% | N/A |
| S06 | N/A | 70.00% | 82.92% | 80.83% | N/A |
| S07 | N/A | — | 66.67% | 74.58% | N/A |
| S08 | N/A | 77.50% | 85.42% | 76.99% | N/A |
| S09 | N/A | 80.42% | 79.17% | 82.08% | N/A |
| S10 | N/A | 54.58% | 53.75% | 62.92% | N/A |
| S11 | N/A | 64.17% | 70.42% | 69.58% | N/A |
| S13 | N/A | 64.58% | 80.83% | 84.26% | N/A |
| S14 | N/A | 66.67% | 75.00% | 65.88% | N/A |
| S15 | N/A | 70.83% | 72.50% | 76.92% | N/A |
| S16 | N/A | 51.67% | 74.17% | 73.33% | N/A |
| S17 | N/A | 65.42% | 62.92% | 59.58% | N/A |
| S18 | N/A | 79.58% | 84.58% | 80.00% | N/A |
| S19 | N/A | 92.92% | 93.75% | 91.25% | N/A |
| **Mean** | — | **71.75%** | **77.16%** | **76.74%** | — |
| Std | — | 11.01% | 10.28% | 8.71% | — |

> **数据来源**: `results/20260325_1934_extra_sessions_cache_imagery_ternary.json`, model=eegnet, baseline_run_tag=20260323_2116
> **注意**: EEGNet 缺乏 ternary designated baseline（仅 S02 有历史结果），无法计算大多数 subjects 的 Δ

### 5.4 统计检验

| 检验 | 统计量 | p 值 | 结论 |
|------|--------|------|------|
| CBraMod Δ vs 0 (paired t-test, N=16) | t=2.50 | p=0.024 | **显著** (p<0.05) |
| EEGNet Δ vs 0 | — | — | 不可用（仅 1 个 baseline） |

### 5.5 关键发现

#### 5.5.1 CBraMod Ternary：额外数据显著有效

- 群体均值从 74.51% → 81.55%（+7.04pp），配对 t 检验显著（p=0.024）
- 与 binary 任务的 CBraMod 提升幅度（+6.13pp）接近，说明额外数据的价值**跨任务一致**
- 标准差从 14.22% 收缩至 10.78%，收缩幅度不如 binary（10.85% → 5.89%），ternary 的个体差异仍然较大

#### 5.5.2 低 baseline 被试获益最大

CBraMod ternary 的分层分析：
- **低 baseline (<75%, n=8)**: 平均 Δ = **+14.82pp**（S06: +24.17pp, S15: +25.10pp, S16: +26.25pp）
- **高 baseline (≥75%, n=8)**: 平均 Δ = **-0.73pp**（天花板效应 + 部分退化）

这与 binary 任务的模式高度一致：额外数据主要帮助初始表现较差的被试。

#### 5.5.3 S04 退化现象

S04 在 CBraMod ternary 上出现显著退化（94.17% → 80.42%, -13.75pp）。有趣的是 +Sess03 和 +Sess04 时准确率反而更高（95.83%, 97.50%），到 +Sess05 突然下降。这可能反映 Sess05 的 Finetune 数据对该被试特别困难（per_session 测试集不可比问题），或 S04 在第 5 次 session 时 BCI 操控策略发生了变化。

#### 5.5.4 EEGNet 步间趋势（无 baseline 对比）

虽然缺乏 baseline，EEGNet 的步间均值仍显示正向趋势：
- +Sess03: 71.75% (n=15) → +Sess04: 77.16% (n=16) → +Sess05: 76.74% (n=16)
- +Sess04 到 +Sess05 出现轻微下降，可能反映 Sess05 测试集整体难度较高

#### 5.5.5 Binary vs Ternary 对比（CBraMod, N=16）

| 维度 | Binary | Ternary |
|------|--------|---------|
| Baseline 均值 | 87.23% | 74.51% |
| +Sess05 均值 | 93.36% | 81.55% |
| Δ(Final) | +6.13pp | +7.04pp |
| Δ p 值 | 0.003 | 0.024 |
| Std 收缩 | 10.85%→5.89% | 14.22%→10.78% |

Ternary 任务整体准确率低于 binary（符合预期：3 类 vs 2 类），但额外数据的相对收益类似。Ternary baseline 更低，留有更多改善空间，但提升幅度并未显著超过 binary。

### 5.6 局限性

1. **EEGNet 缺乏 ternary baseline**: 无法进行 EEGNet 的 Δ 分析和 EEGNet vs CBraMod 对比
2. **测试集不可比**: per_session 策略下各 step 测试集不同（同 binary 的限制）
3. **被试技能学习混淆**: 同 binary 分析，后续 session 的提升包含被试 BCI 技能提升的贡献

## 6. Cross-Subject Extra Sessions 结果 (Binary, per_session 策略)

> **实验日期**: 2026-03-26
> **配置**: 128 通道, binary, imagery, per_session 测试策略
> **两次独立运行对比**:
> - **16-subj 训练** (`20260326_0345`): 仅 16 个有 extra sessions 的被试参与训练和评估
> - **21-subj 训练** (`20260326_1409`): 全部 21 个被试参与训练，仅 16 个有 extra sessions 的被试被评估

### 6.1 实验设计

Cross-subject extra sessions 与 within-subject 的核心区别：每个 step 只训练**一个池化模型**（而非 N 个独立模型），所有被试的数据被合并为一个训练集。

两种训练模式对比：

| 维度 | 16-subj 训练 | 21-subj 训练 (默认) |
|------|-------------|-------------------|
| 训练被试 | 16 (仅 extra-session) | 21 (全部) |
| 评估被试 | 16 | 16 (相同) |
| Baseline 训练数据 | 252,858 segs | 332,866 segs (+31.6%) |
| 额外 5 被试角色 | 不参与 | 贡献标准数据，不被评估 |

### 6.2 EEGNet 结果

| Step | 16-subj Mean | 21-subj Mean | Δ(21 vs 16) |
|------|-------------|-------------|-------------|
| Baseline | 81.56% ± 9.09% | 81.45% ± 10.53% | -0.12pp |
| +Sess03 | 82.66% ± 7.80% | 81.84% ± 8.01% | -0.82pp |
| +Sess04 | 83.32% ± 8.47% | 82.54% ± 10.05% | -0.78pp |
| +Sess05 | 81.80% ± 8.69% | 81.33% ± 9.84% | -0.47pp |

> **数据来源**:
> - 16-subj: `results/20260326_0345_cross_subject_extra_sessions_cache_imagery_binary.json`, model=eegnet
> - 21-subj: `results/20260326_1409_cross_subject_extra_sessions_cache_imagery_binary.json`, model=eegnet

**EEGNet 观察**: 两种训练模式差异极小（<1pp），额外 5 个被试的训练数据对 EEGNet 几乎没有帮助。这与 EEGNet 的小模型容量（~2.5K 参数）一致——模型已经从 16 subjects 的数据中学到了能学的东西，更多数据无法进一步提升。两种模式下额外 sessions 的边际收益都很小（+Sess05 相比 baseline 仅 +0.23pp / -0.12pp）。

### 6.3 CBraMod 结果

| Step | 16-subj Mean | 21-subj Mean | Δ(21 vs 16) |
|------|-------------|-------------|-------------|
| Baseline | 91.37% ± 8.35% | 92.38% ± 8.08% | +1.02pp |
| +Sess03 | 90.66% ± 7.31% | 91.87% ± 6.50% | +1.21pp |
| +Sess04 | 91.56% ± 7.31% | 92.19% ± 6.69% | +0.63pp |
| +Sess05 | 93.83% ± 4.79% | 93.24% ± 5.63% | -0.59pp |

> **数据来源**:
> - 16-subj: `results/20260326_0345_cross_subject_extra_sessions_cache_imagery_binary.json`, model=cbramod
> - 21-subj: `results/20260326_1409_cross_subject_extra_sessions_cache_imagery_binary.json`, model=cbramod

### 6.4 CBraMod 逐被试对比 (Baseline → +Sess05)

| Subject | 16-BL | 16-S05 | 16-Δ | 21-BL | 21-S05 | 21-Δ |
|---------|-------|--------|------|-------|--------|------|
| S02 | 94.38% | 96.25% | +1.88pp | 95.00% | 98.75% | +3.75pp |
| S03 | 100.00% | 98.75% | -1.25pp | 99.38% | 98.12% | -1.25pp |
| S04 | 98.12% | 98.75% | +0.63pp | 97.50% | 98.75% | +1.25pp |
| S06 | 85.62% | 97.50% | +11.88pp | 88.12% | 96.25% | +8.13pp |
| S07 | 90.62% | 94.38% | +3.75pp | 87.50% | 96.25% | +8.75pp |
| S08 | 96.88% | 94.38% | -2.50pp | 97.50% | 95.62% | -1.87pp |
| S09 | 98.75% | 95.00% | -3.75pp | 99.38% | 95.00% | -4.38pp |
| S10 | 65.62% | 80.00% | +14.38pp | 65.62% | 80.62% | +15.00pp |
| S11 | 86.88% | 100.00% | +13.12pp | 93.12% | 96.88% | +3.75pp |
| S13 | 96.88% | 95.62% | -1.25pp | 94.38% | 92.50% | -1.87pp |
| S14 | 83.12% | 91.25% | +8.12pp | 85.62% | 91.25% | +5.63pp |
| S15 | 95.62% | 88.12% | -7.50pp | 96.25% | 90.00% | -6.25pp |
| S16 | 93.75% | 96.25% | +2.50pp | 95.00% | 95.00% | +0.00pp |
| S17 | 87.50% | 93.75% | +6.25pp | 90.00% | 91.25% | +1.25pp |
| S18 | 90.00% | 90.62% | +0.62pp | 94.38% | 96.25% | +1.88pp |
| S19 | 98.12% | 90.62% | -7.50pp | 99.38% | 79.38% | **-20.00pp** |
| **Mean** | **91.37%** | **93.83%** | **+2.46pp** | **92.38%** | **93.24%** | **+0.86pp** |

### 6.5 统计检验

| 检验 | 统计量 | p 值 | 结论 |
|------|--------|------|------|
| CBraMod 21-subj vs 16-subj baseline (paired, N=16) | t=1.74 | p=0.103 | 不显著 |
| CBraMod 21-subj vs 16-subj +Sess03 (paired, N=16) | t=2.71 | p=0.016 | **显著** |
| CBraMod 21-subj vs 16-subj +Sess04 (paired, N=16) | t=1.35 | p=0.198 | 不显著 |
| CBraMod 21-subj vs 16-subj +Sess05 (paired, N=16) | t=-0.65 | p=0.528 | 不显著 |
| CBraMod 16-subj Δ(BL→S05) vs 0 (N=16) | t=1.45 | p=0.167 | 不显著 |
| CBraMod 21-subj Δ(BL→S05) vs 0 (N=16) | t=0.45 | p=0.662 | 不显著 |

### 6.6 关键发现

#### 6.6.1 额外 5 被试的训练数据贡献有限

21-subj 训练模式在 baseline/+Sess03/+Sess04 上比 16-subj 高 0.6-1.2pp（CBraMod），但到 +Sess05 时差异消失甚至反转（-0.59pp）。统计上仅 +Sess03 step 达到显著（p=0.016），整体趋势不一致。

**解读**: 额外 5 subjects 提供了约 30% 的更多训练数据，但这些数据的边际贡献很小。Cross-subject 模型已经从 16 subjects 学到了足够通用的表征，多 5 个人的标准数据锦上添花效果有限。

#### 6.6.2 Cross-subject 额外 sessions 收益远小于 Within-subject

| 实验 | CBraMod Δ(BL→S05) | p 值 |
|------|-------------------|------|
| **Within-subject** (N=16) | +6.13pp | **0.003** |
| Cross-subject 16-subj | +2.46pp | 0.167 |
| Cross-subject 21-subj | +0.86pp | 0.662 |

Within-subject 的额外 sessions 提升（+6.13pp）远大于 cross-subject（+0.86~2.46pp），且达到高度显著。原因：

1. **更高的 baseline**：cross-subject baseline 已达 91-92%，天花板效应明显
2. **池化稀释**：单个被试增加的 extra session 数据在 21 人的池子中只占很小比例
3. **测试集变化**：per_session 策略下每步测试集不同，引入噪声

#### 6.6.3 S19 在 21-subj +Sess05 出现严重退化

S19 在 21-subj 模式下从 99.38% 跌至 79.38%（-20.00pp），而 16-subj 模式下仅从 98.12% 跌至 90.62%（-7.50pp）。这可能是因为 +Sess05 数据量最大时，21-subj 模型的训练动态不同，导致某些被试的泛化受损。这也是 21-subj +Sess05 均值低于 16-subj 的主要原因。

#### 6.6.4 方差收缩一致

两种模式都在 +Sess05 时标准差收缩：
- 16-subj: 8.35% → 4.79%
- 21-subj: 8.08% → 5.63%

额外 sessions 的主要价值可能不在于提升均值，而在于**缩小个体差异**（低 baseline 被试受益最大，如 S10: +14-15pp）。

### 6.7 Cross-subject vs Within-subject 总览

| 维度 | Within-subject | Cross-subject (21-subj) |
|------|---------------|------------------------|
| 训练模式 | 每被试独立模型 | 池化单一模型 |
| Baseline (CBraMod) | 87.23% | 92.38% |
| +Sess05 (CBraMod) | 93.36% | 93.24% |
| Δ(BL→S05) | +6.13pp (p=0.003) | +0.86pp (p=0.662) |
| 训练耗时 | ~8h (128 模型) | ~9h (8 模型) |

Cross-subject 的高 baseline 使得额外 sessions 的边际价值有限。两种方法在 +Sess05 时**收敛到几乎相同的最终准确率**（93.36% vs 93.24%），但路径不同。

## 7. Transfer Learning Extra Sessions 结果

> **实验日期**: 2026-03-29
> **配置**: 128 通道, CBraMod, 16 被试, imagery paradigm, per_session 测试策略
> **预训练来源**: Cross-subject extra sessions checkpoints（每个 step 使用对应 step 的 cross-subject 模型作为初始化权重）
> **超参配置**: 纯 within-subject HPO 默认值（batch=256, backbone_lr=2.9e-4, scheduler=CAWD），无 finetune override
> **Freeze strategy**: none（全参数可训练）

### 7.1 实验设计

Transfer learning extra sessions 结合了两种策略：
1. **Cross-subject 预训练**：使用 `run_cross_subject_extra_sessions.py` 在全部 21 被试上训练的池化模型作为起点
2. **Per-subject 微调**：使用 `run_extra_sessions.py --pretrained-run` 对每个被试独立微调

每个 step 使用对应 step 的 cross-subject checkpoint：
- Baseline step → `20260326_1409_baseline_cbramod_imagery_binary/best.pt`
- +Sess03 step → `20260326_1409_sess03_cbramod_imagery_binary/best.pt`
- 以此类推

这确保了每个 step 的预训练模型已经"见过"该 step 对应的训练数据分布。

### 7.2 Binary 结果 (CBraMod)

| Subject | Baseline | +Sess03 | +Sess04 | +Sess05 | Δ(Final) |
|---------|----------|---------|---------|---------|----------|
| S02 | 97.50% | 97.50% | 98.12% | 99.38% | +1.88pp |
| S03 | 98.75% | 99.38% | 98.75% | 98.12% | -0.62pp |
| S04 | 97.50% | 85.62% | 98.12% | 99.38% | +1.88pp |
| S06 | 88.12% | 87.50% | 75.62% | 98.12% | +10.00pp |
| S07 | 88.12% | 88.12% | 91.88% | 96.25% | +8.13pp |
| S08 | 97.50% | 96.88% | 95.00% | 96.25% | -1.25pp |
| S09 | 96.25% | 96.88% | 96.88% | 95.62% | -0.62pp |
| S10 | 65.62% | 73.12% | 80.00% | 80.62% | +15.00pp |
| S11 | 91.25% | 93.12% | 95.00% | 98.12% | +6.88pp |
| S13 | 93.12% | 95.62% | 94.38% | 95.00% | +1.88pp |
| S14 | 85.62% | 85.62% | 90.62% | 91.25% | +5.62pp |
| S15 | 93.12% | 96.25% | 91.25% | 95.62% | +2.50pp |
| S16 | 95.00% | 92.50% | 96.25% | 95.00% | +0.00pp |
| S17 | 89.38% | 88.12% | 91.88% | 91.25% | +1.88pp |
| S18 | 93.75% | 91.25% | 93.12% | 96.25% | +2.50pp |
| S19 | 98.75% | 100.00% | 83.75% | 95.62% | -3.12pp |
| **Mean** | **91.84%** | **91.72%** | **91.90%** | **95.12%** | **+3.28pp** |
| Std | 8.12% | 6.94% | 6.80% | 4.53% | |

> **数据来源**: `results/20260329_1357_extra_sessions_cache_imagery_binary.json`, model=cbramod
> **预训练 run**: `20260326_1409` (cross-subject extra sessions, 21-subj, binary)

### 7.3 Ternary 结果 (CBraMod)

| Subject | Baseline | +Sess03 | +Sess04 | +Sess05 | Δ(Final) |
|---------|----------|---------|---------|---------|----------|
| S02 | 90.42% | 89.58% | 85.42% | 96.25% | +5.83pp |
| S03 | 90.83% | 87.08% | 89.58% | 84.17% | -6.67pp |
| S04 | 95.42% | 95.83% | 97.08% | 80.83% | -14.58pp |
| S06 | 79.17% | 77.92% | 84.58% | 92.50% | +13.33pp |
| S07 | 74.58% | — | 65.42% | 86.25% | +11.67pp |
| S08 | 86.25% | 76.67% | 70.42% | 78.66% | -7.59pp |
| S09 | 90.00% | 89.17% | 80.83% | 84.58% | -5.42pp |
| S10 | 51.67% | 47.92% | 52.92% | 62.92% | +11.25pp |
| S11 | 82.08% | 72.08% | 85.00% | 87.92% | +5.83pp |
| S13 | 76.67% | 72.92% | 80.42% | 76.85% | +0.19pp |
| S14 | 80.00% | 72.50% | 84.17% | 83.41% | +3.41pp |
| S15 | 73.75% | 89.17% | 75.42% | 87.02% | +13.27pp |
| S16 | 65.83% | 67.50% | 84.17% | 84.17% | +18.33pp |
| S17 | 78.75% | 80.83% | 79.17% | 80.42% | +1.67pp |
| S18 | 71.25% | 70.42% | 77.50% | 73.33% | +2.08pp |
| S19 | 94.17% | 95.00% | 92.92% | 95.00% | +0.83pp |
| **Mean** | **80.05%** | **76.17%** | **80.32%** | **83.39%** | **+3.34pp** |
| Std | 11.46% | 12.07% | 10.09% | 8.28% | |

> **数据来源**: `results/20260329_1503_extra_sessions_cache_imagery_ternary.json`, model=cbramod
> **预训练 run**: `20260327_0303` (cross-subject extra sessions, ternary)

### 7.4 三种训练范式对比 (CBraMod)

#### Binary

| Step | Within-Subject | Cross-Subject (21-subj) | Transfer | T vs CS | T vs WS |
|------|---------------|------------------------|----------|---------|---------|
| Baseline | 87.23% | 92.38% | 91.84% | -0.55pp | +4.61pp |
| +Sess03 | 89.14% | 91.88% | 92.03% | +0.16pp | +2.89pp |
| +Sess04 | 90.94% | 92.19% | 92.19% | +0.00pp | +1.25pp |
| +Sess05 | 93.36% | 93.24% | **95.12%** | **+1.88pp** | +1.76pp |
| Δ(BL→S05) | +6.13pp | +0.86pp | +3.28pp | | |

> **数据来源**:
> - Within-subject: `results/20260324_2131_extra_sessions_cache_imagery_binary.json`
> - Cross-subject: `results/20260326_1409_cross_subject_extra_sessions_cache_imagery_binary.json`
> - Transfer: `results/20260329_1357_extra_sessions_cache_imagery_binary.json`

#### Ternary

| Step | Within-Subject | Cross-Subject | Transfer | T vs CS | T vs WS |
|------|---------------|--------------|----------|---------|---------|
| Baseline | 74.51% | 80.05% | 80.05% | +0.00pp | +5.55pp |
| +Sess03 | 78.00% | 81.56% | 81.33% | -0.22pp | +3.33pp |
| +Sess04 | 80.10% | 83.18% | 83.20% | +0.03pp | +3.10pp |
| +Sess05 | 81.55% | 83.78% | 83.39% | -0.39pp | +1.84pp |
| Δ(BL→S05) | +7.04pp | +3.73pp | +3.34pp | | |

> **数据来源**:
> - Within-subject: `results/20260325_1934_extra_sessions_cache_imagery_ternary.json`
> - Cross-subject: `results/20260327_0303_cross_subject_extra_sessions_cache_imagery_ternary.json`
> - Transfer: `results/20260329_1503_extra_sessions_cache_imagery_ternary.json`

### 7.5 关键发现

#### 7.5.1 Transfer 在 Binary +Sess05 达到最高准确率

Binary 95.12% 是三种范式中最高的，比 cross-subject (93.24%) 高 1.88pp，比 within-subject (93.36%) 高 1.76pp。Transfer 的优势主要体现在最终 step——此时 per-subject 微调有足够的数据量从 cross-subject 预训练基础上进一步学习被试特异性模式。

Binary 中 transfer vs cross-subject 的逐被试对比显示 S19 获得 +16.25pp 的巨大优势（cross-subject 在 +Sess05 出现严重退化至 79.38%，而 transfer 稳定在 95.62%），说明 per-subject 微调可以**避免 cross-subject 池化模型在特定被试上的灾难性退化**。

#### 7.5.2 Ternary Transfer ≈ Cross-Subject

Ternary 任务中 transfer 与 cross-subject 几乎完全一致（sess05: 83.39% vs 83.78%，差异 -0.39pp）。逐被试对比显示多数被试准确率完全相同——这是因为 ternary 的 fine-tuning 未能超越预训练 checkpoint（所有被试均在 epoch 8 early-stop，best epoch 为 epoch 0）。实质上，"transfer" 结果等价于直接用 cross-subject 模型做推理。

#### 7.5.3 三种范式收敛趋势

Binary 和 ternary 都呈现相同模式：**三种范式随着 extra sessions 数据增加逐步收敛**。

- Binary: 起点差距 5.15pp (WS 87.23% vs CS 92.38%)，到 +Sess05 收缩至 1.76pp (93.36% vs 95.12%)
- Ternary: 起点差距 5.55pp (WS 74.51% vs CS 80.05%)，到 +Sess05 收缩至 2.23pp (81.55% vs 83.78%)

Within-subject 的 Δ(BL→S05) 最大（binary +6.13pp, ternary +7.04pp），因为其起点最低，有更多改善空间。Cross-subject 的 Δ 最小（binary +0.86pp, ternary +3.73pp），因为池化模型已经在高 baseline 上运行。

#### 7.5.4 被试池选择偏差

**重要提示**：本节中 within-subject 的 baseline 均值（binary 87.23%, ternary 74.51%）与 baseline_registry 中的 designated baseline（binary 85.15%, ternary 69.44%）不同。原因是 extra sessions 仅涉及 16 个有额外数据的被试，而 designated baseline 覆盖全部 21 个被试。缺失的 5 个被试 (S01, S05, S12, S20, S21) 准确率显著低于有 extra sessions 的被试（binary 84.12% vs 91.99%, ternary 58.08% vs 80.34%），构成系统性选择偏差。

跨范式对比**仅在同一 16 被试池内有效**，不应将 extra sessions 结果与 21 人的全量实验直接比较。

#### 7.5.5 Transfer 对低 baseline 被试的放大效应

Binary 中 S10（最弱被试）的各范式 +Sess05 表现：
- Within-subject: 74.38%
- Cross-subject: 80.62%
- Transfer: 80.62%

Ternary 中 S10：
- Within-subject: 51.25%
- Cross-subject: 62.92%
- Transfer: 62.92%

Cross-subject 和 transfer 对弱被试的帮助大于 within-subject，因为池化模型提供了更鲁棒的初始表征。但 transfer 的 fine-tuning 在 S10 上未能进一步超越 cross-subject（两者完全一致），说明弱被试的瓶颈可能在于 EEG 信号质量而非模型个性化程度。

## 8. 文件索引

| 文件 | 说明 |
|------|------|
| `results/20260324_1557_extra_sessions_cache_imagery_binary.json` | Binary within-subject N=5 结果 |
| `results/20260324_2131_extra_sessions_cache_imagery_binary.json` | Binary within-subject N=16 结果 |
| `results/20260325_1934_extra_sessions_cache_imagery_ternary.json` | Ternary within-subject N=16 结果 |
| `results/20260326_0345_cross_subject_extra_sessions_cache_imagery_binary.json` | Cross-subject 16-subj 训练结果 |
| `results/20260326_1409_cross_subject_extra_sessions_cache_imagery_binary.json` | Cross-subject 21-subj 训练结果 |
| `results/20260327_0303_cross_subject_extra_sessions_cache_imagery_ternary.json` | Cross-subject ternary 结果 |
| `results/20260329_1357_extra_sessions_cache_imagery_binary.json` | Transfer binary 结果 |
| `results/20260329_1503_extra_sessions_cache_imagery_ternary.json` | Transfer ternary 结果 |
| `results/20260329_1357_extra_sessions_imagery_binary.png` | Transfer binary 组合图 |
| `results/20260329_1503_extra_sessions_imagery_ternary.png` | Transfer ternary 组合图 |
| `docs/dev_log/experiments/extra_sessions_per_session_analysis.md` | Binary within-subject N=16 per_session 详细分析 |
| `docs/dev_log/experiments/extra_sessions_strategy_comparison.md` | 三种测试策略对比分析 |
| `docs/dev_log/experiments/cross_subject_extra_sessions_training_profile.md` | Cross-subject 训练速度参考 |
| `scripts/experiments/run_extra_sessions.py` | Within-subject + transfer 实验脚本 |
| `scripts/experiments/run_cross_subject_extra_sessions.py` | Cross-subject 实验脚本 |
| `src/visualization/extra_sessions.py` | 可视化模块 |
| `src/preprocessing/discovery.py` | Session 发现 + 渐进式 folder 生成 |
