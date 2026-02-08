# S10/S20 CBraMod 性能问题分析报告

生成时间: 2026-02-04 19:24:06

---

## 1. 执行摘要

### 1.1 问题概述

CBraMod 在 S10 和 S20 被试上的表现持续较差：

| 被试 | 任务 | EEGNet | CBraMod | 差异 | 问题类型 |
|------|------|--------|---------|------|----------|
| S10 | binary | 0.0% | 55.0% | +55.0% | both_low |
| S10 | ternary | 59.2% | 42.1% | -17.1% | normal |
| S20 | binary | 60.6% | 68.1% | +7.5% | both_low |
| S20 | ternary | 32.9% | 42.1% | +9.2% | both_low |

### 1.2 关键发现

**S10 - CBraMod 特有问题:**
- EEGNet (0.0%) 显著优于 CBraMod (55.0%)
- 说明该被试数据本身是可学习的，问题在于 CBraMod 的适配
- 可能原因：CBraMod 的预处理/频带特征与该被试不匹配

**S20 - 两模型均表现不佳:**
- CBraMod (68.1%) 略优于 EEGNet (60.6%)
- 两个模型都低于正常水平
- 最近的调度器更改后 CBraMod 性能有所提升

---

## 2. 详细分析

### 2.1 S10 分析

#### Binary 任务

**性能对比:**
- EEGNet: 0.0%
- CBraMod: 55.0%
- 问题类型: `both_low`

**CBraMod 诊断:**
- 主要问题: data_quality: severe_distribution_shift
- 数据质量评分: 0.10
- 泛化 Gap: 15.87%
- 训练稳定性: 0.99

**建议:**
- 【S10 两模型均表现不佳】
- EEGNet: 0.0%, CBraMod: 55.0%
- 该被试对两个模型都具有挑战性
- 建议:
- 1. 检查原始 EEG 数据质量 (信噪比、伪迹)
- 2. 该被试可能需要特殊的预处理参数
- 3. 尝试数据增强策略
- 4. 考虑使用跨被试预训练进行迁移学习
- 5. 检查该被试的运动想象能力是否与其他被试不同

#### Ternary 任务

**性能对比:**
- EEGNet: 59.2%
- CBraMod: 42.1%
- 问题类型: `normal`

**CBraMod 诊断:**
- 主要问题: data_quality: distribution_shift
- 数据质量评分: 0.10
- 泛化 Gap: 17.02%
- 训练稳定性: 0.99

**建议:**
- 【S10 性能正常】
- EEGNet: 59.2%, CBraMod: 42.1%
- 无需特殊处理

**模型对比分析:**
- CBraMod 在 S10 上存在特有问题。EEGNet (59.2%) 显著优于 CBraMod (42.1%)，说明该被试数据本身是可学习的。

### 2.2 S20 分析

#### Binary 任务

**性能对比:**
- EEGNet: 60.6%
- CBraMod: 68.1%
- 问题类型: `both_low`

**CBraMod 诊断:**
- 主要问题: low_performance: acc=68.12%
- 数据质量评分: 1.00
- 泛化 Gap: 13.42%
- 训练稳定性: 0.99

**建议:**
- 【S20 两模型均表现不佳】
- EEGNet: 60.6%, CBraMod: 68.1%
- 该被试对两个模型都具有挑战性
- 建议:
- 1. 检查原始 EEG 数据质量 (信噪比、伪迹)
- 2. 该被试可能需要特殊的预处理参数
- 3. 尝试数据增强策略
- 4. 考虑使用跨被试预训练进行迁移学习
- 5. 检查该被试的运动想象能力是否与其他被试不同

**模型对比分析:**
- S20 对两个模型都具有挑战性。CBraMod (68.1%) 略优于 EEGNet (60.6%)，但两者均低于正常水平。

#### Ternary 任务

**性能对比:**
- EEGNet: 32.9%
- CBraMod: 42.1%
- 问题类型: `both_low`

**CBraMod 诊断:**
- 主要问题: low_performance: acc=42.08%
- 数据质量评分: 1.00
- 泛化 Gap: 16.43%
- 训练稳定性: 0.99

**建议:**
- 【S20 两模型均表现不佳】
- EEGNet: 32.9%, CBraMod: 42.1%
- 该被试对两个模型都具有挑战性
- 建议:
- 1. 检查原始 EEG 数据质量 (信噪比、伪迹)
- 2. 该被试可能需要特殊的预处理参数
- 3. 尝试数据增强策略
- 4. 考虑使用跨被试预训练进行迁移学习
- 5. 检查该被试的运动想象能力是否与其他被试不同

**模型对比分析:**
- S20 对两个模型都具有挑战性。CBraMod (42.1%) 略优于 EEGNet (32.9%)，但两者均低于正常水平。

---

## 3. 可视化

### 3.1 Binary 任务

#### CBraMod Loss 曲线对比
![CBraMod Loss](figures/loss_comparison_cbramod_binary.png)

#### EEGNet Loss 曲线对比
![EEGNet Loss](figures/loss_comparison_eegnet_binary.png)

#### 性能概览
![Performance Overview](figures/performance_overview_binary.png)

### 3.2 Ternary 任务

#### CBraMod Loss 曲线对比
![CBraMod Loss](figures/loss_comparison_cbramod_ternary.png)

#### EEGNet Loss 曲线对比
![EEGNet Loss](figures/loss_comparison_eegnet_ternary.png)

#### 性能概览
![Performance Overview](figures/performance_overview_ternary.png)

#### S10 模型对比 (binary)
![S10 Comparison](figures/model_comparison_S10_binary.png)

#### S10 模型对比 (ternary)
![S10 Comparison](figures/model_comparison_S10_ternary.png)

#### S20 模型对比 (binary)
![S20 Comparison](figures/model_comparison_S20_binary.png)

#### S20 模型对比 (ternary)
![S20 Comparison](figures/model_comparison_S20_ternary.png)

---

## 4. 改进建议

### 4.1 针对 S10 的建议 (CBraMod 特有问题)

1. **检查预处理配置:**
   - 对比 S10 的 EEG 频带特征与其他被试的差异
   - CBraMod 使用 0.3-75Hz 滤波，可能不适合 S10

2. **调整模型策略:**
   - 尝试冻结 CBraMod backbone，只微调分类器头
   - 增加正则化 (dropout 从 0.1 提升到 0.3)

3. **数据适配:**
   - 检查 S10 的数据分布是否与预训练数据差异大
   - 考虑对 S10 使用特定的归一化参数

4. **模型选择:**
   - 对于 S10，EEGNet 可能是更好的选择

### 4.2 针对 S20 的建议 (两模型均较差)

1. **数据质量检查:**
   - 检查原始 EEG 数据的信噪比
   - 检查是否有严重的伪迹

2. **迁移学习:**
   - 使用跨被试预训练的权重
   - 在其他被试上预训练，然后在 S20 上微调

3. **训练策略:**
   - 调度器更改已经帮助了 CBraMod
   - 继续优化超参数 (学习率、epochs)

4. **数据增强:**
   - 尝试时间域和频率域的数据增强
   - 增加训练数据的多样性
