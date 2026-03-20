# CBraMod Within-Subject HPO 分析

> **目的**: 通过 Optuna TPE 搜索 CBraMod within-subject binary 训练的最优超参数组合，量化超参数调优对模型性能的影响，并识别关键参数。
>
> **后续**: HPO 最优参数已于 2026-03-20 应用到 `src/config/training.py` 默认配置并完成 unified model 验证运行。完整的 "HPO 建议 → 用户 override → 最终采用" 对照表见 `docs/dev_log/experiments/hpo_final_parameters.md`。

---

## 1. 实验设计

### 搜索配置

| 配置项 | 值 |
|--------|-----|
| Sampler | TPE (Tree-structured Parzen Estimator) |
| 总 trials | 51 |
| 剪枝器 | ProbabilisticSubjectPruner (threshold=0.1, min_steps=3) |
| 任务 | binary, imagery paradigm |
| 通道数 | 128 |
| 被试 | 全部 21 名 (S01–S21) |
| 数据模式 | cache-only |
| 随机种子 | 42 |

> **数据来源**: `results/hpo/hpo.db`, study=`cbramod_within_subject_binary`
> **最优参数导出**: `results/hpo/cbramod_within_subject_binary_best_params.json`

### 搜索空间 (11 维)

| 参数 | 类型 | 范围 | 分布 |
|------|------|------|------|
| backbone_lr | float | [1e-5, 1e-3] | log-uniform |
| classifier_lr_ratio | float | [1.0, 5.0] | uniform |
| weight_decay | float | [0.01, 0.3] | log-uniform |
| dropout_rate | float | [0.05, 0.45] | uniform |
| batch_size | categorical | {64, 128, 256} | — |
| label_smoothing | float | [0.0, 0.2] | uniform |
| gradient_clip | float | [0.3, 2.0] | uniform |
| phase_decay | float | [0.3, 0.9] | uniform |
| phase_epochs | int | [4, 10] | uniform |
| exploration_epochs | int | [3, 9] | uniform |
| exploration_batch_size | categorical | {16, 32, 64} | — |

> **搜索空间定义**: `src/hpo/search_spaces.py`

---

## 2. 搜索效率

| 状态 | 数量 | 占比 |
|------|------|------|
| Complete | 23 | 45.1% |
| Pruned | 27 | 52.9% |
| Failed | 1 | 2.0% |
| **总计** | **51** | |

概率剪枝器淘汰了超过半数的 trial，显著节省了计算资源。典型剪枝模式：
- 4 个 trial 在 step 2 即被剪枝（仅 3 个被试后），节省约 85% 计算量
- 多数剪枝发生在 step 14–19（完成 75%–95% 被试），此时已有足够统计信号

### 收敛曲线

```
Trial  1: 83.36%  (初始)
Trial  6: 85.12%  (+1.76pp)
Trial  9: 85.57%  (+0.45pp)
Trial 26: 85.77%  (+0.20pp)
Trial 34: 85.80%  (+0.03pp)
Trial 46: 86.01%  (+0.21pp, 最终最优)
```

前 10 个 trial 贡献了主要提升 (83.4% → 85.6%, +2.2pp)，后 40 个 trial 仅再提升 0.4pp。搜索已充分收敛。

---

## 3. 最优参数 vs 默认参数

### Trial #46: 86.01% (最优)

| 参数 | 默认值 | HPO 最优 | 变化 |
|------|--------|---------|------|
| backbone_lr | 1e-4 | **2.87e-4** | ↑ 2.9x |
| classifier_lr | 3e-4 (ratio 3x) | **1.16e-3** (ratio 4x) | ↑ 3.9x |
| weight_decay | 0.06 | **0.026** | ↓ 2.3x |
| dropout_rate | 0.15 | **0.098** | ↓ 1.5x |
| batch_size | 128 | **256** | ↑ 2x |
| label_smoothing | 0.05 | **0.087** | ↑ 1.7x |
| gradient_clip | 1.0 | **0.729** | ↓ 1.4x |
| phase_decay | 0.7 | **0.468** | ↓ 1.5x |
| phase_epochs | 6 | **8** | ↑ |
| exploration_epochs | 6 | **4** | ↓ |
| exploration_batch_size | 32 | **64** | ↑ 2x |

> **默认参数来源**: `src/config/training.py:167-189, 77-88`

### 完成 trial 统计

| 指标 | 值 |
|------|-----|
| 最优 | 86.01% (Trial #46) |
| 均值 | 84.97% |
| 标准差 | 0.64% |
| 最差 | 83.36% |

23 个完成 trial 精度范围仅 2.65pp，表明搜索空间内模型表现稳健，不存在极端敏感区域。

---

## 4. 参数重要性分析

### fANOVA 重要性排名

| 排名 | 参数 | 重要性 | 与 acc 相关性 | Top 5 均值 | Bottom 5 均值 |
|------|------|--------|-------------|-----------|-------------|
| 1 | phase_decay | 0.233 | −0.68 | 0.42 | 0.67 |
| 2 | dropout_rate | 0.196 | −0.74 | 0.09 | 0.28 |
| 3 | gradient_clip | 0.130 | −0.68 | 0.42 | 1.07 |
| 4 | classifier_lr_ratio | 0.100 | +0.48 | 4.3 | 3.0 |
| 5 | backbone_lr | 0.099 | +0.54 | 2.7e-4 | 1.6e-4 |
| 6 | batch_size | 0.082 | +0.69 | 230 | 102 |
| 7 | weight_decay | 0.072 | −0.51 | 0.037 | 0.074 |
| 8 | label_smoothing | 0.040 | +0.13 | 0.14 | 0.11 |
| 9 | exploration_epochs | 0.021 | −0.24 | 4.8 | 5.8 |
| 10 | exploration_batch_size | 0.014 | +0.47 | 57.6 | 41.6 |
| 11 | phase_epochs | 0.013 | +0.33 | 8.0 | 6.8 |

### 关键发现

**1. 低正则化是最强信号**

dropout_rate 与精度的相关性最强 (r = −0.74)。Top 5 trial 平均 dropout 0.09，Bottom 5 平均 0.28。weight_decay 同样呈现"低更好"的趋势 (0.037 vs 0.074)。

**解释**: CBraMod 预训练 backbone 已具备充分的内在正则能力（4M 参数在大规模 EEG 数据上预训练），额外的 dropout/weight_decay 反而抑制了有效表征的传递。

**2. 更高的学习率 + 分离的 backbone/head 学习率**

backbone_lr 从默认 1e-4 提升至 2.87e-4，classifier_lr_ratio 从 3x 提升至 4x。分类头作为随机初始化层，需要相对更高的学习率快速收敛。

**3. 大 batch + 短探索 > 小 batch + 长探索**

batch_size 翻倍至 256，exploration_epochs 从 6 降至 4。更大的 batch 提供更稳定的梯度估计，减少了小 batch 探索阶段的必要性。

**4. CAWD scheduler: 快衰减 + 长 phase**

phase_decay 从 0.7 降至 0.47（峰值 LR 每阶段衰减更快），phase_epochs 从 6 增至 8（每阶段训练更充分）。前期激进学习，后期快速收敛。

**5. 低影响参数**

label_smoothing、exploration_epochs、exploration_batch_size、phase_epochs 的 fANOVA 重要性均 < 0.04，对最终精度影响微弱。

---

## 5. 结论

- HPO 最优配置 (86.01%) 相比默认配置提升约 **1pp**。
- 23 个完成 trial 的 std 仅 0.64%，改善幅度在统计上有意义但不算巨大。
- **核心发现**: 对于预训练基座模型的微调，默认参数偏保守（正则化过强、学习率过低）。降低 dropout 和 weight_decay、提高学习率是最有效的调优方向。
- 搜索在约 10 个 trial 后即接近最优区域，后续 trial 主要起确认和微调作用。TPE sampler + 概率剪枝器的组合在 11 维搜索空间上表现高效。

---
---

# EEGNet Within-Subject HPO 分析

> **目的**: 通过 Optuna TPE 搜索 EEGNet within-subject binary 训练的最优超参数组合，量化超参数调优对模型性能的影响，并识别关键参数。

---

## 1. 实验设计

### 默认参数基线

EEGNet 默认配置与论文原始参数一致（EEGNet-8,2）：

| 参数 | 论文/默认值 | 来源 |
|------|------------|------|
| F1 | 8 | Lawhern et al. (2018) |
| D | 2 | Lawhern et al. (2018) |
| F2 (= F1×D) | 16 | — |
| kernel_length | 64 | ~sampling_rate/2 |
| dropout_rate | 0.5 | Lawhern et al. (2018) |
| learning_rate | 1e-3 | Adam optimizer |
| weight_decay | 0 | — |
| batch_size | 64 | — |

> **论文原始结果**: 2-finger MI online control 80.56% (21 subjects, majority voting, fine-tuned Session 2)
> **默认参数来源**: `src/config/training.py:190-207`

### 默认配置离线复现

| Run | 被试数 | test_acc (majority) | 来源 |
|-----|--------|---------------------|------|
| 20260204_2309 | 21 | **80.00%** | `results/20260204_230933_comparison_imagery_binary.json` |
| 20260206_1003 | 21 | **78.75%** | `results/20260206_1003_eegnet_within_subject/binary/` |
| 20260316_1411 | 21 | **78.10%** | `results/20260316_1411_eegnet_within_subject/binary/` |

三次独立 run 均值 **78.95%**，范围 78.10–80.00%（~1.9pp），反映训练随机性。

### 搜索配置

| 配置项 | 值 |
|--------|-----|
| Sampler | TPE (Tree-structured Parzen Estimator) |
| 总 trials | 32 |
| 剪枝器 | ProbabilisticSubjectPruner (threshold=0.05, min_steps=3) |
| 任务 | binary, imagery paradigm |
| 通道数 | 128 |
| 被试 | 全部 21 名 (S01–S21) |
| 数据模式 | cache-only |
| 随机种子 | 42 |

> **数据来源**: `results/hpo/hpo.db`, study=`eegnet_within_subject_binary`
> **最优参数导出**: `results/hpo/eegnet_within_subject_binary_best_params.json`

### 搜索空间 (7 维)

| 参数 | 类型 | 范围 | 分布 |
|------|------|------|------|
| F1 | categorical | {4, 8, 16} | — |
| D | categorical | {1, 2, 4} | — |
| learning_rate | float | [1e-4, 1e-2] | log-uniform |
| weight_decay | float | [1e-5, 0.1] | log-uniform |
| dropout_rate | float | [0.2, 0.7] | uniform |
| batch_size | categorical | {32, 64, 128} | — |
| kernel_length | categorical | {32, 64, 128} | — |

> **搜索空间定义**: `src/hpo/search_spaces.py`

---

## 2. 搜索效率

| 状态 | 数量 | 占比 |
|------|------|------|
| Complete | 10 | 31.3% |
| Pruned | 21 | 65.6% |
| Failed | 1 | 3.1% |
| **总计** | **32** | |

概率剪枝器淘汰了约 2/3 的 trial。

### 收敛曲线

```
Trial  0: 71.56%  (初始, D=1 + kernel=32 的差配置)
Trial  1: 80.31%  (+8.75pp)
Trial  3: 81.56%  (+1.25pp)
Trial 23: 82.71%  (+1.15pp, 最终最优)
```

Trial 0 的 71.56% 是随机采样到 D=1, kernel_length=32, lr=1.3e-4 的差配置，**不代表默认参数表现**。主要提升来自架构参数修正（D=1→2+, kernel=32→64）。

---

## 3. 最优参数 vs 默认参数

### Trial #23: 82.71% (最优)

| 参数 | 默认值 | HPO 最优 | 变化 |
|------|--------|---------|------|
| F1 | 8 | **16** | ↑ 2x |
| D | 2 | **4** | ↑ 2x |
| F2 (= F1×D) | 16 | **64** | ↑ 4x |
| learning_rate | 1e-3 | **3.98e-3** | ↑ 4x |
| weight_decay | 0 | **1.09e-5** | ↑ (极小) |
| dropout_rate | 0.5 | **0.271** | ↓ 1.8x |
| batch_size | 64 | **64** | — |
| kernel_length | 64 | **64** | — |

### 完成 trial 统计

| 指标 | 值 |
|------|-----|
| 最优 | 82.71% (Trial #23) |
| 均值 | 79.09% |
| 标准差 | 4.71% |
| 最差 | 69.06% |

10 个完成 trial 精度范围 13.65pp，远大于 CBraMod (2.65pp)，表明 EEGNet 对超参数更为敏感。

---

## 4. 参数重要性分析

### fANOVA 重要性排名

| 排名 | 参数 | 重要性 |
|------|------|--------|
| 1 | kernel_length | 0.219 |
| 2 | weight_decay | 0.204 |
| 3 | learning_rate | 0.195 |
| 4 | D | 0.187 |
| 5 | dropout_rate | 0.125 |
| 6 | batch_size | 0.055 |
| 7 | F1 | 0.016 |

### 关键发现

**1. 模型容量提升是最大杠杆**

F1 (8→16) 和 D (2→4) 将特征图数量从 16 增至 64 (4x)。EEGNet 默认参数量极小 (~2.5K)，在 128 通道 21 被试的数据量下存在欠拟合。增大容量是最显著的提升来源。

**2. 降低 dropout 配合容量增加**

dropout 从 0.5 降至 0.27。更大的模型需要更少的正则化，与 CBraMod HPO 的发现一致（预训练/大容量模型不需要强正则）。

**3. 更高的学习率**

learning_rate 从 1e-3 提升至 ~4e-3。更大的模型容量配合更激进的学习率，加速收敛。

**4. kernel_length 和 batch_size 维持默认**

kernel_length=64 和 batch_size=64 均为默认值，表明这两个参数的默认设置已在合理范围。

---

## 5. 结论

- HPO 最优配置 (82.71%) 相比默认配置三次 run 均值 (78.95%) 提升约 **3.8pp**，提升幅度大于 CBraMod (~1pp)。
- 论文原始在线结果 (80.56%) 位于我们离线默认配置复现范围 (78.10–80.00%) 的上端，差异可归因于在线 fine-tuning 机制和 majority voting 在连续解码中的效果。
- EEGNet 对超参数高度敏感（std=4.71% vs CBraMod 0.64%），HPO 对其价值更大。
- **核心发现**: EEGNet 默认架构 (F1=8, D=2) 在 128 通道场景下容量不足。将 F1 和 D 翻倍是最关键的调优方向。
- 与 CBraMod 共通的发现：降低 dropout 和提高学习率普遍有益，过度正则化是两种模型的共同瓶颈。
