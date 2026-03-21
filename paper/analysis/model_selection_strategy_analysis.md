# 模型选择策略对比实验分析

> **目的**: 对比 4 种 model selection 策略（combined, val_acc, ema, soup）在 CBraMod binary 训练中对 within-subject、cross-subject 和 transfer learning 三种范式的影响，评估是否能通过改进模型选择标准提升测试准确率。

## 1. 实验背景

### 1.1 问题动机

此前分析发现，当前 `combined_score = (val_acc + majority_acc) / 2.0` 作为模型选择标准存在以下问题：

- **50% 被试** (10/20) 选出的 best epoch 的 test accuracy 低于某个更早 milestone 的 test accuracy
- majority_acc (trial-level) 的 epoch 间波动是 val_acc (segment-level) 的 **1.2-2.6 倍**（离散跳变）
- combined_score 门控 milestone 保存 → 10/21 被试有 val_acc 达到新高但 combined_score 未改善的 epoch，这些 checkpoint **从未被保存**
- 整体 oracle gap: -1.34%（即如果总能选到 test accuracy 最高的 milestone，可多得 1.34%）

### 1.2 HPO 偏置声明

**重要**: 本实验使用的训练超参数来自 HPO Trial #46（`results/hpo/cbramod_within_subject_binary_best_params.json`），该 HPO 的目标函数使用的是 `combined_score`（val_acc + majority_acc 的平均值）作为模型选择标准。

这意味着：
- HPO 搜索到的最优参数是**针对 combined 策略优化**的
- val_acc、ema、soup 策略使用的是为 combined 优化的参数，可能处于**不公平的劣势**
- 例如，HPO 可能倾向于找到让 majority_acc 表现好的参数组合，而这些参数在仅使用 val_acc 时未必最优
- **结论性判断需要在各策略各自完成 HPO 后才能做出**

> **数据来源**: HPO 参数 `results/hpo/cbramod_within_subject_binary_best_params.json` (Trial #46, value=0.8601)
> HPO 详细分析见 `paper/analysis/hpo_within_subject_analysis.md`

### 1.3 HPO 最优参数（所有策略共用）

| 参数 | HPO 最优值 |
|------|-----------|
| backbone_lr | 2.87e-4 |
| classifier_lr_ratio | 4.03 |
| weight_decay | 0.026 |
| dropout_rate | 0.098 |
| batch_size | 256 |
| label_smoothing | 0.087 |
| gradient_clip | 0.729 |
| phase_decay | 0.468 |
| phase_epochs | 8 |
| exploration_epochs | 4 |
| exploration_batch_size | 64 |

## 2. 实验设计

### 2.1 策略定义

| 策略 | selection_score 定义 | milestone 保存触发 | best_state 来源 | 后处理 |
|------|---------------------|-------------------|----------------|--------|
| `combined` (baseline) | `(val_acc + majority_acc) / 2` | selection_score 改善时 | 当前模型权重 | 无 |
| `val_acc` | `val_acc` (segment-level) | selection_score 改善时 | 当前模型权重 | 无 |
| `ema` | `(ema_val_acc + ema_majority_acc) / 2` | selection_score 改善时 | EMA shadow weights | 无 |
| `soup` | `(val_acc + majority_acc) / 2` (同 combined) | 同 combined | top-3 milestone 权重平均 | checkpoint averaging |

### 2.2 实现细节

- **EMA**: decay=0.998, 每 epoch 更新一次 shadow weights, validation 在 EMA 权重下执行
- **Soup**: 训练过程与 combined 完全一致，训练后加载 top-3 milestone checkpoint 取权重算术平均
- 所有策略使用相同的 CAWD scheduler, 相同的 HPO 最优参数, 相同的随机种子 (42)
- CBraMod 128ch, binary classification, within-subject paradigm, 21 subjects

## 3. 结果

### 3.1 总览

| 策略 | Mean | Median | Std | Min | Max | Run Tag |
|------|------|--------|-----|-----|-----|---------|
| **soup** | **85.09%** | 87.50% | **10.46%** | **61.88%** | 99.38% | 20260321_0343 |
| combined (baseline) | 84.85% | 87.50% | 10.73% | 60.62% | 99.38% | 20260320_2316 |
| val_acc | 84.73% | 86.25% | 10.96% | 61.25% | 99.38% | 20260321_0013 |
| ema | 71.90% | 76.88% | 14.19% | 50.00% | 97.50% | 20260321_0227 |

> **数据来源**:
> - combined: `results/20260320_2316_cbramod_within_subject/binary/S{01-21}/results.json`
> - val_acc: `results/20260321_0013_cbramod_within_subject/binary/S{01-21}/results.json`
> - ema: `results/20260321_0227_cbramod_within_subject/binary/S{01-21}/results.json`
> - soup: `results/20260321_0343_cbramod_within_subject/binary/S{01-21}/results.json`

### 3.2 逐被试对比

| Subject | combined | val_acc | ema | soup | 最优策略 |
|---------|----------|---------|-----|------|---------|
| S01 | 83.12% | 83.12% | 77.50% | **83.75%** | soup |
| S02 | **93.75%** | 93.75% | 77.50% | 93.12% | combined |
| S03 | 97.50% | 98.12% | 97.50% | **98.75%** | soup |
| S04 | **91.88%** | 91.88% | 91.25% | 91.88% | combined/soup |
| S05 | **85.00%** | 83.12% | 60.62% | 83.12% | combined |
| S06 | 73.75% | **74.38%** | 68.12% | 73.12% | val_acc |
| S07 | 76.25% | **79.38%** | 68.75% | 77.50% | val_acc |
| S08 | 91.88% | **95.00%** | 87.50% | **95.00%** | val_acc/soup |
| S09 | **99.38%** | 99.38% | 66.25% | 99.38% | combined/val_acc/soup |
| S10 | 63.12% | **64.38%** | 50.00% | 61.88% | val_acc |
| S11 | 87.50% | **89.38%** | 80.00% | **89.38%** | val_acc/soup |
| S12 | **87.50%** | 86.25% | 86.25% | 87.50% | combined/soup |
| S13 | 91.88% | **95.62%** | 88.12% | 93.12% | val_acc |
| S14 | **83.75%** | 83.12% | 77.50% | 83.12% | combined |
| S15 | **91.88%** | 91.88% | 76.88% | 90.62% | combined/val_acc |
| S16 | 73.12% | 70.00% | 52.50% | **74.38%** | soup |
| S17 | **86.25%** | 83.12% | 66.25% | 84.38% | combined |
| S18 | **91.25%** | 90.62% | 50.00% | 88.12% | combined |
| S19 | **98.75%** | 95.62% | 80.00% | **98.75%** | combined/soup |
| S20 | 60.62% | 61.25% | 50.00% | **64.38%** | soup |
| S21 | 73.75% | 70.00% | 57.50% | **75.62%** | soup |

### 3.3 训练行为差异

| 策略 | 平均 best_epoch | 平均 epochs_trained | 备注 |
|------|----------------|-------------------|------|
| combined | 17.3 | 32.9 | baseline |
| val_acc | 19.6 | 34.6 | 选择稍晚（val_acc 比 combined 更缓慢改善） |
| ema | 34.8 | 42.3 | 显著延迟（EMA 权重滞后于训练权重） |
| soup | 17.4 | 33.0 | 与 combined 几乎一致（训练完全相同） |

## 4. 分析

### 4.1 Soup: 微弱但一致的提升

Soup 策略相比 baseline 的逐被试差异（delta）：

- **改善的被试**: S01 (+0.6%), S03 (+1.3%), S07 (+1.3%), S08 (+3.1%), S11 (+1.9%), S13 (+1.3%), S16 (+1.3%), S20 (+3.8%), S21 (+1.9%)
- **退化的被试**: S02 (-0.6%), S05 (-1.9%), S06 (-0.6%), S10 (-1.3%), S14 (-0.6%), S15 (-1.3%), S17 (-1.9%), S18 (-3.1%)
- **不变**: S04, S09, S12, S19
- **Mean delta**: +0.24%

Soup 的优势在于提升了 worst-case（min 从 60.62% → 61.88%）和降低了标准差（10.73% → 10.46%），符合文献预期：checkpoint 权重平均倾向于移向 loss landscape basin 中心，产生更平滑、泛化更好的解。

不过 +0.24% 的提升幅度在统计上可能不显著（21 被试 paired t-test 难以达到 p<0.05）。

### 4.2 val_acc: 与 combined 持平

val_acc 策略的均值 (84.73%) 略低于 combined (84.85%)，差距仅 0.12%。这出乎意料 — 此前基于旧数据（8ch attention_pool, `20260218_2110`）的反事实分析预测 val_acc 应提升 +0.48%。

可能原因：
1. 反事实分析基于"在已有 milestone 中重新选择"，但 val_acc 策略改变了 milestone 保存本身（不同的 epoch 被保存为 milestone）
2. HPO 参数是为 combined 优化的，val_acc 使用这些参数可能不是最优
3. 128ch 与 8ch 的 majority_acc 稳定性不同 — 128ch 下 majority_acc 可能更可靠

### 4.3 EMA: 严重失败

EMA 策略均值仅 71.90%（-12.95%），是明确的失败。

**失败的被试分析**:

| Subject | EMA test | combined test | EMA best_epoch | EMA epochs | 诊断 |
|---------|---------|--------------|----------------|-----------|------|
| S10 | 50.00% | 63.12% | 1 | 17 | EMA 在 epoch 1 就"最好"，之后再也没改善 |
| S18 | 50.00% | 91.25% | 15 | 31 | 选在 epoch 15 但实际 combined=91%，EMA 严重滞后 |
| S20 | 50.00% | 60.62% | 1 | 17 | 同 S10 |
| S16 | 52.50% | 73.12% | 48 | 50 | 训到底但 EMA 仍很差 |

**根因**: `ema_decay=0.998` 意味着 EMA 权重的"半衰期"约 347 步。在每 epoch 仅更新一次 EMA 的实现下（而非每 batch），需要约 347 个 epoch 才能让 EMA 接近当前权重。训练总共只有 50 个 epoch，EMA 权重严重滞后。

**更深层的问题**: 当前实现在每个 epoch 结束时更新一次 EMA，而非每个 batch。对于 50 epoch 的短训练来说，`decay=0.998` 相当于 EMA 权重几乎不动。合理的 per-epoch decay 应该是 `0.5-0.8`（让 EMA 在 ~5-10 个 epoch 内收敛到当前权重），或者改为 per-batch 更新。

### 4.4 HPO 偏置的影响

本实验的关键局限：所有 4 个策略使用的是同一组 HPO 最优参数，而该 HPO 的目标函数使用的是 `combined_score`。

这种偏置可能影响结果的方式：

1. **对 val_acc 的影响**: HPO 可能找到了让 majority_acc 和 val_acc 同时改善的参数（如特定的 dropout/weight_decay 组合），这些参数在 val_acc-only 选择下未必最优。使用 val_acc 作为 HPO 目标可能会发现不同的最优参数区域。

2. **对 EMA 的影响**: HPO 优化了快速收敛的参数组合（combined 策略下 best_epoch 平均 17.3），而 EMA 需要更多 epoch 才能让 shadow weights 收敛。专门为 EMA 做 HPO 可能会发现偏向更长训练、更小 learning rate 的参数。

3. **对 soup 的影响**: Soup 在训练阶段与 combined 完全一致，仅后处理不同，因此 HPO 偏置对 soup 影响最小。这也解释了为什么 soup 是唯一一个表现略优于 baseline 的策略。

## 5. 结论

### 5.1 初步结论

1. **Soup (+0.24%)** 是唯一改善 baseline 的策略，且实现零额外训练成本（仅后处理）。改善集中在 worst-case 被试。
2. **val_acc (-0.12%)** 与 combined 基本持平，未能复现反事实分析预测的 +0.48% 提升。
3. **EMA (-12.95%)** 严重失败，根因是 per-epoch 更新 + decay=0.998 导致 EMA 权重在短训练中无法收敛。

### 5.2 不能下的结论

由于 HPO 偏置（所有策略使用为 combined 优化的参数），以下结论**尚不能做出**：

- "val_acc 不如 combined" — 可能只是参数不匹配
- "EMA 不适合 within-subject" — 可能只是 decay 参数和更新频率不对
- "soup 是最佳策略" — soup 的 +0.24% 可能在统计上不显著

### 5.3 后续实验建议

1. **分策略 HPO**: 对 val_acc 和 soup 策略各自运行独立的 HPO（将 `model_selection_strategy` 加入 HPO 配置），比较各策略在各自最优参数下的表现
2. **EMA 修复**: 两个改进方向：
   - 降低 decay 到 per-epoch 合理范围（0.5-0.8）
   - 或改为 per-batch 更新 + decay=0.999
3. **Soup 变体**: 尝试 top-5 和 greedy soup（只纳入让 val score 提升的 checkpoint）
4. **统计检验**: 对 soup vs combined 做 paired Wilcoxon signed-rank test 确认 +0.24% 是否显著

---

## 6. Cross-Subject 实验

### 6.1 实验设置

将模型选择策略扩展到 cross-subject 训练范式。与 within-subject 不同，cross-subject 使用全部 21 个被试的数据训练**一个全局模型**，然后逐被试测试。

- 训练参数：使用默认 cross-subject 配置（非 within-subject HPO 参数）
- 128ch, binary classification, CBraMod
- 每个策略独立 run，seed=42

### 6.2 HPO 偏置声明（扩展）

Cross-subject 和 transfer learning 实验使用的训练超参数是 cross-subject 管线的默认配置，**并非** within-subject HPO 优化的参数。这些默认参数没有针对任何特定模型选择策略优化，因此偏置程度可能低于 within-subject 实验。

### 6.3 结果总览

| 策略 | Mean | Median | Std | Min | Max | Best Epoch | Epochs | Run Tag |
|------|------|--------|-----|-----|-----|-----------|--------|---------|
| **combined** (baseline) | **90.54%** | 93.75% | **9.25%** | 65.62% | 100.00% | 26 | 46 (ES) | 20260321_0608 |
| val_acc | 90.42% | 93.12% | 9.58% | 64.38% | 100.00% | 30 | 50 (ES) | 20260321_0656 |
| soup | 89.94% | 93.75% | 10.01% | 62.50% | 100.00% | 25 | 45 (ES) | 20260321_0934 |
| ema | 84.40% | 87.50% | 11.55% | 60.62% | 100.00% | 100 | 100 (full) | 20260321_0750 |

> **数据来源**:
> - combined: `results/20260321_0608_cross-subject_cbramod_imagery_binary.json`
> - val_acc: `results/20260321_0656_cross-subject_cbramod_imagery_binary.json`
> - ema: `results/20260321_0750_cross-subject_cbramod_imagery_binary.json`
> - soup: `results/20260321_0934_cross-subject_cbramod_imagery_binary.json`

### 6.4 逐被试对比

| Subject | combined | val_acc | ema | soup | 最优策略 |
|---------|----------|---------|-----|------|---------|
| S01 | **93.12%** | 91.25% | 87.50% | 89.38% | combined |
| S02 | 95.00% | 95.00% | 91.88% | **95.62%** | soup |
| S03 | **100.00%** | 100.00% | 100.00% | 100.00% | all |
| S04 | **98.75%** | 97.50% | 67.50% | 96.88% | combined |
| S05 | 91.88% | **93.12%** | 63.75% | 91.25% | val_acc |
| S06 | 87.50% | **88.75%** | 78.12% | 85.62% | val_acc |
| S07 | **90.00%** | 88.12% | 85.62% | 89.38% | combined |
| S08 | 96.88% | **97.50%** | 96.88% | 96.88% | val_acc |
| S09 | 98.12% | **99.38%** | 96.88% | **99.38%** | val_acc/soup |
| S10 | **66.25%** | 65.00% | 60.62% | 62.50% | combined |
| S11 | **93.75%** | 93.75% | 92.50% | 93.75% | combined/val_acc/soup |
| S12 | 90.00% | **91.88%** | 86.88% | 91.25% | val_acc |
| S13 | 93.75% | 93.75% | 94.38% | **95.00%** | soup |
| S14 | 88.12% | 86.25% | **89.38%** | 85.62% | ema |
| S15 | 94.38% | **96.25%** | 87.50% | 95.00% | val_acc |
| S16 | **95.62%** | 95.00% | 80.62% | 95.00% | combined |
| S17 | **90.62%** | 89.38% | 88.75% | 90.00% | combined |
| S18 | 93.75% | 93.12% | 85.62% | **94.38%** | soup |
| S19 | **99.38%** | 99.38% | 97.50% | 99.38% | combined/val_acc/soup |
| S20 | **65.62%** | 64.38% | 65.00% | 64.38% | combined |
| S21 | 78.75% | **80.00%** | 75.62% | 78.12% | val_acc |

### 6.5 分析

**Combined 最优**: 在 cross-subject 范式下，combined 策略以 90.54% 领先，优于 val_acc (90.42%, -0.12%) 和 soup (89.94%, -0.60%)。这与 within-subject 的结论相反（within-subject 中 soup 最优）。

**EMA 再次失败**: 均值仅 84.40%（-6.14%），虽然比 within-subject (-12.95%) 好，但仍明显最差。cross-subject 训练 100 epochs（EMA 有更多 epoch 收敛），但 selection_score 仅从 0.58 缓慢爬升到 0.66，EMA 权重始终严重滞后于训练权重。

**训练行为差异**: EMA 跑满 100 epochs（selection_score 每 epoch 都在微小改善，永远不触发 early stopping），其余三个策略在 45-50 epochs 触发 early stopping。这表明 EMA 的 per-epoch 更新 + decay=0.998 不仅选模型差，还浪费了约 2 倍的训练时间。

## 7. Transfer Learning 实验

### 7.1 实验设置

**方案 A（隔离 finetune 效果）**: 所有 4 个 transfer runs 使用**同一个** pretrained checkpoint（combined 策略的 cross-subject 产出 `20260321_0608`），仅改变 finetuning 阶段的 model selection strategy。

- Pretrained model: `checkpoints/cross_subject/20260321_0608_cbramod_imagery_binary/best.pt`
- Freeze strategy: backbone
- Finetune epochs: 10 (default)
- 21 subjects, binary classification

### 7.2 结果总览

| 策略 | Mean | Median | Std | Min | Max | Run Tag |
|------|------|--------|-----|-----|-----|---------|
| ema | **90.36%** | 93.12% | **9.11%** | 65.62% | 99.38% | 20260321_1111 |
| combined (baseline) | 90.18% | 93.12% | 9.39% | 65.62% | 100.00% | 20260321_1025 |
| soup | 90.18% | 93.12% | 9.33% | 65.62% | 100.00% | 20260321_1122 |
| val_acc | 90.12% | 93.75% | 9.35% | 65.62% | 99.38% | 20260321_1101 |

> **数据来源**:
> - combined: `results/20260321_1025_transfer_comparison_cache_imagery_binary.json`
> - val_acc: `results/20260321_1101_transfer_comparison_cache_imagery_binary.json`
> - ema: `results/20260321_1111_transfer_comparison_cache_imagery_binary.json`
> - soup: `results/20260321_1122_transfer_comparison_cache_imagery_binary.json`

### 7.3 逐被试对比

| Subject | combined | val_acc | ema | soup | 最优策略 |
|---------|----------|---------|-----|------|---------|
| S01 | 93.12% | 93.12% | 93.12% | 93.12% | all |
| S02 | **98.12%** | 98.12% | 95.00% | 96.88% | combined/val_acc |
| S03 | **100.00%** | 98.12% | 98.12% | **100.00%** | combined/soup |
| S04 | 98.75% | **99.38%** | 98.75% | 98.75% | val_acc |
| S05 | 91.88% | 91.88% | 91.88% | 91.88% | all |
| S06 | 82.50% | 82.50% | **87.50%** | 83.12% | ema |
| S07 | 87.50% | 87.50% | **90.00%** | 87.50% | ema |
| S08 | **94.38%** | 94.38% | 93.12% | 92.50% | combined/val_acc |
| S09 | 97.50% | 97.50% | **98.12%** | **98.12%** | ema/soup |
| S10 | 66.25% | 66.25% | 66.25% | 66.25% | all |
| S11 | 93.75% | 93.75% | 93.75% | 93.75% | all |
| S12 | **91.88%** | 90.00% | 90.00% | **91.88%** | combined/soup |
| S13 | 93.75% | 93.75% | 93.75% | 93.75% | all |
| S14 | 88.12% | 88.12% | 88.12% | 88.12% | all |
| S15 | **95.62%** | 95.62% | 95.62% | 95.00% | combined/val_acc/ema |
| S16 | 91.88% | 93.75% | **95.62%** | 94.38% | ema |
| S17 | 91.25% | 91.25% | 91.25% | 91.25% | all |
| S18 | 93.75% | 93.75% | 93.75% | 93.75% | all |
| S19 | 99.38% | 99.38% | 99.38% | 99.38% | all |
| S20 | 65.62% | 65.62% | 65.62% | 65.62% | all |
| S21 | 78.75% | 78.75% | 78.75% | 78.75% | all |

### 7.4 分析

**策略差异极小**: 4 种策略的均值差距仅 0.24%（90.12%-90.36%），远小于 within-subject（13.19%）和 cross-subject（6.14%）的策略间差距。

**原因分析**: Transfer learning 仅训练 10 个 epoch（finetune 默认），且 backbone 被冻结。在如此短的训练中：
1. **多数被试不改善**: 21 个被试中有 10+ 个被试所有策略选择了 epoch 0（pretrained baseline）作为最佳，因为 finetune 未能超过预训练模型的 combined score
2. **策略差异被稀释**: 当 pretrained baseline 已经很强时，finetuning 的改善空间有限，model selection 策略的差异也随之缩小

**EMA 的反转**: 在 transfer learning 中 EMA 反而以 90.36% 微弱领先。在仅 10 epoch 的短训练中，EMA 的滞后问题反而变成了优势 — 它更保守地选择模型，避免了 finetune 过拟合导致的退化。不过 0.18% 的差异在统计上不显著。

## 8. 跨范式综合分析

### 8.1 总览

| 范式 | combined | val_acc | ema | soup | 最优策略 |
|------|----------|---------|-----|------|---------|
| Within-subject | 84.85% | 84.73% | 71.90% | **85.09%** | soup |
| Cross-subject | **90.54%** | 90.42% | 84.40% | 89.94% | combined |
| Transfer | 90.18% | 90.12% | **90.36%** | 90.18% | ema |

### 8.2 关键发现

1. **EMA 在长训练中系统性失败**: Within-subject（50 epochs, -12.95%）和 cross-subject（100 epochs, -6.14%）中 EMA 均严重落后。根因是 per-epoch 更新 + decay=0.998 导致 EMA 权重永远追不上训练权重。在短训练（transfer, 10 epochs）中反而微弱领先，但差异不显著。

2. **Combined 是最稳健的默认策略**: 在三个范式中排名分别为 #2、#1、#2，从未严重失败。val_acc 和 soup 的表现与 combined 差距在 0.6% 以内，均在噪声范围内。

3. **策略影响随训练长度衰减**: 策略间最大差距在 within-subject（13.19%）、cross-subject（6.14%）和 transfer（0.24%）中依次缩小，与训练 epoch 数 (50→100→10) 和模型选择机会数正相关。

4. **Soup 在 within-subject 中最优但在 cross-subject 中退化**: 这可能是因为 cross-subject 的验证集更大（全局验证），milestone 筛选更准确，checkpoint 平均带来的平滑效果反而引入了偏差。

### 8.3 HPO 偏置声明

- **Within-subject**: HPO 目标函数使用 `combined_score`，直接偏向 combined 策略。详见 §1.2。
- **Cross-subject 和 Transfer**: 使用 cross-subject 管线默认参数，非 HPO 优化产物，偏置程度更低。
- **结论性判断仍需分策略 HPO**，特别是 EMA 需要针对性地调整 decay 参数。
