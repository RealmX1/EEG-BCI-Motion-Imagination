# 模型选择策略实验计划

## Context

当前模型选择标准 `combined_score = (val_acc + majority_acc) / 2.0` (trainer.py:811) 存在问题：
- **50% 被试**选了比 peak test accuracy 更差的 checkpoint
- majority_acc 波动性是 val_acc 的 1.2-2.6 倍（离散跳变）
- combined_score 门控 milestone 保存 → 10/21 被试有 val_acc 新高的 epoch 未被保存
- 整体损失 -1.34% vs oracle；反事实分析显示 val_acc alone 可 +0.48%

> **数据来源**: `results/20260218_2110_cbramod_within_subject/binary/S{01-21}/results.json`

文献调研确认：segment-level metric 做 model selection 是 EEG-BCI 标准做法（Braindecode 框架）；EMA/checkpoint soup 在预训练模型微调中各有 0.3-1.5% 提升。

### 文献参考

- [Dodge et al. (2020) - Fine-Tuning Pretrained Language Models](https://arxiv.org/abs/2002.06305) — 小数据集微调 early stopping 最佳实践
- [Zhang et al. (2021) - Revisiting Few-sample BERT Fine-tuning (ICLR)](https://arxiv.org/abs/2006.05987) — 常见微调 recipe 的 undertraining 问题
- [Wortsman et al. (2022) - Model Soups (ICML)](https://arxiv.org/abs/2203.05482) — checkpoint 权重平均
- [Demir et al. (2024) - Adaptive SWA](https://arxiv.org/abs/2406.19092) — 仅在 val 改善时更新权重平均
- [EMA of Weights in Deep Learning (2024)](https://arxiv.org/abs/2411.18704) — EMA 动态与收益分析
- [Huang et al. (2017) - Snapshot Ensembles](https://arxiv.org/pdf/1704.00109) — 周期性 LR + snapshot
- [Schirrmeister et al. (2017) - Deep Learning with CNNs for EEG Decoding (Braindecode)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5655781/) — segment-level loss 做 early stopping 标准

## 问题分析摘要

### Majority voting 不适合做 model selection

| 指标 | epoch 间波动 (mean) | unique 值数 | 信号特征 |
|------|-------------------|------------|---------|
| val_acc (segment-level) | 0.006-0.023 | 5-29 | 连续、平滑 |
| majority_acc (trial-level) | 0.010-0.039 | 2-21 | 离散、跳变 |
| 波动比 (majority / val_acc) | **1.2-2.6x** | — | — |

原因：~156 val trials 的 hard majority voting，单个 trial 翻转 → 整体准确率跳 ~0.6%。

### 各选择标准的反事实对比

> 基于 milestone test accuracy 的反事实分析（仅能在已保存的 milestone 间选择）

| 标准 | Mean Test Acc | vs Current | vs Oracle |
|------|-------------|-----------|-----------|
| Oracle (不可实现) | 58.33% | +1.34% | — |
| **val_acc only** | **57.47%** | **+0.48%** | -0.86% |
| weighted 0.9/0.1 | 57.08% | +0.09% | -1.25% |
| combined (current) | 56.99% | baseline | -1.34% |
| majority_acc only | 56.70% | -0.30% | -1.64% |
| val_loss only | 55.92% | -1.07% | -2.41% |

### Patience 分析结论

降低 patience 无法改善问题（P=10 到 P=2 均更差）。问题根源在选择标准，不在停止时机。

## 方案概述

引入可配置的 `model_selection_strategy` 参数，实现 3 种新策略与 baseline 对比：

| 策略 | 含义 | 预期收益 | 实现复杂度 |
|------|------|---------|-----------|
| `combined` (default) | 现行行为，不变 | baseline | — |
| `val_acc` | 仅 segment-level val_acc | +0.48% (已验证) | ~15 行 |
| `ema` | EMA shadow weights + val_acc 选择 | +0.3-1.5% (文献) | ~60 行 |
| `soup` | 训练后 top-K milestone 权重平均 | +0.5-0.7% (文献) | ~40 行 |

## 实现分 4 个 Phase

### Phase 1: 基础架构 + val_acc 策略

**目标**: 引入策略框架 + 实现最简单的 val_acc 策略，可立即验证

#### 1.1 `src/training/trainer.py` — 核心改动

**`__init__`** (line 132-150): 新增 2 个参数
```python
model_selection_strategy: str = 'combined',
ema_decay: float = 0.998,
```
存储为 `self.model_selection_strategy`，`self.ema_decay`。
新增 `self.best_selection_score = 0.0`。
验证策略值在 `('combined', 'val_acc', 'ema', 'soup')` 中。

**`train()` 中的 selection_score 逻辑** (line 811 附近):
```python
# 保留 combined_score 用于 history 记录
combined_score = (val_acc + majority_acc) / 2.0

# 根据策略计算 selection_score
if self.model_selection_strategy == 'val_acc':
    selection_score = val_acc
else:  # 'combined', 'ema', 'soup' 训练时都用 combined
    selection_score = combined_score
```

**best model 判定** (line 851-859): `combined_score >` → `selection_score >`
- 更新 `self.best_selection_score`（新增）
- **保留** `self.best_combined_score` 赋值（向后兼容）
- milestone 字典新增 `selection_score` 字段

**ReduceLROnPlateau** (line 849): `self.scheduler.step(selection_score)`

**history**: 新增 `val_selection_score` 列表

**checkpoint_dict** (line 863-870): 新增 `selection_strategy`, `selection_score`

#### 1.2 `src/training/train_within_subject.py`

**`train_single_subject()`** (line 589-632):
- 从 `train_config` 读取 `model_selection_strategy` 和 `ema_decay`
- 传递给 `WithinSubjectTrainer()` 构造函数 (line 617-632)
- results dict 新增 `model_selection_strategy`, `best_selection_score`

#### 1.3 YAML 配置

创建 `configs/model_selection_val_acc.yaml`:
```yaml
training:
  model_selection_strategy: val_acc
```

#### 1.4 验证
```bash
# 单被试快速验证（默认行为不变）
uv run python scripts/run_within_subject_comparison.py --models cbramod --subjects S01

# val_acc 策略
uv run python scripts/run_within_subject_comparison.py --models cbramod --config configs/model_selection_val_acc.yaml
```

---

### Phase 2: EMA 策略

**目标**: 训练时维护 EMA shadow weights，用 EMA 模型做 validation

#### 2.1 `src/training/trainer.py` — EMA 相关新增

**新增属性**: `self.ema_state = None`

**新增方法 `_update_ema()`**:
- 首次调用时初始化 EMA state 为当前 model weights 的 clone
- 后续: `ema[k] = decay * ema[k] + (1-decay) * model[k]`

**新增上下文管理器 `_ema_context()`**:
- 临时用 EMA weights 替换 model weights（validation 时）
- yield 后恢复原始 weights

**`train()` 改动**:
- 每个 epoch 的 `train_epoch()` 后调用 `_update_ema()`
- validate 和 majority_vote 在 `_ema_context()` 内执行
- EMA 策略下 `selection_score = val_acc`（EMA 模型的 val_acc）
- `best_state` 保存 EMA weights（`self.ema_state` 的 clone）

**resume checkpoint**: 保存/恢复 `ema_state`

#### 2.2 YAML 配置

创建 `configs/model_selection_ema.yaml`:
```yaml
training:
  model_selection_strategy: ema
  ema_decay: 0.998
```

#### 2.3 验证
- 确认 EMA 权重与原始权重的 L2 距离随训练递增
- 确认 best.pt 中保存的是 EMA 权重
- 21 被试 full run 对比

---

### Phase 3: Soup 策略

**目标**: 训练后对 top-K milestone checkpoint 权重取平均

#### 3.1 `src/training/trainer.py` — Soup 后处理

**新增方法 `_make_checkpoint_soup(milestones, top_k=3)`**:
- 按 `selection_score` 降序取 top-K milestone
- 加载 state_dict，对所有参数做 `torch.stack().mean()`
- 返回平均后的 state_dict

**新增方法 `_has_batchnorm()`**: 检查模型是否含 BatchNorm 层

**新增方法 `_update_bn_stats(dataloader)`**: 重置 BN 统计后前向传播 50 batch

**`train()` 末尾** (约 line 968 后):
- 如果 `strategy == 'soup'` 且 milestones >= 2:
  - 调用 `_make_checkpoint_soup()`
  - 如果有 BatchNorm，调用 `_update_bn_stats()`
  - 更新 `self.best_state` 和 `best.pt`

#### 3.2 YAML 配置

创建 `configs/model_selection_soup.yaml`:
```yaml
training:
  model_selection_strategy: soup
```

#### 3.3 验证
- 确认 soup 权重 = top-K 权重算术平均
- EEGNet 下确认 BN 统计已更新
- CBraMod 下确认跳过 BN 更新

---

### Phase 4: Cross-paradigm 兼容

#### 4.1 `src/training/train_cross_subject.py`
- 传递 `model_selection_strategy` 和 `ema_decay` 给 trainer

#### 4.2 `src/training/finetune.py`
- 传递策略参数
- baseline 初始化同步更新 `trainer.best_selection_score`

## 关键文件

| 文件 | 变更类型 |
|------|---------|
| `src/training/trainer.py` | 核心: selection_score, EMA, soup |
| `src/training/train_within_subject.py` | 配置读取与传递 |
| `src/training/train_cross_subject.py` | 配置传递 (Phase 4) |
| `src/training/finetune.py` | 配置传递 + baseline (Phase 4) |
| `configs/model_selection_val_acc.yaml` | 新建 |
| `configs/model_selection_ema.yaml` | 新建 |
| `configs/model_selection_soup.yaml` | 新建 |

## 实验运行

```bash
# Baseline (不带 config)
uv run python scripts/run_within_subject_comparison.py --models cbramod

# 各策略
uv run python scripts/run_within_subject_comparison.py --models cbramod --config configs/model_selection_val_acc.yaml
uv run python scripts/run_within_subject_comparison.py --models cbramod --config configs/model_selection_ema.yaml
uv run python scripts/run_within_subject_comparison.py --models cbramod --config configs/model_selection_soup.yaml
```

## 验证清单

- [ ] 默认行为 (无 config override) 数值与改动前一致
- [ ] val_acc 策略: 选择的 best_epoch 与 combined 不同
- [ ] EMA: checkpoint 保存 EMA 权重，resume 恢复 EMA state
- [ ] Soup: 权重是 top-K 的算术平均，EEGNet 更新 BN
- [ ] HPO (`src/hpo/`) 不传 strategy 时走默认 combined
- [ ] Cross-subject 和 transfer 无报错

## 注意事项

- CBraMod = LayerNorm → EMA/soup 无需 BN 更新
- EEGNet = BatchNorm → soup 需要 `_update_bn_stats()`
- EMA 额外内存 ~16MB (4M params × 4 bytes)，可忽略
- Soup 需要 milestone 文件存在于磁盘（当前已保存）
- `finetune.py` 的 baseline 初始化需同步更新 `best_selection_score`
