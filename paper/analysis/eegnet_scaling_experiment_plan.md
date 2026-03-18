# EEGNet 模型容量扩展实验计划

> **目的**: HPO 搜索发现 EEGNet 架构参数 (F1, D) 已触及搜索空间上限，最优 trial 选择了最大可用配置。本实验进一步扩展模型容量，评估更大 EEGNet 在 binary 和 ternary 任务上的性能变化，验证是否存在容量瓶颈或过拟合转折点。

---

## 1. 背景与动机

### HPO 发现

Within-subject binary HPO (19 trials, 6 complete) 的最佳 trial (#3) 选择了搜索空间中最大的架构配置：

| 参数 | 搜索空间 | HPO 最优 | 位置 |
|------|---------|---------|------|
| F1 | {4, 8, 16} | **16** | 上限 |
| D | {1, 2, 4} | **4** | 上限 |
| F2 (=F1×D) | 派生 | **64** | 上限 |
| kernel_length | {32, 64, 128} | **64** | 中间 |

> **数据来源**: `results/hpo/hpo.db`, study=`eegnet_within_subject_binary`, trial #3, value=0.8156

第二名 trial (#11, value=0.8152) 同样选择 D=4，F1=8，进一步确认了 TPE 向更大模型搜索的趋势。

### 默认 vs HPO 最优

| 配置 | F1 | D | F2 | 参数量 | Within-Subject Binary Acc |
|------|-----|---|-----|-------|--------------------------|
| EEGNet-8,2 (默认) | 8 | 2 | 16 | 3,538 | 78.75% |
| EEGNet-16,4 (HPO 最优) | 16 | 4 | 64 | 16,162 | 81.56% |

> **默认配置来源**: `results/20260206_1003_comparison_cache_imagery_binary.json` (21 subjects)
> **HPO 最优来源**: `results/hpo/eegnet_within_subject_binary_best_params.json`

参数量增加 4.6 倍，准确率提升 +2.81pp。模型容量与性能的正相关尚未见饱和迹象。

### 核心问题

1. **容量瓶颈**: 继续扩大 EEGNet 是否还能提升性能？
2. **过拟合转折点**: 在被试内有限数据量下，多大的模型开始过拟合？
3. **跨任务泛化**: 容量扩展对更难的 ternary 任务影响是否一致？
4. **跨被试鲁棒性**: 更大模型在 cross-subject 场景下是否仍然有效，还是会过度拟合个体差异？

---

## 2. 实验设计

### 2.1 模型配置

HPO 搜索空间的 F1 和 D 均采用 2× 倍增序列（F1: {4, 8, **16**}, D: {1, 2, **4**}）。沿用同一递增模式，各扩展一步至 F1=32, D=8。采用 2×2 factorial design 隔离 F1 和 D 各自的贡献：

| 配置名称 | F1 | D | F2 (=F1×D) | kernel_length | 参数量 | vs 默认倍数 |
|----------|-----|---|------------|--------------|--------|------------|
| EEGNet-8,2 (默认) | 8 | 2 | 16 | 64 | 3,538 | 1.0× |
| EEGNet-16,4 (HPO 最优) | 16 | 4 | 64 | 64 | 16,162 | 4.6× |
| EEGNet-32,4 (expand F1) | **32** | 4 | 128 | 64 | 40,514 | 11.5× |
| EEGNet-16,8 (expand D) | 16 | **8** | 128 | 64 | 39,458 | 11.2× |
| EEGNet-32,8 (expand both) | **32** | **8** | 256 | 64 | 111,682 | 31.6× |

扩展逻辑：

```
HPO 搜索空间          扩展后
F1: {4, 8, 16}    →  {4, 8, 16, 32}    (+1 step)
D:  {1, 2, 4}     →  {1, 2, 4, 8}      (+1 step)
```

EEGNet-32,4 与 EEGNet-16,8 的 F2 相同 (128)、参数量接近 (~40K)，但前者更宽 (更多 temporal filters)，后者更深 (每个 temporal filter 更多 spatial filters)。对比两者可以揭示 **宽度 vs 深度** 对 EEG 解码的相对贡献。

注: 所有配置的参数量远小于 CBraMod (~4M)，即使最大的 EEGNet-32,8 也仅为 CBraMod 的 2.8%。

### 2.2 固定训练超参数

来自 HPO 最优 trial #3 的训练配置：

| 参数 | 值 |
|------|-----|
| learning_rate | 1.90e-4 |
| weight_decay | 1.47e-4 |
| dropout_rate | 0.383 |
| batch_size | 64 |
| kernel_length | 64 |
| scheduler | plateau (默认) |
| epochs | 30 (默认) |

> **来源**: `results/hpo/eegnet_within_subject_binary_best_params.json`

### 2.3 实验矩阵

| # | 模型配置 | 范式 | 任务 | 通道数 | 被试 |
|---|---------|------|------|--------|------|
| 1 | EEGNet-16,4 (HPO 最优) | within-subject | binary | 128 | S01–S21 |
| 2 | EEGNet-16,4 (HPO 最优) | within-subject | ternary | 128 | S01–S21 |
| 3 | EEGNet-32,4 (expand F1) | within-subject | binary | 128 | S01–S21 |
| 4 | EEGNet-32,4 (expand F1) | within-subject | ternary | 128 | S01–S21 |
| 5 | EEGNet-16,8 (expand D) | within-subject | binary | 128 | S01–S21 |
| 6 | EEGNet-16,8 (expand D) | within-subject | ternary | 128 | S01–S21 |
| 7 | EEGNet-32,8 (expand both) | within-subject | binary | 128 | S01–S21 |
| 8 | EEGNet-32,8 (expand both) | within-subject | ternary | 128 | S01–S21 |
| 9 | EEGNet-16,4 (HPO 最优) | cross-subject | binary | 128 | S01–S21 |
| 10 | EEGNet-16,4 (HPO 最优) | cross-subject | ternary | 128 | S01–S21 |
| 11 | EEGNet-32,4 (expand F1) | cross-subject | binary | 128 | S01–S21 |
| 12 | EEGNet-32,4 (expand F1) | cross-subject | ternary | 128 | S01–S21 |
| 13 | EEGNet-16,8 (expand D) | cross-subject | binary | 128 | S01–S21 |
| 14 | EEGNet-16,8 (expand D) | cross-subject | ternary | 128 | S01–S21 |
| 15 | EEGNet-32,8 (expand both) | cross-subject | binary | 128 | S01–S21 |
| 16 | EEGNet-32,8 (expand both) | cross-subject | ternary | 128 | S01–S21 |

共 16 组实验 = 4 模型配置 × 2 范式 × 2 任务。

### 2.4 Baseline 对比基准

| 范式 | 任务 | 默认 EEGNet Acc | 来源 |
|------|------|----------------|------|
| within-subject | binary | 78.75% (±11.56%) | `results/20260206_1003_comparison_cache_imagery_binary.json` (21 subjects) |
| within-subject | ternary | 66.5% (±13.1%) | `results/experiments.db`, run `20260114_1939_within_subject_imagery_ternary` (7 subjects, 需补充全量) |
| cross-subject | binary | 71.85% (±9.75%) | `results/experiments.db`, run `20260227_0049_cross_subject_61ch_standard_1010_imagery_binary` (61ch, 需 128ch 基准) |
| cross-subject | ternary | 44.88% (±9.67%) | `results/experiments.db`, run `20260226_2042_cross_subject_32ch_commercial_imagery_ternary` (32ch, 需 128ch 基准) |

**注意**: within-subject ternary 缺乏 21 被试完整运行，cross-subject binary/ternary 缺乏 128ch 基准。本实验将同时补齐这些 baseline（实验 #1 和 #2 使用 HPO 超参数而非默认超参数；如需严格对比默认配置，需额外运行默认超参数 + 128ch 的 cross-subject 和 full within-subject ternary）。

---

## 3. 预期分析

### 3.1 主要指标

- **测试准确率** (mean ± std across subjects): 主要性能指标
- **训练-验证 gap**: 过拟合程度指标
- **参数量 vs 准确率曲线**: 识别收益递减/过拟合转折点

### 3.2 关注的模式

| 预期模式 | 含义 |
|---------|------|
| Within-subject: 准确率随容量单调递增 | 被试内数据未被充分利用，EEGNet 存在容量瓶颈 |
| Within-subject: 准确率先升后降 | 过拟合转折点，HPO 附近即为最优容量 |
| Cross-subject: 大模型性能下降 | 过拟合个体噪声，泛化需要正则化而非容量 |
| Ternary > Binary 的提升幅度 | 更难任务更需要模型容量 |

### 3.3 Scaling law 分析

5 个数据点 (default, HPO, expand F1, expand D, expand both) 可以拟合 log-linear scaling curve:

```
acc = a * log(params) + b
```

观察 EEGNet 在当前数据规模下的 scaling behavior。

### 3.4 宽度 vs 深度分析

EEGNet-32,4 (expand F1) 与 EEGNet-16,8 (expand D) 参数量接近 (~40K)、F2 相同 (128)，但架构不同：

| 维度 | EEGNet-32,4 | EEGNet-16,8 | 含义 |
|------|------------|------------|------|
| Temporal filters (F1) | 32 | 16 | 频率特征分辨率 |
| Spatial filters per F1 (D) | 4 | 8 | 空间模式多样性 |
| 总 spatial maps (F1×D) | 128 | 128 | 相同 |

对比结果将揭示 EEG 解码中 temporal (频率) 分辨率与 spatial (通道间) 分辨率的相对重要性。

---

## 4. 实现方案

### 4.1 配置传递

通过现有的 `config_overrides` 机制传入自定义 F1/D/F2，无需修改训练管线代码：

```python
config_overrides = {
    'model': {
        'F1': 32,
        'D': 4,
        'F2': 128,
        'kernel_length': 64,
        'dropout_rate': 0.383,
    },
    'training': {
        'learning_rate': 1.90e-4,
        'weight_decay': 1.47e-4,
        'batch_size': 64,
    },
}
```

### 4.2 执行命令示例

```bash
# Within-subject binary, EEGNet-32,4
uv run python scripts/run_within_subject_comparison.py \
    --task binary --models eegnet --cache-only \
    --config-overrides '{"model": {"F1": 32, "D": 4, "F2": 128, "dropout_rate": 0.383}, "training": {"learning_rate": 1.9e-4, "weight_decay": 1.47e-4, "batch_size": 64}}'

# Cross-subject binary, EEGNet-32,4
uv run python scripts/run_cross_subject_comparison.py \
    --task binary --models eegnet --cache-only \
    --config-overrides '{"model": {"F1": 32, "D": 4, "F2": 128, "dropout_rate": 0.383}, "training": {"learning_rate": 1.9e-4, "weight_decay": 1.47e-4, "batch_size": 64}}'
```

*注: 实际命令需确认 `--config-overrides` 参数在现有脚本中是否已支持，可能需要添加。*

---

## 5. 风险与注意事项

1. **显存**: EEGNet-32,8 虽然只有 112K 参数，但中间激活的 feature map 更大 (256 channels)。128ch 输入下预计仍在 GPU 显存内。
2. **训练超参数迁移**: HPO 最优超参数是针对 EEGNet-16,4 搜索的，直接用于更大模型可能不是最优。如果大模型表现不佳，需考虑是超参数不匹配还是真正的过拟合。
3. **Cross-subject 学习率**: HPO 搜索仅覆盖 within-subject，cross-subject 可能需要不同的超参数（如更低学习率、更大 batch）。建议先用 HPO within-subject 参数运行，如果结果异常再调整。
4. **Ternary baseline 不完整**: 需要确保 ternary 实验有可比的 21 被试 baseline。
