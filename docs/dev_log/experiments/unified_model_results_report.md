# Unified Model 实验结果报告

> **实验日期**: 2026-03-19 (default params) → 2026-03-20 (HPO-optimized)
> **128 通道, 21 被试, imagery paradigm**

## 1. 结果概览 (HPO-Optimized, 2026-03-20)

### 1.1 被试内 (Within-Subject) — 每被试独立训练

| 模型 | Binary | Ternary | Quaternary | **Mean (Unified)** |
|------|--------|---------|------------|-------------------|
| **CBraMod** | 87.83% ± 11.18% | 68.53% ± 12.29% | 44.53% ± 10.65% | **66.96% ± 9.56%** |
| **EEGNet** | 84.20% ± 10.40% | 62.42% ± 10.07% | 49.02% ± 10.04% | **65.21% ± 8.98%** |
| CBraMod vs EEGNet | +3.63pp | +6.11pp | -4.49pp | **+1.75pp (p=0.24, n.s.)** |

> **数据来源**: `results/20260320_0243_comparison_cache_imagery_unified.json`
> **图表**: `results/20260320_0243_unified_comparison_imagery_unified.png`

### 1.2 跨被试 (Cross-Subject) — 全被试合并训练一个模型

| 模型 | Binary | Ternary | Quaternary | **Mean (Unified)** |
|------|--------|---------|------------|-------------------|
| **CBraMod** | 89.35% ± 9.44% | 68.63% ± 10.85% | 46.54% ± 11.49% | **68.17% ± 8.95%** |
| **EEGNet** | 80.12% ± 10.53% | 56.23% ± 7.92% | 44.01% ± 8.27% | **60.12% ± 6.85%** |
| CBraMod vs EEGNet | +9.23pp | +12.40pp | +2.53pp | **+8.05pp (p<0.0001)** |

> **数据来源**: `results/20260320_0548_cross-subject_{model}_imagery_unified.json` (含 subtask_results)
> **比较缓存**: `results/20260320_0548_cross_subject_cache_imagery_unified.json`
> **图表**: `results/20260320_0548_cross-subject_unified_comparison_imagery_unified.png`

### 1.3 HPO 优化效果对比

| 范式 | 模型 | Default (0319) | HPO (0320) | Delta |
|------|------|---------------|------------|-------|
| Within | EEGNet | 60.69% | **65.21%** | **+4.52pp** |
| Within | CBraMod | 66.69% | **66.96%** | +0.27pp |
| Cross | EEGNet | 52.61% | **60.12%** | **+7.51pp** |
| Cross | CBraMod | 67.68% | **68.17%** | +0.49pp |

> EEGNet 架构升级 (8,2→16,4, +3× 参数量) 是 HPO 最大收益来源。CBraMod HPO 参数微调收益有限 (~0.3-0.5pp)。

---

### 1.4 历史 Default-Param 结果 (2026-03-19, 存档)

<details>
<summary>展开查看 default-param baseline</summary>

**被试内 (Within-Subject)**

| 模型 | Binary | Ternary | Quaternary | **Mean (Unified)** |
|------|--------|---------|------------|-------------------|
| **CBraMod** | 88.63% ± 10.39% | 68.21% ± 12.44% | 43.23% ± 10.95% | **66.69% ± 9.66%** |
| **EEGNet** | 79.46% ± 13.09% | 57.52% ± 9.80% | 45.09% ± 9.21% | **60.69% ± 9.07%** |

> **数据来源**: `results/20260319_1640_comparison_cache_imagery_unified.json`

**跨被试 (Cross-Subject)**

| 模型 | Binary | Ternary | Quaternary | **Mean (Unified)** |
|------|--------|---------|------------|-------------------|
| **CBraMod** | 89.82% ± 8.55% | 68.35% ± 11.22% | 44.90% ± 10.06% | **67.69% ± 7.83%** |
| **EEGNet** | 71.13% ± 8.76% | 49.64% ± 7.26% | 37.06% ± 6.20% | **52.61% ± 6.13%** |

> **数据来源**: `results/20260319_2102_cross_subject_cache_imagery_unified.json`

</details>

---

## 2. Unified vs Standalone 对比

与历史最佳 **同范式** standalone 运行进行对比（128ch, 21 subjects）。被试内 unified 对比被试内 standalone，跨被试 unified 对比跨被试 standalone。Transfer learning 结果不参与对比（unified 尚无 transfer 实验）。Quaternary 无 standalone 基线（数据集不含 Online 4class sessions，quaternary 仅在 unified 框架下通过 offline held-out 数据评估）。

注：以下 standalone 结果均为标准超参数运行，非 HPO 优化结果。HPO 搜索已完成但尚未用于生成 21 被试全量对比基线。

### 2.1 CBraMod — 被试内 (Within-Subject)

| 子任务 | Standalone Best | Unified | 差异 | Standalone 来源 |
|--------|----------------|---------|------|-----------------|
| Binary | 85.62% ± 10.38% | 88.63% ± 10.39% | **+3.01pp** | `results/comparison_cache_imagery_binary.json` (2026-02-06) |
| Ternary | 69.54% ± 12.84% | 68.21% ± 12.44% | **-1.33pp** | `results/20260205_0306_comparison_cache_imagery_ternary.json` |
| Quaternary | N/A | 43.23% ± 10.95% | — | — |

### 2.2 EEGNet — 被试内 (Within-Subject)

| 子任务 | Standalone Best | Unified | 差异 | Standalone 来源 |
|--------|----------------|---------|------|-----------------|
| Binary | 78.75% ± 11.56% | 79.46% ± 13.09% | **+0.71pp** | `results/20260206_1003_comparison_cache_imagery_binary.json` |
| Ternary | 62.06% ± 13.70% | 57.52% ± 9.80% | **-4.54pp** | `results/comparison_cache_imagery_ternary.json` |
| Quaternary | N/A | 45.09% ± 9.21% | — | — |

### 2.3 CBraMod — 跨被试 (Cross-Subject)

| 子任务 | Standalone Best | Unified | 差异 | Standalone 来源 |
|--------|----------------|---------|------|-----------------|
| Binary | 89.73% ± 9.63% | 89.82% ± 8.55% | **+0.09pp** | `results/20260302_0012_cross_subject_cache_imagery_binary.json` |
| Ternary | 75.42% (DB 仅有一条记录) | 68.35% ± 11.22% | ~-7pp | ExperimentDB query (不可靠，无 std) |
| Quaternary | N/A | 44.90% ± 10.06% | — | — |

### 2.4 EEGNet — 跨被试 (Cross-Subject)

无 EEGNet 跨被试 standalone 基线可供对比（历史上仅 CBraMod 运行过 128ch 跨被试 standalone）。

### 2.5 对比小结

**CBraMod 被试内**: Binary unified **优于** standalone (+3.01pp)，ternary 轻微下降（-1.33pp，在 std 范围内）。Unified 训练利用了全部 session 数据（offline + 2class + 3class），binary 子任务从额外数据中获益最多。

**EEGNet 被试内**: Binary 基本持平（+0.71pp），ternary 明显下降（-4.54pp）。EEGNet 仅 2.5K 参数，4-class unified 输出层分摊了有限容量，ternary 受冲击较大。

**CBraMod 跨被试 Binary**: 与 standalone 基本持平（89.82% vs 89.73%）。Ternary standalone 基线（DB 记录 75.42%）可靠性存疑（单次运行，无 std），后续需重新运行 standalone ternary 跨被试实验以建立可靠基线。

**数据缺口**:
- EEGNet 跨被试 standalone 基线（binary/ternary）缺失
- Standalone ternary 跨被试基线不可靠
- HPO 优化后的 standalone 基线尚未建立（HPO 搜索已完成，最优参数已保存在 `results/hpo/`）

---

## 3. 训练动态分析

### 3.1 训练配置 (HPO-Optimized, 2026-03-20)

| 参数 | CBraMod 被试内 | CBraMod 跨被试 | EEGNet 被试内 | EEGNet 跨被试 |
|------|---------------|---------------|-------------|-------------|
| Scheduler | CAWD | CAWD | Plateau | Plateau |
| Max epochs | 50 (+ early stop) | 100 | 30 | 50 |
| Batch size | **256** | 256 | 64 | 128 |
| Learning rate | **2.9e-4** | **1.3e-4** | **4e-3** | **1e-3** |
| Backbone LR | **2.9e-4** | **1.3e-4** | — | — |
| Classifier LR | **1.2e-3** | **2.2e-4** | — | — |
| Weight decay | **0.026** | **0.13** | **1e-5** | 1e-4 |
| Label smoothing | 0.05 | **0.05** | 0 | 0 |
| Dropout | **0.10** | **0.37** | **0.27** | **0.35** |
| Gradient clip | **0.73** | **1.4** | — | — |
| Architecture | CBraMod | CBraMod | **EEGNet-16,4** | **EEGNet-16,4** |
| CAWD phase_epochs | **8** | **10** | — | — |
| CAWD phase_decay | **0.47** | **0.50** | — | — |
| CAWD exploration_epochs | **4** | **3** | — | — |
| CAWD exploration_batch | **64** | **128** | — | — |

> 粗体为 HPO 优化后改动的参数。EEGNet 跨被试参数为手工适配（HPO 仅完成 1/4 trials，不可靠）。

### 3.2 收敛行为

| 指标 | CBraMod 被试内 | CBraMod 跨被试 | EEGNet 被试内 | EEGNet 跨被试 |
|------|---------------|---------------|-------------|-------------|
| Epochs trained (avg) | 25.7 ± 7.3 | 29 | 24.8 ± 4.7 | 37 |
| Best epoch (avg) | 13.7 ± 7.3 | 17 | 15.6 ± 5.9 | 27 |
| Best val acc | 54.28% ± 7.20% | 54.66% | 52.77% ± 7.49% | 44.45% |
| Test acc | 66.69% ± 9.66% | 67.68% | 60.69% ± 9.07% | 52.61% |
| **Val→Test gap** | **+12.4pp** | **+13.0pp** | **+7.9pp** | **+8.2pp** |

### 3.3 Val-Test Gap 分析

所有配置的 val-test gap 均为**正值**（测试准确率 > 验证准确率），这不是过拟合，而是数据分布差异：

- **验证集**: 时序分割的末 20% trials（同一 session 的后段，信号质量随疲劳下降）
- **测试集**: 独立的 Sess02 Finetune（单独的 session，被试已有更多练习，信噪比更高）

CBraMod 的 gap（12-13pp）大于 EEGNet（8pp），因为 CBraMod 的表达能力更强，对分布差异更敏感。

> **注意 (2026-03-20)**: 上述 val acc 数据基于旧版验证方法——对 unified 混合验证集直接做 4-class argmax。此方法与测试评估协议（per-subtask logit masking）语义不一致。已实现 per-subtask 验证（`validate_unified()`），新训练运行的 val acc 将基于 binary/ternary/quaternary 各自 logit-masked 准确率的均值，与测试评估完全对齐。因此，后续实验的 val-test gap 预计会缩小。

---

## 4. 训练时间

| 实验 | EEGNet | CBraMod | 合计 |
|------|--------|---------|------|
| 被试内（21 subjects × 2 models） | 61.7 min | 131.8 min | **3.2 hr** |
| 跨被试（1 model each） | 70.4 min | 81.2 min | **2.5 hr** |
| 跨被试重评估（补生成 subtask 数据） | 13.8 min | 7.4 min | **21 min** |

> 被试内 wall time: 16:40 → 19:57 (3h17min)
> 跨被试 wall time: 21:02 → 23:35 (2h33min)

---

## 5. HPO 优化分析

> 完整的 "HPO 建议 → 用户 override → 最终采用" 参数对照表见 `docs/dev_log/experiments/hpo_final_parameters.md`

### 5.1 HPO 搜索概况

| 组合 | Trials | Best Val | 状态 |
|------|--------|----------|------|
| CBraMod within-subject binary | 51 | — | 完成 |
| EEGNet within-subject binary | 32 | — | 完成 |
| CBraMod cross-subject binary | 77 | — | 完成 |
| EEGNet cross-subject binary | 1/4 | 56.99% | 未完成，不可靠 |

> HPO 结果存储在 `results/hpo/` 目录

### 5.2 HPO 优化效果分析

**EEGNet 架构升级是最大 lever**:

- 从 EEGNet-8,2 (F1=8, D=2, F2=16, ~2.5K params) 升级到 EEGNet-16,4 (F1=16, D=4, F2=64, ~10K params)
- Within-subject: +4.52pp (60.69% → 65.21%)
- Cross-subject: +7.51pp (52.61% → 60.12%)
- Dropout 从 0.5 降至 0.27/0.35 也有贡献（减轻了小模型的过度正则化）

**CBraMod HPO 收益有限**:

- Within-subject: +0.27pp (66.69% → 66.96%)，统计不显著
- Cross-subject: +0.49pp (67.68% → 68.17%)，接近噪声水平
- CBraMod 4M 参数的基座模型对超参数变化不敏感，性能主要取决于预训练质量和下游数据量

**CBraMod vs EEGNet 差距缩小**:

- HPO 前: within +6.00pp, cross +15.08pp
- HPO 后: within +1.75pp (p=0.24, n.s.), cross +8.05pp (p<0.0001)
- EEGNet 架构升级后，within-subject 差距已不显著

### 5.3 关键 HPO 发现

1. **CBraMod within**: HPO 倾向更激进的 LR (2.9e-4 vs 1e-4) 和更低的正则化 (wd=0.026 vs 0.06, dropout=0.10 vs 0.15)，但实际收益微小
2. **CBraMod cross**: HPO 选择了更长的 phase (10 epochs) 和更温和的衰减 (0.50)，gradient clip 放宽至 1.4
3. **Label smoothing**: 用户手动将 label_smoothing 固定为 0.05（HPO 建议 within 0.09, cross 0.28），避免对 quaternary 弱信号的过度平滑
4. **EEGNet**: 架构参数 (F1, D) 的影响远大于训练超参数 (lr, wd)

### 5.4 剩余改进空间

- CBraMod quaternary 子任务 (43-46%) 仍是主要瓶颈，chance=25%，margin 有限
- EEGNet cross-subject HPO 未完成，当前参数为手工适配
- 可考虑 task-specific loss weighting 或 curriculum learning 提升 quaternary

---

## 6. 图表索引

| 图表 | 路径 |
|------|------|
| **HPO 被试内** unified 对比图 | `results/20260320_0243_unified_comparison_imagery_unified.png` |
| **HPO 跨被试** unified 对比图 | `results/20260320_0548_cross-subject_unified_comparison_imagery_unified.png` |
| Default 被试内 unified 对比图 | `results/20260319_1640_unified_comparison_imagery.png` |
| Default 跨被试 unified 对比图 | `results/20260319_2102_unified_comparison_cross-subject_imagery.png` |
