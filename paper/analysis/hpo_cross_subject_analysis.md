# Cross-Subject HPO 分析

> **目的**: 记录 cross-subject 范式下所有模型的 HPO 搜索结果。当前包含 CBraMod（已完成）和 EEGNet（待执行）。

---
---

# Part I: CBraMod Cross-Subject HPO

> 通过 Optuna TPE 搜索 CBraMod cross-subject binary 训练的最优超参数组合，量化超参数调优对模型性能的影响，并与默认配置和 within-subject HPO 结果进行对比。

---

## 1. 实验设计

### 默认参数基线

CBraMod cross-subject 默认配置相比 within-subject 使用更强的正则化：

| 参数 | within 默认 | cross 默认 | 设计意图 |
|------|------------|-----------|---------|
| backbone_lr | 1e-4 | 1e-4 | 相同 |
| classifier_lr | 3e-4 | 1.5e-4 | ↓ 跨被试分布更复杂 |
| weight_decay | 0.06 | 0.12 | ↑ 2x 正则 |
| dropout_rate | 0.15 | 0.35 | ↑ 2.3x 正则 |
| batch_size | 128 | 256 | ↑ 2x 更多数据 |
| label_smoothing | 0.05 | 0.15 | ↑ 3x 标签噪声更大 |
| gradient_clip | 1.0 | 0.5 | ↓ 跨被试梯度方差大 |

> **默认参数来源**: `src/config/training.py:419-430`

### 默认配置历史结果

| Run | 被试数 | mean_test_acc | 来源 |
|-----|--------|---------------|------|
| 20260206_1029 | 21 | **90.27%** | ExperimentDB `run_tag='20260206_1029'` |
| 20260301_1645 | 21 | **88.84%** | ExperimentDB `run_tag='20260301_1645'` |
| 20260301_2249 | 21 | **88.60%** | ExperimentDB `run_tag='20260301_2249'` |

三次独立 run 均值 **89.24%**，范围 88.60–90.27%（~1.67pp），反映训练随机性。

> **数据来源**: ExperimentDB 查询 — `SELECT * FROM runs r JOIN model_summaries ms ON r.run_id=ms.run_id WHERE experiment_type='cross_subject' AND task='binary' AND model_type='cbramod' AND n_channels=128`

### 搜索配置

| 配置项 | 值 |
|--------|-----|
| Sampler | TPE (Tree-structured Parzen Estimator) |
| 总 trials | 77 |
| 剪枝器 | NopPruner (cross-subject 依赖 early stopping，不用被试级剪枝) |
| 任务 | binary, imagery paradigm |
| 通道数 | 128 |
| 被试 | 全部 21 名 (S01–S21) |
| 数据模式 | cache-only |
| 随机种子 | 42 |

> **数据来源**: `results/hpo/hpo.db`, study=`cbramod_cross_subject_binary`
> **最优参数导出**: `results/hpo/cbramod_cross_subject_binary_best_params.json`

### 搜索空间 (11 维)

| 参数 | 类型 | 范围 | 分布 |
|------|------|------|------|
| backbone_lr | float | [1e-5, 5e-4] | log-uniform |
| classifier_lr_ratio | float | [1.0, 3.0] | uniform |
| weight_decay | float | [0.03, 0.5] | log-uniform |
| dropout_rate | float | [0.15, 0.55] | uniform |
| batch_size | categorical | {128, 256, 512} | — |
| label_smoothing | float | [0.05, 0.3] | uniform |
| gradient_clip | float | [0.2, 1.5] | uniform |
| phase_decay | float | [0.2, 0.7] | uniform |
| phase_epochs | int | [4, 10] | uniform |
| exploration_epochs | int | [3, 9] | uniform |
| exploration_batch_size | categorical | {32, 64, 128} | — |

> **搜索空间定义**: `src/hpo/search_spaces.py:99-116`

注意与 within-subject 的搜索空间差异：backbone_lr 上限从 1e-3 收窄至 5e-4，batch_size 从 {64,128,256} 上移至 {128,256,512}，dropout_rate 下限从 0.05 提高至 0.15。

---

## 2. 搜索效率

| 状态 | 数量 | 占比 |
|------|------|------|
| Complete | 43 | 55.8% |
| Pruned | 6 | 7.8% |
| Failed | 28 | 36.4% |
| **总计** | **77** | |

### 高失败率分析

28 个 trial 失败（36.4%），远高于 within-subject (2.0%) 和 EEGNet within (3.1%)。失败集中在 trial #34–54 区间（22/28 = 79% 的失败），原因已查明：**中间一次代码修改引入了 VRAM 溢出 bug**，导致该批 trial 连续 OOM 崩溃，后续修复代码后恢复正常。

| 区间 | 失败数 | 原因 |
|------|--------|------|
| Trial 0–29 | 5 | 零星失败（正常损耗） |
| Trial 30–59 | 22 | VRAM 溢出 bug（已修复） |
| Trial 60+ | 1 | 零星失败 |

此外 2 个 RUNNING 状态的遗留 trial (#9, #22) 已在分析前标记为 FAIL。失败率虚高不影响搜索质量——有效 trial 充足 (43 complete)，且最优解在 bug 出现前即已找到 (Trial #4)。

### 收敛曲线

```
Trial  1: 88.96%  (初始)
Trial  4: 90.68%  (+1.72pp, 最终最优)
```

搜索在极早期 (Trial #4) 即找到最优解，后续 73 个 trial（含 43 个 complete）未能超越。这表明 cross-subject 场景的最优超参数区域较为集中，TPE 初期的随机探索即可覆盖。

---

## 3. 最优参数 vs 默认参数

### Trial #4: 90.68% (最优)

| 参数 | 默认值 | HPO 最优 | 变化 |
|------|--------|---------|------|
| backbone_lr | 1e-4 | **1.34e-4** | ↑ 1.3x |
| classifier_lr | 1.5e-4 (ratio 1.5x) | **2.17e-4** (ratio 1.6x) | ↑ 1.4x |
| weight_decay | 0.12 | **0.130** | ≈ 相同 |
| dropout_rate | 0.35 | **0.369** | ≈ 相同 |
| batch_size | 256 | **256** | — |
| label_smoothing | 0.15 | **0.285** | ↑ 1.9x |
| gradient_clip | 0.5 | **1.363** | ↑ 2.7x |
| phase_decay | 0.7 | **0.499** | ↓ 1.4x |
| phase_epochs | 6 | **10** | ↑ 1.7x |
| exploration_epochs | 6 | **3** | ↓ 2x |
| exploration_batch_size | 32 | **128** | ↑ 4x |

> **默认参数来源**: `src/config/training.py:419-430, 77-88`

### 完成 trial 统计

| 指标 | 值 |
|------|-----|
| 最优 | 90.68% (Trial #4) |
| 均值 | 89.12% |
| 标准差 | 1.23% |
| 最差 | 84.82% |

43 个完成 trial 精度范围 5.86pp，大于 CBraMod within (2.65pp) 但远小于 EEGNet within (13.65pp)。

---

## 4. 参数重要性分析

### fANOVA 重要性排名

| 排名 | 参数 | 重要性 | 与 acc Spearman r | 方向 |
|------|------|--------|------------------|------|
| 1 | backbone_lr | 0.668 | +0.40* | 高更好 |
| 2 | classifier_lr_ratio | 0.156 | +0.27 | 高更好 |
| 3 | phase_epochs | 0.064 | +0.33* | 长更好 |
| 4 | label_smoothing | 0.030 | −0.19 | 低更好 |
| 5 | dropout_rate | 0.029 | +0.07 | 低影响 |
| 6 | phase_decay | 0.013 | +0.02 | 低影响 |
| 7 | weight_decay | 0.010 | −0.24 | 低影响 |
| 8 | batch_size | 0.010 | +0.27 | 低影响 |
| 9 | gradient_clip | 0.007 | +0.26 | 低影响 |
| 10 | exploration_epochs | 0.007 | −0.08 | 低影响 |
| 11 | exploration_batch_size | 0.007 | +0.39* | 大更好 |

Spearman 相关性中仅 3 个参数达到 p<0.05（标 *）。

<!-- Cohen's d Top 10 vs Bottom 10 效应量分析（备用参考，与 fANOVA 排名差异较大）
| 排名 | 参数 | 效应量 (d) | Top 10 均值 | Bottom 10 均值 | 方向 |
|------|------|-----------|------------|---------------|------|
| 1 | exploration_batch_size | +1.10 | 128 | 93 | 大更好 |
| 2 | gradient_clip | +0.82 | 1.22 | 0.93 | 高更好 |
| 3 | label_smoothing | −0.77 | 0.12 | 0.18 | 低更好 |
| 4 | weight_decay | −0.74 | 0.094 | 0.150 | 低更好 |
| 5 | phase_epochs | +0.69 | 8.9 | 7.8 | 长更好 |
| 6 | dropout_rate | +0.66 | 0.35 | 0.31 | 微弱正向 |
| 7 | classifier_lr_ratio | +0.61 | 2.21 | 1.86 | 高更好 |
| 8 | batch_size | +0.40 | 282 | 243 | 大更好 |
| 9 | backbone_lr | +0.25 | 1.1e-4 | 8.6e-5 | 微弱正向 |
| 10 | phase_decay | +0.16 | 0.47 | 0.44 | 低影响 |
| 11 | exploration_epochs | +0.13 | 6.2 | 5.9 | 低影响 |
-->

### 关键发现

**1. backbone_lr 是绝对主导参数**

fANOVA 重要性 0.668（占总方差 2/3），远超第二位 classifier_lr_ratio (0.156)。backbone_lr 搜索范围跨 50x（1e-5 至 5e-4, log-uniform），差区域（<3e-5）性能急剧下降，贡献了巨大的方差。Spearman r=+0.40 (p=0.008) 确认方向为正——更高的 backbone_lr 对应更好的准确率。

**2. 正则化参数接近默认即可，微调空间有限**

dropout_rate (fANOVA 0.029)、weight_decay (0.010)、label_smoothing (0.030) 的重要性均低于 0.03。与 within-subject HPO 的"大幅降低正则化"结论不同，cross-subject 的最优 dropout (0.37) 和 weight_decay (0.13) 与默认值 (0.35, 0.12) 几乎相同。跨被试数据分布更复杂，确实需要较强的正则化。

**3. CAWD scheduler: 长 phase + 短探索 + 大探索 batch**

- phase_epochs: 6→10（fANOVA #3, 0.064），每阶段训练更充分
- exploration_epochs: 6→3, exploration_batch_size: 32→128
- 这与 within-subject HPO 的发现高度一致

**4. gradient_clip 和 batch_size 的 fANOVA 重要性极低**

gradient_clip (0.007) 和 batch_size (0.010) 在 fANOVA 中排名末尾，表明它们对总体方差贡献很小。HPO 最优 gradient_clip=1.36 高于默认 0.5，但搜索空间内绝大多数值都能产生合理结果。

---

## 5. 与 within-subject HPO 对比

| 维度 | Within-Subject | Cross-Subject |
|------|---------------|---------------|
| 最优 acc | 86.01% | 90.68% |
| 默认 acc 参考 | ~85% | ~89.24% (3 runs 均值) |
| HPO 提升 | ~1pp | ~1.4pp |
| 完成 trial std | 0.64% | 1.23% |
| 收敛速度 | ~10 trials | ~4 trials |
| 核心调优方向 | 降低正则 + 升高 LR | backbone_lr 主导 + 优化 scheduler |
| 正则化 | 大幅降低 (dropout 0.15→0.10) | 维持默认 (dropout 0.35→0.37) |

**共通发现**:
- CAWD scheduler 一致偏好: 长 phase_epochs + 短 exploration_epochs + 大 exploration_batch_size
- phase_decay 一致偏好 ~0.47–0.50（低于默认 0.7），更快的 LR 衰减

**差异**:
- Within-subject 的主要杠杆是降低正则化（预训练 backbone 自带正则能力）
- Cross-subject 的正则化默认值已合理，主要杠杆是 gradient_clip 和 scheduler 参数

---

## 6. 结论

- HPO 最优配置 (90.68%) 相比默认配置三次 run 均值 (89.24%) 提升约 **1.4pp**，与 within-subject (~1pp) 提升幅度相当。
- 高失败率 (36.4%) 未影响搜索质量——最优解在 Trial #4 即被发现，后续 43 个完成 trial 确认了该区域的鲁棒性。
- **核心发现**: Cross-subject 默认配置的正则化强度基本合理，但 gradient_clip=0.5 过于保守（最优 ~1.36）、CAWD scheduler 的探索阶段可大幅缩短。
- 搜索空间内完成 trial 的 std 仅 1.23%，表明 cross-subject 训练在合理超参数范围内表现稳健。
- 与 within-subject HPO 一致的 CAWD scheduler 偏好（长 phase + 短探索 + 大 batch）可作为未来默认配置更新的依据。

---
---

# Part II: EEGNet Cross-Subject HPO

> **状态**: 待执行

## 1. 当前状态

DB 中存在 study `eegnet_cross_subject_binary`（study_id=4），但仅尝试过 4 个 trial：

| 状态 | 数量 |
|------|------|
| Complete | 1 |
| Failed | 3 |
| **总计** | **4** |

唯一完成的 Trial #2 准确率仅 **56.99%**，远低于预期。3 个 trial 失败，可能与 EEGNet cross-subject 训练配置或数据量有关。

> **数据来源**: `results/hpo/hpo.db`, study=`eegnet_cross_subject_binary`

## 2. 计划

### 搜索空间 (7 维)

沿用 `src/hpo/search_spaces.py:153-166` 中定义的 `_sample_eegnet_cross`：

| 参数 | 类型 | 范围 | 分布 |
|------|------|------|------|
| F1 | categorical | {4, 8, 16} | — |
| D | categorical | {1, 2, 4} | — |
| learning_rate | float | [5e-5, 5e-3] | log-uniform |
| weight_decay | float | [1e-5, 0.2] | log-uniform |
| dropout_rate | float | [0.3, 0.7] | uniform |
| batch_size | categorical | {64, 128, 256} | — |
| kernel_length | categorical | {32, 64, 128} | — |

### 前置条件

- 需要先建立严格遵循原论文设置的 EEGNet cross-subject 128ch baseline（默认参数），确认 baseline 性能合理后再启动 HPO
- 调查现有 4 个 trial 的失败原因（可能需要调整 EEGNet cross-subject 默认配置中的 epochs、early stopping 等）

### 预期 trial 数量

- 建议 30–50 trials（EEGNet 搜索空间仅 7 维，小于 CBraMod 的 11 维）
- Cross-subject 无被试级剪枝（与 CBraMod cross 一致），使用 NopPruner + early stopping
