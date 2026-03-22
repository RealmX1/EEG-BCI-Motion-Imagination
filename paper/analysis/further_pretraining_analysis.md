# CBraMod Further Pre-training 实验分析

## 1. 实验动机

CBraMod 原始权重在 TUEG (Temple University EEG) 临床数据上预训练，包含休息态、病理等多种 EEG 模式。我们的下游任务是单指运动想象（Motor Imagery, MI）分类——一种与临床 EEG 差异显著的范式。本实验评估：**在外部 MI 数据集上进行 domain-adaptive further pre-training 能否改善下游 finger MI 分类性能？**

## 2. 数据规模对比

| 数据集 | 时长 | 被试数 | 用途 |
|--------|------|--------|------|
| TUEG (原始预训练) | ~15,000+ 小时 | 数千 | CBraMod 自监督预训练 |
| 外部 MI 数据集 (10 个) | ~252 小时 | ~300 | Domain-adaptive further pre-training |
| 自有 finger MI 数据 | ~64 小时 | 21 | 下游评估 |

外部 MI 数据量仅为原始预训练数据的 **~1/60**，自有 finger MI 数据更是仅有 **~1/234**。

## 3. Further Pre-training 数据

### 3.1 数据集构成

使用 10 个 MOABB Motor Imagery 公开数据集，预处理为 CBraMod 输入格式 `(n_channels, 30, 200)` — 即 30 个 1 秒 patch，200 Hz 采样率。

| 数据集 | Segments | 通道数 | 被试数 | 采样权重 | 完整性 |
|--------|:---:|:---:|:---:|:---:|:---:|
| BNCI2014_001 | 1,296 | 22 | 9/9 | 4.3% | 完整 |
| BNCI2015_004 | 1,634 | 30 | 10/10 | 5.4% | 完整 |
| Cho2017 | 2,416 | 64 | 52/52 | 8.0% | 完整 |
| GrosseWentrup2009 | 910 | 128 | 10/10 | 3.0% | 完整 |
| Lee2019_MI | 3,264 | 62 | 54/54 | 10.8% | 完整 |
| Ofner2017 | 1,363 | 61 | 15/15 | 4.5% | 完整 |
| PhysionetMI | 1,000 | 64 | 109/109 | 3.3% | 完整 |
| **Schirrmeister2017** | **927** | 128 | **5/14** | 3.1% | **不完整** |
| Shin2017A | 1,513 | 30 | 29/29 | 5.0% | 完整 |
| **Stieger2021** | **15,959** | 60 | **14/62** | **52.7%** | **不完整** |
| **总计** | **30,282** | — | — | — | — |

### 3.2 数据完整性问题

**Schirrmeister2017**: 仅下载了 5/14 个被试（s1, s2, s6, s12, s13）的原始 EDF 数据，托管在 gin.g-node.org，下载速度极慢（20-300 kB/s）。预计补全后可增加 ~1,000 segments。

**Stieger2021**: 仅下载了 14/62 个被试的原始数据（同样受限于下载速度），但已贡献 15,959 segments，占总数据的 52.7%。补全后预计可增加 ~55,000 segments。

数据采样权重不均衡：Stieger2021 独占 52.7% 的采样概率，可能导致模型过度适配该数据集的分布特征。

### 3.3 数据预处理管线

1. **单位归一化**: 通过 `dataset_metadata.json` 为每个数据集指定权威的 `to_uV_factor`，解决 MOABB 加载器单位不一致问题
   - 特别修正：Cho2017 的 MOABB 加载器错误假设 .mat 数据为 µV（实际为 nV），需额外 ÷1000
2. **重采样**: 统一到 200 Hz
3. **分段**: 30 秒连续段，shape = `(n_channels, 30, 200)`
4. **伪影剔除**: 跳过 run 前 5 秒 + 平均绝对幅值 > 500 µV 的段
5. **存储**: LMDB 格式，pickle 序列化

## 4. 训练配置

| 参数 | 值 |
|------|-----|
| 基础权重 | CBraMod TUEG 预训练权重 (4.92M 参数) |
| 自监督任务 | Masked autoencoding (50% mask ratio, MSE loss) |
| Epochs | 10 |
| Effective batch size | 128 (batch=8 × grad_accum=16) |
| 优化器 | AdamW (lr=5e-5, weight_decay=0.05) |
| LR schedule | Warmup 2 epochs → Cosine decay → 1e-6 |
| AMP | FP16 |
| 总步数 | 2,360 (236 steps/epoch × 10 epochs) |
| 训练时间 | ~48 分钟 |

### 4.1 训练曲线

| Epoch | Loss | LR | Time (s) |
|:---:|:---:|:---:|:---:|
| 1 | 0.027363 | 2.50e-05 | 395 |
| 2 | 0.012323 | 5.00e-05 | 282 |
| 3 | 0.009646 | 4.81e-05 | 276 |
| 4 | 0.008285 | 4.28e-05 | 276 |
| 5 | 0.007073 | 3.49e-05 | 277 |
| 6 | 0.006720 | 2.55e-05 | 275 |
| 7 | 0.006383 | 1.61e-05 | 275 |
| 8 | 0.006261 | 8.18e-06 | 277 |
| **9** | **0.006055** | 2.86e-06 | 274 |
| 10 | 0.006146 | 1.00e-06 | 275 |

Loss 从 0.027 降至 0.006 (4.5x)。Epoch 9 为最佳（loss=0.006055），Epoch 10 微升。Loss 从 Epoch 6 开始趋于平坦，但这可能部分归因于当时使用的 cosine decay schedule 导致 LR 过早衰减——到 Epoch 8 LR 已降至 8e-6，学习率几乎为零。

**注意**: 此运行使用了早期版本的 warmup + cosine decay scheduler。当前代码默认使用 `WarmupConstantScheduler`（warmup 后保持恒定 LR），更适合 domain-adaptive pretraining 场景。此外还新增了 `PhasedCosineWarmupDecayScheduler`（多阶段余弦退火 + ramp-up + 峰值衰减），与主实验的 `CosineAnnealingWarmupDecay` 设计对齐。如需更长训练（50+ epochs），建议使用 `--scheduler phased_cosine`。

> **Best model**: `checkpoints/cbramod/further_pretrain_20260322_0042/best_model.pth` (Epoch 9)

## 5. 下游评估

### 5.1 实验设计

使用标准训练管线，通过 `--pretrained-weights` flag 指定 further-pretrained 权重，与 ExperimentDB 中记录的历史最优 baseline 对比。

- **Within-subject**: 每个被试独立训练和评估
- **Cross-subject**: 全部 21 个被试数据联合训练，按被试评估
- 评估任务: Binary (食指/中指) 和 Ternary (食指/中指/休息)

### 5.2 结果汇总

> **数据来源**: ExperimentDB (`results/experiments.db`) + 评估结果文件

| 范式 | 任务 | Baseline (TUEG) | Further-PT (MI) | 差异 |
|------|------|:---:|:---:|:---:|
| Within-subject | Binary | **85.09%** ± 10.46% | 83.84% ± 10.71% | **-1.25%** |
| Cross-subject | Binary | **90.54%** ± 9.25% | 88.84% ± 9.03% | **-1.70%** |
| Within-subject | Ternary | **69.54%** ± 12.84% | 69.25% ± 14.33% | **-0.29%** |
| Cross-subject | Ternary | 75.42% ± 12.72% | **75.67%** ± 12.91% | **+0.25%** |

> **Baseline 来源**:
> - Binary within: `run_tag=20260321_0343` (post-HPO)
> - Binary cross: `run_tag=20260321_0608` (post-HPO)
> - Ternary within: `run_tag=20260205_0306` (pre-HPO)
> - Ternary cross: `run_tag=20260207_2056` (pre-HPO)
>
> **Further-PT 来源**:
> - Binary within: `results/20260322_1034_cbramod_imagery_binary.json`
> - Binary cross: `results/20260322_1116_cross-subject_cbramod_imagery_binary.json`
> - Ternary within: `results/20260322_1435_cbramod_imagery_ternary.json`
> - Ternary cross: `results/20260322_1543_cross-subject_cbramod_imagery_ternary.json`

### 5.3 逐被试对比 (Binary)

#### Within-Subject Binary

| 被试 | Baseline | Further-PT | Delta |
|:---:|:---:|:---:|:---:|
| S01 | 83.75% | 83.12% | -0.63% |
| S02 | 93.12% | 91.88% | -1.25% |
| S03 | 98.75% | 97.50% | -1.25% |
| S04 | 91.88% | 88.75% | -3.12% |
| S05 | 83.12% | 80.00% | -3.12% |
| S06 | 73.12% | 68.12% | -5.00% |
| S07 | 77.50% | 77.50% | 0.00% |
| S08 | 95.00% | 92.50% | -2.50% |
| S09 | 99.38% | 96.25% | -3.12% |
| S10 | 61.88% | 58.75% | -3.12% |
| S11 | 89.38% | 88.75% | -0.63% |
| S12 | 87.50% | 81.88% | -5.63% |
| S13 | 93.12% | 90.00% | -3.12% |
| S14 | 83.12% | 78.12% | -5.00% |
| S15 | 90.62% | 92.50% | +1.88% |
| S16 | 74.38% | 83.75% | +9.38% |
| S17 | 84.38% | 82.50% | -1.88% |
| S18 | 88.12% | 88.75% | +0.63% |
| S19 | 98.75% | 98.75% | 0.00% |
| S20 | 64.38% | 60.62% | -3.75% |
| S21 | 75.62% | 80.62% | +5.00% |
| **Mean** | **85.09%** | **83.84%** | **-1.25%** |

改善的被试: 4/21 (S15, S16, S18, S21)；退步: 15/21；持平: 2/21。

#### Cross-Subject Binary

| 被试 | Baseline | Further-PT | Delta |
|:---:|:---:|:---:|:---:|
| S01 | 93.12% | 92.50% | -0.63% |
| S02 | 95.00% | 91.25% | -3.75% |
| S03 | 100.00% | 100.00% | 0.00% |
| S04 | 98.75% | 90.62% | -8.12% |
| S05 | 91.88% | 88.75% | -3.12% |
| S06 | 87.50% | 80.62% | -6.88% |
| S07 | 90.00% | 90.00% | 0.00% |
| S08 | 96.88% | 96.88% | 0.00% |
| S09 | 98.12% | 98.12% | 0.00% |
| S10 | 66.25% | 68.75% | +2.50% |
| S11 | 93.75% | 88.12% | -5.63% |
| S12 | 90.00% | 86.25% | -3.75% |
| S13 | 93.75% | 93.75% | 0.00% |
| S14 | 88.12% | 85.62% | -2.50% |
| S15 | 94.38% | 96.88% | +2.50% |
| S16 | 95.62% | 91.25% | -4.38% |
| S17 | 90.62% | 89.38% | -1.25% |
| S18 | 93.75% | 93.12% | -0.63% |
| S19 | 99.38% | 98.12% | -1.25% |
| S20 | 65.62% | 62.50% | -3.12% |
| S21 | 78.75% | 83.12% | +4.38% |
| **Mean** | **90.54%** | **88.84%** | **-1.70%** |

### 5.4 逐被试对比 (Ternary)

#### Within-Subject Ternary

| 被试 | Baseline | Further-PT | Delta |
|:---:|:---:|:---:|:---:|
| S01 | 58.75% | 64.58% | +5.83% |
| S02 | 81.67% | 84.58% | +2.92% |
| S03 | 84.58% | 89.17% | +4.58% |
| S04 | 87.92% | 92.08% | +4.17% |
| S05 | 59.58% | 56.67% | -2.92% |
| S06 | 75.00% | 72.92% | -2.08% |
| S07 | 68.33% | 67.92% | -0.42% |
| S08 | 76.25% | 76.25% | 0.00% |
| S09 | 85.00% | 87.50% | +2.50% |
| S10 | 42.92% | 40.00% | -2.92% |
| S11 | 69.58% | 68.75% | -0.83% |
| S12 | 61.25% | 60.00% | -1.25% |
| S13 | 70.83% | 67.92% | -2.92% |
| S14 | 73.33% | 68.75% | -4.58% |
| S15 | 57.08% | 57.92% | +0.83% |
| S16 | 62.08% | 61.67% | -0.42% |
| S17 | 81.25% | 79.58% | -1.67% |
| S18 | 64.17% | 64.58% | +0.42% |
| S19 | 92.08% | 91.25% | -0.83% |
| S20 | 47.92% | 41.67% | -6.25% |
| S21 | 60.83% | 60.42% | -0.42% |
| **Mean** | **69.54%** | **69.25%** | **-0.29%** |

注意：Ternary baseline 为 pre-HPO 运行（2026-02-05），与 binary baseline (post-HPO, 2026-03-21) 条件不完全一致。

#### Cross-Subject Ternary

| 被试 | Baseline | Further-PT | Delta |
|:---:|:---:|:---:|:---:|
| S01 | 70.83% | 77.50% | +6.67% |
| S02 | 89.17% | 88.75% | -0.42% |
| S03 | 92.08% | 90.00% | -2.08% |
| S04 | 89.17% | 89.17% | 0.00% |
| S05 | 65.42% | 65.00% | -0.42% |
| S06 | 80.83% | 79.17% | -1.67% |
| S07 | 72.50% | 74.58% | +2.08% |
| S08 | 85.42% | 85.42% | 0.00% |
| S09 | 89.58% | 88.75% | -0.83% |
| S10 | 56.25% | 52.08% | -4.17% |
| S11 | 77.50% | 84.58% | +7.08% |
| S12 | 61.25% | 63.75% | +2.50% |
| S13 | 76.67% | 80.83% | +4.17% |
| S14 | 81.67% | 80.00% | -1.67% |
| S15 | 75.42% | 72.08% | -3.33% |
| S16 | 71.25% | 63.33% | -7.92% |
| S17 | 80.83% | 80.83% | 0.00% |
| S18 | 71.25% | 75.42% | +4.17% |
| S19 | 93.75% | 93.75% | 0.00% |
| S20 | 43.75% | 45.42% | +1.67% |
| S21 | 59.17% | 58.75% | -0.42% |
| **Mean** | **75.42%** | **75.67%** | **+0.25%** |

## 6. 分析与讨论

### 6.1 Overall: Further Pre-training 未带来改善

4 个评估条件中，3 个出现退步（-1.25%, -1.70%, -0.29%），1 个微弱提升 (+0.25%)。平均 delta = **-0.75%**。Further pre-training 在当前条件下对下游 finger MI 分类无正面效果。

### 6.2 可能原因分析

**1. Domain 不匹配**

外部 MI 数据集主要涉及左/右手运动想象（粗粒度运动），而下游任务是单指运动想象（细粒度运动）。这一 domain gap 可能导致 further pre-training 学到的表征反而偏离了下游任务需要的精细特征。

**2. 数据量不足**

252 小时 MI 数据（相对 15,000+ 小时 TUEG）不足以显著改变模型表征。仅 2,360 步参数更新（原始预训练可能数十万步），模型权重变化有限。

**3. 数据不均衡**

Stieger2021 占采样权重 52.7%，且仅包含 14/62 个被试。训练可能过度适配该数据集的分布。

**4. 训练不充分 vs 灾难性遗忘**

10 epochs + aggressive cosine decay 导致训练偏短。但同时更长的训练也可能加剧灾难性遗忘——在少量 domain 数据上过度适配会损害原始 TUEG 预训练获得的通用 EEG 表征。这是一个两难问题。

**5. Baseline 已经很强**

Post-HPO binary baseline (85.09%/90.54%) 已经是经过超参数优化的最优结果。在此基础上进一步提升本身就困难。

### 6.3 Ternary Cross-Subject 微弱正面信号

唯一正向的组合 (ternary cross-subject, +0.25%) 可能因为：
- Ternary baseline 为 pre-HPO（2026-02-07），训练条件次优，留有更多提升空间
- Cross-subject 模式下，further pre-training 对"通用 MI 特征"的增强在多被试联合训练时更有价值
- 但 +0.25% 在统计上不显著

## 7. 技术贡献

尽管下游效果有限，本实验贡献了以下可复用的基础设施：

1. **`--pretrained-weights` CLI flag**: `run_single_model.py` 和 `run_cross_subject.py` 新增自定义权重路径支持，便于后续 A/B 对比实验
2. **MI 数据预处理管线**: `preprocess_mi_datasets.py` + `dataset_metadata.json`，支持 10 个 MOABB 数据集的标准化预处理
3. **`further_pretrain.py`**: CBraMod masked autoencoding further pre-training 训练器，支持多数据集混合、动态通道适配 (ACPE)
4. **单位归一化研究**: 发现并修正 Cho2017 的 MOABB 单位错误（nV 被误认为 µV）

## 8. 未来方向

如需进一步探索 further pre-training：

1. **补全数据**: Schirrmeister2017 (9/14 被试缺失) + Stieger2021 (48/62 被试缺失) 可增加 ~56,000 segments
2. **均衡采样**: 使用 sqrt 或 uniform 权重替代按数据量加权
3. **更长训练 + cyclic LR**: 50 epochs + cosine warmup restart，避免 LR 过早衰减
4. **更小 mask ratio**: 30% 替代 50%，让模型学到更精细的时序特征
5. **Finger MI 专属数据**: 直接在自有 64 小时 finger MI 数据上进一步预训练（最贴近下游任务）
6. **Layer-wise fine-tuning**: 冻结底层 encoder，只进一步训练顶层

## 9. 文件索引

| 文件 | 说明 |
|------|------|
| `checkpoints/cbramod/further_pretrain_20260322_0042/` | 训练 checkpoint 目录 |
| `checkpoints/cbramod/further_pretrain_20260322_0042/best_model.pth` | Best model (Epoch 9) |
| `checkpoints/cbramod/further_pretrain_20260322_0042/training_history.json` | 训练曲线 |
| `scripts/pretraining/further_pretrain.py` | Further pre-training 训练脚本 |
| `scripts/pretraining/preprocess_mi_datasets.py` | MI 数据预处理脚本 |
| `scripts/pretraining/dataset_metadata.json` | 数据集单位元数据 |
| `scripts/pretraining/audit_datasets.py` | 数据集审计脚本 |
| `results/20260322_1034_cbramod_imagery_binary.json` | Binary within-subject FT 结果 |
| `results/20260322_1116_cross-subject_cbramod_imagery_binary.json` | Binary cross-subject FT 结果 |
| `results/20260322_1435_cbramod_imagery_ternary.json` | Ternary within-subject FT 结果 |
| `results/20260322_1543_cross-subject_cbramod_imagery_ternary.json` | Ternary cross-subject FT 结果 |
