# 4 通道 Negative Control 实验分析

> **目的**: 验证 4 通道条件下数据驱动通道选择的有效性。通过对比"最优 4 通道"(FDR∩Attention) 与"随机差通道" (negative control)，量化通道选择在极低通道数下的影响。

---

## 1. 实验设计

### Negative Control 通道生成方法

1. 取 5 个 32ch 配置 (commercial, fdr, csp, attention, band_power) 的**并集** → 97 个通道
2. 计算 128ch 全集的**补集** → 31 个通道（不被任何方法选中）
3. 从 31 个补集通道中随机抽取 4 个 (seed=42) → **[6, 11, 47, 51]** (A7, A12, B16, B20)

> **数据来源**: `results/4_channel/channel_selections.json`, `results/5config_complement.json`
> **生成脚本**: `scripts/analysis/generate_5config_complement.py`, `scripts/analysis/generate_4ch_negative_control.py`

### 对比配置

| 配置 | 通道 | 标签 | 来源 |
|------|------|------|------|
| fdr_attention_overlap | [63, 71, 102, 114] | B32, C8, D7, D19 | FDR∩Attention 交集 (数据驱动最优) |
| negative_control | [6, 11, 47, 51] | A7, A12, B16, B20 | 5 配置并集补集随机抽样 |

---

## 2. 结果总览

### 4ch: Negative Control vs FDR∩Attention

| 任务 | 模型 | fdr_attn_overlap | negative_control | 差值 |
|------|------|------------------|------------------|------|
| binary | EEGNet | 66.79% (±12.75) | 57.05% (±5.71) | **-9.74%** |
| binary | CBraMod | 82.86% (±14.55) | 67.62% (±9.15) | **-15.24%** |
| ternary | EEGNet | 45.62% (±11.50) | 38.39% (±6.90) | **-7.23%** |
| ternary | CBraMod | 64.05% (±12.42) | 53.37% (±8.14) | **-10.68%** |

> **数据来源 (fdr_attention_overlap)**:
> - binary EEGNet: `results/4_channel/fdr_attention_overlap/20260302_0243_cross_subject_cache_imagery_binary.json`
> - binary CBraMod: `results/4_channel/fdr_attention_overlap/20260301_2100_cross_subject_cache_imagery_binary.json`
> - ternary: `results/4_channel/fdr_attention_overlap/20260302_2336_cross_subject_cache_imagery_ternary.json`

> **数据来源 (negative_control)**:
> - binary: `results/4_channel/negative_control/20260309_2329_cross_subject_cache_imagery_binary.json`
> - ternary: `results/4_channel/negative_control/20260310_0054_cross_subject_cache_imagery_ternary.json`

### 4ch Negative Control: Transfer Learning 增益

| 任务 | 模型 | cross-subject | transfer | 增益 |
|------|------|---------------|----------|------|
| binary | EEGNet | 57.05% | 62.08% | +5.03% |
| binary | CBraMod | 67.62% | 72.02% | +4.40% |
| ternary | EEGNet | 38.39% | 42.42% | +4.03% |
| ternary | CBraMod | 53.37% | 57.00% | +3.63% |

> **数据来源 (transfer)**:
> - binary: `results/4_channel/negative_control/20260310_0023_transfer_comparison_cache_imagery_binary.json`
> - ternary: `results/4_channel/negative_control/20260310_0206_transfer_comparison_cache_imagery_ternary.json`

### 跨通道数对比: 32ch vs 4ch Negative Control

| 通道数 | 配置 | 任务 | CBraMod cross-subject |
|--------|------|------|-----------------------|
| 32 | negative_control | binary | 84.08% (±9.36) |
| 4 | negative_control | binary | 67.62% (±9.15) |
| 4 | fdr_attn_overlap | binary | 82.86% (±14.55) |

> **数据来源 (32ch negative_control)**:
> `results/32_channel/negative_control/20260302_0141_cross_subject_cache_imagery_binary.json`

---

## 3. 逐被试对比 (Binary Cross-Subject, CBraMod)

| 被试 | 4ch fdr_attn | 4ch neg_ctrl | 32ch neg_ctrl | 4ch 差值 |
|------|-------------|-------------|--------------|---------|
| S01 | 86.88% | 68.13% | 86.25% | -18.75% |
| S02 | 95.00% | 70.63% | 88.13% | -24.37% |
| S03 | 98.75% | 75.00% | 98.13% | -23.75% |
| S04 | 94.38% | 83.75% | 96.25% | -10.63% |
| S05 | 35.63% | 54.38% | 82.50% | +18.75% |
| S06 | 76.88% | 65.63% | 80.63% | -11.25% |
| S07 | 85.63% | 66.88% | 81.25% | -18.75% |
| S08 | 93.75% | 77.50% | 93.13% | -16.25% |
| S09 | 86.25% | 65.00% | 94.38% | -21.25% |
| S10 | 66.88% | 53.75% | 60.00% | -13.13% |
| S11 | 91.25% | 68.75% | 93.75% | -22.50% |
| S12 | 88.13% | 55.63% | 81.25% | -32.50% |
| S13 | 88.13% | 66.25% | 81.88% | -21.88% |
| S14 | 88.75% | 65.00% | 76.25% | -23.75% |
| S15 | 92.50% | 76.25% | 89.38% | -16.25% |
| S16 | 68.75% | 64.38% | 70.00% | -4.37% |
| S17 | 90.63% | 66.25% | 85.63% | -24.38% |
| S18 | 85.00% | 65.63% | 84.38% | -19.37% |
| S19 | 95.00% | 91.88% | 93.75% | -3.12% |
| S20 | 66.25% | 60.00% | 73.13% | -6.25% |
| S21 | 65.63% | 59.38% | 75.63% | -6.25% |
| **Mean** | **82.86%** | **67.62%** | **84.08%** | **-15.24%** |

> **注**: S05 是唯一 negative control 高于 fdr_attn_overlap 的被试。fdr_attn_overlap 的 S05 仅 35.63%（异常低），该被试的个体差异可能导致 FDR∩Attention 通道恰好不适用。

> **数据来源**: 同上各 cross_subject_cache JSON 文件

---

## 4. 关键结论

### 4.1 通道选择在 4ch 下有效

Negative control 在所有条件下都显著低于 fdr_attention_overlap，**drop 幅度 7-15 个百分点**。CBraMod 的 drop 尤其大 (10-15%)，说明基座模型从优质通道中获益更多。

这与 32ch 实验形成鲜明对比——32ch 下各配置准确率差异很小（FDR 88.10% vs 补集 83.18% vs negative control 84.08%），说明 32 个通道提供了足够的信息冗余。

### 4.2 "32 个差通道 ≈ 4 个好通道"

32ch negative control (84.08%) 与 4ch fdr_attention_overlap (82.86%) 几乎持平。这量化了通道数量 vs 通道质量的 trade-off：
- **增加通道数**可以用冗余信息弥补通道质量的不足
- **精选通道**可以用少量高质量通道达到与大量随机通道相当的效果

### 4.3 Transfer Learning 在差通道上依然有效

迁移学习在 negative control 上带来 **3.6-5.0%** 的稳定提升，与好通道配置上的增益幅度类似，说明预训练→微调范式对通道质量不敏感。

### 4.4 实验方法论意义

4ch negative control 作为阴性对照，**确认了 4ch FDR∩Attention 的高准确率并非随机通道都能达到的**，从而验证了数据驱动通道选择方法在低通道数场景下的实际价值。

---

## 5. 实验元数据

| 项目 | 值 |
|------|-----|
| 范式 | Motor Imagery |
| 被试 | 21 (S01-S21) |
| 4ch negative_control 通道 | [6, 11, 47, 51] (A7, A12, B16, B20) |
| 4ch fdr_attn_overlap 通道 | [63, 71, 102, 114] (B32, C8, D7, D19) |
| 32ch negative_control | 31ch 补集 + ch5, seed=42 |
| 随机种子 | 42 |
| Freeze strategy (transfer) | none |
| Scheduler | cosine_annealing_warmup_decay |
