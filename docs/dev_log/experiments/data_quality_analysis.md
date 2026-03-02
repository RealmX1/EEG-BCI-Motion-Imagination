# 逐被试数据质量分析

**Date**: 2026-03-01
**Status**: 完成
**Script**: `scripts/analysis/analyze_data_quality.py`
**Report**: `results/data_quality_report.md`

---

## 概述

对 21 个被试 (S01-S21) 的 HDF5 缓存数据进行系统性质量检查，评估数据污染风险和异常被试。分析直接读取预处理缓存（post-CAR, post-bandpass 4-40 Hz, downsampled 100 Hz, pre-z-score），不参考任何实验结果 JSON，避免确认偏差。

### 数据规模

- **总条目**: 4,376 缓存文件（仅分析 EEGNet 条目，去重后 ~1,980 条）
- **每被试**: 1,900-2,240 trials, 79-96 runs, 9 session folders
- **数据格式**: `[n_trials, 128, 500]` (128 通道 × 5s @ 100 Hz)

---

## 分析方法

### 12 项检查

#### 1. 信号质量

| 检查项 | 方法 | 判定标准 |
|--------|------|----------|
| NaN/Inf | 逐 trial 扫描，区分尾部填充（预期）vs 信号区域（污染）| 信号区域任何 NaN/Inf → critical |
| 死通道 | 通道时间轴方差 < 0.01，超过 50% runs | 死通道数 > 5% → flag |
| 极端振幅 | 排除 NaN 区域后，|值| > mean + 10σ | max > 50,000 → major; >5% trials → minor |
| SNR | ERP 方差 / 残差方差 (dB)，仅用 Offline 数据 | 跨被试 z-score 比较 |

#### 2. 统计异常

| 检查项 | 方法 | 判定标准 |
|--------|------|----------|
| Trial 间方差 | Frobenius 范数的 CV (std/mean) | CV < 0.05 或 > 2.0 → flag |
| 通道间相关 | 50 trial 采样，128×128 相关矩阵均值偏对角 | mean |r| > 0.8 → flag |
| 标签分布 | 每 session 内各类计数比 | session 内 max/min > 2.0 → flag |
| Trial 数量 | 每 run 的 trial 数 vs 预期值 | 异常 run 列出 |

#### 3. 跨 Session 一致性

| 检查项 | 方法 |
|--------|------|
| 振幅偏移 | Session 间通道均值向量 L2 距离 |
| 方差稳定性 | Session 间通道方差 max/min 比，>10x → flag |

#### 4. 污染检测

| 检查项 | 方法 |
|--------|------|
| 重复 trial | 同标签 trial 间 cosine similarity > 0.999 |
| 训练/测试相似性 | 训练 session vs Sess02_Finetune 逐通道 KS 检验 |

### 严重度分级

| 级别 | 含义 | 触发条件 |
|------|------|----------|
| critical | 数据完整性问题 | 信号区域 NaN/Inf |
| major | 严重伪迹，可能影响训练 | max amplitude > 50,000 |
| minor | 中等质量问题 | 重复 trial、>5% 伪迹 trial、CV > 2.0 |
| info | 轻微异常 | 方差不稳定等 |
| clean | 无问题 | 无 flag |

---

## 结果

### 被试分级总览

| 严重度 | 被试 | 数量 |
|--------|------|------|
| **Clean** | S01, S02, S06, S07, S08, S11, S13, S15, S17, S18 | 10/21 |
| **Info** | S12 (方差 20x), S19 (65x), S20 (37x) | 3/21 |
| **Minor** | S03 (7.0% 伪迹), S05 (5.8%), S09 (6.8%), S16 (5.7%), S21 (9.4%) | 5/21 |
| **Major** | S04 (max 306K), S10 (max 268K), S14 (max 126K) | 3/21 |

> **数据来源**: `results/data_quality_report.md`

### 数据污染排查结论

**未发现数据污染**。具体证据：

1. **NaN/Inf**: 所有 NaN 均为预期的 trial 长度填充（尾部 padding），无信号区域 NaN/Inf
2. **死通道**: 128 通道全部正常（方差阈值 0.01）
3. **重复 trial**: 去重加载后无重复（初始误报由缓存索引多条目引起，见下文）
4. **Train/Test 分布**: 大多数被试 KS statistic 0.09-0.29，符合不同 session 采集的预期分布差异

### Major 被试详细分析

#### S04 — 最严重

| 指标 | S04 | 全组均值 | 偏离 |
|------|-----|----------|------|
| Max Amplitude | **306,796** | 37,839 | z=3.2 |
| CV | 3.587 | 1.14 | 极端 |
| 方差比 | 1,367,714x | — | 某 session 方差远超其他 |
| SNR | -21.8 dB | -15.8 dB | 倒数第一 |

**诊断**: 至少一个 session 存在严重的电极脱落或大幅运动伪迹。正常 EEG 振幅（post-CAR, post-bandpass）不会超过数千 µV。

#### S10 — 严重

| 指标 | S10 | 全组均值 | 偏离 |
|------|-----|----------|------|
| Max Amplitude | **267,904** | 37,839 | z=2.7 |
| CV | 4.539 | 1.14 | 全组最高 |
| 方差比 | 822,583x | — | — |
| SNR | -20.3 dB | -15.8 dB | 倒数第三 |

#### S14 — 较严重

| 指标 | S14 | 全组均值 |
|------|-----|----------|
| Max Amplitude | **125,503** | 37,839 |
| CV | 3.898 | 1.14 |
| 方差比 | 68x | — |
| SNR | -19.8 dB | -15.8 dB |

### Train/Test 分布相似性

| 类别 | 被试 | KS Statistic | 说明 |
|------|------|-------------|------|
| 非常相似（潜在关注）| S06, S18, S20 | < 0.10 | 54/45 个通道 train/test 不可区分 |
| 适度不同（正常）| 大多数被试 | 0.10-0.25 | 符合预期 |
| 明显不同（正常）| S05, S12, S16 | > 0.22 | Session 间差异较大 |

### SNR 分布

| 被试 | Mean SNR (dB) | 备注 |
|------|--------------|------|
| S19 | **-9.5** | 全组最高（最好） |
| S09 | -11.2 | |
| S02 | -12.7 | |
| S15 | -13.3 | |
| … | … | |
| S04 | -21.8 | 全组最低（最差） |
| S10 | -20.3 | |
| S14 | -19.8 | |

SNR 最差的 3 个被试恰好是 3 个 major 被试，说明严重伪迹直接拉低了信噪比。

---

## 开发过程中的修正

### 1. NaN 填充的正确处理

**问题**: 初始版本将所有 NaN 视为污染，导致全部 21 个被试被标记为 critical。

**原因**: 缓存中 ~72% 的 trial 末尾有 NaN padding（variable-length trial 对齐到固定长度）。这是 `pipeline.py` 中 `np.pad(..., constant_values=np.nan)` 的预期行为。

**修正**: `check_nan_inf()` 区分尾部连续 NaN（padding）和信号区域 NaN（true contamination）。

### 2. 缓存索引去重

**问题**: 同一 `(subject, run, session_folder)` 在缓存索引中存在 2 个条目（不同 `target_classes` 参数生成不同 cache key）。例如 S01 run 1 的 `OnlineImagery_Sess01_2class_Base` 有 shape `[20, 128, 500]` 和 `[10, 128, 500]` 两个条目。

**影响**: 同一 trial 被加载两次 → duplicate detection 误报 16 对 "重复"。

**修正**: `load_subject_data()` 按 `(session, run)` 去重，保留 trials 数最多的条目。

### 3. 标签分布的正确评估

**问题**: 跨 session 的总体标签分布必然不平衡（Offline 有 4 类，Online 2class 仅有类 1 和 4）。

**修正**: 改为在每个 session 内独立检查类别平衡。

---

## 建议

### 对跨被试训练的影响

| 优先级 | 建议 | 涉及被试 |
|--------|------|----------|
| 高 | 考虑排除或降权 S04/S10/S14 | 3 个 major |
| 中 | 增加基于振幅阈值的 trial rejection | 5 个 minor |
| 低 | 监控 S06 数据量（仅 1,900 trials）| S06 |

### 后续可选工作

1. **Artifact rejection**: 在训练管线中增加 per-trial 振幅检查（阈值建议 P99.9 = ~1000 µV）
2. **被试加权**: 跨被试训练时按数据质量加权采样
3. **排除实验**: 对比排除 S04/S10/S14 前后的跨被试训练性能

---

## Phase 2: 高级分析

**Script**: `scripts/analysis/analyze_data_quality_advanced.py`
**Report**: `results/data_quality_advanced_report.md`

### 新增分析维度

| 分析 | 方法 | 目的 |
|------|------|------|
| 类别可分性 | Fisher 判别比 + Mann-Whitney AUROC (mu+beta 带功率) | 验证信号中是否存在类别信息 |
| 时间漂移 | 逐 run 通道均值 L2 距离 | 检测 session 内信号不平稳性 |
| 频谱特征 | Welch PSD → theta/mu/beta/gamma 带功率 | 频段分布特征 |
| EMG 污染 | 周边 vs 中央通道高频 (20-40 Hz) 功率比 | 检测肌电伪迹 |
| 相邻 Trial 自相关 | 连续 trial 间 Pearson 相关 | 检测 trial 分割质量 |
| 跨被试相似性 | z-scored 10 维特征向量 + 欧氏距离矩阵 | 指导迁移学习被试选择 |

### 关键发现

#### 类别可分性排名

| 排名 | 被试 | Fisher Mean | AUROC Mean | 说明 |
|------|------|------------|------------|------|
| 1 | S09 | 0.1814 | 0.6748 | 最强类别信号 |
| 2 | S13 | 0.0733 | 0.6088 | |
| 3 | S05 | 0.0509 | 0.5834 | |
| ... | | | | |
| 19 | S21 | 0.0031 | 0.5203 | |
| 20 | S04 | 0.0000 | 0.5696 | 伪迹淹没 Fisher，但 AUROC（秩统计量）仍可检测 |
| 21 | S10 | 0.0000 | 0.5253 | 同上 |

> **解读**: 多数被试 AUROC 0.52-0.58，说明 mu+beta 带功率特征本身提供微弱但存在的类别信息。S09 的 Fisher 为全组 40 倍以上，可能因为其 SNR 最高 (-9.5 dB)。S04/S10 的 Fisher≈0 证实伪迹完全掩盖了类别信号。

#### 跨被试聚类

- **最紧密的被试对**: S17↔S03 (0.78), S01↔S20 (0.95), S17↔S21 (0.81)
- **最孤立**: S04 (mean_dist=9.06), S09 (6.16), S10 (5.01)
- **迁移学习建议**: S17/S03/S21 和 S01/S20/S08 形成两个自然聚类，聚类内被试可互相迁移

#### EMG 污染

Post-CAR 数据中周边通道自然高于中央（~2x），仅 S02 (3.38) 和 S09 (3.01) 的 P/C 比超过 3.0 阈值。S09 的高 P/C 比可能与其高 Fisher 比（强类别信号覆盖宽频）相关而非 EMG。

#### 相邻 Trial 自相关

S04 (0.34), S15 (0.36), S16 (0.37) 有中等程度的相邻 trial 相关性，可能由慢漂移引起。多数被试 mean r < 0.1，符合独立 trial 分割预期。

> **数据来源**: `results/data_quality_advanced_report.md`

---

## 脚本使用

```bash
# Phase 1: 基础质量分析（21 被试，~26s）
uv run python scripts/analysis/analyze_data_quality.py

# 聚焦问题被试
uv run python scripts/analysis/analyze_data_quality.py --subjects S04 S10 S14 -v

# 并行度调整
uv run python scripts/analysis/analyze_data_quality.py --workers 8

# Motor Execution 范式（如有缓存）
uv run python scripts/analysis/analyze_data_quality.py --paradigm movement

# Phase 2: 高级分析（21 被试，~130s）
uv run python scripts/analysis/analyze_data_quality_advanced.py

# 指定被试
uv run python scripts/analysis/analyze_data_quality_advanced.py --subjects S01 S04 S09

# 详细输出
uv run python scripts/analysis/analyze_data_quality_advanced.py --workers 8 -v
```
