# 预处理版本历史 (Preprocessing Version History)

本文档记录预处理管线的每个版本变更。版本号存储在 `ExperimentDB.runs.preprocessing_version` 字段和 JSON 缓存的 `metadata.preprocessing_version` 中，用于追溯每次实验运行所使用的数据处理方式。

当前版本常量定义在 `src/config/constants.py` 的 `PREPROCESSING_VERSION` 中。

---

## v2.0 — Trial 振幅拒绝统一为 500 µV (2026-03-02)

> **Git commit**: `5bb2395` (2026-03-02 17:18:47 +0800)

### 变更内容

在训练数据的 trial-to-segment 转换阶段新增/修正振幅拒绝：丢弃 `max(|amplitude|) > 500 µV` 的 trial。

- **EEGNet**: `reject_threshold` 从 `-1.0`（禁用）改为 `500.0`
- **CBraMod**: `reject_threshold` 从 `100.0`（过于激进）改为 `500.0`（统一）
- **仅作用于训练数据**: `trials_to_segments(reject_trials=True)` 默认开启；测试/验证数据 `reject_trials=False`
- 典型拒绝率：< 5%（post-CAR 数据幅值通常在 50-200 µV）

### 变更动机

`data_quality_analysis.md` 中发现部分被试 (S04, S10, S14) 存在高幅值异常 trial。v1.0 中 CBraMod 的 100 µV 阈值过于激进，EEGNet 完全没有拒绝。v2.0 统一为合理的 500 µV。

### 关键参数差异 (相对 v1.0)

| 参数 | v1.0 EEGNet | v1.0 CBraMod | **v2.0 (统一)** |
|------|-------------|--------------|-----------------|
| `reject_threshold` | -1.0 (禁用) | 100.0 (过激) | **500.0** |

### 实现位置

- 逻辑: `src/preprocessing/pipeline.py` `trials_to_segments()` (line ~640)
- 配置: `src/preprocessing/data_loader.py` `PreprocessConfig.paper_aligned()`, `.for_cbramod()`
- 调用: `src/preprocessing/dataset.py` `FingerEEGDataset` (`reject_trials` 参数)

---

## v1.0 — CBraMod 滑动步长优化 + 陷波移除 (2026-01-27)

> **Git commit**: `52f1edf` (2026-01-27 03:11:09 +0800)

### 变更内容 (仅 CBraMod)

基于 ML Engineering 实验 (A6, D3 配置) 的结论：

- **滑动步长**: 125ms (128 samples) → **500ms** (512 samples) — D3 实验：3x 更快训练，+1% 准确率
- **陷波滤波**: 60 Hz → **移除** — A6 实验：对 CBraMod 无影响
- **EEGNet**: 无变化（自项目起始保持不变）
- **Trial 拒绝**: 无变化（EEGNet 禁用，CBraMod 100 µV）

### CBraMod 参数对比

| 参数 | v0.2 | **v1.0** |
|------|------|----------|
| 滑动步长 | 125ms (128 samples) | **500ms (512 samples)** |
| 陷波滤波 | 60 Hz | **无** |
| 其他 | 不变 | 不变 |

---

## v0.2 — CBraMod 128 通道 + Trial-level 缓存 (2026-01-11)

> **Git commit**: `0157fa1` (2026-01-11 01:56:06 +0800)

### 变更内容

- **CBraMod 通道数**: 19ch (10-20 标准) → **128ch** (全部 BioSemi)，通过 ACPE 支持任意通道数
- **缓存系统**: segment-level (v2.0) → **trial-level** (v3.0) — 存储完整 trial，加载时动态分段
- 新增 `for_cbramod_128ch()` 工厂方法、`channel_strategy='C'` (全 128ch)
- **EEGNet**: 无变化

### CBraMod 参数对比

| 参数 | v0.1 | **v0.2** |
|------|------|----------|
| 通道数 | 19 (10-20) | **128 (全部)** |
| `channel_strategy` | 'A' | **'C'** |
| `target_model` | 'cbramod' | **'cbramod_128ch'** |
| 缓存格式 | segment-level (v2.0) | **trial-level (v3.0)** |
| 其他 | 不变 | 不变 |

---

## v0.1 — 初始预处理管线 (项目起始 2025-12-28)

> **Git commit**: `35a1a41` (项目初始提交)

### EEGNet 参数

| 步骤 | 参数 |
|------|------|
| 原始采样率 | 1024 Hz |
| 目标采样率 | 100 Hz (resample_poly) |
| 带通滤波 | 4-40 Hz, 4 阶 Butterworth |
| 陷波滤波 | 无 |
| CAR | 是 (逐 trial) |
| 滑动窗口 | 1s 窗口, 125ms 步长 (128 samples @ 1024 Hz) |
| 归一化 | Z-score per segment (时间轴) |
| Trial 拒绝 | 禁用 (`reject_threshold=-1.0`) |
| 通道数 | 128 (全部 BioSemi) |
| 滤波 padding | 100 samples |

> EEGNet 参数自 v0.1 至 v1.0 未变化，v2.0 仅新增 trial 拒绝。

### CBraMod 参数

| 步骤 | 参数 |
|------|------|
| 原始采样率 | 1024 Hz |
| 目标采样率 | 200 Hz |
| 带通滤波 | 0.3-75 Hz, 4 阶 Butterworth |
| 陷波滤波 | **60 Hz** |
| CAR | 是 |
| 滑动窗口 | 1s 窗口, **125ms 步长** (128 samples @ 1024 Hz) |
| 归一化 | 除以 100 |
| Trial 拒绝 | `reject_threshold=100.0` |
| 通道数 | **19** (10-20 标准电极) |
| `channel_strategy` | **'A'** |
| 滤波 padding | 100 samples |

---

## 版本全局不变参数

以下参数在所有版本中未发生变化：

| 参数 | 值 |
|------|------|
| 原始采样率 | 1024 Hz |
| EEGNet 目标采样率 | 100 Hz |
| CBraMod 目标采样率 | 200 Hz |
| EEGNet 带通 | 4-40 Hz |
| CBraMod 带通 | 0.3-75 Hz |
| 滤波阶数 | 4 阶 Butterworth |
| CAR | 始终启用 |
| Trial 时长 | 5s (offline) / 3s (online) |
| 数据划分 | Trial-level 时序分割 (train 80% / val 20%，test 独立) |

---

## 版本变更总览

| 版本 | 日期 | Commit | EEGNet 变化 | CBraMod 变化 |
|------|------|--------|-------------|--------------|
| v0.1 | 2025-12-28 | `35a1a41` | 初始 | 初始 (19ch, 125ms step, 60Hz notch) |
| v0.2 | 2026-01-11 | `0157fa1` | 无 | 19ch→128ch, 缓存 v2→v3 |
| v1.0 | 2026-01-27 | `52f1edf` | 无 | step 125ms→500ms, notch 移除 |
| v2.0 | 2026-03-02 | `5bb2395` | reject: off→500µV | reject: 100µV→500µV |

---

## 如何查询特定版本的实验

```python
from src.results.experiment_db import ExperimentDB

db = ExperimentDB()

# 查找所有 v2.0 的 completed runs
v2_runs = db.find_runs(preprocessing_version='v2.0', is_complete=True)

# 查找 v1.0 的 binary cross-subject runs
v1_binary = db.find_runs(
    preprocessing_version='v1.0',
    task='binary',
    experiment_type='cross_subject',
)
```
