# 缓存重构计划：存储 Trials 而非 Segments

**状态**: ✅ 已实现 (v3.0)

## 目标

减少缓存大小，同时保持加载速度。

## 问题分析

滑动窗口导致数据膨胀 6.6x：
- 1 个 trial (5s) → 33 个 overlapping segments
- CBraMod 128ch 缓存 ≈ 原始 .mat 大小

## 新策略

```
缓存内容: trials (无重叠)
├── 提取 trials
├── 应用 CAR (per trial, 使用 nanmean 处理 NaN)
└── 降采样到 target_fs (使用 resample_poly)

加载时处理:
├── 滑动窗口分割
├── 带通滤波
└── Z-score 归一化
```

## 实际效果

| 模型 | 旧缓存/run | 新缓存/run | 实际压缩比 |
|------|------------|------------|------------|
| EEGNet (100Hz) | ~51 MB | ~5.1 MB | **10x** |
| CBraMod 128ch (200Hz) | ~101 MB | ~10.2 MB | **10x** |

> 实际压缩比优于预期的 6.6x，因为 lzf 压缩对连续 trial 数据更有效。

## 实现细节

### 1. cache_manager.py

```python
# v3.0: 存储 trials 而非 segments
CACHE_VERSION = "3.0"

# save() 接口变更
def save(self, ..., trials, labels, mat_file_path, ...):
    # HDF5 数据集: "trials" 和 "labels" (移除 "segments" 和 "trial_indices")

# load() 接口变更
def load(self, ...) -> Tuple[np.ndarray, np.ndarray]:
    return trials, labels  # 不再返回 trial_indices
```

### 2. data_loader.py - 新函数

```python
def preprocess_run_to_trials(
    eeg_data, events, metadata, config,
    target_classes=None, store_all_fingers=False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    预处理 run 到 trial 级别（用于缓存）。

    流程:
    1. 提取 trials
    2. 应用 CAR (per trial, 使用 nanmean)
    3. 降采样到 target_fs (使用 resample_poly)

    返回:
    - trials: [n_trials, n_channels, trial_samples_at_target_fs]
    - labels: [n_trials]
    """

def trials_to_segments(
    trials, labels, config
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将 trials 转换为 segments（从缓存加载后调用）。

    流程:
    1. 滑动窗口分割
    2. 带通滤波
    3. Z-score 归一化

    返回:
    - segments: [n_segments, n_channels, segment_samples]
    - seg_labels: [n_segments]
    - trial_indices: [n_segments]
    """
```

### 3. FingerEEGDataset 加载流程

```python
# Cache miss:
trials, labels = preprocess_run_to_trials(...)
cache.save(..., trials, labels, ...)
segments, seg_labels, trial_indices = trials_to_segments(trials, labels, config)
self._store_segments(segments, seg_labels, trial_indices, ...)

# Cache hit:
trials, labels = cache.load(...)
segments, seg_labels, trial_indices = trials_to_segments(trials, labels, config)
self._store_segments(segments, seg_labels, trial_indices, ...)
```

### 4. Bug 修复

- **NaN 处理**: CAR 使用 `np.nanmean()` 而非 `np.mean()` 处理 NaN-padded trials
- **降采样稳定性**: 使用 `scipy.signal.resample_poly()` 替代 `resample()`，避免 FFT artifacts

## 数据泄露防护

Trial-level split 保持不变：
- `get_unique_trials()` 仍然有效
- `get_segment_indices_for_trials()` 仍然有效
- 分割在 trial 级别进行，然后对每个 trial 应用滑动窗口

## 测试检查清单

- [x] 新缓存大小验证 (10x 压缩，优于预期)
- [x] 加载速度对比 (cache hit 3-5x 加速)
- [x] 输出数据形状一致性 (EEGNet: [128, 100], CBraMod: [128, 200])
- [x] Trial-level split 无数据泄露
- [ ] 训练结果可复现 (待验证)

## 迁移说明

旧缓存 (v2.x) 将自动失效，首次运行会重新生成缓存。如需手动清理：

```bash
uv run python scripts/cache_helper.py --all --execute
```
