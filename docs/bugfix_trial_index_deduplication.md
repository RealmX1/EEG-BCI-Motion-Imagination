# Trial Index 重复 Bug 修复报告

## 问题概述

**日期**: 2026-01-02

**问题描述**: `src/preprocessing/data_loader.py` 中的 `_load_run_paper_aligned()` 方法在处理多个 run 时，trial index 在每个 run 中都从 0 开始重复，导致严重的数据泄露问题。

### 影响

- **识别错误**: 300 个 trials (30 runs × 10 trials/run) 被错误识别为只有 10 个唯一 trials
- **数据泄露**: 同一个 trial 的多个 segments 被分散到训练集和验证集中
- **性能异常**:
  - Validation accuracy 以 0.1 为单位跳跃（每个 trial 约占 10%）
  - 最终准确率仅 60% 而非预期的 80.56%

## 根本原因

### 原代码问题 (第 916 行)

```python
# _store_segments() 中
trial_info = TrialInfo(
    ...
    trial_idx=trial_idx,  # BUG: 这里使用的是 run 内部的 local index (0-9)
    ...
)
```

### 问题示例

```
Run 1: trial_idx = [0, 1, 2, ..., 9]
Run 2: trial_idx = [0, 1, 2, ..., 9]  # 重复！
Run 3: trial_idx = [0, 1, 2, ..., 9]  # 重复！
...
Run 30: trial_idx = [0, 1, 2, ..., 9] # 重复！

结果: 只有 10 个唯一的 trial_idx，而非 300 个
```

## 修复方案

### 1. 数据加载层修复 (`data_loader.py`)

#### 1.1 添加全局 Trial Counter

在 `FingerEEGDataset.__init__()` 中添加：

```python
# Global trial counter (to ensure unique trial indices across all runs)
self._global_trial_counter = 0
```

#### 1.2 修改 `_store_segments()` 方法

```python
def _store_segments(
    self,
    segments: np.ndarray,
    seg_labels: np.ndarray,
    trial_indices: np.ndarray,
    session_info: Dict
):
    """
    Store preprocessed segments into the dataset.

    CRITICAL FIX: Use global unique trial indices to prevent data leakage.
    Each trial gets a globally unique ID across all runs.
    """
    # Get unique local trial indices from this run
    unique_local_trials = np.unique(trial_indices)

    # Create mapping from local trial_idx to global trial_idx
    local_to_global = {}
    for local_idx in unique_local_trials:
        local_to_global[local_idx] = self._global_trial_counter
        self._global_trial_counter += 1

    # Store segments with globally unique trial indices
    for i, (segment, label, local_trial_idx) in enumerate(zip(segments, seg_labels, trial_indices)):
        # Map local trial index to global unique index
        global_trial_idx = local_to_global[local_trial_idx]

        trial_info = TrialInfo(
            ...
            trial_idx=global_trial_idx,  # FIXED: Use globally unique trial index
            ...
        )
        ...
```

#### 1.3 添加辅助方法

```python
def get_unique_trials(self) -> List[int]:
    """Get list of unique trial indices in the dataset."""
    unique_trials = sorted(set(info.trial_idx for info in self.trial_infos))
    return unique_trials

def get_segment_indices_for_trials(self, trial_indices: List[int]) -> List[int]:
    """Get segment indices that belong to specific trials."""
    trial_set = set(trial_indices)
    segment_indices = [
        i for i, info in enumerate(self.trial_infos)
        if info.trial_idx in trial_set
    ]
    return segment_indices
```

### 2. 训练层修复 (`train_within_subject.py`)

#### 2.1 从 Segment-level Split 改为 Trial-level Split

**旧代码（错误）**:

```python
# 在 segment 级别分割（导致数据泄露）
all_indices = list(range(len(dataset)))
all_labels = [dataset.labels[i] for i in all_indices]

train_indices, val_indices = train_test_split(
    all_indices, test_size=0.2, stratify=all_labels, random_state=42
)
```

**新代码（正确）**:

```python
# CRITICAL FIX: Train/val split at TRIAL level (not segment level)
# This prevents data leakage where segments from same trial end up in both train and val sets
unique_trials = dataset.get_unique_trials()
logger.info(f"Total unique trials: {len(unique_trials)}")

# Get labels for each trial (use first segment's label per trial)
trial_labels = []
for trial_idx in unique_trials:
    # Find first segment from this trial
    for i, info in enumerate(dataset.trial_infos):
        if info.trial_idx == trial_idx:
            trial_labels.append(dataset.labels[i])
            break

# Split trials (not segments) into train/val (80/20)
train_trials, val_trials = train_test_split(
    unique_trials, test_size=0.2, stratify=trial_labels, random_state=42
)

logger.info(f"Train trials: {len(train_trials)}, Val trials: {len(val_trials)}")

# Get segment indices for each split
train_indices = dataset.get_segment_indices_for_trials(train_trials)
val_indices = dataset.get_segment_indices_for_trials(val_trials)

logger.info(f"Train segments: {len(train_indices)}, Val segments: {len(val_indices)}")
```

## 验证结果

### 修复前

```
Total unique trials: 10  ❌ (错误)
Train trials: 8, Val trials: 2
Validation accuracy jumps: 0.1 increments
Final accuracy: ~60%
```

### 修复后

```
Total unique trials: 300  ✅ (正确)
Train trials: 240, Val trials: 60
Train segments: 7920, Val segments: 1980
Validation accuracy: smooth curve
Final accuracy: 63.33%
```

### 数据泄露验证

```python
# 验证结果
[OK] No trial overlap between train and val sets!
[OK] All train segments belong to train trials!
[OK] All val segments belong to val trials!
[OK] No cross-contamination detected!
[OK] All validation trial indices are unique!
```

## Trial Index 分布

### 修复后的分布

```
Run 1:  trial_idx = [0-9]     (10 trials)
Run 2:  trial_idx = [10-19]   (10 trials)
Run 3:  trial_idx = [20-29]   (10 trials)
...
Run 30: trial_idx = [290-299] (10 trials)

Total: 300 unique trial indices
```

### 数据集统计

```
被试: S01
Total segments: 9900
Total unique trials: 300
Train trials: 240 (80%)
Val trials: 60 (20%)
Train segments: 7920
Val segments: 1980
Segments per trial: ~33 (1s window, 125ms step)
```

## 性能影响

### 训练结果对比

| 指标 | 修复前 | 修复后 | 说明 |
|------|--------|--------|------|
| 识别的 trials | 10 | 300 | ✅ 正确识别全部 trials |
| Val set trials | 2 | 60 | ✅ 20% split 正确 |
| Val accuracy 变化 | 跳跃式 (0.1) | 平滑曲线 | ✅ 无数据泄露 |
| Final accuracy | ~60% | 63.33% | ⚠️ 仍需超参数调优 |

### 为什么准确率还没达到 80.56%？

1. **单被试测试**: 当前只用 S01，论文使用 21 被试的平均值
2. **超参数未优化**: 需要运行 hyperparameter tuning
3. **训练轮数**: 可能需要更多 epochs
4. **随机性**: 需要多次实验取平均

## 后续步骤

1. ✅ **修复完成**: Trial index 去重和 trial-level split
2. ✅ **验证通过**: 无数据泄露
3. ⬜ **超参数优化**: 运行 `hyperparameter_tuner.py`
4. ⬜ **多被试训练**: 训练全部 21 个被试
5. ⬜ **性能对比**: 与论文结果对比

## 关键学习点

### 重要原则

1. **数据分割粒度**: 在有时序依赖或多个样本来自同一源时，必须在源级别（trial-level）而非样本级别（segment-level）分割数据
2. **唯一标识符**: 跨多个数据源时，必须使用全局唯一标识符
3. **验证机制**: 实施后务必验证无数据泄露

### 适用场景

这种修复方案适用于所有类似场景：

- EEG/EMG 数据的滑动窗口分割
- 视频帧序列分析
- 时序数据增强
- 医学影像多切片分析

## 文件清单

### 修改的文件

1. `src/preprocessing/data_loader.py`
   - 添加 `self._global_trial_counter`
   - 修改 `_store_segments()` 方法
   - 添加 `get_unique_trials()` 方法
   - 添加 `get_segment_indices_for_trials()` 方法

2. `src/training/train_within_subject.py`
   - 修改 `train_single_subject()` 中的数据分割逻辑
   - 从 segment-level split 改为 trial-level split

### 依赖修复

- 安装 `mpmath` 包（PyTorch nightly 依赖）

## 测试命令

```bash
# 清除旧缓存（包含错误的 trial indices）
uv run python -c "from src.preprocessing.cache_manager import get_cache; get_cache().clear_all()"

# 测试单被试训练
uv run python -m src.training.train_within_subject --subject S01 --task binary

# 验证数据分割
uv run python -c "
from pathlib import Path
from src.preprocessing.data_loader import FingerEEGDataset, PreprocessConfig
from sklearn.model_selection import train_test_split

data_root = Path('data')
config = PreprocessConfig.paper_aligned(n_class=2)
dataset = FingerEEGDataset(
    str(data_root), ['S01'], config,
    task_types=['OfflineImagery'], target_classes=[1, 4],
    elc_path=str(data_root / 'biosemi128.ELC')
)

unique_trials = dataset.get_unique_trials()
print(f'Total unique trials: {len(unique_trials)}')
print(f'Trial IDs range: {min(unique_trials)} to {max(unique_trials)}')
"
```

## 参考资料

- 原始论文: "EEG-based brain-computer interface enables real-time robotic hand control at individual finger level"
- 相关讨论: Trial-level vs Sample-level splitting in ML
- 数据泄露检测: Cross-validation best practices
