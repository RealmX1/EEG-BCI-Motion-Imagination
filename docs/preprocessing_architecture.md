# 预处理管线架构

本文档详细说明 EEG-BCI 项目的数据预处理管线架构。

## 预处理流程概览

### EEGNet 预处理 (论文对齐)

```
原始 MAT 文件 (128 通道, 1024 Hz)
    ↓
[1] 提取试次 (基于 Target/TrialEnd 事件)
    ↓ 填充至 5s (offline) / 3s (online)
    ↓ 输出: [n_trials x 128 x 5120]
    ↓
[2] Common Average Reference (CAR)
    ↓ 逐 trial 独立应用
    ↓
[3] 滑动窗口分割
    ↓ 窗口: 1024 样本 (1s @ 1024 Hz)
    ↓ 步长: 128 样本 (125ms)
    ↓ 输出: [n_segments x 128 x 1024]
    ↓
[4] 降采样至 100 Hz (resample_poly)
    ↓ 输出: [n_segments x 128 x 100]
    ↓
[5] 带通滤波 4-40 Hz (4阶 Butterworth)
    ↓
[6] Z-score 归一化 (per segment, 时间轴)
    ↓
输出: [n_segments x 128 x 100]
```

### CBraMod 预处理

```
原始 MAT 文件 (128 通道, 1024 Hz)
    ↓
[1] 提取试次 + CAR (同 EEGNet)
    ↓
[2] 降采样至 200 Hz
    ↓
[3] 滑动窗口分割 (1s 窗口, 125ms 步长)
    ↓ 输出: [n_segments x n_channels x 200]
    ↓
[4] 带通滤波 0.3-75 Hz + 陷波 60 Hz
    ↓
[5] 归一化: 除以 100
    ↓
输出: [n_segments x n_channels x 200]
```

**通道模式**:
- 19 通道: 10-20 国际标准电极位置
- 128 通道: 全部 BioSemi 电极 (ACPE 自动适配)

## 缓存系统

### 缓存架构 (v3.0)

缓存存储 **trial 级别** 数据（非 segment），加载时再应用滑动窗口。

```
缓存内容 (HDF5):
├── trials: [n_trials x n_channels x samples_at_target_fs]
└── labels: [n_trials]

加载时处理:
trials → 滑动窗口 → 滤波 → 归一化 → segments
```

**优势**: 相比 segment 级别缓存，存储空间减少约 6.6x。

### 缓存键计算

```python
cache_key = hash(subject, run, session_folder, config, target_classes)
```

### 缓存管理

```bash
# 查看统计
uv run python scripts/cache_helper.py --stats

# 按条件过滤
uv run python scripts/cache_helper.py --paradigm offline
uv run python scripts/cache_helper.py --model cbramod --subject S01

# 执行删除
uv run python scripts/cache_helper.py --model cbramod --execute
```

**过滤选项**:

| 选项 | 说明 | 可选值 |
|------|------|--------|
| `--paradigm` | 采集范式 | `online`, `offline` |
| `--task-type` | 任务类型 | `imagery`, `movement` |
| `--n-classes` | 分类数 | `2`, `3`, `4` |
| `--phase` | 会话阶段 | `base`, `finetune` |
| `--model` | 目标模型 | `eegnet`, `cbramod` |
| `--subject` | 被试 ID | 如 `S01` |

## Trial-level 数据分割

### 问题

滑动窗口将每个 trial 切分为多个 segments（约 33 个）。如果在 segment 级别分割数据，同一 trial 的 segments 可能同时出现在训练集和验证集，导致数据泄露。

### 解决方案

```python
# 获取唯一 trials
unique_trials = dataset.get_unique_trials()

# Trial-level 分割
train_trials, val_trials = train_test_split(unique_trials, test_size=0.2)

# 映射到 segment indices
train_indices = dataset.get_segment_indices_for_trials(train_trials)
val_indices = dataset.get_segment_indices_for_trials(val_trials)
```

### 验证

```python
# 确保无泄露
train_set = set(train_trials)
val_set = set(val_trials)
assert len(train_set & val_set) == 0
```

## API 参考

### PreprocessConfig

```python
# EEGNet 配置
config = PreprocessConfig.paper_aligned(n_class=2)

# CBraMod 配置
config = PreprocessConfig.for_cbramod()              # 128 通道
config = PreprocessConfig.for_cbramod(full_channels=False)  # 19 通道
```

### FingerEEGDataset

```python
dataset = FingerEEGDataset(
    data_root='data',
    subjects=['S01'],
    config=config,
    task_types=['OfflineImagery'],
    target_classes=[1, 4],
    use_cache=True,
    parallel_workers=0,  # 0=自动, -1=禁用
)
```

## 相关文档

- 实验计划: `docs/experiment_plan_v1.md`
- EEGNet 实现对比: `docs/eegnet_implementation_comparison.md`
- Bug 修复记录: `docs/dev_log/bugfixes/`
