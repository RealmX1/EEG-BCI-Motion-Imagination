# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 语言规范

默认中文为主英文为辅——不管是对话还是文档都使用中文。即使用户使用英文提问，也使用中文回答。技术术语和代码相关内容应将英文提供在括号内。

## 项目概述

本项目是一个基于脑电图（EEG）的脑机接口（BCI）研究项目，对比验证 EEG 基座模型（CBraMod）与传统 CNN（EEGNet）在单指级别运动解码任务中的性能。

### 当前状态

- ✅ **Phase 3 进行中** - 统一训练框架已完成，支持三阶段实验协议。

已完成:
- ✅ 数据预处理管线 (通道映射、滤波、重采样)
- ✅ **论文对齐预处理** - Run 级别 CAR → 滑动窗口 → 降采样 → 带通滤波 → Z-score
- ✅ **FingerEEGDataset 更新** - 完全支持论文对齐预处理 (Run 级别处理)
- ✅ EEGNet-8,2 基线模型
- ✅ CBraMod 适配器 (集成官方预训练模型)
- ✅ RTX 5070 (Blackwell) GPU 支持
- ✅ **三阶段实验协议** - Offline → Online_Base (训练) → Online_Finetune (测试)
- ✅ **train_within_subject.py 更新** - 使用新的论文对齐预处理
- ✅ **预处理缓存系统** - HDF5 缓存避免重复计算，加速 20-40x
- ✅ **Trial Index 去重修复** - 修复数据泄露问题，实现 trial-level split (2026-01-02)
- ✅ **统一 CBraMod 训练流程** - train_within_subject.py 支持 --model 参数 (2026-01-02)
- ✅ **全被试模型对比脚本** - scripts/run_full_comparison.py 自动化训练+统计对比 (2026-01-02)
- ✅ **三阶段协议完整实现** - 训练用 Offline+Online_Base，测试用 Online_Finetune (2026-01-03)
- ✅ **时序数据分割** - 训练数据按收集顺序提供，验证集取末尾 20% (2026-01-03)
- ✅ **🔴 严重数据泄露修复** - session_type 现使用完整 folder 名称，训练/测试集完全独立 (2026-01-03)
- ✅ **Quaternary (4指) 任务支持** - 仅使用 Offline 数据，时序分割 60/20/20 (2026-01-03)
- ✅ **🔧 分层时序分割修复** - 修复验证集 100% 来自单一 session 的问题，现在训练/验证集有相似的 session 分布 (2026-01-03)

待完成:
- 完整 21 被试数据训练 (当前有 S01-S05)

## 快速命令

```bash
# 安装
uv sync
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# Within-subject 训练 (推荐)
uv run python -m src.training.train_within_subject --subject S01 --task binary --model eegnet
uv run python -m src.training.train_within_subject --subject S01 --task binary --model cbramod
uv run python -m src.training.train_within_subject --subject S01 --task ternary --model both  # 同时训练两个模型

# Motor Execution (ME) 训练
uv run python -m src.training.train_within_subject --subject S01 --task binary --model eegnet --paradigm movement

# 训练所有被试
uv run python -m src.training.train_within_subject --all-subjects --task binary --model eegnet
uv run python -m src.training.train_within_subject --all-subjects --task binary --model cbramod

# 使用特定配置文件
uv run python -m src.training.train_within_subject --subject S01 --task binary --model cbramod --config configs/cbramod_config.yaml

# 全被试模型对比 (推荐用于最终评估)
uv run python scripts/run_full_comparison.py                          # MI 默认，所有被试，两模型
uv run python scripts/run_full_comparison.py --paradigm movement      # ME 模式
uv run python scripts/run_full_comparison.py --new-run                # 新实验 (保留旧结果)
uv run python scripts/run_full_comparison.py --subjects S01 S02 S03   # 指定被试
uv run python scripts/run_full_comparison.py --models eegnet          # 仅 EEGNet
uv run python scripts/run_full_comparison.py --skip-training          # 查看最新结果

# ZIP 预处理 (将 zip 转换为预处理缓存，默认处理后删除解压文件以节省空间)
uv run python scripts/preprocess_zip.py                               # 默认: 处理 data/ 中所有 zip
uv run python scripts/preprocess_zip.py data/S01.zip data/S02.zip     # 处理指定 zip
uv run python scripts/preprocess_zip.py --keep-extracted              # 保留解压文件
uv run python scripts/preprocess_zip.py --extract-only                # 仅解压 (不删除)
uv run python scripts/preprocess_zip.py --subject S01 --preprocess-only  # 仅预处理 (已解压)
uv run python scripts/preprocess_zip.py --force                       # 强制重新生成缓存
```

## 数据划分协议

遵循原论文的实验设计，支持 Motor Imagery (MI) 和 Motor Execution (ME) 两种范式:

### 数据目录结构

| 目录 | 说明 | 用途 |
|------|------|------|
| `Offline{Imagery,Movement}/` | 离线训练 (30 runs) | **训练** |
| `Online{Imagery,Movement}_Sess01_Xclass_Base/` | Session 1 基础 (8 runs) | **训练** |
| `Online{Imagery,Movement}_Sess01_Xclass_Finetune/` | Session 1 适应 (8 runs) | **训练** |
| `Online{Imagery,Movement}_Sess02_Xclass_Base/` | Session 2 基础 (8 runs) | **训练** |
| `Online{Imagery,Movement}_Sess02_Xclass_Finetune/` | Session 2 适应 (8 runs) | **测试** |

### 数据划分

#### Binary/Ternary 任务 (2/3 指分类)

```
训练数据 (Offline + Sess01 全部 + Sess02 Base)
├── OfflineImagery (30 runs)
├── OnlineImagery_Sess01_Xclass_Base (8 runs)
├── OnlineImagery_Sess01_Xclass_Finetune (8 runs)
└── OnlineImagery_Sess02_Xclass_Base (8 runs)
    ↓
    时序分割 (Temporal Split)
    ↓
├── Train (前 80% trials) → 用于模型训练
└── Val (后 20% trials) → 用于早停

测试数据 (Sess02 Finetune) - 完全独立
└── OnlineImagery_Sess02_Xclass_Finetune (8 runs) → 最终评估
```

#### Quaternary 任务 (4 指分类)

**重要**: 4 指分类数据**仅存在于 Offline 模式**中，不存在 `Online*_4class_*` 文件夹。

```
仅 Offline 数据可用
└── OfflineImagery (30 runs, 4 指全部)
    ↓
    时序分割 (Temporal Split)
    ↓
├── Train (前 60% trials) → 用于模型训练
├── Val (中间 20% trials) → 用于早停
└── Test (后 20% trials) → 最终评估
```

使用方式：
```bash
uv run python scripts/run_full_comparison.py --task quaternary
```

### 关键设计决策

1. **训练顺序**: 每 epoch 随机打乱 (`shuffle=True`)，提升梯度估计质量和泛化能力
2. **验证集分割**: 时序分割 (temporal split)，取训练数据最后 20%，避免数据泄露
3. **测试集独立**: Session 2 Finetune 数据完全不参与训练，作为最终评估指标
4. **范式支持**: 支持 `--paradigm imagery` (MI) 和 `--paradigm movement` (ME)
5. **Quaternary 特殊处理**: 仅使用 Offline 数据，时序分割 60/20/20

## 关键文件

| 文件 | 说明 |
|------|------|
| **数据层** ||
| `src/preprocessing/data_loader.py` | **底层数据加载和预处理** - 含论文对齐预处理管线 |
| `src/preprocessing/cache_manager.py` | **预处理缓存** - HDF5 缓存管理，避免重复计算 |
| **模型层** ||
| `src/models/eegnet.py` | EEGNet-8,2 实现 |
| `src/models/cbramod_adapter.py` | CBraMod 适配器 |
| **训练层** ||
| `src/training/train_within_subject.py` | 被试内训练 (对齐原论文) |
| **实验脚本** ||
| `scripts/run_full_comparison.py` | **全被试模型对比** - 自动训练+统计分析+可视化 |
| `scripts/preprocess_zip.py` | **ZIP 预处理** - 将 zip 解压并生成 4 种缓存 (EEGNet/CBraMod × Binary/Ternary) |
| **配置** ||
| `configs/*.yaml` | 训练配置 |
| `docs/experiment_plan_v1.md` | 详细实验计划 |

## 数据预处理

### 论文对齐预处理 (EEGNet)

**关键更新**: `FingerEEGDataset` 现已完全支持论文对齐预处理！

**方法 1: 使用 FingerEEGDataset (推荐)**

```python
from src.preprocessing.data_loader import (
    FingerEEGDataset,
    PreprocessConfig,
)

# 创建论文对齐配置
config = PreprocessConfig.paper_aligned(n_class=2)  # 二分类

# 创建数据集 (自动在 Run 级别应用预处理)
# parallel_workers=0 自动使用 CPU 核心数-1 进行并行预处理 (首次加载加速 2-3x)
dataset = FingerEEGDataset(
    data_root='data',
    subjects=['S01'],
    config=config,
    task_types=['OfflineImagery'],
    target_classes=[1, 4],  # 拇指 (1) vs 小指 (4)
    elc_path='data/biosemi128.ELC',
    parallel_workers=0,  # 0=自动, -1=禁用并行
)

# 数据集已包含预处理后的 segments (非原始 trials)
# dataset[0] 返回: (segment [128 x 100], label)
print(f"Loaded {len(dataset)} segments")  # 约 330 segments per run (30 trials × 11 segments/trial)
```

**方法 2: 低级 API (用于自定义流程)**

```python
from src.preprocessing.data_loader import (
    PreprocessConfig,
    preprocess_run_paper_aligned,
    load_mat_file
)

# 加载原始数据
eeg_data, events, metadata = load_mat_file('data/S01/OfflineImagery/S01_OfflineImagery_R01.mat')

# 应用论文对齐预处理
config = PreprocessConfig.paper_aligned()
segments, labels, trial_indices = preprocess_run_paper_aligned(
    eeg_data, events, metadata, config,
    target_classes=[1, 4],
    label_mapping={1: 0, 4: 1}
)
# 输出: segments [n_segments x 128 x 100]
```

**论文预处理流程** (严格对齐原论文):
```
原始 MAT 文件 (128 通道, 1024 Hz)
    ↓
[1] 提取试次
    ↓ 基于 Target/TrialEnd 事件
    ↓ 填充至 5s (offline) / 3s (online)
    ↓ 输出: [n_trials x 128 x 5120]
    ↓
[2] Common Average Reference (CAR)
    ↓ **逐 trial 独立应用** (非 Run 级别)
    ↓ trials - trials.mean(axis=1, keepdims=True)
    ↓
[3] 滑动窗口分割
    ↓ 窗口: 1024 样本 (1s @ 1024 Hz)
    ↓ 步长: 128 样本 (125ms @ 1024 Hz)
    ↓ 每个试次 → ~11 个 segments
    ↓ 输出: [n_segments x 128 x 1024]
    ↓
[4] 降采样至 100 Hz
    ↓ scipy.signal.resample
    ↓ 1024 样本 → 100 样本
    ↓ 输出: [n_segments x 128 x 100]
    ↓
[5] 带通滤波 4-40 Hz
    ↓ 4阶 Butterworth, lfilter (因果)
    ↓ 零填充 (padding=100)
    ↓
[6] Z-score 归一化
    ↓ 每个 segment 独立
    ↓ 沿时间轴 (axis=-1)
    ↓
输出: [n_segments x 128 x 100]
```

**重要**:
- 使用 `config.use_sliding_window=True` 时，`FingerEEGDataset` 在 Run 级别处理数据
- **CAR 逐 trial 独立应用**（与原论文 `Functions.py` 完全一致）
- 每个 trial 通过滑动窗口产生多个 segments
- 这完全复刻了原论文的实现

### CBraMod 预处理

使用 `PreprocessConfig.for_cbramod()` (默认启用滑动窗口):

```python
# 默认使用滑动窗口 (推荐，与 EEGNet 公平比较)
config = PreprocessConfig.for_cbramod()  # use_sliding_window=True
dataset = FingerEEGDataset(data_root='data', subjects=['S01'], config=config, ...)
# 流程: 带通 0.3-75 Hz → 陷波 60 Hz → 降采样 200 Hz → 除以 100 → 滑动窗口
# 输出: [n_segments x 19 x 200] (1s @ 200Hz, 19 通道 10-20 系统)

# 不使用滑动窗口 (原始 trial 作为 patch 序列)
config = PreprocessConfig.for_cbramod(use_sliding_window=False)
# 输出: [n_trials x 19 x 1000] (5s @ 200Hz, 19 通道 10-20 系统)
```

**EEGNet vs CBraMod 预处理对比**:

| 特性 | EEGNet | CBraMod |
|------|--------|---------|
| 通道数 | 128 (全部) | 19 (10-20 系统) |
| 采样率 | 100 Hz | 200 Hz |
| 带通滤波 | 4-40 Hz | 0.3-75 Hz |
| 陷波滤波 | 无 | 60 Hz |
| 归一化 | Z-score (时间轴) | 除以 100 |
| 滑动窗口 | 1s, 125ms 步长 | 1s, 125ms 步长 (默认) |
| 输出形状 | [segments, 128, 100] | [segments, 19, 200] |

### 预处理缓存与并行加载

`FingerEEGDataset` 默认启用 HDF5 缓存（lzf 压缩）和并行预处理，大幅加速数据加载。

**性能提升** (以 S01 OfflineImagery 30 runs 为例):
```
首次运行 (串行):     ~32s
首次运行 (并行):     ~14s  (parallel_workers=0, 自动使用多核)
后续运行 (缓存命中): ~5s
首次加速比: 2-3x (并行 + lzf 压缩)
后续加速比: 3-6x (缓存命中)
```

**使用方式**:

```python
# 默认启用缓存
dataset = FingerEEGDataset(
    data_root='data',
    subjects=['S01'],
    config=config,
    use_cache=True,  # 默认值
    cache_dir='caches/preprocessed'  # 默认值
)

# 禁用缓存 (用于调试)
dataset = FingerEEGDataset(..., use_cache=False)
```

**缓存管理**:

```python
from src.preprocessing.cache_manager import get_cache

cache = get_cache()

# 查看缓存统计
stats = cache.get_stats()
print(f"缓存条目: {stats['total_entries']}")
print(f"总大小: {stats['total_size_mb']} MB")

# 清除特定被试缓存
cache.clear_subject('S01')

# 清除所有缓存
cache.clear_all()
```

**缓存失效条件**:
- 源 .mat 文件被修改
- `PreprocessConfig` 参数改变
- `target_classes` 改变
- 缓存版本更新

**缓存位置**:
```
caches/
└── preprocessed/
    ├── {hash}.h5              # HDF5 格式缓存文件
    └── .cache_index.json      # 元数据索引
```

### Trial-level 数据分割

**重要**: 为防止数据泄露，数据分割必须在 trial 级别而非 segment 级别进行。

**问题**: 使用滑动窗口时，每个 trial 产生多个 segments（约 33 个）。如果在 segment 级别分割，同一 trial 的 segments 可能分散到训练集和验证集，导致数据泄露。

**解决方案**:

```python
from sklearn.model_selection import train_test_split

# 获取所有唯一 trials
unique_trials = dataset.get_unique_trials()  # 返回全局唯一的 trial indices

# 获取每个 trial 的标签
trial_labels = []
for trial_idx in unique_trials:
    for i, info in enumerate(dataset.trial_infos):
        if info.trial_idx == trial_idx:
            trial_labels.append(dataset.labels[i])
            break

# Trial-level split (80/20)
train_trials, val_trials = train_test_split(
    unique_trials, test_size=0.2, stratify=trial_labels, random_state=42
)

# 获取对应的 segment indices
train_indices = dataset.get_segment_indices_for_trials(train_trials)
val_indices = dataset.get_segment_indices_for_trials(val_trials)
```

**验证无数据泄露**:

```python
# 验证 train 和 val 的 trial indices 无重叠
train_trial_set = set(train_trials)
val_trial_set = set(val_trials)
assert len(train_trial_set & val_trial_set) == 0, "Data leakage detected!"

# 验证所有 segments 正确归属
train_trial_ids = set(dataset.trial_infos[i].trial_idx for i in train_indices)
val_trial_ids = set(dataset.trial_infos[i].trial_idx for i in val_indices)
assert train_trial_ids == train_trial_set
assert val_trial_ids == val_trial_set
```

**数据集统计** (以 S01 为例):

```
Total segments: 9900
Total unique trials: 300 (30 runs × 10 trials/run)
Train trials: 240 (80%)
Val trials: 60 (20%)
Train segments: 7920 (~33 segments/trial)
Val segments: 1980 (~33 segments/trial)
```

详细修复报告见 `docs/bugfix_trial_index_deduplication.md`

## 数据位置

```
data/
├── S01/                              # 被试数据
│   ├── OfflineImagery/              # Phase 1: 离线训练 (30 runs)
│   ├── OnlineImagery_Sess01_2class_Base/     # Phase 2: 在线训练
│   ├── OnlineImagery_Sess01_2class_Finetune/ # Phase 3: 测试
│   └── ...
├── biosemi128.ELC                    # 电极位置文件
└── channel_mapping.json              # 通道映射表 (自动生成)

github/CBraMod/                       # CBraMod 仓库 (与 EEG-BCI 同级目录)
└── pretrained_weights/
    └── pretrained_weights.pth        # 预训练权重
```

## 模型保存

```
checkpoints/
├── eegnet/{subject}/best.pt    # EEGNet 检查点
└── cbramod/{subject}/best.pt   # CBraMod 检查点

results/
├── results_*.json              # 实验结果
└── optimization_*.json         # 超参数搜索结果
```

## GPU 要求

- **必须使用 NVIDIA GPU**，CPU 模式已禁用
- RTX 50 系列需要 PyTorch nightly + CUDA 12.8
- 如 GPU 不可用，程序立即退出

## Markdown 文档格式规范

1. **章节标题不使用编号**
2. **文档内部引用使用链接**: `详见[数据预处理](#数据预处理)章节`
3. **目录索引后置生成**

## 参考资料

- 数据集: "EEG-based brain-computer interface enables real-time robotic hand control at individual finger level"
- CBraMod: "CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding" (ICLR 2025)
- 实验计划详情: `docs/experiment_plan_v1.md`
- 参考论文: `references/` 目录
