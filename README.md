# EEG-BCI: Foundation Model for Finger-Level BCI Decoding

本项目基于 FINGER-EEG-BCI 数据集，对比验证 EEG 基座模型（CBraMod）与传统 CNN（EEGNet）在单指级别运动解码任务中的性能。

**当前状态**: Phase 3 训练进行中，框架已完成，当前有 7/21 被试数据 (S01-S07)。

## 项目概述

### 研究目标

1. **基座模型的迁移能力**：验证 CBraMod 预训练知识能否有效迁移到手指级 BCI 任务
2. **系统性参数评估**：ML Engineering 实验框架评估预处理参数对性能的影响
3. **公平对比**：严格遵循论文协议，确保 EEGNet 与 CBraMod 的可比性

### 模型对比

| 特性 | EEGNet-8,2 (基线) | CBraMod |
|------|-------------------|---------|
| 类型 | 紧凑 CNN | Criss-Cross Transformer |
| 参数量 | ~2.5K | ~4.0M |
| 预训练 | 无 | 9000+ 小时 TUEG |
| 采样率 | 100 Hz | 200 Hz |
| 通道数 | 128 | 128 (或 19 via ACPE) |
| 滤波 | 4-40 Hz | 0.3-75 Hz |
| 归一化 | Z-score | ÷100 |

### 支持的范式和任务

**实验范式 (Paradigm)**:
- `imagery` - Motor Imagery (MI)，运动想象，默认
- `movement` - Motor Execution (ME)，运动执行

**分类任务 (Task)**:
- `binary` - 拇指 vs 小指 (类别 1, 4)
- `ternary` - 拇指 vs 食指 vs 小指 (类别 1, 2, 4)
- `quaternary` - 全部四指 (类别 1, 2, 3, 4)，仅 Offline 数据

## 安装

### 系统要求

| 要求 | 说明 |
|------|------|
| **Python** | 3.9+ (推荐 3.11.x) |
| **GPU** | NVIDIA GPU **必须**，CPU 模式已禁用 |
| **显存** | 8GB+ (EEGNet) / 12GB+ (CBraMod 128ch) |
| **包管理器** | [uv](https://docs.astral.sh/uv/) (推荐) |

### GPU 兼容性

| GPU 系列 | CUDA 版本 | PyTorch 安装命令 |
|----------|-----------|-----------------|
| RTX 50xx (Blackwell) | 12.8 nightly | `uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128` |
| RTX 40xx / 30xx | 12.4 | `uv pip install torch --index-url https://download.pytorch.org/whl/cu124` |

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/yourusername/EEG-BCI.git
cd EEG-BCI

# 安装依赖
uv sync

# 安装 PyTorch (RTX 50 系列)
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# (可选) 安装 CBraMod
git clone https://github.com/wjq-learning/CBraMod.git
uv pip install -e CBraMod/

# 验证安装
uv run python scripts/verify_installation.py
```

### 数据准备

将 FINGER-EEG-BCI 数据集放置在 `data/` 目录下：

```
data/
├── S01/
│   ├── OfflineImagery/                    # 离线训练 (30 runs)
│   │   └── S01_OfflineImagery_R01.mat
│   ├── OnlineImagery_Sess01_2class_Base/
│   ├── OnlineImagery_Sess01_2class_Finetune/
│   ├── OnlineImagery_Sess02_2class_Base/
│   └── OnlineImagery_Sess02_2class_Finetune/  # 测试集 (独立)
├── S02/ ... S21/
├── biosemi128.ELC                         # 电极位置文件
└── channel_mapping.json                   # 通道映射 (可选)
```

## 快速开始

```bash
# 1. 验证环境
uv run python scripts/verify_installation.py

# 2. 预处理数据 (ZIP → 缓存)
uv run python scripts/preprocess_zip.py

# 3. 运行被试内模型对比
uv run python scripts/run_within_subject_comparison.py
```

## 使用方法

### 被试内模型对比 (推荐)

```bash
# 默认: Motor Imagery, Binary, 双模型对比
uv run python scripts/run_within_subject_comparison.py

# 新实验 (保留历史结果)
uv run python scripts/run_within_subject_comparison.py --new-run

# Motor Execution 范式
uv run python scripts/run_within_subject_comparison.py --paradigm movement

# 仅查看已有结果
uv run python scripts/run_within_subject_comparison.py --skip-training

# 指定模型
uv run python scripts/run_within_subject_comparison.py --models eegnet
uv run python scripts/run_within_subject_comparison.py --models cbramod

# 启用 WandB 追踪
uv run python scripts/run_within_subject_comparison.py --wandb
```

### 单模型训练

```bash
# 训练 EEGNet
uv run python scripts/run_single_model.py --model eegnet

# 训练 CBraMod
uv run python scripts/run_single_model.py --model cbramod

# 指定被试和任务
uv run python scripts/run_single_model.py --model eegnet --subjects S01 S02 --task ternary

# 启用 WandB (交互式提示)
uv run python scripts/run_single_model.py --model eegnet --wandb

# 禁用交互式提示
uv run python scripts/run_single_model.py --model eegnet --wandb --no-wandb-interactive
```

### 数据预处理

```bash
# 处理所有 ZIP (Motor Imagery)
uv run python scripts/preprocess_zip.py

# Motor Execution 范式
uv run python scripts/preprocess_zip.py --paradigm movement

# 仅解压，跳过预处理
uv run python scripts/preprocess_zip.py --extract-only

# 仅预处理已解压数据
uv run python scripts/preprocess_zip.py --preprocess-only --subject S01

# 指定模型/任务
uv run python scripts/preprocess_zip.py --models cbramod --tasks binary
```

### 缓存管理

```bash
# 查看缓存统计
uv run python scripts/cache_helper.py --stats

# 按条件过滤 (干运行)
uv run python scripts/cache_helper.py --model cbramod
uv run python scripts/cache_helper.py --subject S01 --n-classes 2

# 删除匹配的缓存
uv run python scripts/cache_helper.py --model cbramod --execute
uv run python scripts/cache_helper.py --paradigm offline --execute
```

### 结果可视化

```bash
# 从比较结果生成图表
uv run python scripts/visualize_comparison.py results/comparison_*.json

# 指定输出目录
uv run python scripts/visualize_comparison.py results/comparison_*.json --output figures/
```

## 预处理 ML Engineering 实验

系统性评估不同预处理参数对 CBraMod 性能的影响。

### 实验组设计

| 组 | 参数 | 配置 | 说明 |
|----|------|------|------|
| **A** | 滤波 | A1, A6 | 带通/陷波变体 |
| **C** | 归一化 | C1, C2 | ÷100 后的额外归一化 |
| **D** | 窗口 | D1, D2, D3 | 滑动步长 (125/250/500ms) |
| **F** | 质量 | F1, F2 | 振幅阈值控制 |

**固定参数** (CBraMod 论文约束):
- 采样率: 200 Hz
- Patch 长度: 1 秒
- 归一化: ÷100 (强制)
- 通道数: 128

### 运行实验

```bash
# 列出所有实验
uv run python scripts/run_preproc_experiment.py --list

# 运行单个实验
uv run python scripts/run_preproc_experiment.py --exp A2

# 运行整个实验组
uv run python scripts/run_preproc_experiment.py --group A

# 原型验证 (S01-S03, 快速测试)
uv run python scripts/run_preproc_experiment.py --prototype

# 完整实验 + EEGNet 基线
uv run python scripts/run_preproc_experiment.py --all --eegnet-baseline

# 生成实验报告
uv run python scripts/compile_preproc_report.py
```

## 数据划分协议

遵循原论文的严格数据分割协议，防止数据泄露。

### 训练/验证/测试划分

| 数据来源 | 用途 |
|----------|------|
| `Offline*` + `Online*_Sess01_*` + `Online*_Sess02_*_Base` | **训练** (时序分割 80/20) |
| 训练数据最后 20% | **验证** |
| `Online*_Sess02_*_Finetune` | **测试** (完全独立) |

### Trial-Level 分割的重要性

使用滑动窗口时，同一试次 (trial) 会产生多个段 (segment)。**必须**在试次级别分割数据，防止同一试次的 segment 出现在训练集和验证集中。

```
Trial 1 ──► [Seg1, Seg2, Seg3, Seg4, ...] ──► 全部进入 Train 或 Val
Trial 2 ──► [Seg5, Seg6, Seg7, Seg8, ...] ──► 全部进入 Train 或 Val
...
```

### Quaternary 特殊处理

四分类任务仅使用 Offline 数据 (Online 数据不存在 4class 文件夹)：
- 训练: 60%
- 验证: 20%
- 测试: 20%

## 预处理管线

### EEGNet (论文对齐)

```
原始 EEG (128ch, 1024Hz)
    ↓ CAR (per-trial)
    ↓ 滑动窗口 (1s, 125ms step)
    ↓ 降采样 100Hz
    ↓ 4-40Hz 带通滤波
    ↓ Z-score (per-segment)
输出: [n_segments, 128, 100]
```

### CBraMod

```
原始 EEG (128ch, 1024Hz)
    ↓ 0.3-75Hz 带通滤波
    ↓ 60Hz 陷波 (可选)
    ↓ 降采样 200Hz
    ↓ ÷100 归一化
    ↓ [可选] 额外归一化
输出: [n_trials, 128, 200]
```

**关键区别**:
- EEGNet: 降采样 → 滤波
- CBraMod: 滤波 → 降采样
- CBraMod 缓存 trials，segment 在加载时应用

## 缓存系统 (v3.0)

采用 **trial 级别** HDF5 缓存，相比 segment 级别减少约 **5-6 倍**存储空间。

### 缓存策略

- **Offline 数据**: 缓存全部 4 指，加载时过滤到目标类别
- **Online 数据**: 按 target_classes 分类缓存
- **实验隔离**: ML Engineering 实验使用独立缓存目录

### 缓存目录结构

```
caches/preprocessed/
├── {hash}.h5                      # 标准缓存
├── data_preproc_ml_eng/           # 实验隔离
│   ├── A1/
│   ├── A6/
│   └── ...
└── .cache_index.json              # 元数据索引
```

## 项目结构

```
EEG-BCI/
├── src/
│   ├── config/                    # 配置模块
│   │   ├── constants.py           # 全局常量
│   │   ├── training.py            # 训练配置和调度器预设
│   │   └── experiment_config.py   # ML Engineering 实验配置
│   ├── preprocessing/             # 数据预处理
│   │   ├── data_loader.py         # 数据加载入口 (向后兼容层)
│   │   ├── loader.py              # MAT 文件加载
│   │   ├── pipeline.py            # 预处理管线
│   │   ├── dataset.py             # PyTorch Dataset
│   │   ├── discovery.py           # 被试发现
│   │   ├── cache_manager.py       # HDF5 缓存管理 (v3.0)
│   │   ├── filtering.py           # 滤波器实现
│   │   ├── resampling.py          # 重采样
│   │   └── channel_selection.py   # 通道选择
│   ├── models/
│   │   ├── eegnet.py              # EEGNet-8,2 实现
│   │   └── cbramod_adapter.py     # CBraMod 适配器
│   ├── training/                  # 训练模块
│   │   ├── train_within_subject.py # 被试内训练 API
│   │   ├── trainer.py             # WithinSubjectTrainer 类
│   │   ├── schedulers.py          # 学习率调度器
│   │   └── evaluation.py          # 评估函数
│   ├── results/                   # 结果管理
│   │   ├── dataclasses.py         # TrainingResult 等数据类
│   │   ├── cache.py               # 结果缓存
│   │   └── statistics.py          # 统计分析
│   ├── visualization/             # 可视化
│   │   ├── comparison.py          # 模型对比图
│   │   └── single_model.py        # 单模型图
│   ├── evaluation/                # 评估指标 (TODO: 待集成)
│   │   └── metrics.py             # balanced_accuracy, AUROC 等
│   └── utils/
│       ├── table_logger.py        # 彩色表格式 Epoch 日志
│       ├── wandb_logger.py        # WandB 集成
│       ├── device.py              # GPU 检测
│       ├── logging.py             # 日志格式化
│       └── timing.py              # 性能计时
├── scripts/
│   ├── experiments/               # 训练实验脚本
│   │   ├── run_within_subject_comparison.py # 被试内模型对比
│   │   ├── run_single_model.py    # 单模型训练
│   │   └── run_cross_subject.py   # 跨被试预训练
│   ├── preprocessing/             # 数据预处理脚本
│   │   ├── preprocess_zip.py      # ZIP 解压 + 缓存生成
│   │   └── cache_helper.py        # 缓存管理工具
│   ├── analysis/                  # 分析脚本
│   │   ├── run_preproc_experiment.py  # ML Engineering 实验
│   │   └── compile_preproc_report.py  # 实验报告生成
│   ├── tools/                     # 工具脚本
│   │   └── verify_installation.py # 环境验证
│   └── run_*.py                   # 向后兼容 wrapper 脚本
├── docs/
│   ├── preprocessing_architecture.md
│   ├── TROUBLESHOOTING.md
│   └── dev_log/
│       ├── changelog.md
│       └── refactoring/           # 代码重构详细记录
├── data/                          # 数据目录 (不纳入版本控制)
├── caches/                        # 预处理缓存 (不纳入版本控制)
├── checkpoints/                   # 模型检查点 (不纳入版本控制)
├── results/                       # 实验结果 (不纳入版本控制)
├── CLAUDE.md                      # Claude Code 项目指南
└── README.md
```

## WandB 实验追踪

```bash
# 首次使用需登录
wandb login

# 启用追踪 (默认交互式提示)
uv run python scripts/run_single_model.py --model eegnet --wandb

# 禁用交互式提示
uv run python scripts/run_single_model.py --model eegnet --wandb --no-wandb-interactive

# 上传模型文件 (默认不上传以节省带宽)
uv run python scripts/run_single_model.py --model eegnet --wandb --upload-model
```

**交互式模式字段**:
- **Run name**: 运行名称
- **Goal**: 实验目标
- **Hypothesis**: 研究假设
- **Notes**: 额外备注

## 评估指标

- **Segment-Level Accuracy**: 原始段精度
- **Trial-Level Majority Voting**: 主要指标，每个 trial 的 segment 预测投票
- **Combined Score**: `(val_acc + majority_acc) / 2`，用于 early stopping

## 参考资料

1. **Finger-BCI 数据集**: "EEG-based brain-computer interface enables real-time robotic hand control at individual finger level"
2. **CBraMod**: Wang et al., "CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding", ICLR 2025
3. **EEGNet**: Lawhern et al., "EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces"

## 许可证

本项目仅供学术研究使用。

## 致谢

- Carnegie Mellon University 提供的 FINGER-EEG-BCI 数据集
- CBraMod 团队开源的预训练模型
