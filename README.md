# EEG-BCI: Foundation Model for Finger-Level BCI Decoding

本项目基于 "EEG-based brain-computer interface enables real-time robotic hand control at individual finger level" 论文的 FINGER-EEG-BCI 数据集，对比验证 EEG 基座模型（Foundation Model）在精细手指运动解码任务中的有效性。

## 项目概述

### 研究目标

1. **基座模型的迁移能力**：验证预训练知识能否有效迁移到手指级 BCI 任务
2. **低资源场景优势**：探索小样本微调下的性能表现
3. **实时可行性**：评估推理延迟是否满足实时控制需求（<100ms）

### 模型对比

| 特性 | EEGNet-8,2 (基线) | CBraMod (待验证) |
|------|-------------------|------------------|
| 类型 | 紧凑 CNN | Criss-Cross Transformer |
| 参数量 | ~2.5K | ~4.0M |
| 预训练 | ❌ 无 | ✅ 9000+ 小时 TUEG |
| 输入采样率 | 100Hz | 200Hz |
| 输入通道数 | 128 通道 | 19 通道 (10-20) |

### 数据预处理

本项目实现了与原论文完全对齐的预处理管线:

**EEGNet (论文对齐)**:
```
原始 EEG (128ch, 1024Hz) → CAR (per-trial) → 滑动窗口 (1s, 125ms step) →
降采样 100Hz → 4-40Hz 滤波 → Z-score (per-segment)
```

**CBraMod**:
```
原始 EEG (128ch, 1024Hz) → 0.3-75Hz 滤波 → 60Hz 陷波 → 降采样 200Hz → ÷100
```

**重要提示**: 使用滑动窗口时，必须在 **trial 级别**（而非 segment 级别）分割训练集和验证集，以防止数据泄露。详见 `docs/bugfix_trial_index_deduplication.md`。

## 安装

### 环境要求

- Python 3.9+
- PyTorch 2.0+ with CUDA
- NVIDIA GPU with CUDA support
- [uv](https://docs.astral.sh/uv/) (推荐) 或 pip

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/yourusername/EEG-BCI.git
cd EEG-BCI

# 使用 uv 安装（推荐）
uv sync                    # 安装基础依赖

# 安装 PyTorch with CUDA
# RTX 50 系列 (5070/5080/5090) 需要 CUDA 12.8 nightly:
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# RTX 40 系列及更早使用稳定版:
# uv pip install torch --index-url https://download.pytorch.org/whl/cu124

# 可选依赖
uv sync --extra dev        # 开发依赖
uv sync --extra viz        # 可视化依赖

# (可选) 安装 CBraMod
git clone https://github.com/wjq-learning/CBraMod.git
uv pip install -e CBraMod/
```

### 数据准备

将 FINGER-EEG-BCI 数据集放置在 `data/` 目录下：

```
data/
├── S01/
│   ├── OfflineImagery/
│   │   └── S01_OfflineImagery_R01.mat
│   ├── OnlineImagery_Sess01_2class_Base/
│   └── ...
├── S02/
├── ...
├── S21/
├── biosemi128.ELC
└── README.txt
```

## 使用方法

### 运行实验

推荐使用 `scripts/run_full_comparison.py` 进行全被试模型对比：

```bash
# 运行离线对比实验 (默认)
uv run python scripts/run_full_comparison.py

# 仅运行 EEGNet
uv run python scripts/run_full_comparison.py --models eegnet

# 运行 Motor Execution (ME) 范式
uv run python scripts/run_full_comparison.py --paradigm movement
```

### 训练模型

```bash
# 训练 EEGNet（被试内）- 推荐
uv run python -m src.training.train_within_subject --subject S01 --task binary

# 训练所有被试
uv run python -m src.training.train_within_subject --all-subjects --task binary
```

### 数据验证

```bash
# 验证 trial-level split 无数据泄露
uv run python scripts/verify_trial_split.py --subject S01
```

## 项目结构

```
EEG-BCI/
├── configs/                    # 配置文件
│   ├── eegnet_config.yaml     # EEGNet 训练配置
│   ├── cbramod_config.yaml    # CBraMod 训练配置
│   └── experiment_config.yaml # 实验规划配置
├── data/                       # 数据目录 (不纳入版本控制)
│   ├── S01-S21/               # 被试数据
│   └── biosemi128.ELC         # 电极位置文件
├── src/                        # 源代码
│   ├── preprocessing/         # 数据预处理
│   │   ├── data_loader.py     # 核心数据加载和预处理
│   │   ├── cache_manager.py   # HDF5 缓存管理
│   │   ├── channel_selection.py
│   │   ├── filtering.py
│   │   └── resampling.py
│   ├── models/                # 模型定义
│   │   ├── eegnet.py          # EEGNet-8,2 实现
│   │   └── cbramod_adapter.py # CBraMod 适配器
│   ├── training/              # 训练脚本
│   │   └── train_within_subject.py # 核心训练脚本
│   ├── evaluation/            # 评估工具
│   │   └── metrics.py
│   └── utils/                 # 工具函数
│       ├── device.py          # GPU 检测
│       ├── logging.py         # 日志格式化
│       └── timing.py          # 性能计时
├── scripts/                   # 实验脚本
│   ├── run_full_comparison.py # 全被试模型对比
│   ├── preprocess_zip.py      # ZIP 预处理
│   ├── visualize_comparison.py # 结果可视化
│   └── verify_trial_split.py  # 数据分割验证
├── tests/                     # 测试
│   └── test_stratified_split.py
├── docs/                      # 文档
│   ├── experiment_plan_v1.md  # 实验计划
│   ├── bugfix_*.md            # Bug 修复记录
│   └── archive/               # 归档文档
├── references/                # 参考论文
├── checkpoints/               # 模型检查点 (不纳入版本控制)
├── results/                   # 输出结果 (不纳入版本控制)
├── pyproject.toml             # 项目配置与依赖
├── CLAUDE.md                  # Claude Code 项目指南
└── README.md
```

## 实验设计

### 实验一：标准离线性能对比

在离线数据上对比 EEGNet 与 CBraMod 的分类性能。

| 任务 | 类别 | EEGNet 基线 |
|------|------|-------------|
| 二分类 | 拇指 vs 小指 | 80.56% |
| 三分类 | 拇指 vs 食指 vs 小指 | 60.61% |
| 四分类 | 全部四指 | ~45% |

### 实验二：低资源/快速校准

验证基座模型在小样本下的优势：

- 10% 训练数据（约 1-2 个 Run）
- 30% 训练数据（约 10 个 Run）
- 100% 训练数据（全部）

### 实验三：在线微调策略对比

复现原论文的在线微调流程，对比两模型的适应性。

### 实验四：收敛速度分析

量化基座模型的训练效率优势。

## 评估指标

- **二分类**：Balanced Accuracy, AUROC, AUC-PR
- **多分类**：Balanced Accuracy, Cohen's Kappa, Weighted F1

## 参考资料

1. **Finger-BCI 论文**: "EEG-based brain-computer interface enables real-time robotic hand control at individual finger level"
2. **CBraMod 论文**: Wang et al., "CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding", ICLR 2025
3. **EEG Foundation Models 综述**: "A Simple Review of EEG Foundation Models" (arXiv:2504.20069v2)

## 许可证

本项目仅供学术研究使用。

## 致谢

- Carnegie Mellon University 提供的 FINGER-EEG-BCI 数据集
- CBraMod 团队开源的预训练模型
