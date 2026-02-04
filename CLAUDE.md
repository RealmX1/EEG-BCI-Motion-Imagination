# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 语言规范

默认中文为主英文为辅——不管是对话还是文档都使用中文。即使用户使用英文提问，也使用中文回答。技术术语和代码相关内容应将英文提供在括号内。

## 项目概述

本项目是一个基于脑电图（EEG）的脑机接口（BCI）研究项目，对比验证 EEG 基座模型（CBraMod）与传统 CNN（EEGNet）在单指级别运动解码任务中的性能。

**当前状态**: Phase 3 - 全被试训练进行中。框架已完成，**所有 21 个被试数据 (S01-S21) 已合并完成**。详见 `docs/dev_log/changelog.md`。

**缓存状态**: 3640 条预处理缓存（31.4 GB），覆盖所有 21 个被试。合并报告: `caches/MERGE_COMPLETE_REPORT.txt`

## 快速命令

```bash
# 安装
uv sync
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# 验证安装
uv run python scripts/verify_installation.py

# 全被试模型对比 (推荐)
uv run python scripts/run_full_comparison.py                          # 训练所有被试
uv run python scripts/run_full_comparison.py --new-run                # 新实验 (保留旧结果)
uv run python scripts/run_full_comparison.py --skip-training          # 仅查看结果
uv run python scripts/run_full_comparison.py --paradigm movement      # Motor Execution 范式

# 单模型训练
uv run python scripts/run_single_model.py --subject S01 --model eegnet --task binary
uv run python scripts/run_single_model.py --subject S01 --model cbramod --wandb

# 数据预处理 (ZIP -> 缓存)
uv run python scripts/preprocess_zip.py                               # Motor Imagery
uv run python scripts/preprocess_zip.py --paradigm movement           # Motor Execution

# 缓存管理
uv run python scripts/cache_helper.py --stats
uv run python scripts/cache_helper.py --model cbramod --execute

# 跨被试训练与迁移学习
uv run python scripts/run_cross_subject.py --model eegnet                # 跨被试预训练
uv run python scripts/run_cross_subject.py --model cbramod --subjects S01 S02 S03 S04 S05

# 个体微调
uv run python scripts/run_finetune.py \
    --pretrained checkpoints/cross_subject/eegnet_imagery_binary/best.pt \
    --subject S01
uv run python scripts/run_finetune.py \
    --pretrained checkpoints/cross_subject/cbramod_imagery_binary/best.pt \
    --all-subjects --freeze-strategy backbone

# 迁移学习完整对比实验
uv run python scripts/run_transfer_comparison.py --task binary           # 完整对比
uv run python scripts/run_transfer_comparison.py --task binary --models eegnet  # 仅 EEGNet
```

## 数据划分协议

遵循原论文实验设计，支持 Motor Imagery (MI) 和 Motor Execution (ME) 两种范式。

| 数据来源 | 用途 |
|----------|------|
| `Offline*` + `Online*_Sess01_*` + `Online*_Sess02_*_Base` | **训练** (时序分割 80/20) |
| `Online*_Sess02_*_Finetune` | **测试** (完全独立) |

**关键设计决策**:
1. **Trial-level 分割**: 防止同一 trial 的 segments 泄露到验证集
2. **时序分割**: 验证集取训练数据最后 20%
3. **Quaternary 特殊处理**: 仅使用 Offline 数据，时序分割 60/20/20

详细架构说明见 `docs/preprocessing_architecture.md`。

## 关键文件

### src/ 模块

| 文件 | 说明 |
|------|------|
| `src/preprocessing/data_loader.py` | 数据加载和预处理管线 |
| `src/preprocessing/cache_manager.py` | HDF5 预处理缓存 (v3.0) |
| `src/models/eegnet.py` | EEGNet-8,2 实现 |
| `src/models/cbramod_adapter.py` | CBraMod 适配器 (支持 19/128 通道) |
| `src/training/train_within_subject.py` | 被试内训练模块 (API) |
| `src/training/train_cross_subject.py` | 跨被试预训练模块 |
| `src/training/finetune.py` | 个体微调模块 (支持冻结策略) |
| `src/results/` | 结果管理 (缓存、序列化、统计) |
| `src/visualization/` | 可视化模块 (对比图、单模型图) |
| `src/config/` | 配置模块 (常量、预设、实验配置) |
| `src/evaluation/metrics.py` | 评估指标库 (TODO: 待集成到训练流程) |

### scripts/ 目录结构

```
scripts/
├── experiments/                # 训练实验脚本
│   ├── run_full_comparison.py  # 全被试模型对比
│   ├── run_single_model.py     # 单模型训练
│   ├── run_cross_subject.py    # 跨被试预训练
│   ├── run_finetune.py         # 个体微调
│   └── run_transfer_comparison.py
├── preprocessing/              # 数据预处理脚本
│   ├── preprocess_zip.py       # ZIP 解压和预处理
│   ├── cache_helper.py         # 缓存管理
│   └── merge_cache_index.py    # 缓存索引合并
├── tools/                      # 工具脚本
│   ├── verify_installation.py  # 安装验证
│   └── compare_schedulers.py   # 调度器对比
├── analysis/                   # 分析脚本
│   └── research/               # 研究分析
└── internal/                   # 内部工具
```

**向后兼容**: 根目录的 wrapper 脚本 (`scripts/run_*.py`) 仍然有效

## 模型配置

| 模型 | 通道 | 采样率 | 滤波 | 归一化 | 参数量 |
|------|------|--------|------|--------|--------|
| EEGNet | 128 | 100 Hz | 4-40 Hz | Z-score | ~2.5K |
| CBraMod | 128 | 200 Hz | 0.3-75 Hz | ÷100 | ~4.0M |

CBraMod 使用 ACPE（非对称条件位置编码）支持任意通道数输入。

## 数据位置

```
data/
├── S01/                              # 被试数据
│   ├── OfflineImagery/              # 离线训练 (30 runs)
│   └── OnlineImagery_Sess*/         # 在线数据
├── biosemi128.ELC                    # 电极位置文件
└── channel_mapping.json              # 通道映射表

checkpoints/                          # 模型检查点
results/                              # 实验结果
caches/preprocessed/                  # 预处理缓存
```

## GPU 要求

- **必须使用 NVIDIA GPU**，CPU 模式已禁用
- **Blackwell GPU (RTX 5070/5080/5090)**: 原生支持，自动启用 TF32 优化
- CBraMod 128 通道模式显存需求较高 (建议 12GB+)

## 文档结构

| 文档 | 说明 |
|------|------|
| `docs/TROUBLESHOOTING.md` | 故障排除指南 |
| `docs/preprocessing_architecture.md` | 预处理管线详细架构 |
| `docs/dev_log/changelog.md` | 开发历史和变更记录 |
| `docs/dev_log/refactoring/` | 代码重构详细记录 (Phase 1-4) |

## 参考资料

- 数据集论文: "EEG-based brain-computer interface enables real-time robotic hand control at individual finger level"
- CBraMod: "CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding" (ICLR 2025)
