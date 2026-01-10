# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 语言规范

默认中文为主英文为辅——不管是对话还是文档都使用中文。即使用户使用英文提问，也使用中文回答。技术术语和代码相关内容应将英文提供在括号内。

## 项目概述

本项目是一个基于脑电图（EEG）的脑机接口（BCI）研究项目，对比验证 EEG 基座模型（CBraMod）与传统 CNN（EEGNet）在单指级别运动解码任务中的性能。

**当前状态**: Phase 3 进行中 - 统一训练框架已完成，支持三阶段实验协议。详见 `docs/dev_log/changelog.md`。

## 快速命令

```bash
# 安装
uv sync
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# Within-subject 训练
uv run python -m src.training.train_within_subject --subject S01 --task binary --model eegnet
uv run python -m src.training.train_within_subject --subject S01 --task binary --model cbramod
uv run python -m src.training.train_within_subject --subject S01 --task binary --model cbramod --cbramod-channels 19  # 19 通道模式

# 全被试模型对比 (推荐)
uv run python scripts/run_full_comparison.py                          # 训练所有被试
uv run python scripts/run_full_comparison.py --new-run                # 新实验 (保留旧结果)
uv run python scripts/run_full_comparison.py --skip-training          # 仅查看结果

# 数据预处理
uv run python scripts/preprocess_zip.py                               # 处理所有 zip
uv run python scripts/cache_helper.py --stats                         # 缓存统计
uv run python scripts/cache_helper.py --model cbramod --execute       # 清理指定缓存
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

| 文件 | 说明 |
|------|------|
| `src/preprocessing/data_loader.py` | 数据加载和预处理管线 |
| `src/preprocessing/cache_manager.py` | HDF5 预处理缓存 |
| `src/models/eegnet.py` | EEGNet-8,2 实现 |
| `src/models/cbramod_adapter.py` | CBraMod 适配器 (支持 19/128 通道) |
| `src/training/train_within_subject.py` | 被试内训练脚本 |
| `scripts/run_full_comparison.py` | 全被试模型对比 |
| `scripts/cache_helper.py` | 缓存管理工具 |
| `configs/*.yaml` | 训练配置 |

## 模型配置

| 模型 | 通道 | 采样率 | 滤波 | 归一化 |
|------|------|--------|------|--------|
| EEGNet | 128 | 100 Hz | 4-40 Hz | Z-score |
| CBraMod (19ch) | 19 | 200 Hz | 0.3-75 Hz | ÷100 |
| CBraMod (128ch) | 128 | 200 Hz | 0.3-75 Hz | ÷100 |

CBraMod 使用 ACPE（非对称条件位置编码）支持任意通道数输入。128 通道模式显存需求较高。

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
- RTX 50 系列需要 PyTorch nightly + CUDA 12.8

## 文档结构

| 文档 | 说明 |
|------|------|
| `docs/preprocessing_architecture.md` | 预处理管线详细架构 |
| `docs/experiment_plan_v1.md` | 实验计划 |
| `docs/dev_log/changelog.md` | 开发历史和变更记录 |
| `docs/dev_log/bugfixes/` | Bug 修复记录 |

## 参考资料

- 数据集论文: "EEG-based brain-computer interface enables real-time robotic hand control at individual finger level"
- CBraMod: "CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding" (ICLR 2025)
