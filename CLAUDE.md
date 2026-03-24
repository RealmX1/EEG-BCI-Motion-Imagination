# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 语言规范

默认中文为主英文为辅——不管是对话还是文档都使用中文。即使用户使用英文提问，也使用中文回答。技术术语和代码相关内容应将英文提供在括号内。

## 项目概述

EEG 脑机接口研究项目，对比 EEG 基座模型（CBraMod）与传统 CNN（EEGNet）在单指运动解码中的性能。支持 4/8/32/128 通道实验，被试内/跨被试/迁移学习三种训练范式。21 个被试数据已合并（3640 条缓存，31.4 GB）。

实验结果双写：JSON cache + SQLite 注册表 (`ExperimentDB`)。详细命令、文件索引、脚本目录树见 `docs/codebase_reference.md`。

## 常用命令

```bash
uv sync                                                    # 安装依赖
uv run python scripts/run_within_subject_comparison.py     # 被试内对比 (最常用)
uv run python scripts/run_cross_subject_comparison.py      # 跨被试对比
uv run python scripts/run_transfer_comparison.py           # 迁移学习对比
```

运行 `scripts/experiments/` 下的脚本时，默认加 `--cache-only` 避免触发耗时的文件系统扫描：

```bash
uv run python scripts/run_single_model.py --model cbramod --cache-only        # 被试内单模型
uv run python scripts/run_cross_subject.py --model cbramod --cache-only        # 跨被试
```

完整命令参考见 `docs/codebase_reference.md`。

## 数据划分协议

| 数据来源 | 用途 |
|----------|------|
| `Offline*` + `Online*_Sess01_*` + `Online*_Sess02_*_Base` | **训练** (时序分割 80/20) |
| `Online*_Sess02_*_Finetune` | **测试** (完全独立) |

关键约束：Trial-level 分割（防泄露）、时序分割（验证集取末 20%）、Quaternary 仅 Offline 数据 60/20/20。详见 `docs/preprocessing_architecture.md`。

## 模型配置

| 模型 | 通道 | 采样率 | 滤波 | 归一化 | 参数量 |
|------|------|--------|------|--------|--------|
| EEGNet | 128 | 100 Hz | 4-40 Hz | Z-score | ~2.5K |
| CBraMod | 128 | 200 Hz | 0.3-75 Hz | ÷100 | ~4.0M |

CBraMod 使用 ACPE（非对称条件位置编码）支持任意通道数输入。

## 数据与输出位置

```
data/S01-S21/          # 被试原始数据
caches/preprocessed/   # HDF5 预处理缓存
checkpoints/           # 模型检查点
results/               # 实验结果 (experiments.db + JSON + PNG)
```

## GPU 要求

- **必须使用 NVIDIA GPU**，CPU 模式已禁用
- CBraMod 128 通道模式建议 12GB+ 显存

## 实验结果引用规范

**任何时候引用实验数据（对话、文档、分析报告中），都必须标注数据来源**，包括：
1. 结果文件路径（JSON cache 或 SQLite 查询条件）
2. 实验运行标识（时间戳前缀，如 `20260221_0445`）

这确保所有数值可追溯到原始实验输出，防止张冠李戴。

### 格式规范

**在文档/报告中**，使用 blockquote 标注：

```markdown
> **数据来源**: `results/32_channel/fdr/20260221_0445_transfer_comparison_cache_imagery_binary.json`
```

**在对话中引用数据时**，使用内联标注：

```markdown
cross-subject 准确率 88.10% (来源: `results/32_channel/fdr/20260221_0445_transfer_comparison_cache_imagery_binary.json`, model=cbramod)
```

**引用多个实验对比时**，使用表格附带来源列：

```markdown
| 配置 | Binary Acc | 来源 |
|------|-----------|------|
| FDR 32ch | 88.10% | `results/32_channel/fdr/20260221_0445_..._imagery_binary.json` |
| 补集 32ch | 83.18% | `results/32_channel/fdr_complement/20260301_..._imagery_binary.json` |
```

**引用 SQLite 查询结果时**：

```markdown
> **数据来源**: ExperimentDB 查询 — `SELECT * FROM experiments WHERE channel_config='fdr' AND task='binary' AND paradigm='imagery'`
```

### 命名约定

结果文件遵循格式：`{timestamp}_{experiment_type}_{paradigm}_{task}.json`，其中 timestamp 为 `YYYYMMDD_HHMM`，是唯一标识一次运行的关键字段。

## Baseline 管理规范

Baseline 是每个类别 (model + task + experiment_type) 的标准参考运行，通过 ExperimentDB `is_baseline` 列标记。当前 baseline 注册表见 `docs/dev_log/experiments/baseline_registry.md`。

### 变更记录要求

任何 baseline 的新增、替换或移除都必须记录在 `docs/dev_log/experiments/baseline_registry.md` 的"更新历史"中。

### 替换流程

1. **替换必须由开发者明确提出**——Agent 不得自行决定替换某类别的 baseline
2. **执行替换前，Agent 必须与开发者二次确认类别**：明确列出将被替换的 (model, task, experiment_type) 组合及新旧 run_tag
3. **长时间运行任务期间**：如果新 baseline 的标记状态不会直接影响当前正在执行的 run，应等待所有任务完成后再向开发者提出替换确认，避免干扰运行中的实验

## 参考资料

- 数据集论文: "EEG-based brain-computer interface enables real-time robotic hand control at individual finger level"
- CBraMod: "CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding" (ICLR 2025)
