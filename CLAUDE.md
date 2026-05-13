# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 语言规范

默认中文为主英文为辅——不管是对话还是文档都使用中文。即使用户使用英文提问，也使用中文回答。技术术语和代码相关内容应将英文提供在括号内。

## 项目概述

EEG 脑机接口研究项目，对比 EEG 基座模型（CBraMod）与传统 CNN（EEGNet）在单指运动解码中的性能。支持 4/8/32/128 通道实验，被试内/跨被试/迁移学习三种训练范式。21 个被试数据已合并（3640 条缓存，31.4 GB）。

实验结果双写：JSON cache + SQLite 注册表 (`ExperimentDB`)。详细命令、文件索引、脚本目录树见 `docs/codebase_reference.md`。

## 常用命令
默认加 `--cache-only`避免触发耗时的文件系统扫描 -- 同时，因为部分原subject文件并未本地留存，只能通过缓存文件发现以进行完整实验。
```bash
uv sync                                                               # 安装依赖
uv run python scripts/run_within_subject_comparison.py --cache-only   # 被试内对比
uv run python scripts/run_cross_subject_comparison.py --cache-only    # 跨被试对比
uv run python scripts/run_transfer_comparison.py --cache-only         # 迁移学习对比
```

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

## ExperimentDB 使用指引
ExperimentDB 包含了大部分训练和实验的记录。是检索历史训练、实验记录信息的首选对象。

```python
from src.results.experiment_db import ExperimentDB
db = ExperimentDB()  # 默认 results/experiments.db
```

主要表：`runs`（实验运行）、`subject_results`（被试级结果）、`summaries`（汇总统计）、`baseline_refs`（基线引用）。

常用 API（不支持裸 SQL `.query()` 方法）：
```python
db.get_run(run_tag)                          # 按 run_tag 查单次运行
db.find_runs(paradigm=..., task=..., ...)    # 条件搜索运行列表
db.get_results(run_tag)                      # 获取某次运行的被试级结果
db.get_summary(run_tag)                      # 获取汇总统计
db.find_run_by_tag(run_tag)                  # 按 tag 精确查找
db.get_best_run(paradigm, task, model, exp)  # 查最佳运行
db.find_baseline_run(model, task, exp)       # 查 baseline 运行
```

如果出现错误，使用sql获取最新schema并对相关api进行订正

注意：extra sessions 实验结果目前只写入 JSON cache，不写入 ExperimentDB。查询 extra sessions 数据请直接读取 `results/` 下的 JSON 文件。

### purpose 字段（schema v9）

`runs` 表新增 `purpose` 列承载**实验意图**，使用受控词表（定义见 `src/config/constants.py::PURPOSE_VALUES`）：

- `baseline` / `final` / `replication` / `ablation`
- `hpo` / `sweep`
- `sanity_check` / `pilot` / `debug` / `misc`

配合自由文本 `notes` 列使用（之前留空，v9 开始可通过 CLI 写入）。所有训练脚本通过 `--purpose` / `--notes` 采集：

```bash
uv run python scripts/experiments/run_within_subject.py \
    --purpose ablation --notes "drop attention head, compare with baseline 20260321_0343"
```

查询示例：
```python
db.find_runs(purpose='ablation', task='binary')      # 所有 binary 消融
db.find_runs(purpose='sanity_check', limit=10)        # 最近 sanity check
```

历史 baseline 运行（`is_baseline=1`）在 v9 迁移时已自动回填 `purpose='baseline'`。

### 被试数过滤默认值

查询实验结果时，默认只关注覆盖完整被试范围的运行：
- **常规实验**（within_subject / cross_subject / transfer）：**n_subjects = 21**
- **Extra sessions 实验**：**n_subjects = 15**（仅 15 个被试有额外 session 数据）

部分被试的运行（如早期调试运行）不具备统计代表性，除非明确需要否则应过滤掉。

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
> **数据来源**: ExperimentDB 查询 — `SELECT * FROM runs WHERE channel_config='fdr' AND task='binary' AND paradigm='imagery'`
```

### 命名约定

结果文件遵循格式：`{timestamp}_{experiment_type}_{paradigm}_{task}.json`，其中 timestamp 为 `YYYYMMDD_HHMM`，是唯一标识一次运行的关键字段。

## 运行类别 (Run Category)

"类别"指由以下三个维度组成的分类：
- **model**: eegnet / cbramod
- **task**: binary / ternary / quaternary / unified （technically speaking, this is not part of the default category, but an extension experiment type that goes a step further than the cross_subject/transfer learning)
- **experiment_type**: within_subject / cross_subject / transfer

例如 "eegnet + ternary + within_subject" 构成一个类别。Extra sessions、通道数配置等属于实验变体，不属于默认类别维度。

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

## 变更记录

### 2026-03-29: 脚本重命名与清理
- `run_single_model.py` → `run_within_subject.py`（含函数名 `run_single_model()` → `run_within_subject()`）
- 删除独立的 `run_cross_subject.py`，其功能已被 `run_cross_subject_comparison.py` 完全覆盖
- `--freeze-strategy` 默认值恢复为 `none`（原值 `backbone` 从未在实际使用中生效）
