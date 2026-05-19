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

### purpose / notes / superseded_by 字段（schema v9 + v10）

`runs` 表新增三列承载**实验意图与版本演进**：

#### `purpose`（v9）——"为什么跑这次"

受控词表（定义见 `src/config/constants.py::PURPOSE_VALUES`）：

- `baseline` / `final` / `replication` / `ablation`
- `hpo` / `sweep`
- `sanity_check` / `pilot` / `debug` / `misc`

**关键约束：purpose 编码假设（hypothesis），不编码分析（analysis）。**
- ✅ 写入：实验**为什么启动**（验证哪个假设、消融哪个组件、复现哪次旧实验）
- ❌ 不写入：实验**结果如何**（准确率、是否成功、是否被推翻）—— 这些应进 `summaries` 表或 `docs/dev_log/`、`paper/` 报告中
- `notes` 字段可承载假设文本本身（如 `"H1: 4ch 交集足够区分二元 imagery"`），也避免任何事后结论

#### `purpose_provenance`（v10）——"这个 purpose 是怎么来的"

- `'explicit'`：CLI / API 显式写入（用户在跑实验时即设置）
- `'backward_search'`：事后从 docs、commit history、Claude 对话回溯推断（best-effort，**不可作为权威依据**）
- `NULL`：legacy 未标记

查询时按需过滤：
```python
db.find_runs(purpose='ablation', purpose_provenance='explicit')   # 仅信赖明确标注
db.find_runs(purpose='debug')                                      # 两类都看
```

历史 baseline (`is_baseline=1`) v9 迁移自动回填 `purpose='baseline'`，v10 标记为 `'explicit'`（定义性、权威）。

#### `superseded_by`（v10）——"被哪次新 run 取代"

自引用外键，指向取代当前 run 的新 `run_id`。`NULL` 表示活跃 / 未弃用。

- `find_runs()` 默认 `include_deprecated=False`，自动过滤被取代的 run
- `db.mark_superseded(deprecated_run_id, superseding_run_id)` 用于标记替代关系（会检测环路）
- 用例：早期含 bug 的 run 被 fix 后重跑取代；schema 改变后旧 run 与新 run 不可比；方法学更新

```python
# 旧 run 被新 run 完全取代
db.mark_superseded("20260301_1430_within_subject_imagery_binary",
                   "20260315_0900_within_subject_imagery_binary")

# 调取所有被弃用的 run（论文 supplementary 用）
deprecated = db.find_runs(include_deprecated=True)  # then filter superseded_by IS NOT NULL
```

#### CLI 用法

```bash
uv run python scripts/experiments/run_within_subject.py \
    --purpose ablation --notes "H: 移除 attention head 后 binary 准确率不下降"
```

#### API helper

```python
# 事后给历史 run 补 purpose（带 provenance）
db.set_purpose(run_id, purpose='debug', provenance='backward_search',
               notes='H: 32ch 高准确率是否因数据泄露')
```

### 实验队列 `pending_runs`（schema v11）

研究人员可以把"待跑"的命令登记到 `pending_runs` 表，由 `scripts/queue/runner.py` 顺序消化。

#### 状态机

```
pending → claimed → running → completed (terminal)
                           ↓
                  needs_attention → {pending(retry) | skipped | failed} (terminal)
cancelled (terminal, before claim)
```

状态词表定义在 `src/config/constants.py::QUEUE_STATUS_VALUES`，终态集合是 `QUEUE_TERMINAL_STATUSES`。

#### CLI（`scripts/queue/cli.py`）

```bash
# 登记
uv run python scripts/queue/cli.py add \
    --command "uv run python scripts/experiments/run_within_subject.py --subjects S01 --cache-only --no-wandb --purpose ablation --notes 'H: ...'" \
    --purpose ablation \
    --notes "H: ..." \
    --priority 5

# 查看（默认仅非终态）
uv run python scripts/queue/cli.py list
uv run python scripts/queue/cli.py list --all --json   # 给 monitor agent 用

# 取消未跑的条目
uv run python scripts/queue/cli.py rm 42

# 启动 runner（典型情况下由 /long-run skill 包装）
uv run python scripts/queue/cli.py run                  # 持续轮询新条目
uv run python scripts/queue/cli.py run --drain-and-exit # 跑空就退出
```

#### Runner 行为

- **GPU pre-flight**：runner 启动后**只对第一条** entry 做 60 秒 GPU sanity check（avg util < 10% 即放行）；失败则进入 10 分钟 rolling window 等待。之后的 entries 顺序串行，不再 gate（同一时刻最多一个训练在跑，不需要额外协调）。
- **失败时**：runner 把状态翻成 `needs_attention` 并阻塞轮询（每 30 秒一次），等待 monitor agent 通过 `cli.py set <id> --status pending/skipped/failed` 决策。最长等 1 小时（`MAX_NEEDS_ATTENTION_WAIT_S`）无人接手则自动 `failed`。
- **空队列**：默认 `cli.py run` 持续 poll 新条目（每 60 秒一次），`--drain-and-exit` 跑空立即退出。
- **SIGINT**：捕获后 kill 当前 subprocess + 把当前 `running` 条目状态翻回 `pending`，不丢任务。

#### Monitor subagent 工作流（由 Claude 通过 `/long-run` 调度）

- 每 N 分钟先用 `cli.py has-attention`（exit 0 = 有条目要管），命中后再 `cli.py list --status needs_attention --json` 拉详情
- 看到 `needs_attention` 后：
  1. `cli.py show <id>` 读 `error_summary` 和 command
  2. 用最多 30 分钟做 debug：
     - 已知 transient（CUDA OOM / wandb 超时 / 磁盘满）→ 改命令（`cli.py set <id> --command "..." --status pending --increment-debug`）
     - 代码层修复 → 直接 `cli.py set <id> --status pending --increment-debug`
  3. 修不动：写 `docs/handoffs/YYYY-MM-DD_queue_failure_<id>_<slug>.md`（含 command、error_summary、debug 尝试历史、推测原因），然后 `cli.py set <id> --status skipped --handoff-path "docs/handoffs/..."`
  4. runner 自动 unblock，继续下一条

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

## 论文图表生成与版本管理

论文图表（paper figure）有**单一生成入口**和**持久版本链**，不要再用散落的 ad-hoc 脚本或手改路径。

### 单一来源：figure registry

`scripts/paper/figure_registry.py` 是所有论文图表元数据的唯一真源（fig_id →
paper_label / caption / canonical 输出路径 / 生成器）。新增/修改图表时改这里，
其余脚本都从它读取，禁止再硬编码图表路径或时间戳。

### 统一生成入口

```bash
# 单张（fig_id：fig1..fig_s2）
uv run python scripts/paper/generate_paper_figures.py --figure fig4b
# 全部 registry 图（21 张，含 14 主图）
uv run python scripts/paper/generate_paper_figures.py --figure all
# 仅重生成 canonical PNG、不进 staging
uv run python scripts/paper/generate_paper_figures.py --figure fig4 --no-stage-history
```

- 有 native generator 的图在进程内直接调用；timestamp `--replot` 图
  （fig1/fig6/fig6b）由 registry `generator_command` subprocess 生成。
- `--stage-history`（默认开）：生成后把 PNG `propose` 进版本链 staging；与
  当前 trunk tip 字节相同则**静默跳过**（不产生噪音），不同则留待人工裁决。
- 旧 `FIGURE_GENERATORS` short key（如 `channel_scaling`）仍向后兼容，会自动
  映射到 fig_id 并同样走 staging。
- `--figure all` 任一图失败不中断批次，结尾汇总并以非零退出。

### 版本链 + staging UI

每张图的历史在 `paper/figures/_history/<fig_id>/manifest.json`（`trunk[]`
已接受 / `staging[]` 待裁决 / `rejected[]` 软删除；评论按 `(before_sha,
after_sha)` 内容寻址，跨 accept/reject 存活）。schema 见
`.claude/skills/figure-snapshot-diff/references/history_manifest_format.md`。

```bash
# 浏览器里拖动对比任意两版本、Accept/Reject staging、留评论
uv run python .claude/skills/figure-snapshot-diff/scripts/history_server.py --port 8765
# CLI（propose / accept / reject / list / comment-* / context-bundle）
uv run python .claude/skills/figure-snapshot-diff/scripts/history_cli.py list <fig_id>
```

`accept` 会把 trunk tip 复制到该图的 `canonical_output_path`，论文草稿引用
路径因此保持稳定（fig1/2/3c/6/6b 的 canonical 按设计在 `results/<timestamp>...`，
为数据溯源；其余在 `paper/figures/`）。

### 草稿图路径一致性

```bash
uv run python scripts/paper/update_draft_image_paths.py            # 校验（CI 友好，有 MISMATCH 退出 1）
uv run python scripts/paper/update_draft_image_paths.py --apply    # 按 registry 规范修正（先写 .bak）
```

按 alt 文本里的图号匹配 registry `paper_label`，比对草稿相对路径与
`canonical_output_path`。非 registry 图（电极放置图 3a / S3–S6 等）报
`NOT_IN_REGISTRY` 并跳过。

### Deprecated（勿再使用）

`scripts/paper/build_figures_compare_page.py`、根目录散落的
`paper/figures_compare*.html`、skill 内 `scripts/build_compare_page.py` 均已被
上述版本链 + history server 取代，仅作 legacy fallback。

## 参考资料

- 数据集论文: "EEG-based brain-computer interface enables real-time robotic hand control at individual finger level"
- CBraMod: "CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding" (ICLR 2025)

## 变更记录

### 2026-03-29: 脚本重命名与清理
- `run_single_model.py` → `run_within_subject.py`（含函数名 `run_single_model()` → `run_within_subject()`）
- 删除独立的 `run_cross_subject.py`，其功能已被 `run_cross_subject_comparison.py` 完全覆盖
- `--freeze-strategy` 默认值恢复为 `none`（原值 `backbone` 从未在实际使用中生效）
