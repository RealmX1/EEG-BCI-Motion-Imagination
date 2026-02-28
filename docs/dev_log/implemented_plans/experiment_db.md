# SQLite 实验注册表 (ExperimentDB) 实现文档

**日期**: 2026-02-28
**状态**: ✅ 完成 (Phase 1-4)

---

## 背景

原有实验结果管理存在"三重冗余"：

1. **文件名编码元数据** — `20260221_1319_transfer_comparison_cache_imagery_binary.json` 把时间戳、实验类型、范式、任务塞进文件名，查询靠 glob + 字符串解析
2. **目录层级编码** — `results/32_channel/commercial/` 把通道数和配置编码在路径中
3. **每次实验多文件** — 一次运行产生 cache JSON + final JSON + PNG，关系靠命名约定维护
4. **查询逻辑复杂** — `cache.py`（600+ 行）充满 `find_latest_cache`、`find_compatible_*` 等基于文件系统的搜索函数

## 方案

**SQLite 本地注册表 + WandB 远程仪表板**，职责划分：

- **SQLite** (`results/experiments.db`): 实验注册（所有元数据 + 最终指标），结构化查询，运行恢复
- **WandB**: 训练过程监控（epoch 曲线、系统指标），协作分析，模型工件
- **文件系统**: PNG 图表 + JSON cache（双写保留，渐进式淘汰）

```
训练脚本                    存储层                     消费层
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│ run_within  │     │                  │     │ 可视化脚本    │
│ run_cross   │────→│  SQLite DB       │←────│ (matplotlib)  │
│ run_transfer│     │  (实验注册表)     │     │              │
└──────┬──────┘     └────────┬─────────┘     │ 统计分析      │
       │                     │               └──────────────┘
       │              wandb_run_id 关联
       ▼                     ▼
┌──────────────┐     ┌──────────────────┐
│ WandB Cloud  │     │ results/ 目录     │
│ (epoch曲线)  │     │ (PNG + JSON双写)  │
└──────────────┘     └──────────────────┘
```

## 数据库 Schema (v2)

```sql
-- 实验运行（一次脚本执行 = 一行）
runs (
    run_id          TEXT PRIMARY KEY,      -- '{run_tag}_{experiment_type}[_{n_channels}ch][_{channel_config}]_{paradigm}_{task}'
    run_tag         TEXT NOT NULL,          -- '20260221_1319'
    experiment_type TEXT NOT NULL,          -- 'within_subject' | 'cross_subject' | 'transfer'
    paradigm        TEXT NOT NULL,          -- 'imagery' | 'movement'
    task            TEXT NOT NULL,          -- 'binary' | 'ternary' | 'quaternary'
    n_channels      INTEGER DEFAULT 128,
    channel_config  TEXT,                   -- 'motor_cortex' | 'commercial' | 'fdr' | NULL
    n_subjects      INTEGER,
    is_complete     INTEGER DEFAULT 0,
    git_commit      TEXT,
    wandb_group     TEXT,
    created_at      TEXT NOT NULL,          -- ISO 8601, parsed from run_tag
    updated_at      TEXT NOT NULL,
    is_legacy       INTEGER DEFAULT 0,     -- 1 = migrated from JSON
    legacy_source   TEXT                   -- original JSON filename
)

-- 单个被试的训练结果
subject_results (run_id, subject_id, model_type, best_val_acc, test_acc,
                 test_acc_majority, epochs_trained, training_time, wandb_run_id)
    UNIQUE(run_id, subject_id, model_type)  -- 支持 upsert

-- 模型级汇总统计（冗余但加速查询）
model_summaries (run_id, model_type, mean_acc, std_acc, median_acc, min_acc, max_acc, n_subjects)
    UNIQUE(run_id, model_type)

-- 模型间统计对比
comparisons (run_id, model_a, model_b, mean_diff, paired_ttest_t/p, wilcoxon_stat/p,
             better_model, significant)

-- 迁移学习配置
transfer_configs (run_id, freeze_strategy, finetune_epochs, finetune_lr, ...)
```

## 核心 API

```python
from src.results import ExperimentDB

db = ExperimentDB()  # 默认: results/experiments.db

# 写入
run_id = db.create_run('20260221_1319', 'within_subject', 'imagery', 'binary')
db.save_subject_result(run_id, training_result, wandb_run_id='abc123')
db.save_summary(run_id, 'eegnet', stats_dict)
db.save_comparison(run_id, comparison_result)
db.mark_complete(run_id)

# 查询
runs = db.find_runs(paradigm='imagery', task='binary', n_channels=32)
best = db.get_best_run('imagery', 'binary', 'cbramod', 'within_subject')
latest = db.find_latest_run('imagery', 'binary', 'within_subject')
history = db.get_subject_history('S01')

# 高级查询 (绘图用)
hist = db.find_historical_comparison('imagery', 'binary', exclude_run_id=current_id)
ws = db.find_best_within_subject_results('imagery', 'binary', 'eegnet', subjects=subjects_set)
cs = db.find_best_cross_subject_results('imagery', 'binary', 'cbramod', subjects=subjects_set)

# 断点续训
incomplete = db.get_incomplete_run('imagery', 'binary', 'within_subject')
done = db.get_completed_subjects(run_id, 'eegnet')

# 清理
db.close()  # 或使用 with ExperimentDB() as db:
```

## 实现阶段

### Phase 1: 核心模块 (`src/results/experiment_db.py`)

- `ExperimentDB` 类: Schema 管理、CRUD、查询
- WAL 模式 + `PRAGMA foreign_keys=ON`
- Schema 版本化: `schema_info` 表 + `_migrate_to_v2()`
- 41 个单元测试 (`tests/test_experiment_db.py`)

### Phase 2: 训练脚本双写

4 个脚本迁移，每个脚本独立可验证：

1. `run_single_model.py` — 接受 `db`/`db_run_id` 参数
2. `run_within_subject_comparison.py` — 创建 run + 逐被试写入 + summary + comparison + mark_complete
3. `run_cross_subject_comparison.py` — 同上
4. `run_transfer_comparison.py` — 同上 + `transfer_configs`

**双写模式**: DB 写入用 `try/except` 包裹，失败时降级到日志警告，不影响 JSON cache 主流程。

### Phase 3: 可视化数据源迁移

训练脚本的绘图数据源从 `cache.py` 的 `find_*`/`build_*` 函数迁移到 `ExperimentDB` 查询 + `PlotDataSource` 直接构造。

**已标记 deprecated 的函数** (8 个):
- `find_compatible_historical_results()` → `db.find_historical_comparison()`
- `build_data_sources_from_historical()` → 脚本直接构造 `PlotDataSource`
- `find_best_within_subject_for_model()` → `db.find_best_within_subject_results()`
- `build_cross_subject_data_sources()` → 脚本直接构造
- `build_transfer_data_sources()` → 脚本直接构造
- `prepare_combined_plot_data()` → `db.find_historical_comparison()` + `PlotDataSource`
- `find_compatible_within_subject_results()` → `db.find_historical_comparison()`
- `find_compatible_cross_subject_results()` → `db.find_best_cross_subject_results()`

### Phase 4: 历史数据迁移

`scripts/tools/migrate_results_to_db.py`:
- 扫描 `results/` 下 3 种 JSON 格式
- `git log --diff-filter=A` 追溯文件首次提交
- 文件 mtime → `updated_at`
- run_tag → `created_at`
- 结果: 71 runs / 1351 subject_results

## 文件修改清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/results/experiment_db.py` | 新建 | SQLite 注册表核心模块 |
| `scripts/tools/migrate_results_to_db.py` | 新建 | 一次性迁移脚本 |
| `tests/test_experiment_db.py` | 新建 | 41 个单元测试 |
| `src/results/__init__.py` | 修改 | 导出 `ExperimentDB` |
| `src/results/cache.py` | 修改 | 8 个函数标记 deprecated |
| `src/utils/wandb_logger.py` | 修改 | 新增 `run_id` 属性 |
| `scripts/experiments/run_single_model.py` | 修改 | 接受 DB 参数 |
| `scripts/experiments/run_within_subject_comparison.py` | 修改 | DB 双写 + 可视化迁移 |
| `scripts/experiments/run_cross_subject_comparison.py` | 修改 | 同上 |
| `scripts/experiments/run_transfer_comparison.py` | 修改 | 同上 + transfer_configs |
| `.gitignore` | 修改 | 忽略 SQLite 文件 |
