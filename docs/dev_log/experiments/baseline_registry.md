# Baseline Registry

ExperimentDB Schema v6 引入 `is_baseline` 列，显式标记每个类别 (model + task + experiment_type) 的标准参考运行。本文档记录当前所有 baseline 及缺失类别。

> **数据来源**: ExperimentDB 查询 — `SELECT ... FROM runs r JOIN model_summaries ms WHERE r.is_baseline=1 AND r.paradigm='imagery' AND r.n_channels=128`

## 当前 Baseline

### CBraMod Binary (model selection 实验, 每类 4 runs)

| Experiment Type | run_tag | Mean Acc | Std | N | 选择依据 |
|----------------|---------|----------|-----|---|---------|
| within_subject | `20260323_2237` | 85.15% | 11.00% | 21 | 脚本统一重构后全量验证 |
| cross_subject | `20260324_0023` | 90.68% | 9.31% | 21 | 脚本统一重构后全量验证 |
| transfer | `20260321_1025` | 90.18% | 9.39% | 21 | default config 1/4, combined |

备注:
- within/cross binary baselines 于 2026-03-24 替换为脚本统一重构后的全量验证 run，性能持平（+0.06% / +0.14%）
- Transfer baseline `20260321_1025` 保留不变（本次未重新运行 transfer）
- 历史: `20260321_0343` (soup, 85.09%) → `20260323_2237` (85.15%), `20260321_0608` (90.54%) → `20260324_0023` (90.68%)

### CBraMod Ternary (首个 baseline)

| Experiment Type | run_tag | Mean Acc | Std | N | 选择依据 |
|----------------|---------|----------|-----|---|---------|
| within_subject | `20260323_2320` | 69.44% | 15.42% | 21 | 脚本统一重构后首次全量验证 |
| cross_subject | `20260324_0109` | 74.88% | 14.03% | 21 | 脚本统一重构后首次全量验证 |

备注:
- 此前 ternary 无显式 baseline（runs 未标记 complete 或未执行全量 run）
- 本次为首个 128ch ternary baseline，与历史最佳持平（within: -0.10% vs `20260205_0306`, cross: -0.54% vs `20260207_2056`）

### EEGNet Binary (post-EEGNet-HPO)

| Experiment Type | run_tag | Mean Acc | Std | N | 选择依据 |
|----------------|---------|----------|-----|---|---------|
| within_subject | `20260316_1411` | 78.10% | — | 21 | 唯一 post-EEGNet-HPO v2.0 全量 run (1/1) |

备注:
- EEGNet HPO: 32 trials, best Trial #23 = 82.71% (F1=16, D=4 架构升级)
- HPO 最优参数: `results/hpo/eegnet_within_subject_binary_best_params.json`
- EEGNet cross_subject 和 transfer 无 standalone post-HPO run

### Unified Model (HPO 验证实验, 每类 1/1)

同一 run 包含 EEGNet 和 CBraMod 两个模型的结果。

| Experiment Type | run_tag | CBraMod Acc | EEGNet Acc | N |
|----------------|---------|-------------|------------|---|
| within_subject | `20260320_0243` | 66.96% | 65.21% | 21 |
| cross_subject | `20260320_0548` | 68.17% | 60.12% | 21 |

备注:
- Unified model 同时支持 binary/ternary/quaternary 子任务，mean_acc 是三任务加权平均
- 每个类别只有 1 个 run，是 HPO 参数首次全量验证

## 缺失 Baseline 的类别

| Model | Task | Experiment Type | 原因 | 建议 |
|-------|------|----------------|------|------|
| eegnet | binary | cross_subject | 无 standalone post-HPO run | 需运行 `run_cross_subject_comparison.py --models eegnet --baseline` |
| eegnet | binary | transfer | 无 standalone post-HPO run | 需先有 cross_subject checkpoint，再运行 transfer |
| cbramod | ternary | transfer | 无 post-HPO run | 需先完成 ternary cross_subject → transfer |
| eegnet | ternary | within/cross/transfer | 无 post-HPO run | 需先完成 ternary HPO |
| 两者 | quaternary | 全部 | 无数据 | Quaternary 仅限 Offline 数据，样本量小 |

## 查询方法

### 使用 `/find-baseline` skill

```bash
# 查看所有 baseline
uv run python .claude/skills/find-baseline/scripts/query_baseline.py --baseline-only --type all

# 查看特定类别
uv run python .claude/skills/find-baseline/scripts/query_baseline.py --model cbramod --task binary --type within_subject

# 查看 unified baseline
uv run python .claude/skills/find-baseline/scripts/query_baseline.py --task unified --type all --baseline-only

# 查看 run 详情 (逐被试)
uv run python .claude/skills/find-baseline/scripts/query_baseline.py --tag 20260323_2237
```

### 使用 Python API

```python
from src.results.experiment_db import ExperimentDB

db = ExperimentDB()

# 查找 baseline (优先显式标记，fallback 启发式)
r = db.find_baseline_run('imagery', 'binary', 'cbramod', 'within_subject')
print(r['run_tag'], r['baseline_source'])  # 20260323_2237 explicit

# 手动标记/取消 baseline
db.set_baseline('20260323_2237_within_subject_imagery_binary', is_baseline=True)

# 创建新 run 时标记为 baseline
db.create_run(run_tag, 'within_subject', 'imagery', 'binary', is_baseline=True)
```

### 使用 CLI `--baseline` flag

```bash
# 标记新运行为 baseline
uv run python scripts/experiments/run_within_subject_comparison.py --models cbramod --task binary --baseline --cache-only
```

## 更新历史

| 日期 | 变更 |
|------|------|
| 2026-03-23 | Schema v6: 引入 `is_baseline` 列，回填 6 个 baseline runs |
| 2026-03-24 | 脚本统一重构验证: 替换 CBraMod binary within/cross baseline (`20260321_0343` → `20260323_2237`, `20260321_0608` → `20260324_0023`); 新设 CBraMod ternary within/cross 首个 baseline (`20260323_2320`, `20260324_0109`) |
