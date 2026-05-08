# Baseline Registry

ExperimentDB Schema v6 引入 `is_baseline` 列，显式标记每个类别 (model + task + experiment_type) 的标准参考运行。本文档记录当前所有 baseline 及缺失类别。

> **数据来源**: ExperimentDB 查询 — `SELECT ... FROM runs r JOIN model_summaries ms WHERE r.is_baseline=1 AND r.paradigm='imagery' AND r.n_channels=128`

## 当前 Baseline

### CBraMod Binary (model selection 实验, 每类 4 runs)

| Experiment Type | run_tag | Mean Acc | Std | N | 选择依据 |
|----------------|---------|----------|-----|---|---------|
| within_subject | `20260323_2237` | 85.15% | 11.00% | 21 | 脚本统一重构后全量验证 |
| cross_subject | `20260324_0023` | 90.68% | 9.31% | 21 | 脚本统一重构后全量验证 |
| transfer | `20260329_0507` | 90.12% | 8.98% | 21 | 纯 within-subject HPO config，替换旧 finetune defaults |

备注:
- within/cross binary baselines 于 2026-03-24 替换为脚本统一重构后的全量验证 run，性能持平（+0.06% / +0.14%）
- Transfer baseline 于 2026-03-29 替换: `20260321_1025` (90.18%, 旧 finetune config) → `20260329_0507` (90.12%, 纯 HPO config)。性能持平 (-0.06%)，但配置与 within-subject 统一
- 历史: `20260321_0343` (soup, 85.09%) → `20260323_2237` (85.15%), `20260321_0608` (90.54%) → `20260324_0023` (90.68%)

### CBraMod Ternary (首个 baseline)

| Experiment Type | run_tag | Mean Acc | Std | N | 选择依据 |
|----------------|---------|----------|-----|---|---------|
| within_subject | `20260323_2320` | 69.44% | 15.42% | 21 | 脚本统一重构后首次全量验证 |
| cross_subject | `20260324_0109` | 74.88% | 14.03% | 21 | 脚本统一重构后首次全量验证 |
| transfer | `20260329_0521` | 75.04% | 13.97% | 21 | 首个 ternary transfer baseline，纯 within-subject HPO config |

备注:
- 此前 ternary 无显式 baseline（runs 未标记 complete 或未执行全量 run）
- 本次为首个 128ch ternary baseline，与历史最佳持平（within: -0.10% vs `20260205_0306`, cross: -0.54% vs `20260207_2056`）
- Transfer baseline 于 2026-03-29 新设（此前 ternary transfer 无 baseline）。Ternary transfer (75.04%) ≈ cross-subject (74.88%)，fine-tuning 未能超越预训练 checkpoint

### EEGNet Binary (post-EEGNet-HPO)

| Experiment Type | run_tag | Mean Acc | Std | N | 选择依据 |
|----------------|---------|----------|-----|---|---------|
| within_subject | `20260316_1411` | 78.10% | — | 21 | 唯一 post-EEGNet-HPO v2.0 全量 run (1/1) |

备注:
- EEGNet HPO: 32 trials, best Trial #23 = 82.71% (F1=16, D=4 架构升级)
- HPO 最优参数: `results/hpo/eegnet_within_subject_binary_best_params.json`
- EEGNet cross_subject 和 transfer 无 standalone post-HPO run

### EEGNet Ternary

| Experiment Type | run_tag | Mean Acc | Std | N | 选择依据 |
|----------------|---------|----------|-----|---|---------|
| within_subject | `20260329_0056` | 66.81% | 14.50% | 21 | 首个 21 被试全量 run，使用 binary HPO 默认参数 |

### CBraMod Quaternary (首个 baseline)

| Experiment Type | run_tag | Mean Acc | Std | val_acc | val→test gap | N | 选择依据 |
|----------------|---------|----------|-----|---------|--------------|---|---------|
| within_subject | `20260508_1518` | 40.69% | 11.42% | 35.90% | +4.80 pp | 21 | 修复数据泄露后首个 21 被试全量 run |
| cross_subject | `20260508_1221` | 46.30% | 8.86% | 38.29% | +8.01 pp | 21 | 同上（同 run 含 eegnet + cbramod 双 model_summaries）|
| transfer | `20260508_1611` | 45.29% | 12.15% | 38.99% | +6.30 pp | 21 | 同上（pretrained = `20260508_1221_cbramod_imagery_quaternary/best.pt`）|

### EEGNet Quaternary (首个 baseline)

| Experiment Type | run_tag | Mean Acc | Std | val_acc | val→test gap | N | 选择依据 |
|----------------|---------|----------|-----|---------|--------------|---|---------|
| within_subject | `20260508_1538` | 47.81% | 11.35% | 40.80% | +7.02 pp | 21 | 修复数据泄露后首个 21 被试全量 run |
| cross_subject | `20260508_1221` | 43.65% | 7.62% | 37.17% | +6.48 pp | 21 | 同上（与 cbramod 共享 run）|
| transfer | `20260508_1611` | 47.57% | 10.60% | 41.49% | +6.08 pp | 21 | 同上（pretrained = `20260508_1221_eegnet_imagery_quaternary/best.pt`）|

备注:
- **协议差异**：quaternary 仅 OfflineImagery 一个 session（无 Online 4-class 数据），train/val/test 来自同 session 时序 70/15/15 切分；与 binary/ternary 的 Offline+Online_Base→Online_Finetune 协议不直接可比
- **泄露修复后**：所有 6 个 baseline 的 val/test gap 集中在 +4.8 ~ +8.0 pp，与 binary/ternary 的 −24 ~ −27 pp session-shift gap 量级一致——证实修复后的 holdout 即"同分布抽样"
- **跨任务难度对照**：cross-subject CBraMod 的 Acc/Chance 比 binary 1.81× / ternary 2.25× / quaternary 1.85×——quaternary 在该数据集上回落到 binary 量级而非高于 ternary（修复前曾达 3.25× 异常值）
- **模型相对差异范式依赖**：cross 下 CBraMod 微胜 EEGNet (+2.65 pp)；within / transfer 下 EEGNet 反而微胜 CBraMod (−7.12 / −2.28 pp)——首次出现 EEGNet 持平或胜过 CBraMod 的范式

备注:
- 使用当前默认参数（= binary HPO 最优: F1=16, D=4, lr=4e-3），非 ternary HPO 参数
- Ternary HPO 已完成 (31 trials, best=65.58%)，最优参数尚未应用为默认值
- 同 run 的 CBraMod: 69.80% ± 14.18%，差异不显著 (p=0.197)

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
| eegnet | ternary | cross/transfer | 无 standalone run | 需运行 cross_subject comparison + transfer |

## 查询方法

### 使用 `analyze-run` skill 与其 `find-baseline` 子技能

```bash
# 查看所有 baseline
uv run python .agents/skills/analyze-run/subskills/find-baseline/scripts/query_baseline.py --baseline-only --type all

# 查看特定类别
uv run python .agents/skills/analyze-run/subskills/find-baseline/scripts/query_baseline.py --model cbramod --task binary --type within_subject

# 查看 unified baseline
uv run python .agents/skills/analyze-run/subskills/find-baseline/scripts/query_baseline.py --task unified --type all --baseline-only

# 查看单次 run 摘要
uv run python scripts/tools/describe_run.py 0329_1357

# 查看 run 详情 (baseline 子技能的逐被试模式)
uv run python .agents/skills/analyze-run/subskills/find-baseline/scripts/query_baseline.py --tag 20260323_2237
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
| 2026-03-29 | 新设 EEGNet ternary within_subject baseline (`20260329_0056`, 66.81%, n=21)；使用 binary HPO 默认参数 |
| 2026-03-29 | Transfer baseline 更新: 替换 CBraMod binary transfer (`20260321_1025` 90.18% → `20260329_0507` 90.12%, 纯 HPO config); 新设 CBraMod ternary transfer 首个 baseline (`20260329_0521`, 75.04%); 移除 `get_default_finetune_config()` 自动覆盖，transfer 统一使用 within-subject HPO 默认值 |
| 2026-05-05 | 新设 Quaternary cross_subject 首个 baseline (`20260505_0002`): eegnet 48.99% + cbramod 81.23%, n=21; 同一 run 双 model_summaries 同时标记 |
| 2026-05-08 | **撤回** Quaternary cross_subject baseline (`20260505_0002`)：发现 train/val/test 分割数据泄露污染。run 级 + 双 model_summaries 的 is_baseline 均设回 0，notes 字段标记 WITHDRAWN；quaternary 全部类别再次无 baseline，等待按主线 binary/ternary 协议重跑 |
| 2026-05-08 | 完成 quaternary 全 6 个新运行（cross `20260508_1221` / within cbramod `20260508_1518` / within eegnet `20260508_1538` / transfer `20260508_1611`），数据已记入 ExperimentDB；当前**未自动标记 baseline**（按 CLAUDE.md "Agent 不得自行决定替换/新增 baseline" 规范），等待开发者明确确认后再注册 |
| 2026-05-08 | 开发者确认后注册 Quaternary 全范式 6 个首个 baseline：CBraMod within `20260508_1518` (40.69%) / cross `20260508_1221` (46.30%) / transfer `20260508_1611` (45.29%)；EEGNet within `20260508_1538` (47.81%) / cross `20260508_1221` (43.65%) / transfer `20260508_1611` (47.57%)。同时清理被撤回的 `20260505_0002` 及其衍生工件（DB 行 + JSON + checkpoint），见 commit `907f8a6` |
