# 剩余实验 Handoff — 2026-05-05

> **Audience**: 接管夜间实验收尾的长任务监督 agent
> **Origin**: 由主对话生成；继承 `2026-05-04_overnight_experiments.md` 的剩余/延后任务
> **前置事实**：2026-05-04/05 夜间已完成 Tasks A/B/C(32ch only)/D-train，详见
> [2026-05-04_overnight_results.md](2026-05-04_overnight_results.md)。本 handoff
> 只覆盖**尚未完成**的部分。
>
> **时间预算**：基于 2026-05-04 实测，原 plan 预估高估 3–5×。本 handoff 时间
> 估计已对齐实测（一个 cross-subject CBraMod ≈ 30–40 min，binary+ternary
> 串内 ≈ 60–80 min）。

---

## 0. Mission Summary

夜间报告（[2026-05-04_overnight_results.md](2026-05-04_overnight_results.md)）
明确了三类延后/未跑的实验：

| Task | 状态 | 来源 |
|------|------|------|
| Task C-2 — 8ch Band Power transfer | plan §3.2 授权延后；**已转移**至 [paper_review_experiments §3.4](2026-05-05_paper_review_experiments.md) 作为加分项 | 原 handoff §3.2 |
| Task D-2 — V3 4-condition 下游评估 | flag plumbing 选择待定；**已被** 2026-05-05 上午 V3 continue plan 接管 | 原 handoff §5.2 |
| Task D-2b — V2 4-condition 下游评估（apples-to-apples） | 用户已确认 5.5 跳过（dev log） | 原 handoff §5.5 |

> **本 handoff 的实际可执行 task**：均已被后续 plan 接管或确认 skip。
> 论证完整性的 paper-review 补全实验（E-1/E-2/E-3）见独立文件
> [2026-05-05_paper_review_experiments.md](2026-05-05_paper_review_experiments.md)，
> 由第三批长任务 agent 在 V3 evaluation 完成后接力执行。

**核心约束（不变）**：
1. 必须用 `/long-run` skill 启动后台任务（因 `bash` 工具默认 10 分钟 timeout）
2. 启动前 PID baseline snapshot，绝不杀死非自己启动的 Python 进程
3. autonomous bug fixing OK；但 **flag plumbing 选择必须问用户**（见 §2 决策点）
4. 不 commit、不 push、不动 `paper/drafts/paper_draft_v3.md`

---

## 1. Pre-flight 检查清单

### 1.1 工作目录与环境

```bash
cd c:/Users/zhang/Desktop/github/EEG-BCI

# 跳过 uv sync（按 §2.2 of overnight_results.md 教训：会卸载未声明的 torch + lmdb）
# 改为验证依赖：
uv pip list | grep -iE "torch|lmdb"
nvidia-smi
```
- ✅ Pass 条件：`torch` 与 `lmdb` 都存在；GPU free ≥ 10 GB

### 1.2 PID baseline snapshot（PowerShell 版）

```powershell
$ts = (Get-Date).ToString("yyyyMMddHHmm")
Get-Process python -ErrorAction SilentlyContinue |
  Select-Object Id, StartTime, ProcessName, CommandLine |
  Out-File "$env:TEMP\baseline_pids_$ts.txt"
echo "Saved PID baseline to: $env:TEMP\baseline_pids_$ts.txt"
```
- 后续 cleanup 只 kill **不在**这份 baseline 里的 PID

### 1.3 V3 checkpoint 与 V2 checkpoint 完整性

```bash
# V3（夜间训练完成）
ls checkpoints/cbramod/further_pretrain_v3_20260505_0223/best_model.pth
# 期望大小：18.85 MB（19,769,700 bytes）

# V2（生产已用）
ls checkpoints/cbramod/further_pretrain_20260323_0609/best_model.pth
```

### 1.4 8ch Band Power 通道选择确认

```bash
cat results/8_channel/channel_selections.json | head -30
```
- ✅ Pass 条件：JSON 中含 `band_power` key 且 channel list 非空

---

## 2. 决策点（启动前必须解决）

### 2.1 V3/V2 下游评估的 weights 注入路径

**背景**：现有 `run_cross_subject_comparison.py` / `run_within_subject_comparison.py`
**不直接接受**`--pretrained-weights` flag，而是从 `config['model']['pretrained_path']`
读取（见 [src/training/train_cross_subject.py:209-214](../../src/training/train_cross_subject.py#L209-L214)
与 [src/training/train_within_subject.py:609-612](../../src/training/train_within_subject.py#L609-L612)）。

**推荐方案 (b) — 写新 YAML 配置**：

创建 4 个 config 文件（V3/V2 × within/cross），每个含 `model.pretrained_path`
指向相应 checkpoint。其余参数复用 HPO 后的 baseline 默认（来自 ExperimentDB
对应的 baseline run）。**这条路径无需改 Python 代码**。

模板（V3 cross-subject）：

```yaml
# configs/cbramod_v3_cross.yaml
model:
  classifier_type: two_layer
  dropout_rate: 0.37
  freeze_backbone: false
  pretrained_path: checkpoints/cbramod/further_pretrain_v3_20260505_0223/best_model.pth

training:
  scheduler: cosine_annealing_warmup_decay
  epochs: 500
  batch_size: 256
  backbone_lr: 1.3e-4
  classifier_lr: 2.2e-4
  weight_decay: 0.13
  label_smoothing: 0.05
  gradient_clip: 1.4

scheduler_config:
  phase_epochs: 10
  phase_decay: 0.50
  lr_ramp_ratio: 0.1
  eta_min: 1.0e-6
  exploration_epochs: 3
  exploration_batch_size: 128
```

V2 配置同上但 `pretrained_path: checkpoints/cbramod/further_pretrain_20260323_0609/best_model.pth`。
Within-subject 配置使用 paper Table 3 第二列的参数（backbone_lr=2.9e-4 等）。

**备选方案 (a) — flag 透传**：在 `add_shared_train_config_args` 加
`--pretrained-weights PATH`，然后在 [src/training/train_cross_subject.py:211](../../src/training/train_cross_subject.py#L211)
的 elif 分支前加优先级最高的 args 检查。比 (b) 增加 ~10 行代码但更通用。

**选 (b)**——除非用户在主对话明确指示选 (a)。理由：(b) 不动 Python，
回滚干净，且 V2/V3 evaluation 是一次性需求。

### 2.2 是否补 V2 4-condition baseline

ExperimentDB 中现有的 V2 baseline 数字（paper draft Table 16）来源未在 DB
中确认（夜间报告 §5.5）。**两种处理**：

- **路径 X (推荐)**：跑 V2 4-condition 与 V3 4-condition 在同一晚同一管线下，
  对照 V3 vs V2 直接保证 apples-to-apples——**额外 GPU ≈ 60-80 min**。
- **路径 Y**：信赖 paper Table 16 中现有 V2 数字 → 只跑 V3 4-condition（V3
  vs Baseline 严格、V3 vs V2 间接）——节省 ≈ 60-80 min。

**默认选 X**——夜间报告明确点出 baseline 缺口。如果完成时间紧张可降级到 Y。

---

## 3. Tasks 详细规范

### Task C-2 — 8ch Band Power transfer

**目标**：给 paper EDIT T1.1 提供"低密度通道下 transfer 提供显著增益"的第二档证据
（32ch 已得 +0.74 pp 正向；预期 8ch 增益更大）。

**预估时间**：~30 min（21 subject × CBraMod fine-tune；transfer runner
单 model）

**启动命令**：

```bash
uv run python scripts/run_transfer_comparison.py \
    --task binary --paradigm imagery \
    --channels 8 --channel-config band_power \
    --models cbramod \
    --pretrained-cbramod checkpoints/cross_subject/20260331_1950_cbramod_imagery_binary/best.pt \
    --cache-only --no-wandb --no-plot
```

> **--pretrained-cbramod 路径校验**：上述 cross-subject 8ch Band Power best.pt
> 应来自夜间报告引用的 `results/8_channel/band_power/20260331_1950` run。
> 启动前先 `ls checkpoints/cross_subject/ | grep band_power` 找到精确目录名，
> 若略有差异以实际为准。

**通过准则**：
- ✅ N=21 完整运行
- ✅ |Δ vs 84.05% (8ch BP cross baseline)| ≤ 15 pp，方向应为正（+1 至 +5 pp 范围内）

**预期数据文件**：`results/8_channel/band_power/<TS>_transfer_cache_imagery_binary.json`

---

### Task D-2 — V3 4-condition 下游评估

**目标**：闭环 EDIT T2.2——给出 V3（Stieger 30%）下游 4 单元格 vs Baseline
（TUEG）vs V2 的对比。

**预估时间**：~60-80 min（4 个 fine-tune × cross 与 within 内部协议）

**配置文件**（按 §2.1 推荐方案 (b) 创建）：

```bash
# 在 configs/ 下创建 4 个文件
configs/cbramod_v3_within.yaml      # pretrained_path: V3 best_model.pth
configs/cbramod_v3_cross.yaml       # pretrained_path: V3 best_model.pth
configs/cbramod_v2_within.yaml      # pretrained_path: V2 best_model.pth (路径 X)
configs/cbramod_v2_cross.yaml       # pretrained_path: V2 best_model.pth (路径 X)
```

参数取自 [paper/drafts/paper_draft_v3.md](../../paper/drafts/paper_draft_v3.md)
Table 3 与 [Table S5b](../../paper/drafts/paper_draft_v3.md#table-s5b-hpo-超参数变化对照)：

| Field | within | cross |
|-------|--------|-------|
| backbone_lr | 2.9e-4 | 1.3e-4 |
| classifier_lr | 4×backbone = 11.6e-4 | 1.7×backbone = 2.21e-4 |
| weight_decay | 0.026 | 0.13 |
| dropout_rate | 0.10 | 0.37 |
| batch_size | 256 | 256 |
| gradient_clip | 0.73 | 1.4 |
| label_smoothing | 0.05 | 0.05 |
| phase_decay | 0.47 | 0.50 |
| phase_epochs | 8 | 10 |
| exploration_epochs | 4 | 3 |
| exploration_batch_size | 64 | 128 |
| scheduler | cosine_annealing_warmup_decay | cosine_annealing_warmup_decay |

**启动顺序（串行）**：

```bash
# 1. V3 cross binary (~25 min)
uv run python scripts/run_cross_subject_comparison.py \
    --task binary --paradigm imagery \
    --models cbramod \
    --config configs/cbramod_v3_cross.yaml \
    --cache-only --no-wandb --no-plot \
    --run-tag-suffix v3_eval

# 2. V3 cross ternary (~25 min)
uv run python scripts/run_cross_subject_comparison.py \
    --task ternary --paradigm imagery \
    --models cbramod \
    --config configs/cbramod_v3_cross.yaml \
    --cache-only --no-wandb --no-plot \
    --run-tag-suffix v3_eval

# 3. V3 within binary (~15-20 min, smaller per-subject)
uv run python scripts/run_within_subject_comparison.py \
    --task binary --paradigm imagery \
    --models cbramod \
    --config configs/cbramod_v3_within.yaml \
    --cache-only --no-wandb --no-plot \
    --run-tag-suffix v3_eval

# 4. V3 within ternary (~15-20 min)
uv run python scripts/run_within_subject_comparison.py \
    --task ternary --paradigm imagery \
    --models cbramod \
    --config configs/cbramod_v3_within.yaml \
    --cache-only --no-wandb --no-plot \
    --run-tag-suffix v3_eval
```

> **关于 `--run-tag-suffix`**：若 runner 不支持该 flag，直接省略；run_tag 由
> 时间戳决定，不会冲突。事后用 `db.find_runs(...)` 按时间窗 + config 名筛即可
> 区分 V3 vs Baseline。

**通过准则**：
- ✅ 4 个 run 全部 N=21 完成
- ✅ V3 cross binary 与 baseline `20260321_0608` (90.54%) 相差 |Δ| ≤ 5 pp（无论方向）
- ✅ V3 within binary 与 baseline `20260321_0343` (85.09%) 相差 |Δ| ≤ 5 pp
- 数据决定 paper EDIT T2.2 的措辞——**结果填入由主对话**。

**预期数据文件**：`results/<TS>_cross_subject_cache_imagery_{binary,ternary}.json`
与 `results/<TS>_comparison_cache_imagery_{binary,ternary}.json`

---

### Task D-2b — V2 4-condition 下游评估（路径 X）

**目标**：补 V2 在 N=21 完整 cohort 下的 4 单元格下游记录到 ExperimentDB，
作为 V3 比较的精确 apples-to-apples baseline。

**预估时间**：~60-80 min（与 Task D-2 完全对称）

**启动命令**：与 D-2 相同，仅 `--config configs/cbramod_v2_*.yaml` 替换。

**通过准则**：
- ✅ 4 个 run 全部 N=21 完成
- ✅ V2 cross binary 与 paper Table 16 中 89.43% 相差 |Δ| ≤ 3 pp（验证现有
  数字可信）；若差距 >3 pp，需主对话审 paper Table 16 来源

---

## 4. Reporting 模板

任务全部结束后，请在 `docs/handoffs/2026-05-05_remaining_results.md` 写入
（沿用 [2026-05-04_overnight_results.md](2026-05-04_overnight_results.md) 格式）：

```markdown
# 2026-05-05 剩余实验执行报告

## 0. 总览
| Task | 状态 | run_tag | 关键数字 |
| C-2 8ch BP transfer | ✅/⏸/❌ | <TS> | acc / Δ |
| D-2 V3 cross binary | ... | ... | ... |
| D-2 V3 cross ternary | ... | ... | ... |
| D-2 V3 within binary | ... | ... | ... |
| D-2 V3 within ternary | ... | ... | ... |
| D-2b V2 cross binary | ... | ... | ... |
| D-2b V2 cross ternary | ... | ... | ... |
| D-2b V2 within binary | ... | ... | ... |
| D-2b V2 within ternary | ... | ... | ... |

## 1. 配置创建记录
列出新增的 4 个 yaml 文件路径

## 2. Tasks 数字
对每个 task 给完整 mean ± SD，并填入与 baseline 的 Δ

## 3. V3 vs V2 vs Baseline 对照表
| 范式 | 任务 | Baseline | V2 | V3 | Δ(V3 vs BL) | Δ(V3 vs V2) |
| 被试内 | binary | 85.09% | 82.23%* | <填> | <填> | <填> |
| ... |

> *V2 within binary 现有 paper 数字；本次实测 N=21 重跑结果是 <填>，
>  与现有 82.23% 相差 <填> pp。

## 4. 待办
- [ ] paper EDIT T2.2 数字填入
- [ ] V3 配置/checkpoint 是否注册为新 baseline（需用户决策）
```

---

## 5. 执行完整顺序

```text
Pre-flight (5 min)
   ↓
Task C-2 8ch BP transfer (~30 min)
   ↓
[决策点 §2] 创建 4 个 config yaml
   ↓
Task D-2 V3 cross binary (~25 min)   ─┐
Task D-2 V3 cross ternary (~25 min)  │  全程串行（GPU 串行）
Task D-2 V3 within binary (~20 min)  │
Task D-2 V3 within ternary (~20 min) ─┘
   ↓
[路径 X 决策] 跑 V2 4-condition 还是停？
   ↓ (路径 X)
Task D-2b V2 (~80 min)
   ↓
Reporting markdown 写入
```

总预估：**~3.5 hr 路径 X / ~2 hr 路径 Y**（远低于原 plan §5.2 估的 8-12 hr）

---

## 6. 红线（不变）

- ❌ 不 commit / 不 push
- ❌ 不修改 [paper/drafts/paper_draft_v3.md](../../paper/drafts/paper_draft_v3.md)
- ❌ 不 kill baseline PID 列表中任何 python 进程
- ❌ 不修改 channel selection JSON / HPO 默认参数
- ❌ 不为加速禁用 ±500 µV trial 剔除
- ❌ 不修改 `further_pretrain_v3_20260505_0223/` 目录下任何文件（V3 weights 是 read-only artifact）

---

## 7. 故障处理

| 故障 | 处理 |
|------|------|
| `train_cross_subject.py` 在 V3/V2 weights 上报 state_dict 不匹配 | 检查 V2 baseline `further_pretrain_20260323_0609/best_model.pth` 是否能成功加载（已知可用）；若 V3 的 layer name 不一致，对比两份 state_dict.keys() 找出差异，用 `--strict false` 等效项跳过缺失键，但**不要静默删除任何 layer**；汇报给主对话 |
| `--pretrained-cbramod` 路径错误（best.pt 找不到） | 用 `Glob "checkpoints/cross_subject/*band_power*/best.pt"` 找精确路径，或从 ExperimentDB 查 `db.find_runs(channel_config='band_power', channels=8, experiment_type='cross_subject')` 拿 model_path |
| GPU OOM | 把对应 config 的 `batch_size` 降到 128；不动 backbone_lr/classifier_lr |
| 任意 task 卡住 >40 min 无 log | Monitor stagnation 触发；记录最后 10 行 log 后 kill 自己启动的 PID 并报告，**不重试**——汇报给主对话 |
| YAML 解析错（缩进、字符） | 用 [configs/cbramod_cawd_new.yaml](../../configs/cbramod_cawd_new.yaml) 作模板对照 |

---

## 8. 关键文件引用

- 夜间结果：[docs/handoffs/2026-05-04_overnight_results.md](2026-05-04_overnight_results.md)
- 原 handoff：[docs/handoffs/2026-05-04_overnight_experiments.md](2026-05-04_overnight_experiments.md)
- Plan 文件（参考用，不要修改）：`C:\Users\zhang\.claude\plans\dazzling-drifting-candy.md`
- Paper EDIT 参考点：T1.1（reduced-channel transfer）、T2.2（DAPT V3）、
  T2.4（§4.8 synthesis）——见 plan 文件
- Pretrained path 注入逻辑：
  - [src/training/train_cross_subject.py:207-214](../../src/training/train_cross_subject.py#L207-L214)
  - [src/training/train_within_subject.py:608-612](../../src/training/train_within_subject.py#L608-L612)
- CBraMod adapter `_load_pretrained`：
  [src/models/cbramod_adapter.py:435-448](../../src/models/cbramod_adapter.py#L435-L448)

---

## 10. Paper-review 新增需求（已转移）

E-1 / E-2 / E-3 三组实验的完整规范已**转移**到独立 handoff 文件：
[2026-05-05_paper_review_experiments.md](2026-05-05_paper_review_experiments.md)。
该文件包含 wait gate（必须等当前 V3 evaluation agent 完成后再启动）、
真实可调用的 channel selection 脚本（`scripts/analysis/compute_channel_selections.py`，
而非这里早期版本的伪 API），以及自主执行所需的全部决策点。
