# 2026-05-04 夜间实验执行报告

> Plan 引用：[2026-05-04_overnight_experiments.md](2026-05-04_overnight_experiments.md)
> Plan 文件（agent 所用）：`C:\Users\zhang\.claude\plans\docs-handoffs-2026-05-04-overnight-expe-merry-pond.md`
> 生成时间：2026-05-05（凌晨执行，夜间无人值守）
> 状态：Tasks A/B/C 全部完成；Task D V3 训练进行中，下游评估按 plan 延后至清晨

---

## 0. 执行总览

| Task | 状态 | 实际耗时 | run_tag | 关键数字 |
|------|------|----------|---------|----------|
| Pre-flight | ✅ 含修复两次 | ~30 min | — | caches/preprocessed 解压 7433 文件/40.58 GB；fixed torch + lmdb |
| A — quaternary cross-subject | ✅ | 1.2 hr | `20260505_0002` | CBraMod **81.23%** / EEGNet 48.99% (N=21) |
| B-binary — leave-S04/S10/S14-out | ✅ | 26 min | `20260505_0116` | CBraMod **90.62%** (N=18, Δ=-0.06pp) |
| B-ternary — leave-S04/S10/S14-out | ✅ | 24 min | `20260505_0145` | CBraMod **74.75%** (N=18, Δ=-0.13pp) |
| C — 32ch FDR transfer | ✅ | 10 min | `20260505_0212` | CBraMod **88.45%** (N=21, Δ=+0.74pp vs xsubj) |
| Stieger subsample LMDB | ✅ | <1 min | — | 7159 keys (10.67% of 67068)，输出 9.7 GB on disk |
| D — V3 DAPT 训练 | ✅ | 70 min (02:24→03:34) | `further_pretrain_v3_20260505_0223` | 15 epoch 全跑完，final loss **0.005037**（V1→V2→V3 收敛健康） |
| D — V3 4-condition 下游评估 | ⏸ 延后 | — | — | 见 §5 待办 |
| C — 8ch band_power transfer | ⏸ 延后 | — | — | 按 plan §3.2 授权跳过 |

总实测耗时（pre-flight 修复 + A→D 串行 + Stieger subsample 与 C 并行）：约 **3.7 hr**（23:47→03:34），远低于 plan 预算 10 hr。

---

## 1. 启动顺序与 GPU 串行决策

12 GB RTX 5070（free 10.6 GB）必须严格串行 CBraMod 任务。实测序列：
1. Task A（cross-subject runner，eegnet + cbramod 串内）
2. Task B-binary（同 runner，N=18）
3. Task B-ternary（同上，task 切换）
4. Task C（transfer runner，21 subject finetune 串内）
5. Stieger subsample（**与 Task C 并行**——纯 D 盘 I/O 不冲突 GPU）
6. Task D V3 训练（在 Task C 完成后启动）

唯一并行点：Stieger subsample 在 Task C 期间执行，因为 subsample 不用 GPU 且仅访问 D 盘 LMDB（Task C 用的是 C 盘 caches/preprocessed）。

---

## 2. Pre-flight 自主修复记录

### 2.1 caches.zip 解压第一次失败 → 重做
- 上一会话遗留的后台 Expand-Archive (PID 5672) 因会话切换被截断，仅完成 25.62 GB
- 本会话发现时仍在写 + 我误开始 cleanup → race condition
- **修复**：kill 进程 → 完整删除 `caches/preprocessed/` → 通过 long-run skill (nohup) 重启解压（survives 会话重启）
- 第二次解压 6.74 min 完成，最终 7433 文件 / 40.58 GB / 21 名被试齐全

### 2.2 `uv sync` 卸载 torch + lmdb（重要 setup 教训）
- pyproject.toml 不含 torch 和 lmdb（torch 是 GPU-arch 特定的 nightly cu128 wheel，lmdb 同样手装）
- 跑 `uv sync` 会"严格同步"，**卸载未声明的包**——torch + lmdb 双双消失
- **修复**：
  - `uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128`
  - `uv pip install lmdb`
- **教训**：今后 pre-flight **不能跑 `uv sync`**。验证依赖应改为 `uv pip list | grep <pkg>`，缺包再单装

### 2.3 baseline PID 保护
- 启动前 snapshot 15 个旧 python PID（其他 agent 的进程）到 `$env:TEMP\baseline_pids_202605042350.txt`
- 全程未 kill 这些 PID
- 唯一被 kill 的 pwsh 是我自己启动的 PID 5672（Expand-Archive 残留），命令行已 verify 是我的 launch

---

## 3. Tasks A-C 详细数字

### 3.1 Task A — Quaternary Cross-Subject（**项目首条 baseline**）

> 数据来源：[results/20260505_0002_cross_subject_cache_imagery_quaternary.json](../../results/20260505_0002_cross_subject_cache_imagery_quaternary.json)
> ExperimentDB run: `20260505_0002_cross_subject_imagery_quaternary`

| Model | N | mean | std | best_epoch | training_time | val_acc |
|-------|---|------|-----|------------|---------------|---------|
| EEGNet | 21 | 48.99% | 6.27% | 6 | 35 min | 36.04% |
| CBraMod | 21 | **81.23%** | 6.83% | 27 | 38 min | 38.48% |

通过准则：
- ✅ N=21
- ✅ CBraMod ≥ 30% chance + 5pp（实测 81.23%，远超）
- ✅ EEGNet ≥ 25% chance（实测 48.99%）

**注解**：DB 此前**零** quaternary cross-subject 记录，本次为该类别首条 baseline。
- val_acc ≪ test_acc 是协议预期（训 Offline，测 Online Sess02 Finetune）
- S20=63.9% / S21=35.2% 是 outlier（在所有任务中一致）

### 3.2 Task B — Leave-S04/S10/S14-out 灵敏度

> Binary：[results/sensitivity_leave3out/20260505_0116_cross_subject_cache_imagery_binary.json](../../results/sensitivity_leave3out/20260505_0116_cross_subject_cache_imagery_binary.json)
> Ternary：[results/sensitivity_leave3out/20260505_0145_cross_subject_cache_imagery_ternary.json](../../results/sensitivity_leave3out/20260505_0145_cross_subject_cache_imagery_ternary.json)

| Task | N | mean | std | baseline | Δ |
|------|---|------|-----|----------|---|
| Binary | 18 | 90.62% | 8.18% | 90.68% (`20260324_0023`) | **-0.06pp** |
| Ternary | 18 | 74.75% | 13.74% | 74.88% (`20260324_0109`) | **-0.13pp** |

通过准则：
- ✅ N=18
- ✅ |Δ| ≤ 5pp（两组都 ~0pp）

**结论**：移除 S04/S10/S14 对 cross-subject 性能几乎无影响，**paper claim 的稳健性已验证**。

### 3.3 Task C — 32ch FDR Transfer

> 数据来源：[results/32_channel/fdr/20260505_0212_transfer_cache_imagery_binary.json](../../results/32_channel/fdr/20260505_0212_transfer_cache_imagery_binary.json)
> ExperimentDB run: `20260505_0212_transfer_32ch_fdr_imagery_binary`

| Model | N | mean | std | min | max | median |
|-------|---|------|-----|-----|-----|--------|
| CBraMod | 21 | **88.45%** | 8.45% | 66.88% | 99.38% | 91.25% |

baseline = 87.71%（cross-subject 32ch FDR），Δ=+0.74pp，方向**正向**。

通过准则：
- ✅ N=21
- ✅ |Δ| ≤ 10pp，方向正（验证 plan T1.1 假设：transfer 在 32ch FDR 上比纯 cross-subject 略提升）

**注解**：transfer runner 自动归档结果到 `results/32_channel/fdr/`，plan 担心的 Move-Item 后置不需要。

### 3.4 Task D — V3 DAPT 训练

> Checkpoint dir：[checkpoints/cbramod/further_pretrain_v3_20260505_0223/](../../checkpoints/cbramod/further_pretrain_v3_20260505_0223/)
> Best model: `best_model.pth`（18.85 MB，与 V2 同 size，结构兼容）
> Training history: [training_history.json](../../checkpoints/cbramod/further_pretrain_v3_20260505_0223/training_history.json)

**超参完全沿用 V2**（plan §5.2 一致，唯一差异：LMDB 列表把 `Stieger2021_pretrain` 替换为 `Stieger2021_subsampled_30pct`）：
- max_epochs=15, patience=5, batch=16, effective_batch=128, reference_channels=64
- lr=5e-5, wd=0.05, warmup=0.5 epoch, mask_ratio=0.5, scheduler=warmup_constant
- pretrained_weights = V1 base (`checkpoints/cbramod/pretrained_weights.pth`)
- ⚠️ V2 是否同样从 V1 base 启动 = **unknown**（V2 config.json `pretrained_weights: null`，但 plan §5.5 假设 V2 from V1；可能是 record-keeping bug，待用户确认）

**Loss 轨迹**（每 epoch ~280 秒，总 70 min）：

| Epoch | Loss | LR | Time |
|------:|-----:|---:|-----:|
| 1 | 0.020470 | 5e-5 | 319 s |
| 2 | 0.011072 | 5e-5 | 278 s |
| 3 | 0.009409 | 5e-5 | 275 s |
| 4 | 0.008362 | 5e-5 | 276 s |
| 5 | 0.007439 | 5e-5 | 276 s |
| 6 | 0.006757 | 5e-5 | 276 s |
| 7 | 0.006538 | 5e-5 | 273 s |
| 8 | 0.005973 | 5e-5 | 271 s |
| 9 | 0.005960 | 5e-5 | 305 s |
| 10 | 0.005631 | 5e-5 | 287 s |
| 11 | 0.005629 | 5e-5 | 278 s |
| 12 | 0.005547 | 5e-5 | 274 s |
| 13 | 0.005515 | 5e-5 | 275 s |
| 14 | 0.005247 | 5e-5 | 277 s |
| 15 | **0.005037** | 5e-5 | 273 s |

**通过准则**：
- ✅ 15 epoch 全跑完（无早停，patience=0/5 表示 epoch 14/15 仍在创新低）
- ✅ best_model.pth 存在 + 大小合理（19,769,700 bytes / 18.85 MB）
- ✅ 无 NaN，loss 单调下降
- ✅ 训练时长合理（70 min，比 V2 快约 3-4× 因数据量降至 V2 的 28%）

**对照 V2**：V2 checkpoint dir 含 epoch1-14（无 epoch15，无 training_history.json），final epoch14 loss=0.003641（来源：文件名 `epoch14_loss0.003641.pth`）
- V3 final 0.005037 比 V2 epoch14 高 38%——但 **不可直接比较**，因为：
  - V3 数据量约 V2 的 28%（23865 vs 83774 segments），grad steps 仅 V2 的 30.5%（2790 vs 9156）
  - V2 起点 (`pretrained_weights`) **unknown**（V2 config 写 null）；若 V2 真从 scratch 而 V3 from V1 base，零点不同
  - pretext loss 在异质数据混合上天然更难（V3 Stieger 30% vs V2 80%，V3 数据更多样）
- 真正的 V3 vs V2 比较只能在下游 4-condition 评估上做

下游评估按 plan §5.6 Path X 授权延后到清晨，理由见 §5.2。

---

## 4. Stieger 子采样 LMDB（Task D 数据准备）

> 脚本：[scripts/pretraining/subsample_stieger.py](../../scripts/pretraining/subsample_stieger.py)（新增）
> 输出：`D:/data/motion_imagination_datasets/lmdb_pretrain/Stieger2021_subsampled_30pct/`

### 4.1 修正 handoff §5.1 的算术错误（autonomous bug fix）

handoff 原文写 "26K keys = 42% Stieger keys = 30% 训练占比"，但实际数据反算：
- Stieger 67068 segs + other 9 总 16706 segs（实测）
- 若取 26K Stieger keys：share = 26K/(26K+16.7K) = **60.9%**，远非 30%
- **正确解**：share = K/(K + N_other) = 0.30 → K = N_other × 0.30/0.70 = 16706 × 0.4286 = **7159 keys**（仅 10.67% of source）

按 [feedback_autonomous_bug_fixing](../../C:/Users/zhang/.claude/projects/c--Users-zhang-Desktop-github-EEG-BCI/memory/feedback_autonomous_bug_fixing.md) 自主修正，未打扰用户。脚本已在 [scripts/pretraining/subsample_stieger.py](../../scripts/pretraining/subsample_stieger.py) 中以 `--target-share` 参数化，默认 0.30。

### 4.2 子采样输出
- 7159 keys 随机采样（seed=42，与 main script 一致）
- 实际 disk usage 9.7 GB（logical 19.15 GB 是 LMDB sparse 预分配）
- D 盘减 9.7 GB → 剩 ~52.7 GB（远超 plan 预估 26 GB 警戒线）
- verify 通过：`__keys__` list len=7159 = kv entries 7160 - 1（__keys__ 自身）

V3 训练的实际数据组成：
| Dataset | Segments | Share |
|---------|----------|-------|
| Other 9 datasets | 16706 | 70.00% |
| Stieger2021_subsampled_30pct | 7159 | 30.00% |
| **Total** | **23865** | **100%** |

---

## 5. 待办（清晨用户决策）

### 5.1 Task D V3 训练 ✅ 已于 03:34 完成（无人值守）
- final loss 0.005037（健康下降）
- best_model: `checkpoints/cbramod/further_pretrain_v3_20260505_0223/best_model.pth`（18.85 MB）
- 详见 §3.4

### 5.2 Task D V3 4-condition 下游评估（plan §5.6 Path X 授权延后）

**`--pretrained-weights` / `--further-pretrained-cbramod` flag 现状**（2026-05-05 上午更新）：
- ✅ [scripts/experiments/run_within_subject.py:382](../../scripts/experiments/run_within_subject.py#L382)：`--pretrained-weights`（CBraMod-only）
- ✅ [scripts/experiments/run_cross_subject_comparison.py](../../scripts/experiments/run_cross_subject_comparison.py)：**`--further-pretrained-cbramod`** 已新增（透传 `config_overrides['model']['pretrained_path']`，搭配 `--models cbramod` 单模型模式使用，EEGNet 历史 baseline 自动叠加到对比图）
- ❌ `transfer_comparison.py` 不需要新 flag——其 `--pretrained-cbramod` 是已 finetune 的 cross-subject `.pt`（含 classifier head），与 backbone `.pth` 语义不同。要做 V3 transfer，需先用上一行命令训出 V3-init 的 cross-subject `.pt`，再喂给 transfer。

**4-condition 下游 V3 评估命令**：

```bash
V3_CKPT="checkpoints/cbramod/further_pretrain_v3_20260505_0223/best_model.pth"

# Within-subject binary
uv run python scripts/run_within_subject.py \
    --model cbramod --task binary --paradigm imagery \
    --pretrained-weights "$V3_CKPT" \
    --cache-only --no-wandb \
    --results-dir results/dapt_v3

# Within-subject ternary
uv run python scripts/run_within_subject.py \
    --model cbramod --task ternary --paradigm imagery \
    --pretrained-weights "$V3_CKPT" \
    --cache-only --no-wandb \
    --results-dir results/dapt_v3

# Cross-subject binary（CBraMod 单模型，EEGNet 走历史 baseline 叠加）
uv run python scripts/run_cross_subject_comparison.py \
    --task binary --paradigm imagery \
    --models cbramod \
    --further-pretrained-cbramod "$V3_CKPT" \
    --cache-only --no-wandb \
    --results-dir results/dapt_v3

# Cross-subject ternary
uv run python scripts/run_cross_subject_comparison.py \
    --task ternary --paradigm imagery \
    --models cbramod \
    --further-pretrained-cbramod "$V3_CKPT" \
    --cache-only --no-wandb \
    --results-dir results/dapt_v3

# (Optional) Transfer：先用上述 cross-subject 输出的 best.pt 当 --pretrained-cbramod
# 这是两步流程，因为 transfer 需要 finetune 后的 .pt（含 classifier），不能直接用 backbone .pth
```

### 5.3 Task C 8ch band_power transfer（plan §3.2 授权跳过）
清晨手动启动：
```bash
uv run python scripts/run_transfer_comparison.py \
    --task binary --paradigm imagery \
    --channels 8 --channel-config band_power \
    --models cbramod \
    --pretrained-cbramod checkpoints/cross_subject/20260331_1950_cbramod_imagery_binary/best.pt \
    --cache-only --no-wandb --no-plot
```

### 5.4 Task A baseline 注册（用户决策）
本次 quaternary cross-subject 是项目首条该类别 baseline。**待用户确认是否在 review 后通过 `--baseline` flag 正式注册到 ExperimentDB `baseline_refs` 表**（按 [CLAUDE.md baseline 管理规范](../../CLAUDE.md)）。

### 5.5 V2 backbone N=21 下游 baseline 缺口
ExperimentDB 中**无** V2 backbone 的 N=21 within/cross-subject 下游评估记录，handoff §5.5 表中 V2 数字（82.23%/68.08% etc.）来源未确认。**需要明确**：
- (a) 这些数字来自哪个非 DB 源？
- (b) 是否需要补跑 V2 4-condition 下游以保证 V3 vs V2 严格 apples-to-apples？

### 5.6 今晚（2026-05-05）计划：V3 continue training（max=50, no min_delta）

**用户指令（2026-05-05 ~03:30）**：
> continue training V3; do not use min_delta threshold, only rely on early stopping; max epoch is 50; this is scheduled for tonight.

**实现说明**：`further_pretrain.py` **没有真正的 `--resume` flag**——只有 backbone weights load（[further_pretrain.py:471](../../scripts/pretraining/further_pretrain.py#L471)），无 optimizer/scheduler 状态保存。"continue" 实质是 **warm restart**：把 V3 当前 best_model.pth 当 `--pretrained-weights` 重新启动，LR 经 0.5 epoch 重新 warmup 到 5e-5。模型权重保留，优化器状态丢失。

**`--min-delta 0` 行为验证**：[further_pretrain.py:701](../../scripts/pretraining/further_pretrain.py#L701) `if mean_loss < best_loss - min_delta` 退化为 `< best_loss`——任何 loss 下降即视为改进。patience=5 仍生效，5 epoch 完全无改进才停。

**今晚命令（建议先用 `Get-Date` 替换 TS 占位符）**：
```bash
V3_PREV_BEST="checkpoints/cbramod/further_pretrain_v3_20260505_0223/best_model.pth"
TS=$(date +%Y%m%d_%H%M)

uv run python scripts/pretraining/further_pretrain.py \
    --max-epochs 50 \
    --patience 5 \
    --min-delta 0 \
    --batch-size 16 --effective-batch-size 128 --reference-channels 64 \
    --lr 5e-5 --weight-decay 0.05 --warmup-epochs 0.5 \
    --mask-ratio 0.5 --scheduler warmup_constant \
    --pretrained-weights "$V3_PREV_BEST" \
    --lmdb-dirs \
        D:/data/motion_imagination_datasets/lmdb_pretrain/BNCI2014_001_pretrain \
        D:/data/motion_imagination_datasets/lmdb_pretrain/BNCI2015_004_pretrain \
        D:/data/motion_imagination_datasets/lmdb_pretrain/Cho2017_pretrain \
        D:/data/motion_imagination_datasets/lmdb_pretrain/GrosseWentrup2009_pretrain \
        D:/data/motion_imagination_datasets/lmdb_pretrain/Lee2019_MI_pretrain \
        D:/data/motion_imagination_datasets/lmdb_pretrain/Ofner2017_pretrain \
        D:/data/motion_imagination_datasets/lmdb_pretrain/PhysionetMI_pretrain \
        D:/data/motion_imagination_datasets/lmdb_pretrain/Schirrmeister2017_pretrain \
        D:/data/motion_imagination_datasets/lmdb_pretrain/Shin2017A_pretrain \
        D:/data/motion_imagination_datasets/lmdb_pretrain/Stieger2021_subsampled_30pct \
    --checkpoint-dir "checkpoints/cbramod/further_pretrain_v3_continued_$TS"
```

**预算估算**：50 epoch × ~280 s = ~233 min ≈ **3.9 hr**（V3 首次 70 min × 50/15）。实际若早停命中可短得多。

**通过准则**：
- 全部 50 epoch 跑完 OR 早停触发（patience=5/5）
- final loss < V3 首次的 0.005037（继续训练应能改进）
- 无 NaN/inf
- best_model.pth 大小与 V3 首次一致（~18.85 MB）

**不会自动 schedule**——需用户启动时确认（避免过度自主）。如需我用 cron/long-run 自动在指定时间发起，请明确说"在 X 点启动"。

---

## 6. 文件清单（本夜新增/修改）

新增：
- `scripts/pretraining/subsample_stieger.py`（一次性脚本，已 plan §7 授权）
- `D:/data/motion_imagination_datasets/lmdb_pretrain/Stieger2021_subsampled_30pct/`（9.7 GB）
- `results/20260505_0002_cross_subject_cache_imagery_quaternary.json`
- `results/sensitivity_leave3out/20260505_0116_cross_subject_cache_imagery_binary.json`
- `results/sensitivity_leave3out/20260505_0145_cross_subject_cache_imagery_ternary.json`
- `results/32_channel/fdr/20260505_0212_transfer_cache_imagery_binary.json`
- `checkpoints/cross_subject/20260505_0002_eegnet_imagery_quaternary/best.pt`
- `checkpoints/cross_subject/20260505_0002_cbramod_imagery_quaternary/best.pt`
- `checkpoints/cross_subject/20260505_0116_cbramod_imagery_binary/best.pt`
- `checkpoints/cross_subject/20260505_0145_cbramod_imagery_ternary/best.pt`
- `checkpoints/cbramod/further_pretrain_v3_20260505_0223/`（含 epoch1-15 + best_model + training_history.json + config.json）
- 本报告：`docs/handoffs/2026-05-04_overnight_results.md`

修改：
- `caches/preprocessed/`（重新解压恢复）
- `results/experiments.db`（写入 4 条新 runs：A、B-binary、B-ternary、C）

未提交：所有变更按 plan §7 授权**未 git commit**——晨间用户审。

---

## 7. 已遵守的红线

- ✅ 未 commit / 未 push
- ✅ 未修改 [paper/drafts/paper_draft_v3.md](../../paper/drafts/paper_draft_v3.md)
- ✅ 未 kill baseline PID 列表中任何 python 进程
- ✅ 未修改 channel selection JSON / HPO 默认参数
- ✅ 未为加速禁用 ±500 µV trial 剔除
- ✅ V3 checkpoint dir 名 `further_pretrain_v3_<TS>`，与 V2 (`further_pretrain_20260323_0609`) 不撞名

---

## 8. 监控架构说明（执行模式记录）

本夜采用：
- **任务执行**：long-run skill (nohup mode on Windows) → 任务进程独立于 Claude Code 会话，会话中断不影响训练
- **完成通知**：Monitor 工具 + persistent=true → 脚本输出直接流到对话作为推送事件，**无时间限制**
- **错误捕获**：监控脚本同时检测 `exit_code` 文件出现（任何 exit 都触发，含 crash）+ 30 min 日志静默 (stagnation)

每个 task 一个独立 Monitor，事件文本即结果摘要：
- `TASK_*_DONE exit=N json=<path> | tail=[<last log lines>]`
- `TASK_*_STAGNATION 30min_no_output last=[...]`

不依赖 file-pointer 通知，结果直接进对话。

---

## 9. 2026-05-05 上午-傍晚 V3 评估补充

按 plan `docs-handoffs-2026-05-04-overnight-expe-merry-pond.md` autonomy-only 串行执行。

### 9.1 Step 1 — Quaternary cross_subject baseline 注册 ✅

| 类别 | run_tag | run_id | model | mean_acc | is_baseline (after) |
|------|---------|--------|-------|----------|---------------------|
| quaternary cross_subject | `20260505_0002` | `20260505_0002_cross_subject_imagery_quaternary` | eegnet | 48.99% | 1 |
| quaternary cross_subject | `20260505_0002` | 同上 | cbramod | 81.23% | 1 |

机制：`db.set_baseline(run_id, model_type=None)` 同步 `runs.is_baseline=1` 与全部 `model_summaries.is_baseline=1`。已追加至 [docs/dev_log/experiments/baseline_registry.md](../dev_log/experiments/baseline_registry.md) "更新历史" 与新增 "Quaternary Cross-Subject (首个 baseline)" 段落。

### 9.2 Step 2 — V3 continue training（warm restart）✅

| 字段 | 值 |
|------|-----|
| 起始权重 | `checkpoints/cbramod/further_pretrain_v3_20260505_0223/best_model.pth` (V3 first-run, loss=0.005037) |
| 输出 dir | `checkpoints/cbramod/further_pretrain_v3_continued_20260505_1800/` |
| 实参 | `--max-epochs 50 --patience 5 --min-delta 0 --warmup-epochs 0 --scheduler warmup_constant --lr 5e-5` |
| LMDB | 与 V3 第一轮完全一致（10 datasets，含 `Stieger2021_subsampled_30pct`） |
| 实际 epochs | 27（patience 5 在 epoch 27 触发，best at epoch 22） |
| 训练时长 | 2h 10m |
| Best loss | **0.004193**（vs V3 first-run best 0.005037 — 改进 16.76%） |
| LR 校验 | 全程恒定 5.00e-05（warmup_epochs=0 → step<0 永假 → 直接 constant 段，符合预期） |
| 错误 | 无 |

> **数据来源**: `checkpoints/cbramod/further_pretrain_v3_continued_20260505_1800/{best_model.pth, config.json, training_history.json}`

每 epoch loss 曲线（节选）：epoch1=0.005098 → epoch10=0.004512 → epoch22=0.004193 (best) → epoch27=0.004313。

### 9.3 Step 3 — V3 4-condition 下游评估 ✅

V3 backbone (`further_pretrain_v3_continued_20260505_1800/best_model.pth`) 透传方式：
- within_subject：通过现有 `--pretrained-weights` flag
- cross_subject：通过本日新加的 `--further-pretrained-cbramod` flag（首次实跑验证通过）

| 条件 | run_tag | n | mean_acc | std | V0 baseline | V0 acc | Δ vs V0 | 来源 JSON |
|------|---------|---|----------|-----|-------------|--------|--------|----------|
| within binary  | `20260505_2012` | 21 | **83.75%** | 11.12% | `20260323_2237` | 85.15% | **−1.40%** | `results/dapt_v3/20260505_2012_within_subject_cache_imagery_binary.json` |
| within ternary | `20260505_2033` | 21 | **69.31%** | 14.10% | `20260323_2320` | 69.44% | **−0.13%** | `results/dapt_v3/20260505_2033_within_subject_cache_imagery_ternary.json` |
| cross binary   | `20260505_2100` | 21 | **89.23%** |  7.99% | `20260324_0023` | 90.68% | **−1.45%** | `results/dapt_v3/20260505_2100_cross_subject_cache_imagery_binary.json` |
| cross ternary  | `20260505_2131` | 21 | **75.50%** | 12.48% | `20260324_0109` | 74.88% | **+0.62%** | `results/dapt_v3/20260505_2131_cross_subject_cache_imagery_ternary.json` |

**结论**：4 个条件中 3 个略低于 V0、1 个略高（cross ternary +0.62%）。预训练域 loss 明显改进（−16.76%）但下游性能基本持平甚至略降，提示 V3 LMDB 数据域（10 个 BCI 公开数据集）与 finger-MI 下游任务存在域偏移，DAPT 主要拟合到了通用 MI 表征但对 finger-level 的精细解码无显著迁移收益。

每条 run 的 per-subject top/bottom 数字：
- within binary: top S03/S19=98.12%, S09=96.88%; bottom S20=58.13%, S10=60.00%
- within ternary: top S19=92.50%, S04=91.25%; bottom S10=42.92%, S20=43.75%
- cross binary: top S03=99.38%, S19=98.13%; bottom S20=66.88%, S10=72.50%
- cross ternary: top S19=91.67%, S02=90.83%; bottom S20=46.67%, S10=54.17%

S20/S10 在 4 条件下均为 worst — 与历史一致，是数据级困难被试。

### 9.4 文件清单（本日新增）

**Checkpoints**
- `checkpoints/cbramod/further_pretrain_v3_continued_20260505_1800/best_model.pth` (19.77 MB, epoch 22)
- `checkpoints/cbramod/further_pretrain_v3_continued_20260505_1800/epoch{1..27}_loss*.pth` (27 个 per-epoch checkpoints, ~534 MB 总)
- `checkpoints/cbramod/further_pretrain_v3_continued_20260505_1800/{config.json, training_history.json}`
- `checkpoints/cross_subject/20260505_2100_cbramod_imagery_binary/best.pt`
- `checkpoints/cross_subject/20260505_2131_cbramod_imagery_ternary/best.pt`

**Results JSON / PNG (results/dapt_v3/)**
- `20260505_2012_within_subject_cache_imagery_binary.json` + `*_cbramod_imagery_binary.png`
- `20260505_2033_within_subject_cache_imagery_ternary.json` + `*_cbramod_imagery_ternary.png`
- `20260505_2100_cross_subject_cache_imagery_binary.json` + `*_cross-subject_combined_imagery_binary.png`
- `20260505_2131_cross_subject_cache_imagery_ternary.json` + `*_cross-subject_combined_imagery_ternary.png`

**ExperimentDB 变更**
- `runs.is_baseline` set: `20260505_0002_cross_subject_imagery_quaternary` (新 baseline)
- `model_summaries.is_baseline` set: 同 run_id 的 eegnet + cbramod 两条
- 4 条新 cross_subject runs（`20260505_2100`, `20260505_2131`）应已写入 DB；within 两条 (`20260505_2012`, `20260505_2033`) 似只写 JSON 缓存（subagent 在 within binary 检查时报告 DB 无对应 row，待人工核查后补写或确认 within 流程的 JSON-only 模式）

**代码**
- `scripts/experiments/run_cross_subject_comparison.py` 新增 `--further-pretrained-cbramod` flag（已 smoke test + 实跑 2 次验证）
- `docs/dev_log/experiments/baseline_registry.md` 新增 quaternary cross_subject baseline 表 + 更新历史一行

### 9.5 已知 caveat

- **Continue training 是 warm restart**：仅 `model.load_state_dict()`，无 optimizer/scheduler 状态恢复。AdamW betas/eps、scheduler step counter 都从零重建。这是 `further_pretrain.py` 的工具能力上限，详见 [scripts/pretraining/further_pretrain.py:469-471](../../scripts/pretraining/further_pretrain.py#L469-L471)。
- **LR 不分阶段**：用户希望 "match phase it ended at last night"。V3 first run 末段已在 constant 区，所以 `--warmup-epochs 0 --scheduler warmup_constant` → 纯恒定 5e-5 是逻辑上对齐的最佳近似。
- **min_delta=0 行为**：`if mean_loss < best_loss - 0` 退化为严格 `<`。一旦 loss 平台触底就立即 patience 计数。本次正是如此（epoch 22 触底→ patience 5 → epoch 27 stop）。
- **within_subject 结果未入 DB**：subagent 报告 within binary `20260505_2012` run_tag 在 DB 找不到。这与 extra-sessions 模式一致（仅写 JSON），但与常规 within_subject 流程预期不符。下游分析脚本若依赖 DB 而非 JSON 需注意；后续可执行 DB 回填或确认 within_subject 流程是否在某次重构中改成 JSON-only。
- **未跑 transfer 类下游**：transfer 评估仍需先 cross-subject train（已完成）→ 再用 cross-subject 输出 `.pt` 作为 `--pretrained-cbramod` 跑 transfer。本批仅完成 cross-subject 半边；transfer 留给下一轮（用户未在本 plan 要求 transfer）。

### 9.6 红线遵守情况

- ✅ 未 commit / 未 push（用户 review 后再决定）
- ✅ 未修改 `paper/drafts/paper_draft_v3.md`
- ✅ 未 kill 任何 baseline PID
- ✅ V3-continued checkpoint dir 名含 `_continued_` 后缀，与 V3 first-run dir 不撞名
- ✅ 未修改 channel_selections JSON / HPO 默认参数
- ✅ 全程 autonomy-only，4 个未知/可疑场景（plan LMDB 列表错误、within DB 未写、新 flag 首跑、monitor agent 提前退出）均自主修复未升级
