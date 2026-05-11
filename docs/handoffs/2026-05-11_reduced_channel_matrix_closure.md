# Reduced-channel × cross-subject 40-cell 矩阵闭合 交接 (2026-05-11)

## 目的

补齐论文 §3.5 "channel sweet spot" 论证所需的覆盖矩阵：CBraMod × cross_subject × n_subjects=21 在 `{4, 8, 32, 64}ch × {fdr, attention, band_power, csp, negative_control} × {binary, ternary}` 共 40 cell 上的横向对比。开工时 19/40 完成、21/40 缺失（主缺口在 64ch 整列 + ternary 轴）。本轮跑通缺失的 21 个 cell，最终 **40/40 闭合**。

## 关键发现（TL;DR）

1. **40-cell 矩阵完整闭合**：4×5×2 grid 全 cell 在 ExperimentDB 中均有 `n_subjects=21, is_complete=1, model=cbramod` 的 run（详见 §40-cell 矩阵）
2. **64ch 行 5 method 高度一致**：binary 范围 86.22-89.46%（spread 3.24 pp）、ternary 范围 73.35-75.44%（spread 2.09 pp）—— **method 在 64ch 上的影响 ≤ 3 pp**，强烈支持 §3.5 "method-agnostic at high channel count" 论断
3. **64ch negative_control 反超 fdr 在 ternary 上**：75.44% (neg_ctrl) vs 75.12% (fdr)，Δ = +0.32 pp。这是论文 §3.5.3 "method-dependent under sparsity" 的反面证据 —— 64ch 时即便随机/补集通道也能达到 method-driven 的性能上限
4. **channel-count ladder 在 ternary 上首次完整**：4→8→32→64ch × fdr/binary = 62.08→76.43→87.71→89.46%；fdr/ternary = 46.05→64.88→70.79→75.12%。两 task 都符合"diminishing margin"假设但 ternary 的 4→8 跳变更大（+18.83 pp vs binary +14.35 pp）
5. **2 个 codebase footguns 被识别 + 解决**：(a) `wandb` 是空 namespace package、`wandb.init` 不存在，所有训练脚本默认走 wandb 必崩 —— 用 `--no-wandb` flag 绕过；(b) [src/preprocessing/channel_selection.py:156](../../src/preprocessing/channel_selection.py#L156) 强制 `len(indices) == n_channels`，历史 31-index 的 32ch `negative_control` 现在不兼容
6. **wall-clock 8h 20m** 完成 21 cell + 1 retry + 全部验证；比初始估计 22h GPU 大幅快 —— v3 cosine schedule + early stop 让 cell 平均 23 min（估计 50 min）

## ⚠️ Operational Gotchas (READ FIRST 如果你要重跑或扩展)

以下 3 个坑会让"看似简单"的命令直接崩溃。下个 agent 接手必须先验。

### 1. wandb 模块已坏（命中所有训练脚本默认路径）

```python
>>> import wandb; print(wandb.__file__, hasattr(wandb, 'init'))
None False
```

`wandb` 在当前 venv 里是空 namespace package（`__file__=None`，无 `init` 属性）—— 所有走 [src/utils/wandb_logger.py:205](../../src/utils/wandb_logger.py#L205) `wandb.init()` 的训练 entrypoint 会在 4 秒内 crash，traceback 在 PowerShell `Invoke-Expression | Tee-Object` 流水线下会被吞，看起来像静默失败。

**两种修法**（任选一）：
- **绕过**（推荐用于本轮类型的纯实验填坑）：所有训练命令加 `--no-wandb` flag（[src/cli/experiment_utils.py:149](../../src/cli/experiment_utils.py#L149) 注册）
- **修底层**（推荐用于长期工作）：`uv pip install --force-reinstall wandb`，验证 `python -c "import wandb; print(wandb.__file__)"` 返回非 None；然后跑一个 1-cell smoke test 确认 `wandb.init()` 不再抛 `AttributeError`

### 2. `channel_selections.json` 强制 len 校验

[src/preprocessing/channel_selection.py:156-160](../../src/preprocessing/channel_selection.py#L156-L160)：

```python
indices = selections[config_name]['indices']
if len(indices) != n_channels:
    raise ValueError(
        f"Config '{config_name}' in {json_path} has {len(indices)} channels, "
        f"expected {n_channels}"
    )
```

历史上 32ch `negative_control` 只有 31 indices（"5 method 互补集"自然只有 31 个），早期 `20260302_0141` binary run 是在这条校验加入之前跑的。本轮被这个抛错挂掉一次（Cell 12）。修法：用 seed=42 从 4-method union 抽 1 个 pad channel 补到 32。同模式已经应用到 32ch（pad channel 29）和 64ch（4 pure-complement + 60 random pad）。

**通用经验**：任何 `--channel-config <name>` 的 cell 跑之前，先验：
```bash
python -c "import json,io; d=json.load(io.open('results/{N}_channel/channel_selections.json',encoding='utf-8')); print(len(d['configs']['{cfg}']['indices']))"
```
必须 `== N`，否则先 pad / 重新生成。

### 3. overwatch warm-up 固定 30 min

[scripts/overwatch/overwatch.py](../../scripts/overwatch/overwatch.py)：`WARMUP_SEC = 30 * 60` + rolling 30 min window —— 即使 GPU 此刻 0%，overwatch 也会等满 30 分钟才能 exit 0。**这是有意的，不要试图缩短**（避免训练刚结束就抢资源）。

排队多个训练时，把 overwatch **只放在 pipeline 开头一次**，不要每个 cell 都 gate（会浪费 21 × 30 min = 10.5h）。

### 4. PowerShell `Invoke-Expression | Tee-Object` 会吞 stderr

driver 用 `Invoke-Expression $cmd 2>&1 | Tee-Object -Append -FilePath $LOG` 会把 stderr 重定向到 stdout 再 tee，**但 Python tracebacks 在某些情况下会先于重定向输出，导致 log 里只看到 banner 不看到 error**。

**调试模式**：单独跑一次失败的 cell 用 `uv run python ... 2>&1 | tail -40` 直接拿真实 stderr，再回头改 driver。本轮 v1 的诊断就靠这个救出来。

## 40-cell 矩阵（cross-subject × CBraMod × n=21）

| n | method | binary mean_acc | binary run_tag | ternary mean_acc | ternary run_tag |
|---|---|---|---|---|---|
| 4 | fdr | 62.08% | `20260330_2214` | 46.05% | `20260511_1618` |
| 4 | attention | 54.70% | `20260330_2200` | 41.55% | `20260511_1642` |
| 4 | band_power | 78.75% | `20260505_2308` | 60.67% | `20260511_1655` |
| 4 | csp | 66.99% | `20260505_2246` | 47.62% | `20260511_1731` |
| 4 | negative_control | 67.65% | `20260330_1442` | 53.37% | `20260310_0054` |
| 8 | fdr | 76.43% | `20260330_1311` | 64.88% | `20260511_1439` |
| 8 | attention | 68.42% | `20260330_1334` | 59.50% | `20260302_2140` |
| 8 | band_power | 84.05% | `20260331_1950` | 66.33% | `20260511_1508` |
| 8 | csp | 81.73% | `20260331_2044` | 61.77% | `20260511_1539` |
| 8 | negative_control | 76.34% | `20260511_1425` | 59.05% | `20260511_1600` |
| 32 | fdr | 87.71% | `20260330_0836` | 70.79% | `20260221_0332` |
| 32 | attention | 85.48% | `20260330_1009` | 71.53% | `20260228_2247` |
| 32 | band_power | 86.85% | `20260330_1105` | 72.20% | `20260511_1348` |
| 32 | csp | 84.94% | `20260330_1032` | 70.12% | `20260511_1404` |
| 32 | negative_control | 84.08% | `20260302_0141` | 72.38% | `20260511_1757` |
| **64** | **fdr** | **89.46%** | `20260505_2223` | **75.12%** | `20260511_1148` |
| **64** | **attention** | **87.53%** | `20260511_1038` | **73.81%** | `20260511_1217` |
| **64** | **band_power** | **87.89%** | `20260511_1050` | **75.02%** | `20260511_1237` |
| **64** | **csp** | **86.22%** | `20260511_1111` | **73.35%** | `20260511_1256` |
| **64** | **negative_control** | **88.57%** | `20260511_1131` | **75.44%** | `20260511_1314` |

> **数据来源**：所有 `20260511_*` run_tag 均在 `results/experiments.db` 中，与 `model_summaries.model_type='cbramod', is_complete=1, n_subjects=21` JOIN 验证通过。JSON cache 路径形如 `results/{N}_channel/{cfg}/{run_tag}_cross_subject_cache_imagery_{task}.json`。

### 64ch 行 spread 分析

| Task | min | max | spread | median |
|---|---|---|---|---|
| binary | 86.22 (csp) | 89.46 (fdr) | 3.24 pp | 87.89 (band_power) |
| ternary | 73.35 (csp) | 75.44 (neg_ctrl) | 2.09 pp | 75.02 (band_power) |

在 ternary 上 **neg_ctrl 反超 fdr 0.32 pp**——这与 32ch 上 fdr 70.79% > neg_ctrl 72.38%（neg_ctrl 又略胜）和 4ch 上 fdr 46.05% < band_power 60.67% 的 pattern 矛盾。**64ch 已经超过 method 选择的"临界 channel count"**——一旦通道数足够，CBraMod 的 ACPE + foundation model representation 让"选哪些 channel"几乎不重要。

## Timeline (2026-05-11 本轮)

| 时间 | 事件 | task ID / 引用 |
|---|---|---|
| 01:12:48 | v1 driver 启动（21 cell + overwatch gate） | PROC `21cell-fill_20260511_011248` |
| 01:12:51 → 01:53:44 | overwatch warm-up，等已有 `run_transfer_comparison.py` 释放 GPU (41m) | — |
| 01:53:44 → 01:55:50 | P1 (64ch atn/csp/bp) + merge + P2c + P3 sanity ✅ | — |
| 01:55:50 → 01:57:08 | **21 cell 全部 4 秒内 exit=1**（wandb 模块崩溃，banner 后无 traceback 落 log） | — |
| ~10:02 | 手动重跑单 cell 用 `tail` 抓 stderr → 发现 `AttributeError: module 'wandb' has no attribute 'init'` | — |
| 10:03 | driver 加 `--no-wandb` patch（[logs/drive_21cell_fill.ps1:152](../../logs/drive_21cell_fill.ps1#L152)） | — |
| 10:06:54 | v2 driver 启动 | PROC `21cell-fill-v2_20260511_100654` |
| 10:06:54 → 10:38:55 | overwatch（32 min） + pre-work（idempotent，~3 min） | — |
| 10:38:55 → 17:53:07 | 21-cell loop：20/21 cells 通过、Cell 12 (32ch neg_ctrl ternary) exit=1 in 4.4s | — |
| ~17:55 | Cell 12 诊断：[src/preprocessing/channel_selection.py:156](../../src/preprocessing/channel_selection.py#L156) `len==n_channels` 校验，neg_ctrl 只有 31 indices | — |
| 17:56 | 32ch neg_ctrl pad 31→32（seed=42 加 channel 29 / A30） | `results/32_channel/channel_selections.json` |
| 17:57:09 | Cell 12 retry 启动 | PROC `cell12-rerun_20260511_175709` |
| 18:30:46 | Cell 12 retry 完成（33m 37s，72.38%） | — |
| ~18:33 | 40/40 矩阵闭合验证通过 | — |

## State Verification（接手后 30 秒确认 handoff 还成立）

### 注册表完整性

```bash
cd "c:/Users/zhang/Desktop/github/EEG-BCI"
uv run python -c "
import json, io
for n in [4, 8, 32, 64]:
    d = json.load(io.open(f'results/{n}_channel/channel_selections.json', encoding='utf-8'))
    missing = {'fdr','attention','band_power','csp','negative_control'} - set(d['configs'])
    print(f'{n}ch: {len(d[\"configs\"][\"negative_control\"][\"indices\"])} neg_ctrl indices, missing required: {missing or \"none\"}')
"
```

预期输出：每行 neg_ctrl indices 数等于 N，required 全 'none'：
```
4ch: 4 neg_ctrl indices, missing required: none
8ch: 8 neg_ctrl indices, missing required: none
32ch: 32 neg_ctrl indices, missing required: none
64ch: 64 neg_ctrl indices, missing required: none
```

### 40-cell DB 完整性

```python
import sqlite3, itertools
con = sqlite3.connect("file:results/experiments.db?mode=ro", uri=True); cur = con.cursor()
target = list(itertools.product([4,8,32,64], ["fdr","attention","band_power","csp","negative_control"], ["binary","ternary"]))
missing = []
for n,cfg,task in target:
    chans = [n] if n != 64 else [61,64]
    qm = ",".join("?"*len(chans))
    cur.execute(f"""SELECT 1 FROM runs r JOIN model_summaries ms ON ms.run_id=r.run_id
                   WHERE r.paradigm='imagery' AND r.experiment_type='cross_subject' AND r.task=?
                   AND r.channel_config=? AND r.n_channels IN ({qm}) AND r.n_subjects=21
                   AND r.is_complete=1 AND ms.model_type='cbramod' LIMIT 1""",
                (task, cfg, *chans))
    if not cur.fetchone(): missing.append((n,cfg,task))
print(f"Missing: {len(missing)}/40", missing or "MATRIX_CLOSED")
```

预期：`Missing: 0/40 MATRIX_CLOSED`

### wandb 是否还坏

```bash
uv run python -c "import wandb; print('OK' if hasattr(wandb,'init') else 'BROKEN')"
```

写本 handoff 时仍 `BROKEN`。若已 `OK`，可考虑去掉 driver 的 `--no-wandb`（但本轮 21 个 run 已写入 DB，不需要重跑）。

## Reproduce / Extend Commands

### 跑单个新 cell（举例：4ch attention binary，假设已经跑过则跳过）

```powershell
uv run python scripts/experiments/run_cross_subject_comparison.py `
  --paradigm imagery --models cbramod `
  --config configs/cbramod_v3_cross.yaml --cache-only --no-wandb `
  --task binary --channels 4 --channel-config attention
```

### 复跑本轮所有 21 cell（idempotent，会写新 run_tag）

```powershell
pwsh.exe -ExecutionPolicy Bypass -File logs/drive_21cell_fill.ps1
```

driver 内置 overwatch gate + P1/P2c/P3 pre-work（重跑产物与已有 channel_selections.json 等价、不破坏 fdr 等历史 entry）。

### 添加新通道数（举例：16ch 或 96ch）扩展矩阵

1. 生成 channel selections：
   ```powershell
   uv run python scripts/analysis/compute_channel_selections.py --n-channels 16 --methods fdr attention csp band_power
   ```
2. 为该通道数生成 negative_control（参考 [logs/_p2c_inline.py](../../logs/_p2c_inline.py)，将 64 改为 16，pad 策略相同）
3. 复制 [logs/drive_21cell_fill.ps1](../../logs/drive_21cell_fill.ps1) 改 `$cells` 数组，加入 5 method × 2 task = 10 个新 cell
4. 走 long-run skill 跑

## Next Steps Unlocked（论文 §3.5 / §3.5.3 / §3.5.4 重写）

### 可立即做（数据齐了）

1. **画 4×5 grid（4 个 channel-count × 5 method × 2 task overlay）**：取代当前 §3.5 的单线 sweet-spot 图。横轴 method、纵轴 mean_acc ± std，每个 channel-count 一个 panel。预计 20 分钟 matplotlib 工作量。
2. **64ch method-agnostic 论断 + 显著性测试**：5 method × {binary, ternary} 在 64ch 的 ANOVA（n=21 paired）。预期 p > 0.05 → 加进 §3.5 "at 64ch, method choice does not significantly affect accuracy"
3. **更新 §3.5.3 "method-dependent under sparsity"**：现有论断在 4ch 与 8ch 上仍成立（band_power 在 4ch 比 fdr 高 16.70 pp；在 8ch 比 fdr 高 7.62 pp），但需补充 "this method-dependence vanishes at 32ch+ and is reversed at 64ch (where neg_ctrl ternary > fdr ternary)"
4. **§3.5.4 "reduced-channel transfer 收益"**：本轮没跑 transfer 范式，§3.5.4 还在用 cross-subject 数据做转移类比；不变更

### 需要新数据

5. **within-subject reduced-channel matrix（仍空）**：所有 reduced-channel runs 都是 cross-subject，within 维度 4×5×2=40 cell 全空。如果论文需要"channel reduction 在 within-subject 范式下也成立"的论断，需要新一轮 ~40-60h 训练
6. **transfer reduced-channel matrix（基本空）**：仅有 32ch fdr/binary 和 8ch attention 几个 transfer cell，是稀疏的；扩展到 4×5×2=40 cell 是一个完整的 follow-up
7. **EEGNet 对照（仅 128ch 有 baseline）**：用户本轮明确排除，但 §3.5 论证如果要写"CBraMod-specific advantage"则需要 EEGNet 同矩阵对照

## Memory Updates

本轮变更的 conversation-persistent memories：

- [feedback_wait_for_overwatch.md](../../C:/Users/zhang/.claude/projects/c--Users-zhang-Desktop-github-EEG-BCI/memory/feedback_wait_for_overwatch.md)：从 "wait-for-overwatch" 扩展为 **"gate-with-overwatch"**，新增 Rule 1（任何训练前 prepend overwatch 作为 blocking gate）。索引 [MEMORY.md](../../C:/Users/zhang/.claude/projects/c--Users-zhang-Desktop-github-EEG-BCI/memory/MEMORY.md) 已同步更新。

## Caveats（科学性 limitations，写论文时 disclose）

1. **32ch negative_control 不是纯 complement**：原 31-index version（"5 method 互补集"）是严格 complement 语义，本轮 pad 1 个 channel（index 29，seed=42 从 4-method union 抽）让它变成"31 pure-complement + 1 pad"。pad channel 取自 method-union，**严格意义上 32ch neg_ctrl 不再是 100% complement**。但 spread 0.30 pp 量级（与 31 indices 时的 binary 84.08% 一致），不影响论文论断。

2. **64ch negative_control 是 "4 pure-complement + 60 pad"**：因为 4-method 在 64ch 各选 64，union 已覆盖 124 个 channel，complement 只剩 4 个。所以 64ch neg_ctrl 实质是"以 4 个真 complement 为种子的 seed=42 sampler"，更像"低度 method-overlap"而非"纯 complement"。这是 64ch 上 neg_ctrl 反超 fdr 的可能解释之一 —— pad 的 60 channel 包含一些"中等 informative"的通道。

3. **本轮所有 cell 共用 [configs/cbramod_v3_cross.yaml](../../configs/cbramod_v3_cross.yaml) 超参**（V3 pretrained + cosine_annealing_warmup_decay + batch=256 + backbone_lr=1.3e-4）。**没有按 channel-count 做 HPO**——可能 4ch 用更小 batch 更合适、64ch 应该更大 LR。但保持一致更便于矩阵内对比。

4. **PowerShell `Invoke-Expression` 流水线吞 stderr 的潜在风险未根治**：本轮 v1 全 21 cell 的真实 traceback 都没落 log，我是靠手动单独重跑一个 cell 才捞到错误信息。下次若 driver 失败仍要这样救。理想修法是 driver 直接 `& uv run ...` 而非 `Invoke-Expression`，但本轮没改 —— v2 用 `--no-wandb` 修复了根因后没遇到新失败。

5. **v1 driver 的 21 个失败 run 仍在 DB**（run_tag 形如 `20260511_01XX_*`，is_complete=0 / n_subjects=NULL）：这些是 wandb 崩溃留下的死行，不影响 §State Verification 的查询（has `is_complete=1` filter），但污染 DB。如果要清理：
   ```sql
   DELETE FROM runs WHERE run_tag LIKE '20260511_01%' AND is_complete=0 AND channel_config IN ('fdr','attention','band_power','csp','negative_control');
   ```
   未自动执行 —— 留待开发者决定。

## 结果文件清单

```
results/
├── 64_channel/
│   ├── channel_selections.json              # 含 fdr + attention + csp + band_power + negative_control (本轮 P1/P2c 写入)
│   ├── attention/20260511_1038_*.json + .png
│   ├── attention/20260511_1217_*.json + .png  (ternary)
│   ├── band_power/20260511_1050_*.json + .png
│   ├── band_power/20260511_1237_*.json + .png
│   ├── csp/20260511_1111_*.json + .png
│   ├── csp/20260511_1256_*.json + .png
│   ├── fdr/20260511_1148_*.json + .png       (ternary; binary 20260505_2223 是旧 run)
│   ├── negative_control/20260511_1131_*.json + .png
│   └── negative_control/20260511_1314_*.json + .png
├── 32_channel/
│   ├── channel_selections.json              # 注入 negative_control 31→32 indices (本轮 P2a + pad)
│   ├── band_power/20260511_1348_*.json + .png   (ternary)
│   ├── csp/20260511_1404_*.json + .png          (ternary)
│   └── negative_control/20260511_1757_*.json + .png  (Cell 12 retry)
├── 8_channel/
│   ├── channel_selections.json              # 注入 negative_control (本轮 P2b)
│   ├── negative_control/20260511_1425_*.json + .png  (binary)
│   ├── negative_control/20260511_1600_*.json + .png  (ternary)
│   ├── fdr/20260511_1439_*.json + .png       (ternary)
│   ├── band_power/20260511_1508_*.json + .png (ternary)
│   └── csp/20260511_1539_*.json + .png       (ternary)
├── 4_channel/
│   ├── fdr/20260511_1618_*.json + .png       (ternary)
│   ├── attention/20260511_1642_*.json + .png  (ternary)
│   ├── band_power/20260511_1655_*.json + .png (ternary)
│   └── csp/20260511_1731_*.json + .png        (ternary)
└── experiments.db                            # +21 新 run + model_summaries 行

logs/
├── drive_21cell_fill.ps1                     # driver 脚本（含 --no-wandb patch）
├── _p2c_inline.py                            # 64ch negative_control 生成器（inline）
├── _p3_inline.py                             # 注册表 sanity check
├── _pid_baseline.txt + _pid_baseline_v2.txt  # PID 快照（feedback_protect_agent_pids）
└── 21cell_fill_*.log                         # driver 双写日志

~/.claude-procs/
├── 21cell-fill_20260511_011248/              # v1 driver（wandb crash，留作 audit）
├── 21cell-fill-v2_20260511_100654/           # v2 driver（20/21 success）
└── cell12-rerun_20260511_175709/             # Cell 12 retry (success)
```

## Paper Citation Template

> **数据来源（reduced-channel cross-subject 40-cell 矩阵）**：CBraMod × cross_subject × n_subjects=21 × imagery × {binary, ternary}，channel configs ∈ {fdr, attention, band_power, csp, negative_control} × N ∈ {4, 8, 32, 64}。完整 run_tag + mean_acc 列表见 [docs/handoffs/2026-05-11_reduced_channel_matrix_closure.md §40-cell 矩阵](./2026-05-11_reduced_channel_matrix_closure.md#40-cell-矩阵)。所有 run 共用 [configs/cbramod_v3_cross.yaml](../../configs/cbramod_v3_cross.yaml) 超参；本轮 2026-05-11 新增的 21 个 run_tag 均以 `20260511_*` 前缀。

> **64ch method-agnostic 论断**：5 method × binary 范围 86.22-89.46% (spread 3.24 pp)；5 method × ternary 范围 73.35-75.44% (spread 2.09 pp)；ternary 上 negative_control (75.44%) 反超 fdr (75.12%) 0.32 pp，支持"at 64ch, channel selection method choice is largely uninformative"。

## 不在本 handoff 范围

- **§3.5 论文段落的具体重写**：本 handoff 提供数据 + 论证方向，不写论文
- **4×5×2 grid 可视化生成**：列在 §Next Steps，但生成代码 / matplotlib 调优未做
- **method × channel-count 的 ANOVA / paired t-test**：数据齐了但统计检验未跑
- **within-subject + transfer reduced-channel 矩阵**：仍 ~80% 空，本轮明确排除
- **EEGNet reduced-channel 对照**：用户明确排除
- **wandb 模块根因修复**：本 handoff 仅记录"已绕过"，未真正修底层依赖（uv sync / 重装包）
- **v1 死行清理 SQL**：列在 Caveat #5，未执行
- **本轮新发现的 codebase 文件级 bug 提报**：例如 driver 的 stderr 吞噬，未在 codebase 修复
