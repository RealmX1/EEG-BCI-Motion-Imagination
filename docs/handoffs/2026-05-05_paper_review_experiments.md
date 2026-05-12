# Paper-Review 补全实验 Handoff — 2026-05-05

> **Audience**: 第三批长任务监督 agent；**等待**当前正在跑的 V3 evaluation
> 完成后再开工
> **Origin**: 2026-05-05 主对话 paper draft v3 review 阶段确认的论证完整性补全
> 实验，独立于昨夜 (2026-05-04) 与今晨 (V3 continue + 4-condition 下游)
> 两批工作
> **执行模式**：autonomy-only（参考 [feedback_autonomous_bug_fixing](../../C:/Users/zhang/.claude/projects/c--Users-zhang-Desktop-github-EEG-BCI/memory/feedback_autonomous_bug_fixing.md)）
> ——遇 bug 自主修复，不向用户升级；只在涉及修改实验参数语义的边界
> 上停下汇报

---

## 0. Mission Summary

主对话在 review paper draft v3 时识别出三组**论证完整性**补全实验。这些不是
方向性主结论的关键路径（已完成的实验已建立主要 finding），而是为评审提供
更完整的对照证据：

| Task | 类型 | 决定影响的 paper 章节 | 时长 |
|------|------|----------------------|------|
| **E-1** 64ch CBraMod cross-subject (binary) | 关键中间档位 | §3.5 "32ch sweet spot" 弱化论断验证 | ~30 min |
| **E-2** 4ch CSP + Band Power cross-subject (binary) | 方法间矩阵补全 | §3.5.3 "标准方法均失效" 范围扩展 | ~30 min |
| **E-3** EEGNet 128ch transfer (binary + ternary) | 评审对照（**可选**） | §3.3 transfer 表 EEGNet 列 | ~40 min |
| **加分** Task C-2 8ch Band Power transfer (binary) | 旧 handoff 遗留 | §3.5.4 reduced-channel transfer 第二档位 | ~30 min |

**总预算**：~2 hr GPU 时间（不含 E-3 则 ~1.5 hr；含 C-2 则 ~2.5 hr）

---

## 1. 启动前置条件（Wait Gate）

在动手之前**必须**确认以下事实：

### 1.1 当前正在跑的 agent 已结束
当前在跑的是 [V3 continue training + 5.2 4-condition downstream plan](../../C:/Users/zhang/.claude/plans/v3_continued_dapt_evaluation.md)（或同名近期 plan）。其完成信号：

```bash
# (a) 检查 docs/handoffs/2026-05-04_overnight_results.md 是否被追加 §6 段落
grep -c "^## 6\." docs/handoffs/2026-05-04_overnight_results.md
# 期望: ≥ 1（说明 V3 evaluation 报告已写入）

# (b) 检查 nvidia-smi 是否无 python 进程占用 GPU
nvidia-smi | grep python
# 期望: 无输出（GPU 空闲）

# (c) 检查 V3-continued checkpoint 目录是否存在
ls checkpoints/cbramod/further_pretrain_v3_continued_*/best_model.pth 2>&1
# 期望: 找到至少一个；若未找到，前一 agent 仍在跑训练阶段
```

**全部三项 pass** 才能进入 §2 Pre-flight。任意一项 fail（特别是 (b)）则
sleep 30 min 后再 check。最多等待 8 hr；超时仍未 pass 则汇报给主对话停下。

### 1.2 不冲突资源验证
本 handoff 的所有任务都不依赖 V3 weights，且都跑 cross-subject runner（不
同 cohort、不同 channel 配置），与前一 agent 输出**无 file 写冲突**。但
GPU 必须**完全独占**——必须在 (b) 通过后再启动。

---

## 2. Pre-flight 检查清单

### 2.1 工作目录与环境

```bash
cd c:/Users/zhang/Desktop/github/EEG-BCI
uv pip list | grep -iE "torch|lmdb"  # 验证；不要 uv sync（参 §2.2 of overnight_results.md 教训）
nvidia-smi
```
- ✅ Pass 条件：torch + lmdb 都存在；GPU free ≥ 10 GB

### 2.2 PID baseline snapshot（PowerShell）

```powershell
$ts = (Get-Date).ToString("yyyyMMddHHmm")
Get-Process python -ErrorAction SilentlyContinue |
  Select-Object Id, StartTime, ProcessName |
  Out-File "$env:TEMP\baseline_pids_$ts.txt"
echo "Saved: $env:TEMP\baseline_pids_$ts.txt"
```
后续 cleanup 只 kill **不在**这份 baseline 里的 PID（[feedback_protect_agent_pids](../../C:/Users/zhang/.claude/projects/c--Users-zhang-Desktop-github-EEG-BCI/memory/feedback_protect_agent_pids.md)）。

### 2.3 数据缓存与 channel_selections.json 存在性

```bash
# 主缓存
ls caches/preprocessed/ | wc -l  # 期望 21 (S01-S21)

# 现有 channel_selections.json
cat results/32_channel/channel_selections.json | head -5
cat results/8_channel/channel_selections.json | head -5
cat results/4_channel/channel_selections.json | head -5

# 64ch 不存在 — 由 §3.1 step 1 生成
ls results/64_channel/ 2>&1 | head -3
# 期望：No such file or directory
```

### 2.4 EEGNet cross-subject ternary checkpoint（仅 E-3 需要）

```bash
ls checkpoints/cross_subject/20260330_0735_eegnet_imagery_ternary/best.pt
ls checkpoints/cross_subject/20260330_0709_eegnet_imagery_binary/best.pt
```
若有缺失，E-3 跳过；记入报告。

---

## 3. Tasks 详细规范

### 3.1 Task E-1 — 64ch CBraMod cross-subject (binary)

**目标**：填补 §3.5 "32ch sweet spot 弱化版" 论断的关键中间档位。
当前 paper 现有数据：128ch=90.68%, 61ch=89.55%, 32ch FDR=87.71%。
缺 64ch 这一档；如显示 64ch ≈ 90% 则支持"边际增益减弱"假设，
显示 ≥ 91% 则反例需重新审视论断。

**Step 1 — 生成 64ch FDR 通道选择**（一次性，<5 min）

```bash
uv run python scripts/analysis/compute_channel_selections.py \
    --n-channels 64 --methods fdr \
    --paradigm imagery --task binary
```

> 期望产物：`results/64_channel/channel_selections.json` 含 `fdr` key 对应
> 64 个 channel index。该脚本内部会：
> - 调用 `get_session_folders_for_split(paradigm, task, split='train')` 仅用
>   training session 数据计算（防泄露，[scripts/analysis/compute_channel_selections.py:38-45](../../scripts/analysis/compute_channel_selections.py#L38-L45)）
> - 输出与 32ch / 8ch / 4ch 同 schema 的 JSON

**Verify**:
```bash
cat results/64_channel/channel_selections.json | python -c "
import json, sys
d = json.load(sys.stdin)
assert 'fdr' in d, 'fdr key missing'
n = len(d['fdr']['indices'])
assert n == 64, f'expected 64 channels, got {n}'
print(f'OK: 64ch FDR has {n} channels, range [{min(d[\"fdr\"][\"indices\"])}, {max(d[\"fdr\"][\"indices\"])}]')
"
```

**Step 2 — 启动 cross-subject CBraMod**

```bash
uv run python scripts/run_cross_subject_comparison.py \
    --task binary --paradigm imagery \
    --channels 64 --channel-config fdr \
    --models cbramod \
    --cache-only --no-wandb --no-plot
```

**通过准则**：
- ✅ N=21 完整运行
- ✅ 准确率 mean ∈ [85, 92]%（介于 32ch FDR 87.71% 与 128ch 90.68% 间是预期；
  若超出该窗口需汇报）
- 数据决定 paper §3.5 论断的精确措辞

**预期输出**：`results/64_channel/fdr/<TS>_cross_subject_cache_imagery_binary.json`

---

### 3.2 Task E-2 — 4ch CSP + Band Power cross-subject (binary)

**目标**：补全 §3.5.3 表 9 "32→4 通道" 过渡的方法间对比矩阵——
当前只有 FDR top-4 (62.08%) 和 Attention top-4 (54.70%) 与负控制 (67.65%)，
缺 CSP top-4 与 Band Power top-4。这两个数据点决定 §3.5.3 末段
"FDR/Attention top-4 均低于负控制" 主张是否对全部四种方法都成立。

**Step 1 — 生成 4ch CSP + Band Power 通道选择**

```bash
# Verify which methods are present in 4ch JSON
python -c "
import json
with open('results/4_channel/channel_selections.json') as f:
    d = json.load(f)
present = sorted(d.keys())
needed = ['csp', 'band_power']
missing = [m for m in needed if m not in present]
print(f'present: {present}')
print(f'missing: {missing}')
"
```

如 missing 非空，逐个生成：

```bash
uv run python scripts/analysis/compute_channel_selections.py \
    --n-channels 4 --methods csp band_power \
    --paradigm imagery --task binary
```

**Step 2 — 启动两组 cross-subject CBraMod runs**

```bash
# CSP top-4
uv run python scripts/run_cross_subject_comparison.py \
    --task binary --paradigm imagery \
    --channels 4 --channel-config csp \
    --models cbramod \
    --cache-only --no-wandb --no-plot

# Band Power top-4
uv run python scripts/run_cross_subject_comparison.py \
    --task binary --paradigm imagery \
    --channels 4 --channel-config band_power \
    --models cbramod \
    --cache-only --no-wandb --no-plot
```

**通过准则**：
- ✅ 两个 run 各 N=21 完成
- ✅ 与负控制 67.65% 的对照——观察是否两种新方法**均**低于负控制
- 数据决定 §3.5.3 末段措辞：
  - 若两者均 < 67.65%：保留"标准方法均失效"主张并加详细数据
  - 若任一 > 67.65%：弱化为"部分标准方法仍优于负控制"

**预期输出**：
- `results/4_channel/csp/<TS>_cross_subject_cache_imagery_binary.json`
- `results/4_channel/band_power/<TS>_cross_subject_cache_imagery_binary.json`

---

### 3.3 Task E-3 — EEGNet 128ch transfer (binary + ternary)（**可选**）

**目标**：填补 §3.3 transfer learning 对照中 EEGNet 列。当前 paper 已注明
"EEGNet 128ch transfer 未执行"——该 task 是给评审一个直接对照的可选补充，
论文核心论断不依赖该数据。

**前置依赖检查**（已在 §2.4 完成）：cross-subject EEGNet checkpoint 完整。

**Step 1 — Binary**

```bash
uv run python scripts/run_transfer_comparison.py \
    --task binary --paradigm imagery \
    --models eegnet \
    --pretrained-eegnet checkpoints/cross_subject/20260330_0709_eegnet_imagery_binary/best.pt \
    --cache-only --no-wandb --no-plot
```

**Step 2 — Ternary**

```bash
uv run python scripts/run_transfer_comparison.py \
    --task ternary --paradigm imagery \
    --models eegnet \
    --pretrained-eegnet checkpoints/cross_subject/20260330_0735_eegnet_imagery_ternary/best.pt \
    --cache-only --no-wandb --no-plot
```

> 若 `run_transfer_comparison.py` 不接受 `--models eegnet` 单独运行，可改为
> `--models eegnet cbramod`：CBraMod 部分会从已有 cache 中复用，不重新训练
> （[scripts/experiments/run_transfer_comparison.py](../../scripts/experiments/run_transfer_comparison.py) 中 cache hit 逻辑）。
>
> 若 transfer runner 报 EEGNet checkpoint state_dict 不匹配，autonomous 修复
> 优先排查：(a) `best.pt` 是否含 `model_state_dict` 键封装、
> (b) EEGNet 类签名是否变（F1=16, D=4 是当前 paper 标配，若 checkpoint 是
> 旧 F1=8, D=2 需用 HPO 后版本重训——但该重训不是本 task 范围）。

**通过准则**：
- ✅ 两个 run 各 N=21 完成
- ✅ Binary EEGNet transfer 与 76.67% (cross-subject) 相差 |Δ| ≤ 5 pp
  - 显著负 → 与本 paper 假设一致（cross-subject EEGNet 已饱和）
  - 显著正 → 反例，汇报后等待主对话决策
- ✅ Ternary EEGNet transfer 与 61.23% (cross-subject) 相差 |Δ| ≤ 5 pp

**预期输出**：`results/<TS>_transfer_comparison_cache_imagery_{binary,ternary}.json`

---

### 3.4 Task C-2 — 8ch Band Power transfer (binary)（**加分**）

**目标**：填补 §3.5.4 reduced-channel transfer 第二档位。当前 paper §3.5.4
只有 32ch FDR transfer (88.45%, +0.74 pp)；如能加上 8ch Band Power transfer
则可强化"通道越少 transfer 收益越大"假设的趋势性证据。

**前置**：cross-subject 8ch Band Power checkpoint 应来自夜间报告引用的
`20260331_1950` run。

```bash
ls checkpoints/cross_subject/20260331_1950_cbramod_imagery_binary/best.pt
# 若路径名略不同，用 Glob "checkpoints/cross_subject/*band_power*/best.pt" 找
```

**启动命令**：

```bash
uv run python scripts/run_transfer_comparison.py \
    --task binary --paradigm imagery \
    --channels 8 --channel-config band_power \
    --models cbramod \
    --pretrained-cbramod checkpoints/cross_subject/20260331_1950_cbramod_imagery_binary/best.pt \
    --cache-only --no-wandb --no-plot
```

**通过准则**：
- ✅ N=21 完整运行
- ✅ |Δ vs 84.05% (8ch BP cross baseline)| ≤ 15 pp，方向不限
  - 方向为正且 |Δ| > 32ch 的 +0.74 pp → 强化"通道越少收益越大"
  - 方向为负 → 反例需主对话决策
- 此 task 失败不阻塞 E-1/E-2/E-3 的报告产出

**预期输出**：`results/8_channel/band_power/<TS>_transfer_cache_imagery_binary.json`

---

## 4. 执行顺序

```text
[Wait gate §1.1 pass]
   ↓
Pre-flight (5 min)
   ↓
E-1 64ch FDR selection 生成 (<5 min)
   ↓
E-1 64ch CBraMod cross-subject binary (~30 min)
   ↓
E-2 4ch CSP + Band Power selection 验证/生成 (<5 min)
   ↓
E-2 4ch CSP cross-subject binary (~15 min)
   ↓
E-2 4ch Band Power cross-subject binary (~15 min)
   ↓
[决策点] E-3 是否做？默认做；若 §2.4 checkpoint 缺失则跳过
   ↓ (是)
E-3 EEGNet 128ch transfer binary (~20 min)
E-3 EEGNet 128ch transfer ternary (~20 min)
   ↓
[决策点] C-2 是否做？取决于剩余时间预算
   ↓ (是)
C-2 8ch Band Power transfer binary (~30 min)
   ↓
Reporting markdown 写入（§5）
```

总耗时：**~2 hr** (E-1 + E-2 + E-3) / **~2.5 hr** (含 C-2)

---

## 5. Reporting 模板

任务完成后，在 `docs/handoffs/2026-05-05_paper_review_results.md` 写入：

```markdown
# 2026-05-05 Paper-Review 补全实验执行报告

## 0. 总览
| Task | 状态 | run_tag | 关键数字 |
| E-1 64ch CBraMod cross binary | ✅/⏸/❌ | <TS> | mean ± SD / Δ vs 32ch FDR / Δ vs 128ch |
| E-2 4ch CSP cross binary | ... | ... | mean ± SD / vs 67.65% neg ctrl |
| E-2 4ch Band Power cross binary | ... | ... | mean ± SD / vs 67.65% neg ctrl |
| E-3 EEGNet 128ch transfer binary | ✅/⏸/❌/skip | ... | mean ± SD / Δ vs 76.67% |
| E-3 EEGNet 128ch transfer ternary | ... | ... | mean ± SD / Δ vs 61.23% |
| C-2 8ch BP transfer binary | ✅/⏸/❌/skip | ... | mean ± SD / Δ vs 84.05% |

## 1. 通道选择生成记录
- E-1 64ch FDR selection JSON: results/64_channel/channel_selections.json
- E-2 4ch CSP / Band Power selection 是否新生成？

## 2. Tasks 数字与对比
对每个 task 给完整 mean ± SD + 对应 baseline 对比

## 3. Paper §3.5 / §3.5.3 / §3.3 / §3.5.4 论断更新建议
基于本批数字，标注 paper 哪些段落需要主对话补回数字 / 改论断措辞

## 4. 文件清单（新）
- 新 channel_selections.json (64ch + 4ch CSP + 4ch BP)
- 5 个新 JSON cache（E-1 / E-2×2 / E-3×2 / 可选 C-2）
- 5 条新 ExperimentDB runs

## 5. 已知 caveat
- 任何 autonomous fix 的记录
- 任何跳过的 task 与原因

## 6. 汇报触发条件
本批完成后，主对话 review 后再决定是否：
- 把 64ch FDR 注册为 baseline（baseline_registry.md 更新）
- 把 4ch CSP / Band Power 注册为 baseline（同上）
- 是否补 64ch / 4ch 的其他方法（CSP 64ch / Attention 64ch 等）
```

---

## 6. 红线（继承 + 新增）

继承自昨夜 + 今晨 handoff：
- ❌ 不 commit / 不 push
- ❌ 不修改 [paper/drafts/paper_draft_v3.md](../../paper/drafts/paper_draft_v3.md)
- ❌ 不 kill baseline PID 列表中任何 python 进程
- ❌ 不修改 HPO 默认参数；CBraMod cross-subject 沿用 [paper Table 3](../../paper/drafts/paper_draft_v3.md) 第三列
- ❌ 不为加速禁用 ±500 µV trial 剔除

新增（针对本批）：
- ❌ **不**自行决定将任何新 baseline 注册到 ExperimentDB——`db.set_baseline()`
  调用必须由主对话审核后再做（参 [CLAUDE.md baseline 管理规范](../../CLAUDE.md)）
- ❌ **不**修改 V3-continued checkpoint dir 内任何文件（即使本批不读它，
  仍然 read-only artifact）
- ❌ **不**在 wait gate (§1.1) 三项全 pass 之前启动任何 GPU 任务

---

## 7. 故障处理

| 故障 | 处理 |
|------|------|
| `compute_channel_selections.py` 报 cache 不全 | 检查 `caches/preprocessed/` 是否仍有 21 名被试 HDF5；若有解压未完成的迹象（参 [overnight_results.md §2.1](2026-05-04_overnight_results.md)），autonomous 重新解压 |
| `run_cross_subject_comparison.py` 报 channel_config 找不到 | 重 verify `results/{N}_channel/channel_selections.json` 含目标 method key；若仍缺失重跑 §3.1/3.2 step 1 生成 |
| GPU OOM（不太可能在 cross-subject 64ch / 4ch 配置下发生） | 把 `--batch-size` 降到 128（不动 backbone_lr / classifier_lr） |
| `run_transfer_comparison.py` 报 EEGNet checkpoint state_dict 不匹配 | 见 §3.3 内联说明；若是旧 EEGNet-8,2 checkpoint 则 E-3 跳过——不要为本 task 重训 cross-subject EEGNet |
| 任意 task 卡住 >40 min 无 log | Monitor stagnation 触发；记录最后 10 行 log → kill 自己启动的 PID → 跳过该 task → 继续下一项；不重试 |
| 通道选择 JSON 已存在但 method 已存在（如 4ch 的 CSP/BP 之前已生成） | 直接复用现有 selection（**不**覆盖）；进入 §3.2 step 2 |

---

## 8. 关键文件引用

- 上一份报告（昨夜）：[docs/handoffs/2026-05-04_overnight_results.md](2026-05-04_overnight_results.md)
- 上一份 handoff（今晨遗留）：[docs/handoffs/2026-05-05_remaining_experiments.md](2026-05-05_remaining_experiments.md)
- 当前正在跑的 plan（V3 continue + 4-condition）：
  `C:\Users\zhang\.claude\plans\v3_continued_dapt_evaluation.md` 或同名近期 plan
- 通道选择脚本：[scripts/analysis/compute_channel_selections.py](../../scripts/analysis/compute_channel_selections.py)
- Cross-subject runner：[scripts/run_cross_subject_comparison.py](../../scripts/run_cross_subject_comparison.py)
  → [scripts/experiments/run_cross_subject_comparison.py](../../scripts/experiments/run_cross_subject_comparison.py)
- Transfer runner：[scripts/run_transfer_comparison.py](../../scripts/run_transfer_comparison.py)
  → [scripts/experiments/run_transfer_comparison.py](../../scripts/experiments/run_transfer_comparison.py)
- Channel selection API：[src/preprocessing/channel_selection.py:111-162](../../src/preprocessing/channel_selection.py#L111-L162) `get_nch_indices()`
- Memory（关键 feedback）：
  - `feedback_use_longrun.md` — 必须用 long-run skill
  - `feedback_protect_agent_pids.md` — PID baseline 保护
  - `feedback_autonomous_bug_fixing.md` — 无人值守自主修复
  - `feedback_delegate_monitoring.md` — 监控用 subagent，不要自己 poll

---

## 9. Plan 完成度自检

- ✅ Wait gate (§1.1) — 三项 pass 条件均可程序化检测
- ✅ 任务 E-1：channel selection 生成命令已用真实脚本（`compute_channel_selections.py`）
- ✅ 任务 E-2：依赖 4ch JSON 现有结构；先 verify 再 generate
- ✅ 任务 E-3：标记可选；checkpoint state_dict 不匹配的 fallback 路径已写
- ✅ 任务 C-2：标记加分项；失败不阻塞主报告
- ✅ 红线：继承所有 + 新增 baseline 注册门禁
- ✅ 时间预算：~2 hr，远低于一次会话窗口
- ✅ Autonomous-friendly：所有决策点（E-3 跳过 / C-2 跳过）有明确条件，
  不需要用户介入
