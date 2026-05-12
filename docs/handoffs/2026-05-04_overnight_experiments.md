# 夜间实验 Handoff — 2026-05-04

> **Audience**: 一个长任务监督 agent（接管这份 handoff 后将启动并监控四组实验，凌晨完成时报告回主对话）
> **Origin**: 由主对话生成的 plan 文件 `C:\Users\zhang\.claude\plans\dazzling-drifting-candy.md` 中 Batch 4 的展开
> **Mission window**: 当前夜间（约 9–10 小时可用 GPU 时间）

---

## 0. Mission Summary（自包含背景）

本 EEG-BCI 论文（`paper/drafts/paper_draft_v3.md`，对比 CBraMod foundation model 与 EEGNet on finger-MI）在 plan review 阶段确认了四组**出版前必须完成的实验**。这些实验之前被标为 future work，但 plan reviewer 明确要求把它们拉回到出版前 mandatory，因此今晚需要并行/串行启动。

**核心约束**（来自项目 memory）：
1. **必须用 `/long-run` skill 启动后台任务**（`bash <skill-dir>/scripts/launch.sh ...`）；不要用普通 bash background（10 分钟 timeout）
2. **绝不杀死非自己启动的 Python 进程**——启动前务必 snapshot baseline PIDs，仅 cleanup 自己的 PID
3. **autonomous bug fixing OK**：训练脚本若因小 bug 中断（e.g. Windows LMDB MapResizedError、checkpoint 命名冲突）可以自主修复并重启；只在涉及修改实验参数语义时停下来等用户
4. **不要 commit 任何代码 / 不要 push**——所有 commit 等用户清晨 review

**优先级与时长预算**（按用户指示"先短后长"排序）：

| # | Task | 预估时长 | 阻塞依赖 |
|---|------|---------|--------|
| A | Quaternary cross-subject CBraMod + EEGNet（candidate 2） | ~3 hr | 无 |
| B | Leave-S04/S10/S14-out cross-subject (binary + ternary) | ~3 hr | 无 |
| C | Reduced-channel × transfer (32ch FDR + 8ch Band Power, binary) | ~10 hr | T1.1 plan：用现有 32ch FDR / 8ch Band Power 通道列表；先做 32ch，时间允许再做 8ch |
| D | DAPT V3（Stieger2021 占比降至 ~30%）+ 4 个下游评估 | ~12 hr (含下游) | V3 train 完后才能跑下游 |

**总预估**：~28 hr serial。今晚（~10 hr 窗口）能跑完 A + B + C(32ch) + 启动 D 训练（D 训练若不结束则下游评估留给明天）。

启动顺序：**A 与 B 并行**（不同任务，互不抢 GPU 显存——先确认显存预算见 §1）→ **C** → **D 训练**（如剩余时间允许）。

---

## 1. Pre-flight 检查清单

完成下述每一项后再启动任务。失败任一项立即停下并向用户报告。

### 1.1 工作目录与环境
```bash
cd c:/Users/zhang/Desktop/github/EEG-BCI
uv sync   # 确保依赖最新；若失败查 pyproject.toml
nvidia-smi   # 确认 GPU 可用 + 当前显存使用
```
- ✅ Pass 条件：GPU 显存 free ≥ 10 GB；驱动正常

### 1.2 PID baseline snapshot
```bash
# 记录当前所有 python 进程 PID 到本地文件
ps aux | grep -i python | grep -v grep > /tmp/baseline_pids_$(date +%s).txt 2>&1 || \
  tasklist | findstr -i python > /tmp/baseline_pids_$(date +%s).txt
```
- 后续 cleanup 只 kill **不在**这份 baseline 文件里的 PID
- 把 baseline 文件路径记下，handoff 报告时附上

### 1.3 数据缓存与 channel selection 文件确认
```bash
# 主缓存（21 名被试 trial-level）
ls caches/preprocessed/ | head -5

# 通道选择 JSON（32ch FDR / 8ch Band Power 必须存在）
cat results/32_channel/channel_selections.json | head -20
cat results/8_channel/channel_selections.json | head -20
```
- ✅ Pass 条件：32_channel/channel_selections.json 含 `fdr` 配置，8_channel/channel_selections.json 含 `band_power` 配置

### 1.4 DAPT 必要文件
```bash
# V2 baseline 权重（用作 V3 起点对比时引用）
ls checkpoints/cbramod/further_pretrain_20260323_0609/best_model.pth

# CBraMod 原始 TUEG 权重（V3 训练起点）
ls checkpoints/cbramod/pretrained_weights.pth

# LMDB 数据集
ls "D:/data/motion_imagination_datasets/lmdb_pretrain/"
# 期望：10 个 *_pretrain 目录，含 Stieger2021_pretrain
```

### 1.5 ExperimentDB 写入确认
```bash
# 确认 SQLite 可写（被试内/跨被试/迁移会写入 runs 表）
ls -la results/experiments.db
```

---

## 2. Task A — Quaternary cross-subject CBraMod + EEGNet（候选 2）

**Goal**：报告 quaternary 4-class 任务上 CBraMod 与 EEGNet 的 cross-subject 性能；用户已确认采用 candidate 2（cross-subject baseline 对照），不做 4-way ablation（candidate 1）。

**Why**：现有论文 §2.1 任务表只定义 binary/ternary；reviewer 会问"为什么不报告 quaternary"。这次实验跑出 quaternary 数字后，主对话决定是否纳入正文。

### 2.1 命令

```bash
uv run python scripts/run_cross_subject_comparison.py \
    --task quaternary \
    --paradigm imagery \
    --models eegnet cbramod \
    --cache-only \
    --results-dir results
```

> **注**：`--cache-only` 必须；不传 `--config` 则用默认 cross-subject HPO 参数（已在 ExperimentDB 中作为 baseline 注册，无需额外指定）。

### 2.2 用 `/long-run` skill 启动

按 skill 文档的格式启动（agent 的 long-run skill 应该清楚如何调用）：
- session name 建议：`taskA_quaternary_cross_subject`
- 预估超时：4 hr (留 buffer)

### 2.3 期望输出

- JSON: `results/{run_tag}_cross_subject_cache_imagery_quaternary.json`
- 图: `results/{run_tag}_cross-subject_combined_imagery_quaternary.png`
- ExperimentDB: 一条新 `runs` 记录，paradigm=imagery, task=quaternary, experiment_type=cross_subject

### 2.4 验证

```bash
# 拿最新 quaternary cache JSON
ls -lt results/*quaternary*.json | head -3
# 检查 group mean acc + N=21
python -c "
import json; from glob import glob
f = sorted(glob('results/*cross_subject_cache*quaternary*.json'))[-1]
d = json.load(open(f))
print(f'file: {f}')
print(f'subjects: {len(d.get(\"per_subject\", {}))}')
for m in ('eegnet', 'cbramod'):
    accs = [s.get(m, {}).get('test_acc') for s in d.get('per_subject', {}).values() if m in s]
    print(f'{m}: N={len(accs)} mean={sum(accs)/len(accs):.4f}' if accs else f'{m}: no data')
"
```

✅ Pass 条件：N=21（含全部被试），CBraMod mean ≥ 30%（chance=25%），EEGNet mean ≥ 25%

### 2.5 失败处理

- 若 OOM：减小 cbramod batch size（`--batch-size 128`）重试
- 若部分被试缺数据：检查 cache，但不要跳过被试；若 cache 缺失就报告并停（数据集本身可能未完整）

---

## 3. Task B — Leave-S04/S10/S14-out Sensitivity Check

**Goal**：去除 3 名重度伪影被试后重跑 cross-subject CBraMod，验证主结果（21 人 binary 90.68%）不依赖于这三名被试。

**Why**：S04 在 N=21 cross-subject binary 中达 98.12%，但 S04 是重度伪影被试——reviewer 会怀疑 CBraMod 学到的是伪影模式而非神经信号。Sensitivity check 给出"Δ_acc <2 pp 主结果稳健"的具体数字。

### 3.1 命令（binary）

```bash
# Subjects = 全部 21 人减去 S04/S10/S14 = 18 人
uv run python scripts/run_cross_subject_comparison.py \
    --task binary \
    --paradigm imagery \
    --models cbramod \
    --subjects S01 S02 S03 S05 S06 S07 S08 S09 S11 S12 S13 S15 S16 S17 S18 S19 S20 S21 \
    --cache-only \
    --results-dir results/sensitivity_leave3out
```

### 3.2 命令（ternary）

```bash
uv run python scripts/run_cross_subject_comparison.py \
    --task ternary \
    --paradigm imagery \
    --models cbramod \
    --subjects S01 S02 S03 S05 S06 S07 S08 S09 S11 S12 S13 S15 S16 S17 S18 S19 S20 S21 \
    --cache-only \
    --results-dir results/sensitivity_leave3out
```

### 3.3 用 `/long-run` skill 启动（与 Task A 并行可行）

显存预算：cbramod cross-subject batch=256 占用 ~9 GB；如果 Task A 也跑 cbramod cross-subject 同时跑可能 OOM——**建议串行 A → B**，或若显存充足则 A 与 B parallel；agent 自行根据 nvidia-smi 决定。

### 3.4 期望输出

- `results/sensitivity_leave3out/{run_tag}_cross_subject_cache_imagery_binary.json`
- `results/sensitivity_leave3out/{run_tag}_cross_subject_cache_imagery_ternary.json`

### 3.5 验证

```bash
python -c "
import json
for task, baseline in [('binary', 0.9068), ('ternary', 0.7488)]:
    from glob import glob
    files = sorted(glob(f'results/sensitivity_leave3out/*{task}*.json'))
    if not files:
        print(f'{task}: no result')
        continue
    d = json.load(open(files[-1]))
    accs = [s.get('cbramod', {}).get('test_acc') for s in d.get('per_subject', {}).values() if 'cbramod' in s]
    mean = sum(accs)/len(accs)
    delta = (mean - baseline) * 100
    print(f'{task}: N={len(accs)} mean={mean:.4f} (vs N=21 {baseline}) Δ={delta:+.2f}pp')
"
```

✅ Pass 条件：N=18，binary Δ 在 ±5 pp 内（任何方向；不同方向均有解读价值）

---

## 4. Task C — Reduced-channel × Transfer Learning

**Goal**：在 32ch FDR 与 8ch Band Power 配置下做 cross-subject → individual fine-tune，比较 transfer Δ vs 同 channel 的 cross-subject 基线。验证假设："cross-subject 在低密度通道下不再饱和，individual fine-tuning 边际价值放大"。

**Why**：plan EDIT T1.1 要求把"缩减通道下的迁移学习"从 future work 拉回正文，需要数字支撑 §4.4 新增小节。

### 4.1 32ch FDR transfer (binary)

```bash
uv run python scripts/run_transfer_comparison.py \
    --task binary \
    --paradigm imagery \
    --channels 32 \
    --channel-config fdr \
    --cache-only \
    --results-dir results/32_channel/fdr
```

> **注**：transfer comparison 自动从 ExperimentDB 找最佳 32ch FDR cross-subject pretrained checkpoint 作为起点；需先确认该 checkpoint 已存在：
> ```bash
> ls checkpoints/cross_subject/*32*fdr*/best.pt | head -3
> # 应能找到 32ch FDR cross-subject 的 best checkpoint（来自之前的 §3.5.1 实验）
> ```
> 若找不到，先跑 32ch FDR cross-subject 一次：`run_cross_subject_comparison.py --channels 32 --channel-config fdr --task binary --models cbramod --cache-only --results-dir results/32_channel/fdr`

### 4.2 8ch Band Power transfer (binary)

```bash
uv run python scripts/run_transfer_comparison.py \
    --task binary \
    --paradigm imagery \
    --channels 8 \
    --channel-config band_power \
    --cache-only \
    --results-dir results/8_channel/band_power
```

> 8ch Band Power cross-subject pretrained checkpoint 应也已存在（来自 §3.5.2 实验）；若不存在同样先跑一次 cross-subject。

### 4.3 期望输出

- `results/32_channel/fdr/{run_tag}_transfer_cache_imagery_binary.json`
- `results/8_channel/band_power/{run_tag}_transfer_cache_imagery_binary.json`
- ExperimentDB 各一条 transfer run

### 4.4 验证

```bash
python -c "
import json
from glob import glob
for ch, cfg, baseline in [(32, 'fdr', 0.8771), (8, 'band_power', 0.8405)]:
    files = sorted(glob(f'results/{ch}_channel/{cfg}/*transfer_cache*binary*.json'))
    if not files: print(f'{ch}ch {cfg}: no result'); continue
    d = json.load(open(files[-1]))
    accs = [s.get('cbramod', {}).get('test_acc') for s in d.get('per_subject', {}).values() if 'cbramod' in s]
    mean = sum(accs)/len(accs)
    delta = (mean - baseline) * 100
    print(f'{ch}ch {cfg}: N={len(accs)} transfer={mean:.4f} (vs cross-subject {baseline}) Δ={delta:+.2f}pp')
"
```

✅ Pass 条件：N=21，每个配置 transfer mean 与对应 cross-subject baseline 的 Δ 在 ±10 pp 内合理。**关键观察**：32ch / 8ch 的 transfer Δ 是否显著正（plan 假设方向）。

### 4.5 时间不够则只跑 32ch

如果到午夜后估计 8ch + DAPT V3 都跑不完，**只跑 32ch transfer，跳过 8ch**。在报告中明确标注。

---

## 5. Task D — DAPT V3（Stieger2021 占比降至 ~30%）

**Goal**：与 V2 同配置但减少 Stieger2021 主导，验证 plan EDIT T2.2 的假设——V1→V2 负迁移加剧的主要驱动因子是 Stieger2021 占比上升而非数据量本身。

**Why**：plan EDIT T2.2 要求出版前完成此实验；数据来源对论文 §3.6 / §4.5 的因果归因至关重要。

### 5.1 实现思路（agent 自行选择最简洁路径）

V3 的核心要求：把 Stieger2021_pretrain 的有效采样占比从 V2 的 ~79% 降至 ~30%。三种路径（按推荐度）：

**路径 1（推荐）**：写一个一次性脚本生成 Stieger2021 的"子采样 LMDB"，把它当作"独立小数据集"传给训练：
```bash
# 伪代码：
# 1. 打开 D:/data/.../Stieger2021_pretrain LMDB
# 2. 随机抽 ~26K 个 keys（约总量 42%，对应预期 ~30% 训练占比）
# 3. 写到 D:/data/.../Stieger2021_subsampled_30pct LMDB
# 4. 训练时 --lmdb-dirs 指向这个新 LMDB（其他 9 个保持原样）
```
agent 自行写 `scripts/pretraining/subsample_stieger.py` 或 inline。**关键参数**：种子固定（seed=42 与 main script 一致），输出新 LMDB 路径要明确，避免覆盖原始数据。

**路径 2**：直接 hack `MultiDatasetSampler.weights`，给 Stieger2021 手动降权。需要修改 `further_pretrain.py`；改动较大，agent 可能引入 bug。

**路径 3（兜底）**：若路径 1 失败，跳过 Stieger2021 整个数据集（lmdb-dirs 不传它）。这样总 segments 降到 ~17K，但能至少给"无 Stieger 的 DAPT"一个数据点。

### 5.2 V3 训练命令（路径 1 假设已生成 subsampled LMDB）

```bash
uv run python scripts/pretraining/further_pretrain.py \
    --max-epochs 15 \
    --patience 5 \
    --batch-size 16 \
    --effective-batch-size 128 \
    --reference-channels 64 \
    --lr 5e-5 \
    --weight-decay 0.05 \
    --warmup-epochs 0.5 \
    --mask-ratio 0.5 \
    --scheduler warmup_constant \
    --pretrained-weights checkpoints/cbramod/pretrained_weights.pth \
    --lmdb-dirs \
        "D:/data/motion_imagination_datasets/lmdb_pretrain/BNCI2014_001_pretrain" \
        "D:/data/motion_imagination_datasets/lmdb_pretrain/BNCI2015_004_pretrain" \
        "D:/data/motion_imagination_datasets/lmdb_pretrain/Cho2017_pretrain" \
        "D:/data/motion_imagination_datasets/lmdb_pretrain/GrosseWentrup2009_pretrain" \
        "D:/data/motion_imagination_datasets/lmdb_pretrain/Lee2019_MI_pretrain" \
        "D:/data/motion_imagination_datasets/lmdb_pretrain/Ofner2017_pretrain" \
        "D:/data/motion_imagination_datasets/lmdb_pretrain/PhysionetMI_pretrain" \
        "D:/data/motion_imagination_datasets/lmdb_pretrain/Schirrmeister2017_pretrain" \
        "D:/data/motion_imagination_datasets/lmdb_pretrain/Shin2017A_pretrain" \
        "D:/data/motion_imagination_datasets/lmdb_pretrain/Stieger2021_subsampled_30pct" \
    --checkpoint-dir checkpoints/cbramod/further_pretrain_v3_$(date +%Y%m%d_%H%M)
```

> 与 V2 唯一差异：`Stieger2021_pretrain` 替换为 `Stieger2021_subsampled_30pct`。其他超参与 V2 完全一致（lr/epoch/mask_ratio/scheduler）。

### 5.3 V3 下游评估（4 condition）

V3 训练结束（best_model.pth 输出后），用 V3 backbone 跑 4 个下游：

```bash
V3_CKPT=checkpoints/cbramod/further_pretrain_v3_*/best_model.pth   # 取最新

# Within-subject binary
uv run python scripts/run_within_subject_comparison.py \
    --task binary --paradigm imagery --models cbramod \
    --pretrained-weights $V3_CKPT \
    --cache-only --results-dir results/dapt_v3

# Within-subject ternary
uv run python scripts/run_within_subject_comparison.py \
    --task ternary --paradigm imagery --models cbramod \
    --pretrained-weights $V3_CKPT \
    --cache-only --results-dir results/dapt_v3

# Cross-subject binary
uv run python scripts/run_cross_subject_comparison.py \
    --task binary --paradigm imagery --models cbramod \
    --pretrained-weights $V3_CKPT \
    --cache-only --results-dir results/dapt_v3

# Cross-subject ternary
uv run python scripts/run_cross_subject_comparison.py \
    --task ternary --paradigm imagery --models cbramod \
    --pretrained-weights $V3_CKPT \
    --cache-only --results-dir results/dapt_v3
```

> **注**：先 grep `--pretrained-weights` 确认 within/cross_subject_comparison.py 是否支持该参数；若不支持，使用 `--config configs/cbramod_further_pretrained.yaml` 类似的 config override 方式（参考 `configs/cbramod_muon.yaml` 结构）。

### 5.4 期望输出

- `checkpoints/cbramod/further_pretrain_v3_*/best_model.pth`
- `checkpoints/cbramod/further_pretrain_v3_*/training_history.json`
- `results/dapt_v3/{run_tag}_*_imagery_{binary,ternary}.json` × 4 个 condition

### 5.5 与 V1/V2 对比的 baseline 数字（plan reviewer 已要求记录）

| Condition | V1 (10ep) | V2 (12ep) | V3 (待填) | Baseline (TUEG) |
|---|---|---|---|---|
| 被试内 binary | 83.84% | 82.23% | ? | 85.09% |
| 被试内 ternary | 69.25% | 68.08% | ? | 69.54% |
| 跨被试 binary | 88.84% | 89.43% | ? | 90.54% |
| 跨被试 ternary | 75.67% | 75.32% | ? | 75.42% |

V3 报告时填入对应数字。

### 5.6 启动决策

如果到执行 D 时已是凌晨 1 点之后（剩余 < 8 hr），考虑两条路径：
- **Path X**: 跑 V3 训练（~7 hr），下游评估留给明天用户清晨触发
- **Path Y**: 跳过 V3，把残余时间用于 Task C 的 8ch transfer + Task A/B 的复跑（如有失败）

在报告中明确选择了哪条路径。

---

## 6. Reporting Template（凌晨完成时回报）

把以下模板填好后输出到 `docs/handoffs/2026-05-04_overnight_results.md` 并在报告 message 中附 summary：

```markdown
# 夜间实验结果回报 — 2026-05-04 → 2026-05-05

## 总览
| Task | Status | 实际耗时 | 关键数字 |
|------|--------|---------|---------|
| A: Quaternary cross-subject | ✅/❌ | | CBraMod / EEGNet mean acc |
| B: Leave-3-out sensitivity | ✅/❌ | | binary Δ / ternary Δ |
| C: 32ch FDR transfer | ✅/❌ | | transfer mean (cf 87.71%) |
| C: 8ch Band Power transfer | ✅/❌/SKIP | | transfer mean (cf 84.05%) |
| D: V3 training | ✅/❌/SKIP | | best_loss epoch |
| D: V3 downstream (4 condition) | ✅/❌/SKIP | | V3 mean per condition |

## 详细数字
[每个 task 的 per-subject means / SDs / paired-t p values]

## 启动顺序与并行决策
[报告实际执行顺序：哪些并行，哪些串行，原因]

## 中途遇到的问题与处理
[autonomous bug fix 记录：具体改了什么文件、什么行；OOM 处理；MapResizedError 等]

## V3 数据准备路径
[选择了 §5.1 路径 1 / 2 / 3 中的哪个，为什么]

## 待用户决策项
[任何需要用户清晨决定的事项；e.g. V3 下游评估是否需要重跑、某 task 数字是否需要再做 sanity check]

## 文件清单
[新增/修改的文件路径列表，方便用户做 git diff]
```

---

## 7. Constraints & 红线（必须严格遵守）

### 不允许做的事
- ❌ **不要 git commit / git push**——所有改动留给用户清晨 review
- ❌ **不要修改 `paper/drafts/paper_draft_v3.md`**——文字修订是 Batch 1 的事，不在本次 handoff 范围
- ❌ **不要 kill 在 §1.2 baseline snapshot 中存在的 Python PID**
- ❌ **不要修改 channel selection JSON / HPO 默认参数**——本次实验全部用既有 baseline 参数
- ❌ **不要为加速而禁用 ±500 µV trial 剔除**——影响 baseline 可比性
- ❌ **不要把 V3 checkpoint dir 命名为 `further_pretrain_v2_*`**——会与 V2 现有目录冲突；统一用 `further_pretrain_v3_*`

### 自主权范围
- ✅ 修复 Windows LMDB MapResizedError、checkpoint 命名冲突、文件锁等环境性 bug
- ✅ 调整 batch size 应对 OOM（但记录原值）
- ✅ 改变 task 启动顺序（如发现 A 与 B 串行更稳定）
- ✅ Task C 时间不够时只跑 32ch 跳过 8ch
- ✅ Task D 写一次性 subsample 脚本（路径 1）
- ✅ 在 baseline checkpoint 缺失时先补跑 cross-subject（如 §4.1 注释所述）

### 报告频率
- 启动每个 task 时：单行 status update（task name + tmux session + 启动时间）
- task 完成时：单行 summary + 关键数字
- **不要每分钟轮询**——`/long-run` skill 提供 completion notification，依赖它
- 全部完成或被迫停止时：写完 §6 报告并发回主对话

---

## 8. 失败回退策略

按严重度分级：

| 情况 | 处理 |
|---|---|
| 单个 task 数字看起来异常（如 quaternary < 25% chance） | 不重跑，记入报告，标记 ⚠️ 等用户判断 |
| 单个 task 因 bug 中断且无法 5 分钟内修复 | 记入报告，跳到下一个 task |
| 多 task 连续 OOM | 停止所有任务，等待用户介入 |
| GPU 驱动崩溃或 nvidia-smi 报错 | 立即停止，发送 alarm 信息给用户 |
| 发现 baseline channel selection JSON 内容意外修改 | 立即停止；这是 §1.3 之外的篡改信号 |

---

## 9. 关键文件引用（供 agent 自查）

- 主 plan: `C:\Users\zhang\.claude\plans\dazzling-drifting-candy.md`（特别看 EDIT T1.1 / T1.5c / T2.1 / T2.2）
- Project CLAUDE.md: `c:/Users/zhang/Desktop/github/EEG-BCI/CLAUDE.md`（核心约束）
- 项目 memory: `C:\Users\zhang\.claude\projects\c--Users-zhang-Desktop-github-EEG-BCI\memory\MEMORY.md`
  - feedback_use_longrun.md
  - feedback_protect_agent_pids.md
  - feedback_autonomous_bug_fixing.md
- 实验脚本入口:
  - `scripts/run_within_subject_comparison.py` (wrapper)
  - `scripts/run_cross_subject_comparison.py` (wrapper)
  - `scripts/run_transfer_comparison.py` (wrapper → `scripts/experiments/run_transfer_comparison.py`)
  - `scripts/pretraining/further_pretrain.py`
- ExperimentDB API: `src/results/experiment_db.py`（用 `db.find_runs(...)` 查 baseline run_tag）

---

## 10. 主对话用户上下文（agent 完成后向其汇报时引用）

- 用户在 plan review 阶段明确指示三个原 future-work 实验（leave-3-out / reduced-ch transfer / Stieger leave-out）必须在出版前完成
- 用户对 quaternary 选择 candidate 2（cross-subject baseline 对照），不做 4-way ablation
- 用户偏好"先短后长"启动顺序：A → B → C → D
- 用户已明确接受"V3 训练若过半夜跑不完则下游评估留给清晨"

完成报告时引用本 handoff 文件 `docs/handoffs/2026-05-04_overnight_experiments.md` 和 plan 文件 `C:\Users\zhang\.claude\plans\dazzling-drifting-candy.md` 作为来源。
