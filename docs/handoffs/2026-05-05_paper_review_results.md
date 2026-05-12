# 2026-05-05 Paper-Review 补全实验执行报告

> **执行模式**：autonomous（参 `feedback_autonomous_bug_fixing.md`）
> **关联 handoff**：[2026-05-05_paper_review_experiments.md](2026-05-05_paper_review_experiments.md)
> **报告生成**：2026-05-06（首次跨夜执行）；2026-05-07 §5.4 增补 reproducibility 重跑确认与永久代码层 Agg 修复

---

## 0. 总览

| Task | 状态 | run_tag | 关键数字 | Δ vs baseline |
|------|------|---------|---------|---------------|
| E-1 64ch FDR CBraMod cross binary | ✅ | `20260505_2223` | **89.46% ± 8.98%** | +1.75 pp vs 32ch FDR (87.71%); −1.22 pp vs 128ch (90.68%) |
| E-2a 4ch CSP CBraMod cross binary | ✅ | `20260505_2246` | **66.99% ± 8.99%** | −0.66 pp vs neg ctrl (67.65%) |
| E-2b 4ch Band Power CBraMod cross binary | ✅ | `20260505_2308` | **78.75% ± 10.36%** | **+11.10 pp vs neg ctrl** (67.65%) |
| E-3a EEGNet 128ch transfer binary | ✅ | `20260506_2039` | **80.77% ± 11.19%** | +4.10 pp vs cross (76.67%) |
| E-3b EEGNet 128ch transfer ternary | ✅ | `20260506_2112` | **66.23% ± 12.61%** | +5.00 pp vs cross (61.23%) |
| C-2 8ch BP CBraMod transfer binary | ✅ ⚠ | `20260506_2159` | **82.02% ± 10.74%** | **−2.03 pp** vs cross (84.05%) — **反向**迁移 |

---

## 1. 通道选择生成记录

### 1.1 E-1 64ch FDR (新生成)
- **路径**：`results/64_channel/channel_selections.json`
- **方法**：FDR-corrected ANOVA on training-only sessions
- **Top-5 indices**：`[4, 8, 14, 15, 17]`（具体见 JSON）
- **生成命令**：
  ```bash
  uv run python scripts/analysis/compute_channel_selections.py \
      --n-channels 64 --methods fdr --paradigm imagery --task binary
  ```
- ✅ Verify: 64 个 channel index 写入 OK

### 1.2 E-2 4ch CSP + Band Power (合并写入)
- **路径**：`results/4_channel/channel_selections.json`
- **关键操作**：用 `scripts/analysis/_merge_channel_selections.py`（本次会话新增）合并旧
  attention/fdr/fdr_attention_overlap/negative_control 与新 csp/band_power，避免覆盖
- **Top-4 indices**：
  - CSP: `[29, 44, 48, 63]`
  - Band Power: `[59, 86, 106, 122]`
- **最终 JSON keys**：`['attention', 'band_power', 'csp', 'fdr', 'fdr_attention_overlap', 'negative_control']`
- **备份**：`results/4_channel/channel_selections.json.bak_pre_e2`

---

## 2. Tasks 数字与对比

### 2.1 E-1 — 64ch FDR cross binary (89.46% ± 8.98%)

> **数据来源**：`results/64_channel/fdr/20260505_2223_cross_subject_cache_imagery_binary.json`

| 配置 | mean | 来源 |
|------|------|------|
| 128ch | 90.68% | paper §3.5（baseline） |
| **64ch FDR** | **89.46%** | 本批 E-1 |
| 61ch | 89.55% | paper §3.5 |
| 32ch FDR | 87.71% | paper §3.5（baseline） |

**结论**：64ch 落在 [88, 91]% 期望窗口内，且**严格介于 32ch (87.71%) 与 128ch (90.68%) 之间**，
支持 paper §3.5 "边际增益减弱" 论断（每减半通道损失 ~1 pp）。

### 2.2 E-2a — 4ch CSP cross binary (66.99% ± 8.99%)

> **数据来源**：`results/4_channel/csp/20260505_2246_cross_subject_cache_imagery_binary.json`

### 2.3 E-2b — 4ch Band Power cross binary (78.75% ± 10.36%)

> **数据来源**：`results/4_channel/band_power/20260505_2308_cross_subject_cache_imagery_binary.json`

**4ch 对比矩阵（CBraMod cross binary）**：

| 方法 | mean | vs neg ctrl (67.65%) | 来源 |
|------|------|---------------------|------|
| Attention top-4 | 54.70% | −12.95 pp | paper §3.5.3 |
| FDR top-4 | 62.08% | −5.57 pp | paper §3.5.3 |
| Negative control | 67.65% | — | paper §3.5.3 |
| **CSP top-4** | **66.99%** | **−0.66 pp** | 本批 E-2a |
| **Band Power top-4** | **78.75%** | **+11.10 pp** | 本批 E-2b |

**关键观察**：Band Power 显著超过 negative control，**反证 paper §3.5.3 "标准方法均失效" 主张**。
建议措辞改为："基于稀疏统计量（FDR）或注意力（Attention）的方法在 4ch 极端约束下失效，
但物理动机鲜明的 Band Power 标准基线仍能保留可观判别性能。"

CSP 与 negative control 几乎平分秋色（Δ=−0.66 pp），可视为"几无信号"。

### 2.4 E-3a — EEGNet 128ch transfer binary (80.77% ± 11.19%)

> **数据来源**：`results/20260506_2039_transfer_cache_imagery_binary.json`
>
> **修复历程**：
> - v1 (`20260505_2321`) 因 matplotlib Tk backend 在 worker thread GC 中触发
>   `Tcl_AsyncDelete` 致命 abort（S03/S04 完成后 S05 崩溃）
> - v2 (`20260506_2039`) 用 `MPLBACKEND=Agg` 强制软件渲染后端，21/21 完整完成

**Per-subject test accuracy (majority voting, Sess2 Finetune)**：

| Subject | Acc | Subject | Acc | Subject | Acc |
|---------|-----|---------|-----|---------|-----|
| S01 | 72.50% | S08 | 91.88% | S15 | 82.50% |
| S02 | 92.50% | S09 | 96.88% | S16 | 68.12% |
| S03 | 85.00% | S10 | 66.25% | S17 | 76.88% |
| S04 | 94.38% | S11 | 78.12% | S18 | 90.00% |
| S05 | 90.00% | S12 | 76.25% | S19 | 96.25% |
| S06 | 73.12% | S13 | 89.38% | S20 | 55.62% |
| S07 | 80.62% | S14 | 67.50% | S21 | 72.50% |

**结论**：EEGNet 128ch transfer **80.77%** 高于 cross-subject baseline 76.67%（Δ=+4.10 pp，
落在 ±5 pp 期望窗口内）。这与 paper §3.3 假设一致——transfer learning 对 EEGNet 也提供
正向收益但幅度小于 CBraMod。

### 2.5 E-3b — EEGNet 128ch transfer ternary (66.23% ± 12.61%)

> **数据来源**：`results/20260506_2112_transfer_cache_imagery_ternary.json`

**Per-subject test accuracy (majority voting, Sess2 Finetune)**：

| Subject | Acc | Subject | Acc | Subject | Acc |
|---------|-----|---------|-----|---------|-----|
| S01 | 53.33% | S08 | 72.92% | S15 | 68.33% |
| S02 | 77.08% | S09 | 89.17% | S16 | 50.42% |
| S03 | 73.75% | S10 | 57.50% | S17 | 65.00% |
| S04 | 80.83% | S11 | 61.25% | S18 | 79.17% |
| S05 | 55.42% | S12 | 54.58% | S19 | 80.00% |
| S06 | 70.00% | S13 | 80.00% | S20 | 42.92% |
| S07 | 71.25% | S14 | 63.33% | S21 | 44.58% |

**结论**：EEGNet 128ch transfer ternary **66.23%** 高于 cross-subject baseline 61.23%
（Δ=+5.00 pp，恰在 ±5 pp 边界）。与 binary 同向（+正向迁移），但幅度更接近上限。
对 paper §3.3 而言：EEGNet 仍能从 transfer 中获得收益，但相对 cross-subject 已经
接近性能天花板，CBraMod 的 transfer 收益（如有）可能反而更显著。

### 2.6 C-2 — 8ch Band Power CBraMod transfer binary (82.02% ± 10.74%)

> **数据来源**：`results/8_channel/band_power/20260506_2159_transfer_cache_imagery_binary.json`

**Per-subject test accuracy (majority voting, Sess2 Finetune)**：

| Subject | Acc | Subject | Acc | Subject | Acc |
|---------|-----|---------|-----|---------|-----|
| S01 | 76.88% | S08 | 90.00% | S15 | 90.62% |
| S02 | 91.88% | S09 | 90.62% | S16 | 70.00% |
| S03 | 97.50% | S10 | 70.00% | S17 | 77.50% |
| S04 | 98.75% | S11 | 86.88% | S18 | 85.62% |
| S05 | 70.00% | S12 | 83.12% | S19 | 97.50% |
| S06 | 70.00% | S13 | 88.75% | S20 | 60.62% |
| S07 | 79.38% | S14 | 77.50% | S21 | 69.38% |

**结论**：C-2 = 82.02% **低于** 8ch BP cross-subject baseline 84.05%（Δ=−2.03 pp，
方向为负）。这与 paper §3.5.4 第一档位 32ch FDR transfer (+0.74 pp) 趋势**相反**。

**讨论**：原假设"通道越少 transfer 收益越大"在 8ch BP 档位被 ✗ 反例。可能解释：
- 8ch BP cross-subject 84.05% 已经接近该通道配置的容量上限，transfer 微调反而引入
  对单被试的过拟合风险
- 8ch 配置下 fine-tune 数据量（单被试 Online_Sess02_Finetune）相对参数量太小
- Band Power 本身已对单被试低频段相对鲁棒，cross-subject 已抓到主信号

|Δ|=2.03 pp 在通过准则的 ±15 pp 窗口内，run 不需要重跑；但作为反例需主对话审阅
是否在 paper 中如实呈现（对 §3.5.4 的"渐进式增益"假设构成挑战）。



---

## 3. Paper §3.5 / §3.5.3 / §3.3 / §3.5.4 论断更新建议

### 3.1 §3.5 "Channel sweet spot"
保留主论断；新数据点 64ch=89.46% 严格落在 32ch (87.71%) 与 128ch (90.68%) 之间，符合
"边际增益减弱"假设。建议在 §3.5 表格补入 64ch 行。

### 3.2 §3.5.3 "4ch 标准方法对比"
**需修订**。原 "标准方法均低于负控制" 主张被 Band Power (78.75%) 推翻。建议措辞：

> 在 4ch 极端约束下，基于体素稀疏统计的 FDR (62.08%) 与基于模型梯度的 Attention (54.70%)
> 双双失效（均低于均匀分布负控制 67.65%）；CSP 仅勉强匹配负控制 (66.99%)。但物理学动机鲜
> 明的 Band Power 标准基线 (78.75%) 仍能恢复可观判别力，提示 sub-channel 选取应优先考虑
> 频域结构而非纯统计/注意力指标。

### 3.3 §3.3 EEGNet transfer
**新数据可填入对照表**。

| 模型 | Task | Cross | Transfer | Δ |
|------|------|-------|----------|---|
| EEGNet | binary | 76.67% | **80.77%** | +4.10 pp |
| EEGNet | ternary | 61.23% | **66.23%** | +5.00 pp |

两个 task 的 Δ 同向（正向迁移）但幅度温和。建议在 §3.3 加入 EEGNet 列以呈现完整对照。
若 paper 主结论强调"CBraMod transfer 收益显著大于 EEGNet"，则可补一句：

> EEGNet 在 binary / ternary 两 task 上分别从 transfer learning 获得 +4.1 / +5.0 pp 增益，
> 远小于 CBraMod 的 [CBraMod 对应 Δ — paper §3.3 已有] 数值。这与 EEGNet 在 cross-subject
> 设置下已接近性能天花板的解释一致：进一步个性化的边际收益受限于模型容量。

### 3.4 §3.5.4 reduced-channel transfer
**结果挑战原假设**。

| 档位 | Cross | Transfer | Δ |
|------|-------|----------|---|
| 32ch FDR | 87.71% | 88.45% | +0.74 pp |
| **8ch Band Power** | **84.05%** | **82.02%** | **−2.03 pp** |

原假设"通道越少 transfer 收益越大"在 8ch BP 档位被反例。建议措辞修订为：

> 在 32ch FDR 档位 transfer 提供 +0.74 pp 的小幅增益；但在更极端的 8ch Band Power
> 档位下 transfer 反而损失 −2.03 pp。这表明"通道越少则 transfer 收益越大"的简单
> 趋势性假设并不成立——transfer 收益与通道数的关系受 cross-subject baseline 饱和
> 程度、fine-tune 数据量、特征提取方法的鲁棒性多重影响。

或者**不在 paper 中呈现 C-2 结果**，仅保留 32ch transfer 数据，避免反例引发的
解释复杂性。该决策由主对话最终判断。

---

## 4. 文件清单（新）

### 4.1 新 channel_selections JSON
- `results/64_channel/channel_selections.json` — 仅 fdr key
- `results/4_channel/channel_selections.json` — 新增 csp / band_power（保留旧 keys）
- `results/4_channel/channel_selections.json.bak_pre_e2` — 合并前快照

### 4.2 新结果 JSON cache
- `results/64_channel/fdr/20260505_2223_cross_subject_cache_imagery_binary.json` (E-1)
- `results/4_channel/csp/20260505_2246_cross_subject_cache_imagery_binary.json` (E-2a)
- `results/4_channel/band_power/20260505_2308_cross_subject_cache_imagery_binary.json` (E-2b)
- `results/20260506_2039_transfer_cache_imagery_binary.json` (E-3a)
- `results/20260506_2112_transfer_cache_imagery_ternary.json` (E-3b)
- `results/8_channel/band_power/20260506_2159_transfer_cache_imagery_binary.json` (C-2)

### 4.3 ExperimentDB 新 runs
- `20260505_2223_cross_subject_imagery_binary` (E-1, channel_config=fdr@64)
- `20260505_2246_cross_subject_imagery_binary` (E-2a, channel_config=csp@4)
- `20260505_2308_cross_subject_imagery_binary` (E-2b, channel_config=band_power@4)
- `20260506_2039_transfer_imagery_binary` (E-3a, EEGNet 128ch)
- `20260506_2112_transfer_imagery_ternary` (E-3b, EEGNet 128ch)
- `20260506_2159_transfer_imagery_binary` (C-2, CBraMod 8ch BP)

### 4.4 代码改动（autonomous fix，本批引入）
- `src/config/constants.py:83` — `SUPPORTED_CHANNEL_COUNTS` 加入 64
- `src/training/finetune_utils.py:68-84` — `load_pretrained_model` EEGNet 分支从
  `state_dict` 反推 F1/D/F2/kernel_length，避免硬编码 8/2/16 与 16/4/32 checkpoint 不兼容
- `scripts/analysis/_merge_channel_selections.py` — 新增合并辅助脚本
- **2026-05-07 增补**：`scripts/experiments/run_transfer_comparison.py`、
  `run_within_subject.py`、`run_cross_subject_comparison.py` 三个 runner 顶部（任何
  matplotlib 间接 import 之前）插入 `os.environ.setdefault('MPLBACKEND', 'Agg')`，
  把 May 6 的 launch-time `MPLBACKEND=Agg` 环境变量 workaround 升级为代码层永久修复。
  机制：matplotlib 一旦被首次 import 就 lazy-pick backend；TkAgg backend 创建的
  tkinter Variable/Image `__del__` 在 PyTorch DataLoader / CUDA worker 线程被 GC
  时触发 `Tcl_AsyncDelete` cross-thread check → `Tcl_Panic` → `abort()`（exit 3）。
  Agg 是非交互纯像素 backend，不创建 Tk 对象，杜绝该 race。

---

## 5. 已知 caveat

### 5.1 Autonomous fixes 记录

1. **E-1 触发**：`SUPPORTED_CHANNEL_COUNTS = [4, 8, 32, 61, 128]` 不含 64，argparse 拒
   绝 `--channels 64`。修复：加入 64，不破坏现有默认。
2. **E-3a v1 失败**：`load_pretrained_model` 硬编码 `F1=8, D=2, F2=16`，对 16/4/32
   checkpoint 形状不符。修复：从 state_dict 张量形状反推。
3. **E-3a v1 二次失败**：matplotlib Tk backend 在 worker thread 中触发
   `Tcl_AsyncDelete: async handler deleted by the wrong thread` 致命 abort（S05 死亡，
   S03/S04 已完成）。修复：`MPLBACKEND=Agg` 启动。
4. **psutil 缺失**：overwatch.py 首次启动失败。修复：`uv pip install psutil`。

### 5.2 Skipped / failed tasks
- 无 task 跳过；全部 6 个 run 完整 N=21。
- E-3a v1 (`20260505_2321`) 因 Tcl_AsyncDelete 提前死亡；v2 (`20260506_2039`) 修复后完整完成，用 v2 数据。

### 5.3 反例提示（main 对话审阅时关注）
1. **§3.5.3 "标准方法均失效" 论断需修订**：4ch Band Power = 78.75% **超过**负控制 67.65%
   (Δ=+11.10 pp)，反例。
2. **§3.5.4 "通道越少 transfer 收益越大" 假设需修订**：8ch BP transfer = 82.02% **低于**
   cross 84.05%（Δ=−2.03 pp），与 32ch FDR transfer 的 +0.74 pp 趋势相反。

### 5.4 2026-05-07 Reproducibility 重跑确认

May 7 主对话恢复后，将 launch-time `MPLBACKEND=Agg` workaround 升级为永久代码层
修复（详见 §4.4）后，对 E-3a / E-3b / C-2 三个 transfer 实验重跑一次以验证
（a）Agg 修复在源码层稳定生效，（b）May 6 数字非随机偶然产物。

| Task | May 6 mean | May 7 mean | Δ | May 7 run_tag | 一致性 |
|------|-----------|-----------|---|---------------|-------|
| E-3a EEGNet 128ch xfer binary | 80.77% | **82.05%** | +1.28 pp | `20260507_1835` | 在 stochastic 噪声范围 |
| E-3b EEGNet 128ch xfer ternary | 66.23% | **66.33%** | +0.10 pp | `20260507_1913` | 几乎完全一致 |
| C-2 8ch BP CBraMod xfer binary | 82.02% | **82.08%** | +0.06 pp | `20260507_1958` | 几乎完全一致 |

三次重跑均无 Tcl_AsyncDelete / Tcl_Panic（May 6 启动前曾在 `20260505_2318` 与
`20260505_2321` 两次 abort），确认源码层 Agg 修复生效。三个 task 的均值偏移
都 ≤1.3 pp，远小于实验 std (~10 pp)，全部论断、数字与 paper 修订建议保持
不变。本批 canonical 数据点保留 May 6 run（首次成功复现）；May 7 run 作为
独立重现性证据收录于 ExperimentDB（`is_baseline=0`），不替换任何 baseline。

---

## 6. 汇报触发条件

本批完成后，主对话 review 后再决定是否：
- 把 64ch FDR 注册为 baseline
- 把 4ch CSP / Band Power 注册为 baseline（但 CSP 几乎等于负控制，可能不必）
- 是否补 64ch 的其他方法（CSP / Attention / Band Power 64ch）
- 是否针对 §3.5.3 重写措辞

---

## 7. Run 启动命令记录

```bash
# E-1 64ch FDR selection (5 min)
uv run python scripts/analysis/compute_channel_selections.py \
    --n-channels 64 --methods fdr --paradigm imagery --task binary

# E-1 64ch CBraMod cross binary
uv run python scripts/run_cross_subject_comparison.py \
    --task binary --paradigm imagery --channels 64 --channel-config fdr \
    --models cbramod --cache-only --no-wandb --no-plot

# E-2 4ch CSP + Band Power selection (合并到现有 JSON)
uv run python scripts/analysis/compute_channel_selections.py \
    --n-channels 4 --methods csp band_power --paradigm imagery --task binary
# 然后 _merge_channel_selections.py 合并

# E-2a 4ch CSP cross binary
uv run python scripts/run_cross_subject_comparison.py \
    --task binary --paradigm imagery --channels 4 --channel-config csp \
    --models cbramod --cache-only --no-wandb --no-plot

# E-2b 4ch Band Power cross binary
uv run python scripts/run_cross_subject_comparison.py \
    --task binary --paradigm imagery --channels 4 --channel-config band_power \
    --models cbramod --cache-only --no-wandb --no-plot

# E-3a EEGNet 128ch transfer binary (with MPLBACKEND=Agg)
MPLBACKEND=Agg uv run python scripts/run_transfer_comparison.py \
    --task binary --paradigm imagery --models eegnet \
    --pretrained-eegnet checkpoints/cross_subject/20260330_0709_eegnet_imagery_binary/best.pt \
    --cache-only --no-wandb --no-plot

# E-3b EEGNet 128ch transfer ternary (TBD)
MPLBACKEND=Agg uv run python scripts/run_transfer_comparison.py \
    --task ternary --paradigm imagery --models eegnet \
    --pretrained-eegnet checkpoints/cross_subject/20260330_0735_eegnet_imagery_ternary/best.pt \
    --cache-only --no-wandb --no-plot

# C-2 8ch Band Power CBraMod transfer binary (TBD)
MPLBACKEND=Agg uv run python scripts/run_transfer_comparison.py \
    --task binary --paradigm imagery --channels 8 --channel-config band_power \
    --models cbramod \
    --pretrained-cbramod checkpoints/cross_subject/20260331_1950_cbramod_imagery_binary/best.pt \
    --cache-only --no-wandb --no-plot
```
