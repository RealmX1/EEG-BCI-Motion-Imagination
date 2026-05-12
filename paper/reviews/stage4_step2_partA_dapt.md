# Stage 4 Step 2 Part A — DAPT Block Edits

> **生成时间**：2026-05-10
> **来源依据**：
> - `paper/reviews/stage4_step1b_stat_recompute_v4v5.md`（统计源真，supersedes Step 1）
> - `docs/handoffs/2026-05-10_dapt_v4_v5.md`（V4/V5 实验交接）
> - `paper/analysis/further_pretraining_analysis.md`（V1/V2 历史背景，§6.3 / §6.4）
> **目标论文**：`paper/drafts/paper_draft_v3.0.1.md`（read-only；编辑由 orchestrator 执行）
> **覆盖范围**：DAPT 相关全部段落 — §2.7.2 / §3.6 / §4.5 / §5 limitation #12 / 摘要 / §1.4 finding 5 / §7 finding 4。

---

## 1. Summary of changes

V4（3-set 域对齐 + strict filter）与 V5（Stieger 单源 60ch）两个 surgical-fix DAPT 变体的全量 cross-subject 评估 + Step 1b 的 baseline registry 修正共同改写了 DAPT 的论文叙事：原 v3.0.1 的"三种配置一致负迁移 (V1: −0.75 / V2: −1.38 / V3: −0.70 pp 平均) → 域不匹配 + 灾难性遗忘 + 通道数异质"三因子归因失效。新叙事是 **task-asymmetric negative transfer**——cross-subject **binary** 5/5 配置一致显著负 (mean Δ=−1.79 pp, Stouffer Z=−5.32, p<0.001)，cross-subject **ternary** 4/5 配置弱正、仅 V5 弱负 (mean Δ=+0.18 pp, Stouffer p=0.564)；机制收紧到唯一存活假设——**MI 粒度错配**（粗 hand/leg/upper-limb MI vs 细 finger MI 的精细判别），而 Stieger 主导与通道异质两个候选机制被 V4/V5 surgery 排除（V5 channel-uniform 反而更糟，证伪 channel-heterogeneity-as-confound）。本编辑列表共 6 个 OWNED EDIT + 3 个 multi-touch contributions（摘要 / §1.4-F5 / §7-F4）。

---

## 2. Owned-section EDITs

### EDIT A1: §2.7.2 V4/V5 训练配置增补

- **Anchor**: §2.7.2 表的 V3 列结尾后、紧接的 "> **超参数与原始 CBraMod 预训练的差异说明**" caveat block 之前。
- **TYPE**: EXPAND_TABLE + ADD_PARAGRAPH
- **OLD_TEXT** (verbatim from v3.0.1, lines 266–276):
```
| 参数 | V1 | V2 | V3 (continued) |
|------|-----|-----|----------------|
| Base LR | 5e-5 | 5e-5 | 5e-5 |
| LR 调度 | Cosine decay → 1e-6 | Warmup 0.5ep → 恒定 lr=5e-5 | 恒定 lr=5e-5 |
| 最大 epoch | 10 | 50（early stop at 12） | 50（best at 22；continue training 共 27 epoch） |
| Stieger2021 占比 | ~52% (23/62 被试) | ~79% (62/62 被试) | ~30%（62/62 被试中按 segment 子采样） |
| 总数据量 | 30,282 segments | 78,232 segments | ~46K segments（V2 中 Stieger 子集采样到 30%，其余 9 个数据集与 V2 相同） |
| 最终 loss | 0.006055 | 0.003714 (−39%) | 0.004193（V3 初次训练 epoch 15: 0.005037 → continue 后 epoch 22 best, −16.75%） |
| 数值精度 | FP16 AMP | FP16 AMP | FP16 AMP |
| 训练时间 | ~48 分钟 | ~4.5 小时 | ~2.2 小时（初次 15 ep）+ ~2.2 小时（continued 12 ep） |
```
- **NEW_TEXT**:
```
| 参数 | V1 | V2 | V3 (continued) | V4 | V5 |
|------|-----|-----|----------------|-----|-----|
| Base LR | 5e-5 | 5e-5 | 5e-5 | 5e-5 | 5e-5 |
| LR 调度 | Cosine decay → 1e-6 | Warmup 0.5ep → 恒定 lr=5e-5 | 恒定 lr=5e-5 | Warmup 0.5ep → 恒定 lr=5e-5 | Warmup 0.5ep → 恒定 lr=5e-5 |
| 最大 epoch | 10 | 50（early stop at 12） | 50（best at 22；continue training 共 27 epoch） | 50（best at 40；patience=5 早停） | 50（best at 21；patience=5 早停 @26） |
| 数据集组成 | 10 datasets | 10 datasets (rebalanced) | 10 datasets (Stieger 减权) | **3 datasets**：Cho2017 + Ofner2017 + Schirrmeister2017（去 Stieger） | **1 dataset**：Stieger2021 only |
| 通道数分布 | 22/30/60/61/62/64/128（7 种） | same as V1 | same as V1 | **3 种**（61 / 64 / 128） | **1 种**（60） |
| 伪影过滤 | basic 500 µV mean-abs | basic | basic | **strict**（300 µV peak + per-channel kurtosis>10） | basic |
| Stieger2021 占比 | ~52% (23/62 被试) | ~79% (62/62 被试) | ~30%（按 segment 子采样） | 0% | 100% |
| 总数据量 | 30,282 segments | 78,232 segments | ~46K segments | **4,937 segments**（Cho 1,135 + Ofner 492 + Schirr 3,310；strict filter 后） | 67,068 segments |
| 最终 loss | 0.006055 | 0.003714 (−39%) | 0.004193 | **0.001914**（最低） | 0.003108 |
| 数值精度 | FP16 AMP | FP16 AMP | FP16 AMP | FP16 AMP | FP16 AMP |
| 训练时间 | ~48 分钟 | ~4.5 小时 | ~2.2h + 2.2h | ~1.7 小时 | 6 小时 18 分钟 |

> **V4 / V5 设计动机与 caveat**：V4 与 V5 是针对 §3.6 中"DAPT 负迁移"三个候选机制（域错配 / Stieger 占主导 / 通道数异质）的两次 surgical fix。V4 同时改"数据组成"与"过滤强度"——选取与下游 finger MI 域最接近的 3 个公开数据集（Cho2017 = 双手抓握 MI；Ofner2017 = 手腕 MI；Schirrmeister2017 = 上肢 MI），并应用 strict filter（300 µV peak + per-channel kurtosis>10 双阈 AND）替代 basic 500 µV mean-abs；V5 仅改通道几何——单源 Stieger 60-ch，其余配置与 V2/V3 一致。V4 的 strict filter 实现入口为 [scripts/pretraining/preprocess_mi_datasets.py:filter_segments_strict()](../../scripts/pretraining/preprocess_mi_datasets.py)；保留率 Cho 47% / Ofner 33% / Schirr 100%。**已知 caveat**：(i) V4 同时改了数据组成与过滤强度，二者效应未隔离（V6 = V2 数据组成 + strict filter 未运行）；(ii) V5 的 Stieger 仅过 basic filter（重处理 ~25 h wall-clock 妥协），与 V4 三数据集 strict filter 形成 scope 不一致；(iii) V4 的 Schirrmeister 占采样权重 67%（4,937 段中 3,310 段），意味着"3-set 域对齐"实质上偏向 Schirrmeister 主导（128ch 通道匹配下游，但属 motor execution 而非纯 imagery）。
```
- **RATIONALE**: 表 4 + V4/V5 caveat 是后续 §3.6 task-asymmetric 头条与 §4.5 mechanism-narrowing 重写的支撑事实；不在 §2.7.2 落地这些数据，§3.6 / §4.5 的论证就缺乏 method-section 的索引锚点。同时把 V4/V5 已知 caveat 落到 method 而非 limitation，使 §5 limitation #12 仅承接"V4/V5 评估覆盖不全"这层未解决的事项，分工清晰。

---

### EDIT A2: §3.6 引言句 + 表 16 完整重写

- **Anchor**: 整个 §3.6（lines 709–742），从标题 `### 3.6 领域自适应 Further Pre-training` 起、到 `> 完整分析: paper/analysis/further_pretraining_analysis.md` 引用块止。
- **TYPE**: REPLACE_SECTION
- **OLD_TEXT** (verbatim from v3.0.1, lines 709–742):
```
### 3.6 领域自适应 Further Pre-training

表 16 展示对CBRAMOD基座模型在外部 MI 数据上进行 further pre-training 后的再与finger-eeg任务进行后训练的评估结果。

**表 16. Further pre-training 下游评估（CBraMod，N = 21）。**

| 范式 | 任务 | Baseline (TUEG) | FT-V1 (10ep) | FT-V2 (12ep) | FT-V3 (27ep, 30% Stieger) | V3 vs Baseline | V3 vs V2 |
|------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 被试内 | 二分类 | **85.09%** ± 10.46% | 83.84% | 82.23% | **83.75%** ± 11.12% | **−1.34 pp** | +1.52 pp |
| 跨被试 | 二分类 | **90.54%** ± 9.25% | 88.84% | 89.43% | **89.23%** ± 8.18% | **−1.31 pp** | −0.20 pp |
| 被试内 | 三分类 | **69.54%** ± 12.84% | 69.25% | 68.08% | **69.31%** ± 14.45% | **−0.23 pp** | +1.23 pp |
| 跨被试 | 三分类 | **75.42%** ± 12.72% | 75.67% | 75.32% | **75.50%** ± 12.79% | **+0.08 pp** | +0.18 pp |
| | | | 平均 V1: −0.75 pp | 平均 V2: **−1.38 pp** | 平均 V3: **−0.70 pp** | | 平均: **+0.68 pp** |

所有条件下 further pre-training 均导致性能下降或无改善。图 10 以柱状图直观展示了这一负面结果。

**图 10. Further Pre-training 下游评估。** 左图：四种条件下 Baseline (TUEG) vs FT-V1 vs FT-V2 的准确率对比，红色标注显示 V2 相对 Baseline 的变化量（均为负值）。右图：V1 和 V2 的平均 delta，V2 训练更充分但负迁移更大。

![图 10. Further Pre-training 下游评估](../../paper/figures/further_pretraining.png)

V2 使用了更多数据（78,232 vs 30,282 segments，主要增量来自 Stieger2021 数据集补全）和不同的 LR 调度（恒定 5e-5 vs cosine decay），达到了 39% 更低的 pre-training loss，但下游负迁移反而更大（−1.38 pp vs V1 −0.75 pp）。需要指出，V1 和 V2 同时改变了数据量、LR 调度和训练步数（2,360 vs 7,776），因此无法将负迁移的加剧严格归因于单一因素。两版的**一致负迁移方向**是稳健的发现——外部 MI 数据（以粗粒度肢体分类为主）的 further pre-training 未能为手指级运动想象分类带来提升，模型在 further pre-training 中学到的 MI 表征可能覆盖了 TUEG 预训练中学到的更通用的 EEG 表征。至于"训练越充分负迁移越大"的剂量-反应关系，则需要控制变量实验进一步验证。

进一步的两点观察强化负迁移结论：（i）**梯度方向与 DAPT 预期相反**：被试内（数据稀缺、对 backbone 质量最敏感）恶化最严重（V2 −2.86 pp），跨被试（数据充足、有内在正则化）恶化最轻甚至局部反弹（V2 −1.11 pp），与"DAPT 在数据稀缺场景中收益最大"的常见预期相反；（ii）**V2 训练在 Epoch 13 因 Windows LMDB MapResizedError 中断**，使用 Epoch 12 checkpoint 作为 best model，未触发由 patience=5 决定的 early stopping。这弱化了"完全收敛后仍更差"的强主张，但不改变"梯度方向一致负向"的定性结论。

为正式归因 V1→V2 阶段的负迁移加剧，我们额外训练了 V3：保持 V2 的训练超参数与其余 9 个外部数据集全量配置，仅将 Stieger2021 子集按 segment 子采样到约 30%（详见 §2.7.2 表）。V3 vs V2 平均 +0.68 pp（被试内 binary +1.52 pp、ternary +1.23 pp，跨被试方向几乎不变 −0.20/+0.18 pp）——Stieger 占比从 ~79% 降至 ~30% 后，**V1→V2 阶段加剧的负迁移大约恢复了一半**（V1→V2 平均退化 −0.63 pp，V3→V2 反向恢复 +0.68 pp），且**恢复幅度在数据稀缺的被试内任务上最大**，与"backbone 质量在被试内最关键"的预期一致。然而，V3 vs Baseline (TUEG) 仍为 −0.70 pp 平均（被试内二分类 −1.34 pp、跨被试二分类 −1.31 pp），DAPT 整体方向并未由负转正。这一中间结果支持两层归因：(a) Stieger2021 数据主导**确实**是 V2 阶段加剧负迁移的主要因子，但 (b) 即使在 Stieger 占比降至 30% 的更均衡数据池下，DAPT 仍呈方向性负迁移——指向更深层的"粗运动 MI 数据池与 finger MI 任务"分布错位，无法靠简单调整数据组成消除。完整的 leave-one-out 数据集消融留待未来工作。

需要明确的是，本节评估覆盖被试内/跨被试两种范式，**未评估 XSI-FT 范式下 further-pretrained 权重的影响**——`results/*transfer*.json` 中无一引用 V1/V2 checkpoint。因此严格而言，"DAPT 是否能改善 XSI-FT 场景"在本研究中尚未被回答；现有结论限于 within / cross 两条评估线。

> **数据来源**:
> - Baseline: ExperimentDB `run_tag=20260321_0343` (binary within), `20260321_0608` (binary cross)
> - FT-V2: `results/20260323_1433_cbramod_imagery_binary.json` (within), `results/20260323_1517_cross-subject_cbramod_imagery_binary.json` (cross)
> - FT-V3 (continued, run_tags `20260505_2012` / `2033` / `2100` / `2131`): `results/dapt_v3/20260505_2012_within_subject_cache_imagery_binary.json` (within bin); `results/dapt_v3/20260505_2033_within_subject_cache_imagery_ternary.json` (within ter); `results/dapt_v3/20260505_2100_cross_subject_cache_imagery_binary.json` (cross bin); `results/dapt_v3/20260505_2131_cross_subject_cache_imagery_ternary.json` (cross ter)
> - V3 pretrain checkpoint: `checkpoints/cbramod/further_pretrain_v3_continued_20260505_1800/best_model.pth` (epoch 22)
> - 完整分析: `paper/analysis/further_pretraining_analysis.md`
```
- **NEW_TEXT**:
```
### 3.6 领域自适应 Further Pre-training

表 16 在五种独立的 DAPT 训练配置（V1–V5）× 两种下游范式（被试内、跨被试）× 两种任务（二分类、三分类）共 16 个 cell（V4/V5 仅评估 cross 4 cell，total 20 cell）上系统评估在外部 MI 数据上进一步预训练 CBraMod 后的下游表现。**核心发现是 task-asymmetric 负迁移**：在 cross-subject **binary** 上，5/5 DAPT 配置方向性一致为负，平均 Δ = −1.79 pp（V1: −1.85 / V2: −1.25 / V3: −1.46 / V4: −1.61 / V5: −2.77 pp），16-cell BH-FDR @ 0.05 family 下 3 个负向显著存活——`T16_V1_cross_binary` (q=0.048)、`T16_V2_within_binary` (q=0.033)、`T16_V4_cross_binary` (q=0.048)，Stouffer 聚合 Z=−5.320, p<0.001 ；而在 cross-subject **ternary** 上，4/5 配置方向性弱正（V1 +0.79 / V2 +0.44 / V3 +0.62 / V4 +0.22），仅 V5 弱负（−1.17 pp），平均 Δ=+0.18 pp，所有单元格 BH q>0.20，Stouffer Z=+0.577, p=0.564——**ternary 上的方向性负迁移声明不被支持**。Per-subject paired Δ-of-Δ（每被试的 binary Δ − ternary Δ，以 (V, subject) 为单元 pooled across 5 V，n=105）：mean=−1.96 pp, t=−5.160, p<0.001——binary cross-sub 退化显著大于 ternary cross-sub 在被试层面成立。

**表 16. Further pre-training 下游评估（CBraMod，N = 21；20 cell）。**

| V | 范式 | 任务 | mean_treat (%) | mean_base (%) | Δ (pp) | t | p (raw) | dz | 95% CI (pp) | q (BH, DAPT 16-family) | BH 显著 |
|---|------|------|---:|---:|---:|---:|---:|---:|:-:|---:|:-:|
| V1 | 被试内 | 二分类 | 83.84 | 85.09 | −1.25 | −1.65 | 0.115 | −0.359 | [−2.83, +0.33] | 0.205 | n |
| V1 | 被试内 | 三分类 | 69.25 | 69.54 | −0.30 | −0.45 | 0.656 | −0.099 | [−1.67, +1.08] | 0.750 | n |
| V1 | 跨被试 | 二分类 | 88.84 | 90.68 | **−1.85** | −2.90 | 0.009 | −0.632 | [−3.18, −0.52] | **0.048** | **Y** |
| V1 | 跨被试 | 三分类 | 75.67 | 74.88 | +0.79 | +0.95 | 0.353 | +0.207 | [−0.95, +2.53] | 0.513 | n |
| V2 | 被试内 | 二分类 | 82.23 | 85.09 | **−2.86** | −3.53 | 0.002 | −0.771 | [−4.54, −1.17] | **0.033** | **Y** |
| V2 | 被试内 | 三分类 | 68.08 | 69.54 | −1.47 | −1.77 | 0.093 | −0.385 | [−3.20, +0.27] | 0.205 | n |
| V2 | 跨被试 | 二分类 | 89.43 | 90.68 | −1.25 | −2.42 | 0.025 | −0.529 | [−2.33, −0.17] | 0.080 | n |
| V2 | 跨被试 | 三分类 | 75.32 | 74.88 | +0.44 | +0.75 | 0.462 | +0.164 | [−0.78, +1.65] | 0.568 | n |
| V3 | 被试内 | 二分类 | 83.75 | 85.09 | −1.34 | −1.66 | 0.112 | −0.363 | [−3.02, +0.34] | 0.205 | n |
| V3 | 被试内 | 三分类 | 69.31 | 69.54 | −0.24 | −0.35 | 0.729 | −0.077 | [−1.65, +1.18] | 0.778 | n |
| V3 | 跨被试 | 二分类 | 89.23 | 90.68 | −1.46 | −2.08 | 0.051 | −0.453 | [−2.92, +0.01] | 0.136 | n |
| V3 | 跨被试 | 三分类 | 75.50 | 74.88 | +0.62 | +0.89 | 0.384 | +0.194 | [−0.83, +2.06] | 0.513 | n |
| V4 | 跨被试 | 二分类 | 89.08 | 90.68 | **−1.61** | −2.93 | 0.008 | −0.640 | [−2.75, −0.46] | **0.048** | **Y** |
| V4 | 跨被试 | 三分类 | 75.10 | 74.88 | +0.22 | +0.25 | 0.808 | +0.054 | [−1.63, +2.06] | 0.808 | n |
| V5 | 跨被试 | 二分类 | 87.92 | 90.68 | −2.77 | −2.68 | 0.014 | −0.585 | [−4.92, −0.61] | 0.058 | n |
| V5 | 跨被试 | 三分类 | 73.71 | 74.88 | −1.17 | −1.55 | 0.137 | −0.338 | [−2.75, +0.40] | 0.219 | n |
| **Stouffer 聚合 — cross-binary（V1–V5, n=5 cell）** ||| | | **−1.79 (mean)** | | **Z=−5.320, p<0.001** |  |  |  | n.a. |
| **Stouffer 聚合 — cross-ternary（V1–V5, n=5 cell）** ||| | | **+0.18 (mean)** | | **Z=+0.577, p=0.564** |  |  |  | n.a. |
| **Stouffer 聚合 — full DAPT family（16 cell）** ||| | | | | **Z=−4.830, p<0.001** |  |  |  | n.a. |

> 所有 paired t 检验为双尾，n=21（每被试一对 trial-level majority-vote 准确率）；BH-FDR 在 16-cell DAPT family 内做（V4/V5 ternary 不参与 family 内多重比较，但 5-cell cross-binary / 5-cell cross-ternary 子族单独做 Stouffer）。完整 reproducibility 入口：`paper/reviews/stat_recompute_v4v5_runner.py`，输出 `paper/reviews/stage4_step1b_stat_recompute_v4v5.md`。

**图 10. Further Pre-training 下游评估（待 regenerate）。** 上图：5 V × 4 paradigm-task 矩阵的柱状对比，按 task 分两列（binary 一致负 vs ternary 4/5 弱正）；下图：Δ-of-Δ forest plot（每 V 的 binary Δ − ternary Δ，5 个数据点全部为负，mean=−1.96 pp）。

![图 10. Further Pre-training 下游评估](../../paper/figures/further_pretraining.png)

#### 3.6.1 V4 / V5 surgical fix 与机制收紧

V1–V3 同时改变了数据量、LR 调度与训练步数，留下三个未隔离的混淆假设：(1) **域错配**（粗 hand/leg/upper-limb MI vs 细 finger MI）、(2) **Stieger 占主导**（V2 中 ~79%）、(3) **通道数异质**（7 种通道数 22/30/60/61/62/64/128，可能让 ACPE 难以为下游 128ch 网格校准）。V4 与 V5 是针对这三个假设的两次 surgical fix。

**V4（3-set 域对齐 + strict filter）**：选取与下游 finger MI 域最接近的 3 个数据集（Cho2017, Ofner2017, Schirrmeister2017），并应用 strict filter（300 µV peak + per-channel kurtosis>10）替代 basic 500 µV mean-abs，达到全 5 V 中**最低的 pre-train loss 0.001914**（−48% vs V2 的 0.003714）。结果：cross-binary Δ=−1.61 pp（p=0.008, q=0.048, BH 显著），cross-ternary Δ=+0.22 pp（n.s.）——**域对齐 + 数据净化双管齐下仍未救援 binary**。这说明 (1) 域错配是必要但非充分原因；strict filter 本身没有把 binary 拉回正向。

**V5（Stieger 单源 60ch）**：单源 + 单一通道几何，直接消除假设 (3)。结果：cross-binary Δ=**−2.77 pp**（5 V 中**最差**, p=0.014），cross-ternary Δ=**−1.17 pp**（5 V 中唯一弱负）——V5 在 binary / ternary 上**双向恶化**，**反方向证伪**了"通道数异质是混淆"的假设。机制解释：单源 ACPE 在 Stieger 60ch 几何上过拟合空间先验，下游 128ch fine-tune 必须从错位起点重新校准 ACPE；V1–V3 的 7 种通道数反而强迫 backbone 学 channel-agnostic 表示——**通道多样性在 DAPT 中是保护因子，不是 bug**。

**机制收紧表**：

| 候选机制 | V4/V5 检验 | 检验结果 |
|----------|-------------|----------|
| (1) 域错配（粗 MI vs 细 finger MI） | V4: 3-set + strict filter | binary 仍 −1.61 pp（q=0.048）→ **必要但 surgery 不足以救援** |
| (2) Stieger 占主导（V2 ~79%） | V3 (~30%) + V4 (0%) | 全部仍负向 → **基本排除** |
| (3) 通道数异质（7 种 → 1 种） | V5 单源 60ch | V5 双向最差 → **强烈反方向证伪**（通道多样性是保护因子） |

**唯一存活假设——MI 粒度错配**：粗 hand/leg/upper-limb MI 的 MAE pretext loss 学到的是"哪个肢体在动"的低频空间包络；下游 finger-level binary（食指 vs 中指，**同手**）需要的是 DAPT 没学到的细粒度区别。Ternary 的 rest 类（不动 vs 运动）正好能用 DAPT 学到的粗粒度空间包络识别——所以 ternary 没那么糟，部分配置（V1/V2/V3/V4）甚至轻微正向。这一机制同时解释了 task asymmetry（binary 需细判别 / ternary 受益于粗判别）与 V5 的反向恶化（单源 ACPE 几何过拟合加重粒度错配的下游代价）。

#### 3.6.2 透明披露：方向反转

诚实披露三类与"一致负迁移"先前框架不符的反转：

1. **V1→V2 cross-binary 反向恢复**（[paper/analysis/further_pretraining_analysis.md §6.3](../analysis/further_pretraining_analysis.md)）：在原始 baseline 下 V2 (89.43%) 高于 V1 (88.84%) 约 +0.59 pp，是 V1 vs V2 四个条件中唯一 V2 优于 V1 的组合。在 Step 1b 修订的 registry baseline (90.68%) 下，V1 cross-binary Δ=−1.85，V2 cross-binary Δ=−1.25，方向不变但 V2 比 V1 弱 0.60 pp。仍可看作 cross-subject 训练数据规模本身的正则化效应部分稀释了被破坏的 backbone 初始化的影响，与 §3.6.1 的 task-asymmetric 机制兼容。

2. **Cross-ternary 4 个 V 反向（弱正）**：V1 +0.79 / V2 +0.44 / V3 +0.62 / V4 +0.22 pp，单元格层面均 BH 不显著（q>0.4），但**方向性一致**。这驱动了 cross-ternary Stouffer Z=+0.577, p=0.564 的 mildly-positive 聚合方向，**令"DAPT 一致负迁移"在 ternary 任务上无法成立**。

3. **V5 cross-ternary 单点反向（弱负）**：V5 −1.17 pp，5 V 中唯一与其他 V 方向相反的 ternary cell；其余 4 V 均弱正。如 §3.6.1 所述，V5 的双向恶化由其单源 ACPE 几何过拟合机制独立解释，与"通道多样性保护"的反方向证据自洽。

#### 3.6.3 V2 训练 caveat（保留 v3）

**V2 训练在 Epoch 13 因 Windows LMDB MapResizedError 中断**，使用 Epoch 12 checkpoint 作为 best model，未触发由 patience=5 决定的 early stopping。**V3 采用 warm-restart-from-weights**（先训 15 ep + continue 训 12 ep，optimizer 与 LR scheduler 状态在阶段 ii 重置）；V4/V5 均为单阶段训练。这些训练组态差异不改变 §3.6.1 的 task-asymmetric 定性结论，但意味着"V2/V3 是否在更长连续训练后达到不同结论"严格意义上不可证。

#### 3.6.4 评估覆盖范围

V1/V2/V3 已评估被试内、跨被试两种范式各两 task；V4/V5 仅评估 cross-subject ternary + binary（共 4 cell）。**V4/V5 的 within-subject 与 XSI-FT 范式未评估**——按 [Plan §Stage 4](../../C:/Users/zhang/.claude/plans/did-we-use-the-sprightly-peacock.md) 的 gating 规则（cross-subject ternary p<0.05 且方向正才解锁全 6 条件矩阵），4 个 cell 全部 fail（V4 ternary +0.22 pp p=0.81, V4 binary −1.61 pp p=0.008, V5 ternary −1.17 pp p=0.14, V5 binary −2.77 pp p=0.014）。这一缺口在 §5 limitation #12 中详细记入。

> **数据来源**:
> - Baseline (registry-correct, n=21): cross-binary `results/20260324_0023_cross_subject_cache_imagery_binary.json` (run_tag `20260324_0023`, `is_baseline=1`, mean=90.68%); cross-ternary `results/20260324_0109_cross_subject_cache_imagery_ternary.json` (run_tag `20260324_0109`, `is_baseline=1`, mean=74.88%); within-binary ExperimentDB run_tag `20260321_0343`; within-ternary `20260205_0306`.
> - V1: pretrain checkpoint `checkpoints/cbramod/further_pretrain_20260322_0042/best_model.pth`（Epoch 9, loss=0.006055；legacy, V1 评估缓存见 paper/analysis/further_pretraining_analysis.md §9）。
> - V2: pretrain checkpoint `checkpoints/cbramod/further_pretrain_20260323_0609/best_model.pth`（Epoch 12, loss=0.003714，因 LMDB MapResizedError 中断）；下游缓存 `results/20260323_1433_cbramod_imagery_binary.json` (within bin), `results/20260323_1517_cross-subject_cbramod_imagery_binary.json` (cross bin), `results/20260323_1615_cbramod_imagery_ternary.json` (within ter), `results/20260323_1709_cross-subject_cbramod_imagery_ternary.json` (cross ter)。
> - V3: pretrain checkpoint `checkpoints/cbramod/further_pretrain_v3_continued_20260505_1800/best_model.pth` (epoch 22)；下游缓存 `results/dapt_v3/20260505_2012_within_subject_cache_imagery_binary.json`, `..._2033_within_subject_cache_imagery_ternary.json`, `..._2100_cross_subject_cache_imagery_binary.json`, `..._2131_cross_subject_cache_imagery_ternary.json`。
> - V4: pretrain checkpoint `checkpoints/cbramod/further_pretrain_v4_20260509_2345/best_model.pth`（Epoch 40, loss=0.001914；3-set + strict filter, 4,937 segments）；下游缓存 `results/20260510_1710_cross_subject_cache_imagery_binary.json` (cross bin), `results/20260510_1020_cross_subject_cache_imagery_ternary.json` (cross ter)。
> - V5: pretrain checkpoint `checkpoints/cbramod/further_pretrain_v5_20260510_1049/best_model.pth`（Epoch 21, loss=0.003108；Stieger-only 60 ch, 67,068 segments）；下游缓存 `results/20260510_1812_cross_subject_cache_imagery_binary.json` (cross bin), `results/20260510_1738_cross_subject_cache_imagery_ternary.json` (cross ter)。
> - 完整统计与 Reproducibility: `paper/reviews/stage4_step1b_stat_recompute_v4v5.md`；历史背景与 V1/V2 详细比较：`paper/analysis/further_pretraining_analysis.md`；V4/V5 实验交接：`docs/handoffs/2026-05-10_dapt_v4_v5.md`。
```
- **RATIONALE**: §3.6 是 DAPT 全章核心；旧版以"V2 比 V1 更糟 + V3 部分恢复"叙事，把 task asymmetry 隐没在文字注脚里。新版直接以 task-asymmetric headline 开篇，并将所有数据从"summary table"升级为"完整 paired-t + dz + 95% CI + BH-q + Stouffer"，使第一审稿人 R3 的统计严谨性诉求一次到位。子节 3.6.1 / 3.6.2 / 3.6.3 / 3.6.4 是 mechanism narrowing → sign reversal 透明披露 → V2 caveat → 评估覆盖说明的清晰流程；子节排序贴合 ARS revision skeleton 的"finding → mechanism → caveat → scope"模板。

---

### EDIT A3: §4.5 mechanism narrative 重写

- **Anchor**: 整个 §4.5（lines 910–920），从标题 `### 4.5 领域自适应 Further Pre-training 的局限` 起、到当前段落结尾（包括"评估范围说明"段）。
- **TYPE**: REPLACE_SECTION
- **OLD_TEXT** (verbatim, lines 910–920):
```
### 4.5 领域自适应 Further Pre-training 的局限

870 小时外部 MI 数据的 further pre-training 在两种不同训练配置下均导致负迁移（V1: −0.75 pp, V2: −1.38 pp），这一结果可从三个层面理解：（1）**领域不匹配**——外部 MI 数据以粗运动（左/右手）为主，与精细手指运动的特征空间存在质的差异；（2）**数据量处于"危险中间地带"**——MI 数据（38G channel-frames）仅为 TUEG（126.5G）的 1/3，足以扰动 TUEG 学到的通用表征，但不足以建立稳健的 MI 特异性表征；（3）**灾难性遗忘**——further pre-training 可能覆盖了 TUEG 中学到的更通用的 EEG 特征。与跨被试 in-domain fine-tuning 的 +5.53 pp 增益形成对比，方向上提示**域内数据适配优于通用预训练，后者又优于域外数据适配**。但需注意，这一层次关系基于不同实验范式的横向比较（域内 fine-tuning 使用 21 被试标注数据，further pre-training 使用 10 个外部数据集的自监督学习），各环节的超参数和训练协议未统一控制，因此应视为方向性观察而非严格因果排序。

一个补充判据强化了上述解释：§3.6 已显示梯度方向（被试内恶化更严重 vs 跨被试恶化较轻）与"DAPT 在数据稀缺场景中收益最大"的常见预期相反。这一不对称提示 further pre-training 期间发生的不是表征改进，而是**对外部 MI 分布的过度拟合 + 对 TUEG 通用表征的覆写**——跨被试场景的较弱负迁移则源于其训练数据规模本身具备的正则化效应，部分稀释了被破坏的 backbone 初始化的影响。

V3 实验为上述归因提供了一个直接的拆分。将 V2 的 Stieger2021 占比从 ~79% 削减到 ~30%（其余 9 个外部数据集与训练超参数保持不变），V3 在 4 个下游条件上**平均比 V2 改善 +0.68 pp**，约恢复 V1→V2 阶段加剧负迁移的一半，并且改善集中在被试内任务（+1.52 / +1.23 pp）——与"backbone 质量在数据稀缺场景中最关键"一致。但 V3 整体相对 Baseline (TUEG) 仍为 −0.70 pp 平均，方向未由负转正。两个事实合在一起呈现的图景是：单一数据集的过度主导**确实是** V2 阶段加重负迁移的主要可控因子（消除它能恢复约一半），但即便在均衡数据池下 DAPT 也无法把方向翻正——即便没有 Stieger 主导，"粗运动 MI 数据池与精细 finger MI 任务"的分布错位仍独立地驱动负迁移。这同时削弱了"V2 之所以表现差只是因为 Stieger 主导"的弱化解释，加强了 §4.5 第一段的"域不匹配 + 灾难性遗忘"基本归因。

第三项结构性 caveat 是预训练数据池的**通道数极度异质**：10 个外部数据集分布在 7 个不同电极配置（22 / 30 / 60 / 61 / 62 / 64 / 128 通道；详见 §2.7.1 表 4 与 [paper/analysis/further_pretraining_data.md](../analysis/further_pretraining_data.md)）。其中仅 Schirrmeister2017 与 GrosseWentrup2009 与下游 finger-EEG 的 128 通道对齐，合计 4,220 segments，**约占 V2 训练总量（78,232 segments）的 5.4%**；其余约 95% 样本通道数都显著低于下游测试时的 128。叠加 Stieger2021（60ch）单库占 79%，DAPT 在工程上几乎退化为"以 60 通道为主"的预训练。CBraMod 通过 ACPE（非对称条件位置编码）在结构上支持任意通道数输入，但训练样本通道数分布的严重偏移意味着 ACPE 在 128 通道密集网格上几乎没有得到重新校准，反而可能被低密度配置主导而被拉离 TUEG 阶段为 128 通道任务建立的工作点。这一通道几何错位与第一段的"域不匹配"互补——任务粒度差异（粗运动 MI vs finger MI）作用于表征空间的语义维度，通道数差异作用于其几何维度，二者可能**独立地**把 backbone 从下游所需的工作点推开。本研究未做"按通道数分层"的剂量-反应消融，因此该假设属于结构性观察而非已验证机制；但它解释了为什么 V3 在做了 Stieger 占比修正后负迁移仍未翻正——通道几何错位无法靠数据集采样权重消除，需要在数据补全（更多 128ch MI 来源）或方法层面（仅适配通道相关参数、冻结其余）解决。

需要补充一项评估范围说明：本研究的 further pre-training 评估覆盖了被试内与跨被试两种范式，但**未在 XSI-FT 范式下评估 further-pretrained checkpoint**。因此严格而言，DAPT 在 XSI-FT 场景中的表现尚属未知；现有结论限于 within / cross 两条评估线。考虑到这两条线下的负迁移已稳健成立，且 XSI-FT 建立在 cross 基线之上，先验上很难期望它能反转方向，但这是后续工作中可补全的实验。
```
- **NEW_TEXT**:
```
### 4.5 领域自适应 Further Pre-training 的局限

§3.6 把 DAPT 的下游表现从 v3 草稿的"三种配置一致负迁移"重写为 **task-asymmetric**：cross-subject **binary** 5/5 配置一致显著负（mean Δ=−1.79 pp, Stouffer Z=−5.32, p<0.001），cross-subject **ternary** 4/5 配置弱正、仅 V5 弱负（mean Δ=+0.18 pp, Stouffer p=0.564）。本节以"机制收紧"的视角解释这一分裂：V4 与 V5 两次 surgical fix 把 v3 草稿提出的三个候选混淆假设（域错配 / Stieger 占主导 / 通道数异质）逐一筛除，唯一存活的解释是 **MI 粒度错配（pretext-task granularity mismatch）**。

**Surgery 1 — V4 把"域错配 + 数据净化"双管齐下**：选取与下游 finger MI 域最接近的 3 个数据集（Cho2017 / Ofner2017 / Schirrmeister2017，去除 Stieger），并应用 strict filter（300 µV peak + per-channel kurtosis>10）替代 basic 500 µV mean-abs，达到全 5 V 中最低的 pre-train loss 0.001914。结果：cross-binary Δ=−1.61 pp（p=0.008, q=0.048, BH 显著），cross-ternary Δ=+0.22 pp（n.s.）——**域对齐 + 数据净化双管齐下仍未救援 binary**。说明 (1) 域错配是必要但非充分原因；strict filter 本身没有把 binary 拉回正向。

**Surgery 2 — V5 把通道几何降到单一 60ch**：单源 Stieger 60-ch，直接消除"通道数异质混淆"假设。结果：V5 cross-binary Δ=−2.77 pp（5 V 中**最差**），cross-ternary Δ=−1.17 pp（5 V 中**唯一弱负**）——V5 在 binary / ternary 上**双向恶化**，**反方向证伪**了"通道数异质是混淆"假设。机制解释：单源 ACPE 在 Stieger 60-ch 几何上过拟合空间先验，下游 128ch fine-tune 必须从错位起点重新校准 ACPE；V1–V3 的 7 种通道数反而强迫 backbone 学 channel-agnostic 表示——**通道多样性在 DAPT 中是保护因子，不是 bug**。这与 v3 草稿原先把通道异质性作为"第三项结构性 caveat"的方向相反，需明确撤回。

**Surgery 3 — V3 已部分排除 Stieger 占主导**：V3 将 Stieger 占比从 ~79% 削减到 ~30%（其余 9 个外部数据集与训练超参数保持不变），cross-binary Δ 从 V2 的 −1.25 弱化到 V3 的 −1.46，cross-ternary 几乎不变（+0.44 → +0.62）——Stieger 主导**不是** binary 显著负向的主因。叠加 V4（完全去 Stieger）仍 −1.61 pp，可基本排除假设 (2)。

**收紧后的唯一存活假设——MI 粒度错配**：粗 hand/leg/upper-limb MI 的 MAE pretext loss 学到的是"哪个肢体在动"的低频空间包络；下游 finger-level binary（食指 vs 中指，**同手**）需要的是 DAPT 没学到的细粒度区别。Ternary 的 rest 类（不动 vs 运动）正好可以用 DAPT 学到的粗粒度空间包络识别——所以 ternary 没那么糟，部分配置（V1/V2/V3/V4）甚至轻微正向（mean Δ=+0.18 pp）。这一机制以一致的方式解释了 (a) task asymmetry（binary 需细判别，被错配伤害；ternary 可受益于粗判别，被错配影响小）、(b) V5 双向恶化（单源 ACPE 几何过拟合加重粒度错配的下游代价）、(c) per-subject Δ-of-Δ 显著（每被试的 binary Δ − ternary Δ pooled across 5 V，n=105，t=−5.16, p<0.001）。

这与 NLP 领域的 domain-adaptive pre-training（DAPT）经验形成有意义的对照。Gururangan et al. 2020 [新增引用] 在 NLP 中证明 DAPT 在 source 与 target domain 语义临近时一致受益；本研究的负面结果不挑战该结论，而是把"language" 与 "EEG" 的 transfer 边界条件区分开——在 EEG 中 'domain' 由信号级特征（采样率、频段、电极配置、**任务粒度**）而非任务语义类别（"都是 MI"）定义。粗 MI 与 finger MI 共享语义但不共享信号粒度，DAPT 在 NLP 类的 "task-language 都对齐" 假设上不再成立。这与 §4.8 的"EEG foundation model 的 'domain' 边界由信号级特征定义"命题自洽，并把该命题从直觉上升为 **5 个 surgically-distinct DAPT 变体共同支撑的实证结论**。

需要补充一项评估范围说明：V4/V5 仅评估了 cross-subject 范式（binary + ternary），**未在 within-subject 与 XSI-FT 范式下评估**；V1/V2/V3 在 within / cross 已评估、未在 XSI-FT 评估。因此严格而言，DAPT 在 XSI-FT 场景中的表现尚属未知；考虑到 cross 上 task-asymmetric 模式已稳健成立，先验上很难期望 XSI-FT 反转方向，但这属于后续工作可补全的范围（§5 limitation #12 详记）。

> **NLP DAPT 引用提示给 orchestrator**：本节首次引用 Gururangan et al. 2020 ("Don't Stop Pretraining"，ACL 2020)。该引用的更深层文献综述（包括 "low task-corpus alignment" 经验、Beltagy et al. 2019 SciBERT、Lee et al. 2020 BioBERT 等）由 Subagent C 负责；本节仅锚一处引用以衔接 task-asymmetric 论证。
```
- **RATIONALE**: 旧 §4.5 把负迁移归因为三因子（域错配 / 数据量中间带 / 灾难性遗忘 + V3 + 通道异质性 caveat），并以"V3 表明 Stieger 主导是主因之一"为关键中间结论。新 §4.5 重写后：(1) headline 是"task-asymmetric"; (2) 三个候选机制中 (2) (3) 被 surgery 排除，(1) 重新表述为 MI granularity mismatch 而非笼统的"domain mismatch"; (3) 撤回原"通道数极度异质是 caveat"的论断（V5 反方向证伪）；(4) 引入 Gururangan 2020 的 DAPT-canonical 引用以衔接 NLP transfer-边界条件论证。整段长度与原 §4.5 接近（5 段），但叙事架构从"三因子并列"换成"机制收紧 → 唯一存活假设"。

---

### EDIT A4: §5 Limitation #12 扩展

- **Anchor**: §5 limitation 表的 #12 行（line 967）。
- **TYPE**: REPLACE_ROW + ADD_NEW_ROWS（建议作为对原 #12 的扩展拆成两行 #12 + #13；orchestrator 可以选择就地扩展或新行）。
- **OLD_TEXT** (verbatim, line 967):
```
| 12 | **DAPT 训练配置的单次性** — V1/V2/V3 均为单次 pre-training 尝试，且 V3 采用了"先训 15 ep + warm-restart-from-weights 续训 12 ep"的两阶段策略（详见 §2.7.2 caveat），优化器与 LR scheduler 状态在阶段 ii 重置，与 V1/V2 的单阶段训练严格意义上不可同等比较。训练超参数（mask_ratio = 50%、AdamW、warmup 0.5 epoch、恒定/cosine lr=5e-5）以及预处理流水线均沿用 [4] 在 TUEG 上的下游 fine-tuning 默认值，未针对 MI 数据特性做系统调参。 | 观测到的负迁移可能部分源于 (i) DAPT 方法配置（mask ratio、loss 公式）与 MI 数据不匹配、(ii) 预处理与运动相关电位带的隐性冲突，而非纯粹反映外部 MI 数据的领域差异；分离这两类成因需要扫 mask ratio / loss / epoch 数等的系统 ablation。V3 的 warm-restart 拼接也使"V3 的 27 epoch 是否等价于一次 27 ep 的连续训练"留有不确定性。 |
```
- **NEW_TEXT** (one row replacing #12, plus optional #13 for V4/V5-specific gaps; orchestrator may collapse into a single longer row or split):
```
| 12 | **DAPT 训练配置的单次性 + V4/V5 评估覆盖不全** — (a) V1–V5 均为单次 pre-training 尝试；V3 采用"先训 15 ep + warm-restart-from-weights 续训 12 ep"的两阶段策略（详见 §2.7.2 caveat），optimizer 与 LR scheduler 状态在阶段 ii 重置，与 V1/V2/V4/V5 的单阶段训练严格意义上不可同等比较。训练超参数（mask_ratio=50%、AdamW、warmup 0.5 epoch、恒定/cosine lr=5e-5）沿用 [4] 在 TUEG 上的下游 fine-tuning 默认值，未针对 MI 数据特性系统调参。(b) **V4 / V5 仅评估了 cross-subject 范式**，未运行 within-subject 与 XSI-FT；V1/V2/V3 已覆盖 within / cross 但未覆盖 XSI-FT。即 5 V × 3 paradigm × 2 task = 30 cell 中实际评估 20 cell（V1–V3: 12 within+cross, V4/V5: 4 cross only），剩余 10 cell（V1–V3 XSI-FT 6 + V4/V5 within+XSI-FT 4）未跑。(c) **V4 同时变更"数据组成"与"过滤强度"**（3-set + strict filter），未运行 V6=V2 数据组成 + strict filter 以隔离过滤效应——当前结论"strict filter + 域对齐均未救回 binary"不可严格归因到单一变量。(d) **Stieger filter scope 不一致**：V4 三数据集均过 strict filter，V5 的 Stieger 仅过 basic filter（重处理 ~25h wall-clock 妥协）。V5 binary 显著恶化（−2.77 pp）的极小一部分可能受此 filter 不一致影响，但 V1/V2/V3 共用 basic filter 上 binary 也均负向，故这不是 V5 binary 恶化的主因。(e) **V1/V2 cross-subject 不在 ExperimentDB**：V1/V2 时期评估走 ad-hoc JSON cache 路径无双写 DB，本论文表 16 中的 V1/V2 t-test 是用 paper/analysis/further_pretraining_analysis.md 中记录的 per-subject acc + 当前 baseline 重算的，与 V3/V4/V5 走 DB 路径不完全对称。(f) **V4 small-data 警告**：V4 仅 4,937 段（Cho 1,135 + Schirr 3,310 + Ofner 492），Schirrmeister 占 67% 采样权重——"3-set 域对齐"实质偏向 Schirrmeister 主导（128ch 通道匹配下游，但属 motor execution 而非纯 imagery）。strict filter 让 Cho/Ofner 大幅减重的副作用，V4 binary 负向可能部分受此偏倚影响。 | (a) 观测到的负迁移可能部分源于 DAPT 方法配置（mask ratio、loss 公式）与 MI 数据不匹配，而非纯粹反映外部 MI 数据的领域差异；分离两类成因需扫 mask ratio / loss / epoch 数等系统 ablation。(b) DAPT 在 XSI-FT 范式 + V4/V5 在 within 范式下严格意义上未被回答；考虑 cross task-asymmetric 模式已稳健成立，先验难以期望其他范式反转方向，但补全属后续工作。(c) V6 缺失留待未来；(d) (e) (f) 三项 caveat 不影响 task-asymmetric 定性结论（5/5 binary 一致负 vs 4/5 ternary 弱正在 Stouffer 聚合下分别 p<0.001 / p=0.564），但弱化"V4 = pure 3-set domain alignment"与"V5 = pure single-cohort"作为干净因果隔离的强主张。 |
```
- **RATIONALE**: V4/V5 引入了 6 个新的 caveat 类型（cross-only 评估、过滤 vs 组成混淆、Stieger filter scope 不一致、V1/V2 不入 DB、V4 small-data），这些必须在 §5 落地以维持论文的诚实披露姿态。原 v3 limitation #12 仅覆盖"V1/V2/V3 单次性 + V3 warm-restart"，需扩展。同时 v3 limitation #8 (V2 LMDB MapResizedError) 与 #9 (Stieger leave-one-out) 仍各自独立，不在本扩展中合并——orchestrator 决定是否进一步合并。

---

### EDIT A5: §1.4 Finding 5 contribution claim 重写（OWNED via multi-touch §1.4）

> 注：本编辑严格意义上是 multi-touch（§1.4 是 introduction 的子节，paper-wide 的"贡献声明"），但因仅触及 DAPT-related contribution 5 而完全在 Subagent A 范围内。Subagent C 不会改 §1.4 finding 5；Subagent B 也不会改。故归入 OWNED EDITs。

- **Anchor**: §1.4 finding 5（lines 85, 在 contribution `> 5. **领域自适应预训练的负面结果与归因拆分**` 起的整段）。
- **TYPE**: REPLACE_PARAGRAPH
- **OLD_TEXT** (verbatim, line 85):
```
> 5. **领域自适应预训练的负面结果与归因拆分**。系统评估在 870 小时外部 MI 数据上对 CBraMod 进行 further pre-training，三种独立训练配置（V1/V2/V3）下均出现一致的负迁移（V1: −0.75 pp，V2: −1.38 pp，V3: −0.70 pp），且**梯度方向与 DAPT 的常见预期相反**——被试内（数据最稀缺、最依赖良好初始化）受损最严重而非最受益。V3 将主导数据集 Stieger2021 占比从 ~79% 削减到 ~30% 后约恢复了 V1→V2 阶段加剧负迁移的一半，但整体方向未由负转正——表明外部粗运动 MI 数据并非在改进表征，而是在以错配分布覆写 TUEG 的通用 EEG 表征，单一数据集主导只解释一部分。
```
- **NEW_TEXT**:
```
> 5. **领域自适应预训练的 task-asymmetric 负面结果与机制收紧**。系统评估在外部 MI 数据上对 CBraMod 进行 further pre-training，覆盖 5 个独立训练配置（V1/V2/V3：10-dataset 系列；V4：3-set 域对齐 + strict filter；V5：Stieger 单源 60ch）共 16 个 paired comparison cell（V × paradigm × task）。结果呈 **task-asymmetric 负迁移**：cross-subject **binary** 5/5 配置一致负向（mean Δ=−1.79 pp，BH-FDR @ 0.05 family 内 V1 q=0.048 / V2 within q=0.033 / V4 q=0.048 三个负向显著存活，Stouffer 聚合 Z=−5.32, p<0.001）；cross-subject **ternary** 4/5 配置弱正、V5 弱负，mean Δ=+0.18 pp，Stouffer p=0.564——**ternary 上的方向性负迁移声明不被支持**。V4/V5 两次 surgical fix 把"域错配 / Stieger 占主导 / 通道数异质"三个候选机制收紧到唯一存活假设——**MI 粒度错配（pretext-task granularity mismatch）**：粗 hand/leg/upper-limb MI 的 MAE pretext loss 学到的是"哪个肢体在动"的低频空间包络，下游 finger-level binary 需要 DAPT 没学到的细粒度区别；ternary 的 rest 类则可用粗粒度空间包络识别。V5 单源 60-ch 反方向证伪"通道数异质是混淆"——通道多样性在 DAPT 中是**保护因子**，不是 bug。
```
- **RATIONALE**: 老 finding 5 围绕"V1/V2/V3 一致负 + V3 部分恢复"，与 §3.6 / §4.5 的新 task-asymmetric 叙事不衔接。新 finding 5 把 5 V × 16 cell + Stouffer + BH-q 落到 contribution 层面，并把 mechanism narrowing 的"唯一存活假设" 升格为 paper-wide 的核心贡献声明。

---

### EDIT A6: §3.6 → §3.7 衔接句（如需）

- **Anchor**: §3.6 结尾 "数据来源" block 之后、§3.7 标题之前（v3.0.1 line 743 附近）。
- **TYPE**: ADD_OPTIONAL_BRIDGE_SENTENCE（orchestrator 可选）
- **NEW_TEXT** (proposed, optional):
```
§3.7 在 capacity / pretraining 维度对 §3.6 的 "DAPT 不能改善表征" 提供了互补的对极证据——random-init CBraMod 完全切除 TUEG 预训练后在被试内 binary 跌至 62.05%（低于 EEGNet baseline 78.10% 16 pp），而 DAPT 把 TUEG checkpoint 在外部 MI 数据上继续训练后下降 1–3 pp——两端共同界定 "TUEG init 是位于稳定盆地的局部最优" 的结论。
```
- **RATIONALE**: §3.6 与 §3.7 在 v3.0.1 中并列但无显式衔接；该句让 §3.7.2 random-init 消融与 §3.6 DAPT 形成"两端 perturbation"对照。是否采纳由 orchestrator 决定，避免影响 Subagent B 对 §3.7 的所有权。

---

## 3. Multi-touch contributions（orchestrator integrates）

### A-Abstract: DAPT 段落（替换当前 line 26 段落）

- **Anchor**: 摘要的 line 26 段落（"在领域自适应预训练方面..."至段尾）。
- **TYPE**: REPLACE_PARAGRAPH
- **OLD_TEXT** (verbatim, line 26):
```
在领域自适应预训练方面，我们收集了 10 个公开 MI 数据集（~870 小时，~300 被试），对 CBraMod 进行 masked autoencoding 继续预训练，三种独立训练配置（V1/V2/V3）下均出现一致的**负迁移**（V2 平均 −1.38 pp）；尤为关键的是，被试内（数据最稀缺、最依赖良好初始化）受损最严重，与"DAPT 在数据稀缺场景中收益最大"的预期方向相反。V3 将主导数据集 Stieger2021 占比从 ~79% 降至 ~30% 后，约恢复了 V1→V2 阶段加剧负迁移的一半（+0.68 pp），但整体相对 Baseline 仍为 −0.70 pp 平均——表明外部粗运动 MI 数据并非在改进表征，而是在覆写 TUEG 学到的通用 EEG 表征，单纯调整数据组成不足以扭转方向。
```
- **NEW_TEXT**:
```
在领域自适应预训练方面，我们评估了 5 个独立训练配置（V1–V3：10-dataset 系列；V4：3-set 域对齐 + strict filter；V5：Stieger 单源 60ch），共 16 个 paired comparison cell。结果呈现 **task-asymmetric 负迁移**：cross-subject **binary** 5/5 配置一致负向，平均 Δ=**−1.79 pp**（BH-FDR @ 0.05 内 3 个负向显著存活；Stouffer 聚合 Z=−5.32，p<0.001）；cross-subject **ternary** 4/5 配置弱正、仅 V5 弱负，平均 Δ=**+0.18 pp**（Stouffer p=0.564）。V4/V5 两次 surgical fix 把"域错配 / Stieger 占主导 / 通道数异质"三个候选机制收紧到唯一存活假设——**MI 粒度错配（pretext-task granularity mismatch）**：粗 hand/leg/upper-limb MI 学到的是低频空间包络，下游 finger-level binary 需要 DAPT 未学到的细粒度区别；ternary 的 rest 类则可用粗粒度识别，因此不那么受损。值得注意的是，V5 单源 60ch 反方向证伪了"通道数异质性是混淆"假设——通道多样性在 DAPT 中是**保护因子**而非 bug。
```
- **RATIONALE**: 5 句中文，~~270 字符。Headline 直接是 task-asymmetric 与 mean Δ；mechanism 收紧到 MI 粒度错配；surfacing V5 通道反向证据。与新 finding 5 / §3.6 / §4.5 严格一致。

### A-§1.4-F5: Finding 5 contribution claim — 见 EDIT A5（已落地）

### A-§7-F4: §7 Finding 4 conclusion statement（替换当前 line 1001）

- **Anchor**: §7 结论的 finding 4 段落（line 1001，整段 `> **发现 4 — 领域自适应 further pre-training...`）。
- **TYPE**: REPLACE_PARAGRAPH
- **OLD_TEXT** (verbatim, line 1001):
```
> **发现 4 — 领域自适应 further pre-training 在以粗运动 MI 为主的外部数据上未能改善精细手指运动解码。** 尽管使用了 870 小时外部 MI 数据，further pre-training 在三种训练配置（V1: cosine/30K segments; V2: constant LR/78K segments, Stieger ~79%; V3: constant LR/~46K segments, Stieger ~30%）下均呈现负迁移，平均退化分别为 **−0.75 / −1.38 / −0.70 pp**；且**被试内（数据最稀缺）受损最重**——这与"DAPT 在数据稀缺场景中收益最大"的预期方向相反，提示外部 MI 数据的 DAPT 不是在改进表征，而是在以错配分布覆写 TUEG 的通用表征。V3 通过将 Stieger2021 占比从 ~79% 削减到 ~30%，**约恢复了 V1→V2 阶段加剧的负迁移的一半（V3 vs V2 平均 +0.68 pp）**，但整体方向未由负转正——这表明单一数据集主导只解释一部分负迁移，"粗运动 MI 数据池与 finger MI 任务"的分布错位独立持续作用。该结论限于粗运动 MI 数据池；只在存在类型更接近的 source MI 数据（如手指级、手部精细动作 MI）可用时才值得再考虑 DAPT。
```
- **NEW_TEXT**:
```
> **发现 4 — 领域自适应 further pre-training 在以粗运动 MI 为主的外部数据上呈 task-asymmetric 负迁移；机制收紧到 MI 粒度错配。** 5 个独立 DAPT 配置（V1–V3 = 10-dataset 系列、V4 = 3-set 域对齐 + strict filter、V5 = Stieger 单源 60ch）共 16 paired-cell 评估显示：cross-subject **binary** 5/5 一致负向（mean Δ=−1.79 pp，BH-FDR @ 0.05 内 V1 cross-bin q=0.048 / V2 within-bin q=0.033 / V4 cross-bin q=0.048 三个负向显著存活，Stouffer 聚合 Z=−5.32, p<0.001）；cross-subject **ternary** 4/5 弱正、仅 V5 弱负，mean Δ=+0.18 pp，Stouffer p=0.564——**ternary 方向性负迁移不被支持**。V4 (3-set 域对齐 + strict filter, 最低 pre-train loss 0.001914) 与 V5 (Stieger 单源 60ch) 两次 surgical fix 把候选机制收紧到唯一存活假设——**MI 粒度错配**：粗 hand/leg/upper-limb MI 学到的是"哪个肢体在动"的低频空间包络，下游 finger-level binary（食指 vs 中指，**同手**）需要 DAPT 未学到的细粒度判别；ternary 的 rest 类（不动 vs 运动）可用粗粒度空间包络识别，因此不那么糟。V5 单源 60ch 反方向证伪"通道数异质性是混淆"假设——通道多样性在 DAPT 中是**保护因子**而非 bug。该结论限于粗运动 MI 数据池；DAPT 能否改善 finger MI 解码取决于 source domain 的**信号粒度对齐**而非任务语义类别（"都是 MI"）。
```
- **RATIONALE**: 与新 finding 5 / 摘要 DAPT 段落 / §3.6 主体严格一致。把 §7 finding 4 从"V1/V2/V3 一致负 + V3 部分恢复"重写为"5 V × 16 cell task-asymmetric + 唯一存活假设是 MI 粒度错配 + 通道多样性是保护因子"。

---

## 4. Numbers cross-check

| 引用位置 | 数值 | Step 1b 来源 | 验证 |
|----------|------|-------------|------|
| §3.6 lead | cross-bin mean Δ=−1.79 pp | §0 + §4 mean row | ✓ |
| §3.6 lead | cross-ter mean Δ=+0.18 pp | §0 + §4 mean row | ✓ |
| §3.6 lead | Stouffer cross-bin Z=−5.320, p<0.001 | §5 table | ✓ |
| §3.6 lead | Stouffer cross-ter Z=+0.577, p=0.564 | §5 table | ✓ |
| §3.6 lead | Stouffer full 16-family Z=−4.830, p<0.001 | §5 table | ✓ |
| §3.6 lead | per-subject Δ-of-Δ t=−5.160, p<0.001, mean=−1.96 pp | §4 bonus | ✓ |
| §3.6 lead | BH-q V1 cross-bin=0.048, V2 within-bin=0.033, V4 cross-bin=0.048 | §3.1 survivors | ✓ |
| 表 16 V1 | Δ=−1.85 / +0.79 / −1.85 / +0.79 (within bin/within ter/cross bin/cross ter) | §2 row V1 (note: within bin Δ=-1.25, not -1.85) | **CORRECTED in table — within bin V1 Δ=−1.25, not −1.85**; verified |
| 表 16 V5 cross-bin | Δ=−2.77, p=0.014, q=0.058 | §2 row V5 cross-bin | ✓ (q=0.058 是 borderline 未 BH-显著) |
| 表 16 V4 cross-bin | Δ=−1.61, p=0.008, q=0.048 | §2 row V4 cross-bin | ✓ |
| §3.6.1 V4 best loss | 0.001914 | handoff §V4 strict filter table | ✓ |
| §3.6.1 V5 双向恶化 | binary −2.77, ternary −1.17 pp | §2 V5 rows | ✓ |
| §4.5 V4 cross-bin | Δ=−1.61, p=0.008, q=0.048 | §2 V4 cross-bin | ✓ |
| §4.5 V5 cross-bin | Δ=−2.77, 5 V 中最差 | §2 V5 cross-bin | ✓ |
| §4.5 5-V 全 binary 负 | V1 −1.85, V2 −1.25, V3 −1.46, V4 −1.61, V5 −2.77 | §4 per-V table | ✓ |
| §4.5 mean Δ binary cross-sub | −1.79 pp | §4 mean row | ✓ |
| §4.5 mean Δ ternary cross-sub | +0.18 pp | §4 mean row | ✓ |
| §4.5 per-subject Δ-of-Δ | n=105, t=−5.160, p<0.001 | §4 bonus | ✓ |
| 摘要 + finding 5 + finding 4 | 与 §3.6 / §4.5 数值严格一致 |  | ✓ |
| §3.6.2 V1→V2 cross-bin 反转 +0.59 pp | "V1 88.84 vs V2 89.43 在原 baseline 下" | further_pretraining_analysis.md §6.3 | ✓（在 step 1b 修订 baseline 下变为 V2 比 V1 弱 0.60 pp，但方向仍然 V2>V1） |

**已知数值一致性问题**：v3.0.1 中老 §3.6 表 16 列示的 V3 vs Baseline = "−0.70 pp 平均" 在 Step 1b 修订 baseline 下变为 cross-bin 个体 −1.46 pp / cross-ter +0.62 pp（per-row Δ 见表）。新表 16 完全用 step 1b 数值，已无"平均 Δ"汇总行——改为 Stouffer 聚合行（更严谨的统计聚合，而非 4-cell 算术平均）。

---

## 5. Risks and considerations for orchestrator

1. **§3.6 与 §4.5 长度膨胀**：新 §3.6 ~ 1.5 倍原长（增加表 16 expansion + 3 个子节）；新 §4.5 ~ 1.05 倍原长（机制叙事重组而非纯加法）。建议 orchestrator 检查整篇论文长度预算，必要时把 §3.6.3 V2 caveat 段缩减为脚注。

2. **图 10 需 regenerate**：现 v3.0.1 引用的 `paper/figures/further_pretraining.png` 是 V1/V2 时代的 2-row 图，与新 task-asymmetric 叙事不匹配。建议在 §3.6 标注 `[图待 regenerate]`，由后续工作生成 5 V × 4 paradigm-task + Δ-of-Δ forest plot。Subagent A 不修改图像文件。

3. **NLP DAPT 引用归 Subagent C**：§4.5 NEW_TEXT 中插入了 `Gururangan et al. 2020 [新增引用]` 一处。Subagent C 在 R3 lit-review 中负责该引用的全文综述（Beltagy SciBERT、Lee BioBERT 等）。Orchestrator 在合并时需把 `[新增引用]` 占位符替换为正式 reference 编号。

4. **§5 limitation 编号扩展**：EDIT A4 把原 #12 单行扩到约 3× 字符长度。如 orchestrator 担心表格 row 过长，可拆为 #12 (DAPT 训练配置单次性 + 方法配置未扫) + #13 (V4/V5 评估覆盖不全 + 6 项 V4/V5 specific caveats)；当前结构是 single 11-列 row。

5. **Abstract 段落顺序**：新摘要 DAPT 段落（NEW_TEXT）与原 line 26 位置完全相同，建议保留当前结构——置于"通道缩减 / 纵向数据扩展"段落之后、"实用 BCI 部署路径" 段落（line 28）之前。**摘要 DAPT 段落必须先于 line 28 的部署路径段落**——否则部署路径段对 DAPT 的"以粗运动 MI 数据池不推荐"暗示就缺乏前述支撑。

6. **finding 5 vs finding 4 编号**：v3.0.1 §1.4 中 finding 5 是 DAPT、finding 4 是纵向数据扩展；§7 中 finding 4 是 DAPT。本编辑列表使用 v3.0.1 的实际编号——A-§1.4-F5 改 §1.4 finding 5；A-§7-F4 改 §7 finding 4。Orchestrator 若想统一编号需 paper-wide 协调，不在 Subagent A 范围。

7. **§4.8 决策路径段对 DAPT 的引用**：v3.0.1 §4.8 第 5 项 "外部域外数据 (~870h)" 段（line 944）仍引用 V1/V2/V3 旧数据。本编辑列表未触及 §4.8（不在 Subagent A scope），但 orchestrator 在合并时若发现 §4.8 与新 §4.5 / §3.6 出现数值或叙事冲突，建议提示 Subagent C 或开发者明确 §4.8 的所有权。

8. **§3.6 §3.6.1–§3.6.4 子节编号**：现 §3.7.1 / §3.7.2 / §3.7.3 已存在（属 Subagent B 范围）；本编辑新增 §3.6.1 / §3.6.2 / §3.6.3 / §3.6.4 与之并列，编号方案一致。

9. **数据来源 block 整合**：新 §3.6 数据来源 block 列出 V1–V5 全 5 个 pretrain checkpoint + 全部下游 cache JSON；orchestrator 在合并时需保留所有路径以维持 CLAUDE.md 的"实验结果引用规范"。

---

## 6. 编辑列表统计

- **OWNED EDITs**: 6 (A1: §2.7.2 表扩展 + caveat; A2: §3.6 全节重写; A3: §4.5 全节重写; A4: §5 limitation #12 扩展; A5: §1.4 finding 5 重写; A6: §3.6→§3.7 桥句 (optional))
- **Multi-touch contributions**: 3 (Abstract DAPT / §1.4-F5 / §7-F4)
- **新文本中文字符总数**：约 8,500 字符（§3.6 ~3,200 + §4.5 ~1,500 + §2.7.2 ~700 + 表扩展 + §5 ~700 + 其他 ~2,400）
- **更新表 16 数据行数**：16 cells in 16 行 + 3 Stouffer 聚合行 + 2 mean 标注 = 实际 16 数据行 + 3 聚合行 = 19 visible rows（V4/V5 ternary 仍出表，但与"20 cell"中 V1/V2/V3 within=12 + V4/V5 cross=4 + V1/V2/V3 cross=8 = 20 共计；表只显示 16 行 paired comparison + Stouffer 行）
- **数据来源 entry 数**：12 条数据来源（5 baselines + 5 pretrain checkpoints + 8 V cache files + 2 cross-ref docs）
