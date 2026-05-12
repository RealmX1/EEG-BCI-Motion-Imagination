# R1 Methodology Review Report — Stage 3 Phase 1

**Reviewer Role**: R1 (Methodology / Machine Learning)
**Paper**: `paper/drafts/paper_draft_v3.0.1.md`
**Review Date**: 2026-05-10
**Recommendation**: **Major Revision**

---

## 1. Summary of the Methodology (R1 framing)

本文在 21 名被试、128 通道 BioSemi 单指 MI 数据集 [3] 上，将 EEG 基座模型 CBraMod (~30M 参数) 与紧凑 CNN EEGNet-16,4 (~16K 参数) 在三种范式 (within-subject / cross-subject / XSI-FT) 与三种任务 (binary / ternary / quaternary) 下进行比较，并通过四条互补的方法学线索界定基座模型的实用价值：(a) §3.5 系统化通道缩减 (128 → 4 ch，五种 32ch 数据驱动 / 手工方法 + 4ch 负控制)；(b) §3.4 多 session 纵向数据扩展 (N = 16)；(c) §3.6 在 870h 外部 MI 数据上的领域自适应 further pre-training (V1/V2/V3 三种配置) 负面结果；(d) §3.7 容量 / 预训练 / 架构三向消融 (EEGNet 容量阶梯 16K → 30M + random-init CBraMod)。所有报告数字基于 HPO 后超参数；统计采用单次配对 t 检验，未做多重比较校正；引用全部回溯到 ExperimentDB run_tag + JSON cache。**该方法学骨架在工程严谨度上明显高于一般 BCI 类硕士论文**——HPO 对照表 (Table S5b) 完整、负控制实验 (§3.5.3) 设计、ProbabilisticSubjectPruner 的剪枝率与时间预算节省都被显式披露。**但本文最重要的方法学命题——§3.7 三向分解 (架构 ~+35 pp / 预训练 ~+27 pp / 容量 −25 pp)——的隔离严密度与摘要 / §7 Finding 1 的修辞强度不匹配**，下文将以此为核心。

---

## 2. Strengths in Methodology

1. **数据分割协议的严密度高**：trial-level 时序分割 + held-out Sess02_Finetune 测试集 + 验证集取末 20%（§2.3）的设计基本排除了滑窗泄露与时间穿越，这是 BCI 文献中常见而又被很多 EEGNet 复现遗漏的细节。Quaternary 与 binary / ternary 切分协议的差异 (§3.3.1) 被显式标注且不混入主线，做法专业。

2. **HPO 透明化**：Table S5 (fANOVA) + Table S5b (init / HPO best / actual adopted) 把"实际采用值与 HPO 最优值之间的所有 override"全部列出来（最显著的是 label_smoothing 用户 override 0.05 的说明）——这是 R1 在大多数 BCI 投稿里看不到的诚实披露。

3. **ProbabilisticSubjectPruner 的方法学合理化**：剪枝率 52.9–65.6% 与最终 best trial 由"跑完所有被试的运行"决定这两点都被显式说明 (§2.5.1)，避免了"剪枝 = bias"的常见质疑。

4. **负面结果的高密度且分层归因**：§3.6 DAPT 三阶段 (V1 → V2 → V3) 把"Stieger 主导"vs"粗运动 MI 域错配"两个解释拆开，并通过 V3 占比 30% 的对照实验回收了一半负迁移幅度，剩余一半归因为域错配。这种"先归一个变量，再用补充实验隔离"的方式正是方法论严谨的样板。

5. **Random-init seed 复现性检查 (§3.7.2)**：seed=42 → seed=1234 重跑 within ternary，并按"max train_acc > 0.40"的严格判据重新评估"逃出 chance"被试集合 ({S07, S09, S19} → {S13, S19})，是本论文里方法论最干净的一段。

6. **架构 / 预训练 / 容量三向消融的实验设计本身**：在 EEGNet 与 CBraMod 之间补出 EEGNet-Mid (1.90M) / EEGNet-Huge v3 (5.84M) / CBraMod random-init (30.48M) 三个中间锚点，覆盖了 (architecture × pretrain) 矩阵的三个角点。**实验设计合理；本评议的核心异议在于结论的强度 vs. 隔离不严的程度不匹配，而非实验本身设计错误。**

7. **数据来源标注规范**：每个 finding 后跟随 run_tag + JSON cache 路径的双重溯源，使任意数字可在 ExperimentDB 上独立验证。这一规范对本评议过程本身也极有帮助。

---

## 3. Major Concerns (sorted by severity)

### 3.1 §3.7 三向分解的隔离严密度与结论强度严重不匹配（**最严重**）

**位置**：§3.7.1 (lines 748–772), §3.7.2 (lines 774–805), §3.7.3 (lines 807–820), 摘要 (lines 18–22), Finding 1 (line 995)。

**证据**：

- **Caveat 1（已在文中承认但未弱化结论）**：§3.7.1 line 750 明确承认 "baseline → Mid 的首跳同时改变了 conv stem (F1: 16→32, F2: 64→256) 与 MLP 头（单 Linear → 双层 [1024,1024] + LayerNorm + ELU）"。这意味着 baseline 76.67% → Mid 57.65% 这一 **−19.02 pp 的 cross-subject 跳跃中，无法分离 conv stem (F1/F2) 单轴贡献、MLP 头扩展贡献、与 LayerNorm 引入的优化稳定性副作用**。Mid → v3 → v1/v2 的 MLP 单轴扩参链路虽然干净，但占整条阶梯总跌幅 (76.67 → 50.00 = −26.67 pp) 中只有约 (57.65 − 50.00) = −7.65 pp 是干净的。
- **Caveat 2（部分被承认）**：v1 (LR=5e-5) / v2 (LR=5e-4) 在 ~20–30M 量级下两套 HP 间隔 10× 都死锁在 train loss = 0.693。文中将其升格为"容量饱和而非 HP 问题"——但 R1 的判断不同：**两套 HP 都不带 LayerNorm**，而 §3.7.1 footnote 与 handoff 都已承认"v3 通过加 LayerNorm + 缩 MLP 才让模型变得 trainable"。这意味着 v1/v2 的失败更准确地说是 **"30M EEGNet + no LayerNorm + BF16 优化"** 这个组合的失败，**而非"30M 规模本身不可训"**。一个未被实验回答的反事实：30M EEGNet + LayerNorm + 同 v3 HP 是否仍然死锁？文中没有此对照。
- **Caveat 3（未被承认）**：EEGNet-Huge v1 / v2 / v3 / Mid 总共只有 **2 套手调 HP**（v1 用 LR=5e-5、v2 用 LR=5e-4，v3 / Mid 共用 LR = 8e-4 / 1.5e-3 / 5e-4）——**没有任何一档跑过 Optuna 系统化搜索**。CBraMod baseline 经历了 51 trials / 23 complete (within-subject HPO) + 77 trials / 43 complete (cross-subject HPO) 的 TPE 探索 (Table S5b)。EEGNet baseline (16K) 也经历了 32 trials / 10 complete 的 HPO。**EEGNet-Huge / Mid 系列的 HPO 预算 = 0**。因此当文中宣称 "EEGNet 容量内扩参 −25 pp" 时，更精确的描述应该是 "**在没有为该容量量级搜索专属 HP 的前提下**，扩参 EEGNet 跨被试性能 −25 pp"——这在方法论上等价于把"under-search"误判为"capacity is not the bottleneck"。
- **Caveat 4（未被承认）**：CBraMod random-init 同样没有跑专属 HPO，handoff line 240 已明示"random-init 复用了 original-weights CBraMod 的 HP（即 `get_default_config()`）"。这在数学上意味着摘要 line 20 的 "random-init CBraMod cross 86.34% vs EEGNet-Huge v3 51.37% → +34.97 pp 来自架构" 这个论断**两端都用了 sub-optimal HP 的对照**——一端 (random-init CBraMod) 用了"为 pretrained backbone 调出的 HP"作 from-scratch 训练，另一端 (EEGNet-Huge v3) 用了"为该容量量级手调出的 HP，未经 Optuna 搜索"。两个 sub-optimal 估计量之间的差能否升格为"+35 pp 完全来自架构归纳偏置"，**至少需要一个对称的 fairness 论证**：要么承认两端都未充分调优 → 真实差距上限不明，要么补一个对称 HPO 实验。

**所要求行动**：

(a) **必须重写 §3.7.3 与 §7 Finding 1 与摘要的关键句**：把 "+34.97 pp 完全来自架构" 修订为 "**在两端均未做专属 HPO 的对照下，+34.97 pp 是架构差异 + EEGNet-Huge HP 欠搜索 + random-init CBraMod HP 欠搜索的复合估计；其中架构归纳偏置的下界估计 ~ X pp（基于 §3.7.1 的 MLP 单轴 −7.65 pp 与 random-init CBraMod cross 86.34% 的接近 baseline 的事实）**"。

(b) **强烈建议补一组实验**：EEGNet-Huge v3 (5.84M) 跑 8–12 trials 的 TPE HPO（lr × wd × dropout × LayerNorm 位置）；如果 cross-subject 仍然 ≤ 60%，则"+34.97 pp 来自架构"主张稳健；如果 cross-subject 拉到 65–75%，则该主张需要修订为 "+15–25 pp 来自架构 + 余量来自 EEGNet HP 欠搜索"。预算估计：~6–10 GPU-hour（参考 EEGNet baseline 的 ~30 trial HPO 时长）。

(c) **必须显式分离 §3.7.1 baseline → Mid 一跳中的 (F1/F2 conv stem 单轴)、(MLP 头扩展)、(LayerNorm 引入) 三项贡献**——§6 已列为 future work（line 985）但并未在主结论中弱化。Future work 不能替代 main claim 的 caveat 落地。

### 3.2 HPO 预算严重不对称（与 3.1 关联但独立）

**位置**：Table S5b (lines 1224–1277), §2.5.1 (lines 216–225)。

**证据**：

| 模型 / 范式 | Trials launched | Trials complete | 完成率 | 经过 Optuna 系统搜索 |
|---|---|---|---|---|
| EEGNet within | 32 | 10 | 31.3% | ✓ |
| CBraMod within | 51 | 23 | 45.1% | ✓ |
| CBraMod cross | 77 | 43 | 55.8% | ✓ |
| EEGNet cross | (Table S5b 中缺) | — | — | (论文未明确披露) |
| EEGNet-Mid / Huge v1/v2/v3 (3 种容量 × 3 种范式) | **0** | **0** | **0%** | **✗（仅 2 套手调 HP）** |
| CBraMod random-init (3 种范式 × 2 种任务 = 6 condition) | **0** | **0** | **0%** | **✗（复用 pretrained HP）** |

ProbabilisticSubjectPruner 的剪枝率 52.9–65.6% 意味着**被剪掉的 trials 不进入 fANOVA 重要性计算**。但 fANOVA 在 25 trials (Table S5 within) / 43 trials (S5 cross) 的样本上做参数重要性归因——这在 ML 文献里属于"**勉强可信但样本边缘**"的范围（fANOVA 通常需要 ≥ 50 trials 才能稳定估计 Sobol-style 重要性）。`backbone_lr` 在 cross-subject HPO 上占 66.8% 重要性这一报告值，**95% bootstrap CI 至少应做一次报告**——文中没有。

**EEGNet within HPO 完成率 31.3% (10/32) 是另一个红线**。10 个完成 trial 不足以让 TPE 收敛到该搜索空间的局部最优——这意味着 EEGNet baseline 78.10% 的 within-subject baseline 本身可能仍有 1–3 pp 的 HP 改进余地。这个不确定性**会同时影响 §3.7 三向分解的 "EEGNet baseline 78.10%" 这个锚点**。

**剪枝率本身的潜在 bias**：ProbabilisticSubjectPruner 的判据是"当前被试集合上累计准确率超越 best trial 的后验概率 < 10% → 剪枝"。但被试评估顺序是固定的（按 S01 → S21 推进）。**如果 S01–S07 在某些 HP 配置下系统性偏低（如 S01 ternary 在 random-init 下 ep11 即触发 patience，§random_init_handoff line 152），则在那些区域的 trial 会被早剪——产生选择偏差**。文中 §2.5.1 用"被剪掉的 trial 集中在已经低于当前 best 的早期分支"这一句安抚读者，但**并未提供任何被剪 trial 的统计分布作为佐证**（如：被剪 trial 的早期被试 acc 分布 vs 完整跑完 trial 的早期被试 acc 分布）。

**所要求行动**：

(a) **Table S5b 应补 EEGNet cross-subject HPO 的 trial 计数**（论文目前只列了 within，cross 数据缺失）。
(b) **§2.5.1 应补一个 "Pruning bias check" 段**：报告被剪 trial 与未剪 trial 在前 7 名被试上的准确率分布；若两者无显著差异，则剪枝偏差小；若有，则需说明对最终 best trial 选择的影响。
(c) **fANOVA 参数重要性应附 95% bootstrap CI**（用 Optuna `importance_evaluator=FANOVA` 接口），25/43 trials 下 backbone_lr 66.8% 的点估计可能 ±20 pp。
(d) **强烈建议**：把 EEGNet within HPO trial 数从 32 → 50+ 重跑一次，如果完成率从 31% 升至 ≥ 50%，且 best_value 移动 < 0.5 pp，则可声明"EEGNet HPO 已收敛"。

### 3.3 通道选择方法的"轻微泄露"承认与负控制证伪逻辑的失衡

**位置**：§3.5 (lines 549–707)，Limitation #1 (line 956)。

**证据**：

- Limitation #1 承认 "FDR、CSP、Attention、Band Power 指标使用了所有 session 数据（含测试 session 上下文）计算" → 这意味着**通道排序的训练信号包含了测试 session 的 trial**。文中给出的影响估计是"**可能轻微高估通道选择质量**"，但**未量化**这一高估的可能上限。
- §3.5.3 4ch 负控制 (67.65%) **远高于 50% chance** 这一事实在文中被解读为 "数据驱动方法确实捕获了更多信息（+15.06 pp）" + "高准确率并非数据泄露所致（体积传导）"。R1 认为这一论证**只半证伪了泄露假设**：负控制能解释为什么"随机选 4 通道也有 67.65%"，但**不能解释 FDR ∩ Attention 4 ch 配置高出负控制 +15.06 pp 是否完全不含泄露贡献**。一个更紧的负控制应是：**用排除测试 session 的训练数据重新计算 FDR / CSP / Band Power 排序，再跑同一组 4 ch / 8 ch / 32 ch 实验**。这个对照实验本论文没做。
- 32 ch FDR 87.71% 这一最重要的部署推荐数字**直接受这个 caveat 影响**：如果排除测试 session 后重新计算 FDR，32 ch 配置可能从 87.71% 降至 85–86% 区间（R1 估计幅度），**这不会推翻"FDR 是 32 ch 最优"的定性结论，但会修订摘要 line 22 的"96.7% retention"为更弱的描述**。
- 跨方法公平性问题：FDR / CSP / Attention / Band Power 全部使用 32 ch（除 §3.5.2 的 8/4 ch 对照外），**未做"每种方法在其最优通道数下"的对照**。例如 CSP 在 BCI 文献里的典型最优通道数是 6–12 个（用滤波器对，不是简单 top-k），文中把 CSP 强行截断到 32 个 spatial pattern 排序前 32 通道 —— 这并非 CSP 的标准用法。**单一固定 32 ch 比较对 FDR 友好**（ANOVA-style 排序在等通道数下表现稳定），对 CSP 不友好。

**所要求行动**：

(a) **必须补一个 "Train-only channel ranking" 控制实验**：用 §2.3 的 train + val (排除 test session) 重新计算 FDR / Band Power / Attention 的 32ch / 8ch / 4ch ranking，然后跑同样的 cross-subject 实验。如果新结果与文中的差异 < 1 pp，则"轻微高估"的描述成立；如果差异 1–3 pp，则需在 §3.5 / 摘要 / Finding 2 全面修订；如果 > 3 pp，则部分 finding 需重做。预算：~4 GPU-hour 的额外训练 + ~30min 的 ranking 重计算。
(b) **CSP 的对照应改为 "标准 6 / 8 / 12-component CSP + LDA pipeline"** 而非按 32 通道空间模式截断——这是 BCI 领域 (Lotte 2018, Blankertz 2008) 的标准用法。否则 CSP 在表 8 / 表 9 中的相对劣势包含"使用方式不当"的 confound。
(c) **§3.5.2 line 596 的 "4 通道交集 (FDR∩Attention) 82.71%" 应在论文主线中明确不作为推荐配置**——文中已经在 §3.5.3 的"重要说明"段标注其为 favorable outlier，但摘要 line 22 与 Finding 2 line 997 仍提到 4ch BP 78.75%，建议将"4ch BP 仅作 future cohort 验证候选"措辞贯穿到摘要与 Finding。

### 3.4 统计检验的多重比较与效应量缺失

**位置**：§2.8 (lines 283–287), 各表 p value。

**证据**：

- §2.8 line 285 显式声明 "无多重比较校正（结果章节中各表的 p value 均为该独立检验的原始值）"。
- 全文 paired t-test 的"独立检验数"R1 粗略统计 ≥ **20 次**（§3.1 binary / ternary, §3.2 binary / ternary, §3.3 XSI-FT 4 个 condition, §3.4.1 binary 2 个模型, §3.4.2 ternary 2 个模型, §3.4.3 三策略 × 2 模型, §3.4.4 三范式, §3.4.5 binary / ternary cross-subject extra sessions, §3.5.4 三档 channel, §3.6 DAPT 4 个 condition × 3 个 version, §3.7 三向消融）。即便用最宽松的 Benjamini-Hochberg FDR @ 0.05 控制，部分 p ≈ 0.012（如 §3.4.2 CBraMod ternary +Sess05）会从"显著"边缘退到"不显著"。
- **Cohen's d / 95% CI 在主表中完全缺失**。表 11、表 12a/b、表 13a/b、表 15、表 18 全部只报 mean ± SD + p value，**没有任何 effect size 或 CI**。这在 NeurIPS / ICLR / TNSRE / TPAMI 投稿中是基本要求。SD 11.00% (CBraMod within binary) 配 N=21 意味着 95% CI ~ ±5 pp——而表 6 的 +7.05 pp 差距就在这个 CI 边缘。
- 部分 p value 被 hard-bolded 但 effect size 极弱：如 §3.4.5 表 15b CBraMod ternary cross extra sessions Δ = +3.73 pp, p = 0.090 — 这本来就是 marginal，配上"无多重校正"的全局背景，应在文中弱化，而非在 §4.8 决策路径中引用为"额外 session 数据增益 cross-subject 较弱"的支持证据。

**所要求行动**：

(a) **必须**：在所有报告 p value 的表中**附 Cohen's dz** (paired t-test 的 effect size = mean_diff / SD_diff) 与 95% CI of mean difference。Python 实现 ~30 行代码。
(b) **必须**：在 §2.8 增加"由于本文同时进行 ≥ 20 次独立配对检验，所有 p < 0.05 的结论应解读为 nominal significance，未做 family-wise 或 FDR 校正"——并在主表中**增加一列 "BH-adjusted p"** 或在脚注中说明哪些 p 在 BH @ 0.05 下仍显著。
(c) 摘要中 "+6.13 pp, p = 0.007" 等数字保留，但 Finding 1–4 末尾应总体附 "All p values nominal; not adjusted for multiplicity" 一句。

### 3.5 §3.6 DAPT V2 的中断与"完全收敛"主张的不严密

**位置**：§3.6 line 731。

**证据**：line 731 承认 "V2 训练在 Epoch 13 因 Windows LMDB MapResizedError 中断，使用 Epoch 12 checkpoint 作为 best model，未触发由 patience=5 决定的 early stopping。这弱化了'完全收敛后仍更差'的强主张，但不改变'梯度方向一致负向'的定性结论。"

R1 认同该自我评估的方向。**但 V2 的负迁移幅度 −1.38 pp 是 §3.6 / Finding 4 / 摘要里被反复引用的中心数字**。一个 Epoch 12 中断的 checkpoint 与一个跑完 50 epoch 的 V3 (best at ep22) 在 "训练充分度" 上不等价；用 V2 去定量"梯度方向"可以，用 V2 + V3 的差 +0.68 pp 去归因 "Stieger 主导效应回收一半"则需要更强的可比性论证。文中虽然在 §2.7.2 caveat 段（line 281）说明 V3 用了 warm-restart-from-weights，但 **V2 vs V3 的 "+0.68 pp" 是在两段不同训练充分度的 checkpoint 之间比较**——这一警告在 §4.5 / Finding 4 中没被显式重复。

**所要求行动**：

(a) **必须**：在 §3.6 line 731 与 Finding 4 之间显式附加一句 "由于 V2 在 ep12 中断而 V3 best at ep22，V2 vs V3 的 +0.68 pp 包含训练充分度差异；该差异占 +0.68 pp 中的比例无法从现有数据估计"。
(b) **建议**：补一次 V2 retrain（同 HP，跑满 30 epoch 或自然 early-stop），如果 V2 final loss 仍高于 V3 0.4193 → 数据组成主因；如果 V2 final loss ≤ 0.4193 → V2 vs V3 差应主要归因于 Stieger 占比。预算：~5 GPU-hour。

### 3.6 §3.5.4 缩减通道下 XSI-FT 的解释框架基于 N=3 数据点

**位置**：§3.5.4 (lines 685–707)，Limitation #11 (line 966)。

**证据**：表 11c 只有 3 个数据点（128ch / 32ch FDR / 8ch BP），但论文据此提出 "XSI-FT 收益取决于 cross-subject baseline 离 (channel, method) 容量上限的距离" 这一**新方法学命题**。3 个数据点不足以建立任何 scaling law（哪怕是定性的）。Limitation #11 承认这一点（"三档样本不足以系统验证..."），但**正文 §3.5.4 的论述强度（"修订框架"）与 §4.6 决策路径（"低密度 + XSI-FT 不应作为默认推荐"）已经把这个 N=3 框架当作工作假设在用**。

**所要求行动**：

(a) §3.5.4 与 §4.6 / §4.8 应将 "XSI-FT 收益取决于..." 措辞从"修订框架"降级为"基于 3 个数据点的方向性观察 / 工作假设"。
(b) 至少补 8ch FDR XSI-FT 与 32ch BP XSI-FT 两个数据点（~2 GPU-hour），把 N 从 3 提至 5——这是论文应该自己给的最小验证数据。

---

## 4. Minor Concerns

1. **Figure 1 / Figure 6 / Figure 6b 的版本不同步**：line 387 / 391 都承认"图为 2026-03-29 渲染版本，EEGNet XSI-FT 数字已在表 11 中补齐，绘图后续重生成时同步更新"。投稿前必须重生成图表；图与表不一致是审稿人的红旗信号。

2. **CBraMod 参数计数三处不一致**：摘要 (line 18) 写 "~4M"，Table 2b (line 194) 写 30,484,402，handoff `2026-05-09_random_init_ablation.md` line 193 提及"CBraMod 参数计数在跨文档间不一致——CLAUDE.md 写 ~4.0M，eegnet_huge.md 写 30,484,202，runtime 日志报 ~10M"。**摘要的 ~4M 与 Table 2b 的 30M 相差 ~7×**。这是审稿人当场会抓的不一致，必须统一。R1 推测正确数字是 **30.48M（含分类头）**或 **~4M backbone + 26M MLP head** 二者之一，需要论文给出明确口径。

3. **"deepEEGNet" 引用（line 178 / 766）**：文中说 "Ding et al. [3] 测试了更宽更深的 deepEEGNet... 观察到的性能提升仍较有限"，但 [3] 论文里 deepEEGNet 的具体参数量、HP、HPO 范围未在本文中标注。Ding et al. 报告的 deepEEGNet 是几 M？文中按 "+1.21% binary 微弱提升" 引用其结果，但**没有引用具体页码或数字**。这是文献核对项。

4. **Table 2b "EEGNet-16,4 = 16,162 参数"**：line 178 说 "16,162 (~10K 可训练)"，但 16,162 ≈ 16K 而非 10K。文中 16K / 10K / ~16,162 三个数字在不同段落混用，应统一为 "16K (16,162) 总参数"。

5. **§3.1 line 326 引用 "S20 仅略高于随机 (52.50%/61.25%)"**：表 S1 显示 S20 cross-subject binary = 65.62%，within = 61.25%，§3.1 是 within-subject 上下文 → 数字应是 61.25%；52.50% 是 §3.1 line 326 的 EEGNet within-subject (查 S20 EEGNet within-subject = 52.50%, OK)。但行文 "S20 仅略高于随机 (52.50%/61.25%)" 没有明确两个数字分别对应哪个模型，建议改为 "S20 (EEGNet 52.50% / CBraMod 61.25%) 仅略高于随机"。

6. **预处理流水线中 EEGNet 与 CBraMod 不对齐**（§2.2 表 1）：100 Hz vs 200 Hz, 4–40 Hz vs 0.3–75 Hz。这两个差异**会让所有"模型差异"也包含"输入信号差异"**。文中没有报告"用 CBraMod 预处理流水线 (200 Hz, 0.3–75 Hz) 跑 EEGNet"或"用 EEGNet 流水线跑 CBraMod"的对照。EIC / R2 可能不会强求这个对照（因为它属于"贴近源论文 input convention"的合理选择），但 R1 认为至少应在 limitation 中标注此为方法学 confound 之一。

7. **EMA 早停策略 (Table S6) 的灾难性表现 (-12.95 pp)**：脚注承认 "decay=0.998 在 50-epoch 短训练下根本不匹配 (有效半衰期 ~347 epoch)"。但 Table S6 主体仍把 EMA 列为"策略对比"的一项——这容易让读者误以为 EMA 方法本身有问题。**强烈建议**把 EMA 那行加灰底或在标题中改为 "EMA (with mismatched decay; not reliable comparator)"。

8. **§3.7.2 line 798 "from-scratch CBraMod 反而输给 EEGNet"**：这是论文中最 important 的归因 finding（**预训练在 within-subject 贡献 ~+27 pp**），但**所用 within-subject HP 是为 pretrained CBraMod 调出来的 HP**（lr = 2.9e-4 是典型的 fine-tuning lr，而 from-scratch transformer 经验值在 1e-3 ~ 3e-3 量级；handoff line 199 已明确指出此点）。换言之 random-init within ternary 18/21 chance collapse **可能部分是 HP 不适合 from-scratch 训练造成的**，而非"~4M 参数 transformer 在 ~70 trial 下结构性 saddle-lock"。Handoff 已估计 "数据 / 过参数化导致 saddle-lock 70–80% 概率，LR 假设 < 5%"——但**这一概率估计本身只基于轨迹分析，未做实验验证**。摘要里的 "~+27 pp from TUEG pretraining" 应附 caveat。

---

## 5. Reproducibility Assessment

**Code / config / data 可获取性**：
- Code repository (隐含已存在 in repo)：完整。
- HP YAML 配置：`configs/` 目录下完整列出（cbramod_v2_*, eegnet_huge_*, eegnet_mid_*, etc.）。
- 数据：依赖 [3] 的公开 Figshare 数据 (DOI: 10.1184/R1/29104040) + 16 名被试的额外 session 数据。Extra sessions 数据集论文是否公开 R1 未独立验证。
- ExperimentDB schema 及 21 个被试 + 3640 缓存 31.4 GB 信息已在 CLAUDE.md 中说明。

**Seed + split 文档化**：
- 主实验 seed 默认 42，§3.7.2 footnote 显式说明 within ternary 重跑用 seed = 1234。
- §2.3 数据分割协议清晰，trial-level 时序分割可严格复现。
- HPO 采用 Optuna，pruner 实现细节在 §2.5.1 说明。
- **缺失**：fANOVA bootstrap CI 的 random seed。

**独立研究者可否复现主数字**：
- **可以复现**：被试内 / 跨被试 / XSI-FT 主数字（HP 在 Table S5b 完整给出，run_tag 可在 ExperimentDB 反查）。
- **能复现但数字会略有飘移**：DAPT V2 (Windows LMDB MapResizedError 中断 at ep12) — 复现需在 Linux 环境或换 DB 后端。
- **难以独立验证**：通道选择的 ranking（FDR / Attention / Band Power）是从所有 session 计算得到，复现该 ranking 需重新跑各 ranking 脚本，但 channel_selections.json 已 committed。
- **不能复现**：fANOVA 重要性 (Table S5) 的 95% CI（论文未提供）。

**整体评分**：**3.5 / 5**——大部分主结果可复现；HPO seed、bootstrap CI、版本同步图表等元数据缺失但不致命。

---

## 6. Statistical Rigor

- **p value 使用**：paired t-test 实现位置 (line 285 给出 scipy 调用) 透明。
- **多重比较校正**：**完全无**。如 §3.4 所述，全文 ≥ 20 次独立 p value 报告未做 BH/Bonferroni 校正。这是当前论文最容易在审稿环节被点名的统计问题。
- **Effect size**：**全部主表完全无**。无 Cohen's d, 无 Hedges g, 无 95% CI of mean difference。
- **样本量预设 / power analysis**：无。N = 21 的样本量是数据集本身决定的，作者无可选；但**应在 §2.8 附加 "post-hoc power analysis at α = 0.05 detected effect size dz ≥ X with 80% power on N = 21"** 一句，让读者评估各 finding 的统计可靠性边界。
- **Subgroup 报告完整性**：§3.4.1 表 12 a/b 用了 "低基线 < 80%" / "高基线 > 90%" 作为分组——分组阈值是事后选择 (post-hoc) 还是先验定义？文中未说明。**事后阈值选择 + N=3 (CBraMod 低基线) 的 +18.75 pp 增益是 garden-of-forking-paths 式 finding** —— 应明确标注。
- **配对单元格定义**：对照两组被试集合不完全相同时取交集（§2.8 line 285）做法合理。

**整体评分**：**2.5 / 5**——单次 t-test 实现正确，但全文统计严密度远低于 R1 标准。所要求的最小补救：(a) effect size 全表补齐，(b) BH 校正一列，(c) post-hoc power 一句。

---

## 7. Required Revisions (Major Revision)

如果作者愿意做以下补救实验与文本修订，R1 可以在第二轮把推荐改为 Minor Revision。

### 7.1 必做实验（**优先级 A**）

1. **EEGNet-Huge v3 (5.84M) cross-subject 跑 8–12 trial 的 TPE HPO** —— 解决 §3.1 / §3.2 异议。预算 ~6 GPU-hour。**这是 R1 唯一一个"如果不做就不能转 Minor Revision"的 ask。**
2. **Train-only channel ranking 控制实验**（§3.3）—— 排除测试 session 后重新计算 FDR / Band Power / Attention，跑同一 cross-subject CBraMod。预算 ~4 GPU-hour（仅需 32ch / 8ch / 4ch 三档）。
3. **EEGNet within HPO 从 32 → 50+ trial 重跑**（§3.2）—— 提升完成率从 31% 至 ≥ 50%，确认 EEGNet baseline 78.10% 的 HPO 收敛性。预算 ~3 GPU-hour。

### 7.2 强烈建议实验（**优先级 B**）

4. **§3.5.4 补 8ch FDR XSI-FT + 32ch BP XSI-FT 两个数据点**——把 N 从 3 提至 5。预算 ~2 GPU-hour。
5. **DAPT V2 retrain（不中断版本）**——验证 V2 vs V3 的 +0.68 pp 不是训练充分度差异主导。预算 ~5 GPU-hour。

### 7.3 必做文本修订

6. **§3.7.3 / §7 Finding 1 / 摘要**：把 "+34.97 pp 完全来自架构" 措辞修订为 "在两端均未做专属 HPO 的对照下，+34.97 pp 是架构差异 + EEGNet HP 欠搜索 + random-init CBraMod HP 欠搜索的复合估计；在 §3.7.1 的 MLP 单轴干净对照下，架构归纳偏置的下界估计 ~ X pp"（X 待补 7.1 后填）。
7. **所有主表补 Cohen's dz + 95% CI**。
8. **§2.8 + 主表**补 BH-adjusted p value。
9. **§3.6 / Finding 4** 补"V2 ep12 中断对 +0.68 pp 归因的影响"caveat。
10. **§3.5.4 / §4.6 / §4.8** 把 "XSI-FT 收益框架" 降级为"3 个数据点的方向性观察"。
11. **CBraMod 参数计数**：摘要 / Table 2b / 各章节统一为 30.48M（或 4M backbone + 26M MLP head 的拆分口径）。
12. **图表与最新数字同步**（Figure 1, 6, 6b, 4b 全部需要重生成）。

### 7.4 估计总额外成本

实验补救（7.1 + 7.2）：~20 GPU-hour（一夜跑完）；文本修订：~1–2 整工作日。

---

## 8. Confidence in Review

**Confidence: 4.5 / 5**

依据：
- 阅读了 §1–§7 + 关键 Supplementary (S5, S5b, S6, S7) + 两份核心 handoff (random-init ablation + eegnet_huge) + 关键 YAML (cbramod_v2_cross.yaml, eegnet_huge_cross.yaml)。
- HPO trial 计数与剪枝率的报告与论文 Table S5b 一致；fANOVA 重要性数字与 Table S5 一致。
- 三向分解的 caveat (§3.7.1 conv stem 单轴未隔离、~30M EEGNet 无 LayerNorm 失败、random-init / EEGNet-Huge 无专属 HPO) 的判断在文中均能找到对应承认或暗示，并非 R1 自由推断。
- 唯一一处 R1 未独立验证的具体数字：fANOVA 在 25 trial 下 95% CI 的真实宽度（论文未提供，R1 凭经验估计 ±20 pp）。
- ProbabilisticSubjectPruner 的 selection bias 判断属于"可能存在但未被作者用数据反驳"——R1 没有独立证据但所要求的额外检查（被剪 vs 未剪 trial 在前 7 名被试上的分布）应当能在 ~30 min 内由作者本人提供。
- ChannelRank "用全 session 数据计算"的影响幅度估计（1–3 pp）是 R1 的经验估计，作者复现实验后可能修正方向。

---

## ===============================
## ORCHESTRATOR RETURN — 250-word summary
## ===============================

**Recommendation**: **Major Revision**

**Top 3 major methodology concerns**:
1. §3.7 三向分解的隔离严密度严重不足：摘要 / Finding 1 强主张 "+34.97 pp 完全来自架构归纳偏置"，但 EEGNet-Huge 与 random-init CBraMod 双方都未做专属 HPO，且 baseline → Mid 一跳同时改了 conv stem + MLP 头，无法单轴归因。
2. HPO 预算严重不对称：CBraMod 51 + 77 trials 经过 Optuna 系统搜索；EEGNet-Huge / Mid / random-init CBraMod 共 4 套配置 × 3 范式 = 12 condition 全部 0 trial Optuna，仅手调 2 套 HP。"capacity is not the bottleneck" 主张可能是 HP under-search masquerading as architectural finding 的典型表现。
3. 多重比较校正完全缺失（≥ 20 次独立 p value 全无 BH / Bonferroni）+ 所有主表 Cohen's d / 95% CI 全无；§3.5 通道选择的"轻微泄露"承认未量化，主推荐数字 32ch FDR 87.71% 受影响。

**§3.7 三向分解判定**: **Overclaimed**。实验设计本身合理，三个角点 (EEGNet-Huge / random-init CBraMod / CBraMod baseline) 选择正确，但对每个 Δ 的归因强度（+34.97 pp 完全归因架构、−25 pp 完全归因 EEGNet 架构内扩参有害、+27 pp 完全归因预训练）超出当前隔离严密度所能支持的程度。

**HPO Fairness 判定**: **Asymmetric and confounding**。CBraMod 经过 51 + 77 trial 系统搜索，EEGNet-Huge / random-init CBraMod 用手调 HP 与 default config 直接对比。需要 EEGNet-Huge v3 至少 8–12 trial 的对称 HPO 才能让 §3.7.3 的 +34.97 pp 主张成立。

**Path to full report**: `paper/reviews/stage3_r1_methodology_review.md`
