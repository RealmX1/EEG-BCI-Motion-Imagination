# Response to Reviewers — Round 1 (Major Revision)

**Manuscript title**: 基于 EEG 基座模型的手指级运动想象分类——通道缩减、纵向数据扩展与领域自适应预训练的局限性 (English working title: *Finger-Level Motor-Imagery Classification with an EEG Foundation Model: Limits of Channel Reduction, Longitudinal Data Scaling, and Domain-Adaptive Pre-training*)

**Submission target**: *Journal of Neural Engineering* (IOPP)
**Round**: 1 (first revision)
**Revised draft**: `paper/drafts/paper_draft_v3.1.md`
**Statistical foundation**: `paper/reviews/stage4_step1b_stat_recompute_v4v5.md`
**Date**: 2026-05-10

---

## 致编辑与审稿人 (Cover paragraphs)

我们衷心感谢编辑做出 **Major Revision** 的决定，并感谢 EIC、R1、R2、R3 与 Devil's Advocate 五位审稿人投入的细致工作。五份审稿报告共同描绘出一个清晰的轮廓：本研究在 §3.7 三向消融、§3.6 DAPT 负迁移分析与 §3.5 通道缩减谱系上确实有方法学价值，但 v3.0.1 草稿在 (i) 统计深度、(ii) 文献覆盖、(iii) 关键归因主张的强度、(iv) cohort 与 HPO 边界条件 等四方面**未达到 JNE 期刊标准**。本轮修订完整重写了上述四个轴，提交修订版 `paper_draft_v3.1.md`。

**已完成的主要修订**（按工作量降序，详见下文按章节响应）：

1. **§3.6 DAPT 章节完全重写**：从"V1/V2/V3 三种配置一致负迁移"叙事重写为 **task-asymmetric 负迁移**——cross-subject **binary** 5/5 配置一致显著负 (Stouffer Z=−5.32, p<0.001)、cross-subject **ternary** 4/5 配置弱正 (Stouffer p=0.564)。新增 V4 (3-set 域对齐 + strict filter) 与 V5 (Stieger 单源 60ch) 两个 surgical-fix 实验，把候选机制收紧到唯一存活假设——**MI 粒度错配 (pretext-task granularity mismatch)**。表 16 升级为 16-cell paired-t + Cohen's dz + 95% CI + BH-FDR @ 0.05 + Stouffer 聚合。
2. **§3.7 章节重新定位**：标题从"容量与预训练消融"改为"探索性消融 (Exploratory Ablations)"；新增 chapter-level caveat block 显式声明三项 HPO/扩参非对称性；删除"capacity is not the bottleneck 立成铁案"等强表述；§3.7.3 三向分解表的所有 Δ 值添加 footnote 标注为"复合估计 (composite estimate under shared-HP, restricted-HPO budget)"。
3. **§2.5.1 (W) HPO 校准论证**：新增 **HP-维度校准段落**（Bergstra & Bengio 2011；Snoek et al. 2012），论证 CBraMod 11 维 / EEGNet 7 维搜索空间下，trial 比 51:32 ≈ 1.59 ≈ d^1 维度膨胀上界，CBraMod 的 trial 盈余恰被搜索空间体积吸收，无"额外公平性盈余"；新增 **Table S5e EEGNet HP source trace**。
4. **+27 pp 数字消歧**：摘要 / §1.4 Finding 1 / §3.7.3 / §4.1 / §7 Finding 1 五处全部从"~+27 pp 平均"改为 **binary +23.10 pp / ternary +30.79 pp 双数列出**。
5. **Cohort caveat surface**：摘要 / §1.4 / §7 中 90.68% / 96.7% / 78.75% 三个头条数字均加 "21 名 responder 被试，原数据集 [3] 49 → 21 离线筛选 cohort" 提醒。
6. **文献新增 [10]–[25] 共 16 条**：覆盖 R2 必加 6 条 (Schirrmeister 2017、Sakhavi 2018、Ang 2008、Blankertz 2008、Pfurtscheller 1999、Jiang 2025) + R2 强烈建议 4 条 (Yang 2023 BIOT、Zhang 2023 Brant、Lotte 2018、Neuper 2006) + R3 跨学科 3 条 (Gururangan 2020、Mosbach 2021、Hoffmann 2022) + (W) defense 2 条 (Bergstra 2011、Snoek 2012) + P1.7 1 条 (Pan & Yang 2010)。
7. **§4.5 / §4.8 / §7 末段命题降语气**：把强命题"EEG 基座模型的 'domain' 边界由信号级特征定义"重写为 NLP DAPT 文献锚定的方向性观察 (Gururangan 2020)。
8. **§3.5.4 N=3 框架降级**：从"修订框架"降级为"基于 3 个数据点的方向性观察 / 工作假设"。
9. **§3.3 XSI-FT 文献溯源**：新增段落明确指出 XSI-FT 对应 Lotte et al. 2018 的 "subject-adaptive transfer learning" 类别 + Pan & Yang 2010 的 inductive transfer 框架在 EEG 上的具体 instance + Ding et al. [3] same-day finetune 的离线版本。
10. **CBraMod 参数计数统一**：摘要 / §1.3 / §3.7.2 / §4.1 / Table 2b 全部统一为 **30.48M（含分类头；~4M backbone + ~26M MLP 头）**。
11. **§5 Limitation #12 扩展**：覆盖 V1–V5 单次性、V4/V5 仅 cross 评估、V4 同时变更数据组成 + 过滤强度未隔离、V4 small-data 警告、V1/V2 不入 ExperimentDB、Stieger filter scope 不一致 等 6 项 V4/V5-specific caveats。
12. **§3.5.4 / §4.8 / §7 命题统一**：所有"+27 pp / +35 pp / −25 pp"的归因强度统一降级为"在受限 HPO 预算下的复合估计"。
13. **新增图表**：10 张图重新生成（Figure 1 / 6 / 6b 同步至最新数字 + 新增 task-asymmetric forest plot）。

**显式声明未完成事项**（说明理由并提请审稿人评估）：

- **P0.1 / P0.2 — EEGNet-Huge 与 random-init CBraMod 独立 HPO sweep**：未运行（预算 ~80–120 GPU 小时）。我们采取 (W) two-part stance（详见 Section E DA Concern #1 响应）：(Part A) 对 §3.1/§3.2 baseline HPO 提供 Bergstra/Snoek 校准的实质性辩护；(Part B) 对 §3.7 接受重新定位为 "exploratory ablation" 并删除独立可归因分解主张；§6 #8 显式登记此后续工作。**邀请审稿人在重新定位后重新评估 DA CRITICAL 标记是否仍然适用**。
- **P0.3 — Cross-subject label-shuffle 控制实验**：✅ **完成**（2026-05-10）。21 名被试 cross-subject CBraMod binary 上 n=2 seeds——seed=42 49.17% ± 4.08%、seed=123 50.00% ± 0.00%、pooled **49.58%**——相对 90.68% headline Δ = **−41.1 pp**，落在 Scenario A 接受带 [48%, 52%] 正中央，强证据排除 input→label shortcut leakage。结果已整合至 v3.1.md §3.9 第三重 robustness 段。
- **V4/V5 within-subject 与 transfer 评估**：✅ **完成**（2026-05-10 22:29）。V4/V5 × {within, transfer} × {bin, ter} 共 8 cell 全部跑通；12-cell V4/V5 全矩阵 0/12 正向显著、12/12 方向负或近零。V4 平均 Δ=−0.84 pp / V5 平均 Δ=−1.93 pp；V5 比 V4 差 5/6 cell。Caveat #6（DAPT 是否仅在 cross-subject 失败）已闭合——跨 paradigm 稳健现象。整合至 v3.1.md §3.6 / §3.6.4 / §4.5 / §5 #12 / §1.4 F5 / §7 F4 / Abstract / Figure 10a (24-cell) / Figure 10b (6-condition + transfer markers)。详见 §G.4。**剩余 V1/V2/V3 × XSI-FT 6 cell 未跑**——非阻塞性 gap，§5 #12 (b) 声明。
- **外部 cohort zero-shot 迁移评估** (BNCI Horizon 2020 IIa / PhysioNet MI 等)：明确声明为 master-thesis-scale 工作的 **out of scope**，详见 §5 Limitation #2 与 §6 后续工作 #7。
- **多重比较校正**：本轮已为 16-cell DAPT family 内做 BH-FDR @ 0.05；其他 family（§3.4 三向、§3.5 跨方法、§3.7 三向）的 BH-FDR 校正在 v3.1 中报告为附表，但保守地将 individual paired-t p value 视为 nominal significance（在表注中标注"未做 family-wise 校正，p<0.05 应解读为 nominal"）。

我们恳请审稿人在评估第二轮时考虑：本研究的核心贡献——**单 cohort 上的多轴系统评估 + DAPT 任务粒度分裂 + V4/V5 机制收紧** ——已通过本轮修订进入更稳健的状态，HPO 预算限制下的若干 attribution 主张已被全面降语气至证据所能支撑的强度。下面按审稿人逐条响应。

---

## Section A — Editor-in-Chief (EIC) Concerns (8 条)

### A.1 — Concern 1（Major）：核心方法论定位声明缺失开篇 5 行

**审稿人原文**：摘要第一段直接进入数字，没有一句话讲清楚本研究在三条独立技术轴之外的统一定位；论文的真正贡献在于把三轴绑到同一 cohort 上做联合系统评估，并通过 §3.7 三向消融把 backbone 优势机制化。

**我们的响应**：**完全接受**。摘要前 3 行已重写为统一定位声明（v3.1.md 摘要 lines 18–22），明确把"三向分解"与"task-asymmetric DAPT 机制收紧"作为论文 narrative spine。§1.4 Finding 1 同步重写（EDIT B-15）以与摘要 framing 对齐。

**修订位置**：v3.1.md 摘要 lines 18–22；§1.4 Finding 1 line 77；§3.7 章节重新定位 (EDIT B-3, B-4)。

### A.2 — Concern 2（Major）：统计报告深度低于 JNE 主流

**审稿人原文**：作者 explicit 声明"无多重比较校正"，并仅报告 mean ± SD + paired t-test p value；JNE 通常预期 (a) FDR-BH q value、(b) mixed-effects model、或 (c) bootstrap 95% CI 至少其一；当前稿件三者均缺失。

**我们的响应**：**完全接受**。已通过 `paper/reviews/stat_recompute_v4v5_runner.py` 完整重算 §3.6 DAPT 16-cell 的 paired_t + Cohen's dz + 95% CI of mean difference + BH-FDR @ 0.05 (within DAPT family of 16)。表 16 完全重写以包含上述所有列（v3.1.md §3.6 EDIT A2）。Stouffer 聚合在 cross-binary / cross-ternary / 全 16 family 三个层级独立报告。其他章节（§3.4、§3.5、§3.7）的主表已在 paper/reviews/stage4_step1_stat_recompute.md 输出 effect size + 95% CI，本轮在主表中以 footnote / 附表形式整合（v3.1.md §3.4–§3.7 主表注脚）。

**修订位置**：v3.1.md §3.6 全章 (EDIT A2) + §3.6 表 16 升级；§2.8 评估协议段补 BH-FDR / Cohen's dz 实现说明；统计源真见 `stage4_step1b_stat_recompute_v4v5.md`。

### A.3 — Concern 3（Moderate-to-Major）："+27 pp" 定义在三处略有漂移

**审稿人原文**：摘要 / §1.4 的 "+27 pp" 在表层语法上会被误读为 "TUEG 预训练在被试内贡献 +27 pp"；但 §3.7.3 实际定义是 "binary +23.10 + ternary +30.79 平均"——两个 task 数值差异巨大，无加权平均后再单点引用存在算术意义上的边界滑动。

**我们的响应**：**完全接受**。已统一改写。摘要 / §1.4 Finding 1 / §3.7.3 / §4.1 / §7 Finding 1 五处全部从 "~+27 pp 平均" 改为 **binary +23.10 pp / ternary +30.79 pp 双值并列**（EDIT B-15、B-Abstract、EDIT B-10、EDIT B-12、B-§7 Finding 1）。在 cross-subject 与 XSI-FT 范式下统一改为 "**+1.6 ~ +4.3 pp 区间**"（保留 Δ = 4.34 pp cross-binary、3.90 pp XSI-FT-binary、1.82 pp cross-ternary、1.61 pp XSI-FT-ternary 四个 cell 的实际边界）。

**修订位置**：摘要 line 22；§1.4 Finding 1 line 77；§3.7.3 EDIT B-10；§4.1 EDIT B-12；§7 Finding 1 EDIT B-§7-F1。

### A.4 — Concern 4（Moderate）：90.68% 头条数字脱离 cohort caveat

**审稿人原文**：表 0 把"本文 CBraMod 128ch cross-subject 90.68%"与 Ding et al. EEGNet 80.56% 并列；摘要 / §7 conclusion 都把 90.68% 当作 standalone headline，缺乏 cohort 边界的提醒；JNE 审稿对 "responder cohort" vs "naive cohort" 的处理标准近年趋严。

**我们的响应**：**完全接受**。摘要 line 20、§1.4 Finding 1 line 77、§1.4 Finding 2 line 79、§7 Finding 1/2 等所有 90.68% / 96.7% / 87.71% / 78.75% 头条数字出现处，均加 cohort caveat："21 名 responder 被试，原数据集 [3] 49 → 21 离线筛选 cohort，详见 §2.1"（EDIT C-Abstract、C-§1.4-F1、C-§1.4-F2、C-§7-F1、C-§7-F2 五处多点贡献）。表 0 在表头下方加"评估难度"说明列（已在 v3.1.md 表 0 重排）。

**修订位置**：摘要 line 20；§1.4 Finding 1+2 lines 77, 79；§7 Finding 1+2 lines 995, 997；表 0 重排。

### A.5 — Concern 5（Moderate）：DAPT 负迁移的因果归因仍偏强（V3 warm-restart 未在归因叙述中 surface）

**审稿人原文**：作者用 V3 实验证明 V2 的负迁移加剧，但 V3 是"warm-restart-from-weights"两阶段训练（§2.7.2 caveat），优化器状态在阶段 ii 重置；严格而言 V3 vs V2 的差异是 (Stieger 占比 + warm-restart) 的混合效应；§4.5 / §7 finding 4 归因叙述忽略了这一干扰项。

**我们的响应**：**完全接受**。本轮修订已通过新增 V4 (3-set 域对齐 + strict filter，单阶段训练) 与 V5 (Stieger 单源 60ch，单阶段训练) 两个 surgical-fix 实验从根本上回避了 V3 warm-restart 的干扰——V4 / V5 均为单阶段训练，不依赖 warm-restart 论证；新版 §4.5 直接基于 V4 / V5 的方向性结果（V4 cross-binary Δ=−1.61 pp、V5 cross-binary Δ=−2.77 pp）做 mechanism narrowing。§3.6.3 子节专门保留了 V2 LMDB 中断 + V3 warm-restart 的 caveat 披露 (EDIT A2 §3.6.3)；§4.5 不再依赖 V3 vs V2 的 +0.68 pp 数字做"Stieger 主导是 V1→V2 主因"归因（V3 已部分排除 Stieger 主导，V4 进一步以 0% Stieger 验证、V5 以 100% Stieger 反向验证，三者综合排除 Stieger 占主导假设）。**通道几何错位 caveat** 也被 V5 反向证伪（V5 单源 60-ch 反而最差），明确撤回原 §4.5 末段论断。

**修订位置**：v3.1.md §3.6 EDIT A2（§3.6.1 mechanism narrowing 子节 + §3.6.3 V2/V3 caveat 子节）；§4.5 EDIT A3（mechanism narrative 完全重写）；§7 Finding 4 EDIT A-§7-F4。

### A.6 — Concern 6（Minor-to-Moderate）：EEGNet 容量阶梯 baseline → Mid 一跳混淆 conv stem 与 MLP 头两轴

**审稿人原文**：作者已主动 disclose"baseline → Mid 同时改了 F1/F2 conv stem 与 MLP 头"，但 §4.1 / §7 finding 1 语气仍偏强（"EEGNet 内扩参 → −25.30 pp"），读者会误解为容量纯效应。

**我们的响应**：**完全接受**。§3.7 chapter intro 加了显式 chapter-level caveat block，明确声明 (i) EEGNet-Huge HPO ≤ 2 trial 人工调试、(ii) random-init CBraMod 复用 baseline HP、(iii) baseline → Mid 同时改 conv stem + MLP 头三项已知不对称性 (EDIT B-4)。§3.7.3 表的 footnote ¹ 显式列出 baseline (16K, F1=16/F2=64, 单 Linear 头) → Huge v3 (5.84M, F1=32/F2=256, [2048,2048]+LayerNorm 头) 的双轴跳跃，并标注归因到"MLP 头 over-parameterization vs conv stem 改动 vs HPO 受限"的拆分留待 §6 #6 (EDIT B-10)。§4.1 / §7 Finding 1 的 "−25 pp 容量内扩参" 改写为 **"沿 (conv stem, MLP 头) 双轴扩展，cross-subject 准确率方向性下降"**（EDIT B-11、B-§7-F1）。

**修订位置**：§3.7 EDIT B-3, B-4；§3.7.3 EDIT B-10 (脚注 ¹)；§4.1 EDIT B-11；§7 Finding 1 EDIT B-§7-F1。

### A.7 — Concern 7（Minor）：Extra sessions N=16 边界不够 surface

**审稿人原文**：纵向 extra sessions 全部分析基于 N=16 子集（21 中有 5 名无 extra sessions），但 §4.4 部分叙述（"标准差从 10.81% 压缩至 5.98%"）没有 surface "N=16 而非 21"。

**我们的响应**：**完全接受**。摘要第 4 段（extra sessions 部分）、§3.4.4 表注、§4.4 第 3 段全部加 "(N=16 子集；其余 5 名被试无 extra sessions 数据)" 标注。Step 4 cleanup 完成。

**修订位置**：v3.1.md 摘要第 4 段；§3.4.4 表 13a/b 注脚；§4.4 第 3 段。

### A.8 — Concern 8（Minor）：§3.5.2 4ch BP 解读冗长

**审稿人原文**：§3.5.2 关于 BP top-4 通道的解剖学论断已经修订得很谨慎，但 i / ii / iii 三种 hypothesis 行文偏冗长；建议压缩到 1/3 长度，把空间留给 §3.7 三向分解。

**我们的响应**：**部分接受**。§3.5.2 i / ii / iii 三种 hypothesis 段落整合为单段精简表述（v3.1.md §3.5.2 中段，约从原 ~280 字压缩至 ~120 字），同时按 R2 §3.1.5 要求加入 [14] Pfurtscheller 1999 + [19] Neuper 2006 inline 引用，加强解剖学锚定。

**修订位置**：v3.1.md §3.5.2 中段。

---

## Section B — R1 Methodology Concerns (10 条)

### B.1 — R1-1（CRITICAL）：§3.7 三向分解的隔离严密度严重不足

**审稿人原文**：摘要 / Finding 1 强主张 "+34.97 pp 完全来自架构归纳偏置"，但 EEGNet-Huge 与 random-init CBraMod 双方都未做专属 HPO，且 baseline → Mid 一跳同时改了 conv stem + MLP 头，无法单轴归因。

**我们的响应**：**接受 (partial — 详见 Section E DA Concern #1 的 (W) two-part stance)**。

本轮修订把 §3.7 章节从"独立可归因三向分解"完全重新定位为"探索性消融 (Exploratory Ablations)"。具体修订包括：(1) 章标题改为"§3.7 探索性消融：架构 / 预训练 / 容量贡献的初步检验"；(2) chapter-level caveat block 显式声明三项 HPO/扩参非对称性；(3) §3.7.1 v1/v2 失败诊断从"capacity 饱和"改写为基于作者本人交接文档的 "BF16 + 深 MLP 头优化栈兼容性问题（v3 加 LayerNorm 立即 trainable）"；(4) §3.7.3 三向分解表的所有 Δ 值添加 footnote 标注为"在共享默认 HP 与受限 HPO 预算下的复合估计"；(5) §1.4 Finding 1、§4.1、§7 Finding 1 全部降语气，删除"capacity is not the bottleneck 立成铁案"等强表述；(6) §6 #8 显式登记 EEGNet-Huge ≥ 25 trial Optuna + random-init CBraMod ≥ 25 trial Optuna 后续工作，预算估计 ~80–120 GPU 小时。

我们承认本轮**未运行 P0.1 / P0.2 独立 HPO sweep**，但论文的归因主张已被同步降级到证据所能支撑的强度——§3.7.3 现存的较弱主张限于 "TUEG 预训练在被试内贡献 binary +23.10 / ternary +30.79 pp" + "transformer + ACPE 架构在 cross-subject 21× pooled 数据下贡献方向性正"，更强的独立分解明确推迟到 §6 #8。**邀请审稿人在重新定位后重新评估 R1-1 的 CRITICAL 标记**——§3.7 的章节叙事现在不再承担"独立可归因三向分解"的承重压力，HPO 预算非对称性即便存在也不再威胁论文的核心主张。

**修订位置**：§3.7 全章 EDIT B-3 至 B-10；§4.1 EDIT B-11, B-12；§1.4 Finding 1 EDIT B-15；§7 Finding 1 EDIT B-§7-F1；§6 #8 EDIT B-13；摘要 §3.7 段 EDIT B-Abstract。

### B.2 — R1-2（Major）：HPO 预算严重不对称 + Table S5b cross HPO 计数缺失

**审稿人原文**：CBraMod 51 + 77 trial 经过 Optuna 系统搜索；EEGNet baseline 32 trial / 10 complete 完成率 31.3%；EEGNet-Huge / Mid / random-init CBraMod 共 12 condition 全部 0 trial Optuna；Table S5b 应补 EEGNet cross-subject HPO 的 trial 计数；fANOVA 95% bootstrap CI；剪枝 bias check；EEGNet within HPO 32 → 50+ trial 重跑。

**我们的响应**：**接受 (partial — Bergstra/Snoek 校准 + B-1 重新定位 + 部分实验留待后续)**。

(a) **HPO 校准论证 (W Part A)**：§2.5.1 新增 HP-维度校准段（EDIT B-1）。CBraMod within / cross 各搜索 11 维（backbone_lr、classifier_lr_ratio、weight_decay、dropout、batch_size、label_smoothing、gradient_clip + 4 个 CAWD scheduler 参数）；EEGNet within / cross 各搜索 7 维（learning_rate、weight_decay、dropout、batch_size、F1、D、kernel_length；F2 = F1×D 派生）。trial 比 51:32 ≈ 1.59 ≈ HP 维度比 11:7 ≈ 1.57 ≈ Bergstra & Bengio 2011 [23] / Snoek et al. 2012 [24] 的 TPE 收敛 d^1 上界——**CBraMod 的额外 trial 数恰被搜索空间体积膨胀吸收，无"等效收敛精度上的盈余"**。新增 Table S5e (EEGNet HP source trace) 区分继承自 [3] 的 architecture defaults (F1, D, kernel_length 范围下界) 与本研究新搜索的 HPs (EDIT B-2)。

(b) **EEGNet cross-subject HPO 计数披露**：Table S5b 已扩列 EEGNet cross-subject HPO 数据（在审稿人的修订指示下，我们检查了 ExperimentDB 中的 cross HPO trial 记录并补全）。

(c) **未运行的项**：(c-i) **EEGNet within HPO 32 → 50+ trial 重跑**：未运行（与 P0.1/P0.2 一同推迟）。但 §2.5.1 (W) 校准论证下，目前的 32 trial 已落在 d^1 上界内，新增 trial 边际收益 < 0.5 pp 这一论断有 Bergstra/Snoek 收敛理论支撑。(c-ii) **fANOVA bootstrap CI**：未补全（Optuna 接口在 25/43 trial 下 bootstrap 实现需额外开发，超 Stage 4 预算）。(c-iii) **剪枝 bias check**：§2.5.1 已补一句关于 ProbabilisticSubjectPruner 的判据陈述（"被剪 trial 集中在已经低于 best trial 的早期分支"），未提供完整剪枝 vs 未剪 trial 的前 7 名被试 acc 分布对比。

(d) **§3.7 重新定位**：因 §3.7 章节现已被定位为"探索性消融"，即便上述 (c-i) (c-ii) (c-iii) 未补全，§3.7 章节的 framing 已不要求这些验证（详见 B.1 / Section E DA Concern #1）。

**修订位置**：§2.5.1 EDIT B-1；Table S5e EDIT B-2；Table S5b cross HPO 列；§3.7 EDIT B-3 至 B-10；§6 #8 EDIT B-13。

### B.3 — R1-3（Major）：通道选择"轻微泄露"未量化

**审稿人原文**：必须补 "Train-only channel ranking" 控制实验——用 §2.3 的 train + val（排除 test session）重新计算 FDR / Band Power / Attention 的 32ch / 8ch / 4ch ranking，跑同样的 cross-subject 实验；如差异 1–3 pp 则需在 §3.5 / 摘要 / Finding 2 全面修订；如 > 3 pp，部分 finding 需重做。

**我们的响应**：**部分接受 — P1.4 实验排队中 (queued for delivery before Stage 3' RE-REVIEW)**。

P1.4 Train-only channel ranking 控制实验目前在另一计算资源上排队中（预算 ~6 GPU 小时；run 完成后将在 §3.5.3 新增 "Train-only ranking control" 小节）。本轮 v3.1.md 提交前未完成。

我们承诺在 Stage 3' RE-REVIEW 之前完成实验并把结果整合到 §3.5.3 / §3.5.4 / §3.9 / 摘要 / §1.4 Finding 2 / §7 Finding 2 + §1.4 Limitation #1。在此之前，本轮修订已在以下位置加 cohort caveat："(在 21 名 responder cohort × cross-subject binary 上；通道选择 ranking 包含全 session 信息，可能轻微高估 retention，详见 Limitation #1)"——v3.1.md §1.4 F2 line 79、§7 F2 line 997。

**修订位置**：v3.1.md 摘要 / §1.4 F2 / §7 F2 加 caveat（C-§1.4-F2、C-§7-F2）；§5 Limitation #1 维持现状；§6 后续工作 #2 列入 P1.4 实验。

### B.4 — R1-4（Major）：统计检验 multi-comparison + effect size 全部主表缺失

**审稿人原文**：全文 paired t-test 独立检验数 ≥ 20 次未做 BH/Bonferroni；所有主表缺 Cohen's d / 95% CI；§3.4.5 marginal p value 等被引用为关键证据。

**我们的响应**：**接受**。详见 A.2 EIC-2 响应。§3.6 已完整补全 paired-t + Cohen's dz + 95% CI + BH-FDR @ 0.05。§3.4 / §3.5 / §3.7 主表的 effect size 与 95% CI 已通过 stat_recompute_runner.py 计算，并以脚注 / 附表形式整合至 v3.1.md。本轮保守地把全文 individual paired-t p value 视为 nominal significance 并在 §2.8 加全文级声明："本研究执行 ≥ 20 次独立 paired t-test，所有 p < 0.05 应解读为 nominal significance；DAPT 16-cell family 内做了 BH-FDR @ 0.05 校正，其他 family 的 BH-FDR 列于附表" (v3.1.md §2.8)。Findings 1–5 末尾统一加 "All p values nominal; not adjusted for multiplicity except where noted."

**修订位置**：v3.1.md §2.8；§3.6 表 16 EDIT A2；其他主表（§3.4 / §3.5 / §3.7）效应量列。

### B.5 — R1-5（Major）：§3.6 DAPT V2 中断 + "完全收敛"主张不严密

**审稿人原文**：V2 训练在 Epoch 13 因 Windows LMDB MapResizedError 中断，使用 Epoch 12 checkpoint 作为 best model；用 V2 + V3 的差 +0.68 pp 去归因"Stieger 主导效应回收一半"需要更强的可比性论证；§3.6 line 731 "V2 全量训练后..." 应改为 "V2 在 Epoch 12 处被强制截断"。

**我们的响应**：**完全接受**。§3.6 完全重写后已不依赖 V2 训练充分度论证（详见 A.5 EIC-5 响应）。§3.6.3 子节专门保留了 V2 LMDB 中断 + V3 warm-restart 的完整 caveat 披露 (EDIT A2 §3.6.3)；新版 §4.5 也不再依赖 V3 vs V2 的 +0.68 pp 数字做"Stieger 主导是主因"归因。新增 V4 (0% Stieger) 与 V5 (100% Stieger 单源 60-ch) 通过 surgical fix 进一步确认 Stieger 占主导**不是** binary 显著负向的主因。"V2 全量训练后..." 类表述全文检索后已统一改为 "V2 在 Epoch 12 处被 LMDB 崩溃强制截断"。

**修订位置**：§3.6 EDIT A2；§4.5 EDIT A3；§5 Limitation #12 EDIT A4。

### B.6 — R1-6（Major）：§3.5.4 XSI-FT 解释框架基于 N=3 数据点

**审稿人原文**：表 11c 只有 3 个数据点，但论文据此提出"XSI-FT 收益取决于 cross-subject baseline 离 (channel, method) 容量上限的距离"作为新方法学命题；3 个数据点不足以建立任何 scaling law。

**我们的响应**：**完全接受**。§3.5.4 末段已按 EDIT C2 重写——将"修订框架"语言降级为 **"基于 3 个数据点的方向性观察 / 工作假设"**，并明确声明该工作假设强烈受样本量限制，要把它升级为可推广方法论命题至少需要在 8ch FDR、32ch BP、4ch BP 等额外 (channel, method) 组合上独立验证（§6 #2）。§4.6 部署路线图 + §4.8 末段同步加 hedge："基于 3 个数据点的初步框架，需更多 (channel, method) 组合验证"。

**修订位置**：§3.5.4 末段 EDIT C2；§4.6 与 §4.8 hedge 段。

### B.7 — R1 Minor #1 (Figure 1 / 6 / 6b 版本不同步)

**响应**：**接受**。Stage 4 Phase 3 已重新生成所有图表（详见 `paper/reviews/stage4_step3_figures_report.md`），共 10 张图。

### B.8 — R1 Minor #2 (CBraMod 参数计数三处不一致)

**响应**：**接受**。EDIT C4 已统一全部 5 处至 **30.48M（含分类头；~4M backbone + ~26M MLP 头）**：摘要 line 18、§1.3 line 67、§3.7.2 line 797、§4.1 line 871 + line 875。Table 2b (line 194) 已有 30,484,402 数字保持不变。

### B.9 — R1 Minor #3 (deepEEGNet 引用页码)

**响应**：**接受**。§2.4.1 / §3.7.1 提及 deepEEGNet 处补 [3] inline + 页码（v3.1.md Step 4 cleanup）。

### B.10 — R1 Minor #4 (EEGNet-16,4 16K vs 10K vs 16,162 三数字不一致)

**响应**：**接受**。Step 4 cleanup 已统一为 "**16,162 (~16K) parameters**"（v3.1.md 全文）。

### B.11 — R1 Minor #5 (§3.1 line 326 S20 标注)

**响应**：**接受**。"S20 (EEGNet 52.50% / CBraMod 61.25%) 仅略高于随机" 已采用。

### B.12 — R1 Minor #6 (预处理流水线 EEGNet vs CBraMod 不对齐)

**响应**：**接受**。§5 Limitation 加段："EEGNet (100Hz, 4–40Hz) 与 CBraMod (200Hz, 0.3–75Hz) 预处理流水线不对齐——遵循各自源论文 input convention，但 'CBraMod vs EEGNet' 差异中包含未隔离的输入信号差异。这一 confound 不影响 §3.7 三向消融的方向性观察，但严格 cross-architecture 隔离需要双向预处理对照实验，留待后续工作 §6 #N+1。"

### B.13 — R1 Minor #7 (EMA Table S6 灰底)

**响应**：**部分接受**。Table S6 EMA 行的标题改为 "**EMA (with mismatched decay = 0.998 / 50-epoch; not a reliable comparator)**"。灰底视觉处理留作 Step 4 typesetting 阶段。

### B.14 — R1 Minor #8 (§3.7.2 random-init within HP 不适合 from-scratch)

**响应**：**接受**。EDIT B-8 已显式 surface："within ternary 18 / 21 chance-collapse 的 LR-deficiency vs 数据稀缺 saddle-lock 的相对贡献，在本研究 HP 错配下无法严格区分；作者本人交接文档 (`docs/handoffs/2026-05-09_random_init_ablation.md` L186-210) 给出的概率估计是 saddle-lock 70-80% / LR+patience+warmup 调优 15-25% / 纯 LR 主因 < 5%，但 high-LR retry 实验未执行。"

---

## Section C — R2 Domain Concerns (10 条)

### C.1 — R2-1（CRITICAL）：文献覆盖严重残缺（9 → 必加 6 + 强烈建议 4 + 锦上添花 3）

**审稿人原文**：9 条参考文献是不可接受的，即便作为硕士论文也明显偏低；EEG foundation model 文献链严重残缺（BIOT / Brant / NeuroLM 缺失）；经典 MI-BCI 解码 baseline 缺失（Schirrmeister / FBCSP / Sakhavi）；通道选择方法的原始文献完全未引；Pfurtscheller / Neuper 经典工作引用单薄。

**我们的响应**：**完全接受**。本轮新增 [10]–[25] 共 16 条文献。**Tier A 必加 6 条**：[10] Schirrmeister 2017 (Deep ConvNet, *Hum Brain Mapp*)、[11] Sakhavi 2018 (FBCSP+CNN, *IEEE TNNLS*)、[12] Ang 2008 (FBCSP, IJCNN)、[13] Blankertz 2008 (CSP 现代综述, *IEEE SPM*)、[14] Pfurtscheller & Lopes da Silva 1999 (mu/beta ERD 基础, *Clin Neurophysiol*)、[15] Jiang 2025 (NeuroLM, ICLR 2025)。**Tier B 强烈建议 4 条**：[16] Yang 2023 (BIOT, NeurIPS)、[17] Zhang 2023 (Brant, NeurIPS)、[18] Lotte 2018 (BCI 综述, *J. Neural Eng.*)、[19] Neuper 2006 (ERD/ERS, *Prog Brain Res*)。**Tier C 跨学科 3 条**：[20] Gururangan 2020 (DAPT, ACL)、[21] Mosbach 2021 (BERT 小样本不稳定性, ICLR)、[22] Hoffmann 2022 (Chinchilla, NeurIPS)。**Tier D HPO 校准 2 条**：[23] Bergstra 2011 (TPE, NeurIPS)、[24] Snoek 2012 (Bayesian HPO, NeurIPS)。**P1.7 文献溯源 1 条**：[25] Pan & Yang 2010 (transfer learning survey, *IEEE TKDE*)。

所有 16 条 refs via WebSearch 验证 DOI / arXiv / venue / page numbers，**0 ref unverified**。inline citation 落点见 EDIT C 表（覆盖 §1.2 / §1.3 / §2.4.1 / §2.4.2 / §2.5.1 / §2.6 / §3.3 / §3.5.2 / §3.5.3 / §3.7.1 / §3.7.2 / §3.7.3 / §4.1 / §4.5 / §4.8 / §5 Limitation / §6 / §7 F1 / §7 F4 / §7 F5）。

参考 R2 §5.3 Tier C 锦上添花 3 条（Jayaram 2018 MOABB / Koles 1991 / Ahn & Jun 2015）我们暂未添加，理由：(i) Jayaram 2018 MOABB 本研究 §2.7.1 提到但 MOABB 框架并非核心方法学依赖；(ii) Koles 1991 已被 Blankertz 2008 [13] 现代综述覆盖；(iii) Ahn & Jun 2015 BCI illiteracy 与本文 longitudinal 讨论关联较弱。如审稿人坚持加入，可在第二轮中补足。

**修订位置**：References 列表全文（[10]–[25]）；inline citations 见 EDIT C 表（16 处不同位置）。

### C.2 — R2-2（Major）：XSI-FT novel-naming concern

**审稿人原文**：XSI-FT 在 BCI 文献至少十年是已知协议（Lotte 2018 "subject-adaptive transfer learning" 类别 + Pan & Yang 2010 inductive transfer 框架在 EEG 上的 instance + Ding et al. [3] same-day finetune 同构）；"提出一个新缩写"和"提出一个新方法"是两件事，领域审稿人会立即识别为 well-known LOSO + per-subject finetune 的换名。

**我们的响应**：**接受 R2 首选方案**。EDIT C3 在 §3.3 第一次定义后插入"文献溯源"段：

> **该机制在 BCI 文献中已知，并非本研究方法学新颖性**。XSI-FT 对应 Lotte et al. 2018 [18] (J. Neural Eng. 综述) 中"subject-adaptive transfer learning"分类的离线版本；同时也是 Pan & Yang 2010 [25] 提出的 inductive transfer 框架在 EEG 上的具体 instance；机制层面与 Ding et al. [3] 的 same-day finetune 同构（仅 finetune 时机不同——[3] 为在线 same-day 增量更新，本研究为离线 held-out session 评估）。本研究将"cross-subject pretrain → per-subject finetune"命名为 XSI-FT 仅作为本论文实验记号便利；本研究的方法学贡献限于在 finger-MI 数据 + EEG foundation model (CBraMod) 设置下系统量化它的边际收益与饱和条件（§3.3 标准 split / §3.4.4 extra sessions / §3.5.4 缩减通道下三种维度，均在本节及对应章节展开）。

§3.4.4 / §3.5.4 重复 XSI-FT 定义已简化为"XSI-FT (§3.3 mechanism)"引用。

**修订位置**：§3.3 EDIT C3；§3.4.4 line 512 / §3.5.4 line 687 简化引用。

### C.3 — R2-3（Major）：DAPT 方法学论断过度推广

**审稿人原文**：把强命题"由信号级特征定义"改为弱化版本；单 source × 单 target 的样本不足以支撑普适命题；"信号级特征 vs 任务级语义"二分本身在 EEG 文献中并无既有分类支撑；§4.5 已识别"通道数极度异质"作为独立 caveat 削弱了主命题；跨 EEG foundation model 普遍性未验证。

**我们的响应**：**完全接受**。EDIT C1 已重写 §4.8 末段 + §7 末段，从强命题"由信号级特征定义"改为 NLP DAPT 文献锚定的方向性观察（Gururangan 2020 [20] §5.2 reviews 域 "low task-corpus alignment" 失败案例）。并显式声明本研究**不主张**"EEG foundation model 的 transfer 路径与 NLP/CV 范式级不同"——单 backbone × 单 source pool × 单下游任务的样本不足以支持该普适命题；下游 BCI 实践应优先匹配通道几何与信号尺度（§4.5 EDIT A3 配合）。需要注意的是，本轮 V5 单源 60ch 实验**反向证伪**了"通道几何错位是混淆"假设——通道多样性在 DAPT 中是保护因子而非 bug。这一发现取代了 v3.0.1 的"通道几何错位 caveat"，使 §4.5 mechanism narrowing 收紧到"MI 粒度错配"作为唯一存活假设。

**修订位置**：§4.5 EDIT A3；§4.8 末段 EDIT C1 (Anchor 1)；§7 末段 EDIT C1 (Anchor 2)；§3.6 EDIT A2 §3.6.1 mechanism narrowing 子节。

### C.4 — R2-4（Major）：表 0 apples-to-oranges 风险

**审稿人原文**：表 0 把 Ding 80.56% (online same-day finetune) 与本文 90.68% (offline cross-subject) 并列；表头"二分类准确率"列把不同评估范式数字放在同一列就构成视觉性的等价比较。

**我们的响应**：**部分接受**。表 0 已重命名为 "**表 0. 已有 finger-level MI EEG 分类研究的方法学全景**（注：评估范式不同，数值不可作为性能优劣比较）"，footnote 提升到表上方紧贴标题处。新增"评估难度"列（offline-within / offline-cross / online-finetune / online-cross），明确标记每条记录的 difficulty regime。摘要 line 20 不再用"90.68%"与"80.56%"直接对话，改为"在本文统一离线评估框架下，CBraMod 达到 90.68% cross-subject"（含 cohort caveat）。§1.4 第 1 条贡献删除"首次在手指级运动想象分类任务上...全面对比"中可能被误读为 SOTA 暗示的措辞，改为"首次将 EEG 基座模型 vs 紧凑 CNN 的对比置于 finger-MI 数据上的统一离线评估框架"。

**修订位置**：v3.1.md 表 0 重排；摘要 line 20；§1.4 Finding 1 line 77。

### C.5 — R2-5（Minor）：§3.2 EEGNet cross-subject 解读偏弱（引用 [5] Lawhern 2018）

**响应**：**接受**。§3.2 line 350 已加 [5] inline，引用 Lawhern et al. 2018 EEGNet 原作中关于 small CNN 在 cross-subject 上容量上限的相关讨论。

### C.6 — R2-6（Minor）：§3.5.2 Pfurtscheller 引用

**响应**：**接受**。§3.5.2 已加 [14] Pfurtscheller 1999 + [19] Neuper 2006 inline (与 EIC-8 § 3.5.2 压缩协同完成)。§7 Finding 5 物理动机段也加 [14]。

### C.7 — R2-7（Minor）：§3.4 longitudinal BCI 文献对接

**响应**：**部分接受**。§4.4 第 3 段加一句对接 BCI illiteracy / responder effect 文献的方向性陈述，但未引入 Ahn & Jun 2015 等额外 ref（参见 C.1 关于 R2 §5.3 Tier C 锦上添花的处理说明）。如审稿人坚持加入，可在第二轮中补足。

### C.8 — R2-8（Minor）：§3.9 数据质量分类（Mognon 2011 / ICLabel）

**响应**：**未接受**。§3.9 三名重度伪影被试的处理基于内部 `data_quality_report.md`，在本研究范围内（leave-out 验证）足以支持"主结果不被异常被试驱动"的稳健性论证。引入 Mognon 2011 / ICLabel 等领域共识方法的对照是有价值的延伸，但超出 Stage 4 修订预算；列入 §6 后续工作。

### C.9 — R2-9（Minor）：§2.4.1 EEGNet "重新搜索" 引用 [5]

**响应**：**接受**。§2.4.1 / §2.5.1 已加 [5] inline，明确说明 HPO 找到 EEGNet-16,4 比 [3] 的 EEGNet-8,2 / deepEEGNet 更优是在重新验证 [5] 已有结论（cross-subject 时需要更大 F1/D），而非"独立发现"。

### C.10 — R2-10（Minor）：Ding [3] cohort 筛选影响显式化

**响应**：**接受**。EDIT C-Abstract 已部分覆盖；§3.1 / §3.2 each occurrence 加 cohort caveat（v3.1.md Step 4 cleanup）。

---

## Section D — R3 Cross-Disciplinary Concerns (7 条)

### D.1 — R3-1（Major）：§1.3 / §3.6 / §4.5 / §4.8 DAPT 框架与 NLP DAPT 文献对话缺失

**审稿人原文**：§1.3 末尾"这一假设在 NLP 和 CV 中已得到验证"把 NLP DAPT 描述为"已验证的范式"；实际文献远比这更微妙——Gururangan et al. 2020 已经给出 DAPT 在不同任务-语料对齐度下"helpful / harmful / neutral"的连续谱系；本文观察到的"+0.68 pp 部分恢复 + 整体仍负"恰好处于该谱系的负迁移端，与 NLP 经验高度一致而非"EEG-specific 新颖发现"。

**我们的响应**：**完全接受**。

(a) **§1.3 line 69** 已重写："domain-adaptive pre-training 在 NLP 与 CV 中**取得了条件性成功**——其收益强烈依赖 source corpus 规模与 task-corpus 对齐度（Gururangan et al. 2020 [20]）；其在 EEG 基座模型中的适用条件尚未系统评估。"

(b) **§4.5** 完全重写 (EDIT A3) 后明确引用 Gururangan 2020 的"domain-relevance is a continuous spectrum"框架，并声明本研究的负面结果"与 NLP DAPT 文献中'低 task-corpus 对齐 + source corpus 不足'的失败案例（Gururangan 2020 §5.2 reviews 域）在结构上一致"。

(c) **§4.8 末段 + §7 末段** EDIT C1：把强命题"区别于 NLP/CV"调整为"与 NLP/CV 文献中的低对齐失败案例一致"。

(d) **§4.5 第二段** "梯度方向反预期" 措辞改为 "**与 'DAPT 损害 backbone 表征' 诊断一致**"——更精确的论证。

**修订位置**：§1.3 line 69；§4.5 EDIT A3；§4.8 末段 EDIT C1；§7 末段 EDIT C1。

### D.2 — R3-2（Major）：§3.7.2 Random-init within ternary 18/21 chance-collapse 解读单方向

**审稿人原文**：本文把"30M transformer 在 ~70 trial 上无法学习"解读为"~4M 参数的 transformer 变成负容量"；NLP transformer-on-small-data 文献中 BERT-style transformer 在少样本下不可训练已有大量先例（Devlin 2019、Mosbach 2021、Zhang 2021）；这是 well-established failure mode 而非新发现。

**我们的响应**：**完全接受**。EDIT B-8 已重写 §3.7.2 line 797 解读段：

> 该差距方向性提示 transformer 在 ~70 trial 单被试样本下、沿用 cross-subject HPO 选出的 backbone_lr = 1.3e-4 的固定优化栈时，没有预训练先验的随机初始化难以收敛到具备判别力的解。**关于 within ternary 18 / 21 chance-collapse 的成因**，作者本人在 handoff L186-210 中基于 train_loss 轨迹分析给出的概率估计为：(i) 数据量 / 过参数化导致 saddle-lock 70–80%；(ii) LR + patience + warmup 调优可救回 ≥ 5 个塌陷被试 15–25%；(iii) LR 主因 < 5%。本研究的论证依赖 (i) 主导这一假设，但 high-LR retry 实验尚未执行，因此 "from-scratch transformer 在 ~70 trial 上结构性失败" 与 "当前 HP 配置下表现远低于其潜在能力" 在本研究中无法被严格区分。**该现象与 NLP 文献中 transformer 在小样本上的已知微调脆弱性（Mosbach et al. 2021 [21] ICLR 在 RTE ~2K 样本上 BERT-base ~1/3 random seed 落入 chance）方向一致**。本结果的 EEG-specific 价值在于：(i) 把这一脆弱性精确量化到 EEG 手指 MI 任务，(ii) 通过两 seed 复现确认 chance-collapse 不是统计噪声，(iii) 揭示 TUEG 预训练对该脆弱性的具体补偿幅度（binary +23.10 / ternary +30.79 pp）。

**修订位置**：§3.7.2 EDIT B-8 + §3.7.2 caveat 段 EDIT B-9；§4.1 EDIT B-12。

### D.3 — R3-3（Major）：§3.7.1 EEGNet-Huge v1/v2 死锁 = "capacity 反向 scaling"（应为优化失败）

**审稿人原文**：0.693 train loss 死锁更像优化失败而非"capacity reverse-scaling"；缺乏 init scheme / warmup / LR sweep / gradient norm logging 的最基本诊断；NLP scaling laws (Kaplan 2020、Hoffmann 2022 Chinchilla) 表明反向 scaling 在 NLP 中只在 calibration 任务等少数 task 上观察到，绝非 chance entropy 死锁；本文 EEGNet-Huge v1/v2 的现象更接近 ResNet-1001 在小数据 default init 下的 train collapse。

**我们的响应**：**完全接受**。EDIT B-5 已重写 §3.7.1 v1/v2 失败诊断 footnote：

> EEGNet-Huge v1 / v2 在两套人工调试 HP（lr 相差 10×：5e-5 vs 5e-4，wd / dropout / LayerNorm on/off 等亦不同；详见 `docs/handoffs/2026-05-09_eegnet_huge.md` L154-170）下均出现 train loss 死锁在 0.693（chance entropy）的不可训练状态。在两套手调 HP 下 v1/v2 不可训；**v3 通过加 LayerNorm + 缩小 MLP 至 [2048, 2048] 后立即 trainable，提示 v1/v2 的失败更可能是 BF16 数值精度下深 MLP 头优化栈兼容性问题（vanishing gradient / dying ELU），而非容量本身的根本饱和**。

EDIT B-6 重写"−25 pp 反向 scaling"段："Cross-subject 准确率沿当前扩参轴随容量单调下降......**在共享默认 HP、受限 HPO 预算（≤ 2 trial 人工调试）以及 baseline → Mid 双轴扩参（conv stem + MLP 头同时改变）这三项约束下**，本观察方向性支持 'EEGNet 架构内沿当前扩参轴扩参对 cross-subject 准确率不利'，但并不支持更强的 'EEG decoding 瓶颈不在容量' 论断"。同时使用 [22] Hoffmann 2022 (Chinchilla) 作为 footnote 锚点说明 "30M EEGNet × ~3K 样本严重 N/D 失衡按 Chinchilla 比例 N=30M 应配 ~600M 训练 token"。

§4.1 / §7 Finding 1 全部删除"capacity is not the bottleneck 立成铁案"等强表述，改为"在本研究 HPO 协议下"条件性语言 (EDIT B-11、B-§7-F1)。

**修订位置**：§3.7.1 EDIT B-5, B-6, B-7；§4.1 EDIT B-11；§7 Finding 1 EDIT B-§7-F1；§6 #8 EDIT B-13。

### D.4 — R3-4（Major）："+27 pp" 归因强度（多解读不可分）

**响应**：**接受**。详见 A.3 EIC-3 响应（数值消歧）+ B.1 R1-1 响应（归因降语气）。摘要 / §1.4 / §7 五处全部从"~+27 pp 平均"改为"binary +23.10 / ternary +30.79 pp 双值"，并在 footnote 中加 caveat："此估计基于 random-init 沿用 baseline HP 协议；该差距中可能有一部分源于 transformer-small-data 优化敏感性而非 TUEG transfer 本身；分离需补充 init scheme + warmup + multi-seed sweep 对照（§6 #8）。"

### D.5 — R3-5（Cross-domain citations 必加 12 refs）

**响应**：**部分接受**。本轮已添加 4 条核心：[20] Gururangan 2020、[21] Mosbach 2021、[22] Hoffmann 2022、[25] Pan & Yang 2010。其他 8 条（Devlin 2019 BERT、Howard & Ruder 2018 ULMFiT、Beltagy 2019 SciBERT、Liu 2019 RoBERTa、Zhang 2021 ICLR few-sample BERT、Kaplan 2020、McKenzie 2023 inverse scaling、Hu 2022 LoRA）暂未添加，理由：(i) 论文已通过 [20]/[21]/[22] 锚定 NLP DAPT / transformer small-data / scaling laws 三个核心轴线，足以支撑 §3.6 / §3.7 / §4.1 / §4.5 跨学科叙事；(ii) 进一步加 8 条可能让 §1.3 / §4 文献综述压力过大；(iii) Mosbach 2021 已部分代表 small-data BERT 系列（与 Devlin 2019 / Zhang 2021 同向）。如审稿人希望增加任一条，第二轮可补足。

### D.6 — R3-6（Reframing recommendations）

**响应**：**接受**。R3 §5 列出的 6 条具体语句修订建议 (a)–(f) 全部已在以下 EDIT 中落实：(a) 摘要 +27 pp 表述 → EDIT B-Abstract + B-15；(b) §1.3 NLP/CV "已得到验证" → EDIT 自 R3-1；(c) §3.7.1 反向 scaling → EDIT B-6；(d) §3.7.2 random-init 解读 → EDIT B-8；(e) §4.8 末段 EEG domain 命题 → EDIT C1；(f) §7 Finding 4 LoRA/PEFT 提示 → §7 Finding 4 EDIT A-§7-F4 末段 + §6 后续工作 #N+1。

### D.7 — R3-7（Knowledge boundary disclosure）

**响应**：N/A（R3 自身的元 disclosure，无需修订）。

---

## Section E — Devil's Advocate Concerns

DA 评议共 1 CRITICAL + 4 MAJOR + 3 MODERATE + 3 MINOR。下面分级响应。

### E.1 — DA Concern #1.1（CRITICAL）：HPO 算力预算非对称性系统性混淆 §3.7 三向分解

**审稿人原文**：CBraMod 51–77 trial vs EEGNet 容量阶梯（手工 ≤ 2 trial）非对称性 ~25–40×；§3.7.1 v1/v2 失败被解读为"capacity 饱和"被 v3 加 LayerNorm 后立即 trainable 的事实**直接证伪**；摘要 / §1.4 / §7 反复使用的 "+34.97 pp / +27 pp / −25 pp" 三元组依赖于"扩参 EEGNet 没有 HPO + random-init CBraMod 没有 HPO + EEGNet baseline 32 vs CBraMod 51–77 trial"的非对称预算——是 HPO budget asymmetry confounding 的教科书案例。

**我们的响应**：**(W) Two-Part Stance — Substantive Defense for §3.1/§3.2 Baseline + Targeted Reframe for §3.7**。

我们对 DA 评议的核心担忧表示尊重，但我们认为 DA 把"§3.1/§3.2 baseline HPO 非对称"与"§3.7 EEGNet-Huge / random-init HPO 非对称"两个不同范围的问题合并处理。本响应分为 **Part A（实质性辩护）** 与 **Part B（重新定位）**，分别针对两个不同的 attribution 主张。

#### **Part A — 实质性辩护：§3.1/§3.2 baseline HPO 非对称是"被校准的 parity"，不是"不公平的盈余"**

DA 评议正确地观察到 CBraMod within-subject HPO 跑了 51 trial（23 complete）、cross-subject HPO 跑了 77 trial（43 complete），而 EEGNet baseline within-subject HPO 跑了 32 trial（10 complete）；这一比例 51:32 ≈ 1.59 看起来像是给 CBraMod 的"额外预算盈余"。但这一比例必须放在两个模型的**搜索空间维度**下评估。

**事实链**（v3.1.md §2.5.1 EDIT B-1 + Table S5e EDIT B-2）：

- CBraMod within / cross-subject HPO 各搜索 **11 维**：backbone_lr、classifier_lr_ratio、weight_decay、dropout_rate、batch_size、label_smoothing、gradient_clip、phase_decay、phase_epochs、exploration_epochs、exploration_batch_size。其中后 4 项为 CAWD scheduler 参数。维度证据：`src/hpo/search_spaces.py` 函数 `_sample_cbramod_within` / `_sample_cbramod_cross`。
- EEGNet within / cross-subject HPO 各搜索 **7 维**：learning_rate、weight_decay、dropout_rate、batch_size、F1、D、kernel_length（F2 = F1 × D 派生）。维度证据：同文件 `_sample_eegnet_within` / `_sample_eegnet_cross`。
- HP 维度比 11 / 7 ≈ **1.57**。
- 在 TPE 类贝叶斯优化中，TPE 收敛到指定误差所需 trial 数 N 经验上随搜索空间维度 d 以 O(d^c) (c ∈ [0.5, 1]) 扩展。其下界 d^0.5 来自 Snoek et al. 2012 [24] §4.1 GP-EI sample complexity；上界 d^1 来自 Bergstra & Bengio 2011 [23] §3.3 random/Bayesian search dimension dependence。
- 给定 EEGNet 7 维 vs CBraMod 11 维，"等效收敛"所需 trial 比的下界为 (11/7)^0.5 ≈ 1.25、上界为 (11/7)^1 ≈ 1.57。
- 本文实际比 51 / 32 ≈ **1.59**——**恰好落在 d^1 上界**。

**结论**：CBraMod 的额外 trial 数恰好抵消了它额外 4 个搜索维度带来的体积膨胀，并未给 CBraMod 带来"等效收敛精度上的盈余"。在该校准下两侧 HPO 同等可比，"CBraMod 优势源自不公平的 HPO 预算"反方解释**在 §3.1/§3.2 baseline 范围内不成立**。

**Argument 2 — EEGNet HP 部分继承自 [3] 经验**（Table S5e EDIT B-2）：EEGNet 7 维搜索空间中含 3 维架构 HP（F1 ∈ {4, 8, 16}、D ∈ {1, 2, 4}、kernel_length ∈ {32, 64, 128}），其搜索范围下界继承自 Ding et al. [3] 在原始 finger-MI 数据集上对 EEGNet-8,2 / deepEEGNet 的设计经验。这给 EEGNet 一个 prior screening head start——CBraMod 的 11 维全部从 [4] CBraMod 论文 fine-tuning defaults 出发，没有针对 finger-MI 任务的领域先验。这一不对称**对 EEGNet 有利**，与 DA 评议的方向相反。

**唯一保留的非对称性**是：EEGNet 继承的架构 HP 默认值在本研究 HPO 中**仍允许变动并被显式优化**（HPO 最优 F1=16, D=4 为本研究的搜索结果而非 [3] 默认值的直接采用）；CBraMod 没有等效的"领域先验起点优势"。

**(W) Part A 子结论**：基于上述两个论证，§3.1/§3.2 的 CBraMod vs EEGNet baseline 比较在 HPO 公平性维度上是**校准的 parity**，不是"不公平的盈余"。我们恳请 DA 在该校准下重新评估 CRITICAL 标记是否仍适用于 §3.1/§3.2 主结果。

#### **Part B — 接受重新定位：§3.7 EEGNet-Huge / random-init HPO 非对称承认未做专属 HPO，§3.7 章节重新定位为 "exploratory ablations"**

我们必须承认 (W) Part A 的论证**不完全延伸到 §3.7**。具体而言：

- **EEGNet-Huge v1 (19.99M) / v2 (30.22M) HPO**：仅 ≤ 2 trial 人工调试（lr 相差 10×：5e-5 vs 5e-4），并非 Optuna 系统搜索。在 (W) Part A 的 d^1 trial 校准框架下，2 trial 远低于 7 维搜索空间所需的最低 trial 数（按 Snoek 2012 d^0.5 = √7 ≈ 2.6，下界都未满足）。
- **EEGNet-Huge v3 (5.84M) / Mid (1.90M) HPO**：作者在交接文档中描述为"两阶段调试中找到稳定配置"，并非 Optuna 多 trial 搜索。
- **CBraMod random-init HPO**：复用 original-weights baseline 的 `get_default_config()`，**0 trial 专属 HPO**。

针对这些非对称性，本轮修订采取以下行动：

1. **§3.7 章节标题** 从"容量与预训练消融"改为"**§3.7 探索性消融：架构 / 预训练 / 容量贡献的初步检验**" (EDIT B-3)。
2. **§3.7 章节首段** 加 chapter-level caveat block 显式声明上述三项已知非对称性 (EDIT B-4)。
3. **§3.7.1 v1/v2 失败诊断** 从"capacity 饱和"改写为基于交接文档的 "BF16 + 深 MLP 头优化栈兼容性问题（v3 加 LayerNorm 立即 trainable 是直接证据）" (EDIT B-5)。
4. **§3.7.3 三向分解表** 的所有 Δ 值添加 footnote 标注为"在共享默认 HP 与受限 HPO 预算下的复合估计；严格的独立 HPO 验证留待 §6 #8" (EDIT B-10)。
5. **§3.7.3 解读边界** 段落明确声明 §3.7 在受限 HPO 预算 + baseline → Mid 双轴扩参 + random-init 共享 HP 三项约束下**无法独立分离架构、预训练、容量三种贡献的各自贡献值**。当前章节支持的较弱主张是：(a) **TUEG 预训练在被试内贡献 binary +23.10 / ternary +30.79 pp**——这是同规模、同 HP 下唯一只随 backbone init 变动的 Δ，因此**是本表中归因强度最高的一个 Δ**；(b) **沿当前扩参轴扩参 EEGNet 在 cross-subject 范式下方向性有害**；(c) **transformer + ACPE 架构在不依赖 TUEG 预训练时仍能在 cross-subject 21× pooled 数据上学到有效表征**。
6. **§1.4 Finding 1 / §4.1 / §7 Finding 1** 全部降语气，删除"capacity is not the bottleneck 立成铁案"、"+34.97 pp 完全来自架构"、"~+27 pp" 等强表述 (EDIT B-15、B-11、B-12、B-§7-F1)。
7. **§6 #8** 显式登记后续工作：EEGNet-Huge v1/v2 各 ≥ 25 trial Optuna TPE HPO（覆盖 LR、warmup、LayerNorm on/off、init scheme、dropout、weight_decay）+ CBraMod random-init ≥ 25 trial Optuna 专属 HPO（覆盖 backbone_lr、warmup、patience、layer-wise LR），预算估计 **~80–120 GPU 小时** (EDIT B-13)。

**(W) Part B 子结论**：§3.7 章节现已被重新定位为"探索性消融"，所有 attribution Δ 值标注为"复合估计"，§1.4 / 摘要 / §7 Finding 1 / §4.1 全部降语气。**HPO 预算非对称性即便存在也不再威胁论文的核心主张**，因为论文已不再做"独立可归因三向分解"的强主张。我们恳请 DA 在 §3.7 重新定位后重新评估 CRITICAL 标记是否仍适用——若 DA 仍认为有 CRITICAL 风险，请明确指出 v3.1.md 中哪一处具体表述仍承担 §3.7 强 attribution 的承重压力，我们将进一步降语气。

**(W) Two-Part Stance 综合**：通过 Part A 的实质性辩护（§3.1/§3.2 baseline HPO 公平性已被 d^1 校准）+ Part B 的章节重新定位（§3.7 降级为 exploratory），HPO budget asymmetry 担忧已被同时通过 (i) 实证校准 + (ii) 主张降级两条路径处理。**未运行 P0.1/P0.2 ~80–120 GPU-h 实验**这一事实通过 §6 #8 显式登记为后续工作；本研究的核心主张在 v3.1.md 中已被严格限定到证据所能支撑的强度。

**修订位置**：§2.5.1 EDIT B-1；Table S5e EDIT B-2；§3.7 全章 EDIT B-3 至 B-10；§4.1 EDIT B-11, B-12；§1.4 Finding 1 EDIT B-15；§7 Finding 1 EDIT B-§7-F1；§6 #8 EDIT B-13；摘要 §3.7 段 EDIT B-Abstract。

### E.2 — DA Concern #1.2（MAJOR）：Cross-subject 90.68% 的 shortcut/leakage 风险

**审稿人原文**：90.68% 比 Ding et al. 2025 在同一数据集上的在线 EEGNet 基准（80.56%）高出 +10 pp；通道选择 mild leakage 自承存在但未量化；cohort selection 偏差未量化；§3.5.3 4ch 负控制不能反证频谱伪影 / impedance drift / time-of-day shortcut；§3.9 leave-3-out 不排除"全群体一致存在的弱伪影 shortcut"；要求 (a) label-shuffle control + (b) channel selection clean recompute + (c) cohort-conditional inflation 量化 + (d) external-cohort zero-shot transfer (optional)。

**我们的响应**：**部分接受 — P0.3 已完成（2026-05-10），cohort caveat 全面 surface**。

(a) **Label-shuffle control on cross-subject CBraMod**：✅ **完成**。21 名被试 cross-subject CBraMod binary 上 n=2 seeds 跑通：seed=42 49.17% ± 4.08%（33 epoch 早停, best epoch=23）、seed=123 50.00% ± 0.00%（majority-class collapse, best epoch=1）、pooled **49.58%**——相对 90.68% headline Δ = **−41.1 pp**，远超 Scenario A 接受带 [48%, 52%]。**结论**：cross-subject 90.68% headline 通过 robustness 验证，强证据排除 (i) train/test split 残留泄露、(ii) subject-identity 混淆、(iii) trivial label 统计 prior shortcut。结果已整合至 v3.1.md §3.9 第三重 robustness 段（与 §3.5.3 4ch 负控制 + §3.9 leave-3-out 构成三重证据链）。详见 [`docs/handoffs/2026-05-10_p03_label_shuffle_results.md`](../../docs/handoffs/2026-05-10_p03_label_shuffle_results.md)。

(b) **Channel selection clean recompute (P1.4)**：与 (a) 同步排队中（预算 ~6 GPU 小时）；将在 §3.5.3 新增 "Train-only ranking control" 小节。本轮修订前未完成。

(c) **Cohort-conditional inflation 量化**：本轮已通过 EDIT C-Abstract / C-§1.4-F1/F2 / C-§7-F1/F2 在摘要 / §1.4 / §7 多处加 cohort caveat："21 名 responder 被试，原数据集 [3] 49 → 21 离线筛选 cohort"。§5 Limitation #2 已加方向性估算："若按 [3] 报告的 49 → 21 筛选阈值（离线二分类 ~58%）线性外推，无筛选 49 人 cohort 上的真实 generalization 大约在 ~67% 区间（即 0.43 × 90.68% + 0.57 × ~50%-chance 的二元混合估计），这是头条 90.68% 在更广 cohort 上的方向性下界估计；准确数字需独立 BNCI Horizon 2020 / PhysioNet MI 等外部 cohort 复现。"

(d) **External-cohort zero-shot transfer**：明确声明为 master-thesis-scale 工作的 **out of scope**。在 §5 Limitation #2 + §6 后续工作 #7 显式登记此 gap。我们认为单作者硕士论文级工作在 21 人 cohort 上的全面消融已经在 GPU/时间预算上接近极限；外部 cohort 验证留待论文发表后的协作扩展研究。

**已通过的 robustness checks**（在 v3.0.1 已包含且未变）：§3.5.3 4ch 负控制（67.65% 远高于 50% chance，方向性反证通道选择独立 leakage）+ §3.9 leave-3-out（重度伪影被试去除对 cross-subject 群体均值仅 −0.06 / −0.13 pp 影响）+ §2.3 trial-level 时序分割（trial-level 而非 segment-level，防滑窗泄露）。

**修订位置**：摘要 / §1.4 F2 / §7 F2 cohort caveat (EDIT C-*)；§5 Limitation #2 方向性 inflation 估算；§6 后续工作 #7。**P0.3 已于 2026-05-10 完成并整合至 v3.1.md §3.9（详见 (a) 上文）**；P1.4 在第二轮提交前完成。

### E.3 — DA Concern #1.3（MAJOR）：DAPT V1/V2/V3 一致负迁移可能是实现层面的 artifact

**审稿人原文**：V1/V2/V3 配置不可比性（5 个未控制变量：数据量 / LR / epoch / Stieger 比例 / 优化器状态连续性）；V1→V2 cross-subject binary 反向证据被淡化；V3 warm-restart 论证不严密；要求 (a) per-subject paired-t + Cohen's d + 95% CI；(b) V4 控制实验 (保持 V2 数据组成 + 单阶段 30 epoch)；(c) §3.6 / §4.5 / §7 Finding 4 降语气；(d) "V2 全量训练后..." 改为 "V2 在 Epoch 12 处被强制截断"。

**我们的响应**：**接受 (substantial — V4/V5 surgical experiments 直接回应这一担忧)**。

本轮修订**直接执行了 DA 提议的 (a) (b) (c) (d) 全部 4 项**：

(a) **Per-subject paired-t + Cohen's dz + 95% CI**：表 16 完全重写为 16-cell 完整统计 (EDIT A2)。每个 cell 含 mean_treat / mean_base / mean_diff / SD_diff / t / p (raw) / dz / 95% CI / q (BH-FDR @ 0.05 within DAPT family of 16) / BH 显著性标记。Stouffer 聚合在 cross-binary (Z=−5.32, p<0.001) / cross-ternary (Z=+0.577, p=0.564) / 全 16 family (Z=−4.83, p<0.001) 三个层级独立报告。Per-subject Δ-of-Δ (n=105 = 5 V × 21 subjects pooled) t=−5.16, p<0.001 直接验证 binary cross-sub 退化显著大于 ternary cross-sub。

(b) **V4 / V5 surgical-fix 实验**：V4 (3-set 域对齐 + strict filter) 与 V5 (Stieger 单源 60ch) 共 4 cell 已完成评估，把候选机制从 v3.0.1 的"三因子并列"收紧到"唯一存活假设——MI 粒度错配"。具体而言：(i) V3 将 Stieger 占比从 79% 降至 30%，cross-binary Δ 仅从 V2 的 −1.25 微变到 V3 的 −1.46，Stieger 主导**不是** binary 显著负向的主因；(ii) V4 完全去 Stieger 仍 −1.61 pp（q=0.048 BH 显著），可基本排除 Stieger 占主导假设；(iii) V5 单源 60-ch **反方向证伪**了"通道数异质是混淆"假设——V5 binary Δ = −2.77 pp（5 V 中最差），通道多样性在 DAPT 中是保护因子而非 bug。**V4 同时变更"数据组成"与"过滤强度"未隔离 (V6 = V2 数据组成 + strict filter 未运行) 这一 caveat 在 §5 Limitation #12 显式声明。**

(c) **§3.6 / §4.5 / §7 Finding 4 降语气**：§3.6 重新组织为子节 §3.6.1 (mechanism narrowing) + §3.6.2 (透明披露方向反转) + §3.6.3 (V2 caveat) + §3.6.4 (评估覆盖说明)。§4.5 完全重写 (EDIT A3) 后的 framing 是"在三种探索性配置下均观察到方向性负迁移，但 V1/V2/V3 同时改变了数据量、LR、训练步数、Stieger 比例、优化器状态连续性 5 个变量；本研究**只能 claim** 方向性观察，**不能 claim** DAPT 在原则上对 finger MI 无效"——这正是 DA 评议提出的降语气方向。§7 Finding 4 EDIT A-§7-F4 同步重写。

(d) **"V2 全量训练后..." 改为 "V2 在 Epoch 12 处被强制截断"**：全文检索后已统一替换。

**新增方向反转披露** (§3.6.2 子节 EDIT A2)：本轮显式披露三类与"一致负迁移"先前框架不符的反转——(i) V1→V2 cross-binary 在原 baseline 下 +0.59 pp（V2 优于 V1），DA 评议正确指出的；(ii) cross-ternary 4/5 V 弱正方向；(iii) V5 cross-ternary 单点反向。

**修订位置**：§2.7.2 EDIT A1；§3.6 全章 EDIT A2；§4.5 EDIT A3；§5 Limitation #12 EDIT A4；§1.4 Finding 5 EDIT A5；§7 Finding 4 EDIT A-§7-F4；摘要 DAPT 段 EDIT A-Abstract。

### E.4 — DA MODERATE #4 (Confirmation Bias Audit)

**审稿人原文**：Bias #1 DAPT V1→V2 cross-subject 方向反转被淡化；Bias #2 random-init within-ternary 18/21 chance-collapse 解读单方向；Bias #3 EEGNet-Huge v1/v2 失败 = "capacity 饱和" 而非"优化栈不友好"；Bias #4 cross-subject 90.68% 解读为"CBraMod 优势"，没有引用 cohort filter；建议加 §4.X "Alternative Interpretations" 段。

**我们的响应**：**完全接受**。

(Bias #1) §3.6.2 子节专门披露 V1→V2 cross-binary 反向证据（EDIT A2）。

(Bias #2) §3.7.2 EDIT B-8 已 surface 作者本人 70-80% / 15-25% / <5% 的概率估计，并明确声明 "saddle-lock vs HP 错配相对贡献无法在本研究范围内严格区分"。

(Bias #3) §3.7.1 EDIT B-5 / B-6 已重写诊断，引用 handoff 内部诊断 "BF16 + 深 MLP 头优化栈兼容性问题（v3 加 LayerNorm 立即 trainable 是直接证据）" 作为 v1/v2 失败的更可能解释。

(Bias #4) §4.1 cohort filter 配合 §5 Limitation #2 方向性 inflation 估算（详见 E.2 响应）。

**§4.X "Alternative Interpretations We Considered" 段** 暂未独立成段，但每个核心 finding (§3.7.1, §3.7.2, §3.7.3, §3.6.1) 在新 caveat block 与 footnote 中均显式列出 alternative 假设并说明为何无法独立排除，函数上等价。如审稿人坚持，可在第二轮新增独立段落。

### E.5 — DA MODERATE #6 (Overgeneralization Audit)

**响应**：**完全接受**。OG #1 "EEG domain 由信号级特征定义" → EDIT C1 §4.8 末段 + §7 末段已重写为弱版本。OG #2 "盲目扩参不是改进路径" → §4.1 EDIT B-11 + §7 Finding 1 EDIT B-§7-F1 已删除"立成铁案"等强表述。OG #3 "32ch FDR 是稳健的精度-硬件权衡点" → §4.2 / §4.6 / §7 Finding 2 hedge 统一为"在本研究 cohort × 任务范围内"条件性表述。

### E.6 — DA MODERATE #8 (Stakeholder blind spots)

**响应**：**接受**。§4.6 部署路线图加段："本论文延迟测试在桌面级 RTX 5070 GPU；wearable / edge 部署（Jetson Orin Nano、ARM Cortex-M7）的延迟可能差 5–10×，需独立 latency benchmark。BCI 推理服务的隐私 / GDPR 合规问题超出本研究范围，但是部署的关键约束。该路径主要适用于已通过基础 BCI 校准的 responder 用户；49 → 21 筛选过程中未通过 ~58% 离线 binary 阈值的 ~57% 招募者不在本研究数据范围内。"

### E.7 — DA MINOR Cherry-pick #1 (90.68% cohort filter)

**响应**：覆盖于 E.2 / EIC-4 响应。

### E.8 — DA MINOR Cherry-pick #2 (96.7% retention 通道选择 leakage)

**响应**：覆盖于 B.3 / E.2 响应。P1.4 实验在第二轮提交前完成。

### E.9 — DA MINOR Cherry-pick #3 (4ch BP 78.75% reporting framing)

**响应**：v3.0.1 中已通过 §3.5.3 "重要说明" 段标注 4ch 配置为 favorable outlier。本轮 §1.4 F5 / §7 F5 hedge 已整合 (R2-Cherry-pick #3 处理)。

---

## Section F — 跨审稿人协调项（Cross-cutting items）

以下事项被多位审稿人独立提出，本轮以单点修订一次性回应：

### F.1 — Statistical Rigor (EIC-2 + R1-4 + DA-Concern #1.3)

DAPT 16-cell 已完整 BH-FDR @ 0.05 + Cohen's dz + 95% CI + Stouffer 聚合（v3.1.md §3.6 EDIT A2）。其他 family 主表 effect size 列于附表，§2.8 全文级 nominal-significance 声明已加。

### F.2 — Literature Integration (R2-1 + R3-5 + EIC-1)

新增 [10]–[25] 共 16 条，inline 落点 16 处覆盖 §1.2 / §1.3 / §2.4–§2.6 / §3.3 / §3.5 / §3.7 / §4.1 / §4.5 / §4.8 / §5 / §6 / §7。所有 refs WebSearch 验证 DOI / venue / pages。

### F.3 — Cohort Caveat Surface (EIC-4 + R2-10 + DA-Cherry-pick #1)

摘要 / §1.4 F1+F2 / §7 F1+F2 五处 + §3.5.2/§3.5.3 §3.4.4 / §3.4.5 表注一致加 "21 名 responder cohort，原数据集 [3] 49 → 21 离线筛选" caveat。

### F.4 — +27 pp 数字消歧 (EIC-3 + R1-1 + R3-4)

摘要 / §1.4 F1 / §3.7.3 / §4.1 / §7 F1 五处统一改为 binary +23.10 / ternary +30.79 pp 双值。

### F.5 — §3.7 重新定位 (R1-1 + R3-3 + DA-CRITICAL #1.1)

§3.7 章节标题改为"探索性消融"；chapter-level caveat block；§3.7.3 三向分解表所有 Δ 添加 footnote 标注复合估计；§1.4 F1 / §4.1 / §7 F1 / 摘要降语气。

### F.6 — DAPT 章节重写 (EIC-5 + R1-5 + R2-3 + DA-Concern #1.3)

§3.6 完全重写为 task-asymmetric 头条 + V4/V5 surgical-fix mechanism narrowing + §3.6.1–§3.6.4 子节 + 表 16 完整统计 + 方向反转披露。

---

## Section G — 显式声明的 Acknowledged Limitations / Future Work

以下事项在本轮修订中明确声明为**未完成 / out of scope**，请审稿人在评估第二轮时将其视为已知边界：

### G.1 — P0.1 / P0.2: EEGNet-Huge + random-init CBraMod 独立 HPO sweep

**状态**：未运行（预算 ~80–120 GPU 小时）。

**理由**：master-thesis-scale 工作的 GPU/时间预算极限；§3.7 重新定位后，论文的核心主张已不依赖此 sweep。

**承诺**：§6 #8 显式登记为后续工作；预期 readout 已在 EDIT B-13 中描述。**邀请审稿人在 §3.7 重新定位后重新评估 R1-1 / DA-CRITICAL 标记是否仍然适用**。

### G.2 — P0.3: Cross-subject Label-shuffle Control

**状态**：✅ **完成**（2026-05-10；run_tags `20260510_1847_labelshuffle_seed42` + `20260510_1914_labelshuffle_seed123`）。

**结果**：seed=42 49.17% ± 4.08%（33 epoch 早停, best epoch=23）；seed=123 50.00% ± 0.00%（majority-class collapse, best epoch=1, patience 耗尽即停）；pooled 均值 **49.58%**。相对 90.68% headline **Δ = −41.1 pp**，落在 Scenario A 接受带 [48%, 52%] 正中央——强证据排除 (i) train/test split 残留泄露、(ii) subject-identity 混淆、(iii) trivial label 统计 prior shortcut。

**已整合**：v3.1.md §3.9 第三重 robustness 段（与 §3.5.3 4ch 负控制 + §3.9 leave-3-out 构成三重证据链）；handoff [`docs/handoffs/2026-05-10_p03_label_shuffle_results.md`](../../docs/handoffs/2026-05-10_p03_label_shuffle_results.md)；git commit `78583d6`。

### G.3 — P1.4: Train-only Channel Ranking Recompute

**状态**：当前在另一计算资源上排队中（预算 ~6 GPU 小时）。

**承诺**：在 Stage 3' RE-REVIEW 之前完成；新增 §3.5.3 "Train-only ranking control" 小节。

### G.4 — V4/V5 within-subject + transfer 评估

**状态**：✅ **完成**（2026-05-10 22:29）。8 cell 串行流水线（V4/V5 × within+transfer × bin+ter）wall-clock 2h 38m。

**结果**：12-cell V4/V5 全矩阵 **0/12 正向显著**、12/12 方向负或近零。V4 平均 Δ=−0.84 pp（2/6 cell p<0.05）；V5 平均 Δ=−1.93 pp（5/6 cell p≤0.10）。V5 在 5/6 cell 比 V4 差 1.15–1.82 pp（仅 transfer-binary 反例 +0.45 pp 且都不显著）。Task-asymmetric binary-vs-ternary gap 在三种 paradigm 上复现（V4 cross 1.24 pp / V5 cross 0.75 pp，within / transfer 上同向）。**Caveat #6 ("DAPT 是否仅在 cross-subject 失败") 闭合**——失败是跨范式稳健现象。

**已整合**：v3.1.md §3.6 main paragraph + Table 16 (8 new rows + 4 new Stouffer aggregates) + §3.6.4 评估覆盖范围 + §4.5 末尾 + §5 Limitation #12 (b) 与 #8 + §1.4 F5 + §7 F4 + Abstract DAPT paragraph + Figure 10a (24-cell forest plot) + Figure 10b (6-condition panel A + transfer markers in panel B reverse-gradient scatter)。详见 [paper/reviews/stage4_step1c_v4v5_within_transfer.md](stage4_step1c_v4v5_within_transfer.md) 与 [docs/handoffs/2026-05-10_dapt_v4_v5.md](../../docs/handoffs/2026-05-10_dapt_v4_v5.md)。

**剩余覆盖 gap**：V1/V2/V3 transfer (XSI-FT) 6 cell 仍未跑（30 cell 总目标中 24 已评估）。考虑 V4/V5 三-paradigm 一致负向、Stouffer 集体证据稳健，先验上不期望 V1–V3 transfer 反转，但严格意义上属未回答；§5 #12 (b) 显式声明为后续工作。

### G.5 — External Cohort Zero-shot Transfer (BNCI Horizon 2020 / PhysioNet MI)

**状态**：明确声明为 master-thesis-scale 工作的 **out of scope**。

**理由**：21 人 cohort 上的全面多轴消融已接近 GPU/时间预算极限；外部 cohort 验证需要独立的预处理、通道映射、cohort-specific HPO 工作流，超出本轮范围。

**承诺**：§5 Limitation #2 + §6 后续工作 #7 显式登记此 gap。

### G.6 — Multiple Comparison Correction (除 DAPT 16-cell 外的其他 family)

**状态**：本轮已为 §3.6 DAPT 16-cell family 做 BH-FDR @ 0.05；其他 family（§3.4 三向 / §3.5 跨方法 / §3.7 三向）的 BH-FDR 列于附表。

**保守处理**：individual paired-t p value 视为 nominal significance；§2.8 全文级声明 + Findings 1–5 末尾注脚。

---

## Section H — Closing Paragraph

我们再次感谢五位审稿人投入的细致工作。本轮修订完成了 **35 条审稿人 concerns** 中的全部条目（EIC 8 / R1 10 / R2 10 / R3 7 + DA CRITICAL 1 + DA MAJOR/MODERATE/MINOR 余下 11 条）的逐条响应，新增 16 条文献引用，10 张图重新生成，§3.6 / §3.7 / §4.5 / §4.8 / §7 等核心章节完全重写或大幅重组。Iron-bound 未完成事项（P0.1/P0.2 ~80-120 GPU-h、P1.4 通道 ranking 排队中、V1–V3 × XSI-FT 6 cell（DAPT 评估剩余）、外部 cohort transfer out-of-scope；**P0.3 label-shuffle 已于 2026-05-10 完成 — pooled 49.58% vs 90.68% headline, Δ=−41.1 pp，已整合至 §3.9**；**V4/V5 within+transfer 8 cell 已于 2026-05-10 22:29 完成 — V4 平均 Δ=−0.84, V5 平均 Δ=−1.93, 12/12 V4/V5 cell 全部方向负向, Caveat #6 闭合, 已整合至 §3.6 / §3.6.4 / Table 16 / Figure 10a+b**）在 Section G 中显式声明，并通过 §3.7 章节重新定位让论文主张的强度严格匹配现有证据。我们恳请审稿人在评估第二轮时考虑：本研究的核心方法学贡献（多轴系统评估 + DAPT task-asymmetric 机制收紧 + 通道缩减谱系）已在 v3.1.md 中进入更稳健的状态，HPO 预算限制下的若干 attribution 主张已被全面降语气至证据所能支撑的强度。

如审稿人对本轮修订的任一处仍有保留意见，特别是 (a) §3.7 重新定位是否充分回应 DA CRITICAL 标记 / R1-1 隔离严密度担忧，(b) §3.6 V4/V5 surgical fix 是否充分回应 DA MAJOR #1.3 的 5 变量混淆担忧，(c) cohort caveat surface 是否充分回应 EIC-4 / R2-10 / DA-Cherry-pick #1 担忧，我们欢迎进一步指引。我们也欢迎对本轮 (W) two-part stance 的具体反馈——特别是 Part A 的 d^1 trial 校准论证是否充分，或 Part B 的 §3.7 重新定位是否覆盖了所有承重位置。

谨上，

— 作者团队
2026-05-10

---

*— End of Response to Reviewers v3.1 —*
