# Editorial Decision Letter — paper_draft_v3.0.1.md

**Manuscript**: 基于 EEG 基座模型的手指级运动想象分类——通道缩减、纵向数据扩展与领域自适应预训练的局限性
**Author**: Bomin Zhang (单作者)
**Submitted to**: *Journal of Neural Engineering* (recommended target; *IEEE TNSRE* 备选)
**Decision**: **Major Revision**
**Date**: 2026-05-10

Dear Mr. Zhang,

感谢您将稿件提交至 *Journal of Neural Engineering*。本编辑部已收到 5 份独立评审报告（EIC + R1 方法学 + R2 领域 + R3 跨学科 + Devil's Advocate）。**5 位评审一致推荐 Major Revision**，无 Reject、无 Accept。综合 5 份意见，本刊认为稿件具备 *publishable methodological contribution*，但 §3.7 三向消融、§3.6 DAPT 负迁移以及若干头条数字的归因强度，超出了当前隔离严密度所能支撑的程度，必须修订后才可考虑接收。Devil's Advocate 提出 1 项 CRITICAL flag（§3.7 的 HPO 预算非对称），经本编辑部交叉核验您本人的工程交接文档（`docs/handoffs/2026-05-09_eegnet_huge.md`）后**得到确认**——这一点构成本次决定的核心。

**论文核心叙事与作者本人的工程证据存在直接冲突**。论文 §3.7.1（L764）将 EEGNet-Huge v1/v2 的失败诊断为 "提示这并非 HP 调优问题而是容量饱和"，并据此立稳"+34.97 pp 来自架构、+27 pp 来自预训练、−25 pp 来自容量"的三向分解，作为摘要、§1.4 Finding 1、§7 Finding 1 的 prominent quantitative claim。但您本人的 handoff 文档（`2026-05-09_eegnet_huge.md` L156、L195、L256）清楚地陈述：v3 (5.84M) 与 Mid (1.90M) 之所以 trainable，是因为 (a) 加了 LayerNorm，(b) 缩小了 MLP 头；v1/v2 (no LayerNorm + 深 MLP + BF16) 的 loss 死锁是 "BF16 + 深 MLP 头必须 LayerNorm" 的优化栈兼容性问题——v3 加 LN 后立刻 trainable 这一事实直接证伪了"capacity 饱和"的诊断。**这是一个事实层面的归因错误，不是修辞调整能修复的**——它要求或者补做对称的独立 HPO（v1/v2 各 ≥25 trial Optuna 搜索 + LayerNorm on/off 对照），或者将三向分解 framing 全面降级。R1 与 Devil's Advocate 在此点上观点完全一致；R3 从 NLP scaling-law 文献角度独立给出了相同结论。

**HPO 预算非对称是一个独立的、贯穿 §3.7 的方法学缺陷**。CBraMod baseline 享有 51 trial（within）+ 77 trial（cross）的 Optuna 系统搜索；EEGNet baseline 仅 32 trial（within，10 complete），EEGNet cross 在 Table S5b 中根本未列；EEGNet-Mid / Huge v1/v2/v3 与 random-init CBraMod 全部 0 trial 专属 HPO。算力预算非对称性 ~25–40×。作者在 §3.7.1 自承 v1/v2 失败是 "并非 HP 调优问题"——但**两套 HP 远不构成"capacity 饱和"的证据**。在两端均未做专属 HPO 的对照下，"+34.97 pp 完全来自架构归纳偏置"等于把 architecture × HP-undersearch × init-mismatch 的复合估计单独归因到架构上。

**统计报告深度低于 JNE 主流标准**。EIC 与 R1 一致指出：全文 ≥ 20 次独立 paired t-test 无任何多重比较校正（§2.8 自承）；所有主表只报 mean ± SD + p value，无 Cohen's d、无 95% CI。在 N=21（或 N=16 extra-sessions 子集）小样本下，单纯 nominal p 容易让审稿人怀疑 family-wise error。这是"达到 JNE 收稿基线"的最低补丁，必须在主表中补 paired Cohen's dz + 95% CI of mean difference + BH-adjusted q value。

**Cross-subject 90.68% 头条数字的 cohort caveat 不够 surface**。21 名被试是 [3] Ding et al. 在 49 名招募者中按 ~58% 离线 baseline 阈值筛选后保留的 responder cohort。Limitation #2 已承认但未量化；Devil's Advocate 与 EIC 一致要求在摘要 / §1.4 / §7 头条引用 90.68% 时显式标注 cohort 边界。同时，R1 与 Devil's Advocate 都要求至少补一项 **label-shuffle control**（< 6h GPU），独立验证 cross-subject 管线无 input → label shortcut——这是 §3.5.3 4ch 负控制（仅反证通道选择 shortcut）所不能替代的最低防线。

**DAPT V1/V2/V3 的"一致负迁移"主张被 5 个未控制变量与 V2 中断弱化**。Devil's Advocate 详细列出 V1/V2/V3 的不可比性矩阵（Stieger 占比、数据量、LR 调度、epoch 数、warm-restart 优化器状态连续性），并指出 V1→V2 在 cross-subject binary 上其实是 +0.59 pp 方向反转（被论文叙事掩盖）。R1 同时指出 §3.6 表 16 缺 paired-t per-subject p value——以本研究的 effect size，binary cross V3 vs Baseline (−1.31 pp, SD ~9 pp) 大概率达不到 p < 0.05。要求作者在表 16 补 paired-t + Cohen's d，并将 §3.6 / §4.5 / §7 Finding 4 从"一致负迁移"降语气为"三种探索性配置下的方向性观察"。

**领域文献覆盖密度严重不足**。R2 指出本稿仅 9 条参考文献——一篇覆盖 finger-MI / 通道缩减 / 纵向 / DAPT 四条研究线 + EEG foundation model benchmarking 的论文，应有 ~20 条。最严重缺失：(i) 同年 ICLR 2025 NeuroLM 与 BIOT、Brant 等 EEG foundation model；(ii) Schirrmeister 2017、Sakhavi 2018、FBCSP (Ang 2008)、Blankertz 2008 等经典 deep / classical MI baseline；(iii) §2.6 通道选择四种方法 (FDR/CSP/Attention/BP) 全无原始文献；(iv) Pfurtscheller & Neuper 1999 ERD 经典工作仅引 [2] (2001) 一条。R3 同时要求加 ~6 条 NLP DAPT / transformer-small-data / scaling law 跨学科锚定（Gururangan 2020、Mosbach 2021、Zhang 2021、Kaplan 2020、Chinchilla 2022、McKenzie 2023）。两位评审在 NeuroLM、Schirrmeister、Pfurtscheller 三条上有重叠；总计建议添加 10–13 条新文献。

**叙事过度推广问题**。"EEG 基座模型的 'domain' 边界由信号级特征定义、区别于 NLP/CV 的 DAPT 经验"（§4.8 末段、§7 Finding 4 末段）——这是基于单 backbone × 单 source pool × 单下游任务的观察，被升级为方法论命题。R2、R3、Devil's Advocate 三方一致要求弱化为"在本研究 (CBraMod, masked-AE, finger MI) 配置下的方向性观察"。R3 进一步指出：本研究的负迁移与 Gururangan 2020 ACL "low task-corpus alignment + insufficient corpus" 失败案例**结构上完全一致**，应作为 NLP DAPT 文献已有现象的扩展而非"EEG 范式级新发现"。同样，§3.7.1 EEGNet 容量阶梯被命名为"反向 scaling 现象"——按 McKenzie 2023 inverse scaling 定义（task-level miscalibration），本研究的 train loss 死锁不属于 inverse scaling，而是 BERT-style transformer-on-small-data 已知失败模式。

**XSI-FT 术语**：R2 明确指出此为 LOSO + per-subject finetune 的换名，在 BCI 圈是"造词陷阱"。建议保留缩写但在 §3.3 第一次定义时加文献溯源段（Lotte 2018 "subject-adaptive transfer"、Pan & Yang 2010 inductive transfer 框架）。EIC 倾向 R2 立场。

请在修订时优先解决 **Revision Roadmap** 中的 P0 项（共 4 条 must-do 实验 + 6 条 must-do 文本修订），P1 与 P2 可在第二轮审稿中协商。我们建议为本次修订留出 **8–12 周**（含 ~30–60 GPU-hour 实验 + 1–2 周写作）。修订稿需附 R&R Letter，逐条说明对每位评审员意见的回应（Roadmap 提供模板）。本编辑部对该工作的方法论原创性（特别是 §3.7 三向消融的实验设计本身）持肯定态度——若 P0 项妥善完成，第二轮可考虑将推荐改为 Minor Revision 或 Accept。

Sincerely,
*Senior Associate Editor, Journal of Neural Engineering* (IOPP)

---

## Reviewer Recommendation Tally

| Reviewer | Recommendation | Confidence |
|----------|---------------|------------|
| EIC      | Major Revision | 4 / 5 |
| R1 (Methodology / ML) | Major Revision | 4.5 / 5 |
| R2 (Domain / MI-BCI)  | Major Revision | 4 / 5 |
| R3 (Cross-disciplinary / NLP-CV) | Major Revision | 4 / 5 |
| Devil's Advocate      | Major Revision | 4.5 / 5 |

**Tally**: 5/5 Major Revision（无 Accept、无 Minor、无 Reject）。Devil's Advocate 触发 1 项 CRITICAL flag — Iron Rule 2 强制覆盖至 Major Revision 或 worse；当前结论自然落在 Major。

---

## Devil's Advocate CRITICAL Flags

### CRITICAL #1：HPO 预算非对称系统性混淆 §3.7 三向分解

**DA 引用证据**: Table S5b（CBraMod 51+77 trials vs EEGNet baseline 32 trials；EEGNet-Mid/Huge/random-init 0 trial 专属 HPO）+ `2026-05-09_eegnet_huge.md` L154-170（v1 vs v2 仅 LR 相差 10× 的两套手调 HP）+ L249-260（v3 加 LayerNorm 后 trainable，证伪 capacity 饱和诊断）+ `2026-05-09_random_init_ablation.md` L240（random-init 复用 baseline HP）。

**Cross-corroboration**: 
- **R1 §3.1**：完全独立得出同结论——"+34.97 pp 完全来自架构" 主张依赖于"两端均未做专属 HPO 的对照"，应降级为复合估计；R1 §3.2 同样独立列出 HPO 预算不对称表，与 DA 表完全一致。
- **R3 §3.3**：从 NLP scaling-law（Kaplan 2020、Chinchilla 2022）+ optimization 文献角度独立给出"0.693 train loss 死锁更像 optimization failure 而非 capacity reverse-scaling"；要求 LR × warmup × init × seed 完整 sweep。
- **EIC Concern 6**：识别 baseline → Mid 一跳同时改了 conv stem (F1/F2) + MLP 头双轴，自承 caveat 但叙事未弱化。
- **R2 §6.1**：未直接挑战 §3.7 隔离严密度（R2 信心 4/5 时自承"如果 R1 发现 §3.7 有问题，我对该部分 strength 判断需相应下调"——R2 实际把 §3.7 当作 strength）。
- 其余 4 位评审在 §3.7 这一点上有 4/5 corroboration（R2 中性偏正；R1/R3/DA/EIC 均为负向）。

**结论**: **CRITICAL CONFIRMED**——这是一个三方独立交叉验证 + 作者自身工程文档自证的事实层面归因错误。CRITICAL 级处置：升级为 P0 必做项（见 Revision Roadmap P0.1 / P0.2）。

---

## Consensus Findings (3+ reviewers agree)

| 主题 | 评审一致性 | 摘录 |
|------|-----------|------|
| §3.7 三向分解归因强度 over-claim | EIC + R1 + R3 + DA (4/5) | "+34.97 pp / +27 pp / −25 pp" 在 HPO 不对称下不可作为定量分解，需降级 |
| HPO 预算非对称（EEGNet-Huge / random-init 无独立 HPO）| R1 + R3 + DA (3/5) | EEGNet-Huge v1/v2 各 ≥25 trial 独立 HPO；CBraMod random-init 至少 25 trial HPO |
| 多重比较校正缺失 + Cohen's d / 95% CI 缺失 | EIC + R1 (2/5 显式 Major) + DA (隐式 paired-t) | 主表补 paired Cohen's dz + 95% CI + BH-adjusted q |
| Cross-subject 90.68% 的 responder cohort caveat 未 surface | EIC + R2 + DA (3/5) | 摘要 / §1.4 / §7 头条数字旁加 cohort 限定 |
| Channel selection mild leakage 未量化（FDR ranking 用了 test session）| R1 + DA (2/5；EIC 简单提及) | Train-only clean recompute；如 retention drop ≥2 pp 修订 96.7% 数字 |
| DAPT V1/V2/V3 "一致负迁移" 强归因不当 | R1 + DA (2/5) + EIC Concern 5 (隐式) | 表 16 补 paired-t + Cohen's d；弱化为"方向性观察"；surface V2 中断 + V3 warm-restart 干扰 |
| §4.8 / §7 末段 "EEG domain 由信号级特征定义" 过度推广 | EIC + R2 + R3 + DA (4/5) | 限定到"本研究 (CBraMod, MAE, finger MI) 配置下" |
| 文献覆盖严重不足（9 条 → ~20 条）| R2 + R3 (2/5；EIC 默认接受 R2 立场) | 必加 Schirrmeister 2017、NeuroLM 2025、Pfurtscheller 1999、Gururangan 2020、Mosbach 2021 等 ~10 条 |
| "+27 pp" 数值在三处叙述漂移（无加权平均 → 单点回写）| EIC Concern 3 + DA Cherry-pick #3 (2/5) | 改为"binary +23.10 / ternary +30.79 双数列出"或"+23–31 pp 范围" |

---

## Disputed Findings (reviewer disagreement + synthesizer arbitration)

| 议题 | 立场 A | 立场 B | 仲裁裁决 |
|------|--------|--------|----------|
| **EEGNet-Huge v3 独立 HPO 是否必做** | R1 + DA (必做 ≥25 trial Optuna) | R2 (倾向"EEG 领域 HPO 标准本就 ad-hoc")；EIC 倾向 R1 | **支持 R1 + DA**：作者本人 handoff 直接提供 v1/v2 失败属优化栈兼容性问题的证据，CRITICAL flag 已成立——Optuna sweep 不可让步。**P0**。 |
| **Label-shuffle control 是否必做** | R1 隐含；DA 明确（绝对必要）| EIC + R2 + R3 未明确要求 | **支持 DA**：自承 channel selection mild leakage + cross-subject 90.68% 比 [3] 高 +10 pp，单一 label-shuffle (<6h GPU) 是最低防线，无理由不做。**P0**。 |
| **DAPT V4 clean run（保 V2 数据 + 单阶段 30ep）是否必做** | DA 强烈建议；R1 建议 V2 retrain（同效）| EIC Concern 5 仅要求 disclose；R2 / R3 未要求 | **部分支持 DA**：V4 clean run 留待 P1（强烈建议但可协商）；P0 仅强制要求 §3.6 表 16 补 paired-t + 改写 §3.6/§4.5/§7 Finding 4 措辞。 |
| **XSI-FT 是否需重命名** | R2 (强烈建议溯源；最坏情况废弃缩写) | R1 / EIC / R3 / DA 未要求重命名 | **部分支持 R2**：保留缩写，但 §3.3 第一次定义时加文献溯源段（Lotte 2018、Pan & Yang 2010）；不强制改名。**P1**。 |
| **外部 cohort（BNCI IIa / PhysioNet MI）零样本验证是否必做** | EIC §5.4 暗示需要；DA OPTIONAL；R2 隐含 | R1 / R3 未要求 | **部分支持**：master-thesis-scale 之外的工作量；降级为 §5 Limitation 显式说明 + Future Work；**不**列入 P0。**P3**。 |
| **NLP DAPT 类比深度** | R3 (引用 ~6 条具体 NLP 文献并改写 §1.3 / §4.5 / §4.8) | R2 倾向"扯远了" | **支持 R3 但限制深度**：加 Gururangan 2020、Mosbach 2021、Kaplan 2020 三条核心文献作为 §1.3 / §3.7 / §4.5 footnote/段落锚定；不重写章节。**P1**。 |
| **CBraMod 参数计数三处不一致（~4M / 30.48M / ~10M）** | R1 §4.2（必须统一）| 其他评审未提 | **支持 R1**：摘要 / Table 2b / 各章节统一为 30.48M；这是审稿人当场会抓的不一致。**P0**（5 分钟编辑）。 |
| **EEGNet within HPO 32→50 trial 重跑** | R1 §7.1 强烈建议 | 其他评审未提 | **建议但不强制**：完成率 31% 偏低但 baseline 数字已用于全文锚定；P1。 |

---

## Verdict on "Author's Internal Handoff Contradicts Paper Narrative" Finding

**CONFIRMED**——经独立核验 `docs/handoffs/2026-05-09_eegnet_huge.md`（路径存在，确认）：

1. **Handoff L156**: "v3 通过加 LayerNorm + 缩 MLP 才让模型 trainable" — 直接说明 v1/v2 失败的工程根因。
2. **Handoff L195-197（教训段）**: "**BF16 + 深 MLP 头必须 LayerNorm**；MLP 不要超过 sqrt(N_segments) 量级" — 把失败明确归类为优化栈兼容性问题。
3. **Handoff L249-260（"v1/v2 失败的科学解读"）**: "v3 通过两点改动让模型变得 trainable：(1) 缩小 MLP 头到 [2048,2048]（30M → 5.84M）；(2) 每个 Linear 后加 LayerNorm" — 即 v3 的 trainability 既来自缩参也来自 LN，无法将 v1/v2 失败单独归因到容量。

**Vs 论文 §3.7.1 L764**: "...提示这并非 HP 调优问题而是容量饱和" — 这是一个**事实层面的错误叙事**（misalignment between author's own diagnostic notes and published claim）。在该归因下立稳"+34.97 pp 来自架构、−25 pp 来自容量"的三向分解，是 paper 最高优先级修复事项。

此外，`2026-05-09_random_init_ablation.md` L186-210 的"LR-deficiency 假设诊断"中作者本人估算"数据/saddle 假设 70-80%、LR + patience + warmup 15-25%、LR 是主因 < 5%" — 但论文 §3.7.2 把此 70-80% 概率主张写成 100% 因果归因（"~4M transformer 在 ~70 trial 上变成负容量"），同样是 confirmation bias。R3 独立从 NLP transformer-small-data 文献（Mosbach 2021 RTE 数据集 1/3 BERT random seed 落入 chance）给出相同诊断。

**这是本次评审最高利害的 finding**——它不仅触发 CRITICAL 级 P0 修订要求，也指向论文叙事过程中的 confirmation bias 模式（作者倾向把工程层面的 alternative explanation 转写为科学层面的 capacity claim）。修订稿需在 §3.7.1 / §3.7.2 / §4.1 / §7 Finding 1 全面降语气；如算力允许，建议补 §1.1 Action a 的对称 HPO 实验（v1/v2 各 ≥25 trial + LayerNorm on/off 对照）以尝试将归因严格化。

---

*— End of Editorial Decision Letter —*
