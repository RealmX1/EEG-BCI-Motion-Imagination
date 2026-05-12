# Revision Roadmap — paper_draft_v3.0.1 → paper_draft_v3.1

**Decision**: Major Revision (5/5 评审一致)
**Total revision items**: 6 P0 + 8 P1 + 6 P2 + 3 P3 = **23 items**
**Estimated total effort**: ~30–60 GPU-hour 实验 + ~2 周写作 + 0.5 周图表重生成
**Target turnaround**: 8–12 周

> 优先级解释：
> - **P0**：阻塞 acceptance；Devil's Advocate CRITICAL 或 3+ reviewer consensus；不做则第二轮仍 Major
> - **P1**：2 reviewer pair 或单评审 strong；做完显著提升 acceptance 概率
> - **P2**：单评审 good suggestion；锦上添花
> - **P3**：optional / future work / 与 master-thesis 边界冲突的 ask
>
> 每条标注 (Source) 指明评审来源；(Effort) 指实验 GPU-hour 或写作工作日。

---

## P0 — Must address (blocks acceptance)

### P0.1 EEGNet-Huge v1/v2 独立 HPO sweep + LayerNorm on/off 对照

- **Source**: R1 §3.1 + §7.1 优先级 A 项 1 + DA §1.1 CRITICAL Action a + R3 §3.3 + EIC Concern 6（隐含）
- **Severity**: CRITICAL — 决定 §3.7 三向分解的根基稳健性。作者本人 handoff `2026-05-09_eegnet_huge.md` L156/L256 明确：v3 加 LayerNorm 后立即 trainable，证伪了 v1/v2 的 "capacity 饱和" 诊断。
- **Action**: 在 EEGNet-Huge v1 (19.99M) 与 v2 (30.22M) 上各跑 **≥25 trial Optuna TPE HPO**；搜索空间 LR ∈ [5e-5, 5e-3] 对数均匀、warmup ratio ∈ [0, 0.2]、LayerNorm on/off (categorical)、init scheme ∈ {Kaiming, Xavier}、dropout ∈ [0.1, 0.6]、weight_decay ∈ [1e-3, 0.3] 对数均匀。Cross-subject binary 优先。
- **Estimated effort**: ~80–120 GPU-hour（v1/v2 各 25 trial × ~1.5h）。如算力受限，最低可接受版本：每个 v 至少 12 trial + 显式 LN on/off 对照（~50 GPU-hour）。
- **Acceptance criterion**: 
  - 若 100% trial 仍死锁在 train_loss=0.693 且 LayerNorm on 也无救→ 可保留"capacity 饱和"主张（强证据）。
  - 若加 LN 后 ≥1 trial 越过 chance（>55% cross binary）→ 必须重写 §3.7.1 + §3.7.3 + §4.1 + §7 Finding 1，把"−25 pp 完全来自容量"修订为"在受限 HPO 预算下观察"。
- **Output target**: §3.7.1（重写 v1/v2 解读段）；§3.7.3（更新三向分解归因强度）；新增 Table S5c（v1/v2/v3/Mid 独立 HPO 对照）。

### P0.2 CBraMod random-init 独立 HPO（≥25 trial）+ 文档归因降级

- **Source**: R1 §3.1 caveat 4 + DA §1.1 CRITICAL Action d + R3 §3.4 + EIC（隐含 via Concern 6）
- **Severity**: CRITICAL — "+27 pp 预训练贡献" 与 "+34.97 pp 架构贡献" 两个数字都依赖此对照的对称性。当前 random-init 复用 baseline HP（`2026-05-09_random_init_ablation.md` L240）+ 未跑 high-LR retry 实验（同 handoff L212-236）。
- **Action**: 在 random-init CBraMod 上跑 ≥25 trial Optuna 专属 HPO（覆盖 backbone_lr 1e-4 ~ 5e-3 对数均匀、warmup、patience、layer-wise LR）；优先 within ternary（最严重的 18/21 chance-collapse case），其次 within binary。
- **Estimated effort**: ~50 GPU-hour（25 trial × ~2h，within ternary 单 seed）。若仅做 high-LR retry（handoff L213 列出的 ~25 min 量级）则 < 5 GPU-hour，但只能 1 数据点。
- **Acceptance criterion**: 
  - 若 chance-collapse 比例从 18/21 降至 ≤ 8/21 → §3.7.2 / §4.1 / §7 Finding 1 + 摘要必须重写"+27 pp 预训练贡献"为更弱版本。
  - 若仍 ≥ 14/21 chance-collapse → 当前归因近似稳健，但仍需在 §3.7.2 加 Mosbach 2021 / Zhang 2021 文献锚定（见 P1.1）。
- **Output target**: §3.7.2 重写；§3.7.3 + §4.1 + §7 Finding 1 + 摘要更新；Table S5d（random-init 独立 HPO 报告）。

### P0.3 Label-shuffle control on cross-subject CBraMod

- **Source**: DA §1.2 MAJOR Action a（绝对必要）+ R1 §3.3（隐含）+ EIC Concern 4
- **Severity**: MAJOR — 论文最高 headline (90.68%) 比同数据集 [3] 在线 SOTA 高 +10 pp，自承通道选择 mild leakage（Limitation #1），单一 label-shuffle control 是最低防线。
- **Action**: 在 21 名被试 cross-subject CBraMod binary 上，把 trial-level label 随机重排（保持 input 不变），用同一 pipeline 重训。报告 test acc。
- **Estimated effort**: < 6 GPU-hour（cross-subject CBraMod 单跑 ~3h × 1-2 seed）。
- **Acceptance criterion**: test acc ∈ [48%, 52%]（chance ± 2 pp）→ pipeline 无 input→label shortcut；§3.5.3 / §3.9 / §4.1 加一段引用此结果。若 test acc > 55% → 存在 leakage，需深入调查（不期望发生）。
- **Output target**: §3.5.3 末段或 §3.9 末段新增"Label-shuffle control"小节；摘要 / §4.1 引用该结果作为 cross-subject 90.68% 的 robustness 支撑。

### P0.4 §3.7 / §4.1 / §7 Finding 1 / 摘要：归因强度全面降级（文本修订）

- **Source**: EIC Concern 1 + Concern 6 + R1 §7.3 项 6 + R3 §3.3-3.4 + DA §1.1 CRITICAL Action b/c + R2 §6.1
- **Severity**: CRITICAL — P0.1 / P0.2 实验未完成前，所有"+34.97 pp 架构 / +27 pp 预训练 / −25 pp 容量"措辞必须立即降级为受限观察。
- **Action**: 
  - **摘要 L18-22**：把"+34.97 pp 来自架构、+27 pp 来自预训练、−25 pp 来自容量"改写为"在 EEGNet-Huge / random-init CBraMod 均未做专属 HPO 的对照下，CBraMod 相对 EEGNet 的优势可分解为：架构差异（下界 ~X pp，待 P0.1 / P0.2 后填）、预训练贡献（binary +23.10 / ternary +30.79 pp，within-subject 范式）、EEGNet 内扩参在固定优化栈下的损害（~−25 pp）"。
  - **§3.7.1 L764**：把"提示这并非 HP 调优问题而是容量饱和"改写为"在两套手调 HP 下 v1/v2 不可训；v3 通过加 LayerNorm + 缩小 MLP 后立即 trainable，提示 v1/v2 的失败更可能是 BF16 下深 MLP 优化栈兼容性问题。是否在严格 HPO 预算下 30M 量级 EEGNet 仍不可训，留待 §6 后续工作"。
  - **§3.7.3 三向分解表**：所有 Δ 加 footnote "在共享默认 HP 与受限 HPO 预算下观察的复合估计"。
  - **§4.1 第一段**：去除"capacity is not the bottleneck"等类强表述。
  - **§7 Finding 1**：末尾加"该结论限于 CBraMod backbone × 本数据集 × 当前 HPO 预算"。
  - **§1.4 Finding 1**：将 "至 ~+27 pp（被试内）" 改为 "在被试内范式下 TUEG 预训练贡献 binary +23.10 pp / ternary +30.79 pp"。
- **Estimated effort**: 1 工作日写作。
- **Acceptance criterion**: 摘要、§1.4、§3.7.3、§4.1、§7 Finding 1 五处引用统一为带 caveat 的弱化版本；全文不再出现"capacity 饱和"作为 v1/v2 失败诊断的措辞。
- **Output target**: 摘要 / §1.4 / §3.7 / §4.1 / §7 Finding 1。

### P0.5 §3.6 表 16 补 paired-t + Cohen's d；改写 §3.6 / §4.5 / §7 Finding 4 "一致负迁移" 措辞

- **Source**: R1 §3.4 + §3.5 + DA §1.3 MAJOR Action a + EIC Concern 5
- **Severity**: MAJOR — V1/V2/V3 五个未控制变量（Stieger 占比、数据量、LR、epoch、warm-restart 优化器状态）+ V2 ep12 LMDB 中断 + V1→V2 cross binary 实际方向反转 (+0.59 pp，被论文叙事掩盖)。当前"一致负迁移"是 confirmation bias。
- **Action**: 
  - 表 16 升级为报告 paired-t per-subject p value、Cohen's dz、95% CI of mean difference（V1/V2/V3 各 vs Baseline；4 个 paradigm × task condition）。
  - §3.6 末段 + §4.5 第三段 + §7 Finding 4：从"V1/V2/V3 三种独立训练配置一致负迁移"降语气为"在三种探索性配置下均观察到方向性负迁移；V1/V2/V3 同时改变了数据量、LR、epoch 数、Stieger 比例、优化器状态连续性 5 个变量，因此本研究只能 claim 方向性观察"。
  - §3.6 显式 surface V1→V2 cross binary 的 +0.59 pp 反向证据（来自 `further_pretraining_analysis.md` §6.3）；不掩盖。
  - §3.6 / §4.5 显式 surface V3 warm-restart-from-weights 对 V3 vs V2 +0.68 pp 拆分的污染；明确指出 "+0.68 pp" 含 (Stieger 占比变化 + warm-restart 优化随机性) 复合效应。
  - §3.6 line 731 "V2 全量训练后..." 等措辞替换为"V2 在 Epoch 12 处被 LMDB 崩溃强制截断"。
- **Estimated effort**: 1 工作日（paired-t 计算 ~2h + 写作 ~6h）。
- **Acceptance criterion**: 表 16 含 d/CI/q；§3.6 / §4.5 / §7 Finding 4 不再出现"一致负迁移"作为 robust 结论的措辞；V2 中断 + V3 warm-restart + V1→V2 反向证据均显式 surface。
- **Output target**: §3.6 表 16 + 末段 + §4.5 + §7 Finding 4 + Limitation #12 强化。

### P0.6 主表全面补 Cohen's dz + 95% CI of mean difference + BH-adjusted q value

- **Source**: EIC Concern 2 + R1 §3.4 + §7.3 项 7-8
- **Severity**: MAJOR — 全文 ≥ 20 次独立 paired t-test 无任何多重比较校正；所有主表只有 mean ± SD + p value。这是 JNE 收稿基线。
- **Action**: 
  - 在所有报告 p value 的主表（Table 6, 11, 12a/b, 13a/b, 15, 18 等）补一列 paired Cohen's dz (= mean_diff / SD_diff) + 一列 95% CI of mean difference。
  - 在所有 p value 旁加一列 BH-adjusted q value（Benjamini-Hochberg FDR @ 0.05），或在 §2.8 + 主表脚注中显式说明哪些 p 在 BH @ 0.05 下仍显著。
  - §2.8 增加段："由于本文同时进行 ≥ 20 次独立配对检验，所有 p < 0.05 的结论应解读为 nominal significance；以 BH-adjusted q < 0.05 为更严格的显著性判据。Cohen's dz 与 95% CI 的报告允许读者评估各 finding 的实际效应量边界（在 N=21 / N=16 小样本下，单纯 p value 容易掩盖 family-wise error）。"
  - 摘要 / §1.4 / §7 Finding 1-5 末尾整体加一句 "All p values reported are nominal; significance under multiplicity correction is reported as BH-adjusted q in main tables."
- **Estimated effort**: ~1 工作日（Python 实现 Cohen's dz + BH 校正 ~30 行；表更新 ~6h）。
- **Acceptance criterion**: 主表每个 p 旁有 d、CI、q；§2.8 含校正声明；摘要 / §1.4 含 nominal-vs-corrected 提醒。
- **Output target**: §2.8 + Table 6/11/12/13/15/18 + 摘要 + §1.4 + §7。

---

## P1 — Should address

### P1.1 §3.7.1 / §3.7.2 跨学科文献锚定（NLP DAPT + transformer-small-data + scaling laws）

- **Source**: R3 §4 全部 + EIC §6.2 项 1
- **Severity**: 中 — 三个 NLP 锚定文献（Gururangan 2020 ACL、Mosbach 2021 ICLR、Kaplan 2020）能让本文核心 finding 锚定到 NLP transformer 已有失败模式上，从"EEG-specific 新发现"调整为"consistent with NLP 文献"——既不削弱贡献，也避免 R3 类审稿人对过度推广的反对。
- **Action**: 
  - §3.7.2 加一段（line ~798 后）"random-init within ternary 18/21 chance-collapse 的失败模式与 NLP 文献中 transformer 在小样本下的已知脆弱性一致：Mosbach et al. 2021 (ICLR) 在 RTE ~2K 样本上 BERT-base 约 1/3 random seed 落入 chance；Zhang et al. 2021 (ICLR) 给出 top-K layer re-init / long warmup / mixout 等稳定化方法。本研究的 EEG-specific 价值在于 (i) 把这一脆弱性精确量化到 finger MI 任务，(ii) cross-seed 复现确认非统计噪声，(iii) 量化 TUEG 预训练对该脆弱性的补偿幅度。"
  - §3.7.1 加 footnote "EEGNet-Huge v1/v2 的 train loss 死锁在 chance entropy 与 NLP/CV scaling-law 文献中 inverse-scaling 现象（McKenzie 2023）的形式不同——后者是 task-level miscalibration（更大模型更自信地学错），前者是 optimization-level 不收敛。本研究的 v1/v2 现象更接近 Goodfellow 2016 §8.4 描述的 saturated activation + vanishing gradient 失败模式。"
  - §1.3 / §4.5 / §4.8 加 Gururangan 2020 (ACL) 锚定 — 把 DAPT 负迁移描述为"NLP DAPT 文献已识别的 'low task-corpus alignment + insufficient corpus' 失败案例的 EEG 复现"而非"EEG 范式级新发现"。
- **Estimated effort**: 0.5 工作日。
- **Output target**: §1.3 + §3.7.1 + §3.7.2 + §4.5 + §4.8。

### P1.2 文献覆盖扩充（从 9 条 → ~20 条）

- **Source**: R2 §5 + R3 §4
- **Severity**: 中-高 — R2 单独提出"9 条参考是不可接受的"；不补会被领域审稿人直接拒。
- **Action**: 必加 6 条（R2 5.1 必加项）：
  1. Schirrmeister et al. 2017 (Hum. Brain Mapp.) — deep ConvNet
  2. Sakhavi et al. 2018 (IEEE TNNLS) — FBCSP+CNN
  3. Ang et al. 2008 (IJCNN) — FBCSP
  4. Blankertz et al. 2008 (IEEE Signal Proc. Mag.) — CSP / spatial filters
  5. Pfurtscheller & Lopes da Silva 1999 (Clin. Neurophysiol.) — ERD/ERS
  6. Jiang et al. 2025 (ICLR) — NeuroLM

  强烈建议 4 条（R2 5.2）：
  7. Yang et al. 2023 (NeurIPS) — BIOT
  8. Zhang et al. 2023 (NeurIPS) — Brant
  9. Lotte et al. 2018 (J. Neural Eng.) — BCI 算法 10-year update
  10. Neuper et al. 2006 (Prog. Brain Res.) — ERD/ERS sensorimotor

  跨学科加 (R3 §4)：
  11. Gururangan et al. 2020 (ACL) — Don't Stop Pretraining
  12. Mosbach et al. 2021 (ICLR) — BERT 小样本不稳定性
  13. Kaplan et al. 2020 — scaling laws

  锦上添花 (R2 5.3 + R3)：
  14. Jayaram & Barachant 2018 — MOABB
  15. Howard & Ruder 2018 — ULMFiT
  16. Hoffmann et al. 2022 (NeurIPS) — Chinchilla
  17. Hu et al. 2022 (ICLR) — LoRA

  对应文中 §1.3 / §2.4 / §2.6 / §2.7.1 / §3.7 / §4.1 / §4.5 / §4.8 / §6 多处补 inline citation。
- **Estimated effort**: 0.5 工作日（已有完整引用建议；只需 BibTeX 录入 + inline 插入）。
- **Output target**: References section + §1.3 / §2.4 / §2.6 / §2.7.1 / §3.5 / §3.6 / §3.7 / §4.5 / §4.8 各处 inline。

### P1.3 §4.8 / §7 末段 "EEG domain 由信号级特征定义" 命题降语气

- **Source**: EIC §6.2 项 1+3 + R2 §3.3 + R3 §3.1 修订建议 (e) + DA §6 OG #1
- **Severity**: 中-高 — 4/5 评审一致认为该命题基于单 cohort × 单 backbone × 单 source pool × 单下游任务，被升级为方法论命题超出证据。
- **Action**: 把 §4.8 末段 + §7 Finding 4 末段的 "EEG 基座模型的 'domain' 边界由信号级特征（采样率、频段、电极配置）而非任务级语义定义……区别于 NLP/CV 的 domain-adaptive pre-training 经验" 改写为：
  > "本研究观察的负迁移与 NLP DAPT 文献中 'low task-corpus alignment + source corpus 不足' 失败案例（Gururangan et al. 2020 §5.2）在结构上一致；在 CBraMod backbone × masked-AE 预训练目标 × 粗运动 MI source pool × finger MI target 的具体配置下，通道几何错位（target 128ch vs source 95% 低密度）与训练超参数对 DAPT 结果的影响至少与任务粒度相当。下游 BCI 实践应优先匹配通道几何与信号尺度，再考虑任务语义对齐。判断 EEG 基座模型是否需要不同于 NLP/CV 的 transfer 设计原则需要在多 backbone × 多 source corpus × 多预训练目标的矩阵下验证。"
- **Estimated effort**: 0.5 工作日。
- **Output target**: §4.8 末段 + §7 Finding 4 末段 + §6 后续工作 #5 强化。

### P1.4 Channel selection clean recompute（train-only ranking）

- **Source**: R1 §3.3 + §7.1 优先级 A 项 2 + DA §1.2 MAJOR Action b
- **Severity**: 中 — Limitation #1 自承"轻微高估"通道选择质量但未量化；32ch FDR 87.71% 是部署推荐数字。
- **Action**: 用 §2.3 的 train + val (排除 test session Sess02_Finetune) 重新计算 FDR / Band Power / Attention 的 32ch / 8ch / 4ch ranking；跑 cross-subject CBraMod 32ch FDR 一次。
- **Estimated effort**: ~30 min ranking 重计算 + ~4 GPU-hour 训练。
- **Acceptance criterion**: 
  - 若新 32ch FDR 准确率与原 87.71% 落在 ±1 pp → §3.5.3 + 摘要 + Finding 2 不变。
  - 若 drop ≥2 pp → 修订 96.7% retention 数字 + §1.4 Finding 2 + §7 Finding 2。
- **Output target**: §3.5.3 加 "Train-only ranking control" 小节；如有数字变化更新 §1.4 / §7 Finding 2。

### P1.5 摘要 / §1.4 / §7 头条数字加 "responder cohort" + "+27 pp" 双数列出 caveat

- **Source**: EIC Concern 4 + DA Cherry-pick #1 + R2 §4.7 + EIC Concern 3
- **Severity**: 中 — 已自承 (Limitation #2) 但未在头条 surface。
- **Action**: 
  - 摘要中 90.68% 第一次出现处加 "（21 名 responder 被试，原数据集 [3] 49 名招募者中筛选后 cohort，详见 §2.1）"
  - 摘要中 "至 ~+27 pp（被试内）" 改为 "在被试内范式下 binary +23.10 pp / ternary +30.79 pp（平均 ~+27 pp，但 binary/ternary 间幅度差异大，引用时应展开）"
  - §1.4 Finding 1 + §7 Finding 1 类似修订。
- **Estimated effort**: 5 min 编辑。
- **Output target**: 摘要 / §1.4 Finding 1 + 2 / §7 Finding 1 + 2。

### P1.6 §3.5.4 "XSI-FT 收益框架" 降级为 N=3 方向性观察

- **Source**: R1 §3.6 + Limitation #11
- **Severity**: 中 — 3 个数据点支撑"修订框架"叙事过强；自承不足但正文未弱化。
- **Action**: §3.5.4 + §4.6 / §4.8 把"XSI-FT 收益取决于 cross-subject baseline 离 (channel, method) 容量上限的距离"措辞从"修订框架"降级为"基于 3 个数据点的方向性观察 / 工作假设"。强烈建议补 8ch FDR XSI-FT + 32ch BP XSI-FT 两个数据点（~2 GPU-hour），把 N 从 3 提至 5。
- **Estimated effort**: 0.5 工作日写作 + 2 GPU-hour 实验（如做）。
- **Output target**: §3.5.4 + §4.6 + §4.8。

### P1.7 §3.3 XSI-FT 文献溯源段

- **Source**: R2 §3.2（首选方案，保留缩写但加溯源）+ EIC §5.2 默认接受 R2 立场
- **Severity**: 中 — BCI 圈"造词陷阱"会显著减分。
- **Action**: §3.3 第一次定义 XSI-FT 时加段：
  > "该机制在 BCI 文献中已知，对应 Lotte et al. 2018 (J. Neural Eng. 综述) 的 'subject-adaptive transfer learning' 类别 + Pan & Yang 2010 inductive transfer 框架在 EEG 上的 instance + Ding et al. [3] same-day finetune 的离线版本。本研究将 'cross-subject pretrain → per-subject finetune' 命名为 XSI-FT 仅作为本论文实验记号便利；本研究的方法学贡献限于在 finger-MI 数据 + EEG foundation model 设置下系统量化它的边际收益与饱和条件。"
- **Estimated effort**: 0.5h。
- **Output target**: §3.3 引言段；§3.4.4 / §3.5.4 引用处简化（避免重复定义）。

### P1.8 CBraMod 参数计数三处不一致统一

- **Source**: R1 §4.2
- **Severity**: 中 — 摘要 ~4M / Table 2b 30,484,402 / handoff "~10M" 三方不一致；审稿人当场会抓。
- **Action**: 全文统一为 **30.48M（含分类头）** 或 **~4M backbone + 26M MLP head** 二者之一（推荐前者）；摘要 line 18 + Table 2b + handoff 卡片同步。
- **Estimated effort**: 30 min。
- **Output target**: 摘要 / Table 2b / §2.4.2 / handoff 文档。

---

## P2 — Recommended

### P2.1 EEGNet within HPO 32→50+ trial 重跑

- **Source**: R1 §3.2 + §7.1 优先级 A 项 3
- **Action**: EEGNet within-subject HPO 从 32 trial 扩展到 50+ trial，确认完成率 ≥ 50% + best_value 移动 < 0.5 pp。
- **Effort**: ~3 GPU-hour。
- **Output target**: Table S5b 更新；§2.5.1 加 "HPO convergence verification" 段。

### P2.2 §3.5.4 补 8ch FDR XSI-FT + 32ch BP XSI-FT 数据点

- **Source**: R1 §3.6 优先级 B 项 4
- **Action**: 把 §3.5.4 的 N 从 3 提至 5。
- **Effort**: ~2 GPU-hour。
- **Output target**: §3.5.4 表 11c 扩展。

### P2.3 DAPT V4 clean run（保 V2 数据 + 单阶段 30 epoch）

- **Source**: DA §1.3 MAJOR Action b（强烈建议）+ R1 §3.5
- **Action**: 隔离 "Stieger 比例下降" vs "warm-restart 续训" 对 V3 vs V2 +0.68 pp 的贡献。
- **Effort**: ~6 GPU-hour。
- **Output target**: §3.6 / §4.5 + Limitation #12 强化。

### P2.4 §4.X "Alternative Interpretations" 段

- **Source**: DA §4 (Confirmation Bias Audit) Action A7
- **Action**: 加一段 §4.X 显式列出每个 finding 的 NULL 假设并说明为何当前数据不能完全排除。具体覆盖：(a) DAPT V1→V2 cross +0.59 pp 反向证据；(b) random-init 失败的 LR-deficiency 替代假设；(c) EEGNet-Huge 失败的 LayerNorm-optimization 替代假设；(d) cross-subject 90.68% 的 cohort responder filter 解释。
- **Effort**: 0.5 工作日。
- **Output target**: 新增 §4.9 "Alternative Interpretations We Considered"。

### P2.5 Cohort-conditional inflation 量化 (Limitation #2 强化)

- **Source**: DA §1.2 MAJOR Action c
- **Action**: 引用 [3] 49 人原始 cohort 的离线 binary baseline；用线性外推估算"无筛选 49 人 cohort 上的 generalization 大约在哪个区间"。
- **Effort**: 0.5 工作日（无新实验，仅文献查询 + 计算）。
- **Output target**: §5 Limitation #2 末段加估算。

### P2.6 §3.5.2 4ch BP 解剖学讨论压缩 + Pfurtscheller/Neuper 引用

- **Source**: EIC Concern 8 + R2 Minor #2
- **Action**: 把 §3.5.2 中段（i/ii/iii 三种 hypothesis）压缩到 1/3 长度；同时正式引用 Pfurtscheller 1999 + Neuper 2006（已含在 P1.2 文献扩充中）。
- **Effort**: 0.5 工作日。
- **Output target**: §3.5.2。

---

## P3 — Optional / Future Work

### P3.1 BCI Competition IV-2a 等外部 cohort zero-shot transfer

- **Source**: EIC §5.4 + DA §1.2 OPTIONAL d
- **Severity**: 低 — master-thesis-scale 之外的工作量；EIC 同意降级为 limitation 加强 / Future Work。
- **Action**: 在 BNCI Horizon 2020 IIa（9 名被试，左右手 MI）或 PhysioNet MI 上做 zero-shot 评估；如不做则在 §5 Limitation 显式说明为何外部验证在研究边界外。
- **Effort**: ~10 GPU-hour（如做）；0 工作量（如仅写 limitation）。
- **Output target**: §5 Limitation 加段；§6 Future Work 列出。

### P3.2 §4.6 部署主张 wearable / edge 边界

- **Source**: DA §8.1 (Stakeholder blind spots)
- **Action**: §4.6 加一句 "本论文延迟测试在桌面级 RTX 5070；wearable / edge 部署（Jetson Orin Nano、ARM Cortex）需独立 latency benchmark；隐私与端到端 BCI 闭环延迟超出本研究范围"。
- **Effort**: 5 min。
- **Output target**: §4.6 / §7 Finding 6 / §5 Limitation 新增条。

### P3.3 EMA 早停策略行高亮 / 灰底

- **Source**: R1 Minor #7
- **Action**: Table S6 中 EMA decay=0.998 (灾难性 −12.95 pp) 行加灰底或在标题改为 "EMA (with mismatched decay; not reliable comparator)"。
- **Effort**: 5 min。
- **Output target**: Table S6。

---

## Cross-cutting Stage 4 strategy

### 推荐执行顺序

**Phase 1（实验阶段，~2 周，~80–120 GPU-hour）**：
1. P0.1 EEGNet-Huge v1/v2 独立 HPO（需算力最大；优先启动）
2. P0.2 random-init CBraMod 独立 HPO（与 P0.1 并行）
3. P0.3 Label-shuffle control（短任务，可在 GPU 空隙穿插）
4. P1.4 Channel selection clean recompute（短任务，并行）
5. P2.1-P2.3 视算力余量补充

**Phase 2（统计 + 写作，~2 周）**：
6. P0.6 Cohen's dz + 95% CI + BH-q 全表更新（先做，后续依赖）
7. P0.5 §3.6 表 16 + DAPT 措辞改写
8. P0.4 §3.7 / §4.1 / §7 Finding 1 / 摘要全面降语气（依赖 Phase 1 实验数据）
9. P1.3 §4.8 / §7 末段命题降语气
10. P1.5 头条数字加 cohort caveat
11. P1.1 + P1.2 文献扩充 + 跨学科锚定
12. P1.6-P1.8 + P2.4-P2.6 文本修订

**Phase 3（图表 + R&R Letter，~1 周）**：
13. Figure 1 / 6 / 6b / 4b 重生成（同步最新数字）
14. R&R Letter 撰写（per reviewer per comment）
15. 整体 consistency check（摘要 / §1.4 / §3 / §4 / §7 数字 cross-reference）

### 章节联动 (linked sections)

修订时注意以下章节是 quantitatively linked，必须同步更新：

| 数字 | 出现位置 | 联动要求 |
|------|---------|----------|
| "+34.97 pp 架构" | 摘要 / §1.4 Finding 1 / §3.7.3 / §4.1 / §7 Finding 1 | P0.1 + P0.4 同步降语气；引用值要么按 P0.1 实验后填新数，要么改为区间 |
| "+27 pp 预训练" | 摘要 / §1.4 Finding 1 / §3.7.3 / §4.1 / §7 Finding 1 | 改为 "binary +23.10 / ternary +30.79 pp" 双数列出（5 处） |
| "−25 pp 容量" | 摘要 / §3.7.1 / §3.7.3 / §4.1 / §7 Finding 1 | 加 "在固定优化栈下" caveat |
| "90.68% cross-subject" | 摘要 / 表 0 / §1.2 / §1.4 / §3.2 / §7 Finding 2 | 加 "21 名 responder cohort" caveat（≥4 处）|
| "96.7% retention" | 摘要 / §1.4 Finding 2 / §3.5.3 / §4.2 / §7 Finding 2 | 视 P1.4 clean recompute 结果决定是否更新；加 "binary cross-subject" 限定 |
| "DAPT 一致负迁移" | §3.6 末段 / §4.5 / §7 Finding 4 | P0.5 同步降语气；加 V1→V2 反向证据 + V3 warm-restart caveat |
| "EEG domain 由信号级特征定义" | §4.5 / §4.8 末段 / §7 Finding 4 末段 | P1.3 同步降语气 |

---

## R&R Letter Skeleton (for Stage 4)

每条 reviewer comment 的回应模板：

```
**Reviewer [EIC/R1/R2/R3/DA]，Concern [§X.Y]**：[逐字引用评审原文]

**作者回应**: [简明承认 / 部分接受 / 反驳]
- 如承认：明确指出修订位置（§X.Y line Z），引用新增 paragraph
- 如部分接受：明确说明哪些部分接受、哪些不接受
- 如反驳：给出具体证据反驳；说明为何评审建议在本研究边界外

**修订动作**:
- §X.Y line Z 改写为：[新文本]
- 新增 §X.Y' 段：[新段落概述]
- Table T 增加列：[列名]
- 新增引用：[文献 ref number]

**预期审稿人 acceptance**: [High / Medium / Low]
**残留风险**: [若 reviewer 仍不满意可能的 fallback]
```

### Pre-allocated Reviewer Attribution per P0/P1 Action

| Roadmap Item | EIC | R1 | R2 | R3 | DA |
|--------------|-----|-----|-----|-----|----|
| P0.1 EEGNet-Huge HPO | ◑ | ✓✓ | — | ✓ | ✓✓ |
| P0.2 random-init HPO | — | ✓✓ | — | ✓ | ✓✓ |
| P0.3 Label-shuffle | ◑ | ◑ | — | — | ✓✓ |
| P0.4 §3.7 降语气 | ✓ | ✓✓ | ◑ | ✓ | ✓✓ |
| P0.5 §3.6 paired-t + 降语气 | ✓ | ✓✓ | — | — | ✓✓ |
| P0.6 主表 d/CI/q | ✓✓ | ✓✓ | — | — | ◑ |
| P1.1 NLP 锚定文献 | ◑ | — | — | ✓✓ | — |
| P1.2 文献扩充 ~10 条 | ◑ | — | ✓✓ | ✓ | — |
| P1.3 §4.8/§7 降语气 | ✓ | — | ✓✓ | ✓ | ✓ |
| P1.4 Channel clean recompute | — | ✓✓ | — | — | ✓ |
| P1.5 cohort caveat | ✓ | — | ✓ | — | ✓ |
| P1.6 §3.5.4 N=3 降级 | — | ✓✓ | — | — | — |
| P1.7 XSI-FT 溯源 | — | — | ✓✓ | — | — |
| P1.8 CBraMod 参数 | — | ✓✓ | — | — | — |

> ✓✓ = 主要 source；✓ = 强 support；◑ = 隐含或弱 support；— = 未提及

R&R Letter 应按上表分配，确保每位评审收到充分回应（Source 越多的评审，回应越详细）。

---

## Final Checklist (Stage 4 完成判据)

- [ ] P0 项 6/6 全部完成（其中 P0.1/P0.2 至少跑完最低可接受版本）
- [ ] P1 项 ≥ 6/8 完成
- [ ] 摘要 / §1.4 / §3.7 / §4.1 / §4.5 / §4.8 / §7 五处叙事一致性 verified
- [ ] 主表全部含 paired Cohen's dz + 95% CI + BH-q
- [ ] References 从 9 条扩充至 ≥ 18 条
- [ ] Figure 1 / 6 / 6b / 4b 重生成同步最新数字
- [ ] R&R Letter 撰写完成（每位评审每条 comment 至少一段回应）
- [ ] CBraMod 参数计数全文统一（30.48M）
- [ ] Cohort caveat 在 90.68% 头条 ≥ 4 处出现
- [ ] §3.7.1 不再出现 "capacity 饱和" 作为 v1/v2 失败诊断的措辞

完成所有 P0 项后，第二轮审稿 EIC + R1 + DA 大概率会将推荐改为 Minor Revision；R2 / R3 视文献覆盖完成度而定。

---

*— End of Revision Roadmap —*
