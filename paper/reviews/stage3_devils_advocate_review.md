# Devil's Advocate Review — Stage 3 Phase 1

**Reviewer Role**: Devil's Advocate (v1.1)
**Default stance**: skeptical
**Recommendation**: **Major Revision**（不是 Reject——证据基础值得发表；但 §3.7 + §3.6 在当前形式下不能支撑作者所做的归因主张）

> 本评审独立于其他 4 位评审员（EIC / R1 / R2 / R3），未参考其结论，唯一目的是把"最不友善但仍有据可查"的反方证据摆上桌。
>
> 评审基于：`paper_draft_v3.0.1.md`（v3.0.1，2026-05-10）+ `experiments.db`（已交叉核验 9 个核心 run_tag）+ `2026-05-09_random_init_ablation.md` + `2026-05-09_eegnet_huge.md` + `further_pretraining_analysis.md`。

---

## 1. Headline Concerns（最强 3 条）

### 1.1 [CRITICAL] HPO 算力预算的非对称性系统性混淆 §3.7 三向分解

这是本评审中**唯一一条 CRITICAL 级**的反对意见——因为它直接威胁论文摘要、§1.4 Finding 1、§7 Finding 1、§4.1 在三处独立位置反复使用的"+34.97 pp 来自架构、+27 pp 来自预训练、−25 pp 来自容量"这一定量分解。

**事实链 (cite by file:line)**：

1. CBraMod cross-subject baseline 由 **77 trial Optuna / 43 complete** 的 HPO 选出（Table S5b cross-subject 行），within-subject 由 **51 trial / 23 complete** 选出。
2. **EEGNet baseline 16K** 由 **32 trial / 10 complete** 的 HPO 选出（Table S5b within-subject 行；EEGNet cross-subject HPO 在 Table S5b 中**根本没有列出独立 HPO**——只复用了 within-subject 的 F1/D/F2 architecture，将其搬到 cross 范式）。
3. **EEGNet-Huge v1 (19.99M)** 与 **v2 (30.22M)** 的 HPO 是**两套独立 HP**——见 `2026-05-09_eegnet_huge.md` L154-170：v1 用 lr=5e-5、wd=0.2、dropout=0.6、no LN；v2 用 lr=5e-4、wd=0.05、dropout=0.4、no LN。两者都死在 train_loss=0.693（chance entropy for binary），val 50%，全部 21 名被试 test 50%。
4. **EEGNet-Huge v3 (5.84M)** 与 **EEGNet-Mid (1.90M)** 的 HPO 由作者在 handoff 中描述为"两阶段调试中找到稳定配置"（§3.7.1 文字），并非独立 Optuna 多 trial 搜索——见 §3.7.1 L750 引用的 "lr = 8e-4 至 1.5e-3、wd = 0.03–0.05、CAWD" 是单 HP 而非搜索分布。
5. **CBraMod random-init 没有跑专属 HPO**——见 `2026-05-09_random_init_ablation.md` L240-241："random-init 复用了 original-weights CBraMod 的 HP（即 `get_default_config()`）。两者唯一变量是 backbone init。"

**HPO 预算对照表（按 Optuna trial 数量）**：

| 模型 | 参数量 | Optuna trial 数（complete）| 用于 §3.7 哪一个 Δ |
|------|--------|-----------------------------|----------------------|
| CBraMod baseline (within) | 30.5M | 51 / 23 | +TUEG 预训练 +27 pp 主张的分母 |
| CBraMod baseline (cross) | 30.5M | 77 / 43 | +4.34 pp 顶端基线 |
| EEGNet baseline (within) | 16K | 32 / 10 | EEGNet 容量曲线的起点 |
| EEGNet-Mid (1.90M) | 1.90M | "两阶段调试" = **手动 ≤ 6 试** | −19 pp 跳跃 |
| EEGNet-Huge v3 (5.84M) | 5.84M | 同上 | −5 至 −7 pp 跳跃 |
| EEGNet-Huge v1 (19.99M) | 19.99M | **1 套 HP** | "扩参 → chance" 主张 |
| EEGNet-Huge v2 (30.22M) | 30.22M | **1 套 HP** | "扩参 → chance" 主张 |
| CBraMod random-init | 30.5M | **0**（复用 baseline HP）| +34.97 pp / +35 pp 架构主张的关键比较项 |

**问题严重程度**：

- **CBraMod（51–77 trial）vs EEGNet 容量阶梯（手工 ≤2 trial）**：算力预算非对称性 ~**25–40×**。
- 作者在 §3.7.1 自承 v1/v2 失败是 "提示这并非 HP 调优问题而是容量饱和"——但**两套 HP**（lr 相差 10×）远不构成"容量饱和"的证据；NLP 文献一致表明大模型对 HP 极敏感（warmup schedule、init scale、dropout、wd、LR）。BERT-large (340M) 早期同样有人报告过"无法在小数据训得动"，最终用 LayerNorm placement + LR warmup 才解锁。
- **致命的逻辑链**：作者对 v1/v2 失败的解读（"capacity 饱和"）被 v3 的成功（v3 仅是减小 MLP + 加 LayerNorm）所**直接证伪**——v3 比 v1/v2 参数少 5–6×，但 trainable，这恰好说明 v1/v2 的失败是**优化稳定性问题**（缺 LayerNorm、深 MLP 在 BF16 下 dying ELU），不是容量问题本身。`2026-05-09_eegnet_huge.md` L249-260 把这点说得非常清楚——但 §3.7.1 论文正文没有把这一关键归因转述给读者，反而称 v1/v2 "提示这并非 HP 调优问题而是容量饱和"（L764），这是**事实层面的错误叙事**。

**作者必须做的事（CRITICAL，否则论文不能 accept）**：

a. **跑 EEGNet-Huge v1 和 v2 各 ≥25 trial 的独立 Optuna 搜索**（与 CBraMod within-subject HPO 同等规模），覆盖 LR ∈ [5e-5, 5e-3] 对数均匀、warmup ratio ∈ [0, 0.2]、init scale、LayerNorm on/off、dropout ∈ [0.1, 0.6]。如果在这一搜索预算下仍然 100% trial 死锁在 train_loss=0.693，**那时**才能主张"capacity is not the bottleneck"。
b. 把 §3.7.1 当前的 v1/v2 论证降级为"在两套手调 HP 下 v1/v2 不可训，可能是优化稳定性问题（v3 缩 MLP + LN 后 trainable 直接证明此点）；扩参是否在 EEGNet 内有效，需独立 HPO 验证，本研究尚未做"。
c. **把 §1.4 Finding 1、摘要、§4.1、§7 Finding 1 中的 "+34.97 pp"、"+27 pp"、"−25 pp" 全部修订为更弱的 framing**——例如"在 default HP 共享与受限 HPO 预算下观察到的 Δ"。
d. **跑 CBraMod random-init 至少 1 次独立 HPO**（哪怕 25 trial，与 EEGNet within-subject HPO 同规模即可）——否则"+34.97 pp 来自架构"的归因被"random-init 用了 original-weights HP"这一明显次优配置所内蕴削弱。

如果 (a) 不可行（算力上不允许），论文的 §3.7 必须重写为"observation"而非"three-way decomposition"——叙事降级为"在受限 HPO 预算下，扩参 EEGNet 的 ~30M 配置不能达到 trainable 状态；结合 v3 的 trainability，这暗示优化栈对 EEG decoding 的扩参 CNN 不友好——但'capacity is not the bottleneck'尚需更系统的 HPO 验证"。

**这是 CRITICAL 而非 MAJOR 的原因**：摘要、§1.4 Finding 1、§7 Finding 1 都把"+34.97 pp / +27 pp / −25 pp"作为 paper 的**最 prominent**贡献之一。如果这一三向分解不稳健，paper 的核心叙事——"基座模型价值在 cross 由架构主导、在 within 由预训练主导"——的定量基础就被掏空。这不是修辞调整能修复的，必须做实验或重写。

---

### 1.2 [MAJOR] Cross-subject 90.68% 的 shortcut/leakage 风险未被 §3.5.3 + §3.9 充分排除

§3.2 的 cross-subject CBraMod binary 90.68%——比 Ding et al. 2025 [3] 在**同一数据集**上的在线 session-adaptive EEGNet 基准（80.56%）**高出 +10 pp**——是论文最 prominent 的数字之一（出现在摘要、表 0、§1.4 Finding 1、§7 Finding 2、§4.1）。这种超过原研究 +10 pp 且在更难的 cohort-generalization 范式下取得的差距，需要**比当前更严格**的 shortcut/leakage 检验。

**问题事实链**：

1. **作者自承的"通道选择 mild leakage"**（Limitation #1，§5 表）："FDR、CSP、Attention、Band Power 指标使用了所有 session 数据（含测试 session 上下文）计算……可能轻微高估通道选择质量"——这是作者**自己**承认的 leakage 路径。但论文未量化"轻微"——可能是 0.5 pp 也可能是 5 pp，没有 clean recompute 对照。
2. **Cohort selection 偏差**（Limitation #2）：21 名被试是 [3] 在 49 名招募者中按"离线二分类准确率达到 ~58% 阈值"筛选后保留的 responder cohort。这意味着 **21 / 49 ≈ 43%** 的被试已经被预筛掉。"对普通受试者的泛化可能被高估"——但论文没有报告该筛选所导致的 cohort-conditional inflation 估计（即"如果在原始 49 人无筛选 cohort 上重训会落到多少"）。
3. **§3.5.3 4ch 负控制**虽好，但它**只反证了 "通道选择是否构成数据泄露"**——它没有反证：
   - **Trial-onset 频谱伪影泄露**（如 cue onset 后 0–200ms 内的视觉诱发 SSVEP 被模型当作"trial start marker"）
   - **Time-of-day artifacts**（offline session 在上午、Sess02 Finetune 在下午——若被试在不同时段的 baseline 阻抗/伪影模式不同，模型可能学到 session marker）
   - **Impedance drift carrying label hints**（Sess02 Finetune 内部的 trial-level 阻抗漂移与 trial label 的相关性）
4. **§3.9 leave-S04/S10/S14-out** 是好的健壮性检验（Δ < 0.2 pp），但它**只**排除了"重度伪影被试个体"对群体均值的影响，**不**排除"全群体一致存在的弱伪影 shortcut"。

**风险评估（"为什么 90.68% 比 Ding et al. 高 10 pp"的 4 个非互斥假设）**：

| 假设 | 解释 | §3 是否充分排除 |
|------|------|------------------|
| (i) Foundation model 真的更强 | TUEG 预训练 + transformer | §3.2 主张此点，但 §3.6 DAPT 负迁移与 §3.7.2 random-init cross 86.34% 一起说明：架构 + pooled 数据是关键，TUEG 仅 +4 pp |
| (ii) Cohort responder filter | [3] 把 49 人筛到 21 人，留下"BCI-amenable"用户 | Limitation #2 承认，但**没有量化**——若按线性插值，原 49 人 cohort 上准确率可能是 0.43 × 90.68% + 0.57 × ~50% (chance for non-responders) ≈ **67%**，这才是"真实" generalization 数字 |
| (iii) Offline vs online 评估范式差异 | 离线无实时反馈、无 same-day update、无 majority voting 跨长时窗口 | 表 0 的"可比性说明"承认，但**没有**做 majority-vote 评估让两者可比 |
| (iv) Subtle leakage（频谱伪影 / impedance drift / 通道选择穿透）| 见上 | 通道选择 leakage 自承存在，其他路径未测试 |

**作者必须做的事（MAJOR）**：

a. **Label-shuffle control on cross-subject CBraMod**（**绝对必要**）：把 21 名被试的 trial label 在 trial-level 随机重排，然后跑同一 cross-subject pipeline。如果模型 test acc 仍然 > chance + 2 SD，那就有 leakage；如果模型落到 50% ± 2pp，那"管线本身无 shortcut"才被独立确认。这个实验**不是**4ch 负控制能替代的——4ch 负控制反证的是"通道选择 shortcut"，label shuffle 反证的是"任何 input → label 的 shortcut"，两者作用域不同。**预算：cross-subject CBraMod 训练一次 ~3h，label shuffle 1–2 seed 即可，总预算 < 6h GPU**——作者没有任何借口不做这个。
b. **Channel selection clean recompute**：把 FDR/CSP/Attention/BP 排序仅用训练 session（不含 Sess02 Finetune）重算，跑一遍 cross-subject 32ch FDR。如果新 32ch FDR 准确率与原 87.71% 落在 ±1 pp 内，§3.5.3 的"无 leakage"主张可获更强证据；如果 drop ≥2 pp，则需修订 96.7% retention 这一数字。
c. **Cohort-conditional inflation 量化**：在 Limitation #2 中加一段，引用 [3] 的 49 人原始 cohort 的离线 binary baseline，用线性外推估算"无筛选 49 人 cohort 上的 generalization"——这至少给读者一个数量级感（避免读者误以为 90.68% 是无筛选 cohort 上可以达到的）。
d. **External-cohort zero-shot transfer**（理想但 OPTIONAL）：在 BNCI Horizon 2020 Dataset IIa（9 名被试，左/右手 MI）或 PhysioNet MI 上做 zero-shot 评估——这是 EEG foundation model 论文的**期刊默认配置**（CBraMod 原文 [4] 在 10 个公开 benchmark 上评估）。本论文只在单一 cohort 上声称 90.68%，没有任何外部数据集复现，是"basic publication 标准"以下。

只有 (a) 是**绝对不可让步**的——其他三点可在 minor revision 中协商。

**为什么 MAJOR 而非 CRITICAL**：本研究确实做了 §3.5.3 4ch 负控制 + §3.9 leave-3-out + §2.3 trial-level temporal split——已经超过 BCI 文献常见的 leakage 检查门槛。但因为 Limitation #1 自承通道选择确实有"mild leakage"，且核心数字 90.68% 比同数据集前作高 10 pp，单一一项 label-shuffle control 是绝对必要的最低防线。

---

### 1.3 [MAJOR] DAPT V1/V2/V3 的"一致负迁移"可能是实现层面的 artifact 而非 finding

§3.6 + §4.5 + §7 Finding 4 把"V1/V2/V3 三种独立训练配置下均出现一致负迁移（−0.75 / −1.38 / −0.70 pp）"作为对"DAPT 在精细 finger MI 任务上无效"的**核心证据**。但当我把作者自己在 §2.7.2 disclosed 的 caveat 摆出来一一审视，我看到的是**三个互不可比的训练配置 + 一个论文叙事强行套上"一致性"标签**：

**V1 / V2 / V3 配置不可比性矩阵**：

| 维度 | V1 | V2 | V3 |
|------|-----|-----|-----|
| Stieger 占比 | ~52% (23/62 被试) | ~79% (62/62) | ~30% (按 segment 子采样) |
| 总数据量 | 30,282 segments | 78,232 | ~46K |
| LR schedule | Cosine decay → 1e-6 | Warmup 0.5ep → 恒定 5e-5 | 恒定 5e-5 |
| Epoch 数 | 10 (cosine 跑完)| 50 计划 / **Epoch 13 LMDB 崩溃** | 15 ep 初次 + **warm-restart 12 ep 续训** |
| 优化器状态连续性 | 单阶段 | 单阶段，Epoch 13 之后没有数据 | **断裂**（仅 weight 续训，optimizer + scheduler 重置）|
| 训练充分度 | 不充分（cosine 提前衰减）| 提前停（崩溃）| 优化器重置后 12 ep（非自然续训）|

**作者自承的 caveat 来自原文（cited verbatim）**：

- §2.7.2 V1 caveat："V1 使用了部分下载的外部数据集……总计 30,282 segments。"
- §2.7.2 V2 caveat："V2 完成了两个大型数据集的全量下载……V1 和 V2 之间不仅训练配置不同（LR 调度、epoch 数），**数据组成也不同**，下游结果差异不可归因于单一因素。"
- §2.7.2 V3 caveat："V3 训练分两阶段：(i) 初次训练 15 epoch……(ii) 在该 best checkpoint 基础上做 continue training 12 epoch，**采用 warm-restart-from-weights 策略**（仅恢复模型权重，不恢复 optimizer 与 LR scheduler 状态；初始 LR 重置为 5e-5）……V3 与 V1/V2（单阶段训练）的"训练充分度"不严格可比"。
- §4.5 实质内容："V1 和 V2 同时改变了数据量、LR 调度和训练步数（2,360 vs 7,776），因此无法将负迁移的加剧严格归因于单一因素。"
- Limitation #12："V1/V2/V3 均为单次 pre-training 尝试……观测到的负迁移可能部分源于 (i) DAPT 方法配置（mask ratio、loss 公式）与 MI 数据不匹配……"

**作者已经把 V1/V2/V3 的不可比性诚实陈述了三次。**问题是：诚实陈述完之后，§3.6 倒数第二段、§4.5、§7 Finding 4 又**回到了**"V1/V2/V3 一致负迁移"作为 robust 结论。这是一个**有点伤眼的逻辑滑坡**——前文承认三个版本不可严格比较，后文又把它们的方向一致性作为论据。

**对方向一致性的反方解释**：

- **"3 个 negative 数据点"的统计无力性**：V1: −0.75, V2: −1.38, V3: −0.70。这些 deltas 比被试间 SD（~10–14 pp）小一个数量级。论文只报告了 group mean delta，但**没有报告 paired-t per-subject p-value for V vs Baseline**——这是 DAPT 论文的标准报告。论文 §3.6 表 16 列了"V3 vs Baseline = −1.31 pp (cross binary)"，但没有 p value。如果 paired-t p > 0.10（大概率，因为 effect size < SD/4），那"一致负迁移"在 N=21 上根本无统计显著性。
- **唯一一次方向反转的内涵**：V1 vs V2 在 cross-subject binary 上 V2 反而**比** V1 好（V1 = −1.70 pp，V2 = −1.11 pp，§5.3 来自 `further_pretraining_analysis.md`）。但论文 §3.6 + §7 Finding 4 完全略过这个数据点——这就是 **confirmation bias**（详见 §4 节）。
- **V3 warm-restart 的关键性**：作者在 §2.7.2 caveat 中明确：V3 第二阶段 12 epoch 在**优化器重置后**进行——这相当于 "0.5-LR Warmup → 重新开始" 的另一次训练而非自然 continue training。这意味着 V3 的"+0.68 pp 相对 V2 改善"既包含 Stieger 占比变化，也包含 "重启优化器 + 多训 12 ep" 的优化随机性。要 claim "Stieger 占比是负迁移加剧主因"（§3.6 + §4.5 当前的写法），需要 V4 = "保持 V2 数据组成，warm-restart 续训 12 ep" 作为对照——这是当前证据缺口。

**作者必须做的事（MAJOR）**：

a. **Per-subject paired-t**：把 §3.6 表 16 升级为报告 paired-t p value、Cohen's d、95% CI（21 名被试 V3 vs Baseline 的逐被试 delta）。如果 binary cross paired-t p > 0.10（很可能），明确标注"DAPT 三种配置下 group-mean delta 均为负，但**没有一个达到 paired-t 显著**"——这才是诚实表述。
b. **V4 控制实验（强烈建议但可协商）**：跑一次 "保持 V2 数据组成（含 79% Stieger） + 优化器单阶段连续训练 30 epoch"，专门隔离 "V3 改进的多少来自 Stieger 比例下降 vs 多少来自 warm-restart 续训"。预算：~6h GPU（V2 一次完整 = 4.5h，多训 12 ep ≈ 1.2h）。
c. **改写 §3.6 + §4.5 + §7 Finding 4**：从"DAPT 三配置一致负迁移 → 域不匹配 + 灾难性遗忘"的强归因，弱化为"在三种探索性配置下均观察到方向性负迁移，但 V1/V2/V3 同时改变了数据量、LR、训练步数、Stieger 比例、优化器状态连续性 5 个变量；因此本研究**只能 claim** 方向性观察，**不能 claim** DAPT 在原则上对 finger MI 无效——单数据集 leave-one-out 与控制变量 ablation 留待未来工作"。
d. **V2 LMDB 崩溃影响**：Limitation #8 已经承认。但 §3.6 文字依然多处使用"V2 全量训练后……"的说法（如 L729），这与 "Epoch 13 崩溃没让 V2 自然 early-stop" 自相矛盾。把所有"V2 训练充分"的暗示替换为"V2 在 Epoch 12 处被强制截断"。

**这是 MAJOR 而非 CRITICAL 的原因**：DAPT 负迁移的方向性观察本身有价值——它至少提示"naive DAPT 不会自动改进 finger MI"——这对 community 是有用的负面结果。但当前的 strong-claim framing（"一致负迁移"作为定量基础）超过了证据所能承担的份量。MAJOR 修订要求降语气、补 paired-t、对 V4 控制实验做出处理。

---

## 2. Strongest Counter-Argument to the Paper's Headline Claim

**论文核心主张**：CBraMod + FDR 32ch + extra sessions 是 finger-level MI-BCI 的最优部署路径，CBraMod 相对 EEGNet 的优势可以分解为架构 (~+35 pp) + 预训练 (~+4 pp) + 容量 (−25 pp) 三向贡献。

**最强单一反方论证（One-Sentence Form）**：

> 整个 §3.7 三向分解的定量稳健性建立在"扩参 EEGNet 与 random-init CBraMod 都被合理 HPO"的隐式假设之上——但前者只跑了手调 ≤ 2 trial、后者完全没跑 random-init 专属 HPO；在算力预算可能差 25–40× 的对照下宣称"+34.97 pp 来自架构"是**HPO budget asymmetry confounding**的教科书案例，paper 的归因主张必须降级为"在共享默认 HP 与受限 HPO 预算下观察到的差距"。

**为什么这是最致命的**：

- 摘要、§1.4 Finding 1、§4.1、§7 Finding 1 在四处独立位置使用 "+34.97 pp / +27 pp / −25 pp" 这一三元组作为论文的核心 quantitative claim。
- 一旦读者意识到这三个数字依赖于"扩参 EEGNet 没有 HPO + random-init CBraMod 没有 HPO + EEGNet baseline 只有 32 trial / CBraMod baseline 有 51–77 trial"的非对称预算，整个 attribution narrative 立刻塌缩为"在共享 HP 下的观察"——失去了"分解架构 vs 预训练 vs 容量"的力度。
- 作者已经在 `2026-05-09_eegnet_huge.md` 内部 handoff 中**清楚地承认** v1/v2 不收敛是 "BF16 + 深 MLP 没有 LayerNorm" 的优化栈问题（v3 加 LN 后立刻 trainable）——但论文正文 §3.7.1 把这点重写为"提示这并非 HP 调优问题而是容量饱和"，这是 evidence 与 narrative 之间的 misalignment。

如果 R1 评审员看到 §3.7.1 + handoff 的对比，会立刻在审稿意见里写："The paper's headline three-way decomposition is methodologically unsound; the HPO budget for the EEGNet capacity ladder and random-init CBraMod is grossly inferior to the CBraMod baseline, and the failure of v1/v2 is by the author's own diagnostic notes a LayerNorm/optimization issue rather than a capacity ceiling. Major revision required: independent HPO sweeps for v1/v2 and random-init CBraMod."

---

## 3. Cherry-Picking Audit

最被宣传的数字是否是从更大集合中精选出的最 favorable 配置？

| 宣传数字 | 最 favorable 选择？ | 证据 |
|---------|---------------------|------|
| **90.68%** (cross-subject CBraMod binary 128ch) | **NO** — 这是 CBraMod 在 21 人 cohort 上的标准结果，run_tag `20260324_0023` 在 ExperimentDB 中标 `is_baseline=1`，HPO trial #4 / 77 best=90.68%（即 cross-subject HPO 的全局最优）。但 Cohort responder filter（49 → 21）独立"挑了一组容易的被试"——这是 cohort 选择层面的 favorable selection，不是模型选择层面。 |
| **96.7%** retention (FDR 32ch) | **PARTIALLY YES** — 32ch FDR vs 32ch CSP 差 2.77 pp（87.71% vs 84.94%）。论文报告了所有 5 种方法，但表 8 排序后 FDR 居首。**Cherry-picking 风险来自 Limitation #1 的"通道选择 mild leakage"**——FDR 排序使用了 test session 信息，所以 87.71% 可能略有夸大。clean recompute 后 retention 可能落到 92–95% 区间。 |
| **+27 pp 预训练贡献** (within) | **MIXED** — 这是 "binary +23.10 pp / ternary +30.79 pp 的平均"。"~+27 pp"的具体数字在摘要、§1.4、§4.1、§7 Finding 1 出现，但**不同段落对应不同 task 边界**：摘要 L20 把 +27 pp 标为"被试内"，§3.7.3 把它定义为"binary +23.10 / ternary +30.79 平均 ~+27"，§7 Finding 1 重新模糊为"~+27 pp"。这是 commit f309048 修复过的 ambiguity，但残留：摘要 / §1.4 / §7 用 "~+27 pp" 这一**单一数字**的同时，其内涵在 within-binary 与 within-ternary 之间漂移。请问："+27 pp" 是平均还是上限？读者无法 disambiguate。**MINOR overgeneralization**，不至于 cherry-picking，但接近边缘。 |
| **78.75%** (4ch Band Power, ~86.8% retention) | **NO** — 这是 4 种 4ch 方法（FDR / CSP / Attention / BP）中**唯一**超过负控制的，论文已经把所有 4 种 + 负控制都报告，在 §3.5.3 表 10 中。但 §1.4 Finding 5 + §7 Finding 5 把 78.75% 单独 elevated 为 "可部署谱系扩展到 4ch" 的代表数字，淡化了"4ch FDR 62.08% / Attention 54.70% / CSP 66.99% 全部失败"——这是 **reporting selection** 而非 data selection。论文叙事偏向 "4ch BP 仍可部署"。 |

**结论**：Cherry-picking 风险 **MODERATE**——核心数字 90.68% 是合规的（HPO 全局最优 + 完整 cohort），但 96.7% 受 Limitation #1 影响、78.75% 受 reporting framing 影响、+27 pp 受跨段落语义漂移影响。**MODERATE，不构成 reject 理由**，但需要在修订版中 (a) 跑 channel selection clean recompute (§1.2 Action b)、(b) 在摘要 / §1.4 / §7 把 "+27 pp" 显式标注为"binary +23 / ternary +31 的平均"。

---

## 4. Confirmation Bias Audit

论文在哪些地方对证据做了单方向解读，而存在合理的 alternative interpretation？

**Bias #1（最严重）：DAPT V1→V2 cross-subject 方向反转被淡化**

`further_pretraining_analysis.md` §6.3 表清晰报告：V1 cross-subject binary delta = −1.70 pp，V2 = −1.11 pp。**V2 在 cross-subject binary 上比 V1 更好 +0.59 pp**。论文 §3.6 + §4.5 + §7 Finding 4 全部以 "V1: −0.75 / V2: −1.38 / V3: −0.70 平均" 总览呈现，称三配置"一致负迁移"——但 V1→V2 在 cross-subject binary 这一具体 cell 上是**正向移动**。这一证据在 §6.3 caveat 段仅一笔带过（"Binary cross-subject 是唯一 V2 优于 V1 的组合"），但被 §3.6 + §4.5 + §7 Finding 4 的"一致负迁移"叙事完全 overwhelm。**反方解释**：V1 → V2 cross-subject 的 +0.59 pp 暗示"足够数据 + 恒定 LR → DAPT 在数据规模上有 scaling 行为"，而 V1 → V2 within-subject 的 −1.61 pp 是"小样本下 DAPT 的灾难性遗忘"——这是一个**与"一致负迁移"完全不同的故事**：DAPT 在 cross 范式可能 net positive，在 within 范式 net negative，而非"统一负迁移"。论文未探索这个 alternative。

**Bias #2：Random-init CBraMod within-ternary 18/21 chance collapse 的解读单方向**

`2026-05-09_random_init_ablation.md` L186-210 的 LR-deficiency 假设诊断给了非常详细的分析，**作者本人**在 handoff 中估算："数据量/过参数化导致 saddle-lock（结构性，与 LR 量级关系弱）" 70–80%，"LR + patience + warmup 调优可救回 ≥ 5 个塌陷被试" 15–25%。但论文 §3.7.2 + §4.1 把这一塌陷解读为"~4M 参数 transformer 在没有预训练先验时变成'负容量'"——把 70–80% 的概率主张写成 100% 的因果归因。**反方解释**："Random-init within-ternary 失败更可能是 HP 不匹配"（用了 cross-subject HPO 出来的 backbone_lr=1.3e-4 跑 within，但 from-scratch transformer 经验值在 1e-3 ~ 3e-3）——handoff 列出的 high-LR retry 实验 (~25 min GPU) 可证伪 / 证实，但论文未引用、未跑。论文 §3.7.2 在 L799 的 framing "transformer 在 ~70 trial 单被试样本下没有预训练先验时变成'负容量'"——是把 alternative hypothesis 提前定性为"已验证"。

**Bias #3：EEGNet-Huge v1/v2 失败 = "capacity 饱和" 而非"优化栈不友好"**

见 §1.1 论证。这是 confirmation bias 最显眼的实例——handoff 内部诊断与论文外部叙事直接相反。

**Bias #4：Cross-subject 90.68% 解读为"CBraMod 优势"，没有引用 cohort filter**

§4.1 L867-877 解释 CBraMod 优势"~+35 pp 来自架构 + ~+4 pp 来自预训练"，但没有同时讨论"21 名 responder cohort 的天花板比无筛选 cohort 高"这一独立的 explanatory factor。Limitation #2 提到 cohort，但 §4.1 没有把它纳入"+35 pp / +4 pp"分解的 confounder 列表。

**汇总**：4 项 confirmation bias，以 Bias #1 (DAPT V1→V2 cross 方向反转被淡化) 与 Bias #3 (EEGNet-Huge 失败归因) 最严重。建议作者在修订版中加一个 §4.X "Alternative Interpretations We Considered and Why We Discarded Them" 段落，显式列出这些 alternative，并 articulate 为何选择当前 interpretation——这是 negative-result paper 的标准做法。

---

## 5. Logical Chain Validation

对每个主张找出推理链中最弱的一环。

**Claim 1**: "CBraMod 在 cross-subject 范式下相对 EEGNet baseline 的 +14.01 pp 优势可分解为架构 +35 pp + 预训练 +4 pp + 容量 −25 pp 三向贡献"
- **最弱一环**: "+34.97 pp 来自架构" 依赖于 "EEGNet-Huge v3 51.37% vs CBraMod random-init 86.34%" 的对照。但 EEGNet-Huge v3 没有独立 HPO（手调 ≤ 2 trial）、CBraMod random-init 复用 baseline HP（0 trial 专属 HPO）。在 HP 可能错配的双侧，差距 +34.97 pp 是 "架构差异 + HP 错配 + 容量差异" 的混合体。 **推理强度: 弱 — 需要 §1.1 Action 修复。**

**Claim 2**: "Cross-subject CBraMod 90.68% 反映了 TUEG 预训练 + transformer 架构在 21 名被试 pooled 数据上的优势"
- **最弱一环**: 21 名被试是 [3] 在 49 人中筛选的 responder cohort——意味着至少 0–10 pp 的 cohort-conditional inflation 没有量化。**推理强度: 中 — Limitation #2 已经承认，但没有量化让读者评估。**

**Claim 3**: "DAPT 在 finger MI 任务上一致负迁移；外部粗运动 MI 数据池不能改善表征"
- **最弱一环**: V1/V2/V3 三个 negative 数据点的方向一致性，是把"5 个不可比变量"的 effects 强行归一化为"DAPT 性质"。Limitation #12 自承"V1/V2/V3 均为单次 pre-training 尝试"——单次实验 × 多变量 ≠ rigorous evidence。**推理强度: 弱 — 需要 §1.3 Action 修复。**

**Claim 4**: "32ch FDR 保留 96.7% 的性能"
- **最弱一环**: FDR 排序在所有 session（含测试 session 上下文）上计算（Limitation #1）。clean recompute 后 retention 可能下降 1–3 pp。**推理强度: 中 — 自承存在，需要 clean recompute 验证。**

**Claim 5**: "推理延迟 <13 ms 满足实时要求 + 单卡 RTX 5070 可服务 64 用户"
- **最弱一环**: §3.8 测试在桌面级 RTX 5070；临床部署可能在 edge device（Jetson Orin Nano、移动 CPU），延迟可能差 5–10×。论文未讨论 edge deployment scaling。**推理强度: 中 — 不影响主结论但部署主张范围模糊。**

**Claim 6**: "Extra sessions 推动 CBraMod within-subject 到 93.36%；标准差从 10.81% 压缩至 5.98%（−45%）"
- **最弱一环**: 16 名被试是从 21 中**有 extra sessions**的子集。Limitation #6 承认 selection bias——拥有 extra sessions 的被试可能是更愿意参加多 session 的"好 BCI 用户"，baseline 系统性偏高。这一 bias 同时**夸大 baseline 也夸大终点 +Sess05** —— net effect 不确定。**推理强度: 中 — 自承存在，但 §4.4 没有显式处理 selection-on-baseline confounding。**

---

## 6. Overgeneralization Audit

**OG #1（最严重）**: "EEG 基座模型的 'domain' 边界由信号级特征（采样率、频段、电极配置）而非任务级语义定义"（§4.8 末段 + §7 Finding 4 末段）

这是一个从 "DAPT 在 finger MI + 外部粗运动 MI 数据池上失败" 单一观测推广出的**普世性方法论命题**。它声称 EEG foundation model 的 transfer 行为与 NLP/CV 的 domain-adaptive pre-training 不同。**这一推广的证据基础**：单一 cohort × 单一 backbone × 单一 pretraining objective × 一组外部数据集。要 claim "EEG foundation model 通用属性"，需要至少 (a) 多 backbone（LaBraM、BENDR）（b) 多 pretraining objective（contrastive、predictive）（c) 多下游任务（不只 finger MI）。论文 Limitation #7 + 未来工作 #7 已 acknowledge 此缺口，但 §4.8 + §7 Finding 4 末段没有相应降语气。**建议把"EEG foundation model 的 domain 边界由信号级特征定义"降语气为"在本研究的 CBraMod + masked autoencoding + finger MI 场景下观察到的现象，提示后续 EEG foundation model 设计可能需要考虑信号级 domain 对齐"。**

**OG #2**: "盲目扩参不是改进路径"（§4.1 + §7 Finding 1）

证据基础是 "EEGNet 内 16K → 30M 扩参，cross-subject 单调下降"。但 (a) 只有 EEGNet 一个架构 (b) 没有独立 HPO (c) 没有 conv stem 单轴 vs MLP 头单轴隔离（作者自承，Limitation 之前 §3.7.1 末段也坦白）。"盲目扩参不是改进路径" 在 EEGNet 单一架构 + 单一扩展轴下成立——但被论文升级为 "EEG decoding 的瓶颈不在容量"（§7 Finding 1）。**这是单架构观察推广为跨架构方法论的 overgeneralization。**

**OG #3**: "32ch FDR 是稳健的精度-硬件权衡点"（§4.2）

证据基础是单一 cohort × 单一任务（binary cross-subject）。论文确实加了 "本研究评估范围内"、"在本数据集上"、"以下任意条件改变都可能让该排序翻转" 等 hedge 语言（§3.5.3 末段写得非常明确）——这是好的实践。但 §4.2 + §4.6 部署路线图 + §7 Finding 2 用 "32 通道是最优部署目标" 的强表述，与 §3.5.3 末段的 hedge 语言不一致。建议统一调整。

---

## 7. Alternative Explanations（针对 3 大 headline finding 的 NULL 假设）

### Finding A: CBraMod > EEGNet（cross-subject binary +14.01 pp、within-subject binary +7.05 pp）

**最 plausible NULL 假设**: "差距完全来自 EEGNet HPO 受限 + 21 名 responder cohort 选择偏倚 + cross-subject pooling 对小模型本身有害"。

具体分解：
- EEGNet baseline HPO: 32 trial / 10 complete vs CBraMod cross-subject HPO: 77 trial / 43 complete。如果 EEGNet 有同等 HPO 预算，可能恢复 1–3 pp。
- 21 名 responder cohort 把"难学的被试"提前过滤掉——CBraMod 因预训练 backbone 更宽容，在 marginal responder 上更鲁棒；EEGNet 在 marginal responder 上更脆弱。把这一 explanatory variance 算入，CBraMod-EEGNet 差距可能在无筛选 cohort 上落到 +5–8 pp 而非 +14 pp。
- §3.2 已显示 EEGNet 在 cross-subject 下方向性受损（−1.43 pp vs within-subject 78.10%）——这是 EEGNet 特有的"小模型 + 异质 21 名被试 pool 不能受益"现象，而非 CBraMod 优势的对称证据。

**作者已经做的 robustness checks**: §3.7.1 EEGNet 容量阶梯、§3.7.2 random-init CBraMod、§3.9 leave-3-out。**仍需做**: label-shuffle control + 共享同等 HPO 预算的 EEGNet 重训。

### Finding B: FDR 32ch ~96.7% retention

**最 plausible NULL 假设**: "FDR ranking 用了 test-session 数据 → 1–3 pp leakage 夸大；剩余 retention 真实数字在 92–95% 区间，与 CSP / Attention / Band Power 4 种方法在 32ch 上的真实差异 ~1 pp 而非 2.77 pp"。

要排除：clean recompute（仅用 train + val session 重算 FDR ranking 后跑 cross-subject CBraMod）。预算 < 6h GPU。**作者绝对应该做这个 clean recompute**——它直接关系到 §1.4 Finding 2、§7 Finding 2、§4.2 部署主张。

### Finding C: DAPT V1/V2/V3 一致负迁移

**最 plausible NULL 假设**: "DAPT 三个数据点没有一个达到 paired-t p < 0.05；group-mean delta < 1 SD/4；'一致性'是 3 个统计噪声落到同一象限的偶然 + 5 个未控制变量的混合（数据量、LR、epoch、Stieger 比例、optimizer 状态）。可能在控制良好的 V4 实验下方向反转。"

具体：
- §3.6 表 16 没有 paired-t p value——加上之后 V3 vs Baseline cross binary delta = −1.31 pp 大概率不显著（被试间 SD ~9.25 pp，N=21，Cohen's d ≈ 0.14 → effect 极小）。
- 单数据集 leave-one-out（移除 Stieger）尚未做——作者列在 Limitation #9。

要排除：补 V4 实验（保持 V2 数据组成 + 单阶段 30 epoch）+ 报告 paired-t per cell + leave-one-out Stieger 实验。**这其中 paired-t 是 0 cost，必须立即做。**

---

## 8. Stakeholder Blind Spots & "So What?" Test

### 8.1 谁会被 "CBraMod + FDR 32ch 部署路径" 主张所伤？

- **Edge / 移动 BCI 部署者**：论文 §4.6 部署路线图主张"单卡 RTX 5070 可服务 64 用户"——但临床 BCI 大量场景是**离线设备 / wearable**，需要 ARM Cortex-M7 / Jetson Orin Nano / 树莓派级别。RTX 5070 上 12.9 ms 延迟在 Jetson Orin Nano 上可能 80–150 ms（10× 慢估计），逼近实时阈值。**论文应该至少加一句**"本论文延迟测试在桌面级 GPU；wearable 部署需要独立 latency benchmark"。
- **隐私敏感场景**：foundation model 推理需要把 EEG 上传到服务器（多用户共享场景），EEG 信号包含被试身份信息（已被多篇 paper 证实可识别个体）+ 神经状态（疲劳、压力、认知负担）。论文完全没有触及隐私问题——但既然论文 §4.6 推荐云服务部署模式，至少应该 acknowledge"BCI 推理服务的隐私 / GDPR 合规问题超出本研究范围，但是部署的关键约束"。
- **Latency-critical applications**：论文说 13 ms < 100 ms 实时阈值——但 13 ms 是**模型推理延迟**，不是**端到端 BCI 闭环延迟**（含 EEG 采集 + filter + classification + 控制信号 → 设备执行）。完整闭环在 50–200 ms 量级常见。论文没有说明"13 ms 推理延迟"在闭环延迟预算中的占比——这是**部分误导**。
- **Non-responder 用户**：21 名被试是 [3] 筛选后的 responder cohort。论文 §4.6 推荐"采用 CBraMod + FDR 32ch"为部署起点——但 49 → 21 的 ~57% 不能成为 BCI responder 的用户**完全不在 paper 数据范围内**。Limitation #2 承认此点，但 §4.6 部署主张没有添加"该路径主要适用于已通过基础 BCI 校准的 responder 用户"hedge。

### 8.2 "So What?" Test for §1.4 6 项贡献

| Contribution | 实际推动 BCI 领域？ | 评判 |
|--------------|--------------------|------|
| 1. 三向分解（架构/预训练/容量）| **方法论原创** — 但因 §1.1 关注的 HPO 预算非对称未控制，定量分解的可信度受限 | 高潜力 / 当前证据弱 |
| 2. 32ch FDR 96.7% retention | **实用** — 直接对临床 BCI 设计有 actionable 影响 | 高价值（待 clean recompute 验证）|
| 3. 通道选择方法在低密度档位排序翻转 | **新颖** — 4ch BP 78.75% 是 nontrivial 发现 | 中价值（单 cohort 限制范围）|
| 4. 多 session 纵向数据扩展三范式对比 | **复刻** — 与 [3] 在线版本结论一致，离线版本是 incremental | 低-中价值（incremental）|
| 5. DAPT 负迁移 + V3 拆分 | **方向性 negative result** — 对 EEG foundation model 设计有警示价值 | 中价值（证据稳健性需补强）|
| 6. 推理延迟 < 13 ms | **工程基线** — 标准 inference benchmark | 低价值（routine engineering）|

**评判**: 真正推动领域的是 (1) + (2) + (3) + (5)；(4) 是对 [3] 的离线复刻，价值 incremental；(6) 是 routine engineering benchmark。**(1) 是论文最原创的贡献**——这正是 §1.1 关注的 HPO 预算非对称问题如此关键的原因：如果 (1) 不稳健，论文最大的原创性就没了。

---

## 9. Summary of Required Author Actions

| Priority | Action | 估计预算 | 关联章节 |
|----------|--------|----------|----------|
| **CRITICAL** | A1. 跑 EEGNet-Huge v1 / v2 各 ≥25 trial 独立 Optuna HPO（覆盖 LR / warmup / LayerNorm on/off / dropout）；如果 100% trial 仍死锁，**那时**才能 claim "capacity 饱和"。同时跑 random-init CBraMod 至少 25 trial 独立 HPO。 | ~80–120h GPU | §3.7.1 / §3.7.2 / §4.1 / §7 Finding 1 |
| **CRITICAL** | A2. 修订 §1.4 / §3.7 / §4.1 / §7 Finding 1 中的 "+34.97 pp / +27 pp / −25 pp" 为更弱 framing（如"在 default HP 共享下观察到的 Δ"），直至 A1 完成；同步把 §3.7.1 关于 v1/v2 失败的归因从"capacity 饱和"修正为"在两套手调 HP 下不收敛，与 v3 加 LayerNorm 后立即 trainable 一致——优化栈兼容性问题"。 | 1 day 写作 | 摘要 / §1.4 / §3.7 / §4.1 / §7 |
| **MAJOR** | A3. **Label-shuffle control on cross-subject CBraMod**（必做）：trial-level 随机重排标签 → 跑同一 cross-subject pipeline，验证 test acc 落到 50% ± 2pp 区间。 | < 6h GPU | §3.5.3 / §3.9 / §4.1 |
| **MAJOR** | A4. Channel selection clean recompute：仅用 train + val session 重算 FDR/CSP/Attention/BP，跑一次 cross-subject 32ch FDR；如 retention 下降 ≥2 pp，修订 96.7% 数字。 | ~6h GPU | §3.5 / §1.4 Finding 2 / §7 Finding 2 |
| **MAJOR** | A5. §3.6 表 16 加 paired-t per-subject p value、Cohen's d、95% CI；如 binary cross V3 vs Baseline 不显著，明确标注；改写 §3.6 + §4.5 + §7 Finding 4 把 V1/V2/V3 "一致负迁移"降语气为"三种探索性配置下方向性观察，5 个未控制变量"。 | 1 day 分析 + 写作 | §3.6 / §4.5 / §7 Finding 4 |
| **MAJOR** | A6. V4 控制实验（强烈建议）：保持 V2 数据组成 + 单阶段 30 epoch，隔离"Stieger 比例下降"vs"warm-restart 续训"的贡献。 | ~6h GPU | §3.6 / Limitation #12 |
| **MODERATE** | A7. 加 §4.X "Alternative Interpretations" 段：显式列出每个 finding 的 NULL 假设并说明为何当前数据不能完全排除。 | 0.5 day 写作 | §4 |
| **MODERATE** | A8. Cohort-conditional inflation 量化：在 Limitation #2 中估算"无筛选 49 人 cohort 上的 generalization 大约在哪个区间"。 | 0.5 day 分析 | §5 Limitation #2 |
| **MODERATE** | A9. 部署主张降语气：把 §4.6 部署路线图的"单卡 RTX 5070 可服务 64 用户"加上"桌面级 GPU 测试，wearable / edge 部署需独立 benchmark"hedge；把"32ch FDR 是最优部署目标"统一调成 §3.5.3 末段那样的 cohort-conditioned 表述。 | 0.5 day 写作 | §4.2 / §4.6 / §7 Finding 2 |
| **MINOR** | A10. 摘要 / §1.4 / §7 中的 "~+27 pp" 显式标注为 "binary +23.10 / ternary +30.79 平均"。 | 5 min 编辑 | 摘要 / §1.4 / §7 |
| **MINOR** | A11. 添加 EEGNet baseline cross-subject 独立 HPO 表（如果存在的话——目前 Table S5b 只列 within-subject）；如果没做过，至少在 Limitation 中承认 EEGNet cross-subject 没有独立 HPO。 | 0.5 day | Table S5b / Limitation |

**A1–A2 是 CRITICAL**——论文核心 quantitative claim 的稳健性依赖之。
**A3–A6 是 MAJOR**——影响 §3.5 + §3.6 + §4.5 主张的可信度。
**A7–A11 是 MODERATE / MINOR**——提升 paper 完整性但不阻碍接收。

如果 CRITICAL 全部完成：可考虑 Minor Revision → Accept。
如果只完成 A2（叙事降语气，不补 HPO 实验）：仍是 Major Revision。
如果连 A2 / A3 都不愿做：**Reject**——论文核心定量主张不可靠且作者未承认问题。

---

## 10. Confidence in Review (1-5)

**4.5 / 5**

依据：

- **+1.5 来自交叉核验**: ExperimentDB 直接验证了 9 个核心 run_tag 的 mean_acc / std_acc，所有引用数字与 paper 一致。
- **+1 来自 handoff 读到位**: `2026-05-09_random_init_ablation.md`（核心证据）+ `2026-05-09_eegnet_huge.md`（揭示 v1/v2 失败诊断与论文叙事不一致）+ `further_pretraining_analysis.md`（V1→V2 cross +0.59 pp 反向证据）三份 handoff 全文阅读。
- **+1 来自 caveat 自洽性**: 作者在三处独立位置（§2.7.2 V3 caveat、§4.5 V1/V2 不可严格归因、Limitation #12 单次性）已 disclose V1/V2/V3 不可比性——这让 §1.3 challenge 不是"我新发现的问题"，而是"作者承认的问题被叙事 framing 掩盖"。同样地，作者在 `2026-05-09_eegnet_huge.md` handoff 中清楚承认 v1/v2 失败是 LayerNorm 优化问题——让 §1.1 challenge 直接 cite 作者自己的诊断。
- **+1 来自工具/方法学共识**: HPO budget asymmetry、label-shuffle control、paired-t 报告这三项是 ML 方法论 community 多年的标准——任何 R1-tier reviewer 都会问到。
- **−1 来自单作者评审局限**: 我 (Devil's Advocate) 不能完全替代领域专家——例如 cohort-conditional inflation 的具体数字（"49 → 21 的影响是 ~10 pp 还是 ~3 pp"）需要更深的 BCI cohort effects 研究文献支持，我只能提出方向。

**总体**: 我对自己的 challenges 的方向性 99% 有信心；对 specific 数字（如 "+34.97 pp 中有多少来自 HPO 错配 vs 多少来自架构"）的精确量化 < 70% 有信心。这正是为何要求作者跑 A1 / A3 / A4 实验——不是因为我已经证明 paper 错了，而是因为现有证据**不足以排除我所列的 alternative explanations**，而 burden of proof 在作者身上。

---

**评审签名**: Devil's Advocate Reviewer (v1.1)
**评审时间**: 2026-05-10
**输出位置**: `paper/reviews/stage3_devils_advocate_review.md`
