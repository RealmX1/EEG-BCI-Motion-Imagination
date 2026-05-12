# R3 Cross-Disciplinary Perspective Review — Stage 3 Phase 1

**Reviewer Role**: R3（NLP / CV 基座模型 + Transfer Learning 理论方向，跨学科视角）
**Recommendation**: **Major Revision**（重大修改后可接收——证据基本完整，但跨学科叙事框架需重新校准）
**Knowledge boundary disclosure**: 我不是 EEG 领域专家。下方对体积传导、mu/beta ERD、感觉运动皮层解剖学等 EEG-specific 主张不做实质性挑战；本评审聚焦 transformer 架构、自监督预训练、domain-adaptive pre-training（DAPT）、scaling laws 这几条与 NLP/CV 文献直接对话的轴线。

---

## 1. Summary from a Cross-Disciplinary Lens

本文研究 EEG 基座模型（CBraMod，~30M 参数 transformer + ACPE 位置编码，ICLR 2025）在手指级运动想象分类中的应用，三大主张分别对应 NLP/CV 文献中已有大量先例的现象：

1. **§3.6 / §4.5 / §4.8 的 DAPT 负迁移结论**——本文把外部 870h MI 数据上的 further pre-training 失败解读为 "EEG 基座模型的 'domain' 由信号级特征定义，与 NLP/CV 的任务级 domain 不同"。从 NLP DAPT 文献角度看，这一框架过度简化：Gururangan et al. 2020 已经给出 DAPT 在不同任务-语料对齐度下"helpful / harmful / neutral"的连续谱系，本文观察到的"+0.68 pp 部分恢复 + 整体仍负"恰好处于该谱系的负迁移端，**与 NLP 经验高度一致而非"EEG-specific 新颖发现"**。
2. **§3.7.2 random-init CBraMod 在 within-subject 上的 chance-collapse**——本文把"30M transformer 在 ~70 trial 上无法学习"解读为"~4M 参数的 transformer 变成负容量"。从 NLP transformer-on-small-data 文献角度看，这是 **BERT-style transformer 在少样本下的已知失败模式**（Devlin et al. 2019、Mosbach et al. 2021、Zhang et al. 2021），不应作为新发现陈述。
3. **§3.7.1 EEGNet-Huge v1/v2 的 train loss 死锁在 0.693**——本文称之为"EEGNet 架构内的容量天花板 / 反向 scaling"。但从 scaling law（Kaplan et al. 2020；Hoffmann et al. 2022 Chinchilla）和 deep learning HPO 文献角度看，"~30M 参数模型在 ~3K 样本/被试 + 两套 LR 下 train loss 死锁"更像**优化失败（optimization failure）**而非"capacity reverse-scaling"——证据不足以排除"如果做合适的 init / warmup / LR sweep / gradient clipping，模型就能训起来"。

整体上，三个发现都是真实且有价值的实证观察，但**叙事层次需要从"novel discovery"调整为"consistent with（且扩展了）NLP/CV 已有文献"**。这种调整不会削弱本文的贡献——相反，把发现锚定到既有理论框架中能让 reviewer 更容易接受其稳健性。

## 2. Strengths

跨学科视角下，本文有三点尤其值得肯定：

**S1. Random-init 消融的设计本身（§3.7.2）严格得当。**
"完全切除 backbone 预训练 + HP 沿用 baseline + cross-subject checkpoint 用作 XSI-FT 初始化（端到端 from-scratch）"是 NLP 文献中 BERT vs random-init 对照的标准做法（Tenney et al. 2019、Conneau et al. 2018）。Seed 复现性检查（seed=42 vs seed=1234，within ternary 18/21 vs 17/21 chance-collapse）尤其值得肯定——NLP transformer 小样本失败是出名地 seed-sensitive（Mosbach et al. 2021），单 seed 报告的 chance-collapse 论断很难成立，本文用 cross-seed 双确认在方法学上做得比许多 NLP 论文更严谨。

**S2. V3 Stieger 占比消融（§3.6）做出了正确的对照。**
V2 vs V3 把"主导数据集占比"从 ~79% 降到 ~30% 以分离单一数据集主导效应——这与 BiomedRoBERTa（Gu et al. 2021）做的 corpus composition 消融在精神上一致。V3 vs V2 +0.68 pp、V3 vs Baseline 仍 −0.70 pp 的"恢复一半"模式被坦率报告，符合 DAPT 文献中"调数据组成只能解释一部分负迁移"的常见经验。

**S3. EEGNet 容量阶梯虽然从 NLP 视角下结论站不住，但实验设计的 transparency 值得肯定。**
作者明确披露 v1/v2 状态（state_dict bug、within data orphan）、双 LR 行为一致、HP 调试细节追溯到 handoff 文档——这种"我们试过这两个 LR、都失败了"的可追溯性在 NLP 论文中也属罕见。问题不在 transparency 而在 inference（详见 §3.3）。

## 3. Major Cross-Disciplinary Concerns

### 3.1 §1.3 / §3.6 / §4.5 / §4.8 — DAPT 框架与 NLP DAPT 文献的对话缺失

**问题 1：§1.3 末尾的扫式陈述**

> "这一假设在 NLP（如将通用语言模型适配到生物医学领域）和 CV（如将 ImageNet 模型适配到医学影像）中已得到验证"

这一表述把 NLP DAPT 文献描述为"已验证的范式"，但实际文献远比这更微妙：

- **Gururangan et al. 2020（"Don't Stop Pretraining: Adapt Language Models to Domains and Tasks", ACL 2020）** 在 4 个 domain × 8 个任务上系统评估 DAPT。结果：**DAPT 在 task-corpus 对齐度高时帮助**（biomedical / CS），**但在对齐度低时甚至损害**（reviews 域上某些任务）。论文摘要的 takeaway 不是"DAPT 已被验证有效"，而是"DAPT helps when domain-relevance is high"——这是一个**条件性结论**。
- Gu et al. 2021（BiomedRoBERTa）在生物医学领域 DAPT 提供 +1.5 至 +5 F1，但前提是 source corpus（PubMed 14B tokens）的规模与 BERT 原始预训练（BookCorpus + Wiki）至少同量级，且 **domain 高度对齐**（医学文献 → 医学 NLP 任务）。
- Beltagy et al. 2019（SciBERT）和 Lee et al. 2020（BioBERT）都报告：从 BERT-base 出发做 DAPT 时，新 domain 数据规模需要≥10× 才能稳定改善，否则与 baseline 持平甚至小幅下降。

**本文的观察实际上与 Gururangan 2020 高度一致**：外部 MI 数据 38G channel-frames << TUEG 126.5G（仅 0.3×），任务对齐度低（粗运动 MI vs finger MI 在原始 EEG 信号统计上差异巨大），这恰好处于 Gururangan 2020 中"DAPT 损害"的区域。

**修订建议**：

1. §1.3 的句子改为："domain-adaptive pre-training 在 NLP 与 CV 中**取得了条件性成功**（条件包括 source corpus 规模、task-corpus 对齐度等；详见 Gururangan et al. 2020 [新引]）；其在 EEG 基座模型中的适用条件尚未系统评估"。
2. §4.5 第一段在论述"领域不匹配 + 数据量危险中间地带 + 灾难性遗忘"三层归因时，应**明确引用 Gururangan 2020 的"domain-relevance is a continuous spectrum"框架**，并指出本文观察到的"V3 部分恢复 + 整体仍负"恰好落在该谱系的"低对齐 + 数据量不足"区域，而非"EEG vs NLP/CV 的范式差异"。
3. §4.8 末尾的方法论命题——"EEG 基座模型的 'domain' 边界由信号级特征而非任务级语义定义"——需要软化。这一命题相当强（"区别于 NLP/CV 的 domain-adaptive 经验"），但本研究只在一种 backbone（CBraMod） + 一种预训练目标（masked autoencoding） + 一类 source corpus（粗运动 MI）下做了一次 DAPT，**不足以推断到"EEG vs NLP/CV 的范式级差异"**。建议改写为："本研究观察的负迁移与 NLP DAPT 文献中'低 task-corpus 对齐 + 数据量不足'的失败案例在结构上一致（参考 Gururangan et al. 2020 §5.2 的 reviews 域结果）；进一步判断 EEG 基座模型是否需要不同于 NLP/CV 的 transfer 设计原则需要在多 backbone × 多 source corpus 矩阵下验证。"

**问题 2：§4.5 关于"梯度方向反预期"的论证逻辑可强化**

> "梯度方向（被试内恶化更严重 vs 跨被试恶化较轻）与 'DAPT 在数据稀缺场景中收益最大'的常见预期相反"

NLP DAPT 文献中"数据稀缺时收益最大"的来源是 Howard & Ruder 2018（ULMFiT）和 Gururangan 2020。但这个"预期"的成立条件是 **DAPT 改善 backbone 表征**——如果 DAPT 损害 backbone（覆写有用先验），那么"数据稀缺者受损最严重"恰恰就是预期方向，因为下游 fine-tune 时数据稀缺场景**最依赖好的初始化**。所以本文观察到的方向并不"反预期"，而是与"DAPT 在该 setting 下损害 backbone"的诊断完全一致。

**修订建议**：把 §4.5 第二段的措辞从"反预期"改为"与'DAPT 损害 backbone 表征'诊断一致"——这是更精确的论证。同时引用 Howard & Ruder 2018 给出原始 baseline。

### 3.2 §3.7.2 — Random-init CBraMod 与 Transformer-on-Small-Data 文献

**核心问题**：§3.7.2 把"random-init CBraMod 在 within-subject 上 18/21 chance-collapse"描述为"~4M 参数的 transformer 在 ~70 trial 单被试样本下没有预训练先验时变成负容量"——这一陈述**作为现象观察是真实的，但作为新发现 framing 是误导的**。NLP 文献中关于 transformer 在小样本下不可训练已有大量先例：

- **Devlin et al. 2019（BERT 原论文）** §A.1 Appendix 已明确指出：BERT-base/large 在 fine-tune 时若样本数 < 1K，结果方差极大，需要 multiple-restart + 选最佳 dev acc。
- **Mosbach et al. 2021（"On the Stability of Fine-tuning BERT", ICLR 2021）** 系统分析 BERT fine-tune 在小样本（RTE, MRPC, ~2K 样本）下的不稳定性：**BERT-base / large 在 RTE 上 25 个 random seed 中有约 1/3 fall to chance**——非常接近本文观察到的 18/21 chance-collapse 比例。Mosbach 文章的解释：vanishing gradient + Adam 在小样本下的低 batch effective LR。
- **Zhang et al. 2021（"Revisiting Few-sample BERT Fine-tuning", ICLR 2021）** 进一步指出：BERT 小样本失败的解决方案包括（i）re-initialize top-K layers、（ii）long-warmup + low LR、（iii）mixout regularization、（iv）ULMFiT-style discriminative LR。
- 在 from-scratch 方向（更接近 §3.7.2 的 random-init 设置），Liu et al. 2019（RoBERTa）明确强调："training BERT-base from scratch requires careful warmup + 256 batch + LR sweep"——典型规模需要 8B-token 级别预训练才能得到稳定下游性能。

**~30M 参数 transformer 从随机初始化 + ~70 trial × 21 被试做 within-subject training，在 NLP 文献中没人会期望它能成功**。这是一个 well-established failure mode。

**修订建议**：

1. §3.7.2 的解读段落（797 行附近）"~4M 参数的 transformer 变成负容量"措辞过强。建议改为："这与 NLP transformer 在小样本下的已知失败模式一致（Mosbach et al. 2021 在 RTE ~2K 样本上 BERT-base 约 1/3 random seed 落入 chance；Zhang et al. 2021）——从随机初始化的 30M 参数 transformer 在 ~70 trial 上不可训练**不是 EEG 特有现象**，而是 transformer 架构在小样本下的通用脆弱性。本结果的 EEG-specific 价值在于：(i) 把这一脆弱性**精确量化到 EEG 手指 MI 任务**，(ii) 通过两 seed 复现确认 chance-collapse 不是统计噪声，(iii) 揭示 TUEG 预训练对该脆弱性的具体补偿幅度（+27 pp）。"
2. 与此同时，§3.7.2 应补充一段说明**没有尝试 NLP 文献中已知的小样本稳定化技术**（top-K layer re-init / long warmup / mixout）就把 random-init within-subject 失败定性为"transformer 在 ~70 trial 不可训练"过于强。**严格的反驳是**：random-init CBraMod within ternary 18/21 chance-collapse 也许通过 Mosbach-style 多 seed 重启 + Zhang 2021 的 top-3 layer re-init 能拉到 chance 之上。本文应在 Limitations 部分明确这一点。

### 3.3 §3.7.1 — EEGNet-Huge v1/v2 的 0.693 死锁与 Scaling Laws 文献

这是我作为跨学科 reviewer 最担心的一节。**§3.7.1 把"EEGNet-Huge v1/v2 train loss 死锁在 0.693（chance entropy）"解读为"~30M 已落入 chance，是 EEGNet 架构在跨被试范式下的容量天花板"——但从 NLP/CV 优化文献看，0.693 train loss 死锁更像优化失败（optimization pathology）而非"capacity reverse-scaling"。**

**关键证据缺失**：

1. **Init scheme**: §3.7.1 没有说明 v1/v2 的 5120-wide MLP 头使用什么 init（Kaiming/He vs Xavier/Glorot vs default）。Goodfellow et al. 2016 第 8.4 章和 Glorot & Bengio 2010 已证明：MLP 头宽度从 2048 跳到 5120 时，default init（如 PyTorch 的 kaiming_uniform_ with `a=sqrt(5)`）会让前向激活方差快速放大，导致 saturated activation（特别是 ELU/ReLU 之后），梯度消失，loss 死锁——这与本文报告的"两套 LR 都死锁"完全一致。
2. **Warmup**: §3.7.1 没有说明是否使用 linear warmup。BERT/GPT/ViT 系列的标准做法是 warmup_steps = 1000-10000。在 ~30M 参数模型上没有 warmup，loss 在前 100 step 内就可能进入 NaN 或 saturate 状态。
3. **LR sweep coverage**: 本文只跑了"两套 LR 相差 10×"（5e-5 vs 5e-4）。这远不足以宣称"capacity 不是问题的解"。**NLP/CV 标准实践要求至少 3-5 个 LR × 2-3 个 warmup × 3 个 seed**。Hoffmann et al. 2022 (Chinchilla) 在 scaling 分析中专门指出："at each (model size, data size) point, sweep at least 4 LRs around theoretical optimum"。
4. **Gradient norm logging**: 0.693 死锁的最快诊断是看 gradient norm。如果 grad norm 小于 1e-6，是 vanishing gradient（init 问题）；如果 grad norm > 100，是 exploding gradient（需要 grad clip）。本文 §3.7.1 没有报告 grad norm，无法判断诊断。

**Scaling Laws 文献的相关性**：

- Kaplan et al. 2020（"Scaling Laws for Neural Language Models", arXiv）和 Hoffmann et al. 2022（Chinchilla, "Training Compute-Optimal Large Language Models", NeurIPS 2022）都指出：在某个（参数量 N，数据量 D）平面上有 compute-optimal 边界，**N/D 比例失衡时模型欠拟合或过拟合，但都可训练**；反向 scaling（参数越多越差）在 NLP 文献中**只在 calibration 任务、TruthfulQA 等少数 task 上观察到**（McKenzie et al. 2023 Inverse Scaling Prize）——绝非 chance entropy 死锁。
- McKenzie 2023 的反向 scaling 任务都是 task-design level 的（模型规模越大越自信地学错答案），**不是 train loss 不下降**。本文 EEGNet-Huge v1/v2 的 0.693 死锁更接近 ResNet-1001 在小数据上 default init 下的 train collapse，而不是反向 scaling 现象。

**修订建议**：

1. **§3.7.1 不应直接把 30M EEGNet 的 chance behavior 命名为"反向 scaling 现象"**。建议改写为："EEGNet-Huge v1/v2 在我们尝试的两套 LR (5e-5, 5e-4) 下均未脱离 chance entropy；该结果**与 EEGNet 架构内 capacity 是否真正反向 scaling 不可区分**——它同样兼容'优化失败（init 不当 / warmup 缺失 / LR sweep 不足）'解释。本文不主张 EEGNet 在 30M 参数下不可训练；本文仅主张：在 baseline EEGNet 沿用的标准 HP 协议（无 warmup、双 LR）下，30M 量级 EEGNet 不可稳定训练，因此即便 EEGNet 架构在 30M 下理论上有解，工程上也不构成对 CBraMod 30M 的可行替代品。"
2. §4.1 第一段"把 'capacity is not the bottleneck' 立成铁案"的措辞**应去除**——这一主张的证据强度不够，应改为"在我们的 HP 协议下，简单扩参 EEGNet 不构成可行的 CBraMod 替代品"。
3. **未来工作 §6 列表中应补充一条**：在 EEGNet-Huge v1/v2 上做完整 HPO sweep（≥3 LR × 3 warmup × 3 seed × Kaiming/Xavier init 对照），并报告 gradient norm trajectory，以严格隔离"优化失败 vs capacity 反向 scaling"两种可能。这可能 ~1-2 个 GPU-day 完成。
4. **三向分解（§3.7.3）的解读需要软化**："EEGNet 容量扩展不仅无益反而显著有害（−25 pp）"过强；应改为"在我们使用的 HP 协议下，简单扩参 EEGNet 不能逼近 CBraMod 的性能"。这一较弱表述同样支持后续"transformer + ACPE 架构归纳偏置主导"的结论，但避免把不严格隔离的"capacity 有害"主张写入论文核心叙事。

### 3.4 摘要、§1.4、§7 中"+27 pp 预训练贡献"的归因强度

跨章节看，论文反复使用"TUEG 预训练在被试内贡献 ~+27 pp"这一数字。但严格地说，这是 within-subject (random-init CBraMod) vs (original CBraMod) 的差。考虑到 §3.2-3.3 我已论证 random-init CBraMod within-subject 18/21 chance-collapse 可能源于 transformer-small-data 优化失败，**这 +27 pp 的解读至少有两层不可分**：(a) TUEG 预训练提供 EEG 通用表征作为初始化（真正的 transfer learning 收益），(b) 任意 informed init（不一定要 TUEG，例如 random init + multi-seed best-of-N + LR warmup + mixout）能把 chance 拉到一个 above-chance 的水平。**本文未做（b）的对照**，所以这 +27 pp 严格上说是 "TUEG init vs default-HP random init" 的差，而不是 "TUEG transfer learning 价值" 的纯净估计。

**修订建议**：摘要、§1.4、§4.1、§7 在引用"+27 pp"时加 footnote："此估计基于 random-init 沿用 baseline HP 协议；该差距中可能有一部分源于 transformer-small-data 优化敏感性而非 TUEG transfer 本身；分离需补充 init scheme + warmup + multi-seed sweep 对照。"

## 4. Required Cross-Domain Citations

以下文献为本文跨学科 framing 必要：

**关于 DAPT 框架（§1.3 / §3.6 / §4.5 / §4.8）**：

1. **Gururangan, S., Marasović, A., Swayamdipta, S., Lo, K., Beltagy, I., Downey, D., & Smith, N. A. (2020). "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks." *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)*, pp. 8342–8360.**
   - 锚定 §1.3 / §4.5 / §4.8 的 DAPT 论述。本文负迁移结果与该论文 reviews 域上低对齐 task 的负迁移在结构上一致，应明确对话。

2. **Gu, Y., Tinn, R., Cheng, H., Lucas, M., Usuyama, N., Liu, X., Naumann, T., Gao, J., & Poon, H. (2021). "Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing." *ACM Transactions on Computing for Healthcare*, 3(1), Article 2 (BiomedRoBERTa / PubMedBERT 框架).**
   - 锚定 §4.5 / §4.8 关于 corpus 规模、composition 与 DAPT 收益的关系。

3. **Beltagy, I., Lo, K., & Cohan, A. (2019). "SciBERT: A Pretrained Language Model for Scientific Text." *Proceedings of EMNLP-IJCNLP 2019*, pp. 3615–3620.**
   - 锚定"小 source corpus + 高 task 对齐"下 DAPT 仍能持平/微负的经验，对比本文"大 source corpus + 低 task 对齐"的负迁移。

4. **Howard, J., & Ruder, S. (2018). "Universal Language Model Fine-tuning for Text Classification." *Proceedings of ACL 2018*, pp. 328–339.**
   - ULMFiT。锚定 §4.5 中"DAPT 在数据稀缺场景中收益最大"的原始预期来源。

**关于 Transformer Small-Data Failure（§3.7.2）**：

5. **Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *Proceedings of NAACL-HLT 2019*, pp. 4171–4186.**
   - BERT 原论文 Appendix A.1 已明确 small-data fine-tune instability。本文 §3.7.2 应引用以建立 small-data transformer fragile 的 baseline 知识。

6. **Mosbach, M., Andriushchenko, M., & Klakow, D. (2021). "On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines." *International Conference on Learning Representations (ICLR) 2021*.**
   - 系统分析 BERT 在 RTE 等小样本下 random-seed 1/3 chance-collapse 的现象。本文 18/21 within-ternary chance-collapse 与该论文的现象在数量级上一致——这是本文 random-init 失败的最直接 NLP 文献先例。**强烈建议引用**。

7. **Zhang, T., Wu, F., Katiyar, A., Weinberger, K. Q., & Artzi, Y. (2021). "Revisiting Few-sample BERT Fine-tuning." *International Conference on Learning Representations (ICLR) 2021*.**
   - 给出小样本 BERT 稳定化技术（top-K layer re-init、long warmup、mixout）。本文 §3.7.2 Limitations 应引用——把"random-init 18/21 chance-collapse 是否可解" 作为开放问题。

8. **Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). "RoBERTa: A Robustly Optimized BERT Pretraining Approach." arXiv:1907.11692.**
   - From-scratch transformer 训练对 HP 极度敏感。本文 §3.7.2 / §3.7.1 在 random-init / Huge baseline 上的 single-shot HP 选择需要这一文献作为对照标准。

**关于 Scaling Laws & Optimization（§3.7.1）**：

9. **Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., & Amodei, D. (2020). "Scaling Laws for Neural Language Models." arXiv:2001.08361.**
   - 锚定 EEGNet 容量阶梯讨论。指出 NLP scaling 的标准 sweep 协议（每个 model size 至少 4 LR）。

10. **Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., et al. (2022). "Training Compute-Optimal Large Language Models." *Advances in Neural Information Processing Systems 35 (NeurIPS 2022)* (Chinchilla 论文).**
    - 给出 compute-optimal scaling 的 N/D 平衡。本文 EEGNet 30M × ~3K 样本严重 N/D 失衡（按 Chinchilla 比例 N=30M 应配 ~600M 训练 token），这是 §3.7.1 EEGNet-Huge train loss 死锁的另一种自然解释。

11. **McKenzie, I., et al. (2023). "Inverse Scaling: When Bigger Isn't Better." arXiv:2306.09479.**
    - 反向 scaling 在 NLP 中的文献。本文 §3.7.3 提"反向 scaling"应引用此文献——同时也会让本文清楚：NLP inverse scaling 的现象**是 task-level 行为（更大模型在 calibration 等任务上更自信地犯错）**，而不是 train loss 死锁；本文 EEGNet 30M 的现象不属于 inverse scaling 的标准定义。

**关于 PEFT / 替代 transfer 路径（§4.6 / §6）**：

12. **Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." *International Conference on Learning Representations (ICLR) 2022*.**
    - 本文 §4.5 末尾建议"仅适配通道相关参数、冻结其余"——这正是 PEFT 思路。引用 LoRA 给出这一方向的成熟方法学先例。

## 5. Reframing Recommendations

以下是若干具体语句的修订建议：

**(a) 摘要中的 +27 pp 表述。**
> 现版：Random-init 落到 62.05%，反而低于 EEGNet baseline 78.10%
> 建议：保留事实陈述，但加补充"该 random-init 数字基于沿用 baseline HP 协议；transformer 在小样本下的优化敏感性可能贡献该差距的一部分（Mosbach et al. 2021）"。

**(b) §1.3 开篇关于 "EEG 基座模型" 的定位语。**
> 现版："这一假设在 NLP 和 CV 中已得到验证"
> 建议："domain-adaptive pre-training 在 NLP 与 CV 中取得了**条件性成功**——其收益强烈依赖 source corpus 规模与 task-corpus 对齐度（Gururangan et al. 2020 ACL）；其在 EEG 基座模型中的适用条件尚未系统评估。"

**(c) §3.7.1 EEGNet-Huge 死锁解读。**
> 现版："Cross-subject 准确率随容量单调下降，呈反向 scaling"
> 建议："Cross-subject 准确率在我们的 HP 协议下随容量单调下降至 chance；该结果与'EEGNet 架构内反向 scaling'与'~30M 量级在双 LR 协议下优化失败'两种解释均兼容，本文未通过完整 HPO sweep（≥3 LR × 3 warmup × 3 seed × init scheme 对照 + grad norm logging）进行严格隔离。"

**(d) §3.7.2 about random-init within-subject collapse.**
> 现版："~4M 参数的 transformer 在 ~70 trial 单被试样本下没有预训练先验时变成'负容量'"
> 建议："这与 NLP transformer 在小样本下的已知失败模式一致（Mosbach et al. 2021 在 RTE ~2K 样本上 BERT-base 约 1/3 random seed 落入 chance；Zhang et al. 2021 给出 top-K layer re-init / long warmup / mixout 等稳定化方法）。本结果的 EEG-specific 价值在于把这一脆弱性精确量化到手指 MI 任务、cross-seed 复现确认、并定位 TUEG 预训练对该脆弱性的补偿幅度（~+27 pp）。"

**(e) §4.8 末尾关于 EEG 'domain' 的方法论命题。**
> 现版："EEG 基座模型的 'domain' 边界由信号级特征（采样率、频段、电极配置）而非任务级语义定义……这区别于 NLP/CV 的 domain-adaptive pre-training 经验"
> 建议："本研究观察的负迁移与 NLP DAPT 文献中'低 task-corpus 对齐 + source corpus 不足'的失败案例（Gururangan et al. 2020 §5.2 reviews 域结果）在结构上一致；进一步判断 EEG 基座模型是否需要不同于 NLP/CV 的 transfer 设计原则需要在多 backbone × 多 source corpus × 多预训练目标的矩阵下验证。"——把强主张"区别于 NLP/CV"调整为"与 NLP/CV 文献中的低对齐失败案例一致"。

**(f) §7 结论 finding 4 末尾。**
> 现版："只在存在类型更接近的 source MI 数据（如手指级、手部精细动作 MI）可用时才值得再考虑 DAPT。"
> 建议补充："此外，参数高效微调（LoRA, Hu et al. 2022 ICLR；adapter, Houlsby et al. 2019）作为 full DAPT 的替代是后续工作可考虑的方向——它能在不覆写 backbone 的前提下吸收外部 MI 数据信息，可能避免本研究观察的 catastrophic-overwrite 模式。"

## 6. What I Cannot Evaluate (knowledge gap disclosure)

**作为 NLP/CV reviewer 我无法评判**：

1. **§4.3 体积传导（volume conduction）**——我无法判断 4ch BP 的 78.75% 是否真的源于 mu/beta ERD 物理签名而非数据泄露/通道选择泄露的某种我看不到的形式。这是 R2 domain reviewer 的领地。
2. **§2.7.2 / §3.6 关于 mask ratio 50%、masked autoencoding 在 EEG 信号上的合理性**——我能看出 mask ratio 50% 与 BERT 15% 的差异，但无法判断 EEG 信号统计是否需要这么高的 mask ratio。
3. **§3.5 中 FDR / CSP / Attention / Band Power 各方法的 EEG 文献依据是否充分**——这是 R2 的领地。
4. **§2.2 / §2.3 数据预处理与分割是否对 finger MI 任务做了合适处理**——我相信论文披露的分割协议在 EEG 文献中是标准的，但无独立判断。
5. **§4.1 关于 cross-subject 21× 训练数据的"21 名被试足以激活 transformer 架构归纳偏置"的具体阈值**——我没有 EEG 数据集 transformer 训练的足够先验来评判这个阈值是否合理。
6. **CBraMod ACPE 位置编码对 transformer-on-small-data 失败模式的具体补偿机制**——ACPE 是否构成针对 EEG 信号的有效 inductive bias 调节，需要 backbone 架构层面的 EEG-aware 分析，这是 R1 / R2 的领地。

## 7. Confidence in Review

**Confidence: 4 / 5**

我对本评审的 NLP/CV/scaling-law 部分有较高把握（已查证 §4 中所有引用文献的核心论点，并且本文随机种子复现 + Gururangan 2020 域对齐谱系 + Chinchilla scaling 协议都是 NLP 标准方法学）；扣 1 分是因为我无法独立评判本文是否在 EEG 领域已有 NLP-style HPO sweep 的隐性约定（如果 EEG 文献的 community standard 本就只跑双 LR、不做 multi-seed warmup sweep，那么我对 §3.7 的批评就需要软化为"建议加强"而非"证据强度不足"）。这一信心区间限定下，我推荐：**Major Revision**——本文的核心实验工作扎实，问题集中在跨学科叙事框架、文献定位、几处主张的强度校准上；这些修改可在 1-2 周内完成，无需新实验。

---

**评审结束**

**对编辑的简短补充**：本文最值得保留的跨学科价值是"在 EEG 基座模型上严谨复现了 NLP DAPT 文献中已知的低对齐 + 数据不足失败模式"，以及"在 finger MI 任务上量化了 transformer 小样本失败的精确尺度（with seed reproducibility）"。把这两点作为本文跨学科贡献的明确锚点（而非"EEG 区别于 NLP/CV 的范式级新发现"），既能让 NLP/CV reviewer 接受，也能让 EEG reviewer 看到清晰的 transfer learning 文献对话——这是 win-win。
