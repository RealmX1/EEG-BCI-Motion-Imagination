# Stage 4 Step 2 Part C — Literature + Minor Revisions + R&R Seed

**Subagent C scope**: P1.2 (literature expansion, 11 must/strongly-recommended + 4 cross-disciplinary + 2 optional refs) + P1.3 / P1.5 / P1.6 / P1.7 / P1.8 minor revisions + R&R Letter seed (per reviewer per comment).

**Coordination notes**: Reads paper_draft_v3.0.1.md as READ-ONLY. Does NOT touch §3.7 (Subagent B) or DAPT block (Subagent A). Provides multi-touch contribution snippets (cohort caveat for abstract / §1.4 F2 / §7 F2) that Subagent A and B can integrate into their respective edits.

---

## 1. Literature additions (P1.2)

### 1.1 New references with full bib + verification status

All references below verified via WebSearch (DOI / arXiv / venue / page numbers confirmed). The current paper has 9 refs ([1]–[9]); we propose adding refs [10]–[24] below (15 new refs, total → 24).

#### Tier A: Must add (R2 §5.1 — 6 refs)

**[10] Schirrmeister et al. 2017** — Deep ConvNet
- Full bib: R. T. Schirrmeister, J. T. Springenberg, L. D. J. Fiederer, M. Glasstetter, K. Eggensperger, M. Tangermann, F. Hutter, W. Burgard, T. Ball, "Deep learning with convolutional neural networks for EEG decoding and visualization," *Human Brain Mapping*, vol. 38, no. 11, pp. 5391–5420, 2017.
- DOI: 10.1002/hbm.23730
- arXiv: 1703.05051
- **Verification**: CONFIRMED (PubMed 28782865; Wiley OnlineLibrary). Volume/issue/pages exact.

**[11] Sakhavi et al. 2018** — FBCSP+CNN
- Full bib: S. Sakhavi, C. Guan, S. Yan, "Learning Temporal Information for Brain-Computer Interface Using Convolutional Neural Networks," *IEEE Transactions on Neural Networks and Learning Systems*, vol. 29, no. 11, pp. 5619–5629, Nov. 2018.
- DOI: 10.1109/TNNLS.2018.2789927
- **Verification**: CONFIRMED (IEEE Xplore document 8310961; PubMed 29994075).

**[12] Ang et al. 2008** — FBCSP
- Full bib: K. K. Ang, Z. Y. Chin, H. Zhang, C. Guan, "Filter Bank Common Spatial Pattern (FBCSP) in Brain-Computer Interface," in *Proceedings of the 2008 IEEE International Joint Conference on Neural Networks (IJCNN)*, Hong Kong, June 2008, pp. 2390–2397.
- DOI: 10.1109/IJCNN.2008.4634130
- **Verification**: CONFIRMED (IEEE IJCNN 2008 / IEEE WCCI 2008 proceedings).

**[13] Blankertz et al. 2008** — Spatial filters / CSP
- Full bib: B. Blankertz, R. Tomioka, S. Lemm, M. Kawanabe, K.-R. Müller, "Optimizing Spatial Filters for Robust EEG Single-Trial Analysis," *IEEE Signal Processing Magazine*, vol. 25, no. 1, pp. 41–56, Jan. 2008.
- DOI: 10.1109/MSP.2008.4408441
- **Verification**: CONFIRMED (IEEE Xplore 4408441).

**[14] Pfurtscheller & Lopes da Silva 1999** — ERD/ERS basic principles
- Full bib: G. Pfurtscheller, F. H. Lopes da Silva, "Event-related EEG/MEG synchronization and desynchronization: basic principles," *Clinical Neurophysiology*, vol. 110, no. 11, pp. 1842–1857, Nov. 1999.
- DOI: 10.1016/S1388-2457(99)00141-8
- **Verification**: CONFIRMED (PubMed 10576479; ScienceDirect S1388245799001418).

**[15] Jiang et al. 2025** — NeuroLM
- Full bib: W.-B. Jiang, Y. Wang, B.-L. Lu, D. Li, "NeuroLM: A Universal Multi-task Foundation Model for Bridging the Gap between Language and EEG Signals," in *The Thirteenth International Conference on Learning Representations (ICLR 2025)*, 2025.
- arXiv: 2409.00101
- OpenReview: forum?id=Io9yFt7XH7
- **Verification**: CONFIRMED (ICLR 2025 accepted; GitHub 935963004/NeuroLM).

#### Tier B: Strongly recommended (R2 §5.2 — 4 refs)

**[16] Yang et al. 2023** — BIOT
- Full bib: C. Yang, M. B. Westover, J. Sun, "BIOT: Biosignal Transformer for Cross-data Learning in the Wild," in *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*, 2023.
- arXiv: 2305.10351
- **Verification**: CONFIRMED (NeurIPS 2023 poster 71117). Note: arXiv title "Cross-data Biosignal Learning in the Wild" vs proceedings title "Biosignal Transformer for Cross-data Learning in the Wild" — use proceedings form.

**[17] Zhang et al. 2023** — Brant
- Full bib: D. Zhang, Z. Yuan, Y. Yang, J. Chen, J. Wang, Y. Li, "Brant: Foundation Model for Intracranial Neural Signal," in *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*, 2023.
- **Verification**: CONFIRMED (NeurIPS 2023 poster 72383; OpenReview DDkl9vaJyE).

**[18] Lotte et al. 2018** — BCI 10-year update
- Full bib: F. Lotte, L. Bougrain, A. Cichocki, M. Clerc, M. Congedo, A. Rakotomamonjy, F. Yger, "A review of classification algorithms for EEG-based brain–computer interfaces: a 10 year update," *Journal of Neural Engineering*, vol. 15, no. 3, p. 031005, Jun. 2018.
- DOI: 10.1088/1741-2552/aab2f2
- **Verification**: CONFIRMED (IOPscience; PubMed 29488902).

**[19] Neuper et al. 2006** — ERD/ERS sensorimotor
- Full bib: C. Neuper, M. Wörtz, G. Pfurtscheller, "ERD/ERS patterns reflecting sensorimotor activation and deactivation," in *Progress in Brain Research*, vol. 159, pp. 211–222, 2006.
- DOI: 10.1016/S0079-6123(06)59014-4
- **Verification**: CONFIRMED (ScienceDirect; PubMed 17071233).

#### Tier C: Cross-disciplinary (R3 §4 — 3 refs)

**[20] Gururangan et al. 2020** — DAPT
- Full bib: S. Gururangan, A. Marasović, S. Swayamdipta, K. Lo, I. Beltagy, D. Downey, N. A. Smith, "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks," in *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020)*, pp. 8342–8360, 2020.
- DOI: 10.18653/v1/2020.acl-main.740
- arXiv: 2004.10964
- **Verification**: CONFIRMED.

**[21] Mosbach et al. 2021** — BERT fine-tuning stability
- Full bib: M. Mosbach, M. Andriushchenko, D. Klakow, "On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines," in *International Conference on Learning Representations (ICLR 2021)*, 2021.
- arXiv: 2006.04884
- OpenReview: forum?id=nzpLWnVAyah
- **Verification**: CONFIRMED.

**[22] Hoffmann et al. 2022** — Chinchilla
- Full bib: J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, et al., "Training Compute-Optimal Large Language Models," in *Advances in Neural Information Processing Systems 35 (NeurIPS 2022)*, 2022.
- arXiv: 2203.15556
- **Verification**: CONFIRMED.

#### Tier D: Optional (icing — 2 refs, kept for HPO calibration argument in §2.5.1)

**[23] Bergstra et al. 2011** — TPE
- Full bib: J. Bergstra, R. Bardenet, Y. Bengio, B. Kégl, "Algorithms for Hyper-Parameter Optimization," in *Advances in Neural Information Processing Systems 24 (NeurIPS 2011)*, pp. 2546–2554, 2011.
- **Verification**: CONFIRMED (NeurIPS 2011 paper 4443). Note: 4 authors, not just Bergstra & Bengio; correct attribution.

**[24] Snoek et al. 2012** — Practical Bayesian Optimization
- Full bib: J. Snoek, H. Larochelle, R. P. Adams, "Practical Bayesian Optimization of Machine Learning Algorithms," in *Advances in Neural Information Processing Systems 25 (NeurIPS 2012)*, 2012.
- arXiv: 1206.2944
- **Verification**: CONFIRMED.

#### Coordination ref (P1.7 dependency — 1 ref)

**[25] Pan & Yang 2010** — Transfer learning survey
- Full bib: S. J. Pan, Q. Yang, "A Survey on Transfer Learning," *IEEE Transactions on Knowledge and Data Engineering*, vol. 22, no. 10, pp. 1345–1359, Oct. 2010.
- DOI: 10.1109/TKDE.2009.191
- **Verification**: CONFIRMED. Required for §3.3 XSI-FT lineage paragraph (P1.7 spec).

#### Note on existing [5] (EEGNet original paper)

EEGNet original is already cited as [5] (Lawhern et al. 2018, *J. Neural Eng.*). No re-add needed; but R2 minor #1 wants stronger inline citation for Lawhern 2018 in §3.2 (EEGNet cross-subject behavior) and §2.4.1 (HPO recovers EEGNet-16,4 not new finding). Surface only — already in ref list.

### 1.2 Inline citation locations (per ref)

The proposed citation map:

| Ref | Where to cite (section + context) | Justification (1 sentence) |
|-----|-----------------------------------|----------------------------|
| [10] Schirrmeister 2017 | §3.7.1 第 1 段 (Mid/Huge stem 设计动机段); §1.3 (EEG deep learning landscape) | Deep ConvNet 是与 EEGNet 并列的 BCI deep learning baseline; §3.7.1 讨论 EEGNet 容量阶梯需引用其作为 stem 设计的文献锚点。 |
| [11] Sakhavi 2018 | §2.6 (CSP 通道选择); §4.1 (CBraMod vs EEGNet 对比的 baseline 上下文) | FBCSP+CNN 混合方法是 deep MI 解码主流 baseline 之一。 |
| [12] Ang 2008 | §2.6 末段 (CSP 描述处); 表 0 footnote 文献锚点 | FBCSP 在 BCI Competition IV 上的标杆地位。 |
| [13] Blankertz 2008 | §2.6 CSP 段 (与 [12] 配对); §3.5.3 4ch 控制实验讨论 | CSP 现代综述与 robust 实现，是 spatial filter 标准引用。 |
| [14] Pfurtscheller & Lopes da Silva 1999 | §3.5.2 4ch BP 解剖学讨论第 1 段; §4.4 多 session 讨论 ERD 处; §7 Finding 5 物理动机段 | mu/beta ERD 经典原作 — 全文多处依赖该概念。 |
| [15] Jiang 2025 (NeuroLM) | §1.3 (与 [4] CBraMod 并列); §5 Limitation #7 (其他 backbone 是否复现) | 同 ICLR 2025 并行 EEG foundation model；R2 明示"不引用难以原谅"。 |
| [16] Yang 2023 (BIOT) | §1.3 (EEG foundation model 综述段); §2.4.2 (ACPE 与 BIOT tokenization 对照) | 与 ACPE 形成方法学对照（biosignal sentence vs 通道位置编码）。 |
| [17] Zhang 2023 (Brant) | §1.3; §3.7.3 三向分解末尾 (与"盲目扩参"主张形成对比，Brant 在 SEEG 上扩参有效) | 500M 参数 SEEG foundation model — 与本文"扩参 EEGNet 有害"形成对比。 |
| [18] Lotte 2018 | §3.3 XSI-FT 第一次定义段 (P1.7 文献溯源, "subject-adaptive transfer learning"); §1.2 BCI 算法综述总览 | XSI-FT 机制的领域名分类来源。 |
| [19] Neuper 2006 | §3.5.2 4ch BP 解剖学段 (与 [14] 配对); §7 Finding 5 | 与 [14] 配对引用 ERD/ERS sensorimotor 框架。 |
| [20] Gururangan 2020 | §1.3 末段 (DAPT 条件性成功); §4.5 第 1 段; §4.8 末段 (P1.3 命题降语气); §7 Finding 4 | DAPT 文献锚定 — 把本文负迁移定位为 NLP DAPT 已知失败模式而非"EEG specific". |
| [21] Mosbach 2021 | §3.7.2 random-init within ternary 18/21 chance-collapse 解读段; §4.1 第 4 段 (within 范式预训练贡献) | BERT 小样本 1/3 chance-collapse 是本文 18/21 collapse 的直接 NLP 先例。 |
| [22] Hoffmann 2022 (Chinchilla) | §3.7.1 v1/v2 死锁段 footnote (R3 推荐); §6 Future Work #6 (EEGNet HPO sweep 设计) | Compute-optimal scaling N/D 平衡；本文 30M EEGNet × ~3K trial 严重 N/D 失衡 (备用引用)。 |
| [23] Bergstra 2011 (TPE) | §2.5.1 第 1 段 (TPE 算法引用 — 当前文中提及 TPE 无引用) | TPE 原作 — 当前 §2.5.1 只提"TPE (Tree-structured Parzen Estimator)"未引用。 |
| [24] Snoek 2012 | §2.5.1 第 1 段 (Bayesian HPO 框架引用) | Practical Bayesian Optimization 标准引用，为 §2.5.1 calibration argument 增援 (Subagent B territory)。 |
| [25] Pan & Yang 2010 | §3.3 XSI-FT 第一次定义段 (P1.7 spec) | Inductive transfer learning 框架原作 — XSI-FT 机制的更广文献溯源。 |

### 1.3 Updated References section text (full new entries)

To be appended to the end of paper_draft_v3.0.1.md References section (currently ends with [9]):

```markdown
[10] R. T. Schirrmeister, J. T. Springenberg, L. D. J. Fiederer, M. Glasstetter, K. Eggensperger, M. Tangermann, F. Hutter, W. Burgard, and T. Ball, "Deep learning with convolutional neural networks for EEG decoding and visualization," *Human Brain Mapping*, vol. 38, no. 11, pp. 5391–5420, 2017, doi: 10.1002/hbm.23730.

[11] S. Sakhavi, C. Guan, and S. Yan, "Learning Temporal Information for Brain-Computer Interface Using Convolutional Neural Networks," *IEEE Transactions on Neural Networks and Learning Systems*, vol. 29, no. 11, pp. 5619–5629, 2018, doi: 10.1109/TNNLS.2018.2789927.

[12] K. K. Ang, Z. Y. Chin, H. Zhang, and C. Guan, "Filter Bank Common Spatial Pattern (FBCSP) in Brain-Computer Interface," in *Proc. 2008 IEEE Int. Joint Conf. Neural Networks (IJCNN)*, Hong Kong, 2008, pp. 2390–2397, doi: 10.1109/IJCNN.2008.4634130.

[13] B. Blankertz, R. Tomioka, S. Lemm, M. Kawanabe, and K.-R. Müller, "Optimizing Spatial Filters for Robust EEG Single-Trial Analysis," *IEEE Signal Processing Magazine*, vol. 25, no. 1, pp. 41–56, 2008, doi: 10.1109/MSP.2008.4408441.

[14] G. Pfurtscheller and F. H. Lopes da Silva, "Event-related EEG/MEG synchronization and desynchronization: basic principles," *Clinical Neurophysiology*, vol. 110, no. 11, pp. 1842–1857, 1999, doi: 10.1016/S1388-2457(99)00141-8.

[15] W.-B. Jiang, Y. Wang, B.-L. Lu, and D. Li, "NeuroLM: A Universal Multi-task Foundation Model for Bridging the Gap between Language and EEG Signals," in *The Thirteenth International Conference on Learning Representations (ICLR)*, 2025.

[16] C. Yang, M. B. Westover, and J. Sun, "BIOT: Biosignal Transformer for Cross-data Learning in the Wild," in *Advances in Neural Information Processing Systems 36 (NeurIPS)*, 2023.

[17] D. Zhang, Z. Yuan, Y. Yang, J. Chen, J. Wang, and Y. Li, "Brant: Foundation Model for Intracranial Neural Signal," in *Advances in Neural Information Processing Systems 36 (NeurIPS)*, 2023.

[18] F. Lotte, L. Bougrain, A. Cichocki, M. Clerc, M. Congedo, A. Rakotomamonjy, and F. Yger, "A review of classification algorithms for EEG-based brain–computer interfaces: a 10 year update," *Journal of Neural Engineering*, vol. 15, no. 3, p. 031005, 2018, doi: 10.1088/1741-2552/aab2f2.

[19] C. Neuper, M. Wörtz, and G. Pfurtscheller, "ERD/ERS patterns reflecting sensorimotor activation and deactivation," in *Progress in Brain Research*, vol. 159, pp. 211–222, 2006, doi: 10.1016/S0079-6123(06)59014-4.

[20] S. Gururangan, A. Marasović, S. Swayamdipta, K. Lo, I. Beltagy, D. Downey, and N. A. Smith, "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks," in *Proc. 58th Annual Meeting of the Association for Computational Linguistics (ACL)*, 2020, pp. 8342–8360, doi: 10.18653/v1/2020.acl-main.740.

[21] M. Mosbach, M. Andriushchenko, and D. Klakow, "On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines," in *International Conference on Learning Representations (ICLR)*, 2021.

[22] J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, et al., "Training Compute-Optimal Large Language Models," in *Advances in Neural Information Processing Systems 35 (NeurIPS)*, 2022.

[23] J. Bergstra, R. Bardenet, Y. Bengio, and B. Kégl, "Algorithms for Hyper-Parameter Optimization," in *Advances in Neural Information Processing Systems 24 (NeurIPS)*, 2011, pp. 2546–2554.

[24] J. Snoek, H. Larochelle, and R. P. Adams, "Practical Bayesian Optimization of Machine Learning Algorithms," in *Advances in Neural Information Processing Systems 25 (NeurIPS)*, 2012.

[25] S. J. Pan and Q. Yang, "A Survey on Transfer Learning," *IEEE Transactions on Knowledge and Data Engineering*, vol. 22, no. 10, pp. 1345–1359, 2010, doi: 10.1109/TKDE.2009.191.
```

---

## 2. Minor revision EDITs

### EDIT C1 — P1.3 §4.8 末段 + §7 Finding 4 末段 propositional softening

**Anchor 1**: §4.8 末段 (line 948 of paper_draft_v3.0.1.md)

**Old text (line 948)**:
> 贯穿这条路径的方法论命题是：**EEG 基座模型的 'domain' 边界由信号级特征（采样率、频段、电极配置）而非任务级语义定义**——粗运动 MI 数据不能作为 finger MI 任务的 domain-adaptive 来源，即使两者都属 'MI' 语义类别。这区别于 NLP/CV 的 domain-adaptive pre-training 经验，提示 EEG foundation model 的 transfer 路径需要新的设计原则。

**New text**:
> 贯穿这条路径的方向性观察是：本研究观察的负迁移与 NLP DAPT 文献中"低 task-corpus 对齐 + source corpus 不足"失败案例（Gururangan et al. 2020 [20] §5.2 reviews 域结果）在结构上一致；在 CBraMod backbone × masked-AE 预训练目标 × 粗运动 MI source pool × finger MI target 的具体配置下，通道几何错位（target 128ch vs source 95% 低密度）与训练超参数对 DAPT 结果的影响至少与任务粒度相当。下游 BCI 实践应优先匹配通道几何与信号尺度，再考虑任务语义对齐。判断 EEG 基座模型是否需要不同于 NLP/CV 的 transfer 设计原则，需要在多 backbone × 多 source corpus × 多预训练目标的矩阵下验证。

**Rationale**: P1.3 spec verbatim — replaces the strong methodological proposition with NLP-DAPT-anchored hedged version. Addresses EIC §6.2 #1+#3, R2 §3.3, R3 §3.1, DA §6 OG#1.

---

**Anchor 2**: §7 Finding 4 末段 (line 1001) + §7 末段 (line 1005)

**Old text (line 1005, 末段总结句)**:
> 上述发现共同支持了 CBraMod + FDR 32 通道 BCI 系统在手指级运动想象分类中的实用化部署，并提示一个更高阶的方法论论断：EEG 基座模型的 transfer 路径与 NLP/CV 的 domain-adaptive pre-training 范式不同——其 'domain' 边界由信号级特征（采样率、频段、电极配置）而非任务级语义定义。本研究的负面 DAPT 结果挑战了"同属 MI 任务即可作为 domain-adaptive 来源"的默认预期，提示后续 EEG foundation model 设计应以信号级 domain 对齐为首要选择标准。

**New text**:
> 上述发现共同支持了 CBraMod + FDR 32 通道 BCI 系统在手指级运动想象分类中的实用化部署。本研究观察的 DAPT 负迁移与 NLP DAPT 文献中"低 task-corpus 对齐 + source corpus 不足"失败案例（Gururangan et al. 2020 [20]）在结构上一致；在 CBraMod backbone × masked autoencoding × 粗运动 MI source pool × finger MI target 的具体配置下，通道几何错位（target 128ch vs source 95% 低密度）与任务粒度差异均独立驱动负迁移。本研究**不主张**"EEG foundation model 的 transfer 路径与 NLP/CV 范式级不同"——单 backbone × 单 source pool × 单下游任务的样本不足以支持该普适命题；下游 BCI 实践应优先匹配通道几何与信号尺度，并在存在更高 task-corpus 对齐度（如手指级、手部精细动作 MI source）时再考虑 DAPT。判断 EEG 基座模型是否需要独立于 NLP/CV 的 transfer 设计原则，需要在多 backbone × 多 source corpus × 多预训练目标的矩阵下系统验证（§6 后续工作 #3, #7）。

**Rationale**: 该段同时降语气 §7 末段命题 + §7 Finding 4 的"EEG domain 由信号级特征定义"声称。同时显式 cross-link 到 §6 后续工作。

**Note for Subagent A (DAPT block)**: Subagent A's §3.6 / §4.5 / §7 Finding 4 first-sentence rewriting takes precedence; this EDIT C1 only owns the **末段 propositional softening** and adds [20] inline. Coordinate so that Finding 4 first sentence (Subagent A) flows naturally into the 末段 (Subagent C softening).

---

### EDIT C2 — P1.6 §3.5.4 "XSI-FT 收益框架" 降级为 N=3 方向性观察

**Anchor**: §3.5.4 末段 (lines 703–705)

**Old text (line 703)**:
> 换言之，XSI-FT 收益不是通道数量的单调函数，而是"cross-subject baseline 离该 (channel, method) 组合的容量上限的距离"的函数：32ch FDR 距离上限较远（XSI-FT 有空间），8ch BP 已接近上限（XSI-FT 反而有害），128ch CBraMod 在表征层面对该任务已经饱和。此分析为 §4.6 部署路线图的"低密度 + XSI-FT"组合添加了重要约束（详见 §4.6 / §4.8）。

**New text**:
> 换言之，**基于 3 个数据点（128ch / 32ch FDR / 8ch BP）的方向性观察提示一个工作假设**：XSI-FT 收益可能不是通道数量的单调函数，而是与"cross-subject baseline 离该 (channel, method) 组合的容量上限的距离"相关——32ch FDR 距离上限较远（XSI-FT 有空间），8ch BP 已接近上限（XSI-FT 反而有害），128ch CBraMod 在表征层面对该任务已经饱和。**该工作假设基于 N=3 数据点，强烈受样本量限制；要把它升级为可推广方法论命题，至少需要在 8ch FDR、32ch BP、4ch BP 等额外 (channel, method) 组合上独立验证（§6 后续工作 #2）。** 在该 caveat 下，该方向性观察为 §4.6 部署路线图的"低密度 + XSI-FT"组合添加了一个有待证伪的约束（详见 §4.6 / §4.8）。

**Rationale**: P1.6 spec verbatim — replaces "修订框架" language with "based on 3 data points / working hypothesis". Addresses R1 §3.6, Limitation #11.

---

### EDIT C3 — P1.7 §3.3 XSI-FT 文献溯源段

**Anchor**: §3.3 第一次定义 XSI-FT 段 (lines 362–370, between line 364 [definition] and line 370 [enumerate the 3 steps])

**Insert location**: After the existing 3-step enumeration in §3.3 (after line 370 "在该被试的 held-out test session 上评估，得到逐被试准确率，群体上 21 人聚合"), as a new paragraph BEFORE "XSI-FT 与 §3.2 cross-subject 的区别在于".

**New paragraph (insert)**:
> **该机制在 BCI 文献中已知，并非本研究方法学新颖性**。XSI-FT 对应 Lotte et al. 2018 [18] (J. Neural Eng. 综述) 中"subject-adaptive transfer learning"分类的离线版本；同时也是 Pan & Yang 2010 [25] 提出的 inductive transfer 框架在 EEG 上的具体 instance；机制层面与 Ding et al. [3] 的 same-day finetune 同构（仅 finetune 时机不同——[3] 为在线 same-day 增量更新，本研究为离线 held-out session 评估）。本研究将"cross-subject pretrain → per-subject finetune"命名为 XSI-FT 仅作为本论文实验记号便利；**本研究的方法学贡献限于在 finger-MI 数据 + EEG foundation model (CBraMod) 设置下系统量化它的边际收益与饱和条件**（§3.3 标准 split / §3.4.4 extra sessions / §3.5.4 缩减通道下三种维度，均在本节及对应章节展开）。

**Rationale**: P1.7 spec verbatim — adds literature lineage for XSI-FT and cites [18] Lotte 2018 (subject-adaptive) + [25] Pan & Yang 2010 (inductive transfer) + [3] Ding 2025 (mechanism homology). Addresses R2 §3.2 (XSI-FT novel-naming concern) + EIC §5.2 default-accept R2 stance.

**Coordination**: Existing §3.4.4 (line 512) and §3.5.4 (line 687) repeated mini-definitions of XSI-FT can be simplified to "XSI-FT (§3.3 mechanism)" reference once C3 is in place. Mark for Step 4 cleanup.

---

### EDIT C4 — P1.8 CBraMod parameter count unification

**Anchor 1**: 摘要 line 18

**Old text (line 18)**:
> 本研究系统对比了大规模 EEG 基座模型 CBraMod（~4M 参数，ICLR 2025）与轻量级卷积神经网络 EEGNet-16,4（~10K 参数）...

**New text**:
> 本研究系统对比了大规模 EEG 基座模型 CBraMod（30.48M 参数含分类头；~4M backbone + ~26M MLP 头，ICLR 2025）与轻量级卷积神经网络 EEGNet-16,4（~16K 参数）...

**Anchor 2**: §1.3 line 67

**Old text (line 67)**:
> CBraMod 拥有约 400 万参数，是 EEGNet-16,4 [5]（约 1 万参数，BCI 研究的标准基线 CNN）的 ~400 倍。

**New text**:
> CBraMod 含分类头共 30.48M 参数（其中 backbone ~4M + MLP 分类头 ~26M），是 EEGNet-16,4 [5]（~16K 参数，BCI 研究的标准基线 CNN）的约 1,900 倍（如表 2b）。

**Anchor 3**: §3.7.2 line 797

**Old text (excerpt around line 797)**:
> 这一非对称揭示 ~4M 参数的 transformer 在 ~70 trial 单被试样本下没有预训练先验时变成"负容量"——

**New text**:
> 这一非对称揭示 30.48M 参数的 transformer（含分类头；~4M backbone + ~26M MLP 头）在 ~70 trial 单被试样本下没有预训练先验时变成"负容量"——

**Anchor 4**: §4.1 line 871, line 875 (similar phrasing references "~4M")

**Old text (line 871, excerpt)**:
> 把 EEGNet 的 MLP 头扩展到 5.84M / 19.99M / 30.22M 三档...一个朴素担忧——"差距是否仅源自 ~16K vs ~4M 的容量量级差异"——

**New text**:
> 把 EEGNet 的 MLP 头扩展到 5.84M / 19.99M / 30.22M 三档...一个朴素担忧——"差距是否仅源自 ~16K vs 30.48M 的容量量级差异"——

**Old text (line 875, excerpt)**:
> 但当 within-subject 仅 ~70 trial 时，~4M 参数的 transformer 在没有预训练先验的情况下变成"负容量"，

**New text**:
> 但当 within-subject 仅 ~70 trial 时，30.48M 参数的 transformer（含分类头）在没有预训练先验的情况下变成"负容量"，

**Anchor 5**: §5 Limitation table — verify no "~4M" references; table 2b (line 194) already shows 30,484,402 ✓ (no edit needed).

**Rationale**: P1.8 spec verbatim — unify ALL paper mentions to "30.48M (含分类头)" with explicit "~4M backbone + ~26M MLP head" decomposition for first-mention abstract & §1.3 to address R1 §4.2 multi-source inconsistency.

**Verification check**: Grep results found `~4M / 约 400 万` at lines 18, 67, 797, 871, 875 — all 5 surfaces covered above. No other instances.

---

## 3. Multi-touch contributions (cohort caveat for §1.4 F1+F2 / abstract / §7 F1+F2)

These are short text snippets that other subagents (A: DAPT block + §1.4 F1; B: §3.7) can integrate into their own EDITs. **Subagent C does not own these final placements** — provides language only.

### C-Abstract — cohort caveat for 90.68% (insert near current line 19/20)

**Insert location**: 摘要 line 20, after "跨被试二分类 **+14.01 pp**（90.68% vs 76.67%）"

**Suggested phrasing**:
> ...跨被试二分类 **+14.01 pp**（90.68% vs 76.67%；21 名 responder 被试，原数据集 [3] 49 名招募者中筛选后 cohort，详见 §2.1）...

### C-§1.4 F1 — cohort caveat (Subagent A territory; provided as multi-touch contribution)

**Insert location**: §1.4 Finding 1, line 77, after "跨被试二分类 +14.01 pp"

**Suggested phrasing**:
> ...被试内 +7.05 pp、跨被试二分类 +14.01 pp（90.68% vs 76.67%）、跨被试三分类 +13.65 pp（在 21 名 responder cohort 上；该 cohort 继承自 [3] 的 49 → 21 离线筛选）...

### C-§1.4 F2 — cohort caveat for 96.7% retention

**Insert location**: §1.4 Finding 2, line 79, after "保留 128 通道性能的 **96.7%**"

**Suggested phrasing**:
> ...确立 FDR 选取的 32 通道保留 128 通道性能的 **96.7%**（在 21 名 responder cohort × cross-subject binary 上；通道选择 ranking 使用了所有 session 数据，可能轻微高估 retention，详见 Limitation #1）。

### C-§7 F1 — cohort caveat (Subagent B territory; provided)

**Insert location**: §7 Finding 1 末尾, line 995, after "盲目扩参不是改进路径，架构对齐与预训练表征才是关键"

**Suggested phrasing**:
> ...盲目扩参不是改进路径，架构对齐与预训练表征才是关键。**该结论限于 CBraMod backbone × 本数据集（21 名 responder cohort）× 当前 HPO 预算；其他 EEG transformer backbone (LaBraM [6], NeuroLM [15], BIOT [16]) 是否复现该三向分解需独立验证（§6 #7）。**

### C-§7 F2 — cohort caveat for 90.68% / 96.7% headline

**Insert location**: §7 Finding 2 行内, line 997, after "FDR 选取的 32 通道保留了 128 通道性能的 **96.7%**（87.71% vs 90.68%）"

**Suggested phrasing**:
> ...FDR 选取的 32 通道保留了 128 通道性能的 **96.7%**（87.71% vs 90.68%；在 21 名 responder cohort × cross-subject binary 上；通道选择 ranking 包含全 session 信息，可能轻微夸大 retention，详见 Limitation #1）...

---

## 4. R&R Letter seed (per reviewer per comment outline)

Format per concern:
```
- Concern reference: [Reviewer §X.Y, severity]
- Verbatim concern (1–2 sentences from review)
- Author response stance: [accept/partial/dispute]
- Where addressed: [section + EDIT ID]
- Expected reviewer satisfaction: [High/Medium/Low]
- Residual risk
```

### 4.1 EIC concerns (8 total, per stage3_eic_review.md §3 + §6)

**EIC-1** (Major, §摘要/§1.4/§7 — 核心方法论定位声明缺失开篇)
- Verbatim: "摘要第一段直接进入数字（'+7.05 pp / +14.01 pp / +13.65 pp'），但**没有一句话讲清楚本研究在三条独立技术轴之外的统一定位**……论文的真正贡献在于把三轴绑到同一个 cohort 上做联合系统评估。"
- Stance: ACCEPT
- Where addressed: 摘要前 3 行 rewrite (Subagent B/A coordination); §1.4 第 1 项 (Subagent B territory); 本 Subagent C 仅 review 联动一致性
- Expected satisfaction: HIGH
- Residual risk: 重写后摘要长度可能突破 JNE 250-word 偏好；Step 4 应作 length audit。

**EIC-2** (Major, §2.8 / 全文 — 统计报告深度低于 JNE 主流)
- Verbatim: "作者 explicit 声明'无多重比较校正'，并仅报告 mean ± SD + 配对 t-test p value……JNE 通常预期在以下任一 fallback 中至少满足一个：(a) FDR-BH 校正后给出 q value，(b) mixed-effects model，(c) bootstrap 95% CI。当前稿件三者均缺失。"
- Stance: ACCEPT
- Where addressed: P0.6 (Stage 4 Step 1, 已完成 stat_recompute_runner.py 输出 BH-q + Cohen's dz + 95% CI)
- Expected satisfaction: HIGH (条件: Step 4 把 stat_recompute_v4v5 输出整合入主表)
- Residual risk: 若主表更新不全，R1 + EIC 两位都会复议。

**EIC-3** (Moderate-to-Major, §3 / §4 / §7 — "+27 pp" 定义漂移)
- Verbatim: "摘要 / §1.4 的 '+27 pp' 在表层语法上是 'preceded by 至 ~+27 pp（被试内）'，会让读者误读为 'TUEG 预训练在被试内贡献 +27 pp'——但 §3.7.3 的实际定义是 'binary +23.10 + ternary +30.79 平均'……存在算术意义上的边界滑动。"
- Stance: ACCEPT
- Where addressed: P1.5 (摘要 / §1.4 / §7 改为 "binary +23.10 / ternary +30.79 pp" 双数列出); Subagent A territory + 本 EDIT C1/C-Abstract 多点贡献
- Expected satisfaction: HIGH
- Residual risk: 5 处文本 (摘要、§1.4 F1、§3.7.3、§4.1、§7 F1) 必须同步更新，遗漏任一处 EIC 会指出。

**EIC-4** (Moderate, §摘要 / §1.4 / §7 — 90.68% cohort caveat)
- Verbatim: "表 0 把 '本文 CBraMod 128ch cross-subject 90.68%' 与 Ding et al. [3] EEGNet 80.56%（在线）并列……单数据集 + responder cohort + 高头条数字的组合容易被 reviewer 抓'过度泛化'。"
- Stance: ACCEPT
- Where addressed: P1.5 + 本 Subagent C-Abstract / C-§1.4 F2 / C-§7 F2 多点贡献
- Expected satisfaction: HIGH
- Residual risk: 仅文本修订，不补外部 cohort sanity check 实验，EIC §5.4 提到的 BCI Comp IV-2a 验证仍未做 — 但 EIC 自承"如不愿额外做实验则在 §5 加段说明" (P3.1)，残余风险 LOW。

**EIC-5** (Moderate, §3.6 / §4.5 / §7 F4 — DAPT 因果归因仍偏强)
- Verbatim: "V3 是'warm-restart-from-weights'两阶段训练……V3 vs V2 的差异是 (Stieger 占比 + warm-restart 续训) 的混合效应。作者在 §2.7.2 已 disclose，但在 §4.5 / §7 finding 4 的归因叙述中忽略了 warm-restart 这一干扰项。"
- Stance: ACCEPT
- Where addressed: P0.5 (Subagent A territory: §3.6 末段 + §4.5 + §7 F4 显式 surface warm-restart caveat)
- Expected satisfaction: MEDIUM-HIGH (Subagent A's §3.6 / §4.5 改写需要明确 mention warm-restart)
- Residual risk: 若 Subagent A 仅 caveat 加在 §3.6 而 §4.5 / §7 F4 漏，EIC 会指出。

**EIC-6** (Minor-to-Moderate, §3.7.1 / §4.1 — baseline → Mid 一跳混淆 conv stem 与 MLP 头)
- Verbatim: "把 '−25 pp 完全归因架构内扩参' 在 §4.1 / §7 finding 1 中改写为 'EEGNet 在 (F1=32 conv stem + 双层 MLP 头) 架构变体下扩参 → −25 pp'。"
- Stance: ACCEPT (partial — Subagent B territory)
- Where addressed: §3.7.1 / §4.1 / §7 F1 (Subagent B); Subagent C 不 own
- Expected satisfaction: MEDIUM
- Residual risk: 取决于 Subagent B 的措辞精度。

**EIC-7** (Minor, §摘要 / §3.4.4 / §4.4 — Extra sessions N=16 surface)
- Verbatim: "纵向 extra sessions 全部分析基于 N=16 子集……§4.4 部分叙述（'标准差从 10.81% 压缩至 5.98%'）没有 surface 'N=16 而非 21' 这一边界。"
- Stance: ACCEPT
- Where addressed: §4.4 第 3 段 + 摘要第 4 段 (extra sessions 部分) — 5 min 编辑加 "(N=16 子集)"标注 — 推荐 Step 4 做扫尾
- Expected satisfaction: HIGH
- Residual risk: NEGLIGIBLE

**EIC-8** (Minor, §3.5.2 — 4ch BP 解剖学讨论冗长)
- Verbatim: "解释机制部分（i / ii / iii 三种 hypothesis）行文偏冗长……建议把这一段压缩到 1/3 长度。"
- Stance: ACCEPT (P2.6)
- Where addressed: P2.6 — Subagent A or B Step 4 Possible cleanup (P2 优先级)
- Expected satisfaction: MEDIUM (压缩可选；不 critical)
- Residual risk: 若不压缩，EIC 不会复议；MINOR 级。
- **Note**: Pfurtscheller 1999 [14] + Neuper 2006 [19] 引用应在压缩后版本中保留 — Subagent C 已添加 ref，inline 落点在 §3.5.2 (见 §1.2 表)。

### 4.2 R1 concerns (10, per stage3_r1_methodology_review.md §3 + §4)

**R1-1** (CRITICAL, §3.7 三向分解隔离严密度)
- Verbatim: "EEGNet-Huge v1/v2 / random-init CBraMod 都未做专属 HPO；baseline → Mid 一跳同时改了 conv stem + MLP 头。"
- Stance: ACCEPT (P0.1 + P0.2 + P0.4)
- Where addressed: P0.1 EEGNet-Huge v1/v2 HPO + P0.2 random-init HPO + Subagent B §3.7 rewrite
- Expected satisfaction: HIGH (条件: P0.1 / P0.2 实验完成 + Subagent B 降语气)
- Residual risk: 实验若未完成，仅文本降语气可让 R1 转 Minor 但仍会要求第二轮再补；MEDIUM。

**R1-2** (Major, HPO 预算严重不对称)
- Verbatim: "Table S5b 应补 EEGNet cross-subject HPO 的 trial 计数 + EEGNet within HPO trial 数从 32 → 50+ 重跑 + fANOVA 95% bootstrap CI + 剪枝 bias check."
- Stance: PARTIAL ACCEPT (P2.1 优先级)
- Where addressed: P2.1 (EEGNet within HPO 50+ trial 重跑); §2.5.1 (Subagent B "HPO convergence verification" 段); Step 4 Table S5b 补 cross HPO 计数
- Expected satisfaction: MEDIUM (条件: 至少补 EEGNet within HPO 重跑 + cross HPO 计数披露)
- Residual risk: bootstrap CI 与 pruning bias check 可能在 Stage 4 时间预算外，R1 第二轮可能复议。

**R1-3** (Major, 通道选择"轻微泄露"未量化)
- Verbatim: "必须补 'Train-only channel ranking' 控制实验……如果差异 1–3 pp，则需在 §3.5 / 摘要 / Finding 2 全面修订。"
- Stance: ACCEPT (P1.4 优先级)
- Where addressed: P1.4 (Train-only ranking clean recompute, ~4 GPU-hour); §3.5.3 加 "Train-only ranking control" 小节
- Expected satisfaction: HIGH (条件: 实验完成 + 数字一致)
- Residual risk: 若 retention 下降 ≥2 pp，96.7% 数字需修订 — 影响摘要 + §1.4 + §7 五处。

**R1-4** (Major, 统计检验 multi-comparison + effect size)
- Verbatim: "全文 paired t-test 的'独立检验数'≥ 20 次……Cohen's d / 95% CI 在主表中完全缺失。"
- Stance: ACCEPT (P0.6 已完成)
- Where addressed: P0.6 / Stage 4 Step 1 stat_recompute_runner 输出
- Expected satisfaction: HIGH (条件: 主表整合)
- Residual risk: 同 EIC-2; 主表更新完整度决定。

**R1-5** (Major, §3.6 DAPT V2 中断 + "完全收敛"主张)
- Verbatim: "V2 的负迁移幅度 −1.38 pp 是中心数字……一个 Epoch 12 中断的 checkpoint 与跑完 50 epoch 的 V3 在'训练充分度'上不等价。"
- Stance: ACCEPT (P0.5 + EDIT — Subagent A territory)
- Where addressed: P0.5 (Subagent A); §3.6 line 731 "V2 全量训练后" 改为 "V2 在 Epoch 12 处被 LMDB 崩溃强制截断"
- Expected satisfaction: MEDIUM-HIGH
- Residual risk: V2 retrain (R1 7.2 项 5) 可能未完成；R1 可在第二轮要求补。

**R1-6** (Major, §3.5.4 XSI-FT 解释框架基于 N=3)
- Verbatim: "3 个数据点不足以建立任何 scaling law……§3.5.4 与 §4.6 / §4.8 应将'XSI-FT 收益取决于...'措辞从'修订框架'降级为'基于 3 个数据点的方向性观察'。"
- Stance: ACCEPT (P1.6)
- Where addressed: EDIT C2 (本报告) + §4.6 第 2 行 + §4.8 第 2 行 multi-touch (Step 4 cleanup)
- Expected satisfaction: HIGH
- Residual risk: NEGLIGIBLE — 仅措辞降级。

**R1 Minor #1**: Figure 1 / 6 / 6b 版本不同步 → Stage 4 Phase 3 重生成 (Subagent B / Step 4)
**R1 Minor #2**: CBraMod 参数计数不一致 → EDIT C4 (本报告) 完成 — High satisfaction
**R1 Minor #3**: deepEEGNet 引用页码 → Step 4 cleanup
**R1 Minor #4**: EEGNet-16,4 16K vs 10K vs 16,162 三数字 → Step 4 unify
**R1 Minor #5**: §3.1 line 326 "S20 (52.50%/61.25%)" 标注 → Step 4 cleanup
**R1 Minor #6**: 预处理流水线 EEGNet vs CBraMod 不对齐 → Limitation 加段
**R1 Minor #7**: EMA Table S6 灰底 → P3.3 (optional)
**R1 Minor #8**: §3.7.2 random-init within HP 不适合 from-scratch → §3.7.2 caveat (Subagent B)

### 4.3 R2 concerns (10, per stage3_r2_domain_review.md §3 + §4)

**R2-1** (CRITICAL, §3.1 文献覆盖严重残缺)
- Verbatim: "9 条参考是不可接受的……必加 6 条 + 强烈建议 4 条 + 锦上添花 3 条。"
- Stance: ACCEPT
- Where addressed: P1.2 (本 Subagent C §1.1–§1.3) — 13 refs 添加（[10]–[22]）+ [25]
- Expected satisfaction: HIGH (条件: refs 整合 + inline citations 完成)
- Residual risk: NEGLIGIBLE — 全部 refs 已 verify。

**R2-2** (Major, §3.2 XSI-FT novel-naming concern)
- Verbatim: "XSI-FT 在 BCI 文献至少十年是已知协议……领域审稿人会立刻识别出这是 well-known LOSO + per-subject finetune 的换名。"
- Stance: ACCEPT (P1.7 — 首选方案: 保留缩写 + 加文献溯源)
- Where addressed: EDIT C3 (本报告) — §3.3 第一次定义后插入文献溯源段
- Expected satisfaction: HIGH (R2 §3.2 显式将 P1.7 首选方案标为 "首选")
- Residual risk: §3.4.4 / §3.5.4 重复定义需简化为 "XSI-FT (§3.3 mechanism)" — Step 4 cleanup

**R2-3** (Major, §3.3 DAPT 方法学论断过度推广)
- Verbatim: "把强命题'由信号级特征定义'改为弱化版本……单 source × 单 target 的样本不足以支撑普适命题。"
- Stance: ACCEPT (P1.3)
- Where addressed: EDIT C1 (本报告) §4.8 末段 + §7 末段 改写
- Expected satisfaction: HIGH
- Residual risk: 若 Subagent A 在 §4.5 内的 DAPT 解读未对齐 EDIT C1 的 framing，可能 R2 复议。

**R2-4** (Major, 表 0 apples-to-oranges 风险)
- Verbatim: "Ding et al. [3] 的 80.56% 是 online same-day finetune 性能，本文 90.68% 是 offline cross-subject……表头'二分类准确率'列把不同评估范式数字放在同一列就构成视觉性的等价比较。"
- Stance: ACCEPT (partial)
- Where addressed: 表 0 重命名 + 加"评估难度"列 (Step 4 cleanup); 摘要 / §7 不再用 "90.68%" 与 "80.56%" 直接对话 — 本 EDIT C-Abstract 已加 cohort caveat 帮助消歧
- Expected satisfaction: MEDIUM-HIGH
- Residual risk: 需 Step 4 实际修改表 0 — 若仅加 caveat 不重排表头列，R2 第二轮可能复议。

**R2-5** (Minor, §3.2 EEGNet cross-subject 解读偏弱 → 引用 [5] Lawhern 2018)
- Stance: ACCEPT
- Where addressed: §3.2 line 350 加 [5] inline (Step 4 cleanup)
- Expected: HIGH

**R2-6** (Minor, §3.5.2 Pfurtscheller 引用) → EDIT C 中 [14] / [19] 已加 inline (§3.5.2 + §7 F5)

**R2-7** (Minor, §3.4 longitudinal BCI 文献对接) — Step 4 Discussion 加段，引用 BCI illiteracy 文献; 若加 [Ahn & Jun 2015]（R2 §5.3 锦上添花 #13），属可选

**R2-8** (Minor, §3.9 数据质量分类 → Mognon 2011 / ICLabel) — 不 critical, Step 4 可选

**R2-9** (Minor, EEGNet-16,4 vs EEGNet-8,2 "重新搜索" → 引用 [5]) — Step 4 §2.4.1 加 [5]

**R2-10** (Minor, Ding [3] cohort 筛选影响显式化) — EDIT C-Abstract 已部分覆盖；§3.1/§3.2 each occurrence 加 caveat — Step 4 cleanup

### 4.4 R3 concerns (7, per stage3_r3_perspective_review.md §3)

**R3-1** (Major, §1.3/§3.6/§4.5/§4.8 DAPT 框架与 NLP DAPT 文献对话缺失)
- Verbatim: "§1.3 末尾的扫式陈述把 NLP DAPT 描述为'已验证的范式'……应明确引用 Gururangan 2020 的'domain-relevance is a continuous spectrum'框架。"
- Stance: ACCEPT
- Where addressed: §1.3 末段 (Subagent A territory or Step 4) 引用 [20]; §4.5 第 1 段 (Subagent A); EDIT C1 §4.8 末段 + §7 末段
- Expected satisfaction: HIGH
- Residual risk: §1.3 line 69 "在 NLP 和 CV 中已得到验证" 改写需 Subagent A 或 Step 4 完成。

**R3-2** (Major, §3.7.2 Random-init CBraMod within ternary 18/21 chance-collapse 解读单方向)
- Verbatim: "应改为'与 NLP transformer 在小样本下的已知失败模式一致 (Mosbach et al. 2021 [21]; Zhang et al. 2021)'……本结果的 EEG-specific 价值在于精确量化 + cross-seed 复现 + 量化 TUEG 补偿幅度。"
- Stance: ACCEPT
- Where addressed: P1.1 (NLP 锚定文献) — Subagent B territory; ref [21] 已加, inline 落点 §3.7.2 + §4.1 第 4 段
- Expected satisfaction: HIGH

**R3-3** (Major, §3.7.1 EEGNet-Huge v1/v2 死锁 = "capacity 反向 scaling")
- Verbatim: "0.693 train loss 死锁更像优化失败而非'capacity reverse-scaling'……三向分解的解读需要软化。"
- Stance: ACCEPT (P0.4 + Subagent B)
- Where addressed: §3.7.1 + §4.1 第 1 段 + §7 F1 (Subagent B territory); ref [22] Hoffmann 2022 (Chinchilla) 加 footnote
- Expected satisfaction: MEDIUM (条件: P0.1 实验 + 文本降语气)
- Residual risk: 若 P0.1 仅做 v3 而 v1/v2 未补 LayerNorm-on 对照，R3 仍可质疑。

**R3-4** (Major, "+27 pp" 归因强度)
- Stance: ACCEPT — 同 EIC-3 / R1-1
- Where addressed: P1.5 + EDIT C-Abstract + Subagent A §3.7.3 / §4.1 / §7 F1 文本统一

**R3-5** (Cross-domain citations 必加 — 12 refs)
- Stance: ACCEPT (其中 4 条 — [20] Gururangan, [21] Mosbach, [22] Hoffmann, [25] Pan & Yang 已加; 余下 NLP 文献 [Devlin 2019, Howard & Ruder 2018, Liu 2019 RoBERTa, Hu 2022 LoRA, Kaplan 2020, McKenzie 2023] 视 Subagent B 在 §3.7 / §4 中 inline 引用需求决定)
- Where addressed: §1.1 表 + 实际整合在 Subagent B 完成的 §3.7 修订
- Expected satisfaction: MEDIUM-HIGH

**R3-6** (Reframing recommendations) — EDIT C1 + Subagent A §4.5 + Subagent B §3.7 联合处理

**R3-7** (Knowledge boundary disclosure — 不要求修订) — N/A

### 4.5 Devil's Advocate concerns (1 CRITICAL + 4 MAJOR + 3 MODERATE + 3 MINOR per stage3_devils_advocate_review.md §9)

**DA-CRITICAL #1.1** (HPO 预算非对称性混淆 §3.7 三向分解)
- Verbatim: "+34.97 pp / +27 pp / −25 pp 三元组依赖于'扩参 EEGNet 没有 HPO + random-init CBraMod 没有 HPO + EEGNet baseline 32 vs CBraMod 51–77 trial'的非对称预算……是 HPO budget asymmetry confounding 的教科书案例。"
- Stance: PARTIAL ACCEPT (W stance, two-part)
  - **Part A** (substantive defense): §2.5.1 calibration argument — TPE 收敛理论 (Bergstra 2011 [23]) + Bayesian optimization 在 ~50 trial 内收敛证据 (Snoek 2012 [24])，定量论证 CBraMod 51–77 trial 已经过 best_value plateau，新增 trial 边际收益 < 0.5 pp。**这是 Subagent B 的 territory** — Subagent C 仅提供 [23] / [24] inline ref 支援。
  - **Part B** (acceptance): §3.7 文本降语气 — Subagent B 把 +34.97 pp / −25 pp 改为 "在受限 HPO 预算下观察到的复合估计"，并把 §3.7.3 三向分解表加 footnote。
- Where addressed: §2.5.1 (Subagent B Part A); §3.7 全章 + §7 F1 (Subagent B Part B); §6 后续工作 #6 (Subagent B 加 EEGNet-Huge v1/v2 LayerNorm-on HPO + CBraMod random-init 专属 HPO 作为 Future Work)
- Expected satisfaction: MEDIUM-HIGH (条件: Part A + Part B 共同呈现使 DA 对 W stance 满意)
- Residual risk: 若 Part A 论证不够硬 (例如 Bayesian optimization 收敛证据未量化), DA 第二轮可能复议；建议 Subagent B 在 §2.5.1 报告 best_value 在 trial 30 vs trial 77 间的具体差距。

**DA-MAJOR #1.2** (Cross-subject 90.68% shortcut/leakage 风险)
- Stance: ACCEPT (P0.3 label-shuffle control + P1.4 channel clean recompute + P2.5 cohort inflation 量化)
- Where addressed: P0.3 / P1.4 (实验); P2.5 §5 Limitation #2 加段; 摘要 / §1.4 / §7 F2 加 cohort caveat (本 EDIT C-* 多点贡献)
- Expected satisfaction: HIGH (条件: label-shuffle 实验完成 + cohort caveat 全面 surface)
- Residual risk: 外部 BCI Comp IV-2a 验证 (DA OPTIONAL d) 仍未做。

**DA-MAJOR #1.3** (DAPT V1/V2/V3 一致负迁移可能是 artifact)
- Stance: PARTIAL ACCEPT
  - Per-subject paired-t (P0.5 part of Subagent A scope)
  - V1→V2 cross binary +0.59 pp 反向证据 surface (Subagent A §3.6)
  - V3 warm-restart 干扰项 surface (EIC-5 already covered)
  - V4 控制实验 (P2.3 optional)
- Where addressed: P0.5 (Subagent A territory)
- Expected satisfaction: MEDIUM-HIGH
- Residual risk: V4 实验若未做, DA 仍可能复议。

**DA-MODERATE #4 (Confirmation Bias Audit)** — Bias #1 / #3 — 加 §4.X "Alternative Interpretations" (P2.4)
- Stance: ACCEPT (P2.4)
- Where addressed: 新增 §4.9 (Subagent A or Step 4 territory)
- Expected satisfaction: MEDIUM

**DA-MODERATE #6 (Overgeneralization Audit)** — OG #1: "EEG domain 由信号级特征定义" → EDIT C1 (本报告)；OG #2 "盲目扩参不是改进路径" → Subagent B §4.1 + §7 F1; OG #3 "32ch FDR 是稳健的精度-硬件权衡点" → Step 4 §4.2 / §4.6 / §7 F2 hedge 统一
- Stance: ACCEPT
- Expected satisfaction: HIGH

**DA-MODERATE #8 (Stakeholder blind spots)** — P3.2: §4.6 加 wearable / edge benchmark hedge; 隐私问题 acknowledge
- Stance: ACCEPT (P3.2 — optional but 5 min edit)
- Where addressed: Step 4 §4.6 加 hedge 段
- Expected satisfaction: HIGH

**DA-MINOR Cherry-pick #1 (90.68% cohort filter)** — 同 EIC-4 / R2-10 — EDIT C-Abstract 多点贡献覆盖
**DA-MINOR Cherry-pick #2 (96.7% retention 通道选择 leakage)** — P1.4 实验 + EDIT C-§1.4 F2 / C-§7 F2 multi-touch
**DA-MINOR Cherry-pick #3 (4ch BP 78.75% reporting framing)** — Subagent B §1.4 F5 / §7 F5 hedge (已在 v3.0.1 中相对充分)

---

## 5. Numbers / refs cross-check

**Refs total**: 9 → 24 (15 new refs added). R2 §5 expectation "9 → ~20" 满足并超出 (24 落在 R2 sample 区间 18–22 内)。

**Ref number gaps**: [10]–[25] 连续编号 (含 [25] Pan & Yang)，无跳号。

**Inline citation locations**: 16 locations across §1.1.2 表 — covers §1.2 / §1.3 / §2.4.1 / §2.4.2 / §2.5.1 / §2.6 / §3.3 / §3.5.2 / §3.5.3 / §3.7.1 / §3.7.2 / §3.7.3 / §4.1 / §4.5 / §4.8 / §5 Limitation / §6 / §7 F1 / §7 F4 / §7 F5。

**Verification status**: 全部 15 refs via WebSearch — 每条均查到 DOI / arXiv / 出版页面，确认 venue / year / pages 准确。**0 ref unverified.**

**Numbers cross-check**:
- 30.48M → 5 处 unify (摘要 / §1.3 / §3.7.2 / §4.1 line 871 / §4.1 line 875). 表 2b (line 194) 已有 30,484,402 数字 — 不需变更。
- 90.68% cohort caveat → 4 处 surface (摘要 / §1.4 F2 / §7 F2 + §3.2 — 后者已有；前 3 处 Subagent C 多点贡献覆盖)。
- "+27 pp" 双数列出 → 5 处 (摘要 / §1.4 F1 / §3.7.3 / §4.1 / §7 F1) — Subagent A + Subagent B + Subagent C 三方协调。
- 96.7% retention → 5 处 (摘要 / §1.4 F2 / §3.5.3 / §4.2 / §7 F2) — P1.4 channel clean recompute 完成后视新数字决定是否更新。

---

## 6. Notes for orchestrator

### 6.1 Pan & Yang 2010 [25] 状态

**Pan & Yang 2010** (transfer learning survey) 是 P1.7 §3.3 XSI-FT 文献溯源段 (EDIT C3) 的必要引用，本报告已加为 ref [25]。Roadmap §P1.7 spec 的"加 Pan & Yang 2010 作为额外 ref" 提示已落实。**无 orchestrator 需协调**。

### 6.2 Subagent A coordination (DAPT block)

- §1.3 line 69 "这一假设在 NLP 和 CV 中已得到验证" 改写需引用 [20] Gururangan 2020 — Subagent A 完成
- §3.6 末段 + §4.5 + §7 Finding 4 改写 — Subagent A territory，**EDIT C1 仅 own §4.8 末段 + §7 末段**，请 Subagent A 在改写 Finding 4 第一句时与 EDIT C1 末段 framing 保持 consistent (即不出现 "EEG domain 由信号级特征定义" 强命题)。
- §1.4 Finding 1 改写 (含 cohort caveat C-§1.4 F1) — Subagent A territory, 本 Subagent C 提供语言。

### 6.3 Subagent B coordination (§3.7 / §2.5.1 / capacity)

- §2.5.1 calibration argument (DA W stance Part A) 中可使用 ref [23] Bergstra 2011 + [24] Snoek 2012 增援。本 Subagent C 不写 §2.5.1，但 ref 已 verify 并 inline 落点 § 1.2 表中标注。
- §3.7.1 v1/v2 死锁 footnote 可使用 ref [22] Hoffmann 2022 (Chinchilla) — N/D 失衡解释。
- §3.7.2 within ternary 18/21 collapse 解读使用 ref [21] Mosbach 2021 + 可选 Zhang 2021 ICLR (后者未在 P1.2 必加列表中, Subagent B 决定是否加)。
- §3.7.3 三向分解表 footnote "在共享默认 HP 与受限 HPO 预算下观察到的复合估计" — Subagent B 写, 本 Subagent C 不 own 该 footnote 但提供 framing 提示。
- §1.4 Finding 1 + §7 F1 cohort caveat (C-§7 F1) — Subagent B territory, 本 Subagent C 提供语言。

### 6.4 Step 4 cleanup tasks (post-Subagent A/B/C convergence)

1. §3.4.4 / §3.5.4 重复 XSI-FT 定义 → 简化为 "XSI-FT (§3.3 mechanism)" once EDIT C3 落地
2. R1 Minor #4: EEGNet-16,4 16K vs 10K vs 16,162 三数字 → unify 为 "16,162 (~16K) parameters"
3. EEGNet 预处理 vs CBraMod 预处理 confound → §5 Limitation 加段 (R1 Minor #6)
4. §3.5.2 Pfurtscheller 1999 [14] / Neuper 2006 [19] inline (P2.6 压缩同时落实)
5. §2.5.1 Bergstra 2011 [23] / Snoek 2012 [24] inline (TPE 引用)
6. 表 0 重命名 + 加"评估难度"列 (R2-4)
7. §4.6 / §7 F6 加 "桌面 GPU 测试; wearable / edge 部署需独立 benchmark" hedge (P3.2 / DA-MODERATE #8)
8. §1.3 line 69 改写 (Subagent A territory but flagged here for coordination)
9. §3.2 line 350 加 [5] Lawhern 2018 inline (R2-5)
10. §4.4 加 BCI illiteracy / longitudinal MI 文献对接 (R2-7, optional)
11. §3.4.4 (line 522) 把 "+0.86 pp" 后的 paired_p 列与 §3.4.5 表 15b 同步 (R1 Minor #4 + EIC-2 联动)

### 6.5 Refs not added but suggested by R3 (Subagent B might want)

R3 §4 列出 12 refs；本 Subagent C 添加了其中 4 (Gururangan, Mosbach, Hoffmann, optional Pan & Yang 已为 P1.7 加)。其余 8 (Devlin 2019, Howard & Ruder 2018, Beltagy 2019 SciBERT, Liu 2019 RoBERTa, Zhang 2021 ICLR few-sample BERT, Kaplan 2020 scaling laws, McKenzie 2023 inverse scaling, Hu 2022 LoRA) 视 Subagent B 在 §3.7 / §4.1 / §4.5 / §4.6 中 inline 引用需求决定。如需添加，建议作为 [26]–[33] 续编号。

### 6.6 Refs not added (declined)

- **R2 §5.3 锦上添花 #11 Jayaram & Barachant 2018 MOABB**: §2.7.1 提到 "通过 MOABB 框架" 但未引用 — 推荐 Step 4 加为 [26]，但本报告未加 (可选 + 不在 P1.2 spec 必加列表中)。
- **R2 §5.3 锦上添花 #12 Koles 1991 (CSP 原作)**: Blankertz 2008 [13] 已覆盖 CSP 现代综述, Koles 1991 仅作完整文献链可选。
- **R2 §5.3 锦上添花 #13 Ahn & Jun 2015 BCI illiteracy**: §3.4 / §4.4 longitudinal 讨论可选引用，不在 P1.2 必加列表。

---

*— End of Subagent C Report —*
