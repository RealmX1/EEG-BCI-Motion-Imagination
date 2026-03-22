# Further Pre-training 外部 MI 数据集详细调查

> **数据来源**: 各数据集原始论文 PDF（`references/data/`）+ MOABB 文档 + 网络检索
> **审计数据**: `results/pretraining/audit_report.json` (2026-03-21)

## 概述

CBraMod further pre-training 使用了 10 个通过 MOABB 框架下载的公开运动想象 (Motor Imagery, MI) EEG 数据集，总计约 870 小时原始录制、363 名被试。所有数据集均为粗粒度肢体运动想象（手/脚级别），与本项目的单指运动解码 (finger-level MI) 任务存在 domain gap。

---

## Cho2017
**论文**: EEG datasets for motor imagery brain-computer interface (GigaScience, 2017)
**DOI**: 10.1093/gigascience/gix034
**PDF**: `references/data/gigascience_6_7_gix034.pdf`
**数据集概要**: 52 名被试（19 女，平均 24.8±3.86 岁），64 通道 Ag/AgCl 有源电极（Biosemi ActiveTwo，10-10 系统），采样率 512 Hz，同步 4 通道 EMG。单次 session，5 个 MI run + 1 个 online run + 非任务态数据（静息态、眼动、头动、咬牙）。BCI2000 3.0.2 系统采集。
**实验范式**: 左手/右手 MI，想象四根手指依次触碰拇指（动觉想象）。每类 100-120 个 trial，分 5 个 run。Trial 时序：2s 注视十字 → 3s MI（左/右箭头提示）→ 随机休息。
**预处理**: 原始数据以 MATLAB `.mat` 发布，未做滤波。
**数据清洗**: 提供 bad trial 标注（`badtrialindices` 字段）：(1) 8-30 Hz 带通后 ±100 µV 阈值；(2) EMG 50-250 Hz 相关 >0.8 的 trial。原始 trial 均保留。
**数据格式与单位**: `.mat` 文件，µV。
**已知问题**: (1) 14 名被试（26.9%）MI 准确率低于随机水平；(2) **MOABB 单位 bug**——加载器假设 `.mat` 数据为 µV 并乘 1e-6 转 V，但观测到 mean_abs=20,175 µV，异常高（可能 .mat 存储的是 nV）；(3) 被试 s20/s33 为双利手。
**与本项目关系**: 粗粒度左/右手 MI，与单指解码范式差异大。52 被试规模可作为预训练数据源。

---

## GrosseWentrup2009
**论文**: Beamforming in Noninvasive Brain-Computer Interfaces (IEEE Trans. Biomed. Eng., 2009)
**DOI**: 10.1109/TBME.2008.2009768
**PDF**: `references/data/GrosseWentrup2009_multiclass_CSP.pdf` — **注意：PDF 内容错误**，实际包含 Ranji et al. 心肌损伤论文，非 Grosse-Wentrup 论文
**数据集概要**: 10 名健康被试（8 右利手，2 女，平均 25.6±2.5 岁），128 通道 EEG（extended 10-20），采样率 500 Hz，Cz 参考，4 台 BrainAmp，每被试 1 session / 1 run，每类 150 trials（共 300 trials/被试）。
**实验范式**: 2-class haptic MI（左手 vs 右手），被试自选动作（手指屈曲/抓握等）。Trial：0-3s 注视十字，3-10s 箭头提示期间 MI（7s window），无反馈。
**预处理**: 高通滤波（10s 时间常数），离线 CAR 重参考。
**数据清洗**: **未执行任何伪影剔除或校正**。
**数据格式与单位**: EEGLAB `.set/.fdt`，µV，Zenodo 托管（CC BY 4.0），~7.8 GB。
**已知问题**: (1) 通道以数字 "1"-"128" 命名，无标准 10-20 名称；(2) 电极 65-128 头模型信息不完整。
**与本项目关系**: 128 通道与本项目通道配置一致，可直接用于预训练。粗粒度手部 MI。

---

## Lee2019_MI
**论文**: EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI illiteracy (GigaScience, 2019)
**DOI**: 10.1093/gigascience/giz002
**PDF**: `references/data/Lee2019_OpenBMI.pdf`（arxiv 预印本版本）
**数据集概要**: 54 名健康被试（25 女，50 右利手），62 通道 Ag/AgCl（10-05 montage），采样率 1000 Hz，BrainAmp，参考 nasion，地线 AFz。每被试 2 session（不同日），每 session 200 trials（training 100 + test 100）。另含 4ch EMG、静息态、5 种伪迹模板。
**实验范式**: 2-class 左/右手抓握想象。Trial：3s 注视十字 → 4s MI（箭头提示）→ 6±1.5s 休息。每 phase 左右各 50 trials。Test phase 有实时反馈。
**预处理**: 原始发布，未做预处理。
**数据清洗**: 未对发布数据执行 artifact rejection。提供了伪迹模板（眨眼、眼动、咬牙等）供用户自行处理。
**数据格式与单位**: `.mat`，µV。另有 GDF/EDF/SET 等格式。
**已知问题**: (1) MI BCI illiteracy 率 53.7%（70% 阈值），平均准确率仅 71.1±15%；(2) 仅含左右手抓握 MI。
**与本项目关系**: 最大 MI 数据集之一（54 被试），可作为预训练数据源。62 通道与本项目 128ch 不同但 ACPE 可适配。

---

## Ofner2017
**论文**: Upper limb movements can be decoded from the time-domain of low-frequency EEG (PLOS ONE, 2017)
**DOI**: 10.1371/journal.pone.0182578
**PDF**: `references/data/Ofner2017_upper_limb_MI.pdf`
**数据集概要**: 15 名健康被试（9 女，14 右利手，22-40 岁），61 通道 EEG（g.tec 有源电极，右乳突参考），采样率 512 Hz。每被试 2 session（不同天，ME + MI 各一），每 session 10 run × 42 trials，公开于 BNCI Horizon 2020 + Zenodo。
**实验范式**: 6 类上肢运动 + 休息：肘屈/伸、前臂旋前/旋后、手张/握。每类每 session 60 trials。持续性（非重复性）动作。
**预处理**: 原始 GDF 发布（512 Hz）。论文分析使用 0.3-3 Hz Butterworth + CAR + 下采样 256 Hz。
**数据清洗**: ±200 µV 阈值 + 异常联合概率/峰度剔除（5 SD）。噪声通道移除（平均 1.4 通道/被试）。
**数据格式与单位**: GDF 格式，µV。300 个文件，~27.3 GB。
**已知问题**: (1) MOABB 中可能存在双重缩放 bug（`raw._data *= 1e-6` 在 MNE GDF reader 已转 V 后再乘，FIXME 未解决）；(2) MI 准确率显著低于 ME。
**与本项目关系**: **上肢运动 MI/ME，是所有预训练数据集中与单指 MI 最接近的**。包含手张/握动作，同属上肢精细运动 BCI 领域，但粒度仍为肢体级（非单指级）。

---

## Stieger2021
**论文**: Continuous sensorimotor rhythm based brain computer interface learning in a large population (Nature Scientific Data, 2021)
**DOI**: 10.1038/s41597-021-00883-1
**PDF**: `references/data/s41597-021-00883-1.pdf`
**数据集概要**: 62 名健康成年被试，62 通道 EEG（实际使用 60 通道），采样率 1000 Hz，每人 7-11 个 session，共 598 session，269,099 trials，>600 小时录制。
**实验范式**: SMR-based online BCI 连续控制。3 种任务：LR（左/右手 MI，1D 水平控制）、UD（上/下调节，1D 垂直控制）、2D（同时控制）。每 session ~450 trials，trial 含 2s 间隔 + 2s 目标 + ≤6s 反馈控制。被试分为 MBSR 干预组/对照组。
**预处理**: 原始 raw EEG 发布，未经离线预处理。
**数据清洗**: 自动标注：(1) 8-30 Hz 后方差 z>5 的通道标记为噪声；(2) ±100 µV 的 trial 标记为伪迹。**仅标记，未修改数据**。3% session 有 >4 噪声通道，3% session >5% trial 含伪迹。
**数据格式与单位**: `.mat`，µV。每被试每 session 一个文件。
**已知问题**: (1) 部分被试个体电极位置未记录；(2) 数据托管在 Figshare (AWS S3 eu-west-1)，从中国大陆下载建议 VPN 连西欧节点（英国/德国/法国）；(3) 已下载 23/62 被试（25,020 segments，增量脚本 `preprocess_stieger_incremental.py` / GUI 版 `preprocess_stieger_gui.py` 支持断点续传）。
**与本项目关系**: 规模最大的 MI 数据集（>600h），但任务为粗粒度手/方向级别。纵向多 session 设计对预训练有价值。

---

## Schirrmeister2017
**论文**: Deep Learning with Convolutional Neural Networks for Brain Mapping and Decoding of Movement-Related Information from the Human EEG (Human Brain Mapping, 2017)
**DOI**: 10.1002/hbm.23730
**PDF**: `references/data/Schirrmeister2017_deep_learning_EEG.pdf`（arxiv 版本）
**数据集概要**: 本文使用两个数据集。(1) BCI Competition IV 2a（同 BNCI2014_001）。(2) **High-Gamma Dataset (HGD)**（新采集）：20 名健康被试（9 女，4 左利手，27.5±3.2 岁），128 通道 EEG（WaveGuard, ANT），原始 5 kHz 降采样至 250 Hz，每被试 ~1000 trials，13 runs。
**实验范式**: 4-class 运动执行（非想象）：左手连续手指敲击、右手手指敲击、双脚趾握紧、静息。每 trial 4s，trial 间 3-4s 随机。
**预处理**: 最小化预处理支持 end-to-end 学习。HGD 降采样 250 Hz；可选 4 Hz 高通 + exponential moving standardization。
**数据清洗**: 仅移除任意通道 >±800 µV 的 trial。HGD 使用运动皮层 44 通道子集效果最佳（全 128ch 反而下降）。
**数据格式与单位**: BCI2000 格式，µV。Braindecode 库公开。
**已知问题**: (1) 下载不完整——仅 5/14 被试（gin.g-node.org 限速）；(2) 44ch 子集优于 128ch 的发现对通道选择有参考价值。
**与本项目关系**: 128 通道配置与本项目一致。4-class 运动执行含手指敲击，是粗粒度运动中最接近手指级别的任务。Shallow/Deep ConvNet 是 EEGNet 的重要前身。

---

## Shin2017A
**论文**: Open Access Dataset for EEG+NIRS Single-Trial Classification (IEEE Trans. Neural Syst. Rehabil. Eng., 2017)
**DOI**: 10.1109/TNSRE.2016.2628057
**PDF**: `references/data/Shin2017_EEG_NIRS_open_access.pdf` — **注意：PDF 内容错误**，实际包含 Rovini et al. 2017 帕金森综述论文（DOI: 10.3389/fnins.2017.00555），非 Shin 论文
**数据集概要**: 29 名健康右利手被试（平均 28.5 岁），30 通道 EEG（BrainAmp, 10-5 系统，linked mastoids 参考）+ 36 通道 fNIRS。EEG 原始 1000 Hz 降采样至 200 Hz。每被试 3 session，附 VEOG/HEOG/ECG/呼吸带。
**实验范式**: (A) 左/右手 MI（想象动觉性握拳，1 Hz 节奏）；(B) 心算减法 vs 静息。每 trial：2s 视觉提示 → 10s 任务 → 15-17s 休息。每 session 20 trials，3 sessions 共 60 trials/被试。
**预处理**: 带通 0.5-50 Hz + 50 Hz notch + ICA EOG 去除 + 从 1000 Hz 降采样至 200 Hz。
**数据清洗**: ICA-based EOG artifact rejection。
**数据格式与单位**: `.mat`（BBCI 格式），µV。GPL 3.0。
**已知问题**: (1) PDF 文件内容不匹配（见上）；(2) 需 `accept=True` 下载。
**与本项目关系**: 采样率 200 Hz 与 CBraMod 原生一致，无需重采样。左/右手 MI，粗粒度。30 通道较少。

---

## BNCI2014_001
**论文**: Review of the BCI Competition IV (Frontiers in Neuroscience, 2012)
**DOI**: 10.3389/fnins.2012.00055
**PDF**: `references/data/Tangermann2012_BCI_Competition_IV.pdf`
**数据集概要**: 9 名健康被试，22 导 EEG + 3 导 EOG，采样率 250 Hz，Ag/AgCl 电极（10-20 系统，3.5cm 间距），左乳突参考，右乳突地线。每被试 2 session（不同日），每 session 6 runs × 48 trials = 288 trials/session，共 576 trials/被试。
**实验范式**: 经典 4-class MI：左手 (1)、右手 (2)、双脚 (3)、舌头 (4)。Trial：0s 注视十字 + 警告音 → 2s 箭头 cue（1.25s）→ 2-6s MI（无反馈）→ 6s 结束 + 休息。Session 前录制 ~5min EOG 基线。
**预处理**: 带通 0.5-100 Hz + 50 Hz notch。
**数据清洗**: 专家 visual inspection，含伪迹 trial 标记为 event type 1023（`h.ArtifactSelection` 字段）。3 导 EOG 仅供伪迹去除，不得用于分类。
**数据格式与单位**: GDF 格式，µV（EEG）/ mV（EOG）。另有 .mat 版本。
**已知问题**: (1) 被试 A04 的 training session EOG 基线因技术问题不完整；(2) 竞赛冠军 FBCSP 方法 mean kappa=0.57。
**与本项目关系**: MI-BCI 领域最经典的基准数据集之一。22 通道与本项目 128ch 差异大。4-class 粗粒度 MI。

---

## BNCI2015_004
**论文**: Scherer et al., "Individually Adapted Imagery Improves Brain-Computer Interface Performance in End-Users with Disability" (PLoS ONE, 2015)
**DOI**: 10.1371/journal.pone.0123727
**PDF**: 无本地 PDF（可从 PLOS ONE 开放获取下载）
**数据集概要**: 9 名被试（5 SCI + 4 stroke，7 女 2 男，20-57 岁），30 通道 EEG（10-20 系统），采样率 256 Hz，左右乳突参考。每被试 2 session（间隔 ≥5 天），每 session 8 runs × 25 trials。
**实验范式**: 5 类心理任务（非纯运动想象）：(1) 词语联想 (WORD)；(2) 心算减法 (SUB)；(3) 空间导航 (NAV)；(4) 右手 MI (HAND)——想象挤压球；(5) 双脚 MI (FEET)。每 trial：0s 注视十字 → 3s 听觉提示 → 3-10s 想象（7s 窗口）→ 休息 2.5-3.5s。
**预处理**: 硬件滤波 0.5-100 Hz + 50 Hz notch。
**数据清洗**: 专家 visual inspection 排除肌电/眼动污染 trial。噪声通道移除（因人而异）。伪迹排除率差异大（7-57 trials/天）。
**数据格式与单位**: `.mat`，µV，MOABB 加载时 ×1e-6 转 V。CC BY-NC-ND 4.0。
**已知问题**: (1) 所有被试为 SCI/stroke 患者，非健康人群；(2) 经典 hand vs feet MI 准确率比健康人低 ~15%；(3) 被试间变异大。
**与本项目关系**: 含 2 类 MI + 3 类认知任务，为混合范式。临床人群数据可评估模型鲁棒性。30ch / 256 Hz。

---

## PhysionetMI
**论文**: Schalk et al., "BCI2000: A General-Purpose Brain-Computer Interface (BCI) System" (IEEE Trans. Biomed. Eng., 2004)
**DOI**: 10.1109/TBME.2004.827072
**PDF**: 无本地 PDF（IEEE 付费墙）
**数据集概要**: 109 名被试，64 通道 EEG（10-10 系统），采样率 160 Hz，BCI2000 系统。每人 14 runs（2 基线 + 12 任务），共 1500+ 段录制，3.4 GB。
**实验范式**: Run 1-2 静息基线（睁/闭眼各 1min）。Run 3/7/11 执行左/右手握拳；Run 4/8/12 想象左/右手握拳；Run 5/9/13 执行双拳/双脚；Run 6/10/14 想象双拳/双脚。每任务 run ~2min，含 ~15 rest trials + 7-8 T1 + 7-8 T2 trials。
**预处理**: 原始 raw 发布，无任何预处理。
**数据清洗**: 原始发布无 artifact rejection。后续 curation（Hinss et al. 2024）标注 bad trials（8-30 Hz ±100 µV），并剔除 6 名异常被试（S038/S088/S089/S092/S100/S104）。
**数据格式与单位**: EDF+ 格式，µV（EDF header 校准）。
**已知问题**: (1) **采样率 160 Hz 需上采样至 200 Hz**（CBraMod 要求）；(2) S088 采样率为 128 Hz；(3) 6 名被试标注异常需排除。
**与本项目关系**: MI-BCI 最广泛使用的基准（109 被试）。粗粒度手/脚 MI。160 Hz 采样率是唯一需上采样的数据集。

---

## 汇总对比

| 数据集 | 被试 | 通道 | 采样率 | MI 类型 | 总时长 (h) | 预训练 Segments | 优先级 |
|--------|:---:|:---:|:---:|--------|:---:|:---:|:---:|
| Lee2019_MI | 54 | 62 | 1000 Hz | 左/右手抓握 | 44.1 | 3,264 | ★★★ |
| Stieger2021 | 62 (23 downloaded) | 60 | 1000 Hz | 左/右/上/下 SMR | 706.0 | 25,020 | ★★★ |
| PhysionetMI | 109 | 64 | 160 Hz | 左/右拳 + 双拳/双脚 | 22.7 | 1,000 | ★★★ |
| Cho2017 | 52 | 64 | 512 Hz | 左/右手 | 20.2 | 2,416 | ★★ |
| Schirrmeister2017 | 14 | 128 | 500 Hz | 4-class 手指敲击+脚 | 14.3 | 927 | ★★ |
| GrosseWentrup2009 | 10 | 128 | 500 Hz | 左/右手 haptic MI | 8.7 | 910 | ★★ |
| Ofner2017 | 15 | 61 | 512 Hz | 6-class 上肢 MI/ME | 13.6 | 1,363 | ★★ |
| BNCI2015_004 | 9 | 30 | 256 Hz | 5-class 混合心理任务 | 14.0 | 1,634 | ★ |
| Shin2017A | 29 | 30 | 200 Hz | 左/右手 MI + 心算 | 14.5 | 1,513 | ★ |
| BNCI2014_001 | 9 | 22 | 250 Hz | 4-class MI（经典 BCI-IV 2a） | 11.6 | 1,296 | ★ |
| **总计** | **363** | | | | **869.7** | **39,343** | |

## 关键发现与勘误

### PDF 内容错误（需重新下载）

1. **`GrosseWentrup2009_multiclass_CSP.pdf`** — 实际包含 Ranji et al. 2009 心肌损伤论文，与 EEG/BCI 完全无关
2. **`Shin2017_EEG_NIRS_open_access.pdf`** — 实际包含 Rovini et al. 2017 帕金森可穿戴传感器综述

### DOI 勘误

| 数据集 | 此前引用的 DOI | 正确 DOI |
|--------|---------------|----------|
| GrosseWentrup2009 | ~~10.1109/TBME.2008.2006029~~ | **10.1109/TBME.2008.2009768** |
| Shin2017A | ~~10.3389/fnins.2017.00555~~ | **10.1109/TNSRE.2016.2628057** |
| BNCI2015_004 | N/A（此前认为无论文） | **10.1371/journal.pone.0123727** |

### 数据单位问题

- **Cho2017**: MOABB 加载器假设 `.mat` 为 µV 并乘 1e-6 转 V，但实际数据可能为 nV，需额外 ÷1000 修正（`dataset_metadata.json` 中 `to_uV_factor: 1e3`）
- **Ofner2017**: MOABB PR #700 添加 `raw._data *= 1e-6`，但 MNE GDF reader 可能已转 V，存在**双重缩放风险**（FIXME 未解决）
- 其他数据集单位处理正确

### 下载完整性

| 数据集 | 已下载被试 | 总被试 | 备注 |
|--------|:---:|:---:|------|
| Schirrmeister2017 | 5 | 14 | gin.g-node.org 限速 |
| Stieger2021 | 23 | 62 | 增量下载中，支持断点续传 |
| Weibo2014 | 部分 | 10 | zip 解压错误，已排除 |
| Dreyer2023 | 0 | 87 | OSF 404，已排除 |

### 与本项目 Finger MI 的 Domain Gap

所有 10 个预训练数据集均为**粗粒度肢体 MI**（手/脚级别），本项目解码**单指运动**。最接近的数据集是：
1. **Ofner2017** — 上肢 6 类运动（含手张/握），但仍为肢体级
2. **Schirrmeister2017 HGD** — 含手指敲击执行（非想象），128ch 配置一致
3. **Cho2017** — MI 想象内容为四指触碰拇指，但分类仍为左/右手二分类

这一 domain gap 是 further pre-training 未能提升下游性能的主要原因之一（详见 `further_pretraining_analysis.md`）。

## 文件索引

| 文件 | 说明 |
|------|------|
| `references/data/*.pdf` | 各数据集原始论文 PDF（8 个文件，2 个内容有误） |
| `results/pretraining/audit_report.json` | 数据集审计报告 |
| `results/pretraining/preprocess_report.json` | LMDB 预处理报告 |
| `scripts/pretraining/dataset_metadata.json` | 数据集单位元数据（含 `to_uV_factor` 修正） |
| `paper/analysis/further_pretraining_analysis.md` | Further pre-training 下游评估报告 |
| `paper/analysis/mi_dataset_survey.md` | 数据集技术概览（需同步更新 DOI 勘误） |
