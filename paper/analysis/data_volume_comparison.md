# 数据采集量三级对比分析

> **数据来源**:
> - Finger EEG: `data/README.txt` (21 被试, 128ch, 1024 Hz, ~64h 总录制)
> - MI Pretrain: `results/pretraining/audit_report.json` (审计时间 2026-03-21) + `paper/analysis/mi_dataset_survey.md`
> - TUEG Pretrain: `docs/experiment_plan_v1.md` (CBraMod 论文 Table 5: 1,109,545 样本 × 30s)

## 1. 度量定义

| 度量 | 公式 | 含义 |
|------|------|------|
| **Total Time** | 原始录制总时长 | 纯时间维度的数据规模 |
| **Channel-Time** | Total Time × 通道数 | 引入空间维度，反映"通道-小时"数据量 |
| **Channel-Frame** | Channel-Time × 采样率 | 引入时间分辨率，反映总采样点数（最精确的数据量度量） |

Channel-Frame @200Hz 表示统一重采样到 CBraMod 输入采样率后的 channel-frame，是**模型实际看到的数据量**的最公平比较。

## 2. 总览

| 指标 | Finger EEG | MI Pretrain (10 datasets) | TUEG Pretrain |
|------|:---:|:---:|:---:|
| **Total Time** | 64 h | 870 h | 9,246 h |
| **通道数** | 128 | 22–128 (混合) | 19 |
| **采样率** | 1,024 Hz | 160–1,000 Hz | 200 Hz |
| **Channel-Time** | 8,192 ch-h | 52,723 ch-h | 175,678 ch-h |
| **Channel-Frame (raw)** | 30.2 G samples | 173.3 G samples | 126.5 G samples |
| **Channel-Frame @200Hz** | 5.9 G samples | 38.0 G samples | 126.5 G samples |

## 3. 倍率关系

### 3.1 以 Finger EEG 为基准 (Finger = 1×)

| 指标 | MI Pretrain | TUEG Pretrain |
|------|:---:|:---:|
| Total Time | 13.6× | **144×** |
| Channel-Time | 6.4× | **21×** |
| Channel-Frame @200Hz | 6.4× | **21×** |

### 3.2 以 TUEG 为基准 (TUEG = 1×)

| 指标 | Finger EEG | MI Pretrain |
|------|:---:|:---:|
| Total Time | 1/144 | 1/11 |
| Channel-Time | 1/21 | 1/3.3 |
| Channel-Frame @200Hz | 1/21 | 1/3.3 |

## 4. MI 数据集明细

Stieger2021 独占 MI 数据集 **81% 的时长、80% 的 channel-time、88% 的 raw channel-frame**。

| Dataset | Hours | Ch | Fs (Hz) | Ch-Time (ch-h) | Ch-Frame (G) | 占比 |
|---------|------:|---:|--------:|-----------:|----------:|-----:|
| Stieger2021 | 706.04 | 60 | 1,000 | 42,362 | 152.50 | 88.0% |
| Lee2019_MI | 44.06 | 62 | 1,000 | 2,732 | 9.83 | 5.7% |
| Schirrmeister2017 | 14.33 | 128 | 500 | 1,834 | 3.30 | 1.9% |
| Cho2017 | 20.24 | 64 | 512 | 1,295 | 2.39 | 1.4% |
| GrosseWentrup2009 | 8.66 | 128 | 500 | 1,109 | 2.00 | 1.2% |
| Ofner2017 | 13.57 | 61 | 512 | 828 | 1.53 | 0.9% |
| PhysionetMI | 22.71 | 64 | 160 | 1,453 | 0.84 | 0.5% |
| BNCI2015_004 | 13.96 | 30 | 256 | 419 | 0.39 | 0.2% |
| Shin2017A | 14.52 | 30 | 200 | 436 | 0.31 | 0.2% |
| BNCI2014_001 | 11.61 | 22 | 250 | 255 | 0.23 | 0.1% |
| **Total** | **869.70** | — | — | **52,723** | **173.32** | **100%** |

## 5. 关键发现

### 5.1 度量层次揭示不同的"大小"画面

三个度量层次对排序和比例关系有本质影响：

- **Total Time**: TUEG 以 9,246h 绝对碾压（Finger 的 144 倍）
- **Channel-Time**: Finger 的 128ch 远超 TUEG 的 19ch，差距从 144× 压缩到 **21×**
- **Channel-Frame (raw)**: MI 数据集 (173.3G) 竟然**超过** TUEG (126.5G)——因为 Stieger2021 以 1000Hz × 60ch 贡献了 152.5G samples。但这是采样率膨胀造成的假象

### 5.2 @200Hz 标准化是最公平的比较

CBraMod 统一在 200Hz 处理输入。标准化后 Finger:MI:TUEG = **1 : 6.4 : 21**，消除了采样率差异造成的度量偏差。

### 5.3 Further pretraining 负迁移的数据量解释

MI 数据量仅为 TUEG 的 **1/3.3**（@200Hz channel-frame），处于一个尴尬的中间地带：
- **不够多**：不足以建立鲁棒的 MI-specific 表征
- **不够少**：足以破坏 TUEG 预训练学到的通用 EEG 特征

叠加 domain gap（粗粒度肢体 MI → 细粒度 finger MI），进一步训练更长时间反而加剧负迁移（V2 loss ↓39% 但下游 delta 从 -0.75% 恶化到 -1.38%），详见 `further_pretraining_analysis.md`。

### 5.4 Stieger2021 的主导地位

Stieger2021 以 706h (81%) 主导 MI 数据集，其 SMR-based BCI 连续控制任务（左/右/上/下）与 finger MI 任务的 domain gap 尤其大。这意味着 further pretraining 的 ~79% 训练信号来自一个与下游任务距离最远的数据集。

## 6. 参考文件

| 文件 | 说明 |
|------|------|
| `data/README.txt` | Finger EEG 数据集元信息 |
| `docs/experiment_plan_v1.md` | CBraMod TUEG 预训练配置 (Table 5) |
| `paper/analysis/mi_dataset_survey.md` | 10 个 MI 数据集技术概览 |
| `paper/analysis/further_pretraining_data.md` | MI 数据集详细调查 |
| `paper/analysis/further_pretraining_analysis.md` | Further pretraining 下游评估 |
| `results/pretraining/audit_report.json` | 数据集审计原始数据 |
