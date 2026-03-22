# 运动想象 EEG 公开数据集调查报告

> **数据来源**: `results/pretraining/audit_report.json` (审计时间: 2026-03-21T14:03:44)
> **下载脚本**: `D:/data/motion_imagination_datasets/download.py` (MOABB API)
> **存储位置**: `D:/data/motion_imagination_datasets/`

## 1. 数据集总览

通过 MOABB (Mother of All BCI Benchmarks) 框架下载了 13 个运动想象 EEG 数据集，其中 10 个审计成功、2 个报错、3 个被排除。

### 1.1 成功审计的数据集

| # | 数据集 | 优先级 | 被试数 | EEG 通道 | 采样率 | 总时长 (h) | 备注 |
|:---:|--------|:---:|:---:|:---:|:---:|:---:|------|
| 1 | Lee2019_MI | ★★★ | 54 | 62 | 1000 Hz | 44.06 | 最大 MI 数据集 |
| 2 | Stieger2021 | ★★★ | 62 | 60 | 1000 Hz | 706.04 | 纵向多会话 |
| 3 | PhysionetMI | ★★★ | 109 | 64 | 160 Hz | 22.71 | 160Hz 需上采样 |
| 4 | Cho2017 | ★★ | 52 | 64 | 512 Hz | 20.24 | — |
| 5 | Schirrmeister2017 | ★★ | 14 | 128 | 500 Hz | 14.33 | 排除 EOG/EMG |
| 6 | GrosseWentrup2009 | ★★ | 10 | 128 | 500 Hz | 8.66 | — |
| 7 | Ofner2017 | ★★ | 15 | 61 | 512 Hz | 13.57 | 上肢运动 |
| 8 | BNCI2015_004 | ★ | 9 | 30 | 256 Hz | 13.96 | — |
| 9 | Shin2017A | ★ | 29 | 30 | 200 Hz | 14.52 | 需分离 EEG/fNIRS |
| 10 | BNCI2014_001 | ★ | 9 | 22 | 250 Hz | 11.61 | 经典 BCI-IV 2a |
| | **总计** | | **363** | | | **869.70** | |

### 1.2 报错数据集

| 数据集 | 被试数 | 错误 |
|--------|:---:|------|
| Weibo2014 | 10 | `FileNotFoundError: MNE-weibo-2014/data0.zip.unzip` — zip 解压路径问题 |
| Dreyer2023 | 87 | `HTTPError: 404` — OSF 下载链接失效 (`osf.io/download/67c9abecc1b99765d8bb36b0/`) |

### 1.3 排除的数据集

| 数据集 | 排除原因 |
|--------|----------|
| BNCI2014_004 | 仅 3 通道，ACPE 效果极差 |
| Zhou2016 | 仅 4 被试 14 通道，数据量过小 |
| AlexMI | 仅 8 被试 16 通道，数据量极小 |

## 2. 数据质量分析

### 2.1 信号幅值统计 (µV)

| 数据集 | 最大绝对幅值 | 平均最大绝对幅值 | 备注 |
|--------|:---:|:---:|------|
| BNCI2014_001 | 84.3 | 60.5 | 干净 |
| BNCI2015_004 | 100.0 | 88.7 | 干净 |
| Cho2017 | **171,895.0** | **171,895.0** | **MOABB 单位错误**：加载器假设 .mat 为 µV，实际为 nV，需 ÷1000 |
| GrosseWentrup2009 | 418.0 | 418.0 | 正常范围 |
| Lee2019_MI | 493.0 | 480.4 | 正常范围 |
| Ofner2017 | 1,377.4 | 846.8 | 偏高，含 EOG/misc 通道 |
| PhysionetMI | 601.0 | 348.7 | 正常范围 |
| Schirrmeister2017 | 409.7 | 395.2 | 正常范围 |
| Shin2017A | 318.1 | 265.9 | 正常范围 |
| Stieger2021 | **4,489.3** | 630.9 | 少数异常值偏高 |

**关键发现**：
- **Cho2017** 的 MOABB 加载器存在单位错误（详见 `scripts/pretraining/dataset_metadata.json` 中的 `to_uV_factor` 修正）
- **Stieger2021** 存在极端振幅异常值 (4,489 µV)，预处理时需通过 500 µV 阈值过滤
- 大多数数据集信号在 100-600 µV 范围内

### 2.2 非 EEG 通道

| 数据集 | 非 EEG 通道 |
|--------|-------------|
| Lee2019_MI | EMG1-4 (emg), STI 014 (stim) |
| Cho2017 | EMG1-4 (emg), Stim (stim) |
| Schirrmeister2017 | 无（纯 EEG 128ch） |
| GrosseWentrup2009 | STIM (stim) |
| Ofner2017 | EOG ×3, 手指传感器 ×18, STIM |
| Shin2017A | VEOG, HEOG (eog), Stim |
| BNCI2014_001 | EOG1-3, stim |
| 其他 | stim 通道 |

预处理时需剥离非 EEG 通道，仅保留 EEG 用于 CBraMod 输入。

### 2.3 通道名称标准化

| 数据集 | 通道命名 | 问题 |
|--------|----------|------|
| GrosseWentrup2009 | 数字编号 ("1"-"128") | **无标准 10-20 名称**，需外部 mapping |
| 其他 | 标准 10-20/10-10 命名 | 无问题 |

## 3. 数据预处理状态

### 3.1 MOABB 默认行为

MOABB 通过各数据集的 `_get_single_subject_data()` 方法加载原始数据为 MNE `Raw` 对象。MOABB **不做任何预处理**——返回的是原始采集信号，保留原始采样率和通道。用户需自行处理滤波、重采样、分段等。

### 3.2 本项目预处理管线

> **来源**: `results/pretraining/preprocess_report.json` + `scripts/pretraining/preprocess_mi_datasets.py`

| 步骤 | 参数 |
|------|------|
| 单位归一化 | 通过 `dataset_metadata.json` 中 `to_uV_factor` 转换到 µV |
| 重采样 | → 200 Hz |
| 分段 | 30 秒连续段，shape = `(n_channels, 30, 200)` |
| 伪影剔除 | 跳过 run 前 5s + 平均绝对幅值 > 500 µV 过滤 |
| 存储 | LMDB 格式，pickle 序列化 |

### 3.3 LMDB 预处理结果

> **来源**: `D:/data/motion_imagination_datasets/lmdb_pretrain/` 目录

成功预处理为 LMDB 的数据集（10 个）：

| LMDB 目录 | 原始数据集 |
|-----------|-----------|
| BNCI2014_001_pretrain | BNCI2014_001 |
| BNCI2015_004_pretrain | BNCI2015_004 |
| Cho2017_pretrain | Cho2017 |
| GrosseWentrup2009_pretrain | GrosseWentrup2009 |
| Lee2019_MI_pretrain | Lee2019_MI |
| Ofner2017_pretrain | Ofner2017 |
| PhysionetMI_pretrain | PhysionetMI |
| Schirrmeister2017_pretrain | Schirrmeister2017 |
| Shin2017A_pretrain | Shin2017A |
| Stieger2021_pretrain | Stieger2021 |

## 4. 数据集论文引用

| 数据集 | 论文/DOI |
|--------|----------|
| Lee2019_MI | Lee et al. (2019). "EEG dataset and OpenBMI toolbox for three BCI paradigms". doi: 10.1093/gigascience/giz002 |
| Stieger2021 | Stieger et al. (2021). "Continuous sensorimotor rhythm based brain computer interface learning in a large population". doi: 10.1038/s41597-021-00883-1 |
| PhysionetMI | Schalk et al. (2004). "BCI2000: A general-purpose brain-computer interface system". doi: 10.1109/TBME.2004.827072 |
| Cho2017 | Cho et al. (2017). "EEG datasets for motor imagery brain-computer interface". doi: 10.1093/gigascience/gix034 |
| Schirrmeister2017 | Schirrmeister et al. (2017). "Deep learning with convolutional neural networks for EEG decoding and visualization". doi: 10.1002/hbm.23730 |
| GrosseWentrup2009 | Grosse-Wentrup & Buss (2009). "Beamforming in noninvasive brain-computer interfaces". doi: 10.1109/TBME.2008.2009768 |
| Ofner2017 | Ofner et al. (2017). "Upper limb movements can be decoded from the time-domain of low-frequency EEG". doi: 10.1371/journal.pone.0182578 |
| BNCI2015_004 | Scherer et al. (2015). "Individually adapted imagery improves brain-computer interface performance in end-users with disability". doi: 10.1371/journal.pone.0123727 |
| Shin2017A | Shin et al. (2017). "Open access dataset for EEG+NIRS single-trial classification". doi: 10.1109/TNSRE.2016.2628057 |
| BNCI2014_001 | Tangermann et al. (2012). "Review of the BCI Competition IV". doi: 10.3389/fnins.2012.00055 |
| Weibo2014 | Yi et al. (2014). "Evaluation of EEG oscillatory patterns and cognitive process during simple and compound limb motor imagery" |
| Dreyer2023 | Dreyer et al. (2023). — OSF 链接已失效 |

## 5. 数据下载完整性

### 5.1 不完整的下载

| 数据集 | 已下载 | 总被试 | 缺失原因 |
|--------|:---:|:---:|----------|
| Schirrmeister2017 | 5 | 14 | gin.g-node.org 下载速度极慢 (20-300 kB/s) |
| Stieger2021 | 14 | 62 | 同上 |
| Weibo2014 | 部分 | 10 | zip 解压路径错误 |
| Dreyer2023 | 0 | 87 | OSF 404 链接失效 |

### 5.2 补全后预估增量

- **Stieger2021**: +48 被试 → 预计 +~55,000 segments
- **Schirrmeister2017**: +9 被试 → 预计 +~1,000 segments
- 总增量约 **56,000 segments**，当前 30,282 segments 可增加 ~185%

## 6. 参考文件索引

| 文件 | 说明 |
|------|------|
| `D:/data/motion_imagination_datasets/download.py` | MOABB 数据集批量下载脚本 |
| `D:/data/motion_imagination_datasets/运动想象EEG公开数据集完整指南.pdf` | 数据集选择参考指南 |
| `results/pretraining/audit_report.json` | 原始审计数据 (JSON) |
| `results/pretraining/preprocess_report.json` | LMDB 预处理报告 |
| `scripts/pretraining/audit_datasets.py` | 审计脚本 |
| `scripts/pretraining/preprocess_mi_datasets.py` | 预处理脚本 |
| `scripts/pretraining/dataset_metadata.json` | 数据集单位元数据 |
| `paper/analysis/further_pretraining_analysis.md` | Further pre-training 下游评估报告 |
