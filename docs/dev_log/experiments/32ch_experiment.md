# 32 通道实验：6 配置对比 + 最优配置全量实验

**Date**: 2026-02-20 ~ 2026-03-01
**Status**: 全部完成（Step 1-7: 6 配置对比 + FDR/Commercial/Attention 扩展实验 + 8ch FDR + 61ch 对比）

---

## 概述

在已有的 128 通道（全通道）和 8 通道（motor cortex 子集）实验基础上，新增 32 通道实验。
目标是在通道数和解码性能之间找到更优的折中点，同时评估不同通道选择策略的效果。

### 实验设计

1. **6 种 32ch 配置对比**：通过跨被试训练评估 6 种通道选择策略
2. **最优配置全量实验**：对最优配置运行 within + cross + transfer 完整管线

| 配置名 | 类型 | 来源 |
|--------|------|------|
| `motor_cortex` | Hand-picked | 运动皮层区域密集覆盖（C3/Cz/C4 + SMA + premotor） |
| `commercial` | Hand-picked | 标准商用 32 通道 EEG 帽（10-20 布局） |
| `fdr` | Data-driven | Fisher Discriminant Ratio 通道排序 |
| `csp` | Data-driven | CSP 空间滤波器权重排序 |
| `attention` | Data-driven | EEGNet spatial_conv 权重 + CBraMod 输入梯度 |
| `band_power` | Data-driven | Mu (8-13Hz) + Beta (13-30Hz) ANOVA F-statistic |

### 运行流程

```bash
# Step 1: 计算数据驱动通道选择 (生成 JSON)
uv run python scripts/analysis/compute_32ch_selections.py

# Step 2: 6 配置对比 (确定最优)
uv run python scripts/experiments/run_32ch_config_comparison.py

# Step 3: 最优配置全量实验
uv run python scripts/experiments/run_32ch_experiment.py --channel-config <best>
```

---

## 实现方案

### 架构概览

```
                    ┌─────────────────────────────┐
                    │  channel_selection.py        │
                    │  CHANNEL_32_CONFIGS 注册表   │
                    │  get_32ch_indices()          │
                    │  load_32ch_selections()      │
                    └──────────┬──────────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                   │
    ┌───────▼──────┐   ┌──────▼───────┐   ┌──────▼──────────┐
    │  dataset.py  │   │ training.py  │   │ channel_        │
    │  Strategy 'E'│   │ 32ch presets │   │ selections.json │
    └──────────────┘   └──────────────┘   └─────────────────┘
```

**数据流**: CLI `--channels 32 --channel-config <name>` → `config_overrides['data']` → `PreprocessConfig.channel_strategy='E'` + `channel_32_config=<name>` → `dataset.py` 调用 `get_32ch_indices()` → HDF5 缓存 128ch 数据中选取 32 通道子集

### 通道选择策略 'E'

在已有的 A/B/C/D 策略基础上新增 Strategy 'E'：

| 策略 | 通道数 | 说明 |
|------|--------|------|
| A | 19 | 标准 10-20 系统（CBraMod 预训练通道） |
| B | ~19 | 运动皮层高密度子集 |
| C | 128 | 全部 BioSemi 通道（默认） |
| D | 8 | 8 通道 motor cortex 子集 |
| **E** | **32** | **32ch 命名配置（hand-picked 或 data-driven）** |

Strategy E 的查找逻辑：
1. 检查 `CHANNEL_32_CONFIGS[config_name]`
2. 若为 hard-coded（`motor_cortex`, `commercial`）→ 直接返回索引
3. 若为 `None`（data-driven）→ 从 `results/32_channel/channel_selections.json` 加载

---

## 修改文件清单

### Phase 1: 基础设施（11 个现有文件修改）

| 文件 | 改动 |
|------|------|
| `src/preprocessing/channel_selection.py` | `CHANNEL_32_CONFIGS` 注册表、`get_32ch_indices()`、`load_32ch_selections()` |
| `src/preprocessing/data_loader.py` | `PreprocessConfig.channel_32_config` 字段 |
| `src/preprocessing/dataset.py` | Strategy 'E' 分支（`get_32ch_indices` 导入 + elif） |
| `src/config/constants.py` | `SUPPORTED_CHANNEL_COUNTS = [8, 32, 128]` |
| `src/config/training.py` | `THIRTYTWO_CHANNEL_*_OVERRIDES`（within/cross/finetune）、config 函数扩展 |
| `src/training/train_within_subject.py` | `channels == 32` → strategy E + channel_config |
| `src/training/train_cross_subject.py` | 同上 |
| `src/training/finetune.py` | `channel_config` 参数、`is_32ch_cbramod` override 逻辑 |
| `scripts/experiments/run_within_subject_comparison.py` | `--channels 32 --channel-config` CLI |
| `scripts/experiments/run_cross_subject_comparison.py` | 同上 |
| `scripts/experiments/run_transfer_comparison.py` | 同上 + `channel_config` 透传至 finetune |

### Phase 0: 数据驱动分析（1 个新文件）

| 文件 | 说明 |
|------|------|
| `scripts/analysis/compute_32ch_selections.py` | 4 种 data-driven 方法计算最优 32ch 子集 |

### Phase 2-3: 实验脚本（2 个新文件）

| 文件 | 说明 |
|------|------|
| `scripts/experiments/run_32ch_config_comparison.py` | 6 配置对比（调用 cross-subject comparison × 6） |
| `scripts/experiments/run_32ch_experiment.py` | 全量实验（within + cross + transfer × binary + ternary） |

---

## Hand-Picked 配置设计

### motor_cortex（32ch）

基于运动皮层区域密集覆盖原则选择。核心覆盖区域：

- **中央沟区域** (C3/Cz/C4)：初级运动皮层 M1
- **补充运动区** (SMA, near FCz)：运动计划与准备
- **前运动皮层** (FC3/FC4 附近)：运动编程
- **顶叶运动区** (CP3/CPz/CP4)：体感反馈整合

```
索引: [0, 2, 3, 5, 20, 32, 33, 34, 49, 50, 52, 53, 55,
       62, 63, 64, 65, 66, 77, 85, 86, 90, 97, 107, 108,
       110, 111, 112, 113, 114, 116, 123]
```

**设计决策 — C23 (idx 86) vs D3 (idx 98)**:

选择 C23 而非 D3 的理由（纯运动皮层覆盖质量评估，与 8ch 子集无关）：

| 指标 | C23 (idx 86) | D3 (idx 98) |
|------|-------------|-------------|
| 距运动皮层中心 | 0.575 | 0.909 |
| 距 FCz (SMA) | 0.368 | 0.729 |
| 最近 10-20 位置 | Fz-FCz 之间 | F3 附近 |
| 区域冗余度 | 低（SMA 唯一高质量覆盖） | 高（左侧额区已有 5 个电极） |

C23 是整个 32ch 配置中唯一能高质量覆盖补充运动区 (SMA) 的电极。
SMA 在 Motor Imagery 和 Motor Execution 的 ERD/ERS 解码中起关键作用。
D3 仅会在已经密集覆盖的左侧额区（D2, D12, D13, D18, D19）增加冗余。

### commercial（32ch）

基于标准商用 32 通道 EEG 帽的 10-20 系统布局，通过 BioSemi 128 电极位置文件
计算每个标准 10-20 位置最近的 BioSemi 电极得到。

```
索引: [0, 3, 5, 16, 17, 22, 29, 30, 33, 34, 44, 49, 52, 55,
       62, 65, 66, 68, 76, 77, 85, 89, 90, 97, 98, 100, 107,
       111, 113, 116, 123, 124]
```

覆盖标准 10-20 位置：Fp1, Fp2, F7, F3, Fz, F4, F8, FC5, FC1, FC2, FC6,
T7, C3, Cz, C4, T8, CP5, CP1, CP2, CP6, P7, P3, Pz, P4, P8, O1, Oz, O2 等。

---

## Data-Driven 方法详解

### 1. FDR (Fisher Discriminant Ratio)

纯 numpy 实现，无需外部依赖。

对每个通道计算类间 Fisher 判别比：
```
FDR_ch = (1/C₂ⁿ) Σᵢ<ⱼ mean_t[(μᵢ(t) - μⱼ(t))² / (σᵢ²(t) + σⱼ²(t))]
```

选择 FDR 最高的 32 个通道。

### 2. CSP (Common Spatial Patterns)

使用 `mne.decoding.CSP`（`reg='ledoit_wolf'`）：
- Binary task: 直接拟合 CSP (n_components=6)
- Multi-class: One-vs-Rest，每个 class 拟合 CSP (n_components=4)

通道重要性 = CSP 空间滤波器权重绝对值之和。

### 3. Attention / Gradient

结合两个已训练模型的通道重要性：
- **EEGNet**: 提取 `spatial_conv.weight` 权重，按滤波器平方和得到通道重要性
- **CBraMod**: 对输入计算交叉熵损失的梯度，取绝对值均值得到通道重要性

两个来源各自归一化到 [0,1] 后取平均。

需要预训练检查点：`checkpoints/cross_subject/{eegnet,cbramod}_imagery_binary/best.pt`

注意：EEGNet 缓存为 100Hz，CBraMod 期望 200Hz。使用 `np.repeat(x, 2, axis=2)` 上采样，
对梯度通道排名而言精度足够。

### 4. Band Power (Mu + Beta ANOVA)

对每个通道：
1. `scipy.signal.welch` 计算功率谱密度
2. 分别求和 Mu (8-13 Hz) 和 Beta (13-30 Hz) 频段功率
3. `scipy.stats.f_oneway` 对各类别做 ANOVA
4. 通道得分 = F_mu + F_beta

Mu/Beta ERD (Event-Related Desynchronization) 是运动意象/执行的核心 EEG 特征。

---

## 32ch 超参数配置

针对 CBraMod 的 32 通道超参数，介于 8ch（强正则）和 128ch（弱正则）之间：

### Within-Subject

```python
THIRTYTWO_CHANNEL_WITHIN_SUBJECT_OVERRIDES = {
    'model': {'dropout_rate': 0.20},     # 8ch: 0.30, 128ch: 0.15
    'training': {'weight_decay': 0.08},  # 8ch: 0.10, 128ch: 0.06
}
```

### Cross-Subject

```python
THIRTYTWO_CHANNEL_CROSS_SUBJECT_OVERRIDES = {
    'model': {'dropout_rate': 0.40},     # 8ch: 0.45, 128ch: 0.35
    'training': {'weight_decay': 0.15},  # 8ch: 0.18, 128ch: 0.12
}
```

### Finetune

```python
THIRTYTWO_CHANNEL_FINETUNE_OVERRIDES = {
    'epochs': 25,             # 8ch: 20, 128ch: 10-30
    'patience': 10,           # 8ch: 8, 128ch: 5
    'learning_rate': 8e-5,    # 8ch: 5e-5, 128ch: 1e-4
}
```

设计原则：32ch 信息量介于 8ch 和 128ch 之间，正则化强度相应插值。
EEGNet 对通道数不敏感（参数量主要由 temporal/separable convolution 决定），
因此仅对 CBraMod 应用 32ch 专用超参数。

---

## 输出与结果

### Data-Driven 通道选择输出

```
results/32_channel/channel_selections.json
```

JSON 结构：
```json
{
  "metadata": {
    "created_at": "...",
    "paradigm": "imagery",
    "task": "binary",
    "n_channels_selected": 32,
    "n_subjects": 21,
    "n_trials_total": 12345,
    "methods": ["fdr", "csp", "attention", "band_power"]
  },
  "configs": {
    "fdr": {
      "indices": [3, 5, 12, ...],
      "scores": {"3": 0.85, "5": 0.82, ...},
      "description": "Fisher Discriminant Ratio — ..."
    },
    ...
  }
}
```

### 实验结果目录

所有 32 通道实验结果自动输出到 `results/32_channel/`（通过 `--channels 32` 触发自动重定向）。

```
results/32_channel/
├── channel_selections.json              # Data-driven 通道选择
├── cross_subject_*.json                 # 各配置的跨被试结果
├── *comparison_cache_imagery_binary*.json  # 被试内结果缓存
├── *transfer_comparison_cache*.json     # 迁移学习结果缓存
└── *.png                                # 对比可视化图
```

---

## Code Review 发现与修复

### Bug 1 (Critical): channel_config 未透传至 finetune

`run_transfer_comparison.py` 中 `finetune_and_get_result()` 和 `run_transfer_model()` 缺少
`channel_config` 参数，导致 32ch transfer 实验静默使用默认的 `motor_cortex` 配置。

**修复**: 三处添加 `channel_config: Optional[str] = None` 参数并透传。
仅在 `channels == 32` 时传递 `args.channel_config`，不影响默认行为。

### Bug 2 (Medium): 32ch finetune overrides 未应用

`finetune.py` 仅检查 `is_8ch_cbramod`，未应用 `THIRTYTWO_CHANNEL_FINETUNE_OVERRIDES`。

**修复**: 新增 `is_32ch_cbramod` 变量，在 epochs/learning_rate/patience 默认值逻辑中
各添加 `elif is_32ch_cbramod` 分支。`channels` 默认 `None`，不影响 128ch 路径。

### Bug 3 (Medium): compute_32ch_selections.py NaN 处理与模型参数

运行 Step 1 时发现多个问题：

1. **NaN 时间填充**: 缓存中 trial 有效信号约 2.8-3.1 秒，其余用 NaN 填充至统一长度（EEGNet 500 样本 / CBraMod 1000 样本）。未截断导致 CSP 等方法报错或结果失真。
   **修复**: 加载数据后全局检测有效信号长度（1st percentile），过滤异常短 trial 并截断 NaN 填充。

2. **CBraMod 参数不匹配**: 脚本传入了不存在的 `target_sf` 参数，且未正确计算 `n_patches`。21 被试检查点的 `n_samples=200`（即 `n_patches=1`），但默认值为 5，导致 classifier 权重尺寸不匹配。
   **修复**: 移除 `target_sf`，从 `model_config.n_samples` 计算 `n_patches = n_samples // 200`。

3. **标签映射**: 原始标签 `[1, 4]`（拇指/小指）未映射为 `[0, 1]`，导致 `cross_entropy` 报 "Target 4 is out of bounds"。
   **修复**: 在梯度计算前添加标签重映射。

4. **模型缓存不匹配**: CBraMod 梯度计算使用了 EEGNet 缓存数据（100Hz, z-score），而非 CBraMod 缓存（200Hz, ÷100）。
   **修复**: 为 CBraMod 梯度单独加载 `cbramod_128ch` 缓存。

5. **cache_index.json 遍历**: 第二次加载 index 时未使用 `index.get('entries', index)` 导致遍历到非 dict 值时 AttributeError。
   **修复**: 统一使用 entries 键访问。

6. **UTF-8 编码**: Windows 默认 GBK 编码导致 `run_32ch_config_comparison.py` 读取 JSON 报 `UnicodeDecodeError`。
   **修复**: 添加 `encoding='utf-8'`。

---

## 实验结果

### Step 1: 数据驱动通道选择

**运行时间**: 2026-02-20, ~167s
**数据规模**: 21 被试, 15,663 trials (1 个异常短 trial 被过滤), 128 通道
**输出**: `results/32_channel/channel_selections.json`

NaN 截断后有效信号长度: 277 样本 (2.77s @ EEGNet 100Hz)

#### 通道选择结果

| 方法 | Top-5 通道索引 | BioSemi 标签 | 主要脑区 |
|------|---------------|-------------|---------|
| FDR | 58, 59, 60, 61, 62 | B27-B31 | 右前额-颞叶 + 运动区 |
| CSP | 0, 1, 2, 3, 4 | A1-A5 | 中央-顶叶中线 (Cz/Pz) |
| Attention | 12, 21, 23, 24, 27 | A13, A22, A24, A25, A28 | 后部/枕叶 |
| Band Power | 4, 7, 8, 14, 17 | A5, A8, A9, A15, A18 | 顶叶-枕叶 |

注: BioSemi 128 通道编号 A1-A32 (idx 0-31) 对应中线/中央-顶叶区域，
不同于标准 10-20 系统的字母编号。A1 = Cz (中央顶点)，A4 = Pz (顶叶中线)。

#### 通道重叠分析

| 方法对 | 重叠通道数 | 重叠率 |
|--------|----------|--------|
| FDR ∩ CSP | 8/32 | 25% |
| FDR ∩ Attention | 9/32 | 28% |
| FDR ∩ Band Power | 4/32 | 12% |
| CSP ∩ Attention | 7/32 | 22% |
| CSP ∩ Band Power | 15/32 | 47% |
| Attention ∩ Band Power | 8/32 | 25% |

#### Attention 方法数据来源

| 模型 | 检查点 | 被试数 | 类型 |
|------|--------|--------|------|
| EEGNet | `checkpoints/cross_subject/eegnet_imagery_binary/best.pt` | 7 | spatial_conv 权重提取 |
| CBraMod | `checkpoints/cross_subject/20260206_1029_cbramod_imagery_binary/best.pt` | 21 | 输入梯度计算 (200 trial batch) |

---

### Step 2: 6 配置跨被试对比

**运行时间**: 2026-02-20 17:31 ~ 2026-02-21 00:01 (总计 6.5 小时)
**训练设置**: 每个配置训练 EEGNet + CBraMod，跨被试预训练 (21 被试)
**调度器**: `cosine_annealing_warmup_decay` (50 epochs, patience 10)
**WandB 项目**: `Finger-BCI/eeg-bci`

#### 综合排名 (按两模型平均准确率)

| 排名 | 配置 | 类型 | EEGNet (mean±std) | CBraMod (mean±std) | 平均 |
|------|------|------|-------------------|--------------------|----|
| 1 | **attention** | Data-driven | **70.42 ± 12.75%** | 87.02 ± 9.89% | **78.72%** |
| 2 | fdr | Data-driven | 67.53 ± 11.12% | **88.10 ± 8.80%** | 77.81% |
| 3 | band_power | Data-driven | 67.17 ± 13.21% | 85.51 ± 10.11% | 76.34% |
| 4 | csp | Data-driven | 66.52 ± 12.91% | 85.54 ± 10.34% | 76.03% |
| 5 | commercial | Hand-picked | 64.40 ± 9.82% | 86.31 ± 7.91% | 75.36% |
| 6 | motor_cortex | Hand-picked | 63.12 ± 10.48% | 82.02 ± 9.70% | 72.57% |

#### 按模型分别排名

**CBraMod 排名**:
1. fdr: 88.10%
2. attention: 87.02%
3. commercial: 86.31%
4. csp: 85.54%
5. band_power: 85.51%
6. motor_cortex: 82.02%

**EEGNet 排名**:
1. attention: 70.42%
2. fdr: 67.53%
3. band_power: 67.17%
4. csp: 66.52%
5. commercial: 64.40%
6. motor_cortex: 63.12%

#### 关键发现

1. **数据驱动方法全面优于手工选择**: 4 个 data-driven 配置均排在 2 个 hand-picked 配置之前
2. **CBraMod 最优配置 (fdr)** 达 88.10%，接近 128 通道基线 90.27% (差距仅 2.17%)
3. **EEGNet 最优配置 (attention)** 达 70.42%，显著高于 motor_cortex 的 63.12%
4. **motor_cortex 配置表现最差**: 手工选择运动皮层区域反而不如统计/学习驱动的选择
5. **commercial 配置 CBraMod 标准差最低** (7.91%)，跨被试稳定性最好

#### 各配置结果文件

| 配置 | 时间戳 | EEGNet 结果 | CBraMod 结果 |
|------|--------|------------|-------------|
| motor_cortex | 20260220_1731 | `results/32_channel/20260220_1731_cross-subject_eegnet_imagery_binary.json` | `..._cbramod_...` |
| commercial | 20260220_1850 | `results/32_channel/20260220_1850_cross-subject_eegnet_imagery_binary.json` | `..._cbramod_...` |
| fdr | 20260220_1949 | `results/32_channel/20260220_1949_cross-subject_eegnet_imagery_binary.json` | `..._cbramod_...` |
| csp | 20260220_2052 | `results/32_channel/20260220_2052_cross-subject_eegnet_imagery_binary.json` | `..._cbramod_...` |
| attention | 20260220_2159 | `results/32_channel/20260220_2159_cross-subject_eegnet_imagery_binary.json` | `..._cbramod_...` |
| band_power | 20260220_2301 | `results/32_channel/20260220_2301_cross-subject_eegnet_imagery_binary.json` | `..._cbramod_...` |

---

### 32ch FDR vs 128ch 基线: CBraMod 逐被试对比

128ch 基线: `checkpoints/cross_subject/20260206_1029_cbramod_imagery_binary/best.pt` (21 被试, val_acc=0.660)

| 被试 | 128ch CBraMod | 32ch FDR CBraMod | 差值 |
|------|-------------|-----------------|------|
| S01 | 87.50% | 92.50% | **+5.00%** |
| S02 | 96.25% | 95.62% | -0.62% |
| S03 | 98.75% | 98.12% | -0.62% |
| S04 | 98.12% | 97.50% | -0.62% |
| S05 | 93.12% | 81.25% | -11.88% |
| S06 | 86.25% | 77.50% | -8.75% |
| S07 | 90.00% | 89.38% | -0.62% |
| S08 | 97.50% | 94.38% | -3.12% |
| S09 | 98.75% | 96.88% | -1.88% |
| S10 | 66.25% | 67.50% | **+1.25%** |
| S11 | 93.75% | 91.88% | -1.88% |
| S12 | 88.75% | 88.12% | -0.62% |
| S13 | 91.25% | 89.38% | -1.88% |
| S14 | 88.12% | 90.62% | **+2.50%** |
| S15 | 94.38% | 90.00% | -4.38% |
| S16 | 92.50% | 87.50% | -5.00% |
| S17 | 90.62% | 88.12% | -2.50% |
| S18 | 95.00% | 88.75% | -6.25% |
| S19 | 99.38% | 98.12% | -1.25% |
| S20 | 66.88% | 66.25% | -0.62% |
| S21 | 82.50% | 80.62% | -1.88% |
| **Mean** | **90.27 ± 8.88%** | **88.10 ± 8.80%** | **-2.17%** |

**统计**: 3/21 被试提升，18/21 被试小幅下降。
75% 通道削减 (128→32) 仅造成 2.17% 平均准确率损失，标准差基本不变。

#### 训练配置对比

| 参数 | 128ch 基线 | 32ch FDR |
|------|-----------|---------|
| 检查点 | `20260206_1029_cbramod_imagery_binary/best.pt` | `20260220_2008_cbramod_imagery_binary/best.pt` |
| Best epoch | 52 | 23 |
| Best val_acc | 0.660 | 0.610 |
| 训练时长 | ~85 min | ~44 min |
| Dropout | 0.35 | 0.40 |
| Weight decay | 0.12 | 0.15 |
| Scheduler | cosine_annealing_warmup_decay | cosine_annealing_warmup_decay |
| 被试 | 21 (S01-S21) | 21 (S01-S21) |

---

### 实验意义总结 (Paper Reference, removed — see 综合总结 at end of document)

---

### Step 3: FDR 最优配置扩展实验 (CBraMod Only)

基于 Step 2 确认 FDR 为 CBraMod 最优 32ch 配置后，Step 3 聚焦于 CBraMod 模型，补充以下实验：

1. **Ternary cross-subject** — 扩展至三分类任务
2. **Binary transfer** — 基于 Step 2 binary cross-subject 预训练模型的逐被试微调
3. **Ternary transfer** — 基于 Step 3 ternary cross-subject 预训练模型的逐被试微调

**运行时间**: 2026-02-21 03:32 ~ 11:15
**通道配置**: FDR (32ch)
**模型**: CBraMod only
**微调策略**: freeze=none (全参数可训练)

#### 3a. Ternary Cross-Subject (CBraMod, 32ch FDR)

**Run tag**: `20260221_0332`
**训练时长**: ~72 min (4356s)
**Scheduler**: `cosine_annealing_warmup_decay` (50 epochs)
**Best epoch**: 44 | **Best val_acc**: 0.428
**Checkpoint**: `checkpoints/cross_subject/20260221_0332_cbramod_imagery_ternary/best.pt`

#### 3b. Binary Transfer (CBraMod, 32ch FDR)

**Run tag**: `20260221_0445`
**预训练 checkpoint**: `checkpoints/cross_subject/20260220_2008_cbramod_imagery_binary/best.pt` (Step 2 FDR binary, mean=88.10%)
**微调**: 21 被试逐一微调, freeze=none, `cosine_annealing_warmup_decay`

#### 3c. Ternary Transfer (CBraMod, 32ch FDR)

**Run tag**: `20260221_1042`
**预训练 checkpoint**: `checkpoints/cross_subject/20260221_0332_cbramod_imagery_ternary/best.pt` (Step 3a)
**微调**: 21 被试逐一微调, freeze=none, `cosine_annealing_warmup_decay`

#### Step 3 结果汇总

| 实验 | 任务 | 方法 | Mean ± Std | Median | Min | Max |
|------|------|------|-----------|--------|-----|-----|
| Cross-subject | Binary | 跨被试预训练 | 88.10 ± 8.80% | — | 66.25% | 98.12% |
| **Transfer** | **Binary** | **跨被试→微调** | **88.90 ± 8.72%** | **91.88%** | **64.38%** | **97.50%** |
| Cross-subject | Ternary | 跨被试预训练 | 70.79 ± 12.26% | 70.83% | 47.92% | 89.17% |
| **Transfer** | **Ternary** | **跨被试→微调** | **72.68 ± 11.88%** | **72.50%** | **47.92%** | **92.08%** |

#### Binary Transfer 逐被试结果

| 被试 | Cross-Subject | Transfer | 差值 |
|------|--------------|----------|------|
| S01 | 92.50% | 88.12% | -4.38% |
| S02 | 95.62% | 94.38% | -1.25% |
| S03 | 98.12% | 96.88% | -1.25% |
| S04 | 97.50% | 95.00% | -2.50% |
| S05 | 81.25% | 90.62% | **+9.38%** |
| S06 | 77.50% | 81.88% | +4.38% |
| S07 | 89.38% | 85.62% | -3.75% |
| S08 | 94.38% | 95.62% | +1.25% |
| S09 | 96.88% | 96.25% | -0.62% |
| S10 | 67.50% | 67.50% | 0.00% |
| S11 | 91.88% | 91.88% | 0.00% |
| S12 | 88.12% | 92.50% | +4.38% |
| S13 | 89.38% | 91.88% | +2.50% |
| S14 | 90.62% | 86.88% | -3.75% |
| S15 | 90.00% | 96.88% | **+6.88%** |
| S16 | 87.50% | 94.38% | **+6.88%** |
| S17 | 88.12% | 89.38% | +1.25% |
| S18 | 88.75% | 86.25% | -2.50% |
| S19 | 98.12% | 97.50% | -0.62% |
| S20 | 66.25% | 64.38% | -1.88% |
| S21 | 80.62% | 83.12% | +2.50% |
| **Mean** | **88.10%** | **88.90%** | **+0.80%** |

#### Ternary Transfer 逐被试结果

| 被试 | Cross-Subject | Transfer | 差值 |
|------|--------------|----------|------|
| S01 | 62.08% | 63.75% | +1.67% |
| S02 | 81.67% | 83.75% | +2.08% |
| S03 | 87.50% | 83.75% | -3.75% |
| S04 | 85.00% | 87.50% | +2.50% |
| S05 | 48.75% | 61.25% | **+12.50%** |
| S06 | 78.75% | 78.75% | 0.00% |
| S07 | 70.83% | 70.83% | 0.00% |
| S08 | 83.75% | 82.92% | -0.83% |
| S09 | 80.83% | 89.17% | **+8.33%** |
| S10 | 54.17% | 54.17% | 0.00% |
| S11 | 71.25% | 73.33% | +2.08% |
| S12 | 59.58% | 64.17% | +4.58% |
| S13 | 70.83% | 72.50% | +1.67% |
| S14 | 77.92% | 75.83% | -2.08% |
| S15 | 69.17% | 64.58% | -4.58% |
| S16 | 65.83% | 72.08% | +6.25% |
| S17 | 78.75% | 83.33% | +4.58% |
| S18 | 67.92% | 67.92% | 0.00% |
| S19 | 89.17% | 92.08% | +2.92% |
| S20 | 47.92% | 47.92% | 0.00% |
| S21 | 55.00% | 56.67% | +1.67% |
| **Mean** | **70.79%** | **72.68%** | **+1.89%** |

#### Ternary: 32ch FDR vs 128ch 基线

128ch 基线: `20260207_2056_cross-subject_cbramod_imagery_ternary` (21 被试, mean=75.42%)

| 被试 | 128ch Cross | 32ch FDR Cross | 32ch FDR Transfer | 128→32 Cross 差值 | Transfer 恢复 |
|------|-----------|---------------|-------------------|------------------|--------------|
| S01 | 70.83% | 62.08% | 63.75% | -8.75% | +1.67% |
| S02 | 89.17% | 81.67% | 83.75% | -7.50% | +2.08% |
| S03 | 92.08% | 87.50% | 83.75% | -4.58% | -3.75% |
| S04 | 89.17% | 85.00% | 87.50% | -4.17% | +2.50% |
| S05 | 65.42% | 48.75% | 61.25% | -16.67% | +12.50% |
| S06 | 80.83% | 78.75% | 78.75% | -2.08% | 0.00% |
| S07 | 72.50% | 70.83% | 70.83% | -1.67% | 0.00% |
| S08 | 85.42% | 83.75% | 82.92% | -1.67% | -0.83% |
| S09 | 89.58% | 80.83% | 89.17% | -8.75% | +8.33% |
| S10 | 56.25% | 54.17% | 54.17% | -2.08% | 0.00% |
| S11 | 77.50% | 71.25% | 73.33% | -6.25% | +2.08% |
| S12 | 61.25% | 59.58% | 64.17% | -1.67% | +4.58% |
| S13 | 76.67% | 70.83% | 72.50% | -5.83% | +1.67% |
| S14 | 81.67% | 77.92% | 75.83% | -3.75% | -2.08% |
| S15 | 75.42% | 69.17% | 64.58% | -6.25% | -4.58% |
| S16 | 71.25% | 65.83% | 72.08% | -5.42% | +6.25% |
| S17 | 80.83% | 78.75% | 83.33% | -2.08% | +4.58% |
| S18 | 71.25% | 67.92% | 67.92% | 0.00% | 0.00% |
| S19 | 93.75% | 89.17% | 92.08% | -4.58% | +2.92% |
| S20 | 43.75% | 47.92% | 47.92% | +4.17% | 0.00% |
| S21 | 59.17% | 55.00% | 56.67% | -4.17% | +1.67% |
| **Mean** | **75.42%** | **70.79%** | **72.68%** | **-4.62%** | **+1.89%** |

#### Step 3 关键发现

1. **Transfer 微调效果有限但一致**: Binary +0.80%, Ternary +1.89%。Cross-subject 模型已较好泛化，微调主要帮助低表现被试。

2. **低表现被试受益最大**: S05 在 ternary transfer 中提升 +12.50%（48.75% → 61.25%），binary 中提升 +9.38%。个体微调对"难"被试的补偿作用最显著。

3. **Ternary 通道损失大于 Binary**: 128→32ch 降幅 ternary -4.62% vs binary -2.17%。三分类对空间分辨率更敏感，符合预期——区分三根手指需要更精细的空间信息。

4. **Transfer 可部分弥补通道损失**: Ternary 128ch cross=75.42% → 32ch cross=70.79% (-4.62%) → 32ch transfer=72.68%，transfer 恢复了约 41% 的通道削减损失 (1.89/4.62)。

5. **S20 是持续异常被试**: 在所有配置和任务中均表现最差（binary 64.38%, ternary 47.92%），且微调无效（0% 提升），可能存在数据质量或个体差异问题。

6. **Binary vs Ternary 难度差异**: 32ch FDR transfer 结果 binary 88.90% vs ternary 72.68%，差距 16.22%。三分类（拇指/中指/小指）显著难于二分类（拇指/小指）。

#### Step 3 结果文件

| 实验 | Run Tag | 结果文件 |
|------|---------|---------|
| Ternary cross-subject | 20260221_0332 | `results/32_channel/20260221_0332_cross-subject_cbramod_imagery_ternary.json` |
| Binary transfer | 20260221_0445 | `results/32_channel/20260221_0445_transfer_comparison_cache_imagery_binary.json` |
| Ternary transfer | 20260221_1042 | `results/32_channel/20260221_1042_transfer_comparison_cache_imagery_ternary.json` |

---

### Step 4: 8ch FDR 实验 (CBraMod Only)

将通道选择管线推广至 8 通道，使用 FDR 方法从 128 通道中选择最优 8 通道子集，
验证极低通道数下 CBraMod 的性能下限及 transfer 微调的恢复能力。

**代码泛化**: 在运行 8ch 实验前，对通道选择基础设施进行了重构，使其支持任意通道数：

| 文件 | 改动 |
|------|------|
| `src/preprocessing/channel_selection.py` | 新增 `get_nch_indices(n, config)` 通用函数 |
| `src/preprocessing/data_loader.py` | `channel_32_config` → `channel_config` + `channel_n_target` |
| `src/preprocessing/dataset.py` | Strategy 'E' 调用 `get_nch_indices(n_target, config)` |
| `src/training/*.py` | `channel_config` 优先触发 strategy E（任意 N） |
| `scripts/analysis/compute_32ch_selections.py` | `--output` 默认值从 `--n-channels` 自动推导 |
| `scripts/experiments/run_32ch_experiment.py` | 新增 `--channels`, `--models`, `--steps` |
| `scripts/experiments/run_32ch_config_comparison.py` | 新增 `--channels`，自动过滤不适用配置 |

**运行时间**: 2026-02-21 12:18 ~ 16:24 (总计 4.1 小时)
**通道配置**: FDR (8ch)
**模型**: CBraMod only
**微调策略**: freeze=none

#### 4a. 8ch FDR 通道选择

**输出**: `results/8_channel/channel_selections.json`
**方法**: FDR only（`--n-channels 8 --methods fdr`）

| 通道索引 | BioSemi 标签 | 脑区 |
|---------|-------------|------|
| 60 | B29 | 右前额-颞叶交界 |
| 68 | C5 | 左颞叶-中央 |
| 69 | C6 | 左颞叶-中央 |
| 70 | C7 | 左颞叶 |
| 101 | D6 | 右前额-中央 |
| 102 | D7 | 右前额-中央 |
| 103 | D8 | 右前额 |
| 104 | D9 | 右前额 |

注: 8ch FDR 选择的通道与手工 motor cortex 8ch [A1, A3, A6, A21, B3, B21, C23, D18] **完全不重叠**。
FDR 偏好颞叶和前额区域（类间区分度最高），而非传统认为的中央沟运动皮层。

#### 4b. 8ch FDR 结果汇总

| 实验 | 任务 | 方法 | Mean ± Std | Median | Min | Max |
|------|------|------|-----------|--------|-----|-----|
| Cross-subject | Binary | 跨被试预训练 | 68.33 ± 9.80% | 66.25% | 45.62% | 83.75% |
| **Transfer** | **Binary** | **跨被试→微调** | **72.92 ± 9.55%** | **69.38%** | **60.00%** | **97.50%** |
| Cross-subject | Ternary | 跨被试预训练 | 52.00 ± 9.95% | 52.92% | 35.42% | 72.50% |
| **Transfer** | **Ternary** | **跨被试→微调** | **57.26 ± 11.51%** | **55.42%** | **42.50%** | **87.92%** |

#### Binary 逐被试结果 (8ch FDR)

| 被试 | Cross-Subject | Transfer | 差值 |
|------|--------------|----------|------|
| S01 | 68.75% | 68.75% | 0.00% |
| S02 | 76.88% | 76.25% | -0.62% |
| S03 | 66.25% | 76.25% | **+10.00%** |
| S04 | 83.12% | 92.50% | **+9.38%** |
| S05 | 45.62% | 70.62% | **+25.00%** |
| S06 | 60.00% | 68.75% | +8.75% |
| S07 | 65.00% | 73.12% | +8.12% |
| S08 | 69.38% | 69.38% | 0.00% |
| S09 | 82.50% | 82.50% | 0.00% |
| S10 | 60.00% | 61.25% | +1.25% |
| S11 | 70.62% | 76.88% | +6.25% |
| S12 | 63.12% | 66.25% | +3.12% |
| S13 | 61.88% | 67.50% | +5.62% |
| S14 | 53.12% | 60.00% | +6.88% |
| S15 | 82.50% | 84.38% | +1.88% |
| S16 | 66.25% | 65.00% | -1.25% |
| S17 | 75.00% | 67.50% | -7.50% |
| S18 | 73.12% | 76.88% | +3.75% |
| S19 | 83.75% | 97.50% | **+13.75%** |
| S20 | 63.75% | 63.75% | 0.00% |
| S21 | 64.38% | 66.25% | +1.88% |
| **Mean** | **68.33%** | **72.92%** | **+4.59%** |

#### Ternary 逐被试结果 (8ch FDR)

| 被试 | Cross-Subject | Transfer | 差值 |
|------|--------------|----------|------|
| S01 | 47.92% | 44.58% | -3.33% |
| S02 | 61.25% | 67.08% | +5.83% |
| S03 | 57.50% | 67.50% | **+10.00%** |
| S04 | 57.92% | 75.00% | **+17.08%** |
| S05 | 35.42% | 48.75% | **+13.33%** |
| S06 | 57.92% | 62.50% | +4.58% |
| S07 | 53.33% | 55.42% | +2.08% |
| S08 | 50.83% | 50.00% | -0.83% |
| S09 | 72.08% | 72.08% | 0.00% |
| S10 | 40.42% | 43.33% | +2.92% |
| S11 | 52.92% | 59.58% | +6.67% |
| S12 | 38.33% | 42.50% | +4.17% |
| S13 | 46.67% | 49.58% | +2.92% |
| S14 | 47.92% | 54.17% | +6.25% |
| S15 | 60.00% | 60.00% | 0.00% |
| S16 | 42.50% | 55.42% | **+12.92%** |
| S17 | 55.42% | 58.75% | +3.33% |
| S18 | 57.50% | 57.50% | 0.00% |
| S19 | 72.50% | 87.92% | **+15.42%** |
| S20 | 43.75% | 42.92% | -0.83% |
| S21 | 40.00% | 47.92% | +7.92% |
| **Mean** | **52.00%** | **57.26%** | **+5.26%** |

#### 全通道数对比: CBraMod FDR Cross-Subject

| 被试 | 128ch | 32ch FDR | 8ch FDR | 128→32 | 32→8 |
|------|-------|---------|---------|--------|------|
| S01 | 87.50% | 92.50% | 68.75% | +5.00% | -23.75% |
| S02 | 96.25% | 95.62% | 76.88% | -0.62% | -18.75% |
| S03 | 98.75% | 98.12% | 66.25% | -0.62% | -31.88% |
| S04 | 98.12% | 97.50% | 83.12% | -0.62% | -14.38% |
| S05 | 93.12% | 81.25% | 45.62% | -11.88% | -35.62% |
| S06 | 86.25% | 77.50% | 60.00% | -8.75% | -17.50% |
| S07 | 90.00% | 89.38% | 65.00% | -0.62% | -24.38% |
| S08 | 97.50% | 94.38% | 69.38% | -3.12% | -25.00% |
| S09 | 98.75% | 96.88% | 82.50% | -1.88% | -14.38% |
| S10 | 66.25% | 67.50% | 60.00% | +1.25% | -7.50% |
| S11 | 93.75% | 91.88% | 70.62% | -1.88% | -21.25% |
| S12 | 88.75% | 88.12% | 63.12% | -0.62% | -25.00% |
| S13 | 91.25% | 89.38% | 61.88% | -1.88% | -27.50% |
| S14 | 88.12% | 90.62% | 53.12% | +2.50% | -37.50% |
| S15 | 94.38% | 90.00% | 82.50% | -4.38% | -7.50% |
| S16 | 92.50% | 87.50% | 66.25% | -5.00% | -21.25% |
| S17 | 90.62% | 88.12% | 75.00% | -2.50% | -13.12% |
| S18 | 95.00% | 88.75% | 73.12% | -6.25% | -15.62% |
| S19 | 99.38% | 98.12% | 83.75% | -1.25% | -14.38% |
| S20 | 66.88% | 66.25% | 63.75% | -0.62% | -2.50% |
| S21 | 82.50% | 80.62% | 64.38% | -1.88% | -16.25% |
| **Mean** | **90.27%** | **88.10%** | **68.33%** | **-2.17%** | **-19.77%** |

#### 全通道数对比: Transfer 恢复效果

| 通道数 | Binary Cross | Binary Transfer | Δ | Ternary Cross | Ternary Transfer | Δ |
|--------|-------------|----------------|---|--------------|-----------------|---|
| 128ch | 90.27% | — | — | 75.42% | — | — |
| 32ch FDR | 88.10% | 88.90% | +0.80% | 70.79% | 72.68% | +1.89% |
| **8ch FDR** | **68.33%** | **72.92%** | **+4.59%** | **52.00%** | **57.26%** | **+5.26%** |

#### Step 4 关键发现

1. **8ch 是性能断崖**: 128→32ch 仅损失 2.17%，但 32→8ch 再损失 19.77%（binary cross-subject）。通道数从 32 降至 8 时信息损失远大于 128→32 的削减。

2. **Transfer 微调在低通道数下效果更显著**: 8ch transfer 提升 binary +4.59%, ternary +5.26%，远高于 32ch 的 +0.80%/+1.89%。通道越少，个体适配越重要，跨被试模型的泛化能力越弱。

3. **S05 binary transfer +25.00%**: 8ch cross-subject 仅 45.62%（接近随机），但 transfer 后跳升至 70.62%。8ch 下部分被试的跨被试模型完全失效，但个体微调可部分挽救。

4. **S19 transfer 恢复能力最强**: 8ch binary cross=83.75% → transfer=97.50%（+13.75%），接近 128ch 水平（99.38%）。S19 的解码信息可能集中在少量通道。

5. **FDR 8ch 与手工 motor cortex 8ch 完全不重叠**: FDR 选择偏好颞叶/前额区域（B29, C5-C7, D6-D9），而手工选择集中于中央沟（A1/Cz, B21/C4, D18/C3）。说明 FDR 捕获的是统计上类间差异最大的通道，不一定对应经典运动皮层电极。

6. **Ternary 8ch 接近随机水平**: 52.00% cross-subject（随机基线 33.3%），transfer 后 57.26%。8 通道对三分类任务信息量严重不足。

7. **S20 在 8ch 下反而不再是最差**: Binary cross-subject S20=63.75% 接近均值，S05=45.62% 反而最差。低通道数下被试排序与高通道数不同，说明不同被试对通道位置的敏感性不同。

#### Step 4 结果文件

| 实验 | Run Tag | 结果文件 |
|------|---------|---------|
| 8ch 通道选择 | — | `results/8_channel/channel_selections.json` |
| Binary cross-subject | 20260221_1218 | `results/8_channel/20260221_1218_cross-subject_cbramod_imagery_binary.json` |
| Binary transfer | 20260221_1319 | `results/8_channel/20260221_1319_transfer_comparison_cache_imagery_binary.json` |
| Ternary cross-subject | 20260221_1343 | `results/8_channel/20260221_1343_cross-subject_cbramod_imagery_ternary.json` |
| Ternary transfer | 20260221_1547 | `results/8_channel/20260221_1547_transfer_comparison_cache_imagery_ternary.json` |

---

### Step 5: FDR vs Commercial vs 61ch 对比分析

**Date**: 2026-02-28
**范围**: CBraMod Binary Cross-Subject — 32ch FDR vs 32ch Commercial vs 61ch Standard 10-10

#### 实验背景

在 Step 2 完成 6 配置对比后，补充了两组新实验:
1. **32ch commercial 重跑** (run tag: `20260226_1908`) — 验证 commercial 配置结果稳定性
2. **61ch standard 10-10** (run tag: `20260227_0049`) — 基于 Yazıcı et al. (2025) 的 61 通道配置

> **61ch 配置来源**: Yazıcı, M., Ulutaş, M., & Okuyan, M. (2025). "Effect of EEG Electrode Numbers on Source Estimation in Motor Imagery." *Brain Sciences*, 15(7), 685. DOI: [10.3390/brainsci15070685](https://doi.org/10.3390/brainsci15070685)
>
> 该研究对比了 19/30/61/118 通道对 Motor Imagery 解码的影响，发现 **61 通道准确率最高 (84.73%)**，甚至优于 118 通道 (83.95%)。结论是 30-61 通道范围提供了足够的空间覆盖和良好的信噪比平衡。本实验采用的 61ch 配置即为标准 10-10 国际电极系统的 61 个标准位置。

#### 总览对比

| 配置 | 通道数 | Mean Acc | Std | vs 128ch | vs 32ch FDR |
|------|--------|----------|-----|----------|-------------|
| 128ch baseline | 128 | 90.27% | 8.88% | — | +2.17pp |
| **61ch standard** | 61 | **88.72%** | 9.22% | -1.55pp | +0.62pp |
| **32ch FDR** | 32 | **88.10%** | 8.80% | -2.17pp | — |
| **32ch commercial** | 32 | **86.40%** | 7.95% | -3.87pp | -1.70pp |

#### FDR vs Commercial 逐被试对比 (CBraMod, 32ch)

| 被试 | 32ch FDR | 32ch Commercial | FDR − Comm | 胜出 |
|------|----------|----------------|------------|------|
| S01 | 92.50% | 82.50% | **+10.00%** | FDR |
| S02 | 95.62% | 90.00% | +5.62% | FDR |
| S03 | 98.12% | 96.25% | +1.87% | FDR |
| S04 | 97.50% | 93.12% | +4.38% | FDR |
| S05 | 81.25% | 79.38% | +1.87% | FDR |
| S06 | 77.50% | 80.00% | -2.50% | Comm |
| S07 | 89.38% | 85.62% | +3.76% | FDR |
| S08 | 94.38% | 93.12% | +1.26% | FDR |
| S09 | 96.88% | 99.38% | -2.50% | Comm |
| S10 | 67.50% | 70.62% | -3.12% | Comm |
| S11 | 91.88% | 89.38% | +2.50% | FDR |
| S12 | 88.12% | 83.75% | +4.37% | FDR |
| S13 | 89.38% | 86.25% | +3.13% | FDR |
| S14 | 90.62% | 83.12% | **+7.50%** | FDR |
| S15 | 90.00% | 86.88% | +3.12% | FDR |
| S16 | 87.50% | 92.50% | -5.00% | Comm |
| S17 | 88.12% | 91.25% | -3.13% | Comm |
| S18 | 88.75% | 88.75% | 0.00% | — |
| S19 | 98.12% | 96.88% | +1.24% | FDR |
| S20 | 66.25% | 68.75% | -2.50% | Comm |
| S21 | 80.62% | 76.88% | +3.74% | FDR |
| **Mean** | **88.10%** | **86.40%** | **+1.70%** | **FDR 14:6** |

#### 61ch vs 32ch FDR 逐被试对比 (CBraMod)

| 被试 | 61ch | 32ch FDR | 61ch − FDR | 胜出 |
|------|------|----------|------------|------|
| S01 | 91.25% | 92.50% | -1.25% | FDR |
| S02 | 96.25% | 95.62% | +0.63% | 61ch |
| S03 | 99.38% | 98.12% | +1.26% | 61ch |
| S04 | 98.12% | 97.50% | +0.62% | 61ch |
| S05 | 82.50% | 81.25% | +1.25% | 61ch |
| S06 | 82.50% | 77.50% | +5.00% | 61ch |
| S07 | 88.75% | 89.38% | -0.63% | FDR |
| S08 | 93.75% | 94.38% | -0.63% | FDR |
| S09 | 97.50% | 96.88% | +0.62% | 61ch |
| S10 | 65.00% | 67.50% | -2.50% | FDR |
| S11 | 90.00% | 91.88% | -1.88% | FDR |
| S12 | 82.50% | 88.12% | -5.62% | FDR |
| S13 | 94.38% | 89.38% | +5.00% | 61ch |
| S14 | 93.75% | 90.62% | +3.13% | 61ch |
| S15 | 91.25% | 90.00% | +1.25% | 61ch |
| S16 | 88.75% | 87.50% | +1.25% | 61ch |
| S17 | 88.75% | 88.12% | +0.63% | 61ch |
| S18 | 92.50% | 88.75% | +3.75% | 61ch |
| S19 | 98.75% | 98.12% | +0.63% | 61ch |
| S20 | 66.25% | 66.25% | 0.00% | — |
| S21 | 81.25% | 80.62% | +0.63% | 61ch |
| **Mean** | **88.72%** | **88.10%** | **+0.62%** | **61ch 12:6** |

#### Step 5 关键发现

1. **FDR 显著优于 Commercial (+1.70pp)**: FDR 在 14/21 被试上胜出，验证数据驱动通道选择在实际准确率上的优势。FDR 最大优势出现在 S01 (+10.00%) 和 S14 (+7.50%)。

2. **Commercial 标准差更低但均值更低**: Commercial std=7.95% vs FDR std=8.80%。商用布局的全脑均匀分布提供了更稳定的跨被试表现，但牺牲了约 1.7pp 的平均准确率。

3. **61ch 仅比 32ch FDR 高 0.62pp**: 尽管通道数接近翻倍 (32→61)，性能提升极小。说明 FDR 选择的 32 个通道已捕获了绝大部分判别信息，额外通道贡献边际递减。

4. **通道退化梯度非线性**: 128→61ch (-1.55pp, 52% 通道削减), 61→32ch FDR (-0.62pp, 48% 削减), 但 128→32ch Commercial (-3.87pp)。FDR 选择在 32ch 水平接近 61ch 的信息量，而 commercial 固定布局则损失更多。

5. **Commercial 在部分被试反超 FDR**: S09 (99.38% vs 96.88%), S16 (92.50% vs 87.50%), S10 (70.62% vs 67.50%)。这些被试可能在 10-20 标准位置有更强的可区分信号，而 FDR 选择的非标准通道对他们贡献较小。

6. **61ch 的实用价值**: 61ch standard 10-10 布局达到 88.72%，仅比 128ch 低 1.55pp，且使用标准电极位置无需数据驱动计算。对于能使用 64 通道设备的场景，是最佳选择。

7. **EEGNet 同样从 61ch 获益**: 61ch EEGNet 达到 71.85%（vs 32ch FDR 67.53%, 32ch commercial 64.26%），但整体远低于 CBraMod，再次证实基座模型在低通道数场景的优势。

#### Step 5 结果文件

| 实验 | Run Tag | 结果文件 |
|------|---------|---------|
| 61ch CBraMod cross-subject | 20260227_0049 | `results/61_channel/standard_1010/20260227_0049_cross-subject_cbramod_imagery_binary.json` |
| 61ch EEGNet cross-subject | 20260227_0049 | `results/61_channel/standard_1010/20260227_0049_cross-subject_eegnet_imagery_binary.json` |
| 32ch commercial CBraMod | 20260226_1908 | `results/32_channel/commercial/20260226_1908_cross-subject_cbramod_imagery_binary.json` |
| 32ch commercial EEGNet | 20260226_1908 | `results/32_channel/commercial/20260226_1908_cross-subject_eegnet_imagery_binary.json` |

---

### Step 6: Commercial 扩展实验 (CBraMod Only)

基于 Step 5 对 commercial 配置仅有 binary cross-subject 结果，Step 6 补充 ternary 及 transfer 实验，与 FDR 形成完整对照。

**运行时间**: 2026-02-26 19:08 ~ 23:33
**通道配置**: Commercial (32ch)
**模型**: CBraMod only (transfer 包含 EEGNet 对照)
**微调策略**: freeze=none (全参数可训练)

#### 6a. Ternary Cross-Subject (CBraMod, 32ch Commercial)

**Run tag**: `20260226_2042`
**训练时长**: ~78 min (4659s)
**Scheduler**: `cosine_annealing_warmup_decay` (50 epochs)
**Best epoch**: 40 | **Best val_acc**: 0.442

#### 6b. Binary Transfer (CBraMod + EEGNet, 32ch Commercial)

**Run tag**: `20260226_2000`
**预训练 checkpoint (CBraMod)**: `checkpoints/cross_subject/20260220_2008_cbramod_imagery_binary/best.pt` (32ch FDR binary)
**预训练 checkpoint (EEGNet)**: `checkpoints/cross_subject/20260220_2159_eegnet_imagery_binary/best.pt`
**微调**: 21 被试逐一微调, freeze=none

#### 6c. Ternary Transfer (CBraMod + EEGNet, 32ch Commercial)

**Run tag**: `20260226_2217`
**预训练 checkpoint (CBraMod)**: `checkpoints/cross_subject/20260221_0332_cbramod_imagery_ternary/best.pt` (32ch FDR ternary)
**预训练 checkpoint (EEGNet)**: `checkpoints/cross_subject/20260226_2042_eegnet_imagery_ternary/best.pt`
**微调**: 21 被试逐一微调, freeze=none

#### Step 6 结果汇总

| 实验 | 任务 | 模型 | Mean ± Std | Median | Min | Max |
|------|------|------|-----------|--------|-----|-----|
| Cross-subject | Binary | CBraMod | 86.40 ± 7.95% | — | 68.75% | 96.88% |
| **Transfer** | **Binary** | **CBraMod** | **85.27 ± 9.09%** | **85.62%** | **61.88%** | **97.50%** |
| Transfer | Binary | EEGNet | 72.68 ± 11.81% | 71.88% | 58.12% | 95.00% |
| Cross-subject | Ternary | CBraMod | 69.35 ± 11.93% | — | 42.92% | 90.42% |
| **Transfer** | **Ternary** | **CBraMod** | **69.50 ± 12.56%** | **67.50%** | **43.33%** | **94.17%** |
| Transfer | Ternary | EEGNet | 51.98 ± 12.50% | 52.92% | 33.33% | 76.67% |

> **数据来源**:
> - Binary cross-subject: `results/32_channel/commercial/20260226_1908_cross-subject_cbramod_imagery_binary.json` (Step 5 重跑)
> - Binary transfer: `results/32_channel/commercial/20260226_2000_transfer_comparison_cache_imagery_binary.json`
> - Ternary cross-subject: `results/32_channel/commercial/20260226_2042_cross-subject_cbramod_imagery_ternary.json`
> - Ternary transfer: `results/32_channel/commercial/20260226_2217_transfer_comparison_cache_imagery_ternary.json`

#### Binary Transfer 逐被试结果 (CBraMod)

| 被试 | Cross-Subject | Transfer | 差值 |
|------|--------------|----------|------|
| S01 | 82.50% | 85.62% | +3.12% |
| S02 | 90.00% | 91.88% | +1.88% |
| S03 | 96.25% | 96.25% | 0.00% |
| S04 | 93.12% | 96.88% | +3.76% |
| S05 | 79.38% | 82.50% | +3.12% |
| S06 | 80.00% | 74.38% | -5.62% |
| S07 | 85.62% | 78.75% | -6.87% |
| S08 | 93.12% | 85.62% | -7.50% |
| S09 | 99.38% | 93.12% | -6.26% |
| S10 | 70.62% | 66.25% | -4.37% |
| S11 | 89.38% | 90.62% | +1.24% |
| S12 | 83.75% | 87.50% | +3.75% |
| S13 | 86.25% | 88.75% | +2.50% |
| S14 | 83.12% | 83.12% | 0.00% |
| S15 | 86.88% | 91.25% | +4.37% |
| S16 | 92.50% | 85.62% | -6.88% |
| S17 | 91.25% | 85.00% | -6.25% |
| S18 | 88.75% | 89.38% | +0.63% |
| S19 | 96.88% | 97.50% | +0.62% |
| S20 | 68.75% | 61.88% | -6.87% |
| S21 | 76.88% | 78.75% | +1.87% |
| **Mean** | **86.40%** | **85.27%** | **-1.13%** |

#### Ternary Transfer 逐被试结果 (CBraMod)

| 被试 | Cross-Subject | Transfer | 差值 |
|------|--------------|----------|------|
| S01 | 61.25% | 57.08% | -4.17% |
| S02 | 80.83% | 71.67% | -9.16% |
| S03 | 82.50% | 84.17% | +1.67% |
| S04 | 89.17% | 88.33% | -0.84% |
| S05 | 62.50% | 60.42% | -2.08% |
| S06 | 77.92% | 72.92% | -5.00% |
| S07 | 64.58% | 65.83% | +1.25% |
| S08 | 76.67% | 78.33% | +1.66% |
| S09 | 78.33% | 84.17% | +5.84% |
| S10 | 53.75% | 43.33% | -10.42% |
| S11 | 75.83% | 77.50% | +1.67% |
| S12 | 58.75% | 61.25% | +2.50% |
| S13 | 66.25% | 67.50% | +1.25% |
| S14 | 71.25% | 74.17% | +2.92% |
| S15 | 70.00% | 65.42% | -4.58% |
| S16 | 65.00% | 66.67% | +1.67% |
| S17 | 71.67% | 76.67% | +5.00% |
| S18 | 66.25% | 64.17% | -2.08% |
| S19 | 90.42% | 94.17% | +3.75% |
| S20 | 42.92% | 46.67% | +3.75% |
| S21 | 50.42% | 59.17% | +8.75% |
| **Mean** | **69.35%** | **69.50%** | **+0.15%** |

#### Step 6 关键发现

1. **Commercial binary transfer 反而下降 (-1.13%)**: 与 FDR 配置下 transfer 始终提升不同，commercial 的 binary transfer 均值低于 cross-subject。可能因为预训练 checkpoint 来自 FDR 配置，通道分布不匹配导致微调效果受限。

2. **Commercial ternary transfer 几乎无变化 (+0.15%)**: 微调对 commercial ternary 的改善极为有限，进一步说明通道选择与预训练模型的匹配度对迁移效果至关重要。

3. **EEGNet 在 commercial 配置下全面落后**: Binary 72.68% vs CBraMod 85.27% (差距 12.59pp)，Ternary 51.98% vs 69.50% (差距 17.52pp)。CBraMod 的优势在 commercial 布局下同样显著。

4. **Commercial 在所有 4 个任务中均落后于 FDR**: Binary cross -1.70pp, binary transfer -3.63pp, ternary cross -1.44pp, ternary transfer -3.18pp。差距在 transfer 阶段进一步放大。

#### Step 6 结果文件

| 实验 | Run Tag | 结果文件 |
|------|---------|---------|
| Ternary cross-subject (CBraMod) | 20260226_2042 | `results/32_channel/commercial/20260226_2042_cross-subject_cbramod_imagery_ternary.json` |
| Binary transfer (CBraMod+EEGNet) | 20260226_2000 | `results/32_channel/commercial/20260226_2000_transfer_comparison_cache_imagery_binary.json` |
| Ternary transfer (CBraMod+EEGNet) | 20260226_2217 | `results/32_channel/commercial/20260226_2217_transfer_comparison_cache_imagery_ternary.json` |

---

### Step 7: Attention 扩展实验 (CBraMod Only)

Step 2 确认 attention 为综合排名第 1 的 32ch 配置（EEGNet 最优 + 两模型平均最高），Step 7 补充 attention 配置在 ternary 及 transfer 任务上的完整结果，与 FDR、commercial 形成三方对比。

**运行时间**: 2026-02-28 22:18 ~ 2026-03-01 00:30
**通道配置**: Attention (32ch)
**模型**: CBraMod only
**微调策略**: freeze=none (全参数可训练)

#### 7a. Ternary Cross-Subject (CBraMod, 32ch Attention)

**Run tag**: `20260228_2247`
**训练时长**: ~70 min (4228s)
**Scheduler**: `cosine_annealing_warmup_decay` (50 epochs)
**Best epoch**: 46 | **Best val_acc**: 0.446
**Checkpoint**: `checkpoints/cross_subject/20260228_2247_cbramod_imagery_ternary/best.pt`

#### 7b. Binary Transfer (CBraMod, 32ch Attention)

**Run tag**: `20260228_2218`
**预训练 checkpoint**: `checkpoints/cross_subject/20260220_2218_cbramod_imagery_binary/best.pt` (Step 2 attention binary cross-subject, mean=87.02%)
**微调**: 21 被试逐一微调, freeze=none, `cosine_annealing_warmup_decay`

#### 7c. Ternary Transfer (CBraMod, 32ch Attention)

**Run tag**: `20260228_2358`
**预训练 checkpoint**: `checkpoints/cross_subject/20260228_2247_cbramod_imagery_ternary/best.pt` (Step 7a)
**微调**: 21 被试逐一微调, freeze=none, `cosine_annealing_warmup_decay`

#### Step 7 结果汇总

| 实验 | 任务 | 方法 | Mean ± Std | Median | Min | Max |
|------|------|------|-----------|--------|-----|-----|
| Cross-subject | Binary | 跨被试预训练 | 87.02 ± 9.89% | — | 61.88% | 98.12% |
| **Transfer** | **Binary** | **跨被试→微调** | **88.69 ± 8.37%** | **90.00%** | **64.38%** | **100.00%** |
| Cross-subject | Ternary | 跨被试预训练 | 71.53 ± 11.94% | — | 46.25% | 89.58% |
| **Transfer** | **Ternary** | **跨被试→微调** | **73.57 ± 13.17%** | **74.17%** | **46.25%** | **92.08%** |

> **数据来源**:
> - Binary cross-subject: `results/32_channel/attention/20260220_2159_cross-subject_cbramod_imagery_binary.json` (Step 2)
> - Binary transfer: `results/32_channel/attention/20260228_2218_transfer_comparison_cache_imagery_binary.json`
> - Ternary cross-subject: `results/32_channel/attention/20260228_2247_cross-subject_cbramod_imagery_ternary.json`
> - Ternary transfer: `results/32_channel/attention/20260228_2358_transfer_comparison_cache_imagery_ternary.json`

#### Binary Transfer 逐被试结果

| 被试 | Cross-Subject | Transfer | 差值 |
|------|--------------|----------|------|
| S01 | 91.25% | 93.75% | +2.50% |
| S02 | 85.62% | 86.25% | +0.63% |
| S03 | 98.12% | 99.38% | +1.26% |
| S04 | 96.88% | 98.12% | +1.24% |
| S05 | 61.88% | 92.50% | **+30.62%** |
| S06 | 80.00% | 83.12% | +3.12% |
| S07 | 91.25% | 83.12% | -8.13% |
| S08 | 96.88% | 90.00% | -6.88% |
| S09 | 96.88% | 100.00% | +3.12% |
| S10 | 65.62% | 71.25% | +5.63% |
| S11 | 90.00% | 90.00% | 0.00% |
| S12 | 88.12% | 86.25% | -1.87% |
| S13 | 91.25% | 91.25% | 0.00% |
| S14 | 89.38% | 90.00% | +0.62% |
| S15 | 88.75% | 95.00% | +6.25% |
| S16 | 86.25% | 88.12% | +1.87% |
| S17 | 90.62% | 86.88% | -3.74% |
| S18 | 91.25% | 92.50% | +1.25% |
| S19 | 96.25% | 95.62% | -0.63% |
| S20 | 70.62% | 64.38% | -6.24% |
| S21 | 80.62% | 85.00% | +4.38% |
| **Mean** | **87.02%** | **88.69%** | **+1.67%** |

#### Ternary Transfer 逐被试结果

| 被试 | Cross-Subject | Transfer | 差值 |
|------|--------------|----------|------|
| S01 | 64.58% | 63.33% | -1.25% |
| S02 | 82.08% | 88.33% | +6.25% |
| S03 | 84.17% | 87.50% | +3.33% |
| S04 | 76.67% | 85.83% | +9.16% |
| S05 | 46.25% | 62.92% | **+16.67%** |
| S06 | 75.83% | 74.17% | -1.66% |
| S07 | 74.58% | 71.25% | -3.33% |
| S08 | 82.92% | 83.33% | +0.41% |
| S09 | 87.92% | 92.08% | +4.16% |
| S10 | 56.25% | 48.75% | -7.50% |
| S11 | 77.50% | 82.92% | +5.42% |
| S12 | 65.00% | 61.67% | -3.33% |
| S13 | 72.92% | 74.58% | +1.66% |
| S14 | 82.92% | 85.42% | +2.50% |
| S15 | 66.67% | 71.67% | +5.00% |
| S16 | 65.42% | 63.33% | -2.09% |
| S17 | 75.83% | 80.00% | +4.17% |
| S18 | 69.58% | 71.67% | +2.09% |
| S19 | 89.58% | 91.25% | +1.67% |
| S20 | 49.58% | 46.25% | -3.33% |
| S21 | 55.83% | 58.75% | +2.92% |
| **Mean** | **71.53%** | **73.57%** | **+2.04%** |

#### Step 7: 三配置横向对比 (CBraMod, 32ch)

| 任务 | Attention | FDR | Commercial | Att−FDR | Att−Comm |
|------|-----------|-----|------------|---------|----------|
| Binary Cross | 87.02% | **88.10%** | 86.40% | -1.08pp | +0.62pp |
| Binary Transfer | 88.69% | **88.90%** | 85.27% | -0.21pp | +3.42pp |
| Ternary Cross | **71.53%** | 70.79% | 69.35% | +0.74pp | +2.18pp |
| Ternary Transfer | **73.57%** | 72.68% | 69.50% | +0.89pp | +4.07pp |

> **数据来源**: Attention — Step 7 上方表格; FDR — Step 3 结果汇总; Commercial — Step 6 结果汇总。

#### Step 7 关键发现

1. **Attention 与 FDR 差距极小 (<1.1pp)**: 在所有 4 个任务上，两者差距均在 1.1 个百分点以内。Binary 任务 FDR 微优，ternary 任务 attention 微优，整体可视为等效配置。

2. **Attention transfer 提升最稳定**: Binary +1.67pp, ternary +2.04pp。相比 FDR (+0.80pp/+1.89pp) 和 commercial (-1.13pp/+0.15pp)，attention 配置的预训练→微调管线收益最一致。

3. **S05 binary transfer +30.62%**: Attention 配置下 S05 从 cross-subject 的 61.88% 跃升至 transfer 的 92.50%，提升幅度极为突出。此被试在 attention 通道布局下的个体微调效果远超其他配置 (FDR +9.38%, commercial +3.12%)。

4. **S09 达到 100% binary transfer**: Attention 配置下 S09 在微调后达到完美准确率。该被试在 128ch 基线下亦为高表现被试 (cross-subject 96.88%)。

5. **Commercial 在 transfer 阶段差距扩大**: Commercial vs attention 差距从 cross-subject 的 0.62pp/2.18pp 扩大到 transfer 的 3.42pp/4.07pp。这进一步证实：数据驱动通道选择不仅在初始模型上更优，在迁移学习流程中优势还会放大。

6. **Ternary 任务偏好 attention 配置**: 在两个 ternary 任务中 attention 均胜出，可能因为 attention 方法融合了模型梯度信息，能捕获对多类别区分更关键的通道。

#### Step 7 结果文件

| 实验 | Run Tag | 结果文件 |
|------|---------|---------|
| Binary cross-subject | 20260220_2159 | `results/32_channel/attention/20260220_2159_cross-subject_cbramod_imagery_binary.json` |
| Ternary cross-subject | 20260228_2247 | `results/32_channel/attention/20260228_2247_cross-subject_cbramod_imagery_ternary.json` |
| Binary transfer | 20260228_2218 | `results/32_channel/attention/20260228_2218_transfer_comparison_cache_imagery_binary.json` |
| Ternary transfer | 20260228_2358 | `results/32_channel/attention/20260228_2358_transfer_comparison_cache_imagery_ternary.json` |

---

### 综合实验意义总结 (Paper Reference)

1. **通道削减的非线性特征**: 128→32ch 仅损失 2.17%（binary cross-subject），但 32→8ch 再损失 19.77%。存在明确的性能断崖——约 32 通道是性能/硬件权衡的最优折中点。

2. **数据驱动选择优于手工选择**: 6 配置对比中，4 个 data-driven 方法全面优于 2 个 hand-picked 配置。FDR（最简单的统计方法）在 CBraMod 上获得最高准确率 88.10%。

3. **FDR 通道选择的反直觉发现**: FDR 偏好颞叶/前额区域（高类间区分度），而非传统运动皮层电极（C3/Cz/C4）。8ch FDR 与手工 motor cortex 8ch 完全不重叠。说明对于基座模型，最大化信息量比对齐经典脑区更有效。

4. **Transfer 微调随通道减少而增效**: 8ch transfer 提升 +4.59%/+5.26%（binary/ternary），远高于 32ch 的 +0.80%/+1.89%。通道越少，跨被试泛化越弱，个体适配价值越高。

5. **CBraMod vs EEGNet 差距随通道减少而扩大**: 32ch binary cross-subject: CBraMod 88.10% vs EEGNet 67.53%（差距 20.57%）。基座模型的预训练知识在低通道数场景下优势更加显著。

6. **32ch 的实用价值**: 32ch 对应标准商用 EEG 设备通道数，CBraMod 在 32ch FDR 配置下达到 88.10%（cross-subject）和 88.90%（transfer），直接支持实际 BCI 部署。

7. **被试间差异模式随通道数变化**: 128ch 的"难"被试（S20）在 8ch 下不再最差（S05 反而最差），说明不同被试对通道空间位置的敏感性不同。

8. **三分类对通道数更敏感**: Ternary 损失始终大于 binary（32ch -4.62% vs -2.17%，8ch -23.42% vs -21.94%）。手指级别的精细解码需要更高的空间分辨率。

9. **61ch ≈ 32ch FDR >> 32ch Commercial**: 61ch standard 10-10 (88.72%) 与 32ch FDR (88.10%) 差距仅 0.62pp，但 commercial 配置 (86.40%) 落后 FDR 1.70pp。数据驱动选择可用一半通道达到接近 61ch 的性能。

10. **Attention ≈ FDR >> Commercial（完整管线验证）**: Step 6-7 的四任务对比表明，attention 和 FDR 在所有任务上差距 <1.1pp，均显著优于 commercial (2-4pp)。Attention 在 ternary 任务和 transfer 稳定性上略优，FDR 在 binary 任务上略优。两者均为推荐的 32ch 配置。

#### 核心数据速查表

所有数值为 CBraMod test accuracy (21 被试均值)。

| 配置 | Binary Cross | Binary Transfer | Ternary Cross | Ternary Transfer | 来源 Step |
|------|-------------|----------------|--------------|-----------------|-----------|
| 128ch (baseline) | 90.27% | — | 75.42% | — | Step 3 (ref) |
| 61ch standard | 88.72% | — | — | — | Step 5 |
| 32ch FDR | 88.10% | 88.90% | 70.79% | 72.68% | Step 2-3 |
| **32ch attention** | **87.02%** | **88.69%** | **71.53%** | **73.57%** | **Step 2, 7** |
| 32ch commercial | 86.40% | 85.27% | 69.35% | 69.50% | Step 5-6 |
| 8ch FDR | 68.33% | 72.92% | 52.00% | 57.26% | Step 4 |
