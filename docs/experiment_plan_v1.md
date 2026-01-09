# CBraMod 与 EEGNet 对比实验方案 v1.1

> **更新日期**: 2026-01-07
> **状态**: 执行中 (已迁移至 unified training framework)
> **修订说明**: 更新脚本引用以反映 `train_within_subject.py` 的统一架构

---

## 目录
- [实验背景与目标](#实验背景与目标)
  - [研究目标](#研究目标)
  - [模型对比概览](#模型对比概览)
- [CBraMod 技术规格（来自论文）](#cbramod-技术规格来自论文)
  - [模型架构](#模型架构)
  - [预训练配置（来自论文 Table 5）](#预训练配置来自论文-table-5)
  - [预训练使用的 19 通道（关键）](#预训练使用的-19-通道关键)
  - [预训练数据预处理规则](#预训练数据预处理规则)
- [数据集说明](#数据集说明)
  - [FINGER-EEG-BCI 数据集概要](#finger-eeg-bci-数据集概要)
  - [Session 类型](#session-类型)
  - [数据结构](#数据结构)
  - [数据质量说明](#数据质量说明)
- [关键适配挑战与解决方案](#关键适配挑战与解决方案)
  - [核心适配挑战](#核心适配挑战)
  - [通道适配方案](#通道适配方案)
  - [数据预处理流程（符合 CBraMod 规范）](#数据预处理流程符合-cbramod-规范)
- [实验设计](#实验设计)
  - [实验一：标准离线性能对比](#实验一标准离线性能对比)
  - [实验二：低资源/快速校准（核心实验）](#实验二低资源快速校准核心实验)
  - [实验三：在线微调策略对比](#实验三在线微调策略对比)
  - [实验四：收敛速度分析](#实验四收敛速度分析)
- [模型配置](#模型配置)
  - [EEGNet-8,2 配置（基线）](#eegnet-82-配置基线)
  - [CBraMod 微调配置（来自论文 Table 6）](#cbramod-微调配置来自论文-table-6)
- [预期结果与分析框架](#预期结果与分析框架)
  - [CBraMod 在 Motor Imagery 任务的参考性能（来自论文 Table 3）](#cbramod-在-motor-imagery-任务的参考性能来自论文-table-3)
  - [性能对比表（待填充）](#性能对比表待填充)
  - [分析要点](#分析要点)
- [潜在风险与应对措施](#潜在风险与应对措施)
  - [采样率重采样风险](#采样率重采样风险)
  - [过拟合风险](#过拟合风险)
  - [通道信息不足风险](#通道信息不足风险)
  - [数据幅值范围问题](#数据幅值范围问题)
  - [数据质量问题](#数据质量问题)
- [项目目录结构](#项目目录结构)
- [执行优先级与里程碑](#执行优先级与里程碑)
- [参考资料](#参考资料)
- [附录 A：CBraMod 论文关键发现总结](#附录-acbramod-论文关键发现总结)
  - [A.1 注意力机制对比（Figure 4）](#a1-注意力机制对比figure-4)
  - [A.2 位置编码对比（Figure 5）](#a2-位置编码对比figure-5)
  - [A.3 低资源场景（Table 19）](#a3-低资源场景table-19)
  - [A.4 收敛速度（Figure 13）](#a4-收敛速度figure-13)

---

## 实验背景与目标

### 研究目标

验证 EEG 基座模型（Foundation Model）在精细手指运动解码任务中的有效性，重点探索：
1. **基座模型的迁移能力**：预训练知识能否有效迁移到手指级 BCI 任务
2. **低资源场景优势**：小样本微调下的性能表现
3. **实时可行性**：推理延迟是否满足实时控制需求（<100ms）

### 模型对比概览

| 特性 | EEGNet-8,2 (原论文基线) | CBraMod (待验证模型) |
|------|-------------------------|----------------------|
| **类型** | 紧凑 CNN | Criss-Cross Transformer |
| **参数量** | ~2.5K | ~4.0M |
| **预训练** | ❌ 无 | ✅ 9000+ 小时 TUEG |
| **输入采样率** | 100Hz | 200Hz |
| **输入通道数** | 128 通道 | 任意通道 (预训练用 19ch) |
| **Patch 时长** | N/A | 1 秒 (200 点) |
| **时间窗口** | 4s (离线) / 200ms (在线) | 任意长度 |
| **代码仓库** | bfinl/Finger-BCI-Decoding | wjq-learning/CBraMod |

---

## CBraMod 技术规格（来自论文）

### 模型架构

```
CBraMod 架构概览:
┌─────────────────────────────────────────────────────────┐
│  输入: EEG 样本 S ∈ R^(C×T)                              │
│        C = 通道数, T = 时间点数                          │
├─────────────────────────────────────────────────────────┤
│  1. Patching & Masking                                  │
│     - 按 1 秒窗口分割为 Patches: X ∈ R^(C×n×200)        │
│     - 预训练时 50% mask ratio                            │
├─────────────────────────────────────────────────────────┤
│  2. Time-Frequency Patch Encoding                       │
│     ├── 时域分支: 3层 1D CNN + GroupNorm + GELU         │
│     └── 频域分支: FFT + Fully-Connected                 │
│     输出: Patch Embedding E ∈ R^(C×n×200)               │
├─────────────────────────────────────────────────────────┤
│  3. ACPE (非对称条件位置编码)                            │
│     - 2D Depthwise CNN, kernel=(19,7)                   │
│     - 空间维度用更大 kernel 编码长程依赖                  │
│     - 时间维度用较小 kernel 编码短程依赖                  │
├─────────────────────────────────────────────────────────┤
│  4. Criss-Cross Transformer (12 层)                     │
│     - Hidden dim: 200                                   │
│     - Feed-forward dim: 800                             │
│     - 8 头注意力 (4 S-Attention + 4 T-Attention)        │
│     - S-Attention: 同一时间点跨通道建模                  │
│     - T-Attention: 同一通道跨时间建模                    │
├─────────────────────────────────────────────────────────┤
│  5. Task Head (微调时替换)                               │
│     - 分类: Flatten + MLP                               │
│     - 回归: Flatten + MLP                               │
└─────────────────────────────────────────────────────────┘
```

### 预训练配置（来自论文 Table 5）

| 参数 | 值 |
|------|-----|
| **预训练数据** | TUEG (Temple University Hospital EEG Corpus) |
| **数据量** | 1,109,545 样本 (~9,248 小时) |
| **通道数** | 19 (10-20 子集) |
| **采样率** | 200 Hz |
| **样本时长** | 30 秒 |
| **Patch 时长** | 1 秒 (200 点) |
| **Mask Ratio** | 50% |
| **Batch Size** | 128 |
| **Epochs** | 40 |
| **Optimizer** | AdamW |
| **Learning Rate** | 5e-4 |
| **Weight Decay** | 5e-2 |
| **Scheduler** | CosineAnnealingLR |

### 预训练使用的 19 通道（关键）

```
标准 10-20 子集 (与 TUEG 一致):
Fp1, Fp2, F7, F3, Fz, F4, F8,
T3, C3, Cz, C4, T4,
T5, P3, Pz, P4, T6,
O1, O2
```

> **注意**: T3/T4/T5/T6 在某些命名系统中对应 T7/T8/P7/P8

### 预训练数据预处理规则

```python
# CBraMod 预训练数据处理流程 (来自论文 Section 3.1)

def preprocess_for_pretraining(raw_eeg):
    # 1. 移除过短录像 (≤5分钟)
    # 2. 丢弃每段录像的首尾各 1 分钟

    # 3. 带通滤波
    eeg = bandpass_filter(raw_eeg, low=0.3, high=75)  # Hz

    # 4. 陷波滤波 (去除工频干扰)
    eeg = notch_filter(eeg, freq=60)  # Hz (美国数据)

    # 5. 重采样
    eeg = resample(eeg, target_fs=200)  # Hz

    # 6. 分段
    samples = segment(eeg, duration=30)  # 秒, 不重叠

    # 7. 坏样本移除 (关键!)
    # 任何数据点绝对幅值 > 100µV 的样本视为坏样本
    clean_samples = [s for s in samples
                     if np.all(np.abs(s) <= 100)]  # µV

    # 8. 归一化 (以 100µV 为单位)
    normalized = clean_samples / 100.0  # 值域约 [-1, 1]

    return normalized
```

---

## 数据集说明

### FINGER-EEG-BCI 数据集概要

| 属性 | 值 |
|------|-----|
| **数据来源** | Carnegie Mellon University |
| **被试数量** | 21 人 (右利手) |
| **采集设备** | BioSemi ActiveTwo 128 通道 |
| **原始采样率** | 1024 Hz |
| **电极配置文件** | `biosemi128.ELC` |
| **任务类型** | 运动执行 (ME) / 运动想象 (MI) |
| **目标手指** | 拇指(1)、食指(2)、中指(3)、小指(4) |

### Session 类型

#### 离线 Session (Offline)
- **任务**: 重复单指屈伸运动/想象
- **Run 数量**: 32 runs/session (部分被试 30 runs)
- **试次结构**: 每指 5 次试验，随机顺序
- **时间参数**:
  - 试验时长: 5 秒
  - 试验间隔: 2 秒

#### 在线 Session (Online)
- **分类范式**:
  - 二分类: 拇指 vs 小指
  - 三分类: 拇指 vs 食指 vs 小指
- **Run 分布** (32 runs/session):
  - Run 1-8: 三分类 + Base Model
  - Run 9-16: 二分类 + Base Model
  - Run 17-24: 三分类 + Fine-tuned Model
  - Run 25-32: 二分类 + Fine-tuned Model
- **时间参数**:
  - 试验时长: 3 秒
  - 反馈延迟: 1 秒
  - 反馈时长: 2 秒
  - 试验间隔: 2 秒
  - 预测更新: 每 125ms (128 样本点)

### 数据结构

```
SXX/
├── TaskType(_SessYY_Zclass_Model)/
│   └── SXX_TaskType_RXX.mat
│       ├── eeg (struct)
│       │   ├── data        # [128 × time_points] EEG 矩阵
│       │   ├── time        # 时间戳向量 (秒)
│       │   ├── label       # 通道标签
│       │   ├── fsample     # 采样率 (1024 Hz)
│       │   ├── nChans      # 通道数 (128)
│       │   ├── nSamples    # 样本点数
│       │   ├── prediction  # [在线] 预测类别 (1=拇指, 2=食指, 4=小指)
│       │   └── prob_*      # [在线] 各类别概率
│       └── event (struct array)
│           ├── type        # "Target" / "TrialEnd"
│           ├── sample      # 事件时间戳
│           └── value       # 目标类别 (1/2/4)
```

### 数据质量说明

> ⚠️ **注意**: `S07\OnlineImagery_Sess05_3class_Base` 的事件信息缺失，该部分数据不可用。

---

## 关键适配挑战与解决方案

### 核心适配挑战

```
FINGER-EEG-BCI 数据 (128ch, 1024Hz)
              ↓
        需要适配到
              ↓
CBraMod 输入格式 (任意通道, 200Hz, 1秒 Patch)
```

**主要差异分析**:

| 维度 | FINGER-EEG-BCI | CBraMod 预训练 | 影响分析 |
|------|----------------|----------------|----------|
| **通道数** | 128 通道 (高密度) | 19 通道 (10-20) | ACPE 可适应任意通道，但与预训练差异大 |
| **采样率** | 1024 Hz | 200 Hz | 需重采样，先滤波防混叠 |
| **时间窗** | 5s/3s (离线/在线) | 30s (预训练) | 可处理任意长度 |

### 通道适配方案

#### 方案 A: 标准 10-20 映射 (推荐首选)

**策略**: 从 128 通道中提取与 CBraMod 预训练一致的 19 通道。

**目标通道** (与 CBraMod 预训练完全一致):
```
Fp1, Fp2, F7, F3, Fz, F4, F8,
T3, C3, Cz, C4, T4,
T5, P3, Pz, P4, T6,
O1, O2
```

**映射方法**: 使用 MNE 库的标准 montage 进行最近邻匹配：
```python
import mne
import numpy as np

def create_biosemi128_to_1020_mapping(biosemi_elc_path):
    """
    创建 BioSemi 128 到标准 10-20 的通道映射
    """
    # CBraMod 使用的 19 通道
    target_channels = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'T3', 'C3', 'Cz', 'C4', 'T4',
        'T5', 'P3', 'Pz', 'P4', 'T6',
        'O1', 'O2'
    ]

    # 加载 BioSemi 128 电极位置
    montage_128 = mne.channels.read_custom_montage(biosemi_elc_path)

    # 加载标准 10-20 montage
    montage_1020 = mne.channels.make_standard_montage('standard_1020')

    # 获取 3D 坐标
    pos_128 = montage_128.get_positions()['ch_pos']
    pos_1020 = montage_1020.get_positions()['ch_pos']

    # 为每个目标通道找到最近的 BioSemi 电极
    mapping = {}
    for target_ch in target_channels:
        target_pos = pos_1020[target_ch]
        min_dist = float('inf')
        nearest_ch = None
        for bio_ch, bio_pos in pos_128.items():
            dist = np.linalg.norm(target_pos - bio_pos)
            if dist < min_dist:
                min_dist = dist
                nearest_ch = bio_ch
        mapping[target_ch] = nearest_ch

    return mapping
```

**优点**: 最大化利用预训练知识，与预训练数据分布最接近
**缺点**: 丢弃约 85% 的空间信息

#### 方案 B: 运动皮层高密度子集 (备选)

**策略**: 选择 C3/Cz/C4 区域附近的 19 个高密度电极。

**理由**:
- 手指运动控制主要集中在运动皮层 (Motor Cortex)
- BioSemi 128 在 C3/C4 附近有高密度覆盖
- ACPE 可以学习新的空间拓扑关系

**优点**: 保留与任务最相关的空间信息
**缺点**: 与预训练分布差异大，需更多微调

#### 方案 C: 全通道 ACPE 自适应 (实验性)

**策略**: 直接输入全部 128 通道，依赖 ACPE 学习新空间关系。

**论文支持**: CBraMod 论文明确指出 ACPE "can dynamically generate positional encoding, which can well be adapted to arbitrary EEG formats"

**优点**: 保留全部空间信息
**缺点**:
- 与预训练差异最大
- 计算开销显著增加 (128 vs 19 通道)
- 可能需要更多微调数据

### 数据预处理流程（符合 CBraMod 规范）

```python
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, resample_poly

def preprocess_for_cbramod(mat_file_path, channel_mapping, strategy='A'):
    """
    将 FINGER-EEG-BCI 数据预处理为 CBraMod 输入格式

    严格遵循 CBraMod 论文的预处理规范
    """
    # 1. 加载原始数据
    mat_data = sio.loadmat(mat_file_path, struct_as_record=False, squeeze_me=True)
    eeg_struct = mat_data['eeg']
    eeg_data = eeg_struct.data  # [128, time_points]
    fs_original = int(eeg_struct.fsample)  # 1024 Hz

    # 2. 通道选择 (在滤波前执行以减少计算)
    if strategy == 'A':
        # 使用预定义的映射选择 19 通道
        channel_indices = [channel_mapping[ch] for ch in channel_mapping]
        eeg_data = eeg_data[channel_indices, :]
        n_channels = 19
    elif strategy == 'B':
        # 选择运动皮层区域 19 通道 (需要预定义)
        motor_cortex_indices = get_motor_cortex_indices()
        eeg_data = eeg_data[motor_cortex_indices, :]
        n_channels = 19
    else:  # strategy == 'C'
        n_channels = 128

    # 3. 带通滤波 (0.3-75 Hz, 与 CBraMod 预训练一致)
    nyq = fs_original / 2
    low = 0.3 / nyq
    high = 75 / nyq
    b_bp, a_bp = butter(4, [low, high], btype='bandpass')
    eeg_data = filtfilt(b_bp, a_bp, eeg_data, axis=1)

    # 4. 陷波滤波去除工频干扰 (60Hz, 美国数据)
    notch_freq = 60
    notch_low = (notch_freq - 2) / nyq
    notch_high = (notch_freq + 2) / nyq
    b_notch, a_notch = butter(4, [notch_low, notch_high], btype='bandstop')
    eeg_data = filtfilt(b_notch, a_notch, eeg_data, axis=1)

    # 5. 重采样至 200Hz (CBraMod 要求)
    # resample_poly 内置抗混叠滤波
    eeg_data = resample_poly(eeg_data, up=200, down=fs_original, axis=1)

    # 6. 单位转换 (确保为 µV)
    # BioSemi 数据通常以 µV 为单位，需确认
    # 如果是 V，需要 * 1e6

    # 7. 坏样本检测 (可选，用于质量控制)
    # CBraMod 预训练时移除了 |amplitude| > 100µV 的样本
    max_amp = np.max(np.abs(eeg_data))
    if max_amp > 100:
        print(f"警告: 最大幅值 {max_amp:.1f}µV > 100µV, 可能包含伪影")

    # 8. 归一化 (以 100µV 为单位, 与 CBraMod 预训练一致)
    eeg_data = eeg_data / 100.0  # 值域约 [-1, 1]

    # 9. 分段为 1 秒 Patch (200 点)
    patch_size = 200  # 1秒 @ 200Hz
    n_samples = eeg_data.shape[1]
    n_patches = n_samples // patch_size

    # 截断到整数个 patch
    eeg_data = eeg_data[:, :n_patches * patch_size]

    # 重塑为 [n_patches, n_channels, patch_size]
    patches = eeg_data.reshape(n_channels, n_patches, patch_size)
    patches = patches.transpose(1, 0, 2)

    return patches
```

---

## 实验设计

### 实验一：标准离线性能对比

**目标**: 在离线数据上对比 EEGNet 与 CBraMod 的分类性能。

#### 数据划分

采用被试间划分（Subject-independent），与原论文一致：

| 集合 | 被试 | 用途 |
|------|------|------|
| 训练集 | S01-S14 (14人) | 模型训练 |
| 验证集 | S15-S17 (3人) | 超参数调优 |
| 测试集 | S18-S21 (4人) | 最终评估 |

#### 分类任务

| 任务 | 类别 | 原论文基线 (EEGNet) | 说明 |
|------|------|---------------------|------|
| 二分类 | 拇指 vs 小指 | 80.56% (MI) | 最易区分的两指 |
| 三分类 | 拇指 vs 食指 vs 小指 | 60.61% (MI) | 在线控制常用 |
| 四分类 | 全部四指 | ~45% | 挑战性任务 |

#### 评估指标 (与 CBraMod 论文一致)

**二分类任务**:
- Balanced Accuracy
- AUC-PR
- AUROC (主要监控指标)

**多分类任务**:
- Balanced Accuracy
- Cohen's Kappa (主要监控指标)
- Weighted F1

### 实验二：低资源/快速校准（核心实验）

**目标**: 验证基座模型在小样本下的优势——这是 Foundation Model 最大的价值所在。

#### 实验设置

| 设置 | 训练数据量 | 说明 |
|------|-----------|------|
| 极低资源 | 10% | 约 1-2 个 Run |
| 低资源 | 30% | 约 10 个 Run |
| 标准 | 100% | 全部训练数据 |

**参考**: CBraMod 论文 Table 19 显示，在 30% 数据下 CBraMod 仍显著优于 BIOT 和 LaBraM。

#### 快速校准模拟

模拟真实 BCI 场景：仅使用单个 Session 的前 8 个 Run 进行微调。

```
Session 1 结构:
├── Run 1-8:   用于模型训练/微调 (校准阶段)
│              收集约 160 个试次
│
└── Run 9-16:  用于测试 (使用阶段)
               评估校准后的模型性能
```

**预期结果**:
- CBraMod 在极少数据下应快速达到可用性能 (>60% Accuracy)
- EEGNet 可能因数据不足而无法有效收敛

### 实验三：在线微调策略对比

**目标**: 复现原论文的在线微调流程，对比两模型的适应性。

#### 实验流程

```
┌─────────────────────────────────────────────────────┐
│  Phase 1: Base Model (Run 1-8)                      │
│  ├── 使用离线数据预训练的模型                         │
│  ├── 收集在线数据用于微调                            │
│  └── 记录 Base Model 在线性能                        │
├─────────────────────────────────────────────────────┤
│  Phase 2: Fine-tuned Model (Run 9-16)               │
│  ├── 使用 Phase 1 数据微调模型                       │
│  ├── 对比: EEGNet-FT vs CBraMod-FT                  │
│  └── 记录性能提升幅度                                │
└─────────────────────────────────────────────────────┘
```

### 实验四：收敛速度分析

**目标**: 量化基座模型的训练效率优势。

**参考数据** (来自 CBraMod 论文 Appendix R, Figure 13):
- CBraMod: 第 1 个 epoch 即可达到较好结果，~10 epochs 收敛
- EEGConformer: 第 1 个 epoch Cohen's Kappa ≈ 0，~30 epochs 收敛

**指标**:
- 达到 70% 准确率所需的 Epochs 数
- 达到收敛所需的 Wall-clock time
- 学习曲线对比图

---

## 模型配置

### EEGNet-8,2 配置（基线）

```yaml
# configs/eegnet_config.yaml
model:
  name: EEGNet-8,2
  F1: 8              # 时间滤波器数量
  D: 2               # 深度乘数
  F2: 16             # F1 × D
  kernel_length: 64  # 时间卷积核长度
  dropout_rate: 0.5

data:
  sampling_rate: 100     # Hz (与原论文一致)
  n_channels: 128        # 使用全部通道
  window_length: 4.0     # 秒 (离线)

training:
  epochs: 100
  batch_size: 32
  optimizer: Adam
  learning_rate: 1e-3
  weight_decay: 0
  early_stopping_patience: 20
```

### 训练

```bash
# CBraMod 二分类 (拇指 vs 小指)
uv run python -m src.training.train_within_subject --task binary --model cbramod

# CBraMod 三分类
uv run python -m src.training.train_within_subject --task ternary --model cbramod

# EEGNet 训练
uv run python -m src.training.train_within_subject --task binary --model eegnet
```

### 全被试对比

```bash
# 运行完整对比实验
uv run python scripts/run_full_comparison.py
```

### 测试

### CBraMod 微调配置（来自论文 Table 6）

```yaml
# configs/cbramod_config.yaml
model:
  name: CBraMod
  pretrained: true
  pretrained_path: "checkpoints/cbramod_pretrained.pth"

  # 架构参数 (不可修改，与预训练一致)
  architecture:
    patch_size: 200           # 1秒 @ 200Hz
    hidden_dim: 200
    ff_dim: 800               # Feed-forward dimension
    n_layers: 12              # Criss-cross transformer layers
    n_heads: 8                # 4 S-Attention + 4 T-Attention
    dropout: 0.1

data:
  sampling_rate: 200          # Hz (必须)
  patch_duration: 1.0         # 秒 (必须)
  patch_points: 200           # 1s × 200Hz

preprocessing:
  bandpass: [0.3, 75]         # Hz (与预训练一致)
  notch_filter: 60            # Hz (美国工频)
  normalization_unit: 100     # µV (除以此值归一化)

# 微调超参数 (来自论文 Table 6)
training:
  epochs: 50
  batch_size: 64
  dropout: 0.1

  optimizer: AdamW
  learning_rate: 1e-4
  adam_betas: [0.9, 0.999]
  adam_eps: 1e-8
  weight_decay: 5e-2

  scheduler: CosineAnnealingLR
  cosine_cycle_epochs: 50
  min_lr: 1e-6

  clipping_gradient_norm: 1.0

  # 标签平滑 (仅多分类)
  label_smoothing: 0.1

# 损失函数 (来自论文 Appendix D.4)
loss:
  binary_classification: BCEWithLogitsLoss
  multi_class_classification: CrossEntropyLoss  # with label smoothing
  regression: MSELoss
```

---

## 预期结果与分析框架

### CBraMod 在 Motor Imagery 任务的参考性能（来自论文 Table 3）

| 数据集 | 任务 | EEGNet | CBraMod | Δ |
|--------|------|--------|---------|---|
| PhysioNet-MI | 4-class | 0.5814 BA / 0.4468 Kappa | **0.6417 BA / 0.5222 Kappa** | +6.0% BA |
| SHU-MI | 2-class | 0.6283 AUROC | **0.6988 AUROC** | +7.1% |

> **注意**: 这些是 CBraMod 论文中的结果，使用的是公开的 MI 数据集。FINGER-EEG-BCI 数据集的任务更精细（单指级），可能有不同表现。

### 性能对比表（待填充）

#### 表 1: 离线分类准确率

| 任务 | 数据量 | EEGNet (128ch, 100Hz) | CBraMod (19ch, 200Hz) | Δ |
|------|--------|----------------------|----------------------|---|
| 二分类 (Thumb vs Pinky) | 100% | 80.56% (Ref) | **待测** | - |
| 二分类 (Thumb vs Pinky) | 30% | 待测 | **待测** | - |
| 二分类 (Thumb vs Pinky) | 10% | 待测 | **待测** | - |
| 三分类 | 100% | 60.61% (Ref) | **待测** | - |
| 四分类 | 100% | ~45% (Ref) | **待测** | - |

#### 表 2: 收敛速度

| 指标 | EEGNet | CBraMod | 参考值 (CBraMod 论文) |
|------|--------|---------|----------------------|
| 第 1 epoch 性能 | 待测 | 待测 | CBraMod 可达较好结果 |
| 收敛 epochs | 待测 | 待测 | ~30 vs ~10 |
| 总训练时间 | 待测 | 待测 | - |

#### 表 3: 计算资源

| 指标 | EEGNet | CBraMod | 说明 |
|------|--------|---------|------|
| 参数量 | ~2.5K | ~4.0M | - |
| FLOPs (16ch, 10s) | ~9M | ~319M | 来自论文 Table 22 |
| 单次推理延迟 | 待测 | 待测 | 实时要求 <100ms |
| GPU 显存 | 待测 | 待测 | - |

### 分析要点

1. **通道数 vs 预训练知识**: 128 通道的空间细节能否弥补 EEGNet 缺乏预训练的劣势？
2. **任务复杂度影响**: 随着分类类别增加，大模型的优势是否更明显？
3. **低资源场景**: 在 10%/30% 数据下，性能差距如何变化？
4. **收敛效率**: 预训练能带来多少训练效率提升？
5. **ACPE 适应性**: 不同通道配置方案的效果对比

---

## 潜在风险与应对措施

### 采样率重采样风险

**风险**: 从 1024Hz 下采样到 200Hz 可能引入混叠伪影。

**应对**:
- 下采样前必须进行低通滤波 (<100Hz，实际使用 75Hz)
- 使用 `scipy.signal.resample_poly` 内置抗混叠滤波
- 验证：对比重采样前后的功率谱，确保无高频泄漏

### 过拟合风险

**风险**: CBraMod (4M 参数) 在小数据集上极易过拟合。

**应对** (来自论文配置):
- 使用论文推荐的 Dropout (0.1)
- 使用较大的 Weight Decay (5e-2)
- 可选：冻结 Encoder，仅微调分类头 (但论文 Table 18 显示 fixed 效果差)
- 监控验证集 Loss，及时 Early Stopping
- 使用 Label Smoothing (0.1)

### 通道信息不足风险

**风险**: 19 通道可能无法捕捉足够的手指运动空间模式。

**应对**:
- 优先实验方案 A (标准 10-20)
- 如效果不佳，尝试方案 B (运动皮层高密度子集)
- 方案 C (全 128 通道) 作为上限参考
- 记录并分析不同通道配置的性能差异

### 数据幅值范围问题

**风险**: FINGER-EEG-BCI 数据的幅值范围可能与 CBraMod 预训练数据不同。

**应对**:
- 检查原始数据单位 (应为 µV)
- 遵循 CBraMod 归一化规则：除以 100µV
- 对于幅值 > 100µV 的样本，考虑 clip 或移除

### 数据质量问题

**已知问题**: `S07\OnlineImagery_Sess05_3class_Base` 事件信息缺失。

**应对**: 在数据加载时排除该部分数据，并在结果中注明。

---

## 项目目录结构

```
EEG-BCI/
├── data/
│   ├── raw/                        # 原始数据 (S01-S21)
│   │   ├── S01/
│   │   ├── S02/
│   │   └── ...
│   ├── processed/                  # 预处理后数据
│   │   ├── cbramod_format/         # CBraMod 输入格式
│   │   └── eegnet_format/          # EEGNet 输入格式
│   ├── biosemi128.ELC              # 电极位置文件
│   ├── channel_mapping.json        # 通道映射表
│   └── README.txt                  # 数据说明
│
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── channel_selection.py    # 通道选择与映射
│   │   ├── filtering.py            # 滤波处理
│   │   ├── resampling.py           # 重采样
│   │   └── data_loader.py          # 数据加载器
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── eegnet.py               # EEGNet-8,2 实现
│   │   └── cbramod_adapter.py      # CBraMod 适配层
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── train_within_subject.py # 核心训练脚本 (对齐原论文)
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py              # 评估指标计算
│       └── visualization.py        # 结果可视化
│
├── configs/
│   ├── eegnet_config.yaml
│   ├── cbramod_config.yaml
│   └── experiment_config.yaml
│
├── experiments/
│   ├── exp1_offline_comparison/    # 实验一：离线对比
│   ├── exp2_low_resource/          # 实验二：低资源
│   ├── exp3_online_finetune/       # 实验三：在线微调
│   └── exp4_convergence/           # 实验四：收敛分析
│
├── results/
│   ├── figures/                    # 结果图表
│   ├── tables/                     # 结果表格
│   └── logs/                       # 训练日志
│
├── checkpoints/                    # 模型检查点
│   ├── eegnet/
│   └── cbramod/
│
├── docs/
│   └── experiment_plan_v1.md       # 本文档
│
├── CLAUDE.md                       # Claude Code 指引
└── README.md                       # 项目说明
```

---

## 执行优先级与里程碑

### Phase 1: 基础设施搭建 (优先)

- [ ] 数据加载与预处理管线
- [ ] 通道映射表创建与验证 (BioSemi 128 → 10-20)
- [ ] EEGNet 基线复现
- [ ] CBraMod 模型加载与适配

### Phase 2: 核心实验

- [ ] 实验一：离线性能对比 (100% 数据)
- [ ] 实验二：低资源场景 (10%, 30% 数据)
- [ ] 实验四：收敛速度分析

### Phase 3: 扩展实验

- [ ] 实验三：在线微调策略
- [ ] 通道适配方案对比 (A vs B vs C)
- [ ] 消融实验与分析

### Phase 4: 总结与文档

- [ ] 结果汇总与可视化
- [ ] 实验报告撰写
- [ ] 代码整理与开源准备

---

## 参考资料

1. **Finger-BCI 论文**: "EEG-based brain-computer interface enables real-time robotic hand control at individual finger level"
2. **CBraMod 论文**: Wang et al., "CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding", ICLR 2025
3. **EEG Foundation Models 综述**: "A Simple Review of EEG Foundation Models" (arXiv:2504.20069v2)
4. **数据集**: FINGER-EEG-BCI Dataset (Carnegie Mellon University)
5. **代码仓库**:
   - EEGNet: https://github.com/bfinl/Finger-BCI-Decoding
   - CBraMod: https://github.com/wjq-learning/CBraMod

---

## 附录 A：CBraMod 论文关键发现总结

### A.1 注意力机制对比（Figure 4）

| 注意力类型 | 说明 | 效果 |
|-----------|------|------|
| Full Attention | 建模所有 patch 间依赖 | 最差 |
| Axial Attention | 顺序建模空间/时间依赖 | 中等 |
| CCNet Attention | 单 map criss-cross | 略优于 full |
| **Criss-Cross (Ours)** | 并行 S/T attention | **最佳** |

### A.2 位置编码对比（Figure 5）

| 位置编码 | 说明 | 效果 |
|---------|------|------|
| w/o PE | 无位置编码 | 最差 |
| APE | 绝对位置编码 | 中等 |
| CPE | 条件位置编码 (对称) | 较好 |
| **ACPE (Ours)** | 非对称条件位置编码 | **最佳** |

### A.3 低资源场景（Table 19）

在 30% 数据下，CBraMod 仍显著优于其他方法：
- FACED: CBraMod 0.3239 Kappa vs BIOT 0.2573 vs LaBraM 0.2672
- PhysioNet-MI: CBraMod 0.4150 Kappa vs BIOT 0.3477 vs LaBraM 0.3598

### A.4 收敛速度（Figure 13）

- CBraMod: 第 1 epoch 即达到较好结果，~10 epochs 收敛
- EEGConformer: 第 1 epoch Kappa ≈ 0，~30 epochs 收敛
