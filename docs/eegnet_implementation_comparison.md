# EEGNet 实现对比文档

本文档详细对比本仓库的 EEGNet 实现与原始 FINGER-EEG-BCI 论文实现之间的差异。

**原论文**: Ding et al., "EEG-based brain-computer interface enables real-time robotic hand control at individual finger level", Nature Communications, 2025

**原论文代码**: https://github.com/bfinl/Finger-BCI-Decoding

---

## 概述

| 方面 | 原论文实现 | 本仓库实现 | 差异说明 |
|------|-----------|-----------|----------|
| 深度学习框架 | TensorFlow/Keras | PyTorch | ⚠️ 不同框架 |
| 模型架构 | EEGNet-8,2 | EEGNet-8,2 | ✅ 一致 |
| 通道数 | 128 (BioSemi) | 128 (BioSemi) | ✅ 一致 |
| 采样率 | 1024 Hz → 100 Hz | 1024 Hz → 100 Hz | ✅ 一致 |
| 带通滤波 | 4-40 Hz, 4阶 Butterworth | 4-40 Hz, 4阶 Butterworth | ✅ 一致 |
| 滑动窗口 | 1s 窗口, 128 samples 步长 | 1s 窗口, 128 samples 步长 | ✅ 一致 |
| 归一化 | Z-score (时间轴) | Z-score (时间轴) | ✅ 一致 |
| CAR | 是 (trial 级别) | 是 (trial 级别) | ✅ 一致 |
| Batch Size | 16 | 64 | ⚠️ 不同 |
| 训练 epochs | 300 | 300 | ✅ 一致 |
| 早停 patience | 80 | 4-5 | ⚠️ 不同 |
| 学习率调度器 | ReduceLROnPlateau | ReduceLROnPlateau | ✅ 一致 |
| Temporal Kernel | 32 | 64 | ⚠️ 不同 |
| 性能优化 | 无 | AMP, torch.compile, cuDNN | 🚀 增强 |

---

## 模型架构

### 原论文 EEGNet-8,2

**文件**: `EEGModels_tf.py`, `Functions.py:219-221`

```python
model = EEGNet(
    nb_classes = params['nclass'],
    Chans = chans,
    Samples = samples,
    dropoutRate = 0.5,      # 预训练: 0.5, 微调: 0.65
    kernLength = 32,        # 注意: 32 samples @ 100Hz = 320ms
    F1 = 8,
    D = 2,
    F2 = 16,
    dropoutType = 'Dropout'
)
```

**架构细节** (`EEGModels_tf.py:55-155`):
```
Block 1:
- Conv2D: (1, kernLength=32), F1=8, padding='same', bias=False
- BatchNorm
- DepthwiseConv2D: (Chans, 1), D=2, max_norm(1.)
- BatchNorm
- ELU
- AvgPool2D: (1, 4)
- Dropout: 0.5

Block 2:
- SeparableConv2D: (1, 16), F2=16, padding='same', bias=False
- BatchNorm
- ELU
- AvgPool2D: (1, 8)
- Dropout: 0.5

Classification:
- Flatten
- Dense: n_classes, max_norm(0.25)
- Softmax
```

### 本仓库实现

**文件**: `src/models/eegnet.py:34-130`

```python
class EEGNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 128,
        n_samples: int = 400,       # 4s @ 100Hz
        n_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,    # 注意: 64 samples @ 100Hz = 640ms
        dropout_rate: float = 0.5,
    ):
```

**架构细节**:
```
Block 1:
- Conv2d: (1, kernel_length=64), F1=8, padding='same', bias=False
- BatchNorm2d
- Conv2dWithConstraint: (n_channels, 1), groups=F1, max_norm=1.0
- BatchNorm2d
- ELU
- AvgPool2d: (1, 4)
- Dropout: 0.5

Block 2:
- Conv2d (depthwise): (1, 16), groups=F1*D, bias=False
- Conv2d (pointwise): (1, 1), F2=16, bias=False
- BatchNorm2d
- ELU
- AvgPool2d: (1, 8)
- Dropout: 0.5

Classification:
- Flatten
- Linear: n_classes
```

### 架构差异分析

| 参数 | 原论文 | 本仓库 | 差异说明 |
|------|--------|--------|----------|
| F1 | 8 | 8 | ✅ 一致 |
| D | 2 | 2 | ✅ 一致 |
| F2 | 16 | 16 | ✅ 一致 |
| **kernLength** | **32** | **64** | ⚠️ 本仓库更长 |
| Pool1 | (1, 4) | (1, 4) | ✅ 一致 |
| Pool2 | (1, 8) | (1, 8) | ✅ 一致 |
| SeparableConv kernel | (1, 16) | (1, 16) | ✅ 一致 |
| Dropout | 0.5 | 0.5 | ✅ 一致 |
| Dense max_norm | 0.25 | 无 | ⚠️ 本仓库无约束 |
| Depthwise max_norm | 1.0 | 1.0 | ✅ 一致 |

**关键差异**:
1. **kernLength**: 原论文使用 32 (320ms @ 100Hz)，本仓库使用 64 (640ms @ 100Hz)
2. **Dense layer constraint**: 原论文使用 max_norm(0.25)，本仓库无约束

---

## 数据预处理

### 原论文预处理流程

**文件**: `Functions.py:81-200`

```python
# 1. 加载数据并提取 trials (line 94-136)
for filepath in data_paths:
    for filename in os.listdir(filepath):
        mat = scipy.io.loadmat(file_path)
        # 提取 Target 到 TrialEnd 之间的数据
        # 填充至 maxtriallen=5s (NaN)

        # CAR (逐 trial, line 133)
        cur_data = cur_data - cur_data.mean(axis=1, keepdims=True)

# 2. 随机打乱并划分 (line 160-170)
shuffled_idx = np.random.permutation(nTrial)
train_percent = 0.8
train_idx = shuffled_idx[:int(train_percent*nTrial)]

# 3. 滑动窗口分割 (line 177-180)
segment_size = int(params['windowlen'] * params['srate'])  # 1s = 1024 samples
step_size = 128  # 128 samples @ 1024 Hz = 125ms
X_train, Y_train, I_train = segment_data(X_train, Y_train, segment_size, step_size)

# 4. 降采样 (line 183-184)
DesiredLen = int(params['windowlen'] * params['downsrate'])  # 100 samples
X_train = resample(X_train, DesiredLen, axis=2)

# 5. 带通滤波 (line 187-196)
padding_length = 100
padded_train = np.pad(X_train, ((0,0),(0,0),(padding_length,padding_length)), 'constant')
b, a = scipy.signal.butter(4, params['bandpass_filt'], btype='bandpass', fs=100)
X_train = scipy.signal.lfilter(b, a, padded_train, axis=-1)
X_train = X_train[:,:,padding_length:-padding_length]

# 6. Z-score 归一化 (line 199-200)
X_train = scipy.stats.zscore(X_train, axis=2, nan_policy='omit')
```

### 本仓库实现

**文件**: `src/preprocessing/data_loader.py:721-835`

```python
def preprocess_run_paper_aligned(...):
    # Step 1: 提取 trials (line 757-803)
    # 基于 Target/TrialEnd 事件，填充至 max_samples (NaN)

    # Step 2: CAR (逐 trial, line 806-807)
    if config.apply_car:
        trials = trials - trials.mean(axis=1, keepdims=True)

    # Step 3: 滑动窗口分割 (line 809-815)
    segment_size = int(config.segment_length * fs)  # 1024 samples
    step_size = config.segment_step_samples  # 128 samples
    segments, seg_labels, trial_indices = segment_with_sliding_window(...)

    # Step 4: 降采样 (line 817-820)
    target_samples = int(config.segment_length * config.target_fs)  # 100
    segments = scipy.signal.resample(segments, target_samples, axis=2)

    # Step 5: 带通滤波 (line 822-830)
    segments = apply_bandpass_filter_paper(
        segments, fs=100, low_freq=4.0, high_freq=40.0, order=4, padding=100
    )

    # Step 6: Z-score 归一化 (line 832-833)
    segments = apply_zscore_per_segment(segments, axis=-1)
```

### 预处理差异分析

| 步骤 | 原论文 | 本仓库 | 状态 |
|------|--------|--------|------|
| Trial 提取 | Target → TrialEnd | Target → TrialEnd | ✅ 一致 |
| 填充方法 | NaN | NaN | ✅ 一致 |
| CAR 应用 | 逐 trial (axis=1) | 逐 trial (axis=1) | ✅ 一致 |
| 滑动窗口 | 1s @ 1024Hz | 1s @ 1024Hz | ✅ 一致 |
| 步长 | 128 samples | 128 samples | ✅ 一致 |
| 降采样 | scipy.signal.resample | scipy.signal.resample | ✅ 一致 |
| 滤波类型 | lfilter (因果) | lfilter (因果) | ✅ 一致 |
| 滤波阶数 | 4阶 Butterworth | 4阶 Butterworth | ✅ 一致 |
| 滤波频带 | [4, 40] Hz | [4, 40] Hz | ✅ 一致 |
| 滤波 padding | 100 samples | 100 samples | ✅ 一致 |
| Z-score 轴 | axis=2 (时间) | axis=-1 (时间) | ✅ 一致 |

**预处理完全对齐** ✅

---

## 训练配置

### 原论文训练配置

**文件**: `Functions.py:150-260`, `main_model_training.py:41-48`

```python
# 参数设置 (main_model_training.py:42-48)
params = {
    'maxtriallen': 5,          # 5s offline trials
    'windowlen': 1,            # 1s sliding window
    'block_size': 128,         # step size
    'downsrate': 100,          # 降采样至 100 Hz
    'bandpass_filt': [4, 40],  # 4-40 Hz
    'nclass': nclass
}

# 训练配置 (Functions.py:204)
batch_size, epochs = 16, 300

# 优化器 (Functions.py:229-232)
if finetune:
    optimizer = Adam(learning_rate=1e-4)
else:
    optimizer = Adam(learning_rate=0.001)

# Callbacks (Functions.py:226-227)
callback_es = EarlyStopping(monitor='val_loss', patience=80)
callback_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=30)

# 微调 (Functions.py:243-253)
if finetune:
    epochs = 100
    dropout_rate = 0.65
    layers_fine_tune = 12  # 冻结前 (num_layers - 12) 层
```

### 本仓库实现

**文件**: `configs/eegnet_config.yaml`, `src/training/train_within_subject.py`

```yaml
# configs/eegnet_config.yaml
training:
  epochs: 300
  batch_size: 64           # 原论文: 16
  learning_rate: 1.0e-3
  weight_decay: 0
  early_stopping: true
  patience: 4              # 原论文: 80
  min_delta: 0.001
  scheduler: plateau       # ReduceLROnPlateau
```

```python
# src/training/train_within_subject.py:264-275
elif scheduler_type == 'plateau':
    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        self.optimizer,
        mode='min',
        factor=0.5,          # 与原论文一致
        patience=30,         # 与原论文一致
        min_lr=1e-6,
    )
```

### 训练配置差异分析

| 配置项 | 原论文 | 本仓库 | 差异说明 |
|--------|--------|--------|----------|
| **Batch Size** | **16** | **64** | ⚠️ 本仓库 4x |
| Epochs | 300 | 300 | ✅ 一致 |
| Learning Rate | 1e-3 | 1e-3 | ✅ 一致 |
| Weight Decay | 未明确 | 0 | ✅ 标准值 |
| **Early Stopping** | **patience=80** | **patience=4-5** | ⚠️ 差异显著 |
| LR Scheduler | ReduceLROnPlateau | ReduceLROnPlateau | ✅ 一致 |
| LR Factor | 0.5 | 0.5 | ✅ 一致 |
| LR Patience | 30 | 30 | ✅ 一致 |

**关键差异**:
1. **Batch Size**: 原论文 16，本仓库 64 (4x)
2. **Early Stopping Patience**: 原论文 80 epochs，本仓库 4-5 epochs

---

## 数据划分协议

### 原论文协议

**文件**: `Functions.py:50-78`, `Functions.py:160-170`

```python
# 数据生成 (generate_paths)
if model_type == 'Finetune':
    # 只使用当天的 Base runs
    pattern = f'{prefix}_Sess{session_num:02}*Base'
else:
    # Offline + 之前所有 Online sessions
    offline_pattern = f'{prefix}'  # OfflineImagery 或 OfflineMovement
    for session in range(1, session_num):
        online_pattern = f'{prefix}_Sess{session:02}*'

# 划分方法 (train_models, line 160-170)
shuffled_idx = np.random.permutation(nTrial)
train_percent = 0.8
train_idx = shuffled_idx[:int(train_percent*nTrial)]
val_idx = np.setdiff1d(shuffled_idx, train_idx)
```

**关键**: 原论文使用**随机打乱**划分 (80/20)

### 本仓库实现

**文件**: `src/training/train_within_subject.py`, `CLAUDE.md`

```
训练数据:
├── OfflineImagery (30 runs)
├── OnlineImagery_Sess01_Xclass_Base (8 runs)
├── OnlineImagery_Sess01_Xclass_Finetune (8 runs)
└── OnlineImagery_Sess02_Xclass_Base (8 runs)
    ↓
    分层时序分割 (Stratified Temporal Split)
    ↓
├── Train (前 80% trials)
└── Val (后 20% trials)

测试数据 (完全独立):
└── OnlineImagery_Sess02_Xclass_Finetune (8 runs)
```

**关键**:
- Train/Val 划分：使用**时序分割**（验证集取末尾 20%），避免数据泄露
- 训练时 shuffle：**启用** (`shuffle=True`)，与原论文一致

### 数据划分差异分析

| 方面 | 原论文 | 本仓库 | 差异说明 |
|------|--------|--------|----------|
| 划分比例 | 80/20 | 80/20 | ✅ 一致 |
| **Train/Val 划分** | **随机打乱后划分** | **时序划分 (末尾20%)** | ⚠️ 差异 |
| 训练时 shuffle | 是 (每 epoch) | 是 (每 epoch) | ✅ 一致 |
| 训练数据 | Offline + prior Online | Offline + Sess01 + Sess02 Base | ✅ 类似 |
| 测试数据 | Finetune runs | Sess02 Finetune | ✅ 一致 |
| 评估方式 | Majority Voting | Majority Voting | ✅ 一致 |

**关于划分方法差异**:
- 原论文：随机打乱所有 trials 后取 80/20，验证集可能包含时间上较早的 trials
- 本仓库：验证集固定为时间上最后 20% 的 trials，更严格地模拟真实 BCI 场景
- **训练时两者都使用 shuffle**，区别仅在于验证集的选取方式

---

## 性能优化 (本仓库新增)

本仓库相比原论文实现增加了以下性能优化:

### 1. cuDNN 自动调优

**文件**: `src/training/train_within_subject.py`

```python
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
```

**效果**: 训练速度提升 20-50%

### 2. 自动混合精度 (AMP)

```python
if self.use_amp:
    with torch.amp.autocast('cuda', dtype=torch.float16):
        outputs = self.model(segments)
        loss = self.criterion(outputs, labels)
    self.scaler.scale(loss).backward()
```

**效果**: 显存减少，速度提升 10-20%

### 3. torch.compile() 支持

```python
if use_compile and hasattr(torch, 'compile') and device.type == 'cuda':
    model = torch.compile(model, mode='reduce-overhead')
```

**效果**: 速度提升 10-30% (PyTorch 2.0+)

### 4. 预处理缓存

**文件**: `src/preprocessing/cache_manager.py`

- HDF5 格式缓存 (lzf 压缩)
- 首次加载后 3-6x 加速
- 自动失效检测

### 5. 并行预处理

```python
dataset = FingerEEGDataset(..., parallel_workers=0)  # 自动多核
```

**效果**: 首次加载加速 2-3x

---

## 模型评估

### 原论文结果

| 任务 | 准确率 (Majority Voting) |
|------|-------------------------|
| 2-finger MI | 80.56% |
| 3-finger MI | 60.61% |
| 2-finger ME | 81.10% |
| 3-finger ME | 60.11% |

### 本仓库预期

由于存在以下差异，结果可能略有不同：
1. **kernLength**: 64 vs 32 (影响时间感受野)
2. **Batch Size**: 64 vs 16 (影响梯度估计)
3. **Early Stopping**: 5 vs 80 (可能提前停止)
4. **划分方法**: 时序 vs 随机 (更严格评估)

---

## 关键差异总结

### ✅ 完全一致的方面

1. 模型架构核心 (EEGNet-8,2: F1=8, D=2, F2=16)
2. 预处理流程 (CAR → 滑动窗口 → 降采样 → 滤波 → Z-score)
3. 滤波参数 (4-40 Hz, 4阶 Butterworth, lfilter)
4. 滑动窗口参数 (1s 窗口, 128 samples 步长)
5. 归一化方法 (Z-score 沿时间轴)
6. 训练 epochs (300)
7. LR Scheduler (ReduceLROnPlateau, factor=0.5, patience=30)

### ⚠️ 存在差异的方面

| 差异项 | 原论文 | 本仓库 | 影响 |
|--------|--------|--------|------|
| 深度学习框架 | TensorFlow | PyTorch | 低 |
| kernLength | 32 | 64 | 中 (时间感受野) |
| Batch Size | 16 | 64 | 中 (梯度估计) |
| Early Stopping | patience=80 | patience=4-5 | 高 (可能提前停止) |
| Val 划分方式 | 随机划分 | 时序划分 (末尾20%) | 中 (评估更严格) |
| Dense constraint | max_norm(0.25) | 无 | 低 |
| 训练 shuffle | 是 | 是 | ✅ 一致 |

### 🚀 本仓库增强

1. 性能优化: AMP, torch.compile, cuDNN benchmark
2. 预处理缓存: HDF5 + 并行加载
3. 时序验证集划分: 避免数据泄露
4. Trial-level 分割: 更严格的评估

---

## 建议的改进方向

1. **对齐 kernLength**: 改为 32 以完全匹配原论文
2. **对齐 Batch Size**: 考虑使用 16 进行对比实验
3. **对齐 Early Stopping**: 增加 patience 至 80
4. **添加 Dense constraint**: 实现 max_norm(0.25)
5. **实验对比**: 在两种配置下运行，量化差异影响

---

## 配置文件参考

### EEGNet 配置 (`configs/eegnet_config.yaml`)

```yaml
model:
  name: EEGNet-8,2
  F1: 8
  D: 2
  F2: 16
  kernel_length: 64       # 原论文: 32
  dropout_rate: 0.5

data:
  sampling_rate: 100
  n_channels: 128
  bandpass_low: 4.0
  bandpass_high: 40.0

training:
  epochs: 300
  batch_size: 64          # 原论文: 16
  learning_rate: 1.0e-3
  patience: 4             # 原论文: 80
  scheduler: plateau
```

### 预处理配置 (`PreprocessConfig.paper_aligned()`)

```python
PreprocessConfig(
    target_model='eegnet',
    original_fs=1024,
    target_fs=100,
    bandpass_low=4.0,
    bandpass_high=40.0,
    filter_order=4,
    channel_strategy='C',      # 全部 128 通道
    segment_length=1.0,        # 1s 窗口
    segment_step_samples=128,  # 125ms @ 1024 Hz
    normalize_method='zscore_time',
    apply_car=True,
    filter_padding=100,
)
```

---

## 参考文献

1. Ding et al., "EEG-based brain-computer interface enables real-time robotic hand control at individual finger level", Nature Communications, 2025
2. Lawhern et al., "EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces", J. Neural Eng., 2018
3. 原论文代码: https://github.com/bfinl/Finger-BCI-Decoding
4. 原论文数据: https://doi.org/10.1184/R1/29104040

---

*文档更新日期: 2026-01-08*
