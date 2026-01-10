# CBraMod 128 通道适配方案

## 背景

### 当前实现问题

当前 CBraMod 实现将 BioSemi 128 通道数据映射到标准 10-20 系统的 19 通道：

```python
# src/preprocessing/data_loader.py
if config.channel_strategy == 'A' and elc_path:
    mapping = create_biosemi128_to_1020_mapping(elc_path)
    idx_map = get_channel_indices(mapping)
    self.channel_indices = [idx_map[ch] for ch in STANDARD_1020_CHANNELS]
```

**问题**: 这导致丢失了 109 个通道（85%）的信息，特别是对于手指运动解码这种需要精细空间分辨率的任务。

### 论文关键发现

根据 CBraMod 论文 (ICLR 2025) 分析：

1. **ACPE 卷积核**：固定尺寸 `(19, 7)`，通过 `padding=(9, 3)` 可处理**任意通道数**
2. **Patch Encoder**：独立处理每个 patch，不依赖通道数
3. **Transformer**：将通道视为 "token 数量" 而非 "特征深度"，天然支持变长序列

**官方代码验证** (`CBraMod/models/cbramod.py:117`)：
```python
a = torch.randn((8, 16, 10, 200)).cuda()  # 16 通道也能工作！
b = model(a)
```

## 修改方案

### 方案概述

| 组件 | 当前状态 | 修改后 |
|------|----------|--------|
| 通道数 | 19 (10-20 系统) | 128 (全部 BioSemi) |
| 预处理 | `channel_strategy='A'` | `channel_strategy='C'` |
| ACPE | 无需修改 | 无需修改 (卷积自动适配) |
| Patch Encoder | 无需修改 | 无需修改 |
| Classifier | `input_dim = 19 * n_patches * 200` | `input_dim = 128 * n_patches * 200` |

### 详细修改清单

#### 1. 数据预处理 (`src/preprocessing/data_loader.py`)

**新增配置方法**：

```python
@classmethod
def for_cbramod_128ch(cls, use_sliding_window: bool = True) -> 'PreprocessConfig':
    """
    CBraMod 配置，使用全部 128 通道。

    利用 ACPE 的通道适配能力，保留原始空间分辨率。
    """
    return cls(
        target_model='cbramod_128ch',
        original_fs=1024,
        target_fs=200,
        bandpass_low=0.3,
        bandpass_high=75.0,
        notch_freq=60.0,
        normalize_method='divide',
        normalize_value=100.0,
        apply_car=False,
        use_sliding_window=use_sliding_window,
        segment_length=1.0,
        segment_step=0.125,
        channel_strategy='C',  # 使用全部 128 通道
    )
```

**修改 `for_cbramod()` 默认行为**（可选）：

```python
@classmethod
def for_cbramod(cls, use_sliding_window: bool = True, full_channels: bool = False) -> 'PreprocessConfig':
    """
    Args:
        full_channels: 若为 True，使用全部 128 通道；否则使用 19 通道 (10-20)
    """
    channel_strategy = 'C' if full_channels else 'A'
    # ...
```

#### 2. 模型适配器 (`src/models/cbramod_adapter.py`)

**修改 `CBraModForFingerBCI.__init__`**：

```python
def __init__(
    self,
    n_channels: int = 19,  # 支持 19 或 128
    n_patches: int = 5,
    n_classes: int = 2,
    # ... 其他参数
):
    # Classifier 输入维度自动适配
    classifier_input_dim = n_channels * n_patches * d_model
    # n_channels=128 时: 128 * 1 * 200 = 25,600
    # n_channels=19 时:  19 * 1 * 200 = 3,800
```

**无需修改的部分**：
- `get_cbramod_model()` - 官方模型加载
- `PatchEmbedding` - 独立处理每个 patch
- ACPE 卷积 - `padding=(9, 3)` 自动处理任意高度

#### 3. 训练脚本 (`src/training/train_within_subject.py`)

**添加命令行参数**：

```python
parser.add_argument(
    '--cbramod-channels',
    type=int,
    choices=[19, 128],
    default=19,
    help='CBraMod 使用的通道数 (19=10-20系统, 128=全部)'
)
```

**修改模型创建逻辑**：

```python
if model_name == 'cbramod':
    if args.cbramod_channels == 128:
        config = PreprocessConfig.for_cbramod_128ch()
    else:
        config = PreprocessConfig.for_cbramod()

    model = CBraModForFingerBCI(
        n_channels=config.n_channels,  # 从配置获取
        n_patches=n_patches,
        n_classes=n_classes,
        # ...
    )
```

#### 4. 配置文件 (`configs/cbramod_config.yaml`)

**添加新配置选项**：

```yaml
data:
  # 通道配置
  channel_mode: "128"  # "19" (10-20) 或 "128" (全部)
  n_channels: 128      # 与 channel_mode 一致
  sampling_rate: 200

  # 其他设置保持不变
```

### ACPE 卷积适配原理

```
输入: [batch, 128, n_patches, 200]
       ↓
Patch Embedding: [batch, 128, n_patches, 200] (d_model=200)
       ↓
ACPE 卷积:
  - kernel: (19, 7), padding: (9, 3)
  - 输入视角: [batch, 200, 128, n_patches] (permute 后)
  - 在 128 通道上滑动 19x7 窗口
  - 输出: [batch, 200, 128, n_patches]
       ↓
Transformer (Criss-Cross Attention):
  - S-Attention: 128 个 token 两两计算注意力
  - T-Attention: n_patches 个时间步两两计算注意力
       ↓
Classifier:
  - 输入: 128 * n_patches * 200 = 25,600 维
  - 输出: n_classes
```

### 内存与计算量估算

| 配置 | Classifier 输入维度 | Attention Map 大小 | 相对内存 |
|------|---------------------|-------------------|----------|
| 19 通道 | 3,800 | 19×19 = 361 | 1x |
| 128 通道 | 25,600 | 128×128 = 16,384 | ~6.7x |

**建议**: 对于 128 通道，可能需要：
- 减小 batch_size（从 128 → 32-64）
- 使用梯度检查点（gradient checkpointing）

## 实验设计

### 对比实验

| 实验 | CBraMod 通道 | EEGNet 通道 | 目的 |
|------|-------------|-------------|------|
| Baseline | 19 | 128 | 当前基线 |
| CBraMod-128 | 128 | 128 | 测试 128 通道效果 |
| CBraMod-128 vs EEGNet | 128 | 128 | 公平对比 |

### 假设

1. **H1**: CBraMod-128 > CBraMod-19（更多空间信息）
2. **H2**: CBraMod-128 ≈ EEGNet-128（公平对比下）
3. **H3**: 微调后的 ACPE 能学习到适合 128 通道的位置编码

## 实施步骤

```bash
# Step 1: 添加 128 通道预处理配置
# 修改 src/preprocessing/data_loader.py

# Step 2: 清理旧缓存
uv run python scripts/cache_helper.py --model cbramod --execute

# Step 3: 测试新配置
uv run python -c "
from src.preprocessing.data_loader import PreprocessConfig
config = PreprocessConfig.for_cbramod_128ch()
print(f'通道数: {config.channel_strategy}')
"

# Step 4: 单被试测试训练
uv run python -m src.training.train_within_subject \
    --subject S01 --task binary --model cbramod \
    --cbramod-channels 128

# Step 5: 全被试对比实验
uv run python scripts/run_full_comparison.py \
    --models cbramod --cbramod-channels 128
```

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 显存不足 | 训练失败 | 减小 batch_size，使用梯度累积 |
| 预训练权重不适配 | 性能下降 | 全参数微调（论文推荐） |
| 位置编码外推失败 | 边缘通道信息丢失 | 检查 ACPE padding 策略 |

## 附录：ACPE 卷积核分析

```python
# 来自 CBraMod/models/cbramod.py
self.positional_encoding = nn.Conv2d(
    in_channels=d_model,      # 200
    out_channels=d_model,     # 200
    kernel_size=(19, 7),      # 空间×时间
    stride=(1, 1),
    padding=(9, 3),           # 保持输出尺寸不变
    groups=d_model            # Depthwise 卷积
)
```

**Padding 计算**：
- 空间维度: `padding = (19-1)/2 = 9` → 输入 128，输出 128
- 时间维度: `padding = (7-1)/2 = 3` → 输入 n_patches，输出 n_patches

**结论**: ACPE 的设计天然支持任意通道数，无需修改卷积核。
