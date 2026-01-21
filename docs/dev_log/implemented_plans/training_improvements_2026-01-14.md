# 训练系统改进 (2026-01-14)

本文档记录训练系统的多项改进，包括日志可视化、数据路径解析、GPU 兼容性等。

## 概述

| 改进项 | 涉及文件 | 说明 |
|--------|----------|------|
| 表格式训练日志 | `src/utils/table_logger.py` | 统一的可视化训练输出 |
| 数据路径自动解析 | `src/preprocessing/data_loader.py` | 支持跨仓库数据共享 |
| GPU 兼容性 | `src/training/train_within_subject.py` | Blackwell + TF32 优化 |

---

## 1. 表格式训练日志 (TableEpochLogger)

### 背景

原有训练日志分散且格式不统一，难以快速定位训练状态。新的 `TableEpochLogger` 提供统一的表格式输出。

### 特性

- **Train/Val 并排显示**: Loss 和 Accuracy 同时展示训练/验证值
- **条件着色**:
  - 改进时显示 `↑` (绿色)
  - 退步时显示 `↓` (红色)
  - Best epoch 高亮显示
- **动态覆盖 + 周期保留**:
  - 默认覆盖上一行（减少屏幕滚动）
  - 每 5 个 epoch 保留一行
  - Best epoch 和事件自动保留
- **进度条 + ETA**: 实时显示训练进度和预计剩余时间

### 输出示例

```
Training: S01 | CBraMod | Binary                              [GPU: cuda:0]
Progress: [████████████████████░░░░░░░░░░] 67% (20/30) ETA: 23s

 Epoch │    Loss (T/V)     │     Acc (T/V)     │ Maj Acc │    LR    │ Time
───────┼───────────────────┼───────────────────┼─────────┼──────────┼──────
     1 │  0.6932 /  0.7012 │  0.5234 /  0.5012 │  0.4923 │ 1.00e-03 │ 2.3s
     5 │  0.4521 /  0.4789 │  0.7123 /  0.6890 │  0.6512 │ 9.50e-04 │ 2.2s
 * 12  │  0.2456 /  0.2987 │  0.8567 / ↑0.8234 │ ↑0.8123 │ 7.00e-04 │ 2.3s  BEST
    15 │  0.1987 /  0.2654 │  0.8901 /  0.8156 │  0.8056 │ 5.00e-04 │ 2.2s
    20 │  0.1234 /  0.2456 │  0.9234 /  0.8123 │  0.7989 │ 3.00e-04 │ 2.3s  STOP

────────────────────────────────────────────────────────────────────────────
Training Complete | Best: Epoch 12 | Val Acc: 0.8234 | Maj: 0.8123 | 46.2s
```

### API

```python
from src.utils.table_logger import TableEpochLogger

logger = TableEpochLogger(
    total_epochs=100,
    model_name="CBraMod",
    task_name="Binary",
    subject="S01",
    device="cuda:0",
    keep_every=5,       # 每 N 个 epoch 保留一行
    header_every=25,    # 每 N 行重新打印表头
    show_majority=True, # 显示 Majority Voting Accuracy
)

# 每个 epoch 结束时调用
logger.on_epoch_end(
    epoch=epoch,
    train_loss=train_loss,
    train_acc=train_acc,
    val_loss=val_loss,
    val_acc=val_acc,
    majority_acc=majority_acc,  # 可选
    lr=current_lr,              # 可选
    epoch_time=elapsed,         # 可选
    is_best=is_best,
    event="STOP",               # 可选: "BEST", "STOP", "LR↓" 等
)

# 训练结束时打印总结
logger.print_summary()
```

---

## 2. 数据路径自动解析

### 背景

项目可能与 `Finger-BCI-Decoding` 仓库共享数据目录。新增 `resolve_data_root()` 函数自动查找数据位置。

### 实现

```python
def resolve_data_root(default_path: Path) -> Path:
    """
    解析数据根目录，支持回退到兄弟仓库。

    查找顺序:
    1. default_path (通常是 PROJECT_ROOT/data)
    2. ../Finger-BCI-Decoding/data (兄弟仓库)

    验证条件: 目录存在且包含 S01/ 子目录
    """
```

### 使用场景

```
~/Documents/github/
├── EEG-BCI-Motion-Imagination/    # 本项目 (无 data/)
│   └── ...
└── Finger-BCI-Decoding/           # 兄弟仓库
    └── data/                      # 共享数据
        ├── S01/
        ├── S02/
        └── ...
```

### 集成位置

- `train_within_subject.py`: 训练脚本自动查找数据
- `run_full_comparison.py`: 全被试对比脚本

---

## 3. GPU 兼容性改进

### 3.1 Blackwell 架构支持 (RTX 50 系列)

**问题**: RTX 5070/5080/5090 (sm_120 架构) 在使用 `torch.compile()` 时存在 CUDA Graph 兼容性问题。

**解决方案**: 检测 Blackwell GPU 并跳过编译。

```python
if device.type == 'cuda':
    compute_cap = torch.cuda.get_device_capability(device)
    is_blackwell = compute_cap[0] >= 12  # sm_120+

if is_blackwell:
    print_metric("torch.compile", "skipped (Blackwell GPU)", Colors.DIM)
```

### 3.2 TF32 矩阵乘法加速

**优化**: 在 Ampere+ GPU (RTX 30xx/40xx/50xx) 上启用 TensorFloat-32 精度。

```python
torch.set_float32_matmul_precision('high')
```

**效果**: 矩阵乘法约 2x 加速，精度损失可忽略（FP32 mantissa 23→10 bits）。

---

## 4. 其他改进

### 4.1 Majority Voting 计算频率优化

**原行为**: 每 10 个 epoch 计算一次 Majority Voting Accuracy。

**新行为**: 每 5 个 epoch 计算，或在 `val_acc` 改进时立即计算。

```python
compute_majority = (
    (displayed_epoch == 1) or
    (displayed_epoch % 5 == 0) or
    (val_acc > self.best_val_acc)
)
```

### 4.2 Early Stopping 基于 Accuracy

**修改**: Early stopping 判断从 `val_loss` 改为 `val_acc`，更直接反映模型性能。

```python
# 保存最佳模型的条件
if val_acc > self.best_val_acc:
    self.best_val_acc = val_acc
    self.best_epoch = epoch + 1
    no_improve = 0
    # Save checkpoint...
else:
    no_improve += 1
```

---

## 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `src/utils/table_logger.py` | 新增 | 表格式训练日志器 |
| `src/preprocessing/data_loader.py` | 修改 | 新增 `resolve_data_root()` |
| `src/training/train_within_subject.py` | 修改 | 集成 TableEpochLogger + GPU 优化 |
| `scripts/run_full_comparison.py` | 修改 | 集成 `resolve_data_root()` |

---

## 验证命令

```bash
# 测试表格式日志输出
uv run python -m src.training.train_within_subject \
    --subject S01 --task binary --model cbramod

# 验证数据路径解析 (查看日志中的 "Data root:" 行)
uv run python -m src.training.train_within_subject \
    --subject S01 --task binary --model eegnet --dry-run
```
