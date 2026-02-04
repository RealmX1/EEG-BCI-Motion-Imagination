# Phase 2 Implementation Log

**Date**: 2026-02-05
**Status**: Completed
**Goal**: 拆分大文件 (`data_loader.py` 和 `train_within_subject.py`)

---

## Overview

Phase 2 是代码重构计划中的高风险阶段，目标是将两个超过 2000 行的核心文件拆分为更小、更专注的模块，同时保持完全的向后兼容性。

### 拆分前状态

| 文件 | 行数 | 问题 |
|------|------|------|
| `src/preprocessing/data_loader.py` | 2420 | MAT 加载、发现、管线、Dataset 类混在一起 |
| `src/training/train_within_subject.py` | 2462 | 调度器、评估、训练器、配置混在一起 |

---

## Phase 2.1: 拆分 `data_loader.py`

### 新建文件

#### 1. `src/preprocessing/loader.py` (~130 行)

MAT 文件加载工具。

```python
# 主要函数
def load_mat_file(mat_path: str) -> Tuple[np.ndarray, List[Dict], Dict]
def parse_session_path(path: Path) -> Dict
```

**职责**:
- 从 FINGER-EEG-BCI 数据集加载 `.mat` 文件
- 解析文件路径提取 session 信息

#### 2. `src/preprocessing/discovery.py` (~180 行)

被试和 session 发现工具。

```python
# 主要函数
def get_session_folders_for_split(paradigm: str, task: str, split: str) -> List[str]
def discover_available_subjects(data_root: str, paradigm: str, task: str) -> List[str]
def discover_subjects_from_cache_index(cache_index_path: str, paradigm: str, task: str) -> List[str]
```

**职责**:
- 根据论文协议确定 train/test 的 session folders
- 发现可用被试（从文件系统或缓存索引）

#### 3. `src/preprocessing/pipeline.py` (~550 行)

预处理管线函数。

```python
# 主要类和函数
@dataclass
class TrialInfo

def apply_common_average_reference(data: np.ndarray) -> np.ndarray
def segment_with_sliding_window(...) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
def apply_bandpass_filter_paper(...) -> np.ndarray
def apply_zscore_per_segment(data: np.ndarray, axis: int) -> np.ndarray
def preprocess_run_paper_aligned(...) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
def preprocess_run_to_trials(...) -> Tuple[np.ndarray, np.ndarray]
def trials_to_segments(...) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
def _process_single_mat_file(...) -> Tuple  # 用于并行处理
def _process_single_mat_file_to_trials(...) -> Tuple
```

**职责**:
- 实现论文的预处理管线 (CAR, 滑动窗口, 下采样, 滤波, 归一化)
- 支持 EEGNet 和 CBraMod 的不同配置

#### 4. `src/preprocessing/dataset.py` (~700 行)

PyTorch Dataset 实现。

```python
# 主要类和函数
class FingerEEGDataset(Dataset)
def create_dataloaders(...) -> Tuple[DataLoader, DataLoader, DataLoader]
```

**职责**:
- `FingerEEGDataset`: 主要的 PyTorch Dataset 类
- 支持缓存加载、并行预处理、两阶段 batch size
- `create_dataloaders`: 创建 train/val/test DataLoader

### 修改的文件

#### `src/preprocessing/data_loader.py` (~350 行)

变为向后兼容层：
- 保留 `PreprocessConfig` 类（核心配置类）
- 重导出所有从新模块移出的符号
- 保留模块测试代码

```python
# 重导出示例
from .loader import load_mat_file, parse_session_path
from .discovery import get_session_folders_for_split, discover_available_subjects
from .pipeline import preprocess_run_paper_aligned, trials_to_segments, TrialInfo
from .dataset import FingerEEGDataset, create_dataloaders
```

#### `src/preprocessing/__init__.py`

更新导出列表，添加新模块的符号。

---

## Phase 2.2: 拆分 `train_within_subject.py`

### 新建文件

#### 1. `src/training/schedulers.py` (~650 行)

自定义学习率调度器。

```python
# 主要类和函数
class WSDScheduler  # Warmup-Stable-Decay
class CosineDecayRestarts  # 带衰减峰值的余弦重启
class CosineAnnealingWarmupDecay  # 多阶段余弦 + LR ramp-up

def visualize_lr_schedule(scheduler_config: Dict, base_lr: float, ...) -> Optional[Path]
```

**职责**:
- `WSDScheduler`: 四阶段 (warmup-stable-decay-minimum) 调度器
- `CosineDecayRestarts`: 每个周期峰值衰减的余弦重启
- `CosineAnnealingWarmupDecay`: 基于 epoch 的多阶段调度器，每阶段含 LR ramp-up
- `visualize_lr_schedule`: 生成 LR 曲线可视化

#### 2. `src/training/evaluation.py` (~100 行)

评估工具。

```python
# 主要函数
def majority_vote_accuracy(
    model: nn.Module,
    dataset: FingerEEGDataset,
    indices: List[int],
    device: torch.device,
    batch_size: int = 128,
    use_amp: bool = True,
) -> Tuple[float, Dict]
```

**职责**:
- 实现论文的 majority voting 评估方法
- 每个 trial 的多个 segment 预测进行投票

#### 3. `src/training/trainer.py` (~550 行)

训练器类。

```python
# 主要类
class WithinSubjectTrainer:
    def __init__(self, model, dataset, val_indices, device, ...)
    def train_epoch(self, dataloader, epoch, profile) -> Tuple[float, float]
    def validate(self, dataloader) -> Tuple[float, float]
    def train(self, train_loader, val_loader, ...) -> Dict
    def freeze_early_layers(self)
```

**职责**:
- 完整的训练循环，支持 early stopping
- AMP 混合精度训练
- 两阶段 batch size 策略 (exploration + main)
- 多种调度器支持

### 修改的文件

#### `src/training/train_within_subject.py`

保留核心训练函数，添加重导出：

```python
# 重导出
from .schedulers import WSDScheduler, CosineDecayRestarts, CosineAnnealingWarmupDecay
from .evaluation import majority_vote_accuracy
from .trainer import WithinSubjectTrainer
from ..config.training import SCHEDULER_PRESETS, get_default_config

# 保留的函数
def load_subject_data(...)
def create_data_loaders_from_dataset(...)
def get_task_type_patterns(...)
def train_single_subject(...)  # 主训练函数
def train_subject_simple(...)  # 简化 API
```

#### `src/training/__init__.py`

更新导出列表，添加新模块的符号。

---

## 验证结果

### 导入测试

```
[Phase 2.1] data_loader.py split
  src.preprocessing.loader: OK
  src.preprocessing.discovery: OK
  src.preprocessing.pipeline: OK
  src.preprocessing.dataset: OK
  src.preprocessing.data_loader (backward compat): OK

[Phase 2.2] train_within_subject.py split
  src.training.schedulers: OK
  src.training.evaluation: OK
  src.training.trainer: OK
  src.training.train_within_subject (backward compat): OK
```

### 脚本测试

```
run_single_model.py --help: OK
run_full_comparison.py --help: OK
```

---

## 文件统计

### 新建文件

| 文件 | 行数 | 模块 |
|------|------|------|
| `src/preprocessing/loader.py` | ~130 | preprocessing |
| `src/preprocessing/discovery.py` | ~180 | preprocessing |
| `src/preprocessing/pipeline.py` | ~550 | preprocessing |
| `src/preprocessing/dataset.py` | ~700 | preprocessing |
| `src/training/schedulers.py` | ~650 | training |
| `src/training/evaluation.py` | ~100 | training |
| `src/training/trainer.py` | ~550 | training |

### 修改的文件

| 文件 | 变化 |
|------|------|
| `src/preprocessing/data_loader.py` | 2420 → ~350 行 (向后兼容层) |
| `src/preprocessing/__init__.py` | 更新导出 |
| `src/training/train_within_subject.py` | 添加重导出，保留核心函数 |
| `src/training/__init__.py` | 更新导出 |

---

## 向后兼容性

所有现有导入路径继续工作：

```python
# 旧导入方式 - 继续有效
from src.preprocessing.data_loader import FingerEEGDataset, PreprocessConfig
from src.training.train_within_subject import train_subject_simple, SCHEDULER_PRESETS

# 新导入方式 - 推荐
from src.preprocessing.dataset import FingerEEGDataset
from src.preprocessing.pipeline import preprocess_run_paper_aligned
from src.training.schedulers import CosineAnnealingWarmupDecay
from src.config.training import SCHEDULER_PRESETS
```

---

## 后续阶段

- **Phase 3**: 精简脚本层 (`run_full_comparison.py`, `run_single_model.py`)
- **Phase 4**: 脚本目录重组 (`experiments/`, `preprocessing/`, `tools/`, `analysis/`)

详见计划文件: `C:\Users\zhang\.claude\plans\fuzzy-prancing-parnas.md`
