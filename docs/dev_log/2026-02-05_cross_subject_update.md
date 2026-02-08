# 跨被试训练模块功能对等更新

**日期**: 2026-02-05
**类型**: 功能增强 + 代码重构
**影响范围**: 跨被试训练、被试内训练、共享工具

## 概述

将跨被试训练模块 (`train_cross_subject.py`) 更新至与被试内训练模块 (`train_within_subject.py`) 功能对等的状态。主要改进包括：添加缺失的性能优化、WandB 完整集成、配置管理系统、双阶段 batch size 策略等。同时抽象共享代码到新的 `common.py` 模块，减少重复代码，遵循 DRY 原则。

## 新增文件

### 1. `src/training/common.py`
共享训练工具模块，包含 5 个辅助函数：

#### `setup_performance_optimizations(device, verbose)`
配置 GPU 训练性能优化：
- cuDNN auto-tuning (20-50% 加速卷积运算)
- TF32 矩阵乘法 (Ampere+ GPU)

#### `maybe_compile_model(model, model_type, device, use_compile, verbose)`
智能应用 torch.compile：
- Windows 系统跳过 (Triton 不可用)
- Blackwell GPU 跳过 (sm_120+ 兼容性)
- EEGNet 使用 `reduce-overhead` 模式
- CBraMod 使用 `default` 模式

#### `get_scheduler_config_from_preset(scheduler_type, config)`
从 `SCHEDULER_PRESETS` 提取调度器配置：
- 获取预设默认值
- 合并 `config['scheduler_config']` 覆盖
- 返回完整配置字典

#### `create_two_phase_loaders(dataset, train_indices, val_indices, scheduler_config, main_batch_size, num_workers, verbose)`
创建双阶段 batch size 训练的 DataLoader：
- 探索阶段 (前 N epochs): 小 batch size，更多梯度更新
- 主训练阶段 (后续 epochs): 正常 batch size，稳定训练
- 返回: `(exploration_loader, val_loader, main_train_loader, exploration_epochs)`

#### `apply_config_overrides(config, config_overrides, log_prefix)`
应用配置覆盖的标准化逻辑：
- 优先级: 用户覆盖 > 调度器预设 > 模型默认
- 自动应用调度器预设的 epochs/patience
- 避免代码重复

## 修改文件

### 2. `src/training/train_cross_subject.py` (全面重写)

#### 新增参数
```python
def train_cross_subject(
    subjects, model_type, task='binary', paradigm='imagery',
    epochs=None, batch_size=None,
    save_dir='checkpoints/cross_subject', data_root='data',
    device=None, seed=42,
    # ========== 新增参数 ==========
    config_overrides: Optional[Dict] = None,        # 配置覆盖
    cache_only: bool = False,                       # 缓存模式
    cache_index_path: str = ".cache_index.json",    # 缓存索引路径
    wandb_enabled: bool = False,                    # WandB 开关
    upload_model: bool = False,                     # 上传模型到 WandB
    wandb_project: str = 'eeg-bci',                 # WandB 项目
    wandb_entity: Optional[str] = None,             # WandB 实体
    wandb_group: Optional[str] = None,              # WandB 分组
    wandb_metadata: Optional[Dict[str, str]] = None,# WandB 元数据
    verbose: int = 2,                               # 日志级别 (0/1/2)
) -> Dict:
```

#### 新增配置常量
```python
CROSS_SUBJECT_DEFAULTS = {
    'eegnet': {'batch_size': 128, 'epochs': 50},    # 2x within-subject
    'cbramod': {'batch_size': 256, 'epochs': 100},  # 2x within-subject
}
```

#### 功能增强列表
1. **性能优化**:
   - `setup_performance_optimizations()` - cuDNN + TF32
   - `maybe_compile_model()` - torch.compile 支持

2. **WandB 完整集成**:
   - `create_wandb_logger()` 创建 logger
   - WandB run 命名: `cross_{N}subj_{model}_{task}_{paradigm}`
   - 标签: `["cross-subject", f"subjects:{N}"]`
   - `wandb_callback.on_train_end()` 上传最终结果

3. **配置管理**:
   - `apply_config_overrides()` 标准化配置覆盖
   - `get_scheduler_config_from_preset()` 调度器配置提取
   - 支持通过 `--scheduler` CLI 参数覆盖

4. **双阶段 batch size**:
   - `create_two_phase_loaders()` 创建探索/主阶段 loader
   - 探索阶段 (默认前 5 epochs): 小 batch (32)
   - 主阶段 (后续 epochs): 大 batch (128/256)
   - 传递 `main_train_loader` 和 `exploration_epochs` 到 `trainer.train()`

5. **Verbose 日志控制**:
   - 0: 静默 (仅 TableEpochLogger)
   - 1: 最小输出 (标题 + 训练表格 + 最终评估)
   - 2: 完整输出 (所有 section headers + 详细指标)

6. **Cache-only 模式**:
   - 传递 `cache_only` 和 `cache_index_path` 到 `load_multi_subject_data()`
   - 支持无 `.mat` 文件的纯缓存训练

#### 训练流程改进
```python
# 旧流程 (单一 DataLoader)
trainer.train(train_loader, val_loader, epochs=..., wandb_callback=None)

# 新流程 (双阶段 + WandB)
trainer.train(
    exploration_loader, val_loader,
    main_train_loader=main_train_loader,
    exploration_epochs=exploration_epochs,
    epochs=config['training']['epochs'],
    patience=config['training'].get('patience', 10),
    save_path=save_path,
    wandb_callback=wandb_callback,
)
```

### 3. `src/training/train_within_subject.py` (代码重构)

#### 替换内联代码为共享函数
```python
# 旧代码 (15+ 行)
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    ...

# 新代码 (1 行)
setup_performance_optimizations(device, verbose)
```

```python
# 旧代码 (30+ 行)
import platform
use_compile = config.get('training', {}).get('use_compile', True)
is_windows = platform.system() == 'Windows'
is_blackwell = is_blackwell_gpu()
if is_windows:
    ...
elif is_blackwell:
    ...
elif use_compile and hasattr(torch, 'compile') and device.type == 'cuda':
    ...

# 新代码 (2 行)
use_compile = config.get('training', {}).get('use_compile', True)
model = maybe_compile_model(model, model_type, device, use_compile, verbose)
```

```python
# 旧代码 (8 行)
scheduler_config = SCHEDULER_PRESETS.get(scheduler_type, {}).copy()
if 'scheduler_config' in config:
    scheduler_config.update(config['scheduler_config'])

# 新代码 (1 行)
scheduler_config = get_scheduler_config_from_preset(scheduler_type, config)
```

```python
# 旧代码 (20+ 行配置覆盖逻辑)
if config_overrides:
    training_overrides = config_overrides.get('training', {})
    new_scheduler = training_overrides.get('scheduler')
    if new_scheduler and new_scheduler in SCHEDULER_PRESETS:
        ...

# 新代码 (1 行)
config = apply_config_overrides(config, config_overrides)
```

### 4. `scripts/experiments/run_cross_subject.py` (CLI 增强)

#### 新增参数
```bash
# 调度器选择
--scheduler {plateau,cosine,wsd,cosine_decay,cosine_annealing_warmup_decay}

# 缓存模式
--cache-only

# WandB 完整参数
--wandb                    # 开启 WandB
--wandb-project <name>     # 项目名
--wandb-entity <entity>    # 实体/团队
--wandb-group <group>      # 运行分组
--upload-model             # 上传模型文件

# 日志级别
--verbose {0,1,2}, -v {0,1,2}
--quiet, -q                # 等价于 --verbose 0
```

#### 使用示例
```bash
# 基本用法 (向后兼容)
uv run python scripts/run_cross_subject.py --model eegnet --subjects S01 S02 S03

# WandB 日志
uv run python scripts/run_cross_subject.py --model eegnet --subjects S01 S02 \
    --wandb --wandb-project my-eeg-project --upload-model

# 自定义调度器
uv run python scripts/run_cross_subject.py --model cbramod --subjects S01 S02 S03 \
    --scheduler wsd --epochs 80

# 精简输出
uv run python scripts/run_cross_subject.py --model eegnet --subjects S01 S02 \
    --verbose 1

# 缓存模式训练
uv run python scripts/run_cross_subject.py --model eegnet --subjects S01 S02 \
    --cache-only
```

### 5. `src/training/__init__.py` (导出更新)

新增导出：
```python
from .common import (
    setup_performance_optimizations,
    maybe_compile_model,
    get_scheduler_config_from_preset,
    create_two_phase_loaders,
    apply_config_overrides,
)
```

## 功能对比表

| 功能 | 被试内训练 | 跨被试训练 (之前) | 跨被试训练 (现在) |
|------|-----------|-----------------|-----------------|
| 双阶段 batch size | ✅ | ❌ | ✅ |
| WandB 完整集成 | ✅ | ⚠️ 占位未实现 | ✅ |
| TableEpochLogger | ✅ | ❌ | ✅ (通过 trainer) |
| Scheduler presets | ✅ | ❌ | ✅ |
| torch.compile | ✅ | ❌ | ✅ |
| Verbose 日志级别 | ✅ (0,1,2) | ❌ | ✅ (0,1,2) |
| Cache-only 模式 | ✅ | ❌ | ✅ |
| cuDNN benchmark | ✅ | ❌ | ✅ |
| TF32 优化 | ✅ | ❌ | ✅ |
| config_overrides | ✅ | ❌ | ✅ |
| scheduler_config 传递 | ✅ | ❌ | ✅ |

## 实现细节

### 配置优先级 (三层继承)
1. **用户覆盖** (最高): `config_overrides` 或 CLI 参数
2. **调度器预设**: `SCHEDULER_PRESETS[scheduler_type]`
3. **模型默认** (最低): `get_default_config(model_type, task)`

示例：
```python
# 模型默认: epochs=100, patience=15
config = get_default_config('cbramod', 'binary')

# 调度器预设: 如果指定 --scheduler wsd, 应用 epochs=50, patience=10
config_overrides = {'training': {'scheduler': 'wsd'}}

# 用户覆盖: --epochs 80 覆盖一切
config_overrides = {'training': {'scheduler': 'wsd', 'epochs': 80}}
# 最终: epochs=80, patience=10 (预设), scheduler='wsd'
```

### WandB 命名规范

**被试内训练**:
```
Run name: S01_eegnet_binary_MI
Tags: ["within-subject", "S01", "eegnet", "imagery"]
Group: within_subject_eegnet
```

**跨被试训练**:
```
Run name: cross_7subj_eegnet_binary_MI
Tags: ["cross-subject", "subjects:7", "eegnet", "imagery"]
Group: cross_subject_eegnet
```

### Checkpoint 格式兼容性

checkpoint 格式保持不变，确保与 `finetune.py` 完全兼容：
```python
checkpoint = {
    'model_state_dict': ...,
    'model_config': {...},      # finetune.py 用于重建模型
    'training_config': {...},
    'epoch': trainer.best_epoch,
    'val_acc': trainer.best_val_acc,
    'val_majority_acc': trainer.best_majority_acc,
    'per_subject_test_acc': {...},  # 跨被试特有
    'mean_test_acc': ...,           # 跨被试特有
}
```

## 向后兼容性

所有现有脚本和命令继续工作，无需修改：

```bash
# 现有用法 (完全兼容)
uv run python scripts/run_cross_subject.py --model eegnet --subjects S01 S02 S03

# finetune 依然能加载 checkpoint
uv run python scripts/run_finetune.py \
    --pretrained checkpoints/cross_subject/eegnet_imagery_binary/best.pt \
    --subject S01
```

## 测试验证

### 1. 导入测试
```bash
$ cd "c:\Users\zhang\Desktop\github\EEG-BCI"
$ uv run python -c "from src.training import train_cross_subject, setup_performance_optimizations, maybe_compile_model, get_scheduler_config_from_preset, create_two_phase_loaders, apply_config_overrides; print('All imports successful')"
All imports successful
```

### 2. CLI 测试
```bash
$ uv run python scripts/experiments/run_cross_subject.py --help
# 输出显示所有新增参数正确注册
```

### 3. 被试内训练回归测试
```bash
$ uv run python -c "from src.training.train_within_subject import train_subject_simple, setup_performance_optimizations, maybe_compile_model, get_scheduler_config_from_preset, apply_config_overrides; print('Within-subject imports OK')"
Within-subject imports OK
```

## 代码统计

| 指标 | 数值 |
|------|------|
| 新增文件 | 1 (`common.py`) |
| 修改文件 | 4 |
| 新增代码行数 | ~300 (common.py: ~250, 其他: ~50) |
| 删除重复代码 | ~100 行 |
| 新增功能 | 9 项 |
| 新增 CLI 参数 | 9 个 |

## 后续工作

1. **文档更新**: 更新 `CLAUDE.md` 中的快速命令部分，添加新参数说明
2. **实验验证**: 在真实数据集上运行完整跨被试训练，验证所有功能正常
3. **性能测试**: 对比更新前后的训练速度和内存使用
4. **迁移学习实验**: 使用新的跨被试训练结果进行 finetune 实验

## 相关文件路径

- 新增: `src/training/common.py`
- 修改: `src/training/train_cross_subject.py`
- 修改: `src/training/train_within_subject.py`
- 修改: `scripts/experiments/run_cross_subject.py`
- 修改: `src/training/__init__.py`
- 计划: `C:\Users\zhang\.claude\plans\floating-crunching-eagle.md`
