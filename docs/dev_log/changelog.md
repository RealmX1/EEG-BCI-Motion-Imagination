# 开发变更记录

## 2026-02-05

### WandB 集成默认启用

**设计改进**: 将 WandB 集成从 opt-in（需要 `--wandb` 启用）改为 opt-out（默认启用，需要 `--no-wandb` 禁用）

**修改文件**:
- `scripts/experiments/run_cross_subject.py`: 将 `--wandb` 改为 `--no-wandb`
- `CLAUDE.md`: 更新示例命令

**理由**:
- 与 `run_single_model.py` 和 `run_full_comparison.py` 保持一致
- 默认启用实验追踪是现代 ML 最佳实践
- 减少遗漏实验记录的风险

**使用示例**:
```bash
# 默认启用 WandB (无需额外参数)
uv run python scripts/run_cross_subject.py --model cbramod --subjects S01 S02

# 禁用 WandB
uv run python scripts/run_cross_subject.py --model cbramod --subjects S01 S02 --no-wandb
```

---

### 跨被试训练模块功能对等更新

将跨被试训练模块更新至与被试内训练功能对等的状态，并抽象共享代码以减少重复。

**新增文件**:
- `src/training/common.py`: 共享训练工具模块（5 个辅助函数）

**修改文件**:
- `src/training/train_cross_subject.py`: 新增 9 项功能，全面重写
- `src/training/train_within_subject.py`: 使用共享函数替换重复代码
- `scripts/experiments/run_cross_subject.py`: 新增 9 个 CLI 参数
- `src/training/__init__.py`: 导出共享工具函数

**新增功能** (跨被试训练):
1. 双阶段 batch size 策略 (探索阶段 + 主阶段)
2. WandB 完整集成 (项目、实体、分组、模型上传)
3. Scheduler presets 支持
4. torch.compile 支持 (智能跳过不兼容平台)
5. Verbose 日志级别控制 (0=静默, 1=最小, 2=完整)
6. Cache-only 模式
7. cuDNN benchmark + TF32 优化
8. config_overrides 参数
9. scheduler_config 传递到 trainer

**共享工具函数** (`common.py`):
- `setup_performance_optimizations()`: cuDNN + TF32 配置
- `maybe_compile_model()`: torch.compile 智能应用
- `get_scheduler_config_from_preset()`: 调度器配置提取
- `create_two_phase_loaders()`: 双阶段 DataLoader 创建
- `apply_config_overrides()`: 标准化配置覆盖逻辑

**运行命令**:
```bash
# 基本用法 (向后兼容)
uv run python scripts/run_cross_subject.py --model eegnet --subjects S01 S02 S03

# WandB 日志 + 自定义调度器
uv run python scripts/run_cross_subject.py --model cbramod --subjects S01 S02 S03 \
    --wandb --scheduler wsd --upload-model

# 精简输出
uv run python scripts/run_cross_subject.py --model eegnet --subjects S01 S02 \
    --verbose 1 --cache-only
```

**详细文档**: `docs/dev_log/2026-02-05_cross_subject_update.md`

## 2026-01-25

### CBraMod 预处理 ML Engineering 实验框架

实现系统性评估不同预处理参数对 CBraMod 性能影响的实验框架。

**新增文件**:
- `src/preprocessing/experiment_config.py`: 实验配置类，定义 15 个实验配置
- `scripts/run_preproc_experiment.py`: 实验执行脚本
- `scripts/compile_preproc_report.py`: 报告生成脚本（统计分析 + 可视化）

**修改文件**:
- `src/preprocessing/cache_manager.py`: 支持 experiment_tag 参数，实验数据独立缓存
- `src/preprocessing/data_loader.py`: 新增 `extra_normalize` 字段和 `from_experiment()` 工厂方法
- `src/utils/wandb_logger.py`: 支持实验元数据和标签

**实验设计**:
- A 组 (6 配置): 滤波参数（带通、陷波）
- C 组 (4 配置): 归一化策略（额外 z-score、robust 等）
- D 组 (3 配置): 滑动窗口步长
- F 组 (2 配置): 数据质量控制阈值

**固定参数** (CBraMod 论文约束):
- 采样率: 200 Hz
- Patch 长度: 1 秒
- 归一化: ÷100 (强制)
- 通道数: 128

**运行命令**:
```bash
uv run python scripts/run_preproc_experiment.py --list       # 列出配置
uv run python scripts/run_preproc_experiment.py --prototype  # 原型验证
uv run python scripts/run_preproc_experiment.py --all        # 完整实验
uv run python scripts/compile_preproc_report.py              # 生成报告
```

### 训练框架重构

将 `train_within_subject.py` 从 CLI 脚本重构为 API 模块。

**主要变更**:
- 移除 `main()` 函数和 `argparse` CLI 代码
- 新增 `train_subject_simple()` 简化 API，用于程序调用
- 新增 `get_default_config()` 函数，集中管理默认配置

**训练参数调整**:
- CBraMod epochs: 50 → 30 (配合 WSD 调度器快速收敛)
- EEGNet epochs: 300 → 30 (实验发现早期收敛，配合 early stopping)
- Early stopping patience: 统一为 5 (之前 EEGNet 是 20)

### WSD (Warmup-Stable-Decay) 学习率调度器

新增 CBraMod 论文原生的学习率调度策略。

**实现特点**:
- 三阶段调度: warmup (10%) → stable (50%) → decay (40%)
- 线性 warmup + 恒定 + cosine decay
- 支持状态保存/恢复
- 默认用于 CBraMod 训练

**Combined Score 模型选择**:
- 综合 segment 准确率和 majority voting 准确率
- 公式: `0.7 * val_acc + 0.3 * majority_val_acc`
- 更稳定的最佳模型选择

### CosineDecayRestarts 学习率调度器

新增带递减峰值的 cosine warm restarts 调度器。

**问题背景**:
- PyTorch 原生 `CosineAnnealingLR` 在周期结束后会恢复到相同的初始 LR
- 当 `T_max = total_steps // 5` 时，训练后期 (80%) LR 会突然跳回初始值
- 这可能破坏已学习的特征，导致训练不稳定

**解决方案**:
- `CosineDecayRestarts` 调度器在每个周期后按 `decay_factor` 递减峰值 LR
- 默认 `decay_factor=0.7`，每周期峰值减少 30%

**LR 递减示例** (5 个周期):
| 周期 | 峰值 LR |
|------|---------|
| 0 | 1.0e-4 |
| 1 | 7.0e-5 (-30%) |
| 2 | 4.9e-5 (-30%) |
| 3 | 3.4e-5 (-30%) |
| 4 | 2.4e-5 (-30%) |

**使用方式**:
```python
# 在 get_default_config() 或 config_overrides 中设置
config['training']['scheduler'] = 'cosine_decay'
```

### 彩色日志系统

增强训练过程的可读性。

**新增组件**:
- `TableEpochLogger`: 表格式 epoch 日志，自动颜色编码
- `ColoredFormatter`: 通用彩色日志格式器
- 保留 `YellowFormatter` 别名，向后兼容

### 配置文件弃用

YAML 配置文件转为硬编码默认值。

**弃用文件**:
- `configs/cbramod_config.yaml` → `.deprecated`
- `configs/eegnet_config.yaml` → `.deprecated`
- `configs/experiment_config.yaml` → 删除

**设计决策**:
- 简化部署，避免配置与代码不同步
- 配置通过 `get_default_config()` 和函数参数覆盖

## 2026-01-11

### 增量缓存加载修复
- 修复 `run_full_comparison.py` 缓存加载逻辑
- 新增 `find_latest_cache()` 函数，自动查找最新缓存文件（无论是否有标签）
- 不使用 `--new-run` 时，自动加载最新缓存并仅训练新被试
- 添加 UTF-8 编码到所有文件操作

### 文档重构
- 精简 `CLAUDE.md`，移除过多实现细节
- 新增 `docs/preprocessing_architecture.md` 详细架构文档
- 整理 bug 修复记录到 `docs/dev_log/bugfixes/`

## 2026-01-10

### CBraMod 128 通道支持
- 利用 ACPE（非对称条件位置编码）支持任意通道数
- 新增 `--cbramod-channels` 参数 (19 或 128)
- 128 通道模式与 EEGNet 使用相同通道数，公平比较
- 详见 `docs/dev_log/implemented_plans/cbramod_128ch_adaptation.md`

### 缓存系统重构 (v3.0)
- 缓存存储 trial 级别数据（非 segment），减少 ~6.6x 存储空间
- 滑动窗口在加载时应用
- 新增 `scripts/cache_helper.py` 缓存管理工具
- 详见 `docs/dev_log/implemented_plans/cache_refactor.md`

## 2026-01-03

### 🔧 分层时序分割修复
- 修复验证集 100% 来自单一 session 的问题
- 现在训练/验证集有相似的 session 分布
- 详见 `docs/dev_log/bugfixes/stratified_temporal_split.md`

### Quaternary (4指) 任务支持
- 仅使用 Offline 数据（Online 无 4class 文件夹）
- 时序分割 60/20/20 (train/val/test)

### 🔴 严重数据泄露修复
- `session_type` 现使用完整 folder 名称
- 训练/测试集完全独立
- 详见 `docs/dev_log/bugfixes/session_data_leakage.md`

### 三阶段协议完整实现
- 训练: Offline + Online_Base
- 测试: Online_Finetune

## 2026-01-02

### Trial Index 去重修复
- 修复数据泄露问题
- 实现 trial-level split
- 详见 `docs/dev_log/bugfixes/trial_index_deduplication.md`

### 统一 CBraMod 训练流程
- `train_within_subject.py` 支持 `--model` 参数
- 支持 `eegnet`, `cbramod`, `both`

### 全被试模型对比脚本
- `scripts/run_full_comparison.py`
- 自动训练 + 统计分析 + 可视化

## 之前版本

### 已完成功能
- ✅ 数据预处理管线 (通道映射、滤波、重采样)
- ✅ 论文对齐预处理 (Run 级别处理)
- ✅ EEGNet-8,2 基线模型
- ✅ CBraMod 适配器
- ✅ RTX 5070 (Blackwell) GPU 支持
- ✅ HDF5 预处理缓存系统

## 待完成

- 完整 21 被试数据训练 (当前有 S01-S07)
