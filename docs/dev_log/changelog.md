# 开发变更记录

## 2026-03-01

### Commercial & Attention 扩展实验文档补录

**背景**: 32ch 实验文档 (`32ch_experiment.md`) 中 Step 3-5 仅记录了 FDR 配置的完整管线结果。Commercial (2026-02-26) 和 Attention (2026-02-28~03-01) 的 ternary cross-subject / binary transfer / ternary transfer 结果已运行完成但未记入文档。

**新增文档**:
- **Step 6**: Commercial 扩展实验 — ternary cross-subject (69.35%), binary transfer (85.27%), ternary transfer (69.50%)
- **Step 7**: Attention 扩展实验 — ternary cross-subject (71.53%), binary transfer (88.69%), ternary transfer (73.57%)
- 核心数据速查表补充 attention 行、commercial 补齐缺失列
- 综合总结新增第 10 条: Attention ≈ FDR >> Commercial 完整管线验证结论

**关键结论**: Attention 和 FDR 在所有 4 个任务上差距 <1.1pp，均显著优于 commercial (2-4pp)。Attention 在 ternary 任务和 transfer 稳定性上略优。

---

### FDR 补集通道实验（数据泄露排查）

**背景**: 32 通道实验中，6 种不同 channel config（通道重合度极低）均保持较高准确率，引发数据泄露怀疑。

**方法**: 从表现最佳的 FDR 配置 (binary 88.10%) 的 **补集**（128 通道中 FDR 未选的 96 个通道）中随机选取 32 个通道 (seed=42)，运行 CBraMod cross-subject binary classification。

**结果**:

| 配置 | Mean Acc | Median | Std | Min | Max | 说明 |
|------|----------|--------|-----|-----|-----|------|
| FDR (最优 32ch) | **88.10%** | — | — | — | — | Fisher Discriminant Ratio top 32 |
| FDR 补集 (随机 32ch) | **83.18%** | 83.75% | 9.80% | 59.38% | 98.12% | FDR 补集随机抽样，两次独立运行结果一致 |

**代码审查结论**: **未发现 binary 任务数据泄露路径**。
- `FingerEEGDataset` 的 `session_folder` 过滤 (`dataset.py:215-217`) 正确隔离 train/test
- 训练集: `OfflineImagery` + `OnlineImagery_Sess01_*` + `OnlineImagery_Sess02_*_Base`
- 测试集: `OnlineImagery_Sess02_*_Finetune`（完全独立的 session folder）
- cross-subject 模式下 `best_val_acc` 全员一致是正常行为（单全局模型，单全局验证集）

**补集高准确率的合理解释**:
1. **体积传导 (Volume Conduction)**: EEG 信号因颅骨传导在电极间高度相关，即使非最优通道也携带大量判别信息
2. **32/128 = 25%** 的通道比例本身不低，随机 32 通道仍有相当的空间覆盖
3. **CBraMod 基座模型**经大规模 EEG 预训练，对通道选择具有鲁棒性

**新增文件**:
- `scripts/analysis/generate_fdr_complement.py`: FDR 补集通道生成脚本
- `results/32_channel/channel_selections.json`: 新增 `fdr_complement` 配置条目

**修改文件**:
- `src/preprocessing/channel_selection.py`: `CHANNEL_32_CONFIGS` 添加 `'fdr_complement': None`
- `src/training/trainer.py`: 修复 Windows 上 `Path.rename()` 在目标已存在时的 `FileExistsError`（改用 `os.replace()`）

---

## 2026-02-28

### SQLite 实验注册表 (ExperimentDB)

**功能**: 用 SQLite 本地注册表替代基于 JSON 文件名编码的实验结果管理系统，实现结构化查询、消除冗余，同时保留离线能力。

**架构**: SQLite (本地元数据 + 最终指标) + WandB (云端训练曲线) 互补。训练脚本采用双写模式 (JSON cache + SQLite)，确保渐进式迁移。

**新增文件**:
- `src/results/experiment_db.py`: SQLite 注册表核心模块 (`ExperimentDB` 类)
  - 5 张表: `runs`, `subject_results`, `model_summaries`, `comparisons`, `transfer_configs`
  - Schema 版本化 (v2) + 自动迁移
  - WAL 模式并发读、upsert 支持断点续训
  - 高级查询: `find_best_within_subject_results()`, `find_historical_comparison()`, `find_best_cross_subject_results()`, `get_subject_history()` 等
- `scripts/tools/migrate_results_to_db.py`: JSON → SQLite 一次性迁移脚本
  - 解析 3 种文件格式 (comparison_cache, transfer_cache, cross-subject)
  - `git log --diff-filter=A` 追溯每个文件的首次提交作为 `git_commit`
  - 目录路径推断 `n_channels` / `channel_config`
  - `--execute` / `--force` / `--results-dir` CLI
- `tests/test_experiment_db.py`: 41 个单元测试

**修改文件** (8 个):
- `scripts/experiments/run_within_subject_comparison.py`: DB 双写 + 可视化数据源从 DB 查询
- `scripts/experiments/run_cross_subject_comparison.py`: 同上
- `scripts/experiments/run_transfer_comparison.py`: 同上 + `transfer_configs` 保存
- `scripts/experiments/run_single_model.py`: 接受 `db` / `db_run_id` 参数，逐被试写入
- `src/results/__init__.py`: 导出 `ExperimentDB`
- `src/results/cache.py`: 8 个旧查询函数标记 `@deprecated`
- `src/utils/wandb_logger.py`: 新增 `run_id` 属性 (供 DB 关联)
- `.gitignore`: 忽略 `results/experiments.db*`

**查询能力提升**:

| 场景 | 旧方案 | 新方案 |
|------|--------|--------|
| 最新 binary imagery 运行 | `find_latest_cache()` — glob + sort | `db.find_latest_run('imagery', 'binary', 'within_subject')` |
| 32ch FDR 所有 transfer 运行 | 手动拼路径 + glob | `db.find_runs(n_channels=32, channel_config='fdr', experiment_type='transfer')` |
| 某被试跨所有实验的表现 | 不支持 | `db.get_subject_history('S01')` |
| 历史最高准确率 | `find_best_within_subject_for_model()` (60+ 行) | `db.get_best_run(...)` |

**迁移结果**: 71 runs / 1351 subject_results 从 JSON 导入，覆盖 within_subject、cross_subject、transfer 三种实验类型。

---

## 2026-02-20

### 32 通道实验支持

**功能**: 新增 32 通道实验基础设施和实验脚本，支持 6 种通道选择配置对比。

**新增文件**:
- `scripts/analysis/compute_32ch_selections.py`: 数据驱动通道选择（FDR, CSP, Attention/Gradient, Band Power）
- `scripts/experiments/run_32ch_config_comparison.py`: 6 配置对比实验
- `scripts/experiments/run_32ch_experiment.py`: 最优配置全量实验（within + cross + transfer）
- `docs/dev_log/experiments/32ch_experiment.md`: 实验完整记录（原 `implemented_plans/`，已迁移）

**修改文件** (11 个):
- `src/preprocessing/channel_selection.py`: `CHANNEL_32_CONFIGS` 注册表 + `get_32ch_indices()` + `load_32ch_selections()`
- `src/preprocessing/data_loader.py`: `PreprocessConfig.channel_32_config` 字段
- `src/preprocessing/dataset.py`: Strategy 'E' 分支
- `src/config/constants.py`: `SUPPORTED_CHANNEL_COUNTS = [8, 32, 128]`
- `src/config/training.py`: 32ch 超参数覆盖（within/cross/finetune）+ config 函数扩展
- `src/training/train_within_subject.py`: 8/32ch 通道检测泛化
- `src/training/train_cross_subject.py`: 同上
- `src/training/finetune.py`: `channel_config` 参数 + `is_32ch_cbramod` override 逻辑
- `scripts/experiments/run_{within,cross}_subject_comparison.py`: `--channels 32 --channel-config` CLI
- `scripts/experiments/run_transfer_comparison.py`: 同上 + `channel_config` 透传至 finetune

**设计要点**:
- 所有新参数默认 `None`，不影响现有 128ch/8ch 实验默认行为
- HDF5 缓存始终存储 128ch，32ch 选择在加载时应用
- Hand-picked 配置（motor_cortex, commercial）硬编码；data-driven 配置从 JSON 加载

---

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
