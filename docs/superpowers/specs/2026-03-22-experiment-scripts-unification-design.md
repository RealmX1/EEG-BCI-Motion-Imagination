# 实验脚本统一重构设计

## 问题

三种实验范式（被试内、跨被试、迁移学习）的脚本组织方式不一致：

- **被试内（Within-subject）**：干净的两层设计 — `run_single_model.py` 包含全部逻辑，`run_within_subject_comparison.py` 是调用它两次的薄编排器。
- **跨被试（Cross-subject）**：`run_cross_subject_comparison.py` **不**调用 `run_cross_subject.py` — 而是直接调用 `train_cross_subject()`。
- **迁移学习（Transfer learning）**：`run_transfer_comparison.py` 是一个 987 行的单体文件，包含辅助函数、训练循环、DB 写入和绘图。没有对应的单模型脚本。`run_finetune.py` 独立存在但未被任何其他代码调用。

此外，`src/training/` 中的 `finetune_subject()` 与 `train_single_subject()` 有约 70% 的代码重复 — 两者使用相同的 `WithinSubjectTrainer`、相同的数据加载、相同的时序分割和相同的评估。差异如下：

1. **模型初始化**：预训练 checkpoint vs. 从零开始
2. **冻结策略**：`apply_freeze_strategy()` + 自定义 optimizer
3. **训练阶段**：`train_single_subject()` 使用两阶段训练（exploration + main loaders）；`finetune_subject()` 碰巧使用单阶段（非有意设计 — 只是独立编写时未纳入 exploration phase 支持）
4. **预训练 baseline**：`finetune_subject()` 在 epoch 0 评估预训练模型并将其作为初始 best（trainer 仅在 finetune 超越预训练时才保存新 checkpoint）
5. **Config overrides**：`finetune_subject()` 不支持 `config_overrides`；超参通过显式参数传递（`epochs`, `learning_rate`, `batch_size`）
6. **Scheduler**：`finetune_subject()` 硬编码 EEGNet 用 `'plateau'`、CBraMod 用 `'cosine_annealing_warmup_decay'`，不查询 `SCHEDULER_PRESETS`

## 目标

1. **统一 `src/training/`**：将 `finetune_subject()` 合并进 `train_single_subject()`，新增可选的 `pretrained_path` / `freeze_strategy` 参数。
2. **统一脚本范式**：三种范式遵循相同模式 — 单模型脚本 + 薄对比编排器。
3. **减少冗余**：将共享代码（逐被试训练循环、argparse 组、缓存/恢复、DB 初始化）提取到 `_training_utils.py`。
4. **废弃 `run_finetune.py`**：其功能被 `run_single_model.py --pretrained` 吸收。

## 非目标

- 重构 `src/training/train_cross_subject.py` 内部逻辑（跨被试训练在单一模型上训练所有被试，循环结构根本不同）。
- 更改 `ExperimentDB` schema 或缓存格式。
- 添加超出统一范围的新功能。

## 设计

### 第一层：`src/training/train_within_subject.py` — 统一训练

`train_single_subject()` 新增可选的迁移学习参数：

```python
def train_single_subject(
    subject_id, config, data_root, elc_path, save_dir, device,
    model_type, paradigm,
    # 现有参数不变...
    cbramod_channels, preprocess_config, config_overrides,
    cache_only, cache_index_path,
    no_wandb, upload_model, wandb_project, wandb_entity, wandb_group,
    verbose,
    # ---- 新增：迁移学习 ----
    pretrained_path: Optional[str] = None,
    freeze_strategy: Optional[str] = None,  # 'none'/'backbone'/'partial'
) -> Dict:
```

当 `pretrained_path` 被提供时（迁移/微调模式）：
1. 通过 `load_pretrained_model()` 加载 checkpoint，替代从零创建模型。
2. 验证 `n_classes` 与当前 task 匹配。
3. 通过 `apply_freeze_strategy()` 应用冻结策略。
4. 通过 `get_finetune_optimizer()` 创建微调感知的 optimizer 并替换 `trainer.optimizer`。
5. 在 epoch 0 评估预训练 baseline（训练开始前）。设置 `trainer.best_val_acc`、`trainer.best_combined_score`、`trainer.best_epoch = 0`，使 trainer 仅在微调实际超越预训练模型时才保存新 checkpoint。
6. 当 `config_overrides` 未覆盖时，应用微调专用的默认超参：
   - Scheduler：EEGNet 用 `'plateau'`，CBraMod 用 `'cosine_annealing_warmup_decay'`
   - Epochs/LR/batch_size：基于冻结策略和通道数（来自 `get_default_finetune_config()`）
7. **在微调模式下支持 `config_overrides`**（相对于当前 `finetune_subject()` 的新行为）。这使 YAML 配置和 `--scheduler` 在迁移实验中也能工作。
8. 在返回的 dict 中包含 `pretrained_baseline` 和 `milestone_test_results`。

**两阶段训练（exploration phase）**：由 `scheduler_config` 照常控制 — 微调模式不跳过。当前 `finetune_subject()` 碰巧未使用它，但统一函数通过配置使其可用。微调默认的 scheduler 配置设置自己的 `exploration_epochs` 默认值；若为 0，trainer 自然退化为单阶段。

当 `pretrained_path` 为 None（默认）时：现有行为不变。

`train_subject_simple()` 透传新参数：

```python
def train_subject_simple(
    ...,
    pretrained_path=None,
    freeze_strategy=None,
) -> Dict:
    return train_single_subject(..., pretrained_path=pretrained_path, freeze_strategy=freeze_strategy)
```

`finetune_subject()` 变为向后兼容的薄包装器：

```python
def finetune_subject(pretrained_path, subject_id, freeze_strategy='none',
                     epochs=None, learning_rate=None, batch_size=None,
                     model_selection_strategy='combined', ema_decay=0.998, soup_top_k=3,
                     ...) -> Dict:
    """向后兼容包装器。委托给 train_subject_simple()。

    将微调专用的显式参数（epochs, learning_rate, batch_size）
    翻译为统一函数使用的 config_overrides 格式。
    """
    # 从 checkpoint 检测 model_type（train_single_subject 内部也会加载）
    model_type = _detect_model_type_from_checkpoint(pretrained_path)

    # 显式参数 → config_overrides 翻译
    config_overrides = {'training': {}}
    if epochs is not None:
        config_overrides['training']['epochs'] = epochs
    if learning_rate is not None:
        config_overrides['training']['learning_rate'] = learning_rate
    if batch_size is not None:
        config_overrides['training']['batch_size'] = batch_size
    if model_selection_strategy != 'combined':
        config_overrides['training']['model_selection_strategy'] = model_selection_strategy
    config_overrides['training']['ema_decay'] = ema_decay
    config_overrides['training']['soup_top_k'] = soup_top_k

    return train_subject_simple(
        subject_id=subject_id,
        model_type=model_type,
        pretrained_path=pretrained_path,
        freeze_strategy=freeze_strategy,
        config_overrides=config_overrides or None,
        ...
    )

def _detect_model_type_from_checkpoint(pretrained_path: str) -> str:
    """加载 checkpoint 元数据以确定 model_type。"""
    import torch
    ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    return ckpt['model_config']['model_type']
```

注意：`_detect_model_type_from_checkpoint()` 会加载完整 checkpoint，但这可以接受，因为 `train_single_subject()` 会通过 `load_pretrained_model()` 再次加载。避免双重加载的优化（传递预加载的 checkpoint）可以后续添加，不影响正确性。

从 `src/training/finetune.py` 迁移到新文件 `src/training/finetune_utils.py` 的函数：
- `load_pretrained_model()` — checkpoint 加载 + 模型重建
- `apply_freeze_strategy()` — 参数冻结
- `get_finetune_optimizer()` — 微调感知的 optimizer 创建
- `get_default_finetune_config()` — 微调超参默认值（从硬编码值提取）
- 微调专用常量（`EIGHT_CHANNEL_FINETUNE_OVERRIDES` 等）

这些放在独立文件中而非合并进 `train_within_subject.py`，以避免该已经很大的文件（1,053 行）进一步膨胀。`train_within_subject.py` 在 `pretrained_path` 被提供时从 `finetune_utils.py` 导入。

`finetune_all_subjects()` 仅被 `run_finetune.py`（将被废弃）使用，可以移除。

### 第二层：`scripts/_training_utils.py` — 共享抽象

#### 2.1 Argparse 构建器

```python
def add_common_args(parser):
    """共享：--data-root, --paradigm, --task, --seed, --output-dir, --no-plot"""

def add_cache_resume_args(parser):
    """共享：--resume, --force-retrain, --cache-only, --cache-index-path"""

def add_channel_args(parser):
    """共享：--channels, --channel-config"""

def add_training_config_args(parser):
    """共享：--config (YAML), --scheduler, --classifier-type, --no-pretrained"""

def add_model_selection_args(parser):
    """共享：--model-selection-strategy, --ema-decay, --soup-top-k"""

def add_transfer_args(parser):
    """迁移学习专用：--pretrained, --freeze-strategy, --finetune-epochs, --finetune-lr, --auto-discover-pretrained"""
```

#### 2.2 逐被试训练编排器

提取 `run_single_model()` 和 `run_transfer_model()` 之间共享的逐被试训练循环：

```python
def run_model_on_subjects(
    model_type: str,
    subject_ids: List[str],
    train_fn: Callable[[str, ...], TrainingResult],
    train_kwargs: Dict,
    # 缓存
    output_dir: str,
    paradigm: str,
    task: str,
    run_tag: str,
    cache_type: CacheType = CacheType.WITHIN,
    force_retrain: bool = False,
    extra_cache_metadata: Optional[Dict] = None,
    # DB
    db: Optional[ExperimentDB] = None,
    db_run_id: Optional[str] = None,
    # WandB
    wandb_group: Optional[str] = None,
    no_wandb: bool = True,
    # 显示
    verbose_first_only: bool = True,
) -> Tuple[List[TrainingResult], Dict]:
    """
    带缓存、DB 写入和 WandB 的逐被试训练编排。

    1. 加载已有缓存 → 确定哪些被试需要训练
    2. 对每个被试：
       - 检查缓存 → 存在则跳过
       - 调用 train_fn(subject_id, **train_kwargs) → TrainingResult
       - 即时保存到缓存（progressive save）
       - DB 写入（如果提供了 db）
       - WandB 日志
    3. 计算模型统计
    4. 返回 (results, stats)
    """
```

#### 2.3 公共工具函数

```python
def resolve_run_tag(args, paradigm, task, output_dir, cache_type=None) -> str:
    """处理 --resume 逻辑：查找已有 tag 或生成新的。"""

def init_db_run(run_tag, experiment_type, paradigm, task, args) -> Tuple[ExperimentDB, Optional[str]]:
    """创建或恢复 ExperimentDB run。返回 (db, db_run_id)。"""

def finalize_db_run(db, db_run_id, comparison, **extra):
    """保存对比结果、标记完成、关闭 DB。"""

def resolve_output_dir(args) -> str:
    """通道缩减模式下自动重定向到 results/{n}_channel/{config}/。"""

def find_best_checkpoint_path(model_type, paradigm, task, subjects, results_dir, n_channels=None) -> Optional[str]:
    """自动发现最佳跨被试预训练 checkpoint。从 run_transfer_comparison.py 迁移。"""

def validate_checkpoint_compatibility(pretrained_paths, task) -> Dict[str, str]:
    """验证 n_classes 匹配并提取 classifier_types。从 run_transfer_comparison.py 迁移。"""
```

### 第三层：脚本重构

#### `run_single_model.py`（约 250 行，从 602 缩减）

```python
def run_single_model(
    model_type, subject_ids,
    # 公共
    data_root, task, paradigm, output_dir,
    # 缓存/恢复
    force_retrain, run_tag, cache_type=CacheType.WITHIN,
    # 迁移学习（可选）
    pretrained_path=None, freeze_strategy=None,
    # DB 注入
    db=None, db_run_id=None,
    # 配置
    config_overrides=None,
    # WandB、cache-only 等
    **kwargs,
) -> Tuple[List[TrainingResult], Dict]:
    """所有被试的单模型训练。同时支持被试内和迁移学习。"""

    def train_fn(subject_id, verbose):
        return train_and_get_result(
            subject_id, model_type, task, paradigm, data_root,
            pretrained_path=pretrained_path,
            freeze_strategy=freeze_strategy,
            config_overrides=config_overrides,
            verbose=verbose,
            **kwargs,
        )

    return run_model_on_subjects(
        model_type=model_type,
        subject_ids=subject_ids,
        train_fn=train_fn,
        output_dir=output_dir,
        paradigm=paradigm,
        task=task,
        run_tag=run_tag,
        cache_type=cache_type,
        force_retrain=force_retrain,
        db=db, db_run_id=db_run_id,
    )

def main():
    parser = argparse.ArgumentParser(...)
    add_common_args(parser)
    add_cache_resume_args(parser)
    add_channel_args(parser)
    add_training_config_args(parser)
    add_transfer_args(parser)
    add_wandb_args(parser)
    args = parser.parse_args()

    output_dir = resolve_output_dir(args)
    run_tag = resolve_run_tag(args, ...)
    config_overrides = build_config_overrides(args)

    results, stats = run_single_model(
        model_type=args.model,
        subject_ids=subjects,
        pretrained_path=args.pretrained,
        freeze_strategy=args.freeze_strategy,
        ...
    )

    # 单模型绘图
    generate_single_model_plot(results, ...)
```

#### `run_within_subject_comparison.py`（约 200 行，从 632 缩减）

```python
def main():
    parser = argparse.ArgumentParser(...)
    add_common_args(parser)
    add_cache_resume_args(parser)
    add_channel_args(parser)
    add_training_config_args(parser)
    add_wandb_args(parser)
    parser.add_argument('--models', ...)
    parser.add_argument('--skip-training', ...)
    args = parser.parse_args()

    output_dir = resolve_output_dir(args)
    run_tag = resolve_run_tag(args, ...)
    db, db_run_id = init_db_run(run_tag, 'within_subject', ...)
    config_overrides = build_config_overrides(args)

    # 训练每个模型
    results = {}
    for model_type in args.models:
        model_results, stats = run_single_model(
            model_type=model_type,
            db=db, db_run_id=db_run_id,
            config_overrides=config_overrides,
            ...
        )
        results[model_type] = model_results
        db.save_summary(db_run_id, model_type, stats)

    # 对比 + 绘图（2-way：EEGNet vs CBraMod，仅当前运行结果）
    comparison = compare_models(results.get('eegnet'), results.get('cbramod'))
    finalize_db_run(db, db_run_id, comparison)
    save_cache(..., is_complete=True)
    # 绘图数据源：
    #   1. EEGNet (Within) — 当前运行
    #   2. CBraMod (Within) — 当前运行
    generate_combined_plot(data_sources=[
        PlotDataSource(model_type='eegnet', results=results['eegnet'], is_current_run=True),
        PlotDataSource(model_type='cbramod', results=results['cbramod'], is_current_run=True),
    ], ...)
```

#### `run_transfer_comparison.py`（约 250 行，从 987 缩减）

```python
def main():
    parser = argparse.ArgumentParser(...)
    add_common_args(parser)
    add_cache_resume_args(parser)
    add_channel_args(parser)
    add_transfer_args(parser)
    add_wandb_args(parser)
    parser.add_argument('--models', ...)
    parser.add_argument('--pretrained-eegnet', ...)
    parser.add_argument('--pretrained-cbramod', ...)
    parser.add_argument('--no-cross-subject-baseline', ...)
    args = parser.parse_args()

    output_dir = resolve_output_dir(args)
    run_tag = resolve_run_tag(args, ..., cache_type=CacheType.TRANSFER)
    db, db_run_id = init_db_run(run_tag, 'transfer', ...)

    # 发现/验证 checkpoint
    pretrained_paths = discover_or_validate_checkpoints(args, subjects)
    classifier_types = validate_checkpoint_compatibility(pretrained_paths, args.task)

    # 训练每个模型（迁移模式）
    results = {}
    for model_type in args.models:
        if model_type not in pretrained_paths:
            continue
        model_results, stats = run_single_model(
            model_type=model_type,
            pretrained_path=pretrained_paths[model_type],
            freeze_strategy=args.freeze_strategy,
            cache_type=CacheType.TRANSFER,
            db=db, db_run_id=db_run_id,
            ...
        )
        results[model_type] = model_results
        db.save_summary(db_run_id, model_type, stats)

    # 对比 + 绘图（6-way：历史 baseline + 当前迁移结果）
    comparison = compare_models(results.get('eegnet'), results.get('cbramod'))
    finalize_db_run(db, db_run_id, comparison, transfer_config=...)
    save_cache(..., cache_type=CacheType.TRANSFER, is_complete=True)

    # 绘图数据源组装（6 个 PlotDataSource，从 ExperimentDB 查询历史 baseline）：
    #   1. EEGNet (Within)  — db.find_best_within_subject_results()，hatch='///'
    #   2. CBraMod (Within) — db.find_best_within_subject_results()，hatch='///'
    #   3. EEGNet (Cross)   — db.find_best_cross_subject_results()，hatch='...'
    #   4. CBraMod (Cross)  — db.find_best_cross_subject_results()，hatch='...'
    #   5. EEGNet (Transfer)  — 当前运行结果，is_current_run=True
    #   6. CBraMod (Transfer) — 当前运行结果，is_current_run=True
    # 其中 3-4 可通过 --no-cross-subject-baseline 跳过
    data_sources = build_transfer_plot_sources(results, db, args)
    generate_combined_plot(data_sources=data_sources, ...)
```

#### `run_cross_subject_comparison.py`（约 250 行，从 647 缩减）

使用相同的共享工具，但保留自己的训练逻辑，因为跨被试训练在所有被试上训练单一模型（与逐被试循环根本不同）。

采用的共享工具：
- `add_common_args()`, `add_cache_resume_args()`, `add_channel_args()`, `add_wandb_args()` — argparse 构建器
- `resolve_run_tag()` — 恢复/tag 逻辑
- `init_db_run()` / `finalize_db_run()` — DB 生命周期
- `resolve_output_dir()` — 通道目录重定向

不共享（跨被试专用）：
- 训练循环 — 直接调用 `train_cross_subject()`（在所有被试合并数据上训练单一模型，非逐被试）
- 缓存结构 — 扁平的 `{model: result_dict}` 而非 `{model: {subject: result_dict}}`
- 结果转换 — `cross_subject_result_to_training_results()` 用于对比

### 废弃计划

| 文件 | 操作 |
|------|------|
| `scripts/experiments/run_finetune.py` | 删除。功能被 `run_single_model.py --pretrained` 吸收。 |
| `scripts/run_finetune.py` | 删除（上述文件的薄包装器）。 |
| `src/training/finetune.py` | 保留为向后兼容包装器（约 50 行）。核心逻辑迁移到 `src/training/finetune_utils.py`（可复用函数）并集成进 `train_within_subject.py`（统一流程）。`finetune_all_subjects()` 移除。 |
| `src/training/finetune_utils.py` | 新文件。包含 `load_pretrained_model()`、`apply_freeze_strategy()`、`get_finetune_optimizer()`、`get_default_finetune_config()` 和微调常量。在 `pretrained_path` 被提供时由 `train_within_subject.py` 导入。 |

### 设计决策

**WandB 策略**：所有范式（被试内和迁移学习）统一使用 **per-subject WandB runs**。放弃当前迁移学习专用的"所有被试共享单个 WandB run"模式。理由：per-subject runs 更简单、跨范式一致，且共享 run 模式在失败时有清理问题。Per-subject runs 通过 `wandb_group` 分组。

**通道配置传递**：当 `run_single_model()` 以迁移模式和 `--channels` / `--channel-config` 被调用时，这些通过 `config_overrides['data']['channels']` 和 `config_overrides['data']['channel_config']` 传递。统一的 `train_single_subject()` 从 config 读取并传给 `preprocess_config.apply_channel_overrides()`（当前代码第 394-396 行已支持）。无需新的管道。

**两阶段训练（exploration phase）**：微调模式不条件性跳过。exploration phase 由 `scheduler_config['exploration_epochs']` 控制 — 如果微调默认配置设为 0，trainer 自然使用单阶段。如果 config overrides 指定了 exploration epochs，微调同样可用。统一函数中无需特殊处理。

### 迁移安全

- `finetune_subject()` 保持可导入且签名不变 — `src/hpo/objectives.py` 和 `src/training/__init__.py` 继续工作。
- 所有现有 CLI 命令继续工作（顶层薄包装器不变，除 `run_finetune.py` 外）。
- 缓存格式不变 — 已有缓存结果仍然有效。
- DB schema 不变。

### 迁移清单

- [ ] 更新 `src/training/__init__.py`：从 imports 和 `__all__` 中移除 `finetune_all_subjects`
- [ ] 更新 `src/training/__init__.py`：从新位置重新导出 `finetune_subject`（或继续从 `finetune.py` 包装器导入）
- [ ] 更新 `docs/codebase_reference.md`：移除 `run_finetune.py` 引用，添加 `--pretrained` 文档
- [ ] 验证 `src/hpo/objectives.py` 与向后兼容包装器正常工作（相同参数、相同返回格式）
- [ ] 删除 `scripts/experiments/run_finetune.py` 和 `scripts/run_finetune.py`

### 验证

- **向后兼容**：`finetune_subject()` 从 `src/hpo/objectives.py` 以相同参数调用时产生相同行为。通过在修改前后运行单被试 HPO trial 验证。
- **缓存兼容**：来自迁移运行的现有 JSON 缓存仍可被新的 `run_transfer_comparison.py --resume` 加载。
- **冒烟测试**：`run_single_model.py --model cbramod --pretrained <path> --freeze-strategy backbone --subjects S01` 产生与旧的 `run_transfer_comparison.py` 对单模型/单被试等效的结果。

### 指标

| 指标 | 重构前 | 重构后 | 变化 |
|------|--------|--------|------|
| 实验脚本总代码量 | ~3,500 行 | ~1,200 行 | -65% |
| `_training_utils.py` | 233 行 | ~400 行 | +170（吸收共享逻辑） |
| `src/training/finetune.py` | 840 行 | ~50 行（包装器） | -94% |
| `src/training/train_within_subject.py` | 1,053 行 | ~1,150 行 | +~100（吸收微调逻辑） |
| 净总量 | ~5,600 行 | ~2,800 行 | -50% |
| 跨脚本重复代码 | ~1,500 行 | ~0 | 消除 |
