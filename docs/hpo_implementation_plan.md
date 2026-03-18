# Hyperparameter Optimization (HPO) Implementation Plan

## 1. 动机与目标

当前项目的超参数（~19 个可调参数）均为手动设定，不同通道数配置间通过"插值"估算。
本方案引入 Optuna 自动化超参数搜索，在 128ch baseline 上为每种训练范式找到最优超参数，
后续其他通道配置从 baseline 结果派生。

**目标**:
- 在 128ch 配置上为 CBraMod 和 EEGNet 分别找到 within-subject、cross-subject、transfer 的最优超参数
- 使用自定义概率预测式剪枝器（ProbabilisticSubjectPruner）降低搜索成本
- 保持与现有训练管线的兼容性，HPO 通过 `config_overrides` 注入参数，不修改核心训练逻辑

---

## 2. 架构设计

### 2.1 新增文件

```
src/hpo/
    __init__.py                 # 公开 API 导出
    pruner.py                   # ProbabilisticSubjectPruner
    search_spaces.py            # 搜索空间定义（per paradigm × model）
    objectives.py               # Optuna objective 函数（封装现有训练入口）
scripts/
    run_hpo.py                  # CLI 入口
```

### 2.2 不修改的文件

以下核心训练模块**不做任何修改**，HPO 完全通过已有的 `config_overrides` 机制注入参数：

- `src/training/trainer.py` — `WithinSubjectTrainer`
- `src/training/train_within_subject.py` — `train_subject_simple()`, `train_single_subject()`
- `src/training/train_cross_subject.py` — `train_cross_subject()`
- `src/training/finetune.py` — `finetune_subject()`
- `src/config/training.py` — `get_default_config()`, `get_cross_subject_config()`

### 2.3 依赖

```toml
# pyproject.toml 新增
optuna >= 3.0
```

---

## 3. 自定义剪枝器: ProbabilisticSubjectPruner

### 3.1 适用场景

- **Within-subject**: 逐被试训练独立模型，每完成一个被试 report 累积均值
- **Transfer**: 同上（逐被试微调独立模型）
- **Cross-subject**: 单模型训练，每个 epoch report validation accuracy

### 3.2 原理

标准 MedianPruner 在 step N 判断"当前 trial < 历史中位数则砍"，对被试间方差大的场景容易误杀。

ProbabilisticSubjectPruner 在每个 step 后估算：

$$P(\text{final mean} > \text{best historical}) \mid \text{当前已完成的被试/epoch 结果}$$

当此概率低于阈值（默认 10%）时，终止 trial。

### 3.3 统计模型

```
已完成: n_done 个被试，均值 μ_done，标准差 σ_done
剩余: n_remaining = N_total - n_done
目标: best_final（历史最优 trial 的最终均值）

假设剩余被试的表现独立同分布于 N(μ_done, σ_done²)

最终均值的分布:
  final_mean ~ N(μ_done, σ_done² * n_remaining / (N_total²) + σ_done² * n_done / (N_total²))
  简化: final_std ≈ σ_done / √N_total  （当 n_done 足够时）

更精确的表达（考虑已知部分是确定的）:
  final_mean = (n_done * μ_done + n_remaining * μ_remaining) / N_total
  其中 μ_remaining ~ N(μ_done, σ_done² / n_remaining)

  Var(final_mean) = (n_remaining / N_total)² × (σ_done² / n_remaining)
                  = σ_done² × n_remaining / N_total²

  P(final_mean > best) = 1 - Φ((best - μ_estimated) / final_std)
```

### 3.4 接口

```python
class ProbabilisticSubjectPruner(optuna.pruners.BasePruner):
    def __init__(
        self,
        n_total_steps: int,     # 总步数（被试数或总 epoch 数）
        threshold: float = 0.1, # P < 10% 则剪枝
        min_steps: int = 3,     # 至少完成 3 步才开始判断
    ):
        ...

    def prune(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> bool:
        ...
```

### 3.5 Cross-subject 适配

Cross-subject 只有 1 个模型，report 点是 epoch 而非被试。同一个 Pruner 类，
通过 `n_total_steps=epochs` 配置即可。方差估计使用该 trial 历史 epoch 的 val_acc 波动。

---

## 4. 搜索空间定义

### 4.1 CBraMod Within-Subject (128ch, binary)

| 参数 | 类型 | 范围 | 当前默认 | 说明 |
|------|------|------|----------|------|
| `backbone_lr` | log_float | [1e-5, 1e-3] | 1e-4 | |
| `classifier_lr_ratio` | float | [1.0, 5.0] | 3.0 | classifier_lr = backbone_lr × ratio |
| `weight_decay` | log_float | [0.01, 0.3] | 0.06 | |
| `dropout_rate` | float | [0.05, 0.45] | 0.15 | |
| `batch_size` | categorical | {64, 128, 256} | 128 | |
| `label_smoothing` | float | [0.0, 0.2] | 0.05 | |
| `gradient_clip` | float | [0.3, 2.0] | 1.0 | |
| `phase_decay` | float | [0.3, 0.9] | 0.7 | CAWD 阶段间峰值衰减 |
| `phase_epochs` | int | [4, 10] | 6 | CAWD 每阶段轮数 |
| `exploration_epochs` | int | [3, 9] | 6 | 小 batch 探索轮数 |
| `exploration_batch_size` | categorical | {16, 32, 64} | 32 | 探索阶段 batch size |
| `classifier_type` | categorical | {one_layer, two_layer, three_layer} | two_layer | 可选，v1 可固定 |

**总计**: 11-12 个参数

### 4.2 CBraMod Cross-Subject (128ch, binary)

| 参数 | 类型 | 范围 | 当前默认 | 与 within 的差异 |
|------|------|------|----------|-----------------|
| `backbone_lr` | log_float | [1e-5, 5e-4] | 1e-4 | 上限更低 |
| `classifier_lr_ratio` | float | [1.0, 3.0] | 1.5 | 范围更窄 |
| `weight_decay` | log_float | [0.03, 0.5] | 0.12 | 整体偏高 |
| `dropout_rate` | float | [0.15, 0.55] | 0.35 | 整体偏高 |
| `batch_size` | categorical | {128, 256} | 256 | 更大 |
| `label_smoothing` | float | [0.05, 0.3] | 0.15 | 整体偏高 |
| `gradient_clip` | float | [0.2, 1.5] | 0.5 | |
| `phase_decay` | float | [0.2, 0.7] | 0.5 | 更激进 |
| `phase_epochs` | int | [4, 10] | 6 | |
| `exploration_epochs` | int | [3, 9] | 6 | |
| `exploration_batch_size` | categorical | {32, 64, 128} | 64 | |
| `classifier_type` | categorical | {one_layer, two_layer, three_layer} | two_layer | 可选 |

### 4.3 CBraMod Transfer/Finetune (128ch, binary)

| 参数 | 类型 | 范围 | 当前默认 | 说明 |
|------|------|------|----------|------|
| `backbone_lr` | log_float | [1e-6, 5e-4] | 1e-4 | 预训练模型，lr 更低 |
| `classifier_lr_ratio` | float | [1.0, 5.0] | 1.0 | |
| `weight_decay` | log_float | [1e-3, 0.2] | 0.05 | |
| `dropout_rate` | float | [0.1, 0.5] | 0.15 | |
| `batch_size` | categorical | {32, 64, 128} | 128 | 数据少 |
| `label_smoothing` | float | [0.0, 0.15] | 0.05 | |
| `gradient_clip` | float | [0.3, 2.0] | 1.0 | |
| ~~`freeze_strategy`~~ | ~~categorical~~ | ~~{none, backbone, partial}~~ | none | **固定为 none，不搜索** |
| `finetune_epochs` | int | [10, 40] | 15 | |
| `patience` | int | [5, 15] | 10 | |

**注意**: Transfer 不使用 CAWD 调度器，使用简单 cosine 或 plateau。

### 4.4 EEGNet Within-Subject (128ch, binary)

| 参数 | 类型 | 范围 | 当前默认 |
|------|------|------|----------|
| `learning_rate` | log_float | [1e-4, 1e-2] | 1e-3 |
| `weight_decay` | log_float | [1e-5, 0.1] | 0 |
| `dropout_rate` | float | [0.2, 0.7] | 0.5 |
| `batch_size` | categorical | {32, 64, 128} | 64 |
| `F1` | categorical | {4, 8, 16} | 8 |
| `D` | categorical | {1, 2, 4} | 2 |
| `kernel_length` | categorical | {32, 64, 128} | 64 |

**总计**: 7 个参数

### 4.5 EEGNet Cross-Subject (128ch, binary)

| 参数 | 类型 | 范围 | 当前默认 |
|------|------|------|----------|
| `learning_rate` | log_float | [5e-5, 5e-3] | 5e-4 |
| `weight_decay` | log_float | [1e-5, 0.2] | 1e-4 |
| `dropout_rate` | float | [0.3, 0.7] | 0.5 |
| `batch_size` | categorical | {64, 128, 256} | 128 |
| `F1` | categorical | {4, 8, 16} | 8 |
| `D` | categorical | {1, 2, 4} | 2 |
| `kernel_length` | categorical | {32, 64, 128} | 64 |

---

## 5. Objective 函数设计

### 5.1 Within-Subject Objective

```python
def within_subject_objective(trial, model_type, task, paradigm, subjects):
    """
    遍历所有被试，逐个训练独立模型，report 累积均值。
    Pruner 在被试级别判断是否提前终止。
    """
    params = sample_search_space(trial, model_type, 'within_subject')
    config_overrides = params_to_config_overrides(params, model_type)

    accs = []
    for i, subject_id in enumerate(subjects):
        result = train_subject_simple(
            subject_id=subject_id,
            model_type=model_type,
            task=task,
            paradigm=paradigm,
            config_overrides=config_overrides,
            no_wandb=True,       # HPO 阶段禁用 WandB
            verbose=0,           # 静默模式
        )

        acc = result['test_accuracy_majority']
        accs.append(acc)

        # Report 累积均值给 Optuna
        trial.report(np.mean(accs), step=i)

        # 检查是否应该剪枝
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(accs)
```

### 5.2 Cross-Subject Objective

Cross-subject 训练单模型，需要在 epoch 级别 report。这要求修改 `trainer.train()`
的回调机制。**但为了不修改核心训练代码**，采用以下策略：

```python
def cross_subject_objective(trial, model_type, task, paradigm, subjects):
    """
    训练单个跨被试模型。
    由于 train_cross_subject() 内部自带 early stopping，
    直接用最终结果作为 objective，不做 epoch 级剪枝。

    替代方案：通过 wandb_callback 或自定义 callback 注入 epoch 级 report。
    """
    params = sample_search_space(trial, model_type, 'cross_subject')
    config_overrides = params_to_config_overrides(params, model_type)

    result = train_cross_subject(
        subjects=subjects,
        model_type=model_type,
        task=task,
        paradigm=paradigm,
        config_overrides=config_overrides,
        wandb_enabled=False,
        verbose=0,
    )

    return result['mean_test_acc']
```

**Cross-subject epoch 级剪枝（可选增强）**:

为了支持 epoch 级剪枝而不修改 `trainer.py`，可以引入一个轻量级 callback 对象，
通过现有的 `wandb_callback` 参数槽注入。需要创建一个实现 `on_epoch_end()` 的
`OptunaCallback` 类，在 epoch 结束时调用 `trial.report()` 和 `trial.should_prune()`。

```python
class OptunaEpochCallback:
    """伪装为 WandbCallback 的 Optuna epoch-level reporter。"""

    def __init__(self, trial: optuna.Trial):
        self.trial = trial
        self.epoch = 0

    def on_epoch_end(self, epoch, train_loss, train_acc, val_loss, val_acc,
                     majority_acc, combined_score, lr, **kwargs):
        self.trial.report(combined_score, step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()
        self.epoch = epoch

    # WandbCallback 接口的 no-op 方法
    def on_train_start(self, **kwargs): pass
    def on_train_end(self, **kwargs): pass
```

**决策**: v1 先不做 epoch 级剪枝（cross-subject 训练有 early stopping，已经能自动终止差的配置）。
v2 如果需要，通过 `OptunaEpochCallback` 注入。

### 5.3 Transfer Objective

```python
def transfer_objective(trial, model_type, task, paradigm, subjects, pretrained_path):
    """
    逐被试微调预训练模型。结构与 within-subject 相同。
    pretrained_path 来自 cross-subject HPO 的最优模型。
    """
    params = sample_search_space(trial, model_type, 'transfer')

    accs = []
    for i, subject_id in enumerate(subjects):
        result = finetune_subject(
            pretrained_path=pretrained_path,
            subject_id=subject_id,
            freeze_strategy=params.get('freeze_strategy', 'none'),
            epochs=params.get('finetune_epochs', 15),
            learning_rate=params['backbone_lr'],
            batch_size=params.get('batch_size', 128),
            task=task,
            paradigm=paradigm,
            no_wandb=True,
            verbose=0,
        )

        acc = result['test_acc']
        accs.append(acc)

        trial.report(np.mean(accs), step=i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(accs)
```

---

## 6. 参数映射: trial params → config_overrides

HPO 采样的参数需要转换为现有训练管线接受的 `config_overrides` dict 格式。

### 6.1 CBraMod 映射

```python
def params_to_config_overrides(params: dict, model_type: str) -> dict:
    """将 Optuna trial 参数映射为 config_overrides dict。"""

    if model_type == 'cbramod':
        overrides = {
            'model': {
                'dropout_rate': params['dropout_rate'],
            },
            'training': {
                'backbone_lr': params['backbone_lr'],
                'classifier_lr': params['backbone_lr'] * params['classifier_lr_ratio'],
                'learning_rate': params['backbone_lr'],
                'weight_decay': params['weight_decay'],
                'label_smoothing': params['label_smoothing'],
                'gradient_clip': params['gradient_clip'],
                'batch_size': params['batch_size'],
            },
            'scheduler_config': {
                'phase_decay': params.get('phase_decay'),
                'phase_epochs': params.get('phase_epochs'),
                'exploration_epochs': params.get('exploration_epochs'),
                'exploration_batch_size': params.get('exploration_batch_size'),
            },
        }
        if 'classifier_type' in params:
            overrides['model']['classifier_type'] = params['classifier_type']

        return overrides

    else:  # eegnet
        return {
            'model': {
                'F1': params.get('F1', 8),
                'D': params.get('D', 2),
                'F2': params.get('F1', 8) * params.get('D', 2),  # F2 = F1 × D
                'kernel_length': params.get('kernel_length', 64),
                'dropout_rate': params['dropout_rate'],
            },
            'training': {
                'learning_rate': params['learning_rate'],
                'weight_decay': params['weight_decay'],
                'batch_size': params['batch_size'],
            },
        }
```

---

## 7. CLI 入口: `scripts/run_hpo.py`

```
用法:
  uv run python scripts/run_hpo.py \
      --paradigm within_subject \
      --model cbramod \
      --task binary \
      --n-trials 50 \
      --n-channels 128 \
      --study-name cbramod_within_binary \
      --storage sqlite:///results/hpo.db \
      --pruner probabilistic \
      --prune-threshold 0.1

参数:
  --paradigm        within_subject | cross_subject | transfer
  --model           cbramod | eegnet
  --task            binary | ternary | quaternary  (默认: binary)
  --eeg-paradigm    imagery | movement  (默认: imagery)
  --n-trials        搜索次数 (默认: 50)
  --n-channels      通道数 (默认: 128)
  --study-name      Optuna study 名称（用于持久化和恢复）
  --storage         Optuna 存储后端 (默认: sqlite:///results/hpo.db)
  --pruner          probabilistic | median | none (默认: probabilistic)
  --prune-threshold 概率剪枝阈值 (默认: 0.1)
  --seed            随机种子 (默认: 42)
  --pretrained-path Transfer 模式必须提供预训练 checkpoint 路径

Transfer 特殊参数:
  --pretrained-path  Cross-subject 预训练模型的 checkpoint 路径
```

### 7.1 Study 持久化与恢复

使用 SQLite 存储 (`results/hpo.db`)，支持：
- 中断后恢复（`optuna.load_study()`）
- 多次运行累积 trials
- 通过 `optuna-dashboard` 可视化

---

## 8. 执行计划与优先级

### Phase 1: CBraMod 128ch (核心)

| 步骤 | 配置 | Trials | 剪枝方式 | 预计耗时 |
|------|------|--------|----------|----------|
| 1a | CBraMod, within-subject, binary | 50 | 被试级概率剪枝 | ~8-12× 完整运行 |
| 1b | CBraMod, cross-subject, binary | 50 | 无（依赖 early stopping） | ~50× 单次训练 |
| 1c | CBraMod, transfer, binary | 20-30 | 被试级概率剪枝 | ~5-8× 完整运行 |

**依赖关系**: 1a 和 1b 可并行 → 1c 依赖 1b 的最优模型

### Phase 2: EEGNet 128ch

| 步骤 | 配置 | Trials |
|------|------|--------|
| 2a | EEGNet, within-subject, binary | 30 |
| 2b | EEGNet, cross-subject, binary | 30 |

### Phase 3: 其他通道数验证

用 Phase 1 最优参数作为锚点，对 32ch/8ch 做小规模验证搜索（~10 trials）：
- 主要调整 dropout_rate、weight_decay（因信息量减少需要更强正则化）
- 其余参数直接沿用 128ch 结果

### Phase 4: 其他任务 (ternary/quaternary)

验证 binary 最优参数在 ternary/quaternary 上的表现，必要时做小规模搜索。

---

## 9. 结果输出与分析

### 9.1 自动输出

每次 HPO 完成后自动输出：
- 最优参数组合（JSON 格式，可直接作为 YAML config 使用）
- 参数重要性排名（`optuna.importance.get_param_importances()`）
- 搜索历史可视化（optimization history、param importance、parallel coordinate）

### 9.2 结果存储

```
results/hpo/
    hpo.db                                      # Optuna SQLite 存储
    cbramod_within_binary_best_params.json       # 最优参数导出
    cbramod_cross_binary_best_params.json
    ...
```

### 9.3 最优参数回写

HPO 完成后，最优参数可以：
1. 导出为 YAML config 文件（供 `--config` 使用）
2. 直接更新 `src/config/training.py` 中的默认值（手动审核后）

---

## 10. 实现任务清单

### 10.1 基础设施

- [ ] 在 `pyproject.toml` 中添加 `optuna` 依赖
- [ ] 创建 `src/hpo/__init__.py`

### 10.2 核心模块

- [ ] 实现 `src/hpo/pruner.py` — `ProbabilisticSubjectPruner`
- [ ] 实现 `src/hpo/search_spaces.py` — 搜索空间定义 + `sample_search_space()` + `params_to_config_overrides()`
- [ ] 实现 `src/hpo/objectives.py` — 三种范式的 objective 函数

### 10.3 CLI 入口

- [ ] 实现 `scripts/run_hpo.py` — 参数解析 + Study 创建 + 结果输出

### 10.4 验证

- [ ] 单元测试: ProbabilisticSubjectPruner 的数学正确性
- [ ] 冒烟测试: 用 `--n-trials 2` 跑完整流程
- [ ] 验证 config_overrides 正确传递到训练管线

---

## 11. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 概率剪枝器的正态假设不准 | 误剪好的 trial | `min_steps=3` 保守启动 + threshold=0.1 |
| Cross-subject 单次训练太慢 | 50 trials 不可行 | v1 不做 epoch 级剪枝，依赖 early stopping；v2 加 OptunaEpochCallback |
| HPO 过拟合搜索空间 | 找到的参数不泛化 | 用 test_accuracy_majority（trial-level）而非 segment-level 作为 objective |
| 被试顺序影响剪枝决策 | 先遇到难被试导致误杀 | 固定被试顺序（按难度排序或打乱 + 固定 seed） |
| GPU 显存不足（大 batch_size） | Trial 崩溃 | Optuna 自动处理 failed trials，不影响搜索 |

---

## 12. 未来扩展（不在 v1 范围内）

- **Multi-objective**: 同时优化 accuracy 和训练时间
- **Cross-subject epoch 级剪枝**: 通过 `OptunaEpochCallback` 实现
- **Muon 优化器搜索**: 在 AdamW 最优参数确定后，探索 Muon 是否更优
- **WandB Sweeps 集成**: 与现有 WandB 基础设施对齐可视化
- **通道数自适应搜索**: 自动为不同通道数派生超参数
