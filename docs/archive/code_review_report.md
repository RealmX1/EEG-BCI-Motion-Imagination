# Code Review Report

## 1. 关键发现：数据泄露风险 (Critical)

在审查中发现 `src/data/data_provider.py` 和 `src/training/unified_trainer.py` 存在严重的数据泄露风险。

*   **问题描述**: `DataProvider` 在创建训练/验证集时使用了 `random_split` (segment-level split)。由于数据预处理使用了滑动窗口（sliding window），同一个 trial 会产生多个 segments。随机分割导致同一 trial 的 segments 同时出现在训练集和验证集中。
*   **验证**: 编写了 `verify_leakage.py` 脚本进行验证，确认 604 个 trials (S01) 存在数据泄露。
*   **影响范围**: 任何使用 `UnifiedTrainer` 或 `DataProvider` 的训练脚本（目前主要是 `unified_trainer.py` 自身接口）。
*   **安全代码**: `src/training/train_within_subject.py` 和 `scripts/run_full_comparison.py` 是**安全**的。它们使用了独立的 `load_subject_data` 和 `train_single_subject` 函数，实现了正确的 Stratified Temporal Split Logic (Trial-level split)。

**建议**:
1.  **立即修复**: 修改 `src/data/data_provider.py`，移植 `train_within_subject.py` 中的 trial-level split 逻辑。
2.  **弃用警告**: 在 `UnifiedTrainer` 中添加警告，或直到修复前暂时禁用。

## 2. 废弃文件与脚本 (Deprecated)

以下文件似乎已过时，建议删除或归档，以减少维护负担和混淆：

| 文件路径 | 说明 | 替代方案 |
| :--- | :--- | :--- |
| `src/training/train_eegnet.py` | 旧的 EEGNet 训练脚本 | `src/training/train_within_subject.py` |
| `src/training/finetune_cbramod.py` | 旧的 CBraMod 微调脚本 | `src/training/train_within_subject.py` |
| `src/training/trainer.py` | 旧的通用训练器基类 | `src/training/unified_trainer.py` (需修复) 或 `train_within_subject.py` |
| `run_experiments.py` | 根目录下的旧入口脚本，引用了上述两个废弃脚本 | `scripts/run_full_comparison.py` |
| `diagnose_detailed.py` | 临时诊断脚本 | 无 (一次性用途) |
| `diagnose_*.py` (root) | 其他各类诊断脚本 (`diagnose_data_split.py` 等) | 无 (可视情况保留用于 debug，但非核心) |
| `analyze_trials_per_run.py` | 试次分析脚本，用于发现 trial index 问题 | 问题已在 `train_within_subject.py` 中解决 |

## 3. 代码重复与架构问题

发现存在两套并行的 Dataset 实现：
1.  **`src.preprocessing.data_loader.FingerEEGDataset`**: 功能完善，支持 HDF5 缓存，被 `train_within_subject.py` (主要训练脚本) 使用。
2.  **`src.data.data_provider.TrialDataset`**: `DataProvider` 内部定义的 Dataset，实现较简单，缺少 split 辅助方法，被 `UnifiedTrainer` 使用。

这导致了逻辑分裂（Split Logic Split），也是数据泄露问题的根源。

**建议**:
*   重构 `DataProvider` 以直接封装或使用 `FingerEEGDataset`，而不是重新实现一个 `TrialDataset`。
*   统一入口：确保所有训练任务（无论是通过 `scripts/run_full_comparison.py` 还是 `UnifiedTrainer`）最终都通过同一套稳健的数据加载和分割逻辑。

## 4. 修复建议详情 (Fix Plan)

针对 `src/data/data_provider.py` 的修复建议：

1.  修改 `TrialDataset` (或切换使用 `FingerEEGDataset`) 以提供 `get_unique_trials()` 和 `get_trial_info(idx)`。
2.  在 `create_subject_dataloaders` 中，移除 `random_split`。
3.  实现如下逻辑：
    ```python
    unique_trials = dataset.get_unique_trials()
    # Group by session...
    # Split trials temporally (80/20) within sessions...
    # Get segment indices...
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    ```

## 5. 总结

当前项目核心训练流程 (`run_full_comparison.py` -> `train_within_subject.py`) 是**正确且高质量**的（包含严谨的防泄露分割、性能优化等）。问题主要集中在试图建立“统一架构”的新代码 (`data_provider.py` / `unified_trainer.py`) 上，这部分代码尚未成熟且存在 Risks。旧的脚本则应尽快清理。
