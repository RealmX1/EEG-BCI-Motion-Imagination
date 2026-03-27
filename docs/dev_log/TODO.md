# TODO - Future Improvements

## Training - Best Epoch Selection Weights

**Date Added**: 2025-01-25

**Current Implementation** (`src/training/train_within_subject.py:550-568`):
- Best epoch is updated when EITHER `val_acc` OR `majority_acc` improves
- This can cause model updates when metrics conflict

**Proposed Change**:
Test various weighting schemes for best epoch selection:

```python
# Option A: Simple average
avg_acc = (val_acc + majority_acc) / 2

# Option B: Weighted average (prioritize majority_acc as it's the paper's main metric)
weighted_avg = 0.3 * val_acc + 0.7 * majority_acc

# Option C: Other weight combinations to test
# 0.5 / 0.5, 0.4 / 0.6, 0.2 / 0.8, etc.
```

**Rationale**:
- `majority_acc` is the paper's final evaluation metric (trial-level voting)
- `val_acc` is segment-level and more granular
- Weighted average may produce more balanced model selection

**Testing Plan**:
1. Run full comparison with current approach (baseline)
2. Run with simple average (0.5/0.5)
3. Run with weighted average (0.3/0.7)
4. Compare final test accuracies across all subjects

**Status**: Pending

---

## Missing Baselines — EEGNet Ternary

**Date Added**: 2026-03-26

EEGNet ternary 在所有实验类型 (within_subject / cross_subject / transfer) 均无 baseline run。
详见 [`baseline_registry.md` 缺失表](experiments/baseline_registry.md#缺失-baseline-的类别)。

**需要**:
1. 运行 EEGNet ternary within-subject baseline
2. 运行 EEGNet ternary cross-subject baseline
3. 基于 cross-subject checkpoint 运行 transfer baseline

**Status**: Pending

---

## 分析数据加载 vs 训练时间占比 — 评估 Subject 级流水线化可行性

**Date Added**: 2026-03-26

`train_single_subject()` 现在会在每个 subject 训练结束后，将各阶段耗时（`train_data_loading`、`test_data_loading`、`training`、`val_evaluation`、`test_evaluation`）写入 `{save_dir}/timing_breakdown.csv`。

**分析目标**:
1. 统计 `train_data_loading` 占 `total_time` 的比例（across all subjects/models/tasks）
2. 如果数据加载占比 >10%，subject 级流水线化（当前 subject GPU 训练时后台加载下一个 subject 的数据）可带来可观收益
3. 如果占比 <5%，流水线化投入产出比不高，应优先优化训练循环本身

**操作步骤**:
1. 运行一轮完整的 within-subject comparison (binary, 21 subjects, EEGNet + CBraMod)
2. 读取生成的 `timing_breakdown.csv`
3. 计算 `data_loading_ratio = (train_data_loading + test_data_loading) / total_time`
4. 根据结果决定是否实施 P3.10 subject 级 data prefetch

**Status**: Pending — 等待下次完整实验运行后收集数据




## 
ablation test?