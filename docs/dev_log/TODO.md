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
