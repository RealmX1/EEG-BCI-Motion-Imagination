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

**分析脚本**: `scripts/analysis/analyze_timing_breakdown.py`

**分析结果** (2026-03-30):

基于 3 个完整运行（CBraMod binary 21 subjects, CBraMod ternary 21 subjects, EEGNet ternary 21 subjects）的 63 条 subject-run 记录：

| 指标 | Overall | CBraMod | EEGNet |
|------|---------|---------|--------|
| data_loading_ratio (mean) | 10.05% | 9.7% | 10.8% |
| train_data_loading (mean) | 8.74% | 8.46% | 9.32% |
| training (mean) | 85.80% | 85.50% | 86.42% |

> **数据来源**: `scripts/analysis/analyze_timing_breakdown.py --latest-only`，扫描 `results/` + `checkpoints/` 下 13 个 timing_breakdown.csv

**结论**: data_loading_ratio 均值 ~10%，达到 RECOMMENDED 阈值。Subject 级流水线化（当前 subject GPU 训练时后台预加载下一个 subject 数据）可带来约 10% 的 wall-clock 时间节省。

**实施** (2026-03-30):

`SubjectPrefetcher` (`src/training/prefetch.py`) 使用 `ThreadPoolExecutor(max_workers=1)` 在当前 subject GPU 训练期间后台加载下一个 subject 的数据。集成在 `run_within_subject()` 中，transfer learning 自动受益。

**Status**: Done — 分析完成，prefetch 已实施




## Ablation Testing

**Date Added**: 2026-03-27

对 CBraMod 和 EEGNet 进行消融实验，系统性地评估各组件/超参数对最终性能的贡献。

**待定内容**:
- 具体消融维度（预训练权重、数据增强、滤波范围、归一化方式等）
- 实验范围（within-subject / cross-subject / transfer）
- 评估指标与对比基准

**Status**: Pending — 需进一步确定消融方案

---

## Reduced Channel Extra Sessions 实验

**Date Added**: 2026-03-27

在 extra sessions 实验中测试降通道配置（4ch / 8ch / 32ch）的表现，验证通道选择策略在额外 session 数据上的泛化能力。

**动机**:
- 当前 extra sessions 实验仅在 128 通道上运行
- 需要验证 FDR 等通道选择方法在 extra sessions 场景下是否仍然有效
- 对比降通道在标准实验与 extra sessions 实验中的性能差异

**Status**: Pending

---

## Leave-One-Out Transfer Learning 实验

**Date Added**: 2026-03-29

使用 leave-one-out cross-subject 训练 → 对目标被试进行 transfer learning 的范式。对每个被试，用其余 20 个被试的数据训练 cross-subject 模型，然后对该被试进行 per-subject 微调。

**动机**:
- 当前 transfer learning 使用包含目标被试的 cross-subject checkpoint——目标被试的数据同时参与了预训练和微调，存在间接信息泄露风险
- Leave-one-out 是更严格的 transfer learning 评估方案：预训练模型完全未见过目标被试的 EEG 数据
- 对比当前方案（含目标被试预训练）vs leave-one-out（排除目标被试预训练），量化信息泄露对 transfer 性能的影响

**设计思路**:
1. 对每个被试 S_i，使用其余 20 个被试训练 cross-subject 模型
2. 用该模型作为 S_i 的 transfer learning 初始化权重
3. 对 S_i 进行标准 within-subject 微调
4. 可复用现有 cross-subject extra sessions 的 `+Sess05` checkpoint 作为近似方案（该 checkpoint 在 21 subjects 的标准数据 + 16 subjects 的 extra sessions 上训练，包含更多泛化信息但非严格 leave-one-out）

**实现路径**:
- 需新增脚本或修改 `run_cross_subject_comparison.py` 支持 `--leave-out SUBJECT` 参数
- 或利用已有的 cross-subject extra sessions `+Sess05` checkpoint（`20260326_1409_sess05_cbramod_imagery_binary/best.pt`）作为近似 leave-one-out 预训练

**预期结果**:
- 如果当前 transfer 与 leave-one-out 差异很小（<1pp），说明 cross-subject 预训练中目标被试的贡献被其他 20 人稀释，信息泄露风险可忽略
- 如果差异显著，需在论文中明确声明 transfer learning 的评估局限性

**Status**: Pending — 可使用 `run_extra_sessions.py --pretrained-run` 快速运行近似版本

---

## 论文脚本目录重组

**Date Added**: 2026-03-31

当前论文相关脚本分散在多处：
- `scripts/paper/` — 图表生成 (`generate_paper_figures.py`, `generate_extra_sessions_plots.py`) 和统计计算 (`compute_paper_statistics.py`)
- `scripts/analysis/` — 通用分析脚本 (`analyze_timing_breakdown.py` 等)
- `paper/analysis/` — 分析文档（markdown，非脚本）

未来考虑将论文相关的图表生成、统计计算、数据提取脚本统一组织到一个目录下（如 `scripts/paper/` 或 `paper/scripts/`），并与 `paper/analysis/` 中的分析文档建立清晰的对应关系。

**Status**: Pending — 低优先级，当前结构可用