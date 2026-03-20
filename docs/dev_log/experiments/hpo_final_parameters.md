# HPO 最终参数对照表

> **日期**: 2026-03-20
> **状态**: 已应用到 `src/config/training.py` 并完成 unified model 验证运行
> **配置来源**: `src/config/training.py` (commit `fee43c1`, `1a09704`)

本文档是 HPO 超参数搜索的最终参考——记录每个参数从 "HPO 建议" 到 "实际采用" 的完整决策链。

---

## 1. CBraMod Within-Subject

> HPO study: `cbramod_within_subject_binary` (51 trials, 23 complete, best=Trial #46: 86.01%)
> HPO 参数导出: `results/hpo/cbramod_within_subject_binary_best_params.json`
> fANOVA 分析: `paper/analysis/hpo_within_subject_analysis.md`

| 参数 | 旧默认 | HPO 建议 | **实际采用** | 决策 |
|------|--------|---------|-------------|------|
| backbone_lr | 1e-4 | 2.87e-4 | **2.9e-4** | 采纳 HPO，rounded |
| classifier_lr | 3e-4 (ratio 3×) | 1.16e-3 (ratio 4×) | **1.2e-3** | 采纳 HPO，rounded |
| weight_decay | 0.06 | 0.026 | **0.026** | 采纳 HPO |
| dropout_rate | 0.15 | 0.098 | **0.10** | 采纳 HPO，rounded |
| batch_size | 128 | 256 | **256** | 采纳 HPO |
| label_smoothing | 0.05 | 0.087 | **0.05** | **用户 override**: 保持原值，避免 quaternary 弱信号削弱 |
| gradient_clip | — (default 1.0) | 0.729 | **0.73** | 采纳 HPO，rounded |
| phase_decay (CAWD) | 0.7 | 0.468 | **0.47** | 采纳 HPO，rounded |
| phase_epochs (CAWD) | 6 | 8 | **8** | 采纳 HPO |
| exploration_epochs (CAWD) | 6 | 4 | **4** | 采纳 HPO |
| exploration_batch_size (CAWD) | 32 | 64 | **64** | 采纳 HPO |

---

## 2. EEGNet Within-Subject

> HPO study: `eegnet_within_subject_binary` (32 trials, 10 complete, best=Trial #23: 82.71%)
> HPO 参数导出: `results/hpo/eegnet_within_subject_binary_best_params.json`
> fANOVA 分析: `paper/analysis/hpo_within_subject_analysis.md` (Part II)

| 参数 | 旧默认 | HPO 建议 | **实际采用** | 决策 |
|------|--------|---------|-------------|------|
| F1 | 8 | 16 | **16** | 采纳 HPO |
| D | 2 | 4 | **4** | 采纳 HPO |
| F2 (= F1×D) | 16 | 64 | **64** | 自动计算 |
| model name | EEGNet-8,2 | EEGNet-16,4 | **EEGNet-16,4** | 跟随架构 |
| learning_rate | 1e-3 | 3.98e-3 | **4e-3** | 采纳 HPO，rounded |
| weight_decay | 0 | 1.09e-5 | **1e-5** | 采纳 HPO，rounded |
| dropout_rate | 0.5 | 0.271 | **0.27** | 采纳 HPO，rounded |
| batch_size | 64 | 64 | **64** | 不变 |
| kernel_length | 64 | 64 | **64** | 不变 |

---

## 3. CBraMod Cross-Subject

> HPO study: `cbramod_cross_subject_binary` (77 trials, 43 complete, best=Trial #4: 90.68%)
> HPO 参数导出: `results/hpo/cbramod_cross_subject_binary_best_params.json`
> fANOVA 分析: `paper/analysis/hpo_cross_subject_analysis.md`

| 参数 | 旧默认 | HPO 建议 | **实际采用** | 决策 |
|------|--------|---------|-------------|------|
| backbone_lr | 1e-4 | 1.335e-4 | **1.3e-4** | 采纳 HPO，rounded |
| classifier_lr | 1.5e-4 (ratio 1.5×) | 2.17e-4 (ratio 1.6×) | **2.2e-4** | 采纳 HPO，rounded |
| weight_decay | 0.12 | 0.130 | **0.13** | 采纳 HPO |
| dropout_rate | 0.35 | 0.369 | **0.37** | 采纳 HPO，rounded |
| batch_size | 256 | 256 | **256** | 不变 |
| label_smoothing | 0.15 | 0.285 | **0.05** | **用户 override**: 大幅降低，避免 quaternary 弱信号削弱 |
| gradient_clip | 0.5 | 1.363 | **1.4** | 采纳 HPO，rounded |
| epochs | 100 | — | **100** | 不变 |
| phase_decay (CAWD) | 0.5 | 0.499 | **0.50** | 采纳 HPO (本质不变) |
| phase_epochs (CAWD) | 6 | 10 | **10** | 采纳 HPO |
| exploration_epochs (CAWD) | 6 | 3 | **3** | 采纳 HPO |
| exploration_batch_size (CAWD) | 64 | 128 | **128** | 采纳 HPO |

---

## 4. EEGNet Cross-Subject (手工适配)

> **无 HPO 数据**: EEGNet cross-subject HPO 仅完成 1/4 trials (56.99%)，不可靠。
> 以下参数基于 within-subject HPO 架构发现 + 跨被试保守正则化策略手工适配。

| 参数 | 旧默认 | **实际采用** | 适配理由 |
|------|--------|-------------|---------|
| F1 | 8 (继承 within) | **16** | Within HPO: 架构升级是最大 lever (+3.8pp) |
| D | 2 (继承 within) | **4** | Within HPO |
| F2 | 16 (继承 within) | **64** | F1×D |
| model name | EEGNet-8,2 | **EEGNet-16,4** | 跟随架构 |
| dropout_rate | 0.5 (继承 within) | **0.35** | 低于旧 0.5，但高于 within HPO 0.27（cross 需更强正则） |
| learning_rate | 5e-4 | **1e-3** | Within HPO (4e-3) 和旧 cross (5e-4) 之间取中间值 |
| weight_decay | 1e-4 | **1e-4** | 不变 |
| batch_size | 128 | **128** | 不变 |
| epochs | 50 | **50** | 不变 |

---

## 5. 用户 Override 说明

### Label Smoothing 固定为 0.05

**HPO 建议**: within 0.087, cross 0.285
**实际采用**: 全部 0.05

**理由**: Unified model 包含 quaternary 子任务 (chance=25%)。高 label smoothing (如 0.285) 将真实标签概率从 1.0 压至 0.715，uniform 概率为 0.095 (= 0.285/3)。有效信号 margin 从 0.75 降至 0.52，严重削弱本就微弱的 4-class 学习信号。

HPO 搜索在 binary 任务上进行（chance=50%，margin 充裕），其 label smoothing 偏好不能直接迁移到 quaternary 场景。

---

## 6. HPO 优化效果 (Unified Model, 128ch, 21 subjects)

| 范式 | 模型 | Default (0319) | HPO (0320) | Delta | 来源 |
|------|------|---------------|------------|-------|------|
| Within | EEGNet | 60.69% | **65.21%** | **+4.52pp** | `results/20260320_0243_comparison_cache_imagery_unified.json` |
| Within | CBraMod | 66.69% | **66.96%** | +0.27pp | 同上 |
| Cross | EEGNet | 52.61% | **60.12%** | **+7.51pp** | `results/20260320_0548_cross_subject_cache_imagery_unified.json` |
| Cross | CBraMod | 67.68% | **68.17%** | +0.49pp | 同上 |

**结论**: EEGNet 架构升级 (8,2→16,4, ~2.5K→~10K params) 是 HPO 最大收益来源。CBraMod HPO 参数微调收益有限 (~0.3-0.5pp)，符合预期——4M 参数预训练基座模型对超参数不敏感。

---

## 7. 文件索引

| 内容 | 路径 |
|------|------|
| **本文档 (最终参数)** | `docs/dev_log/experiments/hpo_final_parameters.md` |
| HPO 系统设计 | `docs/hpo_implementation_plan.md` |
| fANOVA 分析 (within) | `paper/analysis/hpo_within_subject_analysis.md` |
| fANOVA 分析 (cross) | `paper/analysis/hpo_cross_subject_analysis.md` |
| Unified 实验报告 | `docs/dev_log/experiments/unified_model_results_report.md` |
| 配置代码 (source of truth) | `src/config/training.py` |
| HPO 数据库 | `results/hpo/hpo.db` |
| HPO 最优参数 JSON | `results/hpo/*_best_params.json` |
| HPO 搜索空间定义 | `src/hpo/search_spaces.py` |
