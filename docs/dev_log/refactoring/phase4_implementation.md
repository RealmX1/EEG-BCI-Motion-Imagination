# Phase 4 Implementation Log

**Date**: 2026-02-05
**Status**: Completed
**Goal**: 脚本目录重组 (`scripts/` 子目录化)

---

## Overview

Phase 4 是代码重构计划中的低风险阶段，目标是将 `scripts/` 目录中的 20+ 脚本按功能分类到子目录中，同时保持向后兼容。

### 重构前状态

```
scripts/
├── _training_utils.py          # 共享工具
├── _wandb_setup.py             # WandB 设置
├── run_full_comparison.py      # 混在根目录
├── run_single_model.py
├── run_cross_subject.py
├── run_finetune.py
├── preprocess_zip.py
├── cache_helper.py
├── verify_installation.py
├── ... (15+ 其他脚本)
├── cache_helpers/              # 已存在
└── research/                   # 已存在
```

### 重构后状态

```
scripts/
├── _training_utils.py          # 保留: 共享工具
├── _wandb_setup.py             # 保留: WandB 设置
├── run_*.py                    # Wrapper 脚本 (向后兼容)
├── experiments/                # 训练实验脚本 (4 个)
│   ├── run_full_comparison.py
│   ├── run_single_model.py
│   ├── run_cross_subject.py
│   └── run_finetune.py
├── preprocessing/              # 数据预处理脚本 (4 个)
│   ├── preprocess_zip.py
│   ├── cache_helper.py
│   ├── merge_cache_index.py
│   └── create_preprocess_package.py
├── tools/                      # 工具脚本 (3 个)
│   ├── verify_installation.py
│   ├── verify_trial_split.py
│   └── compare_schedulers.py
├── analysis/                   # 分析脚本 (4 个 + research/)
│   ├── compile_preproc_report.py
│   ├── visualize_comparison.py
│   ├── visualize_preproc_timing.py
│   ├── run_preproc_experiment.py
│   └── research/               # 研究分析子目录
│       └── wandb_analysis/
└── internal/                   # 内部工具 (3 个 + cache_helpers/)
    ├── cleanup_merge_temp.py
    ├── package_caches.py
    ├── test_wandb_setup.py
    └── cache_helpers/
```

---

## 迁移的文件

### experiments/ (4 个脚本)

| 文件 | 原位置 | 说明 |
|------|--------|------|
| `run_full_comparison.py` | scripts/ | 全被试模型对比 |
| `run_single_model.py` | scripts/ | 单模型训练 |
| `run_cross_subject.py` | scripts/ | 跨被试预训练 |
| `run_finetune.py` | scripts/ | 个体微调 |

### preprocessing/ (4 个脚本)

| 文件 | 原位置 | 说明 |
|------|--------|------|
| `preprocess_zip.py` | scripts/ | ZIP 解压和预处理 |
| `cache_helper.py` | scripts/ | 缓存管理 |
| `merge_cache_index.py` | scripts/ | 缓存索引合并 |
| `create_preprocess_package.py` | scripts/ | 创建预处理包 |

### tools/ (3 个脚本)

| 文件 | 原位置 | 说明 |
|------|--------|------|
| `verify_installation.py` | scripts/ | 安装验证 |
| `verify_trial_split.py` | scripts/ | Trial 分割验证 |
| `compare_schedulers.py` | scripts/ | 调度器对比 |

### analysis/ (4 个脚本 + research/)

| 文件 | 原位置 | 说明 |
|------|--------|------|
| `compile_preproc_report.py` | scripts/ | 预处理报告 |
| `visualize_comparison.py` | scripts/ | 可视化对比 |
| `visualize_preproc_timing.py` | scripts/ | 预处理时间分析 |
| `run_preproc_experiment.py` | scripts/ | 预处理实验 |
| `research/` | scripts/ | 研究分析目录 |

### internal/ (3 个脚本 + cache_helpers/)

| 文件 | 原位置 | 说明 |
|------|--------|------|
| `cleanup_merge_temp.py` | scripts/ | 清理临时文件 |
| `package_caches.py` | scripts/ | 打包缓存 |
| `test_wandb_setup.py` | scripts/ | WandB 测试 |
| `cache_helpers/` | scripts/ | 缓存辅助脚本 |

---

## 路径更新

所有迁移的脚本都更新了 `PROJECT_ROOT` 计算方式：

```python
# 旧: scripts/ -> project root
PROJECT_ROOT = Path(__file__).parent.parent

# 新: scripts/experiments/ -> scripts/ -> project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
```

需要导入 `scripts/` 下工具的脚本还添加了：

```python
SCRIPTS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
```

---

## 向后兼容

在 `scripts/` 根目录创建了 wrapper 脚本：

```python
# scripts/run_full_comparison.py (wrapper)
from scripts.experiments.run_full_comparison import main

if __name__ == '__main__':
    main()
```

创建的 wrapper 脚本:
- `scripts/run_full_comparison.py`
- `scripts/run_single_model.py`
- `scripts/run_cross_subject.py`
- `scripts/run_finetune.py`
- `scripts/preprocess_zip.py`
- `scripts/verify_installation.py`
- `scripts/cache_helper.py`

---

## 验证结果

### 新路径测试

```
scripts/experiments/run_full_comparison.py --help: OK
scripts/experiments/run_single_model.py --help: OK
scripts/experiments/run_cross_subject.py --help: OK
scripts/experiments/run_finetune.py --help: OK
scripts/preprocessing/preprocess_zip.py --help: OK
scripts/preprocessing/cache_helper.py --help: OK
scripts/tools/verify_installation.py: OK
```

### Wrapper 测试 (向后兼容)

```
scripts/run_full_comparison.py --help: OK
scripts/run_single_model.py --help: OK
scripts/preprocess_zip.py --help: OK
scripts/cache_helper.py --help: OK
scripts/verify_installation.py: OK
```

---

## CLAUDE.md 更新

更新了关键文件部分，添加了 `scripts/` 目录结构说明。

---

## 重构计划完成状态

| Phase | 目标 | 状态 |
|-------|------|------|
| Phase 1 | 创建新模块 (config/, results/, visualization/) | Completed |
| Phase 2 | 拆分大文件 (暂缓) | Deferred |
| Phase 3 | 精简脚本层 | Completed |
| Phase 4 | 脚本目录重组 | **Completed** |

---

## 统计

| 指标 | 数值 |
|------|------|
| 迁移的脚本 | 16 个 |
| 创建的子目录 | 5 个 |
| 创建的 wrapper | 8 个 |
| 更新的路径 | 10 个脚本 |
