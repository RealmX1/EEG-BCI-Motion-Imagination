# 跨被试训练结果管理与对比系统实施记录

**日期**: 2026-02-05
**状态**: 完成

## 背景

原有的 `run_cross_subject.py` 脚本存在以下问题：
- CLI 参数与 `run_single_model.py` 不一致
- 不生成 JSON 结果文件到 `results/` 目录
- 不生成可视化图表
- 无历史数据对比功能

## 目标

1. **CLI 对齐**: 使 `run_cross_subject.py` 的参数与其他训练脚本保持一致
2. **创建对比脚本**: 新建 `run_cross_subject_comparison.py`
3. **结果管理**: 保存 JSON 结果文件，文件名含 "cross-subject" 标识
4. **可视化集成**: 生成对比图，支持 within-subject 结果作为历史对比

## 实施内容

### Step 1: 结果序列化基础设施

#### 1.1 serialization.py 更新

**文件**: `src/results/serialization.py`

修改 `generate_result_filename()` 函数，添加 `is_cross_subject` 参数：

```python
def generate_result_filename(
    prefix: str,
    paradigm: str,
    task: str,
    ext: str = 'json',
    run_tag: Optional[str] = None,
    is_cross_subject: bool = False,  # 新增参数
) -> str:
```

输出示例：
- Within-subject: `20260205_1737_eegnet_imagery_binary.json`
- Cross-subject: `20260205_1737_cross-subject_eegnet_imagery_binary.json`

#### 1.2 cache.py 新增函数

**文件**: `src/results/cache.py`

新增三个函数：

| 函数 | 用途 |
|------|------|
| `find_compatible_within_subject_results()` | 搜索兼容的 within-subject 历史结果 |
| `find_compatible_cross_subject_results()` | 搜索兼容的 cross-subject 历史结果 |
| `save_cross_subject_result()` | 保存 cross-subject 训练结果到 JSON |

**搜索条件**：
- Within-subject: 文件模式 `*comparison_cache_{paradigm}_{task}.json`（排除 cross-subject）
- Cross-subject: 文件模式 `*cross-subject_*_{paradigm}_{task}.json`
- 被试集合必须覆盖当前运行的被试
- 支持 `BEST_ACCURACY` 和 `NEWEST` 两种选择策略

#### 1.3 results/__init__.py 更新

导出新函数：
```python
from .cache import (
    # ... 原有导出
    find_compatible_within_subject_results,
    find_compatible_cross_subject_results,
    save_cross_subject_result,
)
```

### Step 2: 可视化基础设施

#### 2.1 新建 cross_subject.py

**文件**: `src/visualization/cross_subject.py`

包含三个函数：

| 函数 | 用途 |
|------|------|
| `cross_subject_result_to_plot_data()` | 将 cross-subject 结果转换为 `PlotDataSource` |
| `generate_cross_subject_single_plot()` | 生成单模型结果图（2 子图） |
| `generate_cross_subject_comparison_plot()` | 生成对比图（3 子图，最多 5 个数据源） |

**单模型图布局**:
```
+------------------+------------------+
|  每被试准确率柱状图 |     箱线图        |
|  (可选历史对比)    |                  |
+------------------+------------------+
```

**对比图布局**:
```
+------------------+--------+----------+
|  每被试柱状图     | 箱线图  | 统计摘要  |
|  (最多5组数据)   |        |          |
+------------------+--------+----------+
```

**对比图数据源（最多 5 个）**:
1. EEGNet Within-Subject (历史, 半透明, 斜线填充)
2. CBraMod Within-Subject (历史, 半透明, 斜线填充)
3. CBraMod Cross-Subject (历史, 半透明, 点状填充)
4. EEGNet Cross-Subject (当前, 实心)
5. CBraMod Cross-Subject (当前, 实心)

#### 2.2 visualization/__init__.py 更新

导出新函数：
```python
from .cross_subject import (
    generate_cross_subject_single_plot,
    generate_cross_subject_comparison_plot,
    cross_subject_result_to_plot_data,
)
```

### Step 3: 更新 run_cross_subject.py

**文件**: `scripts/experiments/run_cross_subject.py`

#### 新增 CLI 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `--results-dir` | `str` | JSON 结果保存目录（默认: results） |
| `--no-plot` | `store_true` | 禁用图表生成 |
| `--no-historical` | `store_true` | 禁用历史数据检索 |

#### 新增功能

1. **生成 run_tag**: 训练开始时自动生成时间戳标签
2. **保存 JSON 结果**: 调用 `save_cross_subject_result()` 保存到 `results/` 目录
3. **生成可视化**:
   - 自动搜索 within-subject 历史数据
   - 调用 `generate_cross_subject_single_plot()` 生成图表

### Step 4: 创建 run_cross_subject_comparison.py

#### 4.1 主脚本

**文件**: `scripts/experiments/run_cross_subject_comparison.py`

**功能**:
- 顺序训练 EEGNet 和 CBraMod（双模型对比）
- 执行统计对比（t-test, Wilcoxon）
- 搜索历史数据（within-subject + cross-subject）
- 生成对比图

**CLI 参数**:
| 参数 | 说明 |
|------|------|
| `--models` | 要训练的模型（默认: both） |
| `--no-within-subject-historical` | 禁用 within-subject 历史数据 |
| `--no-cross-subject-historical` | 禁用 cross-subject 历史数据 |

#### 4.2 向后兼容 Wrapper

**文件**: `scripts/run_cross_subject_comparison.py`

```python
from scripts.experiments.run_cross_subject_comparison import main

if __name__ == '__main__':
    main()
```

### Step 5: 文档更新

**文件**: `CLAUDE.md`

更新内容：
1. 快速命令部分添加 `run_cross_subject_comparison.py` 示例
2. scripts 目录结构添加新脚本说明

## 文件命名规范

| 文件类型 | 命名格式 | 示例 |
|---------|---------|------|
| 单模型结果 | `{tag}_cross-subject_{model}_{paradigm}_{task}.json` | `20260205_1737_cross-subject_eegnet_imagery_binary.json` |
| 单模型图 | `{tag}_cross-subject_{model}_{paradigm}_{task}.png` | `20260205_1737_cross-subject_eegnet_imagery_binary.png` |
| 对比图 | `{tag}_cross-subject_combined_{paradigm}_{task}.png` | `20260205_1737_cross-subject_combined_imagery_binary.png` |

## JSON 结果结构

```json
{
  "metadata": {
    "type": "cross-subject",
    "model_type": "eegnet",
    "paradigm": "imagery",
    "task": "binary",
    "subjects": ["S01", "S02", "..."],
    "n_subjects": 21,
    "run_tag": "20260205_1737",
    "timestamp": "2026-02-05T17:37:00"
  },
  "results": {
    "per_subject_test_acc": {"S01": 0.85, "S02": 0.78, "...": "..."},
    "mean_test_acc": 0.8500,
    "std_test_acc": 0.0609,
    "best_val_acc": 0.8234,
    "best_epoch": 42
  },
  "training_info": {
    "training_time": 3456.7,
    "model_path": "checkpoints/cross_subject/eegnet_imagery_binary/best.pt"
  }
}
```

## 修改文件清单

| 文件 | 操作 | 行数变化 |
|------|------|---------|
| `src/results/serialization.py` | 修改 | +25 |
| `src/results/cache.py` | 修改 | +360 |
| `src/results/__init__.py` | 修改 | +6 |
| `src/visualization/cross_subject.py` | **新建** | +380 |
| `src/visualization/__init__.py` | 修改 | +12 |
| `scripts/experiments/run_cross_subject.py` | 修改 | +55 |
| `scripts/experiments/run_cross_subject_comparison.py` | **新建** | +280 |
| `scripts/run_cross_subject_comparison.py` | **新建** | +10 |
| `CLAUDE.md` | 修改 | +5 |

## 使用示例

```bash
# 单模型跨被试训练（自动保存 JSON + 生成图表）
uv run python scripts/run_cross_subject.py --model eegnet

# 单模型跨被试训练（无历史对比）
uv run python scripts/run_cross_subject.py --model cbramod --no-historical

# 双模型跨被试对比
uv run python scripts/run_cross_subject_comparison.py

# 双模型对比（Motor Execution）
uv run python scripts/run_cross_subject_comparison.py --paradigm movement

# 双模型对比（禁用 within-subject 历史）
uv run python scripts/run_cross_subject_comparison.py --no-within-subject-historical
```

## 验证

所有代码已通过语法验证：
- `src/results` 模块导入正常
- `src/visualization` 模块导入正常
- 所有脚本 `--help` 正常工作

## 后续工作

1. 运行完整训练测试验证功能
2. 验证历史数据搜索在实际场景中的表现
3. 考虑添加 `--resume` 功能到 cross-subject 脚本（如需要）
