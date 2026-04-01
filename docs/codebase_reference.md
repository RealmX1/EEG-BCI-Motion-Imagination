# Codebase Reference

供 Claude Code 按需查阅的详细参考信息。主入口见 `CLAUDE.md`。

## 快速命令

```bash
# 安装
uv sync
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# 验证安装
uv run python scripts/verify_installation.py

# 被试内模型对比 (推荐)
uv run python scripts/run_within_subject_comparison.py                # 新运行 (默认)
uv run python scripts/run_within_subject_comparison.py --resume       # 恢复最近运行
uv run python scripts/run_within_subject_comparison.py --resume 20260205  # 恢复特定运行
uv run python scripts/run_within_subject_comparison.py --skip-training    # 仅查看结果
uv run python scripts/run_within_subject_comparison.py --paradigm movement  # Motor Execution

# 单模型训练 (WandB 默认启用)
uv run python scripts/run_within_subject.py --subject S01 --model eegnet --task binary
uv run python scripts/run_within_subject.py --subject S01 --model cbramod --no-wandb  # 禁用 WandB

# 数据预处理 (ZIP -> 缓存)
uv run python scripts/preprocess_zip.py                               # Motor Imagery
uv run python scripts/preprocess_zip.py --paradigm movement           # Motor Execution

# 缓存管理
uv run python scripts/cache_helper.py --stats
uv run python scripts/cache_helper.py --model cbramod --execute

# 跨被试训练与模型对比
uv run python scripts/run_cross_subject_comparison.py                    # 双模型对比 (EEGNet + CBraMod)
uv run python scripts/run_cross_subject_comparison.py --paradigm movement  # Motor Execution
uv run python scripts/run_cross_subject_comparison.py --no-within-subject-historical  # 无历史对比

# 迁移学习对比 (自动查找最优跨被试预训练模型 → 逐被试微调 → 对比)
uv run python scripts/run_transfer_comparison.py                           # 默认: backbone 冻结
uv run python scripts/run_transfer_comparison.py --freeze-strategy partial # 部分冻结
uv run python scripts/run_transfer_comparison.py --resume                  # 恢复运行
uv run python scripts/run_transfer_comparison.py --paradigm movement       # Motor Execution
uv run python scripts/run_transfer_comparison.py \
    --pretrained-eegnet checkpoints/cross_subject/.../best.pt \
    --pretrained-cbramod checkpoints/cross_subject/.../best.pt             # 手动指定检查点

# 迁移学习 (单模型, 通过 run_within_subject --pretrained)
uv run python scripts/run_within_subject.py --model eegnet \
    --pretrained checkpoints/cross_subject/eegnet_imagery_binary/best.pt \
    --freeze-strategy backbone --subjects S01
uv run python scripts/run_within_subject.py --model cbramod \
    --pretrained checkpoints/cross_subject/cbramod_imagery_binary/best.pt \
    --freeze-strategy none --cache-only

# 实验结果数据库
uv run python scripts/tools/migrate_results_to_db.py              # 预览迁移
uv run python scripts/tools/migrate_results_to_db.py --execute    # 执行迁移
uv run python scripts/tools/migrate_results_to_db.py --execute --force  # 重建数据库
uv run python scripts/tools/describe_run.py 0329_1357            # 按 run_tag substring 查看单次 run 摘要

# 通道缩减实验
uv run python scripts/analysis/compute_channel_selections.py                    # 数据驱动通道选择 (任意通道数)
uv run python scripts/experiments/run_32ch_config_comparison.py                # 6 配置对比
uv run python scripts/experiments/run_32ch_config_comparison.py --dry-run      # 仅显示命令
uv run python scripts/experiments/run_reduced_channel_experiment.py                       # 全量实验 (默认 motor_cortex)
uv run python scripts/experiments/run_reduced_channel_experiment.py --channel-config commercial  # 指定配置
```

## 关键文件

### src/ 模块

| 文件 | 说明 |
|------|------|
| `src/preprocessing/data_loader.py` | 数据加载和预处理管线 |
| `src/preprocessing/cache_manager.py` | HDF5 预处理缓存 (v3.0) |
| `src/models/eegnet.py` | EEGNet-8,2 实现 |
| `src/models/cbramod_adapter.py` | CBraMod 适配器 (支持 19/128 通道) |
| `src/training/common.py` | 共享训练工具 (时序分割、配置覆盖、性能优化) |
| `src/training/train_within_subject.py` | 被试内训练模块 (API) |
| `src/training/train_cross_subject.py` | 跨被试预训练模块 |
| `src/training/finetune.py` | 个体微调模块 (支持冻结策略) |
| `src/results/experiment_db.py` | SQLite 实验注册表 (ExperimentDB) — 元数据 + 最终指标 + 结构化查询 |
| `src/results/cache.py` | JSON 结果缓存 (旧系统，查询函数已标记 deprecated，由 ExperimentDB 替代) |
| `src/results/` | 结果管理 (dataclasses、序列化、统计) |
| `src/visualization/` | 可视化模块 (对比图、单模型图) |
| `src/config/` | 配置模块 (常量、预设、实验配置) |
| `src/evaluation/metrics.py` | 评估指标库 (TODO: 待集成到训练流程) |

### scripts/ 目录结构

```
scripts/
├── experiments/                # 训练实验脚本
│   ├── run_within_subject_comparison.py  # 被试内模型对比
│   ├── run_cross_subject_comparison.py   # 跨被试模型对比
│   ├── run_transfer_comparison.py       # 迁移学习对比 (跨被试→微调→对比)
│   ├── run_32ch_config_comparison.py   # 32ch 6 配置对比
│   ├── run_reduced_channel_experiment.py  # N-ch 全量实验 (任意通道数)
│   ├── run_8ch_experiment.py           # 8ch 全量实验
│   ├── run_within_subject.py    # 单模型训练 (被试内)
│   └── (run_finetune.py 已废弃, 功能由 run_within_subject.py --pretrained 吸收)
├── preprocessing/              # 数据预处理脚本
│   ├── preprocess_zip.py       # ZIP 解压和预处理
│   ├── cache_helper.py         # 缓存管理
│   └── merge_cache_index.py    # 缓存索引合并
├── tools/                      # 工具脚本
│   ├── verify_installation.py  # 安装验证
│   ├── compare_schedulers.py   # 调度器对比
│   ├── migrate_results_to_db.py # JSON → SQLite 一次性迁移
│   └── describe_run.py         # 按 run_tag substring 查询 run 摘要 + baseline 对比
├── analysis/                   # 分析脚本
│   ├── compute_channel_selections.py  # 数据驱动 N-ch 通道选择 (FDR/CSP/Attention/BandPower)
│   └── research/               # 研究分析
└── internal/                   # 内部工具
```

**向后兼容**: 根目录的 wrapper 脚本 (`scripts/run_*.py`) 仍然有效

## 文档索引

| 文档 | 说明 |
|------|------|
| `docs/TROUBLESHOOTING.md` | 故障排除指南 |
| `docs/preprocessing_architecture.md` | 预处理管线详细架构 |
| `docs/preprocessing_versions.md` | 预处理版本追踪 |
| `docs/dev_log/changelog.md` | 开发历史和变更记录 |
| `docs/dev_log/refactoring/` | 代码重构详细记录 (Phase 1-4) |
| `docs/dev_log/experiments/32ch_experiment.md` | 32 通道实验完整记录 (Step 1-7) |
| `docs/dev_log/experiments/reduced_channel_experiment_summary.md` | 减通道实验总结 (代码变更 + FDR 方法) |
| `docs/dev_log/implemented_plans/experiment_db.md` | SQLite 实验注册表实现文档 |
