"""
Data classes for training results.

This module defines the core data structures for representing
training results and plot data sources.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TrainingResult:
    """Result from a single training run."""
    subject_id: str
    task_type: str
    model_type: str
    best_val_acc: float
    test_acc: float
    test_acc_majority: float
    epochs_trained: int
    training_time: float


@dataclass
class PlotDataSource:
    """
    绘图数据源配置，用于组合对比图.

    Attributes:
        model_type: 模型类型 ('eegnet' 或 'cbramod')
        results: 训练结果列表
        is_current_run: True = 当前运行, False = 历史数据
        label: 图例标签
    """
    model_type: str           # 'eegnet' 或 'cbramod'
    results: List[TrainingResult]
    is_current_run: bool      # True = 当前运行, False = 历史数据
    label: str                # 图例标签


@dataclass
class ComparisonResult:
    """Result of statistical comparison between two models."""
    n_subjects: int
    eegnet_mean: float
    eegnet_std: float
    cbramod_mean: float
    cbramod_std: float
    difference_mean: float
    difference_std: float
    paired_ttest_t: float
    paired_ttest_p: float
    wilcoxon_stat: Optional[float]
    wilcoxon_p: Optional[float]
    better_model: str
    significant: bool
    # New fields with defaults for backward compatibility
    eegnet_median: float = 0.0
    cbramod_median: float = 0.0
