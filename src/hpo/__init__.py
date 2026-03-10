"""
HPO (Hyperparameter Optimization) 模块。

基于 Optuna 的超参数搜索系统，支持：
- Within-subject / Cross-subject / Transfer 三种训练范式
- ProbabilisticSubjectPruner 概率剪枝
- CBraMod + EEGNet 模型
"""

from .objectives import (
    cross_subject_objective,
    transfer_objective,
    within_subject_objective,
)
from .pruner import ProbabilisticSubjectPruner
from .search_spaces import params_to_config_overrides, sample_search_space
