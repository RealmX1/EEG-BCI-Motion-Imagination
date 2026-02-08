"""
Configuration module for EEG-BCI project.

This module provides centralized configuration management:
- constants.py: Global constants (colors, paradigm configs)
- training.py: Training configurations and scheduler presets
- experiment_config.py: ML engineering experiment configurations (research)

Usage:
    from src.config import MODEL_COLORS, PARADIGM_CONFIG
    from src.config.training import SCHEDULER_PRESETS, get_default_config
    from src.config.experiment_config import get_experiment_config  # research
"""

from .constants import (
    MODEL_COLORS,
    PARADIGM_CONFIG,
    CACHE_FILENAME,
    CACHE_FILENAME_WITH_TAG,
    TASKS,
)
from .training import (
    SCHEDULER_PRESETS,
    CROSS_SUBJECT_SCHEDULER_OVERRIDES,
    get_default_config,
    get_cross_subject_config,
    load_yaml_config,
)
from .experiment_config import (
    ExperimentPreprocessConfig,
    ALL_EXPERIMENTS,
    EXPERIMENT_GROUPS,
    get_experiment_config,
    get_experiments_in_group,
)

__all__ = [
    # Constants
    'MODEL_COLORS',
    'PARADIGM_CONFIG',
    'CACHE_FILENAME',
    'CACHE_FILENAME_WITH_TAG',
    'TASKS',
    # Training
    'SCHEDULER_PRESETS',
    'CROSS_SUBJECT_SCHEDULER_OVERRIDES',
    'get_default_config',
    'get_cross_subject_config',
    'load_yaml_config',
    # Experiment (research)
    'ExperimentPreprocessConfig',
    'ALL_EXPERIMENTS',
    'EXPERIMENT_GROUPS',
    'get_experiment_config',
    'get_experiments_in_group',
]
