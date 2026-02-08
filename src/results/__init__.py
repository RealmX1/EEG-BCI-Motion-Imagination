"""
Results management module for EEG-BCI project.

This module provides centralized result handling:
- dataclasses.py: TrainingResult, PlotDataSource data classes
- serialization.py: Result serialization (to/from dict)
- cache.py: Training results caching (load/save)
- statistics.py: Statistical analysis functions

Usage:
    from src.results import TrainingResult, PlotDataSource
    from src.results import load_cache, save_cache
    from src.results import compute_model_statistics
"""

from .dataclasses import TrainingResult, PlotDataSource, ComparisonResult
from .serialization import (
    result_to_dict,
    dict_to_result,
    generate_result_filename,
)
from .cache import (
    load_cache,
    save_cache,
    find_latest_cache,
    find_cache_by_tag,
    get_cache_path,
    find_compatible_historical_results,
    build_data_sources_from_historical,
    prepare_combined_plot_data,
    SelectionStrategy,
    # Full comparison results IO
    save_full_comparison_results,
    load_comparison_results,
    # Single model results IO
    save_single_model_results,
    load_single_model_results,
)
from .statistics import (
    compute_model_statistics,
    print_model_summary,
    compare_models,
    print_comparison_report,
)

__all__ = [
    # Data classes
    'TrainingResult',
    'PlotDataSource',
    'ComparisonResult',
    # Serialization
    'result_to_dict',
    'dict_to_result',
    'generate_result_filename',
    # Cache
    'load_cache',
    'save_cache',
    'find_latest_cache',
    'find_cache_by_tag',
    'get_cache_path',
    'find_compatible_historical_results',
    'build_data_sources_from_historical',
    'prepare_combined_plot_data',
    'SelectionStrategy',
    # Full comparison results IO
    'save_full_comparison_results',
    'load_comparison_results',
    # Single model results IO
    'save_single_model_results',
    'load_single_model_results',
    # Statistics
    'compute_model_statistics',
    'print_model_summary',
    'compare_models',
    'print_comparison_report',
]
