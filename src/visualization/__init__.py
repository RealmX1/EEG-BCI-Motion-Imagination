"""
Visualization module for EEG-BCI project.

This module provides plotting functions:
- plots.py: Base plotting utilities (colors, styles)
- comparison.py: Model comparison plots
- single_model.py: Single model plots
- cross_subject.py: Cross-subject training plots
- electrode_map.py: Electrode placement visualization (2D/3D)

Usage:
    from src.visualization import generate_combined_plot, generate_single_model_plot
    from src.visualization import generate_cross_subject_single_plot
    from src.visualization.plots import MODEL_COLORS
    from src.visualization.electrode_map import plot_electrode_grid, plot_electrode_overlap
"""

from .comparison import generate_combined_plot, generate_comparison_plot
from .single_model import generate_single_model_plot
from .cross_subject import generate_cross_subject_single_plot, generate_config_comparison_plot
from .milestone import generate_milestone_plot
from .plots import MODEL_COLORS, CHANCE_LEVELS
try:
    from .electrode_map import (
        create_mne_montage,
        create_mne_info,
        plot_electrode_grid,
        plot_electrode_overlap,
        plot_electrode_placement_2d,
        plot_electrode_placement_3d,
        plot_electrode_3d_multiview,
        plot_region_distribution,
        CONFIG_COLORS,
    )
    _HAS_ELECTRODE_MAP = True
except ImportError:
    _HAS_ELECTRODE_MAP = False

__all__ = [
    'generate_combined_plot',
    'generate_comparison_plot',
    'generate_single_model_plot',
    # Cross-subject
    'generate_cross_subject_single_plot',
    'generate_config_comparison_plot',
    # Milestone
    'generate_milestone_plot',
    # Utilities
    'MODEL_COLORS',
    'CHANCE_LEVELS',
]

if _HAS_ELECTRODE_MAP:
    __all__ += [
        'create_mne_montage',
        'create_mne_info',
        'plot_electrode_grid',
        'plot_electrode_overlap',
        'plot_electrode_placement_2d',
        'plot_electrode_placement_3d',
        'plot_electrode_3d_multiview',
        'plot_region_distribution',
        'CONFIG_COLORS',
    ]
