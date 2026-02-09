"""
Visualization module for EEG-BCI project.

This module provides plotting functions:
- plots.py: Base plotting utilities (colors, styles)
- comparison.py: Model comparison plots
- single_model.py: Single model plots
- cross_subject.py: Cross-subject training plots

Usage:
    from src.visualization import generate_combined_plot, generate_single_model_plot
    from src.visualization import generate_cross_subject_single_plot
    from src.visualization.plots import MODEL_COLORS
"""

from .comparison import generate_combined_plot, generate_comparison_plot
from .single_model import generate_single_model_plot
from .cross_subject import generate_cross_subject_single_plot
from .plots import MODEL_COLORS, CHANCE_LEVELS

__all__ = [
    'generate_combined_plot',
    'generate_comparison_plot',
    'generate_single_model_plot',
    # Cross-subject
    'generate_cross_subject_single_plot',
    # Utilities
    'MODEL_COLORS',
    'CHANCE_LEVELS',
]
