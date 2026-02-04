"""
Visualization module for EEG-BCI project.

This module provides plotting functions:
- plots.py: Base plotting utilities (colors, styles)
- comparison.py: Model comparison plots
- single_model.py: Single model plots

Usage:
    from src.visualization import generate_combined_plot, generate_single_model_plot
    from src.visualization.plots import MODEL_COLORS
"""

from .comparison import generate_combined_plot, generate_comparison_plot
from .single_model import generate_single_model_plot
from .plots import MODEL_COLORS, CHANCE_LEVELS

__all__ = [
    'generate_combined_plot',
    'generate_comparison_plot',
    'generate_single_model_plot',
    'MODEL_COLORS',
    'CHANCE_LEVELS',
]
