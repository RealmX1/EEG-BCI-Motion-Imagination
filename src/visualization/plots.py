"""
Base plotting utilities for EEG-BCI project.

This module provides shared constants and helper functions
for visualization across the project.
"""

from typing import Dict

# Re-export from config for convenience
from ..config.constants import MODEL_COLORS

# Chance levels for different task types
CHANCE_LEVELS: Dict[str, float] = {
    'binary': 0.5,
    'ternary': 1/3,
    'quaternary': 0.25,
}


def get_chance_level(task_type: str) -> float:
    """Get chance level for a task type.

    Args:
        task_type: 'binary', 'ternary', or 'quaternary'

    Returns:
        Chance level (0.5, 0.33, or 0.25)
    """
    return CHANCE_LEVELS.get(task_type, 0.5)
