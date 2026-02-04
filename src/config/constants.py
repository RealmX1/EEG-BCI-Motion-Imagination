"""
Global constants for EEG-BCI project.

This module contains shared constants used across the codebase:
- Model colors for visualization
- Paradigm configurations
- Cache filename patterns
- Task definitions
"""

from typing import Dict, List

# ============================================================================
# Model Colors (for visualization)
# ============================================================================

MODEL_COLORS: Dict[str, str] = {
    'eegnet': '#2E86AB',   # Blue
    'cbramod': '#E94F37',  # Red/Coral
}

# ============================================================================
# Paradigm Configuration
# ============================================================================

PARADIGM_CONFIG: Dict[str, Dict[str, str]] = {
    'imagery': {
        'description': 'Motor Imagery (MI)',
    },
    'movement': {
        'description': 'Motor Execution (ME)',
    },
}

# ============================================================================
# Cache Filename Patterns
# ============================================================================

CACHE_FILENAME = 'comparison_cache_{paradigm}_{task}.json'
CACHE_FILENAME_WITH_TAG = '{tag}_comparison_cache_{paradigm}_{task}.json'

# ============================================================================
# Task Definitions
# ============================================================================

TASKS: Dict[str, Dict[str, any]] = {
    'binary': {'classes': [1, 4], 'n_classes': 2},
    'ternary': {'classes': [1, 2, 4], 'n_classes': 3},
    'quaternary': {'classes': [1, 2, 3, 4], 'n_classes': 4},
}

# Class labels for finger targets
FINGER_LABELS: Dict[int, str] = {
    1: 'Thumb',
    2: 'Index',
    3: 'Middle',
    4: 'Pinky',
}
