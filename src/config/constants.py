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
# Cache Type Identifiers
# ============================================================================

class CacheType:
    """缓存文件类型标识."""
    WITHIN_SUBJECT = 'comparison_cache'
    TRANSFER = 'transfer_comparison_cache'
    CROSS_SUBJECT = 'cross_subject_cache'

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
    'unified': {'classes': [1, 2, 3, 4], 'n_classes': 4},
}

# Class labels for finger targets
FINGER_LABELS: Dict[int, str] = {
    1: 'Thumb',
    2: 'Index',
    3: 'Middle',
    4: 'Pinky',
}

# ============================================================================
# Channel Configuration
# ============================================================================

# Total number of channels in the original BioSemi data collection
FULL_N_CHANNELS = 128

# Supported reduced channel counts for experiments
SUPPORTED_CHANNEL_COUNTS = [4, 8, 32, 61, FULL_N_CHANNELS]

# ============================================================================
# Preprocessing Version Tracking
# ============================================================================

# Current preprocessing version — increment when preprocessing logic changes
# Full version history with parameters: docs/preprocessing_versions.md
PREPROCESSING_VERSION = "v2.0"

# Version history for documentation and backfill
PREPROCESSING_VERSION_HISTORY: Dict[str, str] = {
    "v0.1": "CBraMod 19ch, segment-level cache, 125ms step, 60Hz notch",
    "v0.2": "CBraMod 128ch, trial-level cache (v3.0), 125ms step, 60Hz notch",
    "v1.0": "CBraMod 128ch, 500ms step, no notch; EEGNet unchanged throughout v0.x-v1.0",
    "v2.0": "Training trial amplitude rejection >500µV (both models)",
}

# Boundaries for preprocessing version backfill
# commit 0157fa1: CBraMod 128ch + trial-level cache (2026-01-11)
_PREPROCESSING_V0_2_TIMESTAMP = "2026-01-11T01:56:06"
# commit 52f1edf: CBraMod step 125ms→500ms, notch removed (2026-01-27)
_PREPROCESSING_V1_0_TIMESTAMP = "2026-01-27T03:11:09"
# commit 5bb2395: trial amplitude rejection >500µV (2026-03-02)
_PREPROCESSING_V2_0_TIMESTAMP = "2026-03-02T17:18:47"

# Legacy alias for existing code that references this
_PREPROCESSING_V2_TIMESTAMP = _PREPROCESSING_V2_0_TIMESTAMP
