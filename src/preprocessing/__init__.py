# Preprocessing module
"""
Data preprocessing utilities for EEG-BCI project.

Modules:
- channel_selection: BioSemi 128 to standard 10-20 channel mapping
- filtering: Bandpass and notch filtering
- resampling: Resampling to target frequency
- cache_manager: HDF5 preprocessing cache
- loader: MAT file loading utilities
- discovery: Subject and session discovery
- pipeline: Preprocessing pipeline functions
- dataset: PyTorch Dataset and DataLoader
- data_loader: Backward compatibility module (re-exports from above)

Usage:
    # Recommended: import from specific modules
    from src.preprocessing.loader import load_mat_file
    from src.preprocessing.pipeline import preprocess_run_paper_aligned
    from src.preprocessing.dataset import FingerEEGDataset
    from src.preprocessing.discovery import discover_available_subjects

    # Backward compatible: import from data_loader
    from src.preprocessing.data_loader import (
        PreprocessConfig,
        FingerEEGDataset,
        load_mat_file,
    )
"""

from .channel_selection import (
    create_biosemi128_to_1020_mapping,
    get_channel_indices,
    STANDARD_1020_CHANNELS,
    BIOSEMI_128_LABELS,
)
from .filtering import bandpass_filter, notch_filter
from .resampling import resample_eeg

# Re-exports from new modules (also available via data_loader for backward compat)
from .loader import load_mat_file, parse_session_path
from .discovery import (
    discover_available_subjects,
    discover_subjects_from_cache_index,
    get_session_folders_for_split,
)
from .pipeline import (
    TrialInfo,
    preprocess_run_paper_aligned,
    preprocess_run_to_trials,
    trials_to_segments,
)
from .dataset import FingerEEGDataset, create_dataloaders
from .data_loader import PreprocessConfig

__all__ = [
    # Channel selection
    'create_biosemi128_to_1020_mapping',
    'get_channel_indices',
    'STANDARD_1020_CHANNELS',
    'BIOSEMI_128_LABELS',
    # Filtering
    'bandpass_filter',
    'notch_filter',
    # Resampling
    'resample_eeg',
    # MAT file loading
    'load_mat_file',
    'parse_session_path',
    # Discovery
    'discover_available_subjects',
    'discover_subjects_from_cache_index',
    'get_session_folders_for_split',
    # Pipeline
    'TrialInfo',
    'preprocess_run_paper_aligned',
    'preprocess_run_to_trials',
    'trials_to_segments',
    # Dataset
    'FingerEEGDataset',
    'create_dataloaders',
    # Configuration
    'PreprocessConfig',
]
