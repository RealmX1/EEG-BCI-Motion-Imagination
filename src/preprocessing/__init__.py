# Preprocessing module
"""
Data preprocessing utilities for EEG-BCI project.

Modules:
- channel_selection: BioSemi 128 to standard 10-20 channel mapping
- filtering: Bandpass and notch filtering
- resampling: Resampling to target frequency
- data_loader: PyTorch Dataset and DataLoader for FINGER-EEG-BCI
"""

from .channel_selection import (
    create_biosemi128_to_1020_mapping,
    get_channel_indices,
    STANDARD_1020_CHANNELS,
    BIOSEMI_128_LABELS,
)
from .filtering import bandpass_filter, notch_filter
from .resampling import resample_eeg
from .data_loader import FingerEEGDataset, create_dataloaders

__all__ = [
    'create_biosemi128_to_1020_mapping',
    'get_channel_indices',
    'STANDARD_1020_CHANNELS',
    'BIOSEMI_128_LABELS',
    'bandpass_filter',
    'notch_filter',
    'resample_eeg',
    'FingerEEGDataset',
    'create_dataloaders',
]
