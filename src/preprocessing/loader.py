"""
MAT file loading utilities for FINGER-EEG-BCI dataset.

This module provides functions for loading raw .mat files from the dataset.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.io as sio

from .channel_selection import BIOSEMI_128_LABELS
from ..utils.logging import SectionLogger

logger = logging.getLogger(__name__)
log_load = SectionLogger(logger, 'load')


def load_mat_file(mat_path: str) -> Tuple[np.ndarray, List[Dict], Dict]:
    """
    Load a single .mat file from FINGER-EEG-BCI dataset.

    Args:
        mat_path: Path to .mat file

    Returns:
        Tuple of (eeg_data, events, metadata)
        - eeg_data: [128 x time_samples] array
        - events: List of event dicts with 'type', 'sample', 'value'
        - metadata: Dict with 'fsample', 'labels', etc.
    """
    mat = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)

    eeg_struct = mat['eeg']
    event_struct = mat['event']

    # Extract EEG data
    eeg_data = eeg_struct.data  # [128 x time]

    # Extract metadata
    metadata = {
        'fsample': int(eeg_struct.fsample),
        'nChans': int(eeg_struct.nChans),
        'nSamples': int(eeg_struct.nSamples),
        'labels': list(eeg_struct.label) if hasattr(eeg_struct, 'label') else BIOSEMI_128_LABELS,
        'time': eeg_struct.time if hasattr(eeg_struct, 'time') else None,
    }

    # Check for online session predictions
    if hasattr(eeg_struct, 'prediction'):
        metadata['prediction'] = eeg_struct.prediction
    if hasattr(eeg_struct, 'prob_thumb'):
        metadata['prob_thumb'] = eeg_struct.prob_thumb
    if hasattr(eeg_struct, 'prob_index'):
        metadata['prob_index'] = eeg_struct.prob_index
    if hasattr(eeg_struct, 'prob_pinky'):
        metadata['prob_pinky'] = eeg_struct.prob_pinky

    # Extract events
    events = []
    if hasattr(event_struct, '__iter__'):
        for evt in event_struct:
            events.append({
                'type': str(evt.type),
                'sample': int(evt.sample),
                'value': int(evt.value) if hasattr(evt, 'value') else None
            })
    else:
        # Single event
        events.append({
            'type': str(event_struct.type),
            'sample': int(event_struct.sample),
            'value': int(event_struct.value) if hasattr(event_struct, 'value') else None
        })

    return eeg_data, events, metadata


def parse_session_path(path: Path) -> Dict:
    """
    Parse session information from file path.

    Example paths:
    - S01/OfflineMovement/S01_OfflineMovement_R01.mat
    - S01/OnlineMovement_Sess01_2class_Base/S01_OnlineMovement_R01.mat

    Returns:
        Dict with 'subject', 'task_type', 'session', 'n_class', 'model', 'run'
    """
    info = {
        'subject': None,
        'task_type': None,  # 'OfflineMovement', 'OfflineImagery', 'OnlineMovement', etc.
        'session_folder': None,  # FULL folder name, e.g., 'OnlineImagery_Sess01_2class_Base'
        'session': None,
        'n_class': None,
        'model': None,
        'run': None,
        'is_offline': True,
        'is_imagery': False,
    }

    # Get filename
    filename = path.stem  # e.g., 'S01_OfflineMovement_R01'
    parts = filename.split('_')

    if len(parts) >= 3:
        info['subject'] = parts[0]  # S01
        info['task_type'] = parts[1]  # OfflineMovement
        info['run'] = int(parts[-1][1:])  # R01 -> 1

    # Parse task type
    task = info['task_type'] or ''
    info['is_offline'] = task.startswith('Offline')
    info['is_imagery'] = 'Imagery' in task

    # Parse parent folder for online session details
    parent = path.parent.name
    info['session_folder'] = parent  # CRITICAL FIX: Store full folder name for unique session identification
    if 'Sess' in parent:
        # Extract session number
        sess_match = re.search(r'Sess(\d+)', parent)
        if sess_match:
            info['session'] = int(sess_match.group(1))

        # Extract class count
        class_match = re.search(r'(\d)class', parent)
        if class_match:
            info['n_class'] = int(class_match.group(1))

        # Extract model type
        if 'Base' in parent:
            info['model'] = 'base'
        elif 'Finetune' in parent:
            info['model'] = 'finetune'
        elif 'Smooth' in parent:
            info['model'] = 'smooth'

    return info
