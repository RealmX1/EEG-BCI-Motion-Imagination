"""
Data loader for FINGER-EEG-BCI dataset.

Loads .mat files and prepares data for CBraMod and EEGNet training.

Dataset structure:
- 21 subjects (S01-S21)
- Motor Execution (ME) and Motor Imagery (MI)
- Offline and Online sessions
- Target fingers: Thumb (1), Index (2), Middle (3), Pinky (4)

Preprocessing pipeline (aligned with original paper):
1. Common Average Reference (CAR)
2. Extract trials based on events
3. Sliding window segmentation (1s window, 125ms step)
4. Downsample to target rate (100 Hz for EEGNet, 200 Hz for CBraMod)
5. Bandpass filter (4-40 Hz for EEGNet paper alignment)
6. Z-score normalization per segment (along time axis)
"""

import json
import numpy as np
import scipy.io as sio
import scipy.signal
import scipy.stats
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
import os
import warnings

# Suppress RuntimeWarning from runpy when using multiprocessing with -m flag
# This warning is harmless and occurs because worker processes inherit module state
warnings.filterwarnings(
    "ignore",
    message=".*found in sys.modules after import of package.*",
    category=RuntimeWarning,
    module="runpy"
)

from .channel_selection import (
    create_biosemi128_to_1020_mapping,
    get_channel_indices,
    STANDARD_1020_CHANNELS,
    BIOSEMI_128_LABELS,
)
from .filtering import preprocess_filter_chain, check_amplitude_range
from .resampling import resample_eeg, resample_events, create_patches
from .cache_manager import PreprocessingCache, get_cache
from ..utils.logging import SectionLogger

logger = logging.getLogger(__name__)

# Section-specific loggers
log_load = SectionLogger(logger, 'load')
log_prep = SectionLogger(logger, 'prep')
log_cache = SectionLogger(logger, 'cache')
log_chan = SectionLogger(logger, 'chan')


@dataclass
class TrialInfo:
    """Information about a single trial."""
    subject_id: str
    session_type: str
    run_id: int
    trial_idx: int
    target_class: int  # 1=Thumb, 2=Index, 3=Middle, 4=Pinky
    start_sample: int
    end_sample: int


@dataclass
class PreprocessConfig:
    """
    Configuration for preprocessing pipeline.

    Two presets available:
    - 'eegnet': Paper-aligned settings (4-40 Hz, 100 Hz, z-score per segment)
    - 'cbramod': CBraMod settings (0.3-75 Hz, 200 Hz, divide by 100)

    Use PreprocessConfig.paper_aligned() for exact paper replication.
    """
    # Target model
    target_model: str = 'eegnet'  # 'cbramod' or 'eegnet'

    # Sampling rate
    original_fs: int = 1024
    target_fs: int = 100  # EEGNet: 100 Hz (paper), CBraMod: 200 Hz

    # Filtering (applied AFTER downsampling, as per paper)
    bandpass_low: float = 4.0   # Paper: 4 Hz
    bandpass_high: float = 40.0  # Paper: 40 Hz
    notch_freq: Optional[float] = None  # Paper does not use notch filter
    filter_order: int = 4  # 4th order Butterworth

    # Channel selection
    channel_strategy: str = 'C'  # 'A': 10-20, 'B': motor cortex, 'C': all 128 (paper uses all)

    # Trial extraction
    trial_duration: float = 5.0  # seconds for offline (paper: 5s offline, 3s online)
    online_trial_duration: float = 3.0  # seconds for online sessions
    pre_onset: float = 0.0  # seconds before trial onset
    post_offset: float = 0.0  # seconds after trial offset

    # Sliding window segmentation (paper: 1s window, 128 samples @ 1024 Hz = 125ms step)
    segment_length: float = 1.0  # seconds (1s window)
    segment_step_samples: int = 128  # step in samples at original_fs (125ms @ 1024 Hz)
    use_sliding_window: bool = True  # Enable for paper-aligned preprocessing

    # Normalization (paper: z-score per segment along time axis)
    normalize_method: str = 'zscore_time'  # 'divide', 'zscore_channel', 'zscore_time', 'none'
    normalize_by: float = 100.0  # Divide by this (only for 'divide' method)
    reject_threshold: float = -1.0  # Negative = no rejection (paper doesn't reject)

    # Extra normalization (for ML engineering experiments)
    # Applied AFTER primary normalization (e.g., after divide by 100)
    # Options: None, 'zscore_time', 'zscore_channel', 'robust'
    extra_normalize: Optional[str] = None

    # Common Average Reference (paper applies CAR first)
    apply_car: bool = True

    # Patching (CBraMod only)
    patch_duration: float = 1.0  # seconds
    patch_overlap: float = 0.0

    # Filter padding (paper uses zero padding to avoid edge effects)
    filter_padding: int = 100  # samples to pad before filtering

    # Experiment tracking (for ML engineering experiments)
    # When set, caches are isolated per experiment in separate directories
    experiment_id: Optional[str] = None

    def get_experiment_cache_tag(self) -> Optional[str]:
        """Get cache tag for experiment isolation.

        Returns:
            Cache tag string (e.g., 'data_preproc_ml_eng/A1') or None
        """
        if self.experiment_id:
            return f"data_preproc_ml_eng/{self.experiment_id}"
        return None

    @classmethod
    def paper_aligned(cls, n_class: int = 2) -> 'PreprocessConfig':
        """
        Create configuration exactly matching the original paper.

        Paper settings:
        - 128 channels, 100 Hz
        - 4-40 Hz bandpass (4th order Butterworth)
        - Z-score normalization per segment
        - 1s sliding window, 125ms step
        - CAR applied first

        Args:
            n_class: Number of classes (2 or 3)

        Returns:
            PreprocessConfig with paper-aligned settings
        """
        return cls(
            target_model='eegnet',
            original_fs=1024,
            target_fs=100,
            bandpass_low=4.0,
            bandpass_high=40.0,
            notch_freq=None,
            filter_order=4,
            channel_strategy='C',  # All 128 channels
            trial_duration=5.0,  # Offline
            online_trial_duration=3.0,  # Online
            segment_length=1.0,
            segment_step_samples=128,  # 125ms @ 1024 Hz
            use_sliding_window=True,
            normalize_method='zscore_time',
            reject_threshold=-1.0,  # No rejection
            apply_car=True,
            filter_padding=100,
        )

    @classmethod
    def for_cbramod(cls, use_sliding_window: bool = True, full_channels: bool = True) -> 'PreprocessConfig':
        """
        Create configuration for CBraMod model.

        CBraMod settings:
        - 128 channels (full BioSemi, default) or 19 channels (10-20 system)
        - 200 Hz sampling rate
        - 0.3-75 Hz bandpass
        - Divide by 100 normalization

        Optimized based on ML engineering experiments (2026-01):
        - 500ms sliding step (D3 config): 3x faster training, +1% accuracy
        - Notch filter removed (A6 config): no impact on performance

        Args:
            use_sliding_window: If True, use sliding window (1s, 500ms step) for
                more training data. If False, use whole trials as patches.
                Default True for fair comparison with EEGNet.
            full_channels: If True (default), use all 128 channels. If False, use
                19 channels (10-20 system). CBraMod's ACPE handles any channel count.

        Returns:
            PreprocessConfig for CBraMod
        """
        channel_strategy = 'C' if full_channels else 'A'
        target_model = 'cbramod_128ch' if full_channels else 'cbramod'

        return cls(
            target_model=target_model,
            original_fs=1024,
            target_fs=200,
            bandpass_low=0.3,
            bandpass_high=75.0,
            notch_freq=None,  # A6: notch filter has no impact on CBraMod
            filter_order=4,
            channel_strategy=channel_strategy,
            segment_length=1.0,  # 1s segments (= 200 samples @ 200Hz)
            segment_step_samples=512,  # D3: 500ms step @ 1024 Hz (3x faster, +1% acc)
            use_sliding_window=use_sliding_window,
            normalize_method='divide',
            normalize_by=100.0,
            reject_threshold=100.0,
            apply_car=True,
            filter_padding=100,
        )

    @classmethod
    def for_cbramod_128ch(cls, use_sliding_window: bool = True) -> 'PreprocessConfig':
        """
        Create configuration for CBraMod with full 128 channels.

        This leverages CBraMod's ACPE (Asymmetric Conditional Positional Encoding)
        which can dynamically generate position encodings for any number of channels.
        The ACPE uses a (19, 7) convolution kernel with padding=(9, 3), allowing
        it to process inputs of arbitrary spatial dimensions.

        Advantages over 19-channel version:
        - Preserves full spatial resolution (128 vs 19 channels)
        - Better for fine-grained motor decoding (finger-level)
        - Fair comparison with EEGNet (both use 128 channels)

        Note: Requires more GPU memory due to larger attention maps (128x128 vs 19x19).
        Consider reducing batch_size if OOM occurs.

        Returns:
            PreprocessConfig for CBraMod with 128 channels
        """
        return cls.for_cbramod(use_sliding_window=use_sliding_window, full_channels=True)

    @classmethod
    def from_experiment(cls, exp_config: 'ExperimentPreprocessConfig') -> 'PreprocessConfig':
        """
        Create PreprocessConfig from an ExperimentPreprocessConfig.

        This is used for ML engineering experiments to convert experiment
        configurations into preprocessing configurations.

        Args:
            exp_config: ExperimentPreprocessConfig instance

        Returns:
            PreprocessConfig for the experiment
        """
        # Import here to avoid circular imports
        from .experiment_config import ExperimentPreprocessConfig

        return cls(
            target_model='cbramod_128ch',
            original_fs=exp_config.original_fs,
            target_fs=exp_config.target_fs,
            bandpass_low=exp_config.bandpass_low,
            bandpass_high=exp_config.bandpass_high,
            notch_freq=exp_config.notch_freq,
            filter_order=exp_config.filter_order,
            channel_strategy='C',  # All 128 channels
            trial_duration=5.0,  # Offline trials
            online_trial_duration=3.0,  # Online trials
            segment_length=exp_config.segment_length,
            segment_step_samples=exp_config.segment_step_samples,
            use_sliding_window=True,
            normalize_method='divide',
            normalize_by=exp_config.normalize_by,
            reject_threshold=exp_config.amplitude_threshold if exp_config.amplitude_threshold else -1.0,
            apply_car=True,
            filter_padding=100,
            extra_normalize=exp_config.extra_normalize,
            experiment_id=exp_config.experiment_id,  # For cache isolation
        )


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


def apply_common_average_reference(data: np.ndarray) -> np.ndarray:
    """
    Apply Common Average Reference (CAR) to EEG data.

    CAR subtracts the mean across all channels at each time point.
    This is applied as the first preprocessing step (as per paper).

    Args:
        data: EEG data array [channels x time]

    Returns:
        CAR-referenced data array [channels x time]
    """
    # Subtract mean across channels (axis=0) at each time point
    car_data = data - data.mean(axis=0, keepdims=True)
    return car_data


def segment_with_sliding_window(
    data: np.ndarray,
    labels: np.ndarray,
    segment_size: int,
    step_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Segment trials using sliding window (as per paper).

    Paper uses 1-second window with 128-sample step (125ms @ 1024 Hz).

    Args:
        data: Trial data [n_trials x channels x time]
        labels: Trial labels [n_trials]
        segment_size: Segment size in samples
        step_size: Step size in samples

    Returns:
        Tuple of (segmented_data, repeated_labels, trial_indices)
        - segmented_data: [n_segments x channels x segment_size]
        - repeated_labels: [n_segments]
        - trial_indices: [n_segments] original trial index for each segment
    """
    if segment_size <= 0 or step_size <= 0:
        raise ValueError("segment_size and step_size must be positive.")

    n_trials, n_channels, n_samples = data.shape
    segments = []

    # Calculate number of segments per trial
    for start in range(0, n_samples - segment_size + 1, step_size):
        end = start + segment_size
        segments.append(data[:, :, start:end])

    if not segments:
        raise ValueError(
            f"Cannot create segments: data has {n_samples} samples, "
            f"need at least {segment_size}"
        )

    # Concatenate all segments
    segmented_data = np.concatenate(segments, axis=0)

    # Repeat labels for each segment
    n_segments_per_trial = len(segments)
    repeated_labels = np.tile(labels, n_segments_per_trial)

    # Track original trial indices
    trial_indices = np.tile(np.arange(n_trials), n_segments_per_trial)

    # Remove segments containing NaN
    valid_mask = ~np.isnan(segmented_data).any(axis=(1, 2))
    segmented_data = segmented_data[valid_mask]
    repeated_labels = repeated_labels[valid_mask]
    trial_indices = trial_indices[valid_mask]

    return segmented_data, repeated_labels, trial_indices


def apply_bandpass_filter_paper(
    data: np.ndarray,
    fs: float,
    low_freq: float = 4.0,
    high_freq: float = 40.0,
    order: int = 4,
    padding: int = 100
) -> np.ndarray:
    """
    Apply bandpass filter as per paper implementation.

    Paper uses:
    - 4th order Butterworth
    - Zero-padding to avoid edge effects
    - scipy.signal.lfilter (causal filter, not filtfilt)

    Args:
        data: EEG data [n_segments x channels x time] or [channels x time]
        fs: Sampling frequency in Hz
        low_freq: Low cutoff frequency (default 4 Hz)
        high_freq: High cutoff frequency (default 40 Hz)
        order: Filter order (default 4)
        padding: Padding length in samples (default 100)

    Returns:
        Filtered data array
    """
    # Design filter
    b, a = scipy.signal.butter(order, [low_freq, high_freq], btype='bandpass', fs=fs)

    # Handle different input dimensions
    if data.ndim == 2:
        # [channels x time]
        padded = np.pad(data, ((0, 0), (padding, padding)), 'constant', constant_values=0)
        filtered = scipy.signal.lfilter(b, a, padded, axis=-1)
        return filtered[:, padding:-padding]
    elif data.ndim == 3:
        # [n_segments x channels x time]
        padded = np.pad(data, ((0, 0), (0, 0), (padding, padding)), 'constant', constant_values=0)
        filtered = scipy.signal.lfilter(b, a, padded, axis=-1)
        return filtered[:, :, padding:-padding]
    else:
        raise ValueError(f"Unexpected data dimensions: {data.ndim}")


def apply_zscore_per_segment(data: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Apply z-score normalization per segment along time axis.

    Paper uses scipy.stats.zscore with axis=2 (time dimension).

    Args:
        data: EEG data [n_segments x channels x time] or [channels x time]
        axis: Axis along which to compute z-score (default -1, time axis)

    Returns:
        Z-scored data array
    """
    return scipy.stats.zscore(data, axis=axis, nan_policy='omit')


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
        import re
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


def get_session_folders_for_split(
    paradigm: str,
    task: str,
    split: str,
) -> List[str]:
    """
    Get the list of session folder names for a given data split.

    This follows the paper's experimental protocol:
    - For binary/ternary tasks:
        - Training: Offline + Online Session 1 (Base + Finetune) + Online Session 2 Base
        - Test: Online Session 2 Finetune (held out completely)
    - For quaternary (4-finger) task:
        - Only Offline data contains 4-finger trials (no Online 4class folders exist)
        - Both train and test splits use Offline data
        - Temporal split is handled by the caller

    Args:
        paradigm: 'imagery' or 'movement'
        task: 'binary', 'ternary', or 'quaternary'
        split: 'train' or 'test'

    Returns:
        List of folder names to include
    """
    # Map paradigm to prefix
    paradigm_prefix = 'Imagery' if paradigm == 'imagery' else 'Movement'
    offline = f'Offline{paradigm_prefix}'

    # Special case: quaternary task only has Offline data
    # No Online 4class folders exist in the dataset
    if task == 'quaternary':
        # Both train and test use Offline data; temporal split is done by caller
        return [offline]

    # Map task to n_class for binary/ternary
    task_to_nclass = {
        'binary': '2class',
        'ternary': '3class',
    }
    n_class = task_to_nclass.get(task, '2class')

    # Build folder names
    online_prefix = f'Online{paradigm_prefix}'

    if split == 'train':
        # Training: Offline + Sess01 Base + Sess01 Finetune + Sess02 Base
        folders = [
            offline,
            f'{online_prefix}_Sess01_{n_class}_Base',
            f'{online_prefix}_Sess01_{n_class}_Finetune',
            f'{online_prefix}_Sess02_{n_class}_Base',
        ]
    elif split == 'test':
        # Test: Sess02 Finetune only
        folders = [
            f'{online_prefix}_Sess02_{n_class}_Finetune',
        ]
    else:
        raise ValueError(f"Unknown split: {split}. Expected 'train' or 'test'.")

    return folders


def discover_available_subjects(
    data_root: str,
    paradigm: str = 'imagery',
    task: str = 'binary',
) -> List[str]:
    """
    Discover subjects that have the required data for both training and testing.

    Args:
        data_root: Root directory containing subject folders
        paradigm: 'imagery' or 'movement'
        task: 'binary', 'ternary', or 'quaternary'

    Returns:
        List of subject IDs (e.g., ['S01', 'S02', ...])
    """
    data_path = Path(data_root)
    subjects = []

    # Get required folders for test split (most restrictive)
    test_folders = get_session_folders_for_split(paradigm, task, 'test')

    for item in sorted(data_path.iterdir()):
        if item.is_dir() and item.name.startswith('S') and item.name[1:].isdigit():
            # Check if subject has required data folders
            # For binary/ternary: Session 2 Finetune
            # For quaternary: Offline data (only source of 4-finger trials)
            has_required_data = all(
                (item / folder).exists() for folder in test_folders
            )
            if has_required_data:
                subjects.append(item.name)

    return subjects


def discover_subjects_from_cache_index(
    cache_index_path: str = ".cache_index.json",
    paradigm: str = 'imagery',
    task: str = 'binary',
) -> List[str]:
    """
    从缓存索引中发现可用的被试。

    此函数读取预处理缓存索引，提取所有符合指定范式和任务的被试 ID。
    适用于数据已预处理但原始数据文件不在本地的场景。

    Args:
        cache_index_path: 缓存索引文件路径（默认：.cache_index.json）
        paradigm: 'imagery' 或 'movement'
        task: 'binary', 'ternary', 或 'quaternary'

    Returns:
        被试 ID 列表（如 ['S01', 'S02', ...]），按字母顺序排序

    Note:
        - Offline 数据的 n_classes 字段为 null，包含所有 4 个手指的数据
        - Binary/Ternary/Quaternary 任务都接受 n_classes == null 的条目
    """
    # 验证 paradigm 参数
    if paradigm not in ['imagery', 'movement']:
        logger.error(f"Invalid paradigm: {paradigm}. Must be 'imagery' or 'movement'")
        return []

    cache_path = Path(cache_index_path)

    # 检查缓存索引是否存在
    if not cache_path.exists():
        logger.warning(f"Cache index not found at {cache_index_path}, returning empty subject list")
        return []

    try:
        # 读取缓存索引
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)

        entries = cache_data.get('entries', {})
        if not entries:
            logger.warning(f"Cache index at {cache_index_path} contains no entries")
            return []

        # 确定任务对应的 n_classes
        task_to_n_classes = {
            'binary': [2, None],      # 接受 2-class 和 offline (null)
            'ternary': [3, None],     # 接受 3-class 和 offline (null)
            'quaternary': [4, None],  # 接受 4-class 和 offline (null)
        }

        if task not in task_to_n_classes:
            logger.error(f"Invalid task: {task}. Must be 'binary', 'ternary', or 'quaternary'")
            return []

        valid_n_classes = task_to_n_classes[task]

        # 提取符合条件的被试
        subjects_set = set()
        for entry_data in entries.values():
            # 检查 paradigm 匹配
            if entry_data.get('subject_task_type') != paradigm:
                continue

            # 检查 n_classes 匹配
            entry_n_classes = entry_data.get('n_classes')
            if entry_n_classes not in valid_n_classes:
                continue

            # 提取被试 ID
            subject_id = entry_data.get('subject')
            if subject_id:
                subjects_set.add(subject_id)

        subjects = sorted(list(subjects_set))

        if not subjects:
            logger.warning(f"No subjects found in cache index for paradigm={paradigm}, task={task}")
        else:
            logger.debug(f"Found {len(subjects)} subjects in cache index: {subjects}")

        return subjects

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse cache index at {cache_index_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error reading cache index: {e}")
        return []


def extract_trials(
    eeg_data: np.ndarray,
    events: List[Dict],
    metadata: Dict,
    config: PreprocessConfig
) -> List[Tuple[np.ndarray, int, TrialInfo]]:
    """
    Extract individual trials from continuous EEG data.

    Args:
        eeg_data: [channels x time] array
        events: List of event dicts
        metadata: Dict with sampling info
        config: Preprocessing configuration

    Returns:
        List of (trial_data, label, trial_info) tuples
    """
    fs = metadata['fsample']
    trials = []

    # Find trial starts (Target events)
    trial_starts = [e for e in events if e['type'] == 'Target']
    trial_ends = [e for e in events if e['type'] == 'TrialEnd']

    for i, start_evt in enumerate(trial_starts):
        start_sample = start_evt['sample']
        target_class = start_evt['value']

        # Find corresponding end
        if i < len(trial_ends):
            end_sample = trial_ends[i]['sample']
        else:
            # Use fixed duration
            end_sample = start_sample + int(config.trial_duration * fs)

        # Calculate segment bounds
        pre_samples = int(config.pre_onset * fs)
        post_samples = int(config.post_offset * fs)

        seg_start = start_sample - pre_samples
        seg_end = end_sample + post_samples

        # Check bounds
        if seg_start < 0 or seg_end > eeg_data.shape[1]:
            log_prep.warning(f"Trial {i} out of bounds, skip")
            continue

        # Extract segment
        trial_data = eeg_data[:, seg_start:seg_end].copy()

        # Create trial info (will be filled in later with full path info)
        trial_info = TrialInfo(
            subject_id='',
            session_type='',
            run_id=0,
            trial_idx=i,
            target_class=target_class,
            start_sample=start_sample,
            end_sample=end_sample
        )

        trials.append((trial_data, target_class, trial_info))

    return trials


def preprocess_trial(
    trial_data: np.ndarray,
    config: PreprocessConfig,
    channel_indices: Optional[List[int]] = None
) -> Optional[np.ndarray]:
    """
    Apply full preprocessing pipeline to a single trial.

    For paper-aligned preprocessing (EEGNet), use preprocess_trial_paper_aligned().
    This function supports both old CBraMod pipeline and new paper-aligned pipeline.

    Pipeline (paper-aligned when config.target_model == 'eegnet'):
    1. Common Average Reference (CAR)
    2. Channel selection
    3. Resampling (before filtering, as per paper)
    4. Bandpass filtering (4-40 Hz)
    5. Z-score normalization per segment (along time axis)

    Pipeline (CBraMod):
    1. Channel selection
    2. Bandpass + notch filtering
    3. Resampling
    4. Amplitude check
    5. Normalization (divide by 100)

    Args:
        trial_data: [128 x time] array
        config: Preprocessing configuration
        channel_indices: Indices of channels to keep (None = use all)

    Returns:
        Preprocessed trial data or None if rejected
    """
    data = trial_data.copy()

    # Step 1: Common Average Reference (paper applies first)
    if config.apply_car:
        data = apply_common_average_reference(data)

    # Step 2: Channel selection
    if channel_indices is not None:
        data = data[channel_indices, :]

    if config.target_model == 'eegnet':
        # Paper-aligned pipeline: downsample THEN filter

        # Step 3: Resampling (before filtering)
        data = resample_eeg(data, config.original_fs, config.target_fs)

        # Step 4: Bandpass filtering (after downsampling)
        data = apply_bandpass_filter_paper(
            data,
            fs=config.target_fs,
            low_freq=config.bandpass_low,
            high_freq=config.bandpass_high,
            order=config.filter_order,
            padding=config.filter_padding
        )

        # Step 5: Z-score normalization (along time axis)
        if config.normalize_method == 'zscore_time':
            data = apply_zscore_per_segment(data, axis=-1)
        elif config.normalize_method == 'zscore_channel':
            data = apply_zscore_per_segment(data, axis=0)

    else:
        # CBraMod pipeline: filter THEN downsample

        # Step 3: Filtering (before downsampling)
        data = preprocess_filter_chain(
            data,
            fs=config.original_fs,
            bandpass_low=config.bandpass_low,
            bandpass_high=config.bandpass_high,
            notch_freq=config.notch_freq
        )

        # Step 4: Resampling
        data = resample_eeg(data, config.original_fs, config.target_fs)

        # Step 5: Amplitude check
        if config.reject_threshold > 0:
            is_valid, _ = check_amplitude_range(data, config.reject_threshold)
            if not is_valid:
                return None

        # Step 6: Normalization
        if config.normalize_method == 'divide':
            data = data / config.normalize_by
        elif config.normalize_method in ['zscore_time', 'zscore_channel', 'zscore']:
            axis = -1 if config.normalize_method == 'zscore_time' else 0
            data = apply_zscore_per_segment(data, axis=axis)

    return data


def preprocess_run_paper_aligned(
    eeg_data: np.ndarray,
    events: List[Dict],
    metadata: Dict,
    config: PreprocessConfig,
    target_classes: Optional[List[int]] = None,
    label_mapping: Optional[Dict[int, int]] = None,
    store_all_fingers: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess a complete run following the exact paper pipeline.

    This implements the full pipeline from the original paper code (Functions.py):
    1. Extract trials based on events
    2. Pad trials to max length (with NaN for missing)
    3. Apply CAR to each trial (IMPORTANT: trial-level, not run-level)
    4. Segment with sliding window
    5. Downsample segments
    6. Bandpass filter
    7. Z-score normalize

    Args:
        eeg_data: Raw continuous EEG [128 x time]
        events: List of event dicts
        metadata: Dict with 'fsample', etc.
        config: Preprocessing configuration
        target_classes: Classes to keep (e.g., [1, 4] for thumb/pinky)
        label_mapping: Map original labels to new labels (e.g., {1: 0, 4: 1})
        store_all_fingers: If True, keep all trials regardless of target_classes
                          and use original labels (1,2,3,4). Used for Offline
                          data caching where all 4 fingers are cached together.

    Returns:
        Tuple of (segments, labels, trial_indices)
        - segments: [n_segments x channels x samples]
        - labels: [n_segments] (original 1,2,3,4 if store_all_fingers, else mapped)
        - trial_indices: [n_segments] original trial index
    """
    fs = metadata['fsample']

    # Step 1: Extract trials
    trial_starts = [e for e in events if e['type'] == 'Target']
    trial_ends = [e for e in events if e['type'] == 'TrialEnd']

    max_samples = int(config.trial_duration * fs)
    trials = []
    labels = []

    for i, start_evt in enumerate(trial_starts):
        start_sample = start_evt['sample'] - 1  # Convert to 0-indexed (as per paper)
        target_class = start_evt['value']

        # Filter by target class (skip if store_all_fingers is True)
        if not store_all_fingers:
            if target_classes is not None and target_class not in target_classes:
                continue

        # Find end sample
        if i < len(trial_ends):
            end_sample = trial_ends[i]['sample'] - 1  # 0-indexed
        else:
            end_sample = start_sample + max_samples

        # Extract trial data
        trial_data = eeg_data[:, start_sample:end_sample]

        # Pad to max length (with NaN, as per paper)
        actual_len = trial_data.shape[1]
        if actual_len < max_samples:
            pad_width = ((0, 0), (0, max_samples - actual_len))
            trial_data = np.pad(trial_data, pad_width, 'constant', constant_values=np.nan)
        elif actual_len > max_samples:
            trial_data = trial_data[:, :max_samples]

        trials.append(trial_data)

        # Map label: store_all_fingers uses original labels (1,2,3,4)
        if store_all_fingers:
            labels.append(target_class)  # Keep original finger ID
        elif label_mapping is not None:
            labels.append(label_mapping[target_class])
        else:
            labels.append(target_class)

    if not trials:
        return np.array([]), np.array([]), np.array([])

    # Use float32 to reduce memory usage (half of float64)
    trials = np.array(trials, dtype=np.float32)  # [n_trials x channels x time]
    labels = np.array(labels)

    # Step 2: Apply CAR to each trial independently (as per paper)
    # Use nanmean to handle NaN-padded trials correctly
    if config.apply_car:
        trials = trials - np.nanmean(trials, axis=1, keepdims=True)

    # Step 3: Segment with sliding window
    segment_size = int(config.segment_length * fs)
    step_size = config.segment_step_samples

    segments, seg_labels, trial_indices = segment_with_sliding_window(
        trials, labels, segment_size, step_size
    )

    # Step 4: Downsample using resample_poly for better numerical stability
    # resample_poly uses rational resampling which avoids FFT artifacts
    from math import gcd
    common = gcd(config.original_fs, config.target_fs)
    up = config.target_fs // common
    down = config.original_fs // common
    segments = scipy.signal.resample_poly(segments, up, down, axis=2)
    segments = segments.astype(np.float32)  # Ensure float32 after resample

    # Step 5: Bandpass filter
    segments = apply_bandpass_filter_paper(
        segments,
        fs=config.target_fs,
        low_freq=config.bandpass_low,
        high_freq=config.bandpass_high,
        order=config.filter_order,
        padding=config.filter_padding
    )

    # Step 6: Z-score normalize (per segment, along time axis)
    segments = apply_zscore_per_segment(segments, axis=-1)

    return segments, seg_labels, trial_indices


def preprocess_run_to_trials(
    eeg_data: np.ndarray,
    events: List[Dict],
    metadata: Dict,
    config: PreprocessConfig,
    target_classes: Optional[List[int]] = None,
    store_all_fingers: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess a run to trial level for caching (no sliding window).

    This extracts trials, applies CAR, and downsamples to target_fs.
    Sliding window, filtering, and normalization are applied on cache load.

    Pipeline:
    1. Extract trials based on events
    2. Pad trials to max length (with NaN for missing)
    3. Apply CAR to each trial
    4. Downsample to target_fs

    Cache size reduction: ~6.6x compared to segment-level caching.

    Args:
        eeg_data: Raw continuous EEG [128 x time]
        events: List of event dicts
        metadata: Dict with 'fsample', etc.
        config: Preprocessing configuration
        target_classes: Classes to keep (e.g., [1, 4] for thumb/pinky)
        store_all_fingers: If True, keep all trials with original labels (1,2,3,4)

    Returns:
        Tuple of (trials, labels)
        - trials: [n_trials x channels x target_samples] at target_fs
        - labels: [n_trials] (original 1,2,3,4 if store_all_fingers)
    """
    from math import gcd

    fs = metadata['fsample']

    # Step 1: Extract trials
    trial_starts = [e for e in events if e['type'] == 'Target']
    trial_ends = [e for e in events if e['type'] == 'TrialEnd']

    max_samples = int(config.trial_duration * fs)
    trials = []
    labels = []

    for i, start_evt in enumerate(trial_starts):
        start_sample = start_evt['sample'] - 1  # Convert to 0-indexed
        target_class = start_evt['value']

        # Filter by target class (skip if store_all_fingers is True)
        if not store_all_fingers:
            if target_classes is not None and target_class not in target_classes:
                continue

        # Find end sample
        if i < len(trial_ends):
            end_sample = trial_ends[i]['sample'] - 1
        else:
            end_sample = start_sample + max_samples

        # Extract trial data
        trial_data = eeg_data[:, start_sample:end_sample]

        # Pad to max length (with NaN, as per paper)
        actual_len = trial_data.shape[1]
        if actual_len < max_samples:
            pad_width = ((0, 0), (0, max_samples - actual_len))
            trial_data = np.pad(trial_data, pad_width, 'constant', constant_values=np.nan)
        elif actual_len > max_samples:
            trial_data = trial_data[:, :max_samples]

        trials.append(trial_data)
        labels.append(target_class)

    if not trials:
        return np.array([]), np.array([])

    trials = np.array(trials, dtype=np.float32)
    labels = np.array(labels)

    # Step 2: Apply CAR (using nanmean to handle NaN-padded trials)
    # Suppress "Mean of empty slice" warning - expected for all-NaN trials
    if config.apply_car:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Mean of empty slice', RuntimeWarning)
            trials = trials - np.nanmean(trials, axis=1, keepdims=True)

    # Step 3: Downsample to target_fs using resample_poly
    common = gcd(config.original_fs, config.target_fs)
    up = config.target_fs // common
    down = config.original_fs // common
    trials = scipy.signal.resample_poly(trials, up, down, axis=2)
    trials = trials.astype(np.float32)

    return trials, labels


def apply_robust_normalize(data: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Apply robust normalization (x - median) / IQR.

    More robust to outliers than z-score normalization.

    Args:
        data: EEG data array
        axis: Axis along which to compute statistics

    Returns:
        Robustly normalized data
    """
    median = np.median(data, axis=axis, keepdims=True)
    q75 = np.percentile(data, 75, axis=axis, keepdims=True)
    q25 = np.percentile(data, 25, axis=axis, keepdims=True)
    iqr = q75 - q25

    # Avoid division by zero
    iqr = np.where(iqr == 0, 1.0, iqr)

    return (data - median) / iqr


def trials_to_segments(
    trials: np.ndarray,
    labels: np.ndarray,
    config: PreprocessConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert trials to segments (apply sliding window, filter, normalize).

    Called after loading trials from cache. This applies the remaining
    preprocessing steps that were deferred for cache efficiency.

    Pipeline:
    1. Sliding window segmentation (at target_fs)
    2. Bandpass filter
    3. Primary normalization (z-score or divide)
    4. Extra normalization (optional, for ML engineering experiments)

    Args:
        trials: [n_trials x channels x trial_samples] at target_fs
        labels: [n_trials]
        config: Preprocessing configuration

    Returns:
        Tuple of (segments, seg_labels, trial_indices)
        - segments: [n_segments x channels x segment_samples]
        - seg_labels: [n_segments]
        - trial_indices: [n_segments] original trial index for each segment
    """
    # Step 1: Sliding window at target_fs
    # segment_size in samples at target_fs
    segment_size = int(config.segment_length * config.target_fs)

    # step_size: convert from original_fs to target_fs
    # Use ceiling to ensure we don't miss segments
    step_size = int(np.ceil(
        config.segment_step_samples * config.target_fs / config.original_fs
    ))

    segments, seg_labels, trial_indices = segment_with_sliding_window(
        trials, labels, segment_size, step_size
    )

    if len(segments) == 0:
        return np.array([]), np.array([]), np.array([])

    # Step 2: Bandpass filter
    segments = apply_bandpass_filter_paper(
        segments,
        fs=config.target_fs,
        low_freq=config.bandpass_low,
        high_freq=config.bandpass_high,
        order=config.filter_order,
        padding=config.filter_padding
    )

    # Step 3: Primary normalization
    if config.normalize_method == 'divide':
        # CBraMod: divide by normalize_by (default 100)
        segments = segments / config.normalize_by
    elif config.normalize_method == 'zscore_time':
        # EEGNet: z-score along time axis
        segments = apply_zscore_per_segment(segments, axis=-1)
    elif config.normalize_method == 'zscore_channel':
        segments = apply_zscore_per_segment(segments, axis=1)
    # 'none' = no normalization

    # Step 4: Extra normalization (for ML engineering experiments)
    # Applied after primary normalization
    if config.extra_normalize:
        if config.extra_normalize == 'zscore_time':
            segments = apply_zscore_per_segment(segments, axis=-1)
        elif config.extra_normalize == 'zscore_channel':
            segments = apply_zscore_per_segment(segments, axis=1)
        elif config.extra_normalize == 'robust':
            segments = apply_robust_normalize(segments, axis=-1)

    return segments, seg_labels, trial_indices


def _process_single_mat_file(
    mat_path: str,
    config: 'PreprocessConfig',
    target_classes: Optional[List[int]],
    channel_indices: Optional[List[int]],
    store_all_fingers: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Dict, str]:
    """
    Process a single MAT file for parallel execution.

    This is a module-level function (not a method) so it can be pickled
    for use with ProcessPoolExecutor.

    Args:
        mat_path: Path to the MAT file
        config: Preprocessing configuration
        target_classes: Target classes to include
        channel_indices: Channel indices to select (for CBraMod)
        store_all_fingers: If True, keep all trials and original labels (1,2,3,4).
                          Used for Offline data caching.

    Returns:
        Tuple of (segments, labels, trial_indices, session_info, mat_path)
        Returns (None, None, None, session_info, mat_path) on error
    """
    try:
        mat_path = Path(mat_path)
        session_info = parse_session_path(mat_path)

        # Load MAT file
        eeg_data, events, metadata = load_mat_file(str(mat_path))

        # Prepare label mapping (not used if store_all_fingers)
        label_mapping = None
        if not store_all_fingers and target_classes is not None:
            label_mapping = {cls: i for i, cls in enumerate(sorted(target_classes))}

        # Apply paper-aligned preprocessing
        segments, seg_labels, trial_indices = preprocess_run_paper_aligned(
            eeg_data,
            events,
            metadata,
            config,
            target_classes=target_classes if not store_all_fingers else None,
            label_mapping=label_mapping,
            store_all_fingers=store_all_fingers
        )

        if len(segments) == 0:
            return None, None, None, session_info, str(mat_path)

        # Apply channel selection if needed
        if channel_indices is not None:
            segments = segments[:, channel_indices, :]

        return segments, seg_labels, trial_indices, session_info, str(mat_path)

    except Exception as e:
        log_prep.error(f"Error processing {mat_path}: {e}")
        session_info = parse_session_path(Path(mat_path))
        return None, None, None, session_info, str(mat_path)


def _process_single_mat_file_to_trials(
    mat_path: str,
    config: 'PreprocessConfig',
    target_classes: Optional[List[int]],
    channel_indices: Optional[List[int]],
    store_all_fingers: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict, str]:
    """
    Process a single MAT file to trial level for caching.

    This is the trial-level version of _process_single_mat_file.
    Used for v3.0 caching which stores trials instead of segments.

    Args:
        mat_path: Path to the MAT file
        config: Preprocessing configuration
        target_classes: Target classes to include
        channel_indices: Channel indices to select
        store_all_fingers: If True, keep all trials with original labels

    Returns:
        Tuple of (trials, labels, session_info, mat_path)
        Returns (None, None, session_info, mat_path) on error
    """
    try:
        mat_path = Path(mat_path)
        session_info = parse_session_path(mat_path)

        # Load MAT file
        eeg_data, events, metadata = load_mat_file(str(mat_path))

        # Preprocess to trial level (no sliding window)
        trials, labels = preprocess_run_to_trials(
            eeg_data,
            events,
            metadata,
            config,
            target_classes=target_classes if not store_all_fingers else None,
            store_all_fingers=store_all_fingers
        )

        if len(trials) == 0:
            return None, None, session_info, str(mat_path)

        # Apply channel selection if needed
        if channel_indices is not None:
            trials = trials[:, channel_indices, :]

        return trials, labels, session_info, str(mat_path)

    except Exception as e:
        log_prep.error(f"Error processing {mat_path} to trials: {e}")
        session_info = parse_session_path(Path(mat_path))
        return None, None, session_info, str(mat_path)


class FingerEEGDataset(Dataset):
    """
    PyTorch Dataset for FINGER-EEG-BCI data.

    Supports both CBraMod and EEGNet input formats.
    Includes optional caching for preprocessed data.
    """

    def __init__(
        self,
        data_root: str,
        subjects: List[str],
        config: PreprocessConfig,
        task_types: Optional[List[str]] = None,
        target_classes: Optional[List[int]] = None,
        elc_path: Optional[str] = None,
        transform=None,
        use_cache: bool = True,
        cache_dir: str = "caches/preprocessed",
        session_folders: Optional[List[str]] = None,
        preconvert_tensors: bool = True,
        parallel_workers: int = 0,
        cache_only: bool = False,
        cache_index_path: str = ".cache_index.json",
    ):
        """
        Initialize dataset.

        Args:
            data_root: Root directory containing subject folders
            subjects: List of subject IDs (e.g., ['S01', 'S02'])
            config: Preprocessing configuration
            task_types: List of task types to include (e.g., ['OfflineImagery'])
                       None = include all. Deprecated: use session_folders instead.
            target_classes: List of target classes to include (e.g., [1, 4] for thumb/pinky)
                           None = include all
            elc_path: Path to biosemi128.ELC file (required for channel mapping)
            transform: Optional transform to apply to data
            use_cache: Whether to use preprocessing cache (default: True)
            cache_dir: Directory for cache files (default: 'caches/preprocessed')
            session_folders: List of exact folder names to include (e.g.,
                           ['OfflineImagery', 'OnlineImagery_Sess01_2class_Base']).
                           This takes precedence over task_types if specified.
            preconvert_tensors: If True, convert numpy arrays to tensors at load time
                               for faster __getitem__. Uses more memory but ~5-10% faster.
                               (default: True)
            parallel_workers: Number of parallel workers for loading/preprocessing.
                            0 = auto (use cpu_count - 1), -1 = disabled (serial).
                            Parallel loading can speed up first-time preprocessing by 3-4x.
                            (default: 0)
            cache_only: If True, load data exclusively from cache index without
                       scanning filesystem. Useful when original .mat files are not
                       available but preprocessed caches exist. (default: False)
            cache_index_path: Path to cache index file for cache_only mode.
                            (default: '.cache_index.json')
        """
        self.data_root = Path(data_root)
        self.subjects = subjects
        self.config = config
        self.task_types = task_types
        self.target_classes = target_classes
        self.transform = transform
        self.session_folders = session_folders
        self.preconvert_tensors = preconvert_tensors
        self.cache_only = cache_only
        self.cache_index_path = cache_index_path

        # Validate cache_only mode
        if cache_only and not use_cache:
            raise ValueError("cache_only=True requires use_cache=True")

        # Set up parallel workers
        if parallel_workers == 0:
            # Auto: use cpu_count - 1
            # Each worker needs ~400MB for preprocessing, so 8 workers ≈ 3.2GB
            self.parallel_workers = max(1, (os.cpu_count() or 1) - 1)
        elif parallel_workers < 0:
            # Disabled
            self.parallel_workers = 1
        else:
            self.parallel_workers = parallel_workers

        # Initialize cache
        self.cache: Optional[PreprocessingCache] = None
        if use_cache:
            self.cache = get_cache(cache_dir=cache_dir, enabled=True)

        # Set up channel mapping
        self.channel_indices = None
        if config.channel_strategy == 'A' and elc_path:
            mapping = create_biosemi128_to_1020_mapping(elc_path)
            idx_map = get_channel_indices(mapping)
            self.channel_indices = [idx_map[ch] for ch in STANDARD_1020_CHANNELS]
        elif config.channel_strategy == 'C':
            self.channel_indices = None  # Use all 128 channels

        # Load all trials
        self.trials = []
        self.labels = []
        self.trial_infos = []

        # Global trial counter (to ensure unique trial indices across all runs)
        self._global_trial_counter = 0

        self._load_data()

        # Map labels to continuous indices
        self._setup_label_mapping()

        # Pre-convert numpy arrays to tensors for faster __getitem__
        if self.preconvert_tensors and self.trials:
            self._convert_to_tensors()

    def _build_file_list_from_cache_index(self) -> List[Tuple[Path, Dict, bool, bool]]:
        """
        构建文件列表从缓存索引（纯缓存模式）。

        返回与文件系统扫描相同格式的文件列表，但完全基于缓存索引。
        这允许在原始 .mat 文件不可用时仍能加载数据。

        Returns:
            List of (mat_path, session_info, needs_processing, is_offline) tuples
            其中 mat_path 是虚拟路径，needs_processing 始终为 False
        """
        cache_path = Path(self.cache_index_path)

        if not cache_path.exists():
            logger.error(f"Cache index not found at {self.cache_index_path} (cache_only mode)")
            return []

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse cache index: {e}")
            return []

        entries = cache_data.get('entries', {})
        if not entries:
            logger.warning(f"Cache index contains no entries")
            return []

        # 收集符合条件的缓存条目
        # 使用 (subject, run, session_folder) 作为唯一键去重
        unique_files = {}  # key: (subject, run, session_folder), value: entry_data

        for cache_key, entry_data in entries.items():
            subject = entry_data.get('subject')
            run = entry_data.get('run')
            session_folder = entry_data.get('session_folder')
            model = entry_data.get('model')

            # 过滤：被试
            if subject not in self.subjects:
                continue

            # 过滤：模型匹配
            if model != self.config.target_model:
                continue

            # 过滤：session_folders
            if self.session_folders is not None:
                if session_folder not in self.session_folders:
                    continue
            elif self.task_types is not None:
                # Deprecated: task_types 过滤
                task_type = entry_data.get('subject_task_type')
                if task_type not in self.task_types:
                    continue

            # 过滤：target_classes（仅对 online 数据）
            is_offline = self._is_offline_session(session_folder)
            entry_n_classes = entry_data.get('n_classes')

            if not is_offline and self.target_classes is not None:
                # Online 数据：检查 n_classes 是否匹配
                expected_n_classes = len(self.target_classes)
                if entry_n_classes != expected_n_classes:
                    continue

            # 构建唯一键
            file_key = (subject, run, session_folder)

            # 去重：优先保留当前模型的缓存
            if file_key not in unique_files:
                unique_files[file_key] = entry_data

        # 构建文件列表
        files_to_process = []

        for (subject, run, session_folder), entry_data in sorted(unique_files.items()):
            # 构建虚拟 mat_path（用于缓存键生成，但不访问文件系统）
            # 格式: data_root/subject/session_folder/subject_session_folder_R{run:02d}.mat
            virtual_filename = f"{subject}_{session_folder}_R{run:02d}.mat"
            virtual_mat_path = self.data_root / subject / session_folder / virtual_filename

            # 构建 session_info（与 parse_session_path 相同格式）
            session_info = {
                'subject': subject,
                'run': run,
                'task_type': entry_data.get('subject_task_type'),
                'session_folder': session_folder,
            }

            is_offline = self._is_offline_session(session_folder)
            needs_processing = False  # 纯缓存模式下，所有文件都应该已缓存

            files_to_process.append((virtual_mat_path, session_info, needs_processing, is_offline))

        logger.info(f"Cache-only mode: Found {len(files_to_process)} cached files for {len(self.subjects)} subjects")

        return files_to_process

    def _load_data(self):
        """
        Load all data from disk.

        For paper-aligned preprocessing (use_sliding_window=True), this applies
        preprocessing at the Run level (entire .mat file) to match paper methodology.

        For trial-based preprocessing, this processes individual trials.

        Supports parallel loading when parallel_workers > 1, which can speed up
        first-time preprocessing by 3-4x.
        """
        import time
        from src.utils.timing import colored, Colors, format_time

        total_start = time.perf_counter()
        n_cache_hits = 0
        n_cache_misses = 0

        # Phase 1: Collect all files and check cache status
        files_to_process = []  # (mat_path, session_info, needs_processing, is_offline)

        if self.cache_only:
            # 纯缓存模式：从缓存索引构建文件列表
            files_to_process = self._build_file_list_from_cache_index()

            if not files_to_process:
                log_load.error(f"Cache-only mode: No cached files found for subjects {self.subjects}")
                return
        else:
            # 传统模式：扫描文件系统
            for subject in self.subjects:
                subject_dir = self.data_root / subject

                if not subject_dir.exists():
                    log_load.warning(f"Subject dir not found: {subject_dir}")
                    continue

                mat_files = sorted(subject_dir.rglob('*.mat'))  # Sort for reproducibility

                for mat_path in mat_files:
                    session_info = parse_session_path(mat_path)
                    parent_folder = mat_path.parent.name

                    # Filter by session folders (takes precedence over task_types)
                    if self.session_folders is not None:
                        if parent_folder not in self.session_folders:
                            continue
                    elif self.task_types is not None:
                        if session_info['task_type'] not in self.task_types:
                            continue

                    # Skip known bad data
                    if 'S07' in str(mat_path) and 'OnlineImagery_Sess05_3class_Base' in str(mat_path):
                        log_load.info(f"Skip bad data: {mat_path.name}")
                        continue

                    # Check cache
                    # Offline data: cache with target_classes=None (all 4 fingers)
                    # Online data: cache with actual target_classes
                    is_offline = self._is_offline_session(parent_folder)
                    cache_target_classes = None if is_offline else self.target_classes

                    needs_processing = True
                    if self.cache is not None and self.config.use_sliding_window:
                        has_cache = self.cache.has_valid_cache(
                            session_info['subject'],
                            session_info['run'],
                            parent_folder,  # Use folder name, not task_type
                            self.config,
                            str(mat_path),
                            cache_target_classes,  # None for Offline, actual for Online
                            experiment_tag=self.config.get_experiment_cache_tag(),
                        )
                        if has_cache:
                            needs_processing = False

                    files_to_process.append((mat_path, session_info, needs_processing, is_offline))

        # Phase 2: Load from cache (fast, serial)
        # v3.0: Cache stores trials, not segments. Apply sliding window on load.
        cached_files = [(p, s, offline) for p, s, needs, offline in files_to_process if not needs]
        for mat_path, session_info, is_offline in cached_files:
            try:
                parent_folder = mat_path.parent.name
                cache_target_classes = None if is_offline else self.target_classes

                # v3.0: Load trials + labels (not segments)
                trials, labels = self.cache.load(
                    session_info['subject'],
                    session_info['run'],
                    parent_folder,  # Use folder name
                    self.config,
                    cache_target_classes,  # None for Offline
                    experiment_tag=self.config.get_experiment_cache_tag(),
                )

                # Offline data: filter to target_classes before sliding window
                # Online data: just map labels (already filtered during extraction)
                if self.target_classes is not None:
                    if is_offline:
                        # Offline: filter + map labels
                        trials, labels = self._filter_trials_by_classes(
                            trials, labels, self.target_classes
                        )
                    else:
                        # Online: just map labels (already filtered)
                        labels = self._map_labels_to_indices(labels, self.target_classes)

                # v3.0: Apply sliding window, filter, normalize on load
                segments, seg_labels, trial_indices = trials_to_segments(
                    trials, labels, self.config
                )

                # Apply channel selection if needed (after trials_to_segments)
                if self.channel_indices is not None:
                    segments = segments[:, self.channel_indices, :]

                self._store_segments(segments, seg_labels, trial_indices, session_info)
                n_cache_hits += 1
            except Exception as e:
                log_cache.error(f"Cache load failed: {mat_path.name}: {e}")

        # Phase 3: Process uncached files (potentially parallel)
        uncached_files = [(p, s, offline) for p, s, needs, offline in files_to_process if needs]

        if uncached_files:
            if self.cache_only:
                # 纯缓存模式：不应该有未缓存的文件
                log_load.error(
                    f"Cache-only mode: {len(uncached_files)} files have no cache. "
                    f"Cannot process without original .mat files."
                )
                for mat_path, session_info, is_offline in uncached_files:
                    log_load.error(f"  Missing cache: {session_info['subject']} {session_info['session_folder']} R{session_info['run']:02d}")
            else:
                # 传统模式：处理未缓存的文件
                if self.config.use_sliding_window:
                    # Paper-aligned preprocessing with parallel support
                    self._load_uncached_parallel(uncached_files)
                else:
                    # Trial-based preprocessing (serial, less common)
                    for mat_path, session_info, is_offline in uncached_files:
                        try:
                            eeg_data, events, metadata = load_mat_file(str(mat_path))
                            self._load_run_trial_based(
                                eeg_data, events, metadata, session_info, mat_path
                            )
                        except Exception as e:
                            log_load.error(f"Load failed: {mat_path.name}: {e}")

                n_cache_misses = len(uncached_files)

        total_time = time.perf_counter() - total_start

        # Store cache stats as instance attributes for external access
        self.n_cache_hits = n_cache_hits
        self.n_cache_misses = n_cache_misses

        # Log summary
        log_load.debug(f"Load time: {format_time(total_time)} ({n_cache_hits} hits, {n_cache_misses} miss, {self.parallel_workers}w)")
        log_load.info(f"Loaded {len(self.trials)} segs (cache: {'hit' if n_cache_misses == 0 else 'partial'})")

    def _load_uncached_parallel(self, uncached_files: List[Tuple[Path, Dict, bool]]):
        """
        Load and preprocess uncached files using parallel workers.

        v3.0: Uses trial-level caching. Process flow:
        1. Parallel: Extract trials, apply CAR, downsample (preprocess_run_to_trials)
        2. Serial: Store segments after applying sliding window
        3. Parallel: Save trials to cache

        Results are merged in the original file order to ensure reproducibility.

        Args:
            uncached_files: List of (mat_path, session_info, is_offline) tuples
        """
        import time

        n_files = len(uncached_files)
        use_parallel = self.parallel_workers > 1 and n_files > 1

        if use_parallel:
            log_prep.info(f"Parallel preproc: {n_files} files, {self.parallel_workers}w")
            start_time = time.perf_counter()

            # Prepare arguments for parallel execution
            # Sort by path for reproducible ordering
            sorted_files = sorted(uncached_files, key=lambda x: str(x[0]))
            mat_paths = [str(p) for p, _, _ in sorted_files]

            # Build mapping of mat_path -> is_offline
            path_to_offline = {str(p): offline for p, _, offline in sorted_files}

            # Use ProcessPoolExecutor for CPU-bound preprocessing
            # v3.0: Use _process_single_mat_file_to_trials instead of _process_single_mat_file
            results = {}

            with ProcessPoolExecutor(max_workers=self.parallel_workers) as executor:
                # Submit all tasks
                # For offline data: store_all_fingers=True, process all 4 fingers
                # For online data: use actual target_classes
                future_to_path = {
                    executor.submit(
                        _process_single_mat_file_to_trials,
                        path,
                        self.config,
                        self.target_classes,
                        None,  # channel_indices applied later (after cache load)
                        path_to_offline[path]  # store_all_fingers for offline
                    ): path for path in mat_paths
                }

                # Collect results as they complete
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        result = future.result()
                        results[path] = result
                    except MemoryError as e:
                        # OOM errors should propagate - don't silently continue
                        log_prep.critical(f"Out of memory during parallel processing: {Path(path).name}")
                        raise
                    except Exception as e:
                        log_prep.error(f"Parallel failed: {Path(path).name}: {e}")
                        results[path] = (None, None, parse_session_path(Path(path)), path)

            elapsed_preprocess = time.perf_counter() - start_time
            log_prep.info(f"Parallel done: {elapsed_preprocess:.1f}s ({n_files / elapsed_preprocess:.1f} f/s)")

            # Store results in original sorted order (for reproducibility)
            # v3.0: Apply trials_to_segments() before storing
            for mat_path in mat_paths:
                trials, labels, session_info, path = results[mat_path]
                if trials is not None:
                    is_offline = path_to_offline[mat_path]

                    # Offline data: filter to target_classes before sliding window
                    # Online data: just map labels (already filtered during extraction)
                    trials_for_segments = trials
                    labels_for_segments = labels
                    if self.target_classes is not None:
                        if is_offline:
                            # Offline: filter + map labels
                            trials_for_segments, labels_for_segments = self._filter_trials_by_classes(
                                trials, labels, self.target_classes
                            )
                        else:
                            # Online: just map labels (already filtered)
                            labels_for_segments = self._map_labels_to_indices(labels, self.target_classes)

                    # v3.0: Apply sliding window, filter, normalize
                    segments, seg_labels, trial_indices = trials_to_segments(
                        trials_for_segments, labels_for_segments, self.config
                    )

                    # Apply channel selection if needed
                    if self.channel_indices is not None:
                        segments = segments[:, self.channel_indices, :]

                    self._store_segments(segments, seg_labels, trial_indices, session_info)

            # Save to cache in parallel (I/O + compression bound)
            # v3.0: Cache stores TRIALS (not segments), UNFILTERED for offline
            if self.cache is not None:
                cache_start = time.perf_counter()
                cache_tasks = [
                    (results[p], p, path_to_offline[p]) for p in mat_paths
                    if results[p][0] is not None
                ]

                def save_to_cache(args):
                    (trials, labels, session_info, path), mat_path, is_offline = args
                    parent_folder = Path(path).parent.name
                    # Offline: cache all fingers with target_classes=None
                    # Online: cache with actual target_classes
                    cache_target_classes = None if is_offline else self.target_classes
                    self.cache.save(
                        session_info['subject'],
                        session_info['run'],
                        parent_folder,  # Use folder name
                        self.config,
                        trials, labels,  # v3.0: trials, not segments
                        path, cache_target_classes,
                        experiment_tag=self.config.get_experiment_cache_tag(),
                    )

                # Use ThreadPoolExecutor for I/O-bound cache saving
                with ThreadPoolExecutor(max_workers=min(self.parallel_workers, 8)) as executor:
                    list(executor.map(save_to_cache, cache_tasks))

                cache_elapsed = time.perf_counter() - cache_start
                log_cache.debug(f"Cache save: {cache_elapsed:.1f}s")

        else:
            # Serial fallback
            for mat_path, session_info, is_offline in uncached_files:
                try:
                    eeg_data, events, metadata = load_mat_file(str(mat_path))
                    self._load_run_paper_aligned(
                        eeg_data, events, metadata, session_info, mat_path, is_offline
                    )
                except Exception as e:
                    log_load.error(f"Load failed: {mat_path.name}: {e}")

    def _load_run_paper_aligned(
        self,
        eeg_data: np.ndarray,
        events: List[Dict],
        metadata: Dict,
        session_info: Dict,
        mat_path: Path,
        is_offline: bool = False
    ):
        """
        Load a run using paper-aligned preprocessing.

        v3.0: Uses trial-level caching. Process flow:
        1. Extract trials, apply CAR, downsample (preprocess_run_to_trials)
        2. Save trials to cache (smaller than segments)
        3. Apply sliding window, filter, normalize (trials_to_segments)
        4. Store segments in dataset

        Note: Cache loading is handled by _load_data() for timing purposes.
        This method is only called on cache miss.

        Args:
            is_offline: If True, this is Offline data - cache all 4 fingers,
                       filter to target_classes after loading.
        """
        subject = session_info['subject']
        run_id = session_info['run']
        parent_folder = mat_path.parent.name

        # Offline data: store all fingers in cache, filter later
        # Online data: store only target_classes
        store_all_fingers = is_offline

        # v3.0: Preprocess to trial level (no sliding window yet)
        trials, labels = preprocess_run_to_trials(
            eeg_data,
            events,
            metadata,
            self.config,
            target_classes=self.target_classes if not store_all_fingers else None,
            store_all_fingers=store_all_fingers
        )

        if len(trials) == 0:
            return

        # Save to cache (before applying sliding window)
        # Offline: save all fingers with target_classes=None
        # Online: save with actual target_classes
        if self.cache is not None:
            cache_target_classes = None if is_offline else self.target_classes
            self.cache.save(
                subject, run_id, parent_folder, self.config,
                trials, labels,  # v3.0: trials, not segments
                str(mat_path), cache_target_classes,
                experiment_tag=self.config.get_experiment_cache_tag(),
            )

        # Offline data: filter to target_classes before sliding window
        # Online data: just map labels (already filtered during extraction)
        trials_for_segments = trials
        labels_for_segments = labels
        if self.target_classes is not None:
            if is_offline:
                # Offline: filter + map labels
                trials_for_segments, labels_for_segments = self._filter_trials_by_classes(
                    trials, labels, self.target_classes
                )
            else:
                # Online: just map labels (already filtered)
                labels_for_segments = self._map_labels_to_indices(labels, self.target_classes)

        # v3.0: Apply sliding window, filter, normalize
        segments, seg_labels, trial_indices = trials_to_segments(
            trials_for_segments, labels_for_segments, self.config
        )

        # Apply channel selection if needed
        if self.channel_indices is not None:
            segments = segments[:, self.channel_indices, :]

        # Store segments
        self._store_segments(segments, seg_labels, trial_indices, session_info)

    def _store_segments(
        self,
        segments: np.ndarray,
        seg_labels: np.ndarray,
        trial_indices: np.ndarray,
        session_info: Dict
    ):
        """
        Store preprocessed segments into the dataset.

        CRITICAL FIX: Use global unique trial indices to prevent data leakage.
        Each trial gets a globally unique ID across all runs.
        """
        # Get unique local trial indices from this run
        unique_local_trials = np.unique(trial_indices)

        # Create mapping from local trial_idx to global trial_idx
        local_to_global = {}
        for local_idx in unique_local_trials:
            local_to_global[local_idx] = self._global_trial_counter
            self._global_trial_counter += 1

        # Store segments with globally unique trial indices
        for i, (segment, label, local_trial_idx) in enumerate(zip(segments, seg_labels, trial_indices)):
            # Map local trial index to global unique index
            global_trial_idx = local_to_global[local_trial_idx]

            trial_info = TrialInfo(
                subject_id=session_info['subject'],
                session_type=session_info['session_folder'],  # CRITICAL FIX: Use full folder name for unique identification
                run_id=session_info['run'],
                trial_idx=global_trial_idx,  # FIXED: Use globally unique trial index
                target_class=int(self.target_classes[label]) if self.target_classes else label,
                start_sample=0,  # Segment doesn't have original sample info
                end_sample=int(self.config.segment_length * self.config.original_fs),
            )

            self.trials.append(segment)
            self.labels.append(label)
            self.trial_infos.append(trial_info)

    def _map_labels_to_indices(
        self,
        labels: np.ndarray,
        target_classes: List[int],
    ) -> np.ndarray:
        """
        Map original labels (finger IDs) to continuous indices.

        v3.0: Used for Online data where labels are already filtered
        but need to be mapped to continuous indices (0, 1, ..., n_classes-1).

        Args:
            labels: Original labels (finger IDs: 1, 4 for binary)
            target_classes: Target classes (e.g., [1, 4] for binary)

        Returns:
            Mapped labels using continuous indices (0, 1, ..., n_classes-1)
        """
        label_mapping = {cls: i for i, cls in enumerate(sorted(target_classes))}
        return np.array([label_mapping[l] for l in labels])

    def _filter_trials_by_classes(
        self,
        trials: np.ndarray,
        labels: np.ndarray,
        target_classes: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter trials to keep only those matching target_classes.

        v3.0: Used for Offline data where all 4 fingers are cached together,
        but we need to filter to specific classes at load time.

        Args:
            trials: All trials [n_trials x channels x samples]
            labels: Original labels (finger IDs: 1,2,3,4)
            target_classes: Classes to keep (e.g., [1, 4] for binary)

        Returns:
            Tuple of (filtered_trials, mapped_labels)
            where mapped_labels use continuous indices (0, 1, ..., n_classes-1)
        """
        # Create mask for target classes
        mask = np.isin(labels, target_classes)

        # Filter
        filtered_trials = trials[mask]

        # Map labels to continuous indices
        filtered_labels = self._map_labels_to_indices(labels[mask], target_classes)

        return filtered_trials, filtered_labels

    def _filter_segments_by_classes(
        self,
        segments: np.ndarray,
        labels: np.ndarray,
        trial_indices: np.ndarray,
        target_classes: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter segments to keep only those matching target_classes.

        DEPRECATED in v3.0: Use _filter_trials_by_classes instead.
        Kept for backward compatibility with non-sliding-window mode.

        Args:
            segments: All segments [n_segments x channels x samples]
            labels: Original labels (finger IDs: 1,2,3,4)
            trial_indices: Trial indices for each segment
            target_classes: Classes to keep (e.g., [1, 4] for binary)

        Returns:
            Tuple of (filtered_segments, mapped_labels, filtered_trial_indices)
            where mapped_labels use continuous indices (0, 1, ..., n_classes-1)
        """
        # Create mask for target classes
        mask = np.isin(labels, target_classes)

        # Filter
        filtered_segments = segments[mask]
        filtered_trial_indices = trial_indices[mask]

        # Map labels to continuous indices
        label_mapping = {cls: i for i, cls in enumerate(sorted(target_classes))}
        filtered_labels = np.array([label_mapping[l] for l in labels[mask]])

        return filtered_segments, filtered_labels, filtered_trial_indices

    def _is_offline_session(self, folder_name: str) -> bool:
        """Check if the session folder is an Offline session."""
        return folder_name.lower().startswith('offline')

    def _load_run_trial_based(
        self,
        eeg_data: np.ndarray,
        events: List[Dict],
        metadata: Dict,
        session_info: Dict,
        mat_path: Path
    ):
        """
        Load a run using trial-based preprocessing (for CBraMod).

        Processes each trial independently.
        """
        # Extract trials
        file_trials = extract_trials(eeg_data, events, metadata, self.config)

        for trial_data, label, trial_info in file_trials:
            # Filter by target class
            if self.target_classes is not None:
                if label not in self.target_classes:
                    continue

            # Update trial info
            trial_info.subject_id = session_info['subject']
            trial_info.session_type = session_info['session_folder']  # CRITICAL FIX: Use full folder name for unique identification
            trial_info.run_id = session_info['run']

            # Preprocess
            processed = preprocess_trial(
                trial_data, self.config, self.channel_indices
            )

            if processed is not None:
                # Map label if needed
                if self.target_classes is not None:
                    label_mapping = {cls: i for i, cls in enumerate(sorted(self.target_classes))}
                    label = label_mapping[label]

                self.trials.append(processed)
                self.labels.append(label)
                self.trial_infos.append(trial_info)

    def _setup_label_mapping(self):
        """
        Create mapping from original labels to contiguous indices.

        If use_sliding_window=True, labels are already mapped to 0-indexed during loading.
        Otherwise, create mapping from original labels.
        """
        unique_labels = sorted(set(self.labels))

        if self.config.use_sliding_window and self.target_classes is not None:
            # Labels already mapped to 0-indexed during paper-aligned preprocessing
            self.label_to_idx = {i: i for i in unique_labels}
            self.idx_to_label = {i: i for i in unique_labels}
        else:
            # Create new mapping
            self.label_to_idx = {label: i for i, label in enumerate(unique_labels)}
            self.idx_to_label = {i: label for label, i in self.label_to_idx.items()}

        self.n_classes = len(unique_labels)

        log_load.debug(f"Labels: {self.label_to_idx}, n_classes={self.n_classes}")

    def _convert_to_tensors(self):
        """
        Convert numpy arrays to PyTorch tensors for faster __getitem__.

        This trades memory for speed: ~5-10% faster training by avoiding
        numpy->tensor conversion on every batch.
        """
        import time
        start = time.perf_counter()

        # Convert trials to a single stacked tensor for efficiency
        # This also enables potential future optimizations like pinned memory
        self.trials = [torch.from_numpy(t).float() for t in self.trials]

        # Also pre-convert labels to tensor
        self.labels_tensor = torch.tensor(
            [self.label_to_idx[label] for label in self.labels],
            dtype=torch.long
        )

        elapsed = time.perf_counter() - start
        log_load.debug(f"Tensors: {len(self.trials)} trials in {elapsed:.2f}s")

    def __len__(self) -> int:
        return len(self.trials)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Fast path: use pre-converted tensors
        if self.preconvert_tensors and hasattr(self, 'labels_tensor'):
            trial_tensor = self.trials[idx]
            label = self.labels_tensor[idx].item()
        else:
            # Slow path: convert numpy to tensor on-the-fly
            trial = self.trials[idx]
            label = self.label_to_idx[self.labels[idx]]
            trial_tensor = torch.from_numpy(trial).float()

        if self.transform:
            trial_tensor = self.transform(trial_tensor)

        return trial_tensor, label

    def get_trial_info(self, idx: int) -> TrialInfo:
        """Get detailed information about a trial."""
        return self.trial_infos[idx]

    def get_unique_trials(self) -> List[int]:
        """
        Get list of unique trial indices in the dataset.

        Returns:
            Sorted list of unique trial indices (globally unique)
        """
        unique_trials = sorted(set(info.trial_idx for info in self.trial_infos))
        return unique_trials

    def get_segment_indices_for_trials(self, trial_indices: List[int]) -> List[int]:
        """
        Get segment indices that belong to specific trials.

        Args:
            trial_indices: List of trial indices to filter

        Returns:
            List of segment indices belonging to the specified trials
        """
        trial_set = set(trial_indices)
        segment_indices = [
            i for i, info in enumerate(self.trial_infos)
            if info.trial_idx in trial_set
        ]
        return segment_indices


def create_dataloaders(
    data_root: str,
    config: PreprocessConfig,
    elc_path: str,
    train_subjects: List[str],
    val_subjects: List[str],
    test_subjects: List[str],
    task_types: Optional[List[str]] = None,
    target_classes: Optional[List[int]] = None,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders.

    Args:
        data_root: Path to data directory
        config: Preprocessing configuration
        elc_path: Path to electrode file
        train_subjects: List of training subject IDs
        val_subjects: List of validation subject IDs
        test_subjects: List of test subject IDs
        task_types: Task types to include
        target_classes: Target classes to include
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = FingerEEGDataset(
        data_root, train_subjects, config,
        task_types=task_types,
        target_classes=target_classes,
        elc_path=elc_path
    )

    val_dataset = FingerEEGDataset(
        data_root, val_subjects, config,
        task_types=task_types,
        target_classes=target_classes,
        elc_path=elc_path
    )

    test_dataset = FingerEEGDataset(
        data_root, test_subjects, config,
        task_types=task_types,
        target_classes=target_classes,
        elc_path=elc_path
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test data loading with paper-aligned preprocessing
    import sys

    logging.basicConfig(level=logging.INFO)

    data_root = Path(__file__).parent.parent.parent / 'data'
    elc_path = data_root / 'biosemi128.ELC'

    if not data_root.exists():
        print(f"Data root not found: {data_root}")
        sys.exit(1)

    print("=" * 60)
    print("Testing Paper-Aligned Preprocessing")
    print("=" * 60)

    # Test paper-aligned configuration
    config = PreprocessConfig.paper_aligned()
    print(f"\nPaper-aligned config:")
    print(f"  Target model: {config.target_model}")
    print(f"  Sampling rate: {config.target_fs} Hz")
    print(f"  Bandpass: {config.bandpass_low}-{config.bandpass_high} Hz")
    print(f"  Normalization: {config.normalize_method}")
    print(f"  CAR: {config.apply_car}")
    print(f"  Sliding window: {config.segment_length}s, step={config.segment_step_samples} samples")

    # List available subjects
    subjects = sorted([d.name for d in data_root.iterdir() if d.is_dir() and d.name.startswith('S')])
    print(f"\nFound subjects: {subjects}")

    if subjects:
        # Test with first subject's first run
        subject_dir = data_root / subjects[0] / 'OfflineImagery'
        if subject_dir.exists():
            mat_files = sorted(subject_dir.glob('*.mat'))
            if mat_files:
                print(f"\nTesting with: {mat_files[0].name}")

                # Load raw data
                eeg_data, events, metadata = load_mat_file(str(mat_files[0]))
                print(f"  Raw data shape: {eeg_data.shape}")
                print(f"  Sampling rate: {metadata['fsample']} Hz")
                print(f"  Number of events: {len(events)}")

                # Test paper-aligned preprocessing
                segments, labels, trial_indices = preprocess_run_paper_aligned(
                    eeg_data, events, metadata, config,
                    target_classes=[1, 4],  # Thumb vs Pinky
                    label_mapping={1: 0, 4: 1}
                )

                print(f"\n  After preprocessing:")
                print(f"    Segments shape: {segments.shape}")
                print(f"    Expected: [n_segments x 128 x 100]")
                print(f"    Labels shape: {labels.shape}")
                print(f"    Unique labels: {np.unique(labels)}")

                # Verify output dimensions
                if len(segments) > 0:
                    assert segments.shape[1] == 128, f"Expected 128 channels, got {segments.shape[1]}"
                    assert segments.shape[2] == 100, f"Expected 100 samples, got {segments.shape[2]}"
                    print(f"\n  [OK] Output dimensions correct!")

                    # Check z-score normalization (mean ~0, std ~1 along time axis)
                    seg_mean = segments[0].mean(axis=-1).mean()
                    seg_std = segments[0].std(axis=-1).mean()
                    print(f"  [OK] Z-score check: mean={seg_mean:.4f}, std={seg_std:.4f}")

        # Also test Dataset class
        print("\n" + "=" * 60)
        print("Testing FingerEEGDataset")
        print("=" * 60)

        dataset = FingerEEGDataset(
            str(data_root),
            subjects[:1],
            config,
            task_types=['OfflineImagery'],
            target_classes=[1, 4],  # Thumb vs Pinky
            elc_path=str(elc_path) if elc_path.exists() else None
        )

        print(f"\nDataset size: {len(dataset)}")
        print(f"Number of classes: {dataset.n_classes}")
        print(f"Label mapping: {dataset.label_to_idx}")

        if len(dataset) > 0:
            trial, label = dataset[0]
            print(f"\nFirst trial shape: {trial.shape}")
            print(f"First trial label: {label}")
            print(f"Trial info: {dataset.get_trial_info(0)}")

    print("\n" + "=" * 60)
    print("Testing CBraMod Configuration")
    print("=" * 60)

    config_cbramod = PreprocessConfig.for_cbramod()
    print(f"\nCBraMod config:")
    print(f"  Target model: {config_cbramod.target_model}")
    print(f"  Sampling rate: {config_cbramod.target_fs} Hz")
    print(f"  Bandpass: {config_cbramod.bandpass_low}-{config_cbramod.bandpass_high} Hz")
    print(f"  Normalization: {config_cbramod.normalize_method}")
    print(f"  Notch filter: {config_cbramod.notch_freq} Hz")
