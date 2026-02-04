"""
Preprocessing pipeline functions for FINGER-EEG-BCI dataset.

This module implements the preprocessing pipeline from the original paper:
1. Common Average Reference (CAR)
2. Trial extraction based on events
3. Sliding window segmentation (1s window, 125ms step)
4. Downsample to target rate (100 Hz for EEGNet, 200 Hz for CBraMod)
5. Bandpass filter (4-40 Hz for EEGNet)
6. Z-score normalization per segment (along time axis)
"""

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import scipy.signal
import scipy.stats

from .loader import load_mat_file, parse_session_path
from .filtering import preprocess_filter_chain, check_amplitude_range
from .resampling import resample_eeg
from ..utils.logging import SectionLogger

if TYPE_CHECKING:
    from .data_loader import PreprocessConfig

logger = logging.getLogger(__name__)
log_prep = SectionLogger(logger, 'prep')


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


def extract_trials(
    eeg_data: np.ndarray,
    events: List[Dict],
    metadata: Dict,
    config: 'PreprocessConfig'
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
    config: 'PreprocessConfig',
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
    config: 'PreprocessConfig',
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
    from math import gcd

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
    config: 'PreprocessConfig',
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


def trials_to_segments(
    trials: np.ndarray,
    labels: np.ndarray,
    config: 'PreprocessConfig'
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
