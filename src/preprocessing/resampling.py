"""
Resampling utilities for EEG data.

Implements resampling from original sampling rate to target rate.
CBraMod requires 200 Hz input.
EEGNet uses 100 Hz (as per original FINGER-EEG-BCI paper).
"""

import numpy as np
from scipy.signal import resample_poly
from typing import Optional
from math import gcd


def calculate_resample_factors(
    original_fs: int,
    target_fs: int
) -> tuple:
    """
    Calculate up/down sampling factors for resample_poly.

    Args:
        original_fs: Original sampling frequency in Hz
        target_fs: Target sampling frequency in Hz

    Returns:
        Tuple of (up, down) factors
    """
    # Find GCD for rational resampling
    common = gcd(original_fs, target_fs)
    up = target_fs // common
    down = original_fs // common

    return up, down


def resample_eeg(
    data: np.ndarray,
    original_fs: int,
    target_fs: int,
    axis: int = -1
) -> np.ndarray:
    """
    Resample EEG data to target sampling frequency.

    Uses scipy.signal.resample_poly which includes anti-aliasing filter.

    Args:
        data: EEG data array [channels x time] or [time]
        original_fs: Original sampling frequency in Hz (e.g., 1024)
        target_fs: Target sampling frequency in Hz (e.g., 200 for CBraMod)
        axis: Time axis (default -1)

    Returns:
        Resampled data array
    """
    up, down = calculate_resample_factors(original_fs, target_fs)

    # resample_poly handles anti-aliasing internally
    resampled = resample_poly(data, up, down, axis=axis)

    return resampled


def resample_with_timestamps(
    data: np.ndarray,
    timestamps: np.ndarray,
    original_fs: int,
    target_fs: int
) -> tuple:
    """
    Resample EEG data and regenerate timestamps.

    Args:
        data: EEG data array [channels x time]
        timestamps: Original timestamp array
        original_fs: Original sampling frequency in Hz
        target_fs: Target sampling frequency in Hz

    Returns:
        Tuple of (resampled_data, new_timestamps)
    """
    resampled = resample_eeg(data, original_fs, target_fs)

    # Generate new timestamps
    n_samples = resampled.shape[-1]
    if len(timestamps) > 0:
        start_time = timestamps[0]
        new_timestamps = start_time + np.arange(n_samples) / target_fs
    else:
        new_timestamps = np.arange(n_samples) / target_fs

    return resampled, new_timestamps


def resample_events(
    events: list,
    original_fs: int,
    target_fs: int
) -> list:
    """
    Adjust event sample indices after resampling.

    Args:
        events: List of event dicts with 'sample' field
        original_fs: Original sampling frequency
        target_fs: Target sampling frequency

    Returns:
        List of events with adjusted sample indices
    """
    ratio = target_fs / original_fs

    adjusted_events = []
    for event in events:
        new_event = event.copy()
        if 'sample' in new_event:
            new_event['sample'] = int(new_event['sample'] * ratio)
        adjusted_events.append(new_event)

    return adjusted_events


def create_patches(
    data: np.ndarray,
    patch_duration: float = 1.0,
    fs: int = 200,
    overlap: float = 0.0
) -> np.ndarray:
    """
    Segment continuous EEG data into fixed-duration patches.

    CBraMod uses 1-second patches (200 samples @ 200 Hz).

    Args:
        data: EEG data array [channels x time]
        patch_duration: Duration of each patch in seconds (default 1.0)
        fs: Sampling frequency in Hz (default 200)
        overlap: Overlap fraction between patches (default 0.0, no overlap)

    Returns:
        Patches array [n_patches x channels x patch_samples]
    """
    patch_samples = int(patch_duration * fs)
    hop_samples = int(patch_samples * (1 - overlap))

    n_channels, n_time = data.shape
    n_patches = (n_time - patch_samples) // hop_samples + 1

    if n_patches <= 0:
        raise ValueError(
            f"Data too short for patches: {n_time} samples, "
            f"need at least {patch_samples} samples"
        )

    patches = np.zeros((n_patches, n_channels, patch_samples))

    for i in range(n_patches):
        start = i * hop_samples
        end = start + patch_samples
        patches[i] = data[:, start:end]

    return patches


def extract_trial_segment(
    data: np.ndarray,
    start_sample: int,
    duration_seconds: float,
    fs: int,
    pre_onset: float = 0.0,
    post_offset: float = 0.0
) -> Optional[np.ndarray]:
    """
    Extract a trial segment from continuous data.

    Args:
        data: Continuous EEG data [channels x time]
        start_sample: Trial onset sample index
        duration_seconds: Trial duration in seconds
        fs: Sampling frequency
        pre_onset: Time before onset to include (seconds)
        post_offset: Time after offset to include (seconds)

    Returns:
        Trial segment array or None if out of bounds
    """
    pre_samples = int(pre_onset * fs)
    duration_samples = int(duration_seconds * fs)
    post_samples = int(post_offset * fs)

    segment_start = start_sample - pre_samples
    segment_end = start_sample + duration_samples + post_samples

    if segment_start < 0 or segment_end > data.shape[-1]:
        return None

    return data[:, segment_start:segment_end]


if __name__ == '__main__':
    # Test resampling
    print("Resampling test:")

    # Original: 1024 Hz, Target: 200 Hz
    up, down = calculate_resample_factors(1024, 200)
    print(f"  1024 Hz -> 200 Hz: up={up}, down={down}")

    # Original: 1024 Hz, Target: 100 Hz
    up, down = calculate_resample_factors(1024, 100)
    print(f"  1024 Hz -> 100 Hz: up={up}, down={down}")

    # Test with actual data
    original_fs = 1024
    target_fs = 200
    duration = 5  # seconds

    # Create test data
    n_channels = 19
    n_samples = duration * original_fs
    data = np.random.randn(n_channels, n_samples)

    # Resample
    resampled = resample_eeg(data, original_fs, target_fs)

    print(f"\n  Original shape: {data.shape}")
    print(f"  Resampled shape: {resampled.shape}")
    print(f"  Expected samples: {duration * target_fs}")

    # Test patching
    patches = create_patches(resampled, patch_duration=1.0, fs=target_fs)
    print(f"\n  Patches shape: {patches.shape}")
    print(f"  Expected: ({duration}, {n_channels}, {target_fs})")
