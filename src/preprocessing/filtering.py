"""
Filtering utilities for EEG preprocessing.

Implements bandpass and notch filters following CBraMod paper specifications:
- Bandpass: 0.3-75 Hz
- Notch: 60 Hz (US power line frequency)
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from typing import Tuple, Optional


def butter_bandpass_coefficients(
    low_freq: float,
    high_freq: float,
    fs: float,
    order: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Butterworth bandpass filter coefficients.

    Args:
        low_freq: Low cutoff frequency in Hz
        high_freq: High cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order (default 4)

    Returns:
        Tuple of (b, a) filter coefficients
    """
    nyq = fs / 2.0
    low = low_freq / nyq
    high = high_freq / nyq

    # Clamp to valid range
    low = max(0.001, min(low, 0.999))
    high = max(low + 0.001, min(high, 0.999))

    b, a = butter(order, [low, high], btype='bandpass')
    return b, a


def bandpass_filter(
    data: np.ndarray,
    low_freq: float,
    high_freq: float,
    fs: float,
    order: int = 4,
    axis: int = -1
) -> np.ndarray:
    """
    Apply bandpass filter to EEG data.

    Uses zero-phase filtering (filtfilt) to avoid phase distortion.

    Args:
        data: EEG data array [channels x time] or [time]
        low_freq: Low cutoff frequency in Hz (CBraMod: 0.3 Hz)
        high_freq: High cutoff frequency in Hz (CBraMod: 75 Hz)
        fs: Sampling frequency in Hz
        order: Filter order (default 4)
        axis: Axis along which to filter (default -1, time axis)

    Returns:
        Filtered data array
    """
    b, a = butter_bandpass_coefficients(low_freq, high_freq, fs, order)

    # Apply zero-phase filtering
    filtered = filtfilt(b, a, data, axis=axis)

    return filtered


def notch_filter_coefficients(
    notch_freq: float,
    fs: float,
    quality_factor: float = 30.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate notch filter coefficients.

    Args:
        notch_freq: Frequency to remove in Hz
        fs: Sampling frequency in Hz
        quality_factor: Q factor (higher = narrower notch)

    Returns:
        Tuple of (b, a) filter coefficients
    """
    b, a = iirnotch(notch_freq, quality_factor, fs)
    return b, a


def notch_filter(
    data: np.ndarray,
    notch_freq: float,
    fs: float,
    quality_factor: float = 30.0,
    axis: int = -1
) -> np.ndarray:
    """
    Apply notch filter to remove power line interference.

    Args:
        data: EEG data array [channels x time] or [time]
        notch_freq: Frequency to remove in Hz (US: 60 Hz, EU: 50 Hz)
        fs: Sampling frequency in Hz
        quality_factor: Q factor (higher = narrower notch, default 30)
        axis: Axis along which to filter (default -1, time axis)

    Returns:
        Filtered data array
    """
    b, a = notch_filter_coefficients(notch_freq, fs, quality_factor)

    # Apply zero-phase filtering
    filtered = filtfilt(b, a, data, axis=axis)

    return filtered


def preprocess_filter_chain(
    data: np.ndarray,
    fs: float,
    bandpass_low: float = 0.3,
    bandpass_high: float = 75.0,
    notch_freq: Optional[float] = 60.0,
    notch_harmonics: bool = False,
    axis: int = -1
) -> np.ndarray:
    """
    Apply complete filtering chain as per CBraMod specifications.

    Pipeline:
    1. Bandpass filter (0.3-75 Hz)
    2. Notch filter (60 Hz) - optional
    3. Optional: Notch filter for harmonics (120 Hz, 180 Hz, ...)

    Args:
        data: EEG data array [channels x time]
        fs: Sampling frequency in Hz
        bandpass_low: Low cutoff for bandpass (default 0.3 Hz)
        bandpass_high: High cutoff for bandpass (default 75 Hz)
        notch_freq: Power line frequency (default 60 Hz for US), None to skip
        notch_harmonics: Whether to also filter harmonics (default False)
        axis: Time axis (default -1)

    Returns:
        Filtered data array
    """
    # Step 1: Bandpass filter
    filtered = bandpass_filter(
        data, bandpass_low, bandpass_high, fs, axis=axis
    )

    # Step 2: Notch filter for power line (optional)
    if notch_freq is not None and notch_freq > 0 and notch_freq < fs / 2:
        filtered = notch_filter(filtered, notch_freq, fs, axis=axis)

        # Step 3: Optional harmonics
        if notch_harmonics:
            harmonic = notch_freq * 2
            while harmonic < min(bandpass_high, fs / 2):
                filtered = notch_filter(filtered, harmonic, fs, axis=axis)
                harmonic += notch_freq

    return filtered


def check_amplitude_range(
    data: np.ndarray,
    max_amplitude: float = 100.0,
    return_mask: bool = False
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Check if data falls within valid amplitude range.

    CBraMod pretraining removes samples with |amplitude| > 100 µV.

    Args:
        data: EEG data array (should be in µV)
        max_amplitude: Maximum allowed amplitude in µV (default 100)
        return_mask: Whether to return mask of valid samples

    Returns:
        Tuple of (is_valid, mask)
        - is_valid: True if all values within range
        - mask: Boolean mask of valid samples (if return_mask=True)
    """
    is_valid = np.all(np.abs(data) <= max_amplitude)

    mask = None
    if return_mask:
        # Mask is True for samples where ALL channels are within range
        if data.ndim == 1:
            mask = np.abs(data) <= max_amplitude
        else:
            # Assume [channels x time]
            mask = np.all(np.abs(data) <= max_amplitude, axis=0)

    return is_valid, mask


if __name__ == '__main__':
    # Test filtering functions
    import matplotlib.pyplot as plt

    # Generate test signal
    fs = 1024  # Hz
    t = np.arange(0, 5, 1/fs)

    # Mix of frequencies: 10 Hz (alpha), 60 Hz (noise), 80 Hz (should be filtered)
    signal = (
        np.sin(2 * np.pi * 10 * t) +  # 10 Hz alpha
        0.5 * np.sin(2 * np.pi * 60 * t) +  # 60 Hz power line
        0.3 * np.sin(2 * np.pi * 80 * t)  # 80 Hz (above bandpass)
    )

    # Apply filtering
    filtered = preprocess_filter_chain(signal, fs)

    print("Test signal filtering:")
    print(f"  Original: {len(signal)} samples @ {fs} Hz")
    print(f"  After filtering: {len(filtered)} samples")
    print(f"  Max amplitude original: {np.max(np.abs(signal)):.3f}")
    print(f"  Max amplitude filtered: {np.max(np.abs(filtered)):.3f}")

    # Check amplitude range
    is_valid, _ = check_amplitude_range(filtered * 50, max_amplitude=100)
    print(f"  Amplitude check (50 µV): {is_valid}")
