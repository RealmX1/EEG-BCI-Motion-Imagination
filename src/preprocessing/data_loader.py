"""
Data loader for FINGER-EEG-BCI dataset.

DEPRECATED: This module is maintained for backward compatibility.
New code should import directly from the submodules:
- src.preprocessing.loader: load_mat_file, parse_session_path
- src.preprocessing.discovery: discover_available_subjects, discover_subjects_from_cache_index
- src.preprocessing.pipeline: preprocess_run_paper_aligned, trials_to_segments, etc.
- src.preprocessing.dataset: FingerEEGDataset, create_dataloaders

This module re-exports all symbols from the new locations.

Dataset structure:
- 21 subjects (S01-S21)
- Motor Execution (ME) and Motor Imagery (MI)
- Offline and Online sessions
- Target fingers: Thumb (1), Index (2), Middle (3), Pinky (4)
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils.logging import SectionLogger

logger = logging.getLogger(__name__)

# Section-specific loggers (kept for compatibility)
log_load = SectionLogger(logger, 'load')
log_prep = SectionLogger(logger, 'prep')
log_cache = SectionLogger(logger, 'cache')
log_chan = SectionLogger(logger, 'chan')


# ============================================================================
# Re-exports from src.preprocessing.loader
# ============================================================================
from .loader import (
    load_mat_file,
    parse_session_path,
)

# ============================================================================
# Re-exports from src.preprocessing.discovery
# ============================================================================
from .discovery import (
    get_session_folders_for_split,
    discover_available_subjects,
    discover_subjects_from_cache_index,
)

# ============================================================================
# Re-exports from src.preprocessing.pipeline
# ============================================================================
from .pipeline import (
    TrialInfo,
    apply_common_average_reference,
    segment_with_sliding_window,
    apply_bandpass_filter_paper,
    apply_zscore_per_segment,
    apply_robust_normalize,
    extract_trials,
    preprocess_trial,
    preprocess_run_paper_aligned,
    preprocess_run_to_trials,
    trials_to_segments,
    _process_single_mat_file,
    _process_single_mat_file_to_trials,
)

# ============================================================================
# Re-exports from src.preprocessing.dataset
# ============================================================================
from .dataset import (
    FingerEEGDataset,
    create_dataloaders,
)

# ============================================================================
# Re-exports from src.preprocessing.channel_selection
# ============================================================================
from .channel_selection import (
    create_biosemi128_to_1020_mapping,
    get_channel_indices,
    STANDARD_1020_CHANNELS,
    BIOSEMI_128_LABELS,
)


# ============================================================================
# PreprocessConfig - kept here as it's the central configuration class
# ============================================================================
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


# ============================================================================
# Module test code
# ============================================================================
if __name__ == '__main__':
    # Test data loading with paper-aligned preprocessing
    import sys
    from pathlib import Path

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
