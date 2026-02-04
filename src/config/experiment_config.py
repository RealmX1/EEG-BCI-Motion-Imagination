"""
Experiment configuration for CBraMod data preprocessing ML engineering study.

This module defines experiment configurations for systematically evaluating
different preprocessing parameters on CBraMod performance.

Key constraints from CBraMod paper (ICLR 2025):
- Sampling rate: FIXED at 200 Hz (must match pretrained patch length)
- Patch length: FIXED at 1 second (200 samples)
- Normalization: ÷100 is MANDATORY (can add extra normalization after)
- Channels: FIXED at 128 (full BioSemi data)

Flexible parameters (vary across downstream datasets in paper):
- Bandpass filter range
- Notch filter presence/frequency
- Additional normalization after ÷100
- Sliding window step size
- Amplitude rejection threshold

Usage:
    from src.config.experiment_config import get_experiment_config, ALL_EXPERIMENTS

    # Get a specific experiment config
    config = get_experiment_config("A2")

    # Get all experiments in a group
    group_a_configs = {k: v for k, v in ALL_EXPERIMENTS.items() if k.startswith("A")}
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExperimentPreprocessConfig:
    """
    Experiment configuration for preprocessing parameter study.

    Fixed parameters (paper constraints):
    - target_fs: 200 Hz (FIXED)
    - segment_length: 1.0 seconds (FIXED)
    - normalize_by: 100.0 (FIXED)
    - n_channels: 128 (FIXED)

    Tunable parameters:
    - bandpass_low/high: Bandpass filter cutoff frequencies
    - notch_freq: Notch filter frequency (None to disable)
    - extra_normalize: Additional normalization after ÷100
    - segment_step_ms: Sliding window step in milliseconds
    - amplitude_threshold: Amplitude rejection threshold (None to disable)
    """

    # Experiment identification
    experiment_id: str
    experiment_group: str  # A (filter), C (normalize), D (window), F (quality)
    description: str = ""

    # Filtering parameters (Group A)
    bandpass_low: float = 0.3
    bandpass_high: float = 75.0
    notch_freq: Optional[float] = 60.0
    filter_order: int = 4

    # Sampling rate - FIXED at 200 Hz (paper constraint)
    target_fs: int = field(default=200, repr=False)
    original_fs: int = field(default=1024, repr=False)

    # Normalization parameters (Group C)
    # divide_100 is mandatory, extra_normalize is optional
    normalize_by: float = field(default=100.0, repr=False)  # FIXED
    extra_normalize: Optional[str] = None  # 'zscore_time', 'zscore_channel', 'robust', None

    # Window parameters (Group D)
    segment_length: float = field(default=1.0, repr=False)  # FIXED at 1 second
    segment_step_ms: float = 125.0  # Step in milliseconds (tunable)

    # Channel count - FIXED at 128
    n_channels: int = field(default=128, repr=False)

    # Quality control parameters (Group F)
    amplitude_threshold: Optional[float] = 100.0  # μV, None to disable

    # Cache and tracking
    cache_tag: str = field(default="data_preproc_ml_eng", repr=False)

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration against paper constraints."""
        errors = []

        # Check fixed parameters
        if self.target_fs != 200:
            errors.append(f"target_fs must be 200 Hz (got {self.target_fs})")

        if self.segment_length != 1.0:
            errors.append(f"segment_length must be 1.0 seconds (got {self.segment_length})")

        if self.normalize_by != 100.0:
            errors.append(f"normalize_by must be 100.0 (got {self.normalize_by})")

        if self.n_channels != 128:
            errors.append(f"n_channels must be 128 (got {self.n_channels})")

        # Check valid extra_normalize values
        valid_extra = [None, 'zscore_time', 'zscore_channel', 'robust']
        if self.extra_normalize not in valid_extra:
            errors.append(f"extra_normalize must be one of {valid_extra}")

        # Check filter constraints
        if self.bandpass_low >= self.bandpass_high:
            errors.append(f"bandpass_low ({self.bandpass_low}) must be < bandpass_high ({self.bandpass_high})")

        if self.bandpass_high > 100:  # Nyquist limit for 200 Hz
            errors.append(f"bandpass_high ({self.bandpass_high}) exceeds Nyquist frequency (100 Hz)")

        # Check step size
        if self.segment_step_ms <= 0 or self.segment_step_ms > 1000:
            errors.append(f"segment_step_ms ({self.segment_step_ms}) must be between 0 and 1000")

        if errors:
            raise ValueError(f"Invalid experiment config {self.experiment_id}: " + "; ".join(errors))

    def to_cache_tag(self) -> str:
        """Generate cache tag for this experiment."""
        return f"{self.cache_tag}/{self.experiment_id}"

    @property
    def segment_step_samples(self) -> int:
        """
        Convert step from ms to samples at original_fs.

        This matches the expected input format for PreprocessConfig.
        """
        return int(self.segment_step_ms * self.original_fs / 1000)

    def to_preprocess_config_dict(self) -> Dict:
        """
        Convert to dict compatible with PreprocessConfig.

        This creates a configuration dict that can be used to create
        a PreprocessConfig instance for data loading.
        """
        return {
            'target_model': 'cbramod_128ch',
            'original_fs': self.original_fs,
            'target_fs': self.target_fs,
            'bandpass_low': self.bandpass_low,
            'bandpass_high': self.bandpass_high,
            'notch_freq': self.notch_freq,
            'filter_order': self.filter_order,
            'channel_strategy': 'C',  # All 128 channels
            'trial_duration': 5.0,  # Offline trials
            'online_trial_duration': 3.0,  # Online trials
            'segment_length': self.segment_length,
            'segment_step_samples': self.segment_step_samples,
            'use_sliding_window': True,
            'normalize_method': 'divide',  # Will be handled specially
            'normalize_by': self.normalize_by,
            'reject_threshold': self.amplitude_threshold if self.amplitude_threshold else -1.0,
            'apply_car': True,
            'filter_padding': 100,
            # Custom fields for experiment tracking
            '_experiment_id': self.experiment_id,
            '_extra_normalize': self.extra_normalize,
        }

    def get_wandb_tags(self) -> List[str]:
        """Get WandB tags for this experiment."""
        tags = [
            "preproc-ml-eng",
            f"exp:{self.experiment_id}",
            f"group:{self.experiment_group}",
        ]
        return tags

    def get_wandb_config(self) -> Dict:
        """Get WandB config dict for this experiment."""
        return {
            'experiment_id': self.experiment_id,
            'experiment_group': self.experiment_group,
            'description': self.description,
            'bandpass': [self.bandpass_low, self.bandpass_high],
            'notch_freq': self.notch_freq,
            'extra_normalize': self.extra_normalize,
            'segment_step_ms': self.segment_step_ms,
            'amplitude_threshold': self.amplitude_threshold,
            # Fixed parameters (for documentation)
            'target_fs': self.target_fs,
            'segment_length': self.segment_length,
            'normalize_by': self.normalize_by,
            'n_channels': self.n_channels,
        }


# =============================================================================
# Experiment Definitions
# =============================================================================

def _create_experiments() -> Dict[str, ExperimentPreprocessConfig]:
    """Create all experiment configurations."""
    experiments = {}

    # =========================================================================
    # Group A: Filtering Parameters (6 configs)
    # =========================================================================
    experiments['A1'] = ExperimentPreprocessConfig(
        experiment_id='A1',
        experiment_group='A',
        description='Baseline - TUEG pretrain config (0.3-75Hz, 60Hz notch)',
        bandpass_low=0.3,
        bandpass_high=75.0,
        notch_freq=60.0,
    )

    # A2-A5 removed: prototype showed filtering variations have minimal impact
    # Keeping only A6 to test notch filter necessity

    experiments['A6'] = ExperimentPreprocessConfig(
        experiment_id='A6',
        experiment_group='A',
        description='Baseline without notch filter (0.3-75Hz)',
        bandpass_low=0.3,
        bandpass_high=75.0,
        notch_freq=None,
    )

    # =========================================================================
    # Group C: Normalization Strategies (4 configs)
    # Note: A1 = C1 (same config, baseline)
    # =========================================================================
    experiments['C1'] = experiments['A1']  # Alias - divide_100 only (baseline)

    experiments['C2'] = ExperimentPreprocessConfig(
        experiment_id='C2',
        experiment_group='C',
        description='Divide by 100 + z-score along time axis',
        extra_normalize='zscore_time',
    )

    # C3, C4 removed: prototype showed C2 slightly decreased performance (-1.9%)
    # Extra normalization after ÷100 likely disrupts pretrained weight expectations

    # =========================================================================
    # Group D: Window Sliding Strategy (3 configs)
    # Note: A1 = D1 (same config, baseline with 125ms step)
    # =========================================================================
    experiments['D1'] = experiments['A1']  # Alias - 125ms step (baseline)

    experiments['D2'] = ExperimentPreprocessConfig(
        experiment_id='D2',
        experiment_group='D',
        description='Sliding step 250ms (75% overlap)',
        segment_step_ms=250.0,
    )

    experiments['D3'] = ExperimentPreprocessConfig(
        experiment_id='D3',
        experiment_group='D',
        description='Sliding step 500ms (50% overlap)',
        segment_step_ms=500.0,
    )

    # =========================================================================
    # Group F: Data Quality Control (2 configs)
    # Note: A1 = F1 (same config, baseline with 100μV threshold)
    # =========================================================================
    experiments['F1'] = experiments['A1']  # Alias - 100μV threshold (baseline)

    experiments['F2'] = ExperimentPreprocessConfig(
        experiment_id='F2',
        experiment_group='F',
        description='No amplitude rejection (disable threshold)',
        amplitude_threshold=None,
    )

    return experiments


# Global experiment registry
ALL_EXPERIMENTS = _create_experiments()

# Baseline config (A1 = C1 = D1 = F1)
BASELINE_CONFIG = ALL_EXPERIMENTS['A1']

# Group mappings
EXPERIMENT_GROUPS = {
    'A': ['A1', 'A6'],  # Filtering (A2-A5 removed after prototype)
    'C': ['C1', 'C2'],  # Normalization (C3, C4 removed after prototype)
    'D': ['D1', 'D2', 'D3'],  # Window
    'F': ['F1', 'F2'],  # Quality
}

# Unique configs (excluding aliases)
UNIQUE_EXPERIMENT_IDS = [
    'A1',  # Baseline (= C1 = D1 = F1)
    'A6',  # A group: test notch filter (A2-A5 removed after prototype)
    'C2',  # C group (C3, C4 removed after prototype)
    'D2', 'D3',  # D group (non-baseline)
    'F2',  # F group (non-baseline)
]


def get_experiment_config(experiment_id: str) -> ExperimentPreprocessConfig:
    """
    Get experiment configuration by ID.

    Args:
        experiment_id: Experiment ID (e.g., 'A1', 'A2', 'C2', etc.)

    Returns:
        ExperimentPreprocessConfig instance

    Raises:
        KeyError: If experiment_id not found
    """
    if experiment_id not in ALL_EXPERIMENTS:
        available = ', '.join(sorted(ALL_EXPERIMENTS.keys()))
        raise KeyError(f"Unknown experiment ID: {experiment_id}. Available: {available}")

    return ALL_EXPERIMENTS[experiment_id]


def get_experiments_in_group(group: str) -> List[ExperimentPreprocessConfig]:
    """
    Get all experiment configurations in a group.

    Args:
        group: Group ID ('A', 'C', 'D', or 'F')

    Returns:
        List of ExperimentPreprocessConfig instances
    """
    if group not in EXPERIMENT_GROUPS:
        raise KeyError(f"Unknown group: {group}. Available: {list(EXPERIMENT_GROUPS.keys())}")

    return [ALL_EXPERIMENTS[exp_id] for exp_id in EXPERIMENT_GROUPS[group]]


def get_unique_experiments() -> List[ExperimentPreprocessConfig]:
    """
    Get list of unique experiments (excluding aliases).

    Returns:
        List of unique ExperimentPreprocessConfig instances
    """
    return [ALL_EXPERIMENTS[exp_id] for exp_id in UNIQUE_EXPERIMENT_IDS]


def is_baseline(experiment_id: str) -> bool:
    """Check if an experiment ID is the baseline config."""
    return experiment_id in ['A1', 'C1', 'D1', 'F1']


def print_experiment_summary():
    """Print summary of all experiments."""
    print("\n" + "=" * 80)
    print("CBraMod Data Preprocessing ML Engineering Experiments")
    print("=" * 80)

    for group, exp_ids in EXPERIMENT_GROUPS.items():
        print(f"\n--- Group {group} ---")
        for exp_id in exp_ids:
            config = ALL_EXPERIMENTS[exp_id]
            is_alias = exp_id != 'A1' and config is ALL_EXPERIMENTS['A1']
            alias_note = " (= A1 baseline)" if is_alias else ""
            print(f"  {exp_id}: {config.description}{alias_note}")

    print("\n" + "=" * 80)
    print(f"Total unique configurations: {len(UNIQUE_EXPERIMENT_IDS)}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    print_experiment_summary()

    # Test validation
    print("Testing baseline config validation...")
    baseline = get_experiment_config('A1')
    print(f"  Baseline: {baseline}")
    print(f"  Cache tag: {baseline.to_cache_tag()}")
    print(f"  WandB tags: {baseline.get_wandb_tags()}")
    print(f"  Step samples: {baseline.segment_step_samples}")

    print("\n✓ All experiments validated successfully!")
