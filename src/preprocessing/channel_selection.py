"""
Channel selection and mapping from BioSemi 128 to standard 10-20 system.

This module provides utilities for:
1. Loading BioSemi 128 electrode positions from .ELC file
2. Mapping to standard 10-20 positions used by CBraMod (19 channels)
3. Motor cortex high-density subset selection (alternative strategy)
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# CBraMod pretrained 19 channels (standard 10-20 subset from TUEG)
STANDARD_1020_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6',
    'O1', 'O2'
]

# Motor cortex 8-channel subset (verified against biosemi_8channels.elc)
# Used for testing reduced channel configurations (strategy 'D')
MOTOR_8_CHANNELS = {
    'Cz': 'A1', 'Pz': 'A3', 'PO7': 'A6', 'Oz': 'A21',
    'PO8': 'B3', 'C4': 'B21', 'Fz': 'C23', 'C3': 'D18',
}
MOTOR_8_CHANNEL_INDICES = [0, 2, 5, 20, 34, 52, 86, 113]

# ============================================================================
# 32-channel configurations for reduced-channel experiments
# ============================================================================

# Hard-coded 32ch configs (hand-picked)
# - motor_cortex: motor cortex focused selection (superset of 8ch)
# - commercial: standard commercial 32ch cap (10-20 layout)
# Data-driven configs (fdr, csp, attention, band_power) are loaded from JSON
CHANNEL_32_CONFIGS = {
    # Motor cortex focused: dense coverage around C3/Cz/C4 + SMA/premotor
    # Includes C23 (idx 86, near FCz/SMA) instead of D3 (idx 98, near F3):
    #   C23 is the closest electrode to SMA (dist 0.37 to FCz) — critical for MI/ME decoding
    #   D3 would be redundant with 5 existing electrodes already covering left premotor/F3 area
    'motor_cortex': [0, 2, 3, 5, 20, 32, 33, 34, 49, 50, 52, 53, 55,
                     62, 63, 64, 65, 66, 77, 85, 86, 90, 97, 107, 108,
                     110, 111, 112, 113, 114, 116, 123],
    # Standard commercial 32ch cap: 10-20 layout coverage
    # NOTE: This config simulates standard 10-20 layout by selecting the nearest
    # BioSemi 128 electrodes to each standard position. Results obtained with this
    # config do NOT directly generalize to real commercial 32ch EEG devices, which
    # differ in sensor technology (wet vs dry, active vs passive), impedance
    # characteristics, noise floor, analog front-end, on-device signal processing,
    # and physical electrode placement accuracy.
    'commercial': [0, 3, 5, 16, 17, 22, 29, 30, 33, 34, 44, 49, 52, 55,
                   62, 65, 66, 68, 76, 77, 85, 89, 90, 97, 98, 100, 107,
                   111, 113, 116, 123, 124],
    # Data-driven configs: loaded from JSON at runtime
    'fdr': None,
    'csp': None,
    'attention': None,
    'band_power': None,
}
CHANNEL_32_CONFIG_NAMES = list(CHANNEL_32_CONFIGS.keys())

# Default JSON path for data-driven channel selections
CHANNEL_32_SELECTIONS_JSON = 'results/32_channel/channel_selections.json'

# ============================================================================
# 61-channel configuration: standard 10-10 system
# ============================================================================

# Standard 10-10 system channel names (61 channels)
# This is the electrode set used by most commercial 64ch EEG systems
# (e.g., BrainVision actiCHamp Plus, ANT Neuro eego).
# Reference: commonly tested in channel-density studies as the "medium density" config.
STANDARD_1010_CHANNELS = [
    'Fp1', 'Fpz', 'Fp2',
    'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
    'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'POz', 'PO4', 'PO8',
    'O1', 'Oz', 'O2',
]

# Hard-coded 61ch config: nearest BioSemi 128 indices for standard 10-10 positions.
# Computed via greedy nearest-neighbor matching using MNE standard_1005 and biosemi128
# montages. All 61 positions map to unique BioSemi electrodes.
# Max mapping distance: PO8 -> A28 (28.0mm).
CHANNEL_61_CONFIGS = {
    'standard_1010': [
        0, 2, 4, 5, 7, 8, 9, 15, 16, 17, 18, 20, 21, 27, 28, 29,
        31, 33, 34, 35, 37, 38, 42, 44, 45, 51, 53, 55, 58, 60, 62,
        63, 67, 69, 70, 71, 75, 78, 79, 80, 82, 84, 86, 88, 91, 92,
        93, 97, 99, 101, 102, 103, 105, 107, 109, 111, 114, 119, 121,
        124, 126,
    ],
}
CHANNEL_61_CONFIG_NAMES = list(CHANNEL_61_CONFIGS.keys())

# Data-driven config names (valid for any channel count)
DATA_DRIVEN_CONFIG_NAMES = ['fdr', 'csp', 'attention', 'band_power']


def get_nch_indices(n_channels: int, config_name: str, json_path: Optional[str] = None) -> List[int]:
    """
    Get N-channel indices for a named configuration.

    For 32ch: hard-coded configs (motor_cortex, commercial) are returned directly.
    For any N: data-driven configs (fdr, csp, attention, band_power) are loaded
    from results/{N}_channel/channel_selections.json.

    Args:
        n_channels: Target number of channels (e.g. 8, 32)
        config_name: Configuration name (e.g. 'fdr', 'motor_cortex')
        json_path: Override path to channel_selections.json.
                  Defaults to results/{n_channels}_channel/channel_selections.json.

    Returns:
        Sorted list of channel indices (0-127)

    Raises:
        ValueError: If config not found or not available
    """
    # 32ch hard-coded configs
    if n_channels == 32 and config_name in CHANNEL_32_CONFIGS:
        indices = CHANNEL_32_CONFIGS[config_name]
        if indices is not None:
            return sorted(indices)

    # 61ch hard-coded configs
    if n_channels == 61 and config_name in CHANNEL_61_CONFIGS:
        indices = CHANNEL_61_CONFIGS[config_name]
        if indices is not None:
            return sorted(indices)

    # Data-driven configs: load from JSON
    if json_path is None:
        json_path = f'results/{n_channels}_channel/channel_selections.json'

    selections = load_channel_selections(json_path)
    if config_name not in selections:
        raise ValueError(
            f"Config '{config_name}' not found in {json_path}. "
            f"Run: uv run python scripts/analysis/compute_32ch_selections.py "
            f"--n-channels {n_channels} --methods {config_name}"
        )

    indices = selections[config_name]['indices']
    if len(indices) != n_channels:
        raise ValueError(
            f"Config '{config_name}' in {json_path} has {len(indices)} channels, "
            f"expected {n_channels}"
        )

    return sorted(indices)


def get_32ch_indices(config_name: str, json_path: Optional[str] = None) -> List[int]:
    """Backward-compatible wrapper for get_nch_indices(32, ...)."""
    return get_nch_indices(32, config_name, json_path)


def load_channel_selections(json_path: str) -> dict:
    """
    Load channel selections from JSON, merging hard-coded and data-driven.

    Args:
        json_path: Path to channel_selections.json

    Returns:
        Dict mapping config_name to {'indices': [...], 'description': '...', ...}
    """
    import json

    result = {}

    # Include hard-coded configs based on channel count in path
    if '32_channel' in str(json_path):
        for name, indices in CHANNEL_32_CONFIGS.items():
            if indices is not None:
                result[name] = {
                    'indices': sorted(indices),
                    'description': f'Hard-coded {name} configuration',
                }
    elif '61_channel' in str(json_path):
        for name, indices in CHANNEL_61_CONFIGS.items():
            if indices is not None:
                result[name] = {
                    'indices': sorted(indices),
                    'description': f'Hard-coded {name} configuration',
                }

    # Load data-driven configs from JSON if available
    json_file = Path(json_path)
    if json_file.exists():
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        configs = data.get('configs', {})
        for name, config in configs.items():
            if 'indices' in config:
                result[name] = config

    return result


def load_32ch_selections(json_path: str) -> dict:
    """Backward-compatible wrapper for load_channel_selections."""
    return load_channel_selections(json_path)


# Note: T3/T4/T5/T6 are also known as T7/T8/P7/P8 in some systems

# BioSemi 128 electrode labels
BIOSEMI_128_LABELS = [f"{letter}{num}" for letter in ['A', 'B', 'C', 'D'] for num in range(1, 33)]

# Standard 10-20 positions in 3D coordinates (based on standard montage, normalized to unit sphere)
# These are approximate positions in a coordinate system where:
# - X: right (+) to left (-)
# - Y: front (+) to back (-)
# - Z: up (+) to down (-)
STANDARD_1020_POSITIONS = {
    # Frontal pole
    'Fp1': np.array([-0.31, 0.95, 0.00]),
    'Fp2': np.array([0.31, 0.95, 0.00]),
    # Frontal
    'F7': np.array([-0.81, 0.59, 0.00]),
    'F3': np.array([-0.55, 0.67, 0.50]),
    'Fz': np.array([0.00, 0.72, 0.69]),
    'F4': np.array([0.55, 0.67, 0.50]),
    'F8': np.array([0.81, 0.59, 0.00]),
    # Temporal (T3/T4 = T7/T8)
    'T3': np.array([-1.00, 0.00, 0.00]),
    'T7': np.array([-1.00, 0.00, 0.00]),  # Alias
    # Central
    'C3': np.array([-0.71, 0.00, 0.71]),
    'Cz': np.array([0.00, 0.00, 1.00]),
    'C4': np.array([0.71, 0.00, 0.71]),
    # Temporal (T4)
    'T4': np.array([1.00, 0.00, 0.00]),
    'T8': np.array([1.00, 0.00, 0.00]),  # Alias
    # Temporal-Parietal (T5/T6 = P7/P8)
    'T5': np.array([-0.81, -0.59, 0.00]),
    'P7': np.array([-0.81, -0.59, 0.00]),  # Alias
    # Parietal
    'P3': np.array([-0.55, -0.67, 0.50]),
    'Pz': np.array([0.00, -0.72, 0.69]),
    'P4': np.array([0.55, -0.67, 0.50]),
    # Temporal-Parietal (T6)
    'T6': np.array([0.81, -0.59, 0.00]),
    'P8': np.array([0.81, -0.59, 0.00]),  # Alias
    # Occipital
    'O1': np.array([-0.31, -0.95, 0.00]),
    'O2': np.array([0.31, -0.95, 0.00]),
}


def load_biosemi128_positions(elc_path: str) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Load BioSemi 128 electrode positions from .ELC file.

    Args:
        elc_path: Path to biosemi128.ELC file

    Returns:
        Tuple of (positions dict, labels list)
        - positions: Dict mapping label to 3D coordinates
        - labels: List of electrode labels in order
    """
    positions = {}
    labels = []
    coords = []

    with open(elc_path, 'r') as f:
        lines = f.readlines()

    # Parse file
    in_positions = False
    in_labels = False

    for line in lines:
        line = line.strip()
        if line == 'Positions':
            in_positions = True
            continue
        elif line == 'Labels':
            in_positions = False
            in_labels = True
            continue

        if in_positions and line:
            parts = line.split()
            if len(parts) == 3:
                coords.append([float(x) for x in parts])
        elif in_labels and line:
            labels.append(line)

    # Match coords with labels
    for i, label in enumerate(labels):
        if i < len(coords):
            positions[label] = np.array(coords[i])

    return positions, labels


def normalize_positions(positions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Normalize electrode positions to unit sphere centered at origin.

    Args:
        positions: Dict mapping label to 3D coordinates

    Returns:
        Normalized positions dict
    """
    # Stack all positions
    all_pos = np.array(list(positions.values()))

    # Center at origin
    center = all_pos.mean(axis=0)
    centered = all_pos - center

    # Normalize to unit sphere
    max_dist = np.max(np.linalg.norm(centered, axis=1))
    normalized = centered / max_dist

    # Rebuild dict
    return {label: normalized[i] for i, label in enumerate(positions.keys())}


def create_biosemi128_to_1020_mapping(
    elc_path: str,
    target_channels: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Create mapping from standard 10-20 channels to nearest BioSemi 128 electrodes.

    Args:
        elc_path: Path to biosemi128.ELC file
        target_channels: List of target 10-20 channel names.
                        Defaults to STANDARD_1020_CHANNELS (19 channels)

    Returns:
        Dict mapping 10-20 channel name to nearest BioSemi electrode name
        e.g., {'Cz': 'A1', 'C3': 'B21', ...}
    """
    if target_channels is None:
        target_channels = STANDARD_1020_CHANNELS

    # Load BioSemi positions
    bio_positions, bio_labels = load_biosemi128_positions(elc_path)

    # Normalize BioSemi positions
    bio_normalized = normalize_positions(bio_positions)

    # Find nearest BioSemi electrode for each target channel
    mapping = {}
    for target_ch in target_channels:
        if target_ch not in STANDARD_1020_POSITIONS:
            # Try aliases
            if target_ch == 'T7':
                target_ch = 'T3'
            elif target_ch == 'T8':
                target_ch = 'T4'
            elif target_ch == 'P7':
                target_ch = 'T5'
            elif target_ch == 'P8':
                target_ch = 'T6'

        target_pos = STANDARD_1020_POSITIONS[target_ch]

        min_dist = float('inf')
        nearest_ch = None

        for bio_ch, bio_pos in bio_normalized.items():
            dist = np.linalg.norm(target_pos - bio_pos)
            if dist < min_dist:
                min_dist = dist
                nearest_ch = bio_ch

        mapping[target_ch] = nearest_ch

    return mapping


def get_channel_indices(
    mapping: Dict[str, str],
    all_labels: Optional[List[str]] = None
) -> Dict[str, int]:
    """
    Get channel indices for the mapped electrodes.

    Args:
        mapping: Dict mapping 10-20 names to BioSemi names
        all_labels: List of all electrode labels in order.
                   Defaults to BIOSEMI_128_LABELS

    Returns:
        Dict mapping 10-20 channel name to index in data array
    """
    if all_labels is None:
        all_labels = BIOSEMI_128_LABELS

    label_to_idx = {label: i for i, label in enumerate(all_labels)}

    indices = {}
    for ch_1020, ch_bio in mapping.items():
        if ch_bio in label_to_idx:
            indices[ch_1020] = label_to_idx[ch_bio]
        else:
            raise ValueError(f"BioSemi electrode {ch_bio} not found in labels")

    return indices


def get_motor_cortex_subset(elc_path: str, n_channels: int = 19) -> List[int]:
    """
    Get indices of electrodes near motor cortex (C3, Cz, C4 region).

    Alternative strategy (Plan B) when standard 10-20 mapping is insufficient.

    Args:
        elc_path: Path to biosemi128.ELC file
        n_channels: Number of channels to select

    Returns:
        List of channel indices
    """
    bio_positions, bio_labels = load_biosemi128_positions(elc_path)
    bio_normalized = normalize_positions(bio_positions)

    # Motor cortex center (approximate C3-Cz-C4 line)
    motor_center = np.array([0.0, 0.0, 0.9])  # Near vertex

    # Calculate distances to motor cortex center
    distances = []
    for label in BIOSEMI_128_LABELS:
        pos = bio_normalized[label]
        dist = np.linalg.norm(pos - motor_center)
        distances.append((dist, label))

    # Sort by distance and take closest n_channels
    distances.sort(key=lambda x: x[0])
    selected_labels = [d[1] for d in distances[:n_channels]]

    # Get indices
    label_to_idx = {label: i for i, label in enumerate(BIOSEMI_128_LABELS)}
    indices = [label_to_idx[label] for label in selected_labels]

    return sorted(indices)


def save_channel_mapping(mapping: Dict[str, str], output_path: str) -> None:
    """
    Save channel mapping to JSON file.

    Args:
        mapping: Channel mapping dict
        output_path: Output file path
    """
    import json

    # Add metadata
    output = {
        'description': 'BioSemi 128 to standard 10-20 channel mapping',
        'source': 'biosemi128.ELC',
        'target_system': 'Standard 10-20 (CBraMod pretrained channels)',
        'n_channels': len(mapping),
        'mapping': mapping
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)


if __name__ == '__main__':
    # Test the mapping
    import json

    elc_path = Path(__file__).parent.parent.parent / 'data' / 'biosemi128.ELC'

    if elc_path.exists():
        # Create mapping
        mapping = create_biosemi128_to_1020_mapping(str(elc_path))

        print("BioSemi 128 -> Standard 10-20 Mapping:")
        print("-" * 40)
        for ch_1020, ch_bio in mapping.items():
            print(f"  {ch_1020:4s} -> {ch_bio}")

        # Get indices
        indices = get_channel_indices(mapping)
        print("\nChannel Indices:")
        print("-" * 40)
        for ch_1020, idx in indices.items():
            print(f"  {ch_1020:4s} -> index {idx:3d}")

        # Save mapping
        output_path = elc_path.parent / 'channel_mapping.json'
        save_channel_mapping(mapping, str(output_path))
        print(f"\nMapping saved to: {output_path}")
    else:
        print(f"ELC file not found: {elc_path}")
