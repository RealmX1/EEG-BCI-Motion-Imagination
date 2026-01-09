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
