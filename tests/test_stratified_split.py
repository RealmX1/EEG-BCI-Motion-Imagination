"""
Test stratified temporal split to ensure train/val have similar session distributions.

Run with: uv run pytest tests/test_stratified_split.py -v
"""

import sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.data_loader import (
    FingerEEGDataset,
    PreprocessConfig,
    get_session_folders_for_split,
)


def stratified_temporal_split(dataset: FingerEEGDataset, val_ratio: float = 0.2):
    """
    Perform stratified temporal split: split within each session separately.

    This ensures train and val have similar session distributions.
    """
    # Group trials by session
    session_to_trials = defaultdict(list)

    unique_trials = dataset.get_unique_trials()

    for trial_idx in unique_trials:
        # Find which session this trial belongs to
        for seg_idx, info in enumerate(dataset.trial_infos):
            if info.trial_idx == trial_idx:
                session_to_trials[info.session_type].append(trial_idx)
                break

    # For each session, split temporally
    train_trials = []
    val_trials = []

    for session_type, trials in session_to_trials.items():
        # Sort trials by index (chronological order)
        trials = sorted(set(trials))

        n_trials = len(trials)
        n_val = max(1, int(n_trials * val_ratio))
        n_train = n_trials - n_val

        # Temporal split within this session
        train_trials.extend(trials[:n_train])
        val_trials.extend(trials[n_train:])

        print(f"{session_type}:")
        print(f"  Total: {n_trials} trials")
        print(f"  Train: {n_train} trials (first {(1-val_ratio)*100:.0f}%)")
        print(f"  Val: {n_val} trials (last {val_ratio*100:.0f}%)")

    return train_trials, val_trials


def test_stratified_split(subject_id: str = 'S01'):
    """Test the stratified temporal split."""

    print("=" * 80)
    print("TESTING STRATIFIED TEMPORAL SPLIT")
    print("=" * 80)

    data_root = PROJECT_ROOT / 'data'
    elc_path = data_root / 'biosemi128.ELC'

    config = PreprocessConfig.paper_aligned(n_class=2)
    train_folders = get_session_folders_for_split('imagery', 'binary', 'train')

    dataset = FingerEEGDataset(
        str(data_root),
        [subject_id],
        config,
        session_folders=train_folders,
        target_classes=[1, 4],
        elc_path=str(elc_path),
        use_cache=True,
    )

    print(f"\nDataset loaded: {len(dataset)} segments")

    # Original temporal split (global)
    print(f"\n[ORIGINAL GLOBAL TEMPORAL SPLIT]")
    unique_trials = dataset.get_unique_trials()
    n_val = max(1, int(len(unique_trials) * 0.2))
    original_train_trials = unique_trials[:-n_val]
    original_val_trials = unique_trials[-n_val:]

    original_train_indices = dataset.get_segment_indices_for_trials(original_train_trials)
    original_val_indices = dataset.get_segment_indices_for_trials(original_val_trials)

    # Check session distribution
    original_val_sessions = Counter()
    for idx in original_val_indices:
        info = dataset.trial_infos[idx]
        original_val_sessions[info.session_type] += 1

    print(f"Validation session distribution:")
    for session, count in sorted(original_val_sessions.items()):
        pct = count / len(original_val_indices) * 100
        print(f"  {session}: {count} ({pct:.1f}%)")

    # Stratified temporal split
    print(f"\n[STRATIFIED TEMPORAL SPLIT]")
    stratified_train_trials, stratified_val_trials = stratified_temporal_split(dataset, val_ratio=0.2)

    stratified_train_indices = dataset.get_segment_indices_for_trials(stratified_train_trials)
    stratified_val_indices = dataset.get_segment_indices_for_trials(stratified_val_trials)

    # Check session distribution
    stratified_train_sessions = Counter()
    for idx in stratified_train_indices[:2000]:  # Sample
        info = dataset.trial_infos[idx]
        stratified_train_sessions[info.session_type] += 1

    stratified_val_sessions = Counter()
    for idx in stratified_val_indices:
        info = dataset.trial_infos[idx]
        stratified_val_sessions[info.session_type] += 1

    print(f"\nTrain session distribution (sampled 2000):")
    for session, count in sorted(stratified_train_sessions.items()):
        pct = count / len(stratified_train_sessions) * 100
        print(f"  {session}: {count} ({pct:.1f}%)")

    print(f"\nValidation session distribution:")
    for session, count in sorted(stratified_val_sessions.items()):
        pct = count / len(stratified_val_indices) * 100
        print(f"  {session}: {count} ({pct:.1f}%)")

    print(f"\n[COMPARISON]")
    print(f"Train: {len(stratified_train_trials)} trials ({len(stratified_train_indices)} segments)")
    print(f"Val: {len(stratified_val_trials)} trials ({len(stratified_val_indices)} segments)")

    print("\n" + "=" * 80)
    print("Result: Train and val now have similar session distributions!")
    print("=" * 80)


if __name__ == '__main__':
    test_stratified_split('S01')
