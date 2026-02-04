"""
Evaluation utilities for EEG-BCI training.

This module provides evaluation functions for trained models:
- majority_vote_accuracy: Compute accuracy using majority voting over segments per trial
"""

import logging
from collections import Counter
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ..preprocessing.data_loader import FingerEEGDataset
from ..utils.logging import SectionLogger

logger = logging.getLogger(__name__)
log_eval = SectionLogger(logger, 'eval')


def majority_vote_accuracy(
    model: nn.Module,
    dataset: FingerEEGDataset,
    indices: List[int],
    device: torch.device,
    batch_size: int = 128,
    use_amp: bool = True,
) -> Tuple[float, Dict]:
    """
    Compute accuracy using majority voting over segments per trial.

    This follows the paper's evaluation methodology:
    - Each trial has multiple segment predictions
    - Final trial prediction = majority vote

    Args:
        model: Trained model
        dataset: FingerEEGDataset (with trial_infos)
        indices: Indices of segments to evaluate
        device: Device to use
        batch_size: Batch size for evaluation (increased default for speed)
        use_amp: Whether to use automatic mixed precision

    Returns:
        Tuple of (accuracy, detailed_results)
    """
    model.eval()

    # Group segments by original trial
    trial_to_segments = {}
    for idx in indices:
        trial_idx = dataset.trial_infos[idx].trial_idx
        if trial_idx not in trial_to_segments:
            trial_to_segments[trial_idx] = []
        trial_to_segments[trial_idx].append(idx)

    # Collect predictions per trial
    trial_predictions = {}
    trial_labels = {}

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, pin_memory=True)

    segment_preds = []
    segment_labels = []

    use_amp = use_amp and device.type == 'cuda'

    with torch.no_grad():
        for segments, labels in loader:
            segments = segments.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = model(segments)
            else:
                outputs = model(segments)
            preds = outputs.argmax(dim=1).cpu().numpy()
            segment_preds.extend(preds)
            segment_labels.extend(labels.numpy())

    # Map predictions back to trials
    for i, idx in enumerate(indices):
        trial_idx = dataset.trial_infos[idx].trial_idx
        if trial_idx not in trial_predictions:
            trial_predictions[trial_idx] = []
            trial_labels[trial_idx] = segment_labels[i]
        trial_predictions[trial_idx].append(segment_preds[i])

    # Majority voting
    correct = 0
    total = 0
    results = {'per_trial': []}

    for trial_idx in sorted(trial_predictions.keys()):
        preds = trial_predictions[trial_idx]
        true_label = trial_labels[trial_idx]

        # Majority vote
        counter = Counter(preds)
        majority_pred = counter.most_common(1)[0][0]

        is_correct = int(majority_pred == true_label)
        correct += is_correct
        total += 1

        results['per_trial'].append({
            'trial_idx': trial_idx,
            'n_segments': len(preds),
            'predictions': preds,
            'majority_pred': int(majority_pred),
            'true_label': int(true_label),
            'correct': is_correct,
        })

    accuracy = correct / total if total > 0 else 0.0
    results['accuracy'] = accuracy
    results['correct'] = correct
    results['total'] = total

    return accuracy, results
