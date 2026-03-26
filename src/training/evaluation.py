"""
Evaluation utilities for EEG-BCI training.

This module provides evaluation functions for trained models:
- majority_vote_accuracy: Compute accuracy using majority voting over segments per trial
- majority_vote_accuracy_unified: Same, but with logit masking for unified models
- unified_model_evaluate: Evaluate a unified 4-class model on all subtasks
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ..preprocessing.data_loader import FingerEEGDataset, PreprocessConfig
from ..preprocessing.discovery import get_session_folders_for_split
from ..config.constants import DEFAULT_CACHE_INDEX_PATH, TASKS
from ..utils.logging import SectionLogger

logger = logging.getLogger(__name__)
log_eval = SectionLogger(logger, 'eval')


def _get_amp_dtype(device: torch.device) -> torch.dtype:
    """Return the best AMP dtype for the device (BF16 if supported, else FP16)."""
    if device.type == 'cuda' and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_trial_grouping(
    dataset: FingerEEGDataset,
    indices: List[int],
) -> Dict[int, List[int]]:
    """Pre-compute trial-to-segment-position mapping for majority voting.

    Returns a dict mapping trial_idx -> list of positions in ``indices``.
    This can be built once and reused across epochs to avoid repeated
    O(n) Python iteration.

    Args:
        dataset: FingerEEGDataset (with trial_infos)
        indices: Segment indices to group

    Returns:
        Dict mapping trial_idx -> list of positions (0-based into ``indices``)
    """
    trial_to_positions: Dict[int, List[int]] = {}
    for pos, idx in enumerate(indices):
        trial_idx = dataset.trial_infos[idx].trial_idx
        if trial_idx not in trial_to_positions:
            trial_to_positions[trial_idx] = []
        trial_to_positions[trial_idx].append(pos)
    return trial_to_positions


def majority_vote_from_predictions(
    segment_preds,
    segment_labels,
    trial_grouping: Dict[int, List[int]],
) -> Tuple[float, Dict]:
    """Compute majority-vote accuracy from pre-computed predictions.

    This is a lightweight CPU-only function that takes predictions already
    produced by a forward pass (e.g. from ``validate()``) and groups them
    by trial for majority voting.  No model inference is performed.

    Args:
        segment_preds: Array-like of per-segment predicted class indices
        segment_labels: Array-like of per-segment true labels
        trial_grouping: Pre-computed mapping from trial_idx -> list of
            positions into segment_preds/segment_labels (from
            ``build_trial_grouping()``)

    Returns:
        Tuple of (accuracy, detailed_results)
    """
    correct = 0
    total = 0
    results = {'per_trial': []}

    for trial_idx in sorted(trial_grouping.keys()):
        positions = trial_grouping[trial_idx]
        preds = [segment_preds[p] for p in positions]
        true_label = segment_labels[positions[0]]

        counter = Counter(preds)
        majority_pred = counter.most_common(1)[0][0]

        is_correct = int(majority_pred == true_label)
        correct += is_correct
        total += 1

        results['per_trial'].append({
            'trial_idx': trial_idx,
            'n_segments': len(preds),
            'predictions': [int(p) for p in preds],
            'majority_pred': int(majority_pred),
            'true_label': int(true_label),
            'correct': is_correct,
        })

    accuracy = correct / total if total > 0 else 0.0
    results['accuracy'] = accuracy
    results['correct'] = correct
    results['total'] = total

    return accuracy, results


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
    amp_dtype = _get_amp_dtype(device) if use_amp else torch.float16

    with torch.no_grad():
        for segments, labels in loader:
            segments = segments.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
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
            'predictions': [int(p) for p in preds],
            'majority_pred': int(majority_pred),
            'true_label': int(true_label),
            'correct': is_correct,
        })

    accuracy = correct / total if total > 0 else 0.0
    results['accuracy'] = accuracy
    results['correct'] = correct
    results['total'] = total

    return accuracy, results


# Unified class mapping: finger ID -> 0-indexed position in the 4-class output
UNIFIED_CLASS_MAP = {1: 0, 2: 1, 3: 2, 4: 3}


def compute_subtask_val_groups(
    dataset: FingerEEGDataset,
    val_indices: List[int],
) -> Dict[str, Dict]:
    """
    Group validation indices by subtask for unified mode.

    Uses ``session_type`` from trial_infos to classify each val segment
    as binary (2class), ternary (3class), or quaternary (offline).

    Args:
        dataset: FingerEEGDataset with trial_infos
        val_indices: Segment indices assigned to validation

    Returns:
        Dict mapping subtask name -> {'indices': List[int],
        'active_class_indices': List[int]}.  Only subtasks with
        non-empty indices are included.
    """
    groups: Dict[str, List[int]] = {'binary': [], 'ternary': [], 'quaternary': []}

    for idx in val_indices:
        session_type = dataset.trial_infos[idx].session_type.lower()
        if 'offline' in session_type:
            groups['quaternary'].append(idx)
        elif '2class' in session_type:
            groups['binary'].append(idx)
        elif '3class' in session_type:
            groups['ternary'].append(idx)

    result = {}
    for subtask, indices in groups.items():
        if indices:
            active = [UNIFIED_CLASS_MAP[c] for c in TASKS[subtask]['classes']]
            result[subtask] = {
                'indices': indices,
                'active_class_indices': active,
            }

    log_eval.info(
        f"Unified val groups: "
        + ", ".join(f"{k}={len(v['indices'])}" for k, v in result.items())
    )
    return result


def majority_vote_accuracy_unified(
    model: nn.Module,
    dataset: FingerEEGDataset,
    indices: List[int],
    device: torch.device,
    active_class_indices: List[int],
    batch_size: int = 128,
    use_amp: bool = True,
    label_remap: Optional[Dict[int, int]] = None,
) -> Tuple[float, Dict]:
    """
    Compute accuracy using majority voting with logit masking for unified models.

    Evaluates a 4-class unified model on a subtask by selecting only the
    relevant logit columns before argmax.  Predictions are in **local** index
    space (0 … len(active_class_indices)-1).  When the dataset labels are
    also in local space (separate test dataset) no remapping is needed.
    When labels are in unified 4-class space (validation on the training
    dataset), pass ``label_remap`` to convert them to local space.

    Args:
        model: Trained unified model (4-class output)
        dataset: FingerEEGDataset for the subtask's test set
        indices: Indices of segments to evaluate
        device: Device to use
        active_class_indices: Indices of active logit columns in the 4-class output.
            E.g., [0, 3] for binary (Thumb=0, Pinky=3), [0, 1, 3] for ternary.
        batch_size: Batch size for evaluation
        use_amp: Whether to use automatic mixed precision
        label_remap: Optional mapping from dataset label space to local
            prediction space.  E.g. ``{0: 0, 3: 1}`` for binary when
            labels are in unified [0,1,2,3] space.  ``None`` (default)
            means labels are already in local space.

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

    trial_predictions = {}
    trial_labels = {}

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, pin_memory=True)

    segment_preds = []
    segment_labels = []

    use_amp = use_amp and device.type == 'cuda'
    amp_dtype = _get_amp_dtype(device) if use_amp else torch.float16
    active_indices_tensor = torch.tensor(active_class_indices, device=device)

    with torch.no_grad():
        for segments, labels in loader:
            segments = segments.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    outputs = model(segments)
            else:
                outputs = model(segments)
            # Logit masking: select only active class columns
            active_logits = outputs[:, active_indices_tensor]
            preds = active_logits.argmax(dim=1).cpu().numpy()
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
        if label_remap is not None:
            true_label = label_remap.get(int(true_label), true_label)

        counter = Counter(preds)
        majority_pred = counter.most_common(1)[0][0]

        is_correct = int(majority_pred == true_label)
        correct += is_correct
        total += 1

        results['per_trial'].append({
            'trial_idx': trial_idx,
            'n_segments': len(preds),
            'predictions': [int(p) for p in preds],
            'majority_pred': int(majority_pred),
            'true_label': int(true_label),
            'correct': is_correct,
        })

    accuracy = correct / total if total > 0 else 0.0
    results['accuracy'] = accuracy
    results['correct'] = correct
    results['total'] = total

    return accuracy, results


def unified_model_evaluate(
    model: nn.Module,
    data_root: Path,
    subject_ids: List[str],
    config: PreprocessConfig,
    elc_path: Path,
    paradigm: str,
    device: torch.device,
    cache_only: bool = False,
    train_dataset=None,
    offline_test_indices: Optional[List[int]] = None,
) -> Dict:
    """
    Evaluate a unified 4-class model on all subtasks (binary, ternary, quaternary).

    For binary/ternary: loads each subtask's test dataset (Sess02 Finetune)
    and applies logit masking.
    For quaternary: uses pre-computed ``offline_test_indices`` into the
    training dataset (the held-out portion of OfflineImagery from the
    three-way temporal split).

    Args:
        model: Trained unified model (4-class output)
        data_root: Path to data directory
        subject_ids: List of subject IDs to evaluate
        config: Preprocessing configuration
        elc_path: Path to electrode location file
        paradigm: 'imagery' or 'movement'
        device: Device to use
        cache_only: If True, load exclusively from cache index
        train_dataset: The unified training FingerEEGDataset (needed for
            quaternary eval via ``offline_test_indices``)
        offline_test_indices: Segment indices into ``train_dataset`` that
            were held out from training for quaternary evaluation

    Returns:
        Dict with per-subtask results and 'mean_accuracy'
    """
    subtasks = ['binary', 'ternary', 'quaternary']
    subtask_results = {}
    accuracies = []

    for subtask in subtasks:
        task_config = TASKS[subtask]
        subtask_classes = task_config['classes']

        # Quaternary: evaluate on held-out offline test split
        if subtask == 'quaternary':
            if train_dataset is None or not offline_test_indices:
                log_eval.debug("Unified eval: no offline test indices for quaternary")
                subtask_results[subtask] = {
                    'accuracy': 0.0,
                    'n_trials': 0,
                    'n_segments': 0,
                }
                continue

            active_class_indices = [UNIFIED_CLASS_MAP[c] for c in subtask_classes]
            acc, detailed = majority_vote_accuracy_unified(
                model, train_dataset, offline_test_indices, device,
                active_class_indices=active_class_indices,
            )
            n_trials = len(set(
                train_dataset.trial_infos[i].trial_idx for i in offline_test_indices
            ))
            subtask_results[subtask] = {
                'accuracy': acc,
                'n_trials': n_trials,
                'n_segments': len(offline_test_indices),
                'detailed_results': detailed,
            }
            accuracies.append(acc)
            log_eval.info(f"Unified eval [{subtask}]: {acc:.2%} ({n_trials} trials, held-out offline)")
            continue

        # Binary / ternary: load separate test dataset (Sess02 Finetune)
        test_folders = get_session_folders_for_split(paradigm, subtask, 'test')
        test_ds = FingerEEGDataset(
            str(data_root),
            subject_ids,
            config,
            session_folders=test_folders,
            target_classes=subtask_classes,
            elc_path=str(elc_path),
            cache_only=cache_only,
            reject_trials=False,
        )

        if len(test_ds) == 0:
            log_eval.debug(f"Unified eval: no test data for {subtask}")
            subtask_results[subtask] = {
                'accuracy': 0.0,
                'n_trials': 0,
                'n_segments': 0,
            }
            continue

        # Compute active class indices in the unified 4-class output
        active_class_indices = [UNIFIED_CLASS_MAP[c] for c in subtask_classes]

        test_indices = list(range(len(test_ds)))
        acc, detailed = majority_vote_accuracy_unified(
            model, test_ds, test_indices, device,
            active_class_indices=active_class_indices,
        )

        n_trials = len(test_ds.get_unique_trials())
        subtask_results[subtask] = {
            'accuracy': acc,
            'n_trials': n_trials,
            'n_segments': len(test_ds),
            'detailed_results': detailed,
        }
        accuracies.append(acc)
        log_eval.info(f"Unified eval [{subtask}]: {acc:.2%} ({n_trials} trials)")

    mean_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
    subtask_results['mean_accuracy'] = mean_acc

    return subtask_results
