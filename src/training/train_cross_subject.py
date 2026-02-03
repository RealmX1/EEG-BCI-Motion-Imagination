"""
Cross-subject pretraining module for FINGER-EEG-BCI.

Trains a model on data from multiple subjects jointly, then evaluates
on each subject's held-out test set (Session 2 Finetune).

Data Split (consistent with within-subject protocol):
- For each subject:
    - Training data: Offline + Sess01 + Sess02 Base
    - Temporal split: First 80% trials -> global train, Last 20% -> global val
- Test: Each subject's Sess02 Finetune (evaluated separately)

Usage:
    from src.training.train_cross_subject import train_cross_subject

    # Train on all subjects
    results = train_cross_subject(
        subjects=['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07'],
        model_type='eegnet',
        task='binary',
    )

    # Access pretrained model path
    pretrained_path = results['model_path']
"""

import logging
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, ConcatDataset

from src.models.eegnet import EEGNet
from src.models.cbramod_adapter import (
    CBraModForFingerBCI,
    get_default_pretrained_path,
)
from src.preprocessing.data_loader import (
    FingerEEGDataset,
    PreprocessConfig,
    get_session_folders_for_split,
    discover_available_subjects,
)
from src.training.train_within_subject import (
    WithinSubjectTrainer,
    majority_vote_accuracy,
    get_default_config,
    create_data_loaders_from_dataset,
)
from src.utils.device import get_device, set_seed
from src.utils.logging import SectionLogger
from src.utils.timing import Timer, print_section_header, print_metric, colored, Colors

logger = logging.getLogger(__name__)
log_data = SectionLogger(logger, 'data')
log_model = SectionLogger(logger, 'model')
log_train = SectionLogger(logger, 'train')


def load_multi_subject_data(
    data_root: Path,
    subjects: List[str],
    config: PreprocessConfig,
    target_classes: List[int],
    paradigm: str,
    task: str,
    elc_path: Path,
) -> Tuple[FingerEEGDataset, Dict[str, FingerEEGDataset]]:
    """
    Load data for multiple subjects.

    Args:
        data_root: Path to data directory
        subjects: List of subject IDs
        config: Preprocessing configuration
        target_classes: Target classes for the task
        paradigm: 'imagery' or 'movement'
        task: 'binary', 'ternary', or 'quaternary'
        elc_path: Path to electrode location file

    Returns:
        Tuple of (train_dataset, test_datasets_by_subject)
        - train_dataset: Combined training data from all subjects
        - test_datasets_by_subject: Dict mapping subject_id -> test dataset
    """
    # Get session folders
    train_folders = get_session_folders_for_split(paradigm, task, 'train')
    test_folders = get_session_folders_for_split(paradigm, task, 'test')

    log_data.info(f"Train folders: {train_folders}")
    log_data.info(f"Test folders: {test_folders}")

    # Load training data for all subjects together
    train_dataset = FingerEEGDataset(
        str(data_root),
        subjects,
        config,
        session_folders=train_folders,
        target_classes=target_classes,
        elc_path=str(elc_path),
    )

    # Load test data for each subject separately (for per-subject evaluation)
    test_datasets = {}
    for subject_id in subjects:
        test_ds = FingerEEGDataset(
            str(data_root),
            [subject_id],
            config,
            session_folders=test_folders,
            target_classes=target_classes,
            elc_path=str(elc_path),
        )
        if len(test_ds) > 0:
            test_datasets[subject_id] = test_ds

    return train_dataset, test_datasets


def temporal_split_cross_subject(
    dataset: FingerEEGDataset,
    val_ratio: float = 0.2,
) -> Tuple[List[int], List[int]]:
    """
    Perform temporal split on cross-subject dataset.

    For each subject's data within the dataset:
    - Sort trials chronologically
    - First (1-val_ratio) trials -> training
    - Last val_ratio trials -> validation

    This ensures temporal ordering is preserved within each subject,
    preventing data leakage from future trials to past.

    Args:
        dataset: Combined multi-subject dataset
        val_ratio: Fraction of each subject's trials for validation

    Returns:
        Tuple of (train_indices, val_indices)
    """
    # Group trials by subject
    subject_to_trials = defaultdict(set)
    for idx, info in enumerate(dataset.trial_infos):
        subject_to_trials[info.subject_id].add(info.trial_idx)

    train_trials = []
    val_trials = []

    for subject_id, trials in subject_to_trials.items():
        # Sort trials chronologically
        sorted_trials = sorted(trials)
        n_trials = len(sorted_trials)
        n_val = max(1, int(n_trials * val_ratio))

        # Temporal split
        train_trials.extend(sorted_trials[:-n_val])
        val_trials.extend(sorted_trials[-n_val:])

    # Convert trial indices to segment indices
    train_indices = dataset.get_segment_indices_for_trials(train_trials)
    val_indices = dataset.get_segment_indices_for_trials(val_trials)

    return train_indices, val_indices


def create_cross_subject_model(
    model_type: str,
    n_channels: int,
    n_samples: int,
    n_classes: int,
    config: dict,
) -> nn.Module:
    """Create model for cross-subject training."""
    model_config = config['model']

    if model_type == 'cbramod':
        n_patches = n_samples // 200  # 200 samples per patch @ 200Hz
        pretrained_path = get_default_pretrained_path()

        model = CBraModForFingerBCI(
            n_channels=n_channels,
            n_patches=n_patches,
            n_classes=n_classes,
            pretrained_path=pretrained_path,
            freeze_backbone=model_config.get('freeze_backbone', False),
            classifier_type=model_config.get('classifier_type', 'two_layer'),
            dropout=model_config.get('dropout_rate', 0.1),
        )
    else:
        model = EEGNet(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=n_classes,
            F1=model_config['F1'],
            D=model_config['D'],
            F2=model_config['F2'],
            kernel_length=model_config['kernel_length'],
            dropout_rate=model_config['dropout_rate'],
        )

    return model


def train_cross_subject(
    subjects: List[str],
    model_type: str,
    task: str = 'binary',
    paradigm: str = 'imagery',
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    save_dir: str = 'checkpoints/cross_subject',
    data_root: str = 'data',
    device: Optional[torch.device] = None,
    seed: int = 42,
    wandb_enabled: bool = False,
) -> Dict:
    """
    Cross-subject pretraining.

    Trains a single model on combined data from all subjects, using
    the same temporal split protocol as within-subject training.

    Args:
        subjects: List of subject IDs (e.g., ['S01', 'S02', ...])
        model_type: 'eegnet' or 'cbramod'
        task: 'binary', 'ternary', or 'quaternary'
        paradigm: 'imagery' (MI) or 'movement' (ME)
        epochs: Number of training epochs (None = use default)
        batch_size: Batch size (None = use default)
        save_dir: Directory to save pretrained model
        data_root: Path to data directory
        device: Device to use (None = auto-detect)
        seed: Random seed
        wandb_enabled: Enable WandB logging

    Returns:
        Dict with:
        - model_path: Path to saved pretrained model
        - per_subject_test_acc: Dict mapping subject_id -> test accuracy
        - val_acc: Best validation accuracy
        - training_time: Total training time
        - history: Training history
    """
    total_start = time.perf_counter()
    Timer.reset()
    set_seed(seed)

    if device is None:
        device = get_device()

    print()
    print(colored("=" * 70, Colors.BRIGHT_BLUE, bold=True))
    print(colored(f"  Cross-Subject Pretraining: {model_type.upper()}", Colors.BRIGHT_BLUE, bold=True))
    print(colored(f"  Subjects: {', '.join(subjects)}", Colors.BRIGHT_BLUE))
    print(colored("=" * 70, Colors.BRIGHT_BLUE, bold=True))

    # Get config
    config = get_default_config(model_type, task)

    # Override epochs and batch_size if specified
    if epochs is not None:
        config['training']['epochs'] = epochs
    if batch_size is not None:
        config['training']['batch_size'] = batch_size

    # For cross-subject, use larger batch size and potentially more epochs
    if batch_size is None:
        if model_type == 'cbramod':
            config['training']['batch_size'] = 256  # Larger for cross-subject
        else:
            config['training']['batch_size'] = 128

    if epochs is None:
        if model_type == 'cbramod':
            config['training']['epochs'] = 30
        else:
            config['training']['epochs'] = 50

    # Task configuration
    task_config = config['tasks'][task]
    target_classes = task_config['classes']
    n_classes = task_config['n_classes']

    # Setup paths
    data_root_path = Path(data_root)
    elc_path = data_root_path / 'biosemi128.ELC'
    save_path = Path(save_dir) / f'{model_type}_{paradigm}_{task}'
    save_path.mkdir(parents=True, exist_ok=True)

    # Preprocessing config
    if model_type == 'cbramod':
        preprocess_config = PreprocessConfig.for_cbramod(full_channels=True)
    else:
        preprocess_config = PreprocessConfig.paper_aligned(n_class=n_classes)

    # ========== DATA LOADING ==========
    print_section_header("Data Loading (Cross-Subject)")
    print(colored(f"  Subjects: {subjects}", Colors.CYAN))

    with Timer("data_loading", print_on_exit=True):
        train_dataset, test_datasets = load_multi_subject_data(
            data_root_path,
            subjects,
            preprocess_config,
            target_classes,
            paradigm,
            task,
            elc_path,
        )

    print_metric("Total train segments", len(train_dataset), Colors.CYAN)
    print_metric("Subjects with test data", len(test_datasets), Colors.MAGENTA)

    # ========== TEMPORAL SPLIT ==========
    print_section_header("Data Splitting (Temporal per Subject)")

    with Timer("data_splitting", print_on_exit=True):
        train_indices, val_indices = temporal_split_cross_subject(train_dataset)

    print_metric("Train segments", len(train_indices), Colors.GREEN)
    print_metric("Val segments", len(val_indices), Colors.YELLOW)

    # ========== DATALOADER CREATION ==========
    print_section_header("DataLoader Creation")

    with Timer("dataloader_creation", print_on_exit=True):
        train_loader, val_loader = create_data_loaders_from_dataset(
            train_dataset,
            train_indices,
            val_indices,
            batch_size=config['training']['batch_size'],
            num_workers=0,
            shuffle_train=True,
        )

    # Get input dimensions
    sample_segment, _ = train_dataset[0]
    n_channels = sample_segment.shape[0]
    n_samples = sample_segment.shape[1]

    print_metric("Batch size", config['training']['batch_size'], Colors.CYAN)
    print_metric("Input shape", f"[{n_channels}, {n_samples}]", Colors.CYAN)
    print_metric("Train batches", len(train_loader), Colors.GREEN)

    # ========== MODEL CREATION ==========
    print_section_header("Model Creation")

    with Timer("model_creation", print_on_exit=True):
        model = create_cross_subject_model(
            model_type,
            n_channels,
            n_samples,
            n_classes,
            config,
        )

    print_metric("Model", model_type.upper(), Colors.CYAN)
    print_metric("Parameters", f"{model.count_parameters():,}", Colors.CYAN)
    print_metric("Device", str(device), Colors.GREEN)

    # Enable TF32 for faster training
    if device.type == 'cuda' and hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')

    # ========== TRAINER SETUP ==========
    train_config = config['training']
    learning_rate = train_config.get('learning_rate', 1e-3)
    weight_decay = train_config.get('weight_decay', 0.0)
    scheduler_type = train_config.get('scheduler', None)

    if model_type == 'cbramod':
        learning_rate = train_config.get('backbone_lr', 1e-4)
        weight_decay = train_config.get('weight_decay', 0.05)
        if scheduler_type is None:
            scheduler_type = 'wsd'

    trainer = WithinSubjectTrainer(
        model,
        train_dataset,
        val_indices,
        device,
        model_type=model_type,
        n_classes=n_classes,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        scheduler_type=scheduler_type,
        use_amp=True,
        gradient_clip=1.0 if model_type == 'cbramod' else 0.0,
    )

    # ========== TRAINING ==========
    with Timer("training"):
        history = trainer.train(
            train_loader,
            val_loader,
            epochs=config['training']['epochs'],
            patience=config['training'].get('patience', 10),
            save_path=save_path,
            wandb_callback=None,  # TODO: Add WandB support
        )

    # ========== PER-SUBJECT TEST EVALUATION ==========
    print_section_header("Per-Subject Test Evaluation")

    per_subject_test_acc = {}
    for subject_id, test_dataset in test_datasets.items():
        test_indices = list(range(len(test_dataset)))
        test_acc, _ = majority_vote_accuracy(
            model, test_dataset, test_indices, device
        )
        per_subject_test_acc[subject_id] = test_acc

        acc_color = Colors.BRIGHT_GREEN if test_acc > 0.7 else (
            Colors.YELLOW if test_acc > 0.5 else Colors.RED
        )
        print(f"  {subject_id}: {colored(f'{test_acc:.2%}', acc_color)}")

    # Overall test accuracy (mean across subjects)
    mean_test_acc = np.mean(list(per_subject_test_acc.values()))
    std_test_acc = np.std(list(per_subject_test_acc.values()))
    print(f"\n  {colored('Mean Test Accuracy:', Colors.WHITE, bold=True)} "
          f"{colored(f'{mean_test_acc:.2%} +/- {std_test_acc:.2%}', Colors.BRIGHT_GREEN, bold=True)}")

    # ========== SAVE MODEL AND CONFIG ==========
    total_time = time.perf_counter() - total_start

    # Save final checkpoint with full metadata
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'model_type': model_type,
            'n_channels': n_channels,
            'n_samples': n_samples,
            'n_classes': n_classes,
        },
        'training_config': {
            'subjects': subjects,
            'paradigm': paradigm,
            'task': task,
            'epochs': config['training']['epochs'],
            'batch_size': config['training']['batch_size'],
        },
        'epoch': trainer.best_epoch,
        'val_acc': trainer.best_val_acc,
        'val_majority_acc': trainer.best_majority_acc,
        'per_subject_test_acc': per_subject_test_acc,
        'mean_test_acc': mean_test_acc,
    }

    # Save to best.pt (overwrite the one from training with full metadata)
    torch.save(checkpoint, save_path / 'best.pt')
    log_train.info(f"Model saved: {save_path / 'best.pt'}")

    # Save config JSON
    config_to_save = {
        'model_type': model_type,
        'task': task,
        'paradigm': paradigm,
        'subjects': subjects,
        'n_channels': n_channels,
        'n_samples': n_samples,
        'n_classes': n_classes,
        'training_config': config['training'],
        'model_config': config['model'],
        'per_subject_test_acc': per_subject_test_acc,
        'mean_test_acc': mean_test_acc,
        'std_test_acc': std_test_acc,
        'best_val_acc': trainer.best_val_acc,
        'best_epoch': trainer.best_epoch,
        'training_time': total_time,
    }

    with open(save_path / 'config.json', 'w') as f:
        json.dump(config_to_save, f, indent=2)

    # Save training history
    with open(save_path / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    Timer.print_summary("Cross-Subject Training")

    return {
        'model_path': str(save_path / 'best.pt'),
        'per_subject_test_acc': per_subject_test_acc,
        'mean_test_acc': mean_test_acc,
        'std_test_acc': std_test_acc,
        'val_acc': trainer.best_val_acc,
        'val_majority_acc': trainer.best_majority_acc,
        'best_epoch': trainer.best_epoch,
        'training_time': total_time,
        'history': history,
    }


if __name__ == '__main__':
    import sys

    logging.basicConfig(level=logging.INFO)

    # Quick test with available subjects
    from src.preprocessing.data_loader import discover_available_subjects

    data_root = Path(__file__).parent.parent.parent / 'data'
    subjects = discover_available_subjects(str(data_root), 'imagery', 'binary')

    if not subjects:
        print("No subjects found")
        sys.exit(1)

    print(f"Found subjects: {subjects}")

    # Test cross-subject training with EEGNet (faster)
    results = train_cross_subject(
        subjects=subjects[:3],  # Use first 3 subjects for quick test
        model_type='eegnet',
        task='binary',
        epochs=5,  # Quick test
        data_root=str(data_root),
    )

    print("\nResults:")
    print(f"  Model path: {results['model_path']}")
    print(f"  Mean test acc: {results['mean_test_acc']:.2%}")
    print(f"  Per-subject: {results['per_subject_test_acc']}")
