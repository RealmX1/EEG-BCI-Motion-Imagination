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

    # With WandB logging
    results = train_cross_subject(
        subjects=['S01', 'S02', 'S03'],
        model_type='cbramod',
        task='binary',
        wandb_enabled=True,
        wandb_project='eeg-bci',
    )
"""

import logging
import json
import time
from datetime import datetime
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
    create_data_loaders_from_dataset,
)
from src.training.common import (
    setup_performance_optimizations,
    maybe_compile_model,
    get_scheduler_config_from_preset,
    create_two_phase_loaders,
    apply_config_overrides,
)
from src.config.training import SCHEDULER_PRESETS, get_cross_subject_config
from src.utils.device import get_device, set_seed
from src.utils.logging import SectionLogger
from src.utils.timing import Timer, print_section_header, print_metric, colored, Colors
from src.utils.wandb_logger import (
    create_wandb_logger,
    WandbCallback,
)

logger = logging.getLogger(__name__)
log_data = SectionLogger(logger, 'data')
log_model = SectionLogger(logger, 'model')
log_train = SectionLogger(logger, 'train')



# Cross-subject defaults are now centralized in src/config/training.py
# via get_cross_subject_config(). See that module for full parameter documentation.


def load_multi_subject_data(
    data_root: Path,
    subjects: List[str],
    config: PreprocessConfig,
    target_classes: List[int],
    paradigm: str,
    task: str,
    elc_path: Path,
    cache_only: bool = False,
    cache_index_path: str = ".cache_index.json",
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
        cache_only: If True, load exclusively from cache index
        cache_index_path: Path to cache index file

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
        cache_only=cache_only,
        cache_index_path=cache_index_path,
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
            cache_only=cache_only,
            cache_index_path=cache_index_path,
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
    # Training parameters (None = use defaults)
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    save_dir: str = 'checkpoints/cross_subject',
    data_root: str = 'data',
    device: Optional[torch.device] = None,
    seed: int = 42,
    # Run identification
    run_tag: Optional[str] = None,
    # Config overrides (for scheduler experiments)
    config_overrides: Optional[Dict] = None,
    # Cache-only mode
    cache_only: bool = False,
    cache_index_path: str = ".cache_index.json",
    # WandB parameters
    wandb_enabled: bool = False,
    upload_model: bool = False,
    wandb_project: str = 'eeg-bci',
    wandb_entity: Optional[str] = None,
    wandb_group: Optional[str] = None,
    wandb_metadata: Optional[Dict[str, str]] = None,
    # Logging verbosity
    verbose: int = 2,
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
        run_tag: Optional run tag (timestamp) for this experiment (None = auto-generate)
        config_overrides: Optional dict to override config values
        cache_only: If True, load data exclusively from cache index
        cache_index_path: Path to cache index file for cache_only mode
        wandb_enabled: Enable WandB logging
        upload_model: Upload model artifacts to WandB
        wandb_project: WandB project name
        wandb_entity: WandB entity (team/username)
        wandb_group: WandB run group
        wandb_metadata: Pre-collected metadata for WandB
        verbose: Logging verbosity (0=silent, 1=minimal, 2=full)

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

    # Generate run_tag at start of training (if not provided)
    if run_tag is None:
        run_tag = datetime.now().strftime('%Y%m%d_%H%M')

    if device is None:
        device = get_device()

    # ========== WANDB INITIALIZATION ==========
    wandb_config = {
        "model_type": model_type,
        "task": task,
        "paradigm": paradigm,
        "subjects": subjects,
        "n_subjects": len(subjects),
        "training_type": "cross-subject",
    }

    wandb_logger = create_wandb_logger(
        subject_id=f"cross_{len(subjects)}subj",  # e.g., "cross_7subj"
        model_type=model_type,
        task=task,
        paradigm=paradigm,
        config=wandb_config,
        enabled=wandb_enabled,
        project=wandb_project,
        entity=wandb_entity,
        group=wandb_group or f"cross_subject_{model_type}",
        log_model=upload_model,
        metadata=wandb_metadata,
    )

    wandb_callback = WandbCallback(wandb_logger) if wandb_logger.enabled else None

    # ========== PERFORMANCE OPTIMIZATION ==========
    setup_performance_optimizations(device, verbose)

    # Subject header (verbose >= 1)
    if verbose >= 1:
        print()
        print(colored("=" * 70, Colors.BRIGHT_BLUE, bold=True))
        print(colored(f"  Cross-Subject Pretraining: {model_type.upper()}", Colors.BRIGHT_BLUE, bold=True))
        print(colored(f"  Subjects: {', '.join(subjects)}", Colors.BRIGHT_BLUE))
        print(colored("=" * 70, Colors.BRIGHT_BLUE, bold=True))

    # ========== CONFIG SETUP ==========
    config = get_cross_subject_config(model_type, task)

    # Apply CLI overrides (if user explicitly specified)
    if batch_size is not None:
        config['training']['batch_size'] = batch_size
    if epochs is not None:
        config['training']['epochs'] = epochs

    # Apply config overrides (same logic as within-subject)
    config = apply_config_overrides(config, config_overrides, log_prefix="[Cross-Subject] ")

    # Task configuration
    task_config = config['tasks'][task]
    target_classes = task_config['classes']
    n_classes = task_config['n_classes']

    # Setup paths
    data_root_path = Path(data_root)
    elc_path = data_root_path / 'biosemi128.ELC'
    save_path = Path(save_dir) / f'{run_tag}_{model_type}_{paradigm}_{task}'
    save_path.mkdir(parents=True, exist_ok=True)

    # Preprocessing config
    if model_type == 'cbramod':
        preprocess_config = PreprocessConfig.for_cbramod(full_channels=True)
    else:
        preprocess_config = PreprocessConfig.paper_aligned(n_class=n_classes)

    # Update WandB config with resolved values
    if wandb_logger.enabled:
        wandb_logger.update_config({
            "model_config": config.get('model', {}),
            "training_config": config.get('training', {}),
            "batch_size": config['training']['batch_size'],
            "epochs": config['training']['epochs'],
        })

    # ========== DATA LOADING ==========
    if verbose >= 2:
        print_section_header("Data Loading (Cross-Subject)")
        print(colored(f"  Subjects: {subjects}", Colors.CYAN))

    with Timer("data_loading", print_on_exit=(verbose >= 2)):
        train_dataset, test_datasets = load_multi_subject_data(
            data_root_path,
            subjects,
            preprocess_config,
            target_classes,
            paradigm,
            task,
            elc_path,
            cache_only=cache_only,
            cache_index_path=cache_index_path,
        )

    if verbose >= 2:
        print_metric("Total train segments", len(train_dataset), Colors.CYAN)
        print_metric("Subjects with test data", len(test_datasets), Colors.MAGENTA)

    # ========== TEMPORAL SPLIT ==========
    if verbose >= 2:
        print_section_header("Data Splitting (Temporal per Subject)")

    with Timer("data_splitting", print_on_exit=(verbose >= 2)):
        train_indices, val_indices = temporal_split_cross_subject(train_dataset)

    if verbose >= 2:
        print_metric("Train segments", len(train_indices), Colors.GREEN)
        print_metric("Val segments", len(val_indices), Colors.YELLOW)

    # ========== DATALOADER CREATION (Two-Phase) ==========
    if verbose >= 2:
        print_section_header("DataLoader Creation")

    # Get scheduler config for two-phase batch size (with cross-subject overrides)
    scheduler_type = config['training'].get('scheduler', None)
    scheduler_config = get_scheduler_config_from_preset(scheduler_type, config, cross_subject=True)

    with Timer("dataloader_creation", print_on_exit=(verbose >= 2)):
        exploration_loader, val_loader, main_train_loader, exploration_epochs = create_two_phase_loaders(
            train_dataset,
            train_indices,
            val_indices,
            scheduler_config,
            config['training']['batch_size'],
            num_workers=0,
            verbose=verbose,
        )

    # Get input dimensions
    sample_segment, _ = train_dataset[0]
    n_channels = sample_segment.shape[0]
    n_samples = sample_segment.shape[1]

    if verbose >= 2:
        print_metric("Input shape", f"[{n_channels}, {n_samples}]", Colors.CYAN)
        print_metric("Exploration batches/epoch", len(exploration_loader), Colors.GREEN)
        print_metric("Main batches/epoch", len(main_train_loader), Colors.GREEN)
        print_metric("Val batches", len(val_loader), Colors.YELLOW)

    # ========== MODEL CREATION ==========
    if verbose >= 2:
        print_section_header("Model Creation")

    with Timer("model_creation", print_on_exit=(verbose >= 2)):
        model = create_cross_subject_model(
            model_type,
            n_channels,
            n_samples,
            n_classes,
            config,
        )

    if verbose >= 2:
        print_metric("Model", model_type.upper(), Colors.CYAN)
        print_metric("Parameters", f"{model.count_parameters():,}", Colors.CYAN)
        print_metric("Device", str(device), Colors.GREEN)

    # ========== MODEL COMPILATION (PyTorch 2.0+) ==========
    use_compile = config.get('training', {}).get('use_compile', True)
    model = maybe_compile_model(model, model_type, device, use_compile, verbose)

    # ========== TRAINER SETUP ==========
    train_config = config['training']
    learning_rate = train_config.get('learning_rate', 1e-3)
    classifier_lr = train_config.get('classifier_lr', None)
    weight_decay = train_config.get('weight_decay', 0.0)

    if model_type == 'cbramod':
        learning_rate = train_config.get('backbone_lr', 5e-5)
        classifier_lr = train_config.get('classifier_lr', learning_rate * 3)
        weight_decay = train_config.get('weight_decay', 0.12)
        # Default to cosine_annealing_warmup_decay for CBraMod
        if scheduler_type is None:
            scheduler_type = 'cosine_annealing_warmup_decay'
            log_train.info(f"Scheduler: cosine_annealing_warmup_decay (CBraMod default)")

    # Read label_smoothing and gradient_clip from config (cross-subject has different values)
    label_smoothing = train_config.get('label_smoothing', None)
    gradient_clip = train_config.get('gradient_clip', 1.0 if model_type == 'cbramod' else 0.0)

    trainer = WithinSubjectTrainer(
        model,
        train_dataset,
        val_indices,
        device,
        model_type=model_type,
        n_classes=n_classes,
        learning_rate=learning_rate,
        classifier_lr=classifier_lr,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        scheduler_type=scheduler_type,
        scheduler_config=scheduler_config,
        use_amp=True,
        gradient_clip=gradient_clip,
    )

    # ========== TRAINING ==========
    with Timer("training"):
        history = trainer.train(
            exploration_loader,
            val_loader,
            main_train_loader=main_train_loader,
            exploration_epochs=exploration_epochs,
            epochs=config['training']['epochs'],
            patience=config['training'].get('patience', 10),
            save_path=save_path,
            wandb_callback=wandb_callback,
        )

    # ========== PER-SUBJECT TEST EVALUATION ==========
    if verbose >= 1:
        print_section_header("Per-Subject Test Evaluation")

    per_subject_test_acc = {}
    for subject_id, test_dataset in test_datasets.items():
        test_indices = list(range(len(test_dataset)))
        test_acc, _ = majority_vote_accuracy(
            model, test_dataset, test_indices, device
        )
        per_subject_test_acc[subject_id] = test_acc

        if verbose >= 1:
            acc_color = Colors.BRIGHT_GREEN if test_acc > 0.7 else (
                Colors.YELLOW if test_acc > 0.5 else Colors.RED
            )
            print(f"  {subject_id}: {colored(f'{test_acc:.2%}', acc_color)}")

    # Overall test accuracy (mean across subjects)
    if per_subject_test_acc:
        mean_test_acc = float(np.mean(list(per_subject_test_acc.values())))
        std_test_acc = float(np.std(list(per_subject_test_acc.values())))
    else:
        mean_test_acc = 0.0
        std_test_acc = 0.0
        log_train.warning("No test data available for any subject - mean/std set to 0")

    if verbose >= 1:
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

    if verbose >= 2:
        Timer.print_summary("Cross-Subject Training")

    # ========== WANDB FINALIZATION ==========
    if wandb_callback is not None:
        wandb_callback.on_train_end(
            best_epoch=trainer.best_epoch,
            best_val_acc=trainer.best_val_acc,
            test_acc=mean_test_acc,
            test_majority_acc=mean_test_acc,
            model_path=save_path / 'best.pt',
        )
        wandb_logger.finish()

    return {
        'run_tag': run_tag,
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
