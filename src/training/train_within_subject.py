"""
Within-subject training module for FINGER-EEG-BCI.

This module provides the main training functions for single-subject EEG classification.
It imports from submodules for:
- schedulers.py: Learning rate schedulers (WSD, CosineDecayRestarts, CosineAnnealingWarmupDecay)
- evaluation.py: Evaluation functions (majority_vote_accuracy)
- trainer.py: WithinSubjectTrainer class

SCHEDULER_PRESETS and get_default_config are re-exported from src.config.training.

Supports both EEGNet and CBraMod models, and both Motor Imagery (MI) and
Motor Execution (ME) paradigms.

Data Split (follows paper protocol):
- Training: Offline + Session 1 (Base + Finetune) + Session 2 Base
- Validation: Last 20% of training data (temporal split)
- Test: Session 2 Finetune (completely held out)

Usage (programmatic API):
    from src.training.train_within_subject import train_subject_simple

    # Train EEGNet
    results = train_subject_simple('S01', 'eegnet', 'binary')

    # Train CBraMod
    results = train_subject_simple('S01', 'cbramod', 'binary')

    # Train on Motor Execution paradigm
    results = train_subject_simple('S01', 'eegnet', 'binary', paradigm='movement')

For batch training across subjects, use the scripts:
    uv run python scripts/run_single_model.py --model eegnet
    uv run python scripts/run_within_subject_comparison.py
"""

# Suppress RuntimeWarning from multiprocessing workers when using -m flag
# Must be before any other imports to take effect in worker processes
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*found in sys.modules after import of package.*",
    category=RuntimeWarning,
)

import logging
import json
from pathlib import Path
from collections import Counter
from typing import Any, List, Tuple, Dict, Optional
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
from src.utils.device import get_device, check_cuda_available, set_seed, is_blackwell_gpu
from src.utils.logging import YellowFormatter, SectionLogger
from src.utils.wandb_logger import (
    is_wandb_available,
    WandbLogger,
    WandbCallback,
    create_wandb_logger,
)
from src.utils.timing import (
    Timer,
    EpochTimer,
    timed_section,
    print_section_header,
    print_metric,
    format_time,
    colored,
    Colors,
)
from src.utils.table_logger import TableEpochLogger

# ============================================================================
# Re-exports from submodules (for backward compatibility)
# ============================================================================
from .schedulers import (
    WSDScheduler,
    CosineDecayRestarts,
    CosineAnnealingWarmupDecay,
    visualize_lr_schedule,
)
from .evaluation import majority_vote_accuracy
from .trainer import WithinSubjectTrainer
from .common import (
    setup_performance_optimizations,
    maybe_compile_model,
    get_scheduler_config_from_preset,
    apply_config_overrides,
    temporal_split_by_group,
)

# Re-exports from src.config.training
from ..config.training import SCHEDULER_PRESETS, get_default_config


logger = logging.getLogger(__name__)

# Section-specific loggers
log_data = SectionLogger(logger, 'data')
log_model = SectionLogger(logger, 'model')
log_train = SectionLogger(logger, 'train')
log_eval = SectionLogger(logger, 'eval')


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_subject_data(
    data_root: Path,
    subject_id: str,
    session_folders: List[str],
    target_classes: List[int],
    config: PreprocessConfig,
    elc_path: Path,
    cache_only: bool = False,
    cache_index_path: str = ".cache_index.json",
) -> FingerEEGDataset:
    """
    Load data for a single subject using session folder filtering.

    Args:
        data_root: Root directory containing subject folders
        subject_id: Subject ID (e.g., 'S01')
        session_folders: List of session folders to include
        target_classes: List of target classes
        config: Preprocessing configuration
        elc_path: Path to ELC file
        cache_only: If True, load exclusively from cache index (default: False)
        cache_index_path: Path to cache index file (default: '.cache_index.json')

    Returns:
        FingerEEGDataset instance
    """
    dataset = FingerEEGDataset(
        str(data_root),
        [subject_id],
        config,
        session_folders=session_folders,
        target_classes=target_classes,
        elc_path=str(elc_path),
        cache_only=cache_only,
        cache_index_path=cache_index_path,
    )
    return dataset


def create_data_loaders_from_dataset(
    dataset: FingerEEGDataset,
    train_indices: List[int],
    val_indices: List[int],
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle_train: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders from dataset with train/val split.

    Note: dataset already contains segments (not trials) when using paper-aligned preprocessing.

    IMPORTANT: On Windows, num_workers=0 is faster because:
    - Data is already in memory (self.trials list)
    - Windows uses 'spawn' (not 'fork'), so each worker copies all data
    - The overhead of spawning workers exceeds any parallel benefit

    Args:
        dataset: FingerEEGDataset instance
        train_indices: Indices for training samples
        val_indices: Indices for validation samples
        batch_size: Batch size
        num_workers: Number of data loading workers (default: 0 - single thread is faster for in-memory data)
        shuffle_train: Whether to shuffle training data (default: False for chronological order)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create subsets
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_task_type_patterns(task: str, n_class: int, paradigm: str = 'imagery') -> Dict[str, List[str]]:
    """
    Get session folder patterns for paper protocol.

    Data split follows paper:
    - Training: Offline + Session 1 (Base + Finetune) + Session 2 Base
    - Test: Session 2 Finetune (completely held out)

    Args:
        task: Task name ('binary', 'ternary', 'quaternary')
        n_class: Number of classes (2, 3, or 4)
        paradigm: 'imagery' (MI) or 'movement' (ME)

    Returns:
        Dict with 'train' and 'test' session folder lists
    """
    return {
        'train': get_session_folders_for_split(paradigm, task, 'train'),
        'test': get_session_folders_for_split(paradigm, task, 'test'),
    }


# ============================================================================
# Main Training Function
# ============================================================================

def train_single_subject(
    subject_id: str,
    config: dict,
    data_root: Path,
    elc_path: Path,
    save_dir: Path,
    device: torch.device,
    model_type: str = 'eegnet',
    paradigm: str = 'imagery',
    cbramod_channels: int = 128,
    # Custom preprocessing config (for ML engineering experiments)
    preprocess_config: Optional[PreprocessConfig] = None,
    # Cache-only mode (for training without original .mat files)
    cache_only: bool = False,
    cache_index_path: str = ".cache_index.json",
    # WandB parameters
    no_wandb: bool = False,
    upload_model: bool = False,
    wandb_project: str = 'eeg-bci',
    wandb_entity: Optional[str] = None,
    wandb_group: Optional[str] = None,
    # Logging verbosity
    verbose: int = 2,
) -> Dict:
    """
    Train model for a single subject.

    Following paper protocol:
    - Training: Offline + Session 1 (Base + Finetune) + Session 2 Base
    - Validation: Last 20% of training data (temporal split)
    - Test: Session 2 Finetune (completely held out)

    Args:
        subject_id: Subject ID (e.g., 'S01')
        config: Configuration dict
        data_root: Path to data directory
        elc_path: Path to electrode location file
        save_dir: Path to save checkpoints
        device: Device to use (cuda/cpu)
        model_type: 'eegnet' or 'cbramod'
        paradigm: 'imagery' (MI) or 'movement' (ME)
        cbramod_channels: Number of channels for CBraMod (19 or 128).
            128 uses ACPE adaptation for full BioSemi channels.
        preprocess_config: Optional custom PreprocessConfig. If None, uses
            default config for the model type. Used by ML engineering experiments.
        cache_only: If True, load data exclusively from cache index without
            scanning filesystem. Useful when original .mat files are not available.
        cache_index_path: Path to cache index file for cache_only mode.
        no_wandb: Disable wandb logging
        upload_model: Upload model artifacts (.pt) to WandB (default: False)
        wandb_project: WandB project name
        wandb_entity: WandB entity (team/username)
        wandb_group: WandB run group
        verbose: Logging verbosity level (0=silent, 1=minimal, 2=full). Default: 2.
            Level 2: Full output (all sections)
            Level 1: Subject header + training table + final evaluation
            Level 0: Training table only

    Returns:
        Results dict with accuracy and history
    """
    total_start = time.perf_counter()
    Timer.reset()

    # ========== WANDB INITIALIZATION ==========
    wandb_config = {
        "model_type": model_type,
        "model_config": config.get('model', {}),
        "training_config": config.get('training', {}),
        "task": config['task'],
        "paradigm": paradigm,
        "cbramod_channels": cbramod_channels if model_type == 'cbramod' else None,
    }

    wandb_logger = create_wandb_logger(
        subject_id=subject_id,
        model_type=model_type,
        task=config['task'],
        paradigm=paradigm,
        config=wandb_config,
        enabled=not no_wandb,
        project=wandb_project,
        entity=wandb_entity,
        group=wandb_group,
        log_model=upload_model,
    )

    wandb_callback = WandbCallback(wandb_logger) if wandb_logger.enabled else None

    # ========== PERFORMANCE OPTIMIZATION ==========
    setup_performance_optimizations(device, verbose)

    # Subject header (verbose >= 1)
    if verbose >= 1:
        print()
        print(colored("=" * 60, Colors.BRIGHT_BLUE, bold=True))
        print(colored(f"  Training Subject: {subject_id} ({model_type.upper()})", Colors.BRIGHT_BLUE, bold=True))
        print(colored("=" * 60, Colors.BRIGHT_BLUE, bold=True))

    # Task configuration
    task_config = config['tasks'][config['task']]
    target_classes = task_config['classes']
    n_classes = task_config['n_classes']

    # Get session folders for paper protocol
    task_patterns = get_task_type_patterns(config['task'], n_classes, paradigm)
    paradigm_desc = "Motor Imagery (MI)" if paradigm == 'imagery' else "Motor Execution (ME)"

    # ========== DATA LOADING ==========
    if verbose >= 2:
        print_section_header(f"Data Loading ({paradigm_desc})")
        print(colored(f"  Train folders: {task_patterns['train']}", Colors.DIM))
        print(colored(f"  Test folders: {task_patterns['test']}", Colors.DIM))

    # Select preprocessing config based on model type (unless custom config provided)
    if preprocess_config is not None:
        # Use custom config (e.g., from ML engineering experiments)
        log_data.info(f"Preprocess: Custom config ({preprocess_config.target_fs}Hz, "
                      f"{preprocess_config.bandpass_low}-{preprocess_config.bandpass_high}Hz)")
        if verbose >= 2:
            print(colored(f"  Custom preprocessing config provided", Colors.CYAN))
    elif model_type == 'cbramod':
        # Check if using 128 channels (ACPE adaptation)
        use_full_channels = (cbramod_channels == 128)
        preprocess_config = PreprocessConfig.for_cbramod(full_channels=use_full_channels)
        if use_full_channels:
            log_data.info("Preprocess: CBraMod (128ch, 200Hz, 0.3-75Hz) - ACPE adaptation")
            if verbose >= 2:
                print(colored(f"  CBraMod 128-channel mode: Using ACPE for channel adaptation", Colors.CYAN))
        else:
            log_data.info("Preprocess: CBraMod (19ch, 200Hz, 0.3-75Hz)")
    else:
        preprocess_config = PreprocessConfig.paper_aligned(n_class=n_classes)
        log_data.info("Preprocess: EEGNet (128ch, 100Hz, 4-40Hz)")

    data_config = config['data']
    if 'window_length' in data_config:
        preprocess_config.trial_duration = data_config['window_length']

    # Load TRAINING data
    if verbose >= 2:
        print(colored("\n  Loading training data...", Colors.DIM))
    with Timer("train_data_loading", print_on_exit=(verbose >= 2)):
        train_dataset = load_subject_data(
            data_root, subject_id,
            session_folders=task_patterns['train'],
            target_classes=target_classes,
            config=preprocess_config,
            elc_path=elc_path,
            cache_only=cache_only,
            cache_index_path=cache_index_path,
        )

    if len(train_dataset) == 0:
        print(colored(f"  ERROR: No training data found for subject {subject_id}", Colors.RED))
        return {}

    if verbose >= 2:
        print_metric("Train segments (total)", len(train_dataset), Colors.CYAN)
        # Show detailed cache status with hit/miss counts
        if train_dataset.cache:
            hits = getattr(train_dataset, 'n_cache_hits', 0)
            misses = getattr(train_dataset, 'n_cache_misses', 0)
            if misses == 0 and hits > 0:
                cache_status = f"hit ({hits} files)"
            elif hits == 0 and misses > 0:
                cache_status = f"miss ({misses} files)"
            elif hits > 0 and misses > 0:
                cache_status = f"partial ({hits} hit, {misses} miss)"
            else:
                cache_status = "enabled"
            print_metric("Cache", cache_status, Colors.GREEN)
        else:
            print_metric("Cache", "disabled", Colors.DIM)

    # Load TEST data (Session 2 Finetune - completely held out)
    if verbose >= 2:
        print(colored("\n  Loading test data (Session 2 Finetune)...", Colors.DIM))
    with Timer("test_data_loading", print_on_exit=(verbose >= 2)):
        test_dataset = load_subject_data(
            data_root, subject_id,
            session_folders=task_patterns['test'],
            target_classes=target_classes,
            config=preprocess_config,
            elc_path=elc_path,
            cache_only=cache_only,
            cache_index_path=cache_index_path,
        )

    if len(test_dataset) == 0:
        if verbose >= 2:
            print(colored(f"  WARNING: No test data found for subject {subject_id}", Colors.YELLOW))

    if verbose >= 2:
        print_metric("Test segments", len(test_dataset) if test_dataset else 0, Colors.MAGENTA)
        # Show test data cache status
        if test_dataset and test_dataset.cache:
            hits = getattr(test_dataset, 'n_cache_hits', 0)
            misses = getattr(test_dataset, 'n_cache_misses', 0)
            if misses == 0 and hits > 0:
                cache_status = f"hit ({hits} files)"
            elif hits == 0 and misses > 0:
                cache_status = f"miss ({misses} files)"
            elif hits > 0 and misses > 0:
                cache_status = f"partial ({hits} hit, {misses} miss)"
            else:
                cache_status = "enabled"
            print_metric("Cache", cache_status, Colors.GREEN)

    # ========== DATA SPLITTING (Temporal) ==========
    if verbose >= 2:
        print_section_header("Data Splitting (Temporal - Last 20% for Validation)")

    with Timer("data_splitting", print_on_exit=(verbose >= 2)):
        if verbose >= 2:
            n_trials = len(train_dataset.get_unique_trials())
            print_metric("Total training trials", n_trials, Colors.CYAN)

        # Stratified temporal split: split within each session to ensure
        # similar distributions (prevents validation set from being 100%
        # one session type).
        train_indices, val_indices = temporal_split_by_group(
            train_dataset, group_attr='session_type', val_ratio=0.2,
        )

    if verbose >= 2:
        print_metric("Train segments", len(train_indices), Colors.GREEN)
        print_metric("Val segments", len(val_indices), Colors.YELLOW)

    # ========== DATALOADER CREATION ==========
    if verbose >= 2:
        print_section_header("DataLoader Creation")

    # Get scheduler config (from SCHEDULER_PRESETS or config override)
    scheduler_type = config['training'].get('scheduler', None)
    scheduler_config = get_scheduler_config_from_preset(scheduler_type, config)

    # Get exploration phase parameters from scheduler config
    exploration_epochs = scheduler_config.get('exploration_epochs', 5)
    exploration_batch_size = scheduler_config.get('exploration_batch_size', 32)
    main_batch_size = config['training']['batch_size']

    with Timer("dataloader_creation", print_on_exit=(verbose >= 2)):
        # Create exploration loader (small batch for loss landscape exploration)
        exploration_loader, val_loader = create_data_loaders_from_dataset(
            train_dataset, train_indices, val_indices,
            batch_size=exploration_batch_size,
            num_workers=0,
            shuffle_train=True,
        )

        # Create main loader (normal batch for stable training)
        main_train_loader, _ = create_data_loaders_from_dataset(
            train_dataset, train_indices, val_indices,
            batch_size=main_batch_size,
            num_workers=0,
            shuffle_train=True,
        )

    # Get input dimensions
    sample_segment, _ = train_dataset[0]
    n_channels = sample_segment.shape[0]
    n_samples = sample_segment.shape[1]

    if verbose >= 2:
        print_metric("Exploration batch size", f"{exploration_batch_size} (epochs 1-{exploration_epochs})", Colors.CYAN)
        print_metric("Main batch size", f"{main_batch_size} (epochs {exploration_epochs+1}+)", Colors.CYAN)
        print_metric("Input shape", f"[{n_channels}, {n_samples}]", Colors.CYAN)
        print_metric("Exploration batches/epoch", len(exploration_loader), Colors.GREEN)
        print_metric("Main batches/epoch", len(main_train_loader), Colors.GREEN)
        print_metric("Val batches", len(val_loader), Colors.YELLOW)
        print(colored("  Training order: Shuffled (random order per epoch)", Colors.DIM))

    # ========== MODEL CREATION ==========
    if verbose >= 2:
        print_section_header("Model Creation")

    with Timer("model_creation", print_on_exit=(verbose >= 2)):
        model_config = config['model']

        if model_type == 'cbramod':
            # CBraMod expects patches
            n_patches = n_samples // 200  # 200 samples per patch (1s @ 200Hz)
            pretrained_path = get_default_pretrained_path()

            model = CBraModForFingerBCI(
                n_channels=n_channels,
                n_patches=n_patches,
                n_classes=n_classes,
                pretrained_path=pretrained_path,
                freeze_backbone=model_config.get('freeze_backbone', False),
                classifier_type=model_config.get('classifier_type', 'three_layer'),
                dropout=model_config.get('dropout_rate', 0.1),
            )
            model_name = "CBraMod"
        else:
            # EEGNet
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
            model_name = "EEGNet-8,2"

    if verbose >= 2:
        print_metric("Model", model_name, Colors.CYAN)
        print_metric("Parameters", f"{model.count_parameters():,}", Colors.CYAN)
        print_metric("Device", str(device), Colors.GREEN)

    # ========== MODEL COMPILATION (PyTorch 2.0+) ==========
    use_compile = config.get('training', {}).get('use_compile', True)
    model = maybe_compile_model(model, model_type, device, use_compile, verbose)

    # ========== TRAINER SETUP ==========
    with Timer("trainer_setup", print_on_exit=(verbose >= 2)):
        # Get training config (model-specific defaults may override)
        train_config = config['training']
        learning_rate = train_config.get('learning_rate', 1e-3)
        classifier_lr = train_config.get('classifier_lr', None)
        weight_decay = train_config.get('weight_decay', 0.0)
        scheduler_type = train_config.get('scheduler', None)

        # CBraMod specific overrides
        if model_type == 'cbramod':
            learning_rate = train_config.get('backbone_lr', 1e-4)
            classifier_lr = train_config.get('classifier_lr', learning_rate * 3)
            weight_decay = train_config.get('weight_decay', 0.06)
            # Default to cosine_annealing_warmup_decay scheduler for CBraMod
            if scheduler_type is None:
                scheduler_type = 'cosine_annealing_warmup_decay'
                log_train.info(f"Scheduler: cosine_annealing_warmup_decay (CBraMod default)")

        # Performance optimizations
        use_amp = True  # Enable AMP for both models
        gradient_clip = train_config.get('gradient_clip', 1.0 if model_type == 'cbramod' else 0.0)
        label_smoothing = train_config.get('label_smoothing', None)

        trainer = WithinSubjectTrainer(
            model, train_dataset, val_indices, device,
            model_type=model_type,
            n_classes=n_classes,
            learning_rate=learning_rate,
            classifier_lr=classifier_lr,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing,
            scheduler_type=scheduler_type,
            scheduler_config=scheduler_config,
            use_amp=use_amp,
            gradient_clip=gradient_clip,
        )

    # Create save directory
    subject_save_dir = save_dir / subject_id
    subject_save_dir.mkdir(parents=True, exist_ok=True)

    # ========== TRAINING ==========
    with Timer("training"):
        history = trainer.train(
            exploration_loader, val_loader,
            main_train_loader=main_train_loader,
            exploration_epochs=exploration_epochs,
            epochs=config['training']['epochs'],
            patience=config['training']['patience'],
            save_path=subject_save_dir,
            wandb_callback=wandb_callback,
        )

    # ========== FINAL EVALUATION ==========
    if verbose >= 1:
        print_section_header("Final Evaluation (Three-Phase Protocol)")

    # Evaluate on validation set (from training data)
    with Timer("val_evaluation", print_on_exit=(verbose >= 1)):
        val_acc, val_results = majority_vote_accuracy(
            model, train_dataset, val_indices, device
        )

    if verbose >= 1:
        val_color = Colors.BRIGHT_GREEN if val_acc > 0.7 else (Colors.YELLOW if val_acc > 0.5 else Colors.RED)
        print(f"\n  {colored('Validation Accuracy (last 20% of train):', Colors.WHITE)} "
              f"{colored(f'{val_acc:.4f}', val_color)}")

    # Evaluate on TEST set (Online_Finetune - Phase 3)
    test_acc = 0.0
    test_results = {}
    n_test_trials = 0

    if len(test_dataset) > 0:
        with Timer("test_evaluation", print_on_exit=(verbose >= 1)):
            # Get all test indices
            test_indices = list(range(len(test_dataset)))
            test_acc, test_results = majority_vote_accuracy(
                model, test_dataset, test_indices, device
            )
            n_test_trials = len(test_dataset.get_unique_trials())

        if verbose >= 1:
            test_color = Colors.BRIGHT_GREEN if test_acc > 0.7 else (Colors.YELLOW if test_acc > 0.5 else Colors.RED)
            print(f"  {colored('TEST Accuracy (Online_Finetune):', Colors.WHITE, bold=True)} "
                  f"{colored(f'{test_acc:.4f}', test_color, bold=True)}")
            print(f"  {colored(f'Test trials: {n_test_trials}', Colors.DIM)}")
    else:
        if verbose >= 1:
            print(colored("  No test data available (Online_Finetune)", Colors.YELLOW))

    # ========== TIMING SUMMARY ==========
    total_time = time.perf_counter() - total_start
    if verbose >= 2:
        Timer.print_summary(f"Timing Summary - {subject_id}")

    # Save results
    epochs_trained = len(history['train_loss'])
    results = {
        'subject': subject_id,
        'task': config['task'],
        'model_type': model_type,
        'protocol': 'three_phase',
        # Training info
        'n_trials_train': len(train_trials),
        'n_trials_val': len(val_trials),
        'n_trials_test': n_test_trials,
        # Accuracies
        'val_accuracy': val_acc,
        'best_val_acc': trainer.best_val_acc,  # Val segment accuracy at best epoch
        'best_majority_acc': trainer.best_majority_acc,  # Val majority accuracy at best epoch
        'best_combined_score': trainer.best_combined_score,  # (val_acc + majority_acc) / 2 at best epoch
        'test_accuracy': test_acc,  # This is the main metric (Phase 3)
        'test_accuracy_majority': test_acc,  # Alias for compatibility with run_within_subject_comparison
        'final_accuracy': test_acc if test_acc > 0 else val_acc,  # For backwards compatibility
        # Training info
        'best_epoch': trainer.best_epoch,
        'epochs_trained': epochs_trained,
        'best_val_loss': trainer.best_val_loss,
        'training_time': total_time,  # Total training time in seconds
        'history': history,
        'val_evaluation': val_results,
        'test_evaluation': test_results,
    }

    with open(subject_save_dir / 'results.json', 'w') as f:
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            return obj

        json.dump(results, f, indent=2, default=convert)

    # Save history
    with open(subject_save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # ========== WANDB FINALIZATION ==========
    if wandb_callback is not None:
        # Get class names for confusion matrix
        task_config = config['tasks'][config['task']]
        class_names = task_config.get('classes', [str(i) for i in range(task_config['n_classes'])])

        # Extract predictions from test results for confusion matrix
        y_true = None
        y_pred = None
        if test_results and 'trial_predictions' in test_results:
            trial_preds = test_results['trial_predictions']
            y_true = np.array([v['true_label'] for v in trial_preds.values()])
            y_pred = np.array([v['predicted_label'] for v in trial_preds.values()])

        wandb_callback.on_train_end(
            best_epoch=trainer.best_epoch,
            best_val_acc=trainer.best_val_acc,
            test_acc=test_acc,
            test_majority_acc=test_acc,
            model_path=subject_save_dir / 'best.pt',
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
        )
        wandb_logger.finish()

    return results


# ============================================================================
# Simplified Training API
# ============================================================================

def train_subject_simple(
    subject_id: str,
    model_type: str,
    task: str,
    paradigm: str = 'imagery',
    data_root: str = 'data',
    save_dir: str = 'checkpoints',
    device: Optional[torch.device] = None,
    # Run identification
    run_tag: Optional[str] = None,
    cbramod_channels: int = 128,
    # Custom preprocessing config (for ML engineering experiments)
    preprocess_config: Optional[PreprocessConfig] = None,
    # Config overrides (for scheduler comparison experiments)
    config_overrides: Optional[Dict] = None,
    # Cache-only mode
    cache_only: bool = False,
    cache_index_path: str = ".cache_index.json",
    # WandB parameters
    no_wandb: bool = False,
    upload_model: bool = False,
    wandb_project: str = 'eeg-bci',
    wandb_entity: Optional[str] = None,
    wandb_group: Optional[str] = None,
    # Logging verbosity
    verbose: int = 2,
) -> Dict:
    """
    Simplified training function for programmatic use.

    This is a convenience wrapper around train_single_subject() that
    handles configuration and path setup automatically.

    Args:
        subject_id: Subject ID (e.g., 'S01')
        model_type: 'eegnet' or 'cbramod'
        task: 'binary', 'ternary', or 'quaternary'
        paradigm: 'imagery' (MI) or 'movement' (ME)
        data_root: Path to data directory
        save_dir: Path to save checkpoints
        device: Device to use (auto-detect if None)
        cbramod_channels: Number of channels for CBraMod (19 or 128).
            128 uses ACPE adaptation for full BioSemi channels.
        preprocess_config: Optional custom PreprocessConfig. If None, uses
            default config for the model type. Used by ML engineering experiments.
        cache_only: If True, load data exclusively from cache index without filesystem
        cache_index_path: Path to cache index file for cache_only mode
        no_wandb: Disable wandb logging
        upload_model: Upload model artifacts (.pt) to WandB (default: False)
        wandb_project: WandB project name
        wandb_entity: WandB entity (team/username)
        wandb_group: WandB run group
        verbose: Logging verbosity level (0=silent, 1=minimal, 2=full). Default: 2.

    Returns:
        Results dict with keys:
        - subject, task, model_type, protocol
        - val_accuracy, best_val_acc, test_accuracy, test_accuracy_majority
        - epochs_trained, training_time, best_epoch, best_val_loss
        - history, val_evaluation, test_evaluation
    """
    # Generate run_tag at start (if not provided)
    if run_tag is None:
        run_tag = datetime.now().strftime('%Y%m%d_%H%M')

    if device is None:
        device = get_device()

    data_root_path = Path(data_root) if isinstance(data_root, str) else data_root
    elc_path = data_root_path / 'biosemi128.ELC'
    save_path = Path(save_dir) / f'{run_tag}_{model_type}_within_subject' / task

    config = get_default_config(model_type, task)

    # Apply config overrides with scheduler preset support
    config = apply_config_overrides(config, config_overrides)

    return train_single_subject(
        subject_id=subject_id,
        config=config,
        data_root=data_root_path,
        elc_path=elc_path,
        save_dir=save_path,
        device=device,
        model_type=model_type,
        paradigm=paradigm,
        cbramod_channels=cbramod_channels,
        preprocess_config=preprocess_config,
        # Cache-only mode
        cache_only=cache_only,
        cache_index_path=cache_index_path,
        # WandB parameters
        no_wandb=no_wandb,
        upload_model=upload_model,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_group=wandb_group,
        verbose=verbose,
    )
