"""
Within-subject training script for FINGER-EEG-BCI.

Supports both EEGNet and CBraMod models, and both Motor Imagery (MI) and
Motor Execution (ME) paradigms.

Data Split (follows paper protocol):
- Training: Offline + Session 1 (Base + Finetune) + Session 2 Base
- Validation: Last 20% of training data (temporal split)
- Test: Session 2 Finetune (completely held out)

EEGNet (default):
- 128 channels, 100 Hz sampling rate
- 4-40 Hz bandpass filter (4th order Butterworth)
- Z-score normalization per segment
- 1-second sliding window with 125ms step
- Majority voting for trial prediction
- Pre-training: 300 epochs on offline data

CBraMod:
- 19 or 128 channels, 200 Hz sampling rate
- 0.3-75 Hz bandpass, 60 Hz notch filter
- Divide by 100 normalization
- Patch-based processing (1s patches)
- Pre-training: 50 epochs with cosine annealing

Usage:
    # Train EEGNet on Motor Imagery (default)
    uv run python -m src.training.train_within_subject --subject S01 --task binary --model eegnet

    # Train on Motor Execution
    uv run python -m src.training.train_within_subject --subject S01 --task binary --model eegnet --paradigm movement

    # Train CBraMod for subject S01 (default 128 channels)
    uv run python -m src.training.train_within_subject --subject S01 --task binary --model cbramod

    # Train CBraMod with 19 channels (10-20 system, less memory)
    uv run python -m src.training.train_within_subject --subject S01 --task binary --model cbramod --cbramod-channels 19

    # Train all subjects with both models
    uv run python -m src.training.train_within_subject --all-subjects --task binary --model both
"""

# Suppress RuntimeWarning from multiprocessing workers when using -m flag
# Must be before any other imports to take effect in worker processes
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*found in sys.modules after import of package.*",
    category=RuntimeWarning,
)

import argparse
import logging
import yaml
import json
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict, Optional
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


logger = logging.getLogger(__name__)

# Section-specific loggers
log_data = SectionLogger(logger, 'data')
log_model = SectionLogger(logger, 'model')
log_train = SectionLogger(logger, 'train')
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


class WithinSubjectTrainer:
    """
    Trainer for within-subject model training (EEGNet or CBraMod).

    Follows the paper's training protocol:
    - EEGNet: Pre-train on offline data for 300 epochs, Adam optimizer
    - CBraMod: Pre-train for 50 epochs, AdamW with different LR for backbone/classifier
    - Early stopping on validation loss
    - Fine-tuning freezes early layers
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: FingerEEGDataset,
        val_indices: List[int],
        device: torch.device,
        model_type: str = 'eegnet',
        n_classes: int = 2,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        scheduler_type: Optional[str] = None,
        use_amp: bool = True,
        gradient_clip: float = 1.0,
    ):
        self.model = model.to(device)
        self.dataset = dataset
        self.val_indices = val_indices
        self.device = device
        self.model_type = model_type
        self.scheduler_type = scheduler_type

        # Loss function - apply label smoothing for CBraMod multi-class tasks (per paper Table 6)
        if model_type == 'cbramod' and n_classes > 2:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            log_model.info("Label smoothing=0.1 (CBraMod multi-class)")
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Create optimizer based on model type
        if model_type == 'cbramod' and hasattr(model, 'get_parameter_groups'):
            # CBraMod uses different LR for backbone and classifier
            param_groups = model.get_parameter_groups(
                backbone_lr=learning_rate,
                classifier_lr=learning_rate * 5,  # 5x for classifier
            )
            self.optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=weight_decay,
            )
        else:
            # EEGNet uses standard Adam
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )

        # Create scheduler if requested
        self.scheduler = None
        self.scheduler_needs_metric = False  # For ReduceLROnPlateau
        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=50,  # Will be updated in train() for CBraMod
                eta_min=1e-6,
            )
        elif scheduler_type == 'plateau':
            # ReduceLROnPlateau - aggressive decay for faster convergence
            # Note: 'verbose' parameter removed in PyTorch 2.3+
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.3,
                patience=3,
                min_lr=1e-6,
            )
            self.scheduler_needs_metric = True
            log_train.info("Scheduler: ReduceLROnPlateau (factor=0.3, patience=3)")

        # AMP (Automatic Mixed Precision) setup
        self.use_amp = use_amp and device.type == 'cuda'
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
            log_train.info("AMP enabled")
        else:
            self.scaler = None

        # Gradient clipping
        self.gradient_clip = gradient_clip

        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_majority_acc': [],
        }
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0  # Track best validation accuracy (segment-level)
        self.best_majority_acc = 0.0  # Track best validation accuracy (trial-level majority voting)
        self.best_epoch = 0
        self.best_state = None

    def train_epoch(self, dataloader: DataLoader, profile: bool = False) -> Tuple[float, float]:
        """Train for one epoch with AMP, gradient clipping, and per-step scheduler."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # Profiling variables
        if profile:
            import time
            t_data, t_transfer, t_forward, t_backward, t_optim = 0, 0, 0, 0, 0
            t_start = time.perf_counter()

        for segments, labels in dataloader:
            if profile:
                t_data += time.perf_counter() - t_start
                t0 = time.perf_counter()

            segments = segments.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if profile:
                torch.cuda.synchronize()
                t_transfer += time.perf_counter() - t0
                t0 = time.perf_counter()

            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass with AMP
            if self.use_amp:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = self.model(segments)
                    loss = self.criterion(outputs, labels)

                if profile:
                    torch.cuda.synchronize()
                    t_forward += time.perf_counter() - t0
                    t0 = time.perf_counter()

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                if profile:
                    torch.cuda.synchronize()
                    t_backward += time.perf_counter() - t0
                    t0 = time.perf_counter()

                # Gradient clipping (unscale first)
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(segments)
                loss = self.criterion(outputs, labels)

                if profile:
                    torch.cuda.synchronize()
                    t_forward += time.perf_counter() - t0
                    t0 = time.perf_counter()

                loss.backward()

                if profile:
                    torch.cuda.synchronize()
                    t_backward += time.perf_counter() - t0
                    t0 = time.perf_counter()

                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                self.optimizer.step()

            if profile:
                torch.cuda.synchronize()
                t_optim += time.perf_counter() - t0

            # Per-step scheduler update (for CBraMod, aligned with original)
            if self.scheduler is not None and self.model_type == 'cbramod':
                self.scheduler.step()

            total_loss += loss.item() * segments.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += segments.size(0)

            if profile:
                t_start = time.perf_counter()

        # Print profiling results
        if profile:
            t_total = t_data + t_transfer + t_forward + t_backward + t_optim
            print(f"\n  [PROFILE] data={t_data:.2f}s ({100*t_data/t_total:.0f}%) | "
                  f"transfer={t_transfer:.2f}s ({100*t_transfer/t_total:.0f}%) | "
                  f"forward={t_forward:.2f}s ({100*t_forward/t_total:.0f}%) | "
                  f"backward={t_backward:.2f}s ({100*t_backward/t_total:.0f}%) | "
                  f"optim={t_optim:.2f}s ({100*t_optim/t_total:.0f}%)")

        return total_loss / total, correct / total

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Validate (segment-level accuracy) with AMP support."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for segments, labels in dataloader:
            segments = segments.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.use_amp:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = self.model(segments)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(segments)
                loss = self.criterion(outputs, labels)

            total_loss += loss.item() * segments.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += segments.size(0)

        return total_loss / total, correct / total

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 300,
        patience: int = 20,
        save_path: Optional[Path] = None,
        wandb_callback: Optional['WandbCallback'] = None,
    ) -> Dict:
        """
        Full training loop with early stopping.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            epochs: Maximum epochs
            patience: Early stopping patience
            save_path: Path to save best model
            wandb_callback: Optional WandB callback for logging

        Returns:
            Training history
        """
        print_section_header(f"Training ({epochs} epochs, patience={patience})")

        no_improve = 0
        epoch_timer = EpochTimer()
        training_start = time.perf_counter()

        # Initialize table logger
        table_logger = TableEpochLogger(
            total_epochs=epochs,
            model_name=self.model_type.upper(),
            show_majority=True,
            keep_every=10,
            header_every=30,
        )
        table_logger.print_title()

        # Recreate scheduler with correct T_max for per-step scheduling (CBraMod)
        # Use T_max = total_steps / 2 for faster LR decay (reaches min at 50% of training)
        if self.scheduler_type == 'cosine' and self.model_type == 'cbramod':
            total_steps = epochs * len(train_loader)
            t_max = total_steps // 2  # Faster decay: reach min LR at half training
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
                eta_min=1e-6,
            )
            log_train.info(f"Scheduler: CosineAnnealing (T_max={t_max}, fast decay)")

        for epoch in range(epochs):
            epoch_timer.start_epoch()

            # Train (profile only first epoch to diagnose bottlenecks)
            do_profile = (epoch == 0)
            with epoch_timer.phase("train"):
                train_loss, train_acc = self.train_epoch(train_loader, profile=do_profile)

            # Validate
            with epoch_timer.phase("validate"):
                val_loss, val_acc = self.validate(val_loader)

            # Majority voting: compute every epoch for accurate early stopping
            with epoch_timer.phase("majority_vote"):
                majority_acc, _ = majority_vote_accuracy(
                    self.model, self.dataset, self.val_indices, self.device,
                    use_amp=self.use_amp
                )

            # Update scheduler (only for EEGNet - CBraMod uses per-step scheduling in train_epoch)
            if self.scheduler is not None and self.model_type != 'cbramod':
                if self.scheduler_needs_metric:
                    self.scheduler.step(val_loss)  # ReduceLROnPlateau needs metric
                else:
                    self.scheduler.step()

            epoch_timer.end_epoch()

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_majority_acc'].append(majority_acc)

            # Get current learning rate (used by WandB and table logger)
            current_lr = self.optimizer.param_groups[0]['lr']

            # WandB callback
            if wandb_callback is not None:
                wandb_callback.on_epoch_end(
                    epoch=epoch + 1,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    val_majority_acc=majority_acc,
                    learning_rate=current_lr,
                )

            # Determine if this epoch improved
            is_best_epoch = False

            # Early stopping: reset patience when EITHER segment acc OR majority acc improves
            # Save model when either metric reaches new high
            improved = False
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                improved = True
            if majority_acc > self.best_majority_acc:
                self.best_majority_acc = majority_acc
                improved = True

            if improved:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                self.best_state = self.model.state_dict().copy()
                no_improve = 0
                is_best_epoch = True

                if save_path:
                    torch.save({
                        'model_state_dict': self.best_state,
                        'epoch': self.best_epoch,
                        'val_acc': self.best_val_acc,
                        'val_majority_acc': self.best_majority_acc,
                        'val_loss': self.best_val_loss,
                    }, save_path / 'best.pt')
                    log_train.debug(f"Best model saved (val_acc={val_acc:.4f}, maj_acc={majority_acc:.4f})")
            else:
                no_improve += 1

            # Log epoch with table logger
            table_logger.on_epoch_end(
                epoch=epoch + 1,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                majority_acc=majority_acc,
                lr=current_lr,
                epoch_time=epoch_timer.current_epoch.get('total', 0.0),
                is_best=is_best_epoch,
                event="BEST" if is_best_epoch else None,
            )

            # Early stopping check
            if no_improve >= patience:
                break

        # Restore best model (prefer disk checkpoint if available)
        if save_path and (save_path / 'best.pt').exists():
            checkpoint = torch.load(save_path / 'best.pt', map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            log_train.info(f"Loaded best (val_acc={checkpoint.get('val_acc', 'N/A')})")
        elif self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        training_time = time.perf_counter() - training_start

        # Print training summary using table logger
        table_logger.print_summary()

        return self.history

    def freeze_early_layers(self):
        """
        Freeze early layers for fine-tuning.

        EEGNet: Freeze first 4 layers (temporal conv + spatial depthwise conv)
        CBraMod: Freeze backbone (only train classifier)
        """
        if self.model_type == 'cbramod':
            # Freeze backbone for CBraMod
            if hasattr(self.model, 'backbone'):
                for param in self.model.backbone.parameters():
                    param.requires_grad = False
                log_model.info("Frozen: CBraMod backbone")
        else:
            # EEGNet layers order: conv1 -> batchnorm1 -> depthwise -> batchnorm2
            layers_to_freeze = ['conv1', 'batchnorm1', 'depthwise_conv', 'batchnorm2']

            for name, param in self.model.named_parameters():
                for layer_name in layers_to_freeze:
                    if layer_name in name:
                        param.requires_grad = False
                        log_model.debug(f"Frozen: {name}")
                        break
            log_model.info("Frozen: first 4 layers")

        # Update optimizer to only train unfrozen parameters
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(trainable_params, lr=1e-3)


def load_subject_data(
    data_root: Path,
    subject_id: str,
    session_folders: List[str],
    target_classes: List[int],
    config: PreprocessConfig,
    elc_path: Path,
) -> FingerEEGDataset:
    """Load data for a single subject using session folder filtering."""
    dataset = FingerEEGDataset(
        str(data_root),
        [subject_id],
        config,
        session_folders=session_folders,
        target_classes=target_classes,
        elc_path=str(elc_path)
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
    # WandB parameters
    no_wandb: bool = False,
    wandb_project: str = 'eeg-bci',
    wandb_entity: Optional[str] = None,
    wandb_group: Optional[str] = None,
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
        no_wandb: Disable wandb logging
        wandb_project: WandB project name
        wandb_entity: WandB entity (team/username)
        wandb_group: WandB run group

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
    )

    wandb_callback = WandbCallback(wandb_logger) if wandb_logger.enabled else None

    # ========== PERFORMANCE OPTIMIZATION ==========
    # Enable cuDNN auto-tuning for faster convolutions (20-50% speedup)
    # This finds optimal algorithms for the specific input sizes
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        log_train.debug("cuDNN benchmark enabled")

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
    print_section_header(f"Data Loading ({paradigm_desc})")
    print(colored(f"  Train folders: {task_patterns['train']}", Colors.DIM))
    print(colored(f"  Test folders: {task_patterns['test']}", Colors.DIM))

    # Select preprocessing config based on model type
    if model_type == 'cbramod':
        # Check if using 128 channels (ACPE adaptation)
        use_full_channels = (cbramod_channels == 128)
        preprocess_config = PreprocessConfig.for_cbramod(full_channels=use_full_channels)
        if use_full_channels:
            log_data.info("Preprocess: CBraMod (128ch, 200Hz, 0.3-75Hz) - ACPE adaptation")
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
    print(colored("\n  Loading training data...", Colors.DIM))
    with Timer("train_data_loading", print_on_exit=True):
        train_dataset = load_subject_data(
            data_root, subject_id,
            session_folders=task_patterns['train'],
            target_classes=target_classes,
            config=preprocess_config,
            elc_path=elc_path
        )

    if len(train_dataset) == 0:
        print(colored(f"  ERROR: No training data found for subject {subject_id}", Colors.RED))
        return {}

    print_metric("Train segments (total)", len(train_dataset), Colors.CYAN)
    print_metric("Cache status", "enabled" if train_dataset.cache else "disabled", Colors.GREEN)

    # Load TEST data (Session 2 Finetune - completely held out)
    print(colored("\n  Loading test data (Session 2 Finetune)...", Colors.DIM))
    with Timer("test_data_loading", print_on_exit=True):
        test_dataset = load_subject_data(
            data_root, subject_id,
            session_folders=task_patterns['test'],
            target_classes=target_classes,
            config=preprocess_config,
            elc_path=elc_path
        )

    if len(test_dataset) == 0:
        print(colored(f"  WARNING: No test data found for subject {subject_id}", Colors.YELLOW))

    print_metric("Test segments", len(test_dataset) if test_dataset else 0, Colors.MAGENTA)

    # ========== DATA SPLITTING (Temporal) ==========
    print_section_header("Data Splitting (Temporal - Last 20% for Validation)")

    with Timer("data_splitting", print_on_exit=True):
        unique_trials = train_dataset.get_unique_trials()
        n_trials = len(unique_trials)
        print_metric("Total training trials", n_trials, Colors.CYAN)

        # STRATIFIED temporal split: split within each session to ensure similar distributions
        # This prevents validation set from being 100% one session type
        from collections import defaultdict

        # Group trials by session
        session_to_trials = defaultdict(list)

        for trial_idx in unique_trials:
            # Find which session this trial belongs to
            for info in train_dataset.trial_infos:
                if info.trial_idx == trial_idx:
                    session_to_trials[info.session_type].append(trial_idx)
                    break

        # For each session, split temporally (80/20)
        val_ratio = 0.2
        train_trials = []
        val_trials = []

        for session_type, trials in session_to_trials.items():
            # Sort trials by index (chronological order within session)
            trials = sorted(set(trials))
            n_trials_session = len(trials)
            n_val = max(1, int(n_trials_session * val_ratio))

            # Temporal split within this session
            train_trials.extend(trials[:-n_val])
            val_trials.extend(trials[-n_val:])

        # Get segment indices
        train_indices = train_dataset.get_segment_indices_for_trials(train_trials)
        val_indices = train_dataset.get_segment_indices_for_trials(val_trials)

    print_metric("Train trials", len(train_trials), Colors.GREEN)
    print_metric("Val trials", len(val_trials), Colors.YELLOW)
    print_metric("Train segments", len(train_indices), Colors.GREEN)
    print_metric("Val segments", len(val_indices), Colors.YELLOW)

    # ========== DATALOADER CREATION ==========
    print_section_header("DataLoader Creation")

    with Timer("dataloader_creation", print_on_exit=True):
        # Training data: shuffle enabled for better generalization
        train_loader, val_loader = create_data_loaders_from_dataset(
            train_dataset, train_indices, val_indices,
            batch_size=config['training']['batch_size'],
            num_workers=0,
            shuffle_train=True,  # Shuffle for better gradient estimation
        )

    # Get input dimensions
    sample_segment, _ = train_dataset[0]
    n_channels = sample_segment.shape[0]
    n_samples = sample_segment.shape[1]

    print_metric("Batch size", config['training']['batch_size'], Colors.CYAN)
    print_metric("Input shape", f"[{n_channels}, {n_samples}]", Colors.CYAN)
    print_metric("Train batches", len(train_loader), Colors.GREEN)
    print_metric("Val batches", len(val_loader), Colors.YELLOW)
    print(colored("  Training order: Shuffled (random order per epoch)", Colors.DIM))

    # ========== MODEL CREATION ==========
    print_section_header("Model Creation")

    with Timer("model_creation", print_on_exit=True):
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

    print_metric("Model", model_name, Colors.CYAN)
    print_metric("Parameters", f"{model.count_parameters():,}", Colors.CYAN)
    print_metric("Device", str(device), Colors.GREEN)

    # ========== TF32 矩阵乘法优化 ==========
    # TF32 在 Ampere+ GPU 上提供更快的矩阵乘法，精度损失可忽略
    if device.type == 'cuda' and hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')
        print_metric("TF32 matmul", "enabled (high precision)", Colors.GREEN)

    # ========== MODEL COMPILATION (PyTorch 2.0+) ==========
    # torch.compile() requires Triton which is only available on Linux
    # Skip compilation on Windows and Blackwell GPUs (sm_120+) to avoid compatibility issues
    import platform
    use_compile = config.get('training', {}).get('use_compile', True)
    is_windows = platform.system() == 'Windows'
    is_blackwell = is_blackwell_gpu()

    if is_windows:
        print_metric("torch.compile", "skipped (Windows)", Colors.DIM)
    elif is_blackwell:
        print_metric("torch.compile", "skipped (Blackwell GPU)", Colors.DIM)
    elif use_compile and hasattr(torch, 'compile') and device.type == 'cuda':
        try:
            compile_mode = 'reduce-overhead' if model_type == 'eegnet' else 'default'
            model = torch.compile(model, mode=compile_mode)
            print_metric("torch.compile", f"enabled ({compile_mode})", Colors.GREEN)
        except Exception as e:
            log_model.warning(f"torch.compile failed: {e}")
            print_metric("torch.compile", "failed (fallback to eager)", Colors.YELLOW)
    else:
        print_metric("torch.compile", "disabled", Colors.DIM)

    # ========== TRAINER SETUP ==========
    with Timer("trainer_setup", print_on_exit=True):
        # Get training config (model-specific defaults may override)
        train_config = config['training']
        learning_rate = train_config.get('learning_rate', 1e-3)
        weight_decay = train_config.get('weight_decay', 0.0)
        scheduler_type = train_config.get('scheduler', None)

        # CBraMod specific overrides
        if model_type == 'cbramod':
            learning_rate = train_config.get('backbone_lr', 1e-4)
            weight_decay = train_config.get('weight_decay', 0.05)
            scheduler_type = 'cosine'

        # Performance optimizations
        use_amp = True  # Enable AMP for both models
        gradient_clip = 1.0 if model_type == 'cbramod' else 0.0  # Clip for CBraMod only

        trainer = WithinSubjectTrainer(
            model, train_dataset, val_indices, device,
            model_type=model_type,
            n_classes=n_classes,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            scheduler_type=scheduler_type,
            use_amp=use_amp,
            gradient_clip=gradient_clip,
        )

    # Create save directory
    subject_save_dir = save_dir / subject_id
    subject_save_dir.mkdir(parents=True, exist_ok=True)

    # ========== TRAINING ==========
    with Timer("training"):
        history = trainer.train(
            train_loader, val_loader,
            epochs=config['training']['epochs'],
            patience=config['training']['patience'],
            save_path=subject_save_dir,
            wandb_callback=wandb_callback,
        )

    # ========== FINAL EVALUATION ==========
    print_section_header("Final Evaluation (Three-Phase Protocol)")

    # Evaluate on validation set (from training data)
    with Timer("val_evaluation", print_on_exit=True):
        val_acc, val_results = majority_vote_accuracy(
            model, train_dataset, val_indices, device
        )

    val_color = Colors.BRIGHT_GREEN if val_acc > 0.7 else (Colors.YELLOW if val_acc > 0.5 else Colors.RED)
    print(f"\n  {colored('Validation Accuracy (last 20% of train):', Colors.WHITE)} "
          f"{colored(f'{val_acc:.4f}', val_color)}")

    # Evaluate on TEST set (Online_Finetune - Phase 3)
    test_acc = 0.0
    test_results = {}
    n_test_trials = 0

    if len(test_dataset) > 0:
        with Timer("test_evaluation", print_on_exit=True):
            # Get all test indices
            test_indices = list(range(len(test_dataset)))
            test_acc, test_results = majority_vote_accuracy(
                model, test_dataset, test_indices, device
            )
            n_test_trials = len(test_dataset.get_unique_trials())

        test_color = Colors.BRIGHT_GREEN if test_acc > 0.7 else (Colors.YELLOW if test_acc > 0.5 else Colors.RED)
        print(f"  {colored('TEST Accuracy (Online_Finetune):', Colors.WHITE, bold=True)} "
              f"{colored(f'{test_acc:.4f}', test_color, bold=True)}")
        print(f"  {colored(f'Test trials: {n_test_trials}', Colors.DIM)}")
    else:
        print(colored("  No test data available (Online_Finetune)", Colors.YELLOW))

    # ========== TIMING SUMMARY ==========
    total_time = time.perf_counter() - total_start
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
        'test_accuracy': test_acc,  # This is the main metric (Phase 3)
        'test_accuracy_majority': test_acc,  # Alias for compatibility with run_full_comparison
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


def main(args):
    # Setup logging with yellow formatter
    handler = logging.StreamHandler()
    handler.setFormatter(YellowFormatter('train'))
    logging.root.handlers = [handler]
    logging.root.setLevel(logging.INFO)

    # Check CUDA
    if args.device == 'cuda':
        check_cuda_available(required=True)
    device = get_device(allow_cpu=(args.device == 'cpu'))
    log_train.info(f"Device: {device}")

    # Load configuration
    try:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        log_train.error(f"Config not found: {args.config}")
        log_train.info("Available: configs/eegnet_config.yaml, configs/cbramod_config.yaml")
        sys.exit(1)
    except yaml.YAMLError as e:
        log_train.error(f"Invalid YAML: {e}")
        sys.exit(1)

    config['task'] = args.task
    paradigm_desc = "MI" if args.paradigm == 'imagery' else "ME"
    log_train.info(f"Paradigm: {paradigm_desc} | Task: {args.task}")

    # Determine model types to train
    if args.model == 'both':
        model_types = ['eegnet', 'cbramod']
    else:
        model_types = [args.model]

    log_model.info(f"Models: {model_types}")

    # Paths
    data_root = PROJECT_ROOT / 'data'
    elc_path = data_root / 'biosemi128.ELC'

    # Determine subjects to train
    if args.all_subjects:
        # Find all subjects with required data (including test set)
        subjects = discover_available_subjects(str(data_root), args.paradigm, args.task)
    else:
        subjects = [args.subject]

    log_data.info(f"Subjects: {subjects}")

    # Train each combination of model and subject
    all_results = {}
    for model_type in model_types:
        log_train.info(f"{'='*40} {model_type.upper()} {'='*40}")

        # Load model-specific config (auto-select based on model type)
        if args.config == 'configs/eegnet_config.yaml' and model_type == 'cbramod':
            # User didn't specify config, use model-appropriate default
            model_config_path = PROJECT_ROOT / 'configs' / 'cbramod_config.yaml'
            log_model.info(f"Auto-config: {model_config_path.name}")
        elif args.config == 'configs/cbramod_config.yaml' and model_type == 'eegnet':
            model_config_path = PROJECT_ROOT / 'configs' / 'eegnet_config.yaml'
            log_model.info(f"Auto-config: {model_config_path.name}")
        else:
            model_config_path = args.config

        with open(model_config_path) as f:
            model_config = yaml.safe_load(f)
        model_config['task'] = args.task

        save_dir = PROJECT_ROOT / 'checkpoints' / f'{model_type}_within_subject' / args.task
        model_results = {}

        for subject_id in subjects:
            try:
                results = train_single_subject(
                    subject_id=subject_id,
                    config=model_config,
                    data_root=data_root,
                    elc_path=elc_path,
                    save_dir=save_dir,
                    device=device,
                    model_type=model_type,
                    paradigm=args.paradigm,
                    cbramod_channels=args.cbramod_channels,
                    # WandB parameters
                    no_wandb=args.no_wandb,
                    wandb_project=args.wandb_project,
                    wandb_entity=args.wandb_entity,
                    wandb_group=args.wandb_group,
                )
                model_results[subject_id] = results

                if results:
                    test_acc = results.get('test_accuracy', 0)
                    val_acc = results.get('val_accuracy', results.get('final_accuracy', 0))
                    log_eval.info(f"{subject_id}: Test={test_acc:.4f} Val={val_acc:.4f}")
            except Exception as e:
                log_train.error(f"{subject_id} {model_type} failed: {e}")
                import traceback
                traceback.print_exc()

        all_results[model_type] = model_results

    # Summary per model type
    for model_type, model_results in all_results.items():
        test_accuracies = [r.get('test_accuracy', 0) for r in model_results.values() if r]
        val_accuracies = [r.get('val_accuracy', r.get('final_accuracy', 0)) for r in model_results.values() if r]

        if len(test_accuracies) > 1:
            log_eval.info(f"{'='*50}")
            log_eval.info(f"{model_type.upper()} Summary (n={len(test_accuracies)})")
            log_eval.info(f"  TEST: {np.mean(test_accuracies):.4f}+/-{np.std(test_accuracies):.4f} [{np.min(test_accuracies):.4f}-{np.max(test_accuracies):.4f}]")
            log_eval.info(f"  VAL:  {np.mean(val_accuracies):.4f}")

        # Save summary per model
        save_dir = PROJECT_ROOT / 'checkpoints' / f'{model_type}_within_subject' / args.task
        summary_path = save_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump({
                'task': args.task,
                'model': model_type,
                'protocol': 'three_phase',
                'subjects': list(model_results.keys()),
                # Test accuracies (Phase 3 - main metric)
                'test_accuracies': {s: r.get('test_accuracy', None) for s, r in model_results.items()},
                'mean_test_accuracy': float(np.mean(test_accuracies)) if test_accuracies else None,
                'std_test_accuracy': float(np.std(test_accuracies)) if len(test_accuracies) > 1 else None,
                # Val accuracies (for reference)
                'val_accuracies': {s: r.get('val_accuracy', None) for s, r in model_results.items()},
                'mean_val_accuracy': float(np.mean(val_accuracies)) if val_accuracies else None,
                # Backwards compatibility
                'accuracies': {s: r.get('test_accuracy', r.get('final_accuracy', None)) for s, r in model_results.items()},
                'mean_accuracy': float(np.mean(test_accuracies)) if test_accuracies else None,
                'std_accuracy': float(np.std(test_accuracies)) if len(test_accuracies) > 1 else None,
            }, f, indent=2)

        log_train.info(f"{model_type.upper()} saved: {save_dir}")


def get_default_config(model_type: str, task: str) -> dict:
    """
    Get default configuration for a model type and task.

    This function provides hardcoded default configurations for programmatic use,
    enabling training without loading YAML configuration files. These defaults
    are aligned with the values in configs/eegnet_config.yaml and
    configs/cbramod_config.yaml at the time of implementation.

    IMPORTANT: If you modify the YAML config files, these hardcoded values will
    NOT automatically update. For full config file support, use the CLI interface:
        uv run python -m src.training.train_within_subject --config <path>

    This function is primarily used by:
    - train_subject_simple(): Simplified API for external callers
    - scripts/run_full_comparison.py: Batch training without config files

    Args:
        model_type: 'eegnet' or 'cbramod'
        task: 'binary', 'ternary', or 'quaternary'

    Returns:
        Configuration dict compatible with train_single_subject()

    Example:
        >>> config = get_default_config('eegnet', 'binary')
        >>> config['training']['epochs']
        300
    """
    # Task configurations
    tasks = {
        'binary': {'classes': [1, 4], 'n_classes': 2},
        'ternary': {'classes': [1, 2, 4], 'n_classes': 3},
        'quaternary': {'classes': [1, 2, 3, 4], 'n_classes': 4},
    }

    if model_type == 'cbramod':
        config = {
            'model': {
                'name': 'CBraMod',
                'classifier_type': 'two_layer',
                'dropout_rate': 0.1,
                'freeze_backbone': False,
            },
            'training': {
                'epochs': 50,
                'batch_size': 128,
                'learning_rate': 1e-4,
                'backbone_lr': 1e-4,
                'classifier_lr': 5e-4,
                'weight_decay': 0.05,
                'patience': 5,
                'scheduler': 'cosine',
            },
            'data': {},
            'tasks': tasks,
            'task': task,
        }
    else:  # eegnet
        config = {
            'model': {
                'name': 'EEGNet-8,2',
                'F1': 8,
                'D': 2,
                'F2': 16,
                'kernel_length': 64,
                'dropout_rate': 0.5,
            },
            'training': {
                'epochs': 300,
                'batch_size': 64,
                'learning_rate': 1e-3,
                'weight_decay': 0,
                'patience': 5,
                'scheduler': 'plateau',  # ReduceLROnPlateau (official impl)
            },
            'data': {},
            'tasks': tasks,
            'task': task,
        }

    return config


def train_subject_simple(
    subject_id: str,
    model_type: str,
    task: str,
    paradigm: str = 'imagery',
    data_root: str = 'data',
    save_dir: str = 'checkpoints',
    device: Optional[torch.device] = None,
    cbramod_channels: int = 128,
    # WandB parameters
    no_wandb: bool = False,
    wandb_project: str = 'eeg-bci',
    wandb_entity: Optional[str] = None,
    wandb_group: Optional[str] = None,
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
        no_wandb: Disable wandb logging
        wandb_project: WandB project name
        wandb_entity: WandB entity (team/username)
        wandb_group: WandB run group

    Returns:
        Results dict with keys:
        - subject, task, model_type, protocol
        - val_accuracy, best_val_acc, test_accuracy, test_accuracy_majority
        - epochs_trained, training_time, best_epoch, best_val_loss
        - history, val_evaluation, test_evaluation
    """
    if device is None:
        device = get_device()

    data_root_path = Path(data_root) if isinstance(data_root, str) else data_root
    elc_path = data_root_path / 'biosemi128.ELC'
    save_path = Path(save_dir) / f'{model_type}_within_subject' / task

    config = get_default_config(model_type, task)

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
        # WandB parameters
        no_wandb=no_wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_group=wandb_group,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Within-subject training for EEGNet and CBraMod (paper-aligned)'
    )

    parser.add_argument(
        '--config', type=str,
        default='configs/eegnet_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--subject', type=str,
        default='S01',
        help='Subject ID (e.g., S01)'
    )
    parser.add_argument(
        '--all-subjects', action='store_true',
        help='Train all subjects'
    )
    parser.add_argument(
        '--task', type=str,
        default='binary',
        choices=['binary', 'ternary', 'quaternary'],
        help='Classification task'
    )
    parser.add_argument(
        '--model', type=str,
        default='both',
        choices=['eegnet', 'cbramod', 'both'],
        help='Model type to train (eegnet, cbramod, or both)'
    )
    parser.add_argument(
        '--paradigm', type=str,
        default='imagery',
        choices=['imagery', 'movement'],
        help='Experiment paradigm: imagery (MI) or movement (ME) (default: imagery)'
    )
    parser.add_argument(
        '--device', type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--seed', type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--cbramod-channels', type=int,
        choices=[19, 128],
        default=128,
        help='Number of channels for CBraMod (19=10-20 system, 128=full BioSemi). '
             'Default 128 uses all channels with ACPE adaptation. Requires more GPU memory.'
    )

    # WandB arguments
    parser.add_argument(
        '--no-wandb', action='store_true',
        help='Disable wandb logging (default: enabled if wandb is installed)'
    )
    parser.add_argument(
        '--wandb-project', type=str, default='eeg-bci',
        help='WandB project name (default: eeg-bci)'
    )
    parser.add_argument(
        '--wandb-entity', type=str, default=None,
        help='WandB entity (team/username)'
    )
    parser.add_argument(
        '--wandb-group', type=str, default=None,
        help='WandB run group (for grouping multiple subjects)'
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    main(args)
