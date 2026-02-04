"""
Within-subject training module for FINGER-EEG-BCI.

Supports both EEGNet and CBraMod models, and both Motor Imagery (MI) and
Motor Execution (ME) paradigms.

Data Split (follows paper protocol):
- Training: Offline + Session 1 (Base + Finetune) + Session 2 Base
- Validation: Last 20% of training data (temporal split)
- Test: Session 2 Finetune (completely held out)

EEGNet:
- 128 channels, 100 Hz sampling rate
- 4-40 Hz bandpass filter (4th order Butterworth)
- Z-score normalization per segment
- 1-second sliding window with 125ms step
- Majority voting for trial prediction
- Training: 300 epochs with ReduceLROnPlateau

CBraMod:
- 19 or 128 channels, 200 Hz sampling rate
- 0.3-75 Hz bandpass, 60 Hz notch filter
- Divide by 100 normalization
- Patch-based processing (1s patches)
- Training: 50 epochs with cosine annealing

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
    uv run python scripts/run_full_comparison.py
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


class WSDScheduler:
    """
    Warmup-Stable-Decay (WSD) Learning Rate Scheduler.

    Four-phase learning rate schedule:
    1. Warmup: Linear increase from eta_min to peak_lr
    2. Stable: Maintain constant peak_lr (can be 0)
    3. Decay: Cosine decay from peak_lr to eta_min
    4. Minimum: Stay at eta_min for remaining steps

    This scheduler is designed for foundation model fine-tuning,
    as described in various papers including MiniCPM and similar works.

    Args:
        optimizer: PyTorch optimizer
        total_steps: Total number of training steps
        warmup_ratio: Fraction of total steps for warmup phase (default: 0.1)
        stable_ratio: Fraction of total steps for stable phase (default: 0)
        decay_ratio: Fraction of total steps for decay phase (default: 0.2)
        eta_min: Minimum learning rate at end of decay (default: 1e-6)

    Schedule:
        - Steps [0, warmup_steps): Linear warmup from eta_min to peak_lr
        - Steps [warmup_steps, warmup_steps + stable_steps): Constant at peak_lr
        - Steps [warmup+stable, warmup+stable+decay): Cosine decay to eta_min
        - Steps [warmup+stable+decay, total_steps): Constant at eta_min
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_ratio: float = 0.1,
        stable_ratio: float = 0.0,
        decay_ratio: float = 0.2,
        eta_min: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.eta_min = eta_min

        # Calculate phase boundaries
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.stable_steps = int(total_steps * stable_ratio)
        self.decay_steps = int(total_steps * decay_ratio)
        # Remaining steps stay at minimum
        self.min_steps = total_steps - self.warmup_steps - self.stable_steps - self.decay_steps

        # Store initial learning rates (peak LR for each param group)
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        self.current_step = 0
        self._last_lr = [0.0] * len(self.base_lrs)

        # Initialize to zero (warmup starts from 0)
        self._update_lr()

    def _get_lr_scale(self, step: int) -> float:
        """Calculate LR scale factor for a given step."""
        import math

        if step < self.warmup_steps:
            # Warmup phase: linear increase from 0 to 1
            return step / max(1, self.warmup_steps)

        elif step < self.warmup_steps + self.stable_steps:
            # Stable phase: constant at 1
            return 1.0

        elif step < self.warmup_steps + self.stable_steps + self.decay_steps:
            # Decay phase: cosine decay from 1 to 0
            decay_step = step - self.warmup_steps - self.stable_steps
            progress = decay_step / max(1, self.decay_steps)
            return (1 + math.cos(math.pi * progress)) / 2

        else:
            # Minimum phase: stay at eta_min
            return 0.0

    def _update_lr(self):
        """Update learning rates in optimizer."""
        scale = self._get_lr_scale(self.current_step)

        for i, (group, base_lr) in enumerate(zip(self.optimizer.param_groups, self.base_lrs)):
            # Scale between eta_min and base_lr
            new_lr = self.eta_min + (base_lr - self.eta_min) * scale
            group['lr'] = new_lr
            self._last_lr[i] = new_lr

    def step(self):
        """Advance scheduler by one step."""
        self.current_step += 1
        self._update_lr()

    def get_last_lr(self) -> List[float]:
        """Return last computed learning rates."""
        return self._last_lr

    def get_phase(self) -> str:
        """Return current phase name: 'warmup', 'stable', 'decay', or 'minimum'."""
        if self.current_step < self.warmup_steps:
            return 'warmup'
        elif self.current_step < self.warmup_steps + self.stable_steps:
            return 'stable'
        elif self.current_step < self.warmup_steps + self.stable_steps + self.decay_steps:
            return 'decay'
        else:
            return 'minimum'

    def get_phase_progress(self) -> Tuple[str, float]:
        """
        Return current phase and progress within that phase.

        Returns:
            Tuple of (phase_name, progress_ratio) where progress_ratio is 0.0-1.0
        """
        if self.current_step < self.warmup_steps:
            progress = self.current_step / max(1, self.warmup_steps)
            return ('warmup', progress)
        elif self.current_step < self.warmup_steps + self.stable_steps:
            progress = (self.current_step - self.warmup_steps) / max(1, self.stable_steps)
            return ('stable', progress)
        elif self.current_step < self.warmup_steps + self.stable_steps + self.decay_steps:
            decay_step = self.current_step - self.warmup_steps - self.stable_steps
            progress = decay_step / max(1, self.decay_steps)
            return ('decay', min(1.0, progress))
        else:
            min_step = self.current_step - self.warmup_steps - self.stable_steps - self.decay_steps
            progress = min_step / max(1, self.min_steps)
            return ('minimum', min(1.0, progress))

    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing."""
        return {
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'warmup_steps': self.warmup_steps,
            'stable_steps': self.stable_steps,
            'decay_steps': self.decay_steps,
            'min_steps': self.min_steps,
            'base_lrs': self.base_lrs,
            'eta_min': self.eta_min,
        }

    def load_state_dict(self, state_dict: dict):
        """Load scheduler state from checkpoint."""
        self.current_step = state_dict['current_step']
        self.total_steps = state_dict['total_steps']
        self.warmup_steps = state_dict['warmup_steps']
        self.stable_steps = state_dict['stable_steps']
        self.decay_steps = state_dict['decay_steps']
        self.min_steps = state_dict.get('min_steps', 0)
        self.base_lrs = state_dict['base_lrs']
        self.eta_min = state_dict['eta_min']
        self._update_lr()


class CosineDecayRestarts:
    """
    Cosine Annealing with Warm Restarts and Decaying Peaks.

    Unlike PyTorch's CosineAnnealingWarmRestarts which maintains the same peak LR
    after each restart, this scheduler reduces the peak LR by a decay factor
    after each cycle. This provides a more aggressive LR reduction over training.

    Schedule visualization (decay_factor=0.7, T_0=20):
        Cycle 0: LR oscillates between 1.0 and eta_min
        Cycle 1: LR oscillates between 0.7 and eta_min
        Cycle 2: LR oscillates between 0.49 and eta_min
        ...

    Args:
        optimizer: PyTorch optimizer
        T_0: Number of steps for the first (and subsequent) cycles
        decay_factor: Multiplicative factor to reduce peak LR after each cycle (default: 0.7)
        eta_min: Minimum learning rate (default: 1e-6)

    Example:
        >>> scheduler = CosineDecayRestarts(optimizer, T_0=100, decay_factor=0.7)
        >>> for step in range(500):
        ...     train(...)
        ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int,
        decay_factor: float = 0.7,
        eta_min: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.decay_factor = decay_factor
        self.eta_min = eta_min

        # Store initial learning rates (peak LR for each param group)
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        self.current_step = 0
        self._last_lr = list(self.base_lrs)

    def _get_lr_scale(self, step: int) -> float:
        """Calculate LR scale factor for a given step."""
        import math

        cycle = step // self.T_0
        step_in_cycle = step % self.T_0

        # Peak decays each cycle: 1.0, 0.7, 0.49, 0.343, ...
        peak = self.decay_factor ** cycle

        # Cosine within cycle: starts at peak, decays to eta_min ratio
        # cos(0) = 1 (start of cycle, at peak)
        # cos(pi) = -1 (end of cycle, at minimum)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * step_in_cycle / self.T_0))

        # Scale factor relative to base_lr
        # At cycle start: scale = peak
        # At cycle end: scale = eta_min / base_lr (approximately 0)
        return peak * cosine_factor

    def _update_lr(self):
        """Update learning rates in optimizer."""
        scale = self._get_lr_scale(self.current_step)

        for i, (group, base_lr) in enumerate(zip(self.optimizer.param_groups, self.base_lrs)):
            # Interpolate between eta_min and scaled base_lr
            new_lr = self.eta_min + (base_lr - self.eta_min) * scale
            group['lr'] = new_lr
            self._last_lr[i] = new_lr

    def step(self):
        """Advance scheduler by one step.

        PyTorch convention: step() is called AFTER optimizer.step(),
        and sets the LR for the NEXT training step.
        """
        self.current_step += 1
        self._update_lr()

    def get_last_lr(self) -> List[float]:
        """Return last computed learning rates."""
        return self._last_lr

    def get_cycle(self) -> int:
        """Return current cycle number (0-indexed)."""
        return self.current_step // self.T_0

    def get_cycle_progress(self) -> Tuple[int, float]:
        """
        Return current cycle and progress within that cycle.

        Returns:
            Tuple of (cycle_number, progress_ratio) where progress_ratio is 0.0-1.0
        """
        cycle = self.current_step // self.T_0
        step_in_cycle = self.current_step % self.T_0
        progress = step_in_cycle / self.T_0
        return (cycle, progress)

    def get_current_peak(self) -> float:
        """Return the peak LR for the current cycle."""
        cycle = self.current_step // self.T_0
        return self.base_lrs[0] * (self.decay_factor ** cycle)

    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing."""
        return {
            'current_step': self.current_step,
            'T_0': self.T_0,
            'decay_factor': self.decay_factor,
            'base_lrs': self.base_lrs,
            'eta_min': self.eta_min,
        }

    def load_state_dict(self, state_dict: dict):
        """Load scheduler state from checkpoint."""
        self.current_step = state_dict['current_step']
        self.T_0 = state_dict['T_0']
        self.decay_factor = state_dict['decay_factor']
        self.base_lrs = state_dict['base_lrs']
        self.eta_min = state_dict['eta_min']
        self._update_lr()


class CosineAnnealingWarmupDecay:
    """
    多阶段余弦退火调度器，带 warmup 和峰值衰减。

    每个阶段包含 warmup（从 eta_min 上升到峰值）+ 余弦衰减（从峰值下降到 eta_min）。
    每个新阶段的峰值学习率按 phase_decay 衰减。

    Schedule visualization (phase_length=6 epochs, phase_decay=0.7, 30 epochs total):
        Phase 0 (epochs 0-5):   peak = 100%, warmup + cosine decay
        Phase 1 (epochs 6-11):  peak = 70%,  warmup + cosine decay
        Phase 2 (epochs 12-17): peak = 49%,  warmup + cosine decay
        Phase 3 (epochs 18-23): peak = 34%,  warmup + cosine decay
        Phase 4 (epochs 24-29): peak = 24%,  warmup + cosine decay

    Args:
        optimizer: PyTorch optimizer
        total_steps: Total number of training steps
        steps_per_epoch: Steps per epoch (for calculating phase boundaries)
        phase_length: Number of epochs per phase (default: 6)
        phase_decay: Peak LR decay factor between phases (default: 0.7)
        warmup_ratio: Fraction of each phase for warmup (default: 0.1)
        eta_min: Minimum learning rate (default: 1e-6)

    Example:
        >>> scheduler = CosineAnnealingWarmupDecay(
        ...     optimizer,
        ...     total_steps=3000,
        ...     steps_per_epoch=100,
        ...     phase_length=6,
        ...     phase_decay=0.7,
        ... )
        >>> for step in range(3000):
        ...     train(...)
        ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        steps_per_epoch: int,
        phase_length: int = 6,
        phase_decay: float = 0.7,
        warmup_ratio: float = 0.1,
        eta_min: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.steps_per_epoch = steps_per_epoch
        self.phase_length = phase_length
        self.phase_decay = phase_decay
        self.warmup_ratio = warmup_ratio
        self.eta_min = eta_min

        # Store initial learning rates (peak LR for each param group)
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        # Calculate steps per phase
        self.steps_per_phase = steps_per_epoch * phase_length
        self.warmup_steps_per_phase = int(self.steps_per_phase * warmup_ratio)
        self.cosine_steps_per_phase = self.steps_per_phase - self.warmup_steps_per_phase

        # Calculate number of phases
        total_epochs = total_steps // steps_per_epoch
        self.num_phases = total_epochs // phase_length
        if total_epochs % phase_length != 0:
            self.num_phases += 1

        self.current_step = 0
        self._last_lr = list(self.base_lrs)

        # Initialize LR (start of first warmup)
        self._update_lr()

    def _get_phase_info(self, step: int) -> Tuple[int, int, float]:
        """
        Calculate phase information for a given step.

        Returns:
            (phase_idx, step_in_phase, peak_lr_scale)
        """
        phase_idx = min(step // self.steps_per_phase, self.num_phases - 1)
        step_in_phase = step % self.steps_per_phase

        # Last phase may have extra steps
        if phase_idx == self.num_phases - 1:
            step_in_phase = step - phase_idx * self.steps_per_phase

        # Peak LR scale factor for this phase
        peak_lr_scale = self.phase_decay ** phase_idx

        return phase_idx, step_in_phase, peak_lr_scale

    def _get_lr_scale(self, step: int) -> float:
        """Calculate LR scale factor for a given step (relative to base_lr)."""
        import math

        phase_idx, step_in_phase, peak_scale = self._get_phase_info(step)

        if step_in_phase < self.warmup_steps_per_phase:
            # Warmup phase: linear increase from ~0 to peak
            warmup_progress = step_in_phase / max(1, self.warmup_steps_per_phase)
            return warmup_progress * peak_scale
        else:
            # Cosine decay phase: from peak to ~0
            cosine_step = step_in_phase - self.warmup_steps_per_phase
            cosine_progress = cosine_step / max(1, self.cosine_steps_per_phase)
            # Cosine factor: 1 (start) -> 0 (end)
            cosine_factor = (1 + math.cos(math.pi * cosine_progress)) / 2
            return peak_scale * cosine_factor

    def _update_lr(self):
        """Update learning rates in optimizer."""
        scale = self._get_lr_scale(self.current_step)

        for i, (group, base_lr) in enumerate(zip(self.optimizer.param_groups, self.base_lrs)):
            # Interpolate between eta_min and scaled base_lr
            new_lr = self.eta_min + (base_lr - self.eta_min) * scale
            group['lr'] = new_lr
            self._last_lr[i] = new_lr

    def step(self):
        """Advance scheduler by one step."""
        self.current_step += 1
        self._update_lr()

    def get_last_lr(self) -> List[float]:
        """Return last computed learning rates."""
        return self._last_lr

    def get_phase(self) -> int:
        """Return current phase index (0-indexed)."""
        return min(self.current_step // self.steps_per_phase, self.num_phases - 1)

    def get_phase_progress(self) -> Tuple[int, float, str]:
        """
        Return current phase and progress within that phase.

        Returns:
            (phase_idx, progress_ratio, sub_phase_name)
            sub_phase_name: 'warmup' or 'cosine_decay'
        """
        phase_idx, step_in_phase, _ = self._get_phase_info(self.current_step)

        if step_in_phase < self.warmup_steps_per_phase:
            progress = step_in_phase / max(1, self.warmup_steps_per_phase)
            return (phase_idx, progress, 'warmup')
        else:
            cosine_step = step_in_phase - self.warmup_steps_per_phase
            progress = cosine_step / max(1, self.cosine_steps_per_phase)
            return (phase_idx, min(1.0, progress), 'cosine_decay')

    def get_current_peak(self) -> float:
        """Return the peak LR for the current phase."""
        phase_idx = self.get_phase()
        return self.base_lrs[0] * (self.phase_decay ** phase_idx)

    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing."""
        return {
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'steps_per_epoch': self.steps_per_epoch,
            'steps_per_phase': self.steps_per_phase,
            'warmup_steps_per_phase': self.warmup_steps_per_phase,
            'cosine_steps_per_phase': self.cosine_steps_per_phase,
            'num_phases': self.num_phases,
            'phase_length': self.phase_length,
            'phase_decay': self.phase_decay,
            'warmup_ratio': self.warmup_ratio,
            'base_lrs': self.base_lrs,
            'eta_min': self.eta_min,
        }

    def load_state_dict(self, state_dict: dict):
        """Load scheduler state from checkpoint."""
        self.current_step = state_dict['current_step']
        self.total_steps = state_dict['total_steps']
        self.steps_per_epoch = state_dict['steps_per_epoch']
        self.steps_per_phase = state_dict['steps_per_phase']
        self.warmup_steps_per_phase = state_dict['warmup_steps_per_phase']
        self.cosine_steps_per_phase = state_dict['cosine_steps_per_phase']
        self.num_phases = state_dict['num_phases']
        self.phase_length = state_dict['phase_length']
        self.phase_decay = state_dict['phase_decay']
        self.warmup_ratio = state_dict['warmup_ratio']
        self.base_lrs = state_dict['base_lrs']
        self.eta_min = state_dict['eta_min']
        self._update_lr()


class WithinSubjectTrainer:
    """
    Trainer for within-subject model training (EEGNet or CBraMod).

    Follows the paper's training protocol:
    - EEGNet: Pre-train on offline data for 50 epochs, Adam optimizer
    - CBraMod: Pre-train for 25 epochs, AdamW with different LR for backbone/classifier
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
        classifier_lr: Optional[float] = None,
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

        # Loss function - apply label smoothing for regularization
        # Default: 0.05 for CBraMod (moderate regularization)
        label_smoothing = 0.05 if model_type == 'cbramod' else 0.0
        if label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            log_model.info(f"Label smoothing={label_smoothing}")
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Create optimizer based on model type
        if model_type == 'cbramod' and hasattr(model, 'get_parameter_groups'):
            # CBraMod uses different LR for backbone and classifier
            # Default classifier_lr = 3x backbone_lr if not specified
            actual_classifier_lr = classifier_lr if classifier_lr is not None else learning_rate * 3
            param_groups = model.get_parameter_groups(
                backbone_lr=learning_rate,
                classifier_lr=actual_classifier_lr,
            )
            self.optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=weight_decay,
            )
            log_train.info(f"Optimizer: AdamW (backbone_lr={learning_rate}, classifier_lr={actual_classifier_lr})")
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
            # For CBraMod, scheduler is created in train() with correct T_max
            # Creating here would cause PyTorch warning about step order
            if model_type != 'cbramod':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=50,
                    eta_min=1e-6,
                )
        elif scheduler_type == 'plateau':
            # ReduceLROnPlateau - uses combined score (val_acc + majority_acc) / 2
            # mode='max' because we want to maximize the combined accuracy score
            # Note: 'verbose' parameter removed in PyTorch 2.3+
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',  # Maximize combined accuracy score
                factor=0.3,
                patience=2,
                min_lr=1e-6,
            )
            self.scheduler_needs_metric = True
            log_train.info("Scheduler: ReduceLROnPlateau (mode=max, factor=0.3, patience=2, metric=combined_score)")
        elif scheduler_type == 'wsd':
            # WSD scheduler will be created in train() when total_steps is known
            self.scheduler = None  # Placeholder, created in train()
            log_train.info("Scheduler: WSD (will be initialized in train())")
        elif scheduler_type == 'cosine_decay':
            # CosineDecayRestarts scheduler will be created in train() when total_steps is known
            self.scheduler = None  # Placeholder, created in train()
            log_train.info("Scheduler: CosineDecayRestarts (will be initialized in train())")
        elif scheduler_type == 'cosine_annealing_warmup_decay':
            # CosineAnnealingWarmupDecay scheduler will be created in train() when total_steps is known
            self.scheduler = None  # Placeholder, created in train()
            log_train.info("Scheduler: CosineAnnealingWarmupDecay (will be initialized in train())")

        # WSD-specific parameters (stored for later initialization)
        # Schedule: 5 epochs warmup -> 10 epochs decay -> rest at minimum
        self.wsd_warmup_ratio = 0.1   # 10% = 5 epochs (warmup)
        self.wsd_stable_ratio = 0.0   # 0% = no stable phase
        self.wsd_decay_ratio = 0.3    # 20% = 10 epochs (decay)

        # CosineDecayRestarts-specific parameters
        # With 30 epochs and T_0 = total_steps // 5, we get 5 cycles with decaying peaks
        self.cosine_decay_factor = 0.7  # Peak reduces by 30% each cycle
        self.cosine_decay_cycles = 5    # Number of cycles (T_0 = total_steps // cycles)

        # CosineAnnealingWarmupDecay-specific parameters
        # Each phase: warmup (10%) + cosine decay (90%), peak decays by 0.7 between phases
        self.cosine_warmup_phase_length = 6    # Epochs per phase
        self.cosine_warmup_phase_decay = 0.7   # Peak decay between phases (100% -> 70% -> 49%...)
        self.cosine_warmup_warmup_ratio = 0.1  # Warmup ratio within each phase

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
            'val_combined_score': [],  # (val_acc + majority_acc) / 2
        }
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0  # Track best validation accuracy (segment-level)
        self.best_majority_acc = 0.0  # Track best validation accuracy (trial-level majority voting)
        self.best_combined_score = 0.0  # Combined score = (val_acc + majority_acc) / 2
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
        main_train_loader: Optional[DataLoader] = None,
        warmup_epochs: int = 0,
        epochs: int = 50,
        patience: int = 10,
        save_path: Optional[Path] = None,
        wandb_callback: Optional['WandbCallback'] = None,
    ) -> Dict:
        """
        Full training loop with early stopping and two-phase batch size.

        Args:
            train_loader: Training DataLoader for warmup phase (small batch)
            val_loader: Validation DataLoader
            main_train_loader: Training DataLoader for main phase (normal batch).
                              If None, uses train_loader for all epochs.
            warmup_epochs: Number of epochs for warmup phase (small batch).
                          After this, switches to main_train_loader.
            epochs: Maximum epochs
            patience: Early stopping patience
            save_path: Path to save best model
            wandb_callback: Optional WandB callback for logging

        Returns:
            Training history
        """
        # Use single loader if main_train_loader not provided
        if main_train_loader is None:
            main_train_loader = train_loader
            warmup_epochs = 0

        phase_info = f", warmup={warmup_epochs}eps" if warmup_epochs > 0 else ""
        print_section_header(f"Training ({epochs} epochs, patience={patience}{phase_info})")

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

        # Calculate total_steps for schedulers (account for two-phase batch size)
        # warmup phase uses train_loader (small batch), main phase uses main_train_loader
        warmup_steps = warmup_epochs * len(train_loader)
        main_steps = (epochs - warmup_epochs) * len(main_train_loader)
        total_steps = warmup_steps + main_steps

        # Recreate scheduler with correct T_max for per-step scheduling (CBraMod)
        # Use T_max = total_steps / 5 for faster LR decay (reaches min at 50% of training)
        if self.scheduler_type == 'cosine' and self.model_type == 'cbramod':
            t_max = total_steps // 5  # Faster decay: reach min LR at 20% of training
            # Log T_max overwrite (config value is ignored, hardcoded to total_steps // 5)
            log_train.info(
                f"{Colors.BRIGHT_YELLOW}T_max overwrite: "
                f"config value ignored -> {t_max} (total_steps // 5, CBraMod hardcoded){Colors.RESET}"
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
                eta_min=1e-6,
            )
            log_train.info(f"Scheduler: CosineAnnealing (T_max={t_max}, fast decay)")

        # Create WSD scheduler for CBraMod
        if self.scheduler_type == 'wsd' and self.model_type == 'cbramod':
            wsd_warmup_steps = int(total_steps * self.wsd_warmup_ratio)
            wsd_stable_steps = int(total_steps * self.wsd_stable_ratio)
            wsd_decay_steps = total_steps - wsd_warmup_steps - wsd_stable_steps
            log_train.info(
                f"{Colors.BRIGHT_YELLOW}WSD Scheduler: "
                f"warmup={wsd_warmup_steps} ({self.wsd_warmup_ratio*100:.0f}%) | "
                f"stable={wsd_stable_steps} ({self.wsd_stable_ratio*100:.0f}%) | "
                f"decay={wsd_decay_steps} ({(1-self.wsd_warmup_ratio-self.wsd_stable_ratio)*100:.0f}%)"
                f"{Colors.RESET}"
            )
            self.scheduler = WSDScheduler(
                self.optimizer,
                total_steps=total_steps,
                warmup_ratio=self.wsd_warmup_ratio,
                stable_ratio=self.wsd_stable_ratio,
                decay_ratio=self.wsd_decay_ratio,
                eta_min=1e-6,
            )
            log_train.info(f"Scheduler: WSD (total_steps={total_steps}, warmup={self.wsd_warmup_ratio}, decay={self.wsd_decay_ratio})")

        # Create CosineDecayRestarts scheduler for CBraMod
        if self.scheduler_type == 'cosine_decay' and self.model_type == 'cbramod':
            t_0 = total_steps // self.cosine_decay_cycles  # Cycle length
            log_train.info(
                f"{Colors.BRIGHT_YELLOW}CosineDecayRestarts Scheduler: "
                f"T_0={t_0} ({100/self.cosine_decay_cycles:.0f}% per cycle) | "
                f"decay_factor={self.cosine_decay_factor} | "
                f"cycles={self.cosine_decay_cycles}"
                f"{Colors.RESET}"
            )
            self.scheduler = CosineDecayRestarts(
                self.optimizer,
                T_0=t_0,
                decay_factor=self.cosine_decay_factor,
                eta_min=1e-6,
            )
            # Show peak LR progression
            peaks = [self.cosine_decay_factor ** i for i in range(self.cosine_decay_cycles)]
            peak_str = " -> ".join([f"{p:.2f}" for p in peaks])
            log_train.info(f"Scheduler: CosineDecayRestarts (peak progression: {peak_str})")

        # Create CosineAnnealingWarmupDecay scheduler for CBraMod
        if self.scheduler_type == 'cosine_annealing_warmup_decay' and self.model_type == 'cbramod':
            # Use main_train_loader steps for calculation (main phase is dominant)
            steps_per_epoch = len(main_train_loader)
            num_phases = epochs // self.cosine_warmup_phase_length
            if epochs % self.cosine_warmup_phase_length != 0:
                num_phases += 1  # Handle non-divisible case

            log_train.info(
                f"{Colors.BRIGHT_YELLOW}CosineAnnealingWarmupDecay Scheduler: "
                f"phase_length={self.cosine_warmup_phase_length} epochs | "
                f"num_phases={num_phases} | "
                f"phase_decay={self.cosine_warmup_phase_decay} | "
                f"warmup_ratio={self.cosine_warmup_warmup_ratio}"
                f"{Colors.RESET}"
            )
            self.scheduler = CosineAnnealingWarmupDecay(
                self.optimizer,
                total_steps=total_steps,
                steps_per_epoch=steps_per_epoch,
                phase_length=self.cosine_warmup_phase_length,
                phase_decay=self.cosine_warmup_phase_decay,
                warmup_ratio=self.cosine_warmup_warmup_ratio,
                eta_min=1e-6,
            )
            # Show peak LR progression
            peaks = [self.cosine_warmup_phase_decay ** i for i in range(num_phases)]
            peak_str = " -> ".join([f"{p:.0%}" for p in peaks])
            log_train.info(f"Scheduler: CosineAnnealingWarmupDecay (peak progression: {peak_str})")

        for epoch in range(epochs):
            epoch_timer.start_epoch()

            # Select train loader based on epoch (two-phase batch size)
            if epoch < warmup_epochs:
                current_train_loader = train_loader  # Small batch (warmup)
            else:
                current_train_loader = main_train_loader  # Normal batch (main)

            # Train (profile only first epoch to diagnose bottlenecks)
            do_profile = (epoch == 0)
            with epoch_timer.phase("train"):
                train_loss, train_acc = self.train_epoch(current_train_loader, profile=do_profile)

            # Validate
            with epoch_timer.phase("validate"):
                val_loss, val_acc = self.validate(val_loader)

            # Majority voting: compute every epoch for accurate early stopping
            with epoch_timer.phase("majority_vote"):
                majority_acc, _ = majority_vote_accuracy(
                    self.model, self.dataset, self.val_indices, self.device,
                    use_amp=self.use_amp
                )

            epoch_timer.end_epoch()

            # Combined score: average of segment accuracy and majority voting accuracy
            # Early stopping and best model selection based on this combined metric
            combined_score = (val_acc + majority_acc) / 2.0

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_majority_acc'].append(majority_acc)
            self.history['val_combined_score'].append(combined_score)

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

            # Update scheduler (only for EEGNet - CBraMod uses per-step scheduling in train_epoch)
            # ReduceLROnPlateau uses combined_score (mode='max') to decide LR reduction
            if self.scheduler is not None and self.model_type != 'cbramod':
                if self.scheduler_needs_metric:
                    self.scheduler.step(combined_score)  # ReduceLROnPlateau uses combined score
                else:
                    self.scheduler.step()

            if combined_score > self.best_combined_score:
                self.best_combined_score = combined_score
                self.best_val_acc = val_acc
                self.best_majority_acc = majority_acc
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
                        'combined_score': self.best_combined_score,
                        'val_loss': self.best_val_loss,
                    }, save_path / 'best.pt')
                    log_train.debug(f"Best model saved (combined={combined_score:.4f}, val_acc={val_acc:.4f}, maj_acc={majority_acc:.4f})")
            else:
                no_improve += 1

            # Check if early stopping will trigger
            will_stop = no_improve >= patience

            # Determine event: BEST takes priority, then STOP
            if is_best_epoch:
                event = "BEST"
            elif will_stop:
                event = "STOP"
            else:
                event = None

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
                event=event,
            )

            # Early stopping check
            if will_stop:
                break

        # Restore best model (prefer disk checkpoint if available)
        if save_path and (save_path / 'best.pt').exists():
            checkpoint = torch.load(save_path / 'best.pt', map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            log_train.info(f"Loaded best (combined_score={checkpoint.get('combined_score', 'N/A')})")
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
    wandb_interactive: bool = False,
    wandb_metadata: Optional[Dict[str, str]] = None,
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
        wandb_interactive: Prompt for run details interactively
        wandb_metadata: Pre-collected metadata (goal, hypothesis, notes) for batch training

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
        interactive=wandb_interactive,
        metadata=wandb_metadata,
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

    # Select preprocessing config based on model type (unless custom config provided)
    if preprocess_config is not None:
        # Use custom config (e.g., from ML engineering experiments)
        log_data.info(f"Preprocess: Custom config ({preprocess_config.target_fs}Hz, "
                      f"{preprocess_config.bandpass_low}-{preprocess_config.bandpass_high}Hz)")
        print(colored(f"  Custom preprocessing config provided", Colors.CYAN))
    elif model_type == 'cbramod':
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
            elc_path=elc_path,
            cache_only=cache_only,
            cache_index_path=cache_index_path,
        )

    if len(train_dataset) == 0:
        print(colored(f"  ERROR: No training data found for subject {subject_id}", Colors.RED))
        return {}

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
    print(colored("\n  Loading test data (Session 2 Finetune)...", Colors.DIM))
    with Timer("test_data_loading", print_on_exit=True):
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
        print(colored(f"  WARNING: No test data found for subject {subject_id}", Colors.YELLOW))

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

    # Determine warmup epochs based on scheduler type
    scheduler_type = config['training'].get('scheduler', None)
    if scheduler_type == 'cosine_annealing_warmup_decay':
        warmup_epochs = 6  # phase_length for this scheduler
    elif scheduler_type == 'wsd':
        warmup_epochs = int(config['training']['epochs'] * 0.1)  # 10% warmup
    else:
        warmup_epochs = 5  # default

    main_batch_size = config['training']['batch_size']
    warmup_batch_size = 32  # Small batch for warmup phase

    with Timer("dataloader_creation", print_on_exit=True):
        # Create warmup loader (small batch for exploration)
        warmup_train_loader, val_loader = create_data_loaders_from_dataset(
            train_dataset, train_indices, val_indices,
            batch_size=warmup_batch_size,
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

    print_metric("Warmup batch size", f"{warmup_batch_size} (epochs 1-{warmup_epochs})", Colors.CYAN)
    print_metric("Main batch size", f"{main_batch_size} (epochs {warmup_epochs+1}+)", Colors.CYAN)
    print_metric("Input shape", f"[{n_channels}, {n_samples}]", Colors.CYAN)
    print_metric("Warmup batches/epoch", len(warmup_train_loader), Colors.GREEN)
    print_metric("Main batches/epoch", len(main_train_loader), Colors.GREEN)
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
        gradient_clip = 1.0 if model_type == 'cbramod' else 0.0  # Clip for CBraMod only

        trainer = WithinSubjectTrainer(
            model, train_dataset, val_indices, device,
            model_type=model_type,
            n_classes=n_classes,
            learning_rate=learning_rate,
            classifier_lr=classifier_lr,
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
            warmup_train_loader, val_loader,
            main_train_loader=main_train_loader,
            warmup_epochs=warmup_epochs,
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
        'best_combined_score': trainer.best_combined_score,  # (val_acc + majority_acc) / 2 at best epoch
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


def get_default_config(model_type: str, task: str) -> dict:
    """
    Get default configuration for a model type and task.

    This function provides the canonical training configurations used by all
    training scripts. These are the single source of truth for model hyperparameters.

    Used by:
    - train_subject_simple(): Simplified API for external callers
    - scripts/run_full_comparison.py: Batch training
    - scripts/run_single_model.py: Single model training

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
                'dropout_rate': 0.15,  # Slightly higher than v1 (0.1) for mild regularization
                'freeze_backbone': False,
            },
            'training': {
                'epochs': 50,
                'batch_size': 128,
                'learning_rate': 1e-4,  # Restored to v1 value - crucial for hard subjects
                'backbone_lr': 1e-4,
                'classifier_lr': 3e-4,  # 3x backbone
                'weight_decay': 0.06,  # Slightly higher than v1 (0.05)
                'patience': 10,
                'scheduler': 'cosine_annealing_warmup_decay',
                'label_smoothing': 0.05,
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
                'epochs': 30,
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
    wandb_interactive: bool = False,
    wandb_metadata: Optional[Dict[str, str]] = None,
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
        wandb_interactive: Prompt for run details interactively
        wandb_metadata: Pre-collected metadata (goal, hypothesis, notes) for batch training

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

    # Apply config overrides (for scheduler comparison experiments)
    if config_overrides:
        for section, overrides in config_overrides.items():
            if section in config and isinstance(overrides, dict):
                config[section].update(overrides)
            else:
                config[section] = overrides

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
        wandb_interactive=wandb_interactive,
        wandb_metadata=wandb_metadata,
    )
