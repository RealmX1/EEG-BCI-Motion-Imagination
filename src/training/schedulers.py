"""
Learning rate schedulers for EEG-BCI training.

This module provides custom LR schedulers optimized for EEG classification:
- WSDScheduler: Warmup-Stable-Decay schedule for foundation model fine-tuning
- CosineDecayRestarts: Cosine annealing with decaying peak LR
- CosineAnnealingWarmupDecay: Multi-phase cosine with LR ramp-up

Also includes visualize_lr_schedule() for schedule visualization.
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..utils.logging import SectionLogger

logger = logging.getLogger(__name__)
log_train = SectionLogger(logger, 'train')


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
    多阶段余弦退火调度器，带 LR ramp-up 和峰值衰减。

    基于 EPOCH 计算 phase 边界（而非 step），确保 phase 边界始终在正确的 epoch 位置，
    不受 batch size 变化（如 exploration phase）影响。

    每个 phase 包含:
    - LR ramp-up (前 lr_ramp_ratio): 从 eta_min 线性上升到峰值
    - Cosine decay (后 1-lr_ramp_ratio): 从峰值余弦下降到 eta_min

    每个新 phase 的峰值学习率按 phase_decay 衰减。

    Schedule visualization (phase_epochs=6, phase_decay=0.7, 30 epochs total):
        Phase 0 (epochs 0-5):   peak = 100%, ramp-up + cosine decay
        Phase 1 (epochs 6-11):  peak = 70%,  ramp-up + cosine decay
        Phase 2 (epochs 12-17): peak = 49%,  ramp-up + cosine decay
        Phase 3 (epochs 18-23): peak = 34%,  ramp-up + cosine decay
        Phase 4 (epochs 24-29): peak = 24%,  ramp-up + cosine decay

    Args:
        optimizer: PyTorch optimizer
        total_epochs: Total number of training epochs
        phase_epochs: Number of epochs per phase (default: 6)
        phase_decay: Peak LR decay factor between phases (default: 0.7)
        lr_ramp_ratio: Fraction of each phase for LR ramp-up (default: 0.1)
        eta_min: Minimum learning rate (default: 1e-6)

    Example:
        >>> scheduler = CosineAnnealingWarmupDecay(
        ...     optimizer,
        ...     total_epochs=50,
        ...     phase_epochs=6,
        ...     phase_decay=0.7,
        ... )
        >>> for epoch in range(50):
        ...     for batch_idx, batch in enumerate(train_loader):
        ...         train(batch)
        ...         scheduler.step(epoch, batch_idx, len(train_loader))
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_epochs: int,
        phase_epochs: int = 6,
        phase_decay: float = 0.7,
        lr_ramp_ratio: float = 0.1,
        eta_min: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.phase_epochs = phase_epochs
        self.phase_decay = phase_decay
        self.lr_ramp_ratio = lr_ramp_ratio
        self.eta_min = eta_min

        # Store initial learning rates (peak LR for each param group)
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        # Calculate number of phases
        self.num_phases = total_epochs // phase_epochs
        if total_epochs % phase_epochs != 0:
            self.num_phases += 1

        # Track current position (for state_dict and get_phase)
        self.current_epoch = 0
        self.current_step_in_epoch = 0
        self.current_steps_in_epoch = 1  # Default to 1 to avoid div by zero

        self._last_lr = list(self.base_lrs)

        # Initialize LR (start of first ramp-up)
        self._update_lr()

    def get_phase_for_epoch(self, epoch: int) -> int:
        """
        Get the phase index for a given epoch.

        Args:
            epoch: Current epoch (0-indexed)

        Returns:
            Phase index (0-indexed)
        """
        return min(epoch // self.phase_epochs, self.num_phases - 1)

    def _get_progress_in_phase(self, epoch: int, step_in_epoch: int, steps_in_epoch: int) -> float:
        """
        Calculate progress within the current phase (0.0 to 1.0).

        This is the key function that makes phase boundaries independent of batch size.
        Progress is calculated purely from epoch position, with step providing sub-epoch precision.

        Args:
            epoch: Current epoch (0-indexed)
            step_in_epoch: Current step within the epoch (0-indexed)
            steps_in_epoch: Total steps in this epoch

        Returns:
            Progress ratio within the phase (0.0 to 1.0)
        """
        phase_idx = self.get_phase_for_epoch(epoch)
        epoch_in_phase = epoch - phase_idx * self.phase_epochs

        # Sub-epoch progress (0.0 to 1.0 within current epoch)
        epoch_progress = step_in_epoch / max(1, steps_in_epoch)

        # Total progress within phase
        progress = (epoch_in_phase + epoch_progress) / self.phase_epochs

        return min(1.0, progress)

    def _get_lr_scale(self, epoch: int, step_in_epoch: int, steps_in_epoch: int) -> float:
        """
        Calculate LR scale factor (relative to base_lr) for given position.

        Args:
            epoch: Current epoch (0-indexed)
            step_in_epoch: Current step within the epoch
            steps_in_epoch: Total steps in this epoch

        Returns:
            LR scale factor (0.0 to 1.0)
        """
        phase_idx = self.get_phase_for_epoch(epoch)
        progress = self._get_progress_in_phase(epoch, step_in_epoch, steps_in_epoch)

        # Peak LR scale for this phase
        peak_scale = self.phase_decay ** phase_idx

        if progress < self.lr_ramp_ratio:
            # LR ramp-up: linear increase from ~0 to peak
            ramp_progress = progress / self.lr_ramp_ratio
            return ramp_progress * peak_scale
        else:
            # Cosine decay: from peak to ~0
            decay_progress = (progress - self.lr_ramp_ratio) / (1.0 - self.lr_ramp_ratio)
            # Cosine factor: 1 (start) -> 0 (end)
            cosine_factor = (1 + math.cos(math.pi * decay_progress)) / 2
            return peak_scale * cosine_factor

    def _update_lr(self):
        """Update learning rates in optimizer based on current position."""
        scale = self._get_lr_scale(
            self.current_epoch,
            self.current_step_in_epoch,
            self.current_steps_in_epoch
        )

        for i, (group, base_lr) in enumerate(zip(self.optimizer.param_groups, self.base_lrs)):
            # Interpolate between eta_min and scaled base_lr
            new_lr = self.eta_min + (base_lr - self.eta_min) * scale
            group['lr'] = new_lr
            self._last_lr[i] = new_lr

    def step(self, epoch: int, step_in_epoch: int, steps_in_epoch: int):
        """
        Advance scheduler by one step.

        Args:
            epoch: Current epoch (0-indexed)
            step_in_epoch: Current step within the epoch (0-indexed, AFTER this batch)
            steps_in_epoch: Total steps in this epoch
        """
        self.current_epoch = epoch
        self.current_step_in_epoch = step_in_epoch
        self.current_steps_in_epoch = steps_in_epoch
        self._update_lr()

    def get_last_lr(self) -> List[float]:
        """Return last computed learning rates."""
        return self._last_lr

    def get_phase(self) -> int:
        """Return current phase index (0-indexed)."""
        return self.get_phase_for_epoch(self.current_epoch)

    def get_phase_progress(self) -> Tuple[int, float, str]:
        """
        Return current phase and progress within that phase.

        Returns:
            (phase_idx, progress_ratio, sub_phase_name)
            sub_phase_name: 'lr_ramp' or 'cosine_decay'
        """
        phase_idx = self.get_phase()
        progress = self._get_progress_in_phase(
            self.current_epoch,
            self.current_step_in_epoch,
            self.current_steps_in_epoch
        )

        if progress < self.lr_ramp_ratio:
            ramp_progress = progress / self.lr_ramp_ratio
            return (phase_idx, ramp_progress, 'lr_ramp')
        else:
            decay_progress = (progress - self.lr_ramp_ratio) / (1.0 - self.lr_ramp_ratio)
            return (phase_idx, min(1.0, decay_progress), 'cosine_decay')

    def get_current_peak(self) -> float:
        """Return the peak LR for the current phase."""
        phase_idx = self.get_phase()
        return self.base_lrs[0] * (self.phase_decay ** phase_idx)

    def calculate_lr_at_epoch(self, epoch: int, epoch_progress: float = 0.0) -> float:
        """
        Calculate the learning rate at a specific epoch position.

        Useful for logging and visualization.

        Args:
            epoch: Target epoch (0-indexed)
            epoch_progress: Progress within the epoch (0.0 to 1.0)

        Returns:
            Learning rate at the specified position
        """
        # Convert epoch_progress to step representation
        step_in_epoch = int(epoch_progress * 100)
        steps_in_epoch = 100

        scale = self._get_lr_scale(epoch, step_in_epoch, steps_in_epoch)
        return self.eta_min + (self.base_lrs[0] - self.eta_min) * scale

    def get_schedule_data(self, points_per_epoch: int = 2) -> Tuple[List[float], List[float], List[int]]:
        """
        Get LR schedule data for visualization.

        Args:
            points_per_epoch: Number of sample points per epoch

        Returns:
            (epochs, lrs, phase_boundaries)
            - epochs: List of epoch values (float)
            - lrs: List of corresponding learning rates
            - phase_boundaries: List of epoch indices where phases start
        """
        epochs = []
        lrs = []
        phase_boundaries = []

        for epoch in range(self.total_epochs):
            # Mark phase boundaries
            if epoch % self.phase_epochs == 0:
                phase_boundaries.append(epoch)

            # Sample points within each epoch
            for i in range(points_per_epoch):
                progress = i / points_per_epoch
                lr = self.calculate_lr_at_epoch(epoch, progress)
                epochs.append(epoch + progress)
                lrs.append(lr)

        return epochs, lrs, phase_boundaries

    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing."""
        return {
            'current_epoch': self.current_epoch,
            'current_step_in_epoch': self.current_step_in_epoch,
            'current_steps_in_epoch': self.current_steps_in_epoch,
            'total_epochs': self.total_epochs,
            'phase_epochs': self.phase_epochs,
            'phase_decay': self.phase_decay,
            'lr_ramp_ratio': self.lr_ramp_ratio,
            'num_phases': self.num_phases,
            'base_lrs': self.base_lrs,
            'eta_min': self.eta_min,
        }

    def load_state_dict(self, state_dict: dict):
        """Load scheduler state from checkpoint."""
        self.current_epoch = state_dict['current_epoch']
        self.current_step_in_epoch = state_dict['current_step_in_epoch']
        self.current_steps_in_epoch = state_dict['current_steps_in_epoch']
        self.total_epochs = state_dict['total_epochs']
        self.phase_epochs = state_dict['phase_epochs']
        self.phase_decay = state_dict['phase_decay']
        self.lr_ramp_ratio = state_dict['lr_ramp_ratio']
        self.num_phases = state_dict['num_phases']
        self.base_lrs = state_dict['base_lrs']
        self.eta_min = state_dict['eta_min']
        self._update_lr()


def visualize_lr_schedule(
    scheduler_config: Dict[str, Any],
    base_lr: float = 1e-4,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> Optional[Path]:
    """
    Visualize the learning rate schedule and optionally save/display it.

    This function creates a plot showing the LR schedule with phase boundaries,
    intended to be called once at the start of training to verify the schedule.

    Args:
        scheduler_config: Scheduler configuration from SCHEDULER_PRESETS
        base_lr: Base learning rate (backbone_lr for CBraMod)
        output_path: Path to save the image (optional)
        show: Whether to display the image in a non-blocking window

    Returns:
        Path to saved image if output_path was provided, else None
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving
    import matplotlib.pyplot as plt

    # Extract parameters from config
    total_epochs = scheduler_config.get('epochs', 100)
    phase_epochs = scheduler_config.get('phase_epochs', 6)
    phase_decay = scheduler_config.get('phase_decay', 0.7)
    lr_ramp_ratio = scheduler_config.get('lr_ramp_ratio', 0.1)
    eta_min = scheduler_config.get('eta_min', 1e-6)
    exploration_epochs = scheduler_config.get('exploration_epochs', 6)

    # Create a dummy optimizer to instantiate scheduler
    dummy_param = torch.nn.Parameter(torch.zeros(1))
    dummy_optimizer = torch.optim.SGD([dummy_param], lr=base_lr)

    scheduler = CosineAnnealingWarmupDecay(
        dummy_optimizer,
        total_epochs=total_epochs,
        phase_epochs=phase_epochs,
        phase_decay=phase_decay,
        lr_ramp_ratio=lr_ramp_ratio,
        eta_min=eta_min,
    )

    # Get schedule data
    epochs, lrs, phase_boundaries = scheduler.get_schedule_data(points_per_epoch=4)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot LR curve
    ax.plot(epochs, lrs, 'b-', linewidth=1.5, label='Learning Rate')

    # Mark phase boundaries with vertical lines
    for i, boundary in enumerate(phase_boundaries):
        peak_lr = base_lr * (phase_decay ** i)
        ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        if i < 8:  # Only label first 8 phases to avoid clutter
            ax.annotate(
                f'P{i}\n{peak_lr:.1e}',
                xy=(boundary + 0.5, peak_lr * 0.95),
                fontsize=7,
                color='gray',
                ha='left',
                va='top',
            )

    # Mark exploration phase
    ax.axvspan(0, exploration_epochs, alpha=0.15, color='green', label=f'Exploration ({exploration_epochs} epochs)')

    # Labels and title
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Learning Rate', fontsize=11)
    ax.set_title(
        f'CosineAnnealingWarmupDecay Schedule\n'
        f'Total: {total_epochs} epochs | Phase: {phase_epochs} epochs | '
        f'Decay: {phase_decay} | LR Ramp: {lr_ramp_ratio:.0%}',
        fontsize=12,
    )
    ax.set_xlim(0, total_epochs)
    ax.set_ylim(0, base_lr * 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # Add text annotations
    info_text = (
        f"Base LR: {base_lr:.1e}\n"
        f"Min LR: {eta_min:.1e}\n"
        f"Phases: {scheduler.num_phases}"
    )
    ax.text(
        0.02, 0.98, info_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )

    plt.tight_layout()

    # Save if path provided
    saved_path = None
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        saved_path = output_path
        log_train.info(f"LR schedule visualization saved to: {output_path}")

    # Show in non-blocking window
    if show:
        # Switch to interactive backend for display
        plt.switch_backend('TkAgg')
        fig_show, ax_show = plt.subplots(figsize=(14, 6))

        # Re-plot on the new figure
        ax_show.plot(epochs, lrs, 'b-', linewidth=1.5, label='Learning Rate')
        for i, boundary in enumerate(phase_boundaries):
            peak_lr = base_lr * (phase_decay ** i)
            ax_show.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
            if i < 8:
                ax_show.annotate(
                    f'P{i}\n{peak_lr:.1e}',
                    xy=(boundary + 0.5, peak_lr * 0.95),
                    fontsize=7,
                    color='gray',
                    ha='left',
                    va='top',
                )
        ax_show.axvspan(0, exploration_epochs, alpha=0.15, color='green', label=f'Exploration ({exploration_epochs} epochs)')
        ax_show.set_xlabel('Epoch', fontsize=11)
        ax_show.set_ylabel('Learning Rate', fontsize=11)
        ax_show.set_title(
            f'CosineAnnealingWarmupDecay Schedule\n'
            f'Total: {total_epochs} epochs | Phase: {phase_epochs} epochs | '
            f'Decay: {phase_decay} | LR Ramp: {lr_ramp_ratio:.0%}',
            fontsize=12,
        )
        ax_show.set_xlim(0, total_epochs)
        ax_show.set_ylim(0, base_lr * 1.1)
        ax_show.grid(True, alpha=0.3)
        ax_show.legend(loc='upper right')
        ax_show.text(
            0.02, 0.98, info_text,
            transform=ax_show.transAxes,
            fontsize=9,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        )
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)  # Brief pause to ensure window appears

    plt.close(fig)

    return saved_path
