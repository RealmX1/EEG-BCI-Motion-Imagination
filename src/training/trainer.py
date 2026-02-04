"""
Within-subject trainer for EEG-BCI models.

This module provides the WithinSubjectTrainer class for training
EEGNet and CBraMod models on single-subject EEG data.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .schedulers import WSDScheduler, CosineDecayRestarts, CosineAnnealingWarmupDecay
from .evaluation import majority_vote_accuracy
from ..preprocessing.data_loader import FingerEEGDataset
from ..utils.logging import SectionLogger
from ..utils.timing import EpochTimer, print_section_header
from ..utils.table_logger import TableEpochLogger

if TYPE_CHECKING:
    from ..utils.wandb_logger import WandbCallback

logger = logging.getLogger(__name__)
log_model = SectionLogger(logger, 'model')
log_train = SectionLogger(logger, 'train')


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
        scheduler_config: Optional[Dict[str, Any]] = None,
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

        # Scheduler-specific parameters (read from scheduler_config or use defaults)
        # These are stored for later initialization in train() when total_steps/epochs are known
        self.scheduler_config = scheduler_config or {}

        # WSD-specific parameters
        self.wsd_warmup_ratio = self.scheduler_config.get('warmup_ratio', 0.1)
        self.wsd_stable_ratio = self.scheduler_config.get('stable_ratio', 0.0)
        self.wsd_decay_ratio = self.scheduler_config.get('decay_ratio', 0.3)

        # CosineDecayRestarts-specific parameters
        self.cosine_decay_factor = self.scheduler_config.get('decay_factor', 0.7)
        self.cosine_decay_cycles = self.scheduler_config.get('num_cycles', 5)

        # CosineAnnealingWarmupDecay-specific parameters (renamed from warmup -> lr_ramp)
        self.phase_epochs = self.scheduler_config.get('phase_epochs', 6)
        self.phase_decay = self.scheduler_config.get('phase_decay', 0.7)
        self.lr_ramp_ratio = self.scheduler_config.get('lr_ramp_ratio', 0.1)
        self.cawd_eta_min = self.scheduler_config.get('eta_min', 1e-6)

        # Exploration phase parameters (for two-stage batch size strategy)
        self.exploration_epochs = self.scheduler_config.get('exploration_epochs', 5)
        self.exploration_batch_size = self.scheduler_config.get('exploration_batch_size', 32)

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

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int = 0,
        profile: bool = False,
    ) -> Tuple[float, float]:
        """Train for one epoch with AMP, gradient clipping, and per-step scheduler.

        Args:
            dataloader: Training data loader
            epoch: Current epoch index (0-indexed), used for epoch-based schedulers
            profile: Whether to enable performance profiling
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        steps_in_epoch = len(dataloader)

        # Profiling variables
        if profile:
            t_data, t_transfer, t_forward, t_backward, t_optim = 0, 0, 0, 0, 0
            t_start = time.perf_counter()

        for batch_idx, (segments, labels) in enumerate(dataloader):
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

            # Per-step scheduler update (for CBraMod)
            if self.scheduler is not None and self.model_type == 'cbramod':
                if self.scheduler_type == 'cosine_annealing_warmup_decay':
                    # Epoch-based scheduler: pass epoch and step position
                    self.scheduler.step(epoch, batch_idx + 1, steps_in_epoch)
                else:
                    # Step-based schedulers (WSD, CosineDecayRestarts)
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
        exploration_epochs: int = 0,
        epochs: int = 50,
        patience: int = 10,
        save_path: Optional[Path] = None,
        wandb_callback: Optional['WandbCallback'] = None,
    ) -> Dict:
        """
        Full training loop with early stopping and two-phase batch size.

        Two-phase batch size strategy:
        - Exploration phase (first N epochs): small batch size for more gradient updates
        - Main phase (remaining epochs): normal batch size for stable training

        Args:
            train_loader: Training DataLoader for exploration phase (small batch)
            val_loader: Validation DataLoader
            main_train_loader: Training DataLoader for main phase (normal batch).
                              If None, uses train_loader for all epochs.
            exploration_epochs: Number of epochs for exploration phase (small batch).
                               After this, switches to main_train_loader.
            epochs: Maximum epochs
            patience: Early stopping patience
            save_path: Path to save best model
            wandb_callback: Optional WandB callback for logging

        Returns:
            Training history
        """
        from ..utils.timing import Colors

        # Use single loader if main_train_loader not provided
        if main_train_loader is None:
            main_train_loader = train_loader
            exploration_epochs = 0

        phase_info = f", exploration={exploration_epochs}eps" if exploration_epochs > 0 else ""
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
        # exploration phase uses train_loader (small batch), main phase uses main_train_loader
        exploration_steps = exploration_epochs * len(train_loader)
        main_steps = (epochs - exploration_epochs) * len(main_train_loader)
        total_steps = exploration_steps + main_steps

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
            # NEW: Epoch-based phase calculation
            # Phase boundaries are determined by epoch number, not step count.
            # This ensures phase transitions happen at correct epoch positions regardless of batch size.
            num_phases = epochs // self.phase_epochs
            if epochs % self.phase_epochs != 0:
                num_phases += 1

            log_train.info(
                f"{Colors.BRIGHT_YELLOW}CosineAnnealingWarmupDecay Scheduler: "
                f"phase_epochs={self.phase_epochs} | "
                f"num_phases={num_phases} | "
                f"phase_decay={self.phase_decay} | "
                f"lr_ramp_ratio={self.lr_ramp_ratio}"
                f"{Colors.RESET}"
            )
            self.scheduler = CosineAnnealingWarmupDecay(
                self.optimizer,
                total_epochs=epochs,
                phase_epochs=self.phase_epochs,
                phase_decay=self.phase_decay,
                lr_ramp_ratio=self.lr_ramp_ratio,
                eta_min=self.cawd_eta_min,
            )
            # Show peak LR progression
            peaks = [self.phase_decay ** i for i in range(num_phases)]
            peak_str = " -> ".join([f"{p:.0%}" for p in peaks])
            log_train.info(f"Scheduler: CosineAnnealingWarmupDecay (peak progression: {peak_str})")

        for epoch in range(epochs):
            epoch_timer.start_epoch()

            # Select train loader based on epoch (two-phase batch size)
            if epoch < exploration_epochs:
                current_train_loader = train_loader  # Small batch (exploration)
            else:
                current_train_loader = main_train_loader  # Normal batch (main)

            # Train (profile only first epoch to diagnose bottlenecks)
            do_profile = (epoch == 0)
            with epoch_timer.phase("train"):
                train_loss, train_acc = self.train_epoch(
                    current_train_loader,
                    epoch=epoch,
                    profile=do_profile,
                )

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
