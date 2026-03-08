"""
Within-subject trainer for EEG-BCI models.

This module provides the WithinSubjectTrainer class for training
EEGNet and CBraMod models on single-subject EEG data.
"""

import logging
import os
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


def _import_muon():
    """导入 Muon 优化器. 优先 PyTorch 内置，回退到 standalone 包."""
    try:
        from torch.optim import Muon
        return Muon
    except ImportError:
        pass
    try:
        from muon import Muon
        return Muon
    except ImportError:
        pass
    raise ImportError(
        "Muon optimizer not available. Options:\n"
        "  1. PyTorch >= 2.10: pip install --pre torch\n"
        "  2. Standalone: pip install muon-optimizer\n"
        f"  Current PyTorch: {torch.__version__}"
    )


class MuonAdamWHybrid:
    """Muon + AdamW 混合优化器包装器.

    PyTorch 内置 Muon 仅支持 2D 参数，因此需要将 Muon (2D weights)
    和 AdamW (bias, norm, classifier) 组合为统一接口。

    暴露 param_groups, step(), zero_grad(), state_dict(), load_state_dict()
    以兼容 PyTorch 训练管线 (GradScaler, 自定义 scheduler 等)。
    """

    def __init__(self, muon_param_group: dict, adamw_param_groups: list,
                 weight_decay: float = 0.0):
        Muon = _import_muon()

        # Copy to avoid mutating caller's dicts
        muon_param_group = dict(muon_param_group)
        adamw_param_groups = [dict(g) for g in adamw_param_groups]

        muon_lr = muon_param_group.pop('lr', 0.02)
        muon_momentum = muon_param_group.pop('momentum', 0.95)
        muon_ns_steps = muon_param_group.pop('ns_steps', 5)
        muon_param_group.pop('use_muon', None)
        muon_param_group.pop('weight_decay', None)

        self.muon = Muon(
            [muon_param_group],
            lr=muon_lr,
            momentum=muon_momentum,
            ns_steps=muon_ns_steps,
            weight_decay=weight_decay,
        )

        for g in adamw_param_groups:
            g.pop('use_muon', None)
        self.adamw = torch.optim.AdamW(
            adamw_param_groups,
            weight_decay=weight_decay,
        )

    @property
    def param_groups(self):
        return self.muon.param_groups + self.adamw.param_groups

    def step(self, closure=None):
        if closure is not None:
            raise ValueError("MuonAdamWHybrid does not support closure")
        self.muon.step()
        self.adamw.step()

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            'muon': self.muon.state_dict(),
            'adamw': self.adamw.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.muon.load_state_dict(state_dict['muon'])
        self.adamw.load_state_dict(state_dict['adamw'])

# Scheduler classification by stepping frequency
STEP_BASED_SCHEDULERS = {'wsd', 'cosine_decay', 'cosine_annealing_warmup_decay'}
EPOCH_BASED_SCHEDULERS = {'plateau', 'cosine'}


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
        label_smoothing: Optional[float] = None,
        scheduler_type: Optional[str] = None,
        scheduler_config: Optional[Dict[str, Any]] = None,
        use_amp: bool = True,
        gradient_clip: float = 1.0,
        optimizer_type: str = 'adamw',
        muon_config: Optional[Dict[str, Any]] = None,
        verbose: int = 2,
    ):
        self.model = model.to(device)
        self.dataset = dataset
        self.val_indices = val_indices
        self.device = device
        self.model_type = model_type
        self.scheduler_type = scheduler_type
        self.verbose = verbose

        if optimizer_type == 'muon' and scheduler_type == 'plateau':
            log_train.warning(
                "Muon + ReduceLROnPlateau will decay Muon's lr along with AdamW's. "
                "Consider using 'cosine_annealing_warmup_decay' or 'wsd' instead."
            )

        # Loss function - apply label smoothing for regularization
        # None = use model-specific default (0.05 for CBraMod, 0.0 for EEGNet)
        if label_smoothing is None:
            label_smoothing = 0.05 if model_type == 'cbramod' else 0.0
        if label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            if self.verbose >= 2:
                log_model.info(f"Label smoothing={label_smoothing}")
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Create optimizer based on model type and optimizer_type
        self.optimizer_type = optimizer_type

        if optimizer_type == 'muon' and model_type == 'cbramod':
            self.optimizer = self._create_muon_optimizer(
                model, learning_rate, classifier_lr, weight_decay, muon_config,
            )
        elif model_type == 'cbramod' and hasattr(model, 'get_parameter_groups'):
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
            if self.verbose >= 2:
                log_train.info(f"Optimizer: AdamW (backbone_lr={learning_rate}, classifier_lr={actual_classifier_lr})")
        else:
            # EEGNet uses standard Adam
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )

        # All schedulers are created in train() when total_steps/epochs are known.
        # This ensures consistent initialization regardless of model type.
        self.scheduler = None
        self.scheduler_needs_metric = False  # For ReduceLROnPlateau
        if scheduler_type and self.verbose >= 2:
            log_train.info(f"Scheduler: {scheduler_type} (will be initialized in train())")

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
            if self.verbose >= 2:
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

    def _get_sub_optimizers(self) -> list:
        """返回底层优化器列表 (用于 GradScaler 兼容)."""
        if isinstance(self.optimizer, MuonAdamWHybrid):
            return [self.optimizer.muon, self.optimizer.adamw]
        return [self.optimizer]

    def _create_muon_optimizer(
        self,
        model: nn.Module,
        backbone_lr: float,
        classifier_lr: Optional[float],
        weight_decay: float,
        muon_config: Optional[Dict[str, Any]],
    ) -> MuonAdamWHybrid:
        """创建 Muon 混合优化器 (Muon for 2D weights + AdamW for rest)."""
        cfg = muon_config or {}

        muon_lr = cfg.get('muon_lr', 0.02)
        muon_momentum = cfg.get('muon_momentum', 0.95)
        muon_ns_steps = cfg.get('muon_ns_steps', 5)
        adamw_bb_lr = cfg.get('adamw_backbone_lr', backbone_lr)
        adamw_cls_lr = cfg.get('adamw_classifier_lr', classifier_lr or backbone_lr * 3)

        if not hasattr(model, 'get_muon_parameter_groups'):
            raise ValueError("Model does not support Muon parameter groups")

        param_groups = model.get_muon_parameter_groups(
            muon_lr=muon_lr,
            adamw_backbone_lr=adamw_bb_lr,
            adamw_classifier_lr=adamw_cls_lr,
            muon_momentum=muon_momentum,
            muon_ns_steps=muon_ns_steps,
        )

        # 统计参数分布
        n_muon = sum(p.numel() for p in param_groups[0]['params'])
        n_adamw_bb = sum(p.numel() for p in param_groups[1]['params'])
        n_adamw_cls = sum(p.numel() for p in param_groups[2]['params'])

        # Group 0 = Muon, Groups 1-2 = AdamW
        optimizer = MuonAdamWHybrid(
            muon_param_group=param_groups[0],
            adamw_param_groups=param_groups[1:],
            weight_decay=weight_decay,
        )

        if self.verbose >= 2:
            log_train.info(
                f"Optimizer: Muon hybrid "
                f"(muon={n_muon:,} @ lr={muon_lr}, "
                f"adamw_bb={n_adamw_bb:,} @ lr={adamw_bb_lr}, "
                f"adamw_cls={n_adamw_cls:,} @ lr={adamw_cls_lr})"
            )
        return optimizer

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
                # MuonAdamWHybrid wraps two real optimizers; unscale/step each
                sub_opts = self._get_sub_optimizers()
                if self.gradient_clip > 0:
                    for opt in sub_opts:
                        self.scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                for opt in sub_opts:
                    self.scaler.step(opt)
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

            # Per-step scheduler update (WSD, CosineDecayRestarts, CosineAnnealingWarmupDecay)
            if self.scheduler is not None and self.scheduler_type in STEP_BASED_SCHEDULERS:
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

    def save_resume_checkpoint(self, save_path: Path, epoch: int):
        """保存用于恢复训练的完整检查点.

        与 best.pt（仅保存最佳模型权重）不同，resume_checkpoint.pt
        包含所有训练状态：优化器、调度器、AMP scaler、历史记录等。

        使用原子写入防止写入中崩溃损坏文件。

        Args:
            save_path: 检查点保存目录
            epoch: 已完成的 epoch 数（1-indexed）
        """
        # best_state 不保存到 resume checkpoint（best.pt 已在磁盘上），
        # 避免 CBraMod ~4M 参数的权重存储两份
        checkpoint = {
            'resume_version': 1,
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'best_majority_acc': self.best_majority_acc,
            'best_combined_score': self.best_combined_score,
            'best_epoch': self.best_epoch,
            'history': self.history,
            'scheduler_type': self.scheduler_type,
            'model_type': self.model_type,
            'optimizer_type': self.optimizer_type,
        }

        temp_path = save_path / 'resume_checkpoint.pt.tmp'
        final_path = save_path / 'resume_checkpoint.pt'
        torch.save(checkpoint, temp_path)
        os.replace(str(temp_path), str(final_path))
        log_train.debug(f"Resume checkpoint saved: epoch {epoch}")

    def load_resume_checkpoint(self, save_path: Path) -> Optional[int]:
        """从恢复检查点加载训练状态.

        Args:
            save_path: 检查点所在目录

        Returns:
            已完成的 epoch 数（下一个 epoch 从这里开始），
            如果没有找到恢复检查点则返回 None。
        """
        resume_path = save_path / 'resume_checkpoint.pt'
        if not resume_path.exists():
            return None

        log_train.info(f"Loading resume checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=self.device, weights_only=False)

        resume_version = checkpoint.get('resume_version', 0)
        if resume_version < 1:
            log_train.warning("Resume checkpoint version too old, skipping resume")
            return None

        saved_model_type = checkpoint.get('model_type')
        if saved_model_type and saved_model_type != self.model_type:
            log_train.error(f"Model type mismatch: checkpoint={saved_model_type}, current={self.model_type}")
            return None

        saved_optimizer_type = checkpoint.get('optimizer_type', 'adamw')
        saved_scheduler_type = checkpoint.get('scheduler_type')
        optimizer_changed = saved_optimizer_type != self.optimizer_type
        scheduler_changed = saved_scheduler_type != self.scheduler_type

        if optimizer_changed or scheduler_changed:
            reasons = []
            if optimizer_changed:
                reasons.append(f"optimizer: {saved_optimizer_type} -> {self.optimizer_type}")
            if scheduler_changed:
                reasons.append(f"scheduler: {saved_scheduler_type} -> {self.scheduler_type}")
            log_train.warning(
                f"Config changed ({', '.join(reasons)}). Only model weights will be restored."
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self._pending_scheduler_state = None
            self._pending_scaler_state = None
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self._pending_scheduler_state = checkpoint.get('scheduler_state_dict')
            self._pending_scaler_state = checkpoint.get('scaler_state_dict')

        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_majority_acc = checkpoint['best_majority_acc']
        self.best_combined_score = checkpoint['best_combined_score']
        self.best_epoch = checkpoint['best_epoch']
        self.history = checkpoint.get('history', self.history)

        # Restore best_state from best.pt on disk (not stored in resume checkpoint)
        best_pt = save_path / 'best.pt'
        if best_pt.exists():
            best_ckpt = torch.load(best_pt, map_location=self.device, weights_only=True)
            self.best_state = best_ckpt['model_state_dict']
        else:
            self.best_state = None
            log_train.warning("best.pt not found, best_state unavailable until next improvement")

        completed_epoch = checkpoint['epoch']
        log_train.info(f"Resumed from epoch {completed_epoch}, best_epoch={self.best_epoch}, "
                       f"best_combined_score={self.best_combined_score:.4f}")
        return completed_epoch

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
        resume_from_epoch: Optional[int] = None,
        resume_checkpoint_interval: int = 5,
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
            resume_from_epoch: 从指定 epoch 继续训练（跳过已完成的 epoch）
            resume_checkpoint_interval: 每 N 个 epoch 保存恢复检查点（默认 5）

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

        # ============================================================
        # Create scheduler (model-agnostic, based on scheduler_type only)
        # ============================================================
        if self.scheduler_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',  # Maximize combined accuracy score
                factor=0.3,
                patience=2,
                min_lr=1e-6,
            )
            self.scheduler_needs_metric = True
            if self.verbose >= 2:
                log_train.info("Scheduler: ReduceLROnPlateau (mode=max, factor=0.3, patience=2, metric=combined_score)")

        elif self.scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=1e-6,
            )
            if self.verbose >= 2:
                log_train.info(f"Scheduler: CosineAnnealing (T_max={epochs})")

        elif self.scheduler_type == 'wsd':
            wsd_warmup_steps = int(total_steps * self.wsd_warmup_ratio)
            wsd_stable_steps = int(total_steps * self.wsd_stable_ratio)
            wsd_decay_steps = total_steps - wsd_warmup_steps - wsd_stable_steps
            if self.verbose >= 2:
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
            if self.verbose >= 2:
                log_train.info(f"Scheduler: WSD (total_steps={total_steps}, warmup={self.wsd_warmup_ratio}, decay={self.wsd_decay_ratio})")

        elif self.scheduler_type == 'cosine_decay':
            t_0 = total_steps // self.cosine_decay_cycles  # Cycle length
            if self.verbose >= 2:
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
            if self.verbose >= 2:
                peaks = [self.cosine_decay_factor ** i for i in range(self.cosine_decay_cycles)]
                peak_str = " -> ".join([f"{p:.2f}" for p in peaks])
                log_train.info(f"Scheduler: CosineDecayRestarts (peak progression: {peak_str})")

        elif self.scheduler_type == 'cosine_annealing_warmup_decay':
            num_phases = epochs // self.phase_epochs
            if epochs % self.phase_epochs != 0:
                num_phases += 1

            if self.verbose >= 2:
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
            if self.verbose >= 2:
                peaks = [self.phase_decay ** i for i in range(num_phases)]
                peak_str = " -> ".join([f"{p:.0%}" for p in peaks])
                log_train.info(f"Scheduler: CosineAnnealingWarmupDecay (peak progression: {peak_str})")

        # Restore scheduler/scaler state from resume checkpoint (if pending)
        if hasattr(self, '_pending_scheduler_state') and self._pending_scheduler_state is not None:
            if self.scheduler is not None:
                try:
                    self.scheduler.load_state_dict(self._pending_scheduler_state)
                    log_train.info("Scheduler state restored from resume checkpoint")
                except Exception as e:
                    log_train.warning(f"Failed to restore scheduler state: {e}")
            self._pending_scheduler_state = None

        if hasattr(self, '_pending_scaler_state') and self._pending_scaler_state is not None:
            if self.scaler is not None:
                try:
                    self.scaler.load_state_dict(self._pending_scaler_state)
                    log_train.info("AMP scaler state restored from resume checkpoint")
                except Exception as e:
                    log_train.warning(f"Failed to restore scaler state: {e}")
            self._pending_scaler_state = None

        # Determine start epoch for resume
        start_epoch = resume_from_epoch if resume_from_epoch is not None else 0
        if start_epoch > 0:
            # Restore early stopping counter from checkpoint state
            no_improve = max(0, start_epoch - self.best_epoch)
            log_train.info(f"Resuming from epoch {start_epoch}, "
                           f"epochs since last improvement: {no_improve}")

        # Milestone checkpoint tracking
        # Strategy: During the initial continuous best streak, don't save
        # each one individually. When the streak first breaks, save the last
        # streak epoch as the first milestone. After that, save every new best.
        milestones = []  # List of {'epoch', 'combined_score', 'val_acc', 'val_majority_acc', 'path'}
        initial_streak_active = True
        last_streak_epoch = None
        last_streak_info = None  # Holds checkpoint dict for the last streak best

        for epoch in range(start_epoch, epochs):
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
            # Muon: param_groups[0] is the Muon group; report AdamW backbone LR instead
            if self.optimizer_type == 'muon' and len(self.optimizer.param_groups) >= 2:
                current_lr = self.optimizer.param_groups[1]['lr']
            else:
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

            # Update epoch-based schedulers (ReduceLROnPlateau, CosineAnnealingLR)
            # Step-based schedulers are updated per-batch in train_epoch()
            if self.scheduler is not None and self.scheduler_type in EPOCH_BASED_SCHEDULERS:
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

                checkpoint_dict = {
                    'model_state_dict': self.best_state,
                    'epoch': self.best_epoch,
                    'val_acc': self.best_val_acc,
                    'val_majority_acc': self.best_majority_acc,
                    'combined_score': self.best_combined_score,
                    'val_loss': self.best_val_loss,
                }

                if save_path:
                    torch.save(checkpoint_dict, save_path / 'best.pt')
                    log_train.debug(f"Best model saved (combined={combined_score:.4f}, val_acc={val_acc:.4f}, maj_acc={majority_acc:.4f})")

                # Milestone tracking
                if initial_streak_active:
                    # Still in initial continuous best streak — just remember
                    last_streak_epoch = epoch + 1
                    last_streak_info = checkpoint_dict
                elif save_path:
                    # Post-streak: save every new best as a milestone
                    milestone_path = save_path / f'best_epoch{epoch+1:03d}.pt'
                    torch.save(checkpoint_dict, milestone_path)
                    milestones.append({
                        'epoch': epoch + 1,
                        'combined_score': combined_score,
                        'val_acc': val_acc,
                        'val_majority_acc': majority_acc,
                        'path': str(milestone_path),
                    })
                    log_train.debug(f"Milestone saved: epoch {epoch+1} (combined={combined_score:.4f})")
            else:
                no_improve += 1

                # First non-best epoch breaks the initial streak
                if initial_streak_active:
                    initial_streak_active = False
                    if last_streak_epoch is not None and save_path and last_streak_info is not None:
                        milestone_path = save_path / f'best_epoch{last_streak_epoch:03d}.pt'
                        torch.save(last_streak_info, milestone_path)
                        milestones.append({
                            'epoch': last_streak_epoch,
                            'combined_score': last_streak_info['combined_score'],
                            'val_acc': last_streak_info['val_acc'],
                            'val_majority_acc': last_streak_info['val_majority_acc'],
                            'path': str(milestone_path),
                        })
                        log_train.debug(f"Streak milestone saved: epoch {last_streak_epoch}")
                    last_streak_info = None  # Free memory

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

            # Periodic resume checkpoint (for crash recovery)
            if (save_path and resume_checkpoint_interval > 0
                    and (epoch + 1) % resume_checkpoint_interval == 0):
                self.save_resume_checkpoint(save_path, epoch + 1)

            # Early stopping check
            if will_stop:
                break

        # Edge case: training ended while still in initial streak
        # (every epoch was a new best, or early stopping during streak)
        if initial_streak_active and last_streak_epoch is not None and save_path:
            info = last_streak_info or {
                'model_state_dict': self.best_state,
                'epoch': self.best_epoch,
                'val_acc': self.best_val_acc,
                'val_majority_acc': self.best_majority_acc,
                'combined_score': self.best_combined_score,
                'val_loss': self.best_val_loss,
            }
            milestone_path = save_path / f'best_epoch{last_streak_epoch:03d}.pt'
            torch.save(info, milestone_path)
            milestones.append({
                'epoch': last_streak_epoch,
                'combined_score': info['combined_score'],
                'val_acc': info['val_acc'],
                'val_majority_acc': info['val_majority_acc'],
                'path': str(milestone_path),
            })
        last_streak_info = None  # Free memory

        # Add milestone info to history
        self.history['milestones'] = milestones

        # Clean up resume checkpoint (training completed successfully)
        if save_path:
            resume_path = save_path / 'resume_checkpoint.pt'
            if resume_path.exists():
                resume_path.unlink()
                log_train.debug("Resume checkpoint cleaned up (training complete)")

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
