"""
Finetune utility functions extracted from src/training/finetune.py for reuse.

This module contains:
- FreezeStrategy: type alias for freeze strategy literals
- load_pretrained_model(): load a model from a checkpoint file
- apply_freeze_strategy(): freeze/unfreeze model parameters per strategy
- get_finetune_optimizer(): build optimizer with appropriate LR for finetuning
- get_default_finetune_config(): consolidate default hyperparameter logic
- detect_model_type_from_checkpoint(): inspect a checkpoint and return model_type
"""

import logging
from typing import Literal, Optional

import torch
import torch.nn as nn

from src.models.eegnet import EEGNet
from src.models.cbramod_adapter import CBraModForFingerBCI
from src.config.training import (
    EIGHT_CHANNEL_FINETUNE_OVERRIDES,
    THIRTYTWO_CHANNEL_FINETUNE_OVERRIDES,
    SIXTYONE_CHANNEL_FINETUNE_OVERRIDES,
)
from src.utils.logging import SectionLogger

logger = logging.getLogger(__name__)
log_model = SectionLogger(logger, 'model')


FreezeStrategy = Literal['none', 'backbone', 'partial']


def load_pretrained_model(
    pretrained_path: str,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    """
    Load a pretrained model from checkpoint.

    Args:
        pretrained_path: Path to pretrained checkpoint (.pt file)
        device: Device to load model on

    Returns:
        Tuple of (model, checkpoint_dict)
    """
    checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)

    model_config = checkpoint['model_config']
    model_type = model_config['model_type']
    n_channels = model_config['n_channels']
    n_samples = model_config['n_samples']
    n_classes = model_config['n_classes']

    if model_type == 'cbramod':
        n_patches = model_config.get('n_patches', n_samples // 200)
        model = CBraModForFingerBCI(
            n_channels=n_channels,
            n_patches=n_patches,
            n_classes=n_classes,
            pretrained_path=None,  # Don't load pretrained weights again
            freeze_backbone=False,
            classifier_type=model_config.get('classifier_type', 'two_layer'),
            dropout=0.1,
        )
    else:
        # EEGNet - need to get config from checkpoint or use defaults
        model = EEGNet(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=n_classes,
            F1=8,
            D=2,
            F2=16,
            kernel_length=64,
            dropout_rate=0.5,
        )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    log_model.info(f"Loaded pretrained {model_type} from {pretrained_path}")

    return model, checkpoint


def apply_freeze_strategy(
    model: nn.Module,
    model_type: str,
    freeze_strategy: FreezeStrategy,
) -> int:
    """
    Apply freeze strategy to model.

    Args:
        model: Model to freeze
        model_type: 'eegnet' or 'cbramod'
        freeze_strategy: Freeze strategy to apply

    Returns:
        Number of frozen parameters
    """
    if freeze_strategy == 'none':
        # All parameters trainable
        for param in model.parameters():
            param.requires_grad = True
        log_model.info("Freeze strategy: none (all parameters trainable)")
        return 0

    frozen_count = 0

    if model_type == 'cbramod':
        if freeze_strategy == 'backbone':
            # Freeze entire backbone, train only classifier
            if hasattr(model, 'backbone'):
                for param in model.backbone.parameters():
                    param.requires_grad = False
                    frozen_count += param.numel()
            log_model.info("Freeze strategy: backbone (transformer frozen, classifier trainable)")

        elif freeze_strategy == 'partial':
            # Freeze first 6 transformer layers
            if hasattr(model, 'backbone') and hasattr(model.backbone, 'transformer'):
                transformer = model.backbone.transformer
                # CBraMod uses custom transformer structure
                # Freeze encoder layers 0-5 (first half of 12 layers)
                if hasattr(transformer, 'encoder') and hasattr(transformer.encoder, 'layers'):
                    for i, layer in enumerate(transformer.encoder.layers):
                        if i < 6:
                            for param in layer.parameters():
                                param.requires_grad = False
                                frozen_count += param.numel()
            log_model.info("Freeze strategy: partial (first 6 transformer layers frozen)")

    else:  # EEGNet
        if freeze_strategy == 'backbone':
            # Freeze block 1 (temporal + spatial conv)
            layers_to_freeze = ['temporal_conv', 'spatial_conv', 'bn1', 'bn2']
            for name, param in model.named_parameters():
                if any(layer in name for layer in layers_to_freeze):
                    param.requires_grad = False
                    frozen_count += param.numel()
            log_model.info("Freeze strategy: backbone (block1 frozen, block2+fc trainable)")

        elif freeze_strategy == 'partial':
            # Freeze only temporal conv and bn1
            layers_to_freeze = ['temporal_conv', 'bn1']
            for name, param in model.named_parameters():
                if any(layer in name for layer in layers_to_freeze):
                    param.requires_grad = False
                    frozen_count += param.numel()
            log_model.info("Freeze strategy: partial (temporal_conv frozen only)")

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())

    log_model.info(f"Parameters: {trainable_count:,} trainable / {total_count:,} total "
                   f"({frozen_count:,} frozen)")

    return frozen_count


def get_finetune_optimizer(
    model: nn.Module,
    model_type: str,
    freeze_strategy: FreezeStrategy,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """
    Create optimizer with appropriate learning rates for finetuning.

    For 'backbone' freeze strategy, only classifier parameters are optimized.
    For 'partial' freeze strategy, unfrozen layers get lower LR.

    Args:
        model: Model to optimize
        model_type: 'eegnet' or 'cbramod'
        freeze_strategy: Current freeze strategy
        learning_rate: Base learning rate
        weight_decay: Weight decay

    Returns:
        Configured optimizer
    """
    # Get trainable parameters only
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if not trainable_params:
        raise ValueError("No trainable parameters! Check freeze strategy.")

    if model_type == 'cbramod' and freeze_strategy == 'none':
        # Use differential learning rates for CBraMod
        if hasattr(model, 'get_parameter_groups'):
            param_groups = model.get_parameter_groups(
                backbone_lr=learning_rate,
                classifier_lr=learning_rate * 5,
            )
            # Filter to only trainable parameters
            for group in param_groups:
                group['params'] = [p for p in group['params'] if p.requires_grad]
            return torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    # Standard optimizer for other cases
    if model_type == 'cbramod':
        return torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    else:
        return torch.optim.Adam(trainable_params, lr=learning_rate, weight_decay=weight_decay)


def get_default_finetune_config(
    model_type: str,
    freeze_strategy: FreezeStrategy,
    n_channels: int,
) -> dict:
    """
    Consolidate scattered default hyperparameter logic from finetune_subject().

    Returns a dict with keys: epochs, learning_rate, batch_size, scheduler_type.
    Channel-specific overrides are applied for cbramod at 8/32/61 channels.

    Args:
        model_type: 'eegnet' or 'cbramod'
        freeze_strategy: One of 'none', 'backbone', 'partial'
        n_channels: Number of EEG channels in use

    Returns:
        Dict with keys epochs, learning_rate, batch_size, scheduler_type
    """
    is_8ch_cbramod = (n_channels == 8 and model_type == 'cbramod')
    is_32ch_cbramod = (n_channels == 32 and model_type == 'cbramod')
    is_61ch_cbramod = (n_channels == 61 and model_type == 'cbramod')

    # --- epochs (safety ceiling; early stopping controls actual duration) ---
    if is_8ch_cbramod:
        epochs = EIGHT_CHANNEL_FINETUNE_OVERRIDES['epochs']
    elif is_32ch_cbramod:
        epochs = THIRTYTWO_CHANNEL_FINETUNE_OVERRIDES['epochs']
    elif is_61ch_cbramod:
        epochs = SIXTYONE_CHANNEL_FINETUNE_OVERRIDES['epochs']
    else:
        epochs = 500

    # --- learning_rate ---
    if is_8ch_cbramod:
        learning_rate = EIGHT_CHANNEL_FINETUNE_OVERRIDES['learning_rate']
    elif is_32ch_cbramod:
        learning_rate = THIRTYTWO_CHANNEL_FINETUNE_OVERRIDES['learning_rate']
    elif is_61ch_cbramod:
        learning_rate = SIXTYONE_CHANNEL_FINETUNE_OVERRIDES['learning_rate']
    elif freeze_strategy == 'backbone':
        learning_rate = 5e-4  # Higher LR when only training classifier
    elif freeze_strategy == 'partial':
        learning_rate = 1e-4
    else:
        learning_rate = 1e-4 if model_type == 'cbramod' else 1e-4

    # --- batch_size ---
    batch_size = 64 if model_type == 'eegnet' else 128

    # --- scheduler_type ---
    scheduler_type = 'plateau' if model_type == 'eegnet' else 'cosine_annealing_warmup_decay'

    return {
        'epochs': epochs,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'scheduler_type': scheduler_type,
    }


def detect_model_type_from_checkpoint(checkpoint_path: str) -> str:
    """
    Load a checkpoint and return the model_type string.

    Args:
        checkpoint_path: Path to a .pt checkpoint file

    Returns:
        model_type string, e.g. 'cbramod' or 'eegnet'

    Raises:
        KeyError: if the checkpoint does not contain model_config or model_type
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_type = checkpoint['model_config']['model_type']
    log_model.info(f"Detected model_type='{model_type}' from {checkpoint_path}")
    return model_type
