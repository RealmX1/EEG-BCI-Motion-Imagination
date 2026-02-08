"""
Shared training utilities for within-subject and cross-subject training.

This module provides common helper functions used by both training modes
to avoid code duplication and ensure consistent behavior.

Functions:
    setup_performance_optimizations: Configure cuDNN and TF32 for faster training
    maybe_compile_model: Apply torch.compile if supported
    get_scheduler_config_from_preset: Extract scheduler config from SCHEDULER_PRESETS
    create_two_phase_loaders: Create exploration and main phase DataLoaders
"""

import logging
import platform
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils.logging import SectionLogger
from ..utils.timing import print_metric, Colors

logger = logging.getLogger(__name__)
log_train = SectionLogger(logger, 'train')
log_model = SectionLogger(logger, 'model')


def setup_performance_optimizations(
    device: torch.device,
    verbose: int = 2,
) -> None:
    """
    Setup performance optimizations for GPU training.

    Configures:
    - cuDNN auto-tuning for faster convolutions (20-50% speedup)
    - TF32 matrix multiplication for Ampere+ GPUs

    Args:
        device: PyTorch device (cuda or cpu)
        verbose: Logging verbosity (0=silent, 1=minimal, 2=full)
    """
    if device.type != 'cuda':
        return

    # cuDNN auto-tuning - finds optimal algorithms for specific input sizes
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    log_train.debug("cuDNN benchmark enabled")

    # TF32 for Ampere+ GPUs (faster matmul with negligible precision loss)
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')
        if verbose >= 2:
            print_metric("TF32 matmul", "enabled (high precision)", Colors.GREEN)


def maybe_compile_model(
    model: nn.Module,
    model_type: str,
    device: torch.device,
    use_compile: bool = True,
    verbose: int = 2,
) -> nn.Module:
    """
    Apply torch.compile if supported on the current platform.

    torch.compile requires Triton which is only available on Linux.
    Also skips compilation on Blackwell GPUs (sm_120+) to avoid compatibility issues.

    Args:
        model: PyTorch model to compile
        model_type: 'eegnet' or 'cbramod' (affects compile mode)
        device: PyTorch device
        use_compile: Whether to attempt compilation
        verbose: Logging verbosity (0=silent, 1=minimal, 2=full)

    Returns:
        Compiled model (or original if compilation skipped/failed)
    """
    from ..utils.device import is_blackwell_gpu

    is_windows = platform.system() == 'Windows'
    is_blackwell = is_blackwell_gpu()

    if is_windows:
        if verbose >= 2:
            print_metric("torch.compile", "skipped (Windows)", Colors.DIM)
        return model

    if is_blackwell:
        if verbose >= 2:
            print_metric("torch.compile", "skipped (Blackwell GPU)", Colors.DIM)
        return model

    if not use_compile:
        if verbose >= 2:
            print_metric("torch.compile", "disabled", Colors.DIM)
        return model

    if not hasattr(torch, 'compile') or device.type != 'cuda':
        if verbose >= 2:
            print_metric("torch.compile", "not available", Colors.DIM)
        return model

    try:
        # Use reduce-overhead for smaller EEGNet, default for larger CBraMod
        compile_mode = 'reduce-overhead' if model_type == 'eegnet' else 'default'
        model = torch.compile(model, mode=compile_mode)
        if verbose >= 2:
            print_metric("torch.compile", f"enabled ({compile_mode})", Colors.GREEN)
    except Exception as e:
        log_model.warning(f"torch.compile failed: {e}")
        if verbose >= 2:
            print_metric("torch.compile", "failed (fallback to eager)", Colors.YELLOW)

    return model


def get_scheduler_config_from_preset(
    scheduler_type: Optional[str],
    config: Optional[dict] = None,
    cross_subject: bool = False,
) -> dict:
    """
    Extract scheduler configuration from SCHEDULER_PRESETS.

    Merges preset defaults with any overrides from config['scheduler_config'].
    For cross-subject training, applies CROSS_SUBJECT_SCHEDULER_OVERRIDES.

    Args:
        scheduler_type: Scheduler name ('plateau', 'cosine', 'wsd', etc.)
        config: Training config dict (may contain 'scheduler_config' overrides)
        cross_subject: If True, apply cross-subject scheduler overrides

    Returns:
        Merged scheduler configuration dict with all parameters
    """
    from ..config.training import SCHEDULER_PRESETS, CROSS_SUBJECT_SCHEDULER_OVERRIDES

    # Get preset defaults (empty dict if scheduler_type not found)
    scheduler_config = SCHEDULER_PRESETS.get(scheduler_type, {}).copy()

    # Apply cross-subject scheduler overrides (longer phases, more aggressive decay)
    if cross_subject and scheduler_type in CROSS_SUBJECT_SCHEDULER_OVERRIDES:
        scheduler_config.update(CROSS_SUBJECT_SCHEDULER_OVERRIDES[scheduler_type])

    # Apply config overrides if provided (highest priority)
    if config and 'scheduler_config' in config:
        scheduler_config.update(config['scheduler_config'])

    return scheduler_config


def create_two_phase_loaders(
    dataset,
    train_indices: List[int],
    val_indices: List[int],
    scheduler_config: dict,
    main_batch_size: int,
    num_workers: int = 0,
    verbose: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Create DataLoaders for two-phase batch size training strategy.

    Two-phase strategy:
    - Exploration phase (first N epochs): Small batch size for more gradient updates
      and better loss landscape exploration
    - Main phase (remaining epochs): Normal batch size for stable training

    Args:
        dataset: Training dataset (FingerEEGDataset)
        train_indices: Indices for training samples
        val_indices: Indices for validation samples
        scheduler_config: Scheduler config with exploration_epochs, exploration_batch_size
        main_batch_size: Batch size for main training phase
        num_workers: DataLoader workers (default: 0, fastest for in-memory data)
        verbose: Logging verbosity (0=silent, 1=minimal, 2=full)

    Returns:
        Tuple of (exploration_loader, val_loader, main_train_loader, exploration_epochs)
    """
    from .train_within_subject import create_data_loaders_from_dataset

    # Extract exploration parameters from scheduler config
    exploration_epochs = scheduler_config.get('exploration_epochs', 5)
    exploration_batch_size = scheduler_config.get('exploration_batch_size', 32)

    # Create exploration loader (small batch for loss landscape exploration)
    exploration_loader, val_loader = create_data_loaders_from_dataset(
        dataset, train_indices, val_indices,
        batch_size=exploration_batch_size,
        num_workers=num_workers,
        shuffle_train=True,
    )

    # Create main loader (normal batch for stable training)
    main_train_loader, _ = create_data_loaders_from_dataset(
        dataset, train_indices, val_indices,
        batch_size=main_batch_size,
        num_workers=num_workers,
        shuffle_train=True,
    )

    if verbose >= 2:
        print_metric(
            "Exploration batch size",
            f"{exploration_batch_size} (epochs 1-{exploration_epochs})",
            Colors.CYAN
        )
        print_metric(
            "Main batch size",
            f"{main_batch_size} (epochs {exploration_epochs+1}+)",
            Colors.CYAN
        )

    return exploration_loader, val_loader, main_train_loader, exploration_epochs


def apply_config_overrides(
    config: dict,
    config_overrides: Optional[Dict],
    log_prefix: str = "",
) -> dict:
    """
    Apply config overrides with scheduler preset support.

    Priority (high to low):
    1. User-specified config_overrides
    2. Scheduler preset (if new scheduler specified)
    3. Model default config

    Args:
        config: Base configuration dict
        config_overrides: User overrides to apply
        log_prefix: Prefix for log messages

    Returns:
        Updated config dict
    """
    from ..config.training import SCHEDULER_PRESETS

    if not config_overrides:
        return config

    training_overrides = config_overrides.get('training', {})
    new_scheduler = training_overrides.get('scheduler')

    # If a new scheduler is specified, apply its preset first
    if new_scheduler and new_scheduler in SCHEDULER_PRESETS:
        preset = SCHEDULER_PRESETS[new_scheduler]
        applied = []

        # Preset overrides model default, but not user-specified values
        for key in ('epochs', 'patience'):
            if key in preset and key not in training_overrides:
                config['training'][key] = preset[key]
                applied.append(f"{key}={preset[key]}")

        if applied:
            log_train.info(f"{log_prefix}Applied scheduler preset '{new_scheduler}': {', '.join(applied)}")

    # Apply all user overrides (highest priority)
    for section, overrides in config_overrides.items():
        if section in config and isinstance(overrides, dict):
            config[section].update(overrides)
        else:
            config[section] = overrides

    return config
