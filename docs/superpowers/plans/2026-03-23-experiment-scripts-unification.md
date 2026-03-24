# Experiment Scripts Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify the three experiment paradigms (within-subject, cross-subject, transfer learning) to follow a consistent two-tier pattern: single-model script + thin comparison wrapper. Merge `finetune_subject()` into `train_single_subject()` with optional `pretrained_path`/`freeze_strategy` parameters. Extract shared components to `_training_utils.py`.

**Architecture:** Three layers of change: (1) `src/training/` — create `finetune_utils.py`, integrate finetune logic into `train_within_subject.py`, reduce `finetune.py` to backward-compat wrapper. (2) `scripts/_training_utils.py` — extract shared argparse builders, DB lifecycle, run tag resolution, output dir resolution. (3) `scripts/experiments/` — refactor all scripts to use shared components; extend `run_single_model.py` with `--pretrained`/`--freeze-strategy`; rewrite `run_transfer_comparison.py` to call `run_single_model()`; adopt shared argparse/DB/cache in `run_cross_subject_comparison.py`.

**Tech Stack:** Python 3.12, PyTorch, argparse, SQLite (ExperimentDB), JSON caching, WandB

**Spec:** `docs/superpowers/specs/2026-03-22-experiment-scripts-unification-design.md`

---

## File Structure

### New Files
- `src/training/finetune_utils.py` — Extracted finetune utilities: `load_pretrained_model()`, `apply_freeze_strategy()`, `get_finetune_optimizer()`, `get_default_finetune_config()`, finetune constants

### Modified Files
- `src/training/train_within_subject.py` (lines ~257-944, ~951-1053) — Add `pretrained_path`/`freeze_strategy` params to `train_single_subject()` and `train_subject_simple()`; integrate finetune flow when pretrained_path is provided
- `src/training/finetune.py` (entire file) — Reduce to ~50-line backward-compat wrapper delegating to `train_subject_simple()`; remove `finetune_all_subjects()`
- `src/training/__init__.py` — Remove `finetune_all_subjects` from imports/`__all__`
- `scripts/_training_utils.py` — Add shared argparse builders, `resolve_run_tag()`, `init_db_run()`, `finalize_db_run()`, `resolve_output_dir()`, `build_config_overrides()`, `find_best_checkpoint_path()`, `validate_checkpoint_compatibility()`. Extend `train_and_get_result()` with `pretrained_path`/`freeze_strategy` passthrough.
- `scripts/experiments/run_single_model.py` — Add `--pretrained`/`--freeze-strategy` CLI args; pass through to `train_and_get_result()`; use shared argparse/run-tag/output-dir
- `scripts/experiments/run_within_subject_comparison.py` — Use shared argparse builders, `init_db_run()`, `finalize_db_run()`, `resolve_run_tag()`, `resolve_output_dir()`, `build_config_overrides()`
- `scripts/experiments/run_transfer_comparison.py` — Rewrite to call `run_single_model()` with `pretrained_path`/`freeze_strategy`; use shared components; keep unique 6-way plotting
- `scripts/experiments/run_cross_subject_comparison.py` — Use shared argparse builders, `init_db_run()`, `finalize_db_run()`, `resolve_run_tag()`, `resolve_output_dir()`

### Deleted Files
- `scripts/experiments/run_finetune.py` — Functionality absorbed by `run_single_model.py --pretrained`
- `scripts/run_finetune.py` — Top-level thin wrapper for the above

---

## Task 1: Create `src/training/finetune_utils.py`

Extract pure utility functions from `src/training/finetune.py` into a new module. These functions have no side effects beyond model manipulation and are needed by both `train_within_subject.py` and the backward-compat `finetune.py` wrapper.

**Files:**
- Create: `src/training/finetune_utils.py`
- Reference: `src/training/finetune.py:83-260` (functions to extract)
- Reference: `src/config/training.py:265-326` (finetune override constants)

- [ ] **Step 1: Create `finetune_utils.py` with extracted functions**

Create the file with these functions moved verbatim from `finetune.py`:
- `load_pretrained_model()` (finetune.py:83-135)
- `apply_freeze_strategy()` (finetune.py:138-211)
- `get_finetune_optimizer()` (finetune.py:214-259)
- New function: `get_default_finetune_config()` — consolidate the scattered default logic from `finetune_subject()` lines 375-406

```python
"""
Finetune utility functions for EEG-BCI.

Extracted from src/training/finetune.py to be reusable by both
train_within_subject.py (unified flow) and finetune.py (backward-compat wrapper).
"""

import logging
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn

from src.models.eegnet import EEGNet
from src.models.cbramod_adapter import (
    CBraModForFingerBCI,
    get_default_pretrained_path,
)
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
) -> Tuple[nn.Module, dict]:
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
            pretrained_path=None,
            freeze_backbone=False,
            classifier_type=model_config.get('classifier_type', 'two_layer'),
            dropout=0.1,
        )
    else:
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

    Returns:
        Number of frozen parameters
    """
    if freeze_strategy == 'none':
        for param in model.parameters():
            param.requires_grad = True
        log_model.info("Freeze strategy: none (all parameters trainable)")
        return 0

    frozen_count = 0

    if model_type == 'cbramod':
        if freeze_strategy == 'backbone':
            if hasattr(model, 'backbone'):
                for param in model.backbone.parameters():
                    param.requires_grad = False
                    frozen_count += param.numel()
            log_model.info("Freeze strategy: backbone (transformer frozen, classifier trainable)")
        elif freeze_strategy == 'partial':
            if hasattr(model, 'backbone') and hasattr(model.backbone, 'transformer'):
                transformer = model.backbone.transformer
                if hasattr(transformer, 'encoder') and hasattr(transformer.encoder, 'layers'):
                    for i, layer in enumerate(transformer.encoder.layers):
                        if i < 6:
                            for param in layer.parameters():
                                param.requires_grad = False
                                frozen_count += param.numel()
            log_model.info("Freeze strategy: partial (first 6 transformer layers frozen)")
    else:  # EEGNet
        if freeze_strategy == 'backbone':
            layers_to_freeze = ['temporal_conv', 'spatial_conv', 'bn1', 'bn2']
            for name, param in model.named_parameters():
                if any(layer in name for layer in layers_to_freeze):
                    param.requires_grad = False
                    frozen_count += param.numel()
            log_model.info("Freeze strategy: backbone (block1 frozen, block2+fc trainable)")
        elif freeze_strategy == 'partial':
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
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if not trainable_params:
        raise ValueError("No trainable parameters! Check freeze strategy.")

    if model_type == 'cbramod' and freeze_strategy == 'none':
        if hasattr(model, 'get_parameter_groups'):
            param_groups = model.get_parameter_groups(
                backbone_lr=learning_rate,
                classifier_lr=learning_rate * 5,
            )
            for group in param_groups:
                group['params'] = [p for p in group['params'] if p.requires_grad]
            return torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    if model_type == 'cbramod':
        return torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    else:
        return torch.optim.Adam(trainable_params, lr=learning_rate, weight_decay=weight_decay)


def get_default_finetune_config(
    model_type: str,
    freeze_strategy: FreezeStrategy,
    n_channels: Optional[int] = None,
) -> Dict:
    """
    Get default finetune hyperparameters based on model type, freeze strategy, and channel count.

    Returns dict with keys: epochs, learning_rate, batch_size, scheduler_type
    """
    is_8ch = (n_channels == 8 and model_type == 'cbramod')
    is_32ch = (n_channels == 32 and model_type == 'cbramod')
    is_61ch = (n_channels == 61 and model_type == 'cbramod')

    # Epochs
    if is_8ch:
        epochs = EIGHT_CHANNEL_FINETUNE_OVERRIDES['epochs']
    elif is_32ch:
        epochs = THIRTYTWO_CHANNEL_FINETUNE_OVERRIDES['epochs']
    elif is_61ch:
        epochs = SIXTYONE_CHANNEL_FINETUNE_OVERRIDES['epochs']
    elif freeze_strategy == 'backbone':
        epochs = 20 if model_type == 'eegnet' else 10
    else:
        epochs = 30 if model_type == 'eegnet' else 15

    # Learning rate
    if is_8ch:
        learning_rate = EIGHT_CHANNEL_FINETUNE_OVERRIDES['learning_rate']
    elif is_32ch:
        learning_rate = THIRTYTWO_CHANNEL_FINETUNE_OVERRIDES['learning_rate']
    elif is_61ch:
        learning_rate = SIXTYONE_CHANNEL_FINETUNE_OVERRIDES['learning_rate']
    elif freeze_strategy == 'backbone':
        learning_rate = 5e-4
    elif freeze_strategy == 'partial':
        learning_rate = 1e-4
    else:
        learning_rate = 1e-4

    # Batch size
    batch_size = 64 if model_type == 'eegnet' else 128

    # Scheduler
    scheduler_type = 'plateau' if model_type == 'eegnet' else 'cosine_annealing_warmup_decay'

    return {
        'epochs': epochs,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'scheduler_type': scheduler_type,
    }


def detect_model_type_from_checkpoint(pretrained_path: str) -> str:
    """Load checkpoint metadata to determine model_type."""
    checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    return checkpoint['model_config']['model_type']
```

- [ ] **Step 2: Verify the new module imports correctly**

Run: `cd C:\Users\zhang\Desktop\github\EEG-BCI && uv run python -c "from src.training.finetune_utils import load_pretrained_model, apply_freeze_strategy, get_finetune_optimizer, get_default_finetune_config, detect_model_type_from_checkpoint; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/training/finetune_utils.py
git commit -m "refactor: extract finetune utilities to src/training/finetune_utils.py

Extract load_pretrained_model, apply_freeze_strategy, get_finetune_optimizer,
get_default_finetune_config from finetune.py into a reusable module."
```

---

## Task 2: Integrate finetune flow into `train_single_subject()` and `train_subject_simple()`

Add optional `pretrained_path` and `freeze_strategy` parameters to `train_single_subject()`. When `pretrained_path` is provided, use the finetune flow (load checkpoint, apply freeze, replace optimizer, evaluate baseline at epoch 0).

**Files:**
- Modify: `src/training/train_within_subject.py:257-280` (add params to `train_single_subject` signature)
- Modify: `src/training/train_within_subject.py:951-1053` (add params to `train_subject_simple` signature, pass through)
- Reference: `src/training/finetune.py:339-611` (finetune flow to integrate)

- [ ] **Step 1: Add `pretrained_path` and `freeze_strategy` to `train_single_subject()` signature**

At `train_within_subject.py:257`, add the two new optional parameters to the end of the signature, before the closing `) -> Dict:`:

```python
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
    preprocess_config: Optional[PreprocessConfig] = None,
    cache_only: bool = False,
    cache_index_path: str = ".cache_index.json",
    no_wandb: bool = False,
    upload_model: bool = False,
    wandb_project: str = 'eeg-bci',
    wandb_entity: Optional[str] = None,
    wandb_group: Optional[str] = None,
    verbose: int = 2,
    # ---- Transfer learning (optional) ----
    pretrained_path: Optional[str] = None,
    freeze_strategy: Optional[str] = None,
) -> Dict:
```

- [ ] **Step 2: Add finetune import and integration logic inside `train_single_subject()`**

After the model creation block (where `model` is first created and moved to device), add a conditional block. The exact insertion point is after the model is created but before the trainer is created. Look for the pattern where the model is created and `model = model.to(device)` is called.

Add this logic:

```python
    # ========== PRETRAINED MODEL LOADING (Transfer Learning) ==========
    pretrained_baseline = None
    if pretrained_path is not None:
        from src.training.finetune_utils import (
            load_pretrained_model,
            apply_freeze_strategy,
            get_finetune_optimizer,
            get_default_finetune_config,
        )
        from src.config.constants import TASKS

        if verbose >= 2:
            print_section_header("Loading Pretrained Model")

        # Replace the from-scratch model with the pretrained one
        model, checkpoint = load_pretrained_model(pretrained_path, device)
        model_config_ckpt = checkpoint['model_config']

        # Validate n_classes
        task_n_classes = TASKS[config['task']]['n_classes']
        ckpt_n_classes = model_config_ckpt['n_classes']
        if ckpt_n_classes != task_n_classes:
            ckpt_task = checkpoint.get('training_config', {}).get('task', 'unknown')
            raise ValueError(
                f"Checkpoint/task n_classes mismatch: pretrained model has "
                f"n_classes={ckpt_n_classes} (task='{ckpt_task}'), "
                f"but current task '{config['task']}' requires n_classes={task_n_classes}. "
                f"Checkpoint: {pretrained_path}"
            )

        if verbose >= 2:
            print_metric("Pretrained from", Path(pretrained_path).parent.name, Colors.CYAN)

        # Apply freeze strategy
        effective_freeze = freeze_strategy or 'none'
        frozen_count = apply_freeze_strategy(model, model_type, effective_freeze)
```

Note: The exact integration requires careful placement. The key principle: when `pretrained_path` is provided, the model loaded from checkpoint replaces the model created from scratch. The freeze strategy is applied, and then training proceeds as normal through the existing trainer flow.

**Important:** Do NOT apply finetune-specific default configs (epochs, learning_rate, batch_size, scheduler) inside `train_single_subject()`. The `config` dict it receives already has all values set (defaults + overrides merged). Finetune default application happens in `train_subject_simple()` (see Step 6 below), where `config_overrides` is still a separate parameter and we can tell whether the user explicitly overrode a value.

- [ ] **Step 3: Add finetune optimizer replacement after trainer creation**

After the `WithinSubjectTrainer` is instantiated (the `trainer = WithinSubjectTrainer(...)` call), add:

```python
    # Replace optimizer for finetune mode
    if pretrained_path is not None:
        from src.training.finetune_utils import get_finetune_optimizer
        effective_freeze = freeze_strategy or 'none'
        weight_decay = 0.05 if model_type == 'cbramod' else 0.0
        lr = config['training'].get('learning_rate', 1e-4)
        trainer.optimizer = get_finetune_optimizer(
            model, model_type, effective_freeze, lr, weight_decay
        )
```

- [ ] **Step 4: Add pretrained baseline evaluation before training**

Before `trainer.train()` is called, add the epoch 0 baseline evaluation:

```python
    # Evaluate pretrained baseline (epoch 0) before training
    if pretrained_path is not None:
        if verbose >= 2:
            print_section_header("Pretrained Baseline (Epoch 0)")

        baseline_val_loss, baseline_val_acc = trainer.validate(val_loader)
        baseline_majority_acc, _ = majority_vote_accuracy(
            model, train_dataset, val_indices, device, use_amp=True
        )
        baseline_combined = (baseline_val_acc + baseline_majority_acc) / 2.0

        if verbose >= 2:
            print_metric("Val Accuracy (segment)", f"{baseline_val_acc:.2%}", Colors.CYAN)
            print_metric("Val Accuracy (majority)", f"{baseline_majority_acc:.2%}", Colors.CYAN)
            print_metric("Combined Score", f"{baseline_combined:.2%}", Colors.YELLOW)

        # Set trainer's initial best to pretrained baseline
        trainer.best_val_acc = baseline_val_acc
        trainer.best_majority_acc = baseline_majority_acc
        trainer.best_combined_score = baseline_combined
        trainer.best_val_loss = baseline_val_loss
        trainer.best_epoch = 0
        trainer.best_state = model.state_dict().copy()

        # Initialize best_selection_score
        model_selection = config['training'].get('model_selection_strategy', 'combined')
        if model_selection == 'val_acc':
            trainer.best_selection_score = baseline_val_acc
        else:
            trainer.best_selection_score = baseline_combined

        # Save pretrained as initial best.pt
        torch.save({
            'model_state_dict': trainer.best_state,
            'epoch': 0,
            'val_acc': baseline_val_acc,
            'val_majority_acc': baseline_majority_acc,
            'combined_score': baseline_combined,
            'val_loss': baseline_val_loss,
        }, save_path / 'best.pt')

        pretrained_baseline = {
            'val_loss': baseline_val_loss,
            'val_acc': baseline_val_acc,
            'val_majority_acc': baseline_majority_acc,
            'combined_score': baseline_combined,
        }
```

- [ ] **Step 5: Include `pretrained_baseline` in the return dict**

In the results dict returned at the end of `train_single_subject()`, add:

```python
    if pretrained_baseline is not None:
        results['pretrained_baseline'] = pretrained_baseline
```

- [ ] **Step 6: Update `train_subject_simple()` — add params + finetune default application**

At `train_within_subject.py:951`, add `pretrained_path` and `freeze_strategy` to the signature:

```python
def train_subject_simple(
    ...,
    verbose: int = 2,
    # ---- Transfer learning (optional) ----
    pretrained_path: Optional[str] = None,
    freeze_strategy: Optional[str] = None,
) -> Dict:
```

**Critical: Apply finetune-specific defaults HERE, not in `train_single_subject()`.**

`train_subject_simple()` is the only place where both `config_overrides` (the user's explicit overrides) and `pretrained_path` are available as separate parameters. When `pretrained_path` is set, apply finetune defaults for values the user did NOT explicitly override:

```python
    # After get_default_config() but BEFORE apply_config_overrides():
    if pretrained_path is not None:
        from src.training.finetune_utils import get_default_finetune_config
        effective_freeze = freeze_strategy or 'none'
        n_ch = config_overrides.get('data', {}).get('channels') if config_overrides else None
        ft_defaults = get_default_finetune_config(model_type, effective_freeze, n_ch)

        # Build a finetune base config that will be overridden by config_overrides
        ft_overrides = {'training': {}}
        user_training = config_overrides.get('training', {}) if config_overrides else {}
        if 'epochs' not in user_training:
            ft_overrides['training']['epochs'] = ft_defaults['epochs']
        if 'learning_rate' not in user_training:
            ft_overrides['training']['learning_rate'] = ft_defaults['learning_rate']
        if 'batch_size' not in user_training:
            ft_overrides['training']['batch_size'] = ft_defaults['batch_size']
        if 'scheduler' not in user_training:
            ft_overrides['training']['scheduler'] = ft_defaults['scheduler_type']

        # Apply finetune defaults first, then user overrides on top
        config = apply_config_overrides(config, ft_overrides)
```

Then `apply_config_overrides(config, config_overrides)` runs after, so user overrides win.

And in the `return train_single_subject(...)` call, add:
```python
        pretrained_path=pretrained_path,
        freeze_strategy=freeze_strategy,
```

- [ ] **Step 7: Verify import works**

Run: `cd C:\Users\zhang\Desktop\github\EEG-BCI && uv run python -c "from src.training.train_within_subject import train_subject_simple; import inspect; sig = inspect.signature(train_subject_simple); print('pretrained_path' in sig.parameters and 'freeze_strategy' in sig.parameters)"`
Expected: `True`

- [ ] **Step 8: Commit**

```bash
git add src/training/train_within_subject.py
git commit -m "feat: integrate finetune flow into train_single_subject()

Add pretrained_path and freeze_strategy params. When pretrained_path is
provided: load checkpoint, validate n_classes, apply freeze strategy,
replace optimizer, evaluate pretrained baseline at epoch 0."
```

---

## Task 3: Reduce `finetune.py` to backward-compat wrapper

Replace the 840-line `finetune.py` with a ~60-line wrapper that delegates to `train_subject_simple()`. Remove `finetune_all_subjects()`.

**Files:**
- Modify: `src/training/finetune.py` (rewrite to wrapper)
- Modify: `src/training/__init__.py` (remove `finetune_all_subjects`)
- Reference: `src/hpo/objectives.py:326-363` (backward compat caller)

- [ ] **Step 1: Rewrite `finetune.py` as backward-compat wrapper**

Replace the entire file with:

```python
"""
Backward-compatible wrapper for finetune_subject().

The core finetune logic has been unified into train_within_subject.train_single_subject()
with optional pretrained_path/freeze_strategy parameters.

This module preserves the original finetune_subject() API for backward compatibility
with src/hpo/objectives.py and other callers.
"""

import logging
from pathlib import Path
from typing import Dict, Literal, Optional

import torch

from src.training.finetune_utils import (
    FreezeStrategy,
    detect_model_type_from_checkpoint,
    load_pretrained_model,
    apply_freeze_strategy,
    get_finetune_optimizer,
    get_default_finetune_config,
)
from src.training.train_within_subject import train_subject_simple

logger = logging.getLogger(__name__)


def finetune_subject(
    pretrained_path: str,
    subject_id: str,
    freeze_strategy: FreezeStrategy = 'none',
    run_tag: Optional[str] = None,
    epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    batch_size: Optional[int] = None,
    save_dir: str = 'checkpoints/finetuned',
    data_root: str = 'data',
    paradigm: str = 'imagery',
    task: str = 'binary',
    device: Optional[torch.device] = None,
    seed: int = 42,
    channels: Optional[int] = None,
    channel_config: Optional[str] = None,
    cache_only: bool = False,
    cache_index_path: str = ".cache_index.json",
    model_selection_strategy: str = 'combined',
    ema_decay: float = 0.998,
    soup_top_k: int = 3,
    no_wandb: bool = True,
    upload_model: bool = False,
    wandb_project: str = 'eeg-bci',
    wandb_entity: Optional[str] = None,
    wandb_group: Optional[str] = None,
    verbose: int = 2,
) -> Dict:
    """Backward-compatible wrapper. Delegates to train_subject_simple().

    Translates explicit finetune params (epochs, learning_rate, batch_size)
    into the config_overrides format used by the unified training function.
    """
    # Preserve seed behavior from original finetune_subject()
    from src.utils.device import set_seed
    set_seed(seed)

    model_type = detect_model_type_from_checkpoint(pretrained_path)

    # Build config_overrides from explicit params
    config_overrides = {'training': {}}
    if epochs is not None:
        config_overrides['training']['epochs'] = epochs
    if learning_rate is not None:
        config_overrides['training']['learning_rate'] = learning_rate
    if batch_size is not None:
        config_overrides['training']['batch_size'] = batch_size
    if model_selection_strategy != 'combined':
        config_overrides['training']['model_selection_strategy'] = model_selection_strategy
    config_overrides['training']['ema_decay'] = ema_decay
    config_overrides['training']['soup_top_k'] = soup_top_k

    # Channel overrides
    if channels is not None:
        config_overrides.setdefault('data', {})['channels'] = channels
    if channel_config is not None:
        config_overrides.setdefault('data', {})['channel_config'] = channel_config

    config_overrides = config_overrides if config_overrides.get('training') else None

    result_dict = train_subject_simple(
        subject_id=subject_id,
        model_type=model_type,
        task=task,
        paradigm=paradigm,
        data_root=data_root,
        save_dir=save_dir,
        device=device,
        run_tag=run_tag,
        config_overrides=config_overrides,
        cache_only=cache_only,
        cache_index_path=cache_index_path,
        no_wandb=no_wandb,
        upload_model=upload_model,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_group=wandb_group,
        verbose=verbose,
        pretrained_path=pretrained_path,
        freeze_strategy=freeze_strategy,
    )

    if not result_dict:
        raise ValueError(f"Training failed for {subject_id}")

    # Translate result_dict to the format expected by existing callers
    # (src/hpo/objectives.py expects 'test_acc', 'val_acc', etc.)
    return {
        'run_tag': result_dict.get('run_tag', run_tag),
        'model_path': result_dict.get('model_path', ''),
        'test_acc': result_dict.get('test_accuracy_majority', result_dict.get('test_accuracy', 0.0)),
        'val_acc': result_dict.get('best_val_acc', result_dict.get('val_accuracy', 0.0)),
        'val_majority_acc': result_dict.get('val_majority_acc', 0.0),
        'best_epoch': result_dict.get('best_epoch', 0),
        'epochs_trained': result_dict.get('epochs_trained', 0),
        'training_time': result_dict.get('training_time', 0.0),
        'history': result_dict.get('history', {}),
        'pretrained_baseline': result_dict.get('pretrained_baseline'),
        'milestone_test_results': result_dict.get('milestone_test_results', []),
    }
```

- [ ] **Step 2: Update `src/training/__init__.py`**

Remove `finetune_all_subjects` from the import and `__all__`:

Change line 32:
```python
from .finetune import finetune_subject, finetune_all_subjects
```
To:
```python
from .finetune import finetune_subject
```

Remove `'finetune_all_subjects'` from `__all__` list.

- [ ] **Step 3: Verify backward compat with HPO caller**

Run: `cd C:\Users\zhang\Desktop\github\EEG-BCI && uv run python -c "from src.training.finetune import finetune_subject; import inspect; sig = inspect.signature(finetune_subject); print('pretrained_path' in sig.parameters and 'freeze_strategy' in sig.parameters and 'epochs' in sig.parameters)"`
Expected: `True`

Also verify `finetune_all_subjects` is no longer importable from `__init__`:
Run: `cd C:\Users\zhang\Desktop\github\EEG-BCI && uv run python -c "from src.training import finetune_subject; print('OK')" && uv run python -c "try:\n    from src.training import finetune_all_subjects\n    print('FAIL: still importable')\nexcept ImportError:\n    print('OK: removed')"`

- [ ] **Step 4: Commit**

```bash
git add src/training/finetune.py src/training/__init__.py
git commit -m "refactor: reduce finetune.py to backward-compat wrapper

finetune_subject() now delegates to train_subject_simple() with
pretrained_path/freeze_strategy. Remove finetune_all_subjects()."
```

---

## Task 4: Expand `scripts/_training_utils.py` with shared abstractions

Add argparse builders, DB lifecycle helpers, run tag resolution, output dir resolution, config override builder, and transfer-specific helpers.

**Files:**
- Modify: `scripts/_training_utils.py`
- Reference: `scripts/experiments/run_within_subject_comparison.py:161-256` (argparse patterns to extract)
- Reference: `scripts/experiments/run_within_subject_comparison.py:304-338` (DB init pattern)
- Reference: `scripts/experiments/run_within_subject_comparison.py:260-299` (run tag resolution)
- Reference: `scripts/experiments/run_transfer_comparison.py:96-157` (checkpoint discovery)
- Reference: `scripts/experiments/run_transfer_comparison.py:742-765` (checkpoint validation)

- [ ] **Step 1: Add argparse builder functions**

Add after the existing `add_wandb_args()` function. Extract the duplicated argparse patterns from all three comparison scripts:

```python
def add_common_args(parser):
    """Add shared arguments: --data-root, --paradigm, --task, --seed, --output-dir, --no-plot, --subjects."""
    parser.add_argument('--data-root', type=str, default='data',
                        help='Path to data directory (default: data)')
    parser.add_argument('--subjects', nargs='+', default=None,
                        help='Specific subjects to run (default: all available)')
    parser.add_argument('--paradigm', type=str, default='imagery',
                        choices=['imagery', 'movement'],
                        help='Experiment paradigm (default: imagery)')
    parser.add_argument('--task', type=str, default='binary',
                        choices=['binary', 'ternary', 'quaternary', 'unified'],
                        help='Classification task (default: binary)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results (default: results)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Suppress plot generation')


def add_cache_resume_args(parser):
    """Add shared cache/resume arguments."""
    parser.add_argument('--resume', nargs='?', const='', default=None,
                        metavar='TAG',
                        help='Resume a previous run. Without TAG: resume most recent. '
                             'With TAG: resume run matching the datetime substring')
    parser.add_argument('--force-retrain', action='store_true',
                        help='Force retraining, ignore cache')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training, load existing results')
    parser.add_argument('--cache-only', action='store_true',
                        help='Load data exclusively from cache index (no filesystem scan)')
    parser.add_argument('--cache-index-path', type=str, default='.cache_index.json',
                        help='Path to cache index file (default: .cache_index.json)')


def add_channel_args(parser):
    """Add shared channel selection arguments."""
    from src.config.constants import FULL_N_CHANNELS, SUPPORTED_CHANNEL_COUNTS
    parser.add_argument('--channels', type=int, default=FULL_N_CHANNELS,
                        choices=SUPPORTED_CHANNEL_COUNTS,
                        help=f'Number of EEG channels (default: {FULL_N_CHANNELS})')
    parser.add_argument('--channel-config', type=str, default='motor_cortex',
                        help='Channel configuration name (default: motor_cortex)')


def add_training_config_args(parser):
    """Add shared training config arguments: --config, --scheduler, --classifier-type, --no-pretrained."""
    parser.add_argument('--config', type=str, default=None, metavar='YAML_PATH',
                        help='YAML config file path (overrides model defaults, CLI takes priority)')
    parser.add_argument('--scheduler', type=str, default=None,
                        choices=['plateau', 'cosine', 'wsd', 'cosine_decay', 'cosine_annealing_warmup_decay'],
                        help='Learning rate scheduler (default: model-specific)')
    parser.add_argument('--classifier-type', type=str, default=None,
                        choices=['two_layer', 'three_layer', 'one_layer', 'attention_pool'],
                        help='Override CBraMod classifier head type')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Train CBraMod from scratch (default: use pretrained)')


def add_transfer_args(parser):
    """Add transfer learning arguments: --pretrained, --freeze-strategy."""
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained checkpoint for transfer learning')
    parser.add_argument('--freeze-strategy', type=str, default='backbone',
                        choices=['none', 'backbone', 'partial'],
                        help='Freeze strategy for fine-tuning (default: backbone)')
```

- [ ] **Step 2: Add DB lifecycle and run tag helpers**

```python
def resolve_output_dir(args) -> str:
    """Auto-redirect to results/{n}_channel/{config}/ for reduced channel mode."""
    from src.config.constants import FULL_N_CHANNELS
    output_dir = getattr(args, 'output_dir', None) or getattr(args, 'results_dir', 'results')
    channels = getattr(args, 'channels', FULL_N_CHANNELS)
    channel_config = getattr(args, 'channel_config', 'motor_cortex')
    if channels != FULL_N_CHANNELS and output_dir == 'results':
        return f'results/{channels}_channel/{channel_config}'
    return output_dir


def resolve_run_tag(args, paradigm, task, output_dir, cache_type=None) -> str:
    """Handle --resume logic: find existing tag or generate new one."""
    from src.results.cache import find_cache_by_tag

    if args.resume is not None:
        tag_hint = args.resume if args.resume != '' else None
        found = find_cache_by_tag(output_dir, paradigm, task,
                                   tag_substring=tag_hint, cache_type=cache_type)
        if found:
            _, run_tag = found
            log_cache.info(f"Resuming run: {run_tag}")
            return run_tag
        else:
            import sys
            log_cache.error("No previous run found to resume")
            sys.exit(1)
    else:
        from datetime import datetime
        run_tag = datetime.now().strftime("%Y%m%d_%H%M")
        log_cache.info(f"Starting new run: {run_tag}")
        return run_tag


def init_db_run(run_tag, experiment_type, paradigm, task, args):
    """Create or resume ExperimentDB run. Returns (db, db_run_id)."""
    import shlex
    import sqlite3
    import sys
    from src.config.constants import FULL_N_CHANNELS
    from src.results import ExperimentDB

    db = ExperimentDB()
    db_run_id = None
    channels = getattr(args, 'channels', FULL_N_CHANNELS)
    channel_config = getattr(args, 'channel_config', 'motor_cortex')
    is_baseline = getattr(args, 'baseline', False)

    try:
        db_run_id = db.create_run(
            run_tag=run_tag,
            experiment_type=experiment_type,
            paradigm=paradigm,
            task=task,
            n_channels=channels,
            channel_config=channel_config if channels != FULL_N_CHANNELS else None,
            command=" ".join(shlex.quote(a) for a in sys.argv),
            is_baseline=is_baseline,
        )
        log_train.info(f"DB run created: {db_run_id}")
    except sqlite3.IntegrityError:
        existing = db.find_run_by_tag(
            run_tag, paradigm, task, experiment_type=experiment_type,
        )
        if existing:
            db_run_id = existing['run_id']
            log_train.info(f"DB run resumed: {db_run_id}")
            if is_baseline and not existing.get('is_baseline'):
                try:
                    db.set_baseline(db_run_id)
                except Exception:
                    pass
        else:
            log_train.warning(f"DB run creation failed: duplicate but tag not found")
    except Exception as e:
        log_train.warning(f"DB run creation failed: {e}")

    return db, db_run_id


def finalize_db_run(db, db_run_id, comparison, n_subjects, **extra):
    """Save comparison, mark complete, close DB."""
    if db_run_id:
        try:
            if comparison:
                db.save_comparison(db_run_id, comparison)
            db.update_n_subjects(db_run_id, n_subjects)

            # Save transfer config if provided
            transfer_config = extra.get('transfer_config')
            if transfer_config:
                db.save_transfer_config(db_run_id, **transfer_config)

            db.mark_complete(db_run_id)
        except Exception as e:
            log_train.warning(f"DB finalize failed: {e}")
    db.close()


def build_config_overrides(args) -> Optional[Dict]:
    """Build merged config_overrides dict from YAML + CLI args."""
    from src.config.training import load_yaml_config
    from src.config.constants import FULL_N_CHANNELS

    config_overrides = load_yaml_config(args.config) if getattr(args, 'config', None) else {}

    if getattr(args, 'scheduler', None):
        config_overrides.setdefault('training', {})['scheduler'] = args.scheduler
    if getattr(args, 'no_pretrained', False):
        config_overrides.setdefault('model', {})['no_pretrained'] = True

    channels = getattr(args, 'channels', FULL_N_CHANNELS)
    if channels != FULL_N_CHANNELS:
        config_overrides.setdefault('data', {})['channels'] = channels
        config_overrides.setdefault('data', {})['channel_config'] = getattr(args, 'channel_config', 'motor_cortex')

    if getattr(args, 'classifier_type', None):
        config_overrides.setdefault('model', {})['classifier_type'] = args.classifier_type

    return config_overrides or None
```

- [ ] **Step 3: Add checkpoint discovery and validation helpers (for transfer learning)**

```python
def find_best_checkpoint_path(
    model_type: str,
    paradigm: str,
    task: str,
    subjects: List[str],
    results_dir: str = 'results',
    n_channels: Optional[int] = None,
) -> Optional[str]:
    """Auto-discover best cross-subject pretrained checkpoint.

    Migrated from run_transfer_comparison.py.
    """
    import json
    import torch

    from src.results import find_compatible_cross_subject_results

    cross_result = find_compatible_cross_subject_results(
        output_dir=results_dir,
        paradigm=paradigm,
        task=task,
        subjects=subjects,
        model_type=model_type,
        n_channels=n_channels,
    )
    if not cross_result:
        return None

    source_file = cross_result['source_file']
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        model_path = data.get('training_info', {}).get('model_path', '')
        if model_path and Path(model_path).exists():
            log_io.info(f"Found {model_type} checkpoint: {model_path}")
            return model_path
    except (json.JSONDecodeError, OSError):
        pass

    # Fallback: search checkpoints directory
    checkpoint_dir = Path('checkpoints/cross_subject')
    if checkpoint_dir.exists():
        for subdir in sorted(checkpoint_dir.iterdir(), reverse=True):
            if subdir.is_dir() and model_type in subdir.name and paradigm in subdir.name and task in subdir.name:
                best_pt = subdir / 'best.pt'
                if best_pt.exists():
                    if n_channels is not None:
                        try:
                            ckpt = torch.load(best_pt, map_location='cpu', weights_only=False)
                            ckpt_channels = ckpt.get('model_config', {}).get('n_channels')
                            if ckpt_channels is not None and ckpt_channels != n_channels:
                                continue
                        except Exception:
                            continue
                    log_io.info(f"Found {model_type} checkpoint (fallback): {best_pt}")
                    return str(best_pt)

    return None


def validate_checkpoint_compatibility(pretrained_paths: Dict[str, str], task: str) -> Dict[str, str]:
    """Validate n_classes matches and extract classifier_types.

    Migrated from run_transfer_comparison.py.
    """
    import sys
    import torch
    from src.config.constants import TASKS

    classifier_types = {}
    expected_n_classes = TASKS[task]['n_classes']

    for model_type, path in pretrained_paths.items():
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            ct = ckpt.get('model_config', {}).get('classifier_type', 'two_layer')
            classifier_types[model_type] = ct
            ckpt_n_classes = ckpt.get('model_config', {}).get('n_classes')
            ckpt_task = ckpt.get('training_config', {}).get('task', 'unknown')
            if ckpt_n_classes is not None and ckpt_n_classes != expected_n_classes:
                log_train.error(
                    f"Checkpoint/task mismatch for {model_type.upper()}: "
                    f"pretrained n_classes={ckpt_n_classes} (task='{ckpt_task}'), "
                    f"but current task '{task}' expects n_classes={expected_n_classes}. "
                    f"Checkpoint: {path}"
                )
                sys.exit(1)
        except Exception:
            classifier_types[model_type] = 'unknown'

    return classifier_types
```

- [ ] **Step 4: Add `log_io` logger and update `train_and_get_result()` with pretrained passthrough**

Add `log_io = SectionLogger(logger, 'io')` near the existing loggers.

Update `train_and_get_result()` signature to accept and pass through `pretrained_path` and `freeze_strategy`:

```python
def train_and_get_result(
    ...,
    verbose: int = 2,
    # Transfer learning (optional)
    pretrained_path: Optional[str] = None,
    freeze_strategy: Optional[str] = None,
) -> TrainingResult:
```

And in the `train_subject_simple()` call inside, add:
```python
        pretrained_path=pretrained_path,
        freeze_strategy=freeze_strategy,
```

- [ ] **Step 5: Verify all new functions import correctly**

Run: `cd C:\Users\zhang\Desktop\github\EEG-BCI && uv run python -c "from scripts._training_utils import add_common_args, add_cache_resume_args, add_channel_args, add_training_config_args, add_transfer_args, resolve_output_dir, resolve_run_tag, init_db_run, finalize_db_run, build_config_overrides, find_best_checkpoint_path, validate_checkpoint_compatibility; print('OK')"`

Note: This import might need path adjustment. Run from the project root and verify.

- [ ] **Step 6: Commit**

```bash
git add scripts/_training_utils.py
git commit -m "feat: expand _training_utils.py with shared abstractions

Add argparse builders, DB lifecycle helpers, run tag resolution,
output dir resolution, config override builder, checkpoint discovery,
and transfer learning validation. Extend train_and_get_result() with
pretrained_path/freeze_strategy passthrough."
```

---

## Task 5: Refactor `run_single_model.py` — add `--pretrained`/`--freeze-strategy` support

Extend `run_single_model.py` to support transfer learning via `--pretrained` and `--freeze-strategy` CLI args, using shared argparse builders and passing through to the training function.

**Files:**
- Modify: `scripts/experiments/run_single_model.py`

- [ ] **Step 1: Update `run_single_model()` function to accept transfer learning params**

Add `pretrained_path` and `freeze_strategy` to the `run_single_model()` function signature (around line 94):

```python
def run_single_model(
    model_type: str,
    data_root: str,
    subject_ids: List[str],
    task: str,
    paradigm: str,
    output_dir: str,
    force_retrain: bool = False,
    run_tag: Optional[str] = None,
    no_wandb: bool = False,
    upload_model: bool = False,
    wandb_project: str = 'eeg-bci',
    wandb_entity: Optional[str] = None,
    cache_only: bool = False,
    cache_index_path: str = ".cache_index.json",
    config_overrides: Optional[Dict] = None,
    verbose_first_only: bool = True,
    db: Optional[ExperimentDB] = None,
    db_run_id: Optional[str] = None,
    # Transfer learning (optional)
    pretrained_path: Optional[str] = None,
    freeze_strategy: Optional[str] = None,
    # Cache type
    cache_type = None,  # CacheType enum, default None = WITHIN
) -> Tuple[List[TrainingResult], Dict]:
```

- [ ] **Step 2: Thread `cache_type` through all `load_cache`/`save_cache` calls**

Inside `run_single_model()`, update all cache function calls to pass `cache_type`:

At line ~163 (`load_cache` with `run_tag`):
```python
        cache, metadata = load_cache(output_dir, paradigm, task, run_tag, cache_type=cache_type)
```

At line ~172 (`load_cache` with `find_latest`):
```python
        cache, metadata = load_cache(output_dir, paradigm, task, find_latest=True, cache_type=cache_type)
```

At line ~258 (`save_cache` progressive):
```python
            save_cache(output_dir, paradigm, task, cache, run_tag,
                       wandb_groups=cache_wandb_groups,
                       extra_metadata=cache_extra_metadata,
                       cache_type=cache_type)
```

Also add `from src.config.constants import CacheType` to the imports at the top of the file.

This is critical: without this, transfer learning caches use `CacheType.WITHIN` filenames by default, which means transfer results and within-subject results collide, and `--resume` cannot find transfer caches.

- [ ] **Step 3: Pass pretrained params through to `train_and_get_result()`**

In the `train_and_get_result()` call (around line 232), add:
```python
                pretrained_path=pretrained_path,
                freeze_strategy=freeze_strategy,
```

- [ ] **Step 4: Add `--pretrained` and `--freeze-strategy` to CLI**

In `main()`, use the shared builder if already imported, or add manually:
```python
    # Transfer learning (optional)
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained checkpoint for transfer learning')
    parser.add_argument('--freeze-strategy', type=str, default=None,
                        choices=['none', 'backbone', 'partial'],
                        help='Freeze strategy for fine-tuning (default: none)')
```

- [ ] **Step 5: Pass the new args to `run_single_model()` in `main()`**

In the `run_single_model()` call in `main()` (around line 520), add:
```python
                pretrained_path=args.pretrained,
                freeze_strategy=args.freeze_strategy,
```

Also add `cache_type` support: when `--pretrained` is passed, use `CacheType.TRANSFER`:
```python
    from src.config.constants import CacheType
    cache_type = CacheType.TRANSFER if args.pretrained else None
    # ... pass cache_type=cache_type to run_single_model()
```

- [ ] **Step 6: Verify the script can parse the new arguments**

Run: `cd C:\Users\zhang\Desktop\github\EEG-BCI && uv run python scripts/experiments/run_single_model.py --help | grep -A1 "pretrained\|freeze-strategy"`
Expected: Both args appear in help output.

- [ ] **Step 7: Commit**

```bash
git add scripts/experiments/run_single_model.py
git commit -m "feat: add --pretrained/--freeze-strategy to run_single_model.py

Supports transfer learning mode. When --pretrained is specified, the
unified train_single_subject() loads the checkpoint and applies the
freeze strategy before training."
```

---

## Task 6: Refactor `run_within_subject_comparison.py` — use shared components

Replace duplicated argparse, DB init, run tag, output dir, and config override logic with calls to shared `_training_utils` functions. Preserve all existing behavior.

**Files:**
- Modify: `scripts/experiments/run_within_subject_comparison.py`

- [ ] **Step 1: Replace argparse with shared builders**

Replace the manual argparse definitions (lines 161-256) with calls to the shared builders. Keep script-specific args (like `--models`, `--skip-training`, `--baseline`, `--results-file`) inline.

Import the shared functions:
```python
from _training_utils import (
    discover_subjects,
    add_wandb_args,
    add_common_args,
    add_cache_resume_args,
    add_channel_args,
    add_training_config_args,
    resolve_output_dir,
    resolve_run_tag,
    init_db_run,
    finalize_db_run,
    build_config_overrides,
)
```

Replace the parser setup:
```python
    parser = argparse.ArgumentParser(...)
    add_common_args(parser)
    add_cache_resume_args(parser)
    add_channel_args(parser)
    add_training_config_args(parser)
    add_wandb_args(parser)

    # Script-specific args
    parser.add_argument('--models', nargs='+', default=['eegnet', 'cbramod'],
                        choices=['eegnet', 'cbramod'], help='Models to train')
    parser.add_argument('--results-file', type=str, default=None,
                        help='Path to existing results file (with --skip-training)')
    parser.add_argument('--baseline', action='store_true',
                        help='Mark this run as baseline in ExperimentDB')
```

- [ ] **Step 2: Replace inline DB init, run tag resolution, and output dir**

Replace the manual implementations (lines 260-338, 340-352) with:
```python
    output_dir = resolve_output_dir(args)
    run_tag = resolve_run_tag(args, args.paradigm, args.task, output_dir)
    db, db_run_id = init_db_run(run_tag, 'within_subject', args.paradigm, args.task, args)
    config_overrides = build_config_overrides(args)
```

- [ ] **Step 3: Replace inline DB finalization**

Replace the manual DB mark_complete/close block (lines 617-628) with:
```python
    n_subjects = len(set(r.subject_id for mrs in results.values() for r in mrs))
    finalize_db_run(db, db_run_id, comparison, n_subjects)
```

- [ ] **Step 4: Verify script still works with --help**

Run: `cd C:\Users\zhang\Desktop\github\EEG-BCI && uv run python scripts/experiments/run_within_subject_comparison.py --help`
Expected: All existing args present, no errors.

- [ ] **Step 5: Commit**

```bash
git add scripts/experiments/run_within_subject_comparison.py
git commit -m "refactor: use shared _training_utils in run_within_subject_comparison.py

Replace duplicated argparse, DB init/finalize, run tag resolution,
output dir resolution, and config override building with shared functions."
```

---

## Task 7: Refactor `run_transfer_comparison.py` — rewrite to call `run_single_model()`

The biggest change: rewrite the monolithic transfer script to call `run_single_model()` with `pretrained_path`/`freeze_strategy`, instead of reimplementing the training loop internally. Preserve the unique features: checkpoint auto-discovery, n_classes validation, 6-way plotting with DB-queried baselines.

**Files:**
- Modify: `scripts/experiments/run_transfer_comparison.py`

- [ ] **Step 1: Replace imports**

Replace the current imports with shared components:
```python
from _training_utils import (
    discover_subjects,
    add_wandb_args,
    add_common_args,
    add_cache_resume_args,
    add_channel_args,
    add_transfer_args,
    resolve_output_dir,
    resolve_run_tag,
    init_db_run,
    finalize_db_run,
    build_config_overrides,
    find_best_checkpoint_path,
    validate_checkpoint_compatibility,
)
from run_single_model import run_single_model
```

Remove imports that were only needed for the internal training loop:
- `finetune_subject` from `src.training.finetune`
- `print_subject_result`, `result_to_dict`, `dict_to_result` (handled by run_single_model)

- [ ] **Step 2: Remove internal functions**

Delete `finetune_and_get_result()` (lines 163-234) and `run_transfer_model()` (lines 237-450). These are replaced by `run_single_model()` with `pretrained_path`/`freeze_strategy`.

- [ ] **Step 3: Rewrite `main()` argparse**

Use shared builders and keep transfer-specific args:
```python
    parser = argparse.ArgumentParser(...)
    add_common_args(parser)
    add_cache_resume_args(parser)
    add_channel_args(parser)
    add_transfer_args(parser)
    add_wandb_args(parser)

    # Transfer-specific
    parser.add_argument('--models', nargs='+', default=['eegnet', 'cbramod'],
                        choices=['eegnet', 'cbramod'], help='Models to fine-tune')
    parser.add_argument('--pretrained-eegnet', type=str, default=None,
                        help='Manual path to pretrained EEGNet checkpoint')
    parser.add_argument('--pretrained-cbramod', type=str, default=None,
                        help='Manual path to pretrained CBraMod checkpoint')
    parser.add_argument('--finetune-epochs', type=int, default=None,
                        help='Fine-tuning epochs (default: strategy/model-specific)')
    parser.add_argument('--finetune-lr', type=float, default=None,
                        help='Fine-tuning learning rate')
    parser.add_argument('--finetune-batch-size', type=int, default=None,
                        help='Fine-tuning batch size')
    parser.add_argument('--no-cross-subject-baseline', action='store_true',
                        help='Exclude cross-subject baseline from plot')
    parser.add_argument('--baseline', action='store_true',
                        help='Mark as baseline in ExperimentDB')
```

- [ ] **Step 4: Rewrite training loop to use `run_single_model()`**

Replace the per-model training loop with:
```python
    results = {}
    for model_type in args.models:
        if model_type not in pretrained_paths:
            continue
        log_main.info(f"{'='*50} {model_type.upper()} TRANSFER {'='*50}")

        # Build transfer-specific config overrides
        transfer_overrides = dict(config_overrides) if config_overrides else {}
        if args.finetune_epochs:
            transfer_overrides.setdefault('training', {})['epochs'] = args.finetune_epochs
        if args.finetune_lr:
            transfer_overrides.setdefault('training', {})['learning_rate'] = args.finetune_lr
        if args.finetune_batch_size:
            transfer_overrides.setdefault('training', {})['batch_size'] = args.finetune_batch_size

        model_results, stats = run_single_model(
            model_type=model_type,
            data_root=args.data_root,
            subject_ids=subjects,
            task=args.task,
            paradigm=args.paradigm,
            output_dir=output_dir,
            force_retrain=args.force_retrain,
            run_tag=run_tag,
            no_wandb=args.no_wandb,
            upload_model=args.upload_model,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            cache_only=args.cache_only,
            cache_index_path=args.cache_index_path,
            config_overrides=transfer_overrides or None,
            db=db,
            db_run_id=db_run_id,
            pretrained_path=pretrained_paths[model_type],
            freeze_strategy=args.freeze_strategy,
            cache_type=CacheType.TRANSFER,
        )
        results[model_type] = model_results

        if db_run_id and model_results:
            try:
                db.save_summary(db_run_id, model_type, stats)
            except Exception as e:
                log_main.warning(f"DB summary failed: {e}")
```

- [ ] **Step 5: Preserve the 6-way plotting logic**

Keep the visualization section that builds PlotDataSource list with:
1-2: Within-subject baselines from `db.find_best_within_subject_results()` with `hatch='///'`
3-4: Cross-subject baselines from `db.find_best_cross_subject_results()` with `hatch='...'` (skippable via `--no-cross-subject-baseline`)
5-6: Current transfer results with `is_current_run=True`

This section can remain largely unchanged — just ensure it uses `results` from the new `run_single_model()` calls.

- [ ] **Step 6: Verify the script parses args correctly**

Run: `cd C:\Users\zhang\Desktop\github\EEG-BCI && uv run python scripts/experiments/run_transfer_comparison.py --help`
Expected: All existing args present plus the new shared ones.

- [ ] **Step 7: Commit**

```bash
git add scripts/experiments/run_transfer_comparison.py
git commit -m "refactor: rewrite run_transfer_comparison.py to use run_single_model()

Remove internal training loop (finetune_and_get_result, run_transfer_model).
Call run_single_model() with pretrained_path/freeze_strategy. Use shared
argparse, DB lifecycle, run tag resolution. Preserve 6-way plotting."
```

---

## Task 8: Refactor `run_cross_subject_comparison.py` — use shared components

Cross-subject training is fundamentally different (trains one model on all subjects combined), so it keeps its own training logic. But it should use the shared argparse builders, DB lifecycle, run tag resolution, and output dir resolution.

**Files:**
- Modify: `scripts/experiments/run_cross_subject_comparison.py`

- [ ] **Step 1: Replace argparse with shared builders**

Import shared functions and replace duplicated argparse definitions:
```python
from _training_utils import (
    discover_subjects,
    add_wandb_args,
    add_common_args,
    add_cache_resume_args,
    add_channel_args,
    add_training_config_args,
    resolve_output_dir,
    resolve_run_tag,
    init_db_run,
    finalize_db_run,
    build_config_overrides,
)
```

Use shared builders:
```python
    add_common_args(parser)
    add_cache_resume_args(parser)
    add_channel_args(parser)
    add_training_config_args(parser)
    add_wandb_args(parser)
```

Keep cross-subject-specific args inline: `--models`, `--epochs`, `--batch-size`, `--results-dir` (separate from `--output-dir`), `--no-within-subject-historical`, `--no-cross-subject-historical`, `--verbose`, `--quiet`, `--baseline`.

Note: cross_subject uses `--output-dir` for model checkpoints and `--results-dir` for results/plots. The shared `add_common_args()` adds `--output-dir`. Need to either:
- Override the default in parser after calling `add_common_args()`, or
- Keep `--output-dir` and `--results-dir` as separate cross-subject-specific args.

Decision: Keep `--results-dir` as cross-subject-specific (since it differs from the other scripts), and use `add_common_args()` but override `--output-dir` default to `checkpoints/cross_subject`.

- [ ] **Step 2: Replace inline DB init/finalize, run tag, config overrides**

```python
    output_dir = resolve_output_dir(args)  # For results/plots
    run_tag = resolve_run_tag(args, args.paradigm, args.task, output_dir,
                              cache_type=CacheType.CROSS_SUBJECT)
    db, db_run_id = init_db_run(run_tag, 'cross_subject', args.paradigm, args.task, args)
    config_overrides = build_config_overrides(args)
```

Note: `resolve_output_dir()` should use `args.results_dir` for cross-subject. This might need adjustment — the function checks for `args.output_dir` or `args.results_dir`. Ensure it works correctly for cross-subject's `--results-dir` pattern.

- [ ] **Step 3: Replace inline DB finalization**

```python
    n_subjects = len(subjects)
    finalize_db_run(db, db_run_id, comparison, n_subjects)
```

- [ ] **Step 4: Verify the script works with --help**

Run: `cd C:\Users\zhang\Desktop\github\EEG-BCI && uv run python scripts/experiments/run_cross_subject_comparison.py --help`

- [ ] **Step 5: Commit**

```bash
git add scripts/experiments/run_cross_subject_comparison.py
git commit -m "refactor: use shared _training_utils in run_cross_subject_comparison.py

Replace duplicated argparse, DB init/finalize, run tag resolution with
shared functions. Keep cross-subject-specific training loop unchanged."
```

---

## Task 9: Delete deprecated files

Remove `run_finetune.py` (both copies) — functionality absorbed by `run_single_model.py --pretrained`.

**Files:**
- Delete: `scripts/experiments/run_finetune.py`
- Delete: `scripts/run_finetune.py`

- [ ] **Step 1: Verify no other code imports from these files**

Run: `cd C:\Users\zhang\Desktop\github\EEG-BCI && grep -r "run_finetune" --include="*.py" -l`
Expected: Only the two files themselves (no other importers).

- [ ] **Step 2: Delete both files**

```bash
rm scripts/experiments/run_finetune.py scripts/run_finetune.py
```

- [ ] **Step 3: Commit**

```bash
git add -u scripts/experiments/run_finetune.py scripts/run_finetune.py
git commit -m "chore: delete deprecated run_finetune.py

Functionality absorbed by run_single_model.py --pretrained."
```

---

## Task 10: Update docs and migration checklist

Update `docs/codebase_reference.md` to reflect the new structure and run a final verification.

**Files:**
- Modify: `docs/codebase_reference.md` (remove run_finetune references, add --pretrained docs)
- Modify: `src/training/__init__.py` (verify final state)

- [ ] **Step 1: Update `docs/codebase_reference.md`**

Search for `run_finetune` references and replace with `run_single_model --pretrained`. Add documentation for the new `--pretrained`/`--freeze-strategy` args.

- [ ] **Step 2: Final import verification**

Run a comprehensive import check:
```bash
cd C:\Users\zhang\Desktop\github\EEG-BCI
uv run python -c "
from src.training import finetune_subject, train_subject_simple, WithinSubjectTrainer
from src.training.finetune_utils import load_pretrained_model, apply_freeze_strategy
from src.training.train_within_subject import train_subject_simple
import inspect
sig = inspect.signature(train_subject_simple)
assert 'pretrained_path' in sig.parameters
assert 'freeze_strategy' in sig.parameters
print('All imports OK')
"
```

- [ ] **Step 3: Verify HPO backward compat**

```bash
cd C:\Users\zhang\Desktop\github\EEG-BCI
uv run python -c "
from src.hpo.objectives import transfer_objective
print('HPO import OK')
"
```

- [ ] **Step 4: Verify all three comparison scripts parse correctly**

```bash
cd C:\Users\zhang\Desktop\github\EEG-BCI
uv run python scripts/experiments/run_within_subject_comparison.py --help > /dev/null && echo "within: OK"
uv run python scripts/experiments/run_cross_subject_comparison.py --help > /dev/null && echo "cross: OK"
uv run python scripts/experiments/run_transfer_comparison.py --help > /dev/null && echo "transfer: OK"
uv run python scripts/experiments/run_single_model.py --help | grep -q "pretrained" && echo "single: OK"
```

- [ ] **Step 5: Commit**

```bash
git add docs/codebase_reference.md
git commit -m "docs: update codebase_reference.md for unified experiment scripts

Remove run_finetune.py references, document --pretrained/--freeze-strategy."
```

---

## Task 11: Integration smoke test

Run a quick sanity check to ensure the unified flow works end-to-end.

**Files:**
- Reference: All modified files

- [ ] **Step 1: Verify `run_single_model.py --pretrained` works**

This requires an existing checkpoint. Check if one exists:
```bash
ls checkpoints/cross_subject/*/best.pt 2>/dev/null | head -1
```

If a checkpoint exists, run a quick smoke test (1 subject, minimal epochs):
```bash
cd C:\Users\zhang\Desktop\github\EEG-BCI
uv run python scripts/experiments/run_single_model.py \
    --model cbramod \
    --pretrained <CHECKPOINT_PATH> \
    --freeze-strategy backbone \
    --subjects S01 \
    --cache-only \
    --no-wandb \
    --no-plot
```

If no checkpoint exists, skip this step and note it in the commit.

- [ ] **Step 2: Verify backward compat — `finetune_subject()` still works**

```bash
cd C:\Users\zhang\Desktop\github\EEG-BCI
uv run python -c "
from src.training.finetune import finetune_subject
print(f'finetune_subject callable: {callable(finetune_subject)}')
import inspect
sig = inspect.signature(finetune_subject)
# Verify original params are still present
for p in ['pretrained_path', 'subject_id', 'freeze_strategy', 'epochs', 'learning_rate', 'batch_size']:
    assert p in sig.parameters, f'Missing: {p}'
print('Backward compat OK')
"
```

- [ ] **Step 3: Final commit with verification status**

If all checks pass, create a final summary commit:
```bash
git commit --allow-empty -m "chore: experiment scripts unification complete

Verified: imports, argparse, backward compat, docs updated."
```
