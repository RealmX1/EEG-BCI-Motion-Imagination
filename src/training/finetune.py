"""
Backward-compatible wrapper for finetune_subject().

The core finetune logic has been unified into train_within_subject.train_single_subject()
with optional pretrained_path/freeze_strategy parameters.

This module preserves the original finetune_subject() API for backward compatibility
with src/hpo/objectives.py and other callers.
"""

import logging
from typing import Dict, Optional

import torch

from src.training.finetune_utils import (
    FreezeStrategy,
    detect_model_type_from_checkpoint,
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
