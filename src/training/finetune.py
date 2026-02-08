"""
Individual finetuning module for FINGER-EEG-BCI.

Finetunes a pretrained model (from cross-subject training) on a single
subject's data, with support for different freeze strategies.

Freeze Strategies:
- 'none': Full model finetuning (all parameters trainable)
- 'backbone': Freeze feature extractor, train only classifier
    - EEGNet: Freeze temporal_conv, spatial_conv, bn1, bn2
    - CBraMod: Freeze backbone (transformer)
- 'partial': Freeze early layers, train later layers + classifier
    - EEGNet: Freeze temporal_conv, bn1 only
    - CBraMod: Freeze first 6 transformer layers

Usage:
    from src.training.finetune import finetune_subject

    # Full finetuning
    results = finetune_subject(
        pretrained_path='checkpoints/cross_subject/eegnet_imagery_binary/best.pt',
        subject_id='S01',
        freeze_strategy='none',
    )

    # Backbone-frozen finetuning (faster, less overfitting)
    results = finetune_subject(
        pretrained_path='checkpoints/cross_subject/cbramod_imagery_binary/best.pt',
        subject_id='S01',
        freeze_strategy='backbone',
    )
"""

import logging
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.models.eegnet import EEGNet
from src.models.cbramod_adapter import (
    CBraModForFingerBCI,
    get_default_pretrained_path,
)
from src.preprocessing.data_loader import (
    FingerEEGDataset,
    PreprocessConfig,
    get_session_folders_for_split,
)
from src.training.train_within_subject import (
    WithinSubjectTrainer,
    majority_vote_accuracy,
    get_default_config,
    create_data_loaders_from_dataset,
)
from src.utils.device import get_device, set_seed
from src.utils.logging import SectionLogger
from src.utils.timing import Timer, print_section_header, print_metric, colored, Colors

logger = logging.getLogger(__name__)
log_data = SectionLogger(logger, 'data')
log_model = SectionLogger(logger, 'model')
log_train = SectionLogger(logger, 'train')


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
            classifier_type='two_layer',
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


def finetune_subject(
    pretrained_path: str,
    subject_id: str,
    freeze_strategy: FreezeStrategy = 'none',
    run_tag: Optional[str] = None,
    epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    batch_size: Optional[int] = None,
    patience: Optional[int] = None,
    save_dir: str = 'checkpoints/finetuned',
    data_root: str = 'data',
    paradigm: str = 'imagery',
    task: str = 'binary',
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> Dict:
    """
    Finetune a pretrained model on a single subject's data.

    Args:
        pretrained_path: Path to pretrained model checkpoint
        subject_id: Subject ID to finetune on (e.g., 'S01')
        freeze_strategy: 'none', 'backbone', or 'partial'
        run_tag: Optional run tag (timestamp) for this experiment (None = auto-generate)
        epochs: Number of finetuning epochs (None = use default)
        learning_rate: Learning rate (None = use default based on strategy)
        batch_size: Batch size (None = use default)
        patience: Early stopping patience (None = use default)
        save_dir: Directory to save finetuned model
        data_root: Path to data directory
        paradigm: 'imagery' or 'movement'
        task: 'binary', 'ternary', or 'quaternary'
        device: Device to use (None = auto-detect)
        seed: Random seed

    Returns:
        Dict with:
        - model_path: Path to saved finetuned model
        - test_acc: Test accuracy (majority voting)
        - val_acc: Best validation accuracy
        - training_time: Training time
        - history: Training history
    """
    total_start = time.perf_counter()
    Timer.reset()
    set_seed(seed)

    # Generate run_tag at start of finetuning (if not provided)
    if run_tag is None:
        run_tag = datetime.now().strftime('%Y%m%d_%H%M')

    if device is None:
        device = get_device()

    # ========== LOAD PRETRAINED MODEL ==========
    print()
    print(colored("=" * 70, Colors.BRIGHT_MAGENTA, bold=True))
    print(colored(f"  Finetuning: {subject_id} (freeze={freeze_strategy})", Colors.BRIGHT_MAGENTA, bold=True))
    print(colored("=" * 70, Colors.BRIGHT_MAGENTA, bold=True))

    print_section_header("Loading Pretrained Model")

    model, checkpoint = load_pretrained_model(pretrained_path, device)
    model_config = checkpoint['model_config']
    model_type = model_config['model_type']
    n_classes = model_config['n_classes']

    print_metric("Model type", model_type.upper(), Colors.CYAN)
    print_metric("Pretrained from", Path(pretrained_path).parent.name, Colors.CYAN)

    # Apply freeze strategy
    frozen_count = apply_freeze_strategy(model, model_type, freeze_strategy)

    # ========== GET TRAINING CONFIG ==========
    config = get_default_config(model_type, task)
    task_config = config['tasks'][task]
    target_classes = task_config['classes']

    # Set finetuning-specific defaults
    if epochs is None:
        if freeze_strategy == 'backbone':
            epochs = 20 if model_type == 'eegnet' else 10
        else:
            epochs = 30 if model_type == 'eegnet' else 15

    if learning_rate is None:
        if freeze_strategy == 'backbone':
            learning_rate = 5e-4  # Higher LR when only training classifier
        elif freeze_strategy == 'partial':
            learning_rate = 1e-4
        else:
            learning_rate = 1e-4 if model_type == 'cbramod' else 1e-4

    if batch_size is None:
        batch_size = 64 if model_type == 'eegnet' else 128

    if patience is None:
        patience = 5 if model_type == 'cbramod' else 5

    # ========== DATA LOADING ==========
    print_section_header(f"Data Loading ({subject_id})")

    data_root_path = Path(data_root)
    elc_path = data_root_path / 'biosemi128.ELC'

    # Preprocessing config
    if model_type == 'cbramod':
        preprocess_config = PreprocessConfig.for_cbramod(full_channels=True)
    else:
        preprocess_config = PreprocessConfig.paper_aligned(n_class=n_classes)

    # Get session folders
    train_folders = get_session_folders_for_split(paradigm, task, 'train')
    test_folders = get_session_folders_for_split(paradigm, task, 'test')

    # Load training data
    with Timer("train_data_loading", print_on_exit=True):
        train_dataset = FingerEEGDataset(
            str(data_root_path),
            [subject_id],
            preprocess_config,
            session_folders=train_folders,
            target_classes=target_classes,
            elc_path=str(elc_path),
        )

    if len(train_dataset) == 0:
        raise ValueError(f"No training data found for {subject_id}")

    # Load test data
    with Timer("test_data_loading", print_on_exit=True):
        test_dataset = FingerEEGDataset(
            str(data_root_path),
            [subject_id],
            preprocess_config,
            session_folders=test_folders,
            target_classes=target_classes,
            elc_path=str(elc_path),
        )

    print_metric("Train segments", len(train_dataset), Colors.CYAN)
    print_metric("Test segments", len(test_dataset), Colors.MAGENTA)

    # ========== TEMPORAL SPLIT ==========
    print_section_header("Data Splitting (Temporal)")

    # Same split logic as within-subject training
    unique_trials = train_dataset.get_unique_trials()

    # Group by session for stratified split
    session_to_trials = defaultdict(list)
    for trial_idx in unique_trials:
        for info in train_dataset.trial_infos:
            if info.trial_idx == trial_idx:
                session_to_trials[info.session_type].append(trial_idx)
                break

    val_ratio = 0.2
    train_trials = []
    val_trials = []

    for session_type, trials in session_to_trials.items():
        trials = sorted(set(trials))
        n_val = max(1, int(len(trials) * val_ratio))
        train_trials.extend(trials[:-n_val])
        val_trials.extend(trials[-n_val:])

    train_indices = train_dataset.get_segment_indices_for_trials(train_trials)
    val_indices = train_dataset.get_segment_indices_for_trials(val_trials)

    print_metric("Train segments", len(train_indices), Colors.GREEN)
    print_metric("Val segments", len(val_indices), Colors.YELLOW)

    # ========== DATALOADER CREATION ==========
    train_loader, val_loader = create_data_loaders_from_dataset(
        train_dataset,
        train_indices,
        val_indices,
        batch_size=batch_size,
        num_workers=0,
        shuffle_train=True,
    )

    # ========== OPTIMIZER SETUP ==========
    print_section_header("Finetuning Setup")

    weight_decay = 0.05 if model_type == 'cbramod' else 0.0
    optimizer = get_finetune_optimizer(
        model, model_type, freeze_strategy, learning_rate, weight_decay
    )

    print_metric("Epochs", epochs, Colors.CYAN)
    print_metric("Learning rate", f"{learning_rate:.0e}", Colors.CYAN)
    print_metric("Batch size", batch_size, Colors.CYAN)
    print_metric("Freeze strategy", freeze_strategy, Colors.YELLOW)

    # ========== TRAINER SETUP ==========
    # Create trainer (we'll use our custom optimizer)
    trainer = WithinSubjectTrainer(
        model,
        train_dataset,
        val_indices,
        device,
        model_type=model_type,
        n_classes=n_classes,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        scheduler_type='plateau' if model_type == 'eegnet' else 'wsd',
        use_amp=True,
        gradient_clip=1.0 if model_type == 'cbramod' else 0.0,
    )

    # Replace optimizer with our finetuning-aware one
    trainer.optimizer = optimizer

    # Setup save path
    save_path = Path(save_dir) / f'{run_tag}_{model_type}_{paradigm}_{task}' / subject_id
    save_path.mkdir(parents=True, exist_ok=True)

    # ========== TRAINING ==========
    with Timer("finetuning"):
        history = trainer.train(
            train_loader,
            val_loader,
            epochs=epochs,
            patience=patience,
            save_path=save_path,
            wandb_callback=None,
        )

    # ========== TEST EVALUATION ==========
    print_section_header("Test Evaluation")

    test_indices = list(range(len(test_dataset)))
    test_acc, test_results = majority_vote_accuracy(
        model, test_dataset, test_indices, device
    )

    test_color = Colors.BRIGHT_GREEN if test_acc > 0.7 else (
        Colors.YELLOW if test_acc > 0.5 else Colors.RED
    )
    print(f"  {colored('Test Accuracy:', Colors.WHITE, bold=True)} "
          f"{colored(f'{test_acc:.2%}', test_color, bold=True)}")

    # ========== SAVE RESULTS ==========
    total_time = time.perf_counter() - total_start

    results = {
        'subject_id': subject_id,
        'model_type': model_type,
        'task': task,
        'paradigm': paradigm,
        'freeze_strategy': freeze_strategy,
        'pretrained_path': pretrained_path,
        'test_acc': test_acc,
        'val_acc': trainer.best_val_acc,
        'val_majority_acc': trainer.best_majority_acc,
        'best_epoch': trainer.best_epoch,
        'epochs_trained': len(history['train_loss']),
        'training_time': total_time,
    }

    # Save results JSON
    with open(save_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save training history
    with open(save_path / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    Timer.print_summary(f"Finetuning {subject_id}")

    return {
        'run_tag': run_tag,
        'model_path': str(save_path / 'best.pt'),
        'test_acc': test_acc,
        'val_acc': trainer.best_val_acc,
        'val_majority_acc': trainer.best_majority_acc,
        'best_epoch': trainer.best_epoch,
        'training_time': total_time,
        'history': history,
    }


def finetune_all_subjects(
    pretrained_path: str,
    subjects: List[str],
    freeze_strategy: FreezeStrategy = 'none',
    run_tag: Optional[str] = None,
    **kwargs,
) -> Dict[str, Dict]:
    """
    Finetune a pretrained model on multiple subjects.

    Args:
        pretrained_path: Path to pretrained model
        subjects: List of subject IDs
        freeze_strategy: Freeze strategy to use
        run_tag: Optional shared run tag for all subjects (None = auto-generate)
        **kwargs: Additional arguments passed to finetune_subject

    Returns:
        Dict mapping subject_id -> results
    """
    # If run_tag not provided, generate a shared one for all subjects
    if run_tag is None:
        run_tag = datetime.now().strftime('%Y%m%d_%H%M')

    all_results = {}

    for i, subject_id in enumerate(subjects, 1):
        print(f"\n{'='*70}")
        print(f"  [{i}/{len(subjects)}] Finetuning {subject_id}")
        print('='*70)

        try:
            results = finetune_subject(
                pretrained_path=pretrained_path,
                subject_id=subject_id,
                freeze_strategy=freeze_strategy,
                run_tag=run_tag,
                **kwargs,
            )
            all_results[subject_id] = results

        except Exception as e:
            log_train.error(f"Failed to finetune {subject_id}: {e}")
            all_results[subject_id] = {'error': str(e)}

    # Print summary
    successful = {k: v for k, v in all_results.items() if 'error' not in v}
    if successful:
        test_accs = [v['test_acc'] for v in successful.values()]
        print(f"\n{'='*70}")
        print(f"  FINETUNING SUMMARY")
        print('='*70)
        print(f"  Subjects: {len(successful)}/{len(subjects)} successful")
        print(f"  Mean test acc: {np.mean(test_accs):.2%} +/- {np.std(test_accs):.2%}")
        print('='*70)

    return all_results


if __name__ == '__main__':
    import sys

    logging.basicConfig(level=logging.INFO)

    # This requires a pretrained model to exist
    # Run train_cross_subject.py first to create one

    pretrained = Path('checkpoints/cross_subject/eegnet_imagery_binary/best.pt')
    if not pretrained.exists():
        print(f"Pretrained model not found: {pretrained}")
        print("Run train_cross_subject.py first to create a pretrained model.")
        sys.exit(1)

    # Test finetuning
    results = finetune_subject(
        pretrained_path=str(pretrained),
        subject_id='S01',
        freeze_strategy='none',
        epochs=5,  # Quick test
    )

    print("\nResults:")
    print(f"  Test acc: {results['test_acc']:.2%}")
    print(f"  Model path: {results['model_path']}")
