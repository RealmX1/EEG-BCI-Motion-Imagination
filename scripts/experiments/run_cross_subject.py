#!/usr/bin/env python
"""
Cross-Subject Pretraining Script for EEG-BCI Project.

Trains a model on combined data from multiple subjects, creating a
pretrained model that can be finetuned on individual subjects.

Usage:
    # Train on all available subjects (default)
    uv run python scripts/run_cross_subject.py --model eegnet

    # Train on specific subjects
    uv run python scripts/run_cross_subject.py --model cbramod --subjects S01 S02 S03 S04 S05

    # Motor Execution paradigm
    uv run python scripts/run_cross_subject.py --model eegnet --paradigm movement

    # Custom training parameters
    uv run python scripts/run_cross_subject.py --model eegnet --epochs 100 --batch-size 256
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path (scripts/experiments/ -> scripts/ -> project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.device import set_seed, check_cuda_available, get_device
from src.utils.logging import setup_logging
from src.preprocessing.data_loader import discover_available_subjects
from src.training.train_cross_subject import train_cross_subject


setup_logging('cross_subject')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Cross-subject pretraining for EEG-BCI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Train EEGNet on all subjects
  uv run python scripts/run_cross_subject.py --model eegnet

  # Train CBraMod on specific subjects
  uv run python scripts/run_cross_subject.py --model cbramod --subjects S01 S02 S03

  # Motor Execution paradigm with custom epochs
  uv run python scripts/run_cross_subject.py --model eegnet --paradigm movement --epochs 100
'''
    )

    # Required arguments
    parser.add_argument(
        '--model', type=str, required=True,
        choices=['eegnet', 'cbramod'],
        help='Model type to train'
    )

    # Data arguments
    parser.add_argument(
        '--data-root', type=str, default='data',
        help='Path to data directory (default: data)'
    )
    parser.add_argument(
        '--subjects', nargs='+', default=None,
        help='Specific subjects to train on (default: all available)'
    )
    parser.add_argument(
        '--paradigm', type=str, default='imagery',
        choices=['imagery', 'movement'],
        help='Experiment paradigm (default: imagery)'
    )
    parser.add_argument(
        '--task', type=str, default='binary',
        choices=['binary', 'ternary', 'quaternary'],
        help='Classification task (default: binary)'
    )

    # Training arguments
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Number of training epochs (default: 50 for EEGNet, 30 for CBraMod)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Batch size (default: 128 for EEGNet, 256 for CBraMod)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )

    # Output arguments
    parser.add_argument(
        '--output-dir', type=str, default='checkpoints/cross_subject',
        help='Directory to save pretrained model (default: checkpoints/cross_subject)'
    )

    # WandB arguments
    parser.add_argument(
        '--wandb', action='store_true',
        help='Enable WandB logging'
    )

    args = parser.parse_args()

    # Check GPU
    check_cuda_available(required=True)
    device = get_device()
    logger.info(f"Device: {device}")

    # Set seed
    set_seed(args.seed)
    logger.info(f"Seed: {args.seed}")

    # Discover subjects if not specified
    if args.subjects:
        subjects = args.subjects
    else:
        subjects = discover_available_subjects(
            args.data_root, args.paradigm, args.task
        )

    if not subjects:
        logger.error(f"No subjects found in {args.data_root}")
        sys.exit(1)

    logger.info(f"Model: {args.model.upper()}")
    logger.info(f"Paradigm: {args.paradigm}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Subjects: {subjects}")

    # Run cross-subject training
    results = train_cross_subject(
        subjects=subjects,
        model_type=args.model,
        task=args.task,
        paradigm=args.paradigm,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.output_dir,
        data_root=args.data_root,
        device=device,
        seed=args.seed,
        wandb_enabled=args.wandb,
    )

    # Print final summary
    print("\n" + "=" * 70)
    print(" CROSS-SUBJECT PRETRAINING COMPLETE")
    print("=" * 70)
    print(f"  Model saved: {results['model_path']}")
    print(f"  Mean test acc: {results['mean_test_acc']:.2%} +/- {results['std_test_acc']:.2%}")
    print(f"  Best val acc: {results['val_acc']:.2%}")
    print(f"  Training time: {results['training_time']:.1f}s")
    print("\n  Per-subject test accuracy:")
    for subject_id, acc in sorted(results['per_subject_test_acc'].items()):
        print(f"    {subject_id}: {acc:.2%}")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
