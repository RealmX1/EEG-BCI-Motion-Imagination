#!/usr/bin/env python
"""
Individual Finetuning Script for EEG-BCI Project.

Finetunes a pretrained model (from cross-subject training) on individual
subject data with support for different freeze strategies.

Freeze Strategies:
- none: Full model finetuning (all parameters trainable)
- backbone: Freeze feature extractor, train only classifier
- partial: Freeze early layers, train later layers + classifier

Usage:
    # Finetune on a single subject
    uv run python scripts/run_finetune.py \\
        --pretrained checkpoints/cross_subject/eegnet_imagery_binary/best.pt \\
        --subject S01

    # Finetune on multiple subjects
    uv run python scripts/run_finetune.py \\
        --pretrained checkpoints/cross_subject/eegnet_imagery_binary/best.pt \\
        --subjects S01 S02 S03

    # Use backbone freeze strategy (faster, less overfitting)
    uv run python scripts/run_finetune.py \\
        --pretrained checkpoints/cross_subject/cbramod_imagery_binary/best.pt \\
        --subject S01 \\
        --freeze-strategy backbone
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.device import set_seed, check_cuda_available, get_device
from src.utils.logging import setup_logging
from src.preprocessing.data_loader import discover_available_subjects
from src.training.finetune import finetune_subject, finetune_all_subjects


setup_logging('finetune')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Finetune a pretrained model on individual subjects',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Freeze Strategies:
  none      - Full model finetuning, all parameters trainable
  backbone  - Freeze feature extractor, train only classifier (faster)
  partial   - Freeze early layers, train later layers + classifier

Examples:
  # Single subject finetuning
  uv run python scripts/run_finetune.py \\
      --pretrained checkpoints/cross_subject/eegnet_imagery_binary/best.pt \\
      --subject S01

  # Multiple subjects with backbone freeze
  uv run python scripts/run_finetune.py \\
      --pretrained checkpoints/cross_subject/cbramod_imagery_binary/best.pt \\
      --subjects S01 S02 S03 \\
      --freeze-strategy backbone

  # All available subjects
  uv run python scripts/run_finetune.py \\
      --pretrained checkpoints/cross_subject/eegnet_imagery_binary/best.pt \\
      --all-subjects
'''
    )

    # Required arguments
    parser.add_argument(
        '--pretrained', type=str, required=True,
        help='Path to pretrained model checkpoint (.pt file)'
    )

    # Subject selection (mutually exclusive)
    subject_group = parser.add_mutually_exclusive_group(required=True)
    subject_group.add_argument(
        '--subject', type=str,
        help='Single subject ID to finetune on'
    )
    subject_group.add_argument(
        '--subjects', nargs='+',
        help='Multiple subject IDs to finetune on'
    )
    subject_group.add_argument(
        '--all-subjects', action='store_true',
        help='Finetune on all available subjects'
    )

    # Freeze strategy
    parser.add_argument(
        '--freeze-strategy', type=str, default='none',
        choices=['none', 'backbone', 'partial'],
        help='Freeze strategy (default: none)'
    )

    # Data arguments
    parser.add_argument(
        '--data-root', type=str, default='data',
        help='Path to data directory (default: data)'
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
        help='Number of finetuning epochs (default: based on model/strategy)'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=None,
        help='Learning rate (default: based on model/strategy)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Batch size (default: 64 for EEGNet, 128 for CBraMod)'
    )
    parser.add_argument(
        '--patience', type=int, default=None,
        help='Early stopping patience (default: 5)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )

    # Output arguments
    parser.add_argument(
        '--output-dir', type=str, default='checkpoints/finetuned',
        help='Directory to save finetuned models (default: checkpoints/finetuned)'
    )
    parser.add_argument(
        '--results-file', type=str, default=None,
        help='Path to save results JSON (default: results/finetune_results.json)'
    )

    args = parser.parse_args()

    # Validate pretrained path
    pretrained_path = Path(args.pretrained)
    if not pretrained_path.exists():
        logger.error(f"Pretrained model not found: {pretrained_path}")
        sys.exit(1)

    # Check GPU
    check_cuda_available(required=True)
    device = get_device()
    logger.info(f"Device: {device}")

    # Set seed
    set_seed(args.seed)

    # Determine subjects
    if args.subject:
        subjects = [args.subject]
    elif args.subjects:
        subjects = args.subjects
    else:  # --all-subjects
        subjects = discover_available_subjects(
            args.data_root, args.paradigm, args.task
        )
        if not subjects:
            logger.error(f"No subjects found in {args.data_root}")
            sys.exit(1)

    logger.info(f"Pretrained model: {pretrained_path}")
    logger.info(f"Freeze strategy: {args.freeze_strategy}")
    logger.info(f"Subjects to finetune: {subjects}")

    # Common kwargs for finetuning
    finetune_kwargs = {
        'freeze_strategy': args.freeze_strategy,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'patience': args.patience,
        'save_dir': args.output_dir,
        'data_root': args.data_root,
        'paradigm': args.paradigm,
        'task': args.task,
        'device': device,
        'seed': args.seed,
    }

    # Run finetuning
    if len(subjects) == 1:
        results = finetune_subject(
            pretrained_path=str(pretrained_path),
            subject_id=subjects[0],
            **finetune_kwargs,
        )
        all_results = {subjects[0]: results}
    else:
        all_results = finetune_all_subjects(
            pretrained_path=str(pretrained_path),
            subjects=subjects,
            **finetune_kwargs,
        )

    # Compute summary statistics
    successful_results = {k: v for k, v in all_results.items() if 'error' not in v}

    if successful_results:
        test_accs = [v['test_acc'] for v in successful_results.values()]
        mean_acc = np.mean(test_accs)
        std_acc = np.std(test_accs)

        # Print final summary
        print("\n" + "=" * 70)
        print(" FINETUNING COMPLETE")
        print("=" * 70)
        print(f"  Pretrained: {pretrained_path.parent.name}")
        print(f"  Freeze strategy: {args.freeze_strategy}")
        print(f"  Subjects finetuned: {len(successful_results)}/{len(subjects)}")
        print(f"  Mean test acc: {mean_acc:.2%} +/- {std_acc:.2%}")
        print("\n  Per-subject results:")
        for subject_id, result in sorted(successful_results.items()):
            print(f"    {subject_id}: test={result['test_acc']:.2%}, "
                  f"val={result['val_acc']:.2%}, "
                  f"epochs={result.get('epochs_trained', 'N/A')}")
        print("=" * 70)

        # Save summary results
        results_file = args.results_file
        if results_file is None:
            results_dir = Path('results/finetune')
            results_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = results_dir / f'finetune_{args.freeze_strategy}_{timestamp}.json'

        summary = {
            'pretrained_path': str(pretrained_path),
            'freeze_strategy': args.freeze_strategy,
            'paradigm': args.paradigm,
            'task': args.task,
            'n_subjects': len(successful_results),
            'mean_test_acc': mean_acc,
            'std_test_acc': std_acc,
            'per_subject': {
                k: {
                    'test_acc': v['test_acc'],
                    'val_acc': v['val_acc'],
                    'training_time': v.get('training_time', 0),
                }
                for k, v in successful_results.items()
            },
        }

        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Results saved: {results_file}")

    else:
        logger.error("No successful finetuning runs!")
        sys.exit(1)

    return 0


if __name__ == '__main__':
    sys.exit(main())
