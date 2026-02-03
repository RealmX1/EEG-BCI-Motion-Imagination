#!/usr/bin/env python
"""
Transfer Learning Comparison Script for EEG-BCI Project.

Compares three training strategies:
1. Baseline: Train from scratch on each subject (within-subject)
2. Cross-subject: Pretrain on all subjects, evaluate directly
3. Pretrain + Finetune: Pretrain, then finetune on each subject

This script runs the complete experiment and generates comparison reports.

Usage:
    # Run full comparison with default settings
    uv run python scripts/run_transfer_comparison.py --task binary

    # Run for specific model only
    uv run python scripts/run_transfer_comparison.py --task binary --models eegnet

    # Skip training, just compare existing results
    uv run python scripts/run_transfer_comparison.py --task binary --skip-training

    # Use different freeze strategy for finetuning
    uv run python scripts/run_transfer_comparison.py --task binary --freeze-strategy backbone
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.device import set_seed, check_cuda_available, get_device
from src.utils.logging import setup_logging
from src.preprocessing.data_loader import discover_available_subjects
from src.training.train_cross_subject import train_cross_subject
from src.training.finetune import finetune_all_subjects

# Import baseline training
from scripts._training_utils import (
    PARADIGM_CONFIG,
    TrainingResult,
    train_and_get_result,
    result_to_dict,
    compute_model_statistics,
    generate_result_filename,
)


setup_logging('transfer_comparison')
logger = logging.getLogger(__name__)


@dataclass
class TransferResult:
    """Result from a single transfer learning approach."""
    strategy: str  # 'baseline', 'cross_subject', 'finetuned'
    model_type: str
    subject_id: str
    test_acc: float
    val_acc: float
    training_time: float


def run_baseline_training(
    subjects: List[str],
    model_type: str,
    task: str,
    paradigm: str,
    data_root: str,
    output_dir: str,
) -> List[TransferResult]:
    """Run baseline within-subject training for all subjects."""
    results = []

    print("\n" + "=" * 70)
    print(f" BASELINE: {model_type.upper()} (Train from scratch)")
    print("=" * 70)

    for i, subject_id in enumerate(subjects, 1):
        print(f"\n  [{i}/{len(subjects)}] Training {subject_id}...")

        try:
            set_seed(42)
            training_result = train_and_get_result(
                subject_id=subject_id,
                model_type=model_type,
                task=task,
                paradigm=paradigm,
                data_root=data_root,
                save_dir=output_dir,
                no_wandb=True,
            )

            results.append(TransferResult(
                strategy='baseline',
                model_type=model_type,
                subject_id=subject_id,
                test_acc=training_result.test_acc_majority,
                val_acc=training_result.best_val_acc,
                training_time=training_result.training_time,
            ))

            print(f"    Test acc: {training_result.test_acc_majority:.2%}")

        except Exception as e:
            logger.error(f"Baseline training failed for {subject_id}: {e}")

    return results


def run_transfer_comparison(
    subjects: List[str],
    model_type: str,
    task: str,
    paradigm: str,
    data_root: str,
    output_dir: str,
    freeze_strategy: str = 'none',
    skip_baseline: bool = False,
    skip_cross_subject: bool = False,
) -> Dict[str, List[TransferResult]]:
    """
    Run complete transfer learning comparison.

    Args:
        subjects: List of subject IDs
        model_type: 'eegnet' or 'cbramod'
        task: Classification task
        paradigm: 'imagery' or 'movement'
        data_root: Path to data
        output_dir: Output directory
        freeze_strategy: Freeze strategy for finetuning
        skip_baseline: Skip baseline training
        skip_cross_subject: Skip cross-subject pretraining

    Returns:
        Dict mapping strategy -> list of results
    """
    all_results = {
        'baseline': [],
        'cross_subject': [],
        'finetuned': [],
    }

    device = get_device()

    # ========== 1. BASELINE: TRAIN FROM SCRATCH ==========
    if not skip_baseline:
        baseline_results = run_baseline_training(
            subjects=subjects,
            model_type=model_type,
            task=task,
            paradigm=paradigm,
            data_root=data_root,
            output_dir=f'{output_dir}/baseline/{model_type}_{paradigm}_{task}',
        )
        all_results['baseline'] = baseline_results
    else:
        logger.info("Skipping baseline training")

    # ========== 2. CROSS-SUBJECT PRETRAINING ==========
    pretrained_path = None
    if not skip_cross_subject:
        print("\n" + "=" * 70)
        print(f" CROSS-SUBJECT: {model_type.upper()} (Pretrain on all)")
        print("=" * 70)

        cross_subject_results = train_cross_subject(
            subjects=subjects,
            model_type=model_type,
            task=task,
            paradigm=paradigm,
            save_dir=f'{output_dir}/cross_subject',
            data_root=data_root,
            device=device,
        )

        pretrained_path = cross_subject_results['model_path']

        # Store cross-subject results (direct evaluation without finetuning)
        for subject_id, test_acc in cross_subject_results['per_subject_test_acc'].items():
            all_results['cross_subject'].append(TransferResult(
                strategy='cross_subject',
                model_type=model_type,
                subject_id=subject_id,
                test_acc=test_acc,
                val_acc=cross_subject_results['val_acc'],
                training_time=cross_subject_results['training_time'] / len(subjects),
            ))
    else:
        # Try to find existing pretrained model
        pretrained_dir = Path(output_dir) / 'cross_subject' / f'{model_type}_{paradigm}_{task}'
        if (pretrained_dir / 'best.pt').exists():
            pretrained_path = str(pretrained_dir / 'best.pt')
            logger.info(f"Using existing pretrained model: {pretrained_path}")
        else:
            logger.warning("No pretrained model found, skipping finetuning")

    # ========== 3. PRETRAIN + FINETUNE ==========
    if pretrained_path:
        print("\n" + "=" * 70)
        print(f" PRETRAIN + FINETUNE: {model_type.upper()} (freeze={freeze_strategy})")
        print("=" * 70)

        finetune_results = finetune_all_subjects(
            pretrained_path=pretrained_path,
            subjects=subjects,
            freeze_strategy=freeze_strategy,
            save_dir=f'{output_dir}/finetuned',
            data_root=data_root,
            paradigm=paradigm,
            task=task,
            device=device,
        )

        for subject_id, result in finetune_results.items():
            if 'error' not in result:
                all_results['finetuned'].append(TransferResult(
                    strategy='finetuned',
                    model_type=model_type,
                    subject_id=subject_id,
                    test_acc=result['test_acc'],
                    val_acc=result['val_acc'],
                    training_time=result['training_time'],
                ))

    return all_results


def print_comparison_report(
    results: Dict[str, List[TransferResult]],
    model_type: str,
    task: str,
):
    """Print formatted comparison report."""
    print("\n" + "=" * 80)
    print(f" TRANSFER LEARNING COMPARISON: {model_type.upper()} - {task.upper()}")
    print("=" * 80)

    # Collect per-subject data
    subjects = set()
    for strategy_results in results.values():
        for r in strategy_results:
            subjects.add(r.subject_id)
    subjects = sorted(subjects)

    # Create lookup tables
    lookup = {}
    for strategy, strategy_results in results.items():
        lookup[strategy] = {r.subject_id: r for r in strategy_results}

    # Per-subject table
    print(f"\n{'Subject':<10} {'Baseline':<12} {'Cross-Subj':<12} {'Finetuned':<12} {'Best':<15}")
    print("-" * 65)

    for subject_id in subjects:
        baseline_acc = lookup.get('baseline', {}).get(subject_id)
        cross_acc = lookup.get('cross_subject', {}).get(subject_id)
        finetune_acc = lookup.get('finetuned', {}).get(subject_id)

        b_str = f"{baseline_acc.test_acc:.2%}" if baseline_acc else "N/A"
        c_str = f"{cross_acc.test_acc:.2%}" if cross_acc else "N/A"
        f_str = f"{finetune_acc.test_acc:.2%}" if finetune_acc else "N/A"

        # Determine best
        accs = []
        if baseline_acc:
            accs.append(('baseline', baseline_acc.test_acc))
        if cross_acc:
            accs.append(('cross_subject', cross_acc.test_acc))
        if finetune_acc:
            accs.append(('finetuned', finetune_acc.test_acc))

        if accs:
            best_strategy, best_acc = max(accs, key=lambda x: x[1])
            best_str = f"{best_strategy} (+{best_acc - min(a for _, a in accs):.1%})"
        else:
            best_str = "N/A"

        print(f"{subject_id:<10} {b_str:<12} {c_str:<12} {f_str:<12} {best_str:<15}")

    # Summary statistics
    print("\n" + "-" * 65)
    print(f"{'Strategy':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 65)

    for strategy in ['baseline', 'cross_subject', 'finetuned']:
        strategy_results = results.get(strategy, [])
        if strategy_results:
            accs = [r.test_acc for r in strategy_results]
            print(f"{strategy:<20} {np.mean(accs):.2%}       {np.std(accs):.2%}       "
                  f"{np.min(accs):.2%}       {np.max(accs):.2%}")
        else:
            print(f"{strategy:<20} N/A")

    # Statistical comparison (if we have both baseline and finetuned)
    if results.get('baseline') and results.get('finetuned'):
        baseline_by_subj = {r.subject_id: r.test_acc for r in results['baseline']}
        finetune_by_subj = {r.subject_id: r.test_acc for r in results['finetuned']}
        common_subjects = set(baseline_by_subj.keys()) & set(finetune_by_subj.keys())

        if len(common_subjects) >= 2:
            baseline_accs = [baseline_by_subj[s] for s in sorted(common_subjects)]
            finetune_accs = [finetune_by_subj[s] for s in sorted(common_subjects)]

            t_stat, p_value = stats.ttest_rel(finetune_accs, baseline_accs)
            mean_diff = np.mean(finetune_accs) - np.mean(baseline_accs)

            print("\n" + "-" * 65)
            print(" STATISTICAL COMPARISON: Finetuned vs Baseline")
            print("-" * 65)
            print(f"  Mean difference: {mean_diff:+.2%}")
            print(f"  Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
            if p_value < 0.05:
                winner = "Finetuned" if mean_diff > 0 else "Baseline"
                print(f"  Result: {winner} is significantly better (p < 0.05)")
            else:
                print(f"  Result: No significant difference (p = {p_value:.4f})")

    print("=" * 80 + "\n")


def save_comparison_results(
    results: Dict[str, List[TransferResult]],
    model_type: str,
    task: str,
    paradigm: str,
    output_dir: str,
):
    """Save comparison results to JSON."""
    output = {
        'metadata': {
            'model_type': model_type,
            'task': task,
            'paradigm': paradigm,
            'timestamp': datetime.now().isoformat(),
        },
        'strategies': {},
    }

    for strategy, strategy_results in results.items():
        if strategy_results:
            accs = [r.test_acc for r in strategy_results]
            output['strategies'][strategy] = {
                'subjects': [asdict(r) for r in strategy_results],
                'summary': {
                    'mean': float(np.mean(accs)),
                    'std': float(np.std(accs)),
                    'min': float(np.min(accs)),
                    'max': float(np.max(accs)),
                    'n_subjects': len(accs),
                },
            }

    # Save to file
    results_dir = Path(output_dir) / 'transfer_comparison'
    results_dir.mkdir(parents=True, exist_ok=True)

    filename = generate_result_filename(f'transfer_{model_type}', paradigm, task, 'json')
    output_path = results_dir / filename

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved: {output_path}")
    return output_path


def generate_comparison_plot(
    results: Dict[str, List[TransferResult]],
    model_type: str,
    paradigm: str,
    task: str,
    output_dir: str,
):
    """Generate comparison visualization."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping plot")
        return

    # Collect data
    strategies = ['baseline', 'cross_subject', 'finetuned']
    strategy_labels = ['Baseline\n(from scratch)', 'Cross-Subject\n(no finetune)', 'Pretrain +\nFinetune']
    colors = ['#2E86AB', '#A23B72', '#F18F01']

    # Get subjects and accuracies
    subjects = set()
    for strategy_results in results.values():
        for r in strategy_results:
            subjects.add(r.subject_id)
    subjects = sorted(subjects)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Per-subject grouped bar chart
    ax1 = axes[0]
    x = np.arange(len(subjects))
    width = 0.25

    for i, (strategy, color, label) in enumerate(zip(strategies, colors, strategy_labels)):
        strategy_results = results.get(strategy, [])
        lookup = {r.subject_id: r.test_acc for r in strategy_results}
        accs = [lookup.get(s, 0) for s in subjects]

        ax1.bar(x + (i - 1) * width, accs, width, label=label.replace('\n', ' '), color=color, alpha=0.8)

    ax1.set_xlabel('Subject')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title(f'{model_type.upper()} Per-Subject Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(subjects, rotation=45)
    ax1.legend()
    ax1.set_ylim([0, 1])

    # Chance level
    chance = {'binary': 0.5, 'ternary': 1/3, 'quaternary': 0.25}.get(task, 0.5)
    ax1.axhline(y=chance, color='gray', linestyle='--', alpha=0.5)

    # Panel 2: Box plot comparison
    ax2 = axes[1]
    box_data = []
    box_labels = []

    for strategy, label in zip(strategies, strategy_labels):
        strategy_results = results.get(strategy, [])
        if strategy_results:
            accs = [r.test_acc for r in strategy_results]
            box_data.append(accs)
            box_labels.append(label)

    if box_data:
        bp = ax2.boxplot(box_data, tick_labels=box_labels, patch_artist=True,
                        showmeans=True, meanline=True)
        for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax2.set_ylabel('Test Accuracy')
    ax2.set_title(f'{model_type.upper()} Strategy Comparison')
    ax2.axhline(y=chance, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Save plot
    results_dir = Path(output_dir) / 'transfer_comparison'
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_filename = generate_result_filename(f'transfer_{model_type}', paradigm, task, 'png')
    plot_path = results_dir / plot_filename
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"Plot saved: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Transfer learning comparison experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Full comparison for EEGNet
  uv run python scripts/run_transfer_comparison.py --task binary --models eegnet

  # Full comparison for both models
  uv run python scripts/run_transfer_comparison.py --task binary

  # Use backbone freeze strategy
  uv run python scripts/run_transfer_comparison.py --task binary --freeze-strategy backbone
'''
    )

    parser.add_argument(
        '--task', type=str, default='binary',
        choices=['binary', 'ternary', 'quaternary'],
        help='Classification task (default: binary)'
    )
    parser.add_argument(
        '--models', nargs='+', default=['eegnet', 'cbramod'],
        choices=['eegnet', 'cbramod'],
        help='Models to compare (default: both)'
    )
    parser.add_argument(
        '--paradigm', type=str, default='imagery',
        choices=['imagery', 'movement'],
        help='Experiment paradigm (default: imagery)'
    )
    parser.add_argument(
        '--data-root', type=str, default='data',
        help='Path to data directory (default: data)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='results',
        help='Output directory (default: results)'
    )
    parser.add_argument(
        '--subjects', nargs='+', default=None,
        help='Specific subjects (default: all available)'
    )
    parser.add_argument(
        '--freeze-strategy', type=str, default='none',
        choices=['none', 'backbone', 'partial'],
        help='Freeze strategy for finetuning (default: none)'
    )
    parser.add_argument(
        '--skip-baseline', action='store_true',
        help='Skip baseline (from-scratch) training'
    )
    parser.add_argument(
        '--skip-cross-subject', action='store_true',
        help='Skip cross-subject pretraining (use existing)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help='Skip plot generation'
    )

    args = parser.parse_args()

    # Check GPU
    check_cuda_available(required=True)
    device = get_device()
    logger.info(f"Device: {device}")

    # Set seed
    set_seed(args.seed)

    # Discover subjects
    if args.subjects:
        subjects = args.subjects
    else:
        subjects = discover_available_subjects(
            args.data_root, args.paradigm, args.task
        )

    if not subjects:
        logger.error(f"No subjects found in {args.data_root}")
        sys.exit(1)

    logger.info(f"Subjects: {subjects}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Freeze strategy: {args.freeze_strategy}")

    start_time = time.time()

    # Run comparison for each model
    for model_type in args.models:
        print("\n" + "#" * 80)
        print(f"  MODEL: {model_type.upper()}")
        print("#" * 80)

        results = run_transfer_comparison(
            subjects=subjects,
            model_type=model_type,
            task=args.task,
            paradigm=args.paradigm,
            data_root=args.data_root,
            output_dir=args.output_dir,
            freeze_strategy=args.freeze_strategy,
            skip_baseline=args.skip_baseline,
            skip_cross_subject=args.skip_cross_subject,
        )

        # Print report
        print_comparison_report(results, model_type, args.task)

        # Save results
        save_comparison_results(
            results, model_type, args.task, args.paradigm, args.output_dir
        )

        # Generate plot
        if not args.no_plot:
            generate_comparison_plot(
                results, model_type, args.paradigm, args.task, args.output_dir
            )

    total_time = time.time() - start_time
    logger.info(f"Total time: {total_time/60:.1f} minutes")

    return 0


if __name__ == '__main__':
    sys.exit(main())
