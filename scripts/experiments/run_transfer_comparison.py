#!/usr/bin/env python
"""
Transfer Learning Comparison Script for EEG-BCI Project.

This script fine-tunes pretrained cross-subject models (from run_cross_subject_comparison.py)
on individual subjects and compares the transfer learning performance of EEGNet vs CBraMod.

Workflow:
1. Auto-discovers (or accepts manual) best cross-subject pretrained checkpoints
2. Fine-tunes each model on each individual subject
3. Compares fine-tuned EEGNet vs CBraMod (statistical tests)
4. Generates combined plot: cross-subject baseline + transfer learning results

Usage:
    # Auto-discover best pretrained models and fine-tune
    uv run python scripts/run_transfer_comparison.py

    # Specify freeze strategy
    uv run python scripts/run_transfer_comparison.py --freeze-strategy partial

    # Manual pretrained model paths
    uv run python scripts/run_transfer_comparison.py \\
        --pretrained-eegnet checkpoints/cross_subject/.../best.pt \\
        --pretrained-cbramod chec kpoints/cross_subject/.../best.pt

    # Resume a previous run
    uv run python scripts/run_transfer_comparison.py --resume
    uv run python scripts/run_transfer_comparison.py --resume 20260208

    # Motor Execution paradigm, ternary task
    uv run python scripts/run_transfer_comparison.py --paradigm movement --task ternary
"""

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path (scripts/experiments/ -> scripts/ -> project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.constants import PARADIGM_CONFIG
from src.utils.device import set_seed, check_cuda_available, get_device
from src.utils.logging import SectionLogger, setup_logging

from src.results import (
    CacheType,
    TrainingResult,
    compare_models,
    print_comparison_report,
    find_compatible_cross_subject_results,
    find_best_within_subject_for_model,
    generate_result_filename,
    result_to_dict,
    dict_to_result,
    compute_model_statistics,
    print_model_summary,
    cross_subject_result_to_training_results,
    build_transfer_data_sources,
    get_cache_path,
    find_cache_by_tag,
    load_cache,
    save_cache,
)
from src.visualization import generate_combined_plot
from src.training.finetune import finetune_subject

SCRIPTS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from _training_utils import discover_subjects, print_subject_result, add_wandb_args


setup_logging('transfer_comparison')
logger = logging.getLogger(__name__)

log_main = SectionLogger(logger, 'main')
log_train = SectionLogger(logger, 'train')
log_io = SectionLogger(logger, 'io')




# ============================================================================
# Checkpoint Discovery
# ============================================================================

def find_best_checkpoint_path(
    model_type: str,
    paradigm: str,
    task: str,
    subjects: List[str],
    results_dir: str = 'results',
) -> Optional[str]:
    """
    Auto-discover the best cross-subject pretrained checkpoint for a model.

    Searches results/ for compatible cross-subject result JSONs, then extracts
    the checkpoint path from training_info.model_path.
    """
    cross_result = find_compatible_cross_subject_results(
        output_dir=results_dir,
        paradigm=paradigm,
        task=task,
        subjects=subjects,
        model_type=model_type,
    )
    if not cross_result:
        return None

    # Read source JSON to extract model_path
    source_file = cross_result['source_file']
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        model_path = data.get('training_info', {}).get('model_path', '')
        if model_path and Path(model_path).exists():
            log_io.info(f"Found {model_type} checkpoint: {model_path}")
            return model_path
    except (json.JSONDecodeError, OSError) as e:
        log_io.debug(f"Failed to read source file: {e}")

    # Fallback: search checkpoints directory by pattern
    checkpoint_dir = Path('checkpoints/cross_subject')
    if checkpoint_dir.exists():
        # Match patterns like *_{model}_{paradigm}_{task}/best.pt
        for subdir in sorted(checkpoint_dir.iterdir(), reverse=True):
            if subdir.is_dir() and model_type in subdir.name and paradigm in subdir.name and task in subdir.name:
                best_pt = subdir / 'best.pt'
                if best_pt.exists():
                    log_io.info(f"Found {model_type} checkpoint (fallback): {best_pt}")
                    return str(best_pt)

    return None


# ============================================================================
# Training Helpers
# ============================================================================

def finetune_and_get_result(
    subject_id: str,
    model_type: str,
    pretrained_path: str,
    freeze_strategy: str,
    task: str,
    paradigm: str,
    data_root: str,
    run_tag: Optional[str] = None,
    epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    batch_size: Optional[int] = None,
    patience: Optional[int] = None,
    seed: int = 42,
    # Cache-only mode
    cache_only: bool = False,
    cache_index_path: str = ".cache_index.json",
    # WandB
    no_wandb: bool = True,
    upload_model: bool = False,
    wandb_project: str = 'eeg-bci',
    wandb_entity: Optional[str] = None,
    wandb_group: Optional[str] = None,
) -> TrainingResult:
    """
    Fine-tune a pretrained model on a single subject and return TrainingResult.

    Wraps finetune_subject() and converts the result dict to TrainingResult.
    """
    result_dict = finetune_subject(
        pretrained_path=pretrained_path,
        subject_id=subject_id,
        freeze_strategy=freeze_strategy,
        run_tag=run_tag,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience,
        paradigm=paradigm,
        task=task,
        seed=seed,
        data_root=data_root,
        cache_only=cache_only,
        cache_index_path=cache_index_path,
        no_wandb=no_wandb,
        upload_model=upload_model,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_group=wandb_group,
    )

    return TrainingResult(
        subject_id=subject_id,
        task_type=task,
        model_type=model_type,
        best_val_acc=result_dict.get('val_acc', 0.0),
        test_acc=result_dict['test_acc'],
        test_acc_majority=result_dict['test_acc'],  # finetune already uses majority voting
        epochs_trained=result_dict.get('best_epoch', 0),
        training_time=result_dict.get('training_time', 0.0),
    )


def run_transfer_model(
    model_type: str,
    pretrained_path: str,
    subject_ids: List[str],
    freeze_strategy: str,
    task: str,
    paradigm: str,
    data_root: str,
    output_dir: str,
    run_tag: Optional[str] = None,
    force_retrain: bool = False,
    epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    batch_size: Optional[int] = None,
    patience: Optional[int] = None,
    seed: int = 42,
    transfer_config: Optional[Dict] = None,
    # Cache-only mode
    cache_only: bool = False,
    cache_index_path: str = ".cache_index.json",
    # WandB
    no_wandb: bool = True,
    upload_model: bool = False,
    wandb_project: str = 'eeg-bci',
    wandb_entity: Optional[str] = None,
    wandb_group: Optional[str] = None,
) -> Tuple[List[TrainingResult], Dict]:
    """
    Fine-tune a pretrained model on all specified subjects with cache support.

    Follows the same cache-per-subject pattern as run_single_model().
    """
    log_train.info(f"Transfer Learning: {model_type.upper()} | freeze={freeze_strategy}")
    log_train.info(f"Pretrained: {pretrained_path}")

    # Load existing cache
    if force_retrain:
        cache = {}
    else:
        cache, _ = load_cache(output_dir, paradigm, task, run_tag, cache_type=CacheType.TRANSFER)

    if model_type not in cache:
        cache[model_type] = {}

    # Determine subjects to train
    cached_subjects = set(cache[model_type].keys())
    requested_subjects = set(subject_ids)
    subjects_to_train = requested_subjects - cached_subjects if not force_retrain else requested_subjects

    if cached_subjects & requested_subjects:
        already = len(cached_subjects & requested_subjects)
        to_train = len(subjects_to_train)
        if subjects_to_train:
            log_train.info(f"{already} cached, {to_train} to train ({', '.join(sorted(subjects_to_train))})")
        else:
            log_train.info(f"All {already} subjects cached (no training needed)")

    results: List[TrainingResult] = []
    total_subjects = len(subject_ids)

    for idx, subject_id in enumerate(subject_ids, 1):
        progress = f"[{idx}/{total_subjects}]"

        # Check cache
        if subject_id in cache[model_type] and not force_retrain:
            log_train.info(f"{progress} {subject_id}: cached")
            cached_result = dict_to_result(cache[model_type][subject_id])
            results.append(cached_result)
            print_subject_result(subject_id, model_type, cached_result)
            continue

        # Fine-tune
        log_train.info(f"{progress} {subject_id}: fine-tuning {model_type}...")

        try:
            set_seed(seed)

            result = finetune_and_get_result(
                subject_id=subject_id,
                model_type=model_type,
                pretrained_path=pretrained_path,
                freeze_strategy=freeze_strategy,
                task=task,
                paradigm=paradigm,
                data_root=data_root,
                run_tag=run_tag,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                patience=patience,
                seed=seed,
                cache_only=cache_only,
                cache_index_path=cache_index_path,
                no_wandb=no_wandb,
                upload_model=upload_model,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
                wandb_group=wandb_group,
            )

            results.append(result)

            # Save to cache immediately
            cache[model_type][subject_id] = result_to_dict(result)
            save_cache(
                output_dir, paradigm, task, cache, run_tag,
                cache_type=CacheType.TRANSFER,
                extra_metadata={'type': 'transfer-comparison', 'transfer_config': transfer_config or {}},
            )

            print_subject_result(subject_id, model_type, result)

        except Exception as e:
            log_train.error(f"{progress} {subject_id}: FAILED - {e}")
            traceback.print_exc()
            continue

    stats = compute_model_statistics(results)

    if results:
        log_train.info(
            f"{model_type.upper()} transfer done: {stats['mean']:.1%}+/-{stats['std']:.1%} "
            f"(n={stats['n_subjects']}, best={stats['max']:.1%})"
        )

    return results, stats


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Transfer learning comparison: fine-tune cross-subject pretrained models on individual subjects',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Auto-discover best pretrained models and fine-tune
  uv run python scripts/run_transfer_comparison.py

  # Specify freeze strategy
  uv run python scripts/run_transfer_comparison.py --freeze-strategy partial

  # Manual pretrained model paths
  uv run python scripts/run_transfer_comparison.py \\
      --pretrained-eegnet checkpoints/cross_subject/.../best.pt \\
      --pretrained-cbramod checkpoints/cross_subject/.../best.pt

  # Resume a previous run
  uv run python scripts/run_transfer_comparison.py --resume
'''
    )

    # Data arguments
    parser.add_argument(
        '--data-root', type=str, default='data',
        help='Path to data directory (default: data)'
    )
    parser.add_argument(
        '--subjects', nargs='+', default=None,
        help='Specific subjects to run (default: all available)'
    )
    parser.add_argument(
        '--models', nargs='+', default=['eegnet', 'cbramod'],
        choices=['eegnet', 'cbramod'],
        help='Models to fine-tune (default: both)'
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

    # Transfer learning arguments
    parser.add_argument(
        '--freeze-strategy', type=str, default='backbone',
        choices=['none', 'backbone', 'partial'],
        help='Freeze strategy for fine-tuning (default: backbone)'
    )
    parser.add_argument(
        '--pretrained-eegnet', type=str, default=None,
        help='Manual path to pretrained EEGNet checkpoint (.pt)'
    )
    parser.add_argument(
        '--pretrained-cbramod', type=str, default=None,
        help='Manual path to pretrained CBraMod checkpoint (.pt)'
    )
    parser.add_argument(
        '--finetune-epochs', type=int, default=None,
        help='Number of fine-tuning epochs (default: strategy/model-specific)'
    )
    parser.add_argument(
        '--finetune-lr', type=float, default=None,
        help='Fine-tuning learning rate (default: strategy-specific)'
    )
    parser.add_argument(
        '--finetune-batch-size', type=int, default=None,
        help='Fine-tuning batch size (default: model-specific)'
    )
    parser.add_argument(
        '--finetune-patience', type=int, default=None,
        help='Early stopping patience (default: 5)'
    )

    # Cache/resume arguments
    parser.add_argument(
        '--resume', nargs='?', const='', default=None,
        metavar='TAG',
        help='Resume a previous run. Without TAG: resume most recent. '
             'With TAG: resume run matching the datetime substring (e.g., "20260205")'
    )
    parser.add_argument(
        '--force-retrain', action='store_true',
        help='Force retraining, ignore cache'
    )
    parser.add_argument(
        '--results-dir', type=str, default='results',
        help='Directory to save results and plots (default: results)'
    )

    # Output control
    parser.add_argument(
        '--no-plot', action='store_true',
        help='Suppress plot generation'
    )
    parser.add_argument(
        '--no-cross-subject-baseline', action='store_true',
        help='Do not include cross-subject baseline in the plot'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )

    add_wandb_args(parser)

    # Cache index arguments
    parser.add_argument(
        '--cache-only', action='store_true',
        help='Load data exclusively from cache index (no filesystem scan)'
    )
    parser.add_argument(
        '--cache-index-path', type=str, default='.cache_index.json',
        help='Path to cache index file (default: .cache_index.json)'
    )

    args = parser.parse_args()

    # Start timer
    start_time = time.time()

    # Check GPU
    check_cuda_available(required=True)
    device = get_device()
    log_main.info(f"Device: {device}")

    # Set seed
    set_seed(args.seed)
    log_main.info(f"Seed: {args.seed}")

    # Handle --resume vs new run
    if args.resume is not None:
        if args.resume == '':
            found = find_cache_by_tag(args.results_dir, args.paradigm, args.task, cache_type=CacheType.TRANSFER)
            if found:
                _, run_tag = found
                log_main.info(f"Resuming most recent transfer run: {run_tag or '(untagged)'}")
            else:
                log_main.error("No previous transfer run found to resume")
                sys.exit(1)
        else:
            found = find_cache_by_tag(args.results_dir, args.paradigm, args.task, args.resume, cache_type=CacheType.TRANSFER)
            if found:
                _, run_tag = found
                log_main.info(f"Resuming transfer run matching '{args.resume}': {run_tag}")
            else:
                log_main.error(f"No transfer run found matching '{args.resume}'")
                sys.exit(1)
    else:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M")
        log_main.info(f"Starting new transfer comparison run: {run_tag}")

    paradigm_desc = PARADIGM_CONFIG[args.paradigm]['description']
    log_main.info(f"Paradigm: {paradigm_desc} | Task: {args.task} | Freeze: {args.freeze_strategy}")

    # Discover subjects
    if args.subjects:
        subjects = args.subjects
    else:
        subjects = discover_subjects(
            args.data_root, args.paradigm, args.task,
            cache_only=args.cache_only,
            cache_index_path=args.cache_index_path,
        )

    if not subjects:
        log_main.error(f"No subjects found in {args.data_root}")
        sys.exit(1)

    log_main.info(f"Subjects: {subjects} ({len(subjects)} total)")

    # ======================================================================
    # Discover pretrained checkpoints
    # ======================================================================
    pretrained_paths = {}
    manual_overrides = {
        'eegnet': args.pretrained_eegnet,
        'cbramod': args.pretrained_cbramod,
    }

    for model_type in args.models:
        # Manual override takes priority
        if manual_overrides.get(model_type):
            path = manual_overrides[model_type]
            if not Path(path).exists():
                log_main.error(f"Pretrained {model_type} not found: {path}")
                sys.exit(1)
            pretrained_paths[model_type] = path
            log_main.info(f"{model_type.upper()} pretrained (manual): {path}")
        else:
            # Auto-discover best checkpoint
            path = find_best_checkpoint_path(
                model_type=model_type,
                paradigm=args.paradigm,
                task=args.task,
                subjects=subjects,
                results_dir=args.results_dir,
            )
            if path:
                pretrained_paths[model_type] = path
                log_main.info(f"{model_type.upper()} pretrained (auto): {path}")
            else:
                log_main.warning(
                    f"No pretrained {model_type} checkpoint found for "
                    f"{args.paradigm}/{args.task}. Skipping this model. "
                    f"Run 'scripts/run_cross_subject_comparison.py' first to create one."
                )

    if not pretrained_paths:
        log_main.error(
            "No pretrained checkpoints found for any requested model. "
            "Run cross-subject training first:\n"
            "  uv run python scripts/run_cross_subject_comparison.py"
        )
        sys.exit(1)

    # Build transfer config metadata
    transfer_config = {
        'freeze_strategy': args.freeze_strategy,
        'finetune_epochs': args.finetune_epochs,
        'finetune_lr': args.finetune_lr,
        'finetune_batch_size': args.finetune_batch_size,
        'pretrained_paths': {k: str(v) for k, v in pretrained_paths.items()},
    }

    # ======================================================================
    # Fine-tune each model
    # ======================================================================
    results = {}
    all_stats = {}

    for model_type in args.models:
        if model_type not in pretrained_paths:
            continue

        log_main.info(f"{'='*50} {model_type.upper()} TRANSFER {'='*50}")

        model_results, model_stats = run_transfer_model(
            model_type=model_type,
            pretrained_path=pretrained_paths[model_type],
            subject_ids=subjects,
            freeze_strategy=args.freeze_strategy,
            task=args.task,
            paradigm=args.paradigm,
            data_root=args.data_root,
            output_dir=args.results_dir,
            run_tag=run_tag,
            force_retrain=args.force_retrain,
            epochs=args.finetune_epochs,
            learning_rate=args.finetune_lr,
            batch_size=args.finetune_batch_size,
            patience=args.finetune_patience,
            seed=args.seed,
            transfer_config=transfer_config,
            cache_only=args.cache_only,
            cache_index_path=args.cache_index_path,
            no_wandb=args.no_wandb,
            upload_model=args.upload_model,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_group=f"transfer_{model_type}_{args.freeze_strategy}_{run_tag}",
        )

        results[model_type] = model_results
        all_stats[model_type] = model_stats

    # ======================================================================
    # Statistical comparison
    # ======================================================================
    comparison = None
    if 'eegnet' in results and 'cbramod' in results:
        if len(results['eegnet']) >= 2 and len(results['cbramod']) >= 2:
            try:
                comparison = compare_models(results['eegnet'], results['cbramod'])
            except ValueError as e:
                log_main.warning(f"Cannot compare: {e}")

    # Print report
    print_comparison_report(results, comparison, args.task, args.paradigm, run_tag)

    # ======================================================================
    # Save final cache with summary and comparison
    # ======================================================================
    # Build cache results dict
    cache_results = {}
    for model_type, model_results in results.items():
        cache_results[model_type] = {
            r.subject_id: result_to_dict(r) for r in model_results
        }

    # Compute summary
    summary = {}
    for model_type, stats in all_stats.items():
        summary[model_type] = stats

    comparison_dict = None
    if comparison:
        from dataclasses import asdict
        comparison_dict = asdict(comparison)

    n_subjects = len(set(r.subject_id for model_results in results.values() for r in model_results))

    save_cache(
        output_dir=args.results_dir,
        paradigm=args.paradigm,
        task=args.task,
        results=cache_results,
        run_tag=run_tag,
        summary=summary,
        comparison=comparison_dict,
        n_subjects=n_subjects,
        is_complete=True,
        cache_type=CacheType.TRANSFER,
        extra_metadata={'type': 'transfer-comparison', 'transfer_config': transfer_config or {}},
    )

    # ======================================================================
    # Generate visualization
    # ======================================================================
    if not args.no_plot and results:
        subjects_set = set(subjects)

        # Retrieve within-subject baseline results for plotting
        within_subject_results = {}
        for model_type in ['eegnet', 'cbramod']:
            ws_results = find_best_within_subject_for_model(
                output_dir=args.results_dir,
                paradigm=args.paradigm,
                task=args.task,
                model_type=model_type,
                subjects_set=subjects_set,
            )
            if ws_results:
                within_subject_results[model_type] = ws_results
                mean_acc = sum(r.test_acc_majority for r in ws_results) / len(ws_results)
                log_io.info(
                    f"Within-subject baseline for {model_type}: "
                    f"mean={mean_acc:.1%}"
                )

        # Retrieve cross-subject baseline results for plotting
        cross_subject_results = {}

        if not args.no_cross_subject_baseline:
            for model_type in ['eegnet', 'cbramod']:
                cross_result = find_compatible_cross_subject_results(
                    output_dir=args.results_dir,
                    paradigm=args.paradigm,
                    task=args.task,
                    subjects=subjects,
                    model_type=model_type,
                )
                if cross_result:
                    cross_subject_results[model_type] = cross_result
                    log_io.info(
                        f"Cross-subject baseline for {model_type}: "
                        f"mean={cross_result['mean_test_acc']:.1%}"
                    )

        # Build data sources
        data_sources = build_transfer_data_sources(
            transfer_results=results,
            cross_subject_results=cross_subject_results,
            subjects=subjects,
            task=args.task,
            within_subject_results=within_subject_results,
        )

        if data_sources:
            plot_filename = generate_result_filename(
                'transfer_combined', args.paradigm, args.task, 'png', run_tag
            )
            plot_path = Path(args.results_dir) / plot_filename

            generate_combined_plot(
                data_sources=data_sources,
                output_path=str(plot_path),
                task_type=args.task,
                paradigm=args.paradigm,
            )
            log_io.info(f"Transfer comparison plot saved: {plot_path}")
        else:
            log_io.warning("No data sources available for plotting")

    # ======================================================================
    # Total time
    # ======================================================================
    total_time = time.time() - start_time
    if total_time >= 3600:
        log_main.info(f"Total time: {total_time/3600:.1f}h")
    elif total_time >= 60:
        log_main.info(f"Total time: {total_time/60:.1f}m")
    else:
        log_main.info(f"Total time: {total_time:.1f}s")

    return 0


if __name__ == '__main__':
    sys.exit(main())
