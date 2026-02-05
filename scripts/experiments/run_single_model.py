#!/usr/bin/env python
"""
Single Model Training Script for EEG-BCI Project.

This script trains and evaluates a single model (EEGNet or CBraMod) on all
available subjects, generating statistics, cache, and visualizations.

Features:
- Trains one model type across all subjects
- Incremental caching: resumes from where it left off
- Generates 2-panel visualization (bar chart + box plot)
- Can be called programmatically by run_full_comparison.py

Usage:
    # Train EEGNet on all subjects
    uv run python scripts/run_single_model.py --model eegnet

    # Train CBraMod with specific subjects
    uv run python scripts/run_single_model.py --model cbramod --subjects S01 S02 S03

    # Start a new experiment (preserves old results)
    uv run python scripts/run_single_model.py --model eegnet --new-run

    # Skip training, just load existing results
    uv run python scripts/run_single_model.py --model eegnet --skip-training

    # Suppress plot generation
    uv run python scripts/run_single_model.py --model eegnet --no-plot
"""

import argparse
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

# Import from src modules
from src.results import (
    TrainingResult,
    load_cache,
    save_cache,
    prepare_combined_plot_data,
    generate_result_filename,
    result_to_dict,
    dict_to_result,
    compute_model_statistics,
    print_model_summary,
    save_single_model_results,
    load_single_model_results,
)
from src.visualization import generate_combined_plot, generate_single_model_plot
from src.training.train_within_subject import (
    SCHEDULER_PRESETS,
    visualize_lr_schedule,
    get_default_config,
)

# Import from scripts directory
SCRIPTS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from _training_utils import (
    discover_subjects,
    print_subject_result,
    train_and_get_result,
)


setup_logging('single_model')
logger = logging.getLogger(__name__)
log_main = SectionLogger(logger, 'main')
log_train = SectionLogger(logger, 'train')
log_io = SectionLogger(logger, 'io')


# ============================================================================
# Core Training Function
# ============================================================================

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
    wandb_interactive: bool = False,
    wandb_session_metadata: Optional[Dict[str, Optional[str]]] = None,
    cache_only: bool = False,
    cache_index_path: str = ".cache_index.json",
    scheduler: Optional[str] = None,
    verbose_first_only: bool = True,
) -> Tuple[List[TrainingResult], Dict]:
    """
    Train a single model on all specified subjects.

    Args:
        model_type: 'eegnet' or 'cbramod'
        data_root: Path to data directory
        subject_ids: List of subject IDs to train
        task: 'binary', 'ternary', or 'quaternary'
        paradigm: 'imagery' or 'movement'
        output_dir: Directory to save results
        force_retrain: If True, ignore cache and retrain all
        run_tag: Optional datetime tag for new runs
        no_wandb: Disable wandb logging
        upload_model: Upload model artifacts (.pt) to WandB (default: False)
        wandb_interactive: Prompt for run details interactively (only prompts once for batch training)
        wandb_session_metadata: Pre-collected WandB metadata (from run_full_comparison.py).
            When provided, skips interactive prompting.
        cache_only: If True, load data exclusively from cache index
        cache_index_path: Path to cache index file for cache_only mode
        scheduler: Learning rate scheduler type (e.g., 'wsd', 'cosine_annealing_warmup_decay')
        verbose_first_only: If True, only show full verbose output for the first trained subject.
            Subsequent subjects show minimal output (subject header + training table + final eval).
            Default: True.

    Returns:
        Tuple of (results_list, statistics_dict)
    """
    device = get_device()
    paradigm_config = PARADIGM_CONFIG[paradigm]
    log_train.info(f"Model: {model_type.upper()} | Paradigm: {paradigm_config['description']}")

    # Load existing cache and metadata (including wandb_groups)
    wandb_group = None
    cache_wandb_groups = {}

    if force_retrain:
        cache = {}
        log_train.info("Force retrain - ignoring cache")
    elif run_tag:
        cache, metadata = load_cache(output_dir, paradigm, task, run_tag)
        cache_wandb_groups = metadata.get('wandb_groups', {})
        if cache:
            log_train.info(f"Resuming '{run_tag}'")
            # Restore wandb_group from cache if available
            wandb_group = cache_wandb_groups.get(model_type)
        else:
            log_train.info(f"New run '{run_tag}'")
    else:
        cache, metadata = load_cache(output_dir, paradigm, task, find_latest=True)
        cache_wandb_groups = metadata.get('wandb_groups', {})
        # Restore wandb_group from latest cache if available
        wandb_group = cache_wandb_groups.get(model_type)

    # Generate new wandb_group only if not restored from cache
    if not wandb_group:
        if run_tag:
            wandb_group = f"{model_type}_{paradigm}_{task}_{run_tag}"
        else:
            wandb_group = f"{model_type}_{paradigm}_{task}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    # Save wandb_group to cache metadata for future runs
    cache_wandb_groups[model_type] = wandb_group

    # Determine which subjects need training
    cached_subjects = set(cache.get(model_type, {}).keys()) if cache else set()
    requested_subjects = set(subject_ids)
    subjects_to_train = requested_subjects - cached_subjects if not force_retrain else requested_subjects

    # Log cache summary
    if cache and not force_retrain:
        already_cached = cached_subjects & requested_subjects
        if already_cached and subjects_to_train:
            log_train.info(f"{len(already_cached)} cached, {len(subjects_to_train)} to train ({', '.join(sorted(subjects_to_train))})")
        elif already_cached and not subjects_to_train:
            log_train.info(f"All {len(already_cached)} subjects cached (no training needed)")
        elif subjects_to_train:
            log_train.info(f"{len(subjects_to_train)} to train ({', '.join(sorted(subjects_to_train))})")

    # Determine shared WandB metadata for batch training
    shared_wandb_metadata = None
    if wandb_session_metadata is not None:
        # Use pre-collected metadata from run_full_comparison.py
        shared_wandb_metadata = wandb_session_metadata
    else:
        # Single model run: check if we should prompt
        from scripts._wandb_setup import should_prompt_wandb, prompt_batch_session
        if should_prompt_wandb(
            wandb_enabled=not no_wandb,
            interactive=wandb_interactive,
            has_training=bool(subjects_to_train),
        ):
            shared_wandb_metadata = prompt_batch_session(
                models=[model_type],
                task=task,
                paradigm=paradigm,
                subjects_to_train=len(subjects_to_train),
            )
    # Don't prompt again for individual subjects
    wandb_interactive = False

    results: List[TrainingResult] = []

    if model_type not in cache:
        cache[model_type] = {}

    # Track whether we've trained the first subject (for verbose control)
    first_subject_trained = False

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

        # Train
        log_train.info(f"{progress} {subject_id}: training {model_type}...")

        # Determine verbose level: full (2) for first subject, minimal (1) for subsequent
        verbose = 2 if (not first_subject_trained or not verbose_first_only) else 1

        try:
            # Reset seed before each training for reproducibility
            set_seed(42)

            result = train_and_get_result(
                subject_id=subject_id,
                model_type=model_type,
                task=task,
                paradigm=paradigm,
                data_root=data_root,
                save_dir=output_dir,
                no_wandb=no_wandb,
                upload_model=upload_model,
                wandb_group=wandb_group,
                wandb_interactive=wandb_interactive,
                wandb_metadata=shared_wandb_metadata,
                cache_only=cache_only,
                cache_index_path=cache_index_path,
                scheduler=scheduler,
                verbose=verbose,
            )

            # Mark first subject as trained (for subsequent verbose control)
            first_subject_trained = True

            results.append(result)

            # Save to cache immediately (including wandb_groups metadata)
            cache[model_type][subject_id] = result_to_dict(result)
            save_cache(output_dir, paradigm, task, cache, run_tag, wandb_groups=cache_wandb_groups)

            print_subject_result(subject_id, model_type, result)

        except Exception as e:
            log_train.error(f"{progress} {subject_id}: FAILED - {e}")
            traceback.print_exc()
            continue

    # Compute statistics
    stats = compute_model_statistics(results)

    if results:
        log_train.info(f"{model_type.upper()} done: {stats['mean']:.1%}+/-{stats['std']:.1%} "
                      f"(n={stats['n_subjects']}, best={stats['max']:.1%})")

    return results, stats


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Single model training for EEG-BCI (EEGNet or CBraMod)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Train EEGNet on all subjects
  uv run python scripts/run_single_model.py --model eegnet

  # Train CBraMod on specific subjects
  uv run python scripts/run_single_model.py --model cbramod --subjects S01 S02

  # Start a NEW experiment (preserves old results)
  uv run python scripts/run_single_model.py --model eegnet --new-run

  # Load existing results only (no training)
  uv run python scripts/run_single_model.py --model eegnet --skip-training
'''
    )

    # Required
    parser.add_argument(
        '--model', type=str, required=True,
        choices=['eegnet', 'cbramod'],
        help='Model type to train'
    )

    # Data parameters
    parser.add_argument(
        '--data-root', type=str, default='data',
        help='Path to data directory (default: data)'
    )
    parser.add_argument(
        '--subjects', nargs='+', default=None,
        help='Specific subjects to run (default: all available)'
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

    # Cache/output parameters
    parser.add_argument(
        '--output-dir', type=str, default='results',
        help='Directory to save results (default: results)'
    )
    parser.add_argument(
        '--new-run', action='store_true',
        help='Start a NEW experiment run with datetime tag'
    )
    parser.add_argument(
        '--force-retrain', action='store_true',
        help='Force retraining, ignore cache'
    )
    parser.add_argument(
        '--skip-training', action='store_true',
        help='Skip training, load existing results'
    )
    parser.add_argument(
        '--results-file', type=str, default=None,
        help='Path to existing results file (used with --skip-training)'
    )

    # Output control
    parser.add_argument(
        '--no-plot', action='store_true',
        help='Suppress plot generation'
    )
    parser.add_argument(
        '--no-wandb', action='store_true',
        help='Disable wandb logging'
    )
    parser.add_argument(
        '--upload-model', action='store_true',
        help='Upload model artifacts (.pt files) to WandB (default: disabled to save bandwidth)'
    )
    parser.add_argument(
        '--no-wandb-interactive', action='store_true',
        help='Disable interactive prompts for WandB run details (prompts are enabled by default)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--scheduler', type=str, default=None,
        choices=['plateau', 'cosine', 'wsd', 'cosine_decay', 'cosine_annealing_warmup_decay'],
        help='Learning rate scheduler (default: model-specific)'
    )
    parser.add_argument(
        '--use-cache-index', action='store_true',
        help='从缓存索引发现被试，而非扫描 data/ 目录（适用于仅有缓存无原始数据的场景）'
    )
    parser.add_argument(
        '--cache-index-path', type=str, default='.cache_index.json',
        help='缓存索引文件路径（默认：.cache_index.json）'
    )

    # Historical comparison options
    parser.add_argument(
        '--no-historical', action='store_true',
        help='禁用历史数据检索，仅生成单模型图（不检索另一个模型的历史结果）'
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

    # Generate run tag if --new-run
    run_tag = None
    if args.new_run:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M")
        log_main.info(f"New run: {run_tag}")

    paradigm_desc = PARADIGM_CONFIG[args.paradigm]['description']
    log_main.info(f"Model: {args.model.upper()} | Paradigm: {paradigm_desc} | Task: {args.task}")

    # Show LR schedule visualization for CBraMod (non-blocking, once at start)
    if args.model == 'cbramod' and not args.skip_training:
        # Get scheduler config
        scheduler_type = args.scheduler or 'cosine_annealing_warmup_decay'
        if scheduler_type in SCHEDULER_PRESETS:
            scheduler_config = SCHEDULER_PRESETS[scheduler_type]
            default_config = get_default_config('cbramod', args.task)
            base_lr = default_config['training'].get('backbone_lr', 1e-4)

            # Save to results directory
            lr_schedule_path = Path(args.output_dir) / f"lr_schedule_{scheduler_type}.png"
            visualize_lr_schedule(
                scheduler_config=scheduler_config,
                base_lr=base_lr,
                output_path=lr_schedule_path,
                show=True,  # Non-blocking display
            )

    if args.skip_training:
        # Load existing results
        results, stats = load_single_model_results(
            model_type=args.model,
            output_dir=args.output_dir,
            paradigm=args.paradigm,
            task=args.task,
            results_file=args.results_file,
        )
        if not results:
            log_main.error(f"No cached results found for {args.model}")
            sys.exit(1)
        log_io.info(f"Loaded {len(results)} results from cache")
    else:
        # Discover subjects
        if args.subjects:
            subjects = args.subjects
        else:
            subjects = discover_subjects(
                args.data_root,
                args.paradigm,
                args.task,
                use_cache_index=args.use_cache_index,
                cache_index_path=args.cache_index_path
            )

        if not subjects:
            log_main.error(f"No subjects found in {args.data_root}")
            sys.exit(1)

        log_main.info(f"Subjects: {subjects}")

        if not args.force_retrain and not args.new_run:
            log_main.info("Using cache (--new-run for fresh, --force-retrain to overwrite)")

        # Run training
        results, stats = run_single_model(
            model_type=args.model,
            data_root=args.data_root,
            subject_ids=subjects,
            task=args.task,
            paradigm=args.paradigm,
            output_dir=args.output_dir,
            force_retrain=args.force_retrain,
            run_tag=run_tag,
            no_wandb=args.no_wandb,
            upload_model=args.upload_model,
            wandb_interactive=not args.no_wandb_interactive,
            cache_only=args.use_cache_index,
            cache_index_path=args.cache_index_path,
            scheduler=args.scheduler,
        )

    # Print summary
    if results:
        print_model_summary(args.model, stats, results)

        # Save results JSON
        save_single_model_results(
            model_type=args.model,
            results=results,
            statistics=stats,
            paradigm=args.paradigm,
            task=args.task,
            output_dir=args.output_dir,
            run_tag=run_tag,
        )

        # Generate plot
        if not args.no_plot:
            data_sources, hist_timestamp = None, None

            # Try to find compatible historical data (unless disabled)
            if not args.no_historical:
                current_results = {args.model: results}
                data_sources, hist_timestamp = prepare_combined_plot_data(
                    output_dir=args.output_dir,
                    paradigm=args.paradigm,
                    task=args.task,
                    current_results=current_results,
                    current_model=args.model,
                )

            if data_sources:
                log_io.info(f"Generating combined plot with historical comparison")
                plot_filename = generate_result_filename('combined', args.paradigm, args.task, 'png', run_tag)
                plot_path = Path(args.output_dir) / plot_filename
                generate_combined_plot(
                    data_sources=data_sources,
                    output_path=str(plot_path),
                    task_type=args.task,
                    paradigm=args.paradigm,
                    historical_timestamp=hist_timestamp,
                )
            else:
                # No historical data, use single model plot
                plot_filename = generate_result_filename(args.model, args.paradigm, args.task, 'png', run_tag)
                plot_path = Path(args.output_dir) / plot_filename
                generate_single_model_plot(
                    model_type=args.model,
                    results=results,
                    statistics=stats,
                    output_path=str(plot_path),
                    task_type=args.task,
                )

    # Log total time
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
