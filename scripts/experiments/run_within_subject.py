#!/usr/bin/env python
"""
Within-Subject Training Script for EEG-BCI Project.

This script performs within-subject training and evaluation for a single model
(EEGNet or CBraMod) on all available subjects, generating statistics, cache,
and visualizations.

Features:
- Trains one model type across all subjects
- Incremental caching: resumes from where it left off
- Generates 2-panel visualization (bar chart + box plot)
- Can be called programmatically by run_within_subject_comparison.py

Usage:
    # Train EEGNet on all subjects
    uv run python scripts/run_within_subject.py --model eegnet

    # Train CBraMod with specific subjects
    uv run python scripts/run_within_subject.py --model cbramod --subjects S01 S02 S03

    # Start a new experiment (preserves old results)
    uv run python scripts/run_within_subject.py --model eegnet --new-run

    # Skip training, just load existing results
    uv run python scripts/run_within_subject.py --model eegnet --skip-training

    # Suppress plot generation
    uv run python scripts/run_within_subject.py --model eegnet --no-plot
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

from src.config.constants import PARADIGM_CONFIG, CacheType
from src.utils.device import set_seed, check_cuda_available, get_device
from src.utils.logging import SectionLogger, setup_logging

# Import from src modules
from src.results import (
    TrainingResult,
    ExperimentDB,
    load_cache,
    save_cache,
    prepare_combined_plot_data,
    generate_result_filename,
    result_to_dict,
    dict_to_result,
    compute_model_statistics,
    print_model_summary,
)
from src.visualization import generate_combined_plot, generate_single_model_plot
from src.training.train_within_subject import (
    SCHEDULER_PRESETS,
    visualize_lr_schedule,
    get_default_config,
)
from src.training.prefetch import SubjectPrefetcher

from src.cli.experiment_utils import (
    discover_subjects,
    print_subject_result,
    train_and_get_result,
    add_wandb_args,
    add_common_args,
    add_cache_resume_args,
    add_channel_args,
    add_training_config_args,
    add_transfer_args,
    resolve_output_dir,
    resolve_run_tag,
    build_config_overrides,
)


setup_logging('within_subject')
logger = logging.getLogger(__name__)
log_main = SectionLogger(logger, 'main')
log_train = SectionLogger(logger, 'train')
log_io = SectionLogger(logger, 'io')


# ============================================================================
# Core Training Function
# ============================================================================

def run_within_subject(
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
    config_overrides: Optional[Dict] = None,
    verbose_first_only: bool = True,
    db: Optional[ExperimentDB] = None,
    db_run_id: Optional[str] = None,
    # Transfer learning (optional)
    pretrained_path: Optional[str] = None,
    freeze_strategy: Optional[str] = None,
    # Prefetch
    use_prefetch: bool = True,
    # Cache type
    cache_type = None,  # CacheType enum, None defaults to within-subject
) -> Tuple[List[TrainingResult], Dict]:
    """
    Run within-subject training for one model type on all specified subjects.

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
        wandb_project: WandB project name (default: eeg-bci)
        wandb_entity: WandB entity (team/username)
        cache_only: If True, load data exclusively from cache index
        config_overrides: Config overrides dict (from YAML + CLI merge). Passed to train_and_get_result.
        verbose_first_only: If True, only show full verbose output for the first trained subject.
            Subsequent subjects show minimal output (subject header + training table + final eval).
            Default: True.
        db: Optional ExperimentDB instance for SQLite logging (dual-write with JSON cache).
        db_run_id: Optional run ID in the ExperimentDB. If db is provided but db_run_id is None,
            results are not saved to DB.

    Returns:
        Tuple of (results_list, statistics_dict)
    """
    device = get_device()
    paradigm_config = PARADIGM_CONFIG[paradigm]
    log_train.info(f"Model: {model_type.upper()} | Paradigm: {paradigm_config['description']}")

    # Default cache_type
    if cache_type is None:
        cache_type = CacheType.WITHIN_SUBJECT

    # Resolve effective classifier_type for cache metadata
    effective_config = get_default_config(model_type, task)
    if config_overrides and 'model' in config_overrides:
        effective_config['model'].update(config_overrides['model'])
    cache_extra_metadata = {
        'classifier_type': effective_config['model'].get('classifier_type'),
    } if model_type == 'cbramod' else None

    # Load existing cache and metadata (including wandb_groups)
    wandb_group = None
    cache_wandb_groups = {}

    if force_retrain:
        cache = {}
        log_train.info("Force retrain - ignoring cache")
    elif run_tag:
        cache, metadata = load_cache(output_dir, paradigm, task, run_tag, cache_type=cache_type)
        cache_wandb_groups = metadata.get('wandb_groups', {})
        if cache:
            log_train.info(f"Resuming '{run_tag}'")
            # Restore wandb_group from cache if available
            wandb_group = cache_wandb_groups.get(model_type)
        else:
            log_train.info(f"New run '{run_tag}'")
    else:
        cache, metadata = load_cache(output_dir, paradigm, task, find_latest=True, cache_type=cache_type)
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

    # Set up subject prefetcher (background data loading for next subject)
    prefetcher = None
    if use_prefetch and subjects_to_train:
        try:
            prefetcher = SubjectPrefetcher(
                model_type=model_type,
                task=task,
                paradigm=paradigm,
                data_root=Path(data_root),
                elc_path=Path(data_root) / 'biosemi128.ELC',
                cache_only=cache_only,
                config_overrides=config_overrides,
            )
        except Exception as e:
            log_train.warning(f"Prefetch init failed ({e}), continuing without prefetch")

    # Seed prefetch for first non-cached subject
    if prefetcher is not None:
        for sid in subject_ids:
            if sid not in cache.get(model_type, {}) or force_retrain:
                prefetcher.start_prefetch(sid)
                break

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

        # Retrieve prefetched data (if available) and start prefetch for next subject
        precomputed_data = None
        if prefetcher is not None:
            precomputed_data = prefetcher.get_prefetched(subject_id)
            if precomputed_data is not None:
                log_train.info(f"{progress} {subject_id}: using prefetched data")

            # Start prefetch for next non-cached subject
            for future_id in subject_ids[idx:]:  # idx is 1-based, so subject_ids[idx:] = remaining
                if future_id not in cache[model_type] or force_retrain:
                    prefetcher.start_prefetch(future_id)
                    break

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
                run_tag=run_tag,
                no_wandb=no_wandb,
                upload_model=upload_model,
                wandb_group=wandb_group,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
                cache_only=cache_only,
                config_overrides=config_overrides,
                verbose=verbose,
                pretrained_path=pretrained_path,
                freeze_strategy=freeze_strategy,
                precomputed_data=precomputed_data,
            )

            # Mark first subject as trained (for subsequent verbose control)
            first_subject_trained = True

            results.append(result)

            # Save to cache immediately (including wandb_groups metadata)
            cache[model_type][subject_id] = result_to_dict(result)
            save_cache(output_dir, paradigm, task, cache, run_tag,
                       wandb_groups=cache_wandb_groups,
                       extra_metadata=cache_extra_metadata,
                       cache_type=cache_type)

            # Dual-write: save to SQLite DB if available
            if db and db_run_id:
                try:
                    db.save_subject_result(db_run_id, result)
                except Exception as db_err:
                    log_train.warning(f"DB write failed for {subject_id}: {db_err}")

            print_subject_result(subject_id, model_type, result)

        except Exception as e:
            log_train.error(f"{progress} {subject_id}: FAILED - {e}")
            traceback.print_exc()
            # Clean up any active wandb run left by failed training
            try:
                import wandb
                if wandb.run is not None:
                    wandb.finish(exit_code=1, quiet=True)
            except Exception:
                pass
            continue

    # Clean up prefetcher
    if prefetcher is not None:
        prefetcher.shutdown()

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
        description='Within-subject training for EEG-BCI (EEGNet or CBraMod)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Train EEGNet on all subjects
  uv run python scripts/run_within_subject.py --model eegnet

  # Train CBraMod on specific subjects
  uv run python scripts/run_within_subject.py --model cbramod --subjects S01 S02

  # Resume the most recent run
  uv run python scripts/run_within_subject.py --model eegnet --resume

  # Resume a specific run by datetime substring
  uv run python scripts/run_within_subject.py --model eegnet --resume 20260205

  # Load existing results only (no training)
  uv run python scripts/run_within_subject.py --model eegnet --skip-training
'''
    )

    # Script-specific args
    parser.add_argument(
        '--model', type=str, required=True,
        choices=['eegnet', 'cbramod'],
        help='Model type to train'
    )
    parser.add_argument(
        '--pretrained-weights', type=str, default=None,
        help='Path to pretrained weights for CBraMod backbone (default: auto-detect)'
    )
    parser.add_argument(
        '--no-historical', action='store_true',
        help='禁用历史数据检索，仅生成单模型图（不检索另一个模型的历史结果）'
    )

    # Shared args
    add_common_args(parser)
    add_cache_resume_args(parser)
    add_channel_args(parser)
    add_training_config_args(parser)
    add_transfer_args(parser)
    add_wandb_args(parser)

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

    # Auto-redirect output to results/{n}_channel/{config}/ when using reduced channel mode
    args.output_dir = resolve_output_dir(args)

    # Determine cache_type based on --pretrained flag
    cache_type = CacheType.TRANSFER if args.pretrained else None

    # Handle --resume vs new run (default)
    output_dir = args.output_dir
    run_tag = resolve_run_tag(args, args.paradigm, args.task, output_dir,
                              cache_type=cache_type or CacheType.WITHIN_SUBJECT)

    paradigm_desc = PARADIGM_CONFIG[args.paradigm]['description']
    log_main.info(f"Model: {args.model.upper()} | Paradigm: {paradigm_desc} | Task: {args.task}")

    # Build merged config_overrides: YAML → CLI scheduler/channel/classifier override
    config_overrides = build_config_overrides(args) or {}
    if args.pretrained_weights:
        if args.model != 'cbramod':
            parser.error('--pretrained-weights is only supported for CBraMod')
        config_overrides.setdefault('model', {})['pretrained_path'] = args.pretrained_weights
    config_overrides = config_overrides or None

    # Show LR schedule visualization for CBraMod (non-blocking, once at start)
    if args.model == 'cbramod' and not args.skip_training:
        # Determine scheduler type from merged overrides
        if config_overrides and 'training' in config_overrides:
            scheduler_type = config_overrides['training'].get('scheduler', 'cosine_annealing_warmup_decay')
        else:
            scheduler_type = 'cosine_annealing_warmup_decay'
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
        # Load from cache
        cache, _ = load_cache(args.output_dir, args.paradigm, args.task, find_latest=True, cache_type=cache_type)
        if args.model not in cache:
            log_main.error(f"No cached results found for {args.model}")
            sys.exit(1)
        results = [dict_to_result(d) for d in cache[args.model].values()]
        stats = compute_model_statistics(results)
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
                cache_only=args.cache_only,
            )

        if not subjects:
            log_main.error(f"No subjects found in {args.data_root}")
            sys.exit(1)

        log_main.info(f"Subjects: {subjects}")

        if args.resume is not None:
            log_main.info("Resuming from cache (--force-retrain to overwrite)")

        # Run training
        results, stats = run_within_subject(
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
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            cache_only=args.cache_only,
            config_overrides=config_overrides,
            pretrained_path=args.pretrained,
            freeze_strategy=args.freeze_strategy,
            cache_type=cache_type,
        )

    # Print summary
    if results:
        print_model_summary(args.model, stats, results)

        # Save summary to cache (final save with statistics)
        cache_final, cache_meta = load_cache(args.output_dir, args.paradigm, args.task, run_tag, cache_type=cache_type)
        save_cache(args.output_dir, args.paradigm, args.task, cache_final, run_tag,
                   wandb_groups=cache_meta.get('wandb_groups', {}),
                   summary={args.model: stats},
                   is_complete=True,
                   cache_type=cache_type)

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
                # No historical data, use within-subject single model plot
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
