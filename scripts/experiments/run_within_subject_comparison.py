#!/usr/bin/env python
"""
Within-Subject Model Comparison Script for EEG-BCI Project.

This script trains EEGNet and CBraMod using within-subject paradigm
(each subject trained independently) and compares their performance.

Data Split (follows paper protocol):
- Training: Offline + Session 1 (Base + Finetune) + Session 2 Base
- Validation: Last 20% of training data (temporal split)
- Test: Session 2 Finetune (completely held out)

Features:
- Uses run_single_model.py for individual model training
- Performs statistical comparison between models
- Generates comparison visualizations

Usage:
    # Run on Motor Imagery (default paradigm, plots generated automatically)
    uv run python scripts/run_within_subject_comparison.py

    # Run on Motor Execution
    uv run python scripts/run_within_subject_comparison.py --paradigm movement

    # Resume the most recent run
    uv run python scripts/run_within_subject_comparison.py --resume

    # Resume a specific run by datetime substring
    uv run python scripts/run_within_subject_comparison.py --resume 20260205

    # Force retrain (overwrites existing cache)
    uv run python scripts/run_within_subject_comparison.py --force-retrain

    # Run on specific subjects
    uv run python scripts/run_within_subject_comparison.py --subjects S01 S02 S03

    # Run only EEGNet
    uv run python scripts/run_within_subject_comparison.py --models eegnet

    # Load existing results only (no training)
    uv run python scripts/run_within_subject_comparison.py --skip-training
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path (scripts/experiments/ -> scripts/ -> project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.constants import FULL_N_CHANNELS, PARADIGM_CONFIG
from src.utils.device import set_seed, check_cuda_available, get_device, check_vram_utilization
from src.utils.logging import SectionLogger, setup_logging

# Import from src modules
from src.results import (
    PlotDataSource,
    compare_models,
    compute_model_statistics,
    print_comparison_report,
    load_cache,
    save_cache,
    generate_result_filename,
    load_comparison_results,
)
from src.visualization import generate_combined_plot, generate_comparison_plot
from src.visualization.comparison import plot_unified_comparison
from src.training.train_within_subject import (
    SCHEDULER_PRESETS,
    visualize_lr_schedule,
    get_default_config,
)

# Import from scripts directory
SCRIPTS_DIR = Path(__file__).parent.parent
EXPERIMENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(EXPERIMENTS_DIR))

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
from run_single_model import run_single_model


setup_logging('compare')
logger = logging.getLogger(__name__)

log_main = SectionLogger(logger, 'main')
log_stats = SectionLogger(logger, 'stats')
log_io = SectionLogger(logger, 'io')


# ============================================================================
# Helper Functions
# ============================================================================

def compute_summary(results):
    """计算每个模型的统计摘要。

    Args:
        results: Dict[str, List[TrainingResult]] - 模型类型到结果列表的映射

    Returns:
        Dict[str, Dict[str, float]] - 模型类型到统计摘要的映射
    """
    import numpy as np

    summary = {}
    for model_type, model_results in results.items():
        if not model_results:
            continue

        test_accs = [r.test_acc_majority for r in model_results]
        summary[model_type] = {
            'mean': float(np.mean(test_accs)),
            'std': float(np.std(test_accs)),
            'median': float(np.median(test_accs)),
            'min': float(np.min(test_accs)),
            'max': float(np.max(test_accs)),
        }

    return summary


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run within-subject model comparison on all subjects',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run on Motor Imagery (default, plots generated automatically)
  uv run python scripts/run_within_subject_comparison.py

  # Run on Motor Execution
  uv run python scripts/run_within_subject_comparison.py --paradigm movement

  # Resume the most recent run
  uv run python scripts/run_within_subject_comparison.py --resume

  # Resume a specific run by datetime substring
  uv run python scripts/run_within_subject_comparison.py --resume 20260205

  # Force retrain (overwrites existing cache for this paradigm/task)
  uv run python scripts/run_within_subject_comparison.py --force-retrain

  # Suppress plot generation
  uv run python scripts/run_within_subject_comparison.py --no-plot
'''
    )

    add_common_args(parser)
    add_cache_resume_args(parser)
    add_channel_args(parser)
    add_training_config_args(parser)
    add_wandb_args(parser)

    # Script-specific args
    parser.add_argument('--models', nargs='+', default=['eegnet', 'cbramod'],
                        choices=['eegnet', 'cbramod'], help='Models to train (default: both)')
    parser.add_argument('--results-file', type=str, default=None,
                        help='Path to existing results file (used with --skip-training)')
    parser.add_argument('--baseline', action='store_true',
                        help='Mark this run as a designated baseline in ExperimentDB')

    args = parser.parse_args()

    # Auto-redirect output to results/{n}_channel/{config}/ when using reduced channel mode
    args.output_dir = resolve_output_dir(args)

    # Start timer
    start_time = time.time()

    # Check GPU
    check_cuda_available(required=True)
    if not check_vram_utilization():
        sys.exit(0)
    device = get_device()
    log_main.info(f"Device: {device}")

    # Set seed
    set_seed(args.seed)
    log_main.info(f"Seed: {args.seed}")

    # Handle --resume vs new run (default)
    output_dir = args.output_dir
    run_tag = resolve_run_tag(args, args.paradigm, args.task, output_dir)

    paradigm_desc = PARADIGM_CONFIG[args.paradigm]['description']
    log_main.info(f"Paradigm: {paradigm_desc}")

    # Initialize ExperimentDB (dual-write alongside JSON cache)
    db, db_run_id = init_db_run(run_tag, 'within_subject', args.paradigm, args.task, args)

    # Build merged config_overrides: YAML → CLI scheduler override
    config_overrides = build_config_overrides(args)

    # Show LR schedule visualization for CBraMod (non-blocking, once at start)
    if 'cbramod' in args.models and not args.skip_training:
        if config_overrides and 'training' in config_overrides:
            scheduler_type = config_overrides['training'].get('scheduler', 'cosine_annealing_warmup_decay')
        else:
            scheduler_type = args.scheduler or 'cosine_annealing_warmup_decay'
        if scheduler_type in SCHEDULER_PRESETS:
            scheduler_config = SCHEDULER_PRESETS[scheduler_type]
            default_config = get_default_config('cbramod', args.task)
            base_lr = default_config['training'].get('backbone_lr', 1e-4)

            lr_schedule_path = Path(args.output_dir) / f"lr_schedule_{scheduler_type}.png"
            visualize_lr_schedule(
                scheduler_config=scheduler_config,
                base_lr=base_lr,
                output_path=lr_schedule_path,
                show=True,
            )

    if args.skip_training:
        if args.results_file is None:
            results_dir = Path(args.output_dir)
            pattern = f'*comparison_{args.paradigm}_{args.task}*.json'
            result_files = sorted(results_dir.glob(pattern), reverse=True)
            if not result_files:
                result_files = sorted(results_dir.glob(f'comparison_{args.task}_*.json'), reverse=True)
            if not result_files:
                log_main.error("No results files found. Run training first.")
                sys.exit(1)
            args.results_file = str(result_files[0])
            log_io.info(f"Using: {args.results_file}")

        results = load_comparison_results(args.results_file)
        log_io.info(f"Loaded: {args.results_file}")
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
            log_main.error(f"No subjects in {args.data_root}")
            sys.exit(1)

        log_main.info(f"Subjects: {subjects} | Models: {args.models} | Task: {args.task}")

        if args.resume is not None:
            log_main.info("Resuming from cache (--force-retrain to overwrite)")

        # Run training for each model using run_single_model
        results = {}
        for model_type in args.models:
            log_main.info(f"{'='*50} {model_type.upper()} {'='*50}")

            model_results, stats = run_single_model(
                model_type=model_type,
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
                db=db,
                db_run_id=db_run_id,
            )
            results[model_type] = model_results

            # Save model summary to DB
            if db_run_id and model_results:
                try:
                    db_stats = compute_model_statistics(model_results)
                    db.save_summary(db_run_id, model_type, db_stats)
                except Exception as e:
                    log_stats.warning(f"DB summary save failed: {e}")

    # Compare models
    comparison = None
    if 'eegnet' in results and 'cbramod' in results:
        if len(results['eegnet']) >= 2 and len(results['cbramod']) >= 2:
            try:
                comparison = compare_models(results['eegnet'], results['cbramod'])
            except ValueError as e:
                log_stats.warning(f"Cannot compare: {e}")

    print_comparison_report(results, comparison, args.task, args.paradigm, run_tag)

    # Load existing cache to add summary and comparison data
    cache, cache_metadata = load_cache(
        args.output_dir, args.paradigm, args.task, run_tag, find_latest=(run_tag is None)
    )

    # Compute summary statistics
    summary = compute_summary(results)

    # Convert comparison to dict if present
    from dataclasses import asdict
    comparison_dict = asdict(comparison) if comparison else None

    # Save updated cache with summary and comparison
    # Preserve existing timestamp if available
    existing_timestamp = cache_metadata.get('metadata', {}).get('timestamp')

    # Resolve effective classifier_type for cache metadata
    cbramod_config = get_default_config('cbramod', args.task)
    if config_overrides and 'model' in config_overrides:
        cbramod_config['model'].update(config_overrides['model'])
    cache_extra = {
        'classifier_type': cbramod_config['model'].get('classifier_type'),
    }

    output_path = save_cache(
        output_dir=args.output_dir,
        paradigm=args.paradigm,
        task=args.task,
        results=cache,  # Use existing cache
        run_tag=run_tag,
        wandb_groups=cache_metadata.get('wandb_groups', {}),
        summary=summary,
        comparison=comparison_dict,
        n_subjects=len(set(
            r.subject_id for model_results in results.values()
            for r in model_results
        )),
        is_complete=True,
        existing_timestamp=existing_timestamp,
        extra_metadata=cache_extra,
    )

    # Generate plots by default (unless --no-plot is specified)
    if not args.no_plot:
        # Unified task: use plot_unified_comparison with per-subtask breakdown
        if args.task == 'unified':
            unified_plot_data = {}
            for model_type, model_results in results.items():
                per_subject = {}
                for r in model_results:
                    if r.subtask_results:
                        per_subject[r.subject_id] = {}
                        for st in ('binary', 'ternary', 'quaternary'):
                            if st in r.subtask_results and isinstance(r.subtask_results[st], dict):
                                per_subject[r.subject_id][st] = r.subtask_results[st]

                # Aggregate subtask means
                import numpy as np
                subtask_results = {}
                for st in ('binary', 'ternary', 'quaternary'):
                    accs = [
                        per_subject[s][st]['accuracy']
                        for s in per_subject
                        if st in per_subject[s] and per_subject[s][st].get('n_trials', 0) > 0
                    ]
                    subtask_results[st] = {
                        'accuracy': float(np.mean(accs)) if accs else 0,
                        'std': float(np.std(accs)) if accs else 0,
                        'n_subjects': len(accs),
                    }
                all_means = [r.test_acc_majority for r in model_results]
                subtask_results['mean_accuracy'] = float(np.mean(all_means)) if all_means else 0

                unified_plot_data[model_type] = {
                    'subtask_results': subtask_results,
                    'per_subject': per_subject,
                }

            if unified_plot_data:
                plot_filename = generate_result_filename('unified_comparison', args.paradigm, args.task, 'png', run_tag)
                plot_path = Path(args.output_dir) / plot_filename
                n_subj = len(set(r.subject_id for mrs in results.values() for r in mrs))
                fig = plot_unified_comparison(
                    results=unified_plot_data,
                    save_path=str(plot_path),
                    title=f"Unified Model — Within-Subject Comparison ({args.paradigm.capitalize()}, {n_subj} Subjects)",
                )
                if fig:
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                    log_io.info(f"Unified comparison plot saved: {plot_path}")
            else:
                log_io.info("No subtask data available for unified plot")
        else:
            # Non-unified: standard combined/comparison plot
            # Query DB for historical comparison data
            subjects_set = set(
                r.subject_id for model_results in results.values() for r in model_results
            )

            channel_config_filter = args.channel_config if args.channels != FULL_N_CHANNELS else None
            hist_result = db.find_historical_comparison(
                paradigm=args.paradigm,
                task=args.task,
                n_channels=args.channels,
                channel_config=channel_config_filter,
                subjects=subjects_set if subjects_set else None,
                exclude_run_id=db_run_id,
                return_run_id=True,
            )
            historical = None
            hist_run_id = None
            if hist_result is not None:
                historical, hist_run_id = hist_result
                if db_run_id and hist_run_id:
                    db.add_baseline_ref(db_run_id, hist_run_id, 'historical_comparison')

            data_sources = []
            if historical:
                # Add historical data sources (hatched bars)
                for model_type in ['eegnet', 'cbramod']:
                    hist_results = historical.get(model_type, [])
                    if hist_results:
                        data_sources.append(PlotDataSource(
                            model_type=model_type,
                            results=hist_results,
                            is_current_run=False,
                            label=f'{model_type.upper()} (hist)',
                        ))

            # Add current run data sources
            for model_type in ['eegnet', 'cbramod']:
                current = results.get(model_type, [])
                if current:
                    filtered = [r for r in current if r.subject_id in subjects_set]
                    if filtered:
                        data_sources.append(PlotDataSource(
                            model_type=model_type,
                            results=filtered,
                            is_current_run=True,
                            label=model_type.upper(),
                        ))

            if len(data_sources) >= 2:
                log_io.info("Generating combined plot with historical comparison")
                plot_filename = generate_result_filename('combined', args.paradigm, args.task, 'png', run_tag)
                plot_path = Path(args.output_dir) / plot_filename
                generate_combined_plot(
                    data_sources=data_sources,
                    output_path=str(plot_path),
                    task_type=args.task,
                    paradigm=args.paradigm,
                )
            elif comparison:
                # No historical data but have complete two-model comparison
                plot_filename = generate_result_filename('comparison', args.paradigm, args.task, 'png', run_tag)
                plot_path = Path(args.output_dir) / plot_filename
                generate_comparison_plot(results, comparison, str(plot_path), task_type=args.task)
            else:
                log_io.info("No historical data found and insufficient models for comparison plot")

    # Mark DB run complete
    n_subjects = len(set(
        r.subject_id for model_results in results.values()
        for r in model_results
    ))
    finalize_db_run(db, db_run_id, comparison, n_subjects)

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
