#!/usr/bin/env python
"""
Cross-Subject Model Comparison Script for EEG-BCI Project.

This script trains EEGNet and CBraMod using cross-subject paradigm
(all subjects combined in one model) and compares their performance.

Features:
- Trains both models on combined multi-subject data
- Performs statistical comparison between models
- Generates comparison visualizations with historical data
- Supports within-subject results as baseline comparison
- Resume support: skip completed models, resume training from checkpoint

Usage:
    # Run on Motor Imagery (default paradigm)
    uv run python scripts/run_cross_subject_comparison.py

    # Run on Motor Execution
    uv run python scripts/run_cross_subject_comparison.py --paradigm movement

    # Run on specific subjects
    uv run python scripts/run_cross_subject_comparison.py --subjects S01 S02 S03 S04 S05

    # Run only EEGNet
    uv run python scripts/run_cross_subject_comparison.py --models eegnet

    # Resume most recent run (skip completed models)
    uv run python scripts/run_cross_subject_comparison.py --resume

    # Resume specific run by tag
    uv run python scripts/run_cross_subject_comparison.py --resume 20260301

    # Force retrain all models
    uv run python scripts/run_cross_subject_comparison.py --force-retrain

    # Suppress plot generation
    uv run python scripts/run_cross_subject_comparison.py --no-plot

    # Disable within-subject historical comparison
    uv run python scripts/run_cross_subject_comparison.py --no-within-subject-historical

    # Use Muon optimizer for CBraMod (via YAML config)
    uv run python scripts/run_cross_subject_comparison.py --config configs/cbramod_muon.yaml

    # Re-generate plots for a finished run (no training, no DB writes)
    uv run python scripts/experiments/run_cross_subject_comparison.py --replot 20260321_0934
"""

# Force matplotlib non-interactive backend BEFORE any transitive matplotlib import.
# See run_transfer_comparison.py for the full Tcl_AsyncDelete failure mode.
import os
os.environ.setdefault('MPLBACKEND', 'Agg')

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

from src.config.constants import CacheType
from src.results import (
    PlotDataSource,
    compare_models,
    compute_model_statistics,
    print_comparison_report,
    generate_result_filename,
    cross_subject_result_to_training_results,
)
from src.results.cache import load_cache, save_cache
from src.visualization import generate_combined_plot
from src.visualization.comparison import plot_unified_comparison
from src.training.train_cross_subject import train_cross_subject

from src.cli.experiment_utils import (
    discover_subjects,
    add_wandb_args,
    add_common_args,
    add_cache_resume_args,
    add_channel_args,
    add_training_config_args,
    add_replot_arg,
    load_replot_context,
    resolve_run_tag,
    init_db_run,
    finalize_db_run,
    build_config_overrides,
)


setup_logging('cross_subject_comparison')
logger = logging.getLogger(__name__)

log_main = SectionLogger(logger, 'main')
log_stats = SectionLogger(logger, 'stats')
log_io = SectionLogger(logger, 'io')


def _generate_plots(
    results_by_model,
    raw_results,
    task,
    paradigm,
    subjects,
    models,
    run_tag,
    results_dir,
    n_channels,
    channel_config,
    db,
    db_run_id,
    no_within_subject_historical=False,
    no_cross_subject_historical=False,
    historical_selection='baseline',
):
    """
    生成 cross-subject 对比图（提取自 main 以支持 replot）.

    Args:
        results_by_model: {model_type: List[TrainingResult]} — 用于 data_sources
        raw_results: {model_type: training_result_dict} — 用于 unified subtask 数据
        task, paradigm, subjects, models, run_tag, results_dir: 实验参数
        n_channels, channel_config: 通道配置
        db: ExperimentDB 实例 (read-only for historical queries)
        db_run_id: DB run ID (None for replot → skips DB writes)
        no_within_subject_historical: 是否禁用 within-subject 历史叠加
        no_cross_subject_historical: 是否禁用 cross-subject 历史叠加
        historical_selection: 'baseline' | 'best'
    """
    from pathlib import Path

    # Unified task: use plot_unified_comparison with per-subtask breakdown
    if task == 'unified':
        unified_plot_data = {}
        for model_type, model_results in raw_results.items():
            sr = model_results.get('subtask_results')
            if sr and 'per_subject' in sr:
                import numpy as np
                per_subject = {}
                for sid, subj_data in sr['per_subject'].items():
                    per_subject[sid] = {}
                    for st in ('binary', 'ternary', 'quaternary', 'mean_accuracy'):
                        if st in subj_data:
                            per_subject[sid][st] = subj_data[st]

                subtask_results = {}
                for st in ('binary', 'ternary', 'quaternary'):
                    if st in sr:
                        subtask_results[st] = sr[st]
                subtask_results['mean_accuracy'] = sr.get('mean_accuracy', 0)

                unified_plot_data[model_type] = {
                    'subtask_results': subtask_results,
                    'per_subject': per_subject,
                }
            else:
                log_io.info(f"{model_type}: no per-subject subtask data for unified plot")

        if unified_plot_data:
            plot_filename = generate_result_filename(
                'unified_comparison', paradigm, task, 'png', run_tag, is_cross_subject=True
            )
            plot_path = Path(results_dir) / plot_filename
            fig = plot_unified_comparison(
                results=unified_plot_data,
                save_path=str(plot_path),
                title=f"Unified Model — Cross-Subject Comparison ({paradigm.capitalize()}, {len(subjects)} Subjects)",
            )
            if fig:
                import matplotlib.pyplot as plt
                plt.close(fig)
                log_io.info(f"Unified comparison plot saved: {plot_path}")
        else:
            log_io.info("No subtask data for unified plot")
        return

    # Non-unified: standard combined plot
    data_sources = []
    subjects_set = set(subjects)
    channel_config_filter = channel_config if n_channels != FULL_N_CHANNELS else None
    within_query = (
        db.find_baseline_within_subject_results
        if historical_selection == 'baseline'
        else db.find_best_within_subject_results
    )
    cross_query = (
        db.find_baseline_cross_subject_results
        if historical_selection == 'baseline'
        else db.find_best_cross_subject_results
    )
    within_label_suffix = 'Within Baseline' if historical_selection == 'baseline' else 'Within Best'
    cross_label_suffix = 'Cross Baseline' if historical_selection == 'baseline' else 'Cross Best'
    within_ref_type = 'within_subject_baseline' if historical_selection == 'baseline' else 'historical_comparison'
    cross_ref_type = 'cross_subject_baseline' if historical_selection == 'baseline' else 'historical_comparison'

    # 1 & 2: Historical within-subject references (baseline by default, best via flag)
    if not no_within_subject_historical:
        for model_type in ['eegnet', 'cbramod']:
            ws_result = within_query(
                paradigm=paradigm,
                task=task,
                model_type=model_type,
                n_channels=n_channels,
                channel_config=channel_config_filter,
                subjects=subjects_set,
                return_run_id=True,
            )
            if ws_result is not None:
                hist_results, ws_run_id = ws_result
                if db_run_id and ws_run_id:
                    db.add_baseline_ref(db_run_id, ws_run_id, within_ref_type, model_type)
                data_sources.append(PlotDataSource(
                    model_type=model_type,
                    results=hist_results,
                    is_current_run=False,
                    label=f'{model_type.upper()} ({within_label_suffix})',
                    hatch='///',
                ))

    # 3 & 4: Current cross-subject results
    for model_type in ['eegnet', 'cbramod']:
        training_results = results_by_model.get(model_type, [])
        if training_results:
            data_sources.append(PlotDataSource(
                model_type=model_type,
                results=training_results,
                is_current_run=True,
                label=f'{model_type.upper()} (Cross)',
            ))

    # 5: (Optional) Historical cross-subject reference
    if not no_cross_subject_historical:
        search_model = 'cbramod' if 'cbramod' in models else models[0]
        cs_result = cross_query(
            paradigm=paradigm,
            task=task,
            model_type=search_model,
            n_channels=n_channels,
            channel_config=channel_config_filter,
            subjects=subjects_set,
            exclude_run_id=db_run_id,
            return_run_id=True,
        )
        if cs_result is not None:
            hist_cross, cs_run_id = cs_result
            if db_run_id and cs_run_id:
                db.add_baseline_ref(db_run_id, cs_run_id, cross_ref_type, search_model)
            data_sources.append(PlotDataSource(
                model_type=search_model,
                results=hist_cross,
                is_current_run=False,
                label=f'{search_model.upper()} ({cross_label_suffix})',
                hatch='...',
            ))

    if data_sources:
        plot_filename = generate_result_filename(
            'combined', paradigm, task, 'png', run_tag, is_cross_subject=True
        )
        plot_path = Path(results_dir) / plot_filename

        generate_combined_plot(
            data_sources=data_sources,
            output_path=str(plot_path),
            task_type=task,
            paradigm=paradigm,
        )
        log_io.info(f"Comparison plot saved: {plot_path}")
    else:
        log_io.warning("No data sources available for plotting")


def main():
    parser = argparse.ArgumentParser(
        description='Run cross-subject model comparison on all subjects',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run on Motor Imagery (default)
  uv run python scripts/run_cross_subject_comparison.py

  # Run on Motor Execution
  uv run python scripts/run_cross_subject_comparison.py --paradigm movement

  # Run on specific subjects
  uv run python scripts/run_cross_subject_comparison.py --subjects S01 S02 S03 S04 S05

  # Run only CBraMod
  uv run python scripts/run_cross_subject_comparison.py --models cbramod

  # Disable historical comparison
  uv run python scripts/run_cross_subject_comparison.py --no-within-subject-historical

  # Use best-accuracy historical runs instead of designated baselines
  uv run python scripts/run_cross_subject_comparison.py --historical-selection best
'''
    )

    # Shared common args (--data-root, --subjects, --paradigm, --task, --seed, --output-dir, --no-plot)
    add_common_args(parser)
    # Override --output-dir default for cross-subject (checkpoints dir, not results)
    for action in parser._actions:
        if hasattr(action, 'dest') and action.dest == 'output_dir':
            action.default = 'checkpoints/cross_subject'
            action.help = 'Directory to save pretrained models (default: checkpoints/cross_subject)'
            break

    # Cross-subject-specific: models and training params
    parser.add_argument(
        '--models', nargs='+', default=['eegnet', 'cbramod'],
        choices=['eegnet', 'cbramod'],
        help='Models to train (default: both)'
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Number of training epochs (default: model-specific)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Batch size (default: model-specific)'
    )

    # Results dir (separate from --output-dir which holds checkpoints)
    parser.add_argument(
        '--results-dir', type=str, default='results',
        help='Directory to save results and plots (default: results)'
    )

    # Historical data arguments
    parser.add_argument(
        '--no-within-subject-historical', action='store_true',
        help='Disable within-subject historical data in comparison plot'
    )
    parser.add_argument(
        '--no-cross-subject-historical', action='store_true',
        help='Disable cross-subject historical data (previous runs) in comparison plot'
    )
    parser.add_argument(
        '--historical-selection',
        choices=['baseline', 'best'],
        default='baseline',
        help='How to choose historical reference runs for plots (default: baseline)'
    )

    # Shared cache/resume args
    add_cache_resume_args(parser)

    # Shared channel selection args
    add_channel_args(parser)

    # Shared training config args (--config, --scheduler, --classifier-type, --no-pretrained)
    add_training_config_args(parser)

    add_wandb_args(parser)

    # Replot argument
    add_replot_arg(parser)

    # Verbosity arguments
    parser.add_argument(
        '--verbose', '-v', type=int, default=2,
        choices=[0, 1, 2],
        help='Verbosity level: 0=silent, 1=minimal, 2=full (default: 2)'
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Equivalent to --verbose 0'
    )
    parser.add_argument(
        '--baseline', action='store_true',
        help='Mark this run as a designated baseline in ExperimentDB'
    )
    parser.add_argument(
        '--further-pretrained-cbramod', type=str, default=None,
        help='Path to further-pretrained CBraMod backbone weights (.pth) to load before cross-subject training '
             '(only affects cbramod; EEGNet ignores this option)'
    )

    # P0.3 negative-control: within-subject trial-level label permutation
    parser.add_argument(
        '--shuffle-labels', action='store_true',
        help='[Negative control] Within-subject trial-level label permutation. '
             'Train and per-subject test labels are both shuffled. Used to verify '
             'cross-subject pipeline has no input->label leakage (expected: chance acc).'
    )
    parser.add_argument(
        '--shuffle-seed', type=int, default=42,
        help='RNG seed for label shuffle (only used with --shuffle-labels, default: 42)'
    )

    args = parser.parse_args()

    # Negative-control runs must never be marked as baselines
    if args.shuffle_labels and args.baseline:
        parser.error('--shuffle-labels and --baseline are mutually exclusive '
                     '(label-shuffle runs are negative controls, not baselines)')

    # Auto-redirect results to results/{n}_channel/{config}/ when using reduced channel mode
    if args.channels != FULL_N_CHANNELS and args.results_dir == 'results':
        args.results_dir = f'results/{args.channels}_channel/{args.channel_config}'

    # --replot: re-generate plots for a finished run, skip training entirely
    if args.replot:
        ctx = load_replot_context(
            args.replot, 'cross_subject',
            results_dir_override=args.results_dir if args.results_dir != 'results' else None,
        )

        # For unified tasks: load subtask data from cache
        raw_results = {}
        if ctx['task'] == 'unified':
            cache, _ = load_cache(
                ctx['results_dir'], ctx['paradigm'], ctx['task'],
                run_tag=ctx['run_tag'], cache_type=CacheType.CROSS_SUBJECT,
            )
            for mt in ctx['models']:
                if mt in cache:
                    raw_results[mt] = cache[mt]

        _generate_plots(
            results_by_model=ctx['results_by_model'],
            raw_results=raw_results,
            task=ctx['task'],
            paradigm=ctx['paradigm'],
            subjects=ctx['subjects'],
            models=ctx['models'],
            run_tag=ctx['run_tag'],
            results_dir=ctx['results_dir'],
            n_channels=ctx['n_channels'],
            channel_config=ctx['channel_config'],
            db=ctx['db'],
            db_run_id=None,
            no_within_subject_historical=args.no_within_subject_historical,
            no_cross_subject_historical=args.no_cross_subject_historical,
            historical_selection=args.historical_selection,
        )
        ctx['db'].close()
        log_main.info(f"Replot complete for {args.replot}")
        return 0

    # Start timer
    start_time = time.time()

    # Handle verbosity
    verbose = 0 if args.quiet else args.verbose

    # Check GPU
    check_cuda_available(required=True)
    if not check_vram_utilization():
        sys.exit(0)
    device = get_device()
    log_main.info(f"Device: {device}")

    # Set seed
    set_seed(args.seed)
    log_main.info(f"Seed: {args.seed}")

    # Generate or resume run tag
    run_tag = resolve_run_tag(args, args.paradigm, args.task, args.results_dir, cache_type=CacheType.CROSS_SUBJECT)

    # Tag negative-control runs so they are easy to filter out of baseline searches.
    # Skip when resuming an existing run that already has the suffix.
    if args.shuffle_labels and '_labelshuffle_' not in run_tag:
        run_tag = f"{run_tag}_labelshuffle_seed{args.shuffle_seed}"
        log_main.info(f"[Negative Control] Run tag tagged: {run_tag}")

    paradigm_desc = PARADIGM_CONFIG[args.paradigm]['description']
    log_main.info(f"Paradigm: {paradigm_desc}")

    # Initialize ExperimentDB (dual-write)
    db, db_run_id = init_db_run(run_tag, 'cross_subject', args.paradigm, args.task, args)

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

    log_main.info(f"Subjects: {subjects} | Models: {args.models} | Task: {args.task}")

    # Build config_overrides: YAML base → CLI overrides
    config_overrides = build_config_overrides(args)

    if args.further_pretrained_cbramod:
        if 'cbramod' not in args.models:
            parser.error('--further-pretrained-cbramod requires cbramod in --models')
        if not Path(args.further_pretrained_cbramod).exists():
            parser.error(f'--further-pretrained-cbramod path not found: {args.further_pretrained_cbramod}')
        config_overrides = config_overrides or {}
        config_overrides.setdefault('model', {})['pretrained_path'] = args.further_pretrained_cbramod
        log_main.info(f"CBraMod backbone init: {args.further_pretrained_cbramod}")

    # Load existing cache for resume
    cache = {}
    if not args.force_retrain:
        cache, _ = load_cache(
            args.results_dir, args.paradigm, args.task,
            run_tag=run_tag,
            cache_type=CacheType.CROSS_SUBJECT,
        )

    # Train each model (skip cached models on resume)
    results = {}
    for model_type in args.models:
        # Check if model already has cached results
        if model_type in cache and not args.force_retrain:
            log_main.info(f"{'='*50} {model_type.upper()} (CACHED) {'='*50}")
            cached = cache[model_type]
            results[model_type] = {
                'per_subject_test_acc': cached.get('per_subject_test_acc', {}),
                'mean_test_acc': cached.get('mean_test_acc', 0),
                'std_test_acc': cached.get('std_test_acc', 0),
                'val_acc': cached.get('val_acc', 0),
                'val_majority_acc': cached.get('val_majority_acc', 0),
                'best_epoch': cached.get('best_epoch', 0),
                'training_time': cached.get('training_time', 0),
                'model_path': cached.get('model_path', ''),
                'n_channels': cached.get('n_channels'),
                'history': None,
                'run_tag': run_tag,
            }
            log_main.info(f"Skipping {model_type} — cached mean_test_acc: "
                          f"{cached.get('mean_test_acc', 'N/A'):.4f}")
            continue

        log_main.info(f"{'='*50} {model_type.upper()} {'='*50}")

        # Determine if we should attempt epoch-level resume
        should_resume_epoch = (args.resume is not None and not args.force_retrain)

        model_results = train_cross_subject(
            subjects=subjects,
            model_type=model_type,
            task=args.task,
            paradigm=args.paradigm,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_dir=args.output_dir,
            data_root=args.data_root,
            device=device,
            seed=args.seed,
            run_tag=run_tag,
            config_overrides=config_overrides,
            cache_only=args.cache_only,
            wandb_enabled=not args.no_wandb,
            upload_model=args.upload_model,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            verbose=verbose,
            resume_checkpoint=should_resume_epoch,
            shuffle_labels=args.shuffle_labels,
            shuffle_seed=args.shuffle_seed,
        )

        results[model_type] = model_results

        # Progressive cache: save immediately after each model completes
        cache[model_type] = {
            'per_subject_test_acc': model_results.get('per_subject_test_acc', {}),
            'mean_test_acc': model_results.get('mean_test_acc', 0),
            'std_test_acc': model_results.get('std_test_acc', 0),
            'val_acc': model_results.get('val_acc', 0),
            'val_majority_acc': model_results.get('val_majority_acc', 0),
            'best_epoch': model_results.get('best_epoch', 0),
            'training_time': model_results.get('training_time', 0),
            'model_path': model_results.get('model_path', ''),
            'n_channels': model_results.get('n_channels'),
        }
        save_cache(
            output_dir=args.results_dir,
            paradigm=args.paradigm,
            task=args.task,
            results=cache,
            run_tag=run_tag,
            cache_type=CacheType.CROSS_SUBJECT,
        )
        log_io.info(f"{model_type.upper()} cached to cross_subject_cache")

        # Dual-write to DB
        if db_run_id:
            try:
                training_results = cross_subject_result_to_training_results(
                    model_results, model_type, args.task
                )
                db.save_subject_results_batch(db_run_id, training_results)
                db_stats = compute_model_statistics(training_results)
                db.save_summary(db_run_id, model_type, db_stats)
            except Exception as e:
                log_io.warning(f"DB write failed for {model_type}: {e}")

    # Convert to TrainingResult lists for comparison
    results_as_training = {}
    for model_type, model_results in results.items():
        results_as_training[model_type] = cross_subject_result_to_training_results(
            model_results, model_type, args.task
        )

    # Statistical comparison
    comparison = None
    if 'eegnet' in results_as_training and 'cbramod' in results_as_training:
        if len(results_as_training['eegnet']) >= 2 and len(results_as_training['cbramod']) >= 2:
            try:
                comparison = compare_models(
                    results_as_training['eegnet'],
                    results_as_training['cbramod']
                )
            except ValueError as e:
                log_stats.warning(f"Cannot compare: {e}")

    # Final cache save (mark as complete)
    save_cache(
        output_dir=args.results_dir,
        paradigm=args.paradigm,
        task=args.task,
        results=cache,
        run_tag=run_tag,
        cache_type=CacheType.CROSS_SUBJECT,
        is_complete=True,
        n_subjects=len(subjects),
    )

    # Print comparison report
    print_comparison_report(results_as_training, comparison, args.task, args.paradigm, run_tag)

    # Generate visualization
    if not args.no_plot:
        _generate_plots(
            results_by_model=results_as_training,
            raw_results=results,
            task=args.task,
            paradigm=args.paradigm,
            subjects=subjects,
            models=args.models,
            run_tag=run_tag,
            results_dir=args.results_dir,
            n_channels=args.channels,
            channel_config=args.channel_config,
            db=db,
            db_run_id=db_run_id,
            no_within_subject_historical=args.no_within_subject_historical,
            no_cross_subject_historical=args.no_cross_subject_historical,
            historical_selection=args.historical_selection,
        )

    # Save comparison to DB, mark complete, and close
    finalize_db_run(db, db_run_id, comparison, len(subjects))

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
