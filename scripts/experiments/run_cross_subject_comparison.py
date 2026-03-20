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
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path (scripts/experiments/ -> scripts/ -> project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.constants import FULL_N_CHANNELS, SUPPORTED_CHANNEL_COUNTS, PARADIGM_CONFIG
from src.utils.device import set_seed, check_cuda_available, get_device
from src.utils.logging import SectionLogger, setup_logging

from src.config.constants import CacheType
from src.results import (
    ExperimentDB,
    PlotDataSource,
    compare_models,
    compute_model_statistics,
    print_comparison_report,
    save_cross_subject_result,
    generate_result_filename,
    TrainingResult,
    cross_subject_result_to_training_results,
)
from src.results.cache import find_cache_by_tag, load_cache, save_cache
from src.visualization import generate_combined_plot
from src.visualization.comparison import plot_unified_comparison
from src.training.train_cross_subject import train_cross_subject
from src.config.training import SCHEDULER_PRESETS, load_yaml_config

SCRIPTS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from _training_utils import discover_subjects, add_wandb_args


setup_logging('cross_subject_comparison')
logger = logging.getLogger(__name__)

log_main = SectionLogger(logger, 'main')
log_stats = SectionLogger(logger, 'stats')
log_io = SectionLogger(logger, 'io')


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
        help='Models to train (default: both)'
    )
    parser.add_argument(
        '--paradigm', type=str, default='imagery',
        choices=['imagery', 'movement'],
        help='Experiment paradigm (default: imagery)'
    )
    parser.add_argument(
        '--task', type=str, default='binary',
        choices=['binary', 'ternary', 'quaternary', 'unified'],
        help='Classification task (default: binary)'
    )

    # Training arguments
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Number of training epochs (default: model-specific)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Batch size (default: model-specific)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--scheduler', type=str, default=None,
        choices=list(SCHEDULER_PRESETS.keys()),
        help='Learning rate scheduler (default: model-specific)'
    )
    parser.add_argument(
        '--config', type=str, default=None, metavar='YAML_PATH',
        help='YAML config file path (e.g., configs/cbramod_muon.yaml). '
             'Overrides model defaults; CLI args take priority over YAML.'
    )

    # Output arguments
    parser.add_argument(
        '--output-dir', type=str, default='checkpoints/cross_subject',
        help='Directory to save pretrained models (default: checkpoints/cross_subject)'
    )
    parser.add_argument(
        '--results-dir', type=str, default='results',
        help='Directory to save results and plots (default: results)'
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help='Suppress plot generation'
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

    # Cache arguments
    parser.add_argument(
        '--cache-only', action='store_true',
        help='Load data exclusively from cache index (no filesystem scan)'
    )
    parser.add_argument(
        '--cache-index-path', type=str, default='.cache_index.json',
        help='Path to cache index file (default: .cache_index.json)'
    )

    parser.add_argument(
        '--channels', type=int, default=FULL_N_CHANNELS,
        choices=SUPPORTED_CHANNEL_COUNTS,
        help=f'Number of EEG channels to use: {"/".join(str(c) for c in SUPPORTED_CHANNEL_COUNTS)} (default: {FULL_N_CHANNELS})'
    )
    parser.add_argument(
        '--channel-config', type=str, default='motor_cortex',
        help='Channel configuration name (default: motor_cortex). '
             '32ch: motor_cortex, commercial, fdr, csp, attention, band_power; '
             '61ch: standard_1010'
    )
    parser.add_argument(
        '--classifier-type', type=str, default=None,
        choices=['two_layer', 'three_layer', 'one_layer', 'attention_pool'],
        help='Override CBraMod classifier head type (default: use model config)'
    )

    # Resume arguments
    parser.add_argument(
        '--resume', nargs='?', const='', default=None,
        metavar='TAG',
        help='Resume a previous run. Without TAG: resume most recent. '
             'With TAG: resume run matching the datetime substring (e.g., "20260228")'
    )
    parser.add_argument(
        '--force-retrain', action='store_true',
        help='Force retraining all models, ignore cached results'
    )

    add_wandb_args(parser)

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

    args = parser.parse_args()

    # Auto-redirect results to results/{n}_channel/{config}/ when using reduced channel mode
    if args.channels != FULL_N_CHANNELS and args.results_dir == 'results':
        args.results_dir = f'results/{args.channels}_channel/{args.channel_config}'

    # Start timer
    start_time = time.time()

    # Handle verbosity
    verbose = 0 if args.quiet else args.verbose

    # Check GPU
    check_cuda_available(required=True)
    device = get_device()
    log_main.info(f"Device: {device}")

    # Set seed
    set_seed(args.seed)
    log_main.info(f"Seed: {args.seed}")

    # Generate or resume run tag
    if args.resume is not None:
        tag_hint = args.resume if args.resume != '' else None
        found = find_cache_by_tag(
            args.results_dir, args.paradigm, args.task,
            tag_substring=tag_hint,
            cache_type=CacheType.CROSS_SUBJECT,
        )
        if found:
            cache_path, run_tag = found
            log_main.info(f"Resuming cross-subject comparison run: {run_tag}")
            # Warn if current CLI parameters differ from cached run
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_meta = json.load(f).get('metadata', {})
                for key, cli_val in [('paradigm', args.paradigm), ('task', args.task)]:
                    cached_val = cached_meta.get(key)
                    if cached_val and cached_val != cli_val:
                        log_main.warning(
                            f"Parameter mismatch: cached {key}={cached_val}, "
                            f"current --{key}={cli_val}")
            except Exception:
                pass  # Non-critical: best-effort check
        else:
            log_main.error("No previous cross-subject run found to resume")
            sys.exit(1)
    else:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M")
        log_main.info(f"Starting new cross-subject comparison run: {run_tag}")

    paradigm_desc = PARADIGM_CONFIG[args.paradigm]['description']
    log_main.info(f"Paradigm: {paradigm_desc}")

    # Initialize ExperimentDB (dual-write)
    import shlex
    import sqlite3
    db = ExperimentDB()
    db_run_id = None
    try:
        db_run_id = db.create_run(
            run_tag=run_tag,
            experiment_type='cross_subject',
            paradigm=args.paradigm,
            task=args.task,
            n_channels=args.channels,
            channel_config=args.channel_config if args.channels != FULL_N_CHANNELS else None,
            command=" ".join(shlex.quote(a) for a in sys.argv),
        )
        log_main.info(f"DB run created: {db_run_id}")
    except sqlite3.IntegrityError:
        existing = db.find_run_by_tag(
            run_tag, args.paradigm, args.task, experiment_type='cross_subject',
        )
        if existing:
            db_run_id = existing['run_id']
            log_main.info(f"DB run resumed: {db_run_id}")
        else:
            log_main.warning(f"DB run creation failed: duplicate run_id but tag not found")
    except Exception as e:
        log_main.warning(f"DB run creation failed: {e}")

    # Discover subjects
    if args.subjects:
        subjects = args.subjects
    else:
        subjects = discover_subjects(
            args.data_root,
            args.paradigm,
            args.task,
            cache_only=args.cache_only,
            cache_index_path=args.cache_index_path
        )

    if not subjects:
        log_main.error(f"No subjects found in {args.data_root}")
        sys.exit(1)

    log_main.info(f"Subjects: {subjects} | Models: {args.models} | Task: {args.task}")

    # Build config_overrides: YAML base → CLI overrides
    config_overrides = load_yaml_config(args.config) if args.config else {}
    if args.scheduler:
        config_overrides.setdefault('training', {})['scheduler'] = args.scheduler
    if args.channels != FULL_N_CHANNELS:
        config_overrides.setdefault('data', {})['channels'] = args.channels
        config_overrides.setdefault('data', {})['channel_config'] = args.channel_config
    if args.classifier_type:
        config_overrides.setdefault('model', {})['classifier_type'] = args.classifier_type
    config_overrides = config_overrides or None

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
    channel_config_to_save = args.channel_config if args.channels != FULL_N_CHANNELS else None
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
            extra_metadata={'type': 'cross-subject-comparison'},
        )
        log_io.info(f"{model_type.upper()} cached to cross_subject_cache")

        # Also save individual model result file (backward compatible)
        results_path = save_cross_subject_result(
            result=model_results,
            model_type=model_type,
            paradigm=args.paradigm,
            task=args.task,
            output_dir=args.results_dir,
            run_tag=run_tag,
            channel_config=channel_config_to_save,
        )
        log_io.info(f"{model_type.upper()} results saved: {results_path}")

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

    # Save comparison to DB and mark complete
    if db_run_id:
        try:
            if comparison:
                db.save_comparison(db_run_id, comparison)
            db.update_n_subjects(db_run_id, len(subjects))
            db.mark_complete(db_run_id)
        except Exception as e:
            log_stats.warning(f"DB finalize failed: {e}")

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
        extra_metadata={'type': 'cross-subject-comparison'},
    )

    # Print comparison report
    print_comparison_report(results_as_training, comparison, args.task, args.paradigm, run_tag)

    # Generate visualization
    if not args.no_plot:
        # Unified task: use plot_unified_comparison with per-subtask breakdown
        if args.task == 'unified':
            unified_plot_data = {}
            for model_type, model_results in results.items():
                sr = model_results.get('subtask_results')
                if sr and 'per_subject' in sr:
                    # Full subtask data available from training
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
                    'unified_comparison', args.paradigm, args.task, 'png', run_tag, is_cross_subject=True
                )
                plot_path = Path(args.results_dir) / plot_filename
                fig = plot_unified_comparison(
                    results=unified_plot_data,
                    save_path=str(plot_path),
                    title=f"Unified Model — Cross-Subject Comparison ({args.paradigm.capitalize()}, {len(subjects)} Subjects)",
                )
                if fig:
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                    log_io.info(f"Unified comparison plot saved: {plot_path}")
            else:
                log_io.info("No subtask data for unified plot, falling back to standard plot")
        else:
            # Non-unified: standard combined plot
            data_sources = []
            subjects_set = set(subjects)
            channel_config_filter = args.channel_config if args.channels != FULL_N_CHANNELS else None

            # 1 & 2: Historical within-subject baselines (per-model, best accuracy)
            for model_type in ['eegnet', 'cbramod']:
                hist_results = db.find_best_within_subject_results(
                    paradigm=args.paradigm,
                    task=args.task,
                    model_type=model_type,
                    n_channels=args.channels,
                    channel_config=channel_config_filter,
                    subjects=subjects_set,
                )
                if hist_results:
                    data_sources.append(PlotDataSource(
                        model_type=model_type,
                        results=hist_results,
                        is_current_run=False,
                        label=f'{model_type.upper()} (Within)',
                        hatch='///',
                    ))

            # 3 & 4: Current cross-subject results
            for model_type in ['eegnet', 'cbramod']:
                if model_type not in results:
                    continue
                training_results = cross_subject_result_to_training_results(
                    results[model_type], model_type, args.task
                )
                if training_results:
                    data_sources.append(PlotDataSource(
                        model_type=model_type,
                        results=training_results,
                        is_current_run=True,
                        label=f'{model_type.upper()} (Cross)',
                    ))

            # 5: (Optional) Historical cross-subject data
            if not args.no_cross_subject_historical:
                search_model = 'cbramod' if 'cbramod' in args.models else args.models[0]
                hist_cross = db.find_best_cross_subject_results(
                    paradigm=args.paradigm,
                    task=args.task,
                    model_type=search_model,
                    n_channels=args.channels,
                    channel_config=channel_config_filter,
                    subjects=subjects_set,
                    exclude_run_id=db_run_id,
                )
                if hist_cross:
                    data_sources.append(PlotDataSource(
                        model_type=search_model,
                        results=hist_cross,
                        is_current_run=False,
                        label=f'{search_model.upper()} (Cross-Hist)',
                        hatch='...',
                    ))

            if data_sources:
                plot_filename = generate_result_filename(
                    'combined', args.paradigm, args.task, 'png', run_tag, is_cross_subject=True
                )
                plot_path = Path(args.results_dir) / plot_filename

                generate_combined_plot(
                    data_sources=data_sources,
                    output_path=str(plot_path),
                    task_type=args.task,
                    paradigm=args.paradigm,
                )
                log_io.info(f"Comparison plot saved: {plot_path}")
            else:
                log_io.warning("No data sources available for plotting")

    db.close()

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
