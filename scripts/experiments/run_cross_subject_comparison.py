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

Usage:
    # Run on Motor Imagery (default paradigm)
    uv run python scripts/run_cross_subject_comparison.py

    # Run on Motor Execution
    uv run python scripts/run_cross_subject_comparison.py --paradigm movement

    # Run on specific subjects
    uv run python scripts/run_cross_subject_comparison.py --subjects S01 S02 S03 S04 S05

    # Run only EEGNet
    uv run python scripts/run_cross_subject_comparison.py --models eegnet

    # Suppress plot generation
    uv run python scripts/run_cross_subject_comparison.py --no-plot

    # Disable within-subject historical comparison
    uv run python scripts/run_cross_subject_comparison.py --no-within-subject-historical
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path (scripts/experiments/ -> scripts/ -> project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.constants import PARADIGM_CONFIG
from src.utils.device import set_seed, check_cuda_available, get_device
from src.utils.logging import SectionLogger, setup_logging

from src.results import (
    compare_models,
    print_comparison_report,
    save_cross_subject_result,
    find_compatible_cross_subject_results,
    build_cross_subject_data_sources,
    generate_result_filename,
    TrainingResult,
    cross_subject_result_to_training_results,
)
from src.visualization import generate_combined_plot
from src.training.train_cross_subject import train_cross_subject
from src.config.training import SCHEDULER_PRESETS

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
        choices=['binary', 'ternary', 'quaternary'],
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

    # Generate run tag
    run_tag = datetime.now().strftime("%Y%m%d_%H%M")
    log_main.info(f"Starting new cross-subject comparison run: {run_tag}")

    paradigm_desc = PARADIGM_CONFIG[args.paradigm]['description']
    log_main.info(f"Paradigm: {paradigm_desc}")

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

    # Build config_overrides if scheduler specified
    config_overrides = None
    if args.scheduler:
        config_overrides = {'training': {'scheduler': args.scheduler}}

    # Train each model
    results = {}
    for model_type in args.models:
        log_main.info(f"{'='*50} {model_type.upper()} {'='*50}")

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
            config_overrides=config_overrides,
            cache_only=args.cache_only,
            wandb_enabled=not args.no_wandb,
            upload_model=args.upload_model,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            verbose=verbose,
        )

        results[model_type] = model_results

        # Save individual model results
        results_path = save_cross_subject_result(
            result=model_results,
            model_type=model_type,
            paradigm=args.paradigm,
            task=args.task,
            output_dir=args.results_dir,
            run_tag=run_tag,
        )
        log_io.info(f"{model_type.upper()} results saved: {results_path}")

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

    # Print comparison report
    print_comparison_report(results_as_training, comparison, args.task, args.paradigm, run_tag)

    # Generate visualization
    if not args.no_plot:
        # Search for cross-subject historical data
        cross_subject_historical = None

        if not args.no_cross_subject_historical:
            search_model = 'cbramod' if 'cbramod' in args.models else args.models[0]
            cross_subject_historical = find_compatible_cross_subject_results(
                output_dir=args.results_dir,
                paradigm=args.paradigm,
                task=args.task,
                subjects=subjects,
                model_type=search_model,
                exclude_run_tag=run_tag,
            )
            if cross_subject_historical:
                log_io.info(f"Found cross-subject historical: {cross_subject_historical.get('source_file', 'unknown')}")

        # Build data sources (within-subject historical searched per-model internally)
        data_sources = build_cross_subject_data_sources(
            current_results=results,
            output_dir=args.results_dir,
            paradigm=args.paradigm,
            task=args.task,
            cross_subject_historical=cross_subject_historical,
        )

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
