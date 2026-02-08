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
from datetime import datetime
from pathlib import Path

# Add project root to path (scripts/experiments/ -> scripts/ -> project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.constants import PARADIGM_CONFIG
from src.utils.device import set_seed, check_cuda_available, get_device
from src.utils.logging import SectionLogger, setup_logging

# Import from src modules
from src.results import (
    ComparisonResult,
    compare_models,
    print_comparison_report,
    load_cache,
    save_cache,
    find_cache_by_tag,
    prepare_combined_plot_data,
    generate_result_filename,
    save_full_comparison_results,
    load_comparison_results,
)
from src.visualization import generate_combined_plot, generate_comparison_plot
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

from _training_utils import discover_subjects
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

        test_accs = [r.test_acc for r in model_results]
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
        help='Experiment paradigm: imagery (MI) or movement (ME) (default: imagery)'
    )
    parser.add_argument(
        '--task', type=str, default='binary',
        choices=['binary', 'ternary', 'quaternary'],
        help='Classification task (default: binary)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='results',
        help='Directory to save results (default: results)'
    )
    parser.add_argument(
        '--resume', nargs='?', const='', default=None,
        metavar='TAG',
        help='Resume a previous run. Without TAG: resume most recent. '
             'With TAG: resume run matching the datetime substring (e.g., "20260205")'
    )
    parser.add_argument(
        '--force-retrain', action='store_true',
        help='Force retraining, overwriting cached results for this paradigm/task'
    )
    parser.add_argument(
        '--skip-training', action='store_true',
        help='Skip training entirely and load existing results file'
    )
    parser.add_argument(
        '--results-file', type=str, default=None,
        help='Path to existing results file (used with --skip-training)'
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help='Suppress plot generation (plots are generated by default)'
    )
    parser.add_argument(
        '--no-wandb', action='store_true',
        help='Disable wandb logging (default: enabled if wandb is installed)'
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
        '--cache-only', action='store_true',
        help='Load data exclusively from cache index (no filesystem scan)'
    )
    parser.add_argument(
        '--cache-index-path', type=str, default='.cache_index.json',
        help='Path to cache index file (default: .cache_index.json)'
    )
    parser.add_argument(
        '--scheduler', type=str, default=None,
        choices=['plateau', 'cosine', 'wsd', 'cosine_decay', 'cosine_annealing_warmup_decay'],
        help='Learning rate scheduler (default: model-specific)'
    )
    parser.add_argument(
        '--config', type=str, default=None, metavar='YAML_PATH',
        help='YAML 配置文件路径 (覆盖模型默认配置，被 CLI 参数覆盖)'
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

    # Handle --resume vs new run (default)
    if args.resume is not None:
        # --resume was used
        if args.resume == '':
            # --resume without TAG: resume most recent
            found = find_cache_by_tag(args.output_dir, args.paradigm, args.task)
            if found:
                _, run_tag = found
                log_main.info(f"Resuming most recent run: {run_tag or '(untagged)'}")
            else:
                log_main.error("No previous run found to resume")
                sys.exit(1)
        else:
            # --resume TAG: resume matching run
            found = find_cache_by_tag(args.output_dir, args.paradigm, args.task, args.resume)
            if found:
                _, run_tag = found
                log_main.info(f"Resuming run matching '{args.resume}': {run_tag}")
            else:
                log_main.error(f"No run found matching '{args.resume}'")
                sys.exit(1)
    else:
        # Default: start new run
        run_tag = datetime.now().strftime("%Y%m%d_%H%M")
        log_main.info(f"Starting new run: {run_tag}")

    paradigm_desc = PARADIGM_CONFIG[args.paradigm]['description']
    log_main.info(f"Paradigm: {paradigm_desc}")

    # Build merged config_overrides: YAML → CLI scheduler override
    from src.config.training import load_yaml_config
    config_overrides = load_yaml_config(args.config) if args.config else {}
    if args.scheduler:
        config_overrides.setdefault('training', {})['scheduler'] = args.scheduler
    config_overrides = config_overrides or None

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
                cache_index_path=args.cache_index_path
            )

        if not subjects:
            log_main.error(f"No subjects in {args.data_root}")
            sys.exit(1)

        log_main.info(f"Subjects: {subjects} | Models: {args.models} | Task: {args.task}")

        if args.resume is not None:
            log_main.info("Resuming from cache (--force-retrain to overwrite)")

        # Collect WandB metadata once before model loop
        from scripts._wandb_setup import should_prompt_wandb, prompt_batch_session

        # Count subjects that need training (check cache for each model)
        subjects_needing_training = set()
        if args.force_retrain:
            # All subjects need training when force_retrain is set
            subjects_needing_training = set(subjects)
        else:
            for model_type in args.models:
                cache, _ = load_cache(args.output_dir, args.paradigm, args.task, run_tag, find_latest=(run_tag is None))
                cached_subjects = set(cache.get(model_type, {}).keys()) if cache else set()
                subjects_needing_training.update(set(subjects) - cached_subjects)

        wandb_session_metadata = None
        if should_prompt_wandb(
            wandb_enabled=not args.no_wandb,
            interactive=not args.no_wandb_interactive,
            has_training=bool(subjects_needing_training),
        ):
            wandb_session_metadata = prompt_batch_session(
                models=args.models,
                task=args.task,
                paradigm=args.paradigm,
                subjects_to_train=len(subjects_needing_training),
            )

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
                wandb_interactive=False,  # Disable internal prompting
                wandb_session_metadata=wandb_session_metadata,  # Use pre-collected metadata
                cache_only=args.cache_only,
                cache_index_path=args.cache_index_path,
                config_overrides=config_overrides,
            )
            results[model_type] = model_results

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
    )

    # Generate plots by default (unless --no-plot is specified)
    if not args.no_plot:
        # Try to generate combined plot with historical comparison
        current_model = args.models[0] if len(args.models) == 1 else None
        data_sources, hist_timestamp = prepare_combined_plot_data(
            output_dir=args.output_dir,
            paradigm=args.paradigm,
            task=args.task,
            current_results=results,
            current_model=current_model,
        )

        if data_sources:
            log_io.info("Generating combined plot with historical comparison")
            plot_filename = generate_result_filename('combined', args.paradigm, args.task, 'png', run_tag)
            plot_path = Path(args.output_dir) / plot_filename
            generate_combined_plot(
                data_sources=data_sources,
                output_path=str(plot_path),
                task_type=args.task,
                paradigm=args.paradigm,
                historical_timestamp=hist_timestamp,
            )
        elif comparison:
            # No historical data but have complete two-model comparison
            plot_filename = generate_result_filename('comparison', args.paradigm, args.task, 'png', run_tag)
            plot_path = Path(args.output_dir) / plot_filename
            generate_comparison_plot(results, comparison, str(plot_path), task_type=args.task)
        else:
            log_io.info("No historical data found and insufficient models for comparison plot")

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
