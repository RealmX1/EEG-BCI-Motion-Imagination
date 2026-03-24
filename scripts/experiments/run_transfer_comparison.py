#!/usr/bin/env python
"""
Transfer Learning Comparison Script for EEG-BCI Project.

This script fine-tunes pretrained cross-subject models (from run_cross_subject_comparison.py)
on individual subjects and compares the transfer learning performance of EEGNet vs CBraMod.

Workflow:
1. Auto-discovers (or accepts manual) best cross-subject pretrained checkpoints
2. Fine-tunes each model on each individual subject via run_single_model()
3. Compares fine-tuned EEGNet vs CBraMod (statistical tests)
4. Generates 6-way combined plot: within baseline + cross baseline + transfer results

Usage:
    # Auto-discover best pretrained models and fine-tune
    uv run python scripts/experiments/run_transfer_comparison.py --cache-only

    # Specify freeze strategy
    uv run python scripts/experiments/run_transfer_comparison.py --freeze-strategy partial --cache-only

    # Manual pretrained model paths
    uv run python scripts/experiments/run_transfer_comparison.py \\
        --pretrained-eegnet checkpoints/cross_subject/.../best.pt \\
        --pretrained-cbramod checkpoints/cross_subject/.../best.pt --cache-only

    # Resume a previous run
    uv run python scripts/experiments/run_transfer_comparison.py --resume --cache-only

    # Motor Execution paradigm, ternary task
    uv run python scripts/experiments/run_transfer_comparison.py --paradigm movement --task ternary --cache-only
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict

# Add project root to path (scripts/experiments/ -> scripts/ -> project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.constants import FULL_N_CHANNELS, PARADIGM_CONFIG, CacheType
from src.utils.device import set_seed, check_cuda_available, get_device
from src.utils.logging import SectionLogger, setup_logging

from src.results import (
    PlotDataSource,
    compare_models,
    print_comparison_report,
    compute_model_statistics,
    generate_result_filename,
    load_cache,
    save_cache,
)
from src.visualization import generate_combined_plot

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
    add_transfer_args,
    resolve_output_dir,
    resolve_run_tag,
    init_db_run,
    finalize_db_run,
    build_config_overrides,
    find_best_checkpoint_path,
    validate_checkpoint_compatibility,
)
from run_single_model import run_single_model


setup_logging('transfer_comparison')
logger = logging.getLogger(__name__)

log_main = SectionLogger(logger, 'main')
log_train = SectionLogger(logger, 'train')
log_io = SectionLogger(logger, 'io')


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
  uv run python scripts/experiments/run_transfer_comparison.py --cache-only

  # Specify freeze strategy
  uv run python scripts/experiments/run_transfer_comparison.py --freeze-strategy partial --cache-only

  # Manual pretrained model paths
  uv run python scripts/experiments/run_transfer_comparison.py \\
      --pretrained-eegnet checkpoints/cross_subject/.../best.pt \\
      --pretrained-cbramod checkpoints/cross_subject/.../best.pt --cache-only

  # Resume a previous run
  uv run python scripts/experiments/run_transfer_comparison.py --resume --cache-only
'''
    )

    add_common_args(parser)
    add_cache_resume_args(parser)
    add_channel_args(parser)
    add_training_config_args(parser)
    add_transfer_args(parser)
    add_wandb_args(parser)

    # Transfer-specific args (beyond shared builders)
    parser.add_argument('--models', nargs='+', default=['eegnet', 'cbramod'],
                        choices=['eegnet', 'cbramod'], help='Models to fine-tune (default: both)')
    parser.add_argument('--pretrained-eegnet', type=str, default=None,
                        help='Manual path to pretrained EEGNet checkpoint (.pt)')
    parser.add_argument('--pretrained-cbramod', type=str, default=None,
                        help='Manual path to pretrained CBraMod checkpoint (.pt)')
    parser.add_argument('--finetune-epochs', type=int, default=None,
                        help='Number of fine-tuning epochs (default: strategy/model-specific)')
    parser.add_argument('--finetune-lr', type=float, default=None,
                        help='Fine-tuning learning rate (default: strategy-specific)')
    parser.add_argument('--finetune-batch-size', type=int, default=None,
                        help='Fine-tuning batch size (default: model-specific)')
    parser.add_argument('--no-cross-subject-baseline', action='store_true',
                        help='Do not include cross-subject baseline in the plot')
    parser.add_argument('--baseline', action='store_true',
                        help='Mark this run as a designated baseline in ExperimentDB')

    # Override --task choices: transfer does NOT support 'unified'
    for action in parser._actions:
        if hasattr(action, 'dest') and action.dest == 'task':
            action.choices = ['binary', 'ternary', 'quaternary']
            break

    args = parser.parse_args()

    # Resolve output directory (auto-redirect for reduced channel mode)
    output_dir = resolve_output_dir(args)

    # Resolve run tag (handle --resume or start new)
    run_tag = resolve_run_tag(args, args.paradigm, args.task, output_dir,
                              cache_type=CacheType.TRANSFER)

    # Standard init
    start_time = time.time()
    check_cuda_available(required=True)
    set_seed(args.seed)
    log_main.info(f"Device: {get_device()}")

    paradigm_desc = PARADIGM_CONFIG[args.paradigm]['description']
    log_main.info(f"Paradigm: {paradigm_desc} | Task: {args.task} | Freeze: {args.freeze_strategy}")

    # DB init
    db, db_run_id = init_db_run(run_tag, 'transfer', args.paradigm, args.task, args)

    # Discover subjects
    subjects = args.subjects or discover_subjects(
        args.data_root, args.paradigm, args.task,
        cache_only=args.cache_only, cache_index_path=args.cache_index_path)
    if not subjects:
        log_main.error(f"No subjects found in {args.data_root}")
        sys.exit(1)
    log_main.info(f"Subjects: {subjects} ({len(subjects)} total)")

    # Build config overrides (YAML + CLI channels/classifier)
    config_overrides = build_config_overrides(args) or {}

    # Add transfer-specific overrides
    if args.finetune_epochs:
        config_overrides.setdefault('training', {})['epochs'] = args.finetune_epochs
    if args.finetune_lr:
        config_overrides.setdefault('training', {})['learning_rate'] = args.finetune_lr
    if args.finetune_batch_size:
        config_overrides.setdefault('training', {})['batch_size'] = args.finetune_batch_size
    config_overrides = config_overrides or None

    # ======================================================================
    # Discover pretrained checkpoints
    # ======================================================================
    pretrained_paths: Dict[str, str] = {}
    manual_overrides = {
        'eegnet': args.pretrained_eegnet,
        'cbramod': args.pretrained_cbramod,
    }
    n_channels_filter = args.channels if args.channels != FULL_N_CHANNELS else None

    for model_type in args.models:
        if manual_overrides.get(model_type):
            path = manual_overrides[model_type]
            if not Path(path).exists():
                log_main.error(f"Pretrained {model_type} not found: {path}")
                sys.exit(1)
            pretrained_paths[model_type] = path
            log_main.info(f"{model_type.upper()} pretrained (manual): {path}")
        else:
            path = find_best_checkpoint_path(
                model_type=model_type, paradigm=args.paradigm, task=args.task,
                subjects=subjects, results_dir=output_dir, n_channels=n_channels_filter)
            if path:
                pretrained_paths[model_type] = path
                log_main.info(f"{model_type.upper()} pretrained (auto): {path}")
            else:
                log_main.warning(
                    f"No pretrained {model_type} checkpoint found for "
                    f"{args.paradigm}/{args.task}. Skipping this model. "
                    f"Run 'scripts/experiments/run_cross_subject_comparison.py' first."
                )

    if not pretrained_paths:
        log_main.error(
            "No pretrained checkpoints found for any requested model. "
            "Run cross-subject training first:\n"
            "  uv run python scripts/experiments/run_cross_subject_comparison.py"
        )
        sys.exit(1)

    # Validate checkpoint compatibility (n_classes match)
    classifier_types = validate_checkpoint_compatibility(pretrained_paths, args.task)

    # ======================================================================
    # Fine-tune each model via run_single_model()
    # ======================================================================
    results = {}

    for model_type in args.models:
        if model_type not in pretrained_paths:
            continue

        log_main.info(f"{'='*50} {model_type.upper()} TRANSFER {'='*50}")

        model_results, stats = run_single_model(
            model_type=model_type,
            data_root=args.data_root,
            subject_ids=subjects,
            task=args.task,
            paradigm=args.paradigm,
            output_dir=output_dir,
            force_retrain=args.force_retrain,
            run_tag=run_tag,
            no_wandb=args.no_wandb,
            upload_model=args.upload_model,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            cache_only=args.cache_only,
            cache_index_path=args.cache_index_path,
            config_overrides=config_overrides,
            db=db,
            db_run_id=db_run_id,
            pretrained_path=pretrained_paths[model_type],
            freeze_strategy=args.freeze_strategy,
            cache_type=CacheType.TRANSFER,
        )
        results[model_type] = model_results

        # Save model summary to DB
        if db_run_id and model_results:
            try:
                db.save_summary(db_run_id, model_type, stats)
            except Exception as e:
                log_main.warning(f"DB summary failed: {e}")

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

    print_comparison_report(results, comparison, args.task, args.paradigm, run_tag)

    # ======================================================================
    # Save final cache with summary and comparison
    # ======================================================================
    cache, cache_metadata = load_cache(output_dir, args.paradigm, args.task, run_tag,
                                       cache_type=CacheType.TRANSFER)

    from dataclasses import asdict
    comparison_dict = asdict(comparison) if comparison else None

    summary = {}
    for mt, mrs in results.items():
        if mrs:
            summary[mt] = compute_model_statistics(mrs)

    n_subjects = len(set(r.subject_id for mrs in results.values() for r in mrs))

    transfer_config_meta = {
        'freeze_strategy': args.freeze_strategy,
        'finetune_epochs': args.finetune_epochs,
        'finetune_lr': args.finetune_lr,
        'finetune_batch_size': args.finetune_batch_size,
        'pretrained_paths': {k: str(v) for k, v in pretrained_paths.items()},
        'classifier_types': classifier_types,
    }

    save_cache(
        output_dir=output_dir,
        paradigm=args.paradigm,
        task=args.task,
        results=cache,
        run_tag=run_tag,
        summary=summary,
        comparison=comparison_dict,
        n_subjects=n_subjects,
        is_complete=True,
        cache_type=CacheType.TRANSFER,
        extra_metadata={'transfer_config': transfer_config_meta},
    )

    # ======================================================================
    # Generate 6-way visualization (BEFORE finalize_db_run which closes DB)
    # ======================================================================
    if not args.no_plot and results:
        subjects_set = set(subjects)
        channel_config_filter = args.channel_config if args.channels != FULL_N_CHANNELS else None
        data_sources = []

        # 1 & 2: Within-subject baselines (per model, from DB, hatch='///')
        for mt in ['eegnet', 'cbramod']:
            ws_result = db.find_best_within_subject_results(
                paradigm=args.paradigm,
                task=args.task,
                model_type=mt,
                n_channels=args.channels,
                channel_config=channel_config_filter,
                subjects=subjects_set,
                return_run_id=True,
            )
            if ws_result is not None:
                ws_results, ws_run_id = ws_result
                if db_run_id and ws_run_id:
                    db.add_baseline_ref(db_run_id, ws_run_id, 'within_subject_baseline', mt)
                mean_acc = sum(r.test_acc_majority for r in ws_results) / len(ws_results)
                log_io.info(f"Within-subject baseline for {mt}: mean={mean_acc:.1%}")
                data_sources.append(PlotDataSource(
                    model_type=mt,
                    results=ws_results,
                    is_current_run=False,
                    label=f'{mt.upper()} (Within)',
                    hatch='///',
                ))

        # 3 & 4: Cross-subject baselines (per model, from DB, hatch='...')
        if not args.no_cross_subject_baseline:
            for mt in ['eegnet', 'cbramod']:
                cs_result = db.find_best_cross_subject_results(
                    paradigm=args.paradigm,
                    task=args.task,
                    model_type=mt,
                    n_channels=args.channels,
                    channel_config=channel_config_filter,
                    subjects=subjects_set,
                    return_run_id=True,
                )
                if cs_result is not None:
                    cross_results, cs_run_id = cs_result
                    if db_run_id and cs_run_id:
                        db.add_baseline_ref(db_run_id, cs_run_id, 'cross_subject_baseline', mt)
                    mean_acc = sum(r.test_acc_majority for r in cross_results) / len(cross_results)
                    log_io.info(f"Cross-subject baseline for {mt}: mean={mean_acc:.1%}")
                    data_sources.append(PlotDataSource(
                        model_type=mt,
                        results=cross_results,
                        is_current_run=False,
                        label=f'{mt.upper()} (Cross)',
                        hatch='...',
                    ))

        # 5 & 6: Transfer results (current run)
        for mt in ['eegnet', 'cbramod']:
            model_results = results.get(mt, [])
            filtered = [r for r in model_results if r.subject_id in subjects_set]
            if filtered:
                data_sources.append(PlotDataSource(
                    model_type=mt,
                    results=filtered,
                    is_current_run=True,
                    label=f'{mt.upper()} (Transfer)',
                ))

        if data_sources:
            plot_filename = generate_result_filename(
                'transfer_combined', args.paradigm, args.task, 'png', run_tag)
            plot_path = Path(output_dir) / plot_filename

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
    # DB finalize (save comparison + transfer config + mark complete + close)
    # ======================================================================
    finalize_db_run(db, db_run_id, comparison, n_subjects,
                    transfer_config={
                        'freeze_strategy': args.freeze_strategy,
                        'finetune_epochs': args.finetune_epochs,
                        'finetune_lr': args.finetune_lr,
                        'finetune_batch_size': args.finetune_batch_size,
                        'pretrained_eegnet': str(pretrained_paths.get('eegnet', '')),
                        'pretrained_cbramod': str(pretrained_paths.get('cbramod', '')),
                        'classifier_type': next(iter(classifier_types.values()), None),
                    })

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
