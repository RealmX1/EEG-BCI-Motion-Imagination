#!/usr/bin/env python
"""
Cross-Subject Extra Sessions Experiment for EEG-BCI Project.

Evaluates whether adding extra online session data (sessions 3-5) improves
cross-subject model performance. Unlike within-subject extra sessions where
each subject gets its own model, here ONE model is trained on pooled data
from all subjects at each step.

Default mode: ALL subjects participate in training (including those without
extra sessions, who contribute their standard data at every step). Only
subjects with extra sessions are evaluated. Use --extra-only to restrict
training to only subjects with extra sessions.

Protocol (per_session strategy):
- Baseline: Train on all subjects' standard data, test each on Sess02_FT
- +Sess03: Add Sess02_FT + Sess03_Base to training, test on Sess03_FT
- +Sess04: Add Sess03_FT + Sess04_Base to training, test on Sess04_FT
- +Sess05: Add Sess04_FT + Sess05_Base to training, test on Sess05_FT

Usage:
    # Default: all subjects train, extra-session subjects evaluated
    uv run python scripts/experiments/run_cross_subject_extra_sessions.py --cache-only --no-wandb

    # Only extra-session subjects in training pool
    uv run python scripts/experiments/run_cross_subject_extra_sessions.py --extra-only --cache-only --no-wandb
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.constants import CacheType, DEFAULT_CACHE_INDEX_PATH, PARADIGM_CONFIG, TASKS
from src.utils.device import set_seed, check_cuda_available, get_device
from src.utils.logging import SectionLogger, setup_logging
from src.training.train_cross_subject import train_cross_subject

from src.preprocessing.discovery import (
    discover_extra_session_subjects,
    get_progressive_session_folders,
    get_session_folders_for_split,
)

# Import from scripts directory
SCRIPTS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from _training_utils import (
    add_common_args,
    add_cache_resume_args,
    add_wandb_args,
    add_training_config_args,
    build_config_overrides,
    discover_subjects,
    resolve_output_dir,
    resolve_run_tag,
)

setup_logging('cross_subject_extra_sessions')
logger = logging.getLogger(__name__)
log_main = SectionLogger(logger, 'main')
log_train = SectionLogger(logger, 'train')
log_io = SectionLogger(logger, 'io')

CACHE_TYPE = 'cross_subject_extra_sessions_cache'


# ============================================================================
# Cache I/O
# ============================================================================

def _cache_filename(run_tag: str, paradigm: str, task: str) -> str:
    return f'{run_tag}_{CACHE_TYPE}_{paradigm}_{task}.json'


def save_cache(
    output_dir: str,
    paradigm: str,
    task: str,
    run_tag: str,
    all_results: Dict[str, Dict[str, dict]],
    train_subjects: Optional[List[str]] = None,
    eval_subjects: Optional[List[str]] = None,
    extra_only: bool = False,
):
    """Save cross-subject extra sessions results to JSON cache.

    Args:
        all_results: {model_type: {step_key: {per_subject_test_acc, mean_test_acc, ...}}}
        train_subjects: All subjects in the training pool
        eval_subjects: Subjects evaluated (those with extra sessions)
        extra_only: Whether --extra-only mode was used
    """
    cache_data = {
        'metadata': {
            'paradigm': paradigm,
            'task': task,
            'run_tag': run_tag,
            'training_type': 'cross_subject_extra_sessions',
            'test_strategy': 'per_session',
            'training_mode': 'extra_only' if extra_only else 'all_subjects',
            'train_subjects': train_subjects or [],
            'eval_subjects': eval_subjects or [],
            'n_train_subjects': len(train_subjects) if train_subjects else 0,
            'n_eval_subjects': len(eval_subjects) if eval_subjects else 0,
            'timestamp': datetime.now().isoformat(),
        },
        'results': all_results,
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / _cache_filename(run_tag, paradigm, task)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)

    log_io.info(f"Cache saved: {filepath}")
    return str(filepath)


def load_cache(
    output_dir: str, paradigm: str, task: str, run_tag: str,
) -> Optional[Dict]:
    filepath = Path(output_dir) / _cache_filename(run_tag, paradigm, task)
    if not filepath.exists():
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Cross-subject extra sessions experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--models', nargs='+', default=['eegnet', 'cbramod'],
        choices=['eegnet', 'cbramod'],
        help='Models to train (default: both)'
    )
    parser.add_argument(
        '--extra-only', action='store_true',
        help='Only include subjects with extra sessions in training pool '
             '(default: include ALL subjects for training, evaluate only extra-session subjects)'
    )
    add_common_args(parser)
    add_cache_resume_args(parser)
    add_wandb_args(parser)
    add_training_config_args(parser)

    args = parser.parse_args()

    # ====== Setup ======
    start_time = time.time()
    check_cuda_available(required=True)
    device = get_device()
    set_seed(args.seed)

    output_dir = resolve_output_dir(args)
    run_tag = resolve_run_tag(args, args.paradigm, args.task, output_dir, cache_type=CACHE_TYPE)

    paradigm_desc = PARADIGM_CONFIG[args.paradigm]['description']
    log_main.info(f"Cross-Subject Extra Sessions | {paradigm_desc} | {args.task}")
    log_main.info(f"Run tag: {run_tag} | Device: {device}")

    # ====== Discover subjects ======
    # Extra-session subjects (for evaluation + extra data)
    subjects_with_sessions = discover_extra_session_subjects(
        args.data_root, args.paradigm, args.task,
        cache_only=args.cache_only,
        cache_index_path=DEFAULT_CACHE_INDEX_PATH,
    )

    if not subjects_with_sessions:
        log_main.error("No subjects with extra sessions found")
        sys.exit(1)

    if args.subjects:
        subjects_with_sessions = {
            s: v for s, v in subjects_with_sessions.items()
            if s in args.subjects
        }

    if not subjects_with_sessions:
        log_main.error("No specified subjects have extra sessions")
        sys.exit(1)

    eval_subjects = sorted(subjects_with_sessions.keys())

    # Determine common extra sessions across eval subjects
    all_session_sets = [set(sessions) for sessions in subjects_with_sessions.values()]
    common_sessions = sorted(set.intersection(*all_session_sets)) if all_session_sets else []

    if not common_sessions:
        log_main.error("No common extra sessions across eval subjects")
        sys.exit(1)

    # Training subject pool: all subjects (default) or extra-only
    if args.extra_only:
        train_subjects = eval_subjects
        log_main.info(f"Training pool (extra-only): {len(train_subjects)} subjects")
    else:
        all_subjects = discover_subjects(
            args.data_root, args.paradigm, args.task,
            cache_only=args.cache_only,
        )
        if args.subjects:
            all_subjects = [s for s in all_subjects if s in args.subjects or s in eval_subjects]
        train_subjects = sorted(set(all_subjects) | set(eval_subjects))
        non_extra = sorted(set(train_subjects) - set(eval_subjects))
        log_main.info(f"Training pool (all subjects): {len(train_subjects)} subjects "
                      f"({len(eval_subjects)} with extra sessions + {len(non_extra)} standard-only: {non_extra})")

    log_main.info(f"Eval subjects ({len(eval_subjects)}): {eval_subjects}")
    log_main.info(f"Common extra sessions: {common_sessions}")

    # ====== Load existing cache (for resume) ======
    existing_cache = None
    if not args.force_retrain:
        existing_cache = load_cache(output_dir, args.paradigm, args.task, run_tag)
        if existing_cache:
            log_main.info(f"Loaded existing cache for run {run_tag}")

    # ====== Build config overrides ======
    config_overrides = build_config_overrides(args)

    # ====== Train ======
    all_results: Dict[str, Dict[str, dict]] = {}

    # Steps: baseline + each extra session
    step_definitions = [('baseline', 2)] + [(f'sess{s:02d}', s) for s in common_sessions]

    for model_type in args.models:
        log_main.info(f"{'='*50} {model_type.upper()} {'='*50}")
        model_results: Dict[str, dict] = {}

        for step_key, up_to_session in step_definitions:
            # Check cache
            cached = (existing_cache or {}).get('results', {}).get(
                model_type, {}
            ).get(step_key)

            if cached and not args.force_retrain:
                model_results[step_key] = cached
                mean_acc = cached.get('mean_test_acc', 0)
                log_train.info(f"{step_key}: mean={mean_acc:.4f} [cached]")
                continue

            # Get session folders for this step
            # Baseline: None → train_cross_subject uses default session folders
            # Progressive: override with incremental session folders
            if step_key == 'baseline':
                session_folders_override = None
            else:
                session_folders_override = get_progressive_session_folders(
                    args.paradigm, args.task, up_to_session
                )

            if session_folders_override is not None:
                log_train.info(f"Step {step_key}: train folders = {session_folders_override['train']}")
                log_train.info(f"Step {step_key}: test folders = {session_folders_override['test']}")
            else:
                log_train.info(f"Step {step_key}: using default session folders")

            wandb_group = (f'{model_type}_{args.paradigm}_{args.task}_'
                           f'cross_extra_{step_key}_{run_tag}')

            try:
                result = train_cross_subject(
                    subjects=train_subjects,
                    model_type=model_type,
                    task=args.task,
                    paradigm=args.paradigm,
                    save_dir='checkpoints/cross_subject',
                    data_root=args.data_root,
                    device=device,
                    seed=args.seed,
                    run_tag=f'{run_tag}_{step_key}',
                    config_overrides=config_overrides,
                    cache_only=args.cache_only,
                    wandb_enabled=not args.no_wandb,
                    wandb_group=wandb_group,
                    verbose=1,
                    session_folders_override=session_folders_override,
                    resume_checkpoint=True,
                )

                # Filter test results to eval subjects only.
                # For baseline step, train_cross_subject evaluates ALL train_subjects
                # (including those without extra sessions), but we only keep
                # eval_subjects for fair comparison across steps.
                all_tested = set(result['per_subject_test_acc'].keys())
                eval_test_acc = {
                    s: acc for s, acc in result['per_subject_test_acc'].items()
                    if s in eval_subjects
                }
                excluded = all_tested - set(eval_test_acc.keys())
                if excluded:
                    log_train.info(f"  Eval filter: kept {len(eval_test_acc)}/{len(all_tested)} "
                                   f"subjects (excluded {sorted(excluded)} from metrics)")
                eval_mean = float(np.mean(list(eval_test_acc.values()))) if eval_test_acc else 0.0
                eval_std = float(np.std(list(eval_test_acc.values()))) if eval_test_acc else 0.0

                step_result = {
                    'step': step_key,
                    'up_to_session': up_to_session,
                    'per_subject_test_acc': eval_test_acc,
                    'mean_test_acc': eval_mean,
                    'std_test_acc': eval_std,
                    'val_acc': result['val_acc'],
                    'best_epoch': result['best_epoch'],
                    'training_time': result['training_time'],
                    'model_path': result['model_path'],
                    'n_train_subjects': len(train_subjects),
                    'n_eval_subjects': len(eval_test_acc),
                }
                model_results[step_key] = step_result

                baseline_acc = model_results.get('baseline', {}).get('mean_test_acc', 0)
                delta = eval_mean - baseline_acc if baseline_acc else 0
                log_train.info(f"{step_key}: mean={eval_mean:.4f} "
                               f"(Δ={delta:+.4f}, {len(eval_test_acc)} eval subjects)")

            except Exception as e:
                log_train.error(f"{step_key} FAILED: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Incremental cache save after each step
            all_results[model_type] = model_results
            save_cache(output_dir, args.paradigm, args.task, run_tag, all_results,
                       train_subjects=train_subjects, eval_subjects=eval_subjects,
                       extra_only=args.extra_only)

        all_results[model_type] = model_results

    # ====== Summary ======
    print()
    log_main.info("=" * 60)
    log_main.info("CROSS-SUBJECT EXTRA SESSIONS SUMMARY")
    log_main.info("=" * 60)

    for model_type in args.models:
        model_data = all_results.get(model_type, {})
        if not model_data:
            continue

        print(f"\n  {model_type.upper()}:")
        baseline_acc = model_data.get('baseline', {}).get('mean_test_acc', 0)
        for step_key, _ in step_definitions:
            step = model_data.get(step_key, {})
            mean_acc = step.get('mean_test_acc', 0)
            delta = mean_acc - baseline_acc if baseline_acc and step_key != 'baseline' else 0
            delta_str = f" (Δ={delta:+.2%})" if step_key != 'baseline' else ""
            print(f"    {step_key}: {mean_acc:.2%} ± {step.get('std_test_acc', 0):.2%}{delta_str}")

            # Per-subject detail
            per_subj = step.get('per_subject_test_acc', {})
            if per_subj:
                for sid in sorted(per_subj.keys()):
                    bl_subj = model_data.get('baseline', {}).get('per_subject_test_acc', {}).get(sid, 0)
                    d = per_subj[sid] - bl_subj if bl_subj and step_key != 'baseline' else 0
                    d_str = f" (Δ={d:+.2%})" if step_key != 'baseline' else ""
                    print(f"      {sid}: {per_subj[sid]:.2%}{d_str}")

    # ====== Final save ======
    final_path = save_cache(output_dir, args.paradigm, args.task, run_tag, all_results,
                            train_subjects=train_subjects, eval_subjects=eval_subjects,
                            extra_only=args.extra_only)

    # ====== Visualization ======
    if not args.no_plot and all_results:
        from src.visualization.extra_sessions import generate_extra_sessions_combined_plot

        # Transform cross-subject format {model: {step: {per_subject_test_acc: ...}}}
        # into within-subject format {model: {subject: {step: {test_acc_majority: ...}}}}
        plot_results = {}
        for model_type, steps in all_results.items():
            plot_results[model_type] = {}
            for step_key, step_data in steps.items():
                per_subj = step_data.get('per_subject_test_acc', {})
                for sid, acc in per_subj.items():
                    plot_results[model_type].setdefault(sid, {})[step_key] = {
                        'test_acc_majority': acc,
                    }

        # Build subjects_with_sessions from common_sessions
        plot_subjects_with_sessions = {s: common_sessions for s in eval_subjects}

        plot_filename = f'{run_tag}_cross_subject_extra_sessions_{args.paradigm}_{args.task}.png'
        plot_path = Path(output_dir) / plot_filename

        try:
            generate_extra_sessions_combined_plot(
                all_results=plot_results,
                subjects_with_sessions=plot_subjects_with_sessions,
                output_path=str(plot_path),
                paradigm=args.paradigm,
                task=args.task,
            )
            log_main.info(f"Plot saved: {plot_path}")
        except Exception as e:
            log_main.error(f"Plot generation failed: {e}")
            import traceback
            traceback.print_exc()

    total_time = time.time() - start_time
    if total_time >= 3600:
        log_main.info(f"Total time: {total_time/3600:.1f}h")
    elif total_time >= 60:
        log_main.info(f"Total time: {total_time/60:.1f}m")
    else:
        log_main.info(f"Total time: {total_time:.1f}s")

    log_main.info(f"Results: {final_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
