#!/usr/bin/env python
"""
Extra Online Sessions Experiment for EEG-BCI Project.

Evaluates whether adding additional online session data (sessions 3-5)
improves decoding performance for EEGNet and CBraMod. The original paper
observed that EEGNet shows limited improvement with additional data.

Test set strategies (--test-strategy):
- per_session (default): Each step tests on its own session's Finetune data.
    Non-comparable across steps but matches original experimental design.
- fixed_combined: Fixed test set = last 1/4 trials from ALL Finetune sessions
    (Sess02-05). Comparable across steps; first 3/4 FT trials enter training.
- fixed_sess02: Test set = Sess02_Finetune (same as standard baseline).
    Extra session data (Sess03-05) added to training only.

Usage:
    # Default: per_session strategy, binary, both models
    uv run python scripts/experiments/run_extra_sessions.py --no-wandb

    # Fixed combined test set
    uv run python scripts/experiments/run_extra_sessions.py --test-strategy fixed_combined --no-wandb

    # Fixed Sess02 test set
    uv run python scripts/experiments/run_extra_sessions.py --test-strategy fixed_sess02 --no-wandb

    # Ternary task
    uv run python scripts/experiments/run_extra_sessions.py --task ternary --no-wandb

    # Resume previous run
    uv run python scripts/experiments/run_extra_sessions.py --resume --no-wandb
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.constants import CacheType, DEFAULT_CACHE_INDEX_PATH, PARADIGM_CONFIG, TASKS
from src.results import ExperimentDB, TrainingResult, compute_model_statistics
from src.results.serialization import result_to_dict, dict_to_result
from src.utils.device import set_seed, check_cuda_available, get_device
from src.utils.logging import SectionLogger, setup_logging

# Import from scripts directory
SCRIPTS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from _training_utils import (
    add_common_args,
    add_cache_resume_args,
    add_wandb_args,
    add_training_config_args,
    build_config_overrides,
    resolve_output_dir,
    resolve_run_tag,
    init_db_run,
    finalize_db_run,
    train_and_get_result,
    discover_subjects,
)

from src.preprocessing.discovery import (
    discover_extra_session_subjects,
    get_progressive_session_folders,
    get_progressive_session_folders_fixed_sess02,
    get_all_extra_session_folders,
)

setup_logging('extra_sessions')
logger = logging.getLogger(__name__)
log_main = SectionLogger(logger, 'main')
log_train = SectionLogger(logger, 'train')
log_io = SectionLogger(logger, 'io')

# Test strategy constants
STRATEGY_PER_SESSION = 'per_session'
STRATEGY_FIXED_COMBINED = 'fixed_combined'
STRATEGY_FIXED_SESS02 = 'fixed_sess02'


# ============================================================================
# Baseline Loading
# ============================================================================

def load_baseline_results(
    db: ExperimentDB,
    baseline_run: Dict[str, Any],
    model_type: str,
    subjects: List[str],
) -> Dict[str, TrainingResult]:
    """Load baseline per-subject results from ExperimentDB.

    Args:
        baseline_run: Result from db.find_baseline_run() (must not be None).

    Returns:
        {subject_id: TrainingResult} for requested subjects only.
    """
    run_id = baseline_run['run_id']
    all_results = db.get_results(run_id, model_type=model_type)
    return {r.subject_id: r for r in all_results if r.subject_id in subjects}


# ============================================================================
# Cache I/O
# ============================================================================

def _cache_filename(run_tag: str, paradigm: str, task: str, strategy: str) -> str:
    """Build cache filename, encoding strategy for non-default strategies."""
    base = CacheType.EXTRA_SESSIONS
    if strategy == STRATEGY_PER_SESSION:
        return f'{run_tag}_{base}_{paradigm}_{task}.json'
    return f'{run_tag}_{base}_{strategy}_{paradigm}_{task}.json'


def save_extra_sessions_cache(
    output_dir: str,
    paradigm: str,
    task: str,
    run_tag: str,
    all_results: Dict[str, Dict[str, Dict[str, dict]]],
    baseline_run_tags: Dict[str, str],
    strategy: str = STRATEGY_PER_SESSION,
):
    """Save extra sessions results to JSON cache.

    Args:
        all_results: {model_type: {subject_id: {step_key: result_dict}}}
            step_key is 'baseline', 'sess03', 'sess04', 'sess05'
        baseline_run_tags: {model_type: run_tag}
        strategy: Test set strategy identifier
    """
    cache_data = {
        'metadata': {
            'paradigm': paradigm,
            'task': task,
            'run_tag': run_tag,
            'training_type': 'extra_sessions',
            'test_strategy': strategy,
            'timestamp': datetime.now().isoformat(),
        },
        'baseline_run_tags': baseline_run_tags,
        'results': all_results,
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = _cache_filename(run_tag, paradigm, task, strategy)
    filepath = output_path / filename

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)

    log_io.info(f"Cache saved: {filepath}")
    return str(filepath)


def load_extra_sessions_cache(
    output_dir: str,
    paradigm: str,
    task: str,
    run_tag: str,
    strategy: str = STRATEGY_PER_SESSION,
) -> Optional[Dict]:
    """Load existing extra sessions cache if available."""
    output_path = Path(output_dir)
    filename = _cache_filename(run_tag, paradigm, task, strategy)
    filepath = output_path / filename

    if not filepath.exists():
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================================
# Fixed Combined Strategy — Data Preparation
# ============================================================================

def _get_preprocess_config(model_type: str, task: str, config_overrides: Optional[Dict] = None):
    """Build preprocessing config for a model type (mirrors train_single_subject logic)."""
    from src.preprocessing.data_loader import PreprocessConfig

    n_classes = TASKS[task]['n_classes']

    if model_type == 'cbramod':
        config = PreprocessConfig.for_cbramod(full_channels=True)
    else:
        config = PreprocessConfig.paper_aligned(n_class=n_classes)

    if config_overrides:
        data_config = config_overrides.get('data', {})
        channels = data_config.get('channels')
        channel_config = data_config.get('channel_config')
        config.apply_channel_overrides(channels=channels, channel_config=channel_config)

    return config


def prepare_fixed_combined_data(
    full_dataset,
    available_sessions: List[int],
    up_to_session: int,
    paradigm: str,
    task: str,
) -> Dict:
    """Prepare precomputed data dict for fixed_combined strategy.

    Test set = last 1/4 trials from EVERY Finetune session (Sess02-05).
    Train set = all non-test trials from active sessions (up to up_to_session).

    Args:
        full_dataset: FingerEEGDataset loaded with ALL session folders
        available_sessions: Extra session numbers (e.g., [3, 4, 5])
        up_to_session: Current step — sessions up to this number are active.
            2 = baseline (only Sess01+02 active), 3 = +Sess03, etc.
        paradigm: 'imagery' or 'movement'
        task: 'binary' or 'ternary'

    Returns:
        Dict with keys: 'train_dataset', 'train_indices', 'val_indices',
        'test_indices' (no 'test_dataset' — test is index-based).
    """
    from src.training.common import temporal_split_by_group

    paradigm_prefix = 'Imagery' if paradigm == 'imagery' else 'Movement'
    online_prefix = f'Online{paradigm_prefix}'
    n_class = '2class' if task == 'binary' else '3class'

    # 1. Group trials by session_type (folder name)
    session_trials: Dict[str, set] = defaultdict(set)
    for info in full_dataset.trial_infos:
        session_trials[info.session_type].add(info.trial_idx)
    session_trials = {k: sorted(v) for k, v in session_trials.items()}

    # 2. Identify Finetune sessions and split their trials (last 1/4 → test)
    test_trial_set = set()
    trainable_ft_trials: Dict[str, List[int]] = {}  # folder → first 3/4 trial indices

    for sess_num in [2] + sorted(available_sessions):
        ft_folder = f'{online_prefix}_Sess0{sess_num}_{n_class}_Finetune'
        if ft_folder not in session_trials:
            continue
        trials = session_trials[ft_folder]
        n_test = max(1, len(trials) // 4)
        test_trial_set.update(trials[-n_test:])
        trainable_ft_trials[ft_folder] = trials[:-n_test]

    # 3. Determine active folders for this step
    offline_folder = f'Offline{paradigm_prefix}'
    active_folders = {
        offline_folder,
        f'{online_prefix}_Sess01_{n_class}_Base',
        f'{online_prefix}_Sess01_{n_class}_Finetune',
        f'{online_prefix}_Sess02_{n_class}_Base',
        f'{online_prefix}_Sess02_{n_class}_Finetune',
    }
    for sess in range(3, up_to_session + 1):
        active_folders.add(f'{online_prefix}_Sess0{sess}_{n_class}_Base')
        active_folders.add(f'{online_prefix}_Sess0{sess}_{n_class}_Finetune')

    # 4. Collect active non-test trial indices
    active_trials = set()
    for folder, trials in session_trials.items():
        if folder not in active_folders:
            continue
        if folder in trainable_ft_trials:
            # FT folder: only the first 3/4 (trainable portion)
            active_trials.update(trainable_ft_trials[folder])
        else:
            # Non-FT folder: all trials (none are in test set)
            active_trials.update(t for t in trials if t not in test_trial_set)

    # 5. Temporal split: 80/20 within each session group
    # Build per-session group mapping for active trials only
    group_to_trials: Dict[str, set] = defaultdict(set)
    for info in full_dataset.trial_infos:
        if info.trial_idx in active_trials:
            group_to_trials[info.session_type].add(info.trial_idx)

    train_trials_list: List[int] = []
    val_trials_list: List[int] = []
    for _group, trials in group_to_trials.items():
        sorted_t = sorted(trials)
        n_val = max(1, int(len(sorted_t) * 0.2))
        train_trials_list.extend(sorted_t[:-n_val])
        val_trials_list.extend(sorted_t[-n_val:])

    # 6. Convert trial indices to segment indices
    train_indices = full_dataset.get_segment_indices_for_trials(train_trials_list)
    val_indices = full_dataset.get_segment_indices_for_trials(val_trials_list)
    test_indices = full_dataset.get_segment_indices_for_trials(list(test_trial_set))

    n_test_trials = len(test_trial_set)
    n_train_trials = len(train_trials_list)
    n_val_trials = len(val_trials_list)
    log_train.info(f"fixed_combined split (up_to={up_to_session}): "
                   f"train={n_train_trials}t/{len(train_indices)}s, "
                   f"val={n_val_trials}t/{len(val_indices)}s, "
                   f"test={n_test_trials}t/{len(test_indices)}s (fixed)")

    return {
        'train_dataset': full_dataset,
        'test_dataset': None,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices,
    }


def load_full_dataset(
    data_root: str,
    subject_id: str,
    paradigm: str,
    task: str,
    available_sessions: List[int],
    preprocess_config,
    elc_path: Path,
    cache_only: bool = False,
):
    """Load ALL session folders into a single dataset for fixed_combined strategy."""
    from src.training.train_within_subject import load_subject_data

    all_folders = get_all_extra_session_folders(paradigm, task, available_sessions)
    target_classes = TASKS[task]['classes']

    dataset = load_subject_data(
        Path(data_root), subject_id,
        session_folders=all_folders,
        target_classes=target_classes,
        config=preprocess_config,
        elc_path=elc_path,
        # No trial rejection — test trials must be preserved intact
        reject_trials=False,
        cache_only=cache_only,
    )
    log_train.info(f"{subject_id}: loaded full dataset with {len(dataset)} segments "
                   f"from {len(all_folders)} folders")
    return dataset


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extra online sessions experiment: evaluate progressive data addition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Default: per_session strategy, binary, both models
  uv run python scripts/experiments/run_extra_sessions.py --no-wandb

  # Fixed combined test set (comparable across steps)
  uv run python scripts/experiments/run_extra_sessions.py --test-strategy fixed_combined --no-wandb

  # Fixed Sess02 as test set
  uv run python scripts/experiments/run_extra_sessions.py --test-strategy fixed_sess02 --no-wandb
'''
    )

    parser.add_argument(
        '--models', nargs='+', default=['eegnet', 'cbramod'],
        choices=['eegnet', 'cbramod'],
        help='Models to train (default: both)'
    )
    parser.add_argument(
        '--test-strategy', type=str, default=STRATEGY_PER_SESSION,
        choices=[STRATEGY_PER_SESSION, STRATEGY_FIXED_COMBINED, STRATEGY_FIXED_SESS02],
        help='Test set strategy (default: per_session)'
    )
    add_common_args(parser)
    add_cache_resume_args(parser)
    add_wandb_args(parser)
    add_training_config_args(parser)

    args = parser.parse_args()
    test_strategy = args.test_strategy

    # ====== Setup ======
    start_time = time.time()
    check_cuda_available(required=True)
    device = get_device()
    set_seed(args.seed)

    output_dir = resolve_output_dir(args)

    # Strategy-aware resume: find_cache_by_tag glob doesn't match strategy-tagged
    # filenames, so we handle resume directly for non-default strategies.
    if getattr(args, 'resume', None) is not None and test_strategy != STRATEGY_PER_SESSION:
        tag_hint = args.resume if args.resume != '' else None
        # Search output_dir for strategy-tagged cache files
        output_path = Path(output_dir)
        pattern = f'*{CacheType.EXTRA_SESSIONS}_{test_strategy}_{args.paradigm}_{args.task}.json'
        candidates = list(output_path.glob(pattern)) if output_path.exists() else []
        if tag_hint:
            candidates = [c for c in candidates if tag_hint in c.name]
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            # Extract run_tag from filename
            suffix = f'_{CacheType.EXTRA_SESSIONS}_{test_strategy}_{args.paradigm}_{args.task}.json'
            run_tag = candidates[0].name[:-len(suffix)]
            log_main.info(f"Resuming run: {run_tag} (strategy={test_strategy})")
        else:
            log_main.error(f"No previous {test_strategy} run found to resume")
            sys.exit(1)
    else:
        run_tag = resolve_run_tag(args, args.paradigm, args.task, output_dir,
                                  cache_type=CacheType.EXTRA_SESSIONS)

    paradigm_desc = PARADIGM_CONFIG[args.paradigm]['description']
    log_main.info(f"Extra Sessions Experiment | {paradigm_desc} | {args.task}")
    log_main.info(f"Run tag: {run_tag} | Device: {device} | Strategy: {test_strategy}")

    # ====== Discover subjects with extra sessions ======
    subjects_with_sessions = discover_extra_session_subjects(
        args.data_root, args.paradigm, args.task,
        cache_only=args.cache_only,
        cache_index_path=DEFAULT_CACHE_INDEX_PATH,
    )

    if not subjects_with_sessions:
        log_main.error("No subjects with extra sessions found")
        sys.exit(1)

    # Filter to requested subjects if specified
    if args.subjects:
        subjects_with_sessions = {
            s: v for s, v in subjects_with_sessions.items() if s in args.subjects
        }
        if not subjects_with_sessions:
            log_main.error(f"None of {args.subjects} have extra sessions")
            sys.exit(1)

    subject_list = sorted(subjects_with_sessions.keys())
    log_main.info(f"Subjects: {subject_list}")
    for sid, sessions in sorted(subjects_with_sessions.items()):
        log_main.info(f"  {sid}: extra sessions {sessions}")

    # ====== Load existing cache (for resume) ======
    existing_cache = None
    if not args.force_retrain:
        existing_cache = load_extra_sessions_cache(
            output_dir, args.paradigm, args.task, run_tag, test_strategy
        )
        if existing_cache:
            log_main.info(f"Loaded existing cache for run {run_tag}")

    # ====== Initialize DB ======
    db, db_run_id = init_db_run(run_tag, 'extra_sessions', args.paradigm, args.task, args)

    # ====== Build config overrides ======
    config_overrides = build_config_overrides(args)

    # ====== Train ======
    all_results: Dict[str, Dict[str, Dict[str, dict]]] = {}
    baseline_run_tags: Dict[str, str] = {}

    # For fixed_combined, baseline needs training (different test set).
    # For per_session and fixed_sess02, baseline comes from ExperimentDB.
    needs_baseline_training = (test_strategy == STRATEGY_FIXED_COMBINED)

    for model_type in args.models:
        log_main.info(f"{'='*50} {model_type.upper()} {'='*50}")

        # Load DB baseline (for per_session and fixed_sess02)
        baseline_results = {}
        if not needs_baseline_training:
            baseline_run = db.find_baseline_run(
                args.paradigm, args.task, model_type, 'within_subject'
            )

            # per_session requires a complete baseline — abort early if missing
            if test_strategy == STRATEGY_PER_SESSION:
                if baseline_run is None:
                    log_main.error(
                        f"No baseline found for {model_type}/{args.paradigm}"
                        f"/{args.task} (within_subject). Cannot run "
                        f"extra_sessions with 'per_session' strategy "
                        f"without a baseline.\n"
                        f"  Run a baseline first:\n"
                        f"    uv run python scripts/experiments/"
                        f"run_within_subject_comparison.py "
                        f"--task {args.task} --models {model_type} "
                        f"--no-wandb --cache-only"
                    )
                    sys.exit(1)

            if baseline_run is not None:
                source = baseline_run.get('baseline_source', 'unknown')
                mean_acc = baseline_run.get('best_mean_acc', 0)
                log_main.info(
                    f"Baseline for {model_type}: {baseline_run['run_tag']} "
                    f"(mean={mean_acc:.4f}, source={source})"
                )

                baseline_results = load_baseline_results(
                    db, baseline_run, model_type, subject_list,
                )
                baseline_run_tags[model_type] = baseline_run['run_tag']

                # per_session: all subjects must have baseline data
                if test_strategy == STRATEGY_PER_SESSION:
                    missing = [s for s in subject_list
                               if s not in baseline_results]
                    if missing:
                        log_main.error(
                            f"Baseline {baseline_run['run_tag']} "
                            f"(source={source}) covers only "
                            f"{len(baseline_results)}/{len(subject_list)} "
                            f"subjects. Missing: {', '.join(missing)}\n"
                            f"  Run a full baseline first:\n"
                            f"    uv run python scripts/experiments/"
                            f"run_within_subject_comparison.py "
                            f"--task {args.task} --models {model_type} "
                            f"--no-wandb --cache-only"
                        )
                        sys.exit(1)
            else:
                log_main.warning(
                    f"No baseline found for {model_type}/{args.paradigm}"
                    f"/{args.task} — proceeding without baseline"
                )

        # For fixed_combined: prepare preprocess config + elc_path (once per model)
        preprocess_config_fc = None
        elc_path_fc = None
        if test_strategy == STRATEGY_FIXED_COMBINED:
            preprocess_config_fc = _get_preprocess_config(model_type, args.task, config_overrides)
            elc_path_fc = Path(args.data_root) / 'biosemi128.ELC'

        model_results: Dict[str, Dict[str, dict]] = {}

        for subject_id in subject_list:
            available_sessions = subjects_with_sessions[subject_id]
            subject_results: Dict[str, dict] = {}

            # For fixed_combined: load full dataset once per subject, reuse for
            # both baseline and extra session steps
            full_dataset_fc = None
            if test_strategy == STRATEGY_FIXED_COMBINED:
                full_dataset_fc = load_full_dataset(
                    args.data_root, subject_id, args.paradigm, args.task,
                    available_sessions, preprocess_config_fc, elc_path_fc,
                    cache_only=args.cache_only,
                )

            # ---------- Baseline ----------
            if needs_baseline_training:
                # fixed_combined: train baseline step (up_to_session=2)
                step_key = 'baseline'
                cached = (existing_cache or {}).get('results', {}).get(
                    model_type, {}
                ).get(subject_id, {}).get(step_key)

                if cached and not args.force_retrain:
                    subject_results[step_key] = cached
                    acc = cached.get('test_acc_majority', cached.get('test_acc', 0))
                    log_train.info(f"{subject_id} baseline: {acc:.4f} [cached]")
                else:
                    precomputed = prepare_fixed_combined_data(
                        full_dataset_fc, available_sessions, up_to_session=2,
                        paradigm=args.paradigm, task=args.task,
                    )

                    wandb_group = (f'{model_type}_{args.paradigm}_{args.task}_'
                                   f'extra_baseline_{run_tag}')
                    try:
                        result = train_and_get_result(
                            subject_id=subject_id,
                            model_type=model_type,
                            task=args.task,
                            paradigm=args.paradigm,
                            data_root=args.data_root,
                            save_dir='checkpoints',
                            run_tag=run_tag,
                            no_wandb=args.no_wandb,
                            wandb_group=wandb_group,
                            config_overrides=config_overrides,
                            verbose=1,
                            cache_only=args.cache_only,
                            precomputed_data=precomputed,
                        )
                        result_dict = result_to_dict(result)
                        result_dict['extra_session_step'] = 'baseline_fc'
                        subject_results[step_key] = result_dict
                        log_train.info(f"{subject_id} baseline: {result.test_acc_majority:.4f}")
                    except Exception as e:
                        log_train.error(f"{subject_id} baseline FAILED: {e}")
            else:
                # per_session / fixed_sess02: load from ExperimentDB
                # (per_session coverage validated above; fixed_sess02 may
                #  have partial coverage)
                if subject_id in baseline_results:
                    br = baseline_results[subject_id]
                    subject_results['baseline'] = result_to_dict(br)
                    log_train.info(f"{subject_id} baseline: {br.test_acc_majority:.4f}")
                else:
                    log_train.warning(f"{subject_id}: no baseline found for {model_type}")

            # ---------- Extra session steps ----------

            for sess_num in available_sessions:
                step_key = f'sess{sess_num:02d}'

                # Check cache
                cached = (existing_cache or {}).get('results', {}).get(
                    model_type, {}
                ).get(subject_id, {}).get(step_key)

                if cached and not args.force_retrain:
                    subject_results[step_key] = cached
                    acc = cached.get('test_acc_majority', cached.get('test_acc', 0))
                    baseline_acc = subject_results.get('baseline', {}).get('test_acc_majority', 0)
                    delta = acc - baseline_acc if baseline_acc else 0
                    log_train.info(f"{subject_id} +Sess{sess_num:02d}: {acc:.4f} "
                                   f"(Δ={delta:+.4f}) [cached]")
                    continue

                wandb_group = (f'{model_type}_{args.paradigm}_{args.task}_'
                               f'extra_sess{sess_num:02d}_{run_tag}')

                # Prepare data based on strategy
                session_folders_override = None
                precomputed_data = None

                if test_strategy == STRATEGY_PER_SESSION:
                    session_folders_override = get_progressive_session_folders(
                        args.paradigm, args.task, sess_num
                    )
                elif test_strategy == STRATEGY_FIXED_SESS02:
                    session_folders_override = get_progressive_session_folders_fixed_sess02(
                        args.paradigm, args.task, sess_num
                    )
                elif test_strategy == STRATEGY_FIXED_COMBINED:
                    precomputed_data = prepare_fixed_combined_data(
                        full_dataset_fc, available_sessions, up_to_session=sess_num,
                        paradigm=args.paradigm, task=args.task,
                    )

                try:
                    result = train_and_get_result(
                        subject_id=subject_id,
                        model_type=model_type,
                        task=args.task,
                        paradigm=args.paradigm,
                        data_root=args.data_root,
                        save_dir='checkpoints',
                        run_tag=run_tag,
                        no_wandb=args.no_wandb,
                        wandb_group=wandb_group,
                        config_overrides=config_overrides,
                        verbose=1,
                        cache_only=args.cache_only,
                        session_folders_override=session_folders_override,
                        precomputed_data=precomputed_data,
                    )

                    result_dict = result_to_dict(result)
                    result_dict['extra_session_step'] = sess_num
                    result_dict['test_strategy'] = test_strategy
                    subject_results[step_key] = result_dict

                    baseline_acc = subject_results.get('baseline', {}).get('test_acc_majority', 0)
                    delta = result.test_acc_majority - baseline_acc if baseline_acc else 0
                    log_train.info(f"{subject_id} +Sess{sess_num:02d}: "
                                   f"{result.test_acc_majority:.4f} (Δ={delta:+.4f})")

                    # DB dual-write
                    if db and db_run_id:
                        try:
                            db.save_subject_result(db_run_id, result)
                        except Exception as e:
                            log_train.warning(f"DB write failed: {e}")

                except Exception as e:
                    log_train.error(f"{subject_id} +Sess{sess_num:02d} FAILED: {e}")
                    continue

            model_results[subject_id] = subject_results

            # Incremental cache save after each subject (survives interruption)
            all_results[model_type] = model_results
            save_extra_sessions_cache(
                output_dir, args.paradigm, args.task, run_tag,
                all_results, baseline_run_tags, test_strategy,
            )

        all_results[model_type] = model_results

    # ====== Summary ======
    print()
    log_main.info("=" * 60)
    log_main.info(f"EXPERIMENT SUMMARY (strategy={test_strategy})")
    log_main.info("=" * 60)

    for model_type in args.models:
        model_data = all_results.get(model_type, {})
        if not model_data:
            continue

        print(f"\n  {model_type.upper()}:")
        for subject_id in subject_list:
            subj_data = model_data.get(subject_id, {})
            baseline_acc = subj_data.get('baseline', {}).get('test_acc_majority', 0)
            parts = [f"baseline={baseline_acc:.2%}"]
            for sess_num in subjects_with_sessions.get(subject_id, []):
                step = subj_data.get(f'sess{sess_num:02d}', {})
                acc = step.get('test_acc_majority', 0)
                delta = acc - baseline_acc if baseline_acc else 0
                parts.append(f"+S{sess_num:02d}={acc:.2%}({delta:+.1%})")
            print(f"    {subject_id}: {' | '.join(parts)}")

    # ====== Plot ======
    if not args.no_plot:
        try:
            from src.visualization.extra_sessions import generate_extra_sessions_combined_plot

            plot_filename = f'{run_tag}_extra_sessions_{args.paradigm}_{args.task}.png'
            plot_path = Path(output_dir) / plot_filename

            generate_extra_sessions_combined_plot(
                all_results=all_results,
                subjects_with_sessions=subjects_with_sessions,
                output_path=str(plot_path),
                paradigm=args.paradigm,
                task=args.task,
            )
            log_io.info(f"Plot saved: {plot_path}")
        except Exception as e:
            log_io.warning(f"Plot generation failed: {e}")
            import traceback
            traceback.print_exc()

    # ====== Finalize DB ======
    n_subjects = len(subject_list)
    finalize_db_run(db, db_run_id, comparison=None, n_subjects=n_subjects)

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
