"""Shared helpers for experiment-oriented CLI scripts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.preprocessing.data_loader import PreprocessConfig, discover_available_subjects
from src.results.cache import (
    SelectionStrategy,
    build_data_sources_from_historical,
    find_cache_by_tag,
    find_compatible_cross_subject_results,
    find_compatible_historical_results,
    find_latest_cache,
    get_cache_path,
    load_cache,
    prepare_combined_plot_data,
    save_cache,
)
from src.results.dataclasses import PlotDataSource, TrainingResult
from src.results.experiment_db import ExperimentDB
from src.results.serialization import dict_to_result, generate_result_filename, result_to_dict
from src.results.statistics import compute_model_statistics, print_model_summary
from src.training.train_within_subject import train_subject_simple
from src.utils.logging import SectionLogger, setup_logging
from src.visualization.comparison import generate_combined_plot

from src.config.constants import MODEL_COLORS, PARADIGM_CONFIG


setup_logging("training")
logger = logging.getLogger(__name__)
log_cache = SectionLogger(logger, "cache")
log_train = SectionLogger(logger, "train")
log_io = SectionLogger(logger, "io")


def discover_subjects(
    data_root: str,
    paradigm: str = "imagery",
    task: str = "binary",
    cache_only: bool = False,
) -> List[str]:
    """Discover all available subjects."""
    if cache_only:
        from src.preprocessing.data_loader import discover_subjects_from_cache_index

        return discover_subjects_from_cache_index(paradigm, task)
    return discover_available_subjects(data_root, paradigm, task)


def print_subject_result(subject_id: str, model_type: str, result: TrainingResult) -> None:
    """Print a compact completion summary for one subject."""
    print("\n" + "=" * 60)
    print(f" {model_type.upper()} - {subject_id} COMPLETE")
    print("=" * 60)
    print(f"  Validation Accuracy:  {result.best_val_acc:.2%}")
    print(
        "  Test Accuracy:        "
        f"{result.test_acc_majority:.2%} (majority voting, Sess2 Finetune)"
    )
    print(f"  Epochs Trained:       {result.epochs_trained}")
    print(f"  Training Time:        {result.training_time:.1f}s")
    print("=" * 60 + "\n")


def train_and_get_result(
    subject_id: str,
    model_type: str,
    task: str,
    paradigm: str,
    data_root: str,
    save_dir: str,
    run_tag: Optional[str] = None,
    no_wandb: bool = False,
    upload_model: bool = False,
    wandb_group: Optional[str] = None,
    wandb_project: str = "eeg-bci",
    wandb_entity: Optional[str] = None,
    preprocess_config: Optional[PreprocessConfig] = None,
    cache_only: bool = False,
    config_overrides: Optional[Dict] = None,
    verbose: int = 2,
    pretrained_path: Optional[str] = None,
    freeze_strategy: Optional[str] = None,
    session_folders_override: Optional[Dict] = None,
    precomputed_data: Optional[Dict] = None,
) -> TrainingResult:
    """Train a single subject/model pair and return a lightweight TrainingResult."""
    result_dict = train_subject_simple(
        subject_id=subject_id,
        model_type=model_type,
        task=task,
        paradigm=paradigm,
        data_root=data_root,
        save_dir=save_dir,
        run_tag=run_tag,
        no_wandb=no_wandb,
        upload_model=upload_model,
        wandb_group=wandb_group,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        preprocess_config=preprocess_config,
        cache_only=cache_only,
        config_overrides=config_overrides,
        verbose=verbose,
        pretrained_path=pretrained_path,
        freeze_strategy=freeze_strategy,
        session_folders_override=session_folders_override,
        precomputed_data=precomputed_data,
    )

    if not result_dict:
        raise ValueError(f"Training failed for {subject_id}")

    subtask_results = result_dict.get("subtask_results")
    if subtask_results is not None:
        subtask_results = {
            key: (
                {"accuracy": value["accuracy"], "n_trials": value.get("n_trials", 0)}
                if isinstance(value, dict)
                else value
            )
            for key, value in subtask_results.items()
            if key in ("binary", "ternary", "quaternary", "mean_accuracy")
        }

    return TrainingResult(
        subject_id=subject_id,
        task_type=task,
        model_type=model_type,
        best_val_acc=result_dict.get("best_val_acc", result_dict.get("val_accuracy", 0.0)),
        test_acc=result_dict.get("test_accuracy", 0.0),
        test_acc_majority=result_dict.get(
            "test_accuracy_majority",
            result_dict.get("test_accuracy", 0.0),
        ),
        epochs_trained=result_dict.get("epochs_trained", result_dict.get("best_epoch", 0)),
        training_time=result_dict.get("training_time", 0.0),
        subtask_results=subtask_results,
    )


def add_wandb_args(parser) -> None:
    """Add standardized WandB CLI arguments to an argparse parser."""
    group = parser.add_argument_group("WandB")
    group.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    group.add_argument(
        "--upload-model",
        action="store_true",
        help="Upload model artifacts (.pt) to WandB",
    )
    group.add_argument(
        "--wandb-project",
        type=str,
        default="eeg-bci",
        help="WandB project name (default: eeg-bci)",
    )
    group.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB entity (team/username)",
    )


def add_common_args(parser) -> None:
    """Add shared experiment arguments."""
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Path to data directory (default: data)",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Specific subjects to run (default: all available)",
    )
    parser.add_argument(
        "--paradigm",
        type=str,
        default="imagery",
        choices=["imagery", "movement"],
        help="Experiment paradigm (default: imagery)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="binary",
        choices=["binary", "ternary", "quaternary", "unified"],
        help="Classification task (default: binary)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results (default: results)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Suppress plot generation")

    from src.config.constants import PURPOSE_VALUES

    parser.add_argument(
        "--purpose",
        type=str,
        default=None,
        choices=sorted(PURPOSE_VALUES),
        help=(
            "Run intent / hypothesis being tested (controlled vocab). "
            "Encodes WHY the run is launched (e.g., 'ablation', 'debug'), "
            "NOT post-hoc analysis or outcome — put results in dev_log instead. "
            "See PURPOSE_VALUES in src/config/constants.py"
        ),
    )
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help=(
            "Free-form notes — typically the hypothesis text itself or a "
            "one-line description of intent. No analysis/results content "
            "(those belong in dev_log)."
        ),
    )


def add_cache_resume_args(parser) -> None:
    """Add shared cache and resume arguments."""
    parser.add_argument(
        "--resume",
        nargs="?",
        const="",
        default=None,
        metavar="TAG",
        help="Resume a previous run. Without TAG: most recent. With TAG: matching run",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retraining, ignore cache",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training, load existing results",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Load data from cache index only (no filesystem scan)",
    )


def add_channel_args(parser) -> None:
    """Add shared reduced-channel arguments."""
    from src.config.constants import FULL_N_CHANNELS, SUPPORTED_CHANNEL_COUNTS

    parser.add_argument(
        "--channels",
        type=int,
        default=FULL_N_CHANNELS,
        choices=SUPPORTED_CHANNEL_COUNTS,
        help=f"Number of EEG channels (default: {FULL_N_CHANNELS})",
    )
    parser.add_argument(
        "--channel-config",
        type=str,
        default="motor_cortex",
        help="Channel configuration name (default: motor_cortex)",
    )


def add_training_config_args(parser) -> None:
    """Add shared training configuration overrides."""
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="YAML_PATH",
        help="YAML config file path",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        choices=[
            "plateau",
            "cosine",
            "wsd",
            "cosine_decay",
            "cosine_annealing_warmup_decay",
        ],
        help="Learning rate scheduler (default: model-specific)",
    )
    parser.add_argument(
        "--classifier-type",
        type=str,
        default=None,
        choices=["two_layer", "three_layer", "one_layer", "attention_pool"],
        help="Override CBraMod classifier head type",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train CBraMod from scratch (no pretrained weights)",
    )


def add_transfer_args(parser) -> None:
    """Add transfer-learning arguments."""
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained checkpoint for transfer learning",
    )
    parser.add_argument(
        "--freeze-strategy",
        type=str,
        default="none",
        choices=["none", "backbone", "partial"],
        help="Freeze strategy for fine-tuning (default: none)",
    )


def resolve_output_dir(args) -> str:
    """Auto-redirect reduced-channel experiments into nested results directories."""
    from src.config.constants import FULL_N_CHANNELS

    output_dir = getattr(args, "output_dir", None) or getattr(args, "results_dir", "results")
    channels = getattr(args, "channels", FULL_N_CHANNELS)
    channel_config = getattr(args, "channel_config", "motor_cortex")
    if channels != FULL_N_CHANNELS and output_dir == "results":
        return f"results/{channels}_channel/{channel_config}"
    return output_dir


def resolve_run_tag(args, paradigm, task, output_dir, cache_type=None) -> str:
    """Handle ``--resume`` logic or create a fresh timestamp run tag."""
    import sys
    from datetime import datetime

    if getattr(args, "resume", None) is not None:
        tag_hint = args.resume if args.resume != "" else None
        found = find_cache_by_tag(
            output_dir,
            paradigm,
            task,
            tag_substring=tag_hint,
            cache_type=cache_type,
        )
        if found:
            _, run_tag = found
            log_cache.info(f"Resuming run: {run_tag}")
            return run_tag
        log_cache.error("No previous run found to resume")
        sys.exit(1)

    run_tag = datetime.now().strftime("%Y%m%d_%H%M")
    log_cache.info(f"Starting new run: {run_tag}")
    return run_tag


def init_db_run(run_tag, experiment_type, paradigm, task, args):
    """Create or resume an ExperimentDB run. Returns ``(db, db_run_id)``."""
    import shlex
    import sqlite3
    import sys

    from src.config.constants import FULL_N_CHANNELS

    db = ExperimentDB()
    db_run_id = None
    channels = getattr(args, "channels", FULL_N_CHANNELS)
    channel_config = getattr(args, "channel_config", "motor_cortex")
    is_baseline = getattr(args, "baseline", False)

    try:
        db_run_id = db.create_run(
            run_tag=run_tag,
            experiment_type=experiment_type,
            paradigm=paradigm,
            task=task,
            n_channels=channels,
            channel_config=channel_config if channels != FULL_N_CHANNELS else None,
            command=" ".join(shlex.quote(arg) for arg in sys.argv),
            is_baseline=is_baseline,
            purpose=getattr(args, "purpose", None),
            notes=getattr(args, "notes", None),
        )
        log_train.info(f"DB run created: {db_run_id}")
    except sqlite3.IntegrityError:
        existing = db.find_run_by_tag(run_tag, paradigm, task, experiment_type=experiment_type)
        if existing:
            db_run_id = existing["run_id"]
            log_train.info(f"DB run resumed: {db_run_id}")
            if is_baseline and not existing.get("is_baseline"):
                try:
                    db.set_baseline(db_run_id)
                except Exception:
                    pass
        else:
            log_train.warning("DB run creation failed: duplicate but tag not found")
    except Exception as exc:
        log_train.warning(f"DB run creation failed: {exc}")

    return db, db_run_id


def finalize_db_run(db, db_run_id, comparison, n_subjects, **extra) -> None:
    """Persist summary data, mark the run complete, and close the DB handle."""
    if db_run_id:
        try:
            if comparison:
                db.save_comparison(db_run_id, comparison)
            db.update_n_subjects(db_run_id, n_subjects)

            transfer_config = extra.get("transfer_config")
            if transfer_config:
                db.save_transfer_config(db_run_id, **transfer_config)

            db.mark_complete(db_run_id)
        except Exception as exc:
            log_train.warning(f"DB finalize failed: {exc}")
    db.close()


def build_config_overrides(args) -> Optional[Dict]:
    """Build merged ``config_overrides`` from YAML and CLI flags."""
    from src.config.constants import FULL_N_CHANNELS
    from src.config.training import load_yaml_config

    config_overrides = load_yaml_config(args.config) if getattr(args, "config", None) else {}

    if getattr(args, "scheduler", None):
        config_overrides.setdefault("training", {})["scheduler"] = args.scheduler
    if getattr(args, "no_pretrained", False):
        config_overrides.setdefault("model", {})["no_pretrained"] = True

    channels = getattr(args, "channels", FULL_N_CHANNELS)
    if channels != FULL_N_CHANNELS:
        config_overrides.setdefault("data", {})["channels"] = channels
        config_overrides.setdefault("data", {})["channel_config"] = getattr(
            args,
            "channel_config",
            "motor_cortex",
        )

    if getattr(args, "classifier_type", None):
        config_overrides.setdefault("model", {})["classifier_type"] = args.classifier_type

    return config_overrides or None


def find_best_checkpoint_path(
    model_type,
    paradigm,
    task,
    subjects,
    results_dir="results",
    n_channels=None,
):
    """Auto-discover the best compatible cross-subject pretrained checkpoint."""
    import json

    import torch

    cross_result = find_compatible_cross_subject_results(
        output_dir=results_dir,
        paradigm=paradigm,
        task=task,
        subjects=subjects,
        model_type=model_type,
        n_channels=n_channels,
    )
    if not cross_result:
        return None

    source_file = cross_result["source_file"]
    try:
        with open(source_file, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        model_path = data.get("training_info", {}).get("model_path", "")
        if model_path and Path(model_path).exists():
            log_io.info(f"Found {model_type} checkpoint: {model_path}")
            return model_path
    except (json.JSONDecodeError, OSError):
        pass

    checkpoint_dir = Path("checkpoints/cross_subject")
    if checkpoint_dir.exists():
        for subdir in sorted(checkpoint_dir.iterdir(), reverse=True):
            if not (
                subdir.is_dir()
                and model_type in subdir.name
                and paradigm in subdir.name
                and task in subdir.name
            ):
                continue

            best_pt = subdir / "best.pt"
            if not best_pt.exists():
                continue

            if n_channels is not None:
                try:
                    ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
                    ckpt_channels = ckpt.get("model_config", {}).get("n_channels")
                    if ckpt_channels is not None and ckpt_channels != n_channels:
                        continue
                except Exception:
                    continue

            log_io.info(f"Found {model_type} checkpoint (fallback): {best_pt}")
            return str(best_pt)

    return None


def validate_checkpoint_compatibility(pretrained_paths, task):
    """Validate n_classes compatibility and extract classifier types."""
    import sys

    import torch

    from src.config.constants import TASKS

    classifier_types = {}
    expected_n_classes = TASKS[task]["n_classes"]

    for model_type, path in pretrained_paths.items():
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            classifier_types[model_type] = ckpt.get("model_config", {}).get(
                "classifier_type",
                "two_layer",
            )
            ckpt_n_classes = ckpt.get("model_config", {}).get("n_classes")
            ckpt_task = ckpt.get("training_config", {}).get("task", "unknown")
            if ckpt_n_classes is not None and ckpt_n_classes != expected_n_classes:
                log_train.error(
                    f"Checkpoint/task mismatch for {model_type.upper()}: "
                    f"pretrained n_classes={ckpt_n_classes} (task='{ckpt_task}'), "
                    f"but current task '{task}' expects n_classes={expected_n_classes}. "
                    f"Checkpoint: {path}"
                )
                sys.exit(1)
        except Exception:
            classifier_types[model_type] = "unknown"

    return classifier_types


def add_replot_arg(parser) -> None:
    """Add ``--replot`` to comparison scripts."""
    parser.add_argument(
        "--replot",
        type=str,
        default=None,
        metavar="RUN_TAG",
        help=(
            "Re-generate plots for a completed run (no training, no DB writes). "
            "Requires a run tag (e.g., 20260322_1116)."
        ),
    )


def load_replot_context(
    run_tag: str,
    experiment_type: str,
    results_dir_override: Optional[str] = None,
) -> Dict:
    """Load a completed run from ExperimentDB for plot regeneration."""
    from src.config.constants import FULL_N_CHANNELS

    db = ExperimentDB()
    run = db.find_run_by_tag(run_tag, experiment_type=experiment_type)
    if run is None:
        logger.error(
            f"Run '{run_tag}' not found in ExperimentDB "
            f"(experiment_type={experiment_type})"
        )
        db.close()
        raise SystemExit(1)

    run_id = run["run_id"]
    if not run["is_complete"]:
        logger.warning(f"Run '{run_tag}' is not marked complete — replotting anyway")

    results_by_model = db.get_results_by_model(run_id)
    if not results_by_model:
        logger.error(f"No subject results found for run '{run_tag}' (run_id={run_id})")
        db.close()
        raise SystemExit(1)

    models = sorted(results_by_model.keys())
    subjects = sorted(
        {
            result.subject_id
            for model_results in results_by_model.values()
            for result in model_results
        }
    )

    n_channels = run.get("n_channels", FULL_N_CHANNELS)
    channel_config = run.get("channel_config")

    if results_dir_override:
        results_dir = results_dir_override
    elif n_channels != FULL_N_CHANNELS and channel_config:
        results_dir = f"results/{n_channels}_channel/{channel_config}"
    else:
        results_dir = "results"

    logger.info(
        f"Replot context: run_tag={run_tag}, paradigm={run['paradigm']}, "
        f"task={run['task']}, models={models}, {len(subjects)} subjects, "
        f"results_dir={results_dir}"
    )

    return {
        "run_tag": run_tag,
        "run_id": run_id,
        "paradigm": run["paradigm"],
        "task": run["task"],
        "n_channels": n_channels,
        "channel_config": channel_config,
        "models": models,
        "subjects": subjects,
        "results_by_model": results_by_model,
        "results_dir": results_dir,
        "db": db,
    }
