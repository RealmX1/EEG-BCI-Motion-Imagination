# Experiment Scripts Unification Design

## Problem

The three experiment paradigms (within-subject, cross-subject, transfer learning) follow inconsistent organizational patterns:

- **Within-subject**: Clean 2-tier design — `run_single_model.py` contains all logic, `run_within_subject_comparison.py` is a thin orchestrator calling it twice.
- **Cross-subject**: `run_cross_subject_comparison.py` does NOT call `run_cross_subject.py` — it calls `train_cross_subject()` directly.
- **Transfer learning**: `run_transfer_comparison.py` is a 987-line monolith containing helper functions, training loops, DB writes, and plotting. No corresponding single-model script exists. `run_finetune.py` exists separately but is unused by any other code.

Additionally, `finetune_subject()` and `train_single_subject()` in `src/training/` are ~70% duplicated — both use the same `WithinSubjectTrainer`, the same data loading, the same temporal split, and the same evaluation. The only differences are model initialization (pretrained checkpoint vs. scratch) and freeze strategy.

## Goals

1. **Unify `src/training/`**: Merge `finetune_subject()` into `train_single_subject()` with optional `pretrained_path` / `freeze_strategy` parameters.
2. **Unify script paradigm**: All three paradigms follow the same pattern — single-model script + thin comparison orchestrator.
3. **Reduce redundancy**: Extract shared code (per-subject training loop, argparse groups, cache/resume, DB init) into `_training_utils.py`.
4. **Deprecate `run_finetune.py`**: Its functionality is absorbed into `run_single_model.py --pretrained`.

## Non-Goals

- Refactoring `src/training/train_cross_subject.py` internals (cross-subject trains all subjects in a single model, fundamentally different loop).
- Changing the `ExperimentDB` schema or cache format.
- Adding new features beyond unification.

## Design

### Layer 1: `src/training/train_within_subject.py` — Unified Training

`train_single_subject()` gains optional transfer learning parameters:

```python
def train_single_subject(
    subject_id, config, data_root, elc_path, save_dir, device,
    model_type, paradigm,
    # Existing params unchanged...
    cbramod_channels, preprocess_config, config_overrides,
    cache_only, cache_index_path,
    no_wandb, upload_model, wandb_project, wandb_entity, wandb_group,
    verbose,
    # ---- New: Transfer Learning ----
    pretrained_path: Optional[str] = None,
    freeze_strategy: Optional[str] = None,  # 'none'/'backbone'/'partial'
) -> Dict:
```

When `pretrained_path` is provided:
1. Load checkpoint via `load_pretrained_model()` instead of creating a fresh model.
2. Validate `n_classes` matches current task.
3. Apply `freeze_strategy` via `apply_freeze_strategy()`.
4. Create a finetuning-aware optimizer via `get_finetune_optimizer()` and replace `trainer.optimizer`.
5. Evaluate pretrained baseline at epoch 0 (before training starts).
6. Use finetune-specific default hyperparameters (epochs, LR) based on freeze strategy and channel count.
7. Include `pretrained_baseline` and `milestone_test_results` in the returned dict.

When `pretrained_path` is None (default): existing behavior, no changes.

`train_subject_simple()` passes through the new parameters:

```python
def train_subject_simple(
    ...,
    pretrained_path=None,
    freeze_strategy=None,
) -> Dict:
    return train_single_subject(..., pretrained_path=pretrained_path, freeze_strategy=freeze_strategy)
```

`finetune_subject()` becomes a thin backward-compatible wrapper:

```python
def finetune_subject(pretrained_path, subject_id, freeze_strategy='none', ...) -> Dict:
    """Backward-compatible wrapper. Delegates to train_subject_simple()."""
    return train_subject_simple(
        subject_id=subject_id,
        model_type=_detect_model_type(pretrained_path),
        pretrained_path=pretrained_path,
        freeze_strategy=freeze_strategy,
        ...
    )
```

Functions to relocate from `src/training/finetune.py` into `train_within_subject.py`:
- `load_pretrained_model()` — checkpoint loading + model reconstruction
- `apply_freeze_strategy()` — parameter freezing
- `get_finetune_optimizer()` — finetuning-aware optimizer creation
- `get_default_finetune_config()` — finetune hyperparameter defaults (extracted from hardcoded values)
- Finetune-specific constants (`EIGHT_CHANNEL_FINETUNE_OVERRIDES`, etc.)

`finetune_all_subjects()` is only used by `run_finetune.py` (being deprecated) and can be removed.

### Layer 2: `scripts/_training_utils.py` — Shared Abstractions

#### 2.1 Argparse Builders

```python
def add_common_args(parser):
    """Shared: --data-root, --paradigm, --task, --seed, --output-dir, --no-plot"""

def add_cache_resume_args(parser):
    """Shared: --resume, --force-retrain, --cache-only, --cache-index-path"""

def add_channel_args(parser):
    """Shared: --channels, --channel-config"""

def add_training_config_args(parser):
    """Shared: --config (YAML), --scheduler, --classifier-type, --no-pretrained"""

def add_transfer_args(parser):
    """Transfer-specific: --pretrained, --freeze-strategy, --finetune-epochs, --finetune-lr, --auto-discover-pretrained"""
```

#### 2.2 Per-Subject Training Orchestrator

Extracts the per-subject training loop shared between `run_single_model()` and `run_transfer_model()`:

```python
def run_model_on_subjects(
    model_type: str,
    subject_ids: List[str],
    train_fn: Callable[[str, ...], TrainingResult],
    train_kwargs: Dict,
    # Cache
    output_dir: str,
    paradigm: str,
    task: str,
    run_tag: str,
    cache_type: CacheType = CacheType.WITHIN,
    force_retrain: bool = False,
    extra_cache_metadata: Optional[Dict] = None,
    # DB
    db: Optional[ExperimentDB] = None,
    db_run_id: Optional[str] = None,
    # WandB
    wandb_group: Optional[str] = None,
    no_wandb: bool = True,
    # Display
    verbose_first_only: bool = True,
) -> Tuple[List[TrainingResult], Dict]:
    """
    Orchestrate training across subjects with caching, DB writes, and WandB.

    1. Load existing cache → determine which subjects need training
    2. For each subject:
       - Check cache → skip if exists
       - Call train_fn(subject_id, **train_kwargs) → TrainingResult
       - Progressive save to cache
       - DB write (if db provided)
       - WandB log
    3. Compute model statistics
    4. Return (results, stats)
    """
```

#### 2.3 Common Utilities

```python
def resolve_run_tag(args, paradigm, task, output_dir, cache_type=None) -> str:
    """Handle --resume logic: find existing tag or generate new one."""

def init_db_run(run_tag, experiment_type, paradigm, task, args) -> Tuple[ExperimentDB, Optional[str]]:
    """Create or resume an ExperimentDB run. Returns (db, db_run_id)."""

def finalize_db_run(db, db_run_id, comparison, **extra):
    """Save comparison, mark complete, close DB."""

def resolve_output_dir(args) -> str:
    """Auto-redirect to results/{n}_channel/{config}/ for reduced channel mode."""

def find_best_checkpoint_path(model_type, paradigm, task, subjects, results_dir, n_channels=None) -> Optional[str]:
    """Auto-discover best cross-subject pretrained checkpoint. Relocated from run_transfer_comparison.py."""

def validate_checkpoint_compatibility(pretrained_paths, task) -> Dict[str, str]:
    """Validate n_classes match and extract classifier_types. Relocated from run_transfer_comparison.py."""
```

### Layer 3: Script Restructuring

#### `run_single_model.py` (~250 lines, down from 602)

```python
def run_single_model(
    model_type, subject_ids,
    # Common
    data_root, task, paradigm, output_dir,
    # Cache/resume
    force_retrain, run_tag, cache_type=CacheType.WITHIN,
    # Transfer (optional)
    pretrained_path=None, freeze_strategy=None,
    # DB injection
    db=None, db_run_id=None,
    # Config
    config_overrides=None,
    # WandB, cache-only, etc.
    **kwargs,
) -> Tuple[List[TrainingResult], Dict]:
    """Single model training on all subjects. Supports both within-subject and transfer."""

    def train_fn(subject_id, verbose):
        return train_and_get_result(
            subject_id, model_type, task, paradigm, data_root,
            pretrained_path=pretrained_path,
            freeze_strategy=freeze_strategy,
            config_overrides=config_overrides,
            verbose=verbose,
            **kwargs,
        )

    return run_model_on_subjects(
        model_type=model_type,
        subject_ids=subject_ids,
        train_fn=train_fn,
        output_dir=output_dir,
        paradigm=paradigm,
        task=task,
        run_tag=run_tag,
        cache_type=cache_type,
        force_retrain=force_retrain,
        db=db, db_run_id=db_run_id,
    )

def main():
    parser = argparse.ArgumentParser(...)
    add_common_args(parser)
    add_cache_resume_args(parser)
    add_channel_args(parser)
    add_training_config_args(parser)
    add_transfer_args(parser)
    add_wandb_args(parser)
    args = parser.parse_args()

    output_dir = resolve_output_dir(args)
    run_tag = resolve_run_tag(args, ...)
    config_overrides = build_config_overrides(args)

    results, stats = run_single_model(
        model_type=args.model,
        subject_ids=subjects,
        pretrained_path=args.pretrained,
        freeze_strategy=args.freeze_strategy,
        ...
    )

    # Plot (single model)
    generate_single_model_plot(results, ...)
```

#### `run_within_subject_comparison.py` (~200 lines, down from 632)

```python
def main():
    parser = argparse.ArgumentParser(...)
    add_common_args(parser)
    add_cache_resume_args(parser)
    add_channel_args(parser)
    add_training_config_args(parser)
    add_wandb_args(parser)
    parser.add_argument('--models', ...)
    parser.add_argument('--skip-training', ...)
    args = parser.parse_args()

    output_dir = resolve_output_dir(args)
    run_tag = resolve_run_tag(args, ...)
    db, db_run_id = init_db_run(run_tag, 'within_subject', ...)
    config_overrides = build_config_overrides(args)

    # Train each model
    results = {}
    for model_type in args.models:
        model_results, stats = run_single_model(
            model_type=model_type,
            db=db, db_run_id=db_run_id,
            config_overrides=config_overrides,
            ...
        )
        results[model_type] = model_results
        db.save_summary(db_run_id, model_type, stats)

    # Compare + plot
    comparison = compare_models(results.get('eegnet'), results.get('cbramod'))
    finalize_db_run(db, db_run_id, comparison)
    save_cache(..., is_complete=True)
    generate_combined_plot(...)
```

#### `run_transfer_comparison.py` (~250 lines, down from 987)

```python
def main():
    parser = argparse.ArgumentParser(...)
    add_common_args(parser)
    add_cache_resume_args(parser)
    add_channel_args(parser)
    add_transfer_args(parser)
    add_wandb_args(parser)
    parser.add_argument('--models', ...)
    parser.add_argument('--pretrained-eegnet', ...)
    parser.add_argument('--pretrained-cbramod', ...)
    parser.add_argument('--no-cross-subject-baseline', ...)
    args = parser.parse_args()

    output_dir = resolve_output_dir(args)
    run_tag = resolve_run_tag(args, ..., cache_type=CacheType.TRANSFER)
    db, db_run_id = init_db_run(run_tag, 'transfer', ...)

    # Discover/validate checkpoints
    pretrained_paths = discover_or_validate_checkpoints(args, subjects)
    classifier_types = validate_checkpoint_compatibility(pretrained_paths, args.task)

    # Train each model (transfer mode)
    results = {}
    for model_type in args.models:
        if model_type not in pretrained_paths:
            continue
        model_results, stats = run_single_model(
            model_type=model_type,
            pretrained_path=pretrained_paths[model_type],
            freeze_strategy=args.freeze_strategy,
            cache_type=CacheType.TRANSFER,
            db=db, db_run_id=db_run_id,
            ...
        )
        results[model_type] = model_results
        db.save_summary(db_run_id, model_type, stats)

    # Compare + plot with baselines
    comparison = compare_models(results.get('eegnet'), results.get('cbramod'))
    finalize_db_run(db, db_run_id, comparison, transfer_config=...)
    save_cache(..., cache_type=CacheType.TRANSFER, is_complete=True)
    generate_transfer_plot(results, db, ...)  # 6-way with baselines
```

#### `run_cross_subject_comparison.py` (~250 lines, down from 647)

Uses same shared utilities (`resolve_run_tag`, `init_db_run`, `finalize_db_run`, etc.) but keeps its own training logic since cross-subject training is fundamentally different (single model across all subjects, not per-subject loop). Refactored to use shared argparse builders and DB utilities.

### Deprecation

| File | Action |
|------|--------|
| `scripts/experiments/run_finetune.py` | Delete. Functionality absorbed into `run_single_model.py --pretrained`. |
| `scripts/run_finetune.py` | Delete (thin wrapper for above). |
| `src/training/finetune.py` | Retained as backward-compat wrapper. Core logic moved to `train_within_subject.py`. `finetune_all_subjects()` removed. |

### Migration Safety

- `finetune_subject()` remains importable with identical signature — `src/hpo/objectives.py` and `src/training/__init__.py` continue to work.
- All existing CLI commands continue to work (top-level thin wrappers unchanged except `run_finetune.py`).
- Cache format unchanged — existing cached results remain valid.
- DB schema unchanged.

### Metrics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Experiment scripts total LOC | ~3,500 | ~1,200 | -65% |
| `_training_utils.py` | 233 | ~400 | +170 (absorbs shared logic) |
| `src/training/finetune.py` | 840 | ~50 (wrapper) | -94% |
| `src/training/train_within_subject.py` | 1,053 | ~1,150 | +~100 (absorbs finetune logic) |
| Net total | ~5,600 | ~2,800 | -50% |
| Cross-script duplication | ~1,500 lines | ~0 | Eliminated |
