# Experiment Scripts Unification Design

## Problem

The three experiment paradigms (within-subject, cross-subject, transfer learning) follow inconsistent organizational patterns:

- **Within-subject**: Clean 2-tier design — `run_single_model.py` contains all logic, `run_within_subject_comparison.py` is a thin orchestrator calling it twice.
- **Cross-subject**: `run_cross_subject_comparison.py` does NOT call `run_cross_subject.py` — it calls `train_cross_subject()` directly.
- **Transfer learning**: `run_transfer_comparison.py` is a 987-line monolith containing helper functions, training loops, DB writes, and plotting. No corresponding single-model script exists. `run_finetune.py` exists separately but is unused by any other code.

Additionally, `finetune_subject()` and `train_single_subject()` in `src/training/` are ~70% duplicated — both use the same `WithinSubjectTrainer`, the same data loading, the same temporal split, and the same evaluation. The differences are:

1. **Model initialization**: pretrained checkpoint vs. scratch
2. **Freeze strategy**: `apply_freeze_strategy()` + custom optimizer
3. **Training phases**: `train_single_subject()` uses two-phase training (exploration + main loaders); `finetune_subject()` happens to use single-phase (but this is not intentional — it simply was not implemented with exploration phase support)
4. **Pretrained baseline**: `finetune_subject()` evaluates the pretrained model at epoch 0 and uses it as the initial best (trainer only saves if finetuning improves over pretrained)
5. **Config overrides**: `finetune_subject()` does not support `config_overrides`; hyperparameters are passed as explicit arguments (`epochs`, `learning_rate`, `batch_size`)
6. **Scheduler**: `finetune_subject()` hardcodes `scheduler_type='plateau'` for EEGNet and `'cosine_annealing_warmup_decay'` for CBraMod, without consulting `SCHEDULER_PRESETS`

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

When `pretrained_path` is provided (transfer/finetune mode):
1. Load checkpoint via `load_pretrained_model()` instead of creating a fresh model.
2. Validate `n_classes` matches current task.
3. Apply `freeze_strategy` via `apply_freeze_strategy()`.
4. Create a finetuning-aware optimizer via `get_finetune_optimizer()` and replace `trainer.optimizer`.
5. Evaluate pretrained baseline at epoch 0 (before training starts). Set `trainer.best_val_acc`, `trainer.best_combined_score`, `trainer.best_epoch = 0` so that the trainer only saves a new checkpoint if finetuning actually improves over the pretrained model.
6. Apply finetune-specific default hyperparameters when not overridden by `config_overrides`:
   - Scheduler: `'plateau'` for EEGNet, `'cosine_annealing_warmup_decay'` for CBraMod
   - Epochs/LR/batch_size: based on freeze strategy and channel count (from `get_default_finetune_config()`)
7. **Support `config_overrides` in finetune mode** (new behavior vs. current `finetune_subject()`). This enables YAML config and `--scheduler` to work for transfer experiments.
8. Include `pretrained_baseline` and `milestone_test_results` in the returned dict.

**Two-phase training (exploration phase)**: Controlled by `scheduler_config` as usual — not skipped for finetune mode. The current `finetune_subject()` happens to not use it, but the unified function makes it available via config. The finetune default scheduler config sets `exploration_epochs` per its own defaults; if 0, the trainer naturally degrades to single-phase.

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
def finetune_subject(pretrained_path, subject_id, freeze_strategy='none',
                     epochs=None, learning_rate=None, batch_size=None,
                     model_selection_strategy='combined', ema_decay=0.998, soup_top_k=3,
                     ...) -> Dict:
    """Backward-compatible wrapper. Delegates to train_subject_simple().

    Translates finetune-specific explicit parameters (epochs, learning_rate, batch_size)
    into config_overrides format used by the unified function.
    """
    # Detect model_type from checkpoint (loaded inside train_single_subject anyway)
    model_type = _detect_model_type_from_checkpoint(pretrained_path)

    # Translate explicit params → config_overrides
    config_overrides = {'training': {}}
    if epochs is not None:
        config_overrides['training']['epochs'] = epochs
    if learning_rate is not None:
        config_overrides['training']['learning_rate'] = learning_rate
    if batch_size is not None:
        config_overrides['training']['batch_size'] = batch_size
    if model_selection_strategy != 'combined':
        config_overrides['training']['model_selection_strategy'] = model_selection_strategy
    config_overrides['training']['ema_decay'] = ema_decay
    config_overrides['training']['soup_top_k'] = soup_top_k

    return train_subject_simple(
        subject_id=subject_id,
        model_type=model_type,
        pretrained_path=pretrained_path,
        freeze_strategy=freeze_strategy,
        config_overrides=config_overrides or None,
        ...
    )

def _detect_model_type_from_checkpoint(pretrained_path: str) -> str:
    """Load checkpoint metadata to determine model_type. Lightweight: only reads model_config."""
    import torch
    ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    return ckpt['model_config']['model_type']
```

Note: `_detect_model_type_from_checkpoint()` does load the full checkpoint, but this is acceptable because `train_single_subject()` will load it again via `load_pretrained_model()`. An optimization to avoid double-loading can be added later (pass preloaded checkpoint through), but is not required for correctness.

Functions to relocate from `src/training/finetune.py` into a new `src/training/finetune_utils.py`:
- `load_pretrained_model()` — checkpoint loading + model reconstruction
- `apply_freeze_strategy()` — parameter freezing
- `get_finetune_optimizer()` — finetuning-aware optimizer creation
- `get_default_finetune_config()` — finetune hyperparameter defaults (extracted from hardcoded values)
- Finetune-specific constants (`EIGHT_CHANNEL_FINETUNE_OVERRIDES`, etc.)

These stay in a separate file rather than being merged into `train_within_subject.py` to avoid bloating that already-large file (1,053 lines). `train_within_subject.py` imports from `finetune_utils.py` when `pretrained_path` is provided.

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

def add_model_selection_args(parser):
    """Shared: --model-selection-strategy, --ema-decay, --soup-top-k"""

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

Uses same shared utilities but keeps its own training logic since cross-subject trains a single model across all subjects (fundamentally different from per-subject loops).

Shared utilities adopted:
- `add_common_args()`, `add_cache_resume_args()`, `add_channel_args()`, `add_wandb_args()` — argparse builders
- `resolve_run_tag()` — resume/tag logic
- `init_db_run()` / `finalize_db_run()` — DB lifecycle
- `resolve_output_dir()` — channel directory redirect

Not shared (cross-subject specific):
- Training loop — calls `train_cross_subject()` directly (trains one model on all subjects combined, not per-subject)
- Cache structure — flat `{model: result_dict}` not `{model: {subject: result_dict}}`
- Result conversion — `cross_subject_result_to_training_results()` to convert for comparison

### Deprecation

| File | Action |
|------|--------|
| `scripts/experiments/run_finetune.py` | Delete. Functionality absorbed into `run_single_model.py --pretrained`. |
| `scripts/run_finetune.py` | Delete (thin wrapper for above). |
| `src/training/finetune.py` | Retained as backward-compat wrapper (~50 lines). Core logic moved to `src/training/finetune_utils.py` (reusable functions) and integrated into `train_within_subject.py` (unified flow). `finetune_all_subjects()` removed. |
| `src/training/finetune_utils.py` | New file. Contains `load_pretrained_model()`, `apply_freeze_strategy()`, `get_finetune_optimizer()`, `get_default_finetune_config()`, and finetune constants. Imported by `train_within_subject.py` when `pretrained_path` is provided. |

### Design Decisions

**WandB strategy**: Standardize on **per-subject WandB runs** for all paradigms (within-subject and transfer). The current transfer-specific "single shared WandB run across all subjects" pattern is dropped. Rationale: per-subject runs are simpler, consistent across paradigms, and the shared-run pattern has cleanup issues on failures. Per-subject runs are grouped via `wandb_group`.

**Channel config plumbing**: When `run_single_model()` is called in transfer mode with `--channels` / `--channel-config`, these flow through `config_overrides['data']['channels']` and `config_overrides['data']['channel_config']`. The unified `train_single_subject()` reads these from config and passes them to `preprocess_config.apply_channel_overrides()` (already supported at line 394-396 of the current code). No new plumbing needed.

**Two-phase training (exploration phase)**: Not conditionally skipped for finetune mode. The exploration phase is controlled by `scheduler_config['exploration_epochs']` — if the finetune default config sets it to 0, the trainer naturally uses single-phase. If config overrides specify exploration epochs, they work for finetune too. No special-casing needed in the unified function.

### Migration Safety

- `finetune_subject()` remains importable with identical signature — `src/hpo/objectives.py` and `src/training/__init__.py` continue to work.
- All existing CLI commands continue to work (top-level thin wrappers unchanged except `run_finetune.py`).
- Cache format unchanged — existing cached results remain valid.
- DB schema unchanged.

### Migration Checklist

- [ ] Update `src/training/__init__.py`: remove `finetune_all_subjects` from imports and `__all__`
- [ ] Update `src/training/__init__.py`: re-export `finetune_subject` from new location (or keep importing from `finetune.py` wrapper)
- [ ] Update `docs/codebase_reference.md`: remove `run_finetune.py` references, add `--pretrained` docs
- [ ] Verify `src/hpo/objectives.py` works with the backward-compat wrapper (same args, same return format)
- [ ] Delete `scripts/experiments/run_finetune.py` and `scripts/run_finetune.py`

### Verification

- **Backward compatibility**: `finetune_subject()` called from `src/hpo/objectives.py` with the same arguments produces identical behavior. Verify by running a single-subject HPO trial before and after.
- **Cache compatibility**: Existing JSON caches from transfer runs can still be loaded by the new `run_transfer_comparison.py --resume`.
- **Smoke test**: `run_single_model.py --model cbramod --pretrained <path> --freeze-strategy backbone --subjects S01` produces equivalent results to the old `run_transfer_comparison.py` for a single model/subject.

### Metrics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Experiment scripts total LOC | ~3,500 | ~1,200 | -65% |
| `_training_utils.py` | 233 | ~400 | +170 (absorbs shared logic) |
| `src/training/finetune.py` | 840 | ~50 (wrapper) | -94% |
| `src/training/train_within_subject.py` | 1,053 | ~1,150 | +~100 (absorbs finetune logic) |
| Net total | ~5,600 | ~2,800 | -50% |
| Cross-script duplication | ~1,500 lines | ~0 | Eliminated |
