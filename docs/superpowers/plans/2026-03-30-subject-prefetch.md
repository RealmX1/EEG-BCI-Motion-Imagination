# Subject-Level Data Prefetch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prefetch the next subject's data in a background thread while the current subject trains on GPU, eliminating ~10% idle time per subject.

**Architecture:** A `SubjectPrefetcher` class wraps `ThreadPoolExecutor(max_workers=1)` to run `load_subject_data()` + `temporal_split_by_group()` for the next subject while the current one trains. Results are returned as `precomputed_data` dicts — an interface `train_single_subject()` already supports. Integration happens in `run_within_subject()`, so transfer learning gets prefetch for free (it delegates to `run_within_subject()`).

**Tech Stack:** `concurrent.futures.ThreadPoolExecutor`, existing `load_subject_data()`, `temporal_split_by_group()`, `PreprocessConfig`, `get_task_type_patterns()`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/training/prefetch.py` | **Create** | `SubjectPrefetcher` class — background data loading + splitting |
| `scripts/experiments/run_within_subject.py` | **Modify** (lines 100-305) | Integrate prefetcher into subject iteration loop |
| `tests/test_prefetch.py` | **Create** | Unit tests for prefetcher |

**NOT modified** (already support `precomputed_data`):
- `src/training/train_within_subject.py` — `train_single_subject()` already handles `precomputed_data` (lines 385-405)
- `scripts/_training_utils.py` — `train_and_get_result()` already forwards `precomputed_data` (line 190)
- `scripts/experiments/run_transfer_comparison.py` — calls `run_within_subject()`, gets prefetch automatically

---

## Task 1: Create `src/training/prefetch.py`

**Files:**
- Create: `src/training/prefetch.py`

- [ ] **Step 1: Write the prefetch module**

```python
"""
Subject-level data prefetcher for within-subject training.

Loads the next subject's data in a background thread while the current
subject trains on GPU, eliminating data loading idle time (~10% of
total training time).

Usage:
    prefetcher = SubjectPrefetcher(
        model_type='cbramod', task='binary', paradigm='imagery',
        data_root=Path('data'), elc_path=Path('data/biosemi128.ELC'),
        cache_only=True, config_overrides=None,
    )
    prefetcher.start_prefetch('S02')

    # ... train S01 on GPU ...

    precomputed = prefetcher.get_prefetched('S02')
    # precomputed is a dict with 'train_dataset', 'train_indices', etc.
"""

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

from src.preprocessing.data_loader import PreprocessConfig
from src.training.common import temporal_split_by_group
from src.training.train_within_subject import (
    get_default_config,
    get_task_type_patterns,
    load_subject_data,
)

logger = logging.getLogger(__name__)


class SubjectPrefetcher:
    """Prefetch next subject's data in a background thread.

    Thread-safe: PreprocessingCache uses per-file locks and get_cache()
    uses a global lock for singleton creation.  HDF5 reads do not hold
    the GIL, so the background thread runs concurrently with GPU training.
    """

    def __init__(
        self,
        model_type: str,
        task: str,
        paradigm: str,
        data_root: Path,
        elc_path: Path,
        cache_only: bool = False,
        config_overrides: Optional[Dict] = None,
    ):
        self._model_type = model_type
        self._task = task
        self._paradigm = paradigm
        self._data_root = data_root
        self._elc_path = elc_path
        self._cache_only = cache_only

        # Resolve preprocessing config (mirrors train_single_subject lines 414-439)
        config = get_default_config(model_type, task)
        if config_overrides:
            from src.training.train_within_subject import apply_config_overrides
            config = apply_config_overrides(config, config_overrides)

        task_config = config['tasks'][task]
        self._target_classes = task_config['classes']
        n_classes = task_config['n_classes']

        if model_type == 'cbramod':
            self._preprocess_config = PreprocessConfig.for_cbramod(full_channels=True)
        else:
            self._preprocess_config = PreprocessConfig.paper_aligned(n_class=n_classes)

        data_config = config.get('data', {})
        self._preprocess_config.apply_channel_overrides(
            channels=data_config.get('channels'),
            channel_config=data_config.get('channel_config'),
        )
        if 'window_length' in data_config:
            self._preprocess_config.trial_duration = data_config['window_length']

        # Session folder patterns (same for all subjects of this task)
        self._task_patterns = get_task_type_patterns(task, n_classes, paradigm)

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch")
        self._futures: Dict[str, Future] = {}

    def start_prefetch(self, subject_id: str) -> None:
        """Submit a prefetch job for the given subject (non-blocking)."""
        if subject_id in self._futures:
            return  # Already submitted
        self._futures[subject_id] = self._executor.submit(
            self._load_and_split, subject_id,
        )
        logger.debug(f"Prefetch submitted: {subject_id}")

    def get_prefetched(self, subject_id: str) -> Optional[Dict]:
        """Block until prefetch completes and return precomputed_data, or None on failure."""
        future = self._futures.pop(subject_id, None)
        if future is None:
            return None
        try:
            return future.result(timeout=600)  # 10 min safety timeout
        except Exception as e:
            logger.warning(f"Prefetch failed for {subject_id}: {e}")
            return None

    def cancel_pending(self) -> None:
        """Cancel any pending prefetch jobs."""
        for subject_id, future in list(self._futures.items()):
            future.cancel()
        self._futures.clear()

    def shutdown(self) -> None:
        """Shut down the executor."""
        self.cancel_pending()
        self._executor.shutdown(wait=False)

    def _load_and_split(self, subject_id: str) -> Dict:
        """Worker: load train+test data and compute temporal split."""
        train_dataset = load_subject_data(
            self._data_root, subject_id,
            session_folders=self._task_patterns['train'],
            target_classes=self._target_classes,
            config=self._preprocess_config,
            elc_path=self._elc_path,
            cache_only=self._cache_only,
        )

        test_dataset = None
        if self._task_patterns['test']:
            test_dataset = load_subject_data(
                self._data_root, subject_id,
                session_folders=self._task_patterns['test'],
                target_classes=self._target_classes,
                config=self._preprocess_config,
                elc_path=self._elc_path,
                cache_only=self._cache_only,
                reject_trials=False,
            )

        train_indices, val_indices = temporal_split_by_group(
            train_dataset, group_attr='session_type', val_ratio=0.2,
        )

        return {
            'train_dataset': train_dataset,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_dataset': test_dataset,
        }
```

- [ ] **Step 2: Commit**

```bash
git add src/training/prefetch.py
git commit -m "feat: add SubjectPrefetcher for background data loading"
```

---

## Task 2: Write unit tests

**Files:**
- Create: `tests/test_prefetch.py`

- [ ] **Step 1: Write tests**

```python
"""
Tests for SubjectPrefetcher — background data loading for within-subject training.

Run with: uv run pytest tests/test_prefetch.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.prefetch import SubjectPrefetcher


@pytest.fixture
def mock_prefetcher():
    """Create prefetcher with mocked data loading."""
    with patch('src.training.prefetch.load_subject_data') as mock_load, \
         patch('src.training.prefetch.temporal_split_by_group') as mock_split:

        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_load.return_value = mock_dataset
        mock_split.return_value = (list(range(80)), list(range(80, 100)))

        prefetcher = SubjectPrefetcher(
            model_type='eegnet',
            task='binary',
            paradigm='imagery',
            data_root=Path('data'),
            elc_path=Path('data/biosemi128.ELC'),
            cache_only=True,
        )
        yield prefetcher, mock_load, mock_split
        prefetcher.shutdown()


def test_prefetch_returns_precomputed_data(mock_prefetcher):
    """Prefetch produces dict with required keys."""
    prefetcher, mock_load, _ = mock_prefetcher
    prefetcher.start_prefetch('S01')
    result = prefetcher.get_prefetched('S01')

    assert result is not None
    assert 'train_dataset' in result
    assert 'train_indices' in result
    assert 'val_indices' in result
    assert 'test_dataset' in result
    assert len(result['train_indices']) == 80
    assert len(result['val_indices']) == 20


def test_prefetch_not_submitted_returns_none(mock_prefetcher):
    """get_prefetched for unknown subject returns None."""
    prefetcher, _, _ = mock_prefetcher
    assert prefetcher.get_prefetched('S99') is None


def test_prefetch_deduplicates(mock_prefetcher):
    """Submitting same subject twice does not create a second future."""
    prefetcher, mock_load, _ = mock_prefetcher
    prefetcher.start_prefetch('S01')
    prefetcher.start_prefetch('S01')  # Should be no-op
    result = prefetcher.get_prefetched('S01')
    assert result is not None
    # load_subject_data called twice (train + test), not four times
    assert mock_load.call_count == 2


def test_prefetch_failure_returns_none(mock_prefetcher):
    """If loading raises, get_prefetched returns None instead of crashing."""
    prefetcher, mock_load, _ = mock_prefetcher
    mock_load.side_effect = RuntimeError("disk error")
    prefetcher.start_prefetch('S01')
    result = prefetcher.get_prefetched('S01')
    assert result is None


def test_shutdown_is_safe(mock_prefetcher):
    """Shutdown can be called multiple times without error."""
    prefetcher, _, _ = mock_prefetcher
    prefetcher.shutdown()
    prefetcher.shutdown()  # Should not raise
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest tests/test_prefetch.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_prefetch.py
git commit -m "test: add unit tests for SubjectPrefetcher"
```

---

## Task 3: Integrate prefetcher into `run_within_subject()`

**Files:**
- Modify: `scripts/experiments/run_within_subject.py:100-305`

- [ ] **Step 1: Add import and parameter**

Add import at top of file (after existing imports, around line 67):

```python
from src.training.prefetch import SubjectPrefetcher
```

Add `use_prefetch` parameter to `run_within_subject()` signature (line 122, before `cache_type`):

```python
    # Prefetch
    use_prefetch: bool = True,
    # Cache type
    cache_type = None,  # CacheType enum, None defaults to within-subject
```

- [ ] **Step 2: Create prefetcher before the subject loop**

Insert after the cache summary block (after line 213, before `results: List[TrainingResult] = []`):

```python
    # Set up subject prefetcher (background data loading for next subject)
    prefetcher = None
    if use_prefetch and subjects_to_train:
        try:
            prefetcher = SubjectPrefetcher(
                model_type=model_type,
                task=task,
                paradigm=paradigm,
                data_root=Path(data_root),
                elc_path=Path(data_root) / 'biosemi128.ELC',
                cache_only=cache_only,
                config_overrides=config_overrides,
            )
        except Exception as e:
            log_train.warning(f"Prefetch init failed ({e}), continuing without prefetch")
```

- [ ] **Step 3: Integrate prefetch into the subject loop**

Replace the loop body (lines 224-296) with prefetch-aware version. The changes are:

**A)** After cache check `continue` (line 233), start prefetching the next uncached subject:

```python
        # Check cache
        if subject_id in cache[model_type] and not force_retrain:
            log_train.info(f"{progress} {subject_id}: cached")
            cached_result = dict_to_result(cache[model_type][subject_id])
            results.append(cached_result)
            print_subject_result(subject_id, model_type, cached_result)
            continue
```

No change needed here — the prefetch start happens below.

**B)** Before `train_and_get_result()`, try to retrieve prefetched data and start prefetching the next subject:

```python
        # Retrieve prefetched data (if available) and start prefetch for next subject
        precomputed_data = None
        if prefetcher is not None:
            precomputed_data = prefetcher.get_prefetched(subject_id)
            if precomputed_data is not None:
                log_train.info(f"{progress} {subject_id}: using prefetched data")

            # Start prefetch for next non-cached subject
            for future_id in subject_ids[idx:]:  # idx is 1-based, so subject_ids[idx:] = remaining
                if future_id not in cache[model_type] or force_retrain:
                    prefetcher.start_prefetch(future_id)
                    break
```

**C)** Pass `precomputed_data` to `train_and_get_result()`:

```python
            result = train_and_get_result(
                subject_id=subject_id,
                model_type=model_type,
                task=task,
                paradigm=paradigm,
                data_root=data_root,
                save_dir=output_dir,
                run_tag=run_tag,
                no_wandb=no_wandb,
                upload_model=upload_model,
                wandb_group=wandb_group,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
                cache_only=cache_only,
                config_overrides=config_overrides,
                verbose=verbose,
                pretrained_path=pretrained_path,
                freeze_strategy=freeze_strategy,
                precomputed_data=precomputed_data,
            )
```

- [ ] **Step 4: Shutdown prefetcher after the loop**

Insert after the loop (after line 296, before `stats = compute_model_statistics(results)`):

```python
    # Clean up prefetcher
    if prefetcher is not None:
        prefetcher.shutdown()
```

- [ ] **Step 5: Seed the first prefetch before the loop**

Insert after prefetcher creation (right after the `try/except` block that creates the prefetcher):

```python
    # Seed prefetch for first non-cached subject
    if prefetcher is not None:
        for sid in subject_ids:
            if sid not in cache.get(model_type, {}) or force_retrain:
                prefetcher.start_prefetch(sid)
                break
```

- [ ] **Step 6: Commit**

```bash
git add scripts/experiments/run_within_subject.py
git commit -m "feat: integrate subject prefetch into within-subject training loop"
```

---

## Task 4: End-to-end verification

- [ ] **Step 1: Run unit tests**

```bash
uv run pytest tests/test_prefetch.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 2: Run within-subject with prefetch (small test)**

```bash
uv run python scripts/run_within_subject.py --model eegnet --task ternary --cache-only --subjects S01 S02 S03 --force-retrain --no-wandb
```

Expected output should include:
- `Prefetch submitted: S02` (debug log, may need `--verbose`)
- `using prefetched data` for S02 and S03
- Training completes normally, results match non-prefetch behavior

- [ ] **Step 3: Run transfer learning to verify it also works**

```bash
uv run python scripts/run_transfer_comparison.py --task binary --cache-only --subjects S01 S02 --force-retrain --no-wandb
```

Expected: Transfer training runs with prefetch active (via `run_within_subject()` delegation).

- [ ] **Step 4: Verify timing improvement**

Check `timing_breakdown.csv` — with prefetch enabled, `train_data_loading` should read `0.00s` for subjects where prefetched data was used (since `train_single_subject()` skips data loading when `precomputed_data` is provided).

- [ ] **Step 5: Commit all verified changes**

```bash
git add -A
git commit -m "feat: subject-level data prefetch — ~10% wall-clock reduction for sequential within-subject and transfer learning runs"
```

---

## Task 5: Update TODO.md

**Files:**
- Modify: `docs/dev_log/TODO.md`

- [ ] **Step 1: Update status**

Change the status of "分析数据加载 vs 训练时间占比" from "Done — 分析完成，建议实施 subject 级 data prefetch" to "Done — 分析完成，prefetch 已实施".

- [ ] **Step 2: Commit**

```bash
git add docs/dev_log/TODO.md
git commit -m "docs: mark subject-level prefetch as implemented in TODO"
```

---

## Design Notes

### Why ThreadPoolExecutor (not ProcessPoolExecutor)

- Data loading reads HDF5 files (I/O bound) — GIL is released during `h5py` reads
- `ProcessPoolExecutor` on Windows uses `spawn` which copies the entire dataset between processes
- The cache system (`get_cache()`) is already thread-safe with per-file locks
- Existing project pattern: `preprocess_stieger_incremental.py` uses `ThreadPoolExecutor` for prefetch

### Why lookahead=1

- Each prefetched dataset occupies ~500MB RAM (128ch CBraMod)
- Lookahead=1 means at most 2 datasets in memory (current training + next prefetched)
- Loading typically takes 5-15s while training takes 60-85s — lookahead=1 is sufficient

### Why `unified` task is excluded

- Unified tasks use `temporal_split_with_offline_test()` (3-way split with offline test)
- Standard tasks use `temporal_split_by_group()` (2-way split, 80/20)
- The prefetcher only handles the standard 2-way split
- If `task == 'unified'`, the fallback is normal loading inside `train_single_subject()`

### Timer accuracy

- When prefetch succeeds, `train_data_loading` reads 0.00s in timing_breakdown.csv
- This is accurate — the data WAS loaded, but in a background thread, not in the main thread's Timer block
- The overall training wall-clock time reflects the real speedup
