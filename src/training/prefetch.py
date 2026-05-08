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
from typing import Dict, Optional

from src.preprocessing.data_loader import PreprocessConfig
from src.training.common import (
    apply_config_overrides,
    temporal_split_by_group,
    temporal_split_with_offline_test,
)
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
        if task == 'unified':
            raise ValueError("SubjectPrefetcher does not support unified task (requires 3-way split)")

        self._model_type = model_type
        self._task = task
        self._paradigm = paradigm
        self._data_root = data_root
        self._elc_path = elc_path
        self._cache_only = cache_only

        # Resolve preprocessing config (mirrors train_single_subject lines 414-439)
        n_ch = config_overrides.get('data', {}).get('channels') if config_overrides else None
        if n_ch not in (8, 32):
            n_ch = None
        config = get_default_config(model_type, task, n_channels=n_ch)
        if config_overrides:
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

        # Quaternary has no separate Online_Finetune test session — carve a
        # 70/15/15 holdout out of Offline (matches train_within_subject's
        # non-prefetched path). For binary/ternary the test set is the
        # already-loaded Online_Finetune session.
        test_indices: Optional[list] = None
        if self._task == 'quaternary':
            train_indices, val_indices, test_indices = temporal_split_with_offline_test(
                train_dataset, group_attr='session_type',
            )
            # Sanity: holdout must not overlap train/val (regression guard)
            _train_set = set(train_indices)
            _val_set = set(val_indices)
            _test_set = set(test_indices)
            _tt = _test_set & _train_set
            _tv = _test_set & _val_set
            assert not _tt, f"Leakage: {len(_tt)} segs train↔offline_test"
            assert not _tv, f"Leakage: {len(_tv)} segs val↔offline_test"
        else:
            train_indices, val_indices = temporal_split_by_group(
                train_dataset, group_attr='session_type', val_ratio=0.2,
            )

        return {
            'train_dataset': train_dataset,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_dataset': test_dataset,
            'test_indices': test_indices,
        }
