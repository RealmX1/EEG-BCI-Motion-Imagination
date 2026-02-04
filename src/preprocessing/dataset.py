"""
PyTorch Dataset for FINGER-EEG-BCI data.

This module provides the main Dataset class for loading and preprocessing
EEG data for training neural networks.
"""

import json
import logging
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .loader import load_mat_file, parse_session_path
from .pipeline import (
    TrialInfo,
    extract_trials,
    preprocess_trial,
    preprocess_run_to_trials,
    trials_to_segments,
    _process_single_mat_file_to_trials,
)
from .channel_selection import (
    create_biosemi128_to_1020_mapping,
    get_channel_indices,
    STANDARD_1020_CHANNELS,
)
from .cache_manager import PreprocessingCache, get_cache
from ..utils.logging import SectionLogger

if TYPE_CHECKING:
    from .data_loader import PreprocessConfig

logger = logging.getLogger(__name__)
log_load = SectionLogger(logger, 'load')
log_prep = SectionLogger(logger, 'prep')
log_cache = SectionLogger(logger, 'cache')


class FingerEEGDataset(Dataset):
    """
    PyTorch Dataset for FINGER-EEG-BCI data.

    Supports both CBraMod and EEGNet input formats.
    Includes optional caching for preprocessed data.
    """

    def __init__(
        self,
        data_root: str,
        subjects: List[str],
        config: 'PreprocessConfig',
        task_types: Optional[List[str]] = None,
        target_classes: Optional[List[int]] = None,
        elc_path: Optional[str] = None,
        transform=None,
        use_cache: bool = True,
        cache_dir: str = "caches/preprocessed",
        session_folders: Optional[List[str]] = None,
        preconvert_tensors: bool = True,
        parallel_workers: int = 0,
        cache_only: bool = False,
        cache_index_path: str = ".cache_index.json",
    ):
        """
        Initialize dataset.

        Args:
            data_root: Root directory containing subject folders
            subjects: List of subject IDs (e.g., ['S01', 'S02'])
            config: Preprocessing configuration
            task_types: List of task types to include (e.g., ['OfflineImagery'])
                       None = include all. Deprecated: use session_folders instead.
            target_classes: List of target classes to include (e.g., [1, 4] for thumb/pinky)
                           None = include all
            elc_path: Path to biosemi128.ELC file (required for channel mapping)
            transform: Optional transform to apply to data
            use_cache: Whether to use preprocessing cache (default: True)
            cache_dir: Directory for cache files (default: 'caches/preprocessed')
            session_folders: List of exact folder names to include (e.g.,
                           ['OfflineImagery', 'OnlineImagery_Sess01_2class_Base']).
                           This takes precedence over task_types if specified.
            preconvert_tensors: If True, convert numpy arrays to tensors at load time
                               for faster __getitem__. Uses more memory but ~5-10% faster.
                               (default: True)
            parallel_workers: Number of parallel workers for loading/preprocessing.
                            0 = auto (use cpu_count - 1), -1 = disabled (serial).
                            Parallel loading can speed up first-time preprocessing by 3-4x.
                            (default: 0)
            cache_only: If True, load data exclusively from cache index without
                       scanning filesystem. Useful when original .mat files are not
                       available but preprocessed caches exist. (default: False)
            cache_index_path: Path to cache index file for cache_only mode.
                            (default: '.cache_index.json')
        """
        self.data_root = Path(data_root)
        self.subjects = subjects
        self.config = config
        self.task_types = task_types
        self.target_classes = target_classes
        self.transform = transform
        self.session_folders = session_folders
        self.preconvert_tensors = preconvert_tensors
        self.cache_only = cache_only
        self.cache_index_path = cache_index_path

        # Validate cache_only mode
        if cache_only and not use_cache:
            raise ValueError("cache_only=True requires use_cache=True")

        # Set up parallel workers
        if parallel_workers == 0:
            # Auto: use cpu_count - 1
            # Each worker needs ~400MB for preprocessing, so 8 workers ≈ 3.2GB
            self.parallel_workers = max(1, (os.cpu_count() or 1) - 1)
        elif parallel_workers < 0:
            # Disabled
            self.parallel_workers = 1
        else:
            self.parallel_workers = parallel_workers

        # Initialize cache
        self.cache: Optional[PreprocessingCache] = None
        if use_cache:
            self.cache = get_cache(cache_dir=cache_dir, enabled=True)

        # Set up channel mapping
        self.channel_indices = None
        if config.channel_strategy == 'A' and elc_path:
            mapping = create_biosemi128_to_1020_mapping(elc_path)
            idx_map = get_channel_indices(mapping)
            self.channel_indices = [idx_map[ch] for ch in STANDARD_1020_CHANNELS]
        elif config.channel_strategy == 'C':
            self.channel_indices = None  # Use all 128 channels

        # Load all trials
        self.trials = []
        self.labels = []
        self.trial_infos = []

        # Global trial counter (to ensure unique trial indices across all runs)
        self._global_trial_counter = 0

        self._load_data()

        # Map labels to continuous indices
        self._setup_label_mapping()

        # Pre-convert numpy arrays to tensors for faster __getitem__
        if self.preconvert_tensors and self.trials:
            self._convert_to_tensors()

    def _build_file_list_from_cache_index(self) -> List[Tuple[Path, Dict, bool, bool]]:
        """
        构建文件列表从缓存索引（纯缓存模式）。

        返回与文件系统扫描相同格式的文件列表，但完全基于缓存索引。
        这允许在原始 .mat 文件不可用时仍能加载数据。

        Returns:
            List of (mat_path, session_info, needs_processing, is_offline) tuples
            其中 mat_path 是虚拟路径，needs_processing 始终为 False
        """
        cache_path = Path(self.cache_index_path)

        if not cache_path.exists():
            logger.error(f"Cache index not found at {self.cache_index_path} (cache_only mode)")
            return []

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse cache index: {e}")
            return []

        entries = cache_data.get('entries', {})
        if not entries:
            logger.warning(f"Cache index contains no entries")
            return []

        # 收集符合条件的缓存条目
        # 使用 (subject, run, session_folder) 作为唯一键去重
        unique_files = {}  # key: (subject, run, session_folder), value: entry_data

        for cache_key, entry_data in entries.items():
            subject = entry_data.get('subject')
            run = entry_data.get('run')
            session_folder = entry_data.get('session_folder')
            model = entry_data.get('model')

            # 过滤：被试
            if subject not in self.subjects:
                continue

            # 过滤：模型匹配
            if model != self.config.target_model:
                continue

            # 过滤：session_folders
            if self.session_folders is not None:
                if session_folder not in self.session_folders:
                    continue
            elif self.task_types is not None:
                # Deprecated: task_types 过滤
                task_type = entry_data.get('subject_task_type')
                if task_type not in self.task_types:
                    continue

            # 过滤：target_classes（仅对 online 数据）
            is_offline = self._is_offline_session(session_folder)
            entry_n_classes = entry_data.get('n_classes')

            if not is_offline and self.target_classes is not None:
                # Online 数据：检查 n_classes 是否匹配
                expected_n_classes = len(self.target_classes)
                if entry_n_classes != expected_n_classes:
                    continue

            # 构建唯一键
            file_key = (subject, run, session_folder)

            # 去重：优先保留当前模型的缓存
            if file_key not in unique_files:
                unique_files[file_key] = entry_data

        # 构建文件列表
        files_to_process = []

        for (subject, run, session_folder), entry_data in sorted(unique_files.items()):
            # 构建虚拟 mat_path（用于缓存键生成，但不访问文件系统）
            # 格式: data_root/subject/session_folder/subject_session_folder_R{run:02d}.mat
            virtual_filename = f"{subject}_{session_folder}_R{run:02d}.mat"
            virtual_mat_path = self.data_root / subject / session_folder / virtual_filename

            # 构建 session_info（与 parse_session_path 相同格式）
            session_info = {
                'subject': subject,
                'run': run,
                'task_type': entry_data.get('subject_task_type'),
                'session_folder': session_folder,
            }

            is_offline = self._is_offline_session(session_folder)
            needs_processing = False  # 纯缓存模式下，所有文件都应该已缓存

            files_to_process.append((virtual_mat_path, session_info, needs_processing, is_offline))

        logger.info(f"Cache-only mode: Found {len(files_to_process)} cached files for {len(self.subjects)} subjects")

        return files_to_process

    def _load_data(self):
        """
        Load all data from disk.

        For paper-aligned preprocessing (use_sliding_window=True), this applies
        preprocessing at the Run level (entire .mat file) to match paper methodology.

        For trial-based preprocessing, this processes individual trials.

        Supports parallel loading when parallel_workers > 1, which can speed up
        first-time preprocessing by 3-4x.
        """
        from src.utils.timing import format_time

        total_start = time.perf_counter()
        n_cache_hits = 0
        n_cache_misses = 0

        # Phase 1: Collect all files and check cache status
        files_to_process = []  # (mat_path, session_info, needs_processing, is_offline)

        if self.cache_only:
            # 纯缓存模式：从缓存索引构建文件列表
            files_to_process = self._build_file_list_from_cache_index()

            if not files_to_process:
                log_load.error(f"Cache-only mode: No cached files found for subjects {self.subjects}")
                return
        else:
            # 传统模式：扫描文件系统
            for subject in self.subjects:
                subject_dir = self.data_root / subject

                if not subject_dir.exists():
                    log_load.warning(f"Subject dir not found: {subject_dir}")
                    continue

                mat_files = sorted(subject_dir.rglob('*.mat'))  # Sort for reproducibility

                for mat_path in mat_files:
                    session_info = parse_session_path(mat_path)
                    parent_folder = mat_path.parent.name

                    # Filter by session folders (takes precedence over task_types)
                    if self.session_folders is not None:
                        if parent_folder not in self.session_folders:
                            continue
                    elif self.task_types is not None:
                        if session_info['task_type'] not in self.task_types:
                            continue

                    # Skip known bad data
                    if 'S07' in str(mat_path) and 'OnlineImagery_Sess05_3class_Base' in str(mat_path):
                        log_load.info(f"Skip bad data: {mat_path.name}")
                        continue

                    # Check cache
                    # Offline data: cache with target_classes=None (all 4 fingers)
                    # Online data: cache with actual target_classes
                    is_offline = self._is_offline_session(parent_folder)
                    cache_target_classes = None if is_offline else self.target_classes

                    needs_processing = True
                    if self.cache is not None and self.config.use_sliding_window:
                        has_cache = self.cache.has_valid_cache(
                            session_info['subject'],
                            session_info['run'],
                            parent_folder,  # Use folder name, not task_type
                            self.config,
                            str(mat_path),
                            cache_target_classes,  # None for Offline, actual for Online
                            experiment_tag=self.config.get_experiment_cache_tag(),
                        )
                        if has_cache:
                            needs_processing = False

                    files_to_process.append((mat_path, session_info, needs_processing, is_offline))

        # Phase 2: Load from cache (fast, serial)
        # v3.0: Cache stores trials, not segments. Apply sliding window on load.
        cached_files = [(p, s, offline) for p, s, needs, offline in files_to_process if not needs]
        for mat_path, session_info, is_offline in cached_files:
            try:
                parent_folder = mat_path.parent.name
                cache_target_classes = None if is_offline else self.target_classes

                # v3.0: Load trials + labels (not segments)
                trials, labels = self.cache.load(
                    session_info['subject'],
                    session_info['run'],
                    parent_folder,  # Use folder name
                    self.config,
                    cache_target_classes,  # None for Offline
                    experiment_tag=self.config.get_experiment_cache_tag(),
                )

                # Offline data: filter to target_classes before sliding window
                # Online data: just map labels (already filtered during extraction)
                if self.target_classes is not None:
                    if is_offline:
                        # Offline: filter + map labels
                        trials, labels = self._filter_trials_by_classes(
                            trials, labels, self.target_classes
                        )
                    else:
                        # Online: just map labels (already filtered)
                        labels = self._map_labels_to_indices(labels, self.target_classes)

                # v3.0: Apply sliding window, filter, normalize on load
                segments, seg_labels, trial_indices = trials_to_segments(
                    trials, labels, self.config
                )

                # Apply channel selection if needed (after trials_to_segments)
                if self.channel_indices is not None:
                    segments = segments[:, self.channel_indices, :]

                self._store_segments(segments, seg_labels, trial_indices, session_info)
                n_cache_hits += 1
            except Exception as e:
                log_cache.error(f"Cache load failed: {mat_path.name}: {e}")

        # Phase 3: Process uncached files (potentially parallel)
        uncached_files = [(p, s, offline) for p, s, needs, offline in files_to_process if needs]

        if uncached_files:
            if self.cache_only:
                # 纯缓存模式：不应该有未缓存的文件
                log_load.error(
                    f"Cache-only mode: {len(uncached_files)} files have no cache. "
                    f"Cannot process without original .mat files."
                )
                for mat_path, session_info, is_offline in uncached_files:
                    log_load.error(f"  Missing cache: {session_info['subject']} {session_info['session_folder']} R{session_info['run']:02d}")
            else:
                # 传统模式：处理未缓存的文件
                if self.config.use_sliding_window:
                    # Paper-aligned preprocessing with parallel support
                    self._load_uncached_parallel(uncached_files)
                else:
                    # Trial-based preprocessing (serial, less common)
                    for mat_path, session_info, is_offline in uncached_files:
                        try:
                            eeg_data, events, metadata = load_mat_file(str(mat_path))
                            self._load_run_trial_based(
                                eeg_data, events, metadata, session_info, mat_path
                            )
                        except Exception as e:
                            log_load.error(f"Load failed: {mat_path.name}: {e}")

                n_cache_misses = len(uncached_files)

        total_time = time.perf_counter() - total_start

        # Store cache stats as instance attributes for external access
        self.n_cache_hits = n_cache_hits
        self.n_cache_misses = n_cache_misses

        # Log summary
        log_load.debug(f"Load time: {format_time(total_time)} ({n_cache_hits} hits, {n_cache_misses} miss, {self.parallel_workers}w)")
        log_load.info(f"Loaded {len(self.trials)} segs (cache: {'hit' if n_cache_misses == 0 else 'partial'})")

    def _load_uncached_parallel(self, uncached_files: List[Tuple[Path, Dict, bool]]):
        """
        Load and preprocess uncached files using parallel workers.

        v3.0: Uses trial-level caching. Process flow:
        1. Parallel: Extract trials, apply CAR, downsample (preprocess_run_to_trials)
        2. Serial: Store segments after applying sliding window
        3. Parallel: Save trials to cache

        Results are merged in the original file order to ensure reproducibility.

        Args:
            uncached_files: List of (mat_path, session_info, is_offline) tuples
        """
        n_files = len(uncached_files)
        use_parallel = self.parallel_workers > 1 and n_files > 1

        if use_parallel:
            log_prep.info(f"Parallel preproc: {n_files} files, {self.parallel_workers}w")
            start_time = time.perf_counter()

            # Prepare arguments for parallel execution
            # Sort by path for reproducible ordering
            sorted_files = sorted(uncached_files, key=lambda x: str(x[0]))
            mat_paths = [str(p) for p, _, _ in sorted_files]

            # Build mapping of mat_path -> is_offline
            path_to_offline = {str(p): offline for p, _, offline in sorted_files}

            # Use ProcessPoolExecutor for CPU-bound preprocessing
            # v3.0: Use _process_single_mat_file_to_trials instead of _process_single_mat_file
            results = {}

            with ProcessPoolExecutor(max_workers=self.parallel_workers) as executor:
                # Submit all tasks
                # For offline data: store_all_fingers=True, process all 4 fingers
                # For online data: use actual target_classes
                future_to_path = {
                    executor.submit(
                        _process_single_mat_file_to_trials,
                        path,
                        self.config,
                        self.target_classes,
                        None,  # channel_indices applied later (after cache load)
                        path_to_offline[path]  # store_all_fingers for offline
                    ): path for path in mat_paths
                }

                # Collect results as they complete
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        result = future.result()
                        results[path] = result
                    except MemoryError as e:
                        # OOM errors should propagate - don't silently continue
                        log_prep.critical(f"Out of memory during parallel processing: {Path(path).name}")
                        raise
                    except Exception as e:
                        log_prep.error(f"Parallel failed: {Path(path).name}: {e}")
                        results[path] = (None, None, parse_session_path(Path(path)), path)

            elapsed_preprocess = time.perf_counter() - start_time
            log_prep.info(f"Parallel done: {elapsed_preprocess:.1f}s ({n_files / elapsed_preprocess:.1f} f/s)")

            # Store results in original sorted order (for reproducibility)
            # v3.0: Apply trials_to_segments() before storing
            for mat_path in mat_paths:
                trials, labels, session_info, path = results[mat_path]
                if trials is not None:
                    is_offline = path_to_offline[mat_path]

                    # Offline data: filter to target_classes before sliding window
                    # Online data: just map labels (already filtered during extraction)
                    trials_for_segments = trials
                    labels_for_segments = labels
                    if self.target_classes is not None:
                        if is_offline:
                            # Offline: filter + map labels
                            trials_for_segments, labels_for_segments = self._filter_trials_by_classes(
                                trials, labels, self.target_classes
                            )
                        else:
                            # Online: just map labels (already filtered)
                            labels_for_segments = self._map_labels_to_indices(labels, self.target_classes)

                    # v3.0: Apply sliding window, filter, normalize
                    segments, seg_labels, trial_indices = trials_to_segments(
                        trials_for_segments, labels_for_segments, self.config
                    )

                    # Apply channel selection if needed
                    if self.channel_indices is not None:
                        segments = segments[:, self.channel_indices, :]

                    self._store_segments(segments, seg_labels, trial_indices, session_info)

            # Save to cache in parallel (I/O + compression bound)
            # v3.0: Cache stores TRIALS (not segments), UNFILTERED for offline
            if self.cache is not None:
                cache_start = time.perf_counter()
                cache_tasks = [
                    (results[p], p, path_to_offline[p]) for p in mat_paths
                    if results[p][0] is not None
                ]

                def save_to_cache(args):
                    (trials, labels, session_info, path), mat_path, is_offline = args
                    parent_folder = Path(path).parent.name
                    # Offline: cache all fingers with target_classes=None
                    # Online: cache with actual target_classes
                    cache_target_classes = None if is_offline else self.target_classes
                    self.cache.save(
                        session_info['subject'],
                        session_info['run'],
                        parent_folder,  # Use folder name
                        self.config,
                        trials, labels,  # v3.0: trials, not segments
                        path, cache_target_classes,
                        experiment_tag=self.config.get_experiment_cache_tag(),
                    )

                # Use ThreadPoolExecutor for I/O-bound cache saving
                with ThreadPoolExecutor(max_workers=min(self.parallel_workers, 8)) as executor:
                    list(executor.map(save_to_cache, cache_tasks))

                cache_elapsed = time.perf_counter() - cache_start
                log_cache.debug(f"Cache save: {cache_elapsed:.1f}s")

        else:
            # Serial fallback
            for mat_path, session_info, is_offline in uncached_files:
                try:
                    eeg_data, events, metadata = load_mat_file(str(mat_path))
                    self._load_run_paper_aligned(
                        eeg_data, events, metadata, session_info, mat_path, is_offline
                    )
                except Exception as e:
                    log_load.error(f"Load failed: {mat_path.name}: {e}")

    def _load_run_paper_aligned(
        self,
        eeg_data: np.ndarray,
        events: List[Dict],
        metadata: Dict,
        session_info: Dict,
        mat_path: Path,
        is_offline: bool = False
    ):
        """
        Load a run using paper-aligned preprocessing.

        v3.0: Uses trial-level caching. Process flow:
        1. Extract trials, apply CAR, downsample (preprocess_run_to_trials)
        2. Save trials to cache (smaller than segments)
        3. Apply sliding window, filter, normalize (trials_to_segments)
        4. Store segments in dataset

        Note: Cache loading is handled by _load_data() for timing purposes.
        This method is only called on cache miss.

        Args:
            is_offline: If True, this is Offline data - cache all 4 fingers,
                       filter to target_classes after loading.
        """
        subject = session_info['subject']
        run_id = session_info['run']
        parent_folder = mat_path.parent.name

        # Offline data: store all fingers in cache, filter later
        # Online data: store only target_classes
        store_all_fingers = is_offline

        # v3.0: Preprocess to trial level (no sliding window yet)
        trials, labels = preprocess_run_to_trials(
            eeg_data,
            events,
            metadata,
            self.config,
            target_classes=self.target_classes if not store_all_fingers else None,
            store_all_fingers=store_all_fingers
        )

        if len(trials) == 0:
            return

        # Save to cache (before applying sliding window)
        # Offline: save all fingers with target_classes=None
        # Online: save with actual target_classes
        if self.cache is not None:
            cache_target_classes = None if is_offline else self.target_classes
            self.cache.save(
                subject, run_id, parent_folder, self.config,
                trials, labels,  # v3.0: trials, not segments
                str(mat_path), cache_target_classes,
                experiment_tag=self.config.get_experiment_cache_tag(),
            )

        # Offline data: filter to target_classes before sliding window
        # Online data: just map labels (already filtered during extraction)
        trials_for_segments = trials
        labels_for_segments = labels
        if self.target_classes is not None:
            if is_offline:
                # Offline: filter + map labels
                trials_for_segments, labels_for_segments = self._filter_trials_by_classes(
                    trials, labels, self.target_classes
                )
            else:
                # Online: just map labels (already filtered)
                labels_for_segments = self._map_labels_to_indices(labels, self.target_classes)

        # v3.0: Apply sliding window, filter, normalize
        segments, seg_labels, trial_indices = trials_to_segments(
            trials_for_segments, labels_for_segments, self.config
        )

        # Apply channel selection if needed
        if self.channel_indices is not None:
            segments = segments[:, self.channel_indices, :]

        # Store segments
        self._store_segments(segments, seg_labels, trial_indices, session_info)

    def _store_segments(
        self,
        segments: np.ndarray,
        seg_labels: np.ndarray,
        trial_indices: np.ndarray,
        session_info: Dict
    ):
        """
        Store preprocessed segments into the dataset.

        CRITICAL FIX: Use global unique trial indices to prevent data leakage.
        Each trial gets a globally unique ID across all runs.
        """
        # Get unique local trial indices from this run
        unique_local_trials = np.unique(trial_indices)

        # Create mapping from local trial_idx to global trial_idx
        local_to_global = {}
        for local_idx in unique_local_trials:
            local_to_global[local_idx] = self._global_trial_counter
            self._global_trial_counter += 1

        # Store segments with globally unique trial indices
        for i, (segment, label, local_trial_idx) in enumerate(zip(segments, seg_labels, trial_indices)):
            # Map local trial index to global unique index
            global_trial_idx = local_to_global[local_trial_idx]

            trial_info = TrialInfo(
                subject_id=session_info['subject'],
                session_type=session_info['session_folder'],  # CRITICAL FIX: Use full folder name for unique identification
                run_id=session_info['run'],
                trial_idx=global_trial_idx,  # FIXED: Use globally unique trial index
                target_class=int(self.target_classes[label]) if self.target_classes else label,
                start_sample=0,  # Segment doesn't have original sample info
                end_sample=int(self.config.segment_length * self.config.original_fs),
            )

            self.trials.append(segment)
            self.labels.append(label)
            self.trial_infos.append(trial_info)

    def _map_labels_to_indices(
        self,
        labels: np.ndarray,
        target_classes: List[int],
    ) -> np.ndarray:
        """
        Map original labels (finger IDs) to continuous indices.

        v3.0: Used for Online data where labels are already filtered
        but need to be mapped to continuous indices (0, 1, ..., n_classes-1).

        Args:
            labels: Original labels (finger IDs: 1, 4 for binary)
            target_classes: Target classes (e.g., [1, 4] for binary)

        Returns:
            Mapped labels using continuous indices (0, 1, ..., n_classes-1)
        """
        label_mapping = {cls: i for i, cls in enumerate(sorted(target_classes))}
        return np.array([label_mapping[l] for l in labels])

    def _filter_trials_by_classes(
        self,
        trials: np.ndarray,
        labels: np.ndarray,
        target_classes: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter trials to keep only those matching target_classes.

        v3.0: Used for Offline data where all 4 fingers are cached together,
        but we need to filter to specific classes at load time.

        Args:
            trials: All trials [n_trials x channels x samples]
            labels: Original labels (finger IDs: 1,2,3,4)
            target_classes: Classes to keep (e.g., [1, 4] for binary)

        Returns:
            Tuple of (filtered_trials, mapped_labels)
            where mapped_labels use continuous indices (0, 1, ..., n_classes-1)
        """
        # Create mask for target classes
        mask = np.isin(labels, target_classes)

        # Filter
        filtered_trials = trials[mask]

        # Map labels to continuous indices
        filtered_labels = self._map_labels_to_indices(labels[mask], target_classes)

        return filtered_trials, filtered_labels

    def _filter_segments_by_classes(
        self,
        segments: np.ndarray,
        labels: np.ndarray,
        trial_indices: np.ndarray,
        target_classes: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter segments to keep only those matching target_classes.

        DEPRECATED in v3.0: Use _filter_trials_by_classes instead.
        Kept for backward compatibility with non-sliding-window mode.

        Args:
            segments: All segments [n_segments x channels x samples]
            labels: Original labels (finger IDs: 1,2,3,4)
            trial_indices: Trial indices for each segment
            target_classes: Classes to keep (e.g., [1, 4] for binary)

        Returns:
            Tuple of (filtered_segments, mapped_labels, filtered_trial_indices)
            where mapped_labels use continuous indices (0, 1, ..., n_classes-1)
        """
        # Create mask for target classes
        mask = np.isin(labels, target_classes)

        # Filter
        filtered_segments = segments[mask]
        filtered_trial_indices = trial_indices[mask]

        # Map labels to continuous indices
        label_mapping = {cls: i for i, cls in enumerate(sorted(target_classes))}
        filtered_labels = np.array([label_mapping[l] for l in labels[mask]])

        return filtered_segments, filtered_labels, filtered_trial_indices

    def _is_offline_session(self, folder_name: str) -> bool:
        """Check if the session folder is an Offline session."""
        return folder_name.lower().startswith('offline')

    def _load_run_trial_based(
        self,
        eeg_data: np.ndarray,
        events: List[Dict],
        metadata: Dict,
        session_info: Dict,
        mat_path: Path
    ):
        """
        Load a run using trial-based preprocessing (for CBraMod).

        Processes each trial independently.
        """
        # Extract trials
        file_trials = extract_trials(eeg_data, events, metadata, self.config)

        for trial_data, label, trial_info in file_trials:
            # Filter by target class
            if self.target_classes is not None:
                if label not in self.target_classes:
                    continue

            # Update trial info
            trial_info.subject_id = session_info['subject']
            trial_info.session_type = session_info['session_folder']  # CRITICAL FIX: Use full folder name for unique identification
            trial_info.run_id = session_info['run']

            # Preprocess
            processed = preprocess_trial(
                trial_data, self.config, self.channel_indices
            )

            if processed is not None:
                # Map label if needed
                if self.target_classes is not None:
                    label_mapping = {cls: i for i, cls in enumerate(sorted(self.target_classes))}
                    label = label_mapping[label]

                self.trials.append(processed)
                self.labels.append(label)
                self.trial_infos.append(trial_info)

    def _setup_label_mapping(self):
        """
        Create mapping from original labels to contiguous indices.

        If use_sliding_window=True, labels are already mapped to 0-indexed during loading.
        Otherwise, create mapping from original labels.
        """
        unique_labels = sorted(set(self.labels))

        if self.config.use_sliding_window and self.target_classes is not None:
            # Labels already mapped to 0-indexed during paper-aligned preprocessing
            self.label_to_idx = {i: i for i in unique_labels}
            self.idx_to_label = {i: i for i in unique_labels}
        else:
            # Create new mapping
            self.label_to_idx = {label: i for i, label in enumerate(unique_labels)}
            self.idx_to_label = {i: label for label, i in self.label_to_idx.items()}

        self.n_classes = len(unique_labels)

        log_load.debug(f"Labels: {self.label_to_idx}, n_classes={self.n_classes}")

    def _convert_to_tensors(self):
        """
        Convert numpy arrays to PyTorch tensors for faster __getitem__.

        This trades memory for speed: ~5-10% faster training by avoiding
        numpy->tensor conversion on every batch.
        """
        start = time.perf_counter()

        # Convert trials to a single stacked tensor for efficiency
        # This also enables potential future optimizations like pinned memory
        self.trials = [torch.from_numpy(t).float() for t in self.trials]

        # Also pre-convert labels to tensor
        self.labels_tensor = torch.tensor(
            [self.label_to_idx[label] for label in self.labels],
            dtype=torch.long
        )

        elapsed = time.perf_counter() - start
        log_load.debug(f"Tensors: {len(self.trials)} trials in {elapsed:.2f}s")

    def __len__(self) -> int:
        return len(self.trials)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Fast path: use pre-converted tensors
        if self.preconvert_tensors and hasattr(self, 'labels_tensor'):
            trial_tensor = self.trials[idx]
            label = self.labels_tensor[idx].item()
        else:
            # Slow path: convert numpy to tensor on-the-fly
            trial = self.trials[idx]
            label = self.label_to_idx[self.labels[idx]]
            trial_tensor = torch.from_numpy(trial).float()

        if self.transform:
            trial_tensor = self.transform(trial_tensor)

        return trial_tensor, label

    def get_trial_info(self, idx: int) -> TrialInfo:
        """Get detailed information about a trial."""
        return self.trial_infos[idx]

    def get_unique_trials(self) -> List[int]:
        """
        Get list of unique trial indices in the dataset.

        Returns:
            Sorted list of unique trial indices (globally unique)
        """
        unique_trials = sorted(set(info.trial_idx for info in self.trial_infos))
        return unique_trials

    def get_segment_indices_for_trials(self, trial_indices: List[int]) -> List[int]:
        """
        Get segment indices that belong to specific trials.

        Args:
            trial_indices: List of trial indices to filter

        Returns:
            List of segment indices belonging to the specified trials
        """
        trial_set = set(trial_indices)
        segment_indices = [
            i for i, info in enumerate(self.trial_infos)
            if info.trial_idx in trial_set
        ]
        return segment_indices


def create_dataloaders(
    data_root: str,
    config: 'PreprocessConfig',
    elc_path: str,
    train_subjects: List[str],
    val_subjects: List[str],
    test_subjects: List[str],
    task_types: Optional[List[str]] = None,
    target_classes: Optional[List[int]] = None,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders.

    Args:
        data_root: Path to data directory
        config: Preprocessing configuration
        elc_path: Path to electrode file
        train_subjects: List of training subject IDs
        val_subjects: List of validation subject IDs
        test_subjects: List of test subject IDs
        task_types: Task types to include
        target_classes: Target classes to include
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = FingerEEGDataset(
        data_root, train_subjects, config,
        task_types=task_types,
        target_classes=target_classes,
        elc_path=elc_path
    )

    val_dataset = FingerEEGDataset(
        data_root, val_subjects, config,
        task_types=task_types,
        target_classes=target_classes,
        elc_path=elc_path
    )

    test_dataset = FingerEEGDataset(
        data_root, test_subjects, config,
        task_types=task_types,
        target_classes=target_classes,
        elc_path=elc_path
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
