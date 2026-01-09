"""
Preprocessing cache manager for EEG-BCI project.

Implements HDF5-based caching for preprocessed EEG data to avoid
redundant computation during training and hyperparameter optimization.

Cache structure:
    caches/
    ├── preprocessed/
    │   ├── {cache_key}.h5           # Preprocessed segments
    │   └── ...
    └── .cache_index.json            # Metadata index

Features:
    - HDF5 disk caching with gzip compression
    - In-memory LRU cache for fast repeated access
    - Automatic cache invalidation on source file changes

Usage:
    from src.preprocessing.cache_manager import PreprocessingCache

    cache = PreprocessingCache()

    # Check and load from cache
    if cache.has_valid_cache(subject, run, task_type, config, mat_path):
        segments, labels, trial_indices = cache.load(...)
    else:
        segments, labels, trial_indices = preprocess_run_paper_aligned(...)
        cache.save(subject, run, task_type, config, segments, labels, trial_indices, mat_path)
"""

import hashlib
import json
import logging
import time
import threading
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try to import h5py, fall back to numpy if not available
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    logger.warning("h5py not available, using numpy .npz format for caching")


class PreprocessingCache:
    """
    HDF5-based caching for preprocessed EEG runs with in-memory LRU cache.

    Falls back to numpy .npz format if h5py is not installed.

    Two-tier caching:
        1. In-memory LRU cache (fast, limited by memory_limit_gb)
        2. HDF5 disk cache (slower, persistent)

    Attributes:
        cache_dir: Directory for cache files
        enabled: Whether caching is enabled
        compression: HDF5 compression level (0-9, higher = smaller but slower)
        use_memory_cache: Whether to use in-memory caching
        memory_limit_gb: Maximum memory for in-memory cache (default: 16 GB)
    """

    # Cache format version - increment when cache structure changes
    CACHE_VERSION = "1.0"

    def __init__(
        self,
        cache_dir: str = "caches/preprocessed",
        enabled: bool = True,
        compression: str = "lzf",
        use_memory_cache: bool = True,
        memory_limit_gb: float = 16.0,
    ):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory to store cache files
            enabled: Whether caching is enabled (can be disabled for debugging)
            compression: HDF5 compression type: "lzf" (fast, default), "gzip" (smaller),
                        or None (no compression). For gzip, use "gzip:N" where N is 0-9.
            use_memory_cache: Whether to use in-memory LRU cache (default: True)
            memory_limit_gb: Maximum memory for in-memory cache in GB (default: 16)
        """
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self.compression = compression  # "lzf", "gzip", "gzip:N", or None
        self.use_h5py = HAS_H5PY
        self.use_memory_cache = use_memory_cache
        self.memory_limit_bytes = int(memory_limit_gb * 1024 * 1024 * 1024)

        # Thread lock for metadata operations (needed for parallel cache saving)
        self._lock = threading.Lock()

        # In-memory LRU cache: OrderedDict for LRU eviction
        self._memory_cache: OrderedDict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()
        self._memory_cache_size = 0  # Current size in bytes

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.cache_dir / ".cache_index.json"
        self.metadata = self._load_metadata()

        logger.debug(
            f"PreprocessingCache initialized: dir={self.cache_dir}, "
            f"enabled={self.enabled}, format={'h5' if self.use_h5py else 'npz'}, "
            f"memory_cache={use_memory_cache} ({memory_limit_gb:.1f}GB limit)"
        )

    def _load_metadata(self) -> Dict:
        """Load cache metadata index."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cache metadata: {e}")
                return {"version": self.CACHE_VERSION, "entries": {}}
        return {"version": self.CACHE_VERSION, "entries": {}}

    def _save_metadata(self) -> None:
        """Save cache metadata index."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save cache metadata: {e}")

    def _config_to_dict(self, config: Any) -> Dict:
        """Convert PreprocessConfig to dict for hashing."""
        if hasattr(config, "__dataclass_fields__"):
            return asdict(config)
        elif isinstance(config, dict):
            return config
        else:
            raise ValueError(f"Unsupported config type: {type(config)}")

    def get_cache_key(
        self,
        subject: str,
        run: int,
        task_type: str,
        config: Any,
        target_classes: Optional[List[int]] = None,
    ) -> str:
        """
        Generate unique cache key based on all relevant parameters.

        Args:
            subject: Subject ID (e.g., 'S01')
            run: Run number
            task_type: Task type (e.g., 'OfflineImagery')
            config: PreprocessConfig instance or dict
            target_classes: Target classes for filtering

        Returns:
            MD5 hash string as cache key
        """
        config_dict = self._config_to_dict(config)

        # Only include config parameters that affect preprocessing output
        relevant_config = {
            "target_model": config_dict.get("target_model"),
            "target_fs": config_dict.get("target_fs"),
            "bandpass_low": config_dict.get("bandpass_low"),
            "bandpass_high": config_dict.get("bandpass_high"),
            "filter_order": config_dict.get("filter_order"),
            "use_sliding_window": config_dict.get("use_sliding_window"),
            "segment_length": config_dict.get("segment_length"),
            "segment_step_samples": config_dict.get("segment_step_samples"),
            "normalize_method": config_dict.get("normalize_method"),
            "apply_car": config_dict.get("apply_car"),
            "trial_duration": config_dict.get("trial_duration"),
            "filter_padding": config_dict.get("filter_padding"),
        }

        key_data = {
            "version": self.CACHE_VERSION,
            "subject": subject,
            "run": run,
            "task_type": task_type,
            "config": relevant_config,
            "target_classes": sorted(target_classes) if target_classes else None,
        }

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get path for cache file."""
        ext = ".h5" if self.use_h5py else ".npz"
        return self.cache_dir / f"{cache_key}{ext}"

    # ==================== Memory Cache Methods ====================

    def _get_entry_size(self, segments: np.ndarray, labels: np.ndarray, trial_indices: np.ndarray) -> int:
        """Calculate memory size of a cache entry."""
        return segments.nbytes + labels.nbytes + trial_indices.nbytes

    def _evict_lru_if_needed(self, new_entry_size: int) -> None:
        """Evict least recently used entries if memory limit would be exceeded."""
        while (self._memory_cache_size + new_entry_size > self.memory_limit_bytes
               and self._memory_cache):
            # Pop oldest entry (first item in OrderedDict)
            oldest_key, oldest_data = self._memory_cache.popitem(last=False)
            evicted_size = self._get_entry_size(*oldest_data)
            self._memory_cache_size -= evicted_size
            logger.debug(f"Evicted from memory cache: {oldest_key[:8]} ({evicted_size / 1e6:.1f} MB)")

    def _add_to_memory_cache(
        self,
        cache_key: str,
        segments: np.ndarray,
        labels: np.ndarray,
        trial_indices: np.ndarray
    ) -> None:
        """Add entry to memory cache with LRU eviction."""
        if not self.use_memory_cache:
            return

        entry_size = self._get_entry_size(segments, labels, trial_indices)

        # Don't cache if single entry exceeds limit
        if entry_size > self.memory_limit_bytes:
            logger.debug(f"Entry too large for memory cache: {entry_size / 1e6:.1f} MB")
            return

        # Evict if needed
        self._evict_lru_if_needed(entry_size)

        # Add to cache
        self._memory_cache[cache_key] = (segments, labels, trial_indices)
        self._memory_cache_size += entry_size

    def _get_from_memory_cache(
        self,
        cache_key: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Get entry from memory cache, updating LRU order."""
        if not self.use_memory_cache or cache_key not in self._memory_cache:
            return None

        # Move to end (most recently used)
        self._memory_cache.move_to_end(cache_key)
        return self._memory_cache[cache_key]

    def get_memory_cache_stats(self) -> Dict:
        """Get memory cache statistics."""
        return {
            "entries": len(self._memory_cache),
            "size_mb": round(self._memory_cache_size / 1e6, 2),
            "limit_mb": round(self.memory_limit_bytes / 1e6, 2),
            "usage_pct": round(self._memory_cache_size / self.memory_limit_bytes * 100, 1)
                         if self.memory_limit_bytes > 0 else 0,
        }

    def clear_memory_cache(self) -> None:
        """Clear in-memory cache."""
        self._memory_cache.clear()
        self._memory_cache_size = 0
        logger.info("Memory cache cleared")

    def preload_to_memory(
        self,
        subjects: Optional[List[str]] = None,
        max_gb: Optional[float] = None
    ) -> Dict:
        """
        Preload disk cache into memory for faster access.

        Args:
            subjects: List of subjects to preload (None = all cached)
            max_gb: Maximum GB to preload (None = use memory_limit)

        Returns:
            Dict with preload statistics
        """
        if max_gb is not None:
            limit = int(max_gb * 1024 * 1024 * 1024)
        else:
            limit = self.memory_limit_bytes

        loaded_entries = 0
        loaded_bytes = 0
        start_time = time.time()

        for cache_key, entry in self.metadata.get("entries", {}).items():
            # Filter by subject if specified
            if subjects is not None and entry.get("subject") not in subjects:
                continue

            # Check memory limit
            entry_size = int(entry.get("size_mb", 0) * 1e6)
            if loaded_bytes + entry_size > limit:
                logger.info(f"Preload stopped: memory limit reached ({loaded_bytes / 1e9:.2f} GB)")
                break

            # Load from disk to memory
            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists() and cache_key not in self._memory_cache:
                try:
                    if self.use_h5py:
                        with h5py.File(cache_path, "r") as f:
                            segments = f["segments"][:]
                            labels = f["labels"][:]
                            trial_indices = f["trial_indices"][:]
                    else:
                        data = np.load(cache_path)
                        segments = data["segments"]
                        labels = data["labels"]
                        trial_indices = data["trial_indices"]

                    self._add_to_memory_cache(cache_key, segments, labels, trial_indices)
                    loaded_entries += 1
                    loaded_bytes += self._get_entry_size(segments, labels, trial_indices)

                except Exception as e:
                    logger.warning(f"Failed to preload {cache_key[:8]}: {e}")

        elapsed = time.time() - start_time
        stats = {
            "entries_loaded": loaded_entries,
            "size_gb": round(loaded_bytes / 1e9, 2),
            "time_seconds": round(elapsed, 2),
            "speed_mb_per_sec": round(loaded_bytes / 1e6 / elapsed, 1) if elapsed > 0 else 0,
        }

        logger.info(
            f"Preloaded {loaded_entries} entries ({loaded_bytes / 1e9:.2f} GB) "
            f"in {elapsed:.1f}s ({stats['speed_mb_per_sec']:.0f} MB/s)"
        )

        return stats

    # ==================== End Memory Cache Methods ====================

    def has_valid_cache(
        self,
        subject: str,
        run: int,
        task_type: str,
        config: Any,
        mat_file_path: str,
        target_classes: Optional[List[int]] = None,
    ) -> bool:
        """
        Check if valid cache exists for given parameters.

        Validates:
        1. Cache file exists
        2. Source .mat file hasn't been modified since caching
        3. Cache version matches

        Args:
            subject: Subject ID
            run: Run number
            task_type: Task type
            config: PreprocessConfig
            mat_file_path: Path to source .mat file
            target_classes: Target classes

        Returns:
            True if valid cache exists
        """
        if not self.enabled:
            return False

        cache_key = self.get_cache_key(subject, run, task_type, config, target_classes)
        cache_path = self._get_cache_path(cache_key)

        # Check if cache file exists
        if not cache_path.exists():
            return False

        # Check metadata
        if cache_key not in self.metadata.get("entries", {}):
            return False

        entry = self.metadata["entries"][cache_key]

        # Check source file modification time
        mat_path = Path(mat_file_path)
        if mat_path.exists():
            source_mtime = mat_path.stat().st_mtime
            if source_mtime > entry.get("created_at", 0):
                logger.debug(f"Cache invalidated: source file modified ({cache_key[:8]})")
                return False

        # Check cache version
        if entry.get("version") != self.CACHE_VERSION:
            logger.debug(f"Cache invalidated: version mismatch ({cache_key[:8]})")
            return False

        return True

    def load(
        self,
        subject: str,
        run: int,
        task_type: str,
        config: Any,
        target_classes: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load preprocessed data from cache.

        Checks in-memory cache first, then falls back to disk cache.

        Args:
            subject: Subject ID
            run: Run number
            task_type: Task type
            config: PreprocessConfig
            target_classes: Target classes

        Returns:
            Tuple of (segments, labels, trial_indices)

        Raises:
            FileNotFoundError: If cache doesn't exist
            IOError: If cache file is corrupted
        """
        cache_key = self.get_cache_key(subject, run, task_type, config, target_classes)

        # Try memory cache first (fast path)
        memory_result = self._get_from_memory_cache(cache_key)
        if memory_result is not None:
            logger.debug(f"Memory cache hit: {cache_key[:8]}")
            return memory_result

        # Fall back to disk cache
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            raise FileNotFoundError(f"Cache not found: {cache_path}")

        logger.debug(f"Disk cache load: {cache_key[:8]}...")

        try:
            if self.use_h5py:
                with h5py.File(cache_path, "r") as f:
                    segments = f["segments"][:]
                    labels = f["labels"][:]
                    trial_indices = f["trial_indices"][:]
            else:
                data = np.load(cache_path)
                segments = data["segments"]
                labels = data["labels"]
                trial_indices = data["trial_indices"]

            # Add to memory cache for future access
            self._add_to_memory_cache(cache_key, segments, labels, trial_indices)

            return segments, labels, trial_indices

        except Exception as e:
            logger.error(f"Failed to load cache {cache_key[:8]}: {e}")
            # Remove corrupted cache
            self._remove_cache(cache_key)
            raise IOError(f"Cache corrupted: {e}")

    def save(
        self,
        subject: str,
        run: int,
        task_type: str,
        config: Any,
        segments: np.ndarray,
        labels: np.ndarray,
        trial_indices: np.ndarray,
        mat_file_path: str,
        target_classes: Optional[List[int]] = None,
    ) -> None:
        """
        Save preprocessed data to cache.

        Args:
            subject: Subject ID
            run: Run number
            task_type: Task type
            config: PreprocessConfig
            segments: Preprocessed segments array
            labels: Labels array
            trial_indices: Trial indices array
            mat_file_path: Path to source .mat file
            target_classes: Target classes
        """
        if not self.enabled:
            return

        cache_key = self.get_cache_key(subject, run, task_type, config, target_classes)
        cache_path = self._get_cache_path(cache_key)

        logger.debug(
            f"Saving to cache: {cache_key[:8]} "
            f"(segments: {segments.shape}, {segments.nbytes / 1e6:.1f} MB)"
        )

        try:
            # Use per-file lock to allow parallel writes to different files
            # but prevent concurrent writes to the same file
            file_lock_key = str(cache_path)

            # Get or create file-specific lock
            with self._lock:
                if not hasattr(self, '_file_locks'):
                    self._file_locks = {}
                if file_lock_key not in self._file_locks:
                    self._file_locks[file_lock_key] = threading.Lock()
                file_lock = self._file_locks[file_lock_key]

            # Write with file-specific lock
            with file_lock:
                # Double-check if cache already exists
                if cache_path.exists():
                    logger.debug(f"Cache {cache_key[:8]} already exists, skipping")
                    return

                if self.use_h5py:
                    with h5py.File(cache_path, "w") as f:
                        # Parse compression setting
                        if self.compression is None or self.compression == "none":
                            # No compression
                            f.create_dataset("segments", data=segments)
                        elif self.compression == "lzf":
                            # Fast lzf compression (default)
                            f.create_dataset("segments", data=segments, compression="lzf")
                        elif self.compression.startswith("gzip"):
                            # gzip compression: "gzip" or "gzip:N"
                            if ":" in self.compression:
                                level = int(self.compression.split(":")[1])
                            else:
                                level = 4  # default gzip level
                            f.create_dataset(
                                "segments", data=segments,
                                compression="gzip", compression_opts=level
                            )
                        else:
                            # Fallback to lzf
                            f.create_dataset("segments", data=segments, compression="lzf")

                        f.create_dataset("labels", data=labels)
                        f.create_dataset("trial_indices", data=trial_indices)

                        # Store metadata in HDF5 as well
                        f.attrs["subject"] = subject
                        f.attrs["run"] = run
                        f.attrs["task_type"] = task_type
                        f.attrs["version"] = self.CACHE_VERSION
                else:
                    np.savez_compressed(
                        cache_path,
                        segments=segments,
                        labels=labels,
                        trial_indices=trial_indices
                    )

            # Update metadata index (thread-safe)
            with self._lock:
                self.metadata.setdefault("entries", {})[cache_key] = {
                    "subject": subject,
                    "run": run,
                    "task_type": task_type,
                    "source_file": str(mat_file_path),
                    "created_at": time.time(),
                    "version": self.CACHE_VERSION,
                    "shape": list(segments.shape),
                    "size_mb": round(segments.nbytes / 1e6, 2),
                }
                self._save_metadata()

        except Exception as e:
            logger.error(f"Failed to save cache {cache_key[:8]}: {e}")
            # Clean up partial file (thread-safe)
            with self._lock:
                if cache_path.exists():
                    try:
                        cache_path.unlink()
                    except PermissionError:
                        pass  # File may be locked by another thread

    def _remove_cache(self, cache_key: str) -> None:
        """Remove a cache entry (thread-safe)."""
        cache_path = self._get_cache_path(cache_key)
        with self._lock:
            if cache_path.exists():
                try:
                    cache_path.unlink()
                except PermissionError:
                    pass
            if cache_key in self.metadata.get("entries", {}):
                del self.metadata["entries"][cache_key]
                self._save_metadata()

    def clear_all(self) -> int:
        """
        Clear all cached data.

        Returns:
            Number of cache files removed
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.h5"):
            cache_file.unlink()
            count += 1
        for cache_file in self.cache_dir.glob("*.npz"):
            cache_file.unlink()
            count += 1

        self.metadata = {"version": self.CACHE_VERSION, "entries": {}}
        self._save_metadata()

        logger.info(f"Cleared {count} cache files")
        return count

    def clear_subject(self, subject: str) -> int:
        """
        Clear cache for a specific subject.

        Args:
            subject: Subject ID to clear

        Returns:
            Number of cache files removed
        """
        count = 0
        entries_to_remove = []

        for cache_key, entry in self.metadata.get("entries", {}).items():
            if entry.get("subject") == subject:
                entries_to_remove.append(cache_key)

        for cache_key in entries_to_remove:
            self._remove_cache(cache_key)
            count += 1

        logger.info(f"Cleared {count} cache files for subject {subject}")
        return count

    def get_stats(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            Dict with cache statistics
        """
        entries = self.metadata.get("entries", {})

        total_size_mb = sum(e.get("size_mb", 0) for e in entries.values())
        subjects = set(e.get("subject") for e in entries.values())

        return {
            "total_entries": len(entries),
            "total_size_mb": round(total_size_mb, 2),
            "subjects_cached": sorted(subjects),
            "cache_dir": str(self.cache_dir),
            "format": "h5" if self.use_h5py else "npz",
        }


# Global cache instance (lazy initialization)
_global_cache: Optional[PreprocessingCache] = None


def get_cache(cache_dir: str = "caches/preprocessed", enabled: bool = True) -> PreprocessingCache:
    """
    Get or create global cache instance.

    Args:
        cache_dir: Cache directory
        enabled: Whether caching is enabled

    Returns:
        PreprocessingCache instance
    """
    global _global_cache

    if _global_cache is None:
        _global_cache = PreprocessingCache(cache_dir=cache_dir, enabled=enabled)

    return _global_cache


def clear_global_cache() -> None:
    """Clear the global cache instance."""
    global _global_cache
    _global_cache = None
