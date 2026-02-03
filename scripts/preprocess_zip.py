#!/usr/bin/env python
"""
ZIP to Preprocessed Cache Converter for EEG-BCI Project.

This script extracts subject zip files and generates preprocessing caches
for both EEGNet and CBraMod models across binary and ternary classification tasks.

Supports both Motor Imagery (MI) and Motor Execution (ME) paradigms.

For each subject zip file, it generates 4 cache sets per paradigm:
1. EEGNet + Binary (Thumb vs Pinky)
2. EEGNet + Ternary (Thumb vs Index vs Pinky)
3. CBraMod + Binary
4. CBraMod + Ternary

Cache System (v3.0):
- Stores trials (not segments) for ~6.6x size reduction
- Offline data: cached once with all 4 fingers, filtered at load time
- Online data: cached per target_classes (2class/3class)
- Sliding window applied on cache load via trials_to_segments()

Directory structure (after unzipping distribution package):
    <parent_folder>/                   # Parent of package (--zip-dir AND --data-root default)
    ├── S01.zip, S02.zip, ...          # Subject ZIP files here
    ├── S01/                           # Extracted subject folders (parallel to package)
    ├── S02/
    ├── ...
    │
    └── eeg_bci_preprocess_package/    # Unzipped distribution package
        ├── data/
        │   └── biosemi128.ELC         # Electrode file (included in package)
        ├── caches/preprocessed/       # Generated caches (--cache-dir)
        └── scripts/

By default:
- Looks for .zip files in PARENT folder of package
- Extracts to PARENT folder of package (parallel to the package itself)
- Electrode file (biosemi128.ELC) is included in the package's data/ folder
- Deletes extracted files after caching to save storage
- Processes Motor Imagery paradigm

Usage:
    # Default: process all zips in package root, Motor Imagery paradigm
    uv run python scripts/preprocess_zip.py

    # Process Motor Execution paradigm
    uv run python scripts/preprocess_zip.py --paradigm movement

    # Process specific zip files
    uv run python scripts/preprocess_zip.py data/S01.zip data/S02.zip

    # Keep extracted files (don't delete after caching)
    uv run python scripts/preprocess_zip.py --keep-extracted

    # Extract only (no preprocessing, keeps files)
    uv run python scripts/preprocess_zip.py --extract-only

    # Preprocess only (assume already extracted)
    uv run python scripts/preprocess_zip.py --subject S01 --preprocess-only

    # Force re-preprocessing (ignore existing caches)
    uv run python scripts/preprocess_zip.py --force

    # Process only specific models/tasks
    uv run python scripts/preprocess_zip.py --models cbramod --tasks binary
"""

import argparse
import logging
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.data_loader import (
    FingerEEGDataset,
    PreprocessConfig,
    discover_available_subjects,
    get_session_folders_for_split,
)
from src.preprocessing.cache_manager import get_cache


# Paradigm configuration (aligned with run_full_comparison.py)
PARADIGM_CONFIG = {
    'imagery': {
        'description': 'Motor Imagery (MI)',
        'offline_folder': 'OfflineImagery',
        'online_prefix': 'OnlineImagery',
    },
    'movement': {
        'description': 'Motor Execution (ME)',
        'offline_folder': 'OfflineMovement',
        'online_prefix': 'OnlineMovement',
    },
}


# Custom cyan formatter for this script
class CyanFormatter(logging.Formatter):
    """Formatter that outputs cyan-colored logs."""
    CYAN = '\033[96m'
    RESET = '\033[0m'
    DIM = '\033[2m'

    def format(self, record):
        time_str = self.formatTime(record, '%H:%M:%S')
        level = record.levelname[:4]
        msg = record.getMessage()
        return f"{self.CYAN}{time_str} {level}:{self.DIM}[preprocess]{self.RESET}{self.CYAN} {msg}{self.RESET}"


def setup_logging():
    """Set up logging with cyan formatter."""
    handler = logging.StreamHandler()
    handler.setFormatter(CyanFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    root_logger.addHandler(handler)

    return logging.getLogger(__name__)


def _is_macos_metadata(name: str) -> bool:
    """Check if a zip entry is macOS metadata (should be skipped)."""
    # Skip __MACOSX/ directory and ._ resource fork files
    parts = Path(name).parts
    return (
        '__MACOSX' in parts or
        any(p.startswith('._') for p in parts) or
        name.startswith('._')
    )


def _find_7zip() -> Optional[str]:
    """
    Find 7-Zip executable on the system.

    Returns:
        Path to 7z executable, or None if not found
    """
    # Common 7-Zip locations
    candidates = [
        '7z',  # In PATH
        '7za',  # Standalone version
        r'C:\Program Files\7-Zip\7z.exe',
        r'C:\Program Files (x86)\7-Zip\7z.exe',
        '/usr/bin/7z',
        '/usr/local/bin/7z',
        '/opt/homebrew/bin/7z',
    ]

    for candidate in candidates:
        try:
            result = subprocess.run(
                [candidate, '--help'],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    return None


def _extract_with_7zip(
    zip_path: Path, data_root: Path, logger: logging.Logger
) -> Tuple[Optional[str], bool]:
    """
    Extract a ZIP file using 7-Zip.

    Args:
        zip_path: Path to the zip file
        data_root: Data root directory
        logger: Logger instance

    Returns:
        Tuple of (subject_id, was_freshly_extracted)
        subject_id is None if extraction failed
    """
    # Try to determine subject ID from zip contents first (more reliable than filename)
    subject_id_from_zip = None
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            entries = [e for e in zf.namelist() if not _is_macos_metadata(e)]
            if entries:
                # Extract subject ID from first real entry (e.g., "S01/OfflineImagery/...")
                subject_id_from_zip = Path(entries[0]).parts[0]
    except Exception:
        # Fall back to filename if we can't read zip
        pass

    # Use zip contents subject ID if available, otherwise use filename
    subject_id_to_check = subject_id_from_zip or zip_path.stem
    subject_dir = data_root / subject_id_to_check

    # Check if subject is already extracted with expected subdirectories
    if subject_dir.exists() and any((subject_dir / folder).exists()
                                     for folder in ['OfflineImagery', 'OfflineMovement']):
        logger.info(f"  Subject {subject_id_to_check} already extracted, using existing")
        return subject_id_to_check, False

    try:
        # Extract to data_root
        # -y: assume Yes on all queries
        # -o: output directory
        # -x!__MACOSX/*: exclude macOS metadata
        # -x!*/.DS_Store: exclude .DS_Store files
        result = subprocess.run(
            [
                _find_7zip(), 'x', str(zip_path),
                f'-o{data_root}',
                '-y',
                '-x!__MACOSX/*',
                '-x!*/.DS_Store',
                '-x!*/._*',
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode != 0:
            logger.error(f"  7-Zip extraction failed: {result.stderr}")
            return None, False

        # Find subject ID from extracted contents
        # Look for S## directories
        for item in data_root.iterdir():
            if item.is_dir() and item.name.startswith('S') and len(item.name) == 3:
                # Verify it has expected subdirectories
                if any((item / folder).exists() for folder in ['OfflineImagery', 'OfflineMovement']):
                    # Count extracted files
                    file_count = sum(1 for _ in item.rglob('*') if _.is_file())
                    logger.info(f"  Extracted {item.name} to {data_root} ({file_count} files)")
                    return item.name, True

        # If we can't find a subject directory, try to get it from zip name
        subject_id = zip_path.stem  # S01 from S01.zip
        subject_dir = data_root / subject_id
        if subject_dir.exists():
            file_count = sum(1 for _ in subject_dir.rglob('*') if _.is_file())
            logger.info(f"  Extracted {subject_id} to {data_root} ({file_count} files)")
            return subject_id, True

        logger.error("  Could not determine subject ID from extracted contents")
        return None, False

    except subprocess.TimeoutExpired:
        logger.error("  7-Zip extraction timed out")
        return None, False
    except Exception as e:
        logger.error(f"  7-Zip extraction error: {e}")
        return None, False


def extract_zip(zip_path: Path, data_root: Path, logger: logging.Logger) -> Tuple[Optional[str], bool]:
    """
    Extract a subject zip file to the data directory.

    Uses 7-Zip by default (supports more compression methods like Deflate64, LZMA).
    Falls back to Python's zipfile if 7-Zip is not available.

    Args:
        zip_path: Path to the zip file
        data_root: Data root directory
        logger: Logger instance

    Returns:
        Tuple of (subject_id, was_freshly_extracted)
        subject_id is None if extraction failed
    """
    if not zip_path.exists():
        logger.error(f"Zip file not found: {zip_path}")
        return None, False

    logger.info(f"Extracting {zip_path.name}...")

    # Try 7-Zip first (supports more compression methods)
    seven_zip = _find_7zip()
    if seven_zip:
        subject_id, was_fresh = _extract_with_7zip(zip_path, data_root, logger)
        if subject_id:
            return subject_id, was_fresh
        # 7-Zip failed, but don't fall back - it's likely a real error
        return None, False

    # Fall back to Python's zipfile (only if 7-Zip not available)
    logger.info("  7-Zip not found, using Python zipfile...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Get all entries, filtering out macOS metadata
            entries = [e for e in zf.namelist() if not _is_macos_metadata(e)]
            if not entries:
                logger.error(f"Empty zip file (or only macOS metadata): {zip_path}")
                return None, False

            # Get subject ID from first real entry
            first_entry = entries[0]
            # Use Path for cross-platform compatibility
            subject_id = Path(first_entry).parts[0]

            # Check if already extracted
            subject_dir = data_root / subject_id
            if subject_dir.exists():
                logger.info(f"  Subject {subject_id} already extracted, using existing")
                return subject_id, False

            # Extract only non-metadata files
            for member in entries:
                zf.extract(member, data_root)
            logger.info(f"  Extracted {subject_id} to {data_root} ({len(entries)} files)")

            return subject_id, True

    except zipfile.BadZipFile:
        logger.error(f"Invalid zip file: {zip_path}")
        return None, False
    except NotImplementedError as e:
        # Unsupported compression method (e.g., Deflate64, LZMA)
        logger.error(f"Unsupported compression in {zip_path}: {e}")
        logger.error("  Please install 7-Zip: https://www.7-zip.org/download.html")
        return None, False
    except Exception as e:
        logger.error(f"Error extracting {zip_path}: {e}")
        return None, False


def delete_extracted_folder(subject_id: str, data_root: Path, logger: logging.Logger) -> bool:
    """
    Delete an extracted subject folder.

    Args:
        subject_id: Subject ID (e.g., 'S01')
        data_root: Data root directory
        logger: Logger instance

    Returns:
        True if deleted successfully
    """
    subject_dir = data_root / subject_id
    if not subject_dir.exists():
        return True

    try:
        shutil.rmtree(subject_dir)
        logger.info(f"Deleted extracted folder: {subject_dir}")
        return True
    except Exception as e:
        logger.error(f"Error deleting {subject_dir}: {e}")
        return False


def get_all_session_folders_for_task(
    subject_dir: Path,
    paradigm: str,
    task: str
) -> List[str]:
    """
    Get all session folders for a given paradigm and task type.

    Combines train and test folders to cache all relevant data.
    Uses get_session_folders_for_split from data_loader.py for consistency.

    Args:
        subject_dir: Path to subject directory
        paradigm: 'imagery' or 'movement'
        task: Task type ('binary', 'ternary', 'quaternary')

    Returns:
        List of available session folder names
    """
    # Get both train and test folders
    train_folders = get_session_folders_for_split(paradigm, task, 'train')
    test_folders = get_session_folders_for_split(paradigm, task, 'test')

    # Combine and deduplicate
    all_patterns = list(dict.fromkeys(train_folders + test_folders))

    # Filter to existing folders
    available = []
    for pattern in all_patterns:
        folder = subject_dir / pattern
        if folder.exists():
            available.append(pattern)

    return available


def generate_caches(
    subject_id: str,
    data_root: Path,
    cache_dir: Path,
    logger: logging.Logger,
    paradigm: str = 'imagery',
    force: bool = False,
    tasks: List[str] = None,
    models: List[str] = None,
) -> dict:
    """
    Generate preprocessing caches for a subject.

    Args:
        subject_id: Subject ID (e.g., 'S01')
        data_root: Data root directory
        cache_dir: Cache directory
        logger: Logger instance
        paradigm: 'imagery' or 'movement'
        force: Force re-preprocessing even if cache exists
        tasks: List of tasks to process ('binary', 'ternary'), or None for all
        models: List of models to process ('eegnet', 'cbramod'), or None for all

    Returns:
        Dict with cache generation statistics
    """
    subject_dir = data_root / subject_id
    if not subject_dir.exists():
        logger.error(f"Subject directory not found: {subject_dir}")
        return {'error': f"Subject directory not found: {subject_dir}"}

    # Look for electrode file in multiple locations
    # With standard mode: data_root is parent folder, ELC is in package's data/
    # With --in-place mode: data_root is also parent folder, ELC is in package's data/
    package_data_dir = cache_dir.parent.parent / 'data'  # caches/preprocessed -> package -> data
    elc_path = None
    elc_search_paths = [
        package_data_dir / 'biosemi128.ELC',  # Package's data folder (primary)
        package_data_dir / 'biosemi128.elc',
        data_root / 'biosemi128.ELC',         # Data root folder
        data_root / 'biosemi128.elc',
        data_root.parent / 'biosemi128.ELC',  # Parent of data root
        data_root.parent / 'biosemi128.elc',
    ]
    for path in elc_search_paths:
        if path.exists():
            elc_path = path
            break

    if elc_path is None:
        logger.error(f"Electrode file not found. Searched:")
        for p in elc_search_paths[:4]:  # Show first 4 paths
            logger.error(f"  - {p}")
        return {'error': f"Electrode file not found"}

    # Clear cache if force is set
    if force:
        cache = get_cache(cache_dir=str(cache_dir))
        cache.clear_subject(subject_id)
        logger.info(f"Cleared existing cache for {subject_id}")

    stats = {
        'subject': subject_id,
        'paradigm': paradigm,
        'caches_generated': [],
        'segments_total': 0,
        'time_seconds': 0,
    }

    # Model and task configurations
    # Note: PreprocessConfig.paper_aligned() doesn't use n_class parameter
    all_configs = [
        ('eegnet', 'binary', [1, 4], PreprocessConfig.paper_aligned()),
        ('eegnet', 'ternary', [1, 2, 4], PreprocessConfig.paper_aligned()),
        ('cbramod', 'binary', [1, 4], PreprocessConfig.for_cbramod()),
        ('cbramod', 'ternary', [1, 2, 4], PreprocessConfig.for_cbramod()),
    ]

    # Filter configs based on tasks and models arguments
    configs = []
    for model, task, target_classes, config in all_configs:
        # Filter by tasks
        if tasks and 'all' not in tasks and task not in tasks:
            continue
        # Filter by models
        if models and 'all' not in models and model not in models:
            continue
        configs.append((model, task, target_classes, config))

    if not configs:
        logger.warning("No configurations to process after filtering")
        return stats

    start_time = time.time()
    paradigm_desc = PARADIGM_CONFIG[paradigm]['description']

    for model, task, target_classes, config in configs:
        logger.info(f"Generating cache: {subject_id} / {paradigm_desc} / {model} / {task}")

        # Get available session folders for this paradigm and task
        session_folders = get_all_session_folders_for_task(subject_dir, paradigm, task)

        if not session_folders:
            logger.warning(f"  No session folders found for {paradigm}/{task}, skipping")
            continue

        logger.info(f"  Session folders: {session_folders}")

        try:
            # Create dataset - this triggers cache generation
            # v3.0: Caches trials (not segments), ~6.6x size reduction
            dataset = FingerEEGDataset(
                data_root=str(data_root),
                subjects=[subject_id],
                config=config,
                session_folders=session_folders,
                target_classes=target_classes,
                elc_path=str(elc_path),
                use_cache=True,
                cache_dir=str(cache_dir),
                parallel_workers=0,  # Auto-detect
            )

            n_segments = len(dataset)
            cache_hits = getattr(dataset, 'n_cache_hits', 0)
            cache_misses = getattr(dataset, 'n_cache_misses', 0)

            if cache_misses > 0:
                logger.info(f"  Generated {n_segments} segments ({cache_misses} runs cached)")
            else:
                logger.info(f"  Loaded {n_segments} segments (cache hit)")

            stats['caches_generated'].append({
                'model': model,
                'task': task,
                'segments': n_segments,
                'sessions': session_folders,
                'cache_hits': cache_hits,
                'cache_misses': cache_misses,
            })
            stats['segments_total'] += n_segments

        except Exception as e:
            logger.error(f"  Error generating cache: {e}")
            traceback.print_exc()
            stats['caches_generated'].append({
                'model': model,
                'task': task,
                'error': str(e),
            })

    stats['time_seconds'] = round(time.time() - start_time, 2)
    logger.info(f"Completed {subject_id}: {stats['segments_total']} total segments in {stats['time_seconds']}s")

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Convert ZIP files to preprocessed caches for EEG-BCI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('Usage:')[1] if 'Usage:' in __doc__ else '',
    )

    parser.add_argument(
        'zip_files',
        nargs='*',
        type=Path,
        help='ZIP file(s) to process. If not specified, processes all .zip in --zip-dir',
    )

    parser.add_argument(
        '--subject',
        type=str,
        help='Subject ID to preprocess (use with --preprocess-only)',
    )

    parser.add_argument(
        '--zip-dir',
        type=Path,
        default=PROJECT_ROOT.parent,
        help='Directory containing ZIP files (default: parent of package, i.e., Downloads folder)',
    )

    parser.add_argument(
        '--data-root',
        type=Path,
        default=PROJECT_ROOT.parent,
        help='Directory for extracted data (default: parent of package, same as --zip-dir)',
    )

    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=PROJECT_ROOT / 'caches' / 'preprocessed',
        help='Cache directory (default: caches/preprocessed/)',
    )

    parser.add_argument(
        '--extract-only',
        action='store_true',
        help='Only extract ZIP files, skip preprocessing (keeps extracted files)',
    )

    parser.add_argument(
        '--preprocess-only',
        action='store_true',
        help='Only preprocess (assume data already extracted in data/ subfolder)',
    )

    parser.add_argument(
        '--in-place',
        action='store_true',
        help='In-place mode: package is inside data folder (S01/, S02/ are siblings of scripts/)',
    )

    parser.add_argument(
        '--keep-extracted',
        action='store_true',
        help='Keep extracted files after caching (default: delete to save storage)',
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-preprocessing (clear existing cache)',
    )

    parser.add_argument(
        '--tasks',
        nargs='+',
        choices=['binary', 'ternary', 'all'],
        default=['all'],
        help='Task types to preprocess (default: all)',
    )

    parser.add_argument(
        '--models',
        nargs='+',
        choices=['eegnet', 'cbramod', 'all'],
        default=['all'],
        help='Models to preprocess (default: all)',
    )

    parser.add_argument(
        '--paradigm',
        type=str,
        default='imagery',
        choices=['imagery', 'movement'],
        help='Experiment paradigm: imagery (MI) or movement (ME) (default: imagery)',
    )

    args = parser.parse_args()

    logger = setup_logging()

    paradigm_desc = PARADIGM_CONFIG[args.paradigm]['description']

    logger.info("=" * 60)
    logger.info("ZIP to Preprocessed Cache Converter")
    logger.info(f"Paradigm: {paradigm_desc}")
    if args.in_place:
        logger.info("Mode: IN-PLACE (package inside data folder)")
    elif args.preprocess_only:
        logger.info("Mode: PREPROCESS-ONLY (data already extracted)")
    else:
        logger.info("Mode: EXTRACT + PREPROCESS")
    logger.info("=" * 60)
    if not args.in_place:
        logger.info(f"ZIP dir:    {args.zip_dir}")
    logger.info(f"Data root:  {args.data_root}")
    logger.info(f"Cache dir:  {args.cache_dir}")
    logger.info("-" * 60)

    # Ensure directories exist
    args.data_root.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    subjects_to_process = []
    freshly_extracted = set()  # Track which subjects were freshly extracted

    # Handle --in-place mode: subjects are in the package's data/ directory
    # Directory structure: <package>/data/S01/, <package>/data/S02/, ...
    if args.in_place:
        # In this mode, data_root is the package's data/ directory
        args.data_root = PROJECT_ROOT / 'data'
        logger.info(f"In-place mode: using {args.data_root} as data root")

        if args.subject:
            subjects_to_process = [args.subject]
        else:
            # Discover subjects in the parent directory
            subjects_to_process = discover_available_subjects(
                str(args.data_root),
                paradigm=args.paradigm,
                task='binary',
            )

        if not subjects_to_process:
            logger.error(f"No subjects found in {args.data_root} for {paradigm_desc}")
            logger.error("Expected structure: parent_folder/S01/, parent_folder/S02/, ...")
            return 1

        logger.info(f"Found {len(subjects_to_process)} subjects: {subjects_to_process}")

    # Handle preprocess-only mode
    elif args.preprocess_only:
        if args.subject:
            subjects_to_process = [args.subject]
        else:
            # Find all available subjects for this paradigm
            # Use 'binary' as default task since most subjects have binary data
            subjects_to_process = discover_available_subjects(
                str(args.data_root),
                paradigm=args.paradigm,
                task='binary',
            )

        if not subjects_to_process:
            logger.error(f"No subjects found in data directory for {paradigm_desc}")
            return 1

        logger.info(f"Preprocessing {len(subjects_to_process)} subjects: {subjects_to_process}")

    else:
        # Find zip files to process
        zip_files = args.zip_files
        if not zip_files:
            # Default: find all .zip files in zip_dir (package root)
            # Filter out hidden files (macOS creates ._* and .* files)
            zip_files = [
                f for f in args.zip_dir.glob('*.zip')
                if not f.name.startswith('.')
            ]
            if zip_files:
                logger.info(f"Found {len(zip_files)} ZIP files in {args.zip_dir}")
            else:
                logger.error(f"No ZIP files found in {args.zip_dir}")
                logger.error(f"Place S*.zip files in: {args.zip_dir}")
                return 1

        # Extract ZIP files
        for zip_path in zip_files:
            if not zip_path.exists():
                logger.warning(f"ZIP file not found: {zip_path}")
                continue

            subject_id, was_fresh = extract_zip(zip_path, args.data_root, logger)
            if subject_id:
                if was_fresh:
                    freshly_extracted.add(subject_id)
                if not args.extract_only:
                    subjects_to_process.append(subject_id)

        if args.extract_only:
            logger.info("Extraction complete (--extract-only specified)")
            return 0

    # Generate caches
    if not subjects_to_process:
        logger.warning("No subjects to process")
        return 0

    all_stats = []
    total_start = time.time()

    for subject_id in subjects_to_process:
        stats = generate_caches(
            subject_id=subject_id,
            data_root=args.data_root,
            cache_dir=args.cache_dir,
            logger=logger,
            paradigm=args.paradigm,
            force=args.force,
            tasks=args.tasks,
            models=args.models,
        )
        all_stats.append(stats)

        # Delete extracted folder if:
        # 1. Not in --keep-extracted mode
        # 2. Not in --preprocess-only mode (user explicitly wants to keep)
        # 3. Was freshly extracted by this script
        # 4. No errors occurred during cache generation
        if not args.keep_extracted and not args.preprocess_only and subject_id in freshly_extracted:
            if 'error' not in stats:
                delete_extracted_folder(subject_id, args.data_root, logger)
            else:
                logger.warning(f"Keeping {subject_id} due to errors during cache generation")

    # Summary
    total_time = time.time() - total_start
    total_segments = sum(s.get('segments_total', 0) for s in all_stats)
    errors = [s for s in all_stats if 'error' in s]

    logger.info("")
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"Paradigm: {paradigm_desc}")
    logger.info(f"Subjects processed: {len(subjects_to_process)}")
    logger.info(f"Total segments: {total_segments:,}")
    if total_time >= 60:
        logger.info(f"Total time: {total_time/60:.1f}m")
    else:
        logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Cache directory: {args.cache_dir}")

    # Show cache stats
    cache = get_cache(cache_dir=str(args.cache_dir))
    cache_stats = cache.get_stats()
    logger.info(f"Cache entries: {cache_stats.get('total_entries', 0)}")
    logger.info(f"Cache size: {cache_stats.get('total_size_mb', 0):.1f} MB")

    # Show per-subject breakdown
    subjects_cached = cache_stats.get('subjects_cached', [])
    if subjects_cached:
        # Count entries per subject from metadata
        entries = cache.metadata.get('entries', {})
        by_subject = {}
        for entry in entries.values():
            subj = entry.get('subject', 'unknown')
            by_subject[subj] = by_subject.get(subj, 0) + 1

        logger.info(f"Per-subject cache counts:")
        for subj in sorted(by_subject.keys()):
            logger.info(f"  {subj}: {by_subject[subj]} entries")

    if freshly_extracted and not args.keep_extracted and not args.preprocess_only:
        logger.info(f"Cleaned up {len(freshly_extracted)} extracted folder(s) to save storage")

    if errors:
        logger.warning(f"Errors occurred for {len(errors)} subject(s)")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
