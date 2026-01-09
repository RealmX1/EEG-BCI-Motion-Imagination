#!/usr/bin/env python
"""
ZIP to Preprocessed Cache Converter for EEG-BCI Project.

This script extracts subject zip files and generates preprocessing caches
for both EEGNet and CBraMod models across binary and ternary classification tasks.

For each subject zip file, it generates 4 cache sets:
1. EEGNet + Binary (Thumb vs Pinky)
2. EEGNet + Ternary (Thumb vs Index vs Pinky)
3. CBraMod + Binary
4. CBraMod + Ternary

By default:
- Processes all .zip files in data/ directory
- Deletes extracted files after caching to save storage

Usage:
    # Default: process all zips in data/, delete extracted files after caching
    uv run python scripts/preprocess_zip.py

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
"""

import argparse
import logging
import shutil
import sys
import time
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
)
from src.preprocessing.cache_manager import get_cache


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


def extract_zip(zip_path: Path, data_root: Path, logger: logging.Logger) -> Tuple[Optional[str], bool]:
    """
    Extract a subject zip file to the data directory.

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

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Get subject ID from first entry
            entries = zf.namelist()
            if not entries:
                logger.error(f"Empty zip file: {zip_path}")
                return None, False
            first_entry = entries[0]
            # Use Path for cross-platform compatibility
            subject_id = Path(first_entry).parts[0]

            # Check if already extracted
            subject_dir = data_root / subject_id
            if subject_dir.exists():
                logger.info(f"  Subject {subject_id} already extracted, using existing")
                return subject_id, False

            # Extract to data root
            zf.extractall(data_root)
            logger.info(f"  Extracted {subject_id} to {data_root}")

            return subject_id, True

    except zipfile.BadZipFile:
        logger.error(f"Invalid zip file: {zip_path}")
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


def get_session_folders_for_task(subject_dir: Path, task: str) -> List[str]:
    """
    Get available session folders for a given task type.

    Args:
        subject_dir: Path to subject directory
        task: Task type ('binary', 'ternary', 'quaternary')

    Returns:
        List of session folder names
    """
    # Define expected folders for each task
    if task == 'binary':
        patterns = [
            'OfflineImagery',
            'OnlineImagery_Sess01_2class_Base',
            'OnlineImagery_Sess01_2class_Finetune',
            'OnlineImagery_Sess02_2class_Base',
            'OnlineImagery_Sess02_2class_Finetune',
        ]
    elif task == 'ternary':
        patterns = [
            'OfflineImagery',
            'OnlineImagery_Sess01_3class_Base',
            'OnlineImagery_Sess01_3class_Finetune',
            'OnlineImagery_Sess02_3class_Base',
            'OnlineImagery_Sess02_3class_Finetune',
        ]
    elif task == 'quaternary':
        # Quaternary only uses Offline data
        patterns = ['OfflineImagery']
    else:
        raise ValueError(f"Unknown task: {task}")

    # Filter to existing folders
    available = []
    for pattern in patterns:
        folder = subject_dir / pattern
        if folder.exists():
            available.append(pattern)

    return available


def generate_caches(
    subject_id: str,
    data_root: Path,
    cache_dir: Path,
    logger: logging.Logger,
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

    elc_path = data_root / 'biosemi128.ELC'
    if not elc_path.exists():
        # Try lowercase
        elc_path = data_root / 'biosemi128.elc'

    if not elc_path.exists():
        logger.error(f"Electrode file not found: {elc_path}")
        return {'error': f"Electrode file not found"}

    # Clear cache if force is set
    if force:
        cache = get_cache(cache_dir=str(cache_dir))
        cache.clear_subject(subject_id)
        logger.info(f"Cleared existing cache for {subject_id}")

    stats = {
        'subject': subject_id,
        'caches_generated': [],
        'segments_total': 0,
        'time_seconds': 0,
    }

    # Model and task configurations
    all_configs = [
        ('eegnet', 'binary', [1, 4], PreprocessConfig.paper_aligned(n_class=2)),
        ('eegnet', 'ternary', [1, 2, 4], PreprocessConfig.paper_aligned(n_class=3)),
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

    start_time = time.time()

    for model, task, target_classes, config in configs:
        logger.info(f"Generating cache: {subject_id} / {model} / {task}")

        # Get available session folders
        session_folders = get_session_folders_for_task(subject_dir, task)

        if not session_folders:
            logger.warning(f"  No session folders found for {task}, skipping")
            continue

        logger.info(f"  Session folders: {session_folders}")

        try:
            # Create dataset - this triggers cache generation
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
            logger.info(f"  Generated {n_segments} segments")

            stats['caches_generated'].append({
                'model': model,
                'task': task,
                'segments': n_segments,
                'sessions': session_folders,
            })
            stats['segments_total'] += n_segments

        except Exception as e:
            logger.error(f"  Error generating cache: {e}")
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
        help='ZIP file(s) to process. If not specified, processes all .zip in data/',
    )

    parser.add_argument(
        '--subject',
        type=str,
        help='Subject ID to preprocess (use with --preprocess-only)',
    )

    parser.add_argument(
        '--data-root',
        type=Path,
        default=PROJECT_ROOT / 'data',
        help='Data root directory (default: data/)',
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
        help='Only preprocess (assume data already extracted)',
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

    args = parser.parse_args()

    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("ZIP to Preprocessed Cache Converter")
    logger.info("=" * 60)

    # Ensure directories exist
    args.data_root.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    subjects_to_process = []
    freshly_extracted = set()  # Track which subjects were freshly extracted

    # Handle preprocess-only mode
    if args.preprocess_only:
        if args.subject:
            subjects_to_process = [args.subject]
        else:
            # Find all available subjects
            subjects_to_process = discover_available_subjects(str(args.data_root))

        if not subjects_to_process:
            logger.error("No subjects found in data directory")
            return 1

        logger.info(f"Preprocessing {len(subjects_to_process)} subjects: {subjects_to_process}")

    else:
        # Find zip files to process
        zip_files = args.zip_files
        if not zip_files:
            # Default: find all .zip files in data_root
            zip_files = list(args.data_root.glob('*.zip'))
            if zip_files:
                logger.info(f"Found {len(zip_files)} ZIP files in {args.data_root}")
            else:
                logger.error(f"No ZIP files found in {args.data_root}")
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

    logger.info("")
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"Subjects processed: {len(subjects_to_process)}")
    logger.info(f"Total segments: {total_segments}")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Cache directory: {args.cache_dir}")

    # Show cache stats
    cache = get_cache(cache_dir=str(args.cache_dir))
    cache_stats = cache.get_stats()
    logger.info(f"Cache entries: {cache_stats.get('total_entries', 0)}")
    logger.info(f"Cache size: {cache_stats.get('total_size_mb', 0):.1f} MB")

    if freshly_extracted and not args.keep_extracted and not args.preprocess_only:
        logger.info(f"Cleaned up {len(freshly_extracted)} extracted folder(s) to save storage")

    return 0


if __name__ == '__main__':
    sys.exit(main())
