#!/usr/bin/env python
"""
Package preprocessing caches into uploadable ZIP files.

Splits the cache directory into multiple ZIP files, each under a specified
size limit (default: 9 GB) for easy uploading to cloud storage or sharing.

Features:
- Automatic size-based splitting
- Subject range filtering (e.g., S08-S21)
- Preserves directory structure
- Includes cache index (.cache_index.json)
- Creates manifest file for verification

Usage:
    # Default: split into 9GB chunks (all subjects)
    python scripts/package_caches.py

    # Package only specific subjects
    python scripts/package_caches.py --subjects S08 S09 S10

    # Package a range of subjects (S08 to S21)
    python scripts/package_caches.py --subjects S08-S21

    # Combine range and individual subjects
    python scripts/package_caches.py --subjects S01 S03 S08-S15

    # Custom size limit (in GB)
    python scripts/package_caches.py --max-size 4.5

    # Preview without creating files
    python scripts/package_caches.py --dry-run

    # Specific output directory
    python scripts/package_caches.py --output /path/to/output
"""

import argparse
import hashlib
import json
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent


def parse_subject_range(subject_spec: str) -> List[str]:
    """
    Parse a subject specification into a list of subject IDs.

    Supports:
    - Single subject: "S01" -> ["S01"]
    - Range: "S08-S21" -> ["S08", "S09", ..., "S21"]

    Args:
        subject_spec: Subject specification string

    Returns:
        List of subject IDs
    """
    subject_spec = subject_spec.strip()

    if '-' in subject_spec and subject_spec.count('-') == 1:
        # Range format: S08-S21
        start, end = subject_spec.split('-')
        start = start.strip().upper()
        end = end.strip().upper()

        # Extract numbers
        if start.startswith('S') and end.startswith('S'):
            try:
                start_num = int(start[1:])
                end_num = int(end[1:])
                return [f"S{i:02d}" for i in range(start_num, end_num + 1)]
            except ValueError:
                pass

    # Single subject
    return [subject_spec.strip().upper()]


def expand_subjects(subjects_arg: List[str]) -> List[str]:
    """
    Expand a list of subject specifications into individual subject IDs.

    Args:
        subjects_arg: List of subject specs (e.g., ["S01", "S08-S21"])

    Returns:
        Sorted list of unique subject IDs
    """
    all_subjects = set()
    for spec in subjects_arg:
        all_subjects.update(parse_subject_range(spec))
    return sorted(all_subjects)


def get_file_hash(file_path: Path, chunk_size: int = 8192) -> str:
    """Calculate MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def get_cache_files(
    cache_dir: Path,
    subjects: Optional[List[str]] = None,
    verbose: bool = False,
) -> List[Tuple[Path, int]]:
    """
    Get cache files with their sizes, optionally filtered by subject.

    Args:
        cache_dir: Path to cache directory
        subjects: List of subject IDs to include (None = all)
        verbose: Show detailed breakdown by subject/model

    Returns:
        List of (file_path, size_bytes) tuples, sorted by size descending
    """
    files = []

    # Load cache index for subject filtering
    index_file = cache_dir / '.cache_index.json'
    cache_index = {}
    if index_file.exists():
        try:
            with open(index_file, 'r') as f:
                cache_index = json.load(f).get('entries', {})
        except (json.JSONDecodeError, IOError):
            pass

    # If verbose, show breakdown by subject and model
    if verbose and cache_index:
        # Count entries per subject
        by_subject: Dict[str, Dict[str, int]] = {}
        for entry in cache_index.values():
            subj = entry.get('subject', 'unknown')
            model_raw = entry.get('model', 'unknown')
            paradigm = entry.get('collection_paradigm', 'unknown')
            n_cls = entry.get('n_classes', 0)

            # Normalize model name (cbramod_128ch -> cbramod)
            if 'cbramod' in model_raw.lower():
                model = 'cbramod'
            elif 'eegnet' in model_raw.lower():
                model = 'eegnet'
            else:
                model = model_raw

            if subj not in by_subject:
                by_subject[subj] = {'total': 0, 'eegnet': 0, 'cbramod': 0,
                                   'offline': 0, 'online': 0}
            by_subject[subj]['total'] += 1
            by_subject[subj][model] = by_subject[subj].get(model, 0) + 1
            by_subject[subj][paradigm] = by_subject[subj].get(paradigm, 0) + 1

        print("\nCache index breakdown by subject:")
        print(f"{'Subject':<10} {'Total':<8} {'EEGNet':<8} {'CBraMod':<8} {'Offline':<8} {'Online':<8}")
        print("-" * 50)
        for subj in sorted(by_subject.keys()):
            s = by_subject[subj]
            print(f"{subj:<10} {s['total']:<8} {s.get('eegnet',0):<8} "
                  f"{s.get('cbramod',0):<8} {s.get('offline',0):<8} {s.get('online',0):<8}")
        print()

    # Build set of cache keys for selected subjects
    selected_keys = None
    if subjects:
        # Normalize to uppercase for case-insensitive matching
        subjects_set = {s.upper() for s in subjects}
        selected_keys = set()
        unmatched_subjects = set()
        for cache_key, entry in cache_index.items():
            entry_subject = entry.get('subject', '')
            # Case-insensitive comparison
            if entry_subject.upper() in subjects_set:
                selected_keys.add(cache_key)
            else:
                unmatched_subjects.add(entry_subject)
        print(f"Filtering for subjects: {sorted(subjects_set)}")
        print(f"Matched {len(selected_keys)} cache entries")
        # Show what subjects exist in cache but weren't selected
        if unmatched_subjects:
            print(f"Other subjects in cache: {sorted(unmatched_subjects)}")

    # Get all .h5 and .npz files
    all_cache_files = []
    orphaned_files = []  # Files not in index
    for pattern in ['**/*.h5', '**/*.npz']:
        for file_path in cache_dir.glob(pattern):
            if file_path.is_file():
                cache_key = file_path.stem
                all_cache_files.append(cache_key)

                # Check if file is in index
                if cache_key not in cache_index:
                    orphaned_files.append(file_path.name)

                # Filter by subject if specified
                if selected_keys is not None:
                    if cache_key not in selected_keys:
                        continue

                files.append((file_path, file_path.stat().st_size))

    # Report orphaned files (files without index entries)
    if orphaned_files and len(orphaned_files) <= 10:
        print(f"WARNING: {len(orphaned_files)} cache file(s) not in index: {orphaned_files}")
    elif orphaned_files:
        print(f"WARNING: {len(orphaned_files)} cache file(s) not in index (index may be outdated)")

    # Report index vs file count mismatch
    if len(cache_index) != len(all_cache_files):
        print(f"NOTE: Index has {len(cache_index)} entries, found {len(all_cache_files)} cache files")

    # Include cache index (always include for metadata)
    if index_file.exists():
        files.append((index_file, index_file.stat().st_size))

    # Sort by size descending (pack larger files first)
    files.sort(key=lambda x: x[1], reverse=True)

    return files


def split_files_into_chunks(
    files: List[Tuple[Path, int]],
    max_size_bytes: int
) -> List[List[Tuple[Path, int]]]:
    """
    Split files into chunks that fit within max_size_bytes.

    Uses first-fit decreasing bin packing algorithm.

    Args:
        files: List of (file_path, size_bytes) tuples
        max_size_bytes: Maximum size per chunk

    Returns:
        List of chunks, each chunk is a list of (file_path, size_bytes)
    """
    chunks: List[List[Tuple[Path, int]]] = []
    chunk_sizes: List[int] = []

    for file_path, size in files:
        if size > max_size_bytes:
            print(f"WARNING: File {file_path.name} ({size / 1e9:.2f} GB) exceeds max size!")
            # Put oversized file in its own chunk
            chunks.append([(file_path, size)])
            chunk_sizes.append(size)
            continue

        # Find first chunk that can fit this file
        placed = False
        for i, chunk_size in enumerate(chunk_sizes):
            if chunk_size + size <= max_size_bytes:
                chunks[i].append((file_path, size))
                chunk_sizes[i] += size
                placed = True
                break

        if not placed:
            # Create new chunk
            chunks.append([(file_path, size)])
            chunk_sizes.append(size)

    return chunks


def create_manifest(
    chunks: List[List[Tuple[Path, int]]],
    cache_dir: Path,
    output_dir: Path,
) -> Dict:
    """Create manifest with file information for verification."""
    manifest = {
        'created_at': datetime.now().isoformat(),
        'cache_dir': str(cache_dir),
        'total_files': sum(len(chunk) for chunk in chunks),
        'total_chunks': len(chunks),
        'chunks': []
    }

    for i, chunk in enumerate(chunks):
        chunk_info = {
            'part': i + 1,
            'zip_name': f'eeg_bci_cache_part{i+1:02d}.zip',
            'files': []
        }

        total_size = 0
        for file_path, size in chunk:
            rel_path = file_path.relative_to(cache_dir)
            chunk_info['files'].append({
                'path': str(rel_path),
                'size_bytes': size,
                'size_mb': round(size / 1e6, 2),
            })
            total_size += size

        chunk_info['total_size_bytes'] = total_size
        chunk_info['total_size_gb'] = round(total_size / 1e9, 2)
        manifest['chunks'].append(chunk_info)

    manifest['total_size_gb'] = round(
        sum(c['total_size_bytes'] for c in manifest['chunks']) / 1e9, 2
    )

    return manifest


def package_caches(
    cache_dir: Path,
    output_dir: Path,
    max_size_gb: float = 9.0,
    dry_run: bool = False,
    compression: int = zipfile.ZIP_DEFLATED,
    subjects: Optional[List[str]] = None,
    verbose: bool = False,
) -> int:
    """
    Package cache files into split ZIP archives.

    Args:
        cache_dir: Path to cache directory
        output_dir: Path to output directory
        max_size_gb: Maximum size per ZIP in GB
        dry_run: If True, only preview without creating files
        compression: ZIP compression method
        subjects: List of subject IDs to include (None = all)

    Returns:
        0 on success, 1 on error
    """
    if not cache_dir.exists():
        print(f"ERROR: Cache directory not found: {cache_dir}")
        return 1

    max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)

    print(f"Cache directory: {cache_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Max size per ZIP: {max_size_gb} GB")
    if subjects:
        print(f"Subjects filter: {subjects}")
    print()

    # Get cache files (filtered by subjects if specified)
    files = get_cache_files(cache_dir, subjects=subjects, verbose=verbose)

    if not files:
        print("No cache files found!")
        return 1

    total_size = sum(size for _, size in files)
    print(f"Found {len(files)} files ({total_size / 1e9:.2f} GB)")

    # Split into chunks
    chunks = split_files_into_chunks(files, max_size_bytes)
    print(f"Split into {len(chunks)} ZIP file(s)")
    print()

    # Create manifest
    manifest = create_manifest(chunks, cache_dir, output_dir)

    # Preview
    print("=" * 60)
    print("Package Plan")
    print("=" * 60)

    for chunk_info in manifest['chunks']:
        print(f"\n{chunk_info['zip_name']} ({chunk_info['total_size_gb']:.2f} GB)")
        print(f"  Files: {len(chunk_info['files'])}")
        # Show first few files
        for file_info in chunk_info['files'][:5]:
            print(f"    - {file_info['path']} ({file_info['size_mb']:.1f} MB)")
        if len(chunk_info['files']) > 5:
            print(f"    ... and {len(chunk_info['files']) - 5} more")

    print()
    print(f"Total: {manifest['total_size_gb']:.2f} GB in {len(chunks)} file(s)")

    if dry_run:
        print("\n[DRY RUN] No files created.")
        return 0

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create ZIP files
    print()
    print("=" * 60)
    print("Creating ZIP files...")
    print("=" * 60)

    for i, chunk in enumerate(chunks):
        zip_name = f'eeg_bci_cache_part{i+1:02d}.zip'
        zip_path = output_dir / zip_name
        chunk_size = sum(size for _, size in chunk)

        print(f"\nCreating {zip_name} ({chunk_size / 1e9:.2f} GB)...")

        with zipfile.ZipFile(zip_path, 'w', compression) as zf:
            for file_path, size in chunk:
                rel_path = file_path.relative_to(cache_dir)
                # Store under caches/preprocessed/ to match expected structure
                arc_path = Path('caches/preprocessed') / rel_path
                zf.write(file_path, arc_path)

                # Progress indicator for large files
                if size > 100 * 1024 * 1024:  # > 100 MB
                    print(f"  Added: {rel_path} ({size / 1e6:.1f} MB)")

        actual_size = zip_path.stat().st_size
        print(f"  Created: {zip_path.name} ({actual_size / 1e9:.2f} GB)")

    # Save manifest
    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved: {manifest_path}")

    # Create README for the package
    readme_content = f"""# EEG-BCI Cache Package

Created: {manifest['created_at']}

## Contents

Total: {manifest['total_size_gb']:.2f} GB in {len(chunks)} file(s)

| File | Size |
|------|------|
"""
    for chunk_info in manifest['chunks']:
        readme_content += f"| {chunk_info['zip_name']} | {chunk_info['total_size_gb']:.2f} GB |\n"

    readme_content += """
## Installation

1. Extract all ZIP files to the same location:
   ```bash
   # Linux/Mac
   for f in eeg_bci_cache_part*.zip; do unzip -o "$f"; done

   # Windows PowerShell
   Get-ChildItem eeg_bci_cache_part*.zip | ForEach-Object { Expand-Archive $_.FullName -DestinationPath . -Force }
   ```

2. The extracted structure should be:
   ```
   caches/
   └── preprocessed/
       ├── .cache_index.json
       └── *.h5
   ```

3. Copy or move the `caches/` folder to your EEG-BCI project root.

## Verification

Check the manifest.json file for the complete list of included files.
"""

    readme_path = output_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"README saved: {readme_path}")

    print()
    print("=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nOutput files in: {output_dir}")

    return 0


def check_cache_health(cache_dir: Path, subjects_arg: Optional[List[str]] = None) -> int:
    """
    Check cache health: compare index to files, show expected vs actual counts.

    Args:
        cache_dir: Path to cache directory
        subjects_arg: Subject filter arguments (optional)

    Returns:
        0 on success
    """
    print("=" * 60)
    print("Cache Health Check")
    print("=" * 60)
    print(f"Cache directory: {cache_dir}")
    print()

    if not cache_dir.exists():
        print("ERROR: Cache directory not found!")
        return 1

    # Load cache index
    index_file = cache_dir / '.cache_index.json'
    if not index_file.exists():
        print("ERROR: Cache index not found (.cache_index.json)")
        return 1

    try:
        with open(index_file, 'r') as f:
            index_data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"ERROR: Failed to load cache index: {e}")
        return 1

    cache_index = index_data.get('entries', {})
    cache_version = index_data.get('version', 'unknown')

    print(f"Cache version: {cache_version}")
    print(f"Index entries: {len(cache_index)}")

    # Find all cache files on disk
    cache_files = set()
    for pattern in ['*.h5', '*.npz']:
        for f in cache_dir.glob(pattern):
            cache_files.add(f.stem)

    print(f"Cache files on disk: {len(cache_files)}")

    # Check for discrepancies
    in_index_not_on_disk = set(cache_index.keys()) - cache_files
    on_disk_not_in_index = cache_files - set(cache_index.keys())

    if in_index_not_on_disk:
        print(f"\nWARNING: {len(in_index_not_on_disk)} entries in index but NOT on disk (orphaned index entries)")
        if len(in_index_not_on_disk) <= 5:
            for key in sorted(in_index_not_on_disk):
                entry = cache_index.get(key, {})
                print(f"  - {key[:16]}... ({entry.get('subject', '?')}/{entry.get('session_folder', '?')})")

    if on_disk_not_in_index:
        print(f"\nWARNING: {len(on_disk_not_in_index)} files on disk but NOT in index (orphaned files)")
        if len(on_disk_not_in_index) <= 5:
            for key in sorted(on_disk_not_in_index):
                print(f"  - {key[:16]}...")

    # Subject-level breakdown
    print("\n" + "-" * 60)
    print("Per-Subject Cache Breakdown")
    print("-" * 60)

    by_subject: Dict[str, Dict[str, int]] = {}
    for entry in cache_index.values():
        subj = entry.get('subject', 'unknown')
        model_raw = entry.get('model', 'unknown')
        paradigm = entry.get('collection_paradigm', 'unknown')

        # Normalize model name (cbramod_128ch -> cbramod)
        if 'cbramod' in model_raw.lower():
            model = 'cbramod'
        elif 'eegnet' in model_raw.lower():
            model = 'eegnet'
        else:
            model = model_raw

        if subj not in by_subject:
            by_subject[subj] = {
                'total': 0, 'eegnet': 0, 'cbramod': 0,
                'offline': 0, 'online': 0
            }
        by_subject[subj]['total'] += 1
        by_subject[subj][model] = by_subject[subj].get(model, 0) + 1
        if paradigm:
            by_subject[subj][paradigm] = by_subject[subj].get(paradigm, 0) + 1

    # Filter by subjects if specified
    if subjects_arg:
        subjects_set = {s.upper() for s in expand_subjects(subjects_arg)}
        by_subject = {s: v for s, v in by_subject.items() if s.upper() in subjects_set}
        print(f"Filtering for subjects: {sorted(subjects_set)}")
        print()

    print(f"{'Subject':<10} {'Total':<8} {'EEGNet':<8} {'CBraMod':<8} {'Offline':<8} {'Online':<8}")
    print("-" * 58)

    total_caches = 0
    for subj in sorted(by_subject.keys()):
        s = by_subject[subj]
        total_caches += s['total']
        print(f"{subj:<10} {s['total']:<8} {s.get('eegnet',0):<8} "
              f"{s.get('cbramod',0):<8} {s.get('offline',0):<8} {s.get('online',0):<8}")

    print("-" * 58)
    print(f"{'TOTAL':<10} {total_caches:<8}")

    # Expected cache count estimation
    n_subjects = len(by_subject)
    if n_subjects > 0:
        avg_per_subject = total_caches / n_subjects
        print(f"\nAverage caches per subject: {avg_per_subject:.1f}")

        # Expected: ~140 per subject (70 eegnet + 70 cbramod) for full binary+ternary
        expected_per_subject = 140  # approximate
        if avg_per_subject < expected_per_subject * 0.5:
            print(f"WARNING: Low cache count (expected ~{expected_per_subject} per subject)")
            print("  Possible causes:")
            print("  - preprocess_zip.py not run on all model/task combinations")
            print("  - Missing session folders in source data")
            print("  - Errors during preprocessing (check logs)")

    print("\n" + "=" * 60)
    print("Health check complete")
    print("=" * 60)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Package preprocessing caches into uploadable ZIP files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('Usage:')[1] if 'Usage:' in __doc__ else '',
    )

    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=PROJECT_ROOT / 'caches' / 'preprocessed',
        help='Cache directory (default: caches/preprocessed/)',
    )

    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=PROJECT_ROOT / 'dist',
        help='Output directory for ZIP files (default: dist/)',
    )

    parser.add_argument(
        '--max-size',
        type=float,
        default=9.0,
        help='Maximum size per ZIP file in GB (default: 9.0)',
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview without creating files',
    )

    parser.add_argument(
        '--no-compression',
        action='store_true',
        help='Store files without compression (faster but larger)',
    )

    parser.add_argument(
        '--subjects', '-s',
        nargs='+',
        help='Subjects to include. Supports individual IDs (S01 S02) and ranges (S08-S21). '
             'Example: --subjects S08-S21 or --subjects S01 S03 S08-S15',
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed breakdown of cache entries by subject/model',
    )

    parser.add_argument(
        '--check',
        action='store_true',
        help='Check cache health: report index vs files, expected vs actual counts',
    )

    args = parser.parse_args()

    # Check mode: just report cache health, don't package
    if args.check:
        return check_cache_health(args.cache_dir, args.subjects)

    compression = zipfile.ZIP_STORED if args.no_compression else zipfile.ZIP_DEFLATED

    # Expand subject ranges
    subjects = None
    if args.subjects:
        subjects = expand_subjects(args.subjects)

    return package_caches(
        cache_dir=args.cache_dir,
        output_dir=args.output,
        max_size_gb=args.max_size,
        dry_run=args.dry_run,
        compression=compression,
        subjects=subjects,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    sys.exit(main())
