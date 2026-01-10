#!/usr/bin/env python
"""
Cache management utility for the EEG-BCI preprocessing cache.

Allows selective deletion of cached data based on various criteria:
- collection_paradigm: 'online' or 'offline'
- subject_task_type: 'imagery' or 'movement'
- session_num: 1, 2, or None (for offline)
- n_classes: 2, 3, or 4
- session_phase: 'base' or 'finetune'
- model: 'eegnet' or 'cbramod'
- subject: specific subject ID (e.g., 'S01')

Usage examples:
    # Show cache statistics
    uv run python scripts/cache_helper.py --stats

    # List all offline caches (dry run)
    uv run python scripts/cache_helper.py --paradigm offline

    # List all imagery caches for subject S01
    uv run python scripts/cache_helper.py --subject S01 --task-type imagery

    # Delete all 2-class online finetune caches
    uv run python scripts/cache_helper.py --paradigm online --n-classes 2 --phase finetune --execute

    # Delete all CBraMod caches
    uv run python scripts/cache_helper.py --model cbramod --execute

    # Delete ALL caches (use with caution!)
    uv run python scripts/cache_helper.py --all --execute
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_cache_index(cache_dir: str) -> Tuple[Path, Dict[str, Any]]:
    """Load the cache index file."""
    cache_path = Path(cache_dir)
    index_file = cache_path / ".cache_index.json"

    if not index_file.exists():
        print(f"Cache index not found: {index_file}")
        sys.exit(1)

    with open(index_file, "r", encoding="utf-8") as f:
        index = json.load(f)

    return cache_path, index


def filter_entries(
    entries: Dict[str, Dict],
    paradigm: Optional[str] = None,
    task_type: Optional[str] = None,
    session_num: Optional[int] = None,
    n_classes: Optional[int] = None,
    phase: Optional[str] = None,
    model: Optional[str] = None,
    subject: Optional[str] = None,
    version: Optional[str] = None,
) -> List[Tuple[str, Dict, List[str]]]:
    """
    Filter cache entries based on criteria.

    Returns:
        List of (cache_key, entry, matched_criteria) tuples
    """
    matched = []

    for cache_key, entry in entries.items():
        reasons = []

        # Check each filter criterion
        if paradigm is not None:
            entry_paradigm = entry.get("collection_paradigm")
            if entry_paradigm != paradigm:
                continue
            reasons.append(f"paradigm={paradigm}")

        if task_type is not None:
            entry_task_type = entry.get("subject_task_type")
            if entry_task_type != task_type:
                continue
            reasons.append(f"task_type={task_type}")

        if session_num is not None:
            # For offline data, session_num is None in the entry
            # Handle special case: --session-num 0 means "offline (no session)"
            entry_session = entry.get("session_num")
            if session_num == 0:
                # Match entries with no session (offline)
                if entry_session is not None:
                    continue
                reasons.append("session=offline")
            else:
                if entry_session != session_num:
                    continue
                reasons.append(f"session={session_num}")

        if n_classes is not None:
            entry_n_classes = entry.get("n_classes")
            if entry_n_classes != n_classes:
                continue
            reasons.append(f"n_classes={n_classes}")

        if phase is not None:
            entry_phase = entry.get("session_phase")
            if entry_phase != phase:
                continue
            reasons.append(f"phase={phase}")

        if model is not None:
            entry_model = entry.get("model")
            if entry_model != model:
                continue
            reasons.append(f"model={model}")

        if subject is not None:
            entry_subject = entry.get("subject")
            if entry_subject != subject:
                continue
            reasons.append(f"subject={subject}")

        if version is not None:
            entry_version = entry.get("version")
            if entry_version != version:
                continue
            reasons.append(f"version={version}")

        matched.append((cache_key, entry, reasons))

    return matched


def print_stats(entries: Dict[str, Dict]):
    """Print cache statistics."""
    if not entries:
        print("Cache is empty.")
        return

    # Aggregate statistics
    stats = {
        "total_entries": len(entries),
        "total_size_mb": 0,
        "by_subject": {},
        "by_model": {},
        "by_paradigm": {},
        "by_task_type": {},
        "by_n_classes": {},
        "by_phase": {},
        "by_version": {},
    }

    for entry in entries.values():
        stats["total_size_mb"] += entry.get("size_mb", 0)

        # Count by various dimensions
        for key, stat_key in [
            ("subject", "by_subject"),
            ("model", "by_model"),
            ("collection_paradigm", "by_paradigm"),
            ("subject_task_type", "by_task_type"),
            ("n_classes", "by_n_classes"),
            ("session_phase", "by_phase"),
            ("version", "by_version"),
        ]:
            value = entry.get(key, "unknown")
            if value is None:
                value = "N/A"
            stats[stat_key][value] = stats[stat_key].get(value, 0) + 1

    # Print statistics
    print("=" * 60)
    print("CACHE STATISTICS")
    print("=" * 60)
    print(f"Total entries: {stats['total_entries']}")
    print(f"Total size: {stats['total_size_mb']:.1f} MB")
    print()

    def print_breakdown(title: str, data: Dict):
        if not data:
            return
        print(f"{title}:")
        for key, count in sorted(data.items(), key=lambda x: (-x[1], str(x[0]))):
            print(f"  {key}: {count}")
        print()

    print_breakdown("By Subject", stats["by_subject"])
    print_breakdown("By Model", stats["by_model"])
    print_breakdown("By Collection Paradigm", stats["by_paradigm"])
    print_breakdown("By Subject Task Type", stats["by_task_type"])
    print_breakdown("By N Classes", stats["by_n_classes"])
    print_breakdown("By Session Phase", stats["by_phase"])
    print_breakdown("By Cache Version", stats["by_version"])


def print_matched_entries(
    matched: List[Tuple[str, Dict, List[str]]],
    cache_path: Path,
    max_display: int = 20,
):
    """Print matched entries."""
    if not matched:
        print("No entries match the specified criteria.")
        return

    total_size = sum(entry.get("size_mb", 0) for _, entry, _ in matched)

    print(f"Matched entries: {len(matched)}")
    print(f"Total size: {total_size:.1f} MB")
    print()
    print("Entries:")

    for i, (cache_key, entry, reasons) in enumerate(matched[:max_display]):
        subject = entry.get("subject", "?")
        run = entry.get("run", "?")
        folder = entry.get("session_folder", "?")
        model = entry.get("model", "?")
        size = entry.get("size_mb", 0)
        version = entry.get("version", "?")

        # Format run number
        run_str = f"R{run:02d}" if isinstance(run, int) else f"R{run}"

        print(f"  [{subject}/{run_str}] {folder}")
        print(f"    model={model}, size={size:.1f}MB, version={version}")
        if reasons:
            print(f"    matched: {', '.join(reasons)}")

    if len(matched) > max_display:
        print(f"  ... and {len(matched) - max_display} more entries")


def delete_entries(
    matched: List[Tuple[str, Dict, List[str]]],
    cache_path: Path,
    index: Dict[str, Any],
    index_file: Path,
):
    """Delete matched cache entries."""
    entries = index.get("entries", {})

    deleted_count = 0
    deleted_size = 0
    errors = []

    for cache_key, entry, _ in matched:
        # Try to delete the cache file
        cache_file = cache_path / f"{cache_key}.h5"
        if not cache_file.exists():
            # Try .npz format
            cache_file = cache_path / f"{cache_key}.npz"

        if cache_file.exists():
            try:
                size = entry.get("size_mb", 0)
                cache_file.unlink()
                deleted_count += 1
                deleted_size += size
            except Exception as e:
                errors.append(f"{cache_file.name}: {e}")
        else:
            # File doesn't exist, just remove from index
            pass

        # Remove from index
        if cache_key in entries:
            del entries[cache_key]

    # Save updated index
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"Deleted {deleted_count} files ({deleted_size:.1f} MB)")
    print(f"Updated cache index: {len(entries)} entries remaining")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for err in errors[:5]:
            print(f"  {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")


def main():
    parser = argparse.ArgumentParser(
        description="Manage EEG-BCI preprocessing cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Filter options
    filter_group = parser.add_argument_group("Filter Options")
    filter_group.add_argument(
        "--paradigm",
        choices=["online", "offline"],
        help="Filter by collection paradigm",
    )
    filter_group.add_argument(
        "--task-type",
        choices=["imagery", "movement"],
        help="Filter by subject task type (motor imagery vs execution)",
    )
    filter_group.add_argument(
        "--session-num",
        type=int,
        choices=[0, 1, 2],
        help="Filter by session number (0 = offline/no session, 1 = Sess01, 2 = Sess02)",
    )
    filter_group.add_argument(
        "--n-classes",
        type=int,
        choices=[2, 3, 4],
        help="Filter by number of classes",
    )
    filter_group.add_argument(
        "--phase",
        choices=["base", "finetune"],
        help="Filter by session phase",
    )
    filter_group.add_argument(
        "--model",
        choices=["eegnet", "cbramod"],
        help="Filter by target model",
    )
    filter_group.add_argument(
        "--subject",
        help="Filter by subject ID (e.g., S01)",
    )
    filter_group.add_argument(
        "--version",
        help="Filter by cache version (e.g., 2.1)",
    )
    filter_group.add_argument(
        "--all",
        action="store_true",
        help="Select ALL cache entries (use with caution!)",
    )

    # Action options
    action_group = parser.add_argument_group("Actions")
    action_group.add_argument(
        "--stats",
        action="store_true",
        help="Show cache statistics (no filtering)",
    )
    action_group.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete matched entries (default: dry run)",
    )

    # Other options
    parser.add_argument(
        "--cache-dir",
        default="caches/preprocessed",
        help="Path to cache directory (default: caches/preprocessed)",
    )
    parser.add_argument(
        "--max-display",
        type=int,
        default=20,
        help="Maximum entries to display (default: 20)",
    )

    args = parser.parse_args()

    # Load cache index
    cache_path, index = load_cache_index(args.cache_dir)
    entries = index.get("entries", {})
    index_file = cache_path / ".cache_index.json"

    print(f"Cache directory: {cache_path}")
    print(f"Total entries: {len(entries)}")
    print()

    # Show stats if requested
    if args.stats:
        print_stats(entries)
        return

    # Check if any filter is specified
    has_filter = any([
        args.paradigm,
        args.task_type,
        args.session_num is not None,
        args.n_classes,
        args.phase,
        args.model,
        args.subject,
        args.version,
        args.all,
    ])

    if not has_filter:
        print("No filter specified. Use --stats to show statistics,")
        print("or specify filter options to select entries for deletion.")
        print()
        print("Examples:")
        print("  --paradigm offline          # All offline data")
        print("  --model cbramod             # All CBraMod caches")
        print("  --subject S01 --n-classes 2 # S01's binary classification")
        print("  --all                       # Everything (dangerous!)")
        print()
        print("Add --execute to actually delete (default is dry run).")
        return

    # Apply filters
    if args.all:
        matched = [(k, v, ["all"]) for k, v in entries.items()]
    else:
        matched = filter_entries(
            entries,
            paradigm=args.paradigm,
            task_type=args.task_type,
            session_num=args.session_num,
            n_classes=args.n_classes,
            phase=args.phase,
            model=args.model,
            subject=args.subject,
            version=args.version,
        )

    # Display matched entries
    print_matched_entries(matched, cache_path, max_display=args.max_display)
    print()

    if not matched:
        return

    # Execute deletion if requested
    if args.execute:
        # Confirmation for dangerous operations
        if args.all or len(matched) > 50:
            total_size = sum(e.get("size_mb", 0) for _, e, _ in matched)
            print(f"WARNING: About to delete {len(matched)} entries ({total_size:.1f} MB)")
            confirm = input("Type 'yes' to confirm: ")
            if confirm.lower() != "yes":
                print("Aborted.")
                return
            print()

        delete_entries(matched, cache_path, index, index_file)
    else:
        print("Dry run mode. Use --execute to actually delete files.")


if __name__ == "__main__":
    main()
