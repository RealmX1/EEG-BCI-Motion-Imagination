"""
一次性迁移脚本：将现有 JSON 结果文件导入 SQLite 实验注册表。

扫描 results/ 目录下的所有 JSON 结果文件，解析元数据，并导入到
ExperimentDB (results/experiments.db)。

Usage:
    uv run python scripts/tools/migrate_results_to_db.py              # 预览模式
    uv run python scripts/tools/migrate_results_to_db.py --execute    # 执行导入
    uv run python scripts/tools/migrate_results_to_db.py --execute --force  # 重建数据库
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.results.experiment_db import ExperimentDB
from src.results.dataclasses import TrainingResult, ComparisonResult

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def _get_file_first_commit(file_path: Path) -> Optional[str]:
    """Find the git commit where a file was first added to the repository.

    Uses `git log --diff-filter=A` to find the commit that initially
    added the file, which is the closest proxy for "which code version
    produced this result" for legacy runs.

    Returns:
        Short commit hash (12 chars), or None if not tracked / git unavailable.
    """
    try:
        result = subprocess.run(
            ['git', 'log', '--diff-filter=A', '--format=%H', '--', str(file_path)],
            capture_output=True, text=True, timeout=10,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode == 0 and result.stdout.strip():
            # May return multiple lines if file was deleted and re-added;
            # last line = earliest (first) commit
            lines = result.stdout.strip().splitlines()
            return lines[-1][:12]
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


# ============================================================================
# Filename parsing
# ============================================================================

# Pattern: {timestamp}_{type_prefix}_{paradigm}_{task}.json
# Examples:
#   20260206_1003_comparison_cache_imagery_binary.json
#   20260221_1319_transfer_comparison_cache_imagery_binary.json
#   20260220_1731_cross-subject_eegnet_imagery_binary.json

COMPARISON_CACHE_RE = re.compile(
    r'^(\d{8}_\d{4})_comparison_cache_(imagery|movement)_(binary|ternary|quaternary)\.json$'
)

TRANSFER_CACHE_RE = re.compile(
    r'^(\d{8}_\d{4})_transfer_comparison_cache_(imagery|movement)_(binary|ternary|quaternary)\.json$'
)

CROSS_SUBJECT_RE = re.compile(
    r'^(\d{8}_\d{4})_cross-subject_(eegnet|cbramod)_(imagery|movement)_(binary|ternary|quaternary)\.json$'
)


def infer_channel_info(file_path: Path) -> Tuple[int, Optional[str]]:
    """Infer n_channels and channel_config from directory path.

    Examples:
        results/comparison_cache_... -> (128, None)
        results/32_channel/fdr/cross-subject_... -> (32, 'fdr')
        results/8_channel/naive/... -> (8, 'naive')
        results/61_channel/standard_1010/... -> (61, 'standard_1010')
        results/32_channel/cross-subject_... -> (32, 'motor_cortex')  # default
    """
    parts = file_path.relative_to(PROJECT_ROOT / 'results').parts

    if len(parts) == 1:
        return 128, None

    dir_name = parts[0]
    match = re.match(r'^(\d+)_channel$', dir_name)
    if not match:
        return 128, None

    n_channels = int(match.group(1))

    if len(parts) >= 3:
        channel_config = parts[1]
    elif n_channels == 32:
        channel_config = 'motor_cortex'
    elif n_channels == 8:
        channel_config = None
    else:
        channel_config = None

    return n_channels, channel_config


# ============================================================================
# JSON parsers for each file type
# ============================================================================

def parse_comparison_cache(file_path: Path, data: Dict) -> Optional[Dict[str, Any]]:
    """Parse a comparison_cache JSON (within-subject results)."""
    run_tag = data.get('run_tag')
    if not run_tag:
        m = COMPARISON_CACHE_RE.match(file_path.name)
        if m:
            run_tag = m.group(1)
        else:
            return None

    paradigm = data.get('paradigm')
    task = data.get('task')
    if not paradigm or not task:
        return None

    results_data = data.get('results', {})
    if not results_data:
        return None

    n_channels, channel_config = infer_channel_info(file_path)
    metadata = data.get('metadata', {})

    subject_results: List[TrainingResult] = []
    for model_type, subjects in results_data.items():
        if not isinstance(subjects, dict):
            continue
        for subject_id, sdata in subjects.items():
            if not isinstance(sdata, dict):
                continue
            try:
                subject_results.append(TrainingResult(
                    subject_id=sdata.get('subject_id', subject_id),
                    task_type=sdata.get('task_type', task),
                    model_type=sdata.get('model_type', model_type),
                    best_val_acc=float(sdata.get('best_val_acc', 0)),
                    test_acc=float(sdata.get('test_acc', 0)),
                    test_acc_majority=float(sdata.get('test_acc_majority', sdata.get('test_acc', 0))),
                    epochs_trained=int(sdata.get('epochs_trained', 0)),
                    training_time=float(sdata.get('training_time', 0)),
                ))
            except (ValueError, TypeError) as e:
                logger.warning(f"  Skipping malformed result in {file_path.name}: {e}")

    if not subject_results:
        return None

    # Extract summary if present
    summary_data = data.get('summary', {})

    # Extract wandb groups
    wandb_groups = data.get('wandb_groups', {})
    wandb_group = next(iter(wandb_groups.values()), None) if wandb_groups else None

    return {
        'run_tag': run_tag,
        'experiment_type': 'within_subject',
        'paradigm': paradigm,
        'task': task,
        'n_channels': n_channels,
        'channel_config': channel_config,
        'n_subjects': metadata.get('n_subjects', len(set(r.subject_id for r in subject_results))),
        'is_complete': metadata.get('is_complete', True),
        'wandb_group': wandb_group,
        'subject_results': subject_results,
        'summary': summary_data,
    }


def parse_transfer_cache(file_path: Path, data: Dict) -> Optional[Dict[str, Any]]:
    """Parse a transfer_comparison_cache JSON."""
    metadata = data.get('metadata', {})
    run_tag = metadata.get('run_tag') or data.get('run_tag')
    if not run_tag:
        m = TRANSFER_CACHE_RE.match(file_path.name)
        if m:
            run_tag = m.group(1)
        else:
            return None

    paradigm = metadata.get('paradigm', data.get('paradigm'))
    task = metadata.get('task', data.get('task'))
    if not paradigm or not task:
        return None

    results_data = data.get('results', {})
    if not results_data:
        return None

    n_channels, channel_config = infer_channel_info(file_path)

    subject_results: List[TrainingResult] = []
    for model_type, subjects in results_data.items():
        if not isinstance(subjects, dict):
            continue
        for subject_id, sdata in subjects.items():
            if not isinstance(sdata, dict):
                continue
            try:
                subject_results.append(TrainingResult(
                    subject_id=sdata.get('subject_id', subject_id),
                    task_type=sdata.get('task_type', task),
                    model_type=sdata.get('model_type', model_type),
                    best_val_acc=float(sdata.get('best_val_acc', 0)),
                    test_acc=float(sdata.get('test_acc', 0)),
                    test_acc_majority=float(sdata.get('test_acc_majority', sdata.get('test_acc', 0))),
                    epochs_trained=int(sdata.get('epochs_trained', 0)),
                    training_time=float(sdata.get('training_time', 0)),
                ))
            except (ValueError, TypeError) as e:
                logger.warning(f"  Skipping malformed result in {file_path.name}: {e}")

    if not subject_results:
        return None

    # Extract transfer config
    transfer_config = metadata.get('transfer_config', {})
    summary_data = data.get('summary', {})
    wandb_groups = data.get('wandb_groups', {})
    wandb_group = next(iter(wandb_groups.values()), None) if wandb_groups else None

    return {
        'run_tag': run_tag,
        'experiment_type': 'transfer',
        'paradigm': paradigm,
        'task': task,
        'n_channels': n_channels,
        'channel_config': channel_config,
        'n_subjects': metadata.get('n_subjects', len(set(r.subject_id for r in subject_results))),
        'is_complete': metadata.get('is_complete', True),
        'wandb_group': wandb_group,
        'subject_results': subject_results,
        'summary': summary_data,
        'transfer_config': transfer_config,
    }


def parse_cross_subject(file_path: Path, data: Dict) -> Optional[Dict[str, Any]]:
    """Parse a cross-subject JSON (single model pretraining results)."""
    metadata = data.get('metadata', {})
    run_tag = metadata.get('run_tag')
    if not run_tag:
        m = CROSS_SUBJECT_RE.match(file_path.name)
        if m:
            run_tag = m.group(1)
        else:
            return None

    model_type = metadata.get('model_type')
    paradigm = metadata.get('paradigm')
    task = metadata.get('task')
    if not model_type or not paradigm or not task:
        # Try extracting from filename
        m = CROSS_SUBJECT_RE.match(file_path.name)
        if m:
            model_type = model_type or m.group(2)
            paradigm = paradigm or m.group(3)
            task = task or m.group(4)
        if not model_type or not paradigm or not task:
            return None

    results_data = data.get('results', {})
    per_subject = results_data.get('per_subject_test_acc', {})
    if not per_subject:
        return None

    n_channels, channel_config = infer_channel_info(file_path)
    # Override from metadata if present
    if metadata.get('n_channels'):
        n_channels = metadata['n_channels']

    best_val_acc = results_data.get('best_val_acc', 0)
    best_epoch = results_data.get('best_epoch', 0)
    training_info = data.get('training_info', {})
    total_time = training_info.get('training_time', metadata.get('training_time', 0))
    n_subjects = len(per_subject)

    subject_results: List[TrainingResult] = []
    for subject_id, test_acc in per_subject.items():
        subject_results.append(TrainingResult(
            subject_id=subject_id,
            task_type=task,
            model_type=model_type,
            best_val_acc=float(best_val_acc),
            test_acc=float(test_acc),
            test_acc_majority=float(test_acc),
            epochs_trained=int(best_epoch),
            training_time=float(total_time) / max(n_subjects, 1),
        ))

    return {
        'run_tag': run_tag,
        'experiment_type': 'cross_subject',
        'paradigm': paradigm,
        'task': task,
        'n_channels': n_channels,
        'channel_config': channel_config,
        'n_subjects': n_subjects,
        'is_complete': True,
        'wandb_group': None,
        'subject_results': subject_results,
        'summary': {},
        'model_path': training_info.get('model_path'),
    }


# ============================================================================
# File scanner
# ============================================================================

def scan_result_files(results_dir: Path) -> List[Tuple[str, Path, Dict]]:
    """Scan results directory and classify each JSON file.

    Returns:
        List of (file_type, file_path, parsed_data) tuples.
        file_type is one of: 'comparison_cache', 'transfer_cache', 'cross_subject', 'skip'
    """
    entries = []

    for json_path in sorted(results_dir.rglob('*.json')):
        # Skip non-result files
        name = json_path.name
        if name in ('channel_selections.json',):
            continue
        if 'preproc_ml_eng' in name or 'scheduler_comparison' in name:
            continue

        # Skip per-subject training history files
        rel = json_path.relative_to(results_dir)
        if 'within_subject' in str(rel) and name in ('history.json', 'results.json'):
            continue

        # Classify
        if TRANSFER_CACHE_RE.match(name):
            file_type = 'transfer_cache'
        elif COMPARISON_CACHE_RE.match(name):
            file_type = 'comparison_cache'
        elif CROSS_SUBJECT_RE.match(name):
            file_type = 'cross_subject'
        else:
            continue  # Skip unrecognized files

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"Failed to parse {json_path}: {e}")
            continue

        entries.append((file_type, json_path, data))

    return entries


# ============================================================================
# Migration engine
# ============================================================================

def migrate_file(db: ExperimentDB, file_type: str, file_path: Path, data: Dict) -> bool:
    """Migrate a single file to the database.

    Returns True if successfully migrated.
    """
    if file_type == 'comparison_cache':
        parsed = parse_comparison_cache(file_path, data)
    elif file_type == 'transfer_cache':
        parsed = parse_transfer_cache(file_path, data)
    elif file_type == 'cross_subject':
        parsed = parse_cross_subject(file_path, data)
    else:
        return False

    if parsed is None:
        return False

    # Build run_id and check for duplicates
    run_id_candidate = f"{parsed['run_tag']}_{parsed['experiment_type']}"
    if parsed['n_channels'] != 128:
        run_id_candidate += f"_{parsed['n_channels']}ch"
    if parsed.get('channel_config'):
        run_id_candidate += f"_{parsed['channel_config']}"
    run_id_candidate += f"_{parsed['paradigm']}_{parsed['task']}"

    run_already_exists = db.run_exists(run_id_candidate)

    # Derive updated_at from file modification time
    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()

    # Find the git commit where this file was first added to the repo
    file_git_commit = _get_file_first_commit(file_path)

    if run_already_exists:
        # Run exists — append results (e.g., second model for cross-subject)
        run_id = run_id_candidate
    else:
        # Create new run
        try:
            run_id = db.create_run(
                run_tag=parsed['run_tag'],
                experiment_type=parsed['experiment_type'],
                paradigm=parsed['paradigm'],
                task=parsed['task'],
                n_channels=parsed['n_channels'],
                channel_config=parsed.get('channel_config'),
                n_subjects=parsed.get('n_subjects'),
                wandb_group=parsed.get('wandb_group'),
                updated_at=file_mtime,
                is_legacy=True,
                legacy_source=file_path.name,
                git_commit=file_git_commit,
            )
        except Exception as e:
            logger.error(f"  Failed to create run for {file_path.name}: {e}")
            return False

    # Save subject results
    db.save_subject_results_batch(run_id, parsed['subject_results'], updated_at=file_mtime)

    # Save summaries
    summary_data = parsed.get('summary', {})
    for model_type, stats in summary_data.items():
        if isinstance(stats, dict) and any(k in stats for k in ('mean', 'mean_acc', 'mean_test_acc')):
            normalized = {
                'mean': stats.get('mean', stats.get('mean_acc', stats.get('mean_test_acc', 0))),
                'std': stats.get('std', stats.get('std_acc', stats.get('std_test_acc', 0))),
                'median': stats.get('median', stats.get('median_acc', 0)),
                'min': stats.get('min', stats.get('min_acc', 0)),
                'max': stats.get('max', stats.get('max_acc', 0)),
                'n_subjects': stats.get('n_subjects', parsed.get('n_subjects', 0)),
            }
            db.save_summary(run_id, model_type, normalized)

    # Save transfer config if applicable
    if file_type == 'transfer_cache' and parsed.get('transfer_config'):
        tc = parsed['transfer_config']
        pretrained = tc.get('pretrained_paths', {})
        db.save_transfer_config(
            run_id,
            freeze_strategy=tc.get('freeze_strategy'),
            finetune_epochs=tc.get('finetune_epochs'),
            finetune_lr=tc.get('finetune_lr'),
            finetune_batch_size=tc.get('finetune_batch_size'),
            pretrained_eegnet=pretrained.get('eegnet'),
            pretrained_cbramod=pretrained.get('cbramod'),
            classifier_type=next(iter(tc.get('classifier_types', {}).values()), None),
        )

    # Mark complete
    if parsed.get('is_complete', True):
        db.mark_complete(run_id, updated_at=file_mtime)

    return True


def run_migration(results_dir: Path, db_path: str, execute: bool = False, force: bool = False):
    """Run the full migration process."""
    print("=" * 70)
    print(" EEG-BCI 结果迁移工具: JSON → SQLite")
    print("=" * 70)

    if force and execute:
        db_file = Path(db_path)
        if db_file.exists():
            try:
                db_file.unlink()
                # Also clean up WAL/SHM files
                for suffix in ('-wal', '-shm'):
                    wal = db_file.with_name(db_file.name + suffix)
                    if wal.exists():
                        wal.unlink()
                print(f"\n已删除旧数据库: {db_path}")
            except PermissionError:
                # File locked — truncate tables instead
                import sqlite3
                _conn = sqlite3.connect(str(db_file), timeout=10)
                for tbl in ('comparisons', 'model_summaries', 'subject_results',
                            'transfer_configs', 'runs'):
                    _conn.execute(f"DELETE FROM {tbl}")
                _conn.commit()
                _conn.close()
                print(f"\n已清空数据库表 (文件被占用，无法删除): {db_path}")

    # Scan files
    print(f"\n扫描目录: {results_dir}")
    entries = scan_result_files(results_dir)

    type_counts = {}
    for ft, _, _ in entries:
        type_counts[ft] = type_counts.get(ft, 0) + 1

    print(f"\n发现 {len(entries)} 个可迁移文件:")
    for ft, count in sorted(type_counts.items()):
        print(f"  {ft}: {count}")

    if not execute:
        print("\n预览模式 — 不执行写入。使用 --execute 执行迁移。")
        print("\n待迁移文件:")
        for ft, fp, _ in entries:
            n_ch, ch_cfg = infer_channel_info(fp)
            ch_str = f" [{n_ch}ch" + (f"/{ch_cfg}" if ch_cfg else "") + "]" if n_ch != 128 else ""
            print(f"  [{ft:20s}] {fp.name}{ch_str}")
        return

    # Execute migration
    db = ExperimentDB(db_path=db_path)
    migrated = 0
    skipped = 0
    failed = 0

    for ft, fp, data in entries:
        try:
            if migrate_file(db, ft, fp, data):
                migrated += 1
                logger.info(f"  OK: {fp.name}")
            else:
                skipped += 1
        except Exception as e:
            failed += 1
            logger.error(f"  FAIL: {fp.name}: {e}")

    db.close()

    print(f"\n迁移完成:")
    print(f"  导入: {migrated}")
    print(f"  跳过 (已存在或无数据): {skipped}")
    print(f"  失败: {failed}")
    print(f"\n数据库位置: {db_path}")

    # Verify
    db = ExperimentDB(db_path=db_path)
    print(f"\n验证: {db!r}")
    for exp_type in ['within_subject', 'cross_subject', 'transfer']:
        runs = db.find_runs(experiment_type=exp_type)
        if runs:
            print(f"  {exp_type}: {len(runs)} runs")
    db.close()


def main():
    parser = argparse.ArgumentParser(description='将 JSON 结果文件迁移到 SQLite 数据库')
    parser.add_argument('--execute', action='store_true', help='执行迁移 (默认仅预览)')
    parser.add_argument('--force', action='store_true', help='删除并重建数据库')
    parser.add_argument('--results-dir', default=str(PROJECT_ROOT / 'results'),
                        help='结果目录路径')
    parser.add_argument('--db-path', default=str(PROJECT_ROOT / 'results' / 'experiments.db'),
                        help='SQLite 数据库路径')
    args = parser.parse_args()

    run_migration(
        results_dir=Path(args.results_dir),
        db_path=args.db_path,
        execute=args.execute,
        force=args.force,
    )


if __name__ == '__main__':
    main()
