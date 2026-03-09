"""
SQLite-backed experiment registry for EEG-BCI project.

Replaces the ad-hoc JSON file + filename-encoding scheme with structured
SQL storage. Designed to coexist with WandB (cloud metrics) — SQLite handles
local metadata, final metrics, and structured querying.

Usage:
    from src.results.experiment_db import ExperimentDB

    db = ExperimentDB()

    # Create a run
    run_id = db.create_run('20260221_1319', 'within_subject', 'imagery', 'binary')

    # Save per-subject results incrementally
    db.save_subject_result(run_id, training_result, wandb_run_id='abc123')

    # Mark complete
    db.save_summary(run_id, 'eegnet', stats_dict)
    db.mark_complete(run_id)

    # Query
    runs = db.find_runs(paradigm='imagery', task='binary', n_channels=32)
    best = db.get_best_run('imagery', 'binary', 'cbramod', 'within_subject')
"""

from __future__ import annotations

import logging
import sqlite3
import subprocess
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .dataclasses import ComparisonResult, TrainingResult
except ImportError:
    from src.results.dataclasses import ComparisonResult, TrainingResult

logger = logging.getLogger(__name__)

# Default database location
DEFAULT_DB_PATH = 'results/experiments.db'

# Schema version for future migrations
_SCHEMA_VERSION = 5

_SCHEMA_SQL = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_info (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Experiment runs (one row per script invocation)
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    run_tag         TEXT NOT NULL,
    experiment_type TEXT NOT NULL,
    paradigm        TEXT NOT NULL,
    task            TEXT NOT NULL,
    n_channels      INTEGER NOT NULL DEFAULT 128,
    channel_config  TEXT,
    n_subjects      INTEGER,
    is_complete     INTEGER NOT NULL DEFAULT 0,
    git_commit      TEXT,
    wandb_group     TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    notes           TEXT,
    -- Legacy migration flag: 1 = migrated from JSON, 0 = created natively via DB
    is_legacy       INTEGER NOT NULL DEFAULT 0,
    -- [deprecated] Legacy-only columns below — NULL for new runs
    legacy_source   TEXT,   -- original JSON filename, e.g. '20260205_0116_comparison_cache_imagery_binary.json'
    command         TEXT,   -- full terminal command used to launch this run (sys.argv)
    preprocessing_version TEXT  -- e.g., 'v1.0', 'v2.0'; tracks data filtering/cleaning version
);

-- Transfer learning specific configuration
CREATE TABLE IF NOT EXISTS transfer_configs (
    run_id              TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE,
    freeze_strategy     TEXT,
    finetune_epochs     INTEGER,
    finetune_lr         REAL,
    finetune_batch_size INTEGER,
    pretrained_eegnet   TEXT,
    pretrained_cbramod  TEXT,
    classifier_type     TEXT
);

-- Per-subject training results (core data)
CREATE TABLE IF NOT EXISTS subject_results (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    subject_id        TEXT NOT NULL,
    model_type        TEXT NOT NULL,
    best_val_acc      REAL NOT NULL,
    test_acc          REAL NOT NULL,
    test_acc_majority REAL NOT NULL,
    epochs_trained    INTEGER NOT NULL,
    training_time     REAL NOT NULL,
    wandb_run_id      TEXT,
    UNIQUE(run_id, subject_id, model_type)
);

-- Model-level summary statistics (denormalized for fast queries)
CREATE TABLE IF NOT EXISTS model_summaries (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    model_type TEXT NOT NULL,
    mean_acc   REAL,
    std_acc    REAL,
    median_acc REAL,
    min_acc    REAL,
    max_acc    REAL,
    n_subjects INTEGER,
    UNIQUE(run_id, model_type)
);

-- Statistical comparisons between model pairs
CREATE TABLE IF NOT EXISTS comparisons (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    model_a        TEXT NOT NULL,
    model_b        TEXT NOT NULL,
    mean_diff      REAL,
    paired_ttest_t REAL,
    paired_ttest_p REAL,
    wilcoxon_stat  REAL,
    wilcoxon_p     REAL,
    better_model   TEXT,
    significant    INTEGER,
    UNIQUE(run_id, model_a, model_b)
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_runs_query
    ON runs(paradigm, task, n_channels, experiment_type);
CREATE INDEX IF NOT EXISTS idx_runs_tag
    ON runs(run_tag);
CREATE INDEX IF NOT EXISTS idx_results_run
    ON subject_results(run_id);
CREATE INDEX IF NOT EXISTS idx_results_subject
    ON subject_results(subject_id, model_type);
CREATE INDEX IF NOT EXISTS idx_summaries_run
    ON model_summaries(run_id);
"""


def _parse_run_tag(run_tag: str) -> Optional[datetime]:
    """Parse a run_tag like '20260206_1003' into a datetime.

    Returns None if the tag doesn't match the expected format.
    """
    import re
    m = re.match(r'^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})$', run_tag)
    if m:
        try:
            return datetime(
                int(m.group(1)), int(m.group(2)), int(m.group(3)),
                int(m.group(4)), int(m.group(5)),
            )
        except ValueError:
            return None
    return None


def _get_git_commit() -> Optional[str]:
    """Get current git HEAD commit hash, or None if unavailable."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


class ExperimentDB:
    """SQLite-backed experiment registry.

    Thread-safe for single-writer patterns typical in training scripts.
    Uses WAL mode for concurrent read access.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_schema()

    @contextmanager
    def _connection(self):
        """Context manager for database connections with auto-commit."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self._db_path),
                timeout=30,
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.row_factory = sqlite3.Row
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def _ensure_schema(self):
        """Create tables if they don't exist, and apply schema migrations."""
        with self._connection() as conn:
            conn.executescript(_SCHEMA_SQL)

            # Check current schema version
            try:
                row = conn.execute(
                    "SELECT value FROM schema_info WHERE key = 'version'"
                ).fetchone()
                current_version = int(row['value']) if row else 0
            except Exception:
                current_version = 0

            # Apply migrations
            if current_version < 2:
                self._migrate_to_v2(conn)
            if current_version < 3:
                self._migrate_to_v3(conn)
            if current_version < 4:
                self._migrate_to_v4(conn)
            if current_version < 5:
                self._migrate_to_v5(conn)

            conn.execute(
                "INSERT OR REPLACE INTO schema_info (key, value) VALUES (?, ?)",
                ('version', str(_SCHEMA_VERSION)),
            )

    def _migrate_to_v2(self, conn: sqlite3.Connection):
        """v1 -> v2: Add is_legacy and legacy_source columns."""
        # Check if columns already exist (idempotent)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
        if 'is_legacy' not in cols:
            conn.execute("ALTER TABLE runs ADD COLUMN is_legacy INTEGER NOT NULL DEFAULT 0")
            logger.info("Schema migration v2: added runs.is_legacy")
        if 'legacy_source' not in cols:
            conn.execute("ALTER TABLE runs ADD COLUMN legacy_source TEXT")
            logger.info("Schema migration v2: added runs.legacy_source")

    def _migrate_to_v3(self, conn: sqlite3.Connection):
        """v2 -> v3: Add command column to runs."""
        cols = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
        if 'command' not in cols:
            conn.execute("ALTER TABLE runs ADD COLUMN command TEXT")
            logger.info("Schema migration v3: added runs.command")

    def _migrate_to_v4(self, conn: sqlite3.Connection):
        """v3 -> v4: Add preprocessing_version column and backfill."""
        cols = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
        if 'preprocessing_version' not in cols:
            conn.execute("ALTER TABLE runs ADD COLUMN preprocessing_version TEXT")
            logger.info("Schema migration v4: added runs.preprocessing_version")

        # Backfill: commit 5bb2395 (2026-03-02 17:18:47) introduced trial rejection
        from ..config.constants import _PREPROCESSING_V2_TIMESTAMP
        n_v1 = conn.execute(
            "UPDATE runs SET preprocessing_version = 'v1.0' "
            "WHERE preprocessing_version IS NULL AND created_at < ?",
            (_PREPROCESSING_V2_TIMESTAMP,),
        ).rowcount
        n_v2 = conn.execute(
            "UPDATE runs SET preprocessing_version = 'v2.0' "
            "WHERE preprocessing_version IS NULL AND created_at >= ?",
            (_PREPROCESSING_V2_TIMESTAMP,),
        ).rowcount
        if n_v1 or n_v2:
            logger.info(
                f"Schema migration v4: backfilled preprocessing_version "
                f"(v1.0={n_v1}, v2.0={n_v2})"
            )

    def _migrate_to_v5(self, conn: sqlite3.Connection):
        """v4 -> v5: Reclassify early v1.0 runs into v0.1/v0.2."""
        from ..config.constants import (
            _PREPROCESSING_V0_2_TIMESTAMP,
            _PREPROCESSING_V1_0_TIMESTAMP,
        )
        n_v01 = conn.execute(
            "UPDATE runs SET preprocessing_version = 'v0.1' "
            "WHERE preprocessing_version = 'v1.0' AND created_at < ?",
            (_PREPROCESSING_V0_2_TIMESTAMP,),
        ).rowcount
        n_v02 = conn.execute(
            "UPDATE runs SET preprocessing_version = 'v0.2' "
            "WHERE preprocessing_version = 'v1.0' AND created_at < ?",
            (_PREPROCESSING_V1_0_TIMESTAMP,),
        ).rowcount
        if n_v01 or n_v02:
            logger.info(
                f"Schema migration v5: reclassified early runs "
                f"(v0.1={n_v01}, v0.2={n_v02})"
            )

    def close(self):
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ========================================================================
    # Write operations
    # ========================================================================

    def create_run(
        self,
        run_tag: str,
        experiment_type: str,
        paradigm: str,
        task: str,
        n_channels: int = 128,
        channel_config: Optional[str] = None,
        n_subjects: Optional[int] = None,
        wandb_group: Optional[str] = None,
        notes: Optional[str] = None,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        is_legacy: bool = False,
        legacy_source: Optional[str] = None,
        git_commit: Optional[str] = None,
        command: Optional[str] = None,
        preprocessing_version: Optional[str] = None,
    ) -> str:
        """Create a new experiment run.

        Args:
            run_tag: Timestamp tag (e.g., '20260221_1319')
            experiment_type: 'within_subject' | 'cross_subject' | 'transfer' | 'config_comparison'
            paradigm: 'imagery' | 'movement'
            task: 'binary' | 'ternary' | 'quaternary'
            n_channels: Number of EEG channels (8, 32, 61, 128)
            channel_config: Channel selection method (None for 128ch)
            n_subjects: Expected number of subjects
            wandb_group: WandB group name for linking
            notes: Optional human-readable notes
            created_at: Override for creation timestamp (ISO format).
                If None, parsed from run_tag; falls back to now().
            updated_at: Override for update timestamp (ISO format).
                If None, same as created_at.
            is_legacy: True for runs migrated from JSON files (default False)
            legacy_source: [deprecated] Original JSON filename for migrated runs
            git_commit: Override for git commit hash. If None, auto-detected
                from current HEAD (new runs) or left as None (legacy runs).
            command: Full terminal command used to launch this run (from sys.argv).
            preprocessing_version: Override for preprocessing version string.
                If None, auto-populated from PREPROCESSING_VERSION constant
                (new runs) or left as None (legacy runs).

        Returns:
            run_id: Unique identifier for this run
        """
        run_id = f"{run_tag}_{experiment_type}"
        if n_channels != 128:
            run_id += f"_{n_channels}ch"
        if channel_config:
            run_id += f"_{channel_config}"
        run_id += f"_{paradigm}_{task}"

        if created_at is None:
            parsed = _parse_run_tag(run_tag)
            created_at = parsed.isoformat() if parsed else datetime.now().isoformat()
        if updated_at is None:
            updated_at = created_at
        if git_commit is None and not is_legacy:
            git_commit = _get_git_commit()
        if preprocessing_version is None and not is_legacy:
            from ..config.constants import PREPROCESSING_VERSION
            preprocessing_version = PREPROCESSING_VERSION

        with self._connection() as conn:
            conn.execute(
                """INSERT INTO runs
                   (run_id, run_tag, experiment_type, paradigm, task,
                    n_channels, channel_config, n_subjects, is_complete,
                    git_commit, wandb_group, created_at, updated_at, notes,
                    is_legacy, legacy_source, command, preprocessing_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, run_tag, experiment_type, paradigm, task,
                 n_channels, channel_config, n_subjects, git_commit,
                 wandb_group, created_at, updated_at, notes,
                 int(is_legacy), legacy_source, command, preprocessing_version),
            )

        logger.info(f"Created run: {run_id}")
        return run_id

    def save_subject_result(
        self,
        run_id: str,
        result: TrainingResult,
        wandb_run_id: Optional[str] = None,
    ):
        """Save a single subject's training result (upsert).

        Supports incremental saving for resume capability — calling this
        multiple times with the same (run_id, subject_id, model_type)
        will update the existing record.

        Args:
            run_id: Run identifier from create_run()
            result: TrainingResult object
            wandb_run_id: Optional WandB run ID for linking
        """
        now = datetime.now().isoformat()
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO subject_results
                   (run_id, subject_id, model_type, best_val_acc, test_acc,
                    test_acc_majority, epochs_trained, training_time, wandb_run_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(run_id, subject_id, model_type)
                   DO UPDATE SET
                       best_val_acc = excluded.best_val_acc,
                       test_acc = excluded.test_acc,
                       test_acc_majority = excluded.test_acc_majority,
                       epochs_trained = excluded.epochs_trained,
                       training_time = excluded.training_time,
                       wandb_run_id = excluded.wandb_run_id""",
                (run_id, result.subject_id, result.model_type,
                 result.best_val_acc, result.test_acc,
                 result.test_acc_majority, result.epochs_trained,
                 result.training_time, wandb_run_id),
            )
            conn.execute(
                "UPDATE runs SET updated_at = ? WHERE run_id = ?",
                (now, run_id),
            )

    def save_subject_results_batch(
        self,
        run_id: str,
        results: List[TrainingResult],
        wandb_run_ids: Optional[Dict[str, str]] = None,
        updated_at: Optional[str] = None,
    ):
        """Save multiple subject results in a single transaction.

        Args:
            run_id: Run identifier
            results: List of TrainingResult objects
            wandb_run_ids: Optional mapping of '{subject_id}_{model_type}' -> wandb_run_id
            updated_at: Override for updated_at timestamp. If None, uses now().
        """
        wandb_run_ids = wandb_run_ids or {}
        now = updated_at or datetime.now().isoformat()
        with self._connection() as conn:
            for result in results:
                key = f"{result.subject_id}_{result.model_type}"
                wid = wandb_run_ids.get(key)
                conn.execute(
                    """INSERT INTO subject_results
                       (run_id, subject_id, model_type, best_val_acc, test_acc,
                        test_acc_majority, epochs_trained, training_time, wandb_run_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(run_id, subject_id, model_type)
                       DO UPDATE SET
                           best_val_acc = excluded.best_val_acc,
                           test_acc = excluded.test_acc,
                           test_acc_majority = excluded.test_acc_majority,
                           epochs_trained = excluded.epochs_trained,
                           training_time = excluded.training_time,
                           wandb_run_id = excluded.wandb_run_id""",
                    (run_id, result.subject_id, result.model_type,
                     result.best_val_acc, result.test_acc,
                     result.test_acc_majority, result.epochs_trained,
                     result.training_time, wid),
                )
            conn.execute(
                "UPDATE runs SET updated_at = ? WHERE run_id = ?",
                (now, run_id),
            )

    def save_summary(self, run_id: str, model_type: str, stats: Dict[str, Any]):
        """Save model-level summary statistics (upsert).

        Args:
            run_id: Run identifier
            model_type: 'eegnet' or 'cbramod'
            stats: Dict with keys: mean, std, median, min, max, n_subjects
        """
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO model_summaries
                   (run_id, model_type, mean_acc, std_acc, median_acc,
                    min_acc, max_acc, n_subjects)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(run_id, model_type)
                   DO UPDATE SET
                       mean_acc = excluded.mean_acc,
                       std_acc = excluded.std_acc,
                       median_acc = excluded.median_acc,
                       min_acc = excluded.min_acc,
                       max_acc = excluded.max_acc,
                       n_subjects = excluded.n_subjects""",
                (run_id, model_type,
                 stats.get('mean'), stats.get('std'), stats.get('median'),
                 stats.get('min'), stats.get('max'), stats.get('n_subjects')),
            )

    def save_comparison(self, run_id: str, comparison: ComparisonResult):
        """Save statistical comparison between model pairs (upsert).

        Args:
            run_id: Run identifier
            comparison: ComparisonResult from compare_models()
        """
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO comparisons
                   (run_id, model_a, model_b, mean_diff,
                    paired_ttest_t, paired_ttest_p,
                    wilcoxon_stat, wilcoxon_p,
                    better_model, significant)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(run_id, model_a, model_b)
                   DO UPDATE SET
                       mean_diff = excluded.mean_diff,
                       paired_ttest_t = excluded.paired_ttest_t,
                       paired_ttest_p = excluded.paired_ttest_p,
                       wilcoxon_stat = excluded.wilcoxon_stat,
                       wilcoxon_p = excluded.wilcoxon_p,
                       better_model = excluded.better_model,
                       significant = excluded.significant""",
                (run_id, 'eegnet', 'cbramod', comparison.difference_mean,
                 comparison.paired_ttest_t, comparison.paired_ttest_p,
                 comparison.wilcoxon_stat, comparison.wilcoxon_p,
                 comparison.better_model, int(comparison.significant)),
            )

    def save_transfer_config(
        self,
        run_id: str,
        freeze_strategy: Optional[str] = None,
        finetune_epochs: Optional[int] = None,
        finetune_lr: Optional[float] = None,
        finetune_batch_size: Optional[int] = None,
        pretrained_eegnet: Optional[str] = None,
        pretrained_cbramod: Optional[str] = None,
        classifier_type: Optional[str] = None,
    ):
        """Save transfer learning configuration for a run.

        Args:
            run_id: Run identifier (must exist in runs table)
            freeze_strategy: 'backbone' | 'partial' | 'none'
            finetune_epochs: Number of finetuning epochs
            finetune_lr: Finetuning learning rate
            finetune_batch_size: Finetuning batch size
            pretrained_eegnet: Path to pretrained EEGNet checkpoint
            pretrained_cbramod: Path to pretrained CBraMod checkpoint
            classifier_type: 'two_layer' | 'linear'
        """
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO transfer_configs
                   (run_id, freeze_strategy, finetune_epochs, finetune_lr,
                    finetune_batch_size, pretrained_eegnet, pretrained_cbramod,
                    classifier_type)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(run_id) DO UPDATE SET
                       freeze_strategy = excluded.freeze_strategy,
                       finetune_epochs = excluded.finetune_epochs,
                       finetune_lr = excluded.finetune_lr,
                       finetune_batch_size = excluded.finetune_batch_size,
                       pretrained_eegnet = excluded.pretrained_eegnet,
                       pretrained_cbramod = excluded.pretrained_cbramod,
                       classifier_type = excluded.classifier_type""",
                (run_id, freeze_strategy, finetune_epochs, finetune_lr,
                 finetune_batch_size, pretrained_eegnet, pretrained_cbramod,
                 classifier_type),
            )

    def update_wandb_group(self, run_id: str, wandb_group: str):
        """Update the WandB group for a run."""
        now = datetime.now().isoformat()
        with self._connection() as conn:
            conn.execute(
                "UPDATE runs SET wandb_group = ?, updated_at = ? WHERE run_id = ?",
                (wandb_group, now, run_id),
            )

    def mark_complete(self, run_id: str, updated_at: Optional[str] = None):
        """Mark a run as complete."""
        now = updated_at or datetime.now().isoformat()
        with self._connection() as conn:
            conn.execute(
                "UPDATE runs SET is_complete = 1, updated_at = ? WHERE run_id = ?",
                (now, run_id),
            )
        logger.info(f"Run marked complete: {run_id}")

    def update_n_subjects(self, run_id: str, n_subjects: int):
        """Update the subject count for a run."""
        now = datetime.now().isoformat()
        with self._connection() as conn:
            conn.execute(
                "UPDATE runs SET n_subjects = ?, updated_at = ? WHERE run_id = ?",
                (n_subjects, now, run_id),
            )

    # ========================================================================
    # Read / Query operations
    # ========================================================================

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get a single run by ID.

        Returns:
            Run dict or None if not found.
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            return dict(row) if row else None

    def find_runs(
        self,
        paradigm: Optional[str] = None,
        task: Optional[str] = None,
        n_channels: Optional[int] = None,
        experiment_type: Optional[str] = None,
        channel_config: Optional[str] = None,
        is_complete: Optional[bool] = None,
        preprocessing_version: Optional[str] = None,
        order_by: str = 'created_at DESC',
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Find runs matching the given criteria.

        All filter parameters are optional — omitted filters match everything.

        Args:
            paradigm: 'imagery' | 'movement'
            task: 'binary' | 'ternary' | 'quaternary'
            n_channels: 8 | 32 | 61 | 128
            experiment_type: 'within_subject' | 'cross_subject' | 'transfer' | 'config_comparison'
            channel_config: 'motor_cortex' | 'commercial' | 'fdr' | etc.
            is_complete: Filter by completion status
            preprocessing_version: 'v1.0' | 'v2.0' | etc.
            order_by: SQL ORDER BY clause (default: newest first)
            limit: Maximum number of results

        Returns:
            List of run dicts.
        """
        clauses = []
        params: list = []

        if paradigm is not None:
            clauses.append("paradigm = ?")
            params.append(paradigm)
        if task is not None:
            clauses.append("task = ?")
            params.append(task)
        if n_channels is not None:
            clauses.append("n_channels = ?")
            params.append(n_channels)
        if experiment_type is not None:
            clauses.append("experiment_type = ?")
            params.append(experiment_type)
        if channel_config is not None:
            clauses.append("channel_config = ?")
            params.append(channel_config)
        if is_complete is not None:
            clauses.append("is_complete = ?")
            params.append(int(is_complete))
        if preprocessing_version is not None:
            clauses.append("preprocessing_version = ?")
            params.append(preprocessing_version)

        where = " AND ".join(clauses) if clauses else "1=1"

        # Validate order_by to prevent SQL injection
        allowed_columns = {'created_at', 'updated_at', 'run_tag', 'run_id'}
        allowed_dirs = {'ASC', 'DESC'}
        parts = order_by.split()
        if len(parts) == 2 and parts[0] in allowed_columns and parts[1].upper() in allowed_dirs:
            safe_order = f"{parts[0]} {parts[1].upper()}"
        else:
            safe_order = "created_at DESC"

        sql = f"SELECT * FROM runs WHERE {where} ORDER BY {safe_order}"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        with self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def find_latest_run(
        self,
        paradigm: str,
        task: str,
        experiment_type: str,
        n_channels: int = 128,
        channel_config: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Find the most recent completed run matching criteria.

        Replaces find_latest_cache().
        """
        runs = self.find_runs(
            paradigm=paradigm,
            task=task,
            experiment_type=experiment_type,
            n_channels=n_channels,
            channel_config=channel_config,
            is_complete=True,
            order_by='created_at DESC',
            limit=1,
        )
        return runs[0] if runs else None

    def find_run_by_tag(
        self,
        tag_substring: str,
        paradigm: Optional[str] = None,
        task: Optional[str] = None,
        experiment_type: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Find a run by timestamp tag substring.

        Replaces find_cache_by_tag().

        Args:
            tag_substring: Partial timestamp (e.g., '20260205')
            paradigm: Optional filter
            task: Optional filter
            experiment_type: Optional filter

        Returns:
            Most recent matching run dict, or None.
        """
        clauses = ["run_tag LIKE ?"]
        params: list = [f"%{tag_substring}%"]

        if paradigm is not None:
            clauses.append("paradigm = ?")
            params.append(paradigm)
        if task is not None:
            clauses.append("task = ?")
            params.append(task)
        if experiment_type is not None:
            clauses.append("experiment_type = ?")
            params.append(experiment_type)

        where = " AND ".join(clauses)

        with self._connection() as conn:
            row = conn.execute(
                f"SELECT * FROM runs WHERE {where} ORDER BY created_at DESC LIMIT 1",
                params,
            ).fetchone()
            return dict(row) if row else None

    def get_results(
        self,
        run_id: str,
        model_type: Optional[str] = None,
    ) -> List[TrainingResult]:
        """Get all subject results for a run.

        Args:
            run_id: Run identifier
            model_type: Optional filter by model

        Returns:
            List of TrainingResult objects
        """
        if model_type:
            sql = "SELECT * FROM subject_results WHERE run_id = ? AND model_type = ? ORDER BY subject_id"
            params = (run_id, model_type)
        else:
            sql = "SELECT * FROM subject_results WHERE run_id = ? ORDER BY model_type, subject_id"
            params = (run_id,)

        with self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()

        task_type = self._get_run_task(run_id)
        return [
            TrainingResult(
                subject_id=r['subject_id'],
                task_type=task_type,
                model_type=r['model_type'],
                best_val_acc=r['best_val_acc'],
                test_acc=r['test_acc'],
                test_acc_majority=r['test_acc_majority'],
                epochs_trained=r['epochs_trained'],
                training_time=r['training_time'],
            )
            for r in rows
        ]

    def get_results_by_model(
        self,
        run_id: str,
    ) -> Dict[str, List[TrainingResult]]:
        """Get results grouped by model type.

        Returns:
            Dict mapping model_type -> list of TrainingResult.
        """
        all_results = self.get_results(run_id)
        grouped: Dict[str, List[TrainingResult]] = {}
        for r in all_results:
            grouped.setdefault(r.model_type, []).append(r)
        return grouped

    def get_summary(
        self,
        run_id: str,
        model_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get model summaries for a run.

        Args:
            run_id: Run identifier
            model_type: Optional filter

        Returns:
            List of summary dicts with keys: model_type, mean_acc, std_acc, etc.
        """
        if model_type:
            sql = "SELECT * FROM model_summaries WHERE run_id = ? AND model_type = ?"
            params = (run_id, model_type)
        else:
            sql = "SELECT * FROM model_summaries WHERE run_id = ?"
            params = (run_id,)

        with self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def get_comparison(self, run_id: str) -> Optional[ComparisonResult]:
        """Get statistical comparison for a run.

        Returns:
            ComparisonResult or None if no comparison exists.
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM comparisons WHERE run_id = ?", (run_id,)
            ).fetchone()
            if not row:
                return None

            # Need summary data for full ComparisonResult
            summaries = {
                s['model_type']: s
                for s in self.get_summary(run_id)
            }
            eegnet_s = summaries.get('eegnet', {})
            cbramod_s = summaries.get('cbramod', {})

            n_subjects = max(
                eegnet_s.get('n_subjects', 0) or 0,
                cbramod_s.get('n_subjects', 0) or 0,
            )

            return ComparisonResult(
                n_subjects=n_subjects,
                eegnet_mean=eegnet_s.get('mean_acc', 0),
                eegnet_std=eegnet_s.get('std_acc', 0),
                eegnet_median=eegnet_s.get('median_acc', 0),
                cbramod_mean=cbramod_s.get('mean_acc', 0),
                cbramod_std=cbramod_s.get('std_acc', 0),
                cbramod_median=cbramod_s.get('median_acc', 0),
                difference_mean=row['mean_diff'] or 0,
                difference_std=0,  # not stored separately, compute if needed
                paired_ttest_t=row['paired_ttest_t'] or 0,
                paired_ttest_p=row['paired_ttest_p'] or 0,
                wilcoxon_stat=row['wilcoxon_stat'],
                wilcoxon_p=row['wilcoxon_p'],
                better_model=row['better_model'] or 'tie',
                significant=bool(row['significant']),
            )

    def get_best_run(
        self,
        paradigm: str,
        task: str,
        model_type: str,
        experiment_type: str,
        n_channels: int = 128,
        channel_config: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Find the run with the highest mean accuracy for a given model.

        Replaces find_best_within_subject_for_model() and similar functions.

        Returns:
            Run dict with extra 'best_mean_acc' key, or None.
        """
        clauses = [
            "r.paradigm = ?",
            "r.task = ?",
            "r.n_channels = ?",
            "r.experiment_type = ?",
            "r.is_complete = 1",
            "ms.model_type = ?",
        ]
        params: list = [paradigm, task, n_channels, experiment_type, model_type]

        if channel_config is not None:
            clauses.append("r.channel_config = ?")
            params.append(channel_config)

        where = " AND ".join(clauses)

        with self._connection() as conn:
            row = conn.execute(
                f"""SELECT r.*, ms.mean_acc as best_mean_acc
                    FROM runs r
                    JOIN model_summaries ms ON r.run_id = ms.run_id
                    WHERE {where}
                    ORDER BY ms.mean_acc DESC
                    LIMIT 1""",
                params,
            ).fetchone()
            return dict(row) if row else None

    # ========================================================================
    # Resume support
    # ========================================================================

    def get_incomplete_run(
        self,
        paradigm: str,
        task: str,
        experiment_type: str,
        n_channels: int = 128,
        channel_config: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Find the most recent incomplete run for resume.

        Replaces the cache-based resume mechanism.

        Returns:
            Run dict or None.
        """
        runs = self.find_runs(
            paradigm=paradigm,
            task=task,
            experiment_type=experiment_type,
            n_channels=n_channels,
            channel_config=channel_config,
            is_complete=False,
            order_by='created_at DESC',
            limit=1,
        )
        return runs[0] if runs else None

    def get_completed_subjects(
        self,
        run_id: str,
        model_type: Optional[str] = None,
    ) -> List[str]:
        """Get list of subjects that have completed training in a run.

        Used for resume: skip already-trained subjects.

        Args:
            run_id: Run identifier
            model_type: Optional filter

        Returns:
            List of subject IDs (e.g., ['S01', 'S02', ...])
        """
        if model_type:
            sql = "SELECT DISTINCT subject_id FROM subject_results WHERE run_id = ? AND model_type = ?"
            params = (run_id, model_type)
        else:
            sql = "SELECT DISTINCT subject_id FROM subject_results WHERE run_id = ?"
            params = (run_id,)

        with self._connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [r['subject_id'] for r in rows]

    # ========================================================================
    # Cross-experiment queries
    # ========================================================================

    def get_subject_history(
        self,
        subject_id: str,
        model_type: Optional[str] = None,
        paradigm: Optional[str] = None,
        task: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get all results for a specific subject across experiments.

        This query was impossible with the old file-based system.

        Returns:
            List of dicts with run + result info.
        """
        clauses = [
            "sr.subject_id = ?",
            "r.is_complete = 1",
        ]
        params: list = [subject_id]

        if model_type:
            clauses.append("sr.model_type = ?")
            params.append(model_type)
        if paradigm:
            clauses.append("r.paradigm = ?")
            params.append(paradigm)
        if task:
            clauses.append("r.task = ?")
            params.append(task)

        where = " AND ".join(clauses)

        with self._connection() as conn:
            rows = conn.execute(
                f"""SELECT r.run_id, r.run_tag, r.experiment_type,
                           r.paradigm, r.task, r.n_channels, r.channel_config,
                           sr.model_type, sr.test_acc_majority,
                           sr.best_val_acc, sr.epochs_trained, sr.training_time
                    FROM subject_results sr
                    JOIN runs r ON sr.run_id = r.run_id
                    WHERE {where}
                    ORDER BY r.created_at DESC""",
                params,
            ).fetchall()
            return [dict(r) for r in rows]

    def get_model_comparison_across_runs(
        self,
        paradigm: str,
        task: str,
        experiment_type: str,
        n_channels: int = 128,
    ) -> List[Dict[str, Any]]:
        """Get model summaries across all matching runs for trend analysis.

        Returns:
            List of dicts with run_tag, model_type, mean_acc, etc.
        """
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT r.run_tag, r.channel_config, r.created_at,
                          ms.model_type, ms.mean_acc, ms.std_acc, ms.n_subjects
                   FROM runs r
                   JOIN model_summaries ms ON r.run_id = ms.run_id
                   WHERE r.paradigm = ? AND r.task = ?
                     AND r.experiment_type = ? AND r.n_channels = ?
                     AND r.is_complete = 1
                   ORDER BY r.created_at ASC, ms.model_type""",
                (paradigm, task, experiment_type, n_channels),
            ).fetchall()
            return [dict(r) for r in rows]

    # ========================================================================
    # High-level query helpers (for plotting / historical data)
    # ========================================================================

    def find_best_within_subject_results(
        self,
        paradigm: str,
        task: str,
        model_type: str,
        n_channels: int = 128,
        channel_config: Optional[str] = None,
        subjects: Optional[set] = None,
        exclude_run_id: Optional[str] = None,
    ) -> Optional[List['TrainingResult']]:
        """Find the best completed within-subject run for a model and return results.

        Replaces find_best_within_subject_for_model() from cache.py.

        Selects the run with the highest mean_acc for the given model from
        completed within-subject experiments. Optionally filters by subject coverage.

        Args:
            paradigm: 'imagery' | 'movement'
            task: 'binary' | 'ternary' | 'quaternary'
            model_type: 'eegnet' | 'cbramod'
            n_channels: Channel count filter (default 128)
            channel_config: Channel config filter (None for 128ch)
            subjects: If provided, only return runs that cover all these subjects
            exclude_run_id: Exclude this run_id from results

        Returns:
            List of TrainingResult for the best run, filtered to subjects if given.
            None if no compatible run found.
        """
        clauses = [
            "r.paradigm = ?",
            "r.task = ?",
            "r.n_channels = ?",
            "r.experiment_type = 'within_subject'",
            "r.is_complete = 1",
            "ms.model_type = ?",
        ]
        params: list = [paradigm, task, n_channels, model_type]

        if channel_config is not None:
            clauses.append("r.channel_config = ?")
            params.append(channel_config)
        if exclude_run_id is not None:
            clauses.append("r.run_id != ?")
            params.append(exclude_run_id)

        where = " AND ".join(clauses)

        with self._connection() as conn:
            rows = conn.execute(
                f"""SELECT r.run_id, ms.mean_acc
                    FROM runs r
                    JOIN model_summaries ms ON r.run_id = ms.run_id
                    WHERE {where}
                    ORDER BY ms.mean_acc DESC""",
                params,
            ).fetchall()

        for row in rows:
            run_id = row['run_id']
            results = self.get_results(run_id, model_type)
            if not results:
                continue

            # Check subject coverage if required
            if subjects is not None:
                result_subjects = {r.subject_id for r in results}
                if not (subjects <= result_subjects):
                    continue
                # Filter to requested subjects
                results = [r for r in results if r.subject_id in subjects]

            return results

        return None

    def find_historical_comparison(
        self,
        paradigm: str,
        task: str,
        n_channels: int = 128,
        channel_config: Optional[str] = None,
        subjects: Optional[set] = None,
        exclude_run_id: Optional[str] = None,
    ) -> Optional[Dict[str, List['TrainingResult']]]:
        """Find best completed within-subject run with BOTH models for comparison plots.

        Replaces find_compatible_historical_results() from cache.py.

        Finds a completed within-subject run that has results for both eegnet
        and cbramod, covering the given subjects. Selects the run with the
        highest combined mean accuracy.

        Args:
            paradigm: 'imagery' | 'movement'
            task: 'binary' | 'ternary' | 'quaternary'
            n_channels: Channel count filter
            channel_config: Channel config filter
            subjects: If provided, both models must cover these subjects
            exclude_run_id: Exclude this run_id

        Returns:
            Dict mapping model_type -> List[TrainingResult], or None.
        """
        clauses = [
            "r.paradigm = ?",
            "r.task = ?",
            "r.n_channels = ?",
            "r.experiment_type = 'within_subject'",
            "r.is_complete = 1",
        ]
        params: list = [paradigm, task, n_channels]

        if channel_config is not None:
            clauses.append("r.channel_config = ?")
            params.append(channel_config)
        if exclude_run_id is not None:
            clauses.append("r.run_id != ?")
            params.append(exclude_run_id)

        where = " AND ".join(clauses)

        # Find runs that have BOTH models with summaries, ordered by combined mean_acc
        with self._connection() as conn:
            rows = conn.execute(
                f"""SELECT r.run_id,
                           e.mean_acc as eegnet_mean,
                           c.mean_acc as cbramod_mean,
                           (COALESCE(e.mean_acc, 0) + COALESCE(c.mean_acc, 0)) as combined
                    FROM runs r
                    JOIN model_summaries e ON r.run_id = e.run_id AND e.model_type = 'eegnet'
                    JOIN model_summaries c ON r.run_id = c.run_id AND c.model_type = 'cbramod'
                    WHERE {where}
                    ORDER BY combined DESC""",
                params,
            ).fetchall()

        for row in rows:
            run_id = row['run_id']
            grouped = self.get_results_by_model(run_id)

            if 'eegnet' not in grouped or 'cbramod' not in grouped:
                continue

            # Check subject coverage for both models
            if subjects is not None:
                eegnet_subs = {r.subject_id for r in grouped['eegnet']}
                cbramod_subs = {r.subject_id for r in grouped['cbramod']}
                if not (subjects <= eegnet_subs) or not (subjects <= cbramod_subs):
                    continue
                # Filter to requested subjects
                grouped['eegnet'] = [r for r in grouped['eegnet'] if r.subject_id in subjects]
                grouped['cbramod'] = [r for r in grouped['cbramod'] if r.subject_id in subjects]

            return grouped

        return None

    def find_best_cross_subject_results(
        self,
        paradigm: str,
        task: str,
        model_type: str,
        n_channels: int = 128,
        channel_config: Optional[str] = None,
        subjects: Optional[set] = None,
        exclude_run_id: Optional[str] = None,
    ) -> Optional[List['TrainingResult']]:
        """Find best completed cross-subject run for a model and return per-subject results.

        Replaces find_compatible_cross_subject_results() for plotting purposes.

        Args:
            paradigm: 'imagery' | 'movement'
            task: 'binary' | 'ternary' | 'quaternary'
            model_type: 'eegnet' | 'cbramod'
            n_channels: Channel count filter
            channel_config: Channel config filter
            subjects: If provided, only match runs covering all these subjects
            exclude_run_id: Exclude this run_id

        Returns:
            List of TrainingResult for the best cross-subject run, or None.
        """
        clauses = [
            "r.paradigm = ?",
            "r.task = ?",
            "r.n_channels = ?",
            "r.experiment_type = 'cross_subject'",
            "r.is_complete = 1",
            "ms.model_type = ?",
        ]
        params: list = [paradigm, task, n_channels, model_type]

        if channel_config is not None:
            clauses.append("r.channel_config = ?")
            params.append(channel_config)
        if exclude_run_id is not None:
            clauses.append("r.run_id != ?")
            params.append(exclude_run_id)

        where = " AND ".join(clauses)

        with self._connection() as conn:
            rows = conn.execute(
                f"""SELECT r.run_id, ms.mean_acc
                    FROM runs r
                    JOIN model_summaries ms ON r.run_id = ms.run_id
                    WHERE {where}
                    ORDER BY ms.mean_acc DESC""",
                params,
            ).fetchall()

        for row in rows:
            run_id = row['run_id']
            results = self.get_results(run_id, model_type)
            if not results:
                continue

            if subjects is not None:
                result_subjects = {r.subject_id for r in results}
                if not (subjects <= result_subjects):
                    continue
                results = [r for r in results if r.subject_id in subjects]

            return results

        return None

    # ========================================================================
    # Utility / housekeeping
    # ========================================================================

    def delete_run(self, run_id: str):
        """Delete a run and all its associated data (CASCADE)."""
        with self._connection() as conn:
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        logger.info(f"Deleted run: {run_id}")

    def run_exists(self, run_id: str) -> bool:
        """Check if a run ID exists."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            return row is not None

    def count_runs(self, **filters) -> int:
        """Count runs matching optional filters."""
        return len(self.find_runs(**filters))

    def _get_run_task(self, run_id: str) -> str:
        """Get the task type for a run (internal helper)."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT task FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            return row['task'] if row else 'binary'

    def __repr__(self) -> str:
        with self._connection() as conn:
            n_runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            n_results = conn.execute("SELECT COUNT(*) FROM subject_results").fetchone()[0]
        return f"ExperimentDB(path={self._db_path!r}, runs={n_runs}, results={n_results})"
