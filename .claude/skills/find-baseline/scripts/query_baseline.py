#!/usr/bin/env python3
"""Query ExperimentDB for baseline results."""

import argparse
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
DB_PATH = PROJECT_ROOT / "results" / "experiments.db"

# Post-HPO cutoff date
POST_HPO_DATE = "2026-03-20"


def _has_column(db, table, column):
    """Check if a column exists in a table."""
    cols = {row[1] for row in db.execute(f"PRAGMA table_info({table})").fetchall()}
    return column in cols


def _has_table(db, table):
    """Check if a table exists."""
    row = db.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row[0] > 0


def query_top_results(args):
    """Query top results matching the given filters."""
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row

    has_baseline = _has_column(db, "runs", "is_baseline")

    conditions = [
        "ms.model_type = ?",
        "r.task = ?",
        "r.paradigm = ?",
        "ms.n_subjects >= ?",
    ]
    params = [args.model, args.task, args.paradigm, args.subjects]

    if args.type != "all":
        conditions.append("r.experiment_type = ?")
        params.append(args.type)

    if args.channels:
        conditions.append("r.n_channels = ?")
        params.append(args.channels)

    if args.post_hpo:
        conditions.append("r.created_at >= ?")
        params.append(POST_HPO_DATE)

    if args.baseline_only and has_baseline:
        conditions.append("r.is_baseline = 1")

    if not args.include_unified and args.task != 'unified':
        conditions.append("r.task != 'unified'")

    where = " AND ".join(conditions)

    baseline_col = ", r.is_baseline" if has_baseline else ""
    query = f"""
        SELECT r.run_tag, r.experiment_type, r.n_channels, r.created_at,
               ms.mean_acc, ms.std_acc, ms.n_subjects{baseline_col}
        FROM runs r
        JOIN model_summaries ms ON r.run_id = ms.run_id
        WHERE {where}
        ORDER BY ms.mean_acc DESC
        LIMIT ?
    """
    params.append(args.top)

    cursor = db.execute(query, params)
    rows = cursor.fetchall()

    if not rows:
        print(f"No results found for: model={args.model} task={args.task} "
              f"paradigm={args.paradigm} type={args.type} channels={args.channels} "
              f"subjects>={args.subjects} post_hpo={args.post_hpo}")
        # Try without post-HPO filter as fallback hint
        if args.post_hpo:
            conditions_no_hpo = [c for c in conditions if "created_at" not in c]
            params_no_hpo = [p for p, c in zip(params[:-1], conditions) if "created_at" not in c]
            where_no_hpo = " AND ".join(conditions_no_hpo)
            cursor2 = db.execute(
                f"SELECT COUNT(*) FROM runs r JOIN model_summaries ms ON r.run_id = ms.run_id WHERE {where_no_hpo}",
                params_no_hpo,
            )
            count = cursor2.fetchone()[0]
            if count > 0:
                print(f"  (Found {count} results without post-HPO filter. Use --no-post-hpo to include them.)")
        db.close()
        return

    # Print header
    width = 95 if has_baseline else 85
    print(f"\n{'='*width}")
    print(f"  Baseline: {args.model.upper()} | {args.task} | {args.paradigm} | {args.type} | {args.channels}ch")
    filters = []
    if args.post_hpo:
        filters.append(f"post-HPO (>= {POST_HPO_DATE})")
    if args.baseline_only:
        filters.append("baseline-only")
    if args.include_unified:
        filters.append("including unified")
    if filters:
        print(f"  Filter: {', '.join(filters)}")
    print(f"{'='*width}")

    if has_baseline:
        print(f"{'#':>2} | {'run_tag':>16} | {'type':>15} | {'mean':>7} | {'std':>6} | {'n':>3} | {'created':>16} | BL")
    else:
        print(f"{'#':>2} | {'run_tag':>16} | {'type':>15} | {'mean':>7} | {'std':>6} | {'n':>3} | {'created'}")
    print(f"{'-'*width}")

    for i, r in enumerate(rows, 1):
        bl_marker = ""
        if has_baseline:
            bl_marker = " | **" if r['is_baseline'] else " |   "
        print(
            f"{i:>2} | {r['run_tag']:>16} | {r['experiment_type']:>15} | "
            f"{r['mean_acc']*100:>6.2f}% | {r['std_acc']*100:>5.2f}% | "
            f"{r['n_subjects']:>3} | {r['created_at'][:16]}{bl_marker}"
        )

    print(f"{'='*width}")

    # Find the current baseline (latest is_baseline=1 by date) vs best by accuracy
    best_acc = rows[0]
    baseline_rows = [r for r in rows if has_baseline and r['is_baseline']]
    if baseline_rows:
        # Latest baseline by creation date
        current_bl = max(baseline_rows, key=lambda r: r['created_at'])
        print(f"  BASELINE (current): {current_bl['mean_acc']*100:.2f}% +/- {current_bl['std_acc']*100:.2f}% "
              f"(run_tag={current_bl['run_tag']}, {current_bl['created_at'][:10]})")
        if best_acc['run_tag'] != current_bl['run_tag']:
            print(f"  BEST (by accuracy): {best_acc['mean_acc']*100:.2f}% +/- {best_acc['std_acc']*100:.2f}% "
                  f"(run_tag={best_acc['run_tag']}, {best_acc['created_at'][:10]})")
    else:
        print(f"  BEST: {best_acc['mean_acc']*100:.2f}% +/- {best_acc['std_acc']*100:.2f}% "
              f"(run_tag={best_acc['run_tag']}, {best_acc['created_at'][:10]})")
    print()

    db.close()


def query_run_detail(run_tag):
    """Show per-subject detail for a specific run."""
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row

    has_baseline = _has_column(db, "runs", "is_baseline")

    # Find the run
    cursor = db.execute(
        "SELECT * FROM runs WHERE run_tag = ?", (run_tag,)
    )
    run = cursor.fetchone()
    if not run:
        print(f"Run not found: {run_tag}")
        db.close()
        return

    # Get model summary
    cursor = db.execute(
        "SELECT * FROM model_summaries WHERE run_id = ?", (run["run_id"],)
    )
    summaries = cursor.fetchall()

    # Get per-subject results
    cursor = db.execute(
        "SELECT * FROM subject_results WHERE run_id = ? ORDER BY subject_id",
        (run["run_id"],),
    )
    subjects = cursor.fetchall()

    print(f"\n{'='*80}")
    run_label = f"  Run: {run_tag}"
    if has_baseline and run['is_baseline']:
        run_label += "  [BASELINE]"
    print(run_label)
    print(f"  Type: {run['experiment_type']} | Task: {run['task']} | "
          f"Paradigm: {run['paradigm']} | Channels: {run['n_channels']}")
    print(f"  Created: {run['created_at']} | Subjects: {run['n_subjects']}")
    if run["git_commit"]:
        print(f"  Git: {run['git_commit'][:8]}")
    print(f"{'='*80}")

    for ms in summaries:
        print(f"\n  {ms['model_type'].upper()}: "
              f"{ms['mean_acc']*100:.2f}% +/- {ms['std_acc']*100:.2f}% "
              f"(median={ms['median_acc']*100:.2f}%, "
              f"min={ms['min_acc']*100:.2f}%, max={ms['max_acc']*100:.2f}%)")

    if subjects:
        print(f"\n  {'Subject':>8} | {'Val Acc':>8} | {'Test Acc':>8} | {'Epochs':>6} | {'Time':>8}")
        print(f"  {'-'*50}")
        for s in subjects:
            print(
                f"  {s['subject_id']:>8} | {s['best_val_acc']*100:>7.2f}% | "
                f"{s['test_acc']*100:>7.2f}% | {s['epochs_trained']:>6} | "
                f"{s['training_time']:>7.1f}s"
            )

    # Show baseline references (v7+)
    has_refs = _has_table(db, "run_baseline_refs")
    if has_refs:
        refs = db.execute(
            "SELECT br.ref_type, br.model_type, r2.run_tag AS bl_tag "
            "FROM run_baseline_refs br "
            "JOIN runs r2 ON br.baseline_run_id = r2.run_id "
            "WHERE br.run_id = ? ORDER BY br.ref_type, br.model_type",
            (run["run_id"],),
        ).fetchall()
        if refs:
            print(f"\n  Baseline References:")
            for ref in refs:
                mt_label = f" ({ref['model_type']})" if ref['model_type'] else ""
                print(f"    -> {ref['ref_type']}{mt_label}: {ref['bl_tag']}")

        # Show runs that reference this run as baseline
        reverse = db.execute(
            "SELECT r2.run_tag AS ref_tag, br.ref_type, br.model_type "
            "FROM run_baseline_refs br "
            "JOIN runs r2 ON br.run_id = r2.run_id "
            "WHERE br.baseline_run_id = ? ORDER BY r2.run_tag",
            (run["run_id"],),
        ).fetchall()
        if reverse:
            print(f"\n  Referenced as Baseline by:")
            for rev in reverse:
                mt_label = f" ({rev['model_type']})" if rev['model_type'] else ""
                print(f"    <- {rev['ref_tag']} [{rev['ref_type']}{mt_label}]")

    print(f"{'='*80}\n")
    db.close()


def main():
    parser = argparse.ArgumentParser(description="Query ExperimentDB for baseline results")
    parser.add_argument("--model", default="cbramod", choices=["cbramod", "eegnet"])
    parser.add_argument("--task", default="binary", choices=["binary", "ternary", "quaternary", "unified"])
    parser.add_argument("--paradigm", default="imagery", choices=["imagery", "movement"])
    parser.add_argument("--type", default="within_subject",
                        choices=["within_subject", "cross_subject", "transfer", "all"])
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--subjects", type=int, default=21)
    parser.add_argument("--post-hpo", action="store_true", default=True)
    parser.add_argument("--no-post-hpo", dest="post_hpo", action="store_false")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--tag", type=str, default=None, help="Show detail for a specific run_tag")
    parser.add_argument("--baseline-only", action="store_true", default=False,
                        help="Only show runs marked as baseline")
    parser.add_argument("--include-unified", action="store_true", default=False,
                        help="Include unified model results (excluded by default)")

    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"ERROR: ExperimentDB not found at {DB_PATH}")
        sys.exit(1)

    if args.tag:
        query_run_detail(args.tag)
    else:
        query_top_results(args)


if __name__ == "__main__":
    main()
