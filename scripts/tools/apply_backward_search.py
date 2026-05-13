"""Apply backward-search agent findings to ExperimentDB.

Reads JSON outputs from docs/dev_log/backward_search_2026-05-13/ and calls
db.set_purpose(...) with provenance='backward_search' for each (run_tag -> run_ids)
mapping. Skips any run that already has purpose_provenance='explicit' so that
authoritative memory-entry tags and baseline tags are never overwritten.

Self-contained: builds bucket -> run_tag -> run_id index from the live DB,
filtering by the same bucket criteria used when the agents were dispatched.
Idempotent: re-running after a successful apply is a no-op for already-tagged
runs (the explicit-skip guard short-circuits them).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from src.results.experiment_db import ExperimentDB


# Each bucket: (json_basename, SQL where-clause defining the bucket's scope).
# The where-clause matches the bucket criteria used when CSVs were exported
# for the research agents — see scripts/tools/apply_backward_search.py history.
BUCKETS = [
    ("a", "bucket_a_128ch_main",
     "n_channels = 128 AND experiment_type IN "
     "('within_subject', 'cross_subject', 'transfer')"),
    ("b", "bucket_b_extra_sessions",
     "experiment_type = 'extra_sessions'"),
    ("c", "bucket_c_reduced_channel",
     "(n_channels != 128 OR (channel_config IS NOT NULL AND channel_config != ''))"),
]

FINDINGS_ROOT = Path("docs/dev_log/backward_search_2026-05-13")


def load_bucket_index(db: ExperimentDB, where: str) -> Dict[str, List[str]]:
    """Build run_tag -> [run_id, ...] index from the live DB scoped to bucket."""
    idx: Dict[str, List[str]] = defaultdict(list)
    with db._connection() as conn:
        rows = conn.execute(
            f"SELECT run_tag, run_id FROM runs WHERE {where}"
        ).fetchall()
    for row in rows:
        idx[row["run_tag"]].append(row["run_id"])
    return idx


def apply_findings(db: ExperimentDB, bucket_index: Dict[str, List[str]], findings_path: Path):
    with findings_path.open(encoding="utf-8") as fh:
        findings = json.load(fh)

    stats = {"applied": 0, "skipped_explicit": 0, "skipped_not_found": 0,
             "subgroups": 0, "duplicate_overwrite": 0}
    seen_run_ids: Dict[str, str] = {}  # run_id -> subgroup name (for diagnostic)

    for sg in findings["subgroups"]:
        stats["subgroups"] += 1
        purpose = sg["purpose"]
        notes_body = (
            f"{sg['hypothesis']}\n"
            f"---\n"
            f"Group: {sg['name']} (backward-searched, confidence={sg['confidence']})\n"
            f"Rationale: {sg['rationale']}"
        )

        # Resolve run_ids: explicit run_ids preferred, else run_tag lookup
        run_ids: List[str] = []
        if "run_ids" in sg:
            run_ids.extend(sg["run_ids"])
        for tag in sg.get("run_tags", []):
            matches = bucket_index.get(tag, [])
            if not matches:
                stats["skipped_not_found"] += 1
                print(f"  [not-found] {sg['name']}: run_tag={tag} not in bucket")
                continue
            run_ids.extend(matches)

        for run_id in run_ids:
            existing = db.get_run(run_id)
            if existing is None:
                stats["skipped_not_found"] += 1
                print(f"  [not-found] {sg['name']}: run_id={run_id} not in DB")
                continue
            if existing.get("purpose_provenance") == "explicit":
                stats["skipped_explicit"] += 1
                print(f"  [skip-explicit] {run_id} (already authoritative)")
                continue
            if run_id in seen_run_ids:
                stats["duplicate_overwrite"] += 1
                print(f"  [overwrite] {run_id}: was '{seen_run_ids[run_id]}' -> '{sg['name']}'")
            db.set_purpose(
                run_id,
                purpose=purpose,
                provenance="backward_search",
                notes=notes_body,
                notes_mode="replace",
            )
            seen_run_ids[run_id] = sg["name"]
            stats["applied"] += 1

    return stats


def main():
    db = ExperimentDB()
    print(f"=== Backward-search application ===\n")

    grand = defaultdict(int)
    for key, json_base, where in BUCKETS:
        json_path = FINDINGS_ROOT / f"{json_base}.json"
        print(f"\n--- bucket {key}: {json_path.name} ---")
        bucket_index = load_bucket_index(db, where)
        stats = apply_findings(db, bucket_index, json_path)
        for k, v in stats.items():
            grand[k] += v
        print(f"  bucket-{key} totals: {dict(stats)}")

    print(f"\n=== GRAND TOTALS ===")
    for k, v in grand.items():
        print(f"  {k}: {v}")

    # Final consistency report
    print(f"\n=== Final DB state ===")
    with db._connection() as conn:
        rows = conn.execute("""
            SELECT purpose, purpose_provenance, COUNT(*) AS n
            FROM runs
            GROUP BY purpose, purpose_provenance
            ORDER BY n DESC
        """).fetchall()
        for r in rows:
            print(f"  purpose={r['purpose']!r:>20} provenance={r['purpose_provenance']!r:>20} count={r['n']}")

    db.close()


if __name__ == "__main__":
    main()
