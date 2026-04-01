#!/usr/bin/env python3
"""Describe an ExperimentDB run from a run_tag substring."""

from __future__ import annotations

import argparse
import io
import platform
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

if platform.system() == "Windows":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.results.experiment_db import ExperimentDB

DB_PATH = PROJECT_ROOT / "results" / "experiments.db"


def pct(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def pp(delta: Optional[float]) -> str:
    if delta is None:
        return "N/A"
    return f"{delta * 100:+.2f}pp"


def safe_text(value: Any, default: str = "N/A") -> str:
    if value is None or value == "":
        return default
    return str(value)


def format_channel_config(value: Optional[str]) -> str:
    return value if value else "default"


def format_category(run: Dict[str, Any]) -> str:
    return (
        f"{run['experiment_type']} | {run['paradigm']} | {run['task']} | "
        f"{run['n_channels']}ch | channel_config={format_channel_config(run.get('channel_config'))}"
    )


def fetch_one(
    conn: sqlite3.Connection,
    query: str,
    params: Iterable[Any] = (),
) -> Optional[Dict[str, Any]]:
    row = conn.execute(query, tuple(params)).fetchone()
    return dict(row) if row else None


def fetch_all(
    conn: sqlite3.Connection,
    query: str,
    params: Iterable[Any] = (),
) -> List[Dict[str, Any]]:
    rows = conn.execute(query, tuple(params)).fetchall()
    return [dict(row) for row in rows]


def search_runs(
    conn: sqlite3.Connection,
    substring: str,
    paradigm: Optional[str] = None,
    task: Optional[str] = None,
    experiment_type: Optional[str] = None,
    n_channels: Optional[int] = None,
    channel_config: Optional[str] = None,
) -> List[Dict[str, Any]]:
    clauses = ["r.run_tag LIKE ?"]
    params: List[Any] = [f"%{substring}%"]

    if paradigm:
        clauses.append("r.paradigm = ?")
        params.append(paradigm)
    if task:
        clauses.append("r.task = ?")
        params.append(task)
    if experiment_type:
        clauses.append("r.experiment_type = ?")
        params.append(experiment_type)
    if n_channels is not None:
        clauses.append("r.n_channels = ?")
        params.append(n_channels)
    if channel_config is not None:
        if channel_config.lower() == "default":
            clauses.append("r.channel_config IS NULL")
        else:
            clauses.append("r.channel_config = ?")
            params.append(channel_config)

    where = " AND ".join(clauses)
    query = f"""
        SELECT
            r.*,
            GROUP_CONCAT(ms.model_type, ', ') AS model_types
        FROM runs r
        LEFT JOIN model_summaries ms ON ms.run_id = r.run_id
        WHERE {where}
        GROUP BY r.run_id
        ORDER BY r.created_at DESC
    """
    return fetch_all(conn, query, params)


def choose_run(
    matches: List[Dict[str, Any]],
    substring: str,
    strict: bool,
) -> Tuple[Optional[Dict[str, Any]], str]:
    exact = [row for row in matches if row["run_tag"] == substring]
    if len(exact) == 1:
        return exact[0], "exact"
    if len(matches) == 1:
        return matches[0], "unique"
    if strict:
        return None, "ambiguous"
    return matches[0], "latest"


def print_match_table(matches: List[Dict[str, Any]]) -> None:
    print("候选 runs:")
    print(
        "  "
        f"{'#':>2} | {'run_tag':>13} | {'type':>14} | {'paradigm':>8} | {'task':>9} | "
        f"{'ch':>4} | {'config':>14} | {'models':>18} | {'created':>16}"
    )
    print("  " + "-" * 118)
    for idx, row in enumerate(matches, start=1):
        print(
            "  "
            f"{idx:>2} | {row['run_tag']:>13} | {row['experiment_type'][:14]:>14} | "
            f"{row['paradigm'][:8]:>8} | {row['task'][:9]:>9} | {row['n_channels']:>4} | "
            f"{format_channel_config(row.get('channel_config'))[:14]:>14} | "
            f"{safe_text(row.get('model_types'), '-'):>18} | {safe_text(row.get('created_at'), '')[:16]}"
        )


def get_transfer_config(conn: sqlite3.Connection, run_id: str) -> Optional[Dict[str, Any]]:
    return fetch_one(
        conn,
        "SELECT * FROM transfer_configs WHERE run_id = ?",
        (run_id,),
    )


def get_model_summaries(conn: sqlite3.Connection, run_id: str) -> List[Dict[str, Any]]:
    return fetch_all(
        conn,
        "SELECT * FROM model_summaries WHERE run_id = ? ORDER BY model_type",
        (run_id,),
    )


def get_comparison(conn: sqlite3.Connection, run_id: str) -> Optional[Dict[str, Any]]:
    return fetch_one(
        conn,
        "SELECT * FROM comparisons WHERE run_id = ?",
        (run_id,),
    )


def get_subject_results(
    conn: sqlite3.Connection,
    run_id: str,
    model_type: str,
) -> List[Dict[str, Any]]:
    return fetch_all(
        conn,
        """
        SELECT subject_id, best_val_acc, test_acc, test_acc_majority,
               epochs_trained, training_time
        FROM subject_results
        WHERE run_id = ? AND model_type = ?
        ORDER BY subject_id
        """,
        (run_id, model_type),
    )


def get_model_types_from_subject_results(
    conn: sqlite3.Connection,
    run_id: str,
) -> List[str]:
    rows = fetch_all(
        conn,
        """
        SELECT DISTINCT model_type
        FROM subject_results
        WHERE run_id = ?
        ORDER BY model_type
        """,
        (run_id,),
    )
    return [row["model_type"] for row in rows]


def get_summary_row(
    conn: sqlite3.Connection,
    run_id: str,
    model_type: str,
) -> Optional[Dict[str, Any]]:
    return fetch_one(
        conn,
        """
        SELECT *
        FROM model_summaries
        WHERE run_id = ? AND model_type = ?
        """,
        (run_id, model_type),
    )


def compute_summary_from_subject_rows(
    model_type: str,
    subject_rows: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not subject_rows:
        return None

    accs = [row["test_acc_majority"] for row in subject_rows if row.get("test_acc_majority") is not None]
    if not accs:
        return None

    mean_acc = sum(accs) / len(accs)
    variance = sum((acc - mean_acc) ** 2 for acc in accs) / len(accs)
    return {
        "model_type": model_type,
        "mean_acc": mean_acc,
        "std_acc": variance ** 0.5,
        "median_acc": sorted(accs)[len(accs) // 2] if len(accs) % 2 == 1 else (
            sorted(accs)[len(accs) // 2 - 1] + sorted(accs)[len(accs) // 2]
        ) / 2,
        "min_acc": min(accs),
        "max_acc": max(accs),
        "n_subjects": len(accs),
        "is_baseline": 0,
        "derived_from_subject_results": True,
    }


def collect_summaries(
    conn: sqlite3.Connection,
    run_id: str,
) -> List[Dict[str, Any]]:
    summaries = get_model_summaries(conn, run_id)
    summaries_by_model = {summary["model_type"]: summary for summary in summaries}

    for model_type in get_model_types_from_subject_results(conn, run_id):
        if model_type in summaries_by_model:
            continue
        computed = compute_summary_from_subject_rows(
            model_type,
            get_subject_results(conn, run_id, model_type),
        )
        if computed is not None:
            summaries_by_model[model_type] = computed

    return [summaries_by_model[key] for key in sorted(summaries_by_model)]


def find_exact_designated_baseline(
    conn: sqlite3.Connection,
    run: Dict[str, Any],
    model_type: str,
) -> Optional[Dict[str, Any]]:
    clauses = [
        "ms.is_baseline = 1",
        "ms.model_type = ?",
        "r.paradigm = ?",
        "r.task = ?",
        "r.experiment_type = ?",
        "r.n_channels = ?",
        "r.is_complete = 1",
    ]
    params: List[Any] = [
        model_type,
        run["paradigm"],
        run["task"],
        run["experiment_type"],
        run["n_channels"],
    ]

    if run.get("channel_config") is None:
        clauses.append("r.channel_config IS NULL")
    else:
        clauses.append("r.channel_config = ?")
        params.append(run["channel_config"])

    where = " AND ".join(clauses)
    row = fetch_one(
        conn,
        f"""
        SELECT r.*,
               ms.model_type,
               ms.mean_acc,
               ms.std_acc,
               ms.median_acc,
               ms.min_acc,
               ms.max_acc,
               ms.n_subjects AS summary_n_subjects,
               ms.is_baseline
        FROM runs r
        JOIN model_summaries ms ON ms.run_id = r.run_id
        WHERE {where}
        ORDER BY r.created_at DESC
        LIMIT 1
        """,
        params,
    )
    if row:
        row["baseline_source_label"] = "designated baseline in exact category"
    return row


def get_recorded_baseline_candidates(
    conn: sqlite3.Connection,
    run_id: str,
    model_type: str,
) -> List[Dict[str, Any]]:
    return fetch_all(
        conn,
        """
        SELECT br.ref_type,
               br.model_type AS ref_model_type,
               br.resolved_at,
               r.*,
               ms.mean_acc,
               ms.std_acc,
               ms.median_acc,
               ms.min_acc,
               ms.max_acc,
               ms.n_subjects AS summary_n_subjects,
               ms.is_baseline
        FROM run_baseline_refs br
        JOIN runs r ON r.run_id = br.baseline_run_id
        LEFT JOIN model_summaries ms
               ON ms.run_id = r.run_id AND ms.model_type = ?
        WHERE br.run_id = ?
          AND (br.model_type IS NULL OR br.model_type = ?)
        ORDER BY br.resolved_at DESC, r.created_at DESC
        """,
        (model_type, run_id, model_type),
    )


def find_recorded_baseline_for_summary(
    conn: sqlite3.Connection,
    run: Dict[str, Any],
    model_type: str,
) -> Optional[Dict[str, Any]]:
    candidates = get_recorded_baseline_candidates(conn, run["run_id"], model_type)
    if not candidates:
        return None

    priority_by_type = {
        "within_subject": ["within_subject_baseline", "historical_comparison", "cross_subject_baseline"],
        "cross_subject": ["cross_subject_baseline", "within_subject_baseline", "historical_comparison"],
        "transfer": ["cross_subject_baseline", "within_subject_baseline", "historical_comparison"],
        "extra_sessions": ["within_subject_baseline", "historical_comparison", "cross_subject_baseline"],
    }
    priorities = priority_by_type.get(
        run.get("experiment_type"),
        ["within_subject_baseline", "cross_subject_baseline", "historical_comparison"],
    )
    rank = {ref_type: idx for idx, ref_type in enumerate(priorities)}
    selected = sorted(
        candidates,
        key=lambda row: (
            rank.get(row["ref_type"], len(rank)),
            0 if row.get("ref_model_type") == model_type else 1,
        ),
    )[0]

    combined = dict(selected)
    if combined.get("mean_acc") is None:
        computed = compute_summary_from_subject_rows(
            model_type,
            get_subject_results(conn, combined["run_id"], model_type),
        )
        if computed is not None:
            combined.update(computed)

    model_suffix = f" ({model_type})" if combined.get("ref_model_type") == model_type else ""
    combined["baseline_source_label"] = f"recorded baseline ref: {combined['ref_type']}{model_suffix}"
    return combined


def find_baseline_for_summary(
    conn: sqlite3.Connection,
    db: ExperimentDB,
    run: Dict[str, Any],
    model_type: str,
) -> Optional[Dict[str, Any]]:
    exact = find_exact_designated_baseline(conn, run, model_type)
    if exact is not None:
        return exact

    recorded = find_recorded_baseline_for_summary(conn, run, model_type)
    if recorded is not None:
        return recorded

    fallback = db.find_baseline_run(
        paradigm=run["paradigm"],
        task=run["task"],
        model_type=model_type,
        experiment_type=run["experiment_type"],
        n_channels=run["n_channels"],
    )
    if fallback is None:
        return None

    summary = get_summary_row(conn, fallback["run_id"], model_type) or {}
    combined = dict(fallback)
    combined.update(summary)

    fallback_note = safe_text(fallback.get("baseline_source"), "unknown")
    if run.get("channel_config") != fallback.get("channel_config"):
        fallback_note += "; channel_config not matched"
    combined["baseline_source_label"] = f"ExperimentDB fallback ({fallback_note})"
    return combined


def mean_delta_on_overlap(
    current_results: List[Dict[str, Any]],
    baseline_results: List[Dict[str, Any]],
) -> Tuple[Optional[float], int]:
    current_by_subject = {
        row["subject_id"]: row["test_acc_majority"]
        for row in current_results
        if row.get("test_acc_majority") is not None
    }
    baseline_by_subject = {
        row["subject_id"]: row["test_acc_majority"]
        for row in baseline_results
        if row.get("test_acc_majority") is not None
    }
    common_subjects = sorted(set(current_by_subject) & set(baseline_by_subject))
    if not common_subjects:
        return None, 0

    deltas = [
        current_by_subject[subject_id] - baseline_by_subject[subject_id]
        for subject_id in common_subjects
    ]
    return sum(deltas) / len(deltas), len(common_subjects)


def get_baseline_refs(conn: sqlite3.Connection, run_id: str) -> List[Dict[str, Any]]:
    return fetch_all(
        conn,
        """
        SELECT br.ref_type, br.model_type, br.resolved_at,
               r2.run_tag AS baseline_run_tag
        FROM run_baseline_refs br
        JOIN runs r2 ON r2.run_id = br.baseline_run_id
        WHERE br.run_id = ?
        ORDER BY br.ref_type, br.model_type, r2.run_tag
        """,
        (run_id,),
    )


def print_subject_table(
    model_type: str,
    subject_rows: List[Dict[str, Any]],
) -> None:
    print(f"\n[{model_type.upper()} per-subject]")
    print(
        "  "
        f"{'subject':>7} | {'val':>8} | {'test':>8} | {'epochs':>6} | {'time(s)':>8}"
    )
    print("  " + "-" * 53)
    for row in subject_rows:
        print(
            "  "
            f"{row['subject_id']:>7} | {pct(row['best_val_acc']):>8} | "
            f"{pct(row['test_acc_majority']):>8} | {row['epochs_trained']:>6} | "
            f"{row['training_time']:>8.1f}"
        )


def print_header(
    run: Dict[str, Any],
    substring: str,
    selection_reason: str,
    matches: List[Dict[str, Any]],
) -> None:
    print("=" * 100)
    print(f"Run report: {run['run_tag']}  (query substring: {substring})")
    print(
        f"> 数据来源: ExperimentDB `{DB_PATH.as_posix()}` — "
        f"`SELECT ... FROM runs WHERE run_tag LIKE '%{substring}%'`"
    )
    print("=" * 100)

    if len(matches) > 1:
        print_match_table(matches)
        if selection_reason == "latest":
            print(f"\n共命中 {len(matches)} 条；默认使用最新一条: {run['run_tag']}")
        elif selection_reason == "exact":
            print(f"\n共命中 {len(matches)} 条；其中存在 exact match，使用: {run['run_tag']}")
        print()

    print("[Run]")
    print(f"  run_id: {run['run_id']}")
    print(f"  category: {format_category(run)}")
    print(f"  subjects: {safe_text(run.get('n_subjects'))}")
    print(f"  complete: {'yes' if run.get('is_complete') else 'no'}")
    print(f"  preprocessing: {safe_text(run.get('preprocessing_version'))}")
    print(f"  created_at: {safe_text(run.get('created_at'))}")
    print(f"  updated_at: {safe_text(run.get('updated_at'))}")
    print(f"  git_commit: {safe_text(run.get('git_commit'))}")
    print(f"  wandb_group: {safe_text(run.get('wandb_group'))}")
    print(f"  legacy_source: {safe_text(run.get('legacy_source'))}")
    print(f"  command: {safe_text(run.get('command'))}")


def print_transfer_section(transfer_cfg: Optional[Dict[str, Any]]) -> None:
    if not transfer_cfg:
        return

    print("\n[Transfer config]")
    for key in (
        "freeze_strategy",
        "finetune_epochs",
        "finetune_lr",
        "finetune_batch_size",
        "pretrained_eegnet",
        "pretrained_cbramod",
        "classifier_type",
    ):
        print(f"  {key}: {safe_text(transfer_cfg.get(key))}")


def print_model_section(
    conn: sqlite3.Connection,
    db: ExperimentDB,
    run: Dict[str, Any],
    summary: Dict[str, Any],
    show_subjects: bool,
) -> None:
    model_type = summary["model_type"]
    current_results = get_subject_results(conn, run["run_id"], model_type)
    baseline = find_baseline_for_summary(conn, db, run, model_type)

    print(f"\n[{model_type.upper()}]")
    print(
        "  mean test accuracy (majority): "
        f"{pct(summary.get('mean_acc'))} +/- {pct(summary.get('std_acc'))}"
    )
    print(
        "  median/min/max: "
        f"{pct(summary.get('median_acc'))} / "
        f"{pct(summary.get('min_acc'))} / {pct(summary.get('max_acc'))}"
    )
    print(f"  n_subjects (summary): {safe_text(summary.get('n_subjects'))}")
    print(f"  designated baseline flag: {'yes' if summary.get('is_baseline') else 'no'}")
    if summary.get("derived_from_subject_results"):
        print("  summary source: derived from subject_results (model_summaries missing)")

    if baseline is None:
        print("  baseline comparison: no baseline found")
    else:
        baseline_results = get_subject_results(conn, baseline["run_id"], model_type)
        baseline_mean = baseline.get("mean_acc")
        if baseline_mean is None:
            baseline_mean = baseline.get("best_mean_acc")
        summary_delta = None
        if baseline_mean is not None and summary.get("mean_acc") is not None:
            summary_delta = summary["mean_acc"] - baseline_mean
        overlap_delta, overlap_n = mean_delta_on_overlap(current_results, baseline_results)

        print(
            f"  baseline run: {baseline['run_tag']} "
            f"({baseline.get('baseline_source_label', 'baseline')})"
        )
        print(
            "  baseline mean test accuracy: "
            f"{pct(baseline_mean)} +/- {pct(baseline.get('std_acc'))}"
        )
        if baseline["run_id"] == run["run_id"]:
            print("  delta vs baseline: this run is itself the selected baseline")
        else:
            print(f"  delta vs baseline (summary mean): {pp(summary_delta)}")
            if overlap_delta is not None:
                print(
                    "  delta vs baseline (paired overlap): "
                    f"{pp(overlap_delta)} over {overlap_n} shared subjects"
                )

    if show_subjects and current_results:
        print_subject_table(model_type, current_results)


def print_comparison_section(comparison: Optional[Dict[str, Any]]) -> None:
    if comparison is None:
        return

    print("\n[Within-run model comparison]")
    print(f"  model_a: {safe_text(comparison.get('model_a'))}")
    print(f"  model_b: {safe_text(comparison.get('model_b'))}")
    print(f"  mean diff (CBraMod - EEGNet): {pp(comparison.get('mean_diff'))}")
    print(f"  better_model: {safe_text(comparison.get('better_model'))}")
    print(f"  paired_ttest_t: {safe_text(comparison.get('paired_ttest_t'))}")
    print(f"  paired_ttest_p: {safe_text(comparison.get('paired_ttest_p'))}")
    print(f"  wilcoxon_stat: {safe_text(comparison.get('wilcoxon_stat'))}")
    print(f"  wilcoxon_p: {safe_text(comparison.get('wilcoxon_p'))}")
    print(f"  significant: {'yes' if comparison.get('significant') else 'no'}")


def print_baseline_refs_section(refs: List[Dict[str, Any]]) -> None:
    if not refs:
        return

    print("\n[Recorded baseline refs]")
    for ref in refs:
        model_suffix = f" ({ref['model_type']})" if ref.get("model_type") else ""
        print(
            f"  {ref['ref_type']}{model_suffix}: {ref['baseline_run_tag']} "
            f"[resolved_at={safe_text(ref.get('resolved_at'))}]"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query ExperimentDB by run_tag substring and describe the matched run.",
    )
    parser.add_argument(
        "substring",
        help="run_tag substring, e.g. 0329_1357 or 20260329_1357",
    )
    parser.add_argument("--paradigm", default=None, help="Optional exact filter")
    parser.add_argument("--task", default=None, help="Optional exact filter")
    parser.add_argument("--type", dest="experiment_type", default=None, help="Optional exact filter")
    parser.add_argument("--channels", type=int, default=None, help="Optional exact filter")
    parser.add_argument(
        "--channel-config",
        default=None,
        help="Optional exact filter; use 'default' to match NULL channel_config",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of auto-selecting the latest match when multiple runs are found",
    )
    parser.add_argument(
        "--show-subjects",
        action="store_true",
        help="Also print the per-subject result table for each model",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not DB_PATH.exists():
        print(f"ERROR: ExperimentDB not found at {DB_PATH}")
        return 1

    db = ExperimentDB(str(DB_PATH))
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    try:
        matches = search_runs(
            conn,
            substring=args.substring.strip(),
            paradigm=args.paradigm,
            task=args.task,
            experiment_type=args.experiment_type,
            n_channels=args.channels,
            channel_config=args.channel_config,
        )
        if not matches:
            print(f"未找到匹配 run: substring='{args.substring}'")
            print(
                f"> 数据来源: ExperimentDB `{DB_PATH.as_posix()}` — "
                f"`SELECT ... FROM runs WHERE run_tag LIKE '%{args.substring}%'`"
            )
            return 1

        selected_run, selection_reason = choose_run(matches, args.substring.strip(), args.strict)
        if selected_run is None:
            print(f"匹配到 {len(matches)} 条 run，`--strict` 模式下不会自动选择。")
            print_match_table(matches)
            return 1

        summaries = collect_summaries(conn, selected_run["run_id"])
        transfer_cfg = get_transfer_config(conn, selected_run["run_id"])
        comparison = get_comparison(conn, selected_run["run_id"])
        baseline_refs = get_baseline_refs(conn, selected_run["run_id"])

        print_header(selected_run, args.substring.strip(), selection_reason, matches)
        print_transfer_section(transfer_cfg)

        if summaries:
            print("\n[Model summaries]")
            for summary in summaries:
                print_model_section(
                    conn,
                    db,
                    selected_run,
                    summary,
                    show_subjects=args.show_subjects,
                )
        else:
            print("\n[Model summaries]")
            print("  No model_summaries rows found for this run.")

        print_comparison_section(comparison)
        print_baseline_refs_section(baseline_refs)
        print("\n" + "=" * 100)
        return 0
    finally:
        conn.close()
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
