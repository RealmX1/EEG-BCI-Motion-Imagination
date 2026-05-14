"""CLI for the experiment queue (pending_runs table, schema v11).

Subcommands:
    add            Register a new pending run
    list           Show queue contents (default: non-terminal entries only)
    show           Inspect a single entry by id
    rm             Cancel a not-yet-claimed entry
    set            Manually update fields (status, priority, command, notes)
    run            Start the runner (sequential executor)
    has-attention  Probe: exit 0 if any entry needs_attention, else 1

Example:
    # Register
    uv run python scripts/queue/cli.py add \\
        --command "uv run python scripts/experiments/run_within_subject.py \\
                   --subjects S01 --cache-only --no-wandb --no-plot \\
                   --purpose ablation --notes 'H: drop attention head'" \\
        --purpose ablation \\
        --notes "H: drop attention head" \\
        --priority 5

    # Browse
    uv run python scripts/queue/cli.py list
    uv run python scripts/queue/cli.py list --all --json

    # Cancel
    uv run python scripts/queue/cli.py rm 42

    # Run (typically wrapped by /long-run skill)
    uv run python scripts/queue/cli.py run
    uv run python scripts/queue/cli.py run --drain-and-exit
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from src.config.constants import PURPOSE_VALUES, QUEUE_STATUS_VALUES
from src.results.experiment_db import ExperimentDB


# ──────────────────────────────────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────────────────────────────────

_COLUMNS = ("id", "status", "priority", "purpose", "created_at",
            "command", "error_summary")


def _truncate(s: Optional[str], width: int) -> str:
    if s is None:
        return ""
    s = str(s).replace("\n", " ")
    return s if len(s) <= width else s[: width - 3] + "..."


def _print_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("(queue empty)")
        return
    widths = {
        "id": 4, "status": 16, "priority": 4, "purpose": 12,
        "created_at": 19, "command": 60, "error_summary": 30,
    }
    # Header
    hdr = " | ".join(f"{col:<{widths[col]}}" for col in _COLUMNS)
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = " | ".join(
            f"{_truncate(r.get(col), widths[col]):<{widths[col]}}"
            for col in _COLUMNS
        )
        print(line)


# ──────────────────────────────────────────────────────────────────────────
# Subcommands
# ──────────────────────────────────────────────────────────────────────────

def cmd_add(args: argparse.Namespace) -> int:
    db = ExperimentDB()
    try:
        new_id = db.enqueue_run(
            command=args.command,
            purpose=args.purpose,
            notes=args.notes,
            priority=args.priority,
            created_by=args.created_by,
        )
        print(f"enqueued id={new_id} priority={args.priority} "
              f"purpose={args.purpose}")
    finally:
        db.close()
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    db = ExperimentDB()
    try:
        rows = db.list_pending(
            status=args.status,
            include_terminal=args.all,
            limit=args.limit,
        )
        if args.json:
            print(json.dumps(rows, indent=2, default=str))
        else:
            _print_table(rows)
    finally:
        db.close()
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    db = ExperimentDB()
    try:
        row = db.get_pending(args.id)
        if row is None:
            print(f"no pending entry id={args.id}", file=sys.stderr)
            return 1
        print(json.dumps(row, indent=2, default=str))
    finally:
        db.close()
    return 0


def cmd_rm(args: argparse.Namespace) -> int:
    db = ExperimentDB()
    try:
        db.cancel_pending(args.id)
        print(f"cancelled id={args.id}")
    finally:
        db.close()
    return 0


def cmd_set(args: argparse.Namespace) -> int:
    """Manual surgery on a pending entry.

    Most useful for the monitor agent: e.g., after diagnosing a CUDA OOM,
    rewrite the command (`--set command "..."`) and flip status back to
    'pending' (`--set status pending`) so the runner picks it up again.
    """
    db = ExperimentDB()
    try:
        # Only allow whitelisted fields
        kwargs: Dict[str, Any] = {}
        if args.error_summary is not None:
            kwargs["error_summary"] = args.error_summary
        if args.handoff_path is not None:
            kwargs["handoff_path"] = args.handoff_path
        if args.completed_run_id is not None:
            kwargs["completed_run_id"] = args.completed_run_id
        if args.command is not None:
            kwargs["new_command"] = args.command
        if args.increment_debug:
            kwargs["increment_debug_attempts"] = True
        # status is required (state-machine transition is the whole point)
        db.update_pending_status(args.id, args.status, **kwargs)
        print(f"updated id={args.id} -> status={args.status}")
    finally:
        db.close()
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    # Delegate to runner.main(argv=...) so flags map 1:1
    from scripts.queue import runner
    argv: List[str] = []
    if args.drain_and_exit:
        argv.append("--drain-and-exit")
    if args.skip_preflight:
        argv.append("--skip-preflight")
    return runner.main(argv)


def cmd_has_attention(args: argparse.Namespace) -> int:
    """Probe-style check: exit 0 if any entry needs monitor attention, else 1.

    Designed for shell loops:
        if uv run python scripts/queue/cli.py has-attention; then ...
    stdout always prints the count for visibility.
    """
    db = ExperimentDB()
    try:
        rows = db.list_pending(status='needs_attention')
        print(len(rows))
        return 0 if rows else 1
    finally:
        db.close()


# ──────────────────────────────────────────────────────────────────────────
# Argparse wiring
# ──────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="queue", description="Experiment queue CLI (pending_runs table)"
    )
    sub = p.add_subparsers(dest="subcommand", required=True)

    # add
    p_add = sub.add_parser("add", help="Register a new pending run")
    p_add.add_argument("--command", required=True,
                       help="Full shell command line to execute")
    p_add.add_argument("--purpose", choices=sorted(PURPOSE_VALUES),
                       default=None,
                       help="Intent tag (controlled vocab). Encodes WHY the "
                            "run is launched (hypothesis), NOT post-hoc "
                            "analysis or outcome.")
    p_add.add_argument("--notes", default=None,
                       help="Free-form hypothesis text")
    p_add.add_argument("--priority", type=int, default=0,
                       help="Higher number runs first (default 0)")
    p_add.add_argument("--created-by", default=None,
                       help="Optional researcher / agent identifier")
    p_add.set_defaults(func=cmd_add)

    # list
    p_list = sub.add_parser("list", help="Show queue contents")
    p_list.add_argument("--all", action="store_true",
                        help="Include terminal states (completed/failed/...)")
    p_list.add_argument("--status", choices=sorted(QUEUE_STATUS_VALUES),
                        default=None, help="Filter to a single status")
    p_list.add_argument("--limit", type=int, default=None)
    p_list.add_argument("--json", action="store_true",
                        help="Output as JSON (for monitor agent consumption)")
    p_list.set_defaults(func=cmd_list)

    # show
    p_show = sub.add_parser("show", help="Show one entry by id")
    p_show.add_argument("id", type=int)
    p_show.set_defaults(func=cmd_show)

    # rm
    p_rm = sub.add_parser("rm", help="Cancel an unclaimed entry")
    p_rm.add_argument("id", type=int)
    p_rm.set_defaults(func=cmd_rm)

    # set
    p_set = sub.add_parser(
        "set",
        help="Update fields on a pending entry (monitor agent surgery)",
    )
    p_set.add_argument("id", type=int)
    p_set.add_argument("--status", choices=sorted(QUEUE_STATUS_VALUES),
                       required=True)
    p_set.add_argument("--error-summary", default=None)
    p_set.add_argument("--handoff-path", default=None)
    p_set.add_argument("--completed-run-id", default=None)
    p_set.add_argument("--command", default=None,
                       help="Rewrite the command (e.g., post-debug retry)")
    p_set.add_argument("--increment-debug", action="store_true",
                       help="Bump debug_attempts (use when retrying after fix)")
    p_set.set_defaults(func=cmd_set)

    # run
    p_run = sub.add_parser("run", help="Start the queue runner")
    p_run.add_argument("--drain-and-exit", action="store_true",
                       help="Exit once queue is empty")
    p_run.add_argument("--skip-preflight", action="store_true",
                       help="Skip first-entry GPU sanity check")
    p_run.set_defaults(func=cmd_run)

    # has-attention (probe for monitor agents)
    p_attn = sub.add_parser(
        "has-attention",
        help="Exit 0 if any entry needs_attention, else exit 1 "
             "(prints count to stdout)",
    )
    p_attn.set_defaults(func=cmd_has_attention)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
