"""Sequential queue runner for pending_runs (schema v11).

Pulls entries from pending_runs in priority DESC, FIFO order and executes each
command in turn. Cooperates with a monitor subagent via the pending_runs.status
state machine.

GPU coordination policy (per user decision 2026-05-14):
- BEFORE the first entry of this runner session: 1-min GPU sanity check.
  If GPU is busy, fall back to a 10-min idle wait (rolling window check).
- BETWEEN entries: no GPU gating. Sequential dequeue alone guarantees
  single-GPU exclusivity (the previous subprocess has exited by the time we
  read the next entry; nothing else can claim the GPU mid-loop).

State machine (see also src/config/constants.py::QUEUE_STATUS_VALUES):
    pending  -- enqueued, waiting for runner
    claimed  -- runner picked it up, about to spawn subprocess
    running  -- subprocess in flight
    completed       (terminal) -- success
    needs_attention -- subprocess failed; monitor agent owns the decision
    failed          (terminal) -- monitor agent gave up
    skipped         (terminal) -- monitor agent skipped (handoff written)
    cancelled       (terminal) -- researcher cancelled before claim

Run with `--drain-and-exit` to exit when the queue is empty; otherwise the
runner polls forever for new entries (default for /long-run mode).
"""

from __future__ import annotations

import argparse
import shlex
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.results.experiment_db import ExperimentDB

# Lazily imported so module import doesn't depend on nvidia-smi being present
try:
    from scripts.overwatch.gpu_overwatch import query_gpu
except ImportError:
    query_gpu = None  # type: ignore[assignment]


# ── Tuning ─────────────────────────────────────────────────────────────────
SANITY_WINDOW_S = 60          # 1 min sanity check before first entry
SANITY_POLL_S = 5             # sample every 5 s during sanity check
FALLBACK_WINDOW_S = 10 * 60   # 10 min rolling window for fallback wait
FALLBACK_POLL_S = 30          # sample every 30 s during fallback wait
UTIL_THRESHOLD = 10.0         # GPU idle threshold (% utilization, avg over window)

NEEDS_ATTENTION_POLL_S = 30   # how often runner polls DB while paused
EMPTY_QUEUE_POLL_S = 60       # how often runner polls when queue is empty
MAX_NEEDS_ATTENTION_WAIT_S = 60 * 60   # 1 h — auto-fail if no monitor decision

STDERR_TAIL_LIMIT = 200             # max stderr lines retained in memory
ERROR_SUMMARY_MAX_BYTES = 1500      # max bytes stored in error_summary column

# ── Module state for signal handling ───────────────────────────────────────
_current_proc: Optional[subprocess.Popen] = None
_current_pending_id: Optional[int] = None
_db: Optional[ExperimentDB] = None


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ──────────────────────────────────────────────────────────────────────────
# GPU pre-flight check
# ──────────────────────────────────────────────────────────────────────────

def _sample_gpu_utils(duration_s: int, poll_s: int) -> list[float]:
    """Sample GPU utilization N times (N = duration_s // poll_s).

    Returns list of util%. Uses a deterministic count rather than wall-clock
    deadline so the log is predictable ("12 samples" not "11 or 12").
    """
    if query_gpu is None:
        print(f"[{_ts()}] [queue] WARN: gpu_overwatch.query_gpu unavailable; "
              f"skipping GPU pre-flight (assuming idle)")
        return [0.0]
    n_samples = max(1, duration_s // poll_s)
    samples: list[float] = []
    for _ in range(n_samples):
        try:
            util, _vram = query_gpu()
            samples.append(util)
        except Exception as e:
            print(f"[{_ts()}] [queue] nvidia-smi error: {e}")
        time.sleep(poll_s)
    return samples or [0.0]


def quick_gpu_sanity_check() -> bool:
    """1-min sanity check. Returns True if avg utilization < threshold."""
    print(f"[{_ts()}] [queue] pre-flight: {SANITY_WINDOW_S}s GPU sanity check ...")
    samples = _sample_gpu_utils(SANITY_WINDOW_S, SANITY_POLL_S)
    avg = sum(samples) / len(samples)
    print(f"[{_ts()}] [queue] sanity avg util = {avg:.1f}% "
          f"({len(samples)} samples)")
    return avg < UTIL_THRESHOLD


def wait_for_idle_10min(max_wait_s: int = 8 * 3600) -> None:
    """Fallback wait: rolling 10-min window, exit when avg util < threshold.

    Blocks until idle, or raises RuntimeError if max_wait_s exceeded.
    """
    print(f"[{_ts()}] [queue] fallback: waiting for "
          f"{FALLBACK_WINDOW_S // 60}-min rolling avg < {UTIL_THRESHOLD}% "
          f"(max wait {max_wait_s // 60} min)")
    if query_gpu is None:
        print(f"[{_ts()}] [queue] WARN: nvidia-smi unavailable; skipping wait")
        return
    samples: deque[tuple[float, float]] = deque()  # (timestamp, util)
    t_start = time.monotonic()
    while True:
        if time.monotonic() - t_start > max_wait_s:
            raise RuntimeError(
                f"GPU idle wait exceeded {max_wait_s}s; aborting runner"
            )
        try:
            util, _vram = query_gpu()
        except Exception as e:
            print(f"[{_ts()}] [queue] nvidia-smi error: {e}")
            time.sleep(FALLBACK_POLL_S)
            continue
        now = time.monotonic()
        samples.append((now, util))
        # Evict samples older than the rolling window
        cutoff = now - FALLBACK_WINDOW_S
        while samples and samples[0][0] < cutoff:
            samples.popleft()
        window_span = samples[-1][0] - samples[0][0]
        avg = sum(s[1] for s in samples) / len(samples)
        print(f"[{_ts()}] [queue] now: {util:5.1f}% | "
              f"avg({len(samples)}): {avg:5.1f}% | "
              f"window: {int(window_span)}s")
        # Need at least 80% of the window's worth of data
        if window_span >= FALLBACK_WINDOW_S * 0.8 and avg < UTIL_THRESHOLD:
            print(f"[{_ts()}] [queue] GPU idle, proceeding")
            return
        time.sleep(FALLBACK_POLL_S)


def preflight_first_entry() -> None:
    """Run once before claiming the first entry in this runner session."""
    if quick_gpu_sanity_check():
        print(f"[{_ts()}] [queue] pre-flight OK, GPU idle")
        return
    print(f"[{_ts()}] [queue] pre-flight: GPU busy, deferring to "
          f"{FALLBACK_WINDOW_S // 60}-min fallback wait")
    wait_for_idle_10min()


# ──────────────────────────────────────────────────────────────────────────
# Execution & state machine
# ──────────────────────────────────────────────────────────────────────────

def discover_run_id(db: ExperimentDB, entry: Dict[str, Any]) -> Optional[str]:
    """Best-effort: find the run_id this command produced.

    Heuristic: newest runs row with created_at >= entry.claimed_at AND
    command LIKE '%<first_script_name>%'. Returns None if no match.
    """
    claimed_at = entry.get("claimed_at")
    if not claimed_at:
        return None
    # Find a distinctive token in the command (e.g., script name)
    try:
        tokens = shlex.split(entry["command"])
    except ValueError:
        tokens = entry["command"].split()
    script_token = next(
        (t for t in tokens if t.endswith(".py") or t.endswith(".sh")),
        None,
    )
    sql = "SELECT run_id FROM runs WHERE created_at >= ?"
    params: list[Any] = [claimed_at]
    if script_token:
        sql += " AND command LIKE ?"
        params.append(f"%{script_token}%")
    sql += " ORDER BY created_at DESC LIMIT 1"
    with db._connection() as conn:
        row = conn.execute(sql, params).fetchone()
    return row["run_id"] if row else None


def _capture_tail_from_proc(stderr_lines: list[str], n: int = 30) -> str:
    """Last N stderr lines, joined for storage in error_summary."""
    tail = stderr_lines[-n:] if stderr_lines else []
    return "\n".join(tail).strip()


def wait_for_resolution(db: ExperimentDB, pending_id: int) -> str:
    """Poll DB; return new status once monitor flips it.

    If MAX_NEEDS_ATTENTION_WAIT_S elapses with no decision, auto-fail and
    return 'failed' so the runner moves on. The monitor can still come back
    later and re-enqueue if it has a fix.
    """
    print(f"[{_ts()}] [queue] #{pending_id} needs_attention; "
          f"waiting for monitor decision (poll every {NEEDS_ATTENTION_POLL_S}s, "
          f"max {MAX_NEEDS_ATTENTION_WAIT_S // 60} min)")
    t_start = time.monotonic()
    while True:
        time.sleep(NEEDS_ATTENTION_POLL_S)
        row = db.get_pending(pending_id)
        if row is None:
            print(f"[{_ts()}] [queue] #{pending_id} disappeared; treating as failed")
            return "failed"
        if row["status"] != "needs_attention":
            print(f"[{_ts()}] [queue] #{pending_id} monitor decision: {row['status']}")
            return row["status"]
        if time.monotonic() - t_start > MAX_NEEDS_ATTENTION_WAIT_S:
            print(f"[{_ts()}] [queue] #{pending_id} timeout — auto-failing "
                  f"(no monitor decision in {MAX_NEEDS_ATTENTION_WAIT_S // 60} min)")
            db.update_pending_status(
                pending_id,
                "failed",
                error_summary=(row.get("error_summary") or "")
                + "\n[runner-auto] needs_attention timed out without monitor decision",
            )
            return "failed"


def run_one(db: ExperimentDB, entry: Dict[str, Any]) -> str:
    """Execute a single pending entry; returns terminal status name."""
    global _current_proc, _current_pending_id
    pending_id = entry["id"]
    _current_pending_id = pending_id
    db.update_pending_status(pending_id, "running")
    print(f"[{_ts()}] [queue] >>> #{pending_id}: {entry['command']}")

    try:
        args = shlex.split(entry["command"])
    except ValueError as e:
        msg = f"shlex parse error: {e}"
        print(f"[{_ts()}] [queue] {msg}")
        db.update_pending_status(pending_id, "needs_attention", error_summary=msg)
        return wait_for_resolution(db, pending_id)

    # Stream subprocess output live; also retain a tail for error_summary.
    stderr_tail: list[str] = []
    try:
        _current_proc = subprocess.Popen(
            args,
            stdout=sys.stdout,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # Drain stderr line-by-line so we can both echo and capture
        assert _current_proc.stderr is not None
        for line in _current_proc.stderr:
            sys.stderr.write(line)
            stderr_tail.append(line.rstrip())
            if len(stderr_tail) > STDERR_TAIL_LIMIT:
                stderr_tail = stderr_tail[-STDERR_TAIL_LIMIT:]
        rc = _current_proc.wait()
    except FileNotFoundError as e:
        msg = f"command not found: {e}"
        print(f"[{_ts()}] [queue] {msg}")
        db.update_pending_status(pending_id, "needs_attention", error_summary=msg)
        return wait_for_resolution(db, pending_id)
    finally:
        _current_proc = None

    if rc == 0:
        run_id = discover_run_id(db, entry)
        db.update_pending_status(
            pending_id, "completed", completed_run_id=run_id
        )
        print(f"[{_ts()}] [queue] <<< #{pending_id} completed "
              f"(run_id={run_id})")
        _current_pending_id = None
        return "completed"

    tail = _capture_tail_from_proc(stderr_tail)
    err = f"exit code {rc}\n{tail[:ERROR_SUMMARY_MAX_BYTES]}"
    db.update_pending_status(pending_id, "needs_attention", error_summary=err)
    print(f"[{_ts()}] [queue] <<< #{pending_id} failed (rc={rc}), "
          f"awaiting monitor decision")
    result = wait_for_resolution(db, pending_id)
    _current_pending_id = None
    return result


# ──────────────────────────────────────────────────────────────────────────
# Main loop & signal handling
# ──────────────────────────────────────────────────────────────────────────

def _on_sigint(signum, frame):
    """On Ctrl-C / SIGINT: kill subprocess, revert state to 'pending', exit.

    Safety note on the DB write inside this signal handler:
    SIGINT in practice almost always arrives while we're blocked inside
    `subprocess.Popen.wait()` (running an experiment) or `time.sleep()`
    (polling for monitor decision / queue empty). At both of those points
    the `_connection()` context manager has already exited, so no DB
    cursor is mid-transaction. WAL mode + SQLite's single-writer lock
    further guarantee that even a worst-case mid-write interruption leaves
    the file consistent (the transaction either fully commits or rolls
    back). The handler therefore opens a fresh implicit connection via
    `update_pending_status`, which is safe.
    """
    print(f"\n[{_ts()}] [queue] SIGINT received; cleaning up ...")
    if _current_proc is not None and _current_proc.poll() is None:
        print(f"[{_ts()}] [queue] killing subprocess pid={_current_proc.pid}")
        try:
            _current_proc.kill()
        except Exception as e:
            print(f"[{_ts()}] [queue] kill failed: {e}")
    if _db is not None and _current_pending_id is not None:
        try:
            row = _db.get_pending(_current_pending_id)
            if row and row["status"] in ("claimed", "running"):
                _db.update_pending_status(_current_pending_id, "pending")
                print(f"[{_ts()}] [queue] reverted #{_current_pending_id} -> pending")
        except Exception as e:
            print(f"[{_ts()}] [queue] revert failed: {e}")
    sys.exit(130)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point. Pass argv=None to read from sys.argv[1:] (CLI default)."""
    global _db

    parser = argparse.ArgumentParser(
        description="Sequential queue runner for ExperimentDB pending_runs"
    )
    parser.add_argument(
        "--drain-and-exit",
        action="store_true",
        help="Exit immediately once the queue is empty (default: poll forever)",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the first-entry GPU sanity check (use if you've already "
             "manually confirmed idle status)",
    )
    args = parser.parse_args(argv)

    signal.signal(signal.SIGINT, _on_sigint)

    _db = ExperimentDB()
    did_preflight = args.skip_preflight
    print(f"[{_ts()}] [queue] runner started (drain={args.drain_and_exit}, "
          f"skip_preflight={args.skip_preflight})")

    while True:
        entry = _db.claim_next_pending()
        if entry is None:
            if args.drain_and_exit:
                print(f"[{_ts()}] [queue] queue empty + drain mode; exiting")
                return 0
            print(f"[{_ts()}] [queue] queue empty; polling in "
                  f"{EMPTY_QUEUE_POLL_S}s ...")
            time.sleep(EMPTY_QUEUE_POLL_S)
            continue

        if not did_preflight:
            preflight_first_entry()
            did_preflight = True

        run_one(_db, entry)
        # Loop immediately — next claim_next_pending will fetch next entry


if __name__ == "__main__":
    sys.exit(main())
