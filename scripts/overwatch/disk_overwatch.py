#!/usr/bin/env python
"""
Disk Overwatch: wait for disk I/O to become stably idle before yielding control.

Exit code 0 = Disk is idle (safe to start work).
Exit code 1 = error / interrupted.

Idle criteria (evaluated over a rolling 30-minute window):
  - Average disk throughput (read + write) < 5 MB/s

The script enforces a mandatory 30-minute warm-up period before the first
idle check, so a disk that happens to be idle at launch won't trigger
immediately (the current job may just be between I/O bursts).
"""

import argparse
import sys
import time
from collections import deque
from datetime import datetime

try:
    import psutil
except ImportError:
    print("ERROR: psutil is required. Install with: uv pip install psutil")
    sys.exit(1)

# ── Configuration ──────────────────────────────────────────────────────────
POLL_INTERVAL_SEC = 30          # sample every 30 s
WINDOW_SEC = 30 * 60            # 30-minute rolling window
WARMUP_SEC = 30 * 60            # mandatory wait before first alarm check
THROUGHPUT_THRESHOLD = 5.0      # MB/s (combined read + write)
# ───────────────────────────────────────────────────────────────────────────


def list_disks():
    """Return list of (name, read_bytes, write_bytes) for all physical disks."""
    counters = psutil.disk_io_counters(perdisk=True)
    result = []
    for name, dio in counters.items():
        result.append((name, dio.read_bytes, dio.write_bytes))
    return result


def query_disk(disk=None):
    """Return (bytes_read, bytes_written) cumulative counters.

    If disk is None, returns system-wide totals.
    If disk is specified, returns counters for that disk only.
    Raises ValueError if the specified disk is not found.
    """
    if disk is None:
        dio = psutil.disk_io_counters(perdisk=False)
        return dio.read_bytes, dio.write_bytes
    else:
        per_disk = psutil.disk_io_counters(perdisk=True)
        if disk not in per_disk:
            raise ValueError(
                f"Disk {disk!r} not found. "
                f"Available: {list(per_disk.keys())}"
            )
        dio = per_disk[disk]
        return dio.read_bytes, dio.write_bytes


def compute_disk_throughput(prev_counters, curr_counters, interval_sec):
    """Compute disk throughput in MB/s from two consecutive counter readings.

    Returns (total_mbps, read_mbps, write_mbps).
    """
    if interval_sec <= 0:
        return 0.0, 0.0, 0.0
    d_read = curr_counters[0] - prev_counters[0]
    d_write = curr_counters[1] - prev_counters[1]
    read_mbps = d_read / (1024 * 1024) / interval_sec
    write_mbps = d_write / (1024 * 1024) / interval_sec
    return read_mbps + write_mbps, read_mbps, write_mbps


def fmt_duration(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Disk Overwatch: wait for disk I/O to become stably idle."
    )
    parser.add_argument(
        "--disk",
        type=str,
        default=None,
        help="Monitor specific disk, e.g. PhysicalDrive1 (default: all combined)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THROUGHPUT_THRESHOLD,
        help=f"Idle threshold in MB/s (default: {THROUGHPUT_THRESHOLD})",
    )
    parser.add_argument(
        "--list-disks",
        action="store_true",
        help="List available disks and exit",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_disks:
        disks = list_disks()
        print("Available disks:")
        for name, read_b, write_b in disks:
            print(
                f"  {name:20s} read: {read_b / (1024**3):9.1f} GB  "
                f"write: {write_b / (1024**3):9.1f} GB"
            )
        return 0

    threshold = args.threshold
    samples = deque()          # (timestamp, throughput_mbps)
    prev_counters = None       # (bytes_read, bytes_written) from last poll
    prev_time = None           # monotonic time of last poll
    t_start = time.monotonic()

    disk_label = args.disk or "all"

    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] Disk Overwatch started")
    print(f"       Disk: {disk_label}")
    print(f"       Warm-up: {WARMUP_SEC // 60} min | Window: {WINDOW_SEC // 60} min")
    print(f"       Idle criteria: throughput < {threshold} MB/s")
    print(f"       Polling every {POLL_INTERVAL_SEC}s")
    print()

    try:
        while True:
            now = time.monotonic()
            try:
                counters = query_disk(args.disk)
            except Exception as e:
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"[{ts}] psutil error: {e}")
                time.sleep(POLL_INTERVAL_SEC)
                continue

            # Compute throughput (skip first iteration -- no delta yet)
            if prev_counters is not None:
                interval = now - prev_time
                total_mbps, read_mbps, write_mbps = compute_disk_throughput(
                    prev_counters, counters, interval
                )
                samples.append((now, total_mbps))
            else:
                total_mbps = read_mbps = write_mbps = 0.0

            prev_counters = counters
            prev_time = now

            # Evict samples older than the rolling window
            cutoff = now - WINDOW_SEC
            while samples and samples[0][0] < cutoff:
                samples.popleft()

            # ── Status ────────────────────────────────────────────────
            elapsed = now - t_start
            warmup_remaining = max(0, WARMUP_SEC - elapsed)

            if len(samples) >= 2:
                avg_tp = sum(s[1] for s in samples) / len(samples)
            else:
                avg_tp = total_mbps

            ts = datetime.now().strftime("%H:%M:%S")
            if warmup_remaining > 0:
                phase = f"WARM-UP ({fmt_duration(warmup_remaining)} left)"
            else:
                phase = "MONITORING"

            print(
                f"[{ts}] {phase} | "
                f"now: {read_mbps:7.2f}R {write_mbps:7.2f}W MB/s | "
                f"avg({len(samples)}): {avg_tp:7.2f} MB/s"
            )

            # ── Alarm check (only after warm-up) ─────────────────────
            if warmup_remaining <= 0:
                window_span = samples[-1][0] - samples[0][0] if len(samples) >= 2 else 0

                if window_span >= WINDOW_SEC * 0.8:
                    if avg_tp < threshold:
                        print()
                        print(f"[{ts}] DISK IDLE -- avg throughput {avg_tp:.2f} MB/s "
                              f"(window {fmt_duration(window_span)})")
                        print(f"       Overwatch exiting with code 0.")
                        return 0

            time.sleep(POLL_INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\nOverwatch interrupted.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
