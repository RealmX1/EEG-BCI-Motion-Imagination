#!/usr/bin/env python
"""
Network Overwatch: wait for network to become stably idle before yielding control.

Exit code 0 = Network is idle (safe to start work).
Exit code 1 = error / interrupted.

Idle criteria (evaluated over a rolling 30-minute window):
  - Average throughput (sent + recv) < 1 MB/s

The script enforces a mandatory 30-minute warm-up period before the first
idle check, so a network that happens to be idle at launch won't trigger
immediately (the current job may just be between transfers).
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
THROUGHPUT_THRESHOLD = 1.0      # MB/s (combined sent + recv)
# ───────────────────────────────────────────────────────────────────────────


def list_network_interfaces():
    """Return list of (name, bytes_sent, bytes_recv) for all network interfaces."""
    counters = psutil.net_io_counters(pernic=True)
    result = []
    for name, nio in counters.items():
        result.append((name, nio.bytes_sent, nio.bytes_recv))
    return result


def query_network(interface=None):
    """Return (bytes_sent, bytes_recv) cumulative counters.

    If interface is None, returns system-wide totals.
    If interface is specified, returns counters for that interface only.
    Raises ValueError if the specified interface is not found.
    """
    if interface is None:
        nio = psutil.net_io_counters()
        return nio.bytes_sent, nio.bytes_recv
    else:
        per_nic = psutil.net_io_counters(pernic=True)
        if interface not in per_nic:
            raise ValueError(
                f"Interface {interface!r} not found. "
                f"Available: {list(per_nic.keys())}"
            )
        nio = per_nic[interface]
        return nio.bytes_sent, nio.bytes_recv


def compute_throughput(prev_counters, curr_counters, interval_sec):
    """Compute throughput in MB/s from two consecutive counter readings.

    Returns (total_mbps, sent_mbps, recv_mbps).
    """
    if interval_sec <= 0:
        return 0.0, 0.0, 0.0
    d_sent = curr_counters[0] - prev_counters[0]
    d_recv = curr_counters[1] - prev_counters[1]
    sent_mbps = d_sent / (1024 * 1024) / interval_sec
    recv_mbps = d_recv / (1024 * 1024) / interval_sec
    return sent_mbps + recv_mbps, sent_mbps, recv_mbps


def fmt_duration(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Network Overwatch: wait for network to become stably idle."
    )
    parser.add_argument(
        "--interface",
        type=str,
        default=None,
        help="Monitor specific network interface (default: all combined)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THROUGHPUT_THRESHOLD,
        help=f"Idle threshold in MB/s (default: {THROUGHPUT_THRESHOLD})",
    )
    parser.add_argument(
        "--list-interfaces",
        action="store_true",
        help="List available network interfaces and exit",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_interfaces:
        interfaces = list_network_interfaces()
        print("Available network interfaces:")
        for name, sent, recv in interfaces:
            print(f"  {name:40s} sent: {sent / (1024**2):9.1f} MB  recv: {recv / (1024**2):9.1f} MB")
        return 0

    threshold = args.threshold
    samples = deque()          # (timestamp, throughput_mbps)
    prev_counters = None       # (bytes_sent, bytes_recv) from last poll
    prev_time = None           # monotonic time of last poll
    t_start = time.monotonic()

    iface_label = args.interface or "all"

    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] Network Overwatch started")
    print(f"       Interface: {iface_label}")
    print(f"       Warm-up: {WARMUP_SEC // 60} min | Window: {WINDOW_SEC // 60} min")
    print(f"       Idle criteria: throughput < {threshold} MB/s")
    print(f"       Polling every {POLL_INTERVAL_SEC}s")
    print()

    try:
        while True:
            now = time.monotonic()
            try:
                counters = query_network(args.interface)
            except Exception as e:
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"[{ts}] psutil error: {e}")
                time.sleep(POLL_INTERVAL_SEC)
                continue

            # Compute throughput (skip first iteration -- no delta yet)
            if prev_counters is not None:
                interval = now - prev_time
                total_mbps, sent_mbps, recv_mbps = compute_throughput(
                    prev_counters, counters, interval
                )
                samples.append((now, total_mbps))
            else:
                total_mbps = sent_mbps = recv_mbps = 0.0

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
                f"now: {sent_mbps:6.2f}\u2191 {recv_mbps:6.2f}\u2193 MB/s | "
                f"avg({len(samples)}): {avg_tp:6.2f} MB/s"
            )

            # ── Alarm check (only after warm-up) ─────────────────────
            if warmup_remaining <= 0:
                window_span = samples[-1][0] - samples[0][0] if len(samples) >= 2 else 0

                if window_span >= WINDOW_SEC * 0.8:
                    if avg_tp < threshold:
                        print()
                        print(f"[{ts}] NET IDLE -- avg throughput {avg_tp:.2f} MB/s "
                              f"(window {fmt_duration(window_span)})")
                        print(f"       Overwatch exiting with code 0.")
                        return 0

            time.sleep(POLL_INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\nOverwatch interrupted.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
