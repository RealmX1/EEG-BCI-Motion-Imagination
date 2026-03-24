#!/usr/bin/env python
"""
CPU Overwatch: wait for CPU to become stably idle before yielding control.

Exit code 0 = CPU is idle (safe to start work).
Exit code 1 = error / interrupted.

Idle criteria (evaluated over a rolling 30-minute window):
  - Average overall CPU utilization < 15%

The script monitors ALL cores (not just one) and reports the number of
active cores alongside overall utilization.

The script enforces a mandatory 30-minute warm-up period before the first
idle check, so a CPU that happens to be idle at launch won't trigger
immediately (the current job may just be between stages).
"""

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
UTIL_THRESHOLD = 20.0           # average utilization % threshold (higher than GPU
                                # because CPUs always have background OS activity)
ACTIVE_CORE_THRESHOLD = 20.0    # a core is "active" if > 20% utilization
# ───────────────────────────────────────────────────────────────────────────


def query_cpu():
    """Return (overall_util%, active_cores, total_cores, ram_used_GB, ram_total_GB).

    Uses a 1-second measurement interval to get accurate per-core readings.
    Overall utilization is the mean across all cores.
    """
    per_core = psutil.cpu_percent(interval=1, percpu=True)
    overall = sum(per_core) / len(per_core)
    active_cores = sum(1 for c in per_core if c > ACTIVE_CORE_THRESHOLD)
    mem = psutil.virtual_memory()
    return overall, active_cores, len(per_core), mem.used / (1024**3), mem.total / (1024**3)


def fmt_duration(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def main():
    samples = deque()  # (timestamp, overall_util%, ram_used_gb)
    t_start = time.monotonic()

    # Discard first psutil reading (always 0.0 on first call)
    psutil.cpu_percent(percpu=True)

    num_cores = psutil.cpu_count()
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] CPU Overwatch started")
    print(f"       CPUs: {num_cores} cores")
    print(f"       Warm-up: {WARMUP_SEC // 60} min | Window: {WINDOW_SEC // 60} min")
    print(f"       Idle criteria: overall util < {UTIL_THRESHOLD}%")
    print(f"       Polling every {POLL_INTERVAL_SEC}s")
    print()

    try:
        while True:
            # ── Sample ────────────────────────────────────────────────
            now = time.monotonic()
            try:
                overall, active, total, ram_used, ram_total = query_cpu()
            except Exception as e:
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"[{ts}] psutil error: {e}")
                time.sleep(POLL_INTERVAL_SEC)
                continue

            samples.append((now, overall, ram_used))

            # Evict samples older than the rolling window
            cutoff = now - WINDOW_SEC
            while samples and samples[0][0] < cutoff:
                samples.popleft()

            # ── Status ────────────────────────────────────────────────
            elapsed = now - t_start
            warmup_remaining = max(0, WARMUP_SEC - elapsed)

            if len(samples) >= 2:
                avg_util = sum(s[1] for s in samples) / len(samples)
                avg_ram = sum(s[2] for s in samples) / len(samples)
            else:
                avg_util = overall
                avg_ram = ram_used

            ts = datetime.now().strftime("%H:%M:%S")
            if warmup_remaining > 0:
                phase = f"WARM-UP ({fmt_duration(warmup_remaining)} left)"
            else:
                phase = "MONITORING"

            print(
                f"[{ts}] {phase} | "
                f"now: {overall:5.1f}% {active}/{total} cores "
                f"{ram_used:5.1f}/{ram_total:.0f}GB RAM | "
                f"avg({len(samples)}): {avg_util:5.1f}% {avg_ram:5.1f}GB RAM"
            )

            # ── Alarm check (only after warm-up) ─────────────────────
            if warmup_remaining <= 0:
                window_span = samples[-1][0] - samples[0][0] if len(samples) >= 2 else 0

                # Need at least ~25 min of data in the window for a confident check
                if window_span >= WINDOW_SEC * 0.8:
                    if avg_util < UTIL_THRESHOLD:
                        print()
                        print(f"[{ts}] CPU IDLE -- avg util {avg_util:.1f}% "
                              f"(window {fmt_duration(window_span)})")
                        print(f"       Overwatch exiting with code 0.")
                        return 0

            time.sleep(POLL_INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\nOverwatch interrupted.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
