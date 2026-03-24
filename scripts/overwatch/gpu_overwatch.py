#!/usr/bin/env python
"""
GPU Overwatch: wait for GPU to become stably idle before yielding control.

Exit code 0 = GPU is idle (safe to start work).
Exit code 1 = error / interrupted.

Idle criteria (evaluated over a rolling 30-minute window):
  - Average GPU utilization < 10%

The script enforces a mandatory 30-minute warm-up period before the first
idle check, so a GPU that happens to be idle at launch won't trigger
immediately (the current job may just be between batches).
"""

import subprocess
import sys
import time
from collections import deque
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────
POLL_INTERVAL_SEC = 30          # sample every 30 s
WINDOW_SEC = 30 * 60           # 30-minute rolling window
WARMUP_SEC = 30 * 60           # mandatory wait before first alarm check
UTIL_THRESHOLD = 10.0          # average utilization % threshold
# ───────────────────────────────────────────────────────────────────────────


def query_gpu():
    """Return (utilization_%, vram_used_MB) from nvidia-smi."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    # Take first GPU if multiple
    line = result.stdout.strip().splitlines()[0]
    util_str, vram_str = line.split(",")
    return float(util_str.strip()), float(vram_str.strip())


def fmt_duration(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def main():
    samples = deque()  # (timestamp, util%, vram_mb)
    t_start = time.monotonic()

    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] GPU Overwatch started")
    print(f"       Warm-up: {WARMUP_SEC // 60} min | Window: {WINDOW_SEC // 60} min")
    print(f"       Idle criteria: util < {UTIL_THRESHOLD}%")
    print(f"       Polling every {POLL_INTERVAL_SEC}s")
    print()

    try:
        while True:
            # ── Sample ────────────────────────────────────────────────
            now = time.monotonic()
            try:
                util, vram = query_gpu()
            except Exception as e:
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"[{ts}] nvidia-smi error: {e}")
                time.sleep(POLL_INTERVAL_SEC)
                continue

            samples.append((now, util, vram))

            # Evict samples older than the rolling window
            cutoff = now - WINDOW_SEC
            while samples and samples[0][0] < cutoff:
                samples.popleft()

            # ── Status ────────────────────────────────────────────────
            elapsed = now - t_start
            warmup_remaining = max(0, WARMUP_SEC - elapsed)

            if len(samples) >= 2:
                avg_util = sum(s[1] for s in samples) / len(samples)
                avg_vram = sum(s[2] for s in samples) / len(samples)
            else:
                avg_util = util
                avg_vram = vram

            ts = datetime.now().strftime("%H:%M:%S")
            if warmup_remaining > 0:
                phase = f"WARM-UP ({fmt_duration(warmup_remaining)} left)"
            else:
                phase = "MONITORING"

            print(
                f"[{ts}] {phase} | "
                f"now: {util:5.1f}% {vram/1024:5.2f}GB | "
                f"avg({len(samples)}): {avg_util:5.1f}% {avg_vram/1024:5.2f}GB"
            )

            # ── Alarm check (only after warm-up) ─────────────────────
            if warmup_remaining <= 0:
                window_span = samples[-1][0] - samples[0][0] if len(samples) >= 2 else 0

                # Need at least ~25 min of data in the window for a confident check
                if window_span >= WINDOW_SEC * 0.8:
                    if avg_util < UTIL_THRESHOLD:
                        print()
                        print(f"[{ts}] GPU IDLE -- avg util {avg_util:.1f}% "
                              f"(window {fmt_duration(window_span)})")
                        print(f"       Overwatch exiting with code 0.")
                        return 0

            time.sleep(POLL_INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\nOverwatch interrupted.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
