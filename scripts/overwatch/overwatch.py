#!/usr/bin/env python
"""
Combined Overwatch: wait for ALL monitored resources to become stably idle.

Exit code 0 = all enabled resources are idle (safe to start work).
Exit code 1 = error / interrupted.

Reuses query functions from the individual overwatch modules.
Each resource has its own rolling window and idle criteria:
  - GPU: avg utilization < 10% over 30 min
  - CPU: avg utilization < 20% over 30 min
  - NET: avg throughput < 1 MB/s over 30 min   (on by default)
  - DISK: avg throughput < 5 MB/s over 30 min   (off by default, --enable-disk)

All enabled resources must be simultaneously idle (with sufficient window data)
to trigger exit.
"""

import argparse
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

# Allow imports from the scripts/ directory
sys.path.insert(0, str(Path(__file__).parent))

from cpu_overwatch import (
    UTIL_THRESHOLD as CPU_THRESHOLD,
    WINDOW_SEC as CPU_WINDOW_SEC,
    query_cpu,
)
from disk_overwatch import (
    THROUGHPUT_THRESHOLD as DISK_THRESHOLD,
    WINDOW_SEC as DISK_WINDOW_SEC,
    compute_disk_throughput,
    query_disk,
)
from gpu_overwatch import (
    UTIL_THRESHOLD as GPU_THRESHOLD,
    WINDOW_SEC as GPU_WINDOW_SEC,
    query_gpu,
)
from network_overwatch import (
    THROUGHPUT_THRESHOLD as NET_THRESHOLD,
    WINDOW_SEC as NET_WINDOW_SEC,
    compute_throughput as compute_net_throughput,
    query_network,
)

# ── Configuration ──────────────────────────────────────────────────────────
POLL_INTERVAL_SEC = 30          # sample every 30 s
WARMUP_SEC = 30 * 60            # mandatory wait before first alarm check
# Use the longest window across all resources
WINDOW_SEC = max(GPU_WINDOW_SEC, CPU_WINDOW_SEC, NET_WINDOW_SEC, DISK_WINDOW_SEC)
# ───────────────────────────────────────────────────────────────────────────


def fmt_duration(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combined Overwatch: wait for all resources to become stably idle."
    )
    parser.add_argument(
        "--enable-disk",
        action="store_true",
        help="Enable disk I/O monitoring (off by default)",
    )
    parser.add_argument(
        "--disable-network",
        action="store_true",
        help="Disable network monitoring (on by default)",
    )
    parser.add_argument(
        "--interface",
        type=str,
        default=None,
        help="Network interface to monitor (default: all combined)",
    )
    parser.add_argument(
        "--disk",
        type=str,
        default=None,
        help="Disk to monitor, e.g. PhysicalDrive1 (default: all combined)",
    )
    parser.add_argument(
        "--net-threshold",
        type=float,
        default=None,
        help=f"Network idle threshold in MB/s (default: {NET_THRESHOLD})",
    )
    parser.add_argument(
        "--disk-threshold",
        type=float,
        default=None,
        help=f"Disk idle threshold in MB/s (default: {DISK_THRESHOLD})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    gpu_samples = deque()  # (timestamp, util%, vram_mb)
    cpu_samples = deque()  # (timestamp, overall_util%, ram_gb)
    net_samples = deque()  # (timestamp, throughput_mbps)
    disk_samples = deque()  # (timestamp, throughput_mbps)

    net_prev_counters = None
    net_prev_time = None
    disk_prev_counters = None
    disk_prev_time = None

    t_start = time.monotonic()

    net_thr = args.net_threshold if args.net_threshold is not None else NET_THRESHOLD
    disk_thr = args.disk_threshold if args.disk_threshold is not None else DISK_THRESHOLD

    # ── Probe available resources ────────────────────────────────────────
    has_gpu = True
    try:
        query_gpu()
    except Exception:
        has_gpu = False

    has_cpu = True
    try:
        import psutil  # noqa: F401
    except ImportError:
        has_cpu = False

    has_net = not args.disable_network
    if has_net:
        try:
            query_network(args.interface)
        except Exception:
            has_net = False

    has_disk = args.enable_disk
    if has_disk:
        try:
            query_disk(args.disk)
        except Exception as e:
            print(f"WARNING: Disk monitoring requested but failed: {e}")
            has_disk = False

    if not has_gpu and not has_cpu and not has_net and not has_disk:
        print("ERROR: no monitoring resources available.")
        return 1

    # ── Banner ───────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%H:%M:%S")
    resources = []
    if has_gpu:
        resources.append(f"GPU (< {GPU_THRESHOLD}%)")
    if has_cpu:
        resources.append(f"CPU (< {CPU_THRESHOLD}%)")
    if has_net:
        iface_label = args.interface or "all"
        resources.append(f"NET (< {net_thr} MB/s, {iface_label})")
    if has_disk:
        disk_label = args.disk or "all"
        resources.append(f"DISK (< {disk_thr} MB/s, {disk_label})")
    print(f"[{ts}] Combined Overwatch started")
    print(f"       Monitoring: {' + '.join(resources)}")
    print(f"       Warm-up: {WARMUP_SEC // 60} min | Window: {WINDOW_SEC // 60} min")
    print(f"       Polling every {POLL_INTERVAL_SEC}s")
    if not has_gpu:
        print(f"       WARNING: nvidia-smi not found -- GPU monitoring disabled")
    if not has_cpu:
        print(f"       WARNING: psutil not found -- CPU monitoring disabled")
    if not has_net:
        if args.disable_network:
            print(f"       NOTE: Network monitoring disabled via --disable-network")
        else:
            print(f"       WARNING: Network monitoring unavailable")
    if not has_disk and not args.enable_disk:
        print(f"       NOTE: Disk monitoring disabled (use --enable-disk to enable)")
    print()

    try:
        while True:
            now = time.monotonic()
            elapsed = now - t_start
            warmup_remaining = max(0, WARMUP_SEC - elapsed)
            ts = datetime.now().strftime("%H:%M:%S")

            # ── GPU sample ───────────────────────────────────────────
            gpu_str = ""
            if has_gpu:
                try:
                    gpu_util, vram = query_gpu()
                    gpu_samples.append((now, gpu_util, vram))
                    gpu_str = f"GPU: {gpu_util:5.1f}% {vram/1024:5.2f}GB"
                except Exception as e:
                    gpu_str = f"GPU: err ({e})"

            # ── CPU sample (1s blocking measurement) ─────────────────
            cpu_str = ""
            if has_cpu:
                try:
                    overall, active, total, ram_used, ram_total = query_cpu()
                    cpu_samples.append((now, overall, ram_used))
                    cpu_str = f"CPU: {overall:5.1f}% {active}/{total} cores"
                except Exception as e:
                    cpu_str = f"CPU: err ({e})"

            # ── NET sample ───────────────────────────────────────────
            net_str = ""
            net_tp = 0.0
            if has_net:
                try:
                    counters = query_network(args.interface)
                    if net_prev_counters is not None:
                        interval = now - net_prev_time
                        net_tp, sent_mbps, recv_mbps = compute_net_throughput(
                            net_prev_counters, counters, interval
                        )
                        net_samples.append((now, net_tp))
                        net_str = f"NET: {sent_mbps:5.2f}\u2191 {recv_mbps:5.2f}\u2193"
                    else:
                        net_str = "NET: (init)"
                    net_prev_counters = counters
                    net_prev_time = now
                except Exception as e:
                    net_str = f"NET: err ({e})"

            # ── DISK sample ──────────────────────────────────────────
            disk_str = ""
            disk_tp = 0.0
            if has_disk:
                try:
                    counters = query_disk(args.disk)
                    if disk_prev_counters is not None:
                        interval = now - disk_prev_time
                        disk_tp, read_mbps, write_mbps = compute_disk_throughput(
                            disk_prev_counters, counters, interval
                        )
                        disk_samples.append((now, disk_tp))
                        disk_str = f"DISK: {read_mbps:5.2f}R {write_mbps:5.2f}W"
                    else:
                        disk_str = "DISK: (init)"
                    disk_prev_counters = counters
                    disk_prev_time = now
                except Exception as e:
                    disk_str = f"DISK: err ({e})"

            # ── Evict old samples ────────────────────────────────────
            cutoff = now - WINDOW_SEC
            while gpu_samples and gpu_samples[0][0] < cutoff:
                gpu_samples.popleft()
            while cpu_samples and cpu_samples[0][0] < cutoff:
                cpu_samples.popleft()
            while net_samples and net_samples[0][0] < cutoff:
                net_samples.popleft()
            while disk_samples and disk_samples[0][0] < cutoff:
                disk_samples.popleft()

            # ── Compute averages ─────────────────────────────────────
            avg_gpu = (sum(s[1] for s in gpu_samples) / len(gpu_samples)
                       if len(gpu_samples) >= 2 else
                       (gpu_samples[-1][1] if gpu_samples else 0))
            avg_cpu = (sum(s[1] for s in cpu_samples) / len(cpu_samples)
                       if len(cpu_samples) >= 2 else
                       (cpu_samples[-1][1] if cpu_samples else 0))
            avg_net = (sum(s[1] for s in net_samples) / len(net_samples)
                       if len(net_samples) >= 2 else
                       (net_samples[-1][1] if net_samples else 0))
            avg_disk = (sum(s[1] for s in disk_samples) / len(disk_samples)
                        if len(disk_samples) >= 2 else
                        (disk_samples[-1][1] if disk_samples else 0))

            # ── Status line ──────────────────────────────────────────
            if warmup_remaining > 0:
                phase = f"WARM-UP ({fmt_duration(warmup_remaining)} left)"
            else:
                phase = "MONITORING"

            parts = [f"[{ts}] {phase}"]
            if gpu_str:
                parts.append(gpu_str)
            if cpu_str:
                parts.append(cpu_str)
            if net_str:
                parts.append(net_str)
            if disk_str:
                parts.append(disk_str)

            n_gpu = len(gpu_samples)
            n_cpu = len(cpu_samples)
            n_net = len(net_samples)
            n_disk = len(disk_samples)

            avg_parts = []
            if has_gpu:
                avg_parts.append(f"GPU {avg_gpu:4.1f}%({n_gpu})")
            if has_cpu:
                avg_parts.append(f"CPU {avg_cpu:4.1f}%({n_cpu})")
            if has_net:
                avg_parts.append(f"NET {avg_net:4.1f}({n_net})")
            if has_disk:
                avg_parts.append(f"DISK {avg_disk:4.1f}({n_disk})")
            parts.append(f"avg: {' '.join(avg_parts)}")
            print(" | ".join(parts))

            # ── Alarm check (only after warm-up) ─────────────────────
            if warmup_remaining <= 0:
                gpu_idle = True
                cpu_idle = True
                net_idle = True
                disk_idle = True

                if has_gpu and len(gpu_samples) >= 2:
                    gpu_span = gpu_samples[-1][0] - gpu_samples[0][0]
                    if gpu_span >= WINDOW_SEC * 0.8:
                        gpu_idle = avg_gpu < GPU_THRESHOLD
                    else:
                        gpu_idle = False  # not enough data yet
                elif has_gpu:
                    gpu_idle = False

                if has_cpu and len(cpu_samples) >= 2:
                    cpu_span = cpu_samples[-1][0] - cpu_samples[0][0]
                    if cpu_span >= WINDOW_SEC * 0.8:
                        cpu_idle = avg_cpu < CPU_THRESHOLD
                    else:
                        cpu_idle = False  # not enough data yet
                elif has_cpu:
                    cpu_idle = False

                if has_net and len(net_samples) >= 2:
                    net_span = net_samples[-1][0] - net_samples[0][0]
                    if net_span >= WINDOW_SEC * 0.8:
                        net_idle = avg_net < net_thr
                    else:
                        net_idle = False
                elif has_net:
                    net_idle = False

                if has_disk and len(disk_samples) >= 2:
                    disk_span = disk_samples[-1][0] - disk_samples[0][0]
                    if disk_span >= WINDOW_SEC * 0.8:
                        disk_idle = avg_disk < disk_thr
                    else:
                        disk_idle = False
                elif has_disk:
                    disk_idle = False

                if gpu_idle and cpu_idle and net_idle and disk_idle:
                    print()
                    idle_parts = []
                    if has_gpu:
                        idle_parts.append(f"GPU avg {avg_gpu:.1f}%")
                    if has_cpu:
                        idle_parts.append(f"CPU avg {avg_cpu:.1f}%")
                    if has_net:
                        idle_parts.append(f"NET avg {avg_net:.2f} MB/s")
                    if has_disk:
                        idle_parts.append(f"DISK avg {avg_disk:.2f} MB/s")
                    print(f"[{ts}] ALL IDLE -- {', '.join(idle_parts)}")
                    print(f"       Overwatch exiting with code 0.")
                    return 0

            time.sleep(POLL_INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\nOverwatch interrupted.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
