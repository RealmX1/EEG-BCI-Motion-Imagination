"""
Benchmark inference performance: CBraMod vs EEGNet.

Measures:
1. Latency (ms per prediction, single & batched)
2. Throughput (samples/second)
3. FLOPs (floating-point operations)
4. GPU memory (peak allocated)
5. CPU RAM (peak resident)
6. Parameter count & model size

Usage:
    uv run python scripts/benchmark_inference.py
"""

import gc
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

# ── project imports ──────────────────────────────────────────────────
from src.models.eegnet import EEGNet
from src.models.cbramod_adapter import CBraModForFingerBCI, get_default_pretrained_path
from src.config.training import get_default_config


# ── configuration ────────────────────────────────────────────────────

N_CHANNELS = 128
EEGNET_SAMPLES = 400       # 4s @ 100 Hz (paper-aligned)
CBRAMOD_SAMPLES = 1000     # 5s @ 200 Hz
CBRAMOD_PATCHES = 5        # 1000 / 200
WARMUP_ITERS = 50          # GPU warmup iterations
BENCH_ITERS = 200          # timed iterations
BATCH_SIZES = [1, 8, 32, 64]  # test latency at various batch sizes


@dataclass
class BenchmarkResult:
    model_name: str
    task: str
    n_classes: int
    batch_size: int
    # model stats
    param_count_total: int = 0
    param_count_trainable: int = 0
    model_size_mb: float = 0.0
    # latency
    latency_mean_ms: float = 0.0
    latency_std_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    # throughput
    throughput_samples_per_sec: float = 0.0
    # compute
    flops: int = 0
    flops_str: str = ""
    # memory
    gpu_peak_mb: float = 0.0
    gpu_inference_mb: float = 0.0  # incremental memory for inference
    cpu_peak_mb: float = 0.0


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb(model: nn.Module) -> float:
    """Estimate model size in MB (parameters + buffers)."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def measure_flops(model: nn.Module, x: torch.Tensor) -> int:
    """Measure FLOPs for a single forward pass using PyTorch FlopCounterMode."""
    model.eval()
    flop_counter = FlopCounterMode(display=False)
    with flop_counter, torch.no_grad():
        _ = model(x)
    return flop_counter.get_total_flops()


def format_flops(flops: int) -> str:
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPs"
    else:
        return f"{flops / 1e3:.2f} KFLOPs"


def measure_latency(
    model: nn.Module,
    x: torch.Tensor,
    warmup: int = WARMUP_ITERS,
    iters: int = BENCH_ITERS,
) -> list[float]:
    """
    Measure inference latency using CUDA events.
    Returns list of per-iteration times in milliseconds.
    """
    model.eval()
    device = next(model.parameters()).device

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    torch.cuda.synchronize()

    # Timed runs using CUDA events for precise GPU timing
    times = []
    with torch.no_grad():
        for _ in range(iters):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            _ = model(x)
            end_event.record()

            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))  # ms

    return times


def measure_gpu_memory(model: nn.Module, x: torch.Tensor) -> tuple[float, float]:
    """
    Measure GPU memory: peak total and inference-only increment.
    Returns (peak_mb, inference_increment_mb).
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    # Baseline: model loaded on GPU
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()

    # Run inference
    model.eval()
    with torch.no_grad():
        _ = model(x)
    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated()
    inference_increment = peak - baseline

    return peak / (1024 ** 2), inference_increment / (1024 ** 2)


def measure_cpu_memory(model: nn.Module, x_cpu_shape: tuple, device: torch.device) -> float:
    """Measure CPU-side peak RAM during inference (MB)."""
    tracemalloc.start()
    x = torch.randn(*x_cpu_shape, device=device)
    model.eval()
    with torch.no_grad():
        _ = model(x)
    torch.cuda.synchronize()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 ** 2)


def benchmark_model(
    model: nn.Module,
    model_name: str,
    input_shape: tuple,
    task: str,
    n_classes: int,
    device: torch.device,
) -> list[BenchmarkResult]:
    """Run full benchmark suite for a model across all batch sizes."""
    results = []
    total_params, trainable_params = count_parameters(model)
    size_mb = model_size_mb(model)

    for bs in BATCH_SIZES:
        print(f"  batch_size={bs} ...", end=" ", flush=True)
        result = BenchmarkResult(
            model_name=model_name,
            task=task,
            n_classes=n_classes,
            batch_size=bs,
            param_count_total=total_params,
            param_count_trainable=trainable_params,
            model_size_mb=size_mb,
        )

        x = torch.randn(bs, *input_shape, device=device)

        # FLOPs (per sample — use batch_size=1)
        if bs == 1:
            result.flops = measure_flops(model, x)
            result.flops_str = format_flops(result.flops)

        # Latency
        times = measure_latency(model, x)
        t = torch.tensor(times)
        result.latency_mean_ms = t.mean().item()
        result.latency_std_ms = t.std().item()
        result.latency_min_ms = t.min().item()
        result.latency_max_ms = t.max().item()
        result.latency_p50_ms = t.median().item()
        result.latency_p95_ms = t.quantile(0.95).item()
        result.latency_p99_ms = t.quantile(0.99).item()
        result.throughput_samples_per_sec = bs / (result.latency_mean_ms / 1000)

        # GPU memory
        result.gpu_peak_mb, result.gpu_inference_mb = measure_gpu_memory(model, x)

        # CPU memory
        result.cpu_peak_mb = measure_cpu_memory(model, (bs, *input_shape), device)

        # Backfill FLOPs from bs=1
        if bs != 1 and results:
            result.flops = results[0].flops
            result.flops_str = results[0].flops_str

        print(f"latency={result.latency_mean_ms:.2f}ms, "
              f"throughput={result.throughput_samples_per_sec:.0f} samples/s")
        results.append(result)

    return results


def create_eegnet(task: str, device: torch.device) -> tuple[nn.Module, tuple]:
    """Create EEGNet with default config."""
    n_classes = 2 if task == "binary" else 3
    config = get_default_config("eegnet", task)
    model_cfg = config["model"]

    model = EEGNet(
        n_channels=N_CHANNELS,
        n_samples=EEGNET_SAMPLES,
        n_classes=n_classes,
        F1=model_cfg["F1"],
        D=model_cfg["D"],
        F2=model_cfg["F2"],
        kernel_length=model_cfg["kernel_length"],
        dropout_rate=model_cfg["dropout_rate"],
    ).to(device)
    model.eval()

    input_shape = (N_CHANNELS, EEGNET_SAMPLES)  # 3D input
    return model, input_shape


def create_cbramod(task: str, device: torch.device) -> tuple[nn.Module, tuple]:
    """Create CBraMod with default config."""
    n_classes = 2 if task == "binary" else 3
    config = get_default_config("cbramod", task)
    model_cfg = config["model"]

    pretrained_path = get_default_pretrained_path()
    model = CBraModForFingerBCI(
        n_channels=N_CHANNELS,
        n_patches=CBRAMOD_PATCHES,
        n_classes=n_classes,
        pretrained_path=pretrained_path,
        freeze_backbone=False,
        classifier_type=model_cfg["classifier_type"],
        dropout=model_cfg["dropout_rate"],
    ).to(device)
    model.eval()

    input_shape = (N_CHANNELS, CBRAMOD_PATCHES, 200)  # 4D input
    return model, input_shape


def print_comparison_table(all_results: dict[str, list[BenchmarkResult]]):
    """Print a nicely formatted comparison table."""
    print("\n" + "=" * 100)
    print("INFERENCE BENCHMARK COMPARISON: CBraMod vs EEGNet")
    print("=" * 100)

    # Group by task
    tasks = set()
    for results_list in all_results.values():
        for r in results_list:
            tasks.add(r.task)

    for task in sorted(tasks):
        print(f"\n{'─' * 100}")
        print(f"Task: {task.upper()}")
        print(f"{'─' * 100}")

        # Model summary (from bs=1)
        print(f"\n{'Model Summary':^60}")
        print(f"{'':─<60}")
        header = f"{'Metric':<30} {'EEGNet':>14} {'CBraMod':>14}"
        print(header)
        print(f"{'':─<60}")

        eegnet_key = f"eegnet_{task}"
        cbramod_key = f"cbramod_{task}"

        if eegnet_key in all_results and cbramod_key in all_results:
            e1 = all_results[eegnet_key][0]  # bs=1
            c1 = all_results[cbramod_key][0]

            rows = [
                ("Total Parameters", f"{e1.param_count_total:,}", f"{c1.param_count_total:,}"),
                ("Trainable Parameters", f"{e1.param_count_trainable:,}", f"{c1.param_count_trainable:,}"),
                ("Model Size (MB)", f"{e1.model_size_mb:.2f}", f"{c1.model_size_mb:.2f}"),
                ("FLOPs (bs=1)", e1.flops_str, c1.flops_str),
                ("FLOPs ratio", "1.0x", f"{c1.flops / max(e1.flops, 1):.1f}x"),
            ]
            for label, ev, cv in rows:
                print(f"{label:<30} {ev:>14} {cv:>14}")

        # Latency table
        print(f"\n{'Latency & Throughput':^80}")
        print(f"{'':─<80}")
        print(f"{'BS':<5} {'Model':<12} {'Mean(ms)':>10} {'P50(ms)':>10} {'P95(ms)':>10} "
              f"{'P99(ms)':>10} {'Throughput':>14}")
        print(f"{'':─<80}")

        for bs in BATCH_SIZES:
            for key in [eegnet_key, cbramod_key]:
                if key not in all_results:
                    continue
                for r in all_results[key]:
                    if r.batch_size == bs:
                        name = "EEGNet" if "eegnet" in key else "CBraMod"
                        print(
                            f"{bs:<5} {name:<12} {r.latency_mean_ms:>10.3f} "
                            f"{r.latency_p50_ms:>10.3f} {r.latency_p95_ms:>10.3f} "
                            f"{r.latency_p99_ms:>10.3f} {r.throughput_samples_per_sec:>11.0f} s/s"
                        )
            if bs != BATCH_SIZES[-1]:
                print(f"{'':─<80}")

        # Memory table
        print(f"\n{'GPU Memory (MB)':^65}")
        print(f"{'':─<65}")
        print(f"{'BS':<5} {'Model':<12} {'Peak GPU(MB)':>14} {'Inference(MB)':>14} {'CPU RAM(MB)':>14}")
        print(f"{'':─<65}")

        for bs in BATCH_SIZES:
            for key in [eegnet_key, cbramod_key]:
                if key not in all_results:
                    continue
                for r in all_results[key]:
                    if r.batch_size == bs:
                        name = "EEGNet" if "eegnet" in key else "CBraMod"
                        print(
                            f"{bs:<5} {name:<12} {r.gpu_peak_mb:>14.1f} "
                            f"{r.gpu_inference_mb:>14.1f} {r.cpu_peak_mb:>14.2f}"
                        )
            if bs != BATCH_SIZES[-1]:
                print(f"{'':─<65}")

    # Latency ratio summary
    print(f"\n{'=' * 60}")
    print("LATENCY RATIO SUMMARY (CBraMod / EEGNet)")
    print(f"{'=' * 60}")
    print(f"{'Task':<10} {'BS':<5} {'Ratio':>10}")
    print(f"{'':─<30}")
    for task in sorted(tasks):
        eegnet_key = f"eegnet_{task}"
        cbramod_key = f"cbramod_{task}"
        if eegnet_key in all_results and cbramod_key in all_results:
            for e, c in zip(all_results[eegnet_key], all_results[cbramod_key]):
                ratio = c.latency_mean_ms / max(e.latency_mean_ms, 0.001)
                print(f"{task:<10} {e.batch_size:<5} {ratio:>10.1f}x")


def main():
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"\nConfig: {N_CHANNELS}ch, warmup={WARMUP_ITERS}, iters={BENCH_ITERS}")
    print(f"EEGNet input: ({N_CHANNELS}, {EEGNET_SAMPLES}) @ 100Hz")
    print(f"CBraMod input: ({N_CHANNELS}, {CBRAMOD_PATCHES}, 200) @ 200Hz")
    print()

    all_results = {}

    for task in ["binary", "ternary"]:
        n_classes = 2 if task == "binary" else 3

        # ── EEGNet ──
        print(f"[EEGNet] {task} (n_classes={n_classes})")
        model, input_shape = create_eegnet(task, device)
        results = benchmark_model(model, "EEGNet-16,4", input_shape, task, n_classes, device)
        all_results[f"eegnet_{task}"] = results
        del model
        torch.cuda.empty_cache()
        gc.collect()

        # ── CBraMod ──
        print(f"\n[CBraMod] {task} (n_classes={n_classes})")
        model, input_shape = create_cbramod(task, device)
        results = benchmark_model(model, "CBraMod", input_shape, task, n_classes, device)
        all_results[f"cbramod_{task}"] = results
        del model
        torch.cuda.empty_cache()
        gc.collect()

        print()

    # Print comparison
    print_comparison_table(all_results)


if __name__ == "__main__":
    main()
