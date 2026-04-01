#!/usr/bin/env python
"""
论文 v3 专属图表生成脚本.

生成现有 --replot 流程无法覆盖的论文图表：
  - channel_scaling: 通道数 vs 准确率曲线 (128→61→32→8→4)
  - further_pretraining: Further pre-training V1/V2 vs baseline 对比
  - inference_latency: 推理延迟对比柱状图
  - extra_sessions_paradigm: Extra sessions 三范式（within/cross/transfer）总览
  - extra_sessions_strategy: Extra sessions 三种评估策略折线对比

Usage:
    uv run python scripts/paper/generate_paper_figures.py --figure channel_scaling
    uv run python scripts/paper/generate_paper_figures.py --figure further_pretraining
    uv run python scripts/paper/generate_paper_figures.py --figure inference_latency
    uv run python scripts/paper/generate_paper_figures.py --figure extra_sessions_paradigm
    uv run python scripts/paper/generate_paper_figures.py --figure extra_sessions_strategy
    uv run python scripts/paper/generate_paper_figures.py --figure all
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.results.dataclasses import PlotDataSource, TrainingResult
from src.results.serialization import cross_subject_result_to_training_results
from src.visualization.comparison import generate_combined_plot

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path('paper/figures')

# 128ch cross-subject baseline cache paths (used across multiple figure generators)
BASELINE_128CH = {
    'cbramod_cross': 'results/20260324_0023_cross_subject_cache_imagery_binary.json',
    'eegnet_cross': 'results/20260330_0709_cross_subject_cache_imagery_binary.json',
    'cbramod_within': 'results/20260323_2237_comparison_cache_imagery_binary.json',
    'eegnet_within': 'results/20260316_1411_comparison_cache_imagery_binary.json',
}


# =============================================================================
# Data Loading Helpers
# =============================================================================

def load_json_cache(path: str) -> dict:
    """Load a JSON result cache file."""
    with open(path) as f:
        return json.load(f)


def extract_model_accs(cache: dict, model: str) -> List[float]:
    """Extract per-subject test accuracies for a model from JSON cache.

    Supports two cache formats:
      - Cross-subject: data['results'][model]['per_subject_test_acc'][subject] (0-1 scale)
      - Within-subject: data[model][subject]['test_acc_majority'] (0-1 scale)
    """
    # Format 1: cross-subject cache (results -> model -> per_subject_test_acc)
    results = cache.get('results', {})
    model_results = results.get(model, {})
    per_subj = model_results.get('per_subject_test_acc', {})
    if per_subj:
        return [acc * 100 for acc in sorted(per_subj.values())]

    # Format 2: within-subject / extra-sessions cache (model -> subject -> test_acc_majority)
    model_data = cache.get(model, {})
    accs = []
    for subj_id, subj_data in sorted(model_data.items()):
        if subj_id in ('metadata', 'comparison'):
            continue
        if isinstance(subj_data, dict):
            acc = subj_data.get('test_acc_majority', subj_data.get('test_acc'))
            if acc is not None:
                accs.append(acc * 100)
    return accs


def extract_extra_session_step_accs(cache: dict, model: str) -> Dict[str, List[float]]:
    """从 extra_sessions cache 中提取某模型各 step 的逐被试准确率（百分比）."""
    step_accs = {step: [] for step in ['baseline', 'sess03', 'sess04', 'sess05']}
    model_data = cache.get('results', {}).get(model, {})
    for _, subj_data in sorted(model_data.items()):
        if not isinstance(subj_data, dict):
            continue
        for step in step_accs:
            step_data = subj_data.get(step, {})
            acc = step_data.get('test_acc_majority', step_data.get('test_acc'))
            if acc is not None:
                step_accs[step].append(acc * 100)
    return step_accs


def extract_cross_subject_extra_session_step_accs(cache: dict, model: str) -> Dict[str, List[float]]:
    """从 cross_subject_extra_sessions cache 中提取某模型各 step 的逐被试准确率（百分比）."""
    step_accs = {}
    model_data = cache.get('results', {}).get(model, {})
    for step in ['baseline', 'sess03', 'sess04', 'sess05']:
        per_subject = model_data.get(step, {}).get('per_subject_test_acc', {})
        step_accs[step] = [acc * 100 for _, acc in sorted(per_subject.items())]
    return step_accs


def _build_cross_subject_source(
    cache_path: str,
    model: str,
    task: str,
    label: str,
    is_current_run: bool,
    hatch: Optional[str] = None,
) -> Optional[PlotDataSource]:
    """从 cross-subject JSON cache 构建 PlotDataSource."""
    path = Path(cache_path)
    if not path.exists():
        logger.warning(f'Missing cache: {cache_path}')
        return None
    cache = load_json_cache(cache_path)
    model_data = cache.get('results', {}).get(model, {})
    if not model_data.get('per_subject_test_acc'):
        logger.warning(f'No {model} per_subject_test_acc in {cache_path}')
        return None
    results = cross_subject_result_to_training_results(model_data, model, task)
    if not results:
        return None
    return PlotDataSource(
        model_type=model,
        results=results,
        is_current_run=is_current_run,
        label=label,
        hatch=hatch,
    )


def _build_within_subject_source(
    cache_path: str,
    model: str,
    task: str,
    label: str,
    is_current_run: bool,
    hatch: Optional[str] = None,
) -> Optional[PlotDataSource]:
    """从 within-subject comparison JSON cache 构建 PlotDataSource."""
    path = Path(cache_path)
    if not path.exists():
        logger.warning(f'Missing cache: {cache_path}')
        return None
    cache = load_json_cache(cache_path)
    model_data = cache.get('results', {}).get(model, {})
    if not model_data:
        logger.warning(f'No {model} data in {cache_path}')
        return None
    results = []
    for subj_id, subj_data in sorted(model_data.items()):
        if not isinstance(subj_data, dict) or 'test_acc_majority' not in subj_data:
            continue
        results.append(TrainingResult(
            subject_id=subj_data.get('subject_id', subj_id),
            task_type=subj_data.get('task_type', task),
            model_type=subj_data.get('model_type', model),
            best_val_acc=subj_data.get('best_val_acc', 0),
            test_acc=subj_data.get('test_acc', subj_data['test_acc_majority']),
            test_acc_majority=subj_data['test_acc_majority'],
            epochs_trained=subj_data.get('epochs_trained', 0),
            training_time=subj_data.get('training_time', 0),
        ))
    if not results:
        return None
    return PlotDataSource(
        model_type=model,
        results=results,
        is_current_run=is_current_run,
        label=label,
        hatch=hatch,
    )


# =============================================================================
# Figure 1: Channel Scaling Curve
# =============================================================================

def generate_channel_scaling_figure():
    """
    通道数 vs CBraMod 准确率曲线 (cross-subject binary).

    Red solid line = best config at each channel count (envelope).
    Dotted lines = each individual channel selection method tracked across counts.

    数据来源 (post-HPO runs, 20260330-20260331):
      128ch: results/20260324_0023_cross_subject_cache_imagery_binary.json
      61ch:  results/61_channel/standard_1010/20260330_1213_*.json
      32ch:  results/32_channel/{fdr,band_power,commercial,attention,csp}/20260330_*
      8ch:   results/8_channel/{fdr,band_power,attention,csp}/2026033{0,1}_*
      4ch:   results/4_channel/{fdr,attention,fdr_attention_overlap,negative_control}/20260330_*
    """
    import matplotlib.pyplot as plt

    # ── Per-method data: method -> [(n_ch, path)] ──
    method_paths = {
        'FDR': [
            (32, 'results/32_channel/fdr/20260330_0836_cross_subject_cache_imagery_binary.json'),
            (8,  'results/8_channel/fdr/20260330_1311_cross_subject_cache_imagery_binary.json'),
            (4,  'results/4_channel/fdr/20260330_2214_cross_subject_cache_imagery_binary.json'),
        ],
        'Band Power': [
            (32, 'results/32_channel/band_power/20260330_1105_cross_subject_cache_imagery_binary.json'),
            (8,  'results/8_channel/band_power/20260331_1950_cross_subject_cache_imagery_binary.json'),
        ],
        'Attention': [
            (32, 'results/32_channel/attention/20260330_1009_cross_subject_cache_imagery_binary.json'),
            (8,  'results/8_channel/attention/20260330_1334_cross_subject_cache_imagery_binary.json'),
            (4,  'results/4_channel/attention/20260330_2200_cross_subject_cache_imagery_binary.json'),
        ],
        'CSP': [
            (32, 'results/32_channel/csp/20260330_1032_cross_subject_cache_imagery_binary.json'),
            (8,  'results/8_channel/csp/20260331_2044_cross_subject_cache_imagery_binary.json'),
        ],
        'Commercial': [
            (32, 'results/32_channel/commercial/20260330_1142_cross_subject_cache_imagery_binary.json'),
        ],
    }

    # Special 4ch configs (not tracked as multi-count methods)
    special_points = {
        'FDR∩Att': (4, 'results/4_channel/fdr_attention_overlap/20260330_1417_cross_subject_cache_imagery_binary.json'),
        'Neg. Control': (4, 'results/4_channel/negative_control/20260330_1442_cross_subject_cache_imagery_binary.json'),
    }

    # Baseline points (no method selection)
    baseline_paths = [
        (128, 'results/20260324_0023_cross_subject_cache_imagery_binary.json'),
        (61,  'results/61_channel/standard_1010/20260330_1213_cross_subject_cache_imagery_binary.json'),
    ]

    # ── Load all data ──
    def _load_acc(path):
        if not Path(path).exists():
            logger.warning(f'Missing: {path}')
            return None, None
        cache = load_json_cache(path)
        accs = extract_model_accs(cache, 'cbramod')
        if not accs:
            return None, None
        return np.mean(accs), np.std(accs)

    # method -> {n_ch: (mean, std)}
    method_data = {}
    for method, entries in method_paths.items():
        method_data[method] = {}
        for n_ch, path in entries:
            mean, std = _load_acc(path)
            if mean is not None:
                method_data[method][n_ch] = (mean, std)

    # Baseline (128, 61)
    baseline_data = {}
    for n_ch, path in baseline_paths:
        mean, std = _load_acc(path)
        if mean is not None:
            baseline_data[n_ch] = (mean, std)

    # Special points
    special_data = {}
    for label, (n_ch, path) in special_points.items():
        mean, std = _load_acc(path)
        if mean is not None:
            special_data[label] = (n_ch, mean, std)

    # ── Compute best envelope ──
    all_channel_counts = sorted(
        set(list(baseline_data.keys())
            + [ch for md in method_data.values() for ch in md]),
        reverse=True,
    )
    best_channels, best_means, best_stds, best_labels = [], [], [], []
    for n_ch in all_channel_counts:
        if n_ch in baseline_data:
            best_channels.append(n_ch)
            best_means.append(baseline_data[n_ch][0])
            best_stds.append(baseline_data[n_ch][1])
            label = 'Full' if n_ch == 128 else 'Std 10-10'
            best_labels.append(label)
        else:
            best_m, best_v, best_s = None, -1, 0
            for method, data in method_data.items():
                if n_ch in data and data[n_ch][0] > best_v:
                    best_m = method
                    best_v = data[n_ch][0]
                    best_s = data[n_ch][1]
            if best_m is not None:
                best_channels.append(n_ch)
                best_means.append(best_v)
                best_stds.append(best_s)
                best_labels.append(best_m)

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(12, 7))

    method_style = {
        'FDR':        {'color': '#1976D2', 'marker': 's'},
        'Band Power': {'color': '#FF9800', 'marker': '^'},
        'Attention':  {'color': '#4CAF50', 'marker': 'v'},
        'CSP':        {'color': '#9C27B0', 'marker': 'D'},
        'Commercial': {'color': '#795548', 'marker': 'p'},
    }

    # 1) Dotted lines for each method
    for method, data in method_data.items():
        if not data:
            continue
        style = method_style[method]
        chs = sorted(data.keys(), reverse=True)
        ms = [data[c][0] for c in chs]
        ss = [data[c][1] for c in chs]
        ax.errorbar(chs, ms, yerr=ss,
                    linestyle=':', linewidth=1.5, marker=style['marker'],
                    markersize=7, color=style['color'], capsize=3,
                    alpha=0.7, label=method, zorder=2)

    # 2) Red best-envelope line (on top)
    ax.errorbar(best_channels, best_means, yerr=best_stds,
                marker='o', markersize=10, linewidth=2.8,
                color='red', capsize=5, zorder=4,
                label='Best Config')

    # Annotate best-envelope points
    for i, (x, y, lbl) in enumerate(zip(best_channels, best_means, best_labels)):
        offsets = {128: (12, 12), 61: (12, -18), 32: (12, 12), 8: (12, 10), 4: (12, -18)}
        ofs = offsets.get(x, (12, 10))
        ax.annotate(f'{lbl} ({x}ch)\n{y:.1f}%', (x, y),
                    textcoords='offset points', xytext=ofs,
                    fontsize=9, fontweight='bold', color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=0.8),
                    zorder=5)

    # 3) Special single-point markers
    if 'FDR∩Att' in special_data:
        _, m, s = special_data['FDR∩Att']
        ax.errorbar([4.4], [m], yerr=[s], marker='D', markersize=9,
                    linewidth=0, color='darkorange', capsize=4,
                    markeredgewidth=1.5, markeredgecolor='darkorange',
                    label=f'FDR$\\cap$Att (4ch): {m:.1f}%', zorder=3)
        ax.annotate('Favorable\nOutlier', (4.4, m),
                    textcoords='offset points', xytext=(15, -5),
                    fontsize=8, color='darkorange', fontstyle='italic',
                    arrowprops=dict(arrowstyle='->', color='darkorange', lw=0.8))

    if 'Neg. Control' in special_data:
        _, m, s = special_data['Neg. Control']
        ax.errorbar([3.7], [m], yerr=[s], marker='x', markersize=11,
                    linewidth=0, color='gray', capsize=4, markeredgewidth=2,
                    label=f'Neg. Control (4ch): {m:.1f}%', zorder=3)

    # 4) Reference lines
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Chance (50%)')
    ax.axvspan(28, 36, alpha=0.08, color='green', label='32ch Deploy Zone')

    ax.set_xlabel('Number of Channels', fontsize=13)
    ax.set_ylabel('CBraMod Cross-Subject Binary Accuracy (%)', fontsize=13)
    ax.set_title('Channel Scaling: Best Envelope & Per-Method Tracking', fontsize=14)
    ax.set_xscale('log', base=2)
    ax.set_xticks(best_channels)
    ax.set_xticklabels([str(c) for c in best_channels])
    ax.set_ylim(48, 100)
    ax.legend(loc='lower right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    if len(best_means) >= 3:
        idx_128 = best_channels.index(128) if 128 in best_channels else 0
        idx_32 = best_channels.index(32) if 32 in best_channels else 2
        retention = best_means[idx_32] / best_means[idx_128] * 100
        ax.text(0.02, 0.02, f'32ch Retention: {retention:.1f}% of 128ch',
                transform=ax.transAxes, fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / 'channel_scaling_curve.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Saved: {out_path}')


# =============================================================================
# Figure 2: Further Pre-training Comparison
# =============================================================================

def generate_further_pretraining_figure():
    """
    Further pre-training V1/V2 vs baseline 下游性能对比.

    数据来源 (from paper/analysis/further_pretraining_analysis.md):
      Baseline (TUEG):
        within binary:  run_tag=20260321_0343 → 85.09%
        cross binary:   run_tag=20260321_0608 → 90.54%
        within ternary: run_tag=20260205_0306 → 69.54%
        cross ternary:  run_tag=20260207_2056 → 75.42%
      FT-V1 (cosine, 10ep):
        within binary:  results/20260322_1034_cbramod_imagery_binary.json → 83.84%
        cross binary:   results/20260322_1116_cross-subject_cbramod_imagery_binary.json → 88.84%
        within ternary: results/20260322_1435_cbramod_imagery_ternary.json → 69.25%
        cross ternary:  results/20260322_1543_cross-subject_cbramod_imagery_ternary.json → 75.67%
      FT-V2 (constant LR, 12ep):
        within binary:  results/20260323_1433_cbramod_imagery_binary.json → 82.23%
        cross binary:   results/20260323_1517_cross-subject_cbramod_imagery_binary.json → 89.43%
        within ternary: results/20260323_1615_cbramod_imagery_ternary.json → 68.08%
        cross ternary:  results/20260323_1709_cross-subject_cbramod_imagery_ternary.json → 75.32%
    """
    import matplotlib.pyplot as plt

    # Hardcoded from verified analysis — see data sources in docstring
    conditions = ['Within-Subj\nBinary', 'Cross-Subj\nBinary', 'Within-Subj\nTernary', 'Cross-Subj\nTernary']
    baseline = [85.09, 90.54, 69.54, 75.42]
    ft_v1 = [83.84, 88.84, 69.25, 75.67]
    ft_v2 = [82.23, 89.43, 68.08, 75.32]

    x = np.arange(len(conditions))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 1.2]})

    # Left panel: Grouped bar chart
    bars_bl = ax1.bar(x - width, baseline, width, label='Baseline (TUEG)', color='#2196F3', edgecolor='black')
    bars_v1 = ax1.bar(x, ft_v1, width, label='FT-V1 (10ep, cosine)', color='#FF9800', edgecolor='black')
    bars_v2 = ax1.bar(x + width, ft_v2, width, label='FT-V2 (12ep, constant)', color='#F44336', edgecolor='black')

    # Annotate deltas on V2 bars
    for i in range(len(conditions)):
        delta = ft_v2[i] - baseline[i]
        color = 'red' if delta < 0 else 'green'
        ax1.annotate(f'{delta:+.2f}pp',
                     xy=(x[i] + width, ft_v2[i]),
                     xytext=(0, 8), textcoords='offset points',
                     fontsize=9, fontweight='bold', color=color, ha='center')

    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Domain-Adaptive Further Pre-training: Downstream Evaluation', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, fontsize=10)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_ylim(60, 95)
    ax1.grid(axis='y', alpha=0.3)

    # Right panel: Average delta comparison
    avg_v1 = np.mean([v1 - bl for v1, bl in zip(ft_v1, baseline)])
    avg_v2 = np.mean([v2 - bl for v2, bl in zip(ft_v2, baseline)])

    bars = ax2.bar(['V1', 'V2'], [avg_v1, avg_v2],
                   color=['#FF9800', '#F44336'], edgecolor='black', width=0.5)
    ax2.axhline(y=0, color='black', linewidth=0.8)

    for bar, val in zip(bars, [avg_v1, avg_v2]):
        ax2.text(bar.get_x() + bar.get_width() / 2, val - 0.15,
                 f'{val:.2f}pp', ha='center', va='top', fontsize=11, fontweight='bold')

    ax2.set_ylabel('Mean Delta vs Baseline (pp)', fontsize=12)
    ax2.set_title('More Training = More Negative Transfer', fontsize=12)
    ax2.set_ylim(-2.0, 0.5)
    ax2.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / 'further_pretraining.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Saved: {out_path}')


# =============================================================================
# Figure 3: Inference Latency
# =============================================================================

def generate_inference_latency_figure():
    """
    推理延迟对比图.

    数据来源: docs/dev_log/experiments/inference_benchmark_analysis.md
    硬件: NVIDIA RTX 5070 (12 GB), PyTorch 2.11.0, CUDA 13.0
    """
    import matplotlib.pyplot as plt
    from src.config.constants import MODEL_COLORS

    # Data from benchmark (128ch binary, ms)
    batch_sizes = [1, 8, 32, 64]
    eegnet_latency = [0.375, 0.542, 2.058, 4.027]
    cbramod_latency = [12.919, 12.582, 32.729, 71.110]

    x = np.arange(len(batch_sizes))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: Latency comparison (log scale)
    bars1 = ax1.bar(x - width/2, eegnet_latency, width,
                    label='EEGNet-16,4', color=MODEL_COLORS['eegnet'], edgecolor='black')
    bars2 = ax1.bar(x + width/2, cbramod_latency, width,
                    label='CBraMod', color=MODEL_COLORS['cbramod'], edgecolor='black')

    ax1.set_yscale('log')
    ax1.set_ylabel('Latency (ms, log scale)', fontsize=12)
    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_title('Inference Latency Comparison (128ch Binary)', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(batch_sizes)
    ax1.legend(fontsize=10)

    # 100ms real-time threshold line
    ax1.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='100ms 实时阈值')
    ax1.text(0.02, 100, '100ms', fontsize=9, color='red', va='bottom')

    ax1.grid(axis='y', alpha=0.3)

    # Annotate batch=1 latencies
    ax1.annotate(f'{eegnet_latency[0]:.1f}ms', xy=(x[0] - width/2, eegnet_latency[0]),
                 xytext=(0, 8), textcoords='offset points', fontsize=8, ha='center')
    ax1.annotate(f'{cbramod_latency[0]:.1f}ms', xy=(x[0] + width/2, cbramod_latency[0]),
                 xytext=(0, 8), textcoords='offset points', fontsize=8, ha='center')

    # Right: Model stats comparison
    model_stats = {
        'Parameters': [16162, 30484402],
        'FLOPs (M)': [112.73, 5080],
        'Model Size (MB)': [0.06, 116.29],
        'BS=1 Latency (ms)': [0.375, 12.919],
    }

    stats_labels = list(model_stats.keys())
    eegnet_vals = [v[0] for v in model_stats.values()]
    cbramod_vals = [v[1] for v in model_stats.values()]
    ratios = [c / e for e, c in zip(eegnet_vals, cbramod_vals)]

    y_pos = np.arange(len(stats_labels))
    ax2.barh(y_pos, ratios, color=MODEL_COLORS['cbramod'], edgecolor='black', alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(stats_labels, fontsize=10)
    ax2.set_xlabel('CBraMod / EEGNet Ratio', fontsize=12)
    ax2.set_title('Model Scale Comparison', fontsize=13)
    ax2.set_xscale('log')
    ax2.axvline(x=1, color='gray', linestyle='--', alpha=0.5)

    for i, (ratio, ev, cv) in enumerate(zip(ratios, eegnet_vals, cbramod_vals)):
        ax2.text(ratio * 1.1, i, f'{ratio:.0f}×', va='center', fontsize=10, fontweight='bold')

    ax2.grid(axis='x', alpha=0.3)

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / 'inference_latency.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Saved: {out_path}')


# =============================================================================
# Figure 4: 32-Channel Configuration Comparison
# =============================================================================

def generate_32ch_comparison_figure():
    """
    32 通道五种配置 × 双模型分组柱状图 (cross-subject binary).

    数据来源 (20260330 post-HPO runs):
      FDR:        results/32_channel/fdr/20260330_0836_*.json
      Attention:  results/32_channel/attention/20260330_1009_*.json
      CSP:        results/32_channel/csp/20260330_1032_*.json
      Band Power: results/32_channel/band_power/20260330_1105_*.json
      Commercial: results/32_channel/commercial/20260330_1142_*.json
      128ch ref:  results/20260324_0023_cross_subject_cache_imagery_binary.json (CBraMod)
                  results/20260330_0709_cross_subject_cache_imagery_binary.json (EEGNet)
    """
    import matplotlib.pyplot as plt
    from src.config.constants import MODEL_COLORS

    configs = [
        ('FDR', 'results/32_channel/fdr/20260330_0836_cross_subject_cache_imagery_binary.json'),
        ('Band Power', 'results/32_channel/band_power/20260330_1105_cross_subject_cache_imagery_binary.json'),
        ('Commercial', 'results/32_channel/commercial/20260330_1142_cross_subject_cache_imagery_binary.json'),
        ('Attention', 'results/32_channel/attention/20260330_1009_cross_subject_cache_imagery_binary.json'),
        ('CSP', 'results/32_channel/csp/20260330_1032_cross_subject_cache_imagery_binary.json'),
    ]

    # 128ch reference
    ref_128_cbramod = 'results/20260324_0023_cross_subject_cache_imagery_binary.json'
    ref_128_eegnet = 'results/20260330_0709_cross_subject_cache_imagery_binary.json'

    config_names = []
    cbramod_means, cbramod_stds = [], []
    eegnet_means, eegnet_stds = [], []

    for name, path in configs:
        if not Path(path).exists():
            logger.warning(f'Missing: {path}')
            continue
        cache = load_json_cache(path)

        cb_accs = extract_model_accs(cache, 'cbramod')
        eg_accs = extract_model_accs(cache, 'eegnet')

        config_names.append(name)
        cbramod_means.append(np.mean(cb_accs) if cb_accs else 0)
        cbramod_stds.append(np.std(cb_accs) if cb_accs else 0)
        eegnet_means.append(np.mean(eg_accs) if eg_accs else 0)
        eegnet_stds.append(np.std(eg_accs) if eg_accs else 0)

    # Load 128ch references
    ref_cb_mean = None
    if Path(ref_128_cbramod).exists():
        cache = load_json_cache(ref_128_cbramod)
        accs = extract_model_accs(cache, 'cbramod')
        if accs:
            ref_cb_mean = np.mean(accs)

    ref_eg_mean = None
    if Path(ref_128_eegnet).exists():
        cache = load_json_cache(ref_128_eegnet)
        accs = extract_model_accs(cache, 'eegnet')
        if accs:
            ref_eg_mean = np.mean(accs)

    # Plot
    x = np.arange(len(config_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))

    bars_cb = ax.bar(x - width/2, cbramod_means, width, yerr=cbramod_stds,
                     label='CBraMod', color=MODEL_COLORS['cbramod'],
                     edgecolor='black', capsize=3, zorder=3)
    bars_eg = ax.bar(x + width/2, eegnet_means, width, yerr=eegnet_stds,
                     label='EEGNet-16,4', color=MODEL_COLORS['eegnet'],
                     edgecolor='black', capsize=3, zorder=3)

    # 128ch reference lines
    if ref_cb_mean is not None:
        ax.axhline(y=ref_cb_mean, color=MODEL_COLORS['cbramod'],
                   linestyle='--', alpha=0.6, linewidth=1.5,
                   label=f'CBraMod 128ch ({ref_cb_mean:.1f}%)')
    if ref_eg_mean is not None:
        ax.axhline(y=ref_eg_mean, color=MODEL_COLORS['eegnet'],
                   linestyle='--', alpha=0.6, linewidth=1.5,
                   label=f'EEGNet 128ch ({ref_eg_mean:.1f}%)')

    # Value annotations
    for bars in [bars_cb, bars_eg]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + 1.5,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Cross-Subject Binary Accuracy (%)', fontsize=12)
    ax.set_title('32-Channel Configuration Comparison (N = 21)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, fontsize=11)
    ax.set_ylim(60, 100)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Spread annotation
    if cbramod_means:
        spread = max(cbramod_means) - min(cbramod_means)
        ax.text(0.02, 0.02, f'CBraMod method spread: {spread:.2f} pp',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / '32ch_comparison.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Saved: {out_path}')


# =============================================================================
# Figure 9: Extra Sessions Paradigm Summary
# =============================================================================

def generate_extra_sessions_paradigm_figure():
    """
    Extra sessions 三范式总览（CBraMod binary, N = 16）.

    目的：
      1. 将 within-subject / cross-subject / transfer-init 放到同一主文图中
      2. 直观展示“初始点更高 != 增量更大”的范式差异

    数据来源:
      within-subject: results/20260324_2131_extra_sessions_cache_imagery_binary.json
      cross-subject:  results/20260326_1409_cross_subject_extra_sessions_cache_imagery_binary.json
      transfer-init:  results/20260329_1357_extra_sessions_cache_imagery_binary.json
    """
    import matplotlib.pyplot as plt

    step_order = ['baseline', 'sess03', 'sess04', 'sess05']
    step_labels = ['Baseline', '+Sess03', '+Sess04', '+Sess05']
    x = np.arange(len(step_order))

    configs = [
        {
            'label': 'Within-Subject',
            'path': 'results/20260324_2131_extra_sessions_cache_imagery_binary.json',
            'loader': extract_extra_session_step_accs,
            'color': '#1976D2',
            'marker': 'o',
            'label_offset_y': 6,
        },
        {
            'label': 'Cross-Subject (21-subj train)',
            'path': 'results/20260326_1409_cross_subject_extra_sessions_cache_imagery_binary.json',
            'loader': extract_cross_subject_extra_session_step_accs,
            'color': '#EF6C00',
            'marker': 's',
            'label_offset_y': 0,
        },
        {
            'label': 'Transfer-Init',
            'path': 'results/20260329_1357_extra_sessions_cache_imagery_binary.json',
            'loader': extract_extra_session_step_accs,
            'color': '#2E7D32',
            'marker': 'D',
            'label_offset_y': -6,
        },
    ]

    series = []
    for cfg in configs:
        path = Path(cfg['path'])
        if not path.exists():
            logger.warning(f'Missing: {cfg["path"]}')
            continue
        cache = load_json_cache(cfg['path'])
        step_accs = cfg['loader'](cache, 'cbramod')
        means = np.array([
            np.mean(step_accs[step]) if step_accs.get(step) else np.nan
            for step in step_order
        ])
        sds = np.array([
            np.std(step_accs[step]) if step_accs.get(step) else 0.0
            for step in step_order
        ])
        if np.isnan(means).all():
            continue
        series.append({
            **cfg,
            'means': means,
            'sds': sds,
            'delta': means[-1] - means[0],
        })

    if len(series) < 2:
        logger.error('Insufficient data for extra_sessions_paradigm figure')
        return

    fig, (ax_line, ax_gain) = plt.subplots(
        1, 2, figsize=(13, 5.4), gridspec_kw={'width_ratios': [2.2, 1]}
    )

    for item in series:
        ax_line.plot(
            x, item['means'],
            color=item['color'],
            marker=item['marker'],
            linewidth=2.6,
            markersize=8,
            label=item['label'],
            zorder=3,
        )
        ax_line.fill_between(
            x,
            item['means'] - item['sds'],
            item['means'] + item['sds'],
            color=item['color'],
            alpha=0.12,
            zorder=1,
        )
        ax_line.annotate(
            f'{item["means"][-1]:.2f}%',
            xy=(x[-1], item['means'][-1]),
            xytext=(8, item.get('label_offset_y', 0)),
            textcoords='offset points',
            color=item['color'],
            fontsize=9,
            fontweight='bold',
            va='center',
        )

    ax_line.set_xticks(x)
    ax_line.set_xticklabels(step_labels, fontsize=11)
    ax_line.set_ylabel('Mean Accuracy ± SD (%)', fontsize=12)
    ax_line.set_title('A. Accuracy Trajectory', fontsize=13, fontweight='bold')
    ax_line.grid(True, alpha=0.25)
    ax_line.legend(loc='lower right', fontsize=9)

    y_min = min(float(np.nanmin(item['means'] - item['sds'])) for item in series)
    y_max = max(float(np.nanmax(item['means'] + item['sds'])) for item in series)
    ax_line.set_ylim(max(75, y_min - 2.5), min(100, y_max + 2.5))

    gain_x = np.arange(len(series))
    gains = [item['delta'] for item in series]
    gain_bars = ax_gain.bar(
        gain_x,
        gains,
        color=[item['color'] for item in series],
        width=0.65,
        alpha=0.9,
    )
    for bar, item in zip(gain_bars, series):
        height = bar.get_height()
        ax_gain.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.18 if height >= 0 else height - 0.18,
            f'{height:+.2f} pp',
            ha='center',
            va='bottom' if height >= 0 else 'top',
            fontsize=9,
            fontweight='bold',
            color=item['color'],
        )
        ax_gain.text(
            bar.get_x() + bar.get_width() / 2,
            0.2,
            f'Final {item["means"][-1]:.2f}%',
            ha='center',
            va='bottom',
            fontsize=8,
            rotation=90,
            color='#444444',
        )

    ax_gain.axhline(0, color='gray', linewidth=1, alpha=0.6)
    ax_gain.set_xticks(gain_x)
    ax_gain.set_xticklabels(
        ['Within', 'Cross', 'Transfer'],
        rotation=15,
        ha='right',
        fontsize=10,
    )
    ax_gain.set_ylabel('Gain vs Baseline (pp)', fontsize=12)
    ax_gain.set_title('B. Net Gain by +Sess05', fontsize=13, fontweight='bold')
    ax_gain.grid(axis='y', alpha=0.25)
    ax_gain.set_ylim(min(-1.5, min(gains) - 0.8), max(7.5, max(gains) + 1.0))

    fig.suptitle('Extra Sessions Across Training Paradigms (CBraMod Binary, N = 16)', fontsize=15, y=1.02)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / 'extra_sessions_paradigm_binary.png'
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Saved: {out_path}')


# =============================================================================
# Figure S1: Extra Sessions Strategy Comparison
# =============================================================================

def generate_extra_sessions_strategy_figure():
    """
    三种评估策略 (per_session / fixed_combined / fixed_sess02) 折线对比.

    数据来源:
      per_session:    results/20260324_2131_extra_sessions_cache_imagery_binary.json
      fixed_combined: results/20260325_0514_extra_sessions_cache_fixed_combined_imagery_binary.json
      fixed_sess02:   results/20260325_1208_extra_sessions_cache_fixed_sess02_imagery_binary.json
    """
    import matplotlib.pyplot as plt
    from src.config.constants import MODEL_COLORS

    strategy_configs = {
        'per_session': {
            'path': 'results/20260324_2131_extra_sessions_cache_imagery_binary.json',
            'label': 'per_session (default)',
            'linestyle': '-',
            'marker': 'o',
        },
        'fixed_combined': {
            'path': 'results/20260325_0514_extra_sessions_cache_fixed_combined_imagery_binary.json',
            'label': 'fixed_combined',
            'linestyle': '--',
            'marker': '^',
        },
        'fixed_sess02': {
            'path': 'results/20260325_1208_extra_sessions_cache_fixed_sess02_imagery_binary.json',
            'label': 'fixed_sess02',
            'linestyle': ':',
            'marker': 's',
        },
    }
    strategy_order = list(strategy_configs.keys())

    steps = ['baseline', 'sess03', 'sess04', 'sess05']
    step_labels = ['Baseline', '+Sess03', '+Sess04', '+Sess05']
    models = ['eegnet', 'cbramod']
    model_labels = {'eegnet': 'EEGNet-16,4', 'cbramod': 'CBraMod'}

    # Strategy colors (distinct from model colors)
    strategy_colors = {
        'per_session': '#2196F3',
        'fixed_combined': '#FF9800',
        'fixed_sess02': '#4CAF50',
    }

    # Load all data: {strategy: {model: {step: [accs]}}}
    all_data: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for strat_name, cfg in strategy_configs.items():
        path = cfg['path']
        if not Path(path).exists():
            logger.warning(f'Missing: {path}')
            continue
        cache = load_json_cache(path)
        results = cache.get('results', {})
        all_data[strat_name] = {}
        for model in models:
            model_results = results.get(model, {})
            subjects = sorted(k for k in model_results if k.startswith('S'))
            all_data[strat_name][model] = {}
            for step in steps:
                accs = []
                for subj in subjects:
                    subj_data = model_results[subj]
                    step_data = subj_data.get(step, {})
                    acc = step_data.get('test_acc_majority')
                    if acc is not None:
                        accs.append(acc * 100)
                all_data[strat_name][model][step] = accs

    if not all_data:
        logger.error('No strategy data loaded — check file paths')
        return

    def _spread_label_positions(
        target_ys: List[float],
        lower: float,
        upper: float,
        min_gap: float = 1.9,
    ) -> List[float]:
        """Greedy vertical label spreading for a small number of endpoint labels."""
        if not target_ys:
            return []

        adjusted = list(target_ys)
        order = sorted(range(len(adjusted)), key=lambda idx: adjusted[idx])
        for i in range(1, len(order)):
            prev_idx = order[i - 1]
            curr_idx = order[i]
            adjusted[curr_idx] = max(adjusted[curr_idx], adjusted[prev_idx] + min_gap)

        overshoot = adjusted[order[-1]] - upper
        if overshoot > 0:
            adjusted = [y - overshoot for y in adjusted]

        undershoot = lower - adjusted[order[0]]
        if undershoot > 0:
            adjusted = [y + undershoot for y in adjusted]

        for i in range(1, len(order)):
            prev_idx = order[i - 1]
            curr_idx = order[i]
            adjusted[curr_idx] = max(adjusted[curr_idx], adjusted[prev_idx] + min_gap)

        return [float(np.clip(y, lower, upper)) for y in adjusted]

    # Precompute series for each model to support cleaner labeling + exact-value tables.
    series_by_model: Dict[str, List[Dict[str, object]]] = {model: [] for model in models}
    for model in models:
        for strat_name in strategy_order:
            if strat_name not in all_data or model not in all_data[strat_name]:
                continue
            step_data = all_data[strat_name][model]
            means = np.array([
                np.mean(step_data[s]) if step_data[s] else np.nan for s in steps
            ])
            sds = np.array([
                np.std(step_data[s]) if step_data[s] else 0.0 for s in steps
            ])
            if np.isnan(means).all():
                continue
            series_by_model[model].append({
                'name': strat_name,
                'cfg': strategy_configs[strat_name],
                'means': means,
                'sds': sds,
                'delta': float(means[-1] - means[0]),
                'color': strategy_colors[strat_name],
            })

    # Plot: top row = line charts, bottom row = exact-value tables.
    fig = plt.figure(figsize=(15.2, 8.0))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[4.0, 1.45],
        hspace=0.14,
        wspace=0.12,
    )
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
    ]
    axes[1].sharey(axes[0])
    table_axes = [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]

    all_means = []
    for model_series in series_by_model.values():
        for item in model_series:
            means = item['means']
            sds = item['sds']
            all_means.extend((means - sds).tolist())
            all_means.extend((means + sds).tolist())

    y_min = 50.0
    y_max = 100.0
    if all_means:
        y_min = max(50, np.nanmin(all_means) - 2.0)
        y_max = min(100, np.nanmax(all_means) + 2.0)
    label_lower = y_min + 1.2
    label_upper = y_max - 1.2
    x = np.arange(len(steps))

    for ax, ax_table, model in zip(axes, table_axes, models):
        model_series = series_by_model[model]
        for item in model_series:
            cfg = item['cfg']
            means = item['means']
            sds = item['sds']
            color = item['color']

            ax.plot(
                x, means,
                marker=cfg['marker'],
                linestyle=cfg['linestyle'],
                color=color,
                linewidth=2.2,
                markersize=8,
                label=cfg['label'],
                zorder=3,
            )
            ax.fill_between(
                x,
                means - sds,
                means + sds,
                color=color,
                alpha=0.12,
                zorder=1,
            )

        if model_series:
            start_positions = _spread_label_positions(
                [float(item['means'][0]) for item in model_series],
                lower=label_lower,
                upper=label_upper,
            )
            end_positions = _spread_label_positions(
                [float(item['means'][-1]) for item in model_series],
                lower=label_lower,
                upper=label_upper,
            )

            for item, start_y, end_y in zip(model_series, start_positions, end_positions):
                means = item['means']
                color = item['color']

                # Baseline absolute value label.
                ax.plot(
                    [x[0], x[0] - 0.05, x[0] - 0.17],
                    [means[0], means[0], start_y],
                    color=color,
                    linewidth=1.0,
                    alpha=0.9,
                    clip_on=False,
                )
                ax.text(
                    x[0] - 0.2,
                    start_y,
                    f'{means[0]:.2f}%',
                    ha='right',
                    va='center',
                    fontsize=8.6,
                    fontweight='bold',
                    color=color,
                    bbox={
                        'boxstyle': 'round,pad=0.18',
                        'facecolor': 'white',
                        'edgecolor': color,
                        'alpha': 0.88,
                        'linewidth': 0.8,
                    },
                    clip_on=False,
                )

                # Final absolute value + total gain label.
                ax.plot(
                    [x[-1], x[-1] + 0.05, x[-1] + 0.17],
                    [means[-1], means[-1], end_y],
                    color=color,
                    linewidth=1.0,
                    alpha=0.9,
                    clip_on=False,
                )
                ax.text(
                    x[-1] + 0.2,
                    end_y,
                    f'{means[-1]:.2f}%\n({item["delta"]:+.2f} pp)',
                    ha='left',
                    va='center',
                    fontsize=8.6,
                    fontweight='bold',
                    color=color,
                    bbox={
                        'boxstyle': 'round,pad=0.22',
                        'facecolor': 'white',
                        'edgecolor': color,
                        'alpha': 0.88,
                        'linewidth': 0.8,
                    },
                    clip_on=False,
                )

        ax.set_title(model_labels[model], fontsize=14, fontweight='bold')
        ax.set_xticks(np.arange(len(steps)))
        ax.set_xticklabels(step_labels, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)
        ax.set_xlim(-0.45, len(steps) - 1 + 0.62)
        ax.set_ylim(y_min, y_max)

    axes[0].set_ylabel('Mean Accuracy ± SD (%)', fontsize=12)

    for ax_table, model in zip(table_axes, models):
        ax_table.axis('off')
        table_rows = []
        for item in series_by_model[model]:
            means = item['means']
            table_rows.append([
                item['cfg']['label'].replace(' (default)', ''),
                f'{means[0]:.2f}',
                f'{means[1]:.2f}',
                f'{means[2]:.2f}',
                f'{means[3]:.2f}',
                f'{item["delta"]:+.2f}',
            ])

        table = ax_table.table(
            cellText=table_rows,
            colLabels=['Strategy', 'BL', '+S03', '+S04', '+S05', 'Δ'],
            loc='center',
            cellLoc='center',
            colLoc='center',
            colWidths=[0.28, 0.11, 0.11, 0.11, 0.11, 0.1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1.0, 1.28)
        for (row, col), cell in table.get_celld().items():
            cell.set_linewidth(0.6)
            if row == 0:
                cell.set_facecolor('#F2F2F2')
                cell.set_text_props(fontweight='bold')
            elif col == 0:
                cell.set_text_props(
                    fontweight='bold',
                    color=series_by_model[model][row - 1]['color'],
                )
                cell.set_facecolor('#FBFBFB')
    fig.suptitle(
        'Extra Sessions: Evaluation Strategy Comparison (Binary, N = 16)',
        fontsize=15,
        y=0.97,
    )
    fig.subplots_adjust(top=0.86, bottom=0.08, left=0.06, right=0.97)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / 'extra_sessions_strategy_comparison.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Saved: {out_path}')


# =============================================================================
# Baseline Comparison Figures (paper-specific composition)
# =============================================================================

def generate_figure2_128ch_cross_subject():
    """
    Figure 2: 128ch 跨被试 4-way 对比（Within + Cross × EEGNet + CBraMod）.

    修复原问题：原图 (20260330_0709 run) 只包含 EEGNet cross-subject，
    CBraMod cross-subject 完全缺失。本函数从两个独立 run 合并数据。

    数据来源:
      Within EEGNet:  results/20260316_1411_comparison_cache_imagery_binary.json
      Within CBraMod: results/20260323_2237_comparison_cache_imagery_binary.json
      Cross EEGNet:   results/20260330_0709_cross_subject_cache_imagery_binary.json
      Cross CBraMod:  results/20260324_0023_cross_subject_cache_imagery_binary.json
    """
    task = 'binary'
    data_sources = []

    # Within-subject baselines (hatched, historical)
    for model, key in [('eegnet', 'eegnet_within'), ('cbramod', 'cbramod_within')]:
        src = _build_within_subject_source(
            BASELINE_128CH[key], model, task,
            label=f'{model.upper()} (Within)',
            is_current_run=False, hatch='///',
        )
        if src:
            data_sources.append(src)

    # Cross-subject results (solid, current)
    for model, key in [('eegnet', 'eegnet_cross'), ('cbramod', 'cbramod_cross')]:
        src = _build_cross_subject_source(
            BASELINE_128CH[key], model, task,
            label=f'{model.upper()} (Cross)',
            is_current_run=True,
        )
        if src:
            data_sources.append(src)

    if len(data_sources) < 2:
        logger.error('Insufficient data sources for Figure 2')
        return

    out_path = 'results/20260330_0709_cross-subject_combined_imagery_binary.png'
    generate_combined_plot(
        data_sources=data_sources,
        output_path=out_path,
        task_type=task,
        paradigm='imagery',
    )
    logger.info(f'Figure 2 saved: {out_path}')


def _generate_reduced_channel_baseline_figure(
    current_cache_path: str,
    output_path: str,
    channel_label: str,
    task: str = 'binary',
):
    """
    通用：缩减通道 cross-subject 图 + 128ch baseline overlay.

    生成 4 个 PlotDataSource:
      1. EEGNET ({channel_label} Cross) — 当前缩减通道, solid
      2. CBRAMOD ({channel_label} Cross) — 当前缩减通道, solid
      3. EEGNET (128ch Baseline) — 128ch 跨被试参考, dotted hatch
      4. CBRAMOD (128ch Baseline) — 128ch 跨被试参考, dotted hatch
    """
    data_sources = []

    # Current reduced-channel results (solid)
    for model in ['eegnet', 'cbramod']:
        src = _build_cross_subject_source(
            current_cache_path, model, task,
            label=f'{model.upper()} ({channel_label} Cross)',
            is_current_run=True,
        )
        if src:
            data_sources.append(src)

    # 128ch cross-subject baselines (dotted hatch, historical)
    for model, key in [('eegnet', 'eegnet_cross'), ('cbramod', 'cbramod_cross')]:
        src = _build_cross_subject_source(
            BASELINE_128CH[key], model, task,
            label=f'{model.upper()} (128ch Baseline)',
            is_current_run=False, hatch='...',
        )
        if src:
            data_sources.append(src)

    if len(data_sources) < 2:
        logger.error(f'Insufficient data sources for {output_path}')
        return

    generate_combined_plot(
        data_sources=data_sources,
        output_path=output_path,
        task_type=task,
        paradigm='imagery',
    )
    logger.info(f'Saved: {output_path}')


def generate_figure3b_32ch_fdr():
    """
    Figure 3b: 32ch FDR 跨被试对比 + 128ch baseline overlay.

    数据来源:
      32ch FDR: results/32_channel/fdr/20260330_0836_cross_subject_cache_imagery_binary.json
      128ch:    BASELINE_128CH (EEGNet + CBraMod cross-subject)
    """
    _generate_reduced_channel_baseline_figure(
        current_cache_path='results/32_channel/fdr/20260330_0836_cross_subject_cache_imagery_binary.json',
        output_path='results/32_channel/fdr/20260330_0836_cross-subject_combined_imagery_binary.png',
        channel_label='32ch',
    )
    logger.info('Figure 3b done')


def generate_figure5_4ch_control():
    """
    Figure 5a/5b: 4ch 控制实验 + 128ch baseline overlay.

    数据来源:
      5a (FDR∩Att): results/4_channel/fdr_attention_overlap/20260330_1417_*.json
      5b (Neg Ctrl): results/4_channel/negative_control/20260330_1442_*.json
      128ch:         BASELINE_128CH
    """
    configs = [
        (
            'results/4_channel/fdr_attention_overlap/20260330_1417_cross_subject_cache_imagery_binary.json',
            'results/4_channel/fdr_attention_overlap/20260330_1417_cross-subject_combined_imagery_binary.png',
            '4ch',
        ),
        (
            'results/4_channel/negative_control/20260330_1442_cross_subject_cache_imagery_binary.json',
            'results/4_channel/negative_control/20260330_1442_cross-subject_combined_imagery_binary.png',
            '4ch',
        ),
    ]
    for cache_path, out_path, ch_label in configs:
        _generate_reduced_channel_baseline_figure(
            current_cache_path=cache_path,
            output_path=out_path,
            channel_label=ch_label,
        )
    logger.info('Figure 5a/5b done')


def generate_all_baseline_plots():
    """生成所有需要 baseline overlay 修复的论文图表."""
    generate_figure2_128ch_cross_subject()
    generate_figure3b_32ch_fdr()
    generate_figure5_4ch_control()


# =============================================================================
# Main
# =============================================================================

FIGURE_GENERATORS = {
    'channel_scaling': generate_channel_scaling_figure,
    'further_pretraining': generate_further_pretraining_figure,
    'inference_latency': generate_inference_latency_figure,
    '32ch_comparison': generate_32ch_comparison_figure,
    'extra_sessions_paradigm': generate_extra_sessions_paradigm_figure,
    'extra_sessions_strategy': generate_extra_sessions_strategy_figure,
    'figure2': generate_figure2_128ch_cross_subject,
    'figure3b': generate_figure3b_32ch_fdr,
    'figure5': generate_figure5_4ch_control,
    'baseline_plots': generate_all_baseline_plots,
}


def main():
    parser = argparse.ArgumentParser(description='论文 v3 专属图表生成')
    parser.add_argument('--figure', required=True,
                        choices=list(FIGURE_GENERATORS.keys()) + ['all'],
                        help='要生成的图表')
    args = parser.parse_args()

    if args.figure == 'all':
        for name, gen in FIGURE_GENERATORS.items():
            logger.info(f'\n--- Generating: {name} ---')
            gen()
    else:
        FIGURE_GENERATORS[args.figure]()


if __name__ == '__main__':
    main()
