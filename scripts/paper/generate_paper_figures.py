#!/usr/bin/env python
"""
论文 v3 专属图表生成脚本.

生成现有 --replot 流程无法覆盖的论文图表：
  - channel_scaling: 通道数 vs 准确率曲线 (128→61→32→8→4)
  - further_pretraining: Further pre-training V1/V2 vs baseline 对比
  - inference_latency: 推理延迟对比柱状图
  - extra_sessions_paradigm: Extra sessions 三范式（within/cross/transfer）总览
  - extra_sessions_strategy: Extra sessions 三种评估策略折线对比

所有结果文件路径统一从 `paper/run_registry.yaml` 读取，避免论文阶段继续散落硬编码。

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
from src.visualization.paper_style import PAPER_COLORS, FONT_SIZES, apply_paper_style
from src.visualization.plots import force_directed_label_layout
from src.paper.run_registry import get_run_path, resolve_project_path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path('paper/figures')

# 128ch cross-subject baseline cache paths (used across multiple figure generators)
BASELINE_128CH = {
    'cbramod_cross': get_run_path('cross_cbramod_binary'),
    'eegnet_cross': get_run_path('cross_eegnet_binary'),
    'cbramod_within': get_run_path('within_cbramod_binary'),
    'eegnet_within': get_run_path('within_eegnet_binary'),
}


def add_provenance_footer(fig, generator_name: str):
    """Stamp 'generated date + script' provenance to a figure.

    Convention for v3+ paper figures (per editorial review 2026-05-05):
    every newly-authored figure should carry inline provenance so reviewers
    can trace back to the regeneration entrypoint without consulting the
    figure caption text. Backward compat for legacy figures is not required.
    """
    from datetime import datetime
    stamp = (f'Generated {datetime.now().strftime("%Y-%m-%d")} '
             f'by scripts/paper/generate_paper_figures.py --figure {generator_name}')
    fig.text(0.99, 0.005, stamp, ha='right', va='bottom',
             fontsize=7, color='gray', style='italic')


# =============================================================================
# Data Loading Helpers
# =============================================================================

def load_json_cache(path: str) -> dict:
    """Load a JSON result cache file."""
    with open(resolve_project_path(path), encoding='utf-8') as f:
        return json.load(f)


def _extract_accs_from_subject_mapping(subject_mapping: Dict[str, dict]) -> List[float]:
    """Extract percent accuracies from a subject_id -> result mapping."""
    accs = []
    for subject_id, subject_data in sorted(subject_mapping.items()):
        if subject_id in {'metadata', 'comparison', 'summary', 'statistics'}:
            continue
        if not isinstance(subject_data, dict):
            continue
        acc = subject_data.get('test_acc_majority', subject_data.get('test_acc'))
        if acc is not None:
            accs.append(acc * 100)
    return accs


def _extract_accs_from_subject_list(subjects: List[dict]) -> List[float]:
    """Extract percent accuracies from a serialized TrainingResult list."""
    accs = []
    for subject_data in sorted(
        subjects,
        key=lambda item: item.get('subject_id', '') if isinstance(item, dict) else '',
    ):
        if not isinstance(subject_data, dict):
            continue
        acc = subject_data.get('test_acc_majority', subject_data.get('test_acc'))
        if acc is not None:
            accs.append(acc * 100)
    return accs


def extract_model_accs(cache: dict, model: str) -> List[float]:
    """Extract per-subject test accuracies for a model from JSON cache.

    Supports multiple result layouts used across paper figures:
      - Comparison cache: data['results'][model][subject]['test_acc_majority']
      - Cross-subject cache: data['results'][model]['per_subject_test_acc'][subject]
      - Single-model cross-subject result: data['results']['per_subject_test_acc'][subject]
      - Single-model within-subject result: data['subjects'][i]['test_acc_majority']
      - Legacy top-level model map: data[model][subject]['test_acc_majority']
    """
    results = cache.get('results', {})
    model_results = results.get(model, {}) if isinstance(results, dict) else {}
    if isinstance(model_results, dict):
        per_subj = model_results.get('per_subject_test_acc', {})
        if per_subj:
            return [acc * 100 for _, acc in sorted(per_subj.items())]

        accs = _extract_accs_from_subject_mapping(model_results)
        if accs:
            return accs

    # Single-model cross-subject result (results -> per_subject_test_acc)
    if isinstance(results, dict):
        per_subj = results.get('per_subject_test_acc', {})
        metadata_model = cache.get('metadata', {}).get('model_type')
        if per_subj and metadata_model in (None, model):
            return [acc * 100 for _, acc in sorted(per_subj.items())]

    # Single-model within-subject result (subjects list)
    subjects = cache.get('subjects', [])
    if isinstance(subjects, list):
        metadata_model = cache.get('metadata', {}).get('model_type')
        if metadata_model in (None, model):
            accs = _extract_accs_from_subject_list(subjects)
            if accs:
                return accs

    # Legacy top-level model -> subject mapping
    model_data = cache.get(model, {})
    if isinstance(model_data, dict):
        return _extract_accs_from_subject_mapping(model_data)
    return []


def _load_further_pretraining_series() -> Dict[str, List[float]]:
    """Load Figure 2 values from the paper run registry."""
    run_specs = [
        (
            'Within-Subj\nBinary',
            {
                'baseline': 'further_pretraining_baseline_within_binary',
                'ft_v1': 'further_pretraining_v1_within_binary',
                'ft_v2': 'further_pretraining_v2_within_binary',
            },
        ),
        (
            'Cross-Subj\nBinary',
            {
                'baseline': 'further_pretraining_baseline_cross_binary',
                'ft_v1': 'further_pretraining_v1_cross_binary',
                'ft_v2': 'further_pretraining_v2_cross_binary',
            },
        ),
        (
            'Within-Subj\nTernary',
            {
                'baseline': 'further_pretraining_baseline_within_ternary',
                'ft_v1': 'further_pretraining_v1_within_ternary',
                'ft_v2': 'further_pretraining_v2_within_ternary',
            },
        ),
        (
            'Cross-Subj\nTernary',
            {
                'baseline': 'further_pretraining_baseline_cross_ternary',
                'ft_v1': 'further_pretraining_v1_cross_ternary',
                'ft_v2': 'further_pretraining_v2_cross_ternary',
            },
        ),
    ]

    series = {
        'conditions': [],
        'baseline': [],
        'ft_v1': [],
        'ft_v2': [],
    }
    for condition, keys in run_specs:
        series['conditions'].append(condition)
        for series_name, run_key in keys.items():
            cache = load_json_cache(get_run_path(run_key))
            accs = extract_model_accs(cache, 'cbramod')
            if not accs:
                raise ValueError(f'No CBraMod accuracies found for paper run key: {run_key}')
            series[series_name].append(float(np.mean(accs)))

    return series


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
    path = resolve_project_path(cache_path)
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
    path = resolve_project_path(cache_path)
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

def _legacy_unused_generate_channel_scaling_figure():
    """[LEGACY / UNUSED — superseded by generate_channel_scaling_v2_figure, kept for reference only]

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
            (32, get_run_path('reduced_32_fdr_binary')),
            (8,  get_run_path('reduced_8_fdr_binary')),
            (4,  get_run_path('reduced_4_fdr_binary')),
        ],
        'Band Power': [
            (32, get_run_path('reduced_32_band_power_binary')),
            (8,  get_run_path('reduced_8_band_power_binary')),
        ],
        'Attention': [
            (32, get_run_path('reduced_32_attention_binary')),
            (8,  get_run_path('reduced_8_attention_binary')),
            (4,  get_run_path('reduced_4_attention_binary')),
        ],
        'CSP': [
            (32, get_run_path('reduced_32_csp_binary')),
            (8,  get_run_path('reduced_8_csp_binary')),
        ],
        'Commercial': [
            (32, get_run_path('reduced_32_commercial_binary')),
        ],
    }

    # Special 4ch configs (not tracked as multi-count methods)
    special_points = {
        'FDR∩Att': (4, get_run_path('reduced_4_fdr_attention_overlap_binary')),
        'Neg. Control': (4, get_run_path('reduced_4_negative_control_binary')),
    }

    # Baseline points (no method selection)
    baseline_paths = [
        (128, get_run_path('cross_cbramod_binary')),
        (61,  get_run_path('standard_1010_61_cross_binary')),
    ]

    # ── Load all data ──
    def _load_acc(path):
        if not resolve_project_path(path).exists():
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
                    fontsize=9, color='darkorange', fontstyle='italic',
                    arrowprops=dict(arrowstyle='->', color='darkorange', lw=0.8))

    if 'Neg. Control' in special_data:
        _, m, s = special_data['Neg. Control']
        ax.errorbar([3.7], [m], yerr=[s], marker='x', markersize=11,
                    linewidth=0, color='gray', capsize=4, markeredgewidth=2,
                    label=f'Neg. Control (4ch): {m:.1f}%', zorder=3)

    # 4) Reference lines
    ax.axhline(y=50, color=PAPER_COLORS['chance_red'], linestyle='--',
               linewidth=1.0, alpha=0.85, label='Chance (50%)')
    ax.axvspan(28, 36, alpha=0.08, color='green', label='32ch Deploy Zone')

    ax.set_xlabel('Number of Channels', fontsize=11)
    ax.set_ylabel('CBraMod Cross-Subject Binary Accuracy (%)', fontsize=11)
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
                transform=ax.transAxes, fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / 'channel_scaling_curve.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Saved: {out_path}')


# =============================================================================
# Figure 2: Further Pre-training Comparison
# =============================================================================

def _legacy_unused_generate_further_pretraining_figure():
    """[LEGACY / UNUSED — superseded by generate_further_pretraining_v3_figure, kept for reference only]

    Further pre-training V1/V2 vs baseline 下游性能对比.

    数据来源统一由 `paper/run_registry.yaml` 中的
    `further_pretraining_baseline_*`、`further_pretraining_v1_*`、
    `further_pretraining_v2_*` 条目提供，运行时从结果文件读取。
    """
    import matplotlib.pyplot as plt

    series = _load_further_pretraining_series()
    conditions = series['conditions']
    baseline = series['baseline']
    ft_v1 = series['ft_v1']
    ft_v2 = series['ft_v2']

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

    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('Domain-Adaptive Further Pre-training: Downstream Evaluation', fontsize=12)
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
                 f'{val:.2f}pp', ha='center', va='top', fontsize=9, fontweight='bold')

    ax2.set_ylabel('Mean Delta vs Baseline (pp)', fontsize=11)
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

def _legacy_unused_generate_inference_latency_figure():
    """[LEGACY / UNUSED — superseded by generate_inference_latency_v2_figure, kept for reference only]

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
    ax1.set_ylabel('Latency (ms, log scale)', fontsize=11)
    ax1.set_xlabel('Batch Size', fontsize=11)
    ax1.set_title('Inference Latency Comparison (128ch Binary)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(batch_sizes)
    ax1.legend(fontsize=9)

    # 100ms real-time threshold line
    ax1.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='100ms 实时阈值')
    ax1.text(0.02, 100, '100ms', fontsize=9, color='red', va='bottom')

    ax1.grid(axis='y', alpha=0.3)

    # Annotate batch=1 latencies
    ax1.annotate(f'{eegnet_latency[0]:.1f}ms', xy=(x[0] - width/2, eegnet_latency[0]),
                 xytext=(0, 8), textcoords='offset points', fontsize=9, ha='center')
    ax1.annotate(f'{cbramod_latency[0]:.1f}ms', xy=(x[0] + width/2, cbramod_latency[0]),
                 xytext=(0, 8), textcoords='offset points', fontsize=9, ha='center')

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
    ax2.set_xlabel('CBraMod / EEGNet Ratio', fontsize=11)
    ax2.set_title('Model Scale Comparison', fontsize=12)
    ax2.set_xscale('log')
    ax2.axvline(x=1, color='gray', linestyle='--', alpha=0.5)

    for i, (ratio, ev, cv) in enumerate(zip(ratios, eegnet_vals, cbramod_vals)):
        ax2.text(ratio * 1.1, i, f'{ratio:.0f}×', va='center', fontsize=9, fontweight='bold')

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
        ('FDR', get_run_path('reduced_32_fdr_binary')),
        ('Band Power', get_run_path('reduced_32_band_power_binary')),
        ('Commercial', get_run_path('reduced_32_commercial_binary')),
        ('Attention', get_run_path('reduced_32_attention_binary')),
        ('CSP', get_run_path('reduced_32_csp_binary')),
    ]

    # 128ch reference
    ref_128_cbramod = get_run_path('cross_cbramod_binary')
    ref_128_eegnet = get_run_path('cross_eegnet_binary')

    config_names = []
    cbramod_means, cbramod_stds = [], []
    eegnet_means, eegnet_stds = [], []

    for name, path in configs:
        if not resolve_project_path(path).exists():
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
    if resolve_project_path(ref_128_cbramod).exists():
        cache = load_json_cache(ref_128_cbramod)
        accs = extract_model_accs(cache, 'cbramod')
        if accs:
            ref_cb_mean = np.mean(accs)

    ref_eg_mean = None
    if resolve_project_path(ref_128_eegnet).exists():
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
                        f'{height:.1f}', ha='center', va='bottom', fontsize=FONT_SIZES['annotation'])

    ax.set_ylabel('Cross-Subject Binary Accuracy (%)', fontsize=FONT_SIZES['axis_label'])
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, fontsize=FONT_SIZES['tick'])
    ax.set_ylim(60, 100)
    ax.legend(loc='lower right', fontsize=FONT_SIZES['legend'])
    ax.grid(axis='y', alpha=0.3)

    # Spread annotation
    if cbramod_means:
        spread = max(cbramod_means) - min(cbramod_means)
        ax.text(0.02, 0.02, f'CBraMod method spread: {spread:.2f} pp',
                transform=ax.transAxes, fontsize=FONT_SIZES['annotation'],
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.tight_layout()
    apply_paper_style(fig=fig)
    add_provenance_footer(fig, '32ch_comparison')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / '32ch_comparison.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Saved: {out_path}')


# =============================================================================
# Figure 3d: Reduced-channel × method × task 40-cell matrix overview
# =============================================================================

def generate_reduced_channel_grid_figure():
    """Reduced-channel × method × task 40-cell grid panel (cross-subject CBraMod).

    Layout: 2 row × 4 col = 8 panels
      Row 0: binary  | Row 1: ternary
      Col:   4ch    |   8ch   |  32ch  |  64ch
      Per panel: 5 method bars (FDR/Attention/Band Power/CSP/negative_control)
                 with mean ± std error bars, n=21 subjects.

    Reference lines: 128ch CBraMod cross-subject baseline (binary 90.68%, ternary 74.88%).

    Data source: 40 aliases under `paper/run_registry.yaml`, registered 2026-05-12
    as part of the 4×5×2 matrix closure. See [docs/handoffs/2026-05-11_reduced_channel_matrix_closure.md].
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from src.visualization.paper_style import (
        PAPER_COLORS, FONT_SIZES, apply_paper_style, paper_figsize, add_panel_label,
    )

    METHODS = ['fdr', 'attention', 'band_power', 'csp', 'negative_control']
    METHOD_LABELS = {
        'fdr': 'FDR', 'attention': 'Attention', 'band_power': 'Band Power',
        'csp': 'CSP', 'negative_control': 'Neg. Control',
    }
    CHANNEL_COUNTS = [4, 8, 32, 64]
    TASKS = ['binary', 'ternary']

    # --- Load all 40 cells ---
    data = {n: {t: {} for t in TASKS} for n in CHANNEL_COUNTS}
    for n in CHANNEL_COUNTS:
        for t in TASKS:
            for m in METHODS:
                alias = f'reduced_{n}_{m}_{t}'
                cache = load_json_cache(get_run_path(alias))
                accs = extract_model_accs(cache, 'cbramod')
                if len(accs) != 21:
                    raise ValueError(
                        f'{alias}: expected 21 subjects, got {len(accs)}. '
                        f'Check run_registry.yaml + JSON cache integrity.'
                    )
                data[n][t]['mean_' + m] = float(np.mean(accs))
                data[n][t]['std_' + m] = float(np.std(accs))

    # --- 128ch baselines ---
    cache_b = load_json_cache(get_run_path('cross_cbramod_binary'))
    cache_t = load_json_cache(get_run_path('cross_cbramod_ternary'))
    ref_binary = float(np.mean(extract_model_accs(cache_b, 'cbramod')))
    ref_ternary = float(np.mean(extract_model_accs(cache_t, 'cbramod')))

    # --- Plot ---
    fig, axes = plt.subplots(
        2, 4,
        figsize=paper_figsize(rows=2, cols=4, width_in=11.0, row_height_in=3.6),
        sharey='row',
    )

    x = np.arange(len(METHODS))
    bar_colors = [PAPER_COLORS[m] for m in METHODS]
    y_limits = {'binary': (45, 100), 'ternary': (30, 85)}
    panel_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    for row_idx, task in enumerate(TASKS):
        for col_idx, n_ch in enumerate(CHANNEL_COUNTS):
            ax = axes[row_idx, col_idx]
            cell = data[n_ch][task]
            means = [cell['mean_' + m] for m in METHODS]
            stds = [cell['std_' + m] for m in METHODS]

            bars = ax.bar(
                x, means, yerr=stds, color=bar_colors,
                edgecolor='black', linewidth=0.6, capsize=2.5,
                error_kw={'elinewidth': 0.8}, zorder=3,
            )

            # 128ch baseline reference line
            ref = ref_binary if task == 'binary' else ref_ternary
            ax.axhline(y=ref, color='black', linestyle='--', linewidth=0.9,
                       alpha=0.55, zorder=2)

            # Value annotations above bars
            for bar, mean in zip(bars, means):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.2,
                    f'{mean:.1f}',
                    ha='center', va='bottom',
                    fontsize=FONT_SIZES['annotation'] - 1,
                )

            ax.set_xticks(x)
            ax.set_xticklabels([])  # method labels in fig-level legend
            ax.set_ylim(*y_limits[task])
            ax.grid(axis='y', alpha=0.25, zorder=1)

            # Column title (channel count) on top row only
            if row_idx == 0:
                ax.set_title(f'{n_ch} channels', fontsize=FONT_SIZES['title'], pad=8)

            # Row label (task) on leftmost column only
            if col_idx == 0:
                ax.set_ylabel(
                    f'{task.capitalize()} accuracy (%)',
                    fontsize=FONT_SIZES['axis_label'],
                )

            # Panel letter
            idx = row_idx * 4 + col_idx
            add_panel_label(ax, panel_letters[idx], x=-0.06, y=1.02,
                            fontsize=FONT_SIZES['panel_label'])

    # Fig-level legend (5 method swatches + baseline)
    legend_handles = [
        mpatches.Patch(color=PAPER_COLORS[m], label=METHOD_LABELS[m]) for m in METHODS
    ]
    legend_handles.append(plt.Line2D(
        [0], [0], color='black', linestyle='--', linewidth=1.1,
        label='128ch baseline (binary 90.68%, ternary 74.88%)',
    ))
    fig.legend(
        handles=legend_handles, loc='lower center', ncol=6,
        bbox_to_anchor=(0.5, 0.03), frameon=False,
        fontsize=FONT_SIZES['legend'],
    )

    fig.tight_layout(rect=(0, 0.10, 1, 0.97))
    apply_paper_style(fig=fig)
    add_provenance_footer(fig, 'reduced_channel_40cell_grid')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / 'reduced_channel_40cell_grid.png'
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
            'path': get_run_path('extra_sessions_binary'),
            'loader': extract_extra_session_step_accs,
            'color': PAPER_COLORS['secondary_blue'],
            'marker': 'o',
            'label_offset_y': 6,
        },
        {
            'label': 'Cross-Subject (21-subj train)',
            'path': get_run_path('extra_sessions_cross_binary'),
            'loader': extract_cross_subject_extra_session_step_accs,
            'color': '#EF6C00',
            'marker': 's',
            'label_offset_y': 0,
        },
        {
            'label': 'Transfer-Init',
            'path': get_run_path('extra_sessions_transfer_binary'),
            'loader': extract_extra_session_step_accs,
            'color': '#2E7D32',
            'marker': 'D',
            'label_offset_y': -6,
        },
    ]

    series = []
    for cfg in configs:
        path = resolve_project_path(cfg['path'])
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
            fontsize=FONT_SIZES['annotation'],
            fontweight='bold',
            va='center',
        )

    ax_line.set_xticks(x)
    ax_line.set_xticklabels(step_labels, fontsize=FONT_SIZES['tick'])
    ax_line.set_ylabel('Mean Accuracy ± SD (%)', fontsize=FONT_SIZES['axis_label'])
    ax_line.set_title('A. Accuracy Trajectory', fontsize=FONT_SIZES['title'], fontweight='bold')
    ax_line.grid(True, alpha=0.25)
    ax_line.legend(loc='lower right', fontsize=FONT_SIZES['legend'])

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
            fontsize=FONT_SIZES['annotation'],
            fontweight='bold',
            color=item['color'],
        )
        ax_gain.text(
            bar.get_x() + bar.get_width() / 2,
            0.2,
            f'Final {item["means"][-1]:.2f}%',
            ha='center',
            va='bottom',
            fontsize=FONT_SIZES['annotation'],
            rotation=90,
            color='#444444',
        )

    ax_gain.axhline(0, color='gray', linewidth=1, alpha=0.6)
    ax_gain.set_xticks(gain_x)
    ax_gain.set_xticklabels(
        ['Within', 'Cross', 'Transfer'],
        rotation=15,
        ha='right',
        fontsize=FONT_SIZES['tick'],
    )
    ax_gain.set_ylabel('Gain vs Baseline (pp)', fontsize=FONT_SIZES['axis_label'])
    ax_gain.set_title('B. Net Gain by +Sess05', fontsize=FONT_SIZES['title'], fontweight='bold')
    ax_gain.grid(axis='y', alpha=0.25)
    ax_gain.set_ylim(min(-1.5, min(gains) - 0.8), max(7.5, max(gains) + 1.0))

    fig.tight_layout()

    apply_paper_style(fig=fig)
    add_provenance_footer(fig, 'extra_sessions_paradigm')
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
            'path': get_run_path('extra_sessions_binary'),
            'label': 'per_session (default)',
            'linestyle': '-',
            'marker': 'o',
        },
        'fixed_combined': {
            'path': get_run_path('extra_sessions_fixed_combined_binary'),
            'label': 'fixed_combined',
            'linestyle': '--',
            'marker': '^',
        },
        'fixed_sess02': {
            'path': get_run_path('extra_sessions_fixed_sess02_binary'),
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
        if not resolve_project_path(path).exists():
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
                    fontsize=FONT_SIZES['annotation'],
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
                    fontsize=FONT_SIZES['annotation'],
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

        ax.set_title(model_labels[model], fontsize=FONT_SIZES['title'], fontweight='bold')
        ax.set_xticks(np.arange(len(steps)))
        ax.set_xticklabels(step_labels, fontsize=FONT_SIZES['tick'])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=FONT_SIZES['legend'])
        ax.set_xlim(-0.45, len(steps) - 1 + 0.62)
        ax.set_ylim(y_min, y_max)

    axes[0].set_ylabel('Mean Accuracy ± SD (%)', fontsize=FONT_SIZES['axis_label'])

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
    fig.subplots_adjust(top=0.86, bottom=0.08, left=0.06, right=0.97)

    apply_paper_style(fig=fig)
    add_provenance_footer(fig, 'extra_sessions_strategy')
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
        current_cache_path=get_run_path('reduced_32_fdr_binary'),
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
            get_run_path('reduced_4_fdr_attention_overlap_binary'),
            'results/4_channel/fdr_attention_overlap/20260330_1417_cross-subject_combined_imagery_binary.png',
            '4ch',
        ),
        (
            get_run_path('reduced_4_negative_control_binary'),
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
# v3 修订增补图（plan EDITs T3.1 / T3.3）
# =============================================================================

def generate_8ch_ranking_flip_figure():
    """T3.1 — 8ch 方法排序翻转 slope chart.

    把 4 种通道选择方法在 32ch / 8ch / 4ch 三档上的 cross-subject CBraMod 准确率
    画成 slope chart，强调"32ch 第一名（FDR）在 8ch 跌至第三、Band Power 反超"
    的方法选择敏感度现象（v3 Section 3.5.2 末段对应论点）。

    数据来源：
      32ch: reduced_32_{fdr, band_power, csp, attention}_binary
      8ch:  reduced_8_{fdr, band_power, csp, attention}_binary
      4ch:  reduced_4_{fdr, band_power, csp, attention}_binary
            (BP/CSP 来自 2026-05-05 补充实验：20260505_2308 / 20260505_2246)
      负控制（4ch）作虚线天花板对照
    """
    import matplotlib.pyplot as plt

    methods = ['FDR', 'Band Power', 'CSP', 'Attention']
    method_keys = {
        'FDR': ('reduced_32_fdr_binary', 'reduced_8_fdr_binary',
                'reduced_4_fdr_binary'),
        'Band Power': ('reduced_32_band_power_binary',
                       'reduced_8_band_power_binary',
                       'reduced_4_band_power_binary'),
        'CSP': ('reduced_32_csp_binary', 'reduced_8_csp_binary',
                'reduced_4_csp_binary'),
        'Attention': ('reduced_32_attention_binary',
                      'reduced_8_attention_binary',
                      'reduced_4_attention_binary'),
    }
    method_colors = {
        'FDR':        PAPER_COLORS['fdr'],
        'Band Power': PAPER_COLORS['band_power'],
        'CSP':        PAPER_COLORS['csp'],
        'Attention':  PAPER_COLORS['attention'],
    }
    channel_levels = ['32ch', '8ch', '4ch']

    means_table = {m: [] for m in methods}
    for method in methods:
        for level_idx, key in enumerate(method_keys[method]):
            if key is None:
                means_table[method].append(np.nan)
                continue
            cache_path = get_run_path(key)
            if not resolve_project_path(cache_path).exists():
                logger.warning(f'Missing for {method} @ {channel_levels[level_idx]}: {cache_path}')
                means_table[method].append(np.nan)
                continue
            cache = load_json_cache(cache_path)
            accs = extract_model_accs(cache, 'cbramod')
            means_table[method].append(np.mean(accs) if accs else np.nan)

    # 4ch 负控制 baseline（天花板对照）
    neg_ctrl_path = get_run_path('reduced_4_negative_control_binary')
    neg_ctrl_mean = None
    if resolve_project_path(neg_ctrl_path).exists():
        cache = load_json_cache(neg_ctrl_path)
        accs = extract_model_accs(cache, 'cbramod')
        if accs:
            neg_ctrl_mean = np.mean(accs)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    x_positions = np.arange(len(channel_levels))

    for method in methods:
        ys = np.array(means_table[method], dtype=float)
        # 实线连接非缺失段；缺失档位作空心标记
        valid_mask = ~np.isnan(ys)
        ax.plot(x_positions[valid_mask], ys[valid_mask],
                marker='o', markersize=10, linewidth=2.2,
                color=method_colors[method], label=method, zorder=3)
        # 缺失档位空心
        for xi, yi, valid in zip(x_positions, ys, valid_mask):
            if not valid:
                ax.plot(xi, 0, marker='x', markersize=8,
                        color=method_colors[method], alpha=0.4)

    # 4ch 负控制虚线
    if neg_ctrl_mean is not None:
        ax.axhline(y=neg_ctrl_mean, color='gray', linestyle=':', linewidth=1.5,
                   alpha=0.7,
                   label=f'4ch Negative Control ({neg_ctrl_mean:.1f}%)')

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'{lvl}\n(channels reduced)' for lvl in channel_levels],
                       fontsize=FONT_SIZES['tick'])
    ax.set_ylabel('Cross-Subject Binary Accuracy (%)',
                  fontsize=FONT_SIZES['axis_label'])
    ax.set_ylim(50, 100)

    # Sorted ranking stack at top of each channel-level position
    y_top = ax.get_ylim()[1]
    for level_idx, level in enumerate(channel_levels):
        ranking = sorted(
            [(m, means_table[m][level_idx]) for m in methods
             if not np.isnan(means_table[m][level_idx])],
            key=lambda x: x[1], reverse=True,
        )
        stack_text = '\n'.join(
            f'#{r} {m:<11} {v:.1f}%' for r, (m, v) in enumerate(ranking, start=1)
        )
        ax.text(level_idx, y_top * 0.995, stack_text,
                ha='center', va='top',
                fontsize=FONT_SIZES['annotation'] - 1,
                family='monospace',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='white', edgecolor='lightgray', alpha=0.9))

    ax.grid(axis='y', alpha=0.3, zorder=1)
    ax.legend(loc='lower right', fontsize=FONT_SIZES['legend'])

    # Annotation box: ranking flip emphasis
    flip_text = ('At 32ch: FDR > Band Power\n'
                 'At 8ch: Band Power > FDR (reversal)')
    ax.text(0.02, 0.04, flip_text, transform=ax.transAxes,
            fontsize=FONT_SIZES['annotation'],
            bbox=dict(boxstyle='round,pad=0.5',
                      facecolor='lightyellow', alpha=0.8),
            verticalalignment='bottom')

    fig.tight_layout()
    apply_paper_style(fig=fig)
    add_provenance_footer(fig, 'channel_ranking_flip')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / 'channel_method_ranking_flip.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Saved: {out_path}')


def generate_cross_subject_pooling_forest_figure():
    """T3.3 — 跨被试 vs 被试内 forest plot (Finding #1 视觉支持).

    展示 4 个 cells (CBraMod within / cross, EEGNet within / cross) 的均值 ± SD
    + 21 名被试个体散点 + paired-t 配对显著性，让"CBraMod 从 cross-subject pooling
    显著获益、EEGNet 不获益"在一张图上自闭环（v3 Section 4.1 / Finding #1 对应）。

    数据来源：within_{cbramod, eegnet}_binary, cross_{cbramod, eegnet}_binary
    """
    import matplotlib.pyplot as plt
    from src.config.constants import MODEL_COLORS

    cells = [
        ('CBraMod\nWithin-Subject', 'within_cbramod_binary', 'cbramod'),
        ('CBraMod\nCross-Subject',  'cross_cbramod_binary',  'cbramod'),
        ('EEGNet\nWithin-Subject',  'within_eegnet_binary',  'eegnet'),
        ('EEGNet\nCross-Subject',   'cross_eegnet_binary',   'eegnet'),
    ]

    cell_data = []
    for label, registry_key, model in cells:
        cache_path = get_run_path(registry_key)
        if not resolve_project_path(cache_path).exists():
            logger.warning(f'Missing: {cache_path}')
            cell_data.append((label, model, []))
            continue
        cache = load_json_cache(cache_path)
        accs = extract_model_accs(cache, model)
        cell_data.append((label, model, accs))

    fig, ax = plt.subplots(figsize=(10, 6.5))
    y_positions = np.arange(len(cell_data))[::-1]  # top-to-bottom

    for y, (label, model, accs) in zip(y_positions, cell_data):
        if not accs:
            continue
        mean = np.mean(accs)
        std = np.std(accs)
        # SD bar
        ax.errorbar(mean, y, xerr=std, fmt='o', markersize=12,
                    color=MODEL_COLORS[model],
                    ecolor='gray', capsize=6, elinewidth=2,
                    markeredgecolor='black', zorder=4)
        # Per-subject jitter
        jitter = np.random.RandomState(42).uniform(-0.18, 0.18, size=len(accs))
        ax.scatter(accs, np.full_like(accs, y) + jitter,
                   color=MODEL_COLORS[model], alpha=0.4, s=18,
                   edgecolor='none', zorder=3)
        # Mean ± SD label
        ax.text(mean + std + 1.5, y, f'  {mean:.2f} ± {std:.2f}%',
                va='center', fontsize=FONT_SIZES['annotation'], color='black')

    # Δ (cross − within) annotations + paired-t p-values
    from scipy import stats as sp_stats
    cross_idx = {'cbramod': 1, 'eegnet': 3}
    within_idx = {'cbramod': 0, 'eegnet': 2}
    for model in ['cbramod', 'eegnet']:
        wi = within_idx[model]
        ci = cross_idx[model]
        if not cell_data[wi][2] or not cell_data[ci][2]:
            continue
        within_accs = cell_data[wi][2]
        cross_accs = cell_data[ci][2]
        if len(within_accs) != len(cross_accs):
            continue
        delta = np.mean(cross_accs) - np.mean(within_accs)
        t_stat, p_val = sp_stats.ttest_rel(cross_accs, within_accs)
        # Annotate at cross row
        y_cross = y_positions[ci]
        sign_str = ('p < 0.001' if p_val < 0.001
                    else f'p = {p_val:.3f}')
        delta_label = (f'Δ(cross − within) = {delta:+.2f} pp ({sign_str})')
        weight = 'bold' if p_val < 0.05 else 'normal'
        face = 'lightgreen' if p_val < 0.05 else 'whitesmoke'
        ax.text(0.98, y_cross - 0.4, delta_label,
                transform=ax.get_yaxis_transform(),
                ha='right', fontsize=FONT_SIZES['annotation'], fontweight=weight,
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor=face, alpha=0.9))

    ax.set_yticks(y_positions)
    ax.set_yticklabels([cd[0] for cd in cell_data], fontsize=FONT_SIZES['tick'])
    ax.set_xlabel('Binary Accuracy (%)', fontsize=FONT_SIZES['axis_label'])
    ax.set_xlim(40, 105)
    ax.axvline(x=50, color=PAPER_COLORS['chance_red'], linestyle='--',
               linewidth=1.0, alpha=0.85,
               label='Chance (50%)')
    ax.grid(axis='x', alpha=0.3)
    ax.legend(loc='lower right', fontsize=FONT_SIZES['legend'])
    fig.tight_layout()
    apply_paper_style(fig=fig)
    add_provenance_footer(fig, 'cross_subject_pooling_forest')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / 'cross_subject_pooling_forest.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Saved: {out_path}')


# =============================================================================
# Stage 4 Step 3 — v3.1 paper figure additions
# =============================================================================
#
# The following 10 figure generators (NEW-A, NEW-B, T3.4, T3.5, T3.6, T3.2,
# T3.7, T3.8, T3.9, T3.10) implement the "10 figures for v3.1" work item list
# described in `paper/reviews/stage4_step3_figures_report.md`.
#
# Conventions:
#   - V1-V3 numbers (within+cross, 12 cells) come from `stage4_step1b_stat_recompute_v4v5.md`
#     (16-cell family, original v3.1 audit). V4/V5 within+transfer (8 cells) come from
#     `stage4_step1c_v4v5_within_transfer.md`. V1/V2/V3 transfer (6 cells, 2026-05-11
#     补完) come from `stage4_step1d_v1v2v3_transfer.md` (`scripts/internal/recompute_v1v2v3_transfer.py`).
#   - BH-FDR q-values RECOMPUTED over the new 30-cell DAPT family — q-values
#     for the original 24 cells differ from Step 1c values (24→30 family
#     expansion shifts FDR threshold; v3.1 paper text under 24-family had 1 BH
#     survivor (V2_within_binary q=0.048), under 30-family 0/30 survive
#     (V2_within_binary q=0.060). Figure caption discloses "BH-FDR @ 30-cell
#     family"; raw p values unchanged.
#   - Stouffer aggregates: 6 paradigm-level (5V each) computed in Step 1d.
#     Transfer aggregates UPGRADED from V4/V5-only n=2 (Step 1c) to V1-V5 n=5:
#     transfer_binary Z=-2.788 → -3.391 (strengthened); transfer_ternary
#     Z=-1.597 → +0.176 (direction flipped: V1/V2/V3 ternary positive).
#     1 legacy full-DAPT n=16 diamond preserved for v3.1 historical continuity.
#     Total 7 diamonds.
#   - All paths are resolved through `paper/run_registry.yaml` where possible;
#     V4/V5 caches that postdate the registry are cited inline.

# Verified numbers from stat recomputes (single source of truth for figures touching DAPT)
# q values are 30-family BH-FDR from Step 1d (`scripts/internal/recompute_v1v2v3_transfer.py`).
DAPT_V_RESULTS_STEP1B = [
    # (V, paradigm, task, mean_diff_pp, ci_low_pp, ci_high_pp, q_dapt, bh_sig, p)
    # --- V1 (within+cross Step 1b; transfer Step 1d) — q recomputed in 30-cell family ---
    ('V1', 'within',   'binary',  -1.25, -2.83, +0.33, 0.230, False, 0.115),
    ('V1', 'within',   'ternary', -0.30, -1.67, +1.08, 0.757, False, 0.656),
    ('V1', 'cross',    'binary',  -1.85, -3.18, -0.52, 0.090, False, 0.009),
    ('V1', 'cross',    'ternary', +0.79, -0.95, +2.53, 0.504, False, 0.353),
    ('V1', 'transfer', 'binary',  -1.10, -2.72, +0.52, 0.301, False, 0.171),
    ('V1', 'transfer', 'ternary', +0.65, -1.08, +2.39, 0.575, False, 0.441),
    # --- V2 (within+cross Step 1b; transfer Step 1d) — q recomputed in 30-cell family ---
    ('V2', 'within',   'binary',  -2.86, -4.54, -1.17, 0.060, False, 0.002),
    ('V2', 'within',   'ternary', -1.47, -3.20, +0.27, 0.230, False, 0.093),
    ('V2', 'cross',    'binary',  -1.25, -2.33, -0.17, 0.111, False, 0.025),
    ('V2', 'cross',    'ternary', +0.44, -0.78, +1.65, 0.578, False, 0.462),
    ('V2', 'transfer', 'binary',  -0.74, -2.07, +0.58, 0.387, False, 0.255),
    ('V2', 'transfer', 'ternary', +0.18, -1.08, +1.43, 0.796, False, 0.770),
    # --- V3 (within+cross Step 1b; transfer Step 1d) — q recomputed in 30-cell family ---
    ('V3', 'within',   'binary',  -1.34, -3.02, +0.34, 0.230, False, 0.112),
    ('V3', 'within',   'ternary', -0.24, -1.65, +1.18, 0.781, False, 0.729),
    ('V3', 'cross',    'binary',  -1.46, -2.92, +0.01, 0.191, False, 0.051),
    ('V3', 'cross',    'ternary', +0.62, -0.83, +2.06, 0.524, False, 0.384),
    ('V3', 'transfer', 'binary',  -1.01, -2.82, +0.80, 0.387, False, 0.258),
    ('V3', 'transfer', 'ternary', +1.09, -0.28, +2.46, 0.230, False, 0.111),
    # --- V4 (cross Step 1b; within+transfer Step 1c) — q recomputed in 30-cell family ---
    ('V4', 'cross',    'binary',  -1.61, -2.75, -0.46, 0.090, False, 0.008),
    ('V4', 'cross',    'ternary', +0.22, -1.63, +2.06, 0.808, False, 0.808),
    ('V4', 'within',   'binary',  -1.10, -2.81, +0.61, 0.323, False, 0.194),
    ('V4', 'within',   'ternary', -0.56, -2.48, +1.36, 0.664, False, 0.553),
    ('V4', 'transfer', 'binary',  -1.67, -3.11, -0.22, 0.111, False, 0.026),
    ('V4', 'transfer', 'ternary', -0.32, -2.07, +1.43, 0.781, False, 0.709),
    # --- V5 (cross Step 1b; within+transfer Step 1c) — q recomputed in 30-cell family ---
    ('V5', 'cross',    'binary',  -2.77, -4.92, -0.61, 0.105, False, 0.014),
    ('V5', 'cross',    'ternary', -1.17, -2.75, +0.40, 0.257, False, 0.137),
    ('V5', 'within',   'binary',  -2.92, -5.31, -0.52, 0.111, False, 0.020),
    ('V5', 'within',   'ternary', -2.02, -4.30, +0.25, 0.230, False, 0.078),
    ('V5', 'transfer', 'binary',  -1.22, -2.63, +0.19, 0.230, False, 0.086),
    ('V5', 'transfer', 'ternary', -1.47, -3.00, +0.06, 0.197, False, 0.059),
]

# Stouffer aggregates: 6 paradigm-level (5V each, Step 1d) + 1 legacy full DAPT (v3.1 continuity)
STOUFFER_AGG = {
    # --- 6 paradigm-level aggregates (5V each, computed in scripts/internal/recompute_v1v2v3_transfer.py) ---
    'cross_binary':     {'n': 5,  'Z': -5.328, 'p': '<0.001'},
    'cross_ternary':    {'n': 5,  'Z': +0.577, 'p': '0.564'},
    'within_binary':    {'n': 5,  'Z': -4.419, 'p': '<0.0001'},
    'within_ternary':   {'n': 5,  'Z': -2.159, 'p': '0.031'},
    # --- transfer aggregates UPGRADED to 5V (Step 1d, n=30 family); v3.1 had n=2 V4/V5-only ---
    # transfer_binary: V4/V5-only Z=-2.788 → V1-V5 Z=-3.391 (strengthened)
    # transfer_ternary: V4/V5-only Z=-1.597 → V1-V5 Z=+0.176 (direction flipped: V1/V2/V3 ternary positive)
    'transfer_binary':  {'n': 5,  'Z': -3.391, 'p': '0.0007'},
    'transfer_ternary': {'n': 5,  'Z': +0.176, 'p': '0.860'},
    # --- v3.1 legacy 16-cell full DAPT aggregate (preserved as historical reference) ---
    'full_dapt':        {'n': 16, 'Z': -4.830, 'p': '<0.001'},
}


def _data_quality_label(subject_id: str) -> str:
    """Map subject ID -> data quality bin (per §2.9 Table 5)."""
    quality = {
        # 干净 (clean)
        'S01': 'clean', 'S02': 'clean', 'S06': 'clean', 'S07': 'clean',
        'S08': 'clean', 'S11': 'clean', 'S13': 'clean', 'S15': 'clean',
        'S17': 'clean', 'S18': 'clean',
        # 信息性 (informative high-variance)
        'S12': 'informative', 'S19': 'informative', 'S20': 'informative',
        # 轻度 (mild artifact)
        'S03': 'mild', 'S05': 'mild', 'S09': 'mild', 'S16': 'mild', 'S21': 'mild',
        # 重度 (heavy artifact)
        'S04': 'heavy', 'S10': 'heavy', 'S14': 'heavy',
    }
    return quality.get(subject_id, 'unknown')


# -----------------------------------------------------------------------------
# NEW-B / T3.* — Figure: §3.6 V1-V5 DAPT Forest Plot
# -----------------------------------------------------------------------------

def generate_dapt_v1_v5_forest_figure():
    """NEW-B — V1-V5 DAPT 30-cell forest plot (post Step 1d V1/V2/V3 transfer补完).

    Visualizes mean Δ (pp) with 95% CI for each of the 30 V × paradigm × task
    cells (V1-V5 × within+cross+transfer × bin+ter = 30; DAPT 评估全闭合).
    BH-FDR survivors highlighted; 7 Stouffer aggregates (6 paradigm-level 5V
    + 1 legacy v3.1 16-cell full-DAPT) appear as diamonds at the bottom.

    Data sources:
      - V1-V3 within+cross: `paper/reviews/stage4_step1b_stat_recompute_v4v5.md`
      - V4/V5 within+transfer: `paper/reviews/stage4_step1c_v4v5_within_transfer.md`
      - V1/V2/V3 transfer (6 cells, 2026-05-11 补完):
        `paper/reviews/stage4_step1d_v1v2v3_transfer.md`
        (`scripts/internal/recompute_v1v2v3_transfer.py`)
      - BH-FDR q-values RECOMPUTED over the new 30-cell family (q for the
        original 24 cells differs from Step 1c values; 0/30 survive q<0.05).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    rows = list(DAPT_V_RESULTS_STEP1B)
    # Top-to-bottom order: group by V, then per-V:
    # within-bin / within-ter / cross-bin / cross-ter / transfer-bin / transfer-ter
    # (all 5 V now have all 3 paradigms after Step 1d V1/V2/V3 transfer 补完)
    order_map = {('within',   'binary'):  0, ('within',   'ternary'): 1,
                 ('cross',    'binary'):  2, ('cross',    'ternary'): 3,
                 ('transfer', 'binary'):  4, ('transfer', 'ternary'): 5}
    rows = sorted(rows, key=lambda r: (r[0], order_map[(r[1], r[2])]))

    n = len(rows)
    # figsize scales with row count: original (11, 9.5) was for 16 cells + 3
    # diamonds = 19 row-equivalents; new 30 + 7 = 37 ⇒ ~1.95× height.
    fig, ax = plt.subplots(figsize=(11, 18.5))

    y_positions = np.arange(n)[::-1]  # top-to-bottom

    labels = []
    for (V, paradigm, task, _, _, _, _, _, _), y in zip(rows, y_positions):
        # task abbreviation
        t_abbr = 'Bin' if task == 'binary' else 'Ter'
        p_abbr = {'within': 'Within', 'cross': 'Cross', 'transfer': 'Transfer'}[paradigm]
        labels.append(f'{V}  {p_abbr}-{t_abbr}')

    for (V, paradigm, task, mean_d, lo, hi, q, sig, p_raw), y in zip(rows, y_positions):
        color = '#d62728' if sig else '#888888'  # BH sig = red, else gray
        # Direction-of-effect coloring: positive Δ greenish if not sig
        if not sig and mean_d > 0:
            color = '#4caf50'
        ax.errorbar(
            mean_d, y,
            xerr=[[mean_d - lo], [hi - mean_d]],
            fmt='o', markersize=10 if sig else 7,
            color=color, ecolor=color,
            elinewidth=2 if sig else 1.2,
            capsize=4,
            markeredgecolor='black' if sig else 'none',
            markeredgewidth=1.0 if sig else 0,
            zorder=4,
        )
        # Annotate
        sig_marker = ' *' if sig else ''
        ax.text(
            hi + 0.25, y,
            f'  Δ={mean_d:+.2f} pp  q={q:.3f}{sig_marker}',
            va='center', fontsize=9,
            color='black', fontweight='bold' if sig else 'normal',
        )

    # Vertical zero line
    ax.axvline(0, color='black', linestyle='--', alpha=0.6, linewidth=1.0, zorder=2)

    # Stouffer aggregates as diamonds below the cells (7 total: 6 paradigm-level
    # 5V each from Step 1d + 1 legacy v3.1 16-cell full-DAPT). Spaced 1.0 unit apart.
    diamond_y = [-2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]
    diamond_data = [
        # 6 paradigm-level aggregates (5V each, Step 1d)
        ('Stouffer cross-bin (V1-V5, n=5)',           -1.788, STOUFFER_AGG['cross_binary']),
        ('Stouffer cross-ter (V1-V5, n=5)',           +0.180, STOUFFER_AGG['cross_ternary']),
        ('Stouffer within-bin (V1-V5, n=5)',          -1.894, STOUFFER_AGG['within_binary']),
        ('Stouffer within-ter (V1-V5, n=5)',          -0.918, STOUFFER_AGG['within_ternary']),
        # Step 1d upgrade: transfer aggregates V4/V5-only n=2 → V1-V5 n=5
        ('Stouffer transfer-bin (V1-V5, n=5) ★1d',    -1.149, STOUFFER_AGG['transfer_binary']),
        ('Stouffer transfer-ter (V1-V5, n=5) ★1d',    +0.027, STOUFFER_AGG['transfer_ternary']),
        # Legacy: v3.1 full-DAPT 16-cell aggregate (preserved for historical continuity)
        ('Stouffer full DAPT family v3.1 (n=16)',     None,   STOUFFER_AGG['full_dapt']),
    ]
    for (label, mean_or_none, agg), y in zip(diamond_data, diamond_y):
        # Diamond at the (signed) Stouffer Z-projected position (we reuse mean
        # for cross-bin / cross-ter; full has no scalar mean → place at 0)
        x_center = mean_or_none if mean_or_none is not None else 0.0
        diamond = Polygon([
            (x_center - 0.6, y),
            (x_center, y + 0.35),
            (x_center + 0.6, y),
            (x_center, y - 0.35),
        ], closed=True, facecolor='#1f77b4', edgecolor='black', linewidth=1.0,
            alpha=0.9, zorder=4)
        ax.add_patch(diamond)
        ax.text(
            x_center + 1.0, y,
            f'  {label}: Z={agg["Z"]:+.2f}, p={agg["p"]}'
            + (f'  (mean Δ={mean_or_none:+.2f} pp)' if mean_or_none is not None else ''),
            va='center', fontsize=9,
        )

    # y-axis tick labels — 30 cell labels + 7 aggregate placeholders
    ax.set_yticks(list(y_positions) + diamond_y)
    ax.set_yticklabels(list(labels) + ['(aggregate)'] * len(diamond_y),
                       fontsize=9)

    ax.set_xlabel('Mean Δ (DAPT − Baseline, pp)  [95% CI]', fontsize=12)
    ax.set_title('DAPT V1-V5 30-cell forest plot (post Step 1d full closure)\n'
                 '(red filled = BH-FDR q<0.05 within 30-cell DAPT family - 0/30 survive; '
                 'green = directionally positive; '
                 'diamonds = Stouffer aggregate; '
                 'rows V1/V2/V3-Transfer-{Bin,Ter} added 2026-05-11)',
                 fontsize=11)
    ax.set_xlim(-6.5, 7.0)
    ax.set_ylim(-8.7, n - 0.3)
    ax.grid(axis='x', alpha=0.3)
    # Legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker='o', color='w', label='BH-FDR q<0.05 (negative)',
               markerfacecolor='#d62728', markeredgecolor='black', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Directionally positive (n.s.)',
               markerfacecolor='#4caf50', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Other (n.s.)',
               markerfacecolor='#888888', markersize=8),
    ]
    ax.legend(handles=legend_elems, loc='lower right', fontsize=9)

    fig.tight_layout()
    add_provenance_footer(fig, 'dapt_v1_v5_forest')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / 'dapt_v1_v5_forest.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Saved: {out_path}')


# -----------------------------------------------------------------------------
# NEW-A — §3.7 Exploratory Ablation Overview
# -----------------------------------------------------------------------------

def generate_exploratory_ablation_overview_figure():
    """NEW-A — §3.7 capacity x architecture x pretraining overview.

    x-axis: parameter count (log scale)
    y-axis: cross-subject binary accuracy (%)
    Series:
      (i) EEGNet capacity ladder: 16K, 1.90M, 5.84M, 19.99M, 30.22M
      (ii) random-init CBraMod (~30.48M, single point)
      (iii) TUEG-pretrained CBraMod (~30.48M, single point)

    Caveat box: "exploratory observation under shared HP / restricted HPO
    budget" (per §3.7 chapter intro).
    """
    import matplotlib.pyplot as plt

    # Anchors from §3.7.1 / §3.7.2 / §3.7.3 (Step 2b verified)
    eegnet_ladder = [
        ('EEGNet baseline', 16162,    76.67),
        ('EEGNet-Mid',      1.90e6,   57.65),
        ('EEGNet-Huge v3',  5.84e6,   51.37),
        ('EEGNet-Huge v1',  19.99e6,  50.00),
        ('EEGNet-Huge v2',  30.22e6,  50.07),
    ]
    cbramod_random = ('CBraMod random-init', 30.48e6, 86.34)
    cbramod_pretrained = ('CBraMod baseline (TUEG)', 30.48e6, 90.68)

    fig, ax = plt.subplots(figsize=(11, 7))

    # EEGNet ladder series (solid line)
    eg_x = [p for _, p, _ in eegnet_ladder]
    eg_y = [y for _, _, y in eegnet_ladder]
    ax.plot(eg_x, eg_y, marker='s', markersize=11, linewidth=2.4,
            color='#1976D2', label='EEGNet capacity ladder', zorder=3)
    # Annotate each EEGNet point
    annot_offsets = [(8, 12), (8, 12), (8, 12), (-15, -22), (8, -16)]
    for (name, p, y), ofs in zip(eegnet_ladder, annot_offsets):
        ax.annotate(f'{name}\n{y:.2f}%', (p, y),
                    textcoords='offset points', xytext=ofs,
                    fontsize=8.5, color='#1976D2', fontweight='bold')

    # CBraMod random-init (single point)
    rx, ry = cbramod_random[1], cbramod_random[2]
    ax.scatter([rx], [ry], marker='D', s=210, color='#FF9800',
               edgecolor='black', linewidth=1.5,
               label=f'CBraMod random-init (no TUEG)', zorder=4)
    ax.annotate(f'random-init\n{ry:.2f}%', (rx, ry),
                textcoords='offset points', xytext=(-90, 12),
                fontsize=9, color='#FF9800', fontweight='bold')

    # CBraMod pretrained (single point)
    px, py = cbramod_pretrained[1], cbramod_pretrained[2]
    ax.scatter([px], [py], marker='*', s=440, color='#D32F2F',
               edgecolor='black', linewidth=1.5,
               label='CBraMod TUEG-pretrained (baseline)', zorder=5)
    ax.annotate(f'TUEG-pretrained\n{py:.2f}%', (px, py),
                textcoords='offset points', xytext=(-110, -28),
                fontsize=9, color='#D32F2F', fontweight='bold')

    # Decomposition arrows: EEGNet-Huge v3 → random-init CBraMod
    # (this is the +34.97 pp "architecture + free random-init HP" gap)
    eg_v3_p, eg_v3_y = eegnet_ladder[2][1], eegnet_ladder[2][2]
    ax.annotate(
        '', xy=(rx, ry), xytext=(eg_v3_p, eg_v3_y),
        arrowprops=dict(arrowstyle='->', color='#FF9800', lw=2,
                        connectionstyle='arc3,rad=0.18'),
    )
    ax.text(8e6, 70, 'Architecture + free HP\n(EEGNet-Huge v3 → random-init CBraMod)\nΔ ≈ +34.97 pp',
            fontsize=8.5, color='#FF9800', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.85))

    # Random-init → TUEG-pretrained arrow (+4.34 pp, same params, shared HP)
    ax.annotate(
        '', xy=(px, py), xytext=(rx, ry),
        arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=2,
                        connectionstyle='arc3,rad=0.0'),
    )
    ax.text(2.5e7, 88.5, 'TUEG pretraining\nΔ = +4.34 pp\n(shared HP)',
            fontsize=8.5, color='#D32F2F', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff5f5', alpha=0.9))

    # EEGNet baseline → EEGNet-Huge v3 arrow (-25.30 pp; capacity ladder collapse)
    ax.annotate(
        '', xy=(eg_v3_p, eg_v3_y), xytext=(eegnet_ladder[0][1], eegnet_ladder[0][2]),
        arrowprops=dict(arrowstyle='->', color='#1976D2', lw=2,
                        connectionstyle='arc3,rad=-0.2'),
    )
    ax.text(2e5, 60, 'EEGNet capacity ladder\nΔ ≈ −25.30 pp\n(baseline → Huge v3)',
            fontsize=8.5, color='#1976D2', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#e3f2fd', alpha=0.9))

    # Reference lines
    ax.axhline(50, color=PAPER_COLORS['chance_red'], linestyle='--',
               alpha=0.85, linewidth=1.0)
    ax.text(1.5e4, 51, 'Chance (50%)', fontsize=8, color=PAPER_COLORS['chance_red'])

    # Caveat box (top right)
    caveat = (
        'Caveat (§3.7 intro): all Δ shown are observed under\n'
        'shared default HP and restricted HPO budget (≤2 trial\n'
        'manual debug for EEGNet-Huge; CBraMod random-init reuses\n'
        'baseline HP). Strict independent HPO (≥25-trial Optuna)\n'
        'left to future work (§6 #8). Treat as exploratory observation.'
    )
    ax.text(0.98, 0.02, caveat, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fffde7',
                      edgecolor='#fbc02d', linewidth=1.2, alpha=0.92))

    ax.set_xscale('log')
    ax.set_xlim(1e4, 1e8)
    ax.set_ylim(45, 95)
    ax.set_xlabel('Parameter count (log scale)', fontsize=12)
    ax.set_ylabel('Cross-subject binary accuracy (%)', fontsize=12)
    ax.set_title('§3.7 Exploratory Ablation Overview: Architecture × Pretraining × Capacity\n'
                 '(N = 21, 128ch binary, exploratory observation only)',
                 fontsize=12)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(loc='upper left', fontsize=9)

    fig.tight_layout()
    add_provenance_footer(fig, 'exploratory_ablation_overview')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / 'exploratory_ablation_overview.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Saved: {out_path}')


# -----------------------------------------------------------------------------
# T3.8 — Updated Fig 10 Further-pretraining + reverse-gradient panel
# -----------------------------------------------------------------------------

def generate_further_pretraining_v3_figure():
    """T3.8 — UPDATED §3.6 figure with V1-V5 + reverse-gradient panel.

    Panel A: 4-bar chart of mean Δ (pp) for each (paradigm × task) condition,
    with separate bars per V (V1-V5 where available). Negative bars red,
    positive bars green; BH-FDR significant cells get a black border.

    Panel B: (sample_size, Δ) reverse-gradient scatter showing that V1-V5
    Δ becomes more negative as effective training sample shrinks (within-
    subject ~80×21 = 1.7K vs cross ~33K).

    Replaces existing `paper/figures/further_pretraining.png`.
    """
    import matplotlib.pyplot as plt

    # Group Step 1b values by (paradigm, task), then by V
    grouped = {}  # (paradigm, task) -> list of (V, mean_d, sig)
    for (V, paradigm, task, mean_d, _, _, _, sig, _) in DAPT_V_RESULTS_STEP1B:
        grouped.setdefault((paradigm, task), []).append((V, mean_d, sig))

    # 6 conditions (4 v3.1 + 2 transfer added 2026-05-10 from Step 1c).
    # V1/V2/V3 lack transfer data → bars NaN-skipped, only V4/V5 visible there.
    conditions = [
        ('within',   'binary',  'Within\nBinary'),
        ('within',   'ternary', 'Within\nTernary'),
        ('cross',    'binary',  'Cross\nBinary'),
        ('cross',    'ternary', 'Cross\nTernary'),
        ('transfer', 'binary',  'Transfer\nBinary'),
        ('transfer', 'ternary', 'Transfer\nTernary'),
    ]

    v_colors = {
        'V1': PAPER_COLORS['secondary_blue'],
        'V2': '#FF9800',
        'V3': '#4CAF50',
        'V4': '#9C27B0',
        'V5': '#F44336',
    }

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(17, 6.2),
                                   gridspec_kw={'width_ratios': [2.0, 1]})

    # Panel A: grouped bars
    n_v = 5
    width = 0.16
    x = np.arange(len(conditions))
    for vi, V in enumerate(['V1', 'V2', 'V3', 'V4', 'V5']):
        vals = []
        sigs = []
        for paradigm, task, _ in conditions:
            entry = next((e for e in grouped.get((paradigm, task), [])
                          if e[0] == V), None)
            if entry is None:
                vals.append(np.nan)
                sigs.append(False)
            else:
                vals.append(entry[1])
                sigs.append(entry[2])
        offsets = (vi - (n_v - 1) / 2) * width
        bars = axA.bar(x + offsets, vals, width,
                       color=v_colors[V], edgecolor='black',
                       label=V, alpha=0.92)
        # Outline BH-significant
        for bar, sig in zip(bars, sigs):
            if sig:
                bar.set_edgecolor('black')
                bar.set_linewidth(2.5)

    axA.axhline(0, color='black', linewidth=0.8)
    axA.set_xticks(x)
    axA.set_xticklabels([c[2] for c in conditions], fontsize=FONT_SIZES['tick'])
    axA.set_ylabel('Δ (DAPT − Baseline, pp)', fontsize=FONT_SIZES['axis_label'])
    axA.set_title('A. DAPT V1-V5 effect by paradigm × task (30-cell post Step 1d full closure)\n(thick border = BH-FDR q<0.05 within 30-cell DAPT family - 0/30 survive; V1/V2/V3 transfer added 2026-05-11)',
                  fontsize=FONT_SIZES['title'])
    axA.legend(loc='lower right', ncol=5, fontsize=FONT_SIZES['legend'], title='Pretrain config', title_fontsize=FONT_SIZES['legend'])
    axA.grid(axis='y', alpha=0.3)
    axA.set_ylim(-3.5, 1.5)

    # Annotate Stouffer summary in panel A — 6 paradigm-level 5V + 1 legacy.
    summary = (
        'Stouffer aggregates (Step 1d, 5V each):\n'
        'cross-bin (n=5): Z=−5.33, p<0.001\n'
        'cross-ter (n=5): Z=+0.58, p=0.564\n'
        'within-bin (n=5): Z=−4.42, p<0.0001\n'
        'within-ter (n=5): Z=−2.16, p=0.031\n'
        'transfer-bin (n=5) ★1d: Z=−3.39, p=0.0007\n'
        'transfer-ter (n=5) ★1d: Z=+0.18, p=0.860\n'
        '— legacy (v3.1) —\n'
        'full DAPT family (n=16): Z=−4.83, p<0.001'
    )
    axA.text(0.02, 0.02, summary, transform=axA.transAxes,
             fontsize=FONT_SIZES['annotation'], va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#fffde7',
                       edgecolor='#fbc02d', alpha=0.9))

    # Panel B: reverse-gradient scatter — all 3 paradigms.
    # Effective training sample sizes per condition (per-subject ~80 for binary,
    # ~120 for ternary). Transfer = XSI-FT: cross-pretrained init then per-subject
    # fine-tune, so the *discriminative phase* sample size matches within
    # (per-subject); cross init helps mitigate but doesn't change the sample
    # axis. Transfer markers use distinguishing star/diamond markers to signal
    # the cross-init enhancement.
    sample_size = {
        ('within',   'binary'):  80 * 1,        # per-subject training size
        ('within',   'ternary'): 120 * 1,
        ('cross',    'binary'):  80 * 21,       # pooled across 21 subjects
        ('cross',    'ternary'): 120 * 21,
        ('transfer', 'binary'):  80 * 1,        # XSI-FT: per-subject fine-tune;
        ('transfer', 'ternary'): 120 * 1,       # cross-init aids mitigation only
    }
    paradigm_marker = {
        ('within',   'binary'):  'o',
        ('within',   'ternary'): '^',
        ('cross',    'binary'):  'o',
        ('cross',    'ternary'): '^',
        ('transfer', 'binary'):  '*',           # star = transfer-binary
        ('transfer', 'ternary'): 'D',           # diamond = transfer-ternary
    }

    for V in ['V1', 'V2', 'V3', 'V4', 'V5']:
        for paradigm, task, _ in conditions:
            entry = next((e for e in grouped.get((paradigm, task), [])
                          if e[0] == V), None)
            if entry is None:
                continue
            ss = sample_size[(paradigm, task)]
            mean_d = entry[1]
            sig = entry[2]
            marker = paradigm_marker[(paradigm, task)]
            # Slight x-jitter so within (per-subject) and transfer (per-subject)
            # markers don't completely overlap at log10(80) and log10(120).
            ss_jitter = ss * (1.18 if paradigm == 'transfer' else 1.0)
            axB.scatter(ss_jitter, mean_d,
                        s=170 if sig else (110 if paradigm == 'transfer' else 70),
                        marker=marker,
                        color=v_colors[V], edgecolor='black' if sig else 'gray',
                        linewidth=1.5 if sig else 0.6, alpha=0.85)

    # Regression line: all 30 cells now (within + cross + transfer, all 5 V × 6 paradigm-task)
    ss_all = []
    md_all = []
    for V in ['V1', 'V2', 'V3', 'V4', 'V5']:
        for paradigm, task, _ in conditions:
            entry = next((e for e in grouped.get((paradigm, task), [])
                          if e[0] == V), None)
            if entry is None:
                continue
            ss_all.append(np.log10(sample_size[(paradigm, task)]))
            md_all.append(entry[1])

    if len(ss_all) > 2:
        slope, intercept = np.polyfit(ss_all, md_all, 1)
        x_fit = np.linspace(min(ss_all), max(ss_all), 50)
        y_fit = slope * x_fit + intercept
        axB.plot(10**x_fit, y_fit, color='gray', linestyle='--', linewidth=1.5,
                 label=f'OLS slope={slope:+.2f} pp/log10(N)')

    axB.axhline(0, color='black', linewidth=0.8)
    axB.set_xscale('log')
    axB.set_xlabel('Effective training sample size (trials, log scale)', fontsize=FONT_SIZES['axis_label'])
    axB.set_ylabel('Δ (DAPT − Baseline, pp)', fontsize=FONT_SIZES['axis_label'])
    axB.set_title('B. Reverse-gradient scatter — all 3 paradigms\n'
                  '(circle/▲ = within/cross; ★/♦ = transfer; thick border = BH-FDR sig; '
                  'transfer markers x-jittered for visibility)',
                  fontsize=FONT_SIZES['title'])
    axB.grid(True, alpha=0.3)
    axB.legend(loc='lower right', fontsize=FONT_SIZES['legend'])

    # V color legend in panel B
    from matplotlib.lines import Line2D
    legend_v = [Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=v_colors[V],
                       markeredgecolor='black', markersize=8, label=V)
                for V in ['V1', 'V2', 'V3', 'V4', 'V5']]
    legB2 = axB.legend(handles=legend_v, loc='upper right',
                       fontsize=FONT_SIZES['legend'], title='V', title_fontsize=FONT_SIZES['legend'], ncol=2)
    axB.add_artist(legB2)
    # Re-add OLS legend (was overwritten)
    axB.legend(loc='lower right', fontsize=FONT_SIZES['legend'])

    fig.tight_layout()
    apply_paper_style(fig=fig)
    add_provenance_footer(fig, 'further_pretraining')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / 'further_pretraining.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Saved: {out_path}')


# -----------------------------------------------------------------------------
# T3.4 — Fig 9 Extra Sessions Three-Paradigm Overview + per-subject + Δ swarm
# -----------------------------------------------------------------------------

def _legacy_unused_generate_extra_sessions_paradigm_v2_figure():
    """[LEGACY / UNUSED — superseded by generate_extra_sessions_paradigm_figure v1, kept for reference only]

    T3.4 — UPDATED extra-sessions three-paradigm overview.

    Replaces the existing `paper/figures/extra_sessions_paradigm_binary.png`.
    Adds:
      Panel A (existing): mean ± SD trajectory across 4 steps, 3 paradigms
      Panel B (NEW): per-subject baseline → +Sess05 trajectories (within only)
                     colored by baseline acc bin
      Panel C (NEW): Δ(BL → +Sess05) swarm/scatter colored by baseline acc
    """
    import matplotlib.pyplot as plt

    step_order = ['baseline', 'sess03', 'sess04', 'sess05']
    step_labels = ['Baseline', '+Sess03', '+Sess04', '+Sess05']
    x = np.arange(len(step_order))

    configs = [
        {
            'label': 'Within-Subject',
            'path': get_run_path('extra_sessions_binary'),
            'loader': extract_extra_session_step_accs,
            'color': '#1976D2',
            'marker': 'o',
        },
        {
            'label': 'Cross-Subject (21-subj train)',
            'path': get_run_path('extra_sessions_cross_binary'),
            'loader': extract_cross_subject_extra_session_step_accs,
            'color': '#EF6C00',
            'marker': 's',
        },
        {
            'label': 'Transfer-Init',
            'path': get_run_path('extra_sessions_transfer_binary'),
            'loader': extract_extra_session_step_accs,
            'color': '#2E7D32',
            'marker': 'D',
        },
    ]

    series = []
    per_subject_data = {}  # paradigm label -> dict of subject -> (baseline_acc, +Sess05_acc)
    for cfg in configs:
        path = resolve_project_path(cfg['path'])
        if not path.exists():
            logger.warning(f'Missing: {cfg["path"]}')
            continue
        cache = load_json_cache(cfg['path'])
        step_accs = cfg['loader'](cache, 'cbramod')

        # Per-subject for within only (loader returns aggregated lists, so
        # we re-extract from the cache to get subject mapping).
        subj_map = {}
        if cfg['label'] == 'Within-Subject':
            results = cache.get('results', {}).get('cbramod', {})
            for sid, sdata in sorted(results.items()):
                if not isinstance(sdata, dict) or not sid.startswith('S'):
                    continue
                bl = sdata.get('baseline', {}).get('test_acc_majority')
                s5 = sdata.get('sess05', {}).get('test_acc_majority')
                if bl is not None and s5 is not None:
                    subj_map[sid] = (bl * 100, s5 * 100)
        per_subject_data[cfg['label']] = subj_map

        means = np.array([np.mean(step_accs[s]) if step_accs.get(s) else np.nan
                          for s in step_order])
        sds = np.array([np.std(step_accs[s]) if step_accs.get(s) else 0.0
                        for s in step_order])
        if np.isnan(means).all():
            continue
        series.append({**cfg, 'means': means, 'sds': sds,
                       'delta': means[-1] - means[0]})

    fig = plt.figure(figsize=(16, 6.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.0, 1.1], wspace=0.32)
    ax_line = fig.add_subplot(gs[0])
    ax_subj = fig.add_subplot(gs[1])
    ax_swarm = fig.add_subplot(gs[2])

    # Panel A: mean ± SD trajectories
    for item in series:
        ax_line.plot(x, item['means'], color=item['color'],
                     marker=item['marker'], linewidth=2.4, markersize=8,
                     label=item['label'])
        ax_line.fill_between(x, item['means'] - item['sds'],
                             item['means'] + item['sds'],
                             color=item['color'], alpha=0.13)
        ax_line.annotate(f'{item["means"][-1]:.2f}%',
                         xy=(x[-1], item['means'][-1]),
                         xytext=(8, 0), textcoords='offset points',
                         fontsize=9, color=item['color'], fontweight='bold')

    ax_line.set_xticks(x)
    ax_line.set_xticklabels(step_labels)
    ax_line.set_ylabel('Mean accuracy ± SD (%)', fontsize=11)
    ax_line.set_title('A. Trajectory across 3 paradigms (CBraMod binary, N=16)',
                      fontsize=12)
    ax_line.grid(True, alpha=0.25)
    ax_line.legend(loc='lower right', fontsize=9)

    # Panel B: per-subject within-subject trajectories
    within_subj = per_subject_data.get('Within-Subject', {})
    if within_subj:
        bl_values = [v[0] for v in within_subj.values()]
        bl_low_thresh = 80.0
        for sid, (bl, s5) in sorted(within_subj.items()):
            color = '#d62728' if bl < bl_low_thresh else (
                '#2ca02c' if bl > 90 else '#888888')
            ax_subj.plot([0, 1], [bl, s5], color=color, alpha=0.65,
                         linewidth=1.4, marker='o', markersize=5)
            ax_subj.text(1.05, s5, sid, fontsize=9, color=color,
                         va='center')
        ax_subj.set_xticks([0, 1])
        ax_subj.set_xticklabels(['Baseline', '+Sess05'], fontsize=10)
        ax_subj.set_ylabel('Within-subject accuracy (%)', fontsize=11)
        ax_subj.set_title('B. Per-subject within trajectory\n(red=low BL <80%, green=high >90%)',
                          fontsize=12)
        ax_subj.grid(True, alpha=0.25)
    else:
        ax_subj.text(0.5, 0.5, 'No per-subject data', ha='center', va='center',
                     transform=ax_subj.transAxes)

    # Panel C: Δ swarm colored by baseline
    if within_subj:
        bls = []
        deltas = []
        for sid, (bl, s5) in within_subj.items():
            bls.append(bl)
            deltas.append(s5 - bl)
        # Color by baseline (low=red, high=blue)
        cmap = plt.cm.coolwarm_r
        norm = plt.Normalize(vmin=min(bls), vmax=max(bls))
        scatter = ax_swarm.scatter(bls, deltas, c=bls, cmap=cmap, norm=norm,
                                   s=70, edgecolor='black', linewidth=0.7)
        cbar = fig.colorbar(scatter, ax=ax_swarm, pad=0.02)
        cbar.set_label('Baseline accuracy (%)', fontsize=9)
        ax_swarm.axhline(0, color='gray', linestyle='--', alpha=0.6)
        ax_swarm.set_xlabel('Baseline accuracy (%)', fontsize=11)
        ax_swarm.set_ylabel('Δ (BL → +Sess05, pp)', fontsize=11)
        ax_swarm.set_title('C. Δ vs baseline (within-subject)\n(low baselines benefit most)',
                           fontsize=12)
        ax_swarm.grid(True, alpha=0.3)

    fig.tight_layout()
    add_provenance_footer(fig, 'extra_sessions_paradigm')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / 'extra_sessions_paradigm_binary.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Saved: {out_path}')


# -----------------------------------------------------------------------------
# T3.6 — Channel Scaling 3-panel split
# -----------------------------------------------------------------------------

def generate_channel_scaling_v2_figure():
    """T3.6 — Channel Scaling split into 3 sub-panels.

    Replaces existing `paper/figures/channel_scaling_curve.png`.

    Panel A (left, large): main 128 → 4ch envelope (existing chart, condensed)
    Panel B (top right): 8ch focus (4 methods bar chart)
    Panel C (bottom right): 4ch control (FDR/Attention/CSP/Band Power top-4
                           + neg control + FDR∩Att outlier)
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 8.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1.0], height_ratios=[1, 1],
                          hspace=0.32, wspace=0.25)
    axA = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 1])

    def _load_acc(path):
        if not resolve_project_path(path).exists():
            logger.warning(f'Missing: {path}')
            return None, None
        cache = load_json_cache(path)
        accs = extract_model_accs(cache, 'cbramod')
        if not accs:
            return None, None
        return float(np.mean(accs)), float(np.std(accs))

    # --- Panel A: envelope across 128 / 64 / 61 / 32 / 8 / 4 ---
    method_paths = {
        'FDR': [
            (32, get_run_path('reduced_32_fdr_binary')),
            (8,  get_run_path('reduced_8_fdr_binary')),
            (4,  get_run_path('reduced_4_fdr_binary')),
        ],
        'Band Power': [
            (32, get_run_path('reduced_32_band_power_binary')),
            (8,  get_run_path('reduced_8_band_power_binary')),
            (4,  get_run_path('reduced_4_band_power_binary')),
        ],
        'Attention': [
            (32, get_run_path('reduced_32_attention_binary')),
            (8,  get_run_path('reduced_8_attention_binary')),
            (4,  get_run_path('reduced_4_attention_binary')),
        ],
        'CSP': [
            (32, get_run_path('reduced_32_csp_binary')),
            (8,  get_run_path('reduced_8_csp_binary')),
            (4,  get_run_path('reduced_4_csp_binary')),
        ],
    }
    method_style = {
        'FDR':        {'color': PAPER_COLORS['secondary_blue'], 'marker': 's'},
        'Band Power': {'color': '#FF9800', 'marker': '^'},
        'Attention':  {'color': '#4CAF50', 'marker': 'v'},
        'CSP':        {'color': '#9C27B0', 'marker': 'D'},
    }
    method_data = {}
    for method, entries in method_paths.items():
        method_data[method] = {}
        for n_ch, path in entries:
            mean, std = _load_acc(path)
            if mean is not None:
                method_data[method][n_ch] = (mean, std)

    baseline_paths = [
        (128, get_run_path('cross_cbramod_binary')),
        (64,  get_run_path('reduced_64_fdr_binary')),
        (61,  get_run_path('standard_1010_61_cross_binary')),
    ]
    baseline_data = {}
    for n_ch, path in baseline_paths:
        mean, std = _load_acc(path)
        if mean is not None:
            baseline_data[n_ch] = (mean, std)

    # Best envelope
    all_chs = sorted(set(list(baseline_data.keys()) +
                         [c for md in method_data.values() for c in md]),
                     reverse=True)
    best_chs, best_means, best_stds = [], [], []
    for n_ch in all_chs:
        if n_ch in baseline_data:
            best_chs.append(n_ch)
            best_means.append(baseline_data[n_ch][0])
            best_stds.append(baseline_data[n_ch][1])
        else:
            best_v = -1
            best_s = 0
            for m, d in method_data.items():
                if n_ch in d and d[n_ch][0] > best_v:
                    best_v = d[n_ch][0]
                    best_s = d[n_ch][1]
            if best_v > 0:
                best_chs.append(n_ch)
                best_means.append(best_v)
                best_stds.append(best_s)

    for method, data in method_data.items():
        if not data:
            continue
        s = method_style[method]
        chs = sorted(data.keys(), reverse=True)
        ms = [data[c][0] for c in chs]
        ss = [data[c][1] for c in chs]
        axA.errorbar(chs, ms, yerr=ss, linestyle=':', linewidth=1.5,
                     marker=s['marker'], markersize=7, color=s['color'],
                     capsize=3, alpha=0.7, label=method, zorder=2)
    axA.errorbar(best_chs, best_means, yerr=best_stds, marker='o',
                 markersize=11, linewidth=2.6, color='red', capsize=5,
                 zorder=4, label='Best envelope')
    for c, m in zip(best_chs, best_means):
        axA.annotate(f'{c}ch\n{m:.1f}%', (c, m),
                     textcoords='offset points', xytext=(10, 10),
                     fontsize=FONT_SIZES['annotation'], color='red', fontweight='bold')

    axA.axhline(50, color=PAPER_COLORS['chance_red'], linestyle='--',
                alpha=0.85, linewidth=1.0, label='Chance (50%)')
    axA.axvspan(28, 36, alpha=0.08, color='green', label='32ch deploy zone')
    axA.set_xscale('log', base=2)
    axA.set_xticks(best_chs)
    axA.set_xticklabels([str(c) for c in best_chs])
    axA.set_xlabel('Number of channels', fontsize=FONT_SIZES['axis_label'])
    axA.set_ylabel('CBraMod cross-subject binary accuracy (%)', fontsize=FONT_SIZES['axis_label'])
    axA.set_title('A. 128 → 4ch envelope & per-method tracking', fontsize=FONT_SIZES['title'])
    axA.set_ylim(48, 100)
    axA.grid(True, alpha=0.3)
    axA.legend(loc='lower right', fontsize=FONT_SIZES['legend'], ncol=2)

    # --- Panel B: 8ch focus ---
    methods_8 = ['FDR', 'Band Power', 'Attention', 'CSP']
    means_8 = []
    stds_8 = []
    for m in methods_8:
        if 8 in method_data.get(m, {}):
            means_8.append(method_data[m][8][0])
            stds_8.append(method_data[m][8][1])
        else:
            means_8.append(0)
            stds_8.append(0)
    bx = np.arange(len(methods_8))
    bars = axB.bar(bx, means_8, yerr=stds_8, capsize=4,
                   color=[method_style[m]['color'] for m in methods_8],
                   edgecolor='black')
    for b, m, v in zip(bars, methods_8, means_8):
        axB.text(b.get_x() + b.get_width()/2, v + 1.5,
                 f'{v:.1f}%', ha='center', va='bottom', fontsize=FONT_SIZES['annotation'],
                 fontweight='bold')
    axB.set_xticks(bx)
    axB.set_xticklabels(methods_8, fontsize=FONT_SIZES['tick'])
    axB.set_ylabel('CBraMod cross-bin acc (%)', fontsize=FONT_SIZES['axis_label'])
    axB.set_title('B. 8ch focus: 4-method ranking flip\n(BP & CSP overtake FDR)', fontsize=FONT_SIZES['title'])
    axB.set_ylim(60, 90)
    axB.grid(axis='y', alpha=0.3)

    # --- Panel C: 4ch control ---
    # FDR top-4, Attention top-4, neg ctrl, FDR∩Att outlier, BP top-4 (new), CSP top-4 (new)
    ctrl_paths = {
        'FDR top-4':       get_run_path('reduced_4_fdr_binary'),
        'Attention top-4': get_run_path('reduced_4_attention_binary'),
        'Band Power top-4':get_run_path('reduced_4_band_power_binary'),
        'CSP top-4':       get_run_path('reduced_4_csp_binary'),
        'Neg. control':    get_run_path('reduced_4_negative_control_binary'),
        'FDR ∩ Att\n(outlier)': get_run_path('reduced_4_fdr_attention_overlap_binary'),
    }
    ctrl_means, ctrl_stds = [], []
    ctrl_labels = []
    bar_colors = []
    color_map = {
        'FDR top-4': PAPER_COLORS['secondary_blue'],
        'Attention top-4': '#4CAF50',
        'Band Power top-4': '#FF9800',
        'CSP top-4': '#9C27B0',
        'Neg. control': '#888888',
        'FDR ∩ Att\n(outlier)': '#FFC107',
    }
    for name, path in ctrl_paths.items():
        mean, std = _load_acc(path)
        if mean is None:
            continue
        ctrl_means.append(mean)
        ctrl_stds.append(std)
        ctrl_labels.append(name)
        bar_colors.append(color_map.get(name, '#888888'))
    cx = np.arange(len(ctrl_labels))
    cbars = axC.bar(cx, ctrl_means, yerr=ctrl_stds, capsize=3,
                    color=bar_colors, edgecolor='black')
    for b, v in zip(cbars, ctrl_means):
        axC.text(b.get_x() + b.get_width()/2, v + 1.0,
                 f'{v:.1f}%', ha='center', va='bottom', fontsize=FONT_SIZES['annotation'],
                 fontweight='bold')
    axC.set_xticks(cx)
    axC.set_xticklabels(ctrl_labels, fontsize=FONT_SIZES['tick'], rotation=20, ha='right')
    axC.set_ylabel('CBraMod cross-bin acc (%)', fontsize=FONT_SIZES['axis_label'])
    axC.set_title('C. 4ch control: BP top-4 alone exceeds neg. control (+11.10 pp)',
                  fontsize=FONT_SIZES['title'])
    axC.axhline(50, color=PAPER_COLORS['chance_red'], linestyle='--',
                alpha=0.85, linewidth=1.0)
    axC.set_ylim(45, 90)
    axC.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    apply_paper_style(fig=fig)
    add_provenance_footer(fig, 'channel_scaling')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / 'channel_scaling_curve.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Saved: {out_path}')


# -----------------------------------------------------------------------------
# T3.2 — Sensitivity Scaling Curve (NEW)
# -----------------------------------------------------------------------------

def generate_sensitivity_scaling_figure():
    """T3.2 — Sensitivity scaling: spread vs absolute acc across channel counts.

    Dual y-axis chart:
      x: channel count (log: 4, 8, 32)
      left y: method spread (max − min, pp)
      right y: best-method absolute acc (%)
    """
    import matplotlib.pyplot as plt

    # From paper §3.5.3 table: 32ch / 8ch / 4ch spreads and best-method accs
    # Using values from paper §3.5 (Table 9 footer + §3.5.3 body)
    data = {
        32: {'spread': 2.77, 'best_method': 'FDR',         'best_acc': 87.71},
        8:  {'spread': 15.63,'best_method': 'Band Power',  'best_acc': 84.05},
        4:  {'spread': 24.05,'best_method': 'Band Power',  'best_acc': 78.75},
    }

    chs = sorted(data.keys())
    spreads = [data[c]['spread'] for c in chs]
    accs = [data[c]['best_acc'] for c in chs]
    best_methods = [data[c]['best_method'] for c in chs]

    fig, axL = plt.subplots(figsize=(9, 6.2))
    axR = axL.twinx()

    # Left axis: spread (filled area + line)
    line_spread = axL.plot(chs, spreads, marker='s', markersize=11,
                           linewidth=2.6, color=PAPER_COLORS['fdr'],
                           label='Method spread (pp)')
    axL.fill_between(chs, 0, spreads, color=PAPER_COLORS['fdr'], alpha=0.12)

    axL.set_xscale('log', base=2)
    axL.set_xlabel('Channel count (log scale)', fontsize=FONT_SIZES['axis_label'])
    axL.set_ylabel('Method spread (max − min, pp)',
                   fontsize=FONT_SIZES['axis_label'], color=PAPER_COLORS['fdr'])
    axL.tick_params(axis='y', labelcolor=PAPER_COLORS['fdr'])
    axL.set_xticks(chs)
    axL.set_xticklabels([str(c) for c in chs])
    axL.set_ylim(0, max(spreads) * 1.25)
    axL.grid(True, alpha=0.3, axis='both')

    # Right axis: best-method absolute acc
    line_acc = axR.plot(chs, accs, marker='o', markersize=11,
                        linewidth=2.4, color=PAPER_COLORS['secondary_blue'],
                        label='Best-method abs. acc (%)')
    axR.set_ylabel('Best-method absolute accuracy (%)',
                   fontsize=FONT_SIZES['axis_label'],
                   color=PAPER_COLORS['secondary_blue'])
    axR.tick_params(axis='y', labelcolor=PAPER_COLORS['secondary_blue'])
    axR.set_ylim(70, 92)

    # Combined legend
    lines = line_spread + line_acc
    labels = [l.get_label() for l in lines]
    axL.legend(lines, labels, loc='upper left', fontsize=FONT_SIZES['legend'])

    # Materialise axes transforms before force-directed layout
    fig.canvas.draw()

    # Force-directed label placement on axL (spread, red)
    pts_left = np.array(list(zip(chs, spreads)), dtype=float)
    labels_left = [f'{s:.2f} pp' for s in spreads]
    adjusted_left = force_directed_label_layout(
        pts_left, axL,
        w_point=0.0005, w_label=0.0005, w_diagonal=0.0,
        w_spring=50.0, w_edge=0.002, iterations=100,
    )
    for (xa, ya), (xt, yt), txt in zip(pts_left, adjusted_left, labels_left):
        axL.plot([xa, xt], [ya, yt], color='gray', linewidth=0.5,
                 alpha=0.6, zorder=4)
        axL.text(xt, yt, txt,
                 fontsize=FONT_SIZES['annotation'],
                 color=PAPER_COLORS['fdr'], fontweight='bold',
                 ha='center', va='center', zorder=7,
                 bbox=dict(boxstyle='round,pad=0.2',
                           facecolor='white', edgecolor='none', alpha=0.85))

    # Force-directed label placement on axR (best-method acc, blue)
    pts_right = np.array(list(zip(chs, accs)), dtype=float)
    labels_right = [f'{m}\n{a:.2f}%' for m, a in zip(best_methods, accs)]
    adjusted_right = force_directed_label_layout(
        pts_right, axR,
        w_point=0.0005, w_label=0.0008, w_diagonal=0.0,
        w_spring=50.0, w_edge=0.002, iterations=100,
    )
    for (xa, ya), (xt, yt), txt in zip(pts_right, adjusted_right, labels_right):
        axR.plot([xa, xt], [ya, yt], color='gray', linewidth=0.5,
                 alpha=0.6, zorder=4)
        axR.text(xt, yt, txt,
                 fontsize=FONT_SIZES['annotation'],
                 color=PAPER_COLORS['secondary_blue'], fontweight='bold',
                 ha='center', va='center', zorder=7,
                 bbox=dict(boxstyle='round,pad=0.2',
                           facecolor='white', edgecolor='none', alpha=0.85))

    fig.tight_layout()
    apply_paper_style(fig=fig)
    add_provenance_footer(fig, 'sensitivity_scaling')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / 'sensitivity_scaling.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Saved: {out_path}')


# -----------------------------------------------------------------------------
# T3.5 — 21×N Subject Heatmap (Sup Figure S2)
# -----------------------------------------------------------------------------

def generate_subject_heatmap_figure():
    """T3.5 — 21-subject × multi-condition accuracy heatmap.

    Rows: 21 subjects (sorted by primary baseline = CBraMod within binary)
    Cols: 8 conditions across model × paradigm × task + 32ch FDR + extra-sess S05
    Color: accuracy (%)
    Left annotation: data quality label.
    """
    import matplotlib.pyplot as plt

    # Define columns (label, registry key or path, model, extractor)
    cols = [
        ('EEGNet\nWithin Bin',        'within_eegnet_binary',           'eegnet',  'within'),
        ('CBraMod\nWithin Bin',       'within_cbramod_binary',          'cbramod', 'within'),
        ('EEGNet\nCross Bin',         'cross_eegnet_binary',            'eegnet',  'cross'),
        ('CBraMod\nCross Bin',        'cross_cbramod_binary',           'cbramod', 'cross'),
        ('CBraMod\nCross Ter',        'cross_cbramod_ternary',          'cbramod', 'cross'),
        ('CBraMod\nXSI-FT Bin',       'transfer_binary',                'cbramod', 'transfer'),
        ('CBraMod\n32ch FDR Cross',   'reduced_32_fdr_binary',          'cbramod', 'cross'),
        ('CBraMod\n+Sess05 Within',   'extra_sessions_binary',          'cbramod', 'extra_sess05'),
    ]

    subjects = [f'S{i:02d}' for i in range(1, 22)]
    matrix = np.full((len(subjects), len(cols)), np.nan)

    for j, (_, key, model, extractor) in enumerate(cols):
        path = get_run_path(key)
        if not resolve_project_path(path).exists():
            logger.warning(f'Missing: {path}')
            continue
        cache = load_json_cache(path)

        if extractor in ('within', 'transfer'):
            results = cache.get('results', {}).get(model, {})
            for i, sid in enumerate(subjects):
                if sid in results and isinstance(results[sid], dict):
                    acc = (results[sid].get('test_acc_majority') or
                           results[sid].get('test_acc'))
                    if acc is not None:
                        matrix[i, j] = acc * 100
        elif extractor == 'cross':
            results = cache.get('results', {}).get(model, {})
            per_subj = results.get('per_subject_test_acc', {})
            for i, sid in enumerate(subjects):
                if sid in per_subj:
                    matrix[i, j] = per_subj[sid] * 100
        elif extractor == 'extra_sess05':
            results = cache.get('results', {}).get(model, {})
            for i, sid in enumerate(subjects):
                if sid in results and isinstance(results[sid], dict):
                    s5 = results[sid].get('sess05', {})
                    acc = s5.get('test_acc_majority') or s5.get('test_acc')
                    if acc is not None:
                        matrix[i, j] = acc * 100

    # Sort rows by CBraMod cross-binary baseline (col index 3)
    primary_col = 3
    sort_idx = np.argsort(-np.nan_to_num(matrix[:, primary_col], nan=-1))
    matrix_sorted = matrix[sort_idx]
    subjects_sorted = [subjects[i] for i in sort_idx]
    quality_sorted = [_data_quality_label(s) for s in subjects_sorted]

    quality_color = {
        'clean': '#a5d6a7',
        'informative': '#90caf9',
        'mild': '#ffe082',
        'heavy': '#ef9a9a',
        'unknown': '#cccccc',
    }

    fig, ax = plt.subplots(figsize=(13, 9.2))

    cmap = plt.cm.viridis
    im = ax.imshow(matrix_sorted, aspect='auto', cmap=cmap, vmin=40, vmax=100)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Accuracy (%)', fontsize=FONT_SIZES['axis_label'])

    # Cell values
    for i in range(matrix_sorted.shape[0]):
        for j in range(matrix_sorted.shape[1]):
            v = matrix_sorted[i, j]
            if np.isnan(v):
                continue
            color = 'white' if v < 75 else 'black'
            ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                    fontsize=FONT_SIZES['annotation'], color=color)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[0] for c in cols], fontsize=FONT_SIZES['tick'], rotation=0)
    ax.set_yticks(range(len(subjects_sorted)))
    ax.set_yticklabels(subjects_sorted, fontsize=FONT_SIZES['tick'])

    # Quality color strip on the left (separate axis)
    div_ax = fig.add_axes([0.04, ax.get_position().y0,
                           0.022, ax.get_position().height])
    quality_arr = np.zeros((len(subjects_sorted), 1, 3))
    for i, q in enumerate(quality_sorted):
        c = quality_color[q]
        # hex to rgb
        rgb = tuple(int(c[k:k+2], 16)/255 for k in (1, 3, 5))
        quality_arr[i, 0] = rgb
    div_ax.imshow(quality_arr, aspect='auto')
    div_ax.set_xticks([])
    div_ax.set_yticks([])
    div_ax.set_title('Quality', fontsize=FONT_SIZES['title'])

    # Quality legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=quality_color['clean'], edgecolor='black', label='Clean (10)'),
        Patch(facecolor=quality_color['informative'], edgecolor='black',
              label='Informative high-σ (3)'),
        Patch(facecolor=quality_color['mild'], edgecolor='black', label='Mild artifact (5)'),
        Patch(facecolor=quality_color['heavy'], edgecolor='black', label='Heavy artifact (3)'),
    ]
    ax.legend(handles=legend_handles, loc='upper left',
              bbox_to_anchor=(1.18, 1.02), fontsize=FONT_SIZES['legend'],
              title='Data quality (§2.9)', title_fontsize=FONT_SIZES['legend'])

    apply_paper_style(fig=fig)
    add_provenance_footer(fig, 'subject_heatmap')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / 'subject_heatmap.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Saved: {out_path}')


# -----------------------------------------------------------------------------
# T3.7 — Extra Sessions baseline-colored scatter (Figure 7/8 update)
# -----------------------------------------------------------------------------

def _extra_sessions_baseline_colored(task: str):
    """Generate a baseline-colored Δ scatter for extra-sessions binary/ternary."""
    import matplotlib.pyplot as plt

    if task == 'binary':
        path = get_run_path('extra_sessions_binary')
        out_name = 'extra_sessions_binary.png'
    elif task == 'ternary':
        path = get_run_path('extra_sessions_ternary')
        out_name = 'extra_sessions_ternary.png'
    else:
        raise ValueError(task)

    if not resolve_project_path(path).exists():
        logger.warning(f'Missing: {path}')
        return

    cache = load_json_cache(path)
    fig, (ax_traj, ax_delta) = plt.subplots(1, 2, figsize=(14, 5.8),
                                            gridspec_kw={'width_ratios': [1.4, 1]})

    # CBraMod per-subject trajectories
    results = cache.get('results', {}).get('cbramod', {})
    subj_trajs = {}  # sid -> [bl, s3, s4, s5]
    steps = ['baseline', 'sess03', 'sess04', 'sess05']
    for sid, sdata in sorted(results.items()):
        if not isinstance(sdata, dict) or not sid.startswith('S'):
            continue
        seq = []
        for s in steps:
            v = sdata.get(s, {}).get('test_acc_majority')
            seq.append(v * 100 if v is not None else np.nan)
        if not all(np.isnan(seq)):
            subj_trajs[sid] = seq

    if not subj_trajs:
        logger.warning(f'No subject data in {path}')
        return

    # Baseline distribution → colormap
    bls = [v[0] for v in subj_trajs.values() if not np.isnan(v[0])]
    cmap = plt.cm.coolwarm_r
    norm = plt.Normalize(vmin=min(bls), vmax=max(bls))

    x = np.arange(len(steps))
    for sid, seq in subj_trajs.items():
        bl = seq[0]
        if np.isnan(bl):
            continue
        c = cmap(norm(bl))
        # 被试个体轨迹 — 置于最底层 (zorder=1), 让均值线与散点都能在其上方显示
        ax_traj.plot(x, seq, color=c, alpha=0.55, linewidth=1.2,
                     marker='o', markersize=4, zorder=1)

    # 均值线 — 比原版更细 / 半透明 / 白心圆点, 显式置于个体轨迹之上 (zorder=2)
    # 但仍允许下方散点透出, 解决"黑色粗线覆盖被试数据"的可读性问题
    mean_seq = np.nanmean(np.array(list(subj_trajs.values())), axis=0)
    ax_traj.plot(x, mean_seq,
                 color=PAPER_COLORS['mean_marker'],
                 linewidth=2,
                 marker='o',
                 markersize=6,
                 alpha=0.85,
                 zorder=2,
                 markerfacecolor='white',
                 markeredgewidth=1.8,
                 label=f'Mean (N={len(subj_trajs)})')

    ax_traj.set_xticks(x)
    ax_traj.set_xticklabels(['Baseline', '+Sess03', '+Sess04', '+Sess05'])
    ax_traj.set_ylabel('Accuracy (%)', fontsize=FONT_SIZES['axis_label'])
    ax_traj.set_title(f'A. Per-subject trajectory\n(CBraMod, {task}, baseline-colored)',
                      fontsize=FONT_SIZES['title'])
    ax_traj.grid(True, alpha=0.3)
    ax_traj.legend(loc='lower right', fontsize=FONT_SIZES['legend'])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_traj, pad=0.02)
    cbar.set_label('Baseline accuracy (%)', fontsize=FONT_SIZES['annotation'])

    # Δ vs baseline scatter + regression
    bl_arr = []
    delta_arr = []
    for sid, seq in subj_trajs.items():
        if np.isnan(seq[0]) or np.isnan(seq[-1]):
            continue
        bl_arr.append(seq[0])
        delta_arr.append(seq[-1] - seq[0])
    bl_arr = np.array(bl_arr)
    delta_arr = np.array(delta_arr)

    # 被试散点 — 白色描边 + 提升 zorder, 保证在 OLS 拟合线之上可读
    sc = ax_delta.scatter(bl_arr, delta_arr, c=bl_arr, cmap=cmap, norm=norm,
                          s=85, alpha=0.95,
                          edgecolor='white', linewidth=0.8,
                          zorder=3)
    ax_delta.axhline(0, color=PAPER_COLORS['chance_level'],
                     linestyle='--', alpha=0.6, zorder=1)

    if len(bl_arr) > 2:
        slope, intercept = np.polyfit(bl_arr, delta_arr, 1)
        x_fit = np.linspace(bl_arr.min(), bl_arr.max(), 50)
        y_fit = slope * x_fit + intercept
        # OLS 拟合线置于散点之下 (zorder=2), 避免遮挡数据点
        ax_delta.plot(x_fit, y_fit,
                      color=PAPER_COLORS['mean_marker'],
                      linestyle='--', linewidth=1.5,
                      alpha=0.7, zorder=2,
                      label=f'OLS: slope={slope:+.3f} pp/pp')

    ax_delta.set_xlabel('Baseline accuracy (%)', fontsize=FONT_SIZES['axis_label'])
    ax_delta.set_ylabel('Δ (BL → +Sess05, pp)', fontsize=FONT_SIZES['axis_label'])
    ax_delta.set_title(f'B. Δ vs baseline ({task}; high baselines saturate)',
                      fontsize=FONT_SIZES['title'])
    ax_delta.grid(True, alpha=0.3)
    ax_delta.legend(loc='upper right', fontsize=FONT_SIZES['legend'])

    # 移除图级 suptitle — 论文 caption 已是权威标题, 仅保留 panel-level 子标题 A./B.
    fig.tight_layout()
    apply_paper_style(fig=fig)
    add_provenance_footer(fig, f'extra_sessions_{task}')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / out_name
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Saved: {out_path}')


def generate_extra_sessions_binary_v2_figure():
    """T3.7 — Extra Sessions binary baseline-colored scatter."""
    _extra_sessions_baseline_colored('binary')


def generate_extra_sessions_ternary_v2_figure():
    """T3.7 — Extra Sessions ternary baseline-colored scatter."""
    _extra_sessions_baseline_colored('ternary')


# -----------------------------------------------------------------------------
# T3.9 — Inference Latency + Throughput Panel (UPDATED)
# -----------------------------------------------------------------------------

def generate_inference_latency_v2_figure():
    """T3.9 — Inference latency + throughput right panel.

    Replaces existing `paper/figures/inference_latency.png`.
    Right panel changed from "model scale ratios" to "throughput (samples/sec)
    by batch size" bar chart.
    """
    import matplotlib.pyplot as plt
    from src.config.constants import MODEL_COLORS

    # Per Table 17 in §3.8
    batch_sizes = [1, 8, 32, 64]
    eegnet_latency = [0.375, 0.542, 2.058, 4.027]
    cbramod_latency = [12.919, 12.582, 32.729, 71.110]
    eegnet_throughput = [2665, 14756, 15547, 15894]
    cbramod_throughput = [77, 636, 978, 900]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6))

    # Left: Latency comparison (log scale)
    x = np.arange(len(batch_sizes))
    width = 0.36
    ax1.bar(x - width/2, eegnet_latency, width, label='EEGNet-16,4',
            color=MODEL_COLORS['eegnet'], edgecolor='black')
    ax1.bar(x + width/2, cbramod_latency, width, label='CBraMod',
            color=MODEL_COLORS['cbramod'], edgecolor='black')
    ax1.set_yscale('log')
    ax1.set_ylabel('Latency (ms, log)', fontsize=FONT_SIZES['axis_label'])
    ax1.set_xlabel('Batch size', fontsize=FONT_SIZES['axis_label'])
    ax1.set_xticks(x)
    ax1.set_xticklabels(batch_sizes)
    ax1.set_title('A. Inference latency (128ch binary)', fontsize=FONT_SIZES['title'])
    ax1.axhline(100, color='red', linestyle='--', alpha=0.7)
    ax1.text(0.05, 110, '100 ms real-time threshold',
             fontsize=FONT_SIZES['annotation'], color='red')
    for xi, ev, cv in zip(x, eegnet_latency, cbramod_latency):
        ax1.annotate(f'{ev:.2f}', (xi - width/2, ev),
                     textcoords='offset points', xytext=(0, 6),
                     ha='center', fontsize=FONT_SIZES['annotation'])
        ax1.annotate(f'{cv:.1f}', (xi + width/2, cv),
                     textcoords='offset points', xytext=(0, 6),
                     ha='center', fontsize=FONT_SIZES['annotation'])
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend(loc='lower right', fontsize=FONT_SIZES['legend'])

    # Right: Throughput comparison
    ax2.bar(x - width/2, eegnet_throughput, width, label='EEGNet-16,4',
            color=MODEL_COLORS['eegnet'], edgecolor='black')
    ax2.bar(x + width/2, cbramod_throughput, width, label='CBraMod',
            color=MODEL_COLORS['cbramod'], edgecolor='black')
    ax2.set_yscale('log')
    ax2.set_ylabel('Throughput (samples / sec, log)', fontsize=FONT_SIZES['axis_label'])
    ax2.set_xlabel('Batch size', fontsize=FONT_SIZES['axis_label'])
    ax2.set_xticks(x)
    ax2.set_xticklabels(batch_sizes)
    ax2.set_title('B. Throughput (samples/sec, higher = better)', fontsize=FONT_SIZES['title'])
    for xi, ev, cv in zip(x, eegnet_throughput, cbramod_throughput):
        ax2.annotate(f'{ev:,}', (xi - width/2, ev),
                     textcoords='offset points', xytext=(0, 6),
                     ha='center', fontsize=FONT_SIZES['annotation'])
        ax2.annotate(f'{cv:,}', (xi + width/2, cv),
                     textcoords='offset points', xytext=(0, 6),
                     ha='center', fontsize=FONT_SIZES['annotation'])
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend(loc='lower right', fontsize=FONT_SIZES['legend'])

    fig.tight_layout()
    apply_paper_style(fig=fig)
    add_provenance_footer(fig, 'inference_latency')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / 'inference_latency.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Saved: {out_path}')


# -----------------------------------------------------------------------------
# T3.10 — Fig 5a/5b merged 4ch optimal vs neg control
# -----------------------------------------------------------------------------

def generate_fig5_merged_figure():
    """T3.10 — Figure 5: 4-panel comparison of 4ch configurations.

    Panel A: FDR ∩ Att outlier (CBraMod + EEGNet)
    Panel B: Negative Control  (CBraMod + EEGNet)
    Panel C: Band Power top-4  (CBraMod only — 20260505_2308)
    Panel D: CSP top-4         (CBraMod only — 20260505_2246)

    Panels C/D were added in 2026-05-05 补充实验; EEGNet was not run for
    those configs, so only CBraMod bars + mean line are drawn.
    """
    import matplotlib.pyplot as plt
    from src.config.constants import MODEL_COLORS

    configs = [
        ('A. 4ch FDR ∩ Att (optimal outlier)',
         get_run_path('reduced_4_fdr_attention_overlap_binary')),
        ('B. 4ch Negative Control',
         get_run_path('reduced_4_negative_control_binary')),
        ('C. 4ch Band Power top-4',
         get_run_path('reduced_4_band_power_binary')),
        ('D. 4ch CSP top-4',
         get_run_path('reduced_4_csp_binary')),
    ]

    # Also load 128ch baseline for overlay
    baseline_eg = get_run_path('cross_eegnet_binary')
    baseline_cb = get_run_path('cross_cbramod_binary')

    bl_eg_accs = (extract_model_accs(load_json_cache(baseline_eg), 'eegnet')
                  if resolve_project_path(baseline_eg).exists() else [])
    bl_cb_accs = (extract_model_accs(load_json_cache(baseline_cb), 'cbramod')
                  if resolve_project_path(baseline_cb).exists() else [])

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharey=True)
    axes_flat = axes.flatten()

    subjects = [f'S{i:02d}' for i in range(1, 22)]

    for ax, (title, path) in zip(axes_flat, configs):
        if not resolve_project_path(path).exists():
            logger.warning(f'Missing: {path}')
            ax.set_title(f'{title}\n(missing)', fontsize=FONT_SIZES['title'])
            continue
        cache = load_json_cache(path)
        cb_results = cache.get('results', {}).get('cbramod', {})
        eg_results = cache.get('results', {}).get('eegnet', {})
        cb_per = cb_results.get('per_subject_test_acc', {})
        eg_per = eg_results.get('per_subject_test_acc', {})

        has_eegnet = bool(eg_per)

        x = np.arange(len(subjects))
        width = 0.36 if has_eegnet else 0.7
        cb_y = [cb_per.get(s, np.nan) * 100 if cb_per.get(s) is not None else np.nan
                for s in subjects]

        if has_eegnet:
            eg_y = [eg_per.get(s, np.nan) * 100 if eg_per.get(s) is not None else np.nan
                    for s in subjects]
            ax.bar(x - width/2, cb_y, width, label='CBraMod (4ch)',
                   color=MODEL_COLORS['cbramod'], edgecolor='black')
            ax.bar(x + width/2, eg_y, width, label='EEGNet (4ch)',
                   color=MODEL_COLORS['eegnet'], edgecolor='black')
        else:
            ax.bar(x, cb_y, width, label='CBraMod (4ch)',
                   color=MODEL_COLORS['cbramod'], edgecolor='black')

        # Mean lines
        cb_mean = np.nanmean(cb_y)
        ax.axhline(cb_mean, color=MODEL_COLORS['cbramod'], linestyle='--',
                   alpha=0.7, label=f'CBraMod 4ch mean ({cb_mean:.1f}%)')
        if has_eegnet:
            eg_mean = np.nanmean(eg_y)
            ax.axhline(eg_mean, color=MODEL_COLORS['eegnet'], linestyle='--',
                       alpha=0.7, label=f'EEGNet 4ch mean ({eg_mean:.1f}%)')

        # 128ch baseline overlays
        if bl_cb_accs:
            ax.axhline(np.mean(bl_cb_accs), color=MODEL_COLORS['cbramod'],
                       linestyle=':', alpha=0.6,
                       label=f'CBraMod 128ch ({np.mean(bl_cb_accs):.1f}%)')
        if bl_eg_accs:
            ax.axhline(np.mean(bl_eg_accs), color=MODEL_COLORS['eegnet'],
                       linestyle=':', alpha=0.6,
                       label=f'EEGNet 128ch ({np.mean(bl_eg_accs):.1f}%)')

        ax.axhline(50, color=PAPER_COLORS['chance_red'], linestyle='--',
                   alpha=0.85, linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(subjects, fontsize=FONT_SIZES['tick'], rotation=45)
        ax.set_title(title, fontsize=FONT_SIZES['title'])
        ax.set_ylim(40, 105)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='lower right', fontsize=FONT_SIZES['legend'], ncol=1)

    axes[0, 0].set_ylabel('Cross-subject binary accuracy (%)', fontsize=FONT_SIZES['axis_label'])
    axes[1, 0].set_ylabel('Cross-subject binary accuracy (%)', fontsize=FONT_SIZES['axis_label'])

    fig.tight_layout()
    apply_paper_style(fig=fig)
    add_provenance_footer(fig, 'fig5_merged')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / 'fig5_4ch_optimal_vs_neg_control.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Saved: {out_path}')


# =============================================================================
# Main
# =============================================================================

FIGURE_GENERATORS = {
    'channel_scaling': generate_channel_scaling_v2_figure,
    'further_pretraining': generate_further_pretraining_v3_figure,
    'inference_latency': generate_inference_latency_v2_figure,
    '32ch_comparison': generate_32ch_comparison_figure,
    'extra_sessions_paradigm': generate_extra_sessions_paradigm_figure,
    'extra_sessions_strategy': generate_extra_sessions_strategy_figure,
    'figure2': generate_figure2_128ch_cross_subject,
    'figure3b': generate_figure3b_32ch_fdr,
    'figure5': generate_figure5_4ch_control,
    'baseline_plots': generate_all_baseline_plots,
    'channel_ranking_flip': generate_8ch_ranking_flip_figure,
    'cross_subject_pooling_forest': generate_cross_subject_pooling_forest_figure,
    # Stage 4 Step 3 additions:
    'dapt_v1_v5_forest': generate_dapt_v1_v5_forest_figure,
    'exploratory_ablation_overview': generate_exploratory_ablation_overview_figure,
    'sensitivity_scaling': generate_sensitivity_scaling_figure,
    'subject_heatmap': generate_subject_heatmap_figure,
    'extra_sessions_binary_v2': generate_extra_sessions_binary_v2_figure,
    'extra_sessions_ternary_v2': generate_extra_sessions_ternary_v2_figure,
    'fig5_merged': generate_fig5_merged_figure,
    # 2026-05-12: 40-cell matrix overview (commit-time figure)
    'reduced_channel_40cell_grid': generate_reduced_channel_grid_figure,
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
