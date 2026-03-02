#!/usr/bin/env python
"""
Advanced per-subject EEG data quality analysis (Phase 2).

Extends the base quality analysis with deeper investigation axes:
1. Class discriminability (Fisher ratio, per-channel AUROC)
2. Temporal drift (within-session nonstationarity)
3. Spectral features (mu/beta band power)
4. EMG contamination indicators (peripheral vs central high-freq power)
5. Adjacent trial autocorrelation
6. Cross-subject similarity matrix

Data: HDF5 cache (post-CAR, post-bandpass 4-40 Hz, downsampled 100 Hz, pre-z-score)

Usage:
    uv run python scripts/analysis/analyze_data_quality_advanced.py
    uv run python scripts/analysis/analyze_data_quality_advanced.py --subjects S01 S04
    uv run python scripts/analysis/analyze_data_quality_advanced.py --workers 8 -v
"""

import argparse
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import signal, stats

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse data loading from base analysis script
from analyze_data_quality import (
    load_cache_index,
    group_entries_by_subject,
    load_subject_data,
)
from src.preprocessing.channel_selection import BIOSEMI_128_LABELS

# ============================================================================
# Constants
# ============================================================================

FS = 100  # Hz — cache sampling rate (downsampled)

# Motor cortex channel indices (32ch motor_cortex config from channel_selection.py)
CENTRAL_CH = set([
    0, 2, 3, 5, 20, 32, 33, 34, 49, 50, 52, 53, 55,
    62, 63, 64, 65, 66, 77, 85, 86, 90, 97, 107, 108,
    110, 111, 112, 113, 114, 116, 123,
])
PERIPHERAL_CH = set(range(128)) - CENTRAL_CH

# Frequency bands (data is bandpassed 4-40 Hz)
BANDS = {
    'theta': (4, 8),
    'mu': (8, 13),
    'low_beta': (13, 20),
    'high_beta': (20, 30),
    'gamma': (30, 40),
}


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class AdvancedReport:
    subject_id: str
    total_trials: int = 0

    class_discriminability: Dict = field(default_factory=dict)
    temporal_drift: Dict = field(default_factory=dict)
    spectral_features: Dict = field(default_factory=dict)
    emg_indicators: Dict = field(default_factory=dict)
    adjacent_trial_corr: Dict = field(default_factory=dict)

    # Compact feature vector for cross-subject comparison
    feature_vector: np.ndarray = field(default_factory=lambda: np.zeros(10))


# ============================================================================
# Helpers
# ============================================================================

def _get_valid(trial):
    """Extract valid (non-NaN) portion of a trial [128, T] -> [128, T_valid]."""
    valid_mask = ~np.isnan(trial[0])
    if not valid_mask.any():
        return None
    return trial[:, valid_mask]


def _compute_bandpower(valid_trial, fs=FS):
    """Compute per-channel band power from valid trial data [128, T_valid].

    Returns dict of {band_name: [128] array} or None if too short.
    """
    n_samples = valid_trial.shape[1]
    if n_samples < 50:  # Need at least 0.5s
        return None

    nperseg = min(n_samples, 100)
    freqs, psd = signal.welch(valid_trial, fs=fs, nperseg=nperseg, axis=1)
    # psd: [128, n_freqs]

    result = {}
    for band_name, (flo, fhi) in BANDS.items():
        mask = (freqs >= flo) & (freqs <= fhi)
        if mask.any():
            result[band_name] = np.mean(psd[:, mask], axis=1)  # [128]
        else:
            result[band_name] = np.zeros(valid_trial.shape[0])

    return result


def _compute_fisher_auroc(c1: np.ndarray, c2: np.ndarray):
    """Compute Fisher ratio and AUROC arrays for two class arrays [N, D]."""
    mu1, mu2 = np.mean(c1, axis=0), np.mean(c2, axis=0)
    var1, var2 = np.var(c1, axis=0), np.var(c2, axis=0)
    fisher = (mu1 - mu2) ** 2 / (var1 + var2 + 1e-12)

    n_features = c1.shape[1]
    aurocs = np.full(n_features, 0.5)
    for ch in range(n_features):
        try:
            U, _ = stats.mannwhitneyu(c1[:, ch], c2[:, ch], alternative='two-sided')
            auc = U / (len(c1) * len(c2))
            aurocs[ch] = max(auc, 1 - auc)  # Ensure >= 0.5
        except ValueError:
            pass

    return fisher, aurocs


# ============================================================================
# 1. Class Discriminability
# ============================================================================

def check_class_discriminability(data: Dict[str, List]) -> dict:
    """Compute per-channel class discriminability using band power features.

    Fisher ratio = (mu1 - mu2)^2 / (var1 + var2) per channel.
    AUROC via Mann-Whitney U per channel.

    Uses mu+beta band power as features (neurophysiologically motivated for MI/ME).
    Reports results for binary (class 1 vs 4) and all class pairs.
    """
    class_features = defaultdict(list)  # {label: [[128], ...]}

    for session, runs in data.items():
        for trials, labels, run in runs:
            for t_idx in range(trials.shape[0]):
                lbl = int(labels[t_idx])
                valid = _get_valid(trials[t_idx])
                if valid is None:
                    continue
                bp = _compute_bandpower(valid)
                if bp is None:
                    continue
                # Combined mu + beta power as discriminative feature
                feature = bp['mu'] + bp['low_beta'] + bp['high_beta']  # [128]
                class_features[lbl].append(feature)

    available_classes = sorted(class_features.keys())
    if len(available_classes) < 2:
        return {'binary': {}, 'all_pairs_fisher_mean': 0.0,
                'all_pairs_auroc_mean': 0.5, 'n_classes': len(available_classes),
                'classes': available_classes, 'insufficient_data': True}

    class_arrays = {c: np.array(class_features[c]) for c in available_classes}

    # Binary analysis: class 1 (Thumb) vs class 4 (Pinky)
    binary_result = {}
    if 1 in class_arrays and 4 in class_arrays:
        c1, c4 = class_arrays[1], class_arrays[4]
        fisher, aurocs = _compute_fisher_auroc(c1, c4)

        top_k = 10
        top_idx = np.argsort(fisher)[::-1][:top_k]
        top_channels = [(int(i), BIOSEMI_128_LABELS[i], float(fisher[i]))
                        for i in top_idx]
        n_central = sum(1 for i in top_idx if i in CENTRAL_CH)

        binary_result = {
            'fisher_mean': float(np.mean(fisher)),
            'fisher_max': float(np.max(fisher)),
            'auroc_mean': float(np.mean(aurocs)),
            'auroc_max': float(np.max(aurocs)),
            'top_channels': top_channels,
            'top_in_motor_cortex': n_central,
            'n_class1': len(c1),
            'n_class4': len(c4),
        }

    # All-pairs analysis
    all_fisher = []
    all_auroc = []
    for i, ci in enumerate(available_classes):
        for cj in available_classes[i + 1:]:
            f, a = _compute_fisher_auroc(class_arrays[ci], class_arrays[cj])
            all_fisher.append(np.mean(f))
            all_auroc.append(np.mean(a))

    return {
        'binary': binary_result,
        'all_pairs_fisher_mean': float(np.mean(all_fisher)) if all_fisher else 0.0,
        'all_pairs_auroc_mean': float(np.mean(all_auroc)) if all_auroc else 0.5,
        'n_classes': len(available_classes),
        'classes': available_classes,
    }


# ============================================================================
# 2. Temporal Drift
# ============================================================================

def check_temporal_drift(data: Dict[str, List]) -> dict:
    """Track within-session signal drift across sequential runs.

    Computes per-run channel mean vectors and tracks L2 distance
    from the first run in each session. High drift indicates
    electrode impedance changes, fatigue, or movement.
    """
    per_session = {}
    max_drift_overall = 0.0
    max_drift_session = ''

    for session, runs in sorted(data.items()):
        if len(runs) < 2:
            continue

        sorted_runs = sorted(runs, key=lambda x: x[2])  # sort by run_num
        run_means = []
        for trials, labels, run_num in sorted_runs:
            ch_means = np.nanmean(trials, axis=(0, 2))  # [128]
            if not np.any(np.isnan(ch_means)):
                run_means.append(ch_means)

        if len(run_means) < 2:
            continue

        baseline = run_means[0]
        drifts = [float(np.linalg.norm(rm - baseline)) for rm in run_means]
        max_drift = max(drifts)

        per_session[session] = {
            'n_runs': len(run_means),
            'max_drift_l2': max_drift,
            'drift_trajectory': drifts,
            'drift_rate': max_drift / len(run_means),
        }

        if max_drift > max_drift_overall:
            max_drift_overall = max_drift
            max_drift_session = session

    return {
        'per_session': per_session,
        'max_drift': max_drift_overall,
        'max_drift_session': max_drift_session,
        'n_sessions_analyzed': len(per_session),
    }


# ============================================================================
# 3. Spectral Features
# ============================================================================

def check_spectral_features(data: Dict[str, List]) -> dict:
    """Compute per-subject band power distribution.

    Reports theta, mu, beta, gamma band power and mu/beta ratio.
    Uses a random subsample of trials for efficiency.
    """
    rng = np.random.RandomState(42)
    all_bandpowers = {band: [] for band in BANDS}

    # Collect all trials then subsample
    all_trials = []
    for session, runs in data.items():
        for trials, labels, run in runs:
            for t_idx in range(trials.shape[0]):
                all_trials.append(trials[t_idx])

    max_trials = min(len(all_trials), 500)
    indices = (rng.choice(len(all_trials), max_trials, replace=False)
               if len(all_trials) > 500 else range(len(all_trials)))
    n_computed = 0

    for idx in indices:
        valid = _get_valid(all_trials[idx])
        if valid is None:
            continue
        bp = _compute_bandpower(valid)
        if bp is None:
            continue
        for band in BANDS:
            all_bandpowers[band].append(np.mean(bp[band]))  # scalar mean across channels
        n_computed += 1

    if n_computed == 0:
        return {f'{b}_mean': 0.0 for b in BANDS}

    result = {}
    for band in BANDS:
        arr = np.array(all_bandpowers[band])
        result[f'{band}_mean'] = float(np.mean(arr))
        result[f'{band}_std'] = float(np.std(arr))

    mu = result.get('mu_mean', 1e-12)
    beta = result.get('low_beta_mean', 0) + result.get('high_beta_mean', 0)
    result['mu_beta_ratio'] = float(mu / (beta + 1e-12))
    result['n_trials_analyzed'] = n_computed

    return result


# ============================================================================
# 4. EMG Contamination Indicators
# ============================================================================

def check_emg_indicators(data: Dict[str, List]) -> dict:
    """Detect potential EMG contamination via high-frequency power distribution.

    Compares high-frequency (20-40 Hz) power between peripheral and
    central (motor cortex) channels. EMG artifacts manifest as elevated
    broadband power, especially in temporal/frontal channels.

    Note: data is bandpassed 4-40 Hz, limiting detection to low-freq muscle artifacts.
    """
    rng = np.random.RandomState(42)
    central_hf = []
    peripheral_hf = []
    central_total = []
    peripheral_total = []

    central_idx = sorted(CENTRAL_CH)
    peripheral_idx = sorted(PERIPHERAL_CH)

    # Subsample trials
    all_trials = []
    for session, runs in data.items():
        for trials, labels, run in runs:
            for t_idx in range(trials.shape[0]):
                all_trials.append(trials[t_idx])

    max_trials = min(len(all_trials), 300)
    indices = (rng.choice(len(all_trials), max_trials, replace=False)
               if len(all_trials) > 300 else range(len(all_trials)))

    for idx in indices:
        valid = _get_valid(all_trials[idx])
        if valid is None:
            continue
        bp = _compute_bandpower(valid)
        if bp is None:
            continue

        # High-frequency = high_beta + gamma (20-40 Hz)
        hf = bp['high_beta'] + bp['gamma']  # [128]
        total = sum(bp[b] for b in BANDS)

        central_hf.append(np.mean(hf[central_idx]))
        peripheral_hf.append(np.mean(hf[peripheral_idx]))
        central_total.append(np.mean(total[central_idx]))
        peripheral_total.append(np.mean(total[peripheral_idx]))

    if not central_hf:
        return {'peripheral_central_ratio': 1.0, 'flag': None}

    mean_c_hf = float(np.mean(central_hf))
    mean_p_hf = float(np.mean(peripheral_hf))
    ratio = mean_p_hf / (mean_c_hf + 1e-12)

    mean_c_total = float(np.mean(central_total))
    mean_p_total = float(np.mean(peripheral_total))
    hf_frac_central = mean_c_hf / (mean_c_total + 1e-12)
    hf_frac_peripheral = mean_p_hf / (mean_p_total + 1e-12)

    # Threshold 3.0 (not 2.0): post-CAR peripheral channels naturally have
    # higher residual power because they contribute less to the common average.
    # A P/C ratio of ~2.0 is the expected post-CAR baseline.
    flag = None
    if ratio > 3.0:
        flag = 'high_peripheral_hf'

    return {
        'central_hf_mean': mean_c_hf,
        'peripheral_hf_mean': mean_p_hf,
        'peripheral_central_ratio': ratio,
        'hf_frac_central': hf_frac_central,
        'hf_frac_peripheral': hf_frac_peripheral,
        'flag': flag,
    }


# ============================================================================
# 5. Adjacent Trial Autocorrelation
# ============================================================================

def check_adjacent_trial_corr(data: Dict[str, List]) -> dict:
    """Check correlation between consecutive trials within each run.

    High correlation between adjacent trials suggests:
    - Sliding window overlap in trial segmentation
    - Slow signal drift not removed by preprocessing
    Expected: low correlation for independently segmented trials.
    """
    correlations = []

    for session, runs in data.items():
        for trials, labels, run_num in runs:
            n_trials = trials.shape[0]
            if n_trials < 2:
                continue

            for t in range(n_trials - 1):
                v1 = _get_valid(trials[t])
                v2 = _get_valid(trials[t + 1])
                if v1 is None or v2 is None:
                    continue

                # Use channel means as compact representation
                f1 = np.mean(v1, axis=1)  # [128]
                f2 = np.mean(v2, axis=1)  # [128]

                corr = np.corrcoef(f1, f2)[0, 1]
                if not np.isnan(corr):
                    correlations.append(float(corr))

    if not correlations:
        return {'mean_corr': 0.0, 'n_pairs': 0}

    arr = np.array(correlations)

    flag = None
    if np.mean(arr) > 0.6:
        flag = 'very_high_adjacent_corr'
    elif np.mean(arr) > 0.3:
        flag = 'moderate_adjacent_corr'

    return {
        'mean_corr': float(np.mean(arr)),
        'std_corr': float(np.std(arr)),
        'median_corr': float(np.median(arr)),
        'max_corr': float(np.max(arr)),
        'pct_above_0.5': float(np.mean(arr > 0.5) * 100),
        'pct_above_0.8': float(np.mean(arr > 0.8) * 100),
        'n_pairs': len(arr),
        'flag': flag,
    }


# ============================================================================
# Subject-level Analysis
# ============================================================================

def analyze_subject_advanced(
    subject_id: str,
    subject_entries: Dict[str, List[dict]],
    cache_dir: str,
    verbose: bool = False,
) -> AdvancedReport:
    """Run all advanced checks for one subject."""
    report = AdvancedReport(subject_id=subject_id)

    if verbose:
        print(f"  Loading data for {subject_id}...")

    data = load_subject_data(subject_entries, cache_dir)
    if not data:
        return report

    for session, runs in data.items():
        report.total_trials += sum(t.shape[0] for t, l, r in runs)

    if verbose:
        print(f"  {subject_id}: {report.total_trials} trials, running advanced analysis...")

    report.class_discriminability = check_class_discriminability(data)
    report.temporal_drift = check_temporal_drift(data)
    report.spectral_features = check_spectral_features(data)
    report.emg_indicators = check_emg_indicators(data)
    report.adjacent_trial_corr = check_adjacent_trial_corr(data)

    # Build feature vector for cross-subject comparison
    cd = report.class_discriminability.get('binary', {})
    sf = report.spectral_features
    emg = report.emg_indicators
    atc = report.adjacent_trial_corr
    td = report.temporal_drift

    report.feature_vector = np.array([
        cd.get('fisher_mean', 0),
        cd.get('auroc_mean', 0.5),
        cd.get('top_in_motor_cortex', 0),
        sf.get('mu_mean', 0),
        sf.get('low_beta_mean', 0),
        sf.get('high_beta_mean', 0),
        sf.get('mu_beta_ratio', 0),
        emg.get('peripheral_central_ratio', 1),
        atc.get('mean_corr', 0),
        td.get('max_drift', 0),
    ])

    return report


# ============================================================================
# Cross-Subject Similarity Analysis
# ============================================================================

FEATURE_NAMES = [
    'fisher_mean', 'auroc_mean', 'motor_overlap',
    'mu_power', 'low_beta', 'high_beta', 'mu_beta_ratio',
    'emg_ratio', 'adj_trial_corr', 'max_drift',
]


def compute_cross_subject_similarity(reports: List[AdvancedReport]) -> dict:
    """Compute pairwise distance matrix and identify clusters/outliers."""
    subjects = [r.subject_id for r in reports]
    n = len(reports)

    features = np.array([r.feature_vector for r in reports])  # [N, 10]

    # Z-score normalize
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std[std < 1e-10] = 1.0
    normalized = (features - mean) / std

    # Pairwise Euclidean distance
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(normalized[i] - normalized[j]))
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # Nearest neighbors
    nearest = {}
    for i, subj in enumerate(subjects):
        dists = [(subjects[j], dist_matrix[i, j]) for j in range(n) if j != i]
        dists.sort(key=lambda x: x[1])
        nearest[subj] = dists[:3]

    # Most isolated (highest mean distance)
    mean_dists = [(subjects[i], float(np.mean(dist_matrix[i, :])))
                  for i in range(n)]
    mean_dists.sort(key=lambda x: x[1], reverse=True)

    return {
        'distance_matrix': dist_matrix,
        'subjects': subjects,
        'nearest_neighbors': nearest,
        'most_isolated': mean_dists,
        'feature_names': FEATURE_NAMES,
    }


# ============================================================================
# Report Generation
# ============================================================================

def generate_advanced_report(
    reports: List[AdvancedReport],
    cross_subject: dict,
    output_path: str,
    elapsed: float,
):
    """Generate Markdown report and console summary."""
    reports.sort(key=lambda r: r.subject_id)

    # Console summary
    print(f"\n{'='*60}")
    print(f"Advanced Data Quality Analysis Complete ({elapsed:.1f}s)")
    print(f"{'='*60}")

    ranked = sorted(
        reports,
        key=lambda r: r.class_discriminability.get('binary', {}).get('fisher_mean', 0),
        reverse=True,
    )
    print(f"\nClass Discriminability Ranking (Fisher ratio, binary):")
    for r in ranked[:5]:
        cd = r.class_discriminability.get('binary', {})
        print(f"  {r.subject_id}: Fisher={cd.get('fisher_mean', 0):.4f}, "
              f"AUROC={cd.get('auroc_mean', 0.5):.4f}")
    if len(ranked) > 8:
        print(f"  ...")
    for r in ranked[-3:]:
        cd = r.class_discriminability.get('binary', {})
        print(f"  {r.subject_id}: Fisher={cd.get('fisher_mean', 0):.4f}, "
              f"AUROC={cd.get('auroc_mean', 0.5):.4f}")

    emg_flagged = [r for r in reports if r.emg_indicators.get('flag')]
    if emg_flagged:
        print(f"\nEMG contamination flags:")
        for r in emg_flagged:
            ratio = r.emg_indicators.get('peripheral_central_ratio', 0)
            print(f"  {r.subject_id}: P/C ratio={ratio:.2f}")

    adj_flagged = [r for r in reports if r.adjacent_trial_corr.get('flag')]
    if adj_flagged:
        print(f"\nAdjacent trial correlation flags:")
        for r in adj_flagged:
            print(f"  {r.subject_id}: mean_r={r.adjacent_trial_corr.get('mean_corr', 0):.4f} "
                  f"({r.adjacent_trial_corr.get('flag')})")

    print(f"\nMost isolated subjects:")
    for subj, dist in cross_subject.get('most_isolated', [])[:3]:
        print(f"  {subj}: mean_dist={dist:.2f}")

    # ================================================================
    # Markdown report
    # ================================================================
    lines = []
    lines.append("# Advanced EEG Data Quality Report\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Subjects: {len(reports)}")
    lines.append(f"Analysis time: {elapsed:.1f}s")
    lines.append(f"Data: HDF5 cache (post-CAR, post-bandpass 4-40 Hz, 100 Hz, pre-z-score)\n")

    # ---- 1. Class Discriminability ----
    lines.append("## 1. Class Discriminability\n")
    lines.append("Fisher's Linear Discriminant Ratio and AUROC computed on **mu+beta band power** "
                 "features per channel. Measures how well EEG signals separate finger classes "
                 "*without* any trained model.\n")

    lines.append("### 1.1 Binary (Class 1 Thumb vs Class 4 Pinky)\n")
    lines.append("| Subject | Fisher Mean | Fisher Max | AUROC Mean | AUROC Max "
                 "| Top Ch in Motor Cortex | N\u2081 | N\u2084 |")
    lines.append("|---------|-----------|-----------|-----------|----------|"
                 "----------------------|----|----|")
    for r in reports:
        cd = r.class_discriminability.get('binary', {})
        if cd:
            lines.append(
                f"| {r.subject_id} | {cd.get('fisher_mean', 0):.4f} | "
                f"{cd.get('fisher_max', 0):.4f} | "
                f"{cd.get('auroc_mean', 0.5):.4f} | "
                f"{cd.get('auroc_max', 0.5):.4f} | "
                f"{cd.get('top_in_motor_cortex', 0)}/10 | "
                f"{cd.get('n_class1', 0)} | {cd.get('n_class4', 0)} |"
            )
        else:
            lines.append(f"| {r.subject_id} | - | - | - | - | - | - | - |")
    lines.append("")

    lines.append("### 1.2 Top Discriminative Channels (by Fisher Ratio)\n")
    for r in reports:
        cd = r.class_discriminability.get('binary', {})
        top = cd.get('top_channels', [])
        if top:
            ch_str = ", ".join(f"{name}({fr:.3f})" for _, name, fr in top[:5])
            lines.append(f"- **{r.subject_id}**: {ch_str}")
    lines.append("")

    lines.append("### 1.3 All Class Pairs\n")
    lines.append("| Subject | Classes | All-Pairs Fisher | All-Pairs AUROC |")
    lines.append("|---------|---------|-----------------|----------------|")
    for r in reports:
        cd = r.class_discriminability
        classes = ", ".join(str(c) for c in cd.get('classes', []))
        lines.append(
            f"| {r.subject_id} | {classes} | "
            f"{cd.get('all_pairs_fisher_mean', 0):.4f} | "
            f"{cd.get('all_pairs_auroc_mean', 0.5):.4f} |"
        )
    lines.append("")

    # ---- 2. Temporal Drift ----
    lines.append("## 2. Temporal Drift (Within-Session Nonstationarity)\n")
    lines.append("L2 distance of per-run channel mean vectors from first-run baseline. "
                 "High drift indicates electrode impedance changes, fatigue, or movement.\n")
    lines.append("| Subject | Max Drift (L2) | Worst Session | Drift Rate | N Sessions |")
    lines.append("|---------|---------------|--------------|-----------|-----------|")
    for r in reports:
        td = r.temporal_drift
        rate = 0.0
        if td.get('per_session'):
            worst = td.get('max_drift_session', '')
            ps = td['per_session'].get(worst, {})
            rate = ps.get('drift_rate', 0)
        lines.append(
            f"| {r.subject_id} | {td.get('max_drift', 0):.4f} | "
            f"{td.get('max_drift_session', '-')} | "
            f"{rate:.4f} | "
            f"{td.get('n_sessions_analyzed', 0)} |"
        )
    lines.append("")

    # ---- 3. Spectral Features ----
    lines.append("## 3. Spectral Features\n")
    lines.append("Band power (uV^2/Hz, Welch PSD, averaged across channels and trials). "
                 "Data is bandpassed 4-40 Hz so only these bands are available.\n")
    lines.append("| Subject | Theta | Mu (8-13) | Low-\u03b2 (13-20) | "
                 "High-\u03b2 (20-30) | \u03b3 (30-40) | Mu/\u03b2 Ratio |")
    lines.append("|---------|-------|---------|-------------|"
                 "-------------|---------|-----------|")
    for r in reports:
        sf = r.spectral_features
        lines.append(
            f"| {r.subject_id} | {sf.get('theta_mean', 0):.4f} | "
            f"{sf.get('mu_mean', 0):.4f} | "
            f"{sf.get('low_beta_mean', 0):.4f} | "
            f"{sf.get('high_beta_mean', 0):.4f} | "
            f"{sf.get('gamma_mean', 0):.4f} | "
            f"{sf.get('mu_beta_ratio', 0):.3f} |"
        )
    lines.append("")

    # ---- 4. EMG Indicators ----
    lines.append("## 4. EMG Contamination Indicators\n")
    lines.append("Compares high-frequency (20-40 Hz) power between peripheral and "
                 "central (motor cortex) channels. Elevated peripheral high-freq power "
                 "suggests muscle artifact contamination.\n")
    lines.append("**Caveats**:\n"
                 "- Data is bandpassed 4-40 Hz, so only low-frequency EMG "
                 "artifacts (within passband) are detectable.\n"
                 "- Post-CAR peripheral channels naturally have higher residual power "
                 "(~2x) because they contribute less to the common average reference. "
                 "Only P/C ratio > 3.0 is flagged.\n")
    lines.append("| Subject | Central HF | Peripheral HF | P/C Ratio | "
                 "HF% Central | HF% Peripheral | Flag |")
    lines.append("|---------|-----------|-------------|---------|"
                 "-----------|--------------|------|")
    for r in reports:
        emg = r.emg_indicators
        flag = emg.get('flag', '-') or '-'
        lines.append(
            f"| {r.subject_id} | {emg.get('central_hf_mean', 0):.6f} | "
            f"{emg.get('peripheral_hf_mean', 0):.6f} | "
            f"{emg.get('peripheral_central_ratio', 1):.3f} | "
            f"{emg.get('hf_frac_central', 0):.1%} | "
            f"{emg.get('hf_frac_peripheral', 0):.1%} | "
            f"{flag} |"
        )
    lines.append("")

    # ---- 5. Adjacent Trial Autocorrelation ----
    lines.append("## 5. Adjacent Trial Autocorrelation\n")
    lines.append("Pearson correlation between consecutive trial channel-mean vectors "
                 "within runs. High correlation suggests trial overlap or slow drift.\n")
    lines.append("| Subject | Mean r | Median r | Max r | % > 0.5 | % > 0.8 | Flag |")
    lines.append("|---------|--------|---------|-------|---------|---------|------|")
    for r in reports:
        atc = r.adjacent_trial_corr
        flag = atc.get('flag', '-') or '-'
        lines.append(
            f"| {r.subject_id} | {atc.get('mean_corr', 0):.4f} | "
            f"{atc.get('median_corr', 0):.4f} | "
            f"{atc.get('max_corr', 0):.4f} | "
            f"{atc.get('pct_above_0.5', 0):.1f}% | "
            f"{atc.get('pct_above_0.8', 0):.1f}% | "
            f"{flag} |"
        )
    lines.append("")

    # ---- 6. Cross-Subject Similarity ----
    lines.append("## 6. Cross-Subject Similarity\n")
    lines.append("Euclidean distance in z-scored 10-feature space. "
                 "Useful for transfer learning: nearby subjects share similar "
                 "signal characteristics and may benefit from mutual pretraining.\n")
    lines.append(f"Features: {', '.join(FEATURE_NAMES)}\n")

    lines.append("### 6.1 Nearest Neighbors\n")
    lines.append("| Subject | 1st Nearest | 2nd Nearest | 3rd Nearest |")
    lines.append("|---------|------------|------------|------------|")
    nn = cross_subject.get('nearest_neighbors', {})
    for r in reports:
        neighbors = nn.get(r.subject_id, [])
        cols = []
        for subj, dist in neighbors[:3]:
            cols.append(f"{subj} ({dist:.2f})")
        while len(cols) < 3:
            cols.append("-")
        lines.append(f"| {r.subject_id} | {cols[0]} | {cols[1]} | {cols[2]} |")
    lines.append("")

    lines.append("### 6.2 Most Isolated Subjects\n")
    lines.append("| Rank | Subject | Mean Distance | Interpretation |")
    lines.append("|------|---------|--------------|----------------|")
    for rank, (subj, dist) in enumerate(cross_subject.get('most_isolated', []), 1):
        interp = 'outlier' if rank <= 3 else 'normal'
        lines.append(f"| {rank} | {subj} | {dist:.3f} | {interp} |")
    lines.append("")

    # Distance matrix
    subjects = cross_subject.get('subjects', [])
    dm = cross_subject.get('distance_matrix', np.array([]))
    if len(subjects) > 0 and dm.size > 0:
        lines.append("### 6.3 Distance Matrix\n")
        lines.append("```")
        header = "        " + " ".join(f"{s:>6}" for s in subjects)
        lines.append(header)
        for i, subj in enumerate(subjects):
            row = f"{subj:>6}  " + " ".join(f"{dm[i, j]:6.2f}" for j in range(len(subjects)))
            lines.append(row)
        lines.append("```\n")

    # Methodology
    lines.append("## Methodology\n")
    lines.append("- **Class discriminability**: Fisher's LDR and Mann-Whitney AUROC on "
                 "mu+beta band power per channel")
    lines.append("- **Temporal drift**: L2 distance of per-run channel mean vectors "
                 "from first run in each session")
    lines.append("- **Spectral features**: Welch PSD (nperseg=100, up to 500-trial subsample)")
    lines.append("- **EMG indicators**: High-frequency (20-40 Hz) power ratio, "
                 "peripheral vs central (motor cortex 32ch) channels")
    lines.append("- **Adjacent trial correlation**: Pearson r of channel-mean vectors "
                 "between consecutive trials within runs")
    lines.append("- **Cross-subject similarity**: Euclidean distance in z-scored "
                 "10-feature space")
    lines.append("- **Central region**: Motor cortex 32ch configuration "
                 "(C3/Cz/C4 + SMA/premotor, 32 channels)")
    lines.append("- **Peripheral region**: Remaining 96 channels")
    lines.append("")

    # Write report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\nReport saved to: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Advanced per-subject EEG data quality analysis',
    )
    parser.add_argument('--cache-index',
                        default='caches/preprocessed/.cache_index.json')
    parser.add_argument('--cache-dir',
                        default='caches/preprocessed')
    parser.add_argument('--output',
                        default='results/data_quality_advanced_report.md')
    parser.add_argument('--subjects', nargs='+', default=None,
                        help='Specific subjects to analyze (default: all)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Parallel workers')
    parser.add_argument('--paradigm', default='imagery',
                        choices=['imagery', 'movement'])
    parser.add_argument('--model', default='eegnet',
                        help='Cache model filter')
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()

    print("Advanced EEG Data Quality Analysis")
    print(f"{'='*40}")

    print(f"Loading cache index: {args.cache_index}")
    entries = load_cache_index(args.cache_index)
    print(f"  Total entries: {len(entries)}")

    grouped = group_entries_by_subject(entries, model=args.model,
                                       paradigm=args.paradigm)
    print(f"  Subjects with {args.model}/{args.paradigm} data: {len(grouped)}")

    if args.subjects:
        grouped = {s: v for s, v in grouped.items() if s in args.subjects}
        print(f"  Filtered to: {sorted(grouped.keys())}")

    subjects = sorted(grouped.keys())
    print(f"\nAnalyzing {len(subjects)} subjects...")

    start = time.time()

    reports = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                analyze_subject_advanced, subj, grouped[subj],
                args.cache_dir, args.verbose,
            ): subj
            for subj in subjects
        }

        for future in as_completed(futures):
            subj = futures[future]
            try:
                report = future.result()
                reports.append(report)
                cd = report.class_discriminability.get('binary', {})
                print(f"  {subj}: {report.total_trials} trials, "
                      f"Fisher={cd.get('fisher_mean', 0):.4f}, "
                      f"AUROC={cd.get('auroc_mean', 0.5):.4f}")
            except Exception as e:
                import traceback
                print(f"  [E] {subj}: ERROR - {e}")
                traceback.print_exc()

    elapsed_analysis = time.time() - start
    print(f"\nPer-subject analysis: {elapsed_analysis:.1f}s")

    print("Computing cross-subject similarity...")
    cross_subject = compute_cross_subject_similarity(reports)

    elapsed = time.time() - start

    generate_advanced_report(reports, cross_subject, args.output, elapsed)


if __name__ == '__main__':
    main()
