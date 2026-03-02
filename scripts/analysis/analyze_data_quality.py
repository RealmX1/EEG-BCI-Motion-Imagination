#!/usr/bin/env python
"""
Per-subject EEG data quality analysis.

Reads HDF5 cache files directly and checks each subject for:
- Signal quality (NaN/Inf, dead channels, extreme amplitudes, SNR)
- Statistical anomalies (trial variance, channel correlation, label distribution)
- Cross-session consistency (amplitude shift, variance stability)
- Contamination indicators (duplicate trials, train/test similarity)

Generates a Markdown report at results/data_quality_report.md.

Usage:
    uv run python scripts/analysis/analyze_data_quality.py
    uv run python scripts/analysis/analyze_data_quality.py --subjects S01 S06
    uv run python scripts/analysis/analyze_data_quality.py --workers 8
    uv run python scripts/analysis/analyze_data_quality.py -v
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.channel_selection import BIOSEMI_128_LABELS


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class SubjectReport:
    subject_id: str
    # Data overview
    total_trials: int = 0
    total_runs: int = 0
    sessions: Dict[str, int] = field(default_factory=dict)  # session -> n_trials

    # Signal quality
    nan_inf: Dict = field(default_factory=dict)
    dead_channels: Dict = field(default_factory=dict)
    extreme_amplitudes: Dict = field(default_factory=dict)
    snr: Dict = field(default_factory=dict)

    # Statistical anomalies
    inter_trial_variance: Dict = field(default_factory=dict)
    inter_channel_correlation: Dict = field(default_factory=dict)
    label_distribution: Dict = field(default_factory=dict)
    trial_counts: Dict = field(default_factory=dict)

    # Cross-session consistency
    session_amplitude_shift: Dict = field(default_factory=dict)
    session_variance_consistency: Dict = field(default_factory=dict)

    # Contamination
    duplicate_trials: Dict = field(default_factory=dict)
    train_test_similarity: Dict = field(default_factory=dict)

    # Aggregated
    flags: List[str] = field(default_factory=list)
    severity: str = 'clean'


# ============================================================================
# Data Loading
# ============================================================================

def load_cache_index(cache_index_path: str) -> dict:
    """Load .cache_index.json and return entries dict."""
    with open(cache_index_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('entries', data)


def group_entries_by_subject(
    entries: dict,
    model: str = 'eegnet',
    paradigm: str = 'imagery',
) -> Dict[str, Dict[str, List[dict]]]:
    """Group cache entries by subject -> session_folder -> list of entries.

    Only includes entries matching the specified model and paradigm.
    Each entry dict gets 'cache_key' added.
    """
    grouped = defaultdict(lambda: defaultdict(list))

    for key, meta in entries.items():
        if not isinstance(meta, dict):
            continue
        if meta.get('model') != model:
            continue
        if meta.get('subject_task_type') != paradigm:
            continue

        subject = meta.get('subject')
        session = meta.get('session_folder')
        if not subject or not session:
            continue

        entry = dict(meta)
        entry['cache_key'] = key
        grouped[subject][session].append(entry)

    return dict(grouped)


def load_subject_data(
    subject_entries: Dict[str, List[dict]],
    cache_dir: str,
) -> Dict[str, List[Tuple[np.ndarray, np.ndarray, int]]]:
    """Load all HDF5 data for one subject.

    Deduplicates by (session_folder, run) — keeps the entry with the
    most trials when multiple cache entries exist for the same run
    (e.g., different target_classes producing different cache keys).

    Returns:
        {session_folder: [(trials, labels, run_number), ...]}
        Each trials is [n_trials, 128, T].
    """
    result = defaultdict(list)

    for session, entries in subject_entries.items():
        # Deduplicate by run: keep entry with most trials (fullest data)
        run_entries = defaultdict(list)
        for entry in entries:
            run_entries[entry.get('run', 0)].append(entry)

        best_entries = []
        for run_num, run_entry_list in sorted(run_entries.items()):
            # Pick the entry with the largest shape[0] (most trials)
            best = max(run_entry_list,
                       key=lambda e: (e.get('shape', [0])[0] if e.get('shape') else 0))
            best_entries.append(best)

        for entry in best_entries:
            h5_path = Path(cache_dir) / f"{entry['cache_key']}.h5"
            if not h5_path.exists():
                continue
            try:
                with h5py.File(str(h5_path), 'r') as f:
                    trials = f['trials'][:]
                    labels = f['labels'][:]
                result[session].append((trials, labels, entry.get('run', 0)))
            except Exception:
                continue

    return dict(result)


# ============================================================================
# Signal Quality Checks
# ============================================================================

def check_nan_inf(data: Dict[str, List]) -> dict:
    """Check for NaN and Inf values across all trials.

    Distinguishes between:
    - NaN padding at the trailing end of time axis (expected, from variable-length trials)
    - NaN/Inf within the signal region (true contamination)
    """
    padding_nan_trials = 0
    signal_nan_trials = 0
    inf_count = 0
    total_trials = 0
    affected = []

    for session, runs in data.items():
        for trials, labels, run in runs:
            for t_idx in range(trials.shape[0]):
                total_trials += 1
                trial = trials[t_idx]  # [128, T]

                # Check Inf
                n_inf = np.isinf(trial).sum()
                if n_inf > 0:
                    inf_count += int(n_inf)
                    affected.append(f"{session}/R{run:02d}/T{t_idx}(inf)")

                # Check NaN: distinguish padding vs signal contamination
                nan_mask = np.isnan(trial)
                if not nan_mask.any():
                    continue

                # NaN padding = all channels are NaN at the same time indices
                # (trailing end of time axis)
                nan_per_timepoint = nan_mask.any(axis=0)  # [T]
                first_nan_t = np.argmax(nan_per_timepoint)

                if nan_per_timepoint[first_nan_t]:
                    # Check if all NaN is at trailing end
                    trailing_nan = nan_per_timepoint[first_nan_t:]
                    if trailing_nan.all():
                        # All NaN from first_nan_t onwards = padding
                        padding_nan_trials += 1
                    else:
                        # NaN scattered within signal
                        signal_nan_trials += 1
                        affected.append(f"{session}/R{run:02d}/T{t_idx}(signal_nan)")

    has_signal_nan = signal_nan_trials > 0
    has_inf = inf_count > 0

    return {
        'has_signal_nan': has_signal_nan,
        'has_inf': has_inf,
        'padding_nan_trials': padding_nan_trials,
        'signal_nan_trials': signal_nan_trials,
        'inf_count': inf_count,
        'total_trials': total_trials,
        'padding_pct': padding_nan_trials / max(total_trials, 1) * 100,
        'affected_runs': affected[:20],  # Cap output
    }


def check_dead_channels(data: Dict[str, List], var_threshold: float = 0.01) -> dict:
    """Detect channels with near-zero variance (flat lines).

    A channel is flagged if its variance < threshold in >50% of runs.
    """
    n_channels = 128
    channel_low_var_count = np.zeros(n_channels, dtype=int)
    total_runs = 0

    for session, runs in data.items():
        for trials, labels, run in runs:
            total_runs += 1
            # Mean variance across trials and time per channel
            # trials shape: [n_trials, n_channels, n_samples]
            ch_var = np.nanvar(trials, axis=2).mean(axis=0)  # [n_channels]
            channel_low_var_count[ch_var < var_threshold] += 1

    if total_runs == 0:
        return {'dead_channels': [], 'dead_pct': 0.0}

    threshold_runs = total_runs * 0.5
    dead_indices = np.where(channel_low_var_count > threshold_runs)[0].tolist()
    dead_labels = [BIOSEMI_128_LABELS[i] for i in dead_indices] if dead_indices else []

    return {
        'dead_channels': dead_indices,
        'dead_channel_labels': dead_labels,
        'dead_pct': len(dead_indices) / n_channels * 100,
        'total_runs_checked': total_runs,
    }


def check_extreme_amplitudes(data: Dict[str, List]) -> dict:
    """Detect channels/trials with abnormally high amplitudes.

    Excludes NaN-padded regions (uses only valid signal data).
    """
    all_maxabs = []

    for session, runs in data.items():
        for trials, labels, run in runs:
            # trials: [n_trials, 128, T] — may contain NaN padding
            for t_idx in range(trials.shape[0]):
                trial = trials[t_idx]  # [128, T]
                # Find valid (non-NaN) time range
                valid_mask = ~np.isnan(trial[0])  # Use ch0 as reference
                if not valid_mask.any():
                    continue
                valid_data = trial[:, valid_mask]  # [128, T_valid]
                ch_max = np.max(np.abs(valid_data), axis=1)  # [128]
                all_maxabs.append(ch_max)

    if not all_maxabs:
        return {'max_abs_amplitude': 0.0, 'amplitude_percentiles': {}}

    combined = np.array(all_maxabs)  # [total_trials, 128]
    global_flat = combined.flatten()

    p50 = float(np.percentile(global_flat, 50))
    p95 = float(np.percentile(global_flat, 95))
    p99 = float(np.percentile(global_flat, 99))
    p999 = float(np.percentile(global_flat, 99.9))
    max_val = float(np.max(global_flat))

    global_mean = float(np.mean(global_flat))
    global_std = float(np.std(global_flat))

    # Count trials with any channel exceeding 10 sigma
    threshold_10sig = global_mean + 10 * global_std
    n_extreme_trials = int((combined.max(axis=1) > threshold_10sig).sum())

    # Channels that frequently have extreme values (>10% of trials)
    n_total_trials = combined.shape[0]
    ch_extreme_counts = (combined > threshold_10sig).sum(axis=0)  # [128]
    extreme_channels = np.where(ch_extreme_counts > n_total_trials * 0.1)[0].tolist()

    return {
        'max_abs_amplitude': max_val,
        'amplitude_percentiles': {50: p50, 95: p95, 99: p99, 99.9: p999},
        'global_mean': global_mean,
        'global_std': global_std,
        'n_extreme_trials': n_extreme_trials,
        'extreme_channels': extreme_channels,
        'extreme_channel_labels': [BIOSEMI_128_LABELS[i] for i in extreme_channels],
    }


def check_snr(data: Dict[str, List]) -> dict:
    """Estimate SNR using inter-trial coherence per class.

    SNR = var(mean_ERP_across_trials) / mean(var_across_trials) per channel.
    Only uses Offline data (has all 4 classes).
    """
    # Collect trials by label from offline sessions
    class_trials = defaultdict(list)
    for session, runs in data.items():
        if 'Offline' not in session:
            continue
        for trials, labels, run in runs:
            for label_val in np.unique(labels):
                mask = labels == label_val
                class_trials[int(label_val)].append(trials[mask])

    if not class_trials:
        return {'mean_snr_db': float('nan'), 'per_class_snr': {}}

    per_class_snr = {}
    all_channel_snrs = []

    for label_val, trial_list in class_trials.items():
        all_t = np.concatenate(trial_list, axis=0)  # [N_class, 128, T]
        if all_t.shape[0] < 2:
            continue

        # Signal: variance of the mean ERP across trials
        mean_erp = np.nanmean(all_t, axis=0)  # [128, T]
        signal_var = np.nanvar(mean_erp, axis=1)  # [128]

        # Noise: mean of per-trial residual variance
        residuals = all_t - mean_erp[np.newaxis, :, :]
        noise_var = np.nanmean(np.nanvar(residuals, axis=2), axis=0)  # [128]

        ch_snr = signal_var / (noise_var + 1e-12)  # [128]
        snr_db = 10 * np.log10(ch_snr + 1e-12)

        per_class_snr[label_val] = float(np.nanmean(snr_db))
        all_channel_snrs.append(snr_db)

    if all_channel_snrs:
        mean_snr_db = float(np.nanmean(np.stack(all_channel_snrs)))
        per_channel_snr = np.nanmean(np.stack(all_channel_snrs), axis=0).tolist()
    else:
        mean_snr_db = float('nan')
        per_channel_snr = []

    return {
        'mean_snr_db': mean_snr_db,
        'per_class_snr': per_class_snr,
        'per_channel_snr': per_channel_snr,
    }


# ============================================================================
# Statistical Anomaly Checks
# ============================================================================

def check_inter_trial_variance(data: Dict[str, List]) -> dict:
    """Check trial-to-trial variance using coefficient of variation.

    Uses only valid (non-NaN) data per trial.
    """
    per_session_cv = {}

    all_norms = []
    for session, runs in data.items():
        session_norms = []
        for trials, labels, run in runs:
            for t in range(trials.shape[0]):
                trial = trials[t]
                # Exclude NaN padding
                valid = trial[:, ~np.isnan(trial[0])]
                if valid.size == 0:
                    continue
                norm = float(np.linalg.norm(valid))
                session_norms.append(norm)
                all_norms.append(norm)
        if session_norms:
            arr = np.array(session_norms)
            cv = float(np.std(arr) / (np.mean(arr) + 1e-12))
            per_session_cv[session] = cv

    overall_cv = 0.0
    flag = None
    if all_norms:
        arr = np.array(all_norms)
        overall_cv = float(np.std(arr) / (np.mean(arr) + 1e-12))
        if overall_cv < 0.05:
            flag = 'very_low'
        elif overall_cv > 2.0:
            flag = 'very_high'

    return {
        'overall_cv': overall_cv,
        'per_session_cv': per_session_cv,
        'flag': flag,
    }


def check_inter_channel_correlation(data: Dict[str, List], n_sample: int = 50) -> dict:
    """Check for abnormally high global inter-channel correlation.

    Samples n_sample trials and computes mean off-diagonal correlation.
    """
    # Collect all offline trials
    all_trials = []
    for session, runs in data.items():
        if 'Offline' not in session:
            continue
        for trials, labels, run in runs:
            all_trials.append(trials)

    if not all_trials:
        # Fall back to any available session
        for session, runs in data.items():
            for trials, labels, run in runs:
                all_trials.append(trials)

    if not all_trials:
        return {'mean_off_diagonal_corr': 0.0}

    combined = np.concatenate(all_trials, axis=0)
    n_total = combined.shape[0]

    # Sample trials
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(n_total, min(n_sample, n_total), replace=False)
    sampled = combined[sample_idx]  # [n_sample, 128, T]

    # Compute correlation matrix averaged across sampled trials
    corr_matrices = []
    for t in range(sampled.shape[0]):
        trial_data = sampled[t]  # [128, T]
        # Strip NaN padding (all channels share same NaN pattern from tail padding)
        valid_mask = ~np.isnan(trial_data[0])
        if np.sum(valid_mask) < 10:  # Need minimum samples for correlation
            continue
        trial_valid = trial_data[:, valid_mask]
        corr = np.corrcoef(trial_valid)  # [128, 128]
        if not np.any(np.isnan(corr)):
            corr_matrices.append(corr)

    if not corr_matrices:
        return {'mean_off_diagonal_corr': 0.0}

    mean_corr = np.mean(corr_matrices, axis=0)  # [128, 128]
    # Off-diagonal elements
    mask = ~np.eye(128, dtype=bool)
    off_diag = mean_corr[mask]

    mean_off_diag = float(np.mean(np.abs(off_diag)))
    max_off_diag = float(np.max(np.abs(off_diag)))
    n_high = int(np.sum(np.abs(off_diag) > 0.9))

    flag = None
    if mean_off_diag > 0.8:
        flag = 'very_high_correlation'

    return {
        'mean_off_diagonal_corr': mean_off_diag,
        'max_off_diagonal_corr': max_off_diag,
        'n_high_corr_pairs': n_high // 2,  # symmetric
        'flag': flag,
    }


def check_label_distribution(data: Dict[str, List]) -> dict:
    """Check class balance per session.

    Evaluates balance WITHIN each session type separately, since:
    - Offline data has all 4 labels (1,2,3,4)
    - Online 2class has only labels 1,4
    - Online 3class has labels 1,2,4
    Cross-session label counts are expected to be imbalanced.
    """
    per_session = {}
    overall = defaultdict(int)
    imbalanced_sessions = []

    for session, runs in data.items():
        session_dist = defaultdict(int)
        for trials, labels, run in runs:
            for lbl in labels:
                session_dist[int(lbl)] += 1
                overall[int(lbl)] += 1

        per_session[session] = dict(session_dist)

        # Check within-session balance
        if session_dist:
            counts = list(session_dist.values())
            if min(counts) > 0:
                ratio = max(counts) / min(counts)
                if ratio > 2.0:
                    imbalanced_sessions.append(f"{session} ({ratio:.1f}x)")

    flag = None
    if imbalanced_sessions:
        flag = 'within_session_imbalance'

    return {
        'per_session': per_session,
        'overall': dict(overall),
        'imbalanced_sessions': imbalanced_sessions,
        'flag': flag,
    }


def check_trial_counts(data: Dict[str, List]) -> dict:
    """Check for missing runs or unexpected trial counts."""
    per_session = {}
    total_trials = 0
    total_runs = 0
    anomalous_runs = []

    for session, runs in data.items():
        trials_per_run = []
        for trials, labels, run_num in runs:
            n = trials.shape[0]
            trials_per_run.append(n)
            total_trials += n
            total_runs += 1

            # Expected: Offline has ~20 trials/run, Online varies
            if 'Offline' in session and (n < 10 or n > 30):
                anomalous_runs.append(f"{session}/R{run_num:02d} ({n} trials)")

        per_session[session] = {
            'n_runs': len(runs),
            'trials_per_run': trials_per_run,
            'total_trials': sum(trials_per_run),
        }

    return {
        'total_trials': total_trials,
        'total_runs': total_runs,
        'per_session': per_session,
        'anomalous_runs': anomalous_runs,
    }


# ============================================================================
# Cross-Session Consistency Checks
# ============================================================================

def check_session_amplitude_shift(data: Dict[str, List]) -> dict:
    """Check for mean amplitude shifts between sessions."""
    session_means = {}

    for session, runs in data.items():
        all_means = []
        for trials, labels, run in runs:
            ch_mean = np.nanmean(trials, axis=(0, 2))  # [128]
            all_means.append(ch_mean)
        if all_means:
            session_means[session] = np.mean(all_means, axis=0)  # [128]

    if len(session_means) < 2:
        return {'pairwise_shifts': {}, 'max_shift': 0.0}

    sessions = sorted(session_means.keys())
    pairwise = {}
    max_shift = 0.0
    max_pair = ('', '')

    for i in range(len(sessions)):
        for j in range(i + 1, len(sessions)):
            s_a, s_b = sessions[i], sessions[j]
            dist = float(np.linalg.norm(session_means[s_a] - session_means[s_b]))
            pairwise[f"{s_a} vs {s_b}"] = dist
            if dist > max_shift:
                max_shift = dist
                max_pair = (s_a, s_b)

    return {
        'pairwise_shifts': pairwise,
        'max_shift': max_shift,
        'max_shift_pair': max_pair,
    }


def check_session_variance_consistency(data: Dict[str, List]) -> dict:
    """Check if per-channel variance is stable across sessions."""
    session_vars = {}

    for session, runs in data.items():
        all_vars = []
        for trials, labels, run in runs:
            ch_var = np.nanvar(trials, axis=(0, 2))  # [128]
            all_vars.append(ch_var)
        if all_vars:
            session_vars[session] = float(np.mean(all_vars))

    if len(session_vars) < 2:
        return {'per_session_mean_var': session_vars, 'variance_ratio': 1.0}

    values = list(session_vars.values())
    variance_ratio = max(values) / (min(values) + 1e-12)

    flag = None
    if variance_ratio > 10.0:
        flag = 'unstable_variance'

    return {
        'per_session_mean_var': session_vars,
        'variance_ratio': variance_ratio,
        'flag': flag,
    }


# ============================================================================
# Contamination Checks
# ============================================================================

def check_duplicate_trials(data: Dict[str, List], corr_threshold: float = 0.999) -> dict:
    """Detect duplicate or near-duplicate trials.

    Compares trials with the same label within a subject.
    Uses only valid (non-NaN) data regions.
    """
    # Group trials by label — flatten valid (non-NaN) portion only
    label_trials = defaultdict(list)
    label_sources = defaultdict(list)

    for session, runs in data.items():
        for trials, labels, run_num in runs:
            for t_idx in range(trials.shape[0]):
                lbl = int(labels[t_idx])
                trial = trials[t_idx]  # [128, T]
                valid = trial[:, ~np.isnan(trial[0])]
                if valid.size == 0:
                    continue
                label_trials[lbl].append(valid.flatten())
                label_sources[lbl].append(f"{session}/R{run_num:02d}/T{t_idx}")

    n_duplicates = 0
    duplicate_pairs = []

    for lbl, trial_list in label_trials.items():
        n = len(trial_list)
        if n < 2:
            continue

        # For efficiency, subsample if too many
        rng = np.random.RandomState(42)
        max_compare = min(n, 200)
        indices = rng.choice(n, max_compare, replace=False) if n > 200 else np.arange(n)

        # Pad to same length (different trials may have different valid lengths)
        max_len = max(len(trial_list[i]) for i in indices)
        vectors = np.zeros((len(indices), max_len), dtype=np.float32)
        for vi, i in enumerate(indices):
            vec = trial_list[i]
            vectors[vi, :len(vec)] = vec

        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = vectors / norms

        # Pairwise cosine similarity
        sim_matrix = normed @ normed.T

        for i_idx in range(len(indices)):
            for j_idx in range(i_idx + 1, len(indices)):
                if sim_matrix[i_idx, j_idx] > corr_threshold:
                    n_duplicates += 1
                    i_orig = indices[i_idx]
                    j_orig = indices[j_idx]
                    duplicate_pairs.append((
                        label_sources[lbl][i_orig],
                        label_sources[lbl][j_orig],
                        float(sim_matrix[i_idx, j_idx]),
                    ))

    return {
        'n_duplicates': n_duplicates,
        'duplicate_pairs': duplicate_pairs[:20],
        'flag': 'duplicates_found' if n_duplicates > 0 else None,
    }


def check_train_test_similarity(data: Dict[str, List]) -> dict:
    """Compare train vs test session data using KS test.

    Train sessions: OfflineImagery, Sess01, Sess02_Base
    Test sessions: Sess02_Finetune
    """
    train_data = []
    test_data = []

    for session, runs in data.items():
        is_test = 'Finetune' in session and 'Sess02' in session
        for trials, labels, run in runs:
            # Compute channel means using only valid (non-NaN) time samples
            ch_means = np.nanmean(trials, axis=2)  # [n_trials, 128]
            # Skip if all NaN
            valid_rows = ~np.isnan(ch_means).any(axis=1)
            ch_means = ch_means[valid_rows]
            if ch_means.shape[0] == 0:
                continue
            if is_test:
                test_data.append(ch_means)
            else:
                train_data.append(ch_means)

    if not train_data or not test_data:
        return {'mean_ks_statistic': float('nan'), 'interpretation': 'insufficient_data'}

    train_combined = np.concatenate(train_data, axis=0)  # [N_train, 128]
    test_combined = np.concatenate(test_data, axis=0)    # [N_test, 128]

    ks_stats = []
    ks_pvalues = []
    n_channels = min(train_combined.shape[1], test_combined.shape[1])

    for ch in range(n_channels):
        stat, pval = stats.ks_2samp(train_combined[:, ch], test_combined[:, ch])
        ks_stats.append(stat)
        ks_pvalues.append(pval)

    mean_ks = float(np.mean(ks_stats))
    mean_pval = float(np.mean(ks_pvalues))
    n_similar = int(sum(1 for p in ks_pvalues if p > 0.05))
    n_different = n_channels - n_similar

    # Interpretation
    if mean_ks < 0.1:
        interpretation = 'very_similar (potential concern)'
    elif mean_ks < 0.3:
        interpretation = 'moderately_similar (normal)'
    else:
        interpretation = 'different_distributions (expected)'

    return {
        'mean_ks_statistic': mean_ks,
        'mean_ks_pvalue': mean_pval,
        'n_channels_similar': n_similar,
        'n_channels_different': n_different,
        'interpretation': interpretation,
    }


# ============================================================================
# Subject Analysis
# ============================================================================

def analyze_subject(
    subject_id: str,
    subject_entries: Dict[str, List[dict]],
    cache_dir: str,
    verbose: bool = False,
) -> SubjectReport:
    """Run all quality checks for one subject."""
    report = SubjectReport(subject_id=subject_id)

    if verbose:
        print(f"  Loading data for {subject_id}...")

    data = load_subject_data(subject_entries, cache_dir)
    if not data:
        report.flags.append('no_data')
        report.severity = 'critical'
        return report

    # Count totals
    for session, runs in data.items():
        session_trials = sum(t.shape[0] for t, l, r in runs)
        report.sessions[session] = session_trials
        report.total_trials += session_trials
        report.total_runs += len(runs)

    if verbose:
        print(f"  {subject_id}: {report.total_trials} trials, {report.total_runs} runs, "
              f"{len(data)} sessions")

    # Run all checks
    report.nan_inf = check_nan_inf(data)
    report.dead_channels = check_dead_channels(data)
    report.extreme_amplitudes = check_extreme_amplitudes(data)
    report.snr = check_snr(data)
    report.inter_trial_variance = check_inter_trial_variance(data)
    report.inter_channel_correlation = check_inter_channel_correlation(data)
    report.label_distribution = check_label_distribution(data)
    report.trial_counts = check_trial_counts(data)
    report.session_amplitude_shift = check_session_amplitude_shift(data)
    report.session_variance_consistency = check_session_variance_consistency(data)
    report.duplicate_trials = check_duplicate_trials(data)
    report.train_test_similarity = check_train_test_similarity(data)

    # Collect flags — tiered by severity
    # Critical: data integrity issues
    if report.nan_inf.get('has_signal_nan') or report.nan_inf.get('has_inf'):
        report.flags.append('nan_inf_contamination')
    if report.duplicate_trials.get('n_duplicates', 0) > 0:
        report.flags.append(f"duplicate_trials ({report.duplicate_trials['n_duplicates']})")

    # Major: severe artifacts that could impact training
    max_amp = report.extreme_amplitudes.get('max_abs_amplitude', 0)
    if max_amp > 50000:
        report.flags.append(f"severe_artifacts (max={max_amp:.0f})")

    # Minor: moderate quality concerns
    n_extreme = report.extreme_amplitudes.get('n_extreme_trials', 0)
    extreme_pct = n_extreme / max(report.total_trials, 1) * 100
    if extreme_pct > 5 and max_amp <= 50000:
        report.flags.append(f"moderate_artifacts ({extreme_pct:.1f}% trials)")
    if report.dead_channels.get('dead_pct', 0) > 5:
        report.flags.append(f"dead_channels ({report.dead_channels['dead_pct']:.1f}%)")
    if report.inter_trial_variance.get('flag'):
        report.flags.append(f"trial_variance_{report.inter_trial_variance['flag']}")
    if report.inter_channel_correlation.get('flag'):
        report.flags.append('high_channel_correlation')
    if report.session_variance_consistency.get('flag'):
        var_ratio = report.session_variance_consistency.get('variance_ratio', 1)
        report.flags.append(f"unstable_variance ({var_ratio:.0f}x)")
    if report.label_distribution.get('flag'):
        report.flags.append(f"within_session_label_imbalance")
    if report.trial_counts.get('anomalous_runs'):
        report.flags.append('anomalous_trial_counts')

    # Determine severity
    critical_flags = {'nan_inf_contamination'}
    major_flags = set()

    # Check for critical
    has_critical = any(f in report.flags for f in critical_flags)
    has_severe_artifact = any('severe_artifacts' in f for f in report.flags)
    has_duplicates = any('duplicate_trials' in f for f in report.flags)

    if has_critical:
        report.severity = 'critical'
    elif has_severe_artifact:
        report.severity = 'major'
    elif has_duplicates or any(f for f in report.flags
                              if 'moderate_artifacts' in f
                              or 'dead_channels' in f
                              or 'trial_variance_very_high' in f):
        report.severity = 'minor'
    elif report.flags:
        report.severity = 'info'
    else:
        report.severity = 'clean'

    return report


# ============================================================================
# Cross-Subject Outlier Detection
# ============================================================================

def flag_outlier_subjects(reports: List[SubjectReport]) -> Dict[str, List[str]]:
    """Flag subjects that are statistical outliers compared to the group."""
    outlier_flags = defaultdict(list)

    # Metrics to compare
    snrs = [(r.subject_id, r.snr.get('mean_snr_db', float('nan'))) for r in reports]
    cvs = [(r.subject_id, r.inter_trial_variance.get('overall_cv', 0)) for r in reports]
    corrs = [(r.subject_id, r.inter_channel_correlation.get('mean_off_diagonal_corr', 0))
             for r in reports]
    max_amps = [(r.subject_id, r.extreme_amplitudes.get('max_abs_amplitude', 0))
                for r in reports]
    total_trials = [(r.subject_id, r.total_trials) for r in reports]

    for metric_name, values in [
        ('SNR', snrs),
        ('trial_CV', cvs),
        ('channel_corr', corrs),
        ('max_amplitude', max_amps),
        ('total_trials', total_trials),
    ]:
        valid = [(s, v) for s, v in values if not np.isnan(v)]
        if len(valid) < 3:
            continue

        vals = np.array([v for _, v in valid])
        mean = np.mean(vals)
        std = np.std(vals)
        if std < 1e-10:
            continue

        for subj, val in valid:
            z = abs(val - mean) / std
            if z > 2.5:
                direction = 'high' if val > mean else 'low'
                outlier_flags[subj].append(
                    f"{metric_name}_{direction} (z={z:.1f}, val={val:.2f}, "
                    f"group={mean:.2f}±{std:.2f})"
                )

    return dict(outlier_flags)


# ============================================================================
# Report Generation
# ============================================================================

def generate_report(
    reports: List[SubjectReport],
    outlier_flags: Dict[str, List[str]],
    output_path: str,
    elapsed: float,
):
    """Generate Markdown report and console summary."""
    reports.sort(key=lambda r: r.subject_id)

    # Console summary
    n_total = len(reports)
    n_clean = sum(1 for r in reports if r.severity == 'clean')
    n_info = sum(1 for r in reports if r.severity == 'info')
    n_minor = sum(1 for r in reports if r.severity == 'minor')
    n_major = sum(1 for r in reports if r.severity == 'major')
    n_critical = sum(1 for r in reports if r.severity == 'critical')

    print(f"\n{'='*60}")
    print(f"Data Quality Analysis Complete ({elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"  Clean:    {n_clean}/{n_total}")
    print(f"  Info:     {n_info}/{n_total}")
    print(f"  Minor:    {n_minor}/{n_total}")
    print(f"  Major:    {n_major}/{n_total}")
    print(f"  Critical: {n_critical}/{n_total}")

    if any(r.flags for r in reports):
        print(f"\nFlagged subjects:")
        for r in reports:
            if r.flags:
                print(f"  {r.subject_id} [{r.severity}]: {', '.join(r.flags)}")

    if outlier_flags:
        print(f"\nStatistical outliers:")
        for subj, flags in sorted(outlier_flags.items()):
            for f in flags:
                print(f"  {subj}: {f}")

    # Markdown report
    lines = []
    lines.append("# EEG Data Quality Report\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Cache: `caches/preprocessed/.cache_index.json` (EEGNet entries only)")
    lines.append(f"Subjects: {len(reports)} (S01-S21)")
    lines.append(f"Analysis time: {elapsed:.1f}s\n")

    # Executive summary
    lines.append("## Executive Summary\n")
    lines.append(f"- **Clean subjects**: {n_clean}/{n_total}")
    lines.append(f"- **Info (minor notes)**: {n_info}/{n_total}")
    lines.append(f"- **Minor issues**: {n_minor}/{n_total}")
    lines.append(f"- **Major issues**: {n_major}/{n_total}")
    lines.append(f"- **Critical issues**: {n_critical}/{n_total}\n")

    # Flagged subjects table
    flagged = [r for r in reports if r.flags]
    if flagged:
        lines.append("## Flagged Subjects\n")
        lines.append("| Subject | Severity | Issues |")
        lines.append("|---------|----------|--------|")
        for r in flagged:
            lines.append(f"| {r.subject_id} | {r.severity} | {', '.join(r.flags)} |")
        lines.append("")

    # Statistical outliers
    if outlier_flags:
        lines.append("## Statistical Outliers (|z| > 2.5 from group mean)\n")
        lines.append("| Subject | Metric | Detail |")
        lines.append("|---------|--------|--------|")
        for subj, flags in sorted(outlier_flags.items()):
            for f in flags:
                parts = f.split(' ', 1)
                lines.append(f"| {subj} | {parts[0]} | {parts[1] if len(parts) > 1 else ''} |")
        lines.append("")

    # Per-subject summary table
    lines.append("## Per-Subject Overview\n")
    lines.append("| Subject | Trials | Runs | Sessions | Dead Ch | Max|Amp| | SNR (dB) | CV | Dup | Severity |")
    lines.append("|---------|--------|------|----------|---------|---------|----------|------|-----|----------|")
    for r in reports:
        dead_n = len(r.dead_channels.get('dead_channels', []))
        max_amp = r.extreme_amplitudes.get('max_abs_amplitude', 0)
        snr_db = r.snr.get('mean_snr_db', float('nan'))
        cv = r.inter_trial_variance.get('overall_cv', 0)
        dup = r.duplicate_trials.get('n_duplicates', 0)
        snr_str = f"{snr_db:.1f}" if not np.isnan(snr_db) else "N/A"
        lines.append(
            f"| {r.subject_id} | {r.total_trials} | {r.total_runs} | "
            f"{len(r.sessions)} | {dead_n} | {max_amp:.1f} | "
            f"{snr_str} | {cv:.3f} | {dup} | {r.severity} |"
        )
    lines.append("")

    # Detailed sections
    # --- Signal Quality ---
    lines.append("## Detailed Analysis\n")
    lines.append("### 1. Signal Quality\n")

    # NaN/Inf
    lines.append("#### NaN/Inf Analysis\n")
    lines.append("Note: NaN padding at the trailing end of trials is **expected** "
                 "(variable-length trial padding). Only signal-region NaN/Inf is flagged.\n")
    lines.append("| Subject | Padding NaN Trials | Padding % | Signal NaN Trials | Inf Count |")
    lines.append("|---------|-------------------|-----------|-------------------|-----------|")
    for r in reports:
        ni = r.nan_inf
        lines.append(
            f"| {r.subject_id} | {ni.get('padding_nan_trials', 0)} | "
            f"{ni.get('padding_pct', 0):.1f}% | "
            f"{ni.get('signal_nan_trials', 0)} | {ni.get('inf_count', 0)} |"
        )
    lines.append("")

    any_signal_nan = any(r.nan_inf.get('has_signal_nan') or r.nan_inf.get('has_inf')
                         for r in reports)
    if any_signal_nan:
        lines.append("**WARNING**: Signal-region NaN or Inf detected in the following subjects:\n")
        for r in reports:
            if r.nan_inf.get('has_signal_nan') or r.nan_inf.get('has_inf'):
                lines.append(f"- {r.subject_id}: {', '.join(r.nan_inf['affected_runs'])}")
        lines.append("")
    else:
        lines.append("No signal-region NaN or Inf detected. All NaN values are expected padding.\n")

    # Dead channels
    lines.append("#### Dead/Flat Channels\n")
    any_dead = any(r.dead_channels.get('dead_channels') for r in reports)
    if any_dead:
        lines.append("| Subject | Dead Channels | Labels |")
        lines.append("|---------|---------------|--------|")
        for r in reports:
            if r.dead_channels.get('dead_channels'):
                lines.append(
                    f"| {r.subject_id} | {len(r.dead_channels['dead_channels'])} | "
                    f"{', '.join(r.dead_channels['dead_channel_labels'])} |"
                )
    else:
        lines.append("No dead channels detected (variance threshold: 0.01).\n")

    # Extreme amplitudes
    lines.append("#### Extreme Amplitudes\n")
    lines.append("| Subject | Max|Amp| | P50 | P95 | P99 | P99.9 | Extreme Trials |")
    lines.append("|---------|---------|-----|-----|-----|-------|----------------|")
    for r in reports:
        ea = r.extreme_amplitudes
        p = ea.get('amplitude_percentiles', {})
        lines.append(
            f"| {r.subject_id} | {ea.get('max_abs_amplitude', 0):.1f} | "
            f"{p.get(50, 0):.1f} | {p.get(95, 0):.1f} | "
            f"{p.get(99, 0):.1f} | {p.get(99.9, 0):.1f} | "
            f"{ea.get('n_extreme_trials', 0)} |"
        )
    lines.append("")

    # SNR
    lines.append("#### Signal-to-Noise Ratio (dB)\n")
    lines.append("| Subject | Mean SNR | Thumb | Index | Middle | Pinky |")
    lines.append("|---------|----------|-------|-------|--------|-------|")
    for r in reports:
        snr_data = r.snr
        mean_snr = snr_data.get('mean_snr_db', float('nan'))
        pcs = snr_data.get('per_class_snr', {})
        mean_str = f"{mean_snr:.2f}" if not np.isnan(mean_snr) else "N/A"
        t_str = f"{pcs.get(1, float('nan')):.2f}" if 1 in pcs else "N/A"
        i_str = f"{pcs.get(2, float('nan')):.2f}" if 2 in pcs else "N/A"
        m_str = f"{pcs.get(3, float('nan')):.2f}" if 3 in pcs else "N/A"
        p_str = f"{pcs.get(4, float('nan')):.2f}" if 4 in pcs else "N/A"
        lines.append(f"| {r.subject_id} | {mean_str} | {t_str} | {i_str} | {m_str} | {p_str} |")
    lines.append("")

    # --- Statistical Anomalies ---
    lines.append("### 2. Statistical Anomalies\n")

    # Inter-trial variance
    lines.append("#### Inter-Trial Variance (Coefficient of Variation)\n")
    lines.append("| Subject | Overall CV | Flag |")
    lines.append("|---------|-----------|------|")
    for r in reports:
        itv = r.inter_trial_variance
        flag = itv.get('flag', '-') or '-'
        lines.append(f"| {r.subject_id} | {itv.get('overall_cv', 0):.4f} | {flag} |")
    lines.append("")

    # Inter-channel correlation
    lines.append("#### Inter-Channel Correlation\n")
    lines.append("| Subject | Mean |r| | Max |r| | High Pairs (>0.9) | Flag |")
    lines.append("|---------|---------|---------|-------------------|------|")
    for r in reports:
        icc = r.inter_channel_correlation
        flag = icc.get('flag', '-') or '-'
        lines.append(
            f"| {r.subject_id} | {icc.get('mean_off_diagonal_corr', 0):.4f} | "
            f"{icc.get('max_off_diagonal_corr', 0):.4f} | "
            f"{icc.get('n_high_corr_pairs', 0)} | {flag} |"
        )
    lines.append("")

    # Label distribution
    lines.append("#### Label Distribution\n")
    lines.append("Note: Cross-session label imbalance is **expected** (Offline has 4 classes, "
                 "Online 2class has only classes 1,4). Checking within-session balance.\n")
    lines.append("| Subject | Overall Distribution | Within-Session Imbalance |")
    lines.append("|---------|---------------------|--------------------------|")
    for r in reports:
        ld = r.label_distribution
        dist_str = ", ".join(f"{k}:{v}" for k, v in sorted(ld.get('overall', {}).items()))
        imb = ', '.join(ld.get('imbalanced_sessions', [])) or '-'
        lines.append(f"| {r.subject_id} | {dist_str} | {imb} |")
    lines.append("")

    # Trial counts
    lines.append("#### Trial Counts per Session\n")
    lines.append("| Subject | Total Trials | Total Runs | Sessions | Anomalous Runs |")
    lines.append("|---------|-------------|------------|----------|----------------|")
    for r in reports:
        tc = r.trial_counts
        anomalous = ', '.join(tc.get('anomalous_runs', [])) or '-'
        sessions = ', '.join(f"{s}({d['n_runs']})" for s, d in
                            sorted(tc.get('per_session', {}).items()))
        lines.append(
            f"| {r.subject_id} | {tc.get('total_trials', 0)} | "
            f"{tc.get('total_runs', 0)} | {sessions} | {anomalous} |"
        )
    lines.append("")

    # --- Cross-Session Consistency ---
    lines.append("### 3. Cross-Session Consistency\n")

    # Amplitude shift
    lines.append("#### Session Amplitude Shift (L2 distance of channel means)\n")
    lines.append("| Subject | Max Shift | Max Shift Pair |")
    lines.append("|---------|-----------|---------------|")
    for r in reports:
        sas = r.session_amplitude_shift
        pair = sas.get('max_shift_pair', ('', ''))
        pair_str = f"{pair[0]} vs {pair[1]}" if pair[0] else '-'
        lines.append(
            f"| {r.subject_id} | {sas.get('max_shift', 0):.4f} | {pair_str} |"
        )
    lines.append("")

    # Variance consistency
    lines.append("#### Session Variance Consistency\n")
    lines.append("| Subject | Variance Ratio (max/min) | Flag |")
    lines.append("|---------|------------------------|------|")
    for r in reports:
        svc = r.session_variance_consistency
        flag = svc.get('flag', '-') or '-'
        lines.append(
            f"| {r.subject_id} | {svc.get('variance_ratio', 1.0):.2f} | {flag} |"
        )
    lines.append("")

    # --- Contamination Checks ---
    lines.append("### 4. Contamination Checks\n")

    # Duplicates
    lines.append("#### Duplicate Trials (cosine similarity > 0.999)\n")
    any_dup = any(r.duplicate_trials.get('n_duplicates', 0) > 0 for r in reports)
    if any_dup:
        lines.append("| Subject | N Duplicates | Sample Pairs |")
        lines.append("|---------|-------------|-------------|")
        for r in reports:
            dt = r.duplicate_trials
            if dt.get('n_duplicates', 0) > 0:
                sample = "; ".join(f"{a} ~ {b} ({c:.4f})"
                                   for a, b, c in dt.get('duplicate_pairs', [])[:3])
                lines.append(f"| {r.subject_id} | {dt['n_duplicates']} | {sample} |")
    else:
        lines.append("No duplicate trials detected.\n")

    # Train/test similarity
    lines.append("#### Train/Test Distribution Similarity (KS Test)\n")
    lines.append("| Subject | Mean KS Stat | Similar Ch | Different Ch | Interpretation |")
    lines.append("|---------|-------------|------------|--------------|----------------|")
    for r in reports:
        tts = r.train_test_similarity
        ks = tts.get('mean_ks_statistic', float('nan'))
        ks_str = f"{ks:.4f}" if not np.isnan(ks) else "N/A"
        interp = tts.get('interpretation', 'N/A')
        lines.append(
            f"| {r.subject_id} | {ks_str} | "
            f"{tts.get('n_channels_similar', 0)} | "
            f"{tts.get('n_channels_different', 0)} | {interp} |"
        )
    lines.append("")

    # Methodology
    lines.append("## Methodology\n")
    lines.append("- **Data source**: HDF5 cache files (post-CAR, post-bandpass 4-40 Hz, "
                 "downsampled to 100 Hz, pre-z-score)")
    lines.append("- **Model filter**: EEGNet entries only (avoid double-counting with CBraMod)")
    lines.append("- **Dead channel threshold**: Variance < 0.01 in >50% of runs")
    lines.append("- **Extreme amplitude threshold**: |value| > mean + 10*std")
    lines.append("- **SNR**: Inter-trial coherence (signal = ERP variance, noise = residual variance)")
    lines.append("- **Duplicate detection**: Cosine similarity > 0.999 between same-label trials")
    lines.append("- **Train/test similarity**: 2-sample KS test per channel on trial-mean amplitudes")
    lines.append("- **Outlier detection**: |z-score| > 2.5 from group mean across subjects")
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
        description='Per-subject EEG data quality analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--cache-index',
                        default='caches/preprocessed/.cache_index.json')
    parser.add_argument('--cache-dir',
                        default='caches/preprocessed')
    parser.add_argument('--output',
                        default='results/data_quality_report.md')
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

    print(f"EEG Data Quality Analysis")
    print(f"{'='*40}")

    # Load cache index
    print(f"Loading cache index: {args.cache_index}")
    entries = load_cache_index(args.cache_index)
    print(f"  Total entries: {len(entries)}")

    # Group by subject
    grouped = group_entries_by_subject(entries, model=args.model, paradigm=args.paradigm)
    print(f"  Subjects with {args.model}/{args.paradigm} data: {len(grouped)}")

    # Filter subjects if specified
    if args.subjects:
        grouped = {s: v for s, v in grouped.items() if s in args.subjects}
        print(f"  Filtered to: {sorted(grouped.keys())}")

    subjects = sorted(grouped.keys())
    print(f"\nAnalyzing {len(subjects)} subjects...")

    start = time.time()

    # Analyze subjects in parallel
    reports = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                analyze_subject, subj, grouped[subj], args.cache_dir, args.verbose
            ): subj
            for subj in subjects
        }

        for future in as_completed(futures):
            subj = futures[future]
            try:
                report = future.result()
                reports.append(report)
                severity_icon = {
                    'clean': '+', 'info': 'i', 'minor': '~',
                    'major': '!', 'critical': 'X',
                }.get(report.severity, '?')
                print(f"  [{severity_icon}] {subj}: {report.total_trials} trials, "
                      f"{report.severity}"
                      + (f" — {', '.join(report.flags)}" if report.flags else ""))
            except Exception as e:
                print(f"  [E] {subj}: ERROR — {e}")

    elapsed = time.time() - start

    # Cross-subject outlier detection
    outlier_flags = flag_outlier_subjects(reports)

    # Generate report
    generate_report(reports, outlier_flags, args.output, elapsed)


if __name__ == '__main__':
    main()
