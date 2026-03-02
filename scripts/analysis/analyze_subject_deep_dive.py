#!/usr/bin/env python
"""
Deep-dive analysis for specific subjects.

Provides per-session, per-run, per-channel granularity for diagnosing
data quality issues. Designed for focused investigation of flagged subjects.

Usage:
    uv run python scripts/analysis/analyze_subject_deep_dive.py --subjects S10 S20 S05 S21
"""

import argparse
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import signal, stats

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analyze_data_quality import (
    load_cache_index,
    group_entries_by_subject,
    load_subject_data,
)
from src.preprocessing.channel_selection import BIOSEMI_128_LABELS

# ============================================================================
# Constants
# ============================================================================

FS = 100
BANDS = {
    'theta': (4, 8),
    'mu': (8, 13),
    'low_beta': (13, 20),
    'high_beta': (20, 30),
    'gamma': (30, 40),
}
# Motor cortex 32ch indices
CENTRAL_CH = set([
    0, 2, 3, 5, 20, 32, 33, 34, 49, 50, 52, 53, 55,
    62, 63, 64, 65, 66, 77, 85, 86, 90, 97, 107, 108,
    110, 111, 112, 113, 114, 116, 123,
])
PERIPHERAL_CH = set(range(128)) - CENTRAL_CH

# Session ordering (logical temporal order)
SESSION_ORDER = [
    'OfflineImagery',
    'OnlineImagery_Sess01_2class_Base',
    'OnlineImagery_Sess01_2class_Finetune',
    'OnlineImagery_Sess01_3class_Base',
    'OnlineImagery_Sess01_3class_Finetune',
    'OnlineImagery_Sess02_2class_Base',
    'OnlineImagery_Sess02_2class_Finetune',
    'OnlineImagery_Sess02_3class_Base',
    'OnlineImagery_Sess02_3class_Finetune',
]


def _get_valid(trial):
    """Extract valid (non-NaN) portion of a trial [128, T]."""
    valid_mask = ~np.isnan(trial[0])
    if not valid_mask.any():
        return None
    return trial[:, valid_mask]


def _compute_bandpower(valid_trial, fs=FS):
    """Compute per-channel band power [128, T_valid] -> {band: [128]}."""
    n_samples = valid_trial.shape[1]
    if n_samples < 50:
        return None
    nperseg = min(n_samples, 100)
    freqs, psd = signal.welch(valid_trial, fs=fs, nperseg=nperseg, axis=1)
    result = {}
    for band_name, (flo, fhi) in BANDS.items():
        mask = (freqs >= flo) & (freqs <= fhi)
        result[band_name] = np.mean(psd[:, mask], axis=1) if mask.any() else np.zeros(128)
    return result


def sort_sessions(sessions):
    """Sort session names by logical temporal order."""
    order_map = {s: i for i, s in enumerate(SESSION_ORDER)}
    return sorted(sessions, key=lambda s: order_map.get(s, 999))


# ============================================================================
# Per-Session Breakdown
# ============================================================================

def per_session_stats(data: Dict[str, List[Tuple]]) -> Dict[str, dict]:
    """Compute comprehensive per-session statistics."""
    results = {}
    for session in sort_sessions(data.keys()):
        runs = data[session]
        all_trials = np.concatenate([t for t, l, r in runs], axis=0)
        all_labels = np.concatenate([l for t, l, r in runs], axis=0)

        # Amplitude stats (ignoring NaN)
        valid_vals = all_trials[~np.isnan(all_trials)]
        amp_max = float(np.max(np.abs(valid_vals))) if valid_vals.size > 0 else 0
        amp_mean = float(np.mean(np.abs(valid_vals))) if valid_vals.size > 0 else 0
        amp_p99 = float(np.percentile(np.abs(valid_vals), 99)) if valid_vals.size > 0 else 0
        amp_p999 = float(np.percentile(np.abs(valid_vals), 99.9)) if valid_vals.size > 0 else 0

        # Per-channel variance (mean across trials, per-sample)
        ch_var = np.nanvar(all_trials, axis=(0, 2))  # [128]

        # Artifact trial count (any sample > threshold)
        artifact_threshold = 500
        n_artifact = 0
        for i in range(all_trials.shape[0]):
            valid = _get_valid(all_trials[i])
            if valid is not None and np.max(np.abs(valid)) > artifact_threshold:
                n_artifact += 1

        # Label distribution
        unique, counts = np.unique(all_labels.astype(int), return_counts=True)
        label_dist = dict(zip(unique.tolist(), counts.tolist()))

        # SNR estimate: class-conditioned mean variance / residual variance
        snr_db = float('nan')
        if len(label_dist) >= 2:
            class_means = []
            for lbl in sorted(label_dist.keys()):
                mask = all_labels.astype(int) == lbl
                if mask.sum() > 0:
                    class_means.append(np.nanmean(all_trials[mask], axis=0))
            if len(class_means) >= 2:
                signal_var = np.nanvar(np.array(class_means), axis=0).mean()
                noise_var = np.nanvar(all_trials, axis=0).mean()
                if noise_var > 0:
                    snr_db = float(10 * np.log10(signal_var / noise_var + 1e-20))

        results[session] = {
            'n_runs': len(runs),
            'n_trials': all_trials.shape[0],
            'labels': label_dist,
            'amp_max': amp_max,
            'amp_mean': amp_mean,
            'amp_p99': amp_p99,
            'amp_p999': amp_p999,
            'ch_var_mean': float(np.mean(ch_var)),
            'ch_var_max': float(np.max(ch_var)),
            'ch_var_min': float(np.min(ch_var)),
            'n_artifact_trials': n_artifact,
            'pct_artifact': n_artifact / all_trials.shape[0] * 100 if all_trials.shape[0] > 0 else 0,
            'snr_db': snr_db,
        }

    return results


# ============================================================================
# Per-Run Amplitude Profile
# ============================================================================

def per_run_amplitude_profile(data: Dict[str, List[Tuple]]) -> List[dict]:
    """Compute per-run amplitude statistics for identifying bad runs."""
    profiles = []
    for session in sort_sessions(data.keys()):
        runs = sorted(data[session], key=lambda x: x[2])
        for trials, labels, run_num in runs:
            valid_vals = trials[~np.isnan(trials)]
            amp_max = float(np.max(np.abs(valid_vals))) if valid_vals.size > 0 else 0
            amp_mean = float(np.mean(np.abs(valid_vals))) if valid_vals.size > 0 else 0
            amp_std = float(np.std(valid_vals)) if valid_vals.size > 0 else 0

            # Per-trial max amplitude
            trial_maxes = []
            for i in range(trials.shape[0]):
                v = _get_valid(trials[i])
                if v is not None:
                    trial_maxes.append(float(np.max(np.abs(v))))
            trial_maxes = np.array(trial_maxes) if trial_maxes else np.array([0.0])

            profiles.append({
                'session': session,
                'run': run_num,
                'n_trials': trials.shape[0],
                'amp_max': amp_max,
                'amp_mean': amp_mean,
                'amp_std': amp_std,
                'trial_max_median': float(np.median(trial_maxes)),
                'trial_max_p95': float(np.percentile(trial_maxes, 95)),
                'trial_max_max': float(np.max(trial_maxes)),
                'n_trials_above_500': int(np.sum(trial_maxes > 500)),
                'n_trials_above_1000': int(np.sum(trial_maxes > 1000)),
            })

    return profiles


# ============================================================================
# Channel-Level Analysis
# ============================================================================

def channel_quality_map(data: Dict[str, List[Tuple]]) -> dict:
    """Per-channel quality analysis: variance, artifact contribution, band power."""
    # Aggregate all trials
    all_trials_list = []
    for session, runs in data.items():
        for trials, labels, run in runs:
            for i in range(trials.shape[0]):
                all_trials_list.append(trials[i])

    n_total = len(all_trials_list)
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(n_total, min(n_total, 500), replace=False)

    ch_variances = []
    ch_max_amps = []
    ch_bandpowers = {b: [] for b in BANDS}

    for idx in sample_idx:
        valid = _get_valid(all_trials_list[idx])
        if valid is None:
            continue
        ch_variances.append(np.var(valid, axis=1))       # [128]
        ch_max_amps.append(np.max(np.abs(valid), axis=1)) # [128]
        bp = _compute_bandpower(valid)
        if bp:
            for b in BANDS:
                ch_bandpowers[b].append(bp[b])

    if not ch_variances:
        return {}

    ch_var_arr = np.array(ch_variances)     # [N, 128]
    ch_max_arr = np.array(ch_max_amps)      # [N, 128]

    mean_var = np.mean(ch_var_arr, axis=0)  # [128]
    mean_max = np.mean(ch_max_arr, axis=0)  # [128]

    # Find worst channels (by variance)
    worst_by_var = np.argsort(mean_var)[::-1][:10]
    # Find worst channels (by max amplitude)
    worst_by_amp = np.argsort(mean_max)[::-1][:10]

    # Compute mu/beta ratio per channel
    mu_arr = np.mean(ch_bandpowers['mu'], axis=0) if ch_bandpowers['mu'] else np.zeros(128)
    beta_arr = (np.mean(ch_bandpowers['low_beta'], axis=0) +
                np.mean(ch_bandpowers['high_beta'], axis=0)) if ch_bandpowers['low_beta'] else np.zeros(128)
    mu_beta_ratio = mu_arr / (beta_arr + 1e-12)

    # Central vs peripheral stats
    central_idx = sorted(CENTRAL_CH)
    peripheral_idx = sorted(PERIPHERAL_CH)

    return {
        'worst_channels_by_variance': [
            (int(i), BIOSEMI_128_LABELS[i], float(mean_var[i]), 'central' if i in CENTRAL_CH else 'peripheral')
            for i in worst_by_var
        ],
        'worst_channels_by_amplitude': [
            (int(i), BIOSEMI_128_LABELS[i], float(mean_max[i]), 'central' if i in CENTRAL_CH else 'peripheral')
            for i in worst_by_amp
        ],
        'central_mean_var': float(np.mean(mean_var[central_idx])),
        'peripheral_mean_var': float(np.mean(mean_var[peripheral_idx])),
        'central_mean_amp': float(np.mean(mean_max[central_idx])),
        'peripheral_mean_amp': float(np.mean(mean_max[peripheral_idx])),
        'mu_beta_ratio_central_mean': float(np.mean(mu_beta_ratio[central_idx])),
        'mu_beta_ratio_peripheral_mean': float(np.mean(mu_beta_ratio[peripheral_idx])),
        'n_trials_sampled': len(ch_variances),
    }


# ============================================================================
# Per-Session Class Discriminability
# ============================================================================

def per_session_discriminability(data: Dict[str, List[Tuple]]) -> Dict[str, dict]:
    """Compute class discriminability per session (not aggregated)."""
    results = {}

    for session in sort_sessions(data.keys()):
        runs = data[session]
        class_features = defaultdict(list)

        for trials, labels, run in runs:
            for t_idx in range(trials.shape[0]):
                lbl = int(labels[t_idx])
                valid = _get_valid(trials[t_idx])
                if valid is None:
                    continue
                bp = _compute_bandpower(valid)
                if bp is None:
                    continue
                feature = bp['mu'] + bp['low_beta'] + bp['high_beta']
                class_features[lbl].append(feature)

        avail = sorted(class_features.keys())
        if len(avail) < 2:
            results[session] = {
                'n_classes': len(avail),
                'fisher_mean': 0.0,
                'auroc_mean': 0.5,
                'class_counts': {c: len(class_features[c]) for c in avail},
            }
            continue

        # All-pairs Fisher + AUROC
        all_fisher = []
        all_auroc = []
        for i, ci in enumerate(avail):
            for cj in avail[i + 1:]:
                c1 = np.array(class_features[ci])
                c2 = np.array(class_features[cj])
                mu1, mu2 = np.mean(c1, axis=0), np.mean(c2, axis=0)
                var1, var2 = np.var(c1, axis=0), np.var(c2, axis=0)
                fisher = (mu1 - mu2) ** 2 / (var1 + var2 + 1e-12)
                all_fisher.append(np.mean(fisher))

                # Quick AUROC on top-5 Fisher channels
                top5 = np.argsort(fisher)[::-1][:5]
                aurocs = []
                for ch in top5:
                    try:
                        U, _ = stats.mannwhitneyu(c1[:, ch], c2[:, ch], alternative='two-sided')
                        auc = U / (len(c1) * len(c2))
                        aurocs.append(max(auc, 1 - auc))
                    except ValueError:
                        aurocs.append(0.5)
                all_auroc.append(np.mean(aurocs))

        results[session] = {
            'n_classes': len(avail),
            'fisher_mean': float(np.mean(all_fisher)),
            'auroc_mean': float(np.mean(all_auroc)),
            'class_counts': {c: len(class_features[c]) for c in avail},
        }

    return results


# ============================================================================
# Per-Session Spectral Profile
# ============================================================================

def per_session_spectral(data: Dict[str, List[Tuple]]) -> Dict[str, dict]:
    """Compute band power per session for drift comparison."""
    results = {}
    rng = np.random.RandomState(42)

    for session in sort_sessions(data.keys()):
        runs = data[session]
        all_trials = []
        for trials, labels, run in runs:
            for i in range(trials.shape[0]):
                all_trials.append(trials[i])

        max_t = min(len(all_trials), 200)
        indices = rng.choice(len(all_trials), max_t, replace=False) if len(all_trials) > 200 else range(len(all_trials))

        band_vals = {b: [] for b in BANDS}
        for idx in indices:
            valid = _get_valid(all_trials[idx])
            if valid is None:
                continue
            bp = _compute_bandpower(valid)
            if bp is None:
                continue
            for b in BANDS:
                band_vals[b].append(np.mean(bp[b]))

        n = len(band_vals['theta'])
        if n == 0:
            continue

        results[session] = {
            b: float(np.mean(band_vals[b])) for b in BANDS
        }
        mu = results[session].get('mu', 1e-12)
        beta = results[session].get('low_beta', 0) + results[session].get('high_beta', 0)
        results[session]['mu_beta_ratio'] = float(mu / (beta + 1e-12))
        results[session]['n_trials_sampled'] = n

    return results


# ============================================================================
# Train/Test Distribution Comparison (detailed)
# ============================================================================

def train_test_comparison(data: Dict[str, List[Tuple]]) -> dict:
    """Detailed train/test distribution comparison with KS test per channel."""
    train_sessions = set()
    test_sessions = set()
    for session in data.keys():
        # Only Sess02 Finetune sessions are test (matches data split protocol)
        if 'Finetune' in session and 'Sess02' in session:
            test_sessions.add(session)
        else:
            train_sessions.add(session)

    if not train_sessions or not test_sessions:
        return {'error': 'missing train or test sessions'}

    # Collect per-channel mean values
    train_ch_means = []
    test_ch_means = []

    for session, runs in data.items():
        target = train_ch_means if session in train_sessions else test_ch_means
        for trials, labels, run in runs:
            for i in range(trials.shape[0]):
                valid = _get_valid(trials[i])
                if valid is not None:
                    target.append(np.mean(valid, axis=1))  # [128]

    if not train_ch_means or not test_ch_means:
        return {'error': 'insufficient data'}

    train_arr = np.array(train_ch_means)  # [N_train, 128]
    test_arr = np.array(test_ch_means)    # [N_test, 128]

    # Per-channel KS test
    ks_stats = []
    ks_pvals = []
    for ch in range(128):
        stat, pval = stats.ks_2samp(train_arr[:, ch], test_arr[:, ch])
        ks_stats.append(stat)
        ks_pvals.append(pval)

    ks_arr = np.array(ks_stats)
    pv_arr = np.array(ks_pvals)

    # Channels where train/test are statistically indistinguishable (p > 0.05)
    n_indist = int(np.sum(pv_arr > 0.05))

    # Find most similar and most different channels
    most_similar = np.argsort(ks_arr)[:5]
    most_different = np.argsort(ks_arr)[::-1][:5]

    return {
        'n_train_trials': len(train_ch_means),
        'n_test_trials': len(test_ch_means),
        'mean_ks_stat': float(np.mean(ks_arr)),
        'median_ks_stat': float(np.median(ks_arr)),
        'max_ks_stat': float(np.max(ks_arr)),
        'n_indistinguishable_channels': n_indist,
        'most_similar_channels': [
            (int(i), BIOSEMI_128_LABELS[i], float(ks_arr[i]), float(pv_arr[i]))
            for i in most_similar
        ],
        'most_different_channels': [
            (int(i), BIOSEMI_128_LABELS[i], float(ks_arr[i]), float(pv_arr[i]))
            for i in most_different
        ],
        'train_global_mean': float(np.mean(train_arr)),
        'test_global_mean': float(np.mean(test_arr)),
        'train_global_std': float(np.std(train_arr)),
        'test_global_std': float(np.std(test_arr)),
    }


# ============================================================================
# Salvageability Analysis (for bad subjects)
# ============================================================================

def salvageability_analysis(data: Dict[str, List[Tuple]], amp_threshold=500) -> dict:
    """Assess what fraction of data is salvageable after artifact rejection.

    For each session, compute what percentage of trials would survive
    a max-amplitude threshold rejection.
    """
    per_session = {}
    total_trials = 0
    total_clean = 0

    for session in sort_sessions(data.keys()):
        runs = data[session]
        n_trials = 0
        n_clean = 0
        trial_max_amps = []

        for trials, labels, run in runs:
            for i in range(trials.shape[0]):
                n_trials += 1
                valid = _get_valid(trials[i])
                if valid is None:
                    continue
                mx = float(np.max(np.abs(valid)))
                trial_max_amps.append(mx)
                if mx <= amp_threshold:
                    n_clean += 1

        total_trials += n_trials
        total_clean += n_clean

        per_session[session] = {
            'n_trials': n_trials,
            'n_clean': n_clean,
            'pct_clean': n_clean / n_trials * 100 if n_trials > 0 else 0,
            'amp_p50': float(np.median(trial_max_amps)) if trial_max_amps else 0,
            'amp_p90': float(np.percentile(trial_max_amps, 90)) if trial_max_amps else 0,
            'amp_p99': float(np.percentile(trial_max_amps, 99)) if trial_max_amps else 0,
        }

    return {
        'threshold': amp_threshold,
        'total_trials': total_trials,
        'total_clean': total_clean,
        'pct_clean': total_clean / total_trials * 100 if total_trials > 0 else 0,
        'per_session': per_session,
    }


# ============================================================================
# Main Analysis & Report
# ============================================================================

def analyze_subject_deep(subject_id, data, is_primary=True):
    """Run full deep-dive analysis for one subject."""
    print(f"\n{'='*60}")
    print(f"  Analyzing {subject_id} ({'PRIMARY' if is_primary else 'SECONDARY'})")
    print(f"{'='*60}")

    result = {'subject_id': subject_id}

    print(f"  [1/7] Per-session statistics...")
    result['session_stats'] = per_session_stats(data)

    print(f"  [2/7] Per-run amplitude profile...")
    result['run_profiles'] = per_run_amplitude_profile(data)

    print(f"  [3/7] Channel quality map...")
    result['channel_quality'] = channel_quality_map(data)

    print(f"  [4/7] Per-session class discriminability...")
    result['session_discrim'] = per_session_discriminability(data)

    print(f"  [5/7] Per-session spectral profile...")
    result['session_spectral'] = per_session_spectral(data)

    print(f"  [6/7] Train/test distribution comparison...")
    result['train_test'] = train_test_comparison(data)

    print(f"  [7/7] Salvageability analysis...")
    result['salvage'] = salvageability_analysis(data)

    return result


def generate_report(results: List[dict], output_path: Path):
    """Generate comprehensive deep-dive Markdown report."""
    lines = [
        "# Subject Deep-Dive Analysis Report\n",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Subjects: {', '.join(r['subject_id'] for r in results)}",
        "",
    ]

    for result in results:
        sid = result['subject_id']
        lines.append(f"\n---\n\n# {sid}\n")

        # ---- 1. Session Overview ----
        lines.append("## 1. Per-Session Overview\n")
        lines.append("| Session | Runs | Trials | Labels | Amp Max | Amp P99.9 | "
                     "Artifact% | Ch Var Mean | SNR (dB) |")
        lines.append("|---------|------|--------|--------|---------|-----------|"
                     "-----------|-------------|----------|")

        for session, ss in result['session_stats'].items():
            label_str = ', '.join(f"{k}:{v}" for k, v in sorted(ss['labels'].items()))
            snr_str = f"{ss['snr_db']:.1f}" if not np.isnan(ss['snr_db']) else "n/a"
            lines.append(
                f"| {session} | {ss['n_runs']} | {ss['n_trials']} | "
                f"{label_str} | {ss['amp_max']:.1f} | {ss['amp_p999']:.1f} | "
                f"{ss['pct_artifact']:.1f}% | {ss['ch_var_mean']:.2f} | {snr_str} |"
            )
        lines.append("")

        # ---- 2. Run-Level Amplitude Profile ----
        lines.append("## 2. Run-Level Amplitude Profile\n")

        # Show worst runs first
        run_profiles = sorted(result['run_profiles'], key=lambda x: -x['amp_max'])
        lines.append("### Top 10 Worst Runs (by max amplitude)\n")
        lines.append("| Session | Run | Trials | Amp Max | Amp Mean | "
                     "Trial P95 | >500 | >1000 |")
        lines.append("|---------|-----|--------|---------|----------|"
                     "----------|------|-------|")
        for rp in run_profiles[:10]:
            lines.append(
                f"| {rp['session']} | {rp['run']} | {rp['n_trials']} | "
                f"{rp['amp_max']:.1f} | {rp['amp_mean']:.2f} | "
                f"{rp['trial_max_p95']:.1f} | {rp['n_trials_above_500']} | "
                f"{rp['n_trials_above_1000']} |"
            )

        # Session-aggregated run stats
        lines.append("\n### Per-Session Run Summary\n")
        session_runs = defaultdict(list)
        for rp in result['run_profiles']:
            session_runs[rp['session']].append(rp)

        lines.append("| Session | N Runs | Median AmpMax | Max AmpMax | "
                     "Total >500 | Total >1000 |")
        lines.append("|---------|--------|---------------|------------|"
                     "-----------|-------------|")
        for session in sort_sessions(session_runs.keys()):
            rps = session_runs[session]
            amp_maxes = [r['amp_max'] for r in rps]
            total_above_500 = sum(r['n_trials_above_500'] for r in rps)
            total_above_1000 = sum(r['n_trials_above_1000'] for r in rps)
            lines.append(
                f"| {session} | {len(rps)} | {np.median(amp_maxes):.1f} | "
                f"{np.max(amp_maxes):.1f} | {total_above_500} | {total_above_1000} |"
            )
        lines.append("")

        # ---- 3. Channel Quality ----
        lines.append("## 3. Channel Quality Map\n")
        cq = result['channel_quality']
        if cq:
            lines.append(f"Sampled {cq['n_trials_sampled']} trials.\n")
            lines.append("### Worst 10 Channels by Variance\n")
            lines.append("| Rank | Ch Idx | Label | Mean Var | Region |")
            lines.append("|------|--------|-------|----------|--------|")
            for rank, (idx, label, var, region) in enumerate(cq['worst_channels_by_variance'], 1):
                lines.append(f"| {rank} | {idx} | {label} | {var:.4f} | {region} |")

            lines.append("\n### Worst 10 Channels by Max Amplitude\n")
            lines.append("| Rank | Ch Idx | Label | Mean Max Amp | Region |")
            lines.append("|------|--------|-------|-------------|--------|")
            for rank, (idx, label, amp, region) in enumerate(cq['worst_channels_by_amplitude'], 1):
                lines.append(f"| {rank} | {idx} | {label} | {amp:.2f} | {region} |")

            lines.append(f"\n**Central vs Peripheral**:")
            lines.append(f"- Variance: central={cq['central_mean_var']:.4f}, "
                        f"peripheral={cq['peripheral_mean_var']:.4f} "
                        f"(ratio={cq['peripheral_mean_var'] / (cq['central_mean_var'] + 1e-12):.2f}x)")
            lines.append(f"- Max Amp: central={cq['central_mean_amp']:.2f}, "
                        f"peripheral={cq['peripheral_mean_amp']:.2f}")
            lines.append(f"- Mu/Beta Ratio: central={cq['mu_beta_ratio_central_mean']:.3f}, "
                        f"peripheral={cq['mu_beta_ratio_peripheral_mean']:.3f}")
        lines.append("")

        # ---- 4. Per-Session Class Discriminability ----
        lines.append("## 4. Per-Session Class Discriminability\n")
        lines.append("| Session | Classes | Fisher Mean | AUROC Mean (top5) | Class Counts |")
        lines.append("|---------|---------|-----------|-----------------|-------------|")
        for session, sd in result['session_discrim'].items():
            cc = ', '.join(f"{k}:{v}" for k, v in sorted(sd['class_counts'].items()))
            lines.append(
                f"| {session} | {sd['n_classes']} | {sd['fisher_mean']:.4f} | "
                f"{sd['auroc_mean']:.4f} | {cc} |"
            )
        lines.append("")

        # ---- 5. Per-Session Spectral Profile ----
        lines.append("## 5. Per-Session Spectral Profile\n")
        lines.append("| Session | Theta | Mu | Low-β | High-β | γ | Mu/β Ratio |")
        lines.append("|---------|-------|-----|-------|--------|---|-----------|")
        for session, sp in result['session_spectral'].items():
            lines.append(
                f"| {session} | {sp['theta']:.4f} | {sp['mu']:.4f} | "
                f"{sp['low_beta']:.4f} | {sp['high_beta']:.4f} | "
                f"{sp['gamma']:.4f} | {sp['mu_beta_ratio']:.3f} |"
            )
        lines.append("")

        # ---- 6. Train/Test Distribution ----
        lines.append("## 6. Train/Test Distribution Comparison\n")
        tt = result['train_test']
        if 'error' not in tt:
            lines.append(f"- Train trials: {tt['n_train_trials']}, Test trials: {tt['n_test_trials']}")
            lines.append(f"- Train global mean: {tt['train_global_mean']:.4f} "
                        f"(std={tt['train_global_std']:.4f})")
            lines.append(f"- Test global mean: {tt['test_global_mean']:.4f} "
                        f"(std={tt['test_global_std']:.4f})")
            lines.append(f"- **Mean KS statistic**: {tt['mean_ks_stat']:.4f}")
            lines.append(f"- Median KS statistic: {tt['median_ks_stat']:.4f}")
            lines.append(f"- Max KS statistic: {tt['max_ks_stat']:.4f}")
            lines.append(f"- Channels where train/test indistinguishable (p>0.05): "
                        f"**{tt['n_indistinguishable_channels']}/128**\n")

            lines.append("### Most Similar Channels (lowest KS)\n")
            lines.append("| Ch Idx | Label | KS Stat | p-value |")
            lines.append("|--------|-------|---------|---------|")
            for idx, label, ks, pv in tt['most_similar_channels']:
                lines.append(f"| {idx} | {label} | {ks:.4f} | {pv:.4f} |")

            lines.append("\n### Most Different Channels (highest KS)\n")
            lines.append("| Ch Idx | Label | KS Stat | p-value |")
            lines.append("|--------|-------|---------|---------|")
            for idx, label, ks, pv in tt['most_different_channels']:
                lines.append(f"| {idx} | {label} | {ks:.4f} | {pv:.4f} |")
        else:
            lines.append(f"Error: {tt['error']}")
        lines.append("")

        # ---- 7. Salvageability ----
        lines.append("## 7. Salvageability Analysis\n")
        sv = result['salvage']
        lines.append(f"**Amplitude threshold**: {sv['threshold']} µV")
        lines.append(f"**Overall**: {sv['total_clean']}/{sv['total_trials']} trials clean "
                    f"(**{sv['pct_clean']:.1f}%**)\n")

        lines.append("| Session | Total | Clean | Clean% | Amp P50 | Amp P90 | Amp P99 |")
        lines.append("|---------|-------|-------|--------|---------|---------|---------|")
        for session, sp in sv['per_session'].items():
            lines.append(
                f"| {session} | {sp['n_trials']} | {sp['n_clean']} | "
                f"{sp['pct_clean']:.1f}% | {sp['amp_p50']:.1f} | "
                f"{sp['amp_p90']:.1f} | {sp['amp_p99']:.1f} |"
            )
        lines.append("")

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Subject deep-dive analysis')
    parser.add_argument('--subjects', nargs='+', default=['S10', 'S20', 'S05', 'S21'])
    parser.add_argument('--paradigm', default='imagery')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    cache_dir = PROJECT_ROOT / 'caches' / 'preprocessed'
    cache_index_path = cache_dir / '.cache_index.json'

    if not cache_index_path.exists():
        print(f"ERROR: Cache index not found: {cache_index_path}")
        sys.exit(1)

    t0 = time.time()
    print("Loading cache index...")
    entries = load_cache_index(str(cache_index_path))
    all_subjects = group_entries_by_subject(entries, paradigm=args.paradigm)

    # Validate requested subjects
    primary = args.subjects[:2] if len(args.subjects) >= 2 else args.subjects
    secondary = args.subjects[2:] if len(args.subjects) > 2 else []

    results = []
    for sid in args.subjects:
        if sid not in all_subjects:
            print(f"WARNING: {sid} not found in cache, skipping")
            continue
        data = load_subject_data(all_subjects[sid], str(cache_dir))
        is_primary = sid in primary
        result = analyze_subject_deep(sid, data, is_primary)
        results.append(result)

    if not results:
        print("No subjects analyzed.")
        sys.exit(1)

    output_path = Path(args.output) if args.output else (
        PROJECT_ROOT / 'results' / 'subject_deep_dive_report.md'
    )
    generate_report(results, output_path)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == '__main__':
    main()
