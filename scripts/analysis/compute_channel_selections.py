#!/usr/bin/env python
"""
Data-driven N-channel selection analysis.

Computes optimal channel subsets from 128-channel EEG data using multiple methods:
- FDR: Fisher Discriminant Ratio
- CSP: Common Spatial Patterns
- Attention: CBraMod input gradient magnitude
- Band Power: Mu/Beta ANOVA F-statistic

Supports any target channel count (e.g., 4, 8, 16, 32, 61).

Usage:
    uv run python scripts/analysis/compute_channel_selections.py --n-channels 8 --methods attention
    uv run python scripts/analysis/compute_channel_selections.py --n-channels 32
    uv run python scripts/analysis/compute_channel_selections.py --n-channels 32 --methods fdr csp
    uv run python scripts/analysis/compute_channel_selections.py --paradigm movement
"""

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Data Loading
# ============================================================================

def _get_train_session_folders(paradigm, task):
    """Get the set of session folders that belong to the training split.

    Uses the canonical split definition from discovery.py to ensure
    channel selection only uses training data (no test leakage).
    """
    from src.preprocessing.discovery import get_session_folders_for_split
    return set(get_session_folders_for_split(paradigm, task, split='train'))


TASK_TARGET_CLASSES = {
    'binary': [1, 4],       # Thumb vs Pinky
    'ternary': [1, 3, 4],   # Thumb vs Middle vs Pinky
    'quaternary': [1, 2, 3, 4],
}


def load_all_trials(cache_index_path, paradigm, task, model='eegnet',
                    train_only=True):
    """Load all trials from HDF5 cache for analysis.

    Includes OfflineImagery data (n_classes=None in cache) by filtering
    to target classes after loading. This ensures channel selection uses
    the same training data as the actual training pipeline.

    Args:
        cache_index_path: Path to .cache_index.json
        paradigm: 'imagery' or 'movement'
        task: 'binary', 'ternary', or 'quaternary'
        model: 'eegnet' or 'cbramod_128ch'
        train_only: If True (default), only load training-split sessions
                    to prevent test data leakage into channel selection.

    Returns:
        X: np.ndarray [N, 128, T] — all trials concatenated
        y: np.ndarray [N] — labels (raw class indices, e.g. 1/4 for binary)
    """
    import h5py

    with open(cache_index_path, 'r') as f:
        index = json.load(f)

    task_map = {'binary': 2, 'ternary': 3, 'quaternary': 4}
    n_classes = task_map[task]
    target_classes = TASK_TARGET_CLASSES[task]

    train_sessions = _get_train_session_folders(paradigm, task) if train_only else None

    all_trials = []
    all_labels = []
    loaded_files = 0
    offline_files = 0
    skipped_sessions = set()

    entries = index.get('entries', index)  # v3.0 has 'entries' key
    for key, meta in entries.items():
        if not isinstance(meta, dict):
            continue
        if meta.get('model') != model:
            continue
        if meta.get('subject_task_type') != paradigm:
            continue

        # Accept matching n_classes (Online) or None (Offline, contains all 4 classes)
        entry_n_classes = meta.get('n_classes')
        if entry_n_classes is not None and entry_n_classes != n_classes:
            continue

        # Filter by training sessions to prevent test data leakage
        if train_sessions is not None:
            session_folder = meta.get('session_folder', '')
            if session_folder not in train_sessions:
                skipped_sessions.add(session_folder)
                continue

        h5_path = f"caches/preprocessed/{key}.h5"
        try:
            with h5py.File(h5_path, 'r') as f:
                trials = f['trials'][:]  # [n_trials, 128, n_samples]
                labels = f['labels'][:]

            # Offline data: filter to target classes
            if entry_n_classes is None:
                mask = np.isin(labels, target_classes)
                if mask.sum() == 0:
                    continue
                trials = trials[mask]
                labels = labels[mask]
                offline_files += 1

            all_trials.append(trials)
            all_labels.append(labels)
            loaded_files += 1
        except Exception as e:
            print(f"  Warning: Failed to load {h5_path}: {e}")

    if skipped_sessions:
        print(f"  Excluded non-training sessions: {sorted(skipped_sessions)}")

    if not all_trials:
        raise RuntimeError(
            f"No matching cache entries found for model={model}, "
            f"paradigm={paradigm}, task={task} (n_classes={n_classes})"
        )

    print(f"  Loaded {loaded_files} cache files ({offline_files} Offline, model={model})")

    X = np.concatenate(all_trials, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y


# ============================================================================
# Method 1: Fisher Discriminant Ratio
# ============================================================================

def compute_fdr_scores(X, y):
    """Fisher Discriminant Ratio per channel.

    FDR = sum_{i<j} (mu_i - mu_j)^2 / (sigma_i^2 + sigma_j^2)
    averaged over time samples.
    """
    n_channels = X.shape[1]
    classes = np.unique(y)
    scores = np.zeros(n_channels)

    for ch in range(n_channels):
        ch_data = X[:, ch, :]  # [N, T]
        fdr_sum = 0.0
        n_pairs = 0
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                mask_i = y == classes[i]
                mask_j = y == classes[j]
                mu_i = ch_data[mask_i].mean(axis=0)  # [T]
                mu_j = ch_data[mask_j].mean(axis=0)
                var_i = ch_data[mask_i].var(axis=0)
                var_j = ch_data[mask_j].var(axis=0)
                denom = var_i + var_j + 1e-10
                fdr = ((mu_i - mu_j) ** 2 / denom).mean()
                fdr_sum += fdr
                n_pairs += 1
        scores[ch] = fdr_sum / max(n_pairs, 1)

    return scores


# ============================================================================
# Method 2: Common Spatial Patterns
# ============================================================================

def compute_csp_scores(X, y):
    """CSP spatial filter weight magnitudes per channel.

    Uses mne.decoding.CSP with reg='ledoit_wolf', one-vs-rest for multi-class.
    Channel importance = sum of absolute CSP filter weights.

    Uses EEGNet z-scored data (channel-standardized, ideal for CSP covariance)
    with NaN time-padding truncated.
    """
    from mne.decoding import CSP

    classes = np.unique(y)
    n_channels = X.shape[1]
    scores = np.zeros(n_channels)

    if len(classes) == 2:
        csp = CSP(n_components=6, reg='ledoit_wolf', log=True)
        csp.fit(X, y)
        # filters_ shape: [n_components, n_channels]
        scores = np.sum(np.abs(csp.filters_[:6]), axis=0)
    else:
        # One-vs-rest
        for cls in classes:
            y_binary = (y == cls).astype(int)
            try:
                csp = CSP(n_components=4, reg='ledoit_wolf', log=True)
                csp.fit(X, y_binary)
                scores += np.sum(np.abs(csp.filters_[:4]), axis=0)
            except Exception as e:
                print(f"    CSP OVR class {cls} failed: {e}")

    return scores


# ============================================================================
# Method 3: Attention / Gradient
# ============================================================================

def _find_best_checkpoint(model_type, paradigm, task, checkpoint_tag='baseline'):
    """Auto-discover a 128ch cross-subject checkpoint.

    Search order:
    1. Timestamped dirs: checkpoints/cross_subject/YYYYMMDD_HHMM_{tag}_{model}_{paradigm}_{task}/best.pt
       (only 128ch — skip dirs whose config.json shows n_channels != 128)
    2. Legacy dir: checkpoints/cross_subject/{model}_{paradigm}_{task}/best.pt

    Args:
        model_type: 'eegnet' or 'cbramod'
        paradigm: 'imagery' or 'movement'
        task: 'binary' or 'ternary'
        checkpoint_tag: Tag to match in directory name (default: 'baseline').
                       Use 'any' to accept any checkpoint (newest first).

    Returns:
        Path to best.pt, or None if not found.
    """
    base = PROJECT_ROOT / 'checkpoints' / 'cross_subject'
    suffix = f'{model_type}_{paradigm}_{task}'

    # Timestamped dirs — sorted descending so newest first
    candidates = sorted(base.glob(f'*_{suffix}'), reverse=True)

    # Filter by tag: skip extra-sessions checkpoints (sess03/sess04/sess05)
    # unless explicitly requested
    if checkpoint_tag != 'any':
        tagged = [d for d in candidates if f'_{checkpoint_tag}_' in d.name]
        # Fall back to all candidates if no tagged match
        candidates = tagged if tagged else candidates

    for d in candidates:
        best_pt = d / 'best.pt'
        if not best_pt.exists():
            continue
        # Prefer 128ch checkpoints for channel scoring
        config_json = d / 'config.json'
        if config_json.exists():
            try:
                with open(config_json, 'r') as f:
                    cfg = json.load(f)
                if cfg.get('n_channels', 128) != 128:
                    continue
            except Exception:
                pass
        return best_pt

    # Legacy non-timestamped dir
    legacy = base / suffix / 'best.pt'
    if legacy.exists():
        return legacy

    return None


def compute_attention_scores(X, y, cache_index_path=None, paradigm='imagery',
                             task='binary', checkpoint_tag='baseline'):
    """CBraMod input gradient magnitude as channel importance scores.

    Computes the gradient of cross-entropy loss w.r.t. input for a trained
    CBraMod checkpoint, averaged over a batch of real training data.
    Channels with higher gradient magnitude are more important for the task.

    Checkpoints are auto-discovered (most recent 128ch cross-subject baseline).
    """
    import torch

    n_channels = X.shape[1]

    # --- CBraMod input gradient ---
    cbramod_path = _find_best_checkpoint('cbramod', paradigm, task, checkpoint_tag)
    if not cbramod_path:
        raise RuntimeError(
            f"No CBraMod checkpoint found for {paradigm}/{task}. "
            "Run cross-subject training first."
        )

    from src.models.cbramod_adapter import CBraModForFingerBCI

    ckpt = torch.load(str(cbramod_path), map_location='cpu', weights_only=False)
    model_config = ckpt.get('model_config', {})

    # Compute n_patches from n_samples / patch_size (200)
    n_samples_model = model_config.get('n_samples', 200)
    n_patches = max(1, n_samples_model // 200)

    model = CBraModForFingerBCI(
        n_channels=model_config.get('n_channels', 128),
        n_classes=model_config.get('n_classes', 2),
        n_patches=n_patches,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Load CBraMod-preprocessed data (200 Hz, correct filtering/normalization)
    if cache_index_path is None:
        from src.config.constants import DEFAULT_CACHE_INDEX_PATH
        cache_index_path = DEFAULT_CACHE_INDEX_PATH
    print(f"    Loading CBraMod cache data (200 Hz)...")
    X_cbramod, y_cbramod = load_all_trials(
        cache_index_path, paradigm, task, model='cbramod_128ch'
    )

    # Use a batch of real CBraMod data for gradient computation
    n_batch = min(200, len(X_cbramod))
    rng = np.random.RandomState(42)
    indices = rng.choice(len(X_cbramod), n_batch, replace=False)
    # Trim to n_patches * 200 samples (valid signal, before NaN padding)
    batch = X_cbramod[indices][:, :, :n_patches * 200]

    batch_tensor = torch.tensor(batch, dtype=torch.float32)
    batch_tensor.requires_grad_(True)
    # Remap raw labels (e.g. [1, 4]) to [0, n_classes-1] for cross_entropy
    raw_labels = y_cbramod[indices]
    unique_classes = np.sort(np.unique(y_cbramod))
    label_map = {int(c): i for i, c in enumerate(unique_classes)}
    mapped_labels = np.array([label_map[int(l)] for l in raw_labels])
    batch_labels = torch.tensor(mapped_labels, dtype=torch.long)

    outputs = model(batch_tensor)
    loss = torch.nn.functional.cross_entropy(outputs, batch_labels)
    loss.backward()

    grad = batch_tensor.grad.abs().mean(dim=(0, 2)).numpy()
    print(f"    CBraMod gradient: computed from {cbramod_path}")

    return grad


# ============================================================================
# Method 4: Band Power (Mu + Beta ANOVA)
# ============================================================================

def compute_band_power_scores(X, y, fs=100):
    """Mu (8-13 Hz) + Beta (13-30 Hz) band power with ANOVA F-statistic.

    For each channel:
    1. Compute PSD via scipy.signal.welch
    2. Sum power in mu and beta bands
    3. Compute ANOVA F-statistic across classes
    4. Score = F_mu + F_beta
    """
    from scipy.signal import welch
    from scipy.stats import f_oneway

    n_channels = X.shape[1]
    classes = np.unique(y)
    scores = np.zeros(n_channels)

    for ch in range(n_channels):
        ch_data = X[:, ch, :]  # [N, T]

        freqs, psd = welch(ch_data, fs=fs, nperseg=min(256, ch_data.shape[1]), axis=1)

        mu_mask = (freqs >= 8) & (freqs <= 13)
        beta_mask = (freqs >= 13) & (freqs <= 30)

        mu_power = psd[:, mu_mask].sum(axis=1)
        beta_power = psd[:, beta_mask].sum(axis=1)

        try:
            mu_groups = [mu_power[y == c] for c in classes]
            f_mu, _ = f_oneway(*mu_groups)
        except Exception:
            f_mu = 0.0

        try:
            beta_groups = [beta_power[y == c] for c in classes]
            f_beta, _ = f_oneway(*beta_groups)
        except Exception:
            f_beta = 0.0

        scores[ch] = (f_mu if np.isfinite(f_mu) else 0.0) + (f_beta if np.isfinite(f_beta) else 0.0)

    return scores


# ============================================================================
# Method Registry
# ============================================================================

METHOD_REGISTRY = {
    'fdr': {
        'fn': compute_fdr_scores,
        'description': 'Fisher Discriminant Ratio — top {n} channels by inter-class separability',
    },
    'csp': {
        'fn': compute_csp_scores,
        'description': 'Common Spatial Patterns — top {n} channels by CSP filter weights (OVR)',
    },
    'attention': {
        'fn': compute_attention_scores,
        'description': 'CBraMod gradient — input gradient magnitude from cross-subject checkpoint',
    },
    'band_power': {
        'fn': compute_band_power_scores,
        'description': 'Band Power — Mu(8-13Hz) + Beta(13-30Hz) ANOVA F-statistic',
    },
}


def select_top_channels(scores, n=32):
    """Select top-n channels by score, return sorted indices."""
    top_indices = np.argsort(scores)[::-1][:n]
    return sorted(top_indices.tolist())


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compute data-driven N-channel selections from 128ch EEG data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # 8-channel attention selection
  uv run python scripts/analysis/compute_channel_selections.py --n-channels 8 --methods attention

  # 32-channel selection with all methods
  uv run python scripts/analysis/compute_channel_selections.py --n-channels 32

  # Only FDR and CSP
  uv run python scripts/analysis/compute_channel_selections.py --n-channels 32 --methods fdr csp

  # Motor Execution paradigm
  uv run python scripts/analysis/compute_channel_selections.py --paradigm movement
'''
    )
    parser.add_argument(
        '--methods', nargs='+',
        default=list(METHOD_REGISTRY.keys()),
        choices=list(METHOD_REGISTRY.keys()),
        help='Methods to compute (default: all 4)',
    )
    parser.add_argument(
        '--output', type=str,
        default=None,
        help='Output JSON path (default: results/{n}_channel/channel_selections.json)',
    )
    parser.add_argument(
        '--paradigm', type=str, default='imagery',
        choices=['imagery', 'movement'],
        help='Experiment paradigm (default: imagery)',
    )
    parser.add_argument(
        '--task', type=str, default='binary',
        choices=['binary', 'ternary'],
        help='Classification task (default: binary)',
    )
    parser.add_argument(
        '--n-channels', type=int, default=32,
        help='Number of channels to select (default: 32)',
    )
    parser.add_argument(
        '--cache-index-path', type=str, default='caches/preprocessed/.cache_index.json',
        help='Path to cache index file (default: caches/preprocessed/.cache_index.json)',
    )
    parser.add_argument(
        '--checkpoint-tag', type=str, default='baseline',
        help='Checkpoint tag for attention method (default: baseline). '
             'Use "any" to pick the newest checkpoint regardless of tag.',
    )
    args = parser.parse_args()

    # Auto-derive output path from --n-channels if not explicitly set
    if args.output is None:
        args.output = f'results/{args.n_channels}_channel/channel_selections.json'

    print(f"\n{'='*60}")
    print(f"  {args.n_channels}-Channel Selection Analysis")
    print(f"  Paradigm: {args.paradigm} | Task: {args.task}")
    print(f"  Methods: {args.methods}")
    print(f"  Selecting: {args.n_channels} channels")
    print(f"  Output: {args.output}")
    print(f"{'='*60}\n")

    start_time = time.time()

    # Load data (EEGNet cache: 100 Hz, z-scored)
    print("Loading data from HDF5 cache...")
    X, y = load_all_trials(args.cache_index_path, args.paradigm, args.task)
    print(f"  Loaded: {X.shape[0]} trials, {X.shape[1]} channels, {X.shape[2]} samples")

    # Truncate NaN time-padding globally (trials padded with NaN beyond actual signal)
    nan_per_sample = np.isnan(X[:, 0, :]).astype(int)
    first_nan_per_trial = np.argmax(nan_per_sample, axis=1)
    has_nan = nan_per_sample.any(axis=1)
    first_nan_per_trial[~has_nan] = X.shape[2]
    min_valid = int(np.percentile(first_nan_per_trial, 1))
    valid_mask = first_nan_per_trial >= min_valid
    n_removed = (~valid_mask).sum()
    if n_removed > 0:
        print(f"  Removed {n_removed} outlier trials (valid length < {min_valid})")
        X = X[valid_mask]
        y = y[valid_mask]
    if min_valid < X.shape[2]:
        print(f"  Truncated NaN padding: {X.shape[2]} -> {min_valid} samples ({min_valid/100:.2f}s @ 100Hz)")
        X = X[:, :, :min_valid]

    class_counts = dict(zip(*np.unique(y, return_counts=True)))
    print(f"  Final: {X.shape[0]} trials, {X.shape[2]} samples | Classes: {class_counts}")

    # Compute each method
    results = {}
    for method_name in args.methods:
        method_info = METHOD_REGISTRY[method_name]
        print(f"\nComputing {method_name}...")
        method_start = time.time()

        try:
            if method_name == 'attention':
                scores = method_info['fn'](
                    X, y,
                    cache_index_path=args.cache_index_path,
                    paradigm=args.paradigm,
                    task=args.task,
                    checkpoint_tag=args.checkpoint_tag,
                )
            else:
                scores = method_info['fn'](X, y)
            indices = select_top_channels(scores, args.n_channels)

            scores_dict = {str(i): float(scores[i]) for i in indices}

            results[method_name] = {
                'indices': indices,
                'scores': scores_dict,
                'description': method_info['description'].format(n=args.n_channels),
            }

            elapsed = time.time() - method_start
            print(f"  Done in {elapsed:.1f}s")
            print(f"  Top-5 channels: {indices[:5]}")

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()

    if not results:
        print("\nNo methods succeeded. Exiting.")
        sys.exit(1)

    # Count unique subjects (including Offline entries with n_classes=None)
    with open(args.cache_index_path, 'r') as f:
        index = json.load(f)
    task_map = {'binary': 2, 'ternary': 3, 'quaternary': 4}
    subjects = set()
    entries = index.get('entries', index)
    for meta in entries.values():
        if not isinstance(meta, dict):
            continue
        if (meta.get('model') in ('eegnet', 'cbramod_128ch') and
                meta.get('subject_task_type') == args.paradigm and
                meta.get('n_classes') in (task_map[args.task], None)):
            subjects.add(meta.get('subject', ''))

    # Build output JSON
    output = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'paradigm': args.paradigm,
            'task': args.task,
            'n_channels_selected': args.n_channels,
            'n_subjects': len(subjects),
            'n_trials_total': int(X.shape[0]),
            'methods': list(results.keys()),
            'checkpoint_tag': args.checkpoint_tag,
            'note': 'Train-only data (no test leakage), includes OfflineImagery, '
                    'attention uses CBraMod gradient only (no EEGNet)',
        },
        'configs': results,
    }

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Results saved to: {args.output}")
    print(f"  Methods completed: {list(results.keys())}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"{'='*60}")

    # Print overlap analysis
    if len(results) >= 2:
        print(f"\n  Channel Overlap Analysis:")
        method_names = list(results.keys())
        for i in range(len(method_names)):
            for j in range(i + 1, len(method_names)):
                a = set(results[method_names[i]]['indices'])
                b = set(results[method_names[j]]['indices'])
                overlap = len(a & b)
                print(f"    {method_names[i]} \u2229 {method_names[j]}: "
                      f"{overlap}/{args.n_channels} ({overlap / args.n_channels:.0%})")

    return 0


if __name__ == '__main__':
    sys.exit(main())
