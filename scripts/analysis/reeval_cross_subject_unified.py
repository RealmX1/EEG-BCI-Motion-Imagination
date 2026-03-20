"""
Re-evaluate saved cross-subject unified model checkpoints to recover
per-subject subtask breakdown data, then generate proper unified plots.

This is needed because the original training pipeline discarded subtask_results
during serialization. The temporal split is deterministic, so we can reconstruct
the exact same offline_test_indices from the training dataset.

Usage:
    uv run python scripts/analysis/reeval_cross_subject_unified.py
    uv run python scripts/analysis/reeval_cross_subject_unified.py --run-tag 20260319_2102
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.constants import TASKS
from src.models.eegnet import EEGNet
from src.models.cbramod_adapter import CBraModForFingerBCI, get_default_pretrained_path
from src.preprocessing.data_loader import PreprocessConfig
from src.training.train_cross_subject import load_multi_subject_data
from src.training.common import temporal_split_with_offline_test
from src.training.evaluation import unified_model_evaluate
from src.visualization.comparison import plot_unified_comparison
from src.utils.device import get_device

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_dir: Path, device: torch.device):
    """Load model architecture + weights from checkpoint directory."""
    config_path = checkpoint_dir / 'config.json'
    best_pt = checkpoint_dir / 'best.pt'

    if not config_path.exists() or not best_pt.exists():
        raise FileNotFoundError(f"Missing config.json or best.pt in {checkpoint_dir}")

    with open(config_path) as f:
        config = json.load(f)

    model_type = config['model_type']
    n_channels = config['n_channels']
    n_samples = config['n_samples']
    n_classes = config['n_classes']
    model_config = config.get('model_config', {})

    if model_type == 'cbramod':
        n_patches = n_samples // 200
        model = CBraModForFingerBCI(
            n_channels=n_channels,
            n_patches=n_patches,
            n_classes=n_classes,
            pretrained_path=None,  # Don't load pretrained, we load checkpoint
            freeze_backbone=False,
            classifier_type=model_config.get('classifier_type', 'two_layer'),
            dropout=model_config.get('dropout_rate', 0.1),
        )
    else:
        model = EEGNet(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=n_classes,
            F1=model_config.get('F1', 8),
            D=model_config.get('D', 2),
            F2=model_config.get('F2', 16),
            kernel_length=model_config.get('kernel_length', 64),
            dropout_rate=model_config.get('dropout_rate', 0.5),
        )

    # Load weights
    checkpoint = torch.load(best_pt, map_location=device, weights_only=True)
    # Cross-subject checkpoints store state_dict inside a larger dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    # Handle compiled model state dicts
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, config


def evaluate_checkpoint(checkpoint_dir: Path, device: torch.device):
    """Re-evaluate a single cross-subject unified checkpoint."""
    logger.info(f"Loading checkpoint: {checkpoint_dir.name}")

    model, config = load_model_from_checkpoint(checkpoint_dir, device)
    model_type = config['model_type']
    subjects = config['subjects']
    paradigm = config['paradigm']
    n_channels = config['n_channels']

    # Reconstruct PreprocessConfig
    if model_type == 'cbramod':
        preprocess_config = PreprocessConfig.for_cbramod(full_channels=(n_channels == 128))
    else:
        preprocess_config = PreprocessConfig.paper_aligned()

    data_root = PROJECT_ROOT / 'data'
    elc_path = data_root / 'biosemi128.ELC'

    # Load unified training dataset (same as during training)
    logger.info(f"Loading unified dataset for {len(subjects)} subjects ({model_type})...")
    t0 = time.time()
    train_dataset, _ = load_multi_subject_data(
        data_root, subjects, preprocess_config,
        target_classes=TASKS['unified']['classes'],
        paradigm=paradigm, task='unified',
        elc_path=elc_path,
        cache_only=True,
        cache_index_path='.cache_index.json',
        unified_mode=True,
    )
    logger.info(f"  Loaded {len(train_dataset)} segments in {time.time()-t0:.1f}s")

    # Reconstruct temporal split (deterministic)
    _, _, offline_test_indices = temporal_split_with_offline_test(
        train_dataset, group_attr='subject_id',
    )
    logger.info(f"  Offline test indices: {len(offline_test_indices)} segments")

    # Per-subject evaluation
    per_subject_subtask = {}
    per_subject_test_acc = {}

    for i, subject_id in enumerate(subjects):
        # Filter offline_test_indices for this subject
        subj_offline_test = [
            idx for idx in offline_test_indices
            if train_dataset.trial_infos[idx].subject_id == subject_id
        ]

        subj_results = unified_model_evaluate(
            model, data_root, [subject_id], preprocess_config, elc_path,
            paradigm, device, cache_only=True, cache_index_path='.cache_index.json',
            train_dataset=train_dataset,
            offline_test_indices=subj_offline_test,
        )
        per_subject_subtask[subject_id] = subj_results
        per_subject_test_acc[subject_id] = subj_results['mean_accuracy']

        parts = []
        for st in ['binary', 'ternary', 'quaternary']:
            if st in subj_results and subj_results[st].get('n_trials', 0) > 0:
                parts.append(f"{st[0].upper()}={subj_results[st]['accuracy']:.2%}")
        mean_s = subj_results['mean_accuracy']
        print(f"  [{i+1}/{len(subjects)}] {subject_id}: {' | '.join(parts)} (mean={mean_s:.2%})")

    # Aggregate
    subtask_results_all = {'per_subject': per_subject_subtask}
    for st in ['binary', 'ternary', 'quaternary']:
        st_accs = [
            r[st]['accuracy'] for r in per_subject_subtask.values()
            if st in r and r[st].get('n_trials', 0) > 0
        ]
        subtask_results_all[st] = {
            'accuracy': float(np.mean(st_accs)) if st_accs else 0.0,
            'std': float(np.std(st_accs)) if st_accs else 0.0,
            'n_subjects': len(st_accs),
        }
    subtask_results_all['mean_accuracy'] = float(np.mean(list(per_subject_test_acc.values())))

    return model_type, subtask_results_all


def main():
    parser = argparse.ArgumentParser(description='Re-evaluate cross-subject unified checkpoints')
    parser.add_argument('--run-tag', type=str, default='20260319_2102',
                        help='Run tag to re-evaluate')
    parser.add_argument('--models', nargs='+', default=['eegnet', 'cbramod'],
                        choices=['eegnet', 'cbramod'])
    args = parser.parse_args()

    device = get_device()
    checkpoint_base = PROJECT_ROOT / 'checkpoints' / 'cross_subject'
    results_dir = PROJECT_ROOT / 'results'

    all_results = {}

    for model_type in args.models:
        ckpt_dir = checkpoint_base / f'{args.run_tag}_{model_type}_imagery_unified'
        if not ckpt_dir.exists():
            logger.warning(f"Checkpoint not found: {ckpt_dir}")
            continue

        t0 = time.time()
        model_type_out, subtask_results = evaluate_checkpoint(ckpt_dir, device)
        elapsed = time.time() - t0
        logger.info(f"{model_type_out} evaluation done in {elapsed:.1f}s")

        # Print summary
        for st in ['binary', 'ternary', 'quaternary']:
            sr = subtask_results[st]
            print(f"  {st}: {sr['accuracy']:.2%} +/- {sr['std']:.2%} ({sr['n_subjects']} subjects)")
        print(f"  mean: {subtask_results['mean_accuracy']:.2%}")

        all_results[model_type_out] = {
            'subtask_results': subtask_results,
            'per_subject': subtask_results['per_subject'],
        }

        # Also update the result JSON with subtask data
        result_json = results_dir / f'{args.run_tag}_cross-subject_{model_type}_imagery_unified.json'
        if result_json.exists():
            with open(result_json) as f:
                data = json.load(f)
            # Add subtask_results (strip heavy detailed_results)
            lightweight = {}
            for key in ('binary', 'ternary', 'quaternary', 'mean_accuracy'):
                if key in subtask_results:
                    val = subtask_results[key]
                    if isinstance(val, dict):
                        lightweight[key] = {k: v for k, v in val.items() if k != 'detailed_results'}
                    else:
                        lightweight[key] = val
            lightweight['per_subject'] = {}
            for sid, subj_data in subtask_results['per_subject'].items():
                lightweight['per_subject'][sid] = {}
                for st in ('binary', 'ternary', 'quaternary', 'mean_accuracy'):
                    if st in subj_data:
                        val = subj_data[st]
                        if isinstance(val, dict):
                            lightweight['per_subject'][sid][st] = {
                                k: v for k, v in val.items() if k != 'detailed_results'
                            }
                        else:
                            lightweight['per_subject'][sid][st] = val
            data['subtask_results'] = lightweight
            with open(result_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Updated {result_json.name} with subtask_results")

    # Generate unified comparison plot
    if len(all_results) >= 2:
        plot_path = results_dir / f'{args.run_tag}_unified_comparison_cross-subject_imagery.png'
        n_subjects = len(next(iter(all_results.values()))['per_subject'])
        fig = plot_unified_comparison(
            results=all_results,
            save_path=str(plot_path),
            title=f"Unified Model — Cross-Subject Comparison (Imagery, {n_subjects} Subjects)",
        )
        if fig:
            import matplotlib.pyplot as plt
            plt.close(fig)
            logger.info(f"Plot saved: {plot_path.name}")
    elif len(all_results) == 1:
        logger.info("Only one model evaluated, skipping comparison plot")

    print("\nDone.")


if __name__ == '__main__':
    main()
