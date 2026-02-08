#!/usr/bin/env python
"""
Quick WandB data analysis for CBraMod vs EEGNet comparison.

Usage:
    uv run python scripts/analysis/analyze_wandb_data.py \
        --data analysis_data/wandb_data.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def analyze_training_dynamics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze training dynamics from a single run."""
    train_loss = data.get('train_loss', [])
    val_loss = data.get('val_loss', [])
    train_acc = data.get('train_acc', [])
    val_acc = data.get('val_acc', [])

    result = {
        'total_epochs': len(train_loss),
        'test_accuracy': data.get('test_accuracy', 0),
        'test_majority_accuracy': data.get('test_majority_accuracy', 0),
        'best_val_accuracy': data.get('best_val_accuracy', 0),
    }

    if len(train_acc) >= 3 and len(val_acc) >= 3:
        # Final train-val gap (overfitting indicator)
        final_train = np.mean(train_acc[-3:])
        final_val = np.mean(val_acc[-3:])
        result['final_train_val_gap'] = final_train - final_val

        # Convergence analysis
        if len(val_loss) >= 5:
            last_5_val_loss = val_loss[-5:]
            result['val_loss_stable'] = np.std(last_5_val_loss) < 0.05

    if len(val_loss) >= 2:
        result['final_val_loss'] = val_loss[-1]
        result['min_val_loss'] = min(val_loss)

    return result


def print_subject_analysis(subjects_data: Dict[str, List[Dict]], task: str = 'binary'):
    """Print analysis for all subjects."""
    print(f"\n{'='*80}")
    print(f"  WandB Training Analysis - {task.upper()} Task")
    print(f"{'='*80}")

    # Collect data
    eegnet_results = {}
    cbramod_results = {}

    for key, runs in subjects_data.items():
        if not runs or f'_{task}' not in key:
            continue

        parts = key.split('_')
        subject_id = parts[0]
        model_type = parts[1]

        latest_run = runs[0]  # Most recent
        analysis = analyze_training_dynamics(latest_run)

        if model_type == 'eegnet':
            eegnet_results[subject_id] = analysis
        elif model_type == 'cbramod':
            cbramod_results[subject_id] = analysis

    # Print comparison table
    all_subjects = sorted(set(eegnet_results.keys()) | set(cbramod_results.keys()))

    print(f"\n{'Subject':<8} {'EEGNet Acc':<12} {'CBraMod Acc':<12} {'Diff':<10} "
          f"{'EEGNet Ep':<10} {'CBraMod Ep':<10} {'CBraMod Gap':<12}")
    print("-" * 80)

    differences = []
    for subject in all_subjects:
        e_data = eegnet_results.get(subject, {})
        c_data = cbramod_results.get(subject, {})

        e_acc = e_data.get('test_accuracy', 0)
        c_acc = c_data.get('test_accuracy', 0)
        e_epochs = e_data.get('total_epochs', 0)
        c_epochs = c_data.get('total_epochs', 0)
        c_gap = c_data.get('final_train_val_gap', 0)

        diff = c_acc - e_acc
        differences.append(diff)

        winner = "CBraMod" if diff > 0.01 else ("EEGNet" if diff < -0.01 else "Tie")

        print(f"{subject:<8} {e_acc:<12.2%} {c_acc:<12.2%} {diff:+.2%}     "
              f"{e_epochs:<10} {c_epochs:<10} {c_gap:+.2%}")

    # Summary statistics
    print("\n" + "-" * 80)
    print("SUMMARY STATISTICS")
    print("-" * 80)

    e_accs = [eegnet_results[s]['test_accuracy'] for s in all_subjects if s in eegnet_results]
    c_accs = [cbramod_results[s]['test_accuracy'] for s in all_subjects if s in cbramod_results]

    if e_accs:
        print(f"EEGNet:  Mean={np.mean(e_accs):.2%}, Std={np.std(e_accs):.2%}, "
              f"Min={np.min(e_accs):.2%}, Max={np.max(e_accs):.2%}")
    if c_accs:
        print(f"CBraMod: Mean={np.mean(c_accs):.2%}, Std={np.std(c_accs):.2%}, "
              f"Min={np.min(c_accs):.2%}, Max={np.max(c_accs):.2%}")

    if differences:
        print(f"\nMean difference (CBraMod - EEGNet): {np.mean(differences):+.2%}")

    # Identify problem subjects
    print("\n" + "-" * 80)
    print("PROBLEM SUBJECTS (CBraMod < 70% or much worse than EEGNet)")
    print("-" * 80)

    for subject in all_subjects:
        c_data = cbramod_results.get(subject, {})
        e_data = eegnet_results.get(subject, {})

        c_acc = c_data.get('test_accuracy', 0)
        e_acc = e_data.get('test_accuracy', 0)
        diff = c_acc - e_acc

        if c_acc < 0.70 or diff < -0.10:
            c_gap = c_data.get('final_train_val_gap', 0)
            c_epochs = c_data.get('total_epochs', 0)
            c_val_loss = c_data.get('final_val_loss', 0)

            print(f"\n{subject}:")
            print(f"  CBraMod: {c_acc:.2%} | EEGNet: {e_acc:.2%} | Diff: {diff:+.2%}")
            print(f"  Train-Val Gap: {c_gap:+.2%}")
            print(f"  Epochs trained: {c_epochs}")
            if c_val_loss:
                print(f"  Final val loss: {c_val_loss:.4f}")

            # Diagnosis
            if c_gap > 0.15:
                print(f"  [!] Severe overfitting detected")
            elif c_gap > 0.08:
                print(f"  [!] Moderate overfitting detected")

            if c_acc < e_acc * 0.8:
                print(f"  [!] CBraMod-specific issue (EEGNet works well)")


def main():
    parser = argparse.ArgumentParser(description='Analyze WandB training data')
    parser.add_argument('--data', type=str, default='analysis_data/wandb_data.json',
                        help='Path to WandB data JSON')
    parser.add_argument('--task', type=str, default='binary',
                        choices=['binary', 'ternary'],
                        help='Task type to analyze')
    args = parser.parse_args()

    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)

    print(f"Loading data from: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        subjects_data = json.load(f)

    print(f"Loaded {len(subjects_data)} run groups")

    # Analyze
    print_subject_analysis(subjects_data, args.task)

    # Also analyze ternary if binary was selected
    if args.task == 'binary':
        print_subject_analysis(subjects_data, 'ternary')


if __name__ == '__main__':
    main()
