#!/usr/bin/env python
"""
Visualization script for EEGNet vs CBraMod (Fine-tuned Foundation Model) comparison.

Generates publication-quality figures comparing binary classification performance.

Usage:
    uv run python scripts/visualize_comparison.py results/comparison_binary_20260103_010614.json
    uv run python scripts/visualize_comparison.py results/comparison_binary_20260103_010614.json --output figures/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_results(results_file: str) -> Dict:
    """Load results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def create_visualizations(data: Dict, output_dir: Path):
    """Create all visualization figures."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("Error: matplotlib is required. Install with: uv pip install matplotlib")
        sys.exit(1)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['figure.dpi'] = 150

    # Get task type and calculate chance level
    task_type = data.get('metadata', {}).get('task_type', 'binary')
    chance_levels = {'binary': 50.0, 'ternary': 100/3, 'quaternary': 25.0}
    chance_level = chance_levels.get(task_type, 50.0)

    # Extract data
    eegnet_data = data['models']['eegnet']['subjects']
    cbramod_data = data['models']['cbramod']['subjects']
    comparison = data['comparison']

    subjects = [s['subject_id'] for s in eegnet_data]
    eegnet_acc = [s['test_acc_majority'] for s in eegnet_data]
    cbramod_acc = [s['test_acc_majority'] for s in cbramod_data]
    eegnet_val = [s['best_val_acc'] for s in eegnet_data]
    cbramod_val = [s['best_val_acc'] for s in cbramod_data]
    eegnet_time = [s['training_time'] for s in eegnet_data]
    cbramod_time = [s['training_time'] for s in cbramod_data]
    eegnet_epochs = [s['epochs_trained'] for s in eegnet_data]
    cbramod_epochs = [s['epochs_trained'] for s in cbramod_data]

    # Colors
    eegnet_color = '#2E86AB'  # Blue
    cbramod_color = '#E94F37'  # Red/Coral

    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Figure 1: Main Comparison (4-panel figure)
    # =========================================================================
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Panel A: Per-subject accuracy comparison (bar chart)
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(subjects))
    width = 0.35

    bars1 = ax1.bar(x - width/2, [a*100 for a in eegnet_acc], width,
                    label='EEGNet', color=eegnet_color, edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, [a*100 for a in cbramod_acc], width,
                    label='CBraMod', color=cbramod_color, edgecolor='black', linewidth=0.5)

    ax1.axhline(y=chance_level, color='gray', linestyle='--', alpha=0.7,
                label=f'Chance ({chance_level:.1f}%)')
    ax1.set_xlabel('Subject')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('A. Per-Subject Classification Accuracy', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(subjects)
    ax1.set_ylim([0, 100])
    ax1.legend(loc='upper right')

    # Add value labels on bars
    for bar, val in zip(bars1, eegnet_acc):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val*100:.1f}', ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, cbramod_acc):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val*100:.1f}', ha='center', va='bottom', fontsize=8)

    # Panel B: Box plot comparison
    ax2 = fig.add_subplot(gs[0, 1])
    bp = ax2.boxplot([[a*100 for a in eegnet_acc], [a*100 for a in cbramod_acc]],
                     labels=['EEGNet', 'CBraMod'],
                     patch_artist=True,
                     widths=0.6)

    bp['boxes'][0].set_facecolor(eegnet_color)
    bp['boxes'][1].set_facecolor(cbramod_color)
    for box in bp['boxes']:
        box.set_alpha(0.7)

    # Add individual points
    for i, (acc, color) in enumerate([(eegnet_acc, eegnet_color), (cbramod_acc, cbramod_color)]):
        x_jitter = np.random.normal(i+1, 0.04, len(acc))
        ax2.scatter(x_jitter, [a*100 for a in acc], color=color,
                   edgecolor='black', linewidth=0.5, s=60, zorder=3, alpha=0.8)

    ax2.axhline(y=chance_level, color='gray', linestyle='--', alpha=0.7)
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('B. Accuracy Distribution', fontweight='bold')
    # Adjust y-axis lower limit based on chance level
    ax2.set_ylim([max(0, chance_level - 15), 100])

    # Add significance annotation
    y_max = max(max(eegnet_acc), max(cbramod_acc)) * 100 + 5
    ax2.plot([1, 1, 2, 2], [y_max, y_max+2, y_max+2, y_max], 'k-', linewidth=1)
    sig_text = f"p = {comparison['paired_ttest_p']:.3f}"
    if comparison['significant']:
        sig_text += " *"
    ax2.text(1.5, y_max+3, sig_text, ha='center', fontsize=10)

    # Panel C: Paired scatter plot
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter([a*100 for a in eegnet_acc], [a*100 for a in cbramod_acc],
               s=120, c='#333333', edgecolor='white', linewidth=1.5, zorder=3)

    # Add subject labels
    for i, subj in enumerate(subjects):
        ax3.annotate(subj, (eegnet_acc[i]*100 + 1, cbramod_acc[i]*100 + 1),
                    fontsize=9, alpha=0.8)

    # Diagonal line (equal performance)
    lims = [40, 100]
    ax3.plot(lims, lims, 'k--', alpha=0.5, label='Equal Performance')
    ax3.fill_between(lims, lims, [100, 100], alpha=0.1, color=cbramod_color, label='CBraMod Better')
    ax3.fill_between(lims, [40, 40], lims, alpha=0.1, color=eegnet_color, label='EEGNet Better')

    ax3.set_xlabel('EEGNet Accuracy (%)')
    ax3.set_ylabel('CBraMod Accuracy (%)')
    ax3.set_title('C. Paired Subject Comparison', fontweight='bold')
    ax3.set_xlim(lims)
    ax3.set_ylim(lims)
    ax3.set_aspect('equal')
    ax3.legend(loc='upper left', fontsize=9)

    # Panel D: Summary statistics
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    # Create summary table
    summary_text = f"""
    BINARY CLASSIFICATION RESULTS
    ============================================

    Task: Thumb (1) vs Pinky (4) Discrimination
    Subjects: {comparison['n_subjects']}

    +-------------------------------------------+
    | MODEL PERFORMANCE (Majority Voting)       |
    +-------------------------------------------+
    | EEGNet:   {comparison['eegnet_mean']*100:5.1f}% +/- {comparison['eegnet_std']*100:4.1f}%           |
    | CBraMod:  {comparison['cbramod_mean']*100:5.1f}% +/- {comparison['cbramod_std']*100:4.1f}%           |
    |                                           |
    | Difference: {comparison['difference_mean']*100:+5.1f}% +/- {comparison['difference_std']*100:4.1f}%        |
    +-------------------------------------------+

    STATISTICAL ANALYSIS
    --------------------------------------------
    Paired t-test:
      t = {comparison['paired_ttest_t']:.3f}, p = {comparison['paired_ttest_p']:.4f}

    Wilcoxon signed-rank:
      W = {comparison['wilcoxon_stat']:.1f}, p = {comparison['wilcoxon_p']:.4f}

    ============================================
    CONCLUSION: {'EEGNet significantly outperforms CBraMod' if comparison['significant'] else 'No significant difference'}
    {'(p < 0.05)' if comparison['significant'] else ''}
    """

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('EEGNet vs CBraMod: Binary Finger Classification Comparison',
                fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(output_dir / 'comparison_main.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'comparison_main.pdf', bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print(f"Saved: {output_dir / 'comparison_main.png'}")
    plt.close()

    # =========================================================================
    # Figure 2: Training Dynamics
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel A: Training time comparison
    ax = axes[0]
    x = np.arange(len(subjects))
    bars1 = ax.bar(x - width/2, eegnet_time, width, label='EEGNet', color=eegnet_color)
    bars2 = ax.bar(x + width/2, cbramod_time, width, label='CBraMod', color=cbramod_color)
    ax.set_xlabel('Subject')
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('A. Training Time per Subject', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.legend()

    # Panel B: Epochs to convergence
    ax = axes[1]
    bars1 = ax.bar(x - width/2, eegnet_epochs, width, label='EEGNet', color=eegnet_color)
    bars2 = ax.bar(x + width/2, cbramod_epochs, width, label='CBraMod', color=cbramod_color)
    ax.set_xlabel('Subject')
    ax.set_ylabel('Epochs (Early Stopping)')
    ax.set_title('B. Convergence Speed', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.legend()

    # Panel C: Validation vs Test accuracy (generalization gap)
    ax = axes[2]
    eegnet_gap = [v - t for v, t in zip(eegnet_val, eegnet_acc)]
    cbramod_gap = [v - t for v, t in zip(cbramod_val, cbramod_acc)]

    bars1 = ax.bar(x - width/2, [g*100 for g in eegnet_gap], width,
                   label='EEGNet', color=eegnet_color)
    bars2 = ax.bar(x + width/2, [g*100 for g in cbramod_gap], width,
                   label='CBraMod', color=cbramod_color)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Subject')
    ax.set_ylabel('Val - Test Accuracy (%)')
    ax.set_title('C. Generalization Gap', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.legend()

    plt.suptitle('Training Dynamics: EEGNet vs CBraMod', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'training_dynamics.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'training_dynamics.png'}")
    plt.close()

    # =========================================================================
    # Figure 3: Effect Size Visualization
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 6))

    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt((comparison['eegnet_std']**2 + comparison['cbramod_std']**2) / 2)
    cohens_d = (comparison['eegnet_mean'] - comparison['cbramod_mean']) / pooled_std

    # Forest plot style
    differences = [e - c for e, c in zip(eegnet_acc, cbramod_acc)]
    y_positions = np.arange(len(subjects))

    # Individual subject differences
    ax.barh(y_positions, [d*100 for d in differences], height=0.6,
           color=[eegnet_color if d > 0 else cbramod_color for d in differences],
           edgecolor='black', linewidth=0.5, alpha=0.7)

    # Mean difference line
    mean_diff = comparison['difference_mean'] * 100
    ax.axvline(x=-mean_diff, color='black', linestyle='-', linewidth=2,
              label=f'Mean: {-mean_diff:.1f}%')

    # Zero line
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(subjects)
    ax.set_xlabel('EEGNet - CBraMod Accuracy Difference (%)')
    ax.set_title(f'Per-Subject Performance Difference\n(Cohen\'s d = {cohens_d:.2f}, "Large" effect)',
                fontweight='bold')

    # Add legend patches
    eegnet_patch = mpatches.Patch(color=eegnet_color, alpha=0.7, label='EEGNet Better')
    cbramod_patch = mpatches.Patch(color=cbramod_color, alpha=0.7, label='CBraMod Better')
    ax.legend(handles=[eegnet_patch, cbramod_patch], loc='lower right')

    plt.tight_layout()
    plt.savefig(output_dir / 'effect_size.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'effect_size.png'}")
    plt.close()

    print(f"\nAll figures saved to: {output_dir}")


def print_summary(data: Dict):
    """Print text summary of results."""
    comparison = data['comparison']

    print("\n" + "="*60)
    print(" BINARY CLASSIFICATION: EEGNet vs CBraMod (Foundation Model)")
    print("="*60)

    print("\n[Performance Summary]")
    print(f"  EEGNet:   {comparison['eegnet_mean']*100:.1f}% +/- {comparison['eegnet_std']*100:.1f}%")
    print(f"  CBraMod:  {comparison['cbramod_mean']*100:.1f}% +/- {comparison['cbramod_std']*100:.1f}%")
    print(f"  Delta:    {-comparison['difference_mean']*100:.1f}% (EEGNet advantage)")

    print("\n[Statistical Tests]")
    print(f"  Paired t-test: t={comparison['paired_ttest_t']:.3f}, p={comparison['paired_ttest_p']:.4f}")
    if comparison['wilcoxon_stat']:
        print(f"  Wilcoxon:      W={comparison['wilcoxon_stat']:.1f}, p={comparison['wilcoxon_p']:.4f}")

    # Effect size
    pooled_std = np.sqrt((comparison['eegnet_std']**2 + comparison['cbramod_std']**2) / 2)
    cohens_d = (comparison['eegnet_mean'] - comparison['cbramod_mean']) / pooled_std
    effect_label = "Small" if abs(cohens_d) < 0.5 else ("Medium" if abs(cohens_d) < 0.8 else "Large")
    print(f"  Cohen's d:     {cohens_d:.2f} ({effect_label} effect)")

    print("\n[Key Findings]")
    if comparison['significant']:
        print("  [*] EEGNet SIGNIFICANTLY outperforms CBraMod (p < 0.05)")
    else:
        print("  [ ] No significant difference between models")

    print("\n[Interpretation]")
    print("  The task-specific EEGNet architecture outperforms the")
    print("  fine-tuned CBraMod foundation model on this finger BCI task.")
    print("  This may be due to:")
    print("    1. Domain mismatch (CBraMod pretrained on different EEG tasks)")
    print("    2. Channel mismatch (128 vs 19 channels)")
    print("    3. Insufficient fine-tuning data/epochs")
    print("    4. Task-specific architectures may be optimal for narrow tasks")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize EEGNet vs CBraMod comparison results'
    )
    parser.add_argument(
        'results_file',
        help='Path to comparison results JSON file'
    )
    parser.add_argument(
        '--output', '-o', type=str, default='figures',
        help='Output directory for figures (default: figures)'
    )

    args = parser.parse_args()

    # Load data
    data = load_results(args.results_file)

    # Print summary
    print_summary(data)

    # Create visualizations
    output_dir = Path(args.output)
    create_visualizations(data, output_dir)


if __name__ == '__main__':
    main()
