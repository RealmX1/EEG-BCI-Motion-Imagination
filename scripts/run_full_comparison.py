#!/usr/bin/env python
"""
Full Model Comparison Script for EEG-BCI Project.

This script runs training and evaluation of both EEGNet and CBraMod
on all available subjects, then compares their performance.

Data Split (follows paper protocol):
- Training: Offline + Session 1 (Base + Finetune) + Session 2 Base
- Validation: Last 20% of training data (temporal split)
- Test: Session 2 Finetune (completely held out)

Features:
- Uses run_single_model.py for individual model training
- Performs statistical comparison between models
- Generates comparison visualizations

Usage:
    # Run on Motor Imagery (default paradigm, plots generated automatically)
    uv run python scripts/run_full_comparison.py

    # Run on Motor Execution
    uv run python scripts/run_full_comparison.py --paradigm movement

    # Start a NEW experiment (preserves old results/plots with datetime tag)
    uv run python scripts/run_full_comparison.py --new-run

    # Force retrain (overwrites existing cache)
    uv run python scripts/run_full_comparison.py --force-retrain

    # Run on specific subjects
    uv run python scripts/run_full_comparison.py --subjects S01 S02 S03

    # Run only EEGNet
    uv run python scripts/run_full_comparison.py --models eegnet

    # Load existing results only (no training)
    uv run python scripts/run_full_comparison.py --skip-training
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.device import set_seed, check_cuda_available, get_device
from src.utils.logging import SectionLogger, setup_logging

# Import from scripts directory
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

from _training_utils import (
    PARADIGM_CONFIG,
    TrainingResult,
    discover_subjects,
    load_cache,
    result_to_dict,
    dict_to_result,
    generate_result_filename,
)

# Import single model training function
from run_single_model import run_single_model


setup_logging('compare')
logger = logging.getLogger(__name__)

log_main = SectionLogger(logger, 'main')
log_stats = SectionLogger(logger, 'stats')
log_io = SectionLogger(logger, 'io')


# ============================================================================
# Comparison-Specific Data Classes
# ============================================================================

@dataclass
class ComparisonResult:
    """Result of model comparison."""
    n_subjects: int
    eegnet_mean: float
    eegnet_std: float
    cbramod_mean: float
    cbramod_std: float
    difference_mean: float
    difference_std: float
    paired_ttest_t: float
    paired_ttest_p: float
    wilcoxon_stat: Optional[float]
    wilcoxon_p: Optional[float]
    better_model: str
    significant: bool
    # New fields with defaults for backward compatibility
    eegnet_median: float = 0.0
    cbramod_median: float = 0.0


# ============================================================================
# Statistical Comparison
# ============================================================================

def compare_models(
    eegnet_results: List[TrainingResult],
    cbramod_results: List[TrainingResult],
) -> ComparisonResult:
    """Perform statistical comparison between EEGNet and CBraMod."""
    eegnet_by_subject = {r.subject_id: r for r in eegnet_results}
    cbramod_by_subject = {r.subject_id: r for r in cbramod_results}

    common_subjects = set(eegnet_by_subject.keys()) & set(cbramod_by_subject.keys())

    if len(common_subjects) < 2:
        raise ValueError("Need at least 2 subjects for comparison")

    if len(common_subjects) < 5:
        log_stats.warning(f"Small sample (n={len(common_subjects)}): stats may be unreliable")

    eegnet_accs = []
    cbramod_accs = []

    for subject_id in sorted(common_subjects):
        eegnet_accs.append(eegnet_by_subject[subject_id].test_acc_majority)
        cbramod_accs.append(cbramod_by_subject[subject_id].test_acc_majority)

    eegnet_accs = np.array(eegnet_accs)
    cbramod_accs = np.array(cbramod_accs)
    differences = cbramod_accs - eegnet_accs

    # Paired t-test
    t_stat, t_pvalue = stats.ttest_rel(cbramod_accs, eegnet_accs)

    # Wilcoxon signed-rank test
    try:
        w_stat, w_pvalue = stats.wilcoxon(cbramod_accs, eegnet_accs)
    except ValueError:
        w_stat, w_pvalue = None, None

    # Determine which model has higher mean accuracy
    if np.mean(cbramod_accs) > np.mean(eegnet_accs):
        higher_mean_model = 'cbramod'
    elif np.mean(eegnet_accs) > np.mean(cbramod_accs):
        higher_mean_model = 'eegnet'
    else:
        higher_mean_model = 'tie'

    return ComparisonResult(
        n_subjects=len(common_subjects),
        eegnet_mean=float(np.mean(eegnet_accs)),
        eegnet_std=float(np.std(eegnet_accs)),
        eegnet_median=float(np.median(eegnet_accs)),
        cbramod_mean=float(np.mean(cbramod_accs)),
        cbramod_std=float(np.std(cbramod_accs)),
        cbramod_median=float(np.median(cbramod_accs)),
        difference_mean=float(np.mean(differences)),
        difference_std=float(np.std(differences)),
        paired_ttest_t=float(t_stat),
        paired_ttest_p=float(t_pvalue),
        wilcoxon_stat=float(w_stat) if w_stat is not None else None,
        wilcoxon_p=float(w_pvalue) if w_pvalue is not None else None,
        better_model=higher_mean_model,
        significant=bool(t_pvalue < 0.05),
    )


# ============================================================================
# Report Generation
# ============================================================================

def print_comparison_report(
    results: Dict[str, List[TrainingResult]],
    comparison: Optional[ComparisonResult],
    task_type: str,
    paradigm: str = 'imagery',
    run_tag: Optional[str] = None,
):
    """Print a detailed comparison report."""
    paradigm_desc = PARADIGM_CONFIG[paradigm]['description']
    print("\n" + "=" * 70)
    print(" EEG-BCI MODEL COMPARISON REPORT")
    print("=" * 70)
    print(f"\nParadigm: {paradigm_desc}")
    print(f"Task: {task_type.upper()}")
    if run_tag:
        print(f"Run Tag: {run_tag}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n" + "-" * 70)
    print(" PER-SUBJECT RESULTS (Test on Session 2 Finetune, Majority Voting)")
    print("-" * 70)
    print(f"{'Subject':<10} {'EEGNet':<15} {'CBraMod':<15} {'Difference':<15}")
    print("-" * 70)

    eegnet_by_subject = {r.subject_id: r for r in results.get('eegnet', [])}
    cbramod_by_subject = {r.subject_id: r for r in results.get('cbramod', [])}

    all_subjects = sorted(set(eegnet_by_subject.keys()) | set(cbramod_by_subject.keys()))

    for subject_id in all_subjects:
        eegnet_acc = eegnet_by_subject.get(subject_id)
        cbramod_acc = cbramod_by_subject.get(subject_id)

        e_str = f"{eegnet_acc.test_acc_majority:.2%}" if eegnet_acc else "N/A"
        c_str = f"{cbramod_acc.test_acc_majority:.2%}" if cbramod_acc else "N/A"

        if eegnet_acc and cbramod_acc:
            diff = cbramod_acc.test_acc_majority - eegnet_acc.test_acc_majority
            diff_str = f"{diff:+.2%}"
            if diff > 0:
                diff_str += " (CBraMod)"
            elif diff < 0:
                diff_str += " (EEGNet)"
        else:
            diff_str = "N/A"

        print(f"{subject_id:<10} {e_str:<15} {c_str:<15} {diff_str:<15}")

    print("\n" + "-" * 70)
    print(" SUMMARY STATISTICS")
    print("-" * 70)
    print(f"{'Model':<15} {'Mean':<12} {'Median':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 70)

    for model_type in ['eegnet', 'cbramod']:
        model_results = results.get(model_type, [])
        if model_results:
            accs = [r.test_acc_majority for r in model_results]
            print(f"{model_type.upper():<15} {np.mean(accs):.2%}      {np.median(accs):.2%}      {np.std(accs):.2%}      "
                  f"{np.min(accs):.2%}      {np.max(accs):.2%}")

    if comparison:
        print("\n" + "-" * 70)
        print(" STATISTICAL COMPARISON")
        print("-" * 70)
        print(f"Subjects compared: {comparison.n_subjects}")
        if comparison.n_subjects < 5:
            print(f"  WARNING: Small sample size - results may be unreliable")
        print(f"Difference (CBraMod - EEGNet): {comparison.difference_mean:+.2%} +/- {comparison.difference_std:.2%}")
        print(f"\nPaired t-test:")
        print(f"  t-statistic: {comparison.paired_ttest_t:.4f}")
        print(f"  p-value: {comparison.paired_ttest_p:.4f}")

        if comparison.wilcoxon_stat is not None:
            print(f"\nWilcoxon signed-rank test:")
            print(f"  W-statistic: {comparison.wilcoxon_stat:.4f}")
            print(f"  p-value: {comparison.wilcoxon_p:.4f}")
        else:
            print(f"\nWilcoxon signed-rank test: N/A (requires larger sample or non-zero differences)")

        print(f"\nConclusion:")
        if comparison.significant:
            print(f"  {comparison.better_model.upper()} has significantly higher mean accuracy (p < 0.05)")
        else:
            print(f"  No significant difference between models (p = {comparison.paired_ttest_p:.4f})")

        if comparison.n_subjects < 10:
            print(f"  Note: With n={comparison.n_subjects}, statistical power is limited.")

    print("\n" + "=" * 70)


# ============================================================================
# Visualization
# ============================================================================

def generate_plot(
    results: Dict[str, List[TrainingResult]],
    comparison: ComparisonResult,
    output_path: str,
    task_type: str = 'binary',
):
    """Generate comparison plots (3-panel)."""
    chance_levels = {'binary': 0.5, 'ternary': 1/3, 'quaternary': 0.25}
    chance_level = chance_levels.get(task_type, 0.5)

    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        log_io.warning("matplotlib not installed, skipping plots")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    eegnet_results = results.get('eegnet', [])
    cbramod_results = results.get('cbramod', [])

    eegnet_by_subj = {r.subject_id: r for r in eegnet_results}
    cbramod_by_subj = {r.subject_id: r for r in cbramod_results}
    common = sorted(set(eegnet_by_subj.keys()) & set(cbramod_by_subj.keys()))

    if not common:
        log_io.warning("No common subjects for plotting")
        return

    eegnet_accs = [eegnet_by_subj[s].test_acc_majority for s in common]
    cbramod_accs = [cbramod_by_subj[s].test_acc_majority for s in common]

    # Plot 1: Bar chart
    ax1 = axes[0]
    x = np.arange(len(common))
    width = 0.35

    bars1 = ax1.bar(x - width/2, eegnet_accs, width, label='EEGNet', color='steelblue')
    bars2 = ax1.bar(x + width/2, cbramod_accs, width, label='CBraMod', color='coral')
    ax1.set_xlabel('Subject')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Per-Subject Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(common, rotation=45)
    ax1.legend()
    ax1.set_ylim([0, 1])
    ax1.axhline(y=chance_level, color='gray', linestyle='--', alpha=0.5,
                label=f'Chance ({chance_level*100:.1f}%)')

    for bar, val in zip(bars1, eegnet_accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val*100:.1f}', ha='center', va='bottom', fontsize=7)
    for bar, val in zip(bars2, cbramod_accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val*100:.1f}', ha='center', va='bottom', fontsize=7)

    # Plot 2: Box plot
    ax2 = axes[1]
    median_color = 'black'
    mean_color = '#E63946'

    bp = ax2.boxplot([eegnet_accs, cbramod_accs], tick_labels=['EEGNet', 'CBraMod'],
                     patch_artist=True,
                     showmeans=True, meanline=True,
                     meanprops={'color': mean_color, 'linewidth': 2,
                               'linestyle': (0, (3, 2))})
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('coral')
    bp['boxes'][1].set_alpha(0.7)
    for median in bp['medians']:
        median.set_color(median_color)
        median.set_linewidth(2)
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Accuracy Distribution')
    ax2.axhline(y=chance_level, color='gray', linestyle='--', alpha=0.5)

    eegnet_mean = np.mean(eegnet_accs)
    eegnet_median = np.median(eegnet_accs)
    cbramod_mean = np.mean(cbramod_accs)
    cbramod_median = np.median(cbramod_accs)

    x_offset = 0.35
    ax2.text(1 + x_offset, eegnet_mean, f'{eegnet_mean*100:.1f}',
             ha='left', va='center', fontsize=7, color=mean_color, fontweight='bold')
    ax2.text(1 + x_offset, eegnet_median, f'{eegnet_median*100:.1f}',
             ha='left', va='center', fontsize=7, color=median_color, fontweight='bold')
    ax2.text(2 + x_offset, cbramod_mean, f'{cbramod_mean*100:.1f}',
             ha='left', va='center', fontsize=7, color=mean_color, fontweight='bold')
    ax2.text(2 + x_offset, cbramod_median, f'{cbramod_median*100:.1f}',
             ha='left', va='center', fontsize=7, color=median_color, fontweight='bold')

    legend_elements = [
        Line2D([0], [0], color=median_color, linewidth=2, linestyle='-', label='Median'),
        Line2D([0], [0], color=mean_color, linewidth=2, linestyle=(0, (3, 2)), label='Mean')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=7)

    # Plot 3: Scatter plot
    ax3 = axes[2]
    ax3.scatter(eegnet_accs, cbramod_accs, s=100, alpha=0.7)
    for i, subj in enumerate(common):
        ax3.annotate(subj, (eegnet_accs[i], cbramod_accs[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

    lims = [min(min(eegnet_accs), min(cbramod_accs)) - 0.05,
            max(max(eegnet_accs), max(cbramod_accs)) + 0.05]
    ax3.plot(lims, lims, 'k--', alpha=0.5, label='Equal')
    ax3.set_xlabel('EEGNet Accuracy')
    ax3.set_ylabel('CBraMod Accuracy')
    ax3.set_title('Paired Comparison')
    ax3.set_xlim(lims)
    ax3.set_ylim(lims)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    log_io.info(f"Plot saved: {output_path}")
    plt.close()


# ============================================================================
# Result Saving
# ============================================================================

def save_full_results(
    results: Dict[str, List[TrainingResult]],
    comparison: Optional[ComparisonResult],
    task_type: str,
    paradigm: str,
    output_dir: str,
    run_tag: Optional[str] = None,
):
    """Save comprehensive results to JSON."""
    output = {
        'metadata': {
            'paradigm': paradigm,
            'paradigm_description': PARADIGM_CONFIG[paradigm]['description'],
            'task_type': task_type,
            'run_tag': run_tag,
            'timestamp': datetime.now().isoformat(),
            'n_subjects': len(set(
                r.subject_id for model_results in results.values()
                for r in model_results
            )),
        },
        'models': {},
        'comparison': None,
    }

    for model_type, model_results in results.items():
        accs = [r.test_acc_majority for r in model_results]
        output['models'][model_type] = {
            'subjects': [result_to_dict(r) for r in model_results],
            'summary': {
                'mean': float(np.mean(accs)) if accs else 0,
                'std': float(np.std(accs)) if accs else 0,
                'min': float(np.min(accs)) if accs else 0,
                'max': float(np.max(accs)) if accs else 0,
            }
        }

    if comparison:
        output['comparison'] = asdict(comparison)

    filename = generate_result_filename('comparison', paradigm, task_type, 'json', run_tag)

    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    log_io.info(f"Results saved: {output_path}")
    return output_path


def load_existing_results(results_file: str) -> Dict[str, List[TrainingResult]]:
    """Load results from a previous run."""
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = {}
    for model_type, model_data in data.get('models', data).items():
        subjects_data = model_data.get('subjects', model_data)
        if isinstance(subjects_data, dict):
            subjects_data = subjects_data.get('subjects', [])

        results[model_type] = [dict_to_result(s) for s in subjects_data]

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run full model comparison on all subjects',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run on Motor Imagery (default, plots generated automatically)
  uv run python scripts/run_full_comparison.py

  # Run on Motor Execution
  uv run python scripts/run_full_comparison.py --paradigm movement

  # Start a NEW experiment (preserves old results/plots with datetime tag)
  uv run python scripts/run_full_comparison.py --new-run

  # Force retrain (overwrites existing cache for this paradigm/task)
  uv run python scripts/run_full_comparison.py --force-retrain

  # Suppress plot generation
  uv run python scripts/run_full_comparison.py --no-plot
'''
    )

    parser.add_argument(
        '--data-root', type=str, default='data',
        help='Path to data directory (default: data)'
    )
    parser.add_argument(
        '--subjects', nargs='+', default=None,
        help='Specific subjects to run (default: all available)'
    )
    parser.add_argument(
        '--models', nargs='+', default=['eegnet', 'cbramod'],
        choices=['eegnet', 'cbramod'],
        help='Models to train (default: both)'
    )
    parser.add_argument(
        '--paradigm', type=str, default='imagery',
        choices=['imagery', 'movement'],
        help='Experiment paradigm: imagery (MI) or movement (ME) (default: imagery)'
    )
    parser.add_argument(
        '--task', type=str, default='binary',
        choices=['binary', 'ternary', 'quaternary'],
        help='Classification task (default: binary)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='results',
        help='Directory to save results (default: results)'
    )
    parser.add_argument(
        '--new-run', action='store_true',
        help='Start a NEW experiment run with datetime tag (preserves old results)'
    )
    parser.add_argument(
        '--force-retrain', action='store_true',
        help='Force retraining, overwriting cached results for this paradigm/task'
    )
    parser.add_argument(
        '--skip-training', action='store_true',
        help='Skip training entirely and load existing results file'
    )
    parser.add_argument(
        '--results-file', type=str, default=None,
        help='Path to existing results file (used with --skip-training)'
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help='Suppress plot generation (plots are generated by default)'
    )
    parser.add_argument(
        '--no-wandb', action='store_true',
        help='Disable wandb logging (default: enabled if wandb is installed)'
    )
    parser.add_argument(
        '--upload-model', action='store_true',
        help='Upload model artifacts (.pt files) to WandB (default: disabled to save bandwidth)'
    )
    parser.add_argument(
        '--no-wandb-interactive', action='store_true',
        help='Disable interactive prompts for WandB run details (prompts are enabled by default)'
    )
    parser.add_argument(
        '--use-cache-index', action='store_true',
        help='从缓存索引发现被试，而非扫描 data/ 目录（适用于仅有缓存无原始数据的场景）'
    )
    parser.add_argument(
        '--cache-index-path', type=str, default='.cache_index.json',
        help='缓存索引文件路径（默认：.cache_index.json）'
    )

    args = parser.parse_args()

    # Start timer
    start_time = time.time()

    # Check GPU
    check_cuda_available(required=True)
    device = get_device()
    log_main.info(f"Device: {device}")

    # Set seed
    set_seed(args.seed)
    log_main.info(f"Seed: {args.seed}")

    # Generate run tag if --new-run is specified
    run_tag = None
    if args.new_run:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M")
        log_main.info(f"New run: {run_tag}")

    paradigm_desc = PARADIGM_CONFIG[args.paradigm]['description']
    log_main.info(f"Paradigm: {paradigm_desc}")

    if args.skip_training:
        if args.results_file is None:
            results_dir = Path(args.output_dir)
            pattern = f'*comparison_{args.paradigm}_{args.task}*.json'
            result_files = sorted(results_dir.glob(pattern), reverse=True)
            if not result_files:
                result_files = sorted(results_dir.glob(f'comparison_{args.task}_*.json'), reverse=True)
            if not result_files:
                log_main.error("No results files found. Run training first.")
                sys.exit(1)
            args.results_file = str(result_files[0])
            log_io.info(f"Using: {args.results_file}")

        results = load_existing_results(args.results_file)
        log_io.info(f"Loaded: {args.results_file}")
    else:
        # Discover subjects
        if args.subjects:
            subjects = args.subjects
        else:
            subjects = discover_subjects(
                args.data_root,
                args.paradigm,
                args.task,
                use_cache_index=args.use_cache_index,
                cache_index_path=args.cache_index_path
            )

        if not subjects:
            log_main.error(f"No subjects in {args.data_root}")
            sys.exit(1)

        log_main.info(f"Subjects: {subjects} | Models: {args.models} | Task: {args.task}")

        if not args.force_retrain and not args.new_run:
            log_main.info("Using cache (--new-run for fresh, --force-retrain to overwrite)")

        # Collect WandB metadata once before model loop
        from scripts._wandb_setup import should_prompt_wandb, prompt_batch_session

        # Count subjects that need training (check cache for each model)
        subjects_needing_training = set()
        if args.force_retrain:
            # All subjects need training when force_retrain is set
            subjects_needing_training = set(subjects)
        else:
            for model_type in args.models:
                cache, _ = load_cache(args.output_dir, args.paradigm, args.task, run_tag, find_latest=(run_tag is None))
                cached_subjects = set(cache.get(model_type, {}).keys()) if cache else set()
                subjects_needing_training.update(set(subjects) - cached_subjects)

        wandb_session_metadata = None
        if should_prompt_wandb(
            wandb_enabled=not args.no_wandb,
            interactive=not args.no_wandb_interactive,
            has_training=bool(subjects_needing_training),
        ):
            wandb_session_metadata = prompt_batch_session(
                models=args.models,
                task=args.task,
                paradigm=args.paradigm,
                subjects_to_train=len(subjects_needing_training),
            )

        # Run training for each model using run_single_model
        results = {}
        for model_type in args.models:
            log_main.info(f"{'='*50} {model_type.upper()} {'='*50}")

            model_results, stats = run_single_model(
                model_type=model_type,
                data_root=args.data_root,
                subject_ids=subjects,
                task=args.task,
                paradigm=args.paradigm,
                output_dir=args.output_dir,
                force_retrain=args.force_retrain,
                run_tag=run_tag,
                no_wandb=args.no_wandb,
                upload_model=args.upload_model,
                wandb_interactive=False,  # Disable internal prompting
                wandb_session_metadata=wandb_session_metadata,  # Use pre-collected metadata
                cache_only=args.use_cache_index,
                cache_index_path=args.cache_index_path,
            )
            results[model_type] = model_results

    # Compare models
    comparison = None
    if 'eegnet' in results and 'cbramod' in results:
        if len(results['eegnet']) >= 2 and len(results['cbramod']) >= 2:
            try:
                comparison = compare_models(results['eegnet'], results['cbramod'])
            except ValueError as e:
                log_stats.warning(f"Cannot compare: {e}")

    print_comparison_report(results, comparison, args.task, args.paradigm, run_tag)

    output_path = save_full_results(
        results, comparison, args.task, args.paradigm, args.output_dir, run_tag
    )

    # Generate plots by default (unless --no-plot is specified)
    if not args.no_plot and comparison:
        plot_filename = generate_result_filename('comparison', args.paradigm, args.task, 'png', run_tag)
        plot_path = Path(args.output_dir) / plot_filename
        generate_plot(results, comparison, str(plot_path), task_type=args.task)

    # Log total time
    total_time = time.time() - start_time
    if total_time >= 3600:
        log_main.info(f"Total time: {total_time/3600:.1f}h")
    elif total_time >= 60:
        log_main.info(f"Total time: {total_time/60:.1f}m")
    else:
        log_main.info(f"Total time: {total_time:.1f}s")

    return 0


if __name__ == '__main__':
    sys.exit(main())
