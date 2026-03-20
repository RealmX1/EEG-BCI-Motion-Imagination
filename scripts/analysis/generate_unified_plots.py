"""
Generate proper unified model plots using plot_unified_comparison().

Loads per-subject subtask data from individual results.json files
(within-subject) or cross-subject cache, then generates the 3+1+1x3 layout.

Usage:
    uv run python scripts/analysis/generate_unified_plots.py
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.visualization.comparison import plot_unified_comparison
import matplotlib
matplotlib.use('Agg')


def build_within_subject_unified_data(results_dir: Path, run_tag: str):
    """
    Build plot_unified_comparison input from per-subject results.json files.

    Scans results/{run_tag}_{model}_within_subject/unified/{subject}/results.json
    for each model and subject.

    Returns:
        Dict matching plot_unified_comparison's expected format, or None if no data found.
    """
    plot_data = {}

    for model_type in ['eegnet', 'cbramod']:
        model_dir = results_dir / f'{run_tag}_{model_type}_within_subject' / 'unified'
        if not model_dir.exists():
            print(f"  Warning: {model_dir} not found")
            continue

        per_subject = {}
        subtask_accs = {'binary': [], 'ternary': [], 'quaternary': []}

        for subj_dir in sorted(model_dir.iterdir()):
            results_file = subj_dir / 'results.json'
            if not results_file.exists():
                continue

            with open(results_file) as f:
                data = json.load(f)

            sr = data.get('subtask_results')
            if not sr:
                continue

            subject_id = subj_dir.name
            per_subject[subject_id] = {}

            for subtask in ['binary', 'ternary', 'quaternary']:
                if subtask in sr and isinstance(sr[subtask], dict):
                    acc = sr[subtask].get('accuracy', 0)
                    n_trials = sr[subtask].get('n_trials', 0)
                    per_subject[subject_id][subtask] = {
                        'accuracy': acc,
                        'n_trials': n_trials,
                    }
                    if n_trials > 0:
                        subtask_accs[subtask].append(acc)

        if not per_subject:
            continue

        # Build subtask_results (aggregate means)
        import numpy as np
        subtask_results = {}
        all_means = []
        for subtask in ['binary', 'ternary', 'quaternary']:
            accs = subtask_accs[subtask]
            if accs:
                mean_acc = float(np.mean(accs))
                subtask_results[subtask] = {
                    'accuracy': mean_acc,
                    'std': float(np.std(accs)),
                    'n_subjects': len(accs),
                }
                all_means.append(mean_acc)
            else:
                subtask_results[subtask] = {'accuracy': 0, 'std': 0, 'n_subjects': 0}

        subtask_results['mean_accuracy'] = float(np.mean(all_means)) if all_means else 0

        plot_data[model_type] = {
            'subtask_results': subtask_results,
            'per_subject': per_subject,
        }

        n_subjects = len(per_subject)
        print(f"  {model_type}: {n_subjects} subjects loaded")
        for st in ['binary', 'ternary', 'quaternary']:
            sr_st = subtask_results[st]
            print(f"    {st}: {sr_st['accuracy']:.2%} +/- {sr_st.get('std', 0):.2%} ({sr_st['n_subjects']} subjects)")

    return plot_data if plot_data else None


def build_cross_subject_unified_data(results_dir: Path, run_tag: str):
    """
    Build plot_unified_comparison input for cross-subject runs.

    Checks for subtask_results field (added by reeval script or fixed pipeline).
    Falls back to mean-only if not available.
    """
    plot_data = {}

    for model_type in ['eegnet', 'cbramod']:
        result_file = results_dir / f'{run_tag}_cross-subject_{model_type}_imagery_unified.json'
        if not result_file.exists():
            print(f"  Warning: {result_file.name} not found")
            continue

        with open(result_file) as f:
            data = json.load(f)

        sr = data.get('subtask_results')
        if sr and 'per_subject' in sr:
            # Full subtask data available (from reeval or fixed pipeline)
            import numpy as np
            per_subject = {}
            for sid, subj_data in sr['per_subject'].items():
                per_subject[sid] = {}
                for st in ('binary', 'ternary', 'quaternary'):
                    if st in subj_data and isinstance(subj_data[st], dict):
                        per_subject[sid][st] = subj_data[st]

            subtask_results = {}
            for st in ('binary', 'ternary', 'quaternary'):
                if st in sr:
                    subtask_results[st] = sr[st]
            subtask_results['mean_accuracy'] = sr.get('mean_accuracy', 0)

            plot_data[model_type] = {
                'subtask_results': subtask_results,
                'per_subject': per_subject,
            }

            n_subjects = len(per_subject)
            print(f"  {model_type}: {n_subjects} subjects loaded (with subtask breakdown)")
            for st in ('binary', 'ternary', 'quaternary'):
                if st in subtask_results:
                    sr_st = subtask_results[st]
                    print(f"    {st}: {sr_st.get('accuracy', 0):.2%} +/- {sr_st.get('std', 0):.2%}")
        else:
            results = data.get('results', {})
            plot_data[model_type] = {
                'subtask_results': {'mean_accuracy': results.get('mean_test_acc', 0)},
                'per_subject': {},
            }
            print(f"  {model_type}: mean={results.get('mean_test_acc', 0):.2%} "
                  f"(no subtask breakdown — run reeval script first)")

    return plot_data if plot_data else None


def main():
    results_dir = project_root / 'results'

    # =============================================
    # Within-subject unified (has full subtask data)
    # =============================================
    print("=" * 60)
    print("Within-Subject Unified (20260319_1640)")
    print("=" * 60)

    within_data = build_within_subject_unified_data(results_dir, '20260319_1640')
    if within_data:
        plot_path = results_dir / '20260319_1640_unified_comparison_imagery.png'
        fig = plot_unified_comparison(
            results=within_data,
            save_path=str(plot_path),
            title="Unified Model — Within-Subject Comparison (Imagery, 21 Subjects)",
        )
        if fig:
            import matplotlib.pyplot as plt
            plt.close(fig)
            print(f"\n  Plot saved: {plot_path.name}")
        else:
            print("\n  Plot generation failed")
    else:
        print("  No within-subject unified data found")

    # =============================================
    # Cross-subject unified (no subtask breakdown saved)
    # =============================================
    print()
    print("=" * 60)
    print("Cross-Subject Unified (20260319_2102)")
    print("=" * 60)

    cross_data = build_cross_subject_unified_data(results_dir, '20260319_2102')
    if cross_data:
        # Check if any model has per_subject subtask data
        has_subtask = any(cross_data[m].get('per_subject') for m in cross_data)
        if has_subtask:
            plot_path = results_dir / '20260319_2102_unified_comparison_cross-subject_imagery.png'
            fig = plot_unified_comparison(
                results=cross_data,
                save_path=str(plot_path),
                title="Unified Model — Cross-Subject Comparison (Imagery, 21 Subjects)",
            )
            if fig:
                import matplotlib.pyplot as plt
                plt.close(fig)
                print(f"\n  Plot saved: {plot_path.name}")
            else:
                print("\n  Plot generation failed")
        else:
            print("\n  No subtask breakdown available. Run reeval script first:")
            print("  uv run python scripts/analysis/reeval_cross_subject_unified.py")
    else:
        print("  No cross-subject unified data found")

    print("\nDone.")


if __name__ == '__main__':
    main()
