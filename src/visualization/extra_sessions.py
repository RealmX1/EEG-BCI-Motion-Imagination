"""
Extra online sessions experiment visualization.

Generates plots showing how model performance changes as additional
online session data is progressively added to training.

Follows the project's standard plotting conventions from comparison.py:
- PlotDataSource-based data flow
- MODEL_COLORS for consistent coloring
- Alpha/hatch conventions: current=solid, historical=faded+hatched
- Standard font sizes and axis configuration
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from ..config.constants import MODEL_COLORS
from ..results.dataclasses import PlotDataSource, TrainingResult
from .plots import (
    CHANCE_LEVELS, annotate_bars_with_leaders, accuracy_ylim,
    separate_paired_labels, draw_label_with_leader,
)

logger = logging.getLogger(__name__)


def _build_plot_data_sources(
    all_results: Dict,
    subjects_with_sessions: Dict[str, List[int]],
) -> List[PlotDataSource]:
    """Convert extra sessions results into PlotDataSource objects.

    Each (model, step) combination becomes one PlotDataSource.
    Baseline step is marked as historical (alpha=0.4, hatched).
    Progressive steps are marked as current (alpha=1.0, solid).

    Returns sorted list: baseline sources first, then progressive steps.
    """
    subjects = sorted(subjects_with_sessions.keys())

    # Determine all session steps
    all_sessions = sorted(set(s for ss in subjects_with_sessions.values() for s in ss))
    step_keys = ['baseline'] + [f'sess{s:02d}' for s in all_sessions]
    step_labels = ['Baseline'] + [f'+Sess{s:02d}' for s in all_sessions]

    # Hatch patterns: baseline densest, progressively lighter to show data growth
    # baseline='///', step1='..', step2='.', final='' (solid = most data)
    n_progressive = len(all_sessions)
    step_hatches = ['///']  # baseline
    if n_progressive >= 3:
        step_hatches += ['...', '..', '']
    elif n_progressive == 2:
        step_hatches += ['..', '']
    elif n_progressive == 1:
        step_hatches += ['']
    # Trim to actual number of steps
    step_hatches = step_hatches[:len(step_keys)]

    # Alpha: baseline faded, progressive steps increasingly solid
    step_alphas = [0.4]  # baseline
    if n_progressive >= 1:
        for i in range(n_progressive):
            # Ramp from 0.55 to 1.0
            alpha = 0.55 + 0.45 * (i / max(n_progressive - 1, 1))
            step_alphas.append(alpha)

    sources = []
    for model_type, model_data in sorted(all_results.items()):
        for si, (step_key, step_label) in enumerate(zip(step_keys, step_labels)):
            results = []
            for subject_id in subjects:
                subj = model_data.get(subject_id, {})
                step = subj.get(step_key, {})
                acc = step.get('test_acc_majority', step.get('test_acc'))
                if acc is not None:
                    results.append(TrainingResult(
                        subject_id=subject_id,
                        task_type=step.get('task_type', ''),
                        model_type=model_type,
                        best_val_acc=step.get('best_val_acc', 0),
                        test_acc=step.get('test_acc', acc),
                        test_acc_majority=acc,
                        epochs_trained=step.get('epochs_trained', 0),
                        training_time=step.get('training_time', 0),
                    ))

            if results:
                is_baseline = (step_key == 'baseline')
                sources.append(PlotDataSource(
                    model_type=model_type,
                    results=results,
                    is_current_run=not is_baseline,
                    label=f'{model_type.upper()} {step_label}',
                    hatch=step_hatches[si],
                ))

    return sources


def _extract_step_accuracies(
    all_results: Dict,
    subjects_with_sessions: Dict[str, List[int]],
) -> Dict[str, Dict[str, List[float]]]:
    """Extract per-step accuracies for line plot.

    Returns:
        {model_type: {step_label: [acc_per_subject]}}
    """
    subjects = sorted(subjects_with_sessions.keys())
    all_sessions = sorted(set(s for ss in subjects_with_sessions.values() for s in ss))
    step_keys = ['baseline'] + [f'sess{s:02d}' for s in all_sessions]
    step_labels = ['Baseline'] + [f'+Sess{s:02d}' for s in all_sessions]

    output = {}
    for model_type, model_data in all_results.items():
        step_accs = {}
        for step_key, step_label in zip(step_keys, step_labels):
            accs = []
            for subject_id in subjects:
                subj = model_data.get(subject_id, {})
                step = subj.get(step_key, {})
                acc = step.get('test_acc_majority', step.get('test_acc'))
                accs.append(acc if acc is not None else float('nan'))
            step_accs[step_label] = accs
        output[model_type] = step_accs

    return output


def generate_extra_sessions_combined_plot(
    all_results: Dict,
    subjects_with_sessions: Dict[str, List[int]],
    output_path: str,
    paradigm: str = 'imagery',
    task: str = 'binary',
):
    """
    Generate combined plot for extra sessions experiment.

    Layout (3 rows, standard project conventions):
    ┌──────────────────────────────────────┐
    │  Row 1: Bar chart per subject        │  (full width, standard style)
    ├──────────────────────────────────────┤
    │  Row 2: Progression line plot        │  (full width, core finding)
    ├──────────────────┬───────────────────┤
    │  Row 3L: Box plot │  Row 3R: Scatter │  (standard panels)
    └──────────────────┴───────────────────┘

    Args:
        all_results: {model_type: {subject_id: {step_key: result_dict}}}
        subjects_with_sessions: {subject_id: [session_numbers]}
        output_path: Path to save PNG
        paradigm: 'imagery' or 'movement'
        task: 'binary' or 'ternary'
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.lines import Line2D
    except ImportError:
        logger.warning("matplotlib not installed, skipping plot")
        return

    chance_level = CHANCE_LEVELS.get(task, 0.5)
    colors = MODEL_COLORS
    subjects = sorted(subjects_with_sessions.keys())
    n_subjects = len(subjects)

    # Build data sources and step data
    data_sources = _build_plot_data_sources(all_results, subjects_with_sessions)
    step_accs = _extract_step_accuracies(all_results, subjects_with_sessions)
    models = [m for m in ['eegnet', 'cbramod'] if m in step_accs]

    if not data_sources:
        logger.warning("No data for plotting")
        return

    # ====== Figure: nested GridSpec for independent row gaps ======
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    fig = plt.figure(figsize=(14, 18))
    fig.subplots_adjust(top=0.95)
    # Outer: top half (bar + line) and bottom half (box + scatter)
    outer = GridSpec(2, 1, height_ratios=[1.8, 1.4], hspace=0.10,
                     top=0.95, bottom=0.05, left=0.06, right=0.96)
    # Top section: bar and line stacked with moderate gap
    inner_top = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], hspace=0.22)
    ax_bar = fig.add_subplot(inner_top[0])
    ax_line = fig.add_subplot(inner_top[1])
    # Bottom section: box and scatter side by side
    inner_bottom = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.25)
    ax_box = fig.add_subplot(inner_bottom[0])
    ax_scatter = fig.add_subplot(inner_bottom[1])
    ax_box.set_box_aspect(1)
    ax_scatter.set_box_aspect(1)

    # =========================================================================
    # Panel 1: Bar Chart (per-subject accuracy, standard style)
    # =========================================================================
    n_sources = len(data_sources)
    bar_width = 0.8 / n_sources
    x_base = np.arange(n_subjects)

    # Map hatch density → alpha for progressive visual distinction
    _HATCH_ALPHA = {'///': 0.4, '...': 0.55, '..': 0.75, '.': 0.85, '': 1.0}

    bar_entries = []
    for i, source in enumerate(data_sources):
        x_positions = x_base + (i - (n_sources - 1) / 2) * bar_width

        result_by_subj = {r.subject_id: r.test_acc_majority for r in source.results}
        accs = [result_by_subj.get(s, float('nan')) for s in subjects]

        hatch = source.hatch if source.hatch is not None else ('' if source.is_current_run else '///')
        alpha = _HATCH_ALPHA.get(hatch, 1.0 if source.is_current_run else 0.4)
        edgecolor = 'black' if hatch == '' else 'gray'
        linewidth = 1.5 if hatch == '' else 0.5

        bars = ax_bar.bar(
            x_positions, accs, bar_width,
            label=source.label,
            color=colors[source.model_type],
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=linewidth,
            hatch=hatch,
        )
        bar_entries.append((bars, accs, source.is_current_run))

    annotate_bars_with_leaders(ax_bar, bar_entries)

    ax_bar.set_xlabel('Subject')
    ax_bar.set_ylabel('Test Accuracy')
    title = f'Per-Subject Accuracy by Training Data Stage ({paradigm.title()} {task.title()})'
    ax_bar.set_title(title)
    ax_bar.set_xticks(x_base)
    ax_bar.set_xticklabels(subjects, rotation=45, ha='right')
    ax_bar.axhline(y=chance_level, color='gray', linestyle='--', alpha=0.5,
                   label=f'Chance ({chance_level*100:.1f}%)')
    ax_bar.set_ylim(accuracy_ylim(task))
    ax_bar.legend(loc='lower right', fontsize=8, ncol=2)

    # =========================================================================
    # Panel 2: Progression Line Plot (core finding)
    # =========================================================================
    step_labels = list(next(iter(step_accs.values())).keys())
    n_steps = len(step_labels)
    x_steps = np.arange(n_steps)

    model_mean_data = {}  # {model_type: (step_means, step_ns, color)}
    for model_type in models:
        model_steps = step_accs[model_type]
        color = colors[model_type]

        # Per-subject thin lines (alpha=0.3, dashed — like historical overlay)
        for subj_idx, subject_id in enumerate(subjects):
            y_vals = [list(model_steps.values())[si][subj_idx] for si in range(n_steps)]
            ax_line.plot(x_steps, y_vals, color=color, alpha=0.3, linewidth=1,
                         linestyle='--', marker='o', markersize=3)

        # Mean ± SE (bold line — like current run)
        step_means = []
        step_ses = []
        step_ns = []
        for si in range(n_steps):
            vals = [v for v in list(model_steps.values())[si] if not np.isnan(v)]
            mean = np.mean(vals) if vals else float('nan')
            se = np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            step_means.append(mean)
            step_ses.append(se)
            step_ns.append(len(vals))

        step_means_arr = np.array(step_means)
        step_ses_arr = np.array(step_ses)

        ax_line.plot(x_steps, step_means_arr, color=color, linewidth=2.5,
                     marker='o', markersize=7, label=f'{model_type.upper()} (mean)',
                     zorder=5)
        ax_line.fill_between(x_steps, step_means_arr - step_ses_arr,
                             step_means_arr + step_ses_arr,
                             color=color, alpha=0.15)
        model_mean_data[model_type] = (step_means, step_ns, color)

    # Annotate mean values: higher model above, lower model below
    _label_pad = 0.015
    _label_bbox = dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1.5)
    for xi in range(n_steps):
        entries = [(model_mean_data[m][0][xi], model_mean_data[m][1][xi],
                    model_mean_data[m][2], m)
                   for m in models if not np.isnan(model_mean_data[m][0][xi])]
        if len(entries) == 2:
            # Sort: higher value first
            entries_sorted = sorted(entries, key=lambda e: e[0], reverse=True)
            hi_mv, hi_n, hi_clr, _ = entries_sorted[0]
            lo_mv, lo_n, lo_clr, _ = entries_sorted[1]
            ax_line.text(xi, hi_mv + _label_pad, f'{hi_mv:.1%} (n={hi_n})',
                         ha='center', va='bottom', fontsize=8,
                         color=hi_clr, fontweight='bold', bbox=_label_bbox)
            ax_line.text(xi, lo_mv - _label_pad, f'{lo_mv:.1%} (n={lo_n})',
                         ha='center', va='top', fontsize=8,
                         color=lo_clr, fontweight='bold', bbox=_label_bbox)
        else:
            for mv, n_valid, clr, _ in entries:
                ax_line.text(xi, mv + _label_pad, f'{mv:.1%} (n={n_valid})',
                             ha='center', va='bottom', fontsize=8,
                             color=clr, fontweight='bold', bbox=_label_bbox)

    ax_line.set_xticks(x_steps)
    ax_line.set_xticklabels(step_labels, fontsize=10)
    ax_line.set_ylabel('Test Accuracy')
    ax_line.set_ylim(accuracy_ylim(task, top_pad=0.08))
    ax_line.axhline(y=chance_level, color='gray', linestyle='--', alpha=0.5,
                    label=f'Chance ({chance_level*100:.1f}%)')
    ax_line.legend(loc='lower right', fontsize=8)
    ax_line.set_title('Performance Progression with Additional Training Data')
    ax_line.grid(axis='y', alpha=0.3)

    # =========================================================================
    # Panel 3L: Box Plot (standard style from comparison.py)
    # =========================================================================
    median_color = 'black'
    mean_color = '#E63946'

    box_data = []
    box_labels = []
    box_colors = []
    box_alphas = []
    box_hatches = []

    for source in data_sources:
        accs = [r.test_acc_majority for r in source.results]
        if accs:
            box_data.append(accs)
            box_labels.append(source.label)
            box_colors.append(colors[source.model_type])
            box_alphas.append(1.0 if source.is_current_run else 0.4)
            box_hatches.append(source.hatch if source.hatch is not None
                               else ('' if source.is_current_run else '///'))

    if box_data:
        # Two-row x-axis: Row 1 = step name, Row 2 = model name (centered)
        # Extract step names from source labels (e.g. "CBRAMOD Baseline" → "Baseline")
        step_only_labels = []
        model_spans = {}  # {model_type: (first_idx, last_idx)}
        for i, source in enumerate(data_sources):
            parts = source.label.split(maxsplit=1)
            step_name = parts[1] if len(parts) > 1 else parts[0]
            n_subj = len(box_data[i]) if i < len(box_data) else 0
            step_only_labels.append(f'{step_name}\n(n={n_subj})')
            mt = source.model_type
            if mt not in model_spans:
                model_spans[mt] = [i, i]
            else:
                model_spans[mt][1] = i

        bp = ax_box.boxplot(
            box_data, labels=step_only_labels, patch_artist=True,
            showmeans=True, meanline=True,
            meanprops={'color': mean_color, 'linewidth': 2, 'linestyle': (0, (3, 2))},
        )
        ax_box.tick_params(axis='x', labelsize=7, pad=2)

        # Add model name row below tick labels
        for mt, (first, last) in model_spans.items():
            center_x = (first + 1 + last + 1) / 2  # boxplot positions are 1-indexed
            ax_box.text(center_x, -0.08, mt.upper(),
                        ha='center', va='top', fontsize=8, fontweight='bold',
                        color=colors[mt], transform=ax_box.get_xaxis_transform())

        for patch, color_val, alpha, hatch in zip(bp['boxes'], box_colors, box_alphas, box_hatches):
            patch.set_facecolor(color_val)
            patch.set_alpha(alpha)
            patch.set_hatch(hatch)

        for median in bp['medians']:
            median.set_color(median_color)
            median.set_linewidth(2)

        # Annotate mean/median values (leader starts at box right edge)
        for i, (source, accs_list) in enumerate(zip(data_sources, box_data)):
            mean_val = np.mean(accs_list)
            median_val = np.median(accs_list)
            # Box right edge = position + half box width
            box_right = max(v[0] for v in bp['boxes'][i].get_path().vertices)
            fontweight = 'bold' if source.is_current_run else 'normal'
            adj_mean, adj_med = separate_paired_labels(mean_val, median_val, min_gap=0.02)
            draw_label_with_leader(
                ax_box, mean_val, adj_mean, box_right,
                f'{mean_val*100:.1f}', color=mean_color, fontsize=7,
                fontweight=fontweight)
            draw_label_with_leader(
                ax_box, median_val, adj_med, box_right,
                f'{median_val*100:.1f}', color=median_color, fontsize=7,
                fontweight=fontweight)

        legend_elements = [
            Line2D([0], [0], color=median_color, linewidth=2, linestyle='-', label='Median'),
            Line2D([0], [0], color=mean_color, linewidth=2, linestyle=(0, (3, 2)), label='Mean'),
        ]
        ax_box.legend(handles=legend_elements, loc='lower right', fontsize=7)

    ax_box.set_ylabel('Test Accuracy')
    ax_box.set_title('Accuracy Distribution')
    ax_box.axhline(y=chance_level, color='gray', linestyle='--', alpha=0.5)
    ax_box.set_ylim(accuracy_ylim(task, top_pad=0.08))

    # =========================================================================
    # Panel 3R: Paired Scatter (EEGNet vs CBraMod at each step)
    # =========================================================================
    # Use final progressive step for the paired comparison
    eegnet_sources = [s for s in data_sources if s.model_type == 'eegnet']
    cbramod_sources = [s for s in data_sources if s.model_type == 'cbramod']

    eegnet_baseline = next((s for s in eegnet_sources if not s.is_current_run), None)
    eegnet_final = next((s for s in reversed(eegnet_sources) if s.is_current_run), None)
    cbramod_baseline = next((s for s in cbramod_sources if not s.is_current_run), None)
    cbramod_final = next((s for s in reversed(cbramod_sources) if s.is_current_run), None)

    all_accs = []
    has_any_pair = False
    hist_markers = ['s', 'D', '^', 'v']

    # Use EEGNet (x) vs CBraMod (y) at baseline and final step
    if len(models) == 2:
        # Plot baseline pair (historical style — faded squares)
        if cbramod_baseline and eegnet_baseline:
            ee_base = {r.subject_id: r.test_acc_majority for r in eegnet_baseline.results}
            cb_base = {r.subject_id: r.test_acc_majority for r in cbramod_baseline.results}
            common = sorted(set(ee_base.keys()) & set(cb_base.keys()))
            if common:
                ex = [ee_base[s] for s in common]
                cy = [cb_base[s] for s in common]
                all_accs.extend(ex + cy)
                ax_scatter.scatter(ex, cy, s=80, alpha=0.4, c='#E94F37',
                                   label='Baseline', edgecolors='gray',
                                   linewidths=0.5, marker='s')
                has_any_pair = True

        # Plot final step pair (current style — solid circles)
        scatter_points = []  # for force-directed labels
        scatter_labels = []
        if cbramod_final and eegnet_final:
            ee_final = {r.subject_id: r.test_acc_majority for r in eegnet_final.results}
            cb_final = {r.subject_id: r.test_acc_majority for r in cbramod_final.results}
            common = sorted(set(ee_final.keys()) & set(cb_final.keys()))
            if common:
                ex = [ee_final[s] for s in common]
                cy = [cb_final[s] for s in common]
                all_accs.extend(ex + cy)
                final_step = eegnet_final.label.split()[-1]
                ax_scatter.scatter(ex, cy, s=100, alpha=0.9, c='#E94F37',
                                   label=final_step, edgecolors='black', linewidths=1)
                scatter_points = list(zip(ex, cy))
                scatter_labels = list(common)
                has_any_pair = True

        if has_any_pair and all_accs:
            lims = [min(all_accs) - 0.05, max(all_accs) + 0.05]
            ax_scatter.plot(lims, lims, 'k--', alpha=0.5, label='Equal')
            ax_scatter.set_xlim(lims)
            ax_scatter.set_ylim(lims)
            ax_scatter.set_xlabel('EEGNet Accuracy')
            ax_scatter.set_ylabel('CBraMod Accuracy')
            ax_scatter.legend(loc='lower right', fontsize=7)

            # Force-directed label placement
            if scatter_points:
                from src.visualization.plots import force_directed_label_layout
                pts_arr = np.array(scatter_points)
                fig.canvas.draw()
                label_pos = force_directed_label_layout(pts_arr, ax_scatter)
                for k, subj in enumerate(scatter_labels):
                    lx, ly = label_pos[k]
                    ox, oy = pts_arr[k]
                    ax_scatter.plot([ox, lx], [oy, ly], color='gray',
                                   linewidth=0.5, alpha=0.6, zorder=4)
                    ax_scatter.text(lx, ly, subj, fontsize=7, ha='center', va='center',
                                   bbox=dict(facecolor='white', alpha=0.7,
                                             edgecolor='none', pad=1),
                                   zorder=7)
        else:
            ax_scatter.text(0.5, 0.5, 'No common subjects\nfor paired comparison',
                            ha='center', va='center', transform=ax_scatter.transAxes)
    elif len(models) == 1:
        # Single model: show delta scatter (final vs baseline)
        model = models[0]
        baseline_src = next((s for s in data_sources if s.model_type == model and not s.is_current_run), None)
        final_src = next((s for s in reversed(data_sources) if s.model_type == model and s.is_current_run), None)

        if baseline_src and final_src:
            base_by_subj = {r.subject_id: r.test_acc_majority for r in baseline_src.results}
            final_by_subj = {r.subject_id: r.test_acc_majority for r in final_src.results}
            common = sorted(set(base_by_subj.keys()) & set(final_by_subj.keys()))

            if common:
                bx = [base_by_subj[s] for s in common]
                fy = [final_by_subj[s] for s in common]
                all_accs = bx + fy
                ax_scatter.scatter(bx, fy, s=100, alpha=0.9,
                                   c=colors[model], edgecolors='black', linewidths=1)
                lims = [min(all_accs) - 0.05, max(all_accs) + 0.05]
                ax_scatter.plot(lims, lims, 'k--', alpha=0.5, label='Equal')
                ax_scatter.set_xlim(lims)
                ax_scatter.set_ylim(lims)
                ax_scatter.set_xlabel(f'{model.upper()} Baseline Accuracy')
                ax_scatter.set_ylabel(f'{model.upper()} {final_src.label.split()[-1]} Accuracy')
                ax_scatter.legend(loc='lower right', fontsize=7)

                # Force-directed label placement
                from src.visualization.plots import force_directed_label_layout
                pts_arr = np.array(list(zip(bx, fy)))
                fig.canvas.draw()
                label_pos = force_directed_label_layout(pts_arr, ax_scatter)
                for k, subj in enumerate(common):
                    lx, ly = label_pos[k]
                    ox, oy = pts_arr[k]
                    ax_scatter.plot([ox, lx], [oy, ly], color='gray',
                                   linewidth=0.5, alpha=0.6, zorder=4)
                    ax_scatter.text(lx, ly, subj, fontsize=7, ha='center', va='center',
                                   bbox=dict(facecolor='white', alpha=0.7,
                                             edgecolor='none', pad=1),
                                   zorder=7)
            else:
                ax_scatter.text(0.5, 0.5, 'Insufficient data',
                                ha='center', va='center', transform=ax_scatter.transAxes)
        else:
            ax_scatter.text(0.5, 0.5, 'Insufficient data',
                            ha='center', va='center', transform=ax_scatter.transAxes)
    else:
        ax_scatter.text(0.5, 0.5, 'Insufficient data\nfor paired comparison',
                        ha='center', va='center', transform=ax_scatter.transAxes)

    ax_scatter.set_title('Paired Comparison (Baseline vs Final Step)')

    # ====== Save ======
    paradigm_label = 'Motor Imagery' if paradigm == 'imagery' else 'Motor Execution'
    # Compute actual session range from subjects_with_sessions
    all_sess_nums = sorted(set(s for ss in subjects_with_sessions.values() for s in ss))
    if all_sess_nums:
        sess_range = f"sessions {min(all_sess_nums)}-{max(all_sess_nums)}"
    else:
        sess_range = "extra sessions"
    fig.suptitle(f'Extra Online Sessions — {paradigm_label}, {task.title()}\n'
                 f'({n_subjects} subjects with {sess_range})',
                 fontsize=14, fontweight='bold', y=0.99)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Extra sessions plot saved: {output_path}")
