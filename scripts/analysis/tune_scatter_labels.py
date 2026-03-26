#!/usr/bin/env python
"""
Interactive tuning tool for force-directed label placement on paired comparison scatter plots.

Provides sliders for all force weights plus preset data scenarios. When the window
is closed, the final parameter values are printed in a copy-paste-ready format.

Usage:
    uv run python scripts/analysis/tune_scatter_labels.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.constants import MODEL_COLORS
from src.visualization.plots import force_directed_label_layout  # noqa: F401

# ---------------------------------------------------------------------------
# Interactive backend (TkAgg preferred, fallback to default)
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use('TkAgg')
except Exception:
    pass

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider


# ============================================================================
# Data presets
# ============================================================================

def _load_json(filename: str) -> dict:
    """Load a result JSON, return empty dict if missing."""
    path = PROJECT_ROOT / 'results' / filename
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[warn] {path} not found")
        return {}


# ---------------------------------------------------------------------------
# Real-data presets
# ---------------------------------------------------------------------------

def _load_within_subject_binary() -> List[Tuple[float, float, str]]:
    """Within-subject CBraMod binary (+ simulated EEGNet, seed 42)."""
    data = _load_json('20260323_2237_comparison_cache_imagery_binary.json')
    cbramod = data.get('results', {}).get('cbramod', {})
    if not cbramod:
        return _preset_clustered_high()
    rng = np.random.RandomState(42)
    out = []
    for sid in sorted(cbramod.keys()):
        cb_acc = cbramod[sid]['test_acc']
        ee_acc = float(np.clip(cb_acc + rng.uniform(-0.15, 0.05), 0.45, 1.0))
        out.append((ee_acc, cb_acc, sid))
    return out


def _load_extra_sessions_baseline() -> List[Tuple[float, float, str]]:
    """Extra sessions: EEGNet baseline (x) vs CBraMod baseline (y)."""
    data = _load_json(
        '20260325_0514_extra_sessions_cache_fixed_combined_imagery_binary.json')
    results = data.get('results', {})
    ee_data = results.get('eegnet', {})
    cb_data = results.get('cbramod', {})
    common = sorted(set(ee_data.keys()) & set(cb_data.keys()))
    out = []
    for sid in common:
        ee_acc = ee_data[sid].get('baseline', {}).get('test_acc_majority', 0)
        cb_acc = cb_data[sid].get('baseline', {}).get('test_acc_majority', 0)
        if ee_acc > 0 and cb_acc > 0:
            out.append((ee_acc, cb_acc, sid))
    return out or _preset_clustered_high()


def _load_extra_sessions_final() -> List[Tuple[float, float, str]]:
    """Extra sessions: EEGNet +Sess05 (x) vs CBraMod +Sess05 (y)."""
    data = _load_json(
        '20260325_0514_extra_sessions_cache_fixed_combined_imagery_binary.json')
    results = data.get('results', {})
    ee_data = results.get('eegnet', {})
    cb_data = results.get('cbramod', {})
    common = sorted(set(ee_data.keys()) & set(cb_data.keys()))
    # Find the last session step
    out = []
    for sid in common:
        ee_steps = sorted(k for k in ee_data[sid] if k.startswith('sess'))
        cb_steps = sorted(k for k in cb_data[sid] if k.startswith('sess'))
        if not ee_steps or not cb_steps:
            continue
        ee_acc = ee_data[sid][ee_steps[-1]].get('test_acc_majority', 0)
        cb_acc = cb_data[sid][cb_steps[-1]].get('test_acc_majority', 0)
        if ee_acc > 0 and cb_acc > 0:
            out.append((ee_acc, cb_acc, sid))
    return out or _preset_clustered_high()


def _load_extra_sessions_eegnet_progression() -> List[Tuple[float, float, str]]:
    """Extra sessions: EEGNet baseline (x) vs EEGNet final (y) — single model."""
    data = _load_json(
        '20260325_0514_extra_sessions_cache_fixed_combined_imagery_binary.json')
    ee_data = data.get('results', {}).get('eegnet', {})
    out = []
    for sid in sorted(ee_data.keys()):
        base_acc = ee_data[sid].get('baseline', {}).get('test_acc_majority', 0)
        steps = sorted(k for k in ee_data[sid] if k.startswith('sess'))
        if not steps or base_acc <= 0:
            continue
        final_acc = ee_data[sid][steps[-1]].get('test_acc_majority', 0)
        if final_acc > 0:
            out.append((base_acc, final_acc, sid))
    return out or _preset_diagonal_heavy()


def _load_extra_sessions_cbramod_progression() -> List[Tuple[float, float, str]]:
    """Extra sessions: CBraMod baseline (x) vs CBraMod final (y) — single model."""
    data = _load_json(
        '20260325_0514_extra_sessions_cache_fixed_combined_imagery_binary.json')
    cb_data = data.get('results', {}).get('cbramod', {})
    out = []
    for sid in sorted(cb_data.keys()):
        base_acc = cb_data[sid].get('baseline', {}).get('test_acc_majority', 0)
        steps = sorted(k for k in cb_data[sid] if k.startswith('sess'))
        if not steps or base_acc <= 0:
            continue
        final_acc = cb_data[sid][steps[-1]].get('test_acc_majority', 0)
        if final_acc > 0:
            out.append((base_acc, final_acc, sid))
    return out or _preset_diagonal_heavy()


def _load_within_subject_ternary() -> List[Tuple[float, float, str]]:
    """Within-subject CBraMod ternary (+ simulated EEGNet, seed 43)."""
    data = _load_json('20260323_2320_comparison_cache_imagery_ternary.json')
    cbramod = data.get('results', {}).get('cbramod', {})
    if not cbramod:
        return _preset_spread()
    rng = np.random.RandomState(43)
    out = []
    for sid in sorted(cbramod.keys()):
        cb_acc = cbramod[sid]['test_acc']
        ee_acc = float(np.clip(cb_acc + rng.uniform(-0.20, 0.05), 0.30, 1.0))
        out.append((ee_acc, cb_acc, sid))
    return out


# ---------------------------------------------------------------------------
# Synthetic presets
# ---------------------------------------------------------------------------

def _preset_clustered_high() -> List[Tuple[float, float, str]]:
    """Most points clustered in 0.85-0.95 range (hard overlap case)."""
    rng = np.random.RandomState(1)
    out = []
    for i in range(21):
        x = float(np.clip(rng.normal(0.90, 0.03), 0.80, 0.98))
        y = float(np.clip(rng.normal(0.91, 0.03), 0.80, 0.98))
        out.append((x, y, f'S{i+1:02d}'))
    return out


def _preset_spread() -> List[Tuple[float, float, str]]:
    """Points spread across 0.5-1.0 range (easier case)."""
    rng = np.random.RandomState(2)
    out = []
    for i in range(21):
        x = float(rng.uniform(0.50, 1.00))
        y = float(rng.uniform(0.50, 1.00))
        out.append((x, y, f'S{i+1:02d}'))
    return out


def _preset_diagonal_heavy() -> List[Tuple[float, float, str]]:
    """Points close to y=x line."""
    rng = np.random.RandomState(3)
    out = []
    for i in range(21):
        base = float(rng.uniform(0.60, 0.95))
        offset = float(rng.normal(0, 0.02))
        out.append((base, float(np.clip(base + offset, 0.50, 1.0)), f'S{i+1:02d}'))
    return out


def _preset_outlier() -> List[Tuple[float, float, str]]:
    """Most points clustered, 1-2 extreme outliers."""
    rng = np.random.RandomState(4)
    out = []
    for i in range(19):
        x = float(np.clip(rng.normal(0.88, 0.03), 0.80, 0.96))
        y = float(np.clip(rng.normal(0.90, 0.03), 0.82, 0.98))
        out.append((x, y, f'S{i+1:02d}'))
    out.append((0.55, 0.92, 'S20'))
    out.append((0.90, 0.58, 'S21'))
    return out


PRESETS: Dict[str, callable] = {
    # Real experiment data
    'Within-subj binary':         _load_within_subject_binary,
    'Within-subj ternary':        _load_within_subject_ternary,
    'ExtraSess baseline EE/CB':    _load_extra_sessions_baseline,
    'ExtraSess final EE/CB':       _load_extra_sessions_final,
    'ExtraSess EEGNet base>fin':   _load_extra_sessions_eegnet_progression,
    'ExtraSess CBraMod base>fin':  _load_extra_sessions_cbramod_progression,
    # Synthetic edge cases
    'Synth: clustered':           _preset_clustered_high,
    'Synth: spread':              _preset_spread,
    'Synth: diagonal':            _preset_diagonal_heavy,
    'Synth: outlier':             _preset_outlier,
}


# ============================================================================
# Interactive application
# ============================================================================

class ScatterLabelTuner:
    """Interactive matplotlib application for tuning force-directed label placement."""

    # Default parameter values
    DEFAULTS = {
        'w_point': 0.0005,
        'w_label': 0.0005,
        'w_diagonal': 0.0005,
        'w_spring': 50.0,
        'w_edge': 0.0005,
        'iterations': 100,
        'label_fontsize': 8,
    }

    def __init__(self):
        self.current_preset = list(PRESETS.keys())[0]
        self.params = dict(self.DEFAULTS)
        self._build_ui()
        self._update(None)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.manager.set_window_title(
            'Scatter Label Tuner - Force-Directed Placement'
        )

        # Main scatter axes (top portion)
        self.ax = self.fig.add_axes([0.08, 0.38, 0.55, 0.56])

        # Radio buttons for presets (right side — taller for 10 presets)
        ax_radio = self.fig.add_axes([0.66, 0.38, 0.32, 0.56])
        ax_radio.set_title('Data Preset', fontsize=9, fontweight='bold')
        labels = list(PRESETS.keys())
        self.radio = RadioButtons(ax_radio, labels, active=0)
        for lbl in self.radio.labels:
            lbl.set_fontsize(8)
        self.radio.on_clicked(self._on_preset_change)

        # Sliders (bottom panel)
        slider_specs = [
            ('w_point',       'Point repulsion',    0.0, 0.005, self.DEFAULTS['w_point']),
            ('w_label',       'Label repulsion',    0.0, 0.005, self.DEFAULTS['w_label']),
            ('w_diagonal',    'Diagonal repulsion', 0.0, 0.005, self.DEFAULTS['w_diagonal']),
            ('w_spring',      'Spring to origin',   0.0, 50.0,  self.DEFAULTS['w_spring']),
            ('w_edge',        'Edge repulsion',     0.0, 0.005, self.DEFAULTS['w_edge']),
            ('iterations',    'Iterations',         10,  500, self.DEFAULTS['iterations']),
            ('label_fontsize','Font size',          5,   12,  self.DEFAULTS['label_fontsize']),
        ]

        self.sliders: Dict[str, Slider] = {}
        n_sliders = len(slider_specs)
        slider_left = 0.12
        slider_width = 0.50
        slider_height = 0.025
        slider_bottom_start = 0.26
        slider_gap = 0.035

        for i, (key, label, vmin, vmax, vinit) in enumerate(slider_specs):
            y = slider_bottom_start - i * slider_gap
            ax_slider = self.fig.add_axes([slider_left, y, slider_width, slider_height])
            valfmt = '%d' if key in ('iterations', 'label_fontsize') else ('%.1f' if key == 'w_spring' else '%.6f')
            valstep = 1 if key in ('iterations', 'label_fontsize') else None
            slider = Slider(
                ax_slider, label, vmin, vmax,
                valinit=vinit, valfmt=valfmt, valstep=valstep,
            )
            slider.on_changed(self._update)
            self.sliders[key] = slider

        # Reset button
        ax_reset = self.fig.add_axes([0.72, 0.38, 0.10, 0.05])
        self.btn_reset = Button(ax_reset, 'Reset', hovercolor='0.85')
        self.btn_reset.on_clicked(self._on_reset)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def _update(self, _val):
        """Re-run layout and redraw the scatter plot."""
        # Read current slider values
        for key, slider in self.sliders.items():
            self.params[key] = int(slider.val) if key in ('iterations', 'label_fontsize') else slider.val

        # Load data for current preset
        data = PRESETS[self.current_preset]()
        xs = np.array([d[0] for d in data])
        ys = np.array([d[1] for d in data])
        labels = [d[2] for d in data]

        ax = self.ax
        ax.clear()

        # Axis limits
        all_vals = np.concatenate([xs, ys])
        lo = max(all_vals.min() - 0.05, 0.0)
        hi = min(all_vals.max() + 0.05, 1.05)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        # Diagonal line
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.4, linewidth=1, label='Equal Performance')

        # Scatter points
        ax.scatter(
            xs, ys, s=80, alpha=0.9, zorder=5,
            c=MODEL_COLORS['cbramod'], edgecolors='black', linewidths=0.8,
        )

        # Force-directed label placement
        points = np.column_stack([xs, ys])
        label_pos = force_directed_label_layout(
            points, ax,
            w_point=self.params['w_point'],
            w_label=self.params['w_label'],
            w_diagonal=self.params['w_diagonal'],
            w_spring=self.params['w_spring'],
            w_edge=self.params['w_edge'],
            iterations=self.params['iterations'],
        )

        fontsize = self.params['label_fontsize']

        for i, lbl in enumerate(labels):
            lx, ly = label_pos[i]
            px, py = xs[i], ys[i]

            # Thin leader line from data point to label
            ax.plot(
                [px, lx], [py, ly],
                color='gray', linewidth=0.6, alpha=0.6, zorder=3,
            )

            # Label text
            ax.text(
                lx, ly, lbl,
                fontsize=fontsize, ha='center', va='center',
                zorder=6, fontweight='medium',
                bbox=dict(
                    boxstyle='round,pad=0.15',
                    facecolor='white', edgecolor='gray',
                    alpha=0.8, linewidth=0.5,
                ),
            )

        # Axis decoration
        ax.set_xlabel(f'EEGNet Accuracy', fontsize=10, color=MODEL_COLORS['eegnet'])
        ax.set_ylabel(f'CBraMod Accuracy', fontsize=10, color=MODEL_COLORS['cbramod'])
        ax.set_title(
            f'Paired Comparison — {self.current_preset}',
            fontsize=12, fontweight='bold',
        )
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.2)

        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _on_preset_change(self, label: str):
        self.current_preset = label
        self._update(None)

    def _on_reset(self, _event):
        """Reset all sliders to default values."""
        for key, default_val in self.DEFAULTS.items():
            self.sliders[key].set_val(default_val)
        # The set_val calls will trigger _update via on_changed

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self):
        """Show the interactive window; print final params on close."""
        plt.show()
        self._print_final_params()

    def _print_final_params(self):
        """Print copy-paste-ready parameter dict to stdout."""
        print()
        print('# Tuned scatter label parameters')
        print('SCATTER_LABEL_PARAMS = {')
        for key in ('w_point', 'w_label', 'w_diagonal', 'w_spring', 'w_edge',
                     'iterations', 'label_fontsize'):
            val = self.params[key]
            if key in ('iterations', 'label_fontsize'):
                print(f'    {key!r}: {int(val)},')
            else:
                print(f'    {key!r}: {val:.6f},')
        print('}')
        print()


# ============================================================================
# Entry point
# ============================================================================

def main():
    tuner = ScatterLabelTuner()
    tuner.run()


if __name__ == '__main__':
    main()
