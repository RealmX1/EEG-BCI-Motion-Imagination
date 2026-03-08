#!/usr/bin/env python
"""
Generate a comprehensive multi-config comparison plot for N-channel experiments.

Loads per-config cross-subject JSON results and produces a single plot comparing
all configurations side-by-side (mean accuracy bar chart + per-model box plots).

Auto-detection (new JSON format with channel_config in metadata):
    uv run python scripts/analysis/plot_config_comparison.py --channels 32

Manual mapping for legacy JSONs (no channel_config field):
    uv run python scripts/analysis/plot_config_comparison.py --channels 32 \\
        --config-timestamps motor_cortex:20260220_1731 commercial:20260220_1850 \\
            fdr:20260220_1949 csp:20260220_2052 \\
            attention:20260220_2159 band_power:20260220_2301

With 128ch and/or 61ch baseline reference lines:
    uv run python scripts/analysis/plot_config_comparison.py --channels 32 \\
        --baseline-eegnet 0.8988 --baseline-cbramod 0.9027

    # Auto-discover 61ch results from results/61_channel/:
    uv run python scripts/analysis/plot_config_comparison.py --channels 32 --auto-61ch-baseline

    # Or specify manually:
    uv run python scripts/analysis/plot_config_comparison.py --channels 32 \\
        --baseline-61ch-eegnet 0.88 --baseline-61ch-cbramod 0.90
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.cross_subject import generate_config_comparison_plot


ALL_CONFIGS_ORDER = ['commercial', 'fdr', 'csp', 'attention', 'band_power']


def load_json(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_results_by_channel_config(
    results_dir: Path,
    paradigm: str,
    task: str,
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Auto-detect: scan JSONs with channel_config in metadata.
    Returns {config_name: {model_type: {subject_id: acc}}}.

    Searches both:
    - New structure: results_dir/{config_name}/*.json (subdirectories per config)
    - Legacy structure: results_dir/*.json (flat, all configs in one directory)
    """
    config_results: dict[str, dict[str, dict[str, float]]] = {}
    glob_pattern = f'*_cross-subject_*_{paradigm}_{task}.json'

    # Scan subdirectories (new structure: results/{N}_channel/{config}/)
    json_files = sorted(results_dir.glob(f'*/{glob_pattern}'))
    # Also scan flat directory (legacy structure: results/{N}_channel/)
    json_files.extend(sorted(results_dir.glob(glob_pattern)))

    for json_path in json_files:
        try:
            data = load_json(json_path)
        except (json.JSONDecodeError, OSError):
            continue

        meta = data.get('metadata', {})
        channel_config = meta.get('channel_config')
        model_type = meta.get('model_type')
        if not channel_config or not model_type:
            continue

        per_subject = data.get('results', {}).get('per_subject_test_acc', {})
        if not per_subject:
            continue

        # If a config already has results from a subdirectory, don't overwrite with flat
        if channel_config in config_results and model_type in config_results[channel_config]:
            continue

        config_results.setdefault(channel_config, {})[model_type] = per_subject

    return config_results


def find_results_by_timestamps(
    results_dir: Path,
    paradigm: str,
    task: str,
    config_timestamps: dict[str, str],
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Manual mapping: load JSONs by (config_name → run_tag timestamp).
    Returns {config_name: {model_type: {subject_id: acc}}}.
    """
    config_results: dict[str, dict[str, dict[str, float]]] = {}

    for config_name, run_tag in config_timestamps.items():
        config_results[config_name] = {}
        for model_type in ['eegnet', 'cbramod']:
            pattern = f'{run_tag}_cross-subject_{model_type}_{paradigm}_{task}.json'
            candidates = list(results_dir.glob(pattern))
            if not candidates:
                print(f"  [WARN] Not found: {results_dir / pattern}")
                continue
            try:
                data = load_json(candidates[0])
                per_subject = data.get('results', {}).get('per_subject_test_acc', {})
                if per_subject:
                    config_results[config_name][model_type] = per_subject
                else:
                    print(f"  [WARN] Empty per_subject_test_acc in {candidates[0].name}")
            except (json.JSONDecodeError, OSError) as e:
                print(f"  [WARN] Failed to load {candidates[0]}: {e}")

    return config_results


def discover_61ch_baselines(
    paradigm: str,
    task: str,
    results_base: Path = PROJECT_ROOT / 'results',
) -> dict[str, float]:
    """
    Auto-discover 61ch cross-subject results from results/61_channel/.

    Scans for the most recent cross-subject JSONs with channel_config='standard_1010'
    and computes mean accuracy per model.

    Returns:
        {model_type: mean_accuracy} or empty dict if not found.
    """
    dir_61ch = results_base / '61_channel'
    if not dir_61ch.exists():
        return {}

    baselines: dict[str, float] = {}
    glob_pattern = f'*_cross-subject_*_{paradigm}_{task}.json'

    # Search both subdirectories (new structure) and flat directory (legacy)
    json_files = sorted(dir_61ch.glob(f'*/{glob_pattern}'), reverse=True)
    json_files.extend(sorted(dir_61ch.glob(glob_pattern), reverse=True))

    for json_path in json_files:
        try:
            data = load_json(json_path)
        except (json.JSONDecodeError, OSError):
            continue

        meta = data.get('metadata', {})
        model_type = meta.get('model_type')
        if not model_type or model_type in baselines:
            continue

        per_subject = data.get('results', {}).get('per_subject_test_acc', {})
        if per_subject:
            baselines[model_type] = sum(per_subject.values()) / len(per_subject)
            print(f"  61ch baseline: {model_type} = {baselines[model_type]*100:.2f}% "
                  f"(from {json_path.name})")

    return baselines


def sort_configs(config_results: dict, preferred_order: list[str]) -> dict:
    """Sort configs by preferred order, appending any extras at the end."""
    ordered = [c for c in preferred_order if c in config_results]
    extras = [c for c in config_results if c not in preferred_order]
    keys = ordered + extras
    return {k: config_results[k] for k in keys}


def main():
    parser = argparse.ArgumentParser(
        description='Generate comprehensive N-channel config comparison plot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Auto-detect (new JSONs with channel_config in metadata)
  uv run python scripts/analysis/plot_config_comparison.py --channels 32

  # With auto-discovered 61ch baseline
  uv run python scripts/analysis/plot_config_comparison.py --channels 32 --auto-61ch-baseline

  # With manual 128ch and 61ch baselines
  uv run python scripts/analysis/plot_config_comparison.py --channels 32 \\
      --baseline-eegnet 0.8988 --baseline-cbramod 0.9027 \\
      --baseline-61ch-eegnet 0.88 --baseline-61ch-cbramod 0.90

  # Legacy: specify run_tag per config
  uv run python scripts/analysis/plot_config_comparison.py --channels 32 \\
      --config-timestamps motor_cortex:20260220_1731 commercial:20260220_1850 \\
          fdr:20260220_1949 csp:20260220_2052 \\
          attention:20260220_2159 band_power:20260220_2301
''',
    )
    parser.add_argument('--channels', type=int, default=32,
                        help='Number of EEG channels (default: 32)')
    parser.add_argument('--paradigm', type=str, default='imagery',
                        choices=['imagery', 'movement'],
                        help='Experiment paradigm (default: imagery)')
    parser.add_argument('--task', type=str, default='binary',
                        choices=['binary', 'ternary', 'quaternary'],
                        help='Classification task (default: binary)')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Results directory (default: results/{channels}_channel)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output PNG path (default: auto-generated in results-dir)')
    parser.add_argument(
        '--config-timestamps', nargs='+', default=None,
        metavar='CONFIG:TIMESTAMP',
        help='Manual mapping of config names to run_tag timestamps, '
             'e.g. fdr:20260220_1949 csp:20260220_2052',
    )
    # 128ch baselines
    parser.add_argument('--baseline-eegnet', type=float, default=None,
                        help='EEGNet 128ch baseline mean accuracy (0-1)')
    parser.add_argument('--baseline-cbramod', type=float, default=None,
                        help='CBraMod 128ch baseline mean accuracy (0-1)')
    # 61ch baselines
    parser.add_argument('--auto-61ch-baseline', action='store_true',
                        help='Auto-discover 61ch baselines from results/61_channel/')
    parser.add_argument('--baseline-61ch-eegnet', type=float, default=None,
                        help='EEGNet 61ch baseline mean accuracy (0-1)')
    parser.add_argument('--baseline-61ch-cbramod', type=float, default=None,
                        help='CBraMod 61ch baseline mean accuracy (0-1)')
    args = parser.parse_args()

    # Resolve directories
    if args.results_dir is None:
        args.results_dir = f'results/{args.channels}_channel'
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"[ERROR] Results directory not found: {results_dir}")
        sys.exit(1)

    # Load data
    if args.config_timestamps:
        config_ts_map: dict[str, str] = {}
        for item in args.config_timestamps:
            parts = item.split(':', 1)
            if len(parts) != 2:
                print(f"[ERROR] Invalid --config-timestamps entry: {item!r} (expected CONFIG:TIMESTAMP)")
                sys.exit(1)
            config_ts_map[parts[0]] = parts[1]
        print(f"Loading results by explicit timestamps: {list(config_ts_map.keys())}")
        config_results = find_results_by_timestamps(results_dir, args.paradigm, args.task, config_ts_map)
    else:
        print(f"Auto-detecting results in {results_dir} ...")
        config_results = find_results_by_channel_config(results_dir, args.paradigm, args.task)

    if not config_results:
        print("[ERROR] No config results found. Use --config-timestamps to specify manually.")
        sys.exit(1)

    # Sort configs in canonical order
    config_results = sort_configs(config_results, ALL_CONFIGS_ORDER)

    # Print summary
    print(f"\nConfigs found: {list(config_results.keys())}")
    for cfg, models in config_results.items():
        for mt, subjects in models.items():
            mean_acc = sum(subjects.values()) / len(subjects) if subjects else 0
            print(f"  {cfg:15s} | {mt:7s} | n={len(subjects):2d} | mean={mean_acc*100:.2f}%")

    # 128ch baseline
    baseline_accs = {}
    if args.baseline_eegnet is not None:
        baseline_accs['eegnet'] = args.baseline_eegnet
    if args.baseline_cbramod is not None:
        baseline_accs['cbramod'] = args.baseline_cbramod

    # 61ch baseline (manual values take priority over auto-discovery)
    baseline_61ch_accs = {}
    if args.baseline_61ch_eegnet is not None:
        baseline_61ch_accs['eegnet'] = args.baseline_61ch_eegnet
    if args.baseline_61ch_cbramod is not None:
        baseline_61ch_accs['cbramod'] = args.baseline_61ch_cbramod

    if args.auto_61ch_baseline and not baseline_61ch_accs:
        print("\nAuto-discovering 61ch baselines ...")
        baseline_61ch_accs = discover_61ch_baselines(args.paradigm, args.task)
        if not baseline_61ch_accs:
            print("  No 61ch results found in results/61_channel/")

    # Output path
    if args.output is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M')
        fname = f'{ts}_{args.channels}ch_config_comparison_{args.paradigm}_{args.task}.png'
        output_path = str(results_dir / fname)
    else:
        output_path = args.output

    print(f"\nGenerating plot → {output_path}")
    generate_config_comparison_plot(
        config_results=config_results,
        output_path=output_path,
        task_type=args.task,
        paradigm=args.paradigm,
        n_channels=args.channels,
        baseline_accs=baseline_accs or None,
        baseline_61ch_accs=baseline_61ch_accs or None,
    )
    print(f"[OK] Saved: {output_path}")


if __name__ == '__main__':
    main()
