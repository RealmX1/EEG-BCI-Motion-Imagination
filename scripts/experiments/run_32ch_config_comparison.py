#!/usr/bin/env python
"""
32-channel 6-configuration comparison experiment.

Runs cross-subject comparison for each of the 6 channel configurations
to determine the optimal 32-channel subset.

Configurations:
- motor_cortex: Motor cortex focused (hand-picked, superset of 8ch)
- commercial: Standard 32ch cap 10-20 layout (hand-picked)
- fdr: Fisher Discriminant Ratio (data-driven)
- csp: Common Spatial Patterns (data-driven)
- attention: EEGNet spatial_conv + CBraMod gradient (data-driven)
- band_power: Mu/Beta band power ANOVA (data-driven)

Usage:
    uv run python scripts/experiments/run_32ch_config_comparison.py
    uv run python scripts/experiments/run_32ch_config_comparison.py --configs motor_cortex commercial
    uv run python scripts/experiments/run_32ch_config_comparison.py --task ternary
    uv run python scripts/experiments/run_32ch_config_comparison.py --dry-run
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ALL_CONFIGS = ['motor_cortex', 'commercial', 'fdr', 'csp', 'attention', 'band_power']
HAND_PICKED_CONFIGS = ['motor_cortex', 'commercial']
DATA_DRIVEN_CONFIGS = ['fdr', 'csp', 'attention', 'band_power']


def run_command(cmd: list, dry_run: bool = False) -> bool:
    """Run a command and return success status."""
    cmd_str = ' '.join(cmd)
    print(f"\n{'='*70}")
    print(f"  CMD: {cmd_str}")
    print(f"{'='*70}\n")

    if dry_run:
        print("  [DRY RUN] Skipping execution")
        return True

    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start

    if elapsed >= 3600:
        time_str = f"{elapsed/3600:.1f}h"
    elif elapsed >= 60:
        time_str = f"{elapsed/60:.1f}m"
    else:
        time_str = f"{elapsed:.1f}s"

    if result.returncode != 0:
        print(f"\n  [FAILED] Return code: {result.returncode} (took {time_str})")
        return False

    print(f"\n  [OK] Completed in {time_str}")
    return True


def check_data_driven_configs(configs: list, n_channels: int = 32) -> list:
    """Check which data-driven configs are available in channel_selections.json."""
    json_path = Path(f'results/{n_channels}_channel/channel_selections.json')
    # Hand-picked configs based on channel count
    if n_channels == 32:
        available = set(HAND_PICKED_CONFIGS)
    elif n_channels == 61:
        available = {'standard_1010'}
    else:
        available = set()

    if json_path.exists():
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for name, config in data.get('configs', {}).items():
                if 'indices' in config:
                    available.add(name)
        except (json.JSONDecodeError, OSError):
            pass

    missing = [c for c in configs if c not in available]
    if missing:
        print(f"\n  WARNING: Configs not available for {n_channels}ch: {missing}")
        print(f"  Run: uv run python scripts/analysis/compute_channel_selections.py --n-channels {n_channels}")
        print(f"  Skipping missing configs.\n")

    return [c for c in configs if c in available]


def main():
    parser = argparse.ArgumentParser(
        description='Run 32-channel 6-configuration comparison experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run all 6 configs
  uv run python scripts/experiments/run_32ch_config_comparison.py

  # Only hand-picked configs
  uv run python scripts/experiments/run_32ch_config_comparison.py --configs motor_cortex commercial

  # Ternary task
  uv run python scripts/experiments/run_32ch_config_comparison.py --task ternary

  # Dry run (show commands only)
  uv run python scripts/experiments/run_32ch_config_comparison.py --dry-run
'''
    )
    parser.add_argument(
        '--channels', type=int, default=32,
        help='Number of EEG channels (default: 32)',
    )
    parser.add_argument(
        '--configs', nargs='+', default=None,
        help=f'Channel configurations to compare (default: all 6 for 32ch, data-driven only for other counts)',
    )
    parser.add_argument(
        '--task', type=str, default='binary',
        choices=['binary', 'ternary'],
        help='Classification task (default: binary)',
    )
    parser.add_argument(
        '--paradigm', type=str, default='imagery',
        choices=['imagery', 'movement'],
        help='Experiment paradigm (default: imagery)',
    )
    parser.add_argument(
        '--models', nargs='+', default=['eegnet', 'cbramod'],
        choices=['eegnet', 'cbramod'],
        help='Models to train per config (default: both)',
    )
    parser.add_argument(
        '--scheduler', type=str, default='cosine_annealing_warmup_decay',
        help='LR scheduler for training (default: cosine_annealing_warmup_decay)',
    )
    parser.add_argument(
        '--classifier-type', type=str, default=None,
        choices=['two_layer', 'three_layer', 'one_layer', 'attention_pool'],
        help='Override CBraMod classifier head type',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print commands without executing',
    )
    parser.add_argument(
        '--no-wandb', action='store_true',
        help='Disable WandB logging',
    )
    args = parser.parse_args()

    # Default configs based on channel count
    if args.configs is None:
        if args.channels == 32:
            args.configs = ALL_CONFIGS
        elif args.channels == 61:
            args.configs = ['standard_1010']
        else:
            # For other channel counts, only data-driven configs are valid
            args.configs = DATA_DRIVEN_CONFIGS

    # Check data-driven config availability
    configs = check_data_driven_configs(args.configs, args.channels)
    if not configs:
        print("No configurations available. Exiting.")
        sys.exit(1)

    total_start = time.time()
    total_steps = len(configs)
    step = 0
    failures = []
    successes = []

    print(f"\n{'#'*70}")
    print(f"  32-Channel Configuration Comparison")
    print(f"  Paradigm: {args.paradigm} | Task: {args.task}")
    print(f"  Configs: {configs}")
    print(f"  Models: {args.models}")
    print(f"  Total steps: {total_steps}")
    print(f"{'#'*70}")

    for config_name in configs:
        step += 1
        config_type = "hand-picked" if config_name in HAND_PICKED_CONFIGS else "data-driven"
        print(f"\n{'*'*70}")
        print(f"  [{step}/{total_steps}] Config: {config_name} ({config_type})")
        print(f"{'*'*70}")

        cmd = [
            sys.executable,
            'scripts/experiments/run_cross_subject_comparison.py',
            '--task', args.task,
            '--paradigm', args.paradigm,
            '--channels', str(args.channels),
            '--channel-config', config_name,
            '--cache-only',
            '--scheduler', args.scheduler,
            '--models', *args.models,
        ]

        if args.classifier_type:
            cmd.extend(['--classifier-type', args.classifier_type])

        if args.no_wandb:
            cmd.append('--no-wandb')

        success = run_command(cmd, dry_run=args.dry_run)
        if success:
            successes.append(config_name)
        else:
            failures.append(config_name)

    # Summary
    total_elapsed = time.time() - total_start
    if total_elapsed >= 3600:
        total_str = f"{total_elapsed/3600:.1f}h"
    elif total_elapsed >= 60:
        total_str = f"{total_elapsed/60:.1f}m"
    else:
        total_str = f"{total_elapsed:.1f}s"

    print(f"\n{'#'*70}")
    print(f"  32-CHANNEL CONFIG COMPARISON COMPLETE")
    print(f"  Total time: {total_str}")
    print(f"  Configs: {len(successes)}/{total_steps} succeeded")
    if failures:
        print(f"  Failures:")
        for f in failures:
            print(f"    - {f}")
    print(f"{'#'*70}")

    # Print ranking summary if not dry run
    if not args.dry_run and successes:
        print(f"\n{'='*70}")
        print(f"  RANKING SUMMARY")
        print(f"{'='*70}")
        results_dir = f'results/{args.channels}_channel'
        print(f"  Results base: {results_dir}/")
        for cfg in successes:
            print(f"    - {results_dir}/{cfg}/")
        print(f"  Completed configs: {successes}")
        if failures:
            print(f"  Failed configs: {failures}")
        print(f"{'='*70}")

        # Generate combined config comparison plot
        plot_cmd = [
            sys.executable,
            'scripts/analysis/plot_config_comparison.py',
            '--channels', str(args.channels),
            '--task', args.task,
            '--paradigm', args.paradigm,
        ]
        print(f"\n  Generating combined comparison plot ...")
        run_command(plot_cmd, dry_run=False)

    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
