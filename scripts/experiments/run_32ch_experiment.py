#!/usr/bin/env python
"""
Reduced-channel experiment automation.

Runs cross-subject and transfer learning comparisons for specified tasks
using --channels N --channel-config <name>. Supports any channel count
(8, 32, etc.) with data-driven or hand-picked channel configurations.

Use run_32ch_config_comparison.py first to determine the best channel config,
then use this script to run the full experiment pipeline with that config.

Usage:
    # 32ch FDR full pipeline (cross-subject + transfer, binary + ternary)
    uv run python scripts/experiments/run_32ch_experiment.py --channel-config fdr

    # CBraMod only, ternary cross-subject + binary/ternary transfer
    uv run python scripts/experiments/run_32ch_experiment.py --channel-config fdr --models cbramod --tasks ternary binary

    # 8ch FDR experiment
    uv run python scripts/experiments/run_32ch_experiment.py --channels 8 --channel-config fdr

    # Include within-subject step
    uv run python scripts/experiments/run_32ch_experiment.py --channel-config fdr --steps within cross transfer

    # Dry run
    uv run python scripts/experiments/run_32ch_experiment.py --channel-config fdr --dry-run
"""

import argparse
import subprocess
import sys
import time


TASKS = ['binary', 'ternary']
ALL_STEPS = ['cross', 'transfer']  # Default: skip within-subject


def get_step_scripts(channels: int, channel_config: str, steps: list):
    """Build script list for the requested experiment steps."""
    ch_args = ['--channels', str(channels), '--channel-config', channel_config]
    scripts = []

    if 'within' in steps:
        scripts.append({
            'name': 'Within-Subject Comparison',
            'script': 'scripts/experiments/run_within_subject_comparison.py',
            'args': [
                '--cache-only',
                '--scheduler', 'cosine_annealing_warmup_decay',
                *ch_args,
            ],
        })

    if 'cross' in steps:
        scripts.append({
            'name': 'Cross-Subject Comparison',
            'script': 'scripts/experiments/run_cross_subject_comparison.py',
            'args': [
                '--cache-only',
                '--scheduler', 'cosine_annealing_warmup_decay',
                *ch_args,
            ],
        })

    if 'transfer' in steps:
        scripts.append({
            'name': 'Transfer Learning Comparison',
            'script': 'scripts/experiments/run_transfer_comparison.py',
            'args': [
                '--cache-only',
                '--freeze-strategy', 'none',
                *ch_args,
            ],
        })

    return scripts


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


def main():
    parser = argparse.ArgumentParser(
        description='Run reduced-channel experiment (cross-subject + transfer)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # 32ch FDR, CBraMod only (as used in 32ch Step 3)
  uv run python scripts/experiments/run_32ch_experiment.py --channel-config fdr --models cbramod

  # 8ch FDR experiment
  uv run python scripts/experiments/run_32ch_experiment.py --channels 8 --channel-config fdr

  # Full pipeline including within-subject
  uv run python scripts/experiments/run_32ch_experiment.py --channel-config fdr --steps within cross transfer

  # Only binary task
  uv run python scripts/experiments/run_32ch_experiment.py --channel-config fdr --tasks binary

  # Dry run
  uv run python scripts/experiments/run_32ch_experiment.py --channel-config fdr --dry-run
'''
    )
    parser.add_argument(
        '--channels', type=int, default=32,
        help='Number of EEG channels (default: 32)',
    )
    parser.add_argument(
        '--channel-config', type=str, default='motor_cortex',
        help='Channel configuration name (default: motor_cortex). '
             'Data-driven: fdr, csp, attention, band_power. '
             'Hand-picked (32ch only): motor_cortex, commercial.',
    )
    parser.add_argument(
        '--models', nargs='+', default=['eegnet', 'cbramod'],
        choices=['eegnet', 'cbramod'],
        help='Models to train (default: both)',
    )
    parser.add_argument(
        '--tasks', nargs='+', default=TASKS,
        choices=TASKS,
        help=f'Tasks to run (default: {TASKS})',
    )
    parser.add_argument(
        '--steps', nargs='+', default=ALL_STEPS,
        choices=['within', 'cross', 'transfer'],
        help=f'Experiment steps to run (default: {ALL_STEPS})',
    )
    parser.add_argument(
        '--paradigm', type=str, default='imagery',
        choices=['imagery', 'movement'],
        help='Experiment paradigm (default: imagery)',
    )
    parser.add_argument(
        '--classifier-type', type=str, default=None,
        choices=['two_layer', 'three_layer', 'one_layer', 'attention_pool'],
        help='Override CBraMod classifier head type (default: use model config)',
    )
    parser.add_argument(
        '--no-wandb', action='store_true',
        help='Disable WandB logging',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print commands without executing',
    )
    args = parser.parse_args()

    scripts = get_step_scripts(args.channels, args.channel_config, args.steps)
    total_start = time.time()
    total_steps = len(args.tasks) * len(scripts)
    step = 0
    failures = []

    print(f"\n{'#'*70}")
    print(f"  {args.channels}-Channel Experiment")
    print(f"  Channel Config: {args.channel_config}")
    print(f"  Models: {args.models}")
    print(f"  Paradigm: {args.paradigm}")
    print(f"  Tasks: {args.tasks}")
    print(f"  Steps: {args.steps}")
    print(f"  Total steps: {total_steps}")
    print(f"{'#'*70}")

    for task in args.tasks:
        print(f"\n{'*'*70}")
        print(f"  TASK: {task}")
        print(f"{'*'*70}")

        for script_info in scripts:
            step += 1
            print(f"\n  [{step}/{total_steps}] {script_info['name']} ({task})")

            cmd = [
                sys.executable,
                script_info['script'],
                '--task', task,
                '--paradigm', args.paradigm,
                '--models', *args.models,
                *script_info['args'],
            ]

            # Pass --classifier-type to within/cross scripts (not transfer)
            if args.classifier_type and 'transfer' not in script_info['name'].lower():
                cmd.extend(['--classifier-type', args.classifier_type])

            if args.no_wandb:
                cmd.append('--no-wandb')

            success = run_command(cmd, dry_run=args.dry_run)
            if not success:
                failures.append(f"{script_info['name']} ({task})")

    # Summary
    total_elapsed = time.time() - total_start
    if total_elapsed >= 3600:
        total_str = f"{total_elapsed/3600:.1f}h"
    elif total_elapsed >= 60:
        total_str = f"{total_elapsed/60:.1f}m"
    else:
        total_str = f"{total_elapsed:.1f}s"

    print(f"\n{'#'*70}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"  {args.channels}ch | Config: {args.channel_config} | Models: {args.models}")
    print(f"  Total time: {total_str}")
    print(f"  Steps: {total_steps - len(failures)}/{total_steps} succeeded")
    if failures:
        print(f"  Failures:")
        for f in failures:
            print(f"    - {f}")
    print(f"{'#'*70}")

    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
