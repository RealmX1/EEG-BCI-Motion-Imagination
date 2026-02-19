"""
8-channel motor cortex experiment automation.

Runs within-subject, cross-subject, and transfer learning comparisons
for both binary and ternary tasks using --channels 8 --cache-only.

Channels: Cz (A1), Pz (A3), PO7 (A6), Oz (A21), PO8 (B3), C4 (B21), Fz (C23), C3 (D18)

Usage:
    uv run python scripts/experiments/run_8ch_experiment.py
    uv run python scripts/experiments/run_8ch_experiment.py --tasks binary       # Only binary
    uv run python scripts/experiments/run_8ch_experiment.py --tasks ternary      # Only ternary
    uv run python scripts/experiments/run_8ch_experiment.py --dry-run            # Show commands only
"""

import argparse
import subprocess
import sys
import time


TASKS = ['binary', 'ternary']

SCRIPTS = [
    {
        'name': 'Within-Subject Comparison',
        'script': 'scripts/experiments/run_within_subject_comparison.py',
        'args': ['--cache-only', '--scheduler', 'cosine_annealing_warmup_decay', '--channels', '8'],
    },
    {
        'name': 'Cross-Subject Comparison',
        'script': 'scripts/experiments/run_cross_subject_comparison.py',
        'args': ['--cache-only', '--scheduler', 'cosine_annealing_warmup_decay', '--channels', '8'],
    },
    {
        'name': 'Transfer Learning Comparison',
        'script': 'scripts/experiments/run_transfer_comparison.py',
        'args': ['--cache-only', '--freeze-strategy', 'none', '--channels', '8'],
    },
]


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
        description='Run full 8-channel motor cortex experiment (within-subject + cross-subject + transfer)',
    )
    parser.add_argument(
        '--tasks', nargs='+', default=TASKS,
        choices=TASKS,
        help=f'Tasks to run (default: {TASKS})',
    )
    parser.add_argument(
        '--classifier-type', type=str, default=None,
        choices=['two_layer', 'three_layer', 'one_layer', 'attention_pool'],
        help='Override CBraMod classifier head type (default: use model config)',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print commands without executing',
    )
    args = parser.parse_args()

    total_start = time.time()
    total_steps = len(args.tasks) * len(SCRIPTS)
    step = 0
    failures = []

    print(f"\n{'#'*70}")
    print(f"  8-Channel Motor Cortex Experiment")
    print(f"  Tasks: {args.tasks}")
    print(f"  Total steps: {total_steps}")
    print(f"{'#'*70}")

    for task in args.tasks:
        print(f"\n{'*'*70}")
        print(f"  TASK: {task}")
        print(f"{'*'*70}")

        for script_info in SCRIPTS:
            step += 1
            print(f"\n  [{step}/{total_steps}] {script_info['name']} ({task})")

            cmd = [
                sys.executable,
                script_info['script'],
                '--task', task,
                *script_info['args'],
            ]

            # Pass --classifier-type to within/cross scripts (not transfer, which reads from checkpoint)
            if args.classifier_type and 'transfer' not in script_info['name'].lower():
                cmd.extend(['--classifier-type', args.classifier_type])

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
