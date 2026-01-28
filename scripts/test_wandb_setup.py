#!/usr/bin/env python
"""
Quick test script for WandB session management.

Tests the prompt logic without running actual training.

Usage:
    # Test single model scenario
    uv run python scripts/test_wandb_setup.py --mode single

    # Test full comparison scenario (default)
    uv run python scripts/test_wandb_setup.py --mode comparison

    # Test with WandB disabled
    uv run python scripts/test_wandb_setup.py --no-wandb

    # Test with interactive disabled
    uv run python scripts/test_wandb_setup.py --no-wandb-interactive
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

from _wandb_setup import should_prompt_wandb, prompt_batch_session


def test_should_prompt_wandb():
    """Test should_prompt_wandb logic."""
    print("\n" + "=" * 60)
    print("  Testing should_prompt_wandb()")
    print("=" * 60)

    test_cases = [
        # (wandb_enabled, interactive, has_training, expected_if_tty)
        (True, True, True, True),
        (False, True, True, False),
        (True, False, True, False),
        (True, True, False, False),
        (False, False, False, False),
    ]

    is_tty = sys.stdin.isatty()
    print(f"stdin.isatty() = {is_tty}\n")

    for wandb_enabled, interactive, has_training, expected_if_tty in test_cases:
        result = should_prompt_wandb(wandb_enabled, interactive, has_training)
        expected = expected_if_tty and is_tty
        status = "PASS" if result == expected else "FAIL"
        print(f"  should_prompt_wandb({wandb_enabled}, {interactive}, {has_training}) = {result} [{status}]")

    print()


def test_prompt_single_model(no_wandb: bool, no_interactive: bool):
    """Simulate single model scenario."""
    print("\n" + "=" * 60)
    print("  Simulating: run_single_model.py --model eegnet")
    print("=" * 60)

    model_type = "eegnet"
    task = "binary"
    paradigm = "imagery"
    subjects_to_train = 3  # Simulated

    print(f"\nParameters:")
    print(f"  --no-wandb: {no_wandb}")
    print(f"  --no-wandb-interactive: {no_interactive}")
    print(f"  Subjects to train: {subjects_to_train}")

    should_prompt = should_prompt_wandb(
        wandb_enabled=not no_wandb,
        interactive=not no_interactive,
        has_training=subjects_to_train > 0,
    )

    print(f"\nshould_prompt_wandb() returned: {should_prompt}")

    if should_prompt:
        print("\n>>> Would show WandB prompt:")
        metadata = prompt_batch_session(
            models=[model_type],
            task=task,
            paradigm=paradigm,
            subjects_to_train=subjects_to_train,
        )
        print(f"\nCollected metadata: {metadata}")
    else:
        print("\n>>> No prompt shown (skipped)")


def test_prompt_comparison(no_wandb: bool, no_interactive: bool):
    """Simulate full comparison scenario."""
    print("\n" + "=" * 60)
    print("  Simulating: run_full_comparison.py")
    print("=" * 60)

    models = ["eegnet", "cbramod"]
    task = "binary"
    paradigm = "imagery"
    subjects_to_train = 5  # Simulated

    print(f"\nParameters:")
    print(f"  --models: {models}")
    print(f"  --no-wandb: {no_wandb}")
    print(f"  --no-wandb-interactive: {no_interactive}")
    print(f"  Subjects needing training: {subjects_to_train}")

    should_prompt = should_prompt_wandb(
        wandb_enabled=not no_wandb,
        interactive=not no_interactive,
        has_training=subjects_to_train > 0,
    )

    print(f"\nshould_prompt_wandb() returned: {should_prompt}")

    wandb_session_metadata = None
    if should_prompt:
        print("\n>>> Would show WandB prompt (ONCE before model loop):")
        wandb_session_metadata = prompt_batch_session(
            models=models,
            task=task,
            paradigm=paradigm,
            subjects_to_train=subjects_to_train,
        )
        print(f"\nCollected metadata: {wandb_session_metadata}")
    else:
        print("\n>>> No prompt shown (skipped)")

    # Simulate model loop
    print("\n" + "-" * 60)
    print("  Simulating model loop:")
    print("-" * 60)

    for model_type in models:
        print(f"\n  [{model_type.upper()}] run_single_model() called with:")
        print(f"    wandb_interactive=False")
        print(f"    wandb_session_metadata={wandb_session_metadata}")
        print(f"    -> No additional prompt (uses pre-collected metadata)")


def main():
    parser = argparse.ArgumentParser(
        description='Test WandB session management without training'
    )
    parser.add_argument(
        '--mode', choices=['single', 'comparison', 'unit'],
        default='comparison',
        help='Test mode: single (run_single_model), comparison (run_full_comparison), unit (logic tests only)'
    )
    parser.add_argument(
        '--no-wandb', action='store_true',
        help='Simulate --no-wandb flag'
    )
    parser.add_argument(
        '--no-wandb-interactive', action='store_true',
        help='Simulate --no-wandb-interactive flag'
    )

    args = parser.parse_args()

    print("\n" + "#" * 60)
    print("  WandB Session Management Test")
    print("#" * 60)

    if args.mode == 'unit':
        test_should_prompt_wandb()
    elif args.mode == 'single':
        test_prompt_single_model(args.no_wandb, args.no_wandb_interactive)
    else:  # comparison
        test_prompt_comparison(args.no_wandb, args.no_wandb_interactive)

    print("\n" + "#" * 60)
    print("  Test Complete")
    print("#" * 60 + "\n")


if __name__ == '__main__':
    main()
