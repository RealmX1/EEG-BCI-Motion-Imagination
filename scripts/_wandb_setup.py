"""
WandB session management for batch training.

This module provides functions to manage WandB prompts across multiple training runs,
ensuring users are only prompted once at the beginning of batch training.

Used by:
- run_within_subject_comparison.py: Prompts once before model loop
- run_single_model.py: Prompts once at batch start (when called directly)
"""

import sys
from typing import Dict, List, Optional


def should_prompt_wandb(
    wandb_enabled: bool,
    interactive: bool,
    has_training: bool,
) -> bool:
    """
    Determine if WandB interactive prompt should be shown.

    Args:
        wandb_enabled: Whether WandB logging is enabled (not --no-wandb)
        interactive: Whether interactive prompts are enabled (not --no-wandb-interactive)
        has_training: Whether there are subjects that need training (not all cached)

    Returns:
        True if prompt should be displayed
    """
    if not wandb_enabled:
        return False
    if not interactive:
        return False
    if not has_training:
        return False
    if not sys.stdin.isatty():
        return False
    return True


def prompt_batch_session(
    models: List[str],
    task: str,
    paradigm: str,
    subjects_to_train: int,
    default_name: Optional[str] = None,
) -> Optional[Dict[str, Optional[str]]]:
    """
    Collect WandB batch session metadata (name, goal, hypothesis, notes).

    This function should be called ONCE before batch training starts,
    and the returned metadata passed to each training run.

    Args:
        models: List of model types (e.g., ['eegnet', 'cbramod'])
        task: Task type ('binary', 'ternary', 'quaternary')
        paradigm: Paradigm ('imagery' or 'movement')
        subjects_to_train: Number of subjects that need training
        default_name: Optional custom default name

    Returns:
        Dictionary with collected metadata:
        - name: Run name prefix
        - goal: Experiment goal
        - hypothesis: Research hypothesis
        - notes: Additional notes
        Returns None if user cancels (Ctrl+C)
    """
    paradigm_short = "MI" if paradigm == "imagery" else "ME"
    models_str = "+".join(models)

    if default_name is None:
        default_name = f"{models_str}_{task}_{paradigm_short}"

    print("\n" + "=" * 60)
    print("  WandB Batch Session Configuration")
    print("=" * 60)
    print(f"  Models: {', '.join(m.upper() for m in models)}")
    print(f"  Task: {task} | Paradigm: {paradigm_short}")
    print(f"  Subjects to train: {subjects_to_train}")
    print("-" * 60)
    print("Press Enter to accept default/skip optional fields.\n")

    def _get_input(
        prompt: str,
        default: Optional[str] = None,
        optional: bool = False,
    ) -> Optional[str]:
        """Get input with default value support."""
        opt_marker = " (optional)" if optional else ""
        if default:
            display = f"{prompt}{opt_marker} [{default}]: "
        else:
            display = f"{prompt}{opt_marker}: "

        try:
            user_input = input(display).strip()
            if user_input:
                return user_input
            return default
        except (EOFError, KeyboardInterrupt):
            print("\n(Skipped)")
            return default

    try:
        # 1. Run name (required - has default)
        name = _get_input("Run name", default_name)

        # 2. Goal - optional
        goal = _get_input("Goal", optional=True)

        # 3. Hypothesis - optional
        hypothesis = _get_input("Hypothesis", optional=True)

        # 4. Notes - optional
        notes = _get_input("Notes", optional=True)

        print("-" * 60 + "\n")

        return {
            "name": name,
            "goal": goal,
            "hypothesis": hypothesis,
            "notes": notes,
        }
    except KeyboardInterrupt:
        print("\n(Cancelled)")
        return None
