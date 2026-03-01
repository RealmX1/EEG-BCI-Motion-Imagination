#!/usr/bin/env python
"""
Generate FDR complement channel configuration for data leakage investigation.

Takes the complement of the best-performing FDR 32-channel config (indices NOT
in FDR), randomly selects 32 channels from the 96 complement channels, and
saves the result to channel_selections.json as 'fdr_complement'.

If this complement config achieves similar accuracy to FDR, it strongly suggests
data leakage. If accuracy drops significantly, channel selection is meaningful.

Usage:
    uv run python scripts/analysis/generate_fdr_complement.py
    uv run python scripts/analysis/generate_fdr_complement.py --seed 42
    uv run python scripts/analysis/generate_fdr_complement.py --dry-run
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.channel_selection import BIOSEMI_128_LABELS
from src.config.constants import FULL_N_CHANNELS

SELECTIONS_JSON = PROJECT_ROOT / 'results' / '32_channel' / 'channel_selections.json'
N_SELECT = 32


def main():
    parser = argparse.ArgumentParser(description='Generate FDR complement channel config')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--dry-run', action='store_true', help='Print result without saving')
    args = parser.parse_args()

    # Load existing channel selections
    if not SELECTIONS_JSON.exists():
        print(f"ERROR: {SELECTIONS_JSON} not found.")
        print("Run: uv run python scripts/analysis/compute_32ch_selections.py")
        sys.exit(1)

    with open(SELECTIONS_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Get FDR indices
    fdr_config = data.get('configs', {}).get('fdr')
    if fdr_config is None:
        print("ERROR: 'fdr' config not found in channel_selections.json")
        sys.exit(1)

    fdr_indices = set(fdr_config['indices'])
    print(f"FDR indices ({len(fdr_indices)} channels): {sorted(fdr_indices)}")

    # Compute complement
    full_set = set(range(FULL_N_CHANNELS))
    complement = sorted(full_set - fdr_indices)
    print(f"Complement set: {len(complement)} channels")

    # Randomly select 32 from complement
    rng = random.Random(args.seed)
    selected = sorted(rng.sample(complement, N_SELECT))
    print(f"\nSelected {N_SELECT} channels (seed={args.seed}):")
    print(f"  Indices: {selected}")
    print(f"  Labels:  {[BIOSEMI_128_LABELS[i] for i in selected]}")

    # Verify no overlap
    overlap = set(selected) & fdr_indices
    assert len(overlap) == 0, f"BUG: overlap with FDR: {overlap}"
    print(f"\nOverlap with FDR: {len(overlap)} (verified zero)")

    if args.dry_run:
        print("\n[Dry run] No changes saved.")
        return

    # Save to channel_selections.json
    data['configs']['fdr_complement'] = {
        'indices': selected,
        'description': (
            f'FDR complement — random 32 channels from the 96 NOT in FDR '
            f'(seed={args.seed}, for data leakage investigation)'
        ),
        'seed': args.seed,
        'created_at': datetime.now().isoformat(),
    }

    with open(SELECTIONS_JSON, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved 'fdr_complement' config to {SELECTIONS_JSON}")


if __name__ == '__main__':
    main()
