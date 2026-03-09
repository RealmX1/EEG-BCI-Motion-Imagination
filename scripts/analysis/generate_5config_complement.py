#!/usr/bin/env python
"""
Compute the complement of the union of the 5 good 32-channel configs.

The 5 configs are: commercial, fdr, csp, attention, band_power.
The complement is the set of 128 channels NOT selected by ANY of these methods.

Usage:
    uv run python scripts/analysis/generate_5config_complement.py
    uv run python scripts/analysis/generate_5config_complement.py --dry-run
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.channel_selection import (
    BIOSEMI_128_LABELS,
    CHANNEL_32_CONFIGS,
)
from src.config.constants import FULL_N_CHANNELS

SELECTIONS_32CH_JSON = PROJECT_ROOT / 'results' / '32_channel' / 'channel_selections.json'

# The 5 tested configs whose union we take the complement of
GOOD_CONFIGS = ['commercial', 'fdr', 'csp', 'attention', 'band_power']


def compute_complement() -> list[int]:
    """Compute complement of the 5 good 32ch configs' union."""
    # Load 32ch JSON for data-driven configs
    if not SELECTIONS_32CH_JSON.exists():
        print(f"ERROR: {SELECTIONS_32CH_JSON} not found.")
        print("Run: uv run python scripts/analysis/compute_channel_selections.py")
        sys.exit(1)

    with open(SELECTIONS_32CH_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    configs = data.get('configs', {})

    # Build union of all 5 configs
    union = set()
    for name in GOOD_CONFIGS:
        # Hard-coded configs
        if name in CHANNEL_32_CONFIGS and CHANNEL_32_CONFIGS[name] is not None:
            indices = CHANNEL_32_CONFIGS[name]
        elif name in configs:
            indices = configs[name]['indices']
        else:
            print(f"ERROR: config '{name}' not found")
            sys.exit(1)

        print(f"  {name:12s}: {len(indices)} channels")
        union.update(indices)

    print(f"\n  Union: {len(union)} unique channels")

    # Complement
    full_set = set(range(FULL_N_CHANNELS))
    complement = sorted(full_set - union)
    print(f"  Complement: {len(complement)} channels")

    return complement


def main():
    parser = argparse.ArgumentParser(
        description='Compute complement of 5 good 32ch configs'
    )
    parser.add_argument('--dry-run', action='store_true', help='Print without saving')
    args = parser.parse_args()

    print(f"Computing complement of {len(GOOD_CONFIGS)} configs: {GOOD_CONFIGS}\n")
    complement = compute_complement()

    print(f"\nComplement indices ({len(complement)}):")
    print(f"  {complement}")
    print(f"  Labels: {[BIOSEMI_128_LABELS[i] for i in complement]}")

    # Verify against existing 32ch negative_control
    if SELECTIONS_32CH_JSON.exists():
        with open(SELECTIONS_32CH_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
        existing_nc = data.get('configs', {}).get('negative_control')
        if existing_nc:
            existing_set = set(existing_nc['indices'])
            complement_set = set(complement)
            only_in_existing = existing_set - complement_set
            only_in_complement = complement_set - existing_set
            print(f"\n  Cross-check with existing 32ch negative_control:")
            print(f"    Existing has {len(existing_set)} channels")
            print(f"    Pure complement has {len(complement_set)} channels")
            if only_in_existing:
                print(f"    In existing but NOT in complement: {sorted(only_in_existing)}")
                print(f"      (these were the extra channels added to reach 32)")
            if only_in_complement:
                print(f"    In complement but NOT in existing: {sorted(only_in_complement)}")

    if args.dry_run:
        print("\n[Dry run] No changes saved.")
        return

    # Save complement to a standalone JSON
    output_path = PROJECT_ROOT / 'results' / '5config_complement.json'
    output = {
        'metadata': {
            'description': (
                f'Complement of the union of {len(GOOD_CONFIGS)} 32ch configs: '
                f'{", ".join(GOOD_CONFIGS)}'
            ),
            'n_channels_complement': len(complement),
            'n_channels_union': FULL_N_CHANNELS - len(complement),
            'source_configs': GOOD_CONFIGS,
            'created_at': datetime.now().isoformat(),
        },
        'complement_indices': complement,
        'complement_labels': [BIOSEMI_128_LABELS[i] for i in complement],
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
