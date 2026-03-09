#!/usr/bin/env python
"""
Generate 4-channel negative control from the 5-config complement.

Randomly selects 4 channels from the complement of the union of the 5 good
32-channel configs (commercial, fdr, csp, attention, band_power). These are
channels NOT selected by any data-driven or standard method — a negative
control for validating that channel selection is meaningful.

Prerequisites:
    uv run python scripts/analysis/generate_5config_complement.py

Usage:
    uv run python scripts/analysis/generate_4ch_negative_control.py
    uv run python scripts/analysis/generate_4ch_negative_control.py --seed 42
    uv run python scripts/analysis/generate_4ch_negative_control.py --dry-run
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

COMPLEMENT_JSON = PROJECT_ROOT / 'results' / '5config_complement.json'
SELECTIONS_4CH_JSON = PROJECT_ROOT / 'results' / '4_channel' / 'channel_selections.json'
N_SELECT = 4


def main():
    parser = argparse.ArgumentParser(
        description='Generate 4ch negative control from 5-config complement'
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--dry-run', action='store_true', help='Print without saving')
    args = parser.parse_args()

    # Load complement
    if not COMPLEMENT_JSON.exists():
        print(f"ERROR: {COMPLEMENT_JSON} not found.")
        print("Run: uv run python scripts/analysis/generate_5config_complement.py")
        sys.exit(1)

    with open(COMPLEMENT_JSON, 'r', encoding='utf-8') as f:
        complement_data = json.load(f)

    complement = complement_data['complement_indices']
    print(f"Complement: {len(complement)} channels")
    print(f"  {complement}")

    if len(complement) < N_SELECT:
        print(f"ERROR: complement has only {len(complement)} channels, need {N_SELECT}")
        sys.exit(1)

    # Randomly select 4
    rng = random.Random(args.seed)
    selected = sorted(rng.sample(complement, N_SELECT))
    print(f"\nSelected {N_SELECT} channels (seed={args.seed}):")
    print(f"  Indices: {selected}")
    print(f"  Labels:  {[BIOSEMI_128_LABELS[i] for i in selected]}")

    # Cross-check: no overlap with existing 4ch config
    if SELECTIONS_4CH_JSON.exists():
        with open(SELECTIONS_4CH_JSON, 'r', encoding='utf-8') as f:
            data_4ch = json.load(f)
        for name, cfg in data_4ch.get('configs', {}).items():
            overlap = set(selected) & set(cfg['indices'])
            print(f"\n  Overlap with {name}: {len(overlap)} channels")
            if overlap:
                print(f"    {sorted(overlap)}")

    if args.dry_run:
        print("\n[Dry run] No changes saved.")
        return

    # Save to 4ch channel_selections.json
    if not SELECTIONS_4CH_JSON.exists():
        print(f"ERROR: {SELECTIONS_4CH_JSON} not found.")
        sys.exit(1)

    with open(SELECTIONS_4CH_JSON, 'r', encoding='utf-8') as f:
        data_4ch = json.load(f)

    data_4ch['configs']['negative_control'] = {
        'indices': selected,
        'description': (
            f'Negative control — {N_SELECT} random channels from the complement '
            f'of all 5 tested 32ch configs (commercial, fdr, csp, attention, '
            f'band_power) union (seed={args.seed})'
        ),
        'seed': args.seed,
        'biosemi_labels': [BIOSEMI_128_LABELS[i] for i in selected],
        'created_at': datetime.now().isoformat(),
    }

    with open(SELECTIONS_4CH_JSON, 'w', encoding='utf-8') as f:
        json.dump(data_4ch, f, indent=2, ensure_ascii=False)

    print(f"\nSaved 'negative_control' to {SELECTIONS_4CH_JSON}")


if __name__ == '__main__':
    main()
