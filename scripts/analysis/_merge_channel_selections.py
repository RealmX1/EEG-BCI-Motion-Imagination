#!/usr/bin/env python
"""Merge newly computed channel-selection methods into an existing JSON.

`compute_channel_selections.py` overwrites the entire JSON with whatever
methods were requested. This helper merges a freshly produced JSON's
`configs.<method>` entries on top of an existing JSON, preserving all
prior methods.

Usage:
    python scripts/analysis/_merge_channel_selections.py \
        --new path/to/freshly_written.json \
        --existing path/to/old_backup.json \
        --output path/to/final.json
"""

import argparse
import io
import json
from datetime import datetime
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--new', required=True, help='New JSON (only the freshly computed methods)')
    p.add_argument('--existing', required=True, help='Existing JSON to preserve')
    p.add_argument('--output', required=True, help='Path to write merged JSON')
    args = p.parse_args()

    with io.open(args.new, encoding='utf-8') as f:
        new = json.load(f)
    with io.open(args.existing, encoding='utf-8') as f:
        existing = json.load(f)

    merged_configs = dict(existing.get('configs', {}))
    new_configs = new.get('configs', {})
    overlapping = sorted(set(merged_configs) & set(new_configs))
    added = sorted(set(new_configs) - set(merged_configs))
    for k, v in new_configs.items():
        merged_configs[k] = v

    merged = {
        'metadata': {
            **existing.get('metadata', {}),
            'merged_at': datetime.now().isoformat(),
            'merged_methods_added': added,
            'merged_methods_overwritten': overlapping,
            'note': existing.get('metadata', {}).get('note', '') + ' | merged via _merge_channel_selections.py',
        },
        'configs': merged_configs,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with io.open(out, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"Wrote merged JSON: {out}")
    print(f"  Configs in result: {sorted(merged_configs.keys())}")
    print(f"  Newly added:       {added}")
    print(f"  Overwritten:       {overlapping}")


if __name__ == '__main__':
    main()
