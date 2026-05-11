import json, io, random
from datetime import datetime
import sys
sys.path.insert(0, ".")
from src.preprocessing.channel_selection import BIOSEMI_128_LABELS

with io.open("results/64_channel/channel_selections.json", encoding="utf-8") as f:
    data = json.load(f)
methods_for_complement = ["fdr", "attention", "csp", "band_power"]
union = set()
for m in methods_for_complement:
    if m not in data["configs"]:
        print(f"FATAL: 64ch missing method {m}; aborting P2c"); sys.exit(2)
    union.update(data["configs"][m]["indices"])
all_chans = set(range(128))
complement = sorted(all_chans - union)
print(f"Union of 4 methods @64ch: {len(union)} channels")
print(f"Pure complement: {len(complement)} channels = {complement}")

SEED, N_TARGET = 42, 64
rng = random.Random(SEED)
if len(complement) >= N_TARGET:
    selected = sorted(rng.sample(complement, N_TARGET))
    breakdown = f"{N_TARGET} random (seed={SEED}) from {len(complement)}-channel pure complement"
else:
    pad_pool = sorted(union)
    n_pad = N_TARGET - len(complement)
    pad = rng.sample(pad_pool, n_pad)
    selected = sorted(set(complement) | set(pad))
    while len(selected) < N_TARGET:
        c = rng.choice(pad_pool)
        if c not in selected: selected = sorted(set(selected) | {c})
    breakdown = (f"{len(complement)} pure-complement + {n_pad} seed={SEED} random pad "
                 f"from method-union (complement<{N_TARGET})")
print(f"\n64ch neg_ctrl: {len(selected)} indices = {selected}")
print(f"  Breakdown: {breakdown}")

data["configs"]["negative_control"] = {
    "indices": selected,
    "description": (f"Negative control - {breakdown}; methods used for complement: "
                    f"{methods_for_complement}"),
    "seed": SEED,
    "biosemi_labels": [BIOSEMI_128_LABELS[i] for i in selected],
    "created_at": datetime.now().isoformat(),
}
with io.open("results/64_channel/channel_selections.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print(f"\nWROTE: results/64_channel/channel_selections.json")
print(f"Final 64ch configs: {sorted(data['configs'].keys())}")
