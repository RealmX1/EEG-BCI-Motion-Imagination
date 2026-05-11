# Driver: 21-cell reduced-channel cross-subject CBraMod fill.
# Steps: overwatch gate -> P1 (64ch attn/csp/bp) -> merge -> P2c (64ch neg_ctrl) -> P3 sanity -> 21 runs.
# Sentinel "ALL_21_DONE" appended to log on full success.

$ErrorActionPreference = "Continue"
$ts = Get-Date -Format "yyyyMMdd_HHmm"
$LOG = "logs/21cell_fill_${ts}.log"
"=== driver started at $(Get-Date -Format o) ===" | Out-File -Append $LOG -Encoding utf8

function LogStep($msg) {
    $line = "[$(Get-Date -Format o)] $msg"
    $line | Tee-Object -Append -FilePath $LOG
}

function RunOrAbort($desc, $cmd) {
    LogStep "STEP-START $desc :: $cmd"
    Invoke-Expression $cmd 2>&1 | Tee-Object -Append -FilePath $LOG
    if ($LASTEXITCODE -ne 0) {
        LogStep "STEP-FAIL  $desc exit=$LASTEXITCODE"
        LogStep "DRIVER_ABORTED"
        exit 1
    }
    LogStep "STEP-OK    $desc"
}

# --- Step 0: overwatch gate (blocks until GPU/CPU/NET stably idle) ---
RunOrAbort "overwatch gate" "uv run python scripts/overwatch/overwatch.py --disable-network"

# --- Step 1: P1 - compute 64ch attention/csp/band_power ---
RunOrAbort "P1 64ch attn/csp/bp" "uv run python scripts/analysis/compute_channel_selections.py --n-channels 64 --methods attention csp band_power --output results/64_channel/_new_methods.json"

# --- Step 2: merge with existing fdr ---
RunOrAbort "P1-merge preserve fdr" "uv run python scripts/analysis/_merge_channel_selections.py --new results/64_channel/_new_methods.json --existing results/64_channel/channel_selections.json --output results/64_channel/channel_selections.json"

# --- Step 3: P2c - 64ch negative_control via union-complement (with seed=42 padding fallback) ---
LogStep "STEP-START P2c 64ch negative_control"
$p2cScript = @'
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
'@
$p2cScript | Out-File -FilePath logs/_p2c_inline.py -Encoding utf8
RunOrAbort "P2c 64ch neg_ctrl" "uv run python logs/_p2c_inline.py"

# --- Step 4: P3 - sanity check all registries ---
LogStep "STEP-START P3 sanity check"
$p3Script = @'
import json, io
required = {"fdr","attention","band_power","csp","negative_control"}
fail = False
for n in [4,8,32,64]:
    p = f"results/{n}_channel/channel_selections.json"
    with io.open(p, encoding="utf-8") as f:
        d = json.load(f)
    have = set(d["configs"].keys())
    missing = required - have
    print(f"{n}ch: have={sorted(have)}  missing_required={sorted(missing) or 'none'}")
    if missing: fail = True
if fail:
    print("SANITY_FAIL"); raise SystemExit(2)
print("SANITY_OK")
'@
$p3Script | Out-File -FilePath logs/_p3_inline.py -Encoding utf8
RunOrAbort "P3 sanity" "uv run python logs/_p3_inline.py"

# --- Step 5: 21-cell loop ---
$cells = @(
  # Block A: 64ch (9)
  @{n=64; cfg='attention';        task='binary'},
  @{n=64; cfg='band_power';       task='binary'},
  @{n=64; cfg='csp';              task='binary'},
  @{n=64; cfg='negative_control'; task='binary'},
  @{n=64; cfg='fdr';              task='ternary'},
  @{n=64; cfg='attention';        task='ternary'},
  @{n=64; cfg='band_power';       task='ternary'},
  @{n=64; cfg='csp';              task='ternary'},
  @{n=64; cfg='negative_control'; task='ternary'},
  # Block B: 32ch ternary (3)
  @{n=32; cfg='band_power';       task='ternary'},
  @{n=32; cfg='csp';              task='ternary'},
  @{n=32; cfg='negative_control'; task='ternary'},
  # Block C: 8ch (5)
  @{n=8;  cfg='negative_control'; task='binary'},
  @{n=8;  cfg='fdr';              task='ternary'},
  @{n=8;  cfg='band_power';       task='ternary'},
  @{n=8;  cfg='csp';              task='ternary'},
  @{n=8;  cfg='negative_control'; task='ternary'},
  # Block D: 4ch ternary (4)
  @{n=4;  cfg='fdr';              task='ternary'},
  @{n=4;  cfg='attention';        task='ternary'},
  @{n=4;  cfg='band_power';       task='ternary'},
  @{n=4;  cfg='csp';              task='ternary'}
)

$total = $cells.Count
LogStep "21-CELL-LOOP-START total=$total"
for ($i = 0; $i -lt $total; $i++) {
    $c = $cells[$i]
    $idx = $i + 1
    $cmd = "uv run python scripts/experiments/run_cross_subject_comparison.py --paradigm imagery --models cbramod --config configs/cbramod_v3_cross.yaml --cache-only --no-wandb --task $($c.task) --channels $($c.n) --channel-config $($c.cfg)"
    LogStep "CELL $idx/$total START n=$($c.n) cfg=$($c.cfg) task=$($c.task)"
    Invoke-Expression $cmd 2>&1 | Tee-Object -Append -FilePath $LOG
    $rc = $LASTEXITCODE
    LogStep "CELL $idx/$total END   n=$($c.n) cfg=$($c.cfg) task=$($c.task) exit=$rc"
}

LogStep "ALL_21_DONE"
"=== driver finished at $(Get-Date -Format o) ===" | Out-File -Append $LOG -Encoding utf8
