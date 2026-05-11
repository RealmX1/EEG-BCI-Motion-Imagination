#!/bin/bash
# V1/V2/V3 transfer evaluation (close §6.4.2(A) gap from further_pretraining_analysis.md)
# Each transfer init weights = corresponding V<n> cross-subject checkpoint at same task

set +e

V1_CROSS_BIN="checkpoints/cross_subject/20260322_1116_cbramod_imagery_binary/best.pt"
V1_CROSS_TER="checkpoints/cross_subject/20260322_1543_cbramod_imagery_ternary/best.pt"
V2_CROSS_BIN="checkpoints/cross_subject/20260323_1517_cbramod_imagery_binary/best.pt"
V2_CROSS_TER="checkpoints/cross_subject/20260323_1709_cbramod_imagery_ternary/best.pt"
V3_CROSS_BIN="checkpoints/cross_subject/20260505_2100_cbramod_imagery_binary/best.pt"
V3_CROSS_TER="checkpoints/cross_subject/20260505_2131_cbramod_imagery_ternary/best.pt"

export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

echo "=== [0/6] Waiting for CPU+GPU idle ==="; date
uv run python scripts/overwatch/overwatch.py --disable-network
echo "Overwatch released at: $(date)"; echo

echo "=== [1/6] V1 transfer binary ==="; date
echo c | uv run python scripts/experiments/run_transfer_comparison.py --models cbramod --task binary --paradigm imagery \
  --pretrained-cbramod "$V1_CROSS_BIN" --cache-only --no-wandb --output-dir results/dapt_v1

echo; echo "=== [2/6] V1 transfer ternary ==="; date
echo c | uv run python scripts/experiments/run_transfer_comparison.py --models cbramod --task ternary --paradigm imagery \
  --pretrained-cbramod "$V1_CROSS_TER" --cache-only --no-wandb --output-dir results/dapt_v1

echo; echo "=== [3/6] V2 transfer binary ==="; date
echo c | uv run python scripts/experiments/run_transfer_comparison.py --models cbramod --task binary --paradigm imagery \
  --pretrained-cbramod "$V2_CROSS_BIN" --cache-only --no-wandb --output-dir results/dapt_v2

echo; echo "=== [4/6] V2 transfer ternary ==="; date
echo c | uv run python scripts/experiments/run_transfer_comparison.py --models cbramod --task ternary --paradigm imagery \
  --pretrained-cbramod "$V2_CROSS_TER" --cache-only --no-wandb --output-dir results/dapt_v2

echo; echo "=== [5/6] V3 transfer binary ==="; date
echo c | uv run python scripts/experiments/run_transfer_comparison.py --models cbramod --task binary --paradigm imagery \
  --pretrained-cbramod "$V3_CROSS_BIN" --cache-only --no-wandb --output-dir results/dapt_v3

echo; echo "=== [6/6] V3 transfer ternary ==="; date
echo c | uv run python scripts/experiments/run_transfer_comparison.py --models cbramod --task ternary --paradigm imagery \
  --pretrained-cbramod "$V3_CROSS_TER" --cache-only --no-wandb --output-dir results/dapt_v3

echo
echo "=== ALL_DONE: $(date) ==="
