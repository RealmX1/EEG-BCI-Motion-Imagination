#!/bin/bash
# Wrapper: wait for CPU+GPU idle, then run V4/V5 within + transfer (4 cells each, 8 total)
# Each line is independent — semicolon chaining so one failure doesn't kill the rest

set +e  # don't abort on individual eval failure

V4_DAPT="checkpoints/cbramod/further_pretrain_v4_20260509_2345/best_model.pth"
V5_DAPT="checkpoints/cbramod/further_pretrain_v5_20260510_1049/best_model.pth"
V4_CROSS_BIN="checkpoints/cross_subject/20260510_1710_cbramod_imagery_binary/best.pt"
V4_CROSS_TER="checkpoints/cross_subject/20260510_1020_cbramod_imagery_ternary/best.pt"
V5_CROSS_BIN="checkpoints/cross_subject/20260510_1812_cbramod_imagery_binary/best.pt"
V5_CROSS_TER="checkpoints/cross_subject/20260510_1738_cbramod_imagery_ternary/best.pt"

export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

echo "=== [0/8] Waiting for CPU+GPU idle (overwatch, --disable-network) ==="
date
uv run python scripts/overwatch/overwatch.py --disable-network
echo "Overwatch released at: $(date)"
echo

echo "=== [1/8] V4 within binary ==="; date
echo c | uv run python scripts/run_within_subject.py --model cbramod --task binary --paradigm imagery \
  --pretrained-weights "$V4_DAPT" --cache-only --no-wandb --output-dir results/dapt_v4

echo; echo "=== [2/8] V4 within ternary ==="; date
echo c | uv run python scripts/run_within_subject.py --model cbramod --task ternary --paradigm imagery \
  --pretrained-weights "$V4_DAPT" --cache-only --no-wandb --output-dir results/dapt_v4

echo; echo "=== [3/8] V4 transfer binary ==="; date
echo c | uv run python scripts/experiments/run_transfer_comparison.py --models cbramod --task binary --paradigm imagery \
  --pretrained-cbramod "$V4_CROSS_BIN" --cache-only --no-wandb --output-dir results/dapt_v4

echo; echo "=== [4/8] V4 transfer ternary ==="; date
echo c | uv run python scripts/experiments/run_transfer_comparison.py --models cbramod --task ternary --paradigm imagery \
  --pretrained-cbramod "$V4_CROSS_TER" --cache-only --no-wandb --output-dir results/dapt_v4

echo; echo "=== [5/8] V5 within binary ==="; date
echo c | uv run python scripts/run_within_subject.py --model cbramod --task binary --paradigm imagery \
  --pretrained-weights "$V5_DAPT" --cache-only --no-wandb --output-dir results/dapt_v5

echo; echo "=== [6/8] V5 within ternary ==="; date
echo c | uv run python scripts/run_within_subject.py --model cbramod --task ternary --paradigm imagery \
  --pretrained-weights "$V5_DAPT" --cache-only --no-wandb --output-dir results/dapt_v5

echo; echo "=== [7/8] V5 transfer binary ==="; date
echo c | uv run python scripts/experiments/run_transfer_comparison.py --models cbramod --task binary --paradigm imagery \
  --pretrained-cbramod "$V5_CROSS_BIN" --cache-only --no-wandb --output-dir results/dapt_v5

echo; echo "=== [8/8] V5 transfer ternary ==="; date
echo c | uv run python scripts/experiments/run_transfer_comparison.py --models cbramod --task ternary --paradigm imagery \
  --pretrained-cbramod "$V5_CROSS_TER" --cache-only --no-wandb --output-dir results/dapt_v5

echo
echo "=== ALL_DONE: $(date) ==="
