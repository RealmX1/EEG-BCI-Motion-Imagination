#!/usr/bin/env bash
# P0.3 Cross-Subject CBraMod Label-Shuffle Control — pipeline (overwatch gate removed per user request)
#
# Stage 1: smoke test direct call (no DB write) — 1 epoch, S01+S02
# Stage 2: full seed=42 run (21 subjects, writes DB + JSON cache)
# Stage 3: full seed=123 run (21 subjects, writes DB + JSON cache)
#
# Any stage failure aborts the chain (set -e). Stdout flushed for incremental
# monitor.sh reads (PYTHONUNBUFFERED=1 already set by long-run launcher).

set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root

echo
echo "============================================================"
echo "[$(date '+%F %T')] STAGE 1: smoke test — direct train_cross_subject() call"
echo "  (1 epoch, 2 subjects, NO DB write — sanity-check shuffle integration)"
echo "============================================================"
MPLBACKEND=Agg uv run python -c '
import sys
from src.training.train_cross_subject import train_cross_subject

res = train_cross_subject(
    subjects=["S01", "S02"],
    model_type="cbramod",
    task="binary",
    paradigm="imagery",
    epochs=1,
    cache_only=True,
    wandb_enabled=False,
    verbose=2,
    shuffle_labels=True,
    shuffle_seed=42,
    run_tag="SMOKE_LABELSHUFFLE",
)
test_acc = res.get("mean_test_acc", -1.0)
val_acc = res.get("val_acc", -1.0)
per_subj = res.get("per_subject_test_acc", {})
print(f"\n[SMOKE RESULT] mean_test_acc={test_acc:.4f}  val_acc={val_acc:.4f}", flush=True)
print(f"[SMOKE RESULT] per_subject_test_acc={per_subj}", flush=True)
if test_acc < 0:
    print("[SMOKE FAIL] mean_test_acc not reported", flush=True)
    sys.exit(2)
print("[SMOKE OK] integration test passed", flush=True)
'

echo
echo "============================================================"
echo "[$(date '+%F %T')] STAGE 2: full run seed=42 (21 subjects)"
echo "============================================================"
uv run python scripts/experiments/run_cross_subject_comparison.py \
    --task binary --paradigm imagery --models cbramod --cache-only \
    --shuffle-labels --shuffle-seed 42 --no-wandb

echo
echo "============================================================"
echo "[$(date '+%F %T')] STAGE 3: full run seed=123 (21 subjects)"
echo "============================================================"
uv run python scripts/experiments/run_cross_subject_comparison.py \
    --task binary --paradigm imagery --models cbramod --cache-only \
    --shuffle-labels --shuffle-seed 123 --no-wandb

echo
echo "============================================================"
echo "[$(date '+%F %T')] ALL STAGES COMPLETE — P0.3 done"
echo "============================================================"
