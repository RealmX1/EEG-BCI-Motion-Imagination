#!/usr/bin/env bash
# Pure retry of run_tag 20260509_0102 (within_subject ternary --no-pretrained).
#
# Goal: test reproducibility of the 18/21 chance-collapse pattern. Same command
# as the original — only stochastic difference is weight init / shuffle seed.
# HP unchanged from get_default_config('cbramod', 'ternary').

set -uo pipefail

REPO="/c/Users/zhang/Desktop/github/EEG-BCI"
cd "$REPO" || { echo "FATAL: cannot cd to $REPO"; exit 2; }

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }

log "================================================================"
log " Within-subject ternary random-init RETRY (reproducibility check)"
log " Original run_tag: 20260509_0102 (38.65% mean, 18/21 collapsed)"
log "================================================================"

# ── Step 0: overwatch idle gate ────────────────────────────────────────
log "[0/2] Waiting for GPU+CPU idle (network disabled)..."
uv run python scripts/overwatch/overwatch.py --disable-network
rc=$?
if [ $rc -ne 0 ]; then
  log "[0/2] overwatch exited with code ${rc} — aborting."
  exit "$rc"
fi
log "[0/2] OK — resources idle."

# ── Step 1: within_subject ternary --no-pretrained ─────────────────────
log ""
log "----------------------------------------------------------------"
log "[1/2] within_subject ternary --no-pretrained"
log "----------------------------------------------------------------"
uv run python scripts/experiments/run_within_subject_comparison.py \
  --models cbramod --task ternary --no-pretrained \
  --no-wandb --cache-only --force-retrain
rc=$?
if [ $rc -ne 0 ]; then
  log "[1/2] FAILED (exit ${rc})"
  exit "$rc"
fi
log "[1/2] OK"

log ""
log "================================================================"
log " Retry finished. Compare new run_tag against 20260509_0102:"
log "   - mean accuracy"
log "   - per-subject test_acc (count of subjects at chance ±2pp)"
log "   - which subjects escaped vs collapsed"
log "================================================================"
