#!/usr/bin/env bash
# EEGNet-Mid ablation — final capacity-curve datapoint between original paper's
# deepEEGNet (~1M upper estimate) and our successful v3 (5.84M).
#
# Architecture (configs/eegnet_mid_*.yaml):
#   F1=32, D=4, F2=256, kernel_length=64, mlp_hidden_dims=[1024, 1024], LayerNorm
#   → 1,897,282 params (117x EEGNet baseline 16K, 32.5% of v3 5.84M, 6.2% of CBraMod 30.5M)
#
# Same conv stem and HP as v3 — only mlp_hidden_dims changes [2048,2048]→[1024,1024]
# for a clean single-variable scaling study.
#
# Idle gate intentionally bypassed (user pre-authorized after first launch).

set -uo pipefail

REPO="/c/Users/zhang/Desktop/github/EEG-BCI"
cd "$REPO" || { echo "FATAL: cannot cd to $REPO"; exit 2; }

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }

log "================================================================"
log " EEGNet-Mid ablation"
log " Plan: 3 runs = within / cross / transfer  (binary only)"
log " Model: EEGNet-Mid (~1.9M params, 117x baseline) | Channels: 128"
log " HP: configs/eegnet_mid_{within,cross,transfer}.yaml"
log "================================================================"

# ── Step 0: GPU state snapshot (no gate — proceed immediately) ────────
log ""
log "[0/3] No idle gate — proceeding immediately."
nvidia_state=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1)
log "  Current GPU state: ${nvidia_state}"

run_step() {
  local label=$1
  shift
  log ""
  log "----------------------------------------------------------------"
  log " ${label}"
  log " CMD: $*"
  log "----------------------------------------------------------------"
  local rc=0
  "$@" || rc=$?
  if [ $rc -ne 0 ]; then
    log "[${label}] FAILED (exit ${rc})"
  else
    log "[${label}] OK"
  fi
  return $rc
}

pick_latest_cross_ckpt() {
  local task=$1
  local guard=$2
  local found
  found=$(find checkpoints/cross_subject \
            -path "*_eegnet_imagery_${task}/best.pt" \
            -newer "$guard" 2>/dev/null \
            | sort | tail -1)
  if [ -z "${found:-}" ] || [ ! -f "$found" ]; then
    return 1
  fi
  echo "$found"
}

# ── Step 1: cross_subject binary  (transfer binary depends on this) ────
GUARD_BIN=$(mktemp 2>/dev/null || echo "/tmp/eegnet_mid_guard_bin_$$")
: > "$GUARD_BIN" && touch "$GUARD_BIN"
sleep 1
run_step "1/3 cross_subject binary  EEGNet-Mid" \
  uv run python scripts/experiments/run_cross_subject_comparison.py \
    --models eegnet --task binary \
    --config configs/eegnet_mid_cross.yaml \
    --no-wandb --cache-only --force-retrain
CROSS_BIN_CKPT=$(pick_latest_cross_ckpt binary "$GUARD_BIN" 2>/dev/null || true)
log "  -> cross_subject_binary checkpoint: ${CROSS_BIN_CKPT:-<NOT FOUND>}"

# ── Step 2: within_subject binary  (independent) ───────────────────────
run_step "2/3 within_subject binary  EEGNet-Mid" \
  uv run python scripts/experiments/run_within_subject_comparison.py \
    --models eegnet --task binary \
    --config configs/eegnet_mid_within.yaml \
    --no-wandb --cache-only --force-retrain

# ── Step 3: transfer binary  (depends on step 1 checkpoint) ────────────
if [ -n "${CROSS_BIN_CKPT:-}" ] && [ -f "${CROSS_BIN_CKPT}" ]; then
  run_step "3/3 transfer binary       EEGNet-Mid" \
    uv run python scripts/experiments/run_transfer_comparison.py \
      --models eegnet --task binary \
      --config configs/eegnet_mid_transfer.yaml \
      --pretrained-eegnet "$CROSS_BIN_CKPT" \
      --no-wandb --cache-only --force-retrain
else
  log ""
  log "[3/3] SKIPPED: cross_subject binary checkpoint missing — transfer cannot run."
fi

log ""
log "================================================================"
log " EEGNet-Mid wrapper finished. Capacity-curve datapoint complete."
log " Compare against:"
log "   EEGNet-Huge v3 (5.84M): see eegnet-huge-v3-restart_20260509_084718"
log "     within  20260509_0928 (67.71%)"
log "     cross   20260509_0847 (51.37%)"
log "     transfer 20260509_1030 (80.62%)"
log "   CBraMod baseline (30M): within 85.15% / cross 90.68% / transfer 90.12%"
log "   EEGNet baseline (16K): query find_runs(model='eegnet', is_baseline=1)"
log "================================================================"
