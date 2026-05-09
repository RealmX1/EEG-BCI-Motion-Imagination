#!/usr/bin/env bash
# EEGNet-Huge-v3 ablation — 3 sequential runs (binary only, 3 paradigms).
#
# Goal: pair with CBraMod random-init ablation in paper draft v3 to disentangle
#   (capacity vs. architecture vs. pretraining). After v1 ([4096,4096], 19.99M,
#   no LayerNorm) and v2 ([5120,5120], 30.22M, no LayerNorm) both failed to
#   escape chance loss, v3 narrows the MLP head and adds LayerNorm to make the
#   model trainable while keeping params 2 orders of magnitude above baseline.
#
# Architecture (configs/eegnet_huge_*.yaml, all v3):
#   F1=32, D=4, F2=256, kernel_length=64, mlp_hidden_dims=[2048, 2048] + LayerNorm
#   → 5,837,634 params (361× EEGNet baseline 16K, 19.2% of CBraMod 30.5M)
#   Dropout 0.4, weight_decay 0.05 (cross) / 0.03 (within/transfer)
#   LR 8e-4 (cross) / 1.5e-3 (within) / 5e-4 (transfer) — peak LRs tuned per paradigm
#
# Design choices (decided 2026-05-09):
#   - 128 channels (matches all current EEGNet baselines at n_subjects=21).
#   - Binary task only (matches user scoping decision).
#   - HP source: configs/eegnet_huge_{within,cross,transfer}.yaml (no defaults).
#   - Transfer paradigm uses the cross_subject checkpoint produced by THIS run,
#     not historical EEGNet baseline checkpoints.
#   - Order: cross BEFORE transfer (transfer depends on cross checkpoint);
#     within sandwiched in between (independent).
#   - --no-plot is NOT used (project convention: comparison scripts plot).
#   - EEGNet has no foundation-model backbone, so --no-pretrained is a no-op
#     and is omitted (it's a cbramod-only flag for skipping the .pth load).
#   - GPU idle gate is BYPASSED below — invoke only when GPU is verified free
#     or another concurrent training job is acceptable.

set -uo pipefail

REPO="/c/Users/zhang/Desktop/github/EEG-BCI"
cd "$REPO" || { echo "FATAL: cannot cd to $REPO"; exit 2; }

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }

log "================================================================"
log " EEGNet-Huge ablation"
log " Plan: 3 runs = within / cross / transfer  (binary only)"
log " Model: EEGNet-Huge-v3 (~5.84M params, 361x baseline) | Channels: 128"
log " HP: configs/eegnet_huge_{within,cross,transfer}.yaml"
log "================================================================"

# ── Step 0: GPU/CPU idle gate (BYPASSED for v3 — user gave green light) ──
log ""
log "[0/3] GPU idle gate BYPASSED — user authorized immediate start (v3 launch)."
nvidia_state=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1)
log "  Current GPU state: ${nvidia_state}"

# ── Helper: run a step, log status, never abort the wrapper ────────────
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

# ── Helper: pick the most recent eegnet cross checkpoint for a task ───
# Uses a guard mtime file so we only accept checkpoints created during THIS run.
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
GUARD_BIN=$(mktemp 2>/dev/null || echo "/tmp/eegnet_huge_guard_bin_$$")
: > "$GUARD_BIN" && touch "$GUARD_BIN"
sleep 1
run_step "1/3 cross_subject binary  EEGNet-Huge" \
  uv run python scripts/experiments/run_cross_subject_comparison.py \
    --models eegnet --task binary \
    --config configs/eegnet_huge_cross.yaml \
    --no-wandb --cache-only --force-retrain
CROSS_BIN_CKPT=$(pick_latest_cross_ckpt binary "$GUARD_BIN" 2>/dev/null || true)
log "  -> cross_subject_binary checkpoint: ${CROSS_BIN_CKPT:-<NOT FOUND>}"

# ── Step 2: within_subject binary  (independent) ───────────────────────
run_step "2/3 within_subject binary  EEGNet-Huge" \
  uv run python scripts/experiments/run_within_subject_comparison.py \
    --models eegnet --task binary \
    --config configs/eegnet_huge_within.yaml \
    --no-wandb --cache-only --force-retrain

# ── Step 3: transfer binary  (depends on step 1 checkpoint) ────────────
if [ -n "${CROSS_BIN_CKPT:-}" ] && [ -f "${CROSS_BIN_CKPT}" ]; then
  run_step "3/3 transfer binary       EEGNet-Huge" \
    uv run python scripts/experiments/run_transfer_comparison.py \
      --models eegnet --task binary \
      --config configs/eegnet_huge_transfer.yaml \
      --pretrained-eegnet "$CROSS_BIN_CKPT" \
      --no-wandb --cache-only --force-retrain
else
  log ""
  log "[3/3] SKIPPED: cross_subject binary checkpoint missing — transfer cannot run."
fi

log ""
log "================================================================"
log " Wrapper finished. Review results/ JSON cache and ExperimentDB."
log " Compare against:"
log "   CBraMod baseline binary:"
log "     within  20260323_2237 (85.15%)"
log "     cross   20260324_0023 (90.68%)"
log "     transfer 20260329_0507 (90.12%)"
log "   CBraMod random-init binary: see run_random_init_ablation.sh outputs"
log "   EEGNet baseline: query ExperimentDB"
log "     find_runs(model='eegnet', task='binary', is_baseline=1)"
log "================================================================"
