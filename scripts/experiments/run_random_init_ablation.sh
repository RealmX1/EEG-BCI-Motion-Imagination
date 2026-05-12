#!/usr/bin/env bash
# Random-init CBraMod ablation — 6 sequential runs (3 paradigms x 2 tasks).
#
# Goal: produce a from-scratch (random initialization) CBraMod result matrix to
# pair against the historical original-weights CBraMod baselines and the EEGNet
# baselines. Addresses paper draft v3 Limitation #7 ("only one base model and
# one pretraining objective tested").
#
# Design choices (decided 2026-05-08):
#   - 128 channels (matches all current cbramod is_baseline runs at n_subjects=21).
#   - HP source: get_default_config() in src/config/training.py — i.e. NO --config
#     flag, identical HP to historical original-weights baseline. Only difference
#     vs baseline = backbone init (random vs original CBraMod weights).
#   - Transfer paradigm uses the cross_subject checkpoint produced by THIS run,
#     never historical original-weights checkpoints, so transfer is also
#     end-to-end from-scratch.
#   - Order: cross BEFORE transfer (transfer depends on cross checkpoint);
#     within sandwiched in between (independent).
#   - --no-plot is NOT used (project convention: comparison scripts plot).

set -uo pipefail

REPO="/c/Users/zhang/Desktop/github/EEG-BCI"
cd "$REPO" || { echo "FATAL: cannot cd to $REPO"; exit 2; }

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }

log "================================================================"
log " Random-init CBraMod ablation"
log " Plan: 6 runs = 3 paradigms (within / cross / transfer) x 2 tasks (binary / ternary)"
log " Models: cbramod ONLY  |  Channels: 128  |  HP: defaults (no --config)"
log "================================================================"

# ── Step 0: overwatch idle gate REMOVED 2026-05-08 (no other compute users) ──
log "[0/7] Skipping overwatch idle gate — assuming exclusive GPU/CPU access."

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

# ── Helper: pick the most recent cbramod cross checkpoint for a task ──
# Uses a guard mtime file so we only accept checkpoints created during THIS run,
# not stale historical checkpoints with the same naming pattern.
pick_latest_cross_ckpt() {
  local task=$1
  local guard=$2
  local found
  found=$(find checkpoints/cross_subject \
            -path "*_cbramod_imagery_${task}/best.pt" \
            -newer "$guard" 2>/dev/null \
            | sort | tail -1)
  if [ -z "${found:-}" ] || [ ! -f "$found" ]; then
    return 1
  fi
  echo "$found"
}

# ── Step 1: cross_subject binary  (transfer binary depends on this) ────
GUARD_BIN=$(mktemp 2>/dev/null || echo "/tmp/random_init_guard_bin_$$")
: > "$GUARD_BIN" && touch "$GUARD_BIN"
sleep 1
run_step "1/6 cross_subject binary  --no-pretrained" \
  uv run python scripts/experiments/run_cross_subject_comparison.py \
    --models cbramod --task binary --no-pretrained \
    --no-wandb --cache-only --force-retrain
CROSS_BIN_CKPT=$(pick_latest_cross_ckpt binary "$GUARD_BIN" 2>/dev/null || true)
log "  -> cross_subject_binary checkpoint: ${CROSS_BIN_CKPT:-<NOT FOUND>}"

# ── Step 2: cross_subject ternary  (transfer ternary depends on this) ──
GUARD_TER=$(mktemp 2>/dev/null || echo "/tmp/random_init_guard_ter_$$")
: > "$GUARD_TER" && touch "$GUARD_TER"
sleep 1
run_step "2/6 cross_subject ternary --no-pretrained" \
  uv run python scripts/experiments/run_cross_subject_comparison.py \
    --models cbramod --task ternary --no-pretrained \
    --no-wandb --cache-only --force-retrain
CROSS_TER_CKPT=$(pick_latest_cross_ckpt ternary "$GUARD_TER" 2>/dev/null || true)
log "  -> cross_subject_ternary checkpoint: ${CROSS_TER_CKPT:-<NOT FOUND>}"

# ── Step 3: within_subject binary  (independent) ───────────────────────
run_step "3/6 within_subject binary  --no-pretrained" \
  uv run python scripts/experiments/run_within_subject_comparison.py \
    --models cbramod --task binary --no-pretrained \
    --no-wandb --cache-only --force-retrain

# ── Step 4: within_subject ternary  (independent) ──────────────────────
run_step "4/6 within_subject ternary --no-pretrained" \
  uv run python scripts/experiments/run_within_subject_comparison.py \
    --models cbramod --task ternary --no-pretrained \
    --no-wandb --cache-only --force-retrain

# ── Step 5: transfer binary  (depends on step 1 checkpoint) ────────────
if [ -n "${CROSS_BIN_CKPT:-}" ] && [ -f "${CROSS_BIN_CKPT}" ]; then
  run_step "5/6 transfer binary       --no-pretrained" \
    uv run python scripts/experiments/run_transfer_comparison.py \
      --models cbramod --task binary --no-pretrained \
      --pretrained-cbramod "$CROSS_BIN_CKPT" \
      --no-wandb --cache-only --force-retrain
else
  log ""
  log "[5/6] SKIPPED: cross_subject binary checkpoint missing — transfer binary cannot run from-scratch."
fi

# ── Step 6: transfer ternary  (depends on step 2 checkpoint) ───────────
if [ -n "${CROSS_TER_CKPT:-}" ] && [ -f "${CROSS_TER_CKPT}" ]; then
  run_step "6/6 transfer ternary      --no-pretrained" \
    uv run python scripts/experiments/run_transfer_comparison.py \
      --models cbramod --task ternary --no-pretrained \
      --pretrained-cbramod "$CROSS_TER_CKPT" \
      --no-wandb --cache-only --force-retrain
else
  log ""
  log "[6/6] SKIPPED: cross_subject ternary checkpoint missing — transfer ternary cannot run from-scratch."
fi

log ""
log "================================================================"
log " Wrapper finished. Review results/ JSON cache and ExperimentDB."
log " Compare against original-weights CBraMod baselines:"
log "   within_subject  binary  20260323_2237 / ternary 20260323_2320"
log "   cross_subject   binary  20260324_0023 / ternary 20260324_0109"
log "   transfer        binary  20260329_0507 / ternary 20260329_0521"
log "================================================================"
