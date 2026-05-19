#!/usr/bin/env bash
# 16-channel 5-config sweep (binary + ternary) + 61ch standard_1010 ternary makeup.
#
# Gate strategy (refined 2026-05-13 after first attempt stalled in overwatch for
# 1h+ — desktop CPU/NET rolling avg almost never drops below overwatch defaults):
#   1. PRIMARY gate: nvidia-smi query for foreign python compute processes
#      (fast, deterministic, exactly the thing we care about).
#   2. SECONDARY (best-effort): overwatch.py with --disable-network, capped at
#      30 min — if it doesn't release, we proceed anyway because the primary
#      gate already proved GPU is free for compute.
#   3. No per-run gating: once we've verified GPU is free at the very start,
#      we run all 11 experiments back-to-back. The sweep itself uses the GPU,
#      so a per-run check would trivially fail.
#
# Plan (11 runs total, cross-subject CBraMod only):
#   1. 61ch standard_1010 ternary           (binary 20260330_1213 already present)
#   2-6.  16ch {fdr, csp, attention, band_power, negative_control} binary
#   7-11. 16ch {fdr, csp, attention, band_power, negative_control} ternary
# set -u catches unset-var typos; set -e is INTENTIONALLY omitted so that a
# single failing run is logged (FAIL_*) and the sweep still runs the rest —
# do not add `set -e` or `|| exit` to the run_one calls below.
set -u
cd "$(dirname "$0")/../.." || exit 1

WRAP="uv run python scripts/experiments/run_reduced_channel_experiment.py"
LOG_DIR="logs/16ch_sweep"
mkdir -p "$LOG_DIR"

echo "===== SWEEP START $(date '+%Y-%m-%d %H:%M:%S') ====="
echo "(start)" > "$LOG_DIR/_results.txt"

# ── Primary gate: GPU compute apps ─────────────────────────────────────────
echo ""
echo "############################################################"
echo "# [$(date '+%Y-%m-%d %H:%M:%S')] PRIMARY GATE: GPU compute apps"
echo "############################################################"
FOREIGN_GPU_PY=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -iE "python|train" || true)
if [[ -n "$FOREIGN_GPU_PY" ]]; then
  echo "ABORT: foreign python compute on GPU:"
  echo "$FOREIGN_GPU_PY"
  echo "ABORT_PRE_SWEEP_gpu_busy=1" >> "$LOG_DIR/_results.txt"
  exit 1
fi
echo "GPU compute apps: clear (no python on GPU)"

# ── Secondary best-effort gate: overwatch with timeout ─────────────────────
echo ""
echo "############################################################"
echo "# [$(date '+%Y-%m-%d %H:%M:%S')] SECONDARY GATE: overwatch (cap 30 min)"
echo "############################################################"
OW_LOG="$LOG_DIR/_overwatch.log"
# If `uv` is not on PATH (e.g. under nohup with a stripped env), the overwatch
# subprocess would exit 127 instantly, the kill -0 loop would end without ever
# hitting the timeout branch, and we'd silently "proceed". That's still SAFE
# because the primary gate above already proved the GPU is free for compute —
# but a broken environment must NOT be inferred from a cryptic "exit 127". So:
# surface it loudly, skip the (pointless) dead-PID dance, and rely on the
# primary gate. We do NOT exit here: the training command also uses `uv run`,
# so if uv were truly gone every run would FAIL_* and be obvious anyway.
if ! command -v uv >/dev/null 2>&1; then
  echo "WARNING: 'uv' not on PATH — SECONDARY GATE SKIPPED (overwatch can't run)."
  echo "         Relying on PRIMARY gate (already passed). If uv is genuinely"
  echo "         missing, every run below will also FAIL_ (training uses uv run)."
  echo "ABORT_SECONDARY_GATE_uv_missing=skipped_primary_gate_ok" >> "$LOG_DIR/_results.txt"
else
  # Run overwatch with --disable-network in background, kill after 30 min
  uv run python scripts/overwatch/overwatch.py --disable-network > "$OW_LOG" 2>&1 &
  OW_PID=$!
  OW_TIMEOUT=1800
  OW_START=$(date +%s)
  while kill -0 "$OW_PID" 2>/dev/null; do
    ELAPSED=$(($(date +%s) - OW_START))
    if [[ $ELAPSED -ge $OW_TIMEOUT ]]; then
      echo "Overwatch timeout at ${OW_TIMEOUT}s — killing and proceeding (primary gate already passed)"
      kill -TERM "$OW_PID" 2>/dev/null || true
      sleep 3
      kill -KILL "$OW_PID" 2>/dev/null || true
      break
    fi
    sleep 30
  done
  wait "$OW_PID" 2>/dev/null
  OW_RC=$?
  if [[ $OW_RC -eq 0 ]]; then
    echo "Overwatch released cleanly"
  else
    echo "Overwatch exit $OW_RC (proceeding — primary gate passed)"
  fi
  echo "(see $OW_LOG for full overwatch trace)"
fi

# ── Run helper ──────────────────────────────────────────────────────────────
run_one () {
  local n_ch="$1"; local cfg="$2"; local task="$3"
  local tag="${n_ch}ch_${cfg}_${task}"
  local log="$LOG_DIR/${tag}.log"

  echo ""
  echo "############################################################"
  echo "# [$(date '+%Y-%m-%d %H:%M:%S')] Launching ${tag}"
  echo "#   log: ${log}"
  echo "############################################################"
  $WRAP --channels "$n_ch" --channel-config "$cfg" \
        --models cbramod --tasks "$task" --steps cross \
        > "$log" 2>&1
  local rc=$?
  tail -20 "$log"
  if [[ "$rc" -eq 0 ]]; then
    echo "OK_${tag}=success" >> "$LOG_DIR/_results.txt"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [OK] ${tag}"
  else
    echo "FAIL_${tag}=rc${rc}" >> "$LOG_DIR/_results.txt"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [FAIL ${rc}] ${tag}"
  fi
  return "$rc"
}

# ── Sweep ──────────────────────────────────────────────────────────────────
run_one 61 standard_1010 ternary

for cfg in fdr csp attention band_power negative_control; do
  run_one 16 "$cfg" binary
done

for cfg in fdr csp attention band_power negative_control; do
  run_one 16 "$cfg" ternary
done

echo ""
echo "===== SWEEP END $(date '+%Y-%m-%d %H:%M:%S') ====="
echo "Summary:"
cat "$LOG_DIR/_results.txt"
