#!/bin/bash
# Post-HPO 全量重跑脚本 — 仅 cross-subject（无 transfer）
# Phase 1: 128ch EEGNet baselines → Phase 2: 32ch comparison → Phase 3: channel scaling
set -e
export EEG_NONINTERACTIVE=1

echo "=============================================="
echo "Post-HPO Re-run (cross-subject only): $(date)"
echo "=============================================="

# ============================================
# Phase 1: 128ch EEGNet Cross-Subject Baselines
# ============================================
echo ""
echo "=== Phase 1: 128ch EEGNet Cross-Subject Baselines ==="
echo ""

echo "--- [1/12] EEGNet binary cross-subject 128ch (baseline) ---"
uv run python scripts/run_cross_subject_comparison.py \
  --models eegnet --task binary --cache-only --baseline

echo "--- [2/12] EEGNet ternary cross-subject 128ch (baseline) ---"
uv run python scripts/run_cross_subject_comparison.py \
  --models eegnet --task ternary --cache-only --baseline

# ============================================
# Phase 2: 32ch Binary Comparison (5 configs)
# ============================================
echo ""
echo "=== Phase 2: 32ch Binary Comparison ==="
echo ""

echo "--- [3/12] 32ch FDR cross-subject (CBraMod + EEGNet) ---"
uv run python scripts/run_cross_subject_comparison.py \
  --models eegnet cbramod --task binary --channels 32 --channel-config fdr --cache-only

echo "--- [4/12] 32ch Attention cross-subject (CBraMod only) ---"
uv run python scripts/run_cross_subject_comparison.py \
  --models cbramod --task binary --channels 32 --channel-config attention --cache-only

echo "--- [5/12] 32ch CSP cross-subject (CBraMod + EEGNet) ---"
uv run python scripts/run_cross_subject_comparison.py \
  --models eegnet cbramod --task binary --channels 32 --channel-config csp --cache-only

echo "--- [6/12] 32ch Band Power cross-subject (CBraMod + EEGNet) ---"
uv run python scripts/run_cross_subject_comparison.py \
  --models eegnet cbramod --task binary --channels 32 --channel-config band_power --cache-only

echo "--- [7/12] 32ch Commercial cross-subject (CBraMod + EEGNet) ---"
uv run python scripts/run_cross_subject_comparison.py \
  --models eegnet cbramod --task binary --channels 32 --channel-config commercial --cache-only

# ============================================
# Phase 3: Binary Channel Scaling (61/8/4ch)
# ============================================
echo ""
echo "=== Phase 3: Binary Channel Scaling ==="
echo ""

echo "--- [8/12] 61ch standard_1010 cross-subject (CBraMod + EEGNet) ---"
uv run python scripts/run_cross_subject_comparison.py \
  --models eegnet cbramod --task binary --channels 61 --channel-config standard_1010 --cache-only

echo "--- [9/12] 8ch FDR cross-subject (CBraMod + EEGNet) ---"
uv run python scripts/run_cross_subject_comparison.py \
  --models eegnet cbramod --task binary --channels 8 --channel-config fdr --cache-only

echo "--- [10/12] 8ch Attention cross-subject (CBraMod only) ---"
uv run python scripts/run_cross_subject_comparison.py \
  --models cbramod --task binary --channels 8 --channel-config attention --cache-only

echo "--- [11/12] 4ch FDR-Attention-Overlap cross-subject (CBraMod + EEGNet) ---"
uv run python scripts/run_cross_subject_comparison.py \
  --models eegnet cbramod --task binary --channels 4 --channel-config fdr_attention_overlap --cache-only

echo "--- [12/12] 4ch Negative Control cross-subject (CBraMod + EEGNet) ---"
uv run python scripts/run_cross_subject_comparison.py \
  --models eegnet cbramod --task binary --channels 4 --channel-config negative_control --cache-only

echo ""
echo "=============================================="
echo "ALL PHASES COMPLETE: $(date)"
echo "=============================================="
