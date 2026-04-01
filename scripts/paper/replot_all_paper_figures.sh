#!/bin/bash
# =============================================================================
# 论文 v3 全量图表重绘脚本
#
# 使用 --replot <run_tag> 从 ExperimentDB 重绘所有论文所需图表。
# 不执行训练，不写入 DB，仅重新生成 PNG。
#
# 用法:
#   bash scripts/paper/replot_all_paper_figures.sh
# =============================================================================
set -euo pipefail

run_tail() {
  "$@" 2>&1 | tail -3
}

echo "=============================================="
echo "Paper v3 Figure Re-generation: $(date)"
echo "=============================================="

# ============================================
# Part 1: 128ch 被试内对比 (Section 3.1)
# ============================================
echo ""
echo "=== [1/6] 128ch Within-Subject Comparison ==="

echo "  Binary (20260323_2237)..."
run_tail uv run python scripts/experiments/run_within_subject_comparison.py \
  --replot 20260323_2237 --cache-only

echo "  Ternary (20260329_0056)..."
run_tail uv run python scripts/experiments/run_within_subject_comparison.py \
  --replot 20260329_0056 --cache-only

# ============================================
# Part 2: 128ch 跨被试对比 (Section 3.2)
# ============================================
echo ""
echo "=== [2/6] 128ch Cross-Subject Comparison ==="

echo "  Binary Figure 2 (20260330_0709 + 20260324_0023)..."
run_tail uv run python scripts/paper/generate_paper_figures.py --figure figure2

echo "  Ternary (20260330_0735)..."
run_tail uv run python scripts/experiments/run_cross_subject_comparison.py \
  --replot 20260330_0735 --cache-only

# ============================================
# Part 3: 32ch 配置对比 + Channel Scaling (Section 3.3)
# ============================================
echo ""
echo "=== [3/6] 32ch Configs + Channel Scaling ==="

# 32ch paper figure + underlying config plots
echo "  32ch comparison summary figure..."
run_tail uv run python scripts/paper/generate_paper_figures.py --figure 32ch_comparison

echo "  32ch FDR Figure 3b (with 128ch baseline overlay)..."
run_tail uv run python scripts/paper/generate_paper_figures.py --figure figure3b

echo "  32ch Attention (20260330_1009)..."
run_tail uv run python scripts/experiments/run_cross_subject_comparison.py \
  --replot 20260330_1009 --cache-only

echo "  32ch CSP (20260330_1032)..."
run_tail uv run python scripts/experiments/run_cross_subject_comparison.py \
  --replot 20260330_1032 --cache-only

echo "  32ch Band Power (20260330_1105)..."
run_tail uv run python scripts/experiments/run_cross_subject_comparison.py \
  --replot 20260330_1105 --cache-only

echo "  32ch Commercial (20260330_1142)..."
run_tail uv run python scripts/experiments/run_cross_subject_comparison.py \
  --replot 20260330_1142 --cache-only

# Channel scaling: 61ch, 8ch, 4ch
echo "  61ch Standard 10-10 (20260330_1213)..."
run_tail uv run python scripts/experiments/run_cross_subject_comparison.py \
  --replot 20260330_1213 --cache-only

echo "  8ch FDR (20260330_1311)..."
run_tail uv run python scripts/experiments/run_cross_subject_comparison.py \
  --replot 20260330_1311 --cache-only

echo "  8ch Attention (20260330_1334)..."
run_tail uv run python scripts/experiments/run_cross_subject_comparison.py \
  --replot 20260330_1334 --cache-only

echo "  4ch Figure 5a/5b (with 128ch baseline overlay)..."
run_tail uv run python scripts/paper/generate_paper_figures.py --figure figure5

# ============================================
# Part 4: 128ch 迁移学习对比 (Section 3.4)
# ============================================
echo ""
echo "=== [4/6] 128ch Transfer Comparison ==="

echo "  Binary (20260329_0507)..."
run_tail uv run python scripts/experiments/run_transfer_comparison.py \
  --replot 20260329_0507 --cache-only

echo "  Ternary (20260329_0521)..."
run_tail uv run python scripts/experiments/run_transfer_comparison.py \
  --replot 20260329_0521 --cache-only

# ============================================
# Part 5: Extra Sessions (Section 3.5)
# 无 --replot 支持，从 JSON cache 重绘
# ============================================
echo ""
echo "=== [5/6] Extra Sessions (from JSON cache) ==="

echo "  Binary (20260329_1357)..."
run_tail uv run python scripts/paper/generate_extra_sessions_plots.py \
  --task binary --run-tag 20260329_1357

echo "  Ternary (20260329_1503)..."
run_tail uv run python scripts/paper/generate_extra_sessions_plots.py \
  --task ternary --run-tag 20260329_1503

# ============================================
# Part 6: 论文专属新图表 (Sections 3.3, 3.6, 3.7)
# ============================================
echo ""
echo "=== [6/6] Paper-specific New Figures ==="

echo "  Channel scaling curve..."
run_tail uv run python scripts/paper/generate_paper_figures.py --figure channel_scaling

echo "  Further pretraining comparison..."
run_tail uv run python scripts/paper/generate_paper_figures.py --figure further_pretraining

echo "  Inference latency..."
run_tail uv run python scripts/paper/generate_paper_figures.py --figure inference_latency

echo "  Extra sessions strategy comparison..."
run_tail uv run python scripts/paper/generate_paper_figures.py --figure extra_sessions_strategy

echo ""
echo "=============================================="
echo "ALL PAPER FIGURES COMPLETE: $(date)"
echo "=============================================="
echo ""
echo "Output locations:"
echo "  128ch within:     results/20260323_2237_*.png, results/20260329_0056_*.png"
echo "  128ch cross:      results/20260330_0709_*.png, results/20260330_0735_*.png"
echo "  32ch configs:     results/32_channel/{fdr,attention,csp,band_power,commercial}/*.png"
echo "  Channel scaling:  results/{61,8,4}_channel/*/*.png"
echo "  128ch transfer:   results/20260329_0507_*.png, results/20260329_0521_*.png"
echo "  Extra sessions:   paper/figures/extra_sessions_*.png"
echo "  Paper figures:    paper/figures/channel_scaling.png"
echo "                    paper/figures/further_pretraining.png"
echo "                    paper/figures/inference_latency.png"
echo "                    paper/figures/extra_sessions_strategy_comparison.png"
echo "                    results/20260330_0709_cross-subject_combined_imagery_binary.png"
echo "                    results/32_channel/fdr/20260330_0836_cross-subject_combined_imagery_binary.png"
echo "                    results/4_channel/{fdr_attention_overlap,negative_control}/*.png"
