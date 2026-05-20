"""Single source of truth for paper-figure metadata.

Maps each fig_id (e.g. ``fig4b``, ``fig_s1``) to:
- The paper label used in the draft (e.g. "图 4b")
- The canonical output filename (basename) and output directory
- The primary generator script + command
- Optional secondary generator commands (legacy / alternate sources)

Consumed by:
- ``scripts/paper/generate_paper_figures.py`` (dispatch + history staging)
- ``scripts/paper/audit_draft_figures.py`` (future centralized audit)
- ``.claude/skills/figure-snapshot-diff/scripts/history_cli.py`` (import-snapshots ↔ filename ↔ fig_id mapping)

To add a new figure: append a ``FigureSpec(...)`` to ``FIGURES`` and ensure the
generator command actually produces ``output_dir/canonical_filename``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SecondaryGenerator:
    """A non-canonical alternate command that produces the same figure.

    Used for legacy generators we want to keep visible during audits but
    don't treat as authoritative.
    """
    label: str
    script: str
    command: str


@dataclass(frozen=True)
class FigureSpec:
    fig_id: str
    paper_label: str
    caption: str
    canonical_filename: str
    output_dir: str
    generator_script: str
    generator_command: str
    figure_generators_key: Optional[str] = None
    secondary_generators: tuple[SecondaryGenerator, ...] = field(default_factory=tuple)

    @property
    def canonical_output_path(self) -> str:
        return f"{self.output_dir}/{self.canonical_filename}"


_FIGS: list[FigureSpec] = [
    FigureSpec(
        fig_id="fig1",
        paper_label="图 1",
        caption="被试内 128ch 二分类逐被试对比",
        canonical_filename="20260323_2237_combined_imagery_binary.png",
        output_dir="results",
        generator_script="scripts/experiments/run_within_subject_comparison.py",
        generator_command="uv run python scripts/experiments/run_within_subject_comparison.py --replot 20260323_2237 --cache-only",
    ),
    # Phase 5 (2026-05-19, user-approved Option A): the audited `--replot
    # 20260330_0709` command produces a figure missing the CBraMod cross-subject
    # series (documented bug). The user kept the frozen original in Phase 4 and
    # rejected the raw replot. Authoritative generator is now the bug-fixed
    # native `generate_figure2_128ch_cross_subject` (key ``figure2``), which
    # merges CBraMod + EEGNet from two runs. The old replot is kept as a
    # secondary generator for audit provenance only.
    FigureSpec(
        fig_id="fig2",
        paper_label="图 2",
        caption="跨被试 128ch 二分类逐被试对比",
        canonical_filename="20260330_0709_cross-subject_combined_imagery_binary.png",
        output_dir="results",
        generator_script="scripts/paper/generate_paper_figures.py",
        generator_command="uv run python scripts/paper/generate_paper_figures.py --figure fig2",
        figure_generators_key="figure2",
        secondary_generators=(
            SecondaryGenerator(
                label="audit_claimed_replot_buggy",
                script="scripts/experiments/run_cross_subject_comparison.py",
                command="uv run python scripts/experiments/run_cross_subject_comparison.py --replot 20260330_0709 --cache-only",
            ),
        ),
    ),
    FigureSpec(
        fig_id="fig2b",
        paper_label="图 2b",
        caption="跨被试 vs 被试内 pooling 增益 forest plot",
        canonical_filename="cross_subject_pooling_forest.png",
        output_dir="paper/figures",
        generator_script="scripts/paper/generate_paper_figures.py",
        generator_command="uv run python scripts/paper/generate_paper_figures.py --figure cross_subject_pooling_forest",
        figure_generators_key="cross_subject_pooling_forest",
    ),
    FigureSpec(
        fig_id="fig3b",
        paper_label="图 3b",
        caption="32 通道五种配置双模型对比",
        canonical_filename="32ch_comparison.png",
        output_dir="paper/figures",
        generator_script="scripts/paper/generate_paper_figures.py",
        generator_command="uv run python scripts/paper/generate_paper_figures.py --figure 32ch_comparison",
        figure_generators_key="32ch_comparison",
    ),
    # Phase 5 (2026-05-19, user-approved Option A): the audited `--replot
    # 20260330_0836` regeneration was explicitly REJECTED by the user in Phase 4
    # (kept frozen original). Authoritative generator is now the bug-fixed
    # native `generate_figure3b_32ch_fdr` (key ``figure3b``; the function name
    # is historical — it writes fig3c's canonical path, 32ch-FDR cross-subject
    # with 128ch baseline overlay). Old replot kept as secondary for provenance.
    FigureSpec(
        fig_id="fig3c",
        paper_label="图 3c",
        caption="32 通道 FDR 配置跨被试二分类逐被试对比",
        canonical_filename="20260330_0836_cross-subject_combined_imagery_binary.png",
        output_dir="results/32_channel/fdr",
        generator_script="scripts/paper/generate_paper_figures.py",
        generator_command="uv run python scripts/paper/generate_paper_figures.py --figure fig3c",
        figure_generators_key="figure3b",
        secondary_generators=(
            SecondaryGenerator(
                label="audit_claimed_replot_rejected",
                script="scripts/experiments/run_cross_subject_comparison.py",
                command="uv run python scripts/experiments/run_cross_subject_comparison.py --replot 20260330_0836 --cache-only",
            ),
        ),
    ),
    FigureSpec(
        fig_id="fig3d",
        paper_label="图 3d",
        caption="40-cell reduced-channel matrix 全景",
        canonical_filename="reduced_channel_40cell_grid.png",
        output_dir="paper/figures",
        generator_script="scripts/paper/generate_paper_figures.py",
        generator_command="uv run python scripts/paper/generate_paper_figures.py --figure reduced_channel_40cell_grid",
        figure_generators_key="reduced_channel_40cell_grid",
    ),
    FigureSpec(
        fig_id="fig4",
        paper_label="图 4",
        caption="通道缩放曲线: CBraMod 跨被试二分类准确率随通道数变化",
        canonical_filename="channel_scaling_curve.png",
        output_dir="paper/figures",
        generator_script="scripts/paper/generate_paper_figures.py",
        generator_command="uv run python scripts/paper/generate_paper_figures.py --figure channel_scaling",
        figure_generators_key="channel_scaling",
    ),
    FigureSpec(
        fig_id="fig4b",
        paper_label="图 4b",
        caption="通道选择方法排序翻转 (32ch → 8ch → 4ch)",
        canonical_filename="channel_method_ranking_flip.png",
        output_dir="paper/figures",
        generator_script="scripts/paper/generate_paper_figures.py",
        generator_command="uv run python scripts/paper/generate_paper_figures.py --figure channel_ranking_flip",
        figure_generators_key="channel_ranking_flip",
    ),
    FigureSpec(
        fig_id="fig4c",
        paper_label="图 4c",
        caption="通道选择方法敏感度随通道数缩放",
        canonical_filename="sensitivity_scaling.png",
        output_dir="paper/figures",
        generator_script="scripts/paper/generate_paper_figures.py",
        generator_command="uv run python scripts/paper/generate_paper_figures.py --figure sensitivity_scaling",
        figure_generators_key="sensitivity_scaling",
    ),
    FigureSpec(
        fig_id="fig5",
        paper_label="图 5",
        caption="4 通道最优配置 vs 负控制逐被试对比",
        canonical_filename="fig5_4ch_optimal_vs_neg_control.png",
        output_dir="paper/figures",
        generator_script="scripts/paper/generate_paper_figures.py",
        generator_command="uv run python scripts/paper/generate_paper_figures.py --figure fig5_merged",
        figure_generators_key="fig5_merged",
    ),
    FigureSpec(
        fig_id="fig6",
        paper_label="图 6",
        caption="128 通道 XSI-FT 对比 (二分类)",
        canonical_filename="20260329_0507_transfer_combined_imagery_binary.png",
        output_dir="results",
        generator_script="scripts/experiments/run_transfer_comparison.py",
        generator_command="uv run python scripts/experiments/run_transfer_comparison.py --replot 20260329_0507 --merge-cache 20260507_1835 --cache-only",
    ),
    FigureSpec(
        fig_id="fig6b",
        paper_label="图 6b",
        caption="128 通道 XSI-FT 对比 (三分类)",
        canonical_filename="20260329_0448_transfer_combined_imagery_ternary.png",
        output_dir="results",
        generator_script="scripts/experiments/run_transfer_comparison.py",
        generator_command="uv run python scripts/experiments/run_transfer_comparison.py --replot 20260329_0448 --merge-cache 20260507_1913 --cache-only",
    ),
    FigureSpec(
        fig_id="fig7",
        paper_label="图 7",
        caption="Extra Sessions 二分类被试内对比",
        canonical_filename="extra_sessions_binary.png",
        output_dir="paper/figures",
        generator_script="scripts/paper/generate_paper_figures.py",
        generator_command="uv run python scripts/paper/generate_paper_figures.py --figure extra_sessions_binary_v2",
        figure_generators_key="extra_sessions_binary_v2",
        secondary_generators=(
            SecondaryGenerator(
                label="legacy",
                script="scripts/paper/generate_extra_sessions_comparison.py",
                command="uv run python scripts/paper/generate_extra_sessions_comparison.py --task binary",
            ),
        ),
    ),
    FigureSpec(
        fig_id="fig8",
        paper_label="图 8",
        caption="Extra Sessions 三分类对比",
        canonical_filename="extra_sessions_ternary.png",
        output_dir="paper/figures",
        generator_script="scripts/paper/generate_paper_figures.py",
        generator_command="uv run python scripts/paper/generate_paper_figures.py --figure extra_sessions_ternary_v2",
        figure_generators_key="extra_sessions_ternary_v2",
        secondary_generators=(
            SecondaryGenerator(
                label="legacy",
                script="scripts/paper/generate_extra_sessions_comparison.py",
                command="uv run python scripts/paper/generate_extra_sessions_comparison.py --task ternary",
            ),
        ),
    ),
    FigureSpec(
        fig_id="fig9",
        paper_label="图 9",
        caption="Extra Sessions 三范式总览",
        canonical_filename="extra_sessions_paradigm_binary.png",
        output_dir="paper/figures",
        generator_script="scripts/paper/generate_paper_figures.py",
        generator_command="uv run python scripts/paper/generate_paper_figures.py --figure extra_sessions_paradigm",
        figure_generators_key="extra_sessions_paradigm",
    ),
    FigureSpec(
        fig_id="fig10a",
        paper_label="图 10a",
        caption="DAPT V1-V5 by paradigm × task small-multiples (95% CI; 0/30 BH-FDR sig; ★ = V3 transfer-ternary outlier Δ=+1.09 pp)",
        canonical_filename="dapt_v1_v5_smallmultiples.png",
        output_dir="paper/figures",
        generator_script="scripts/paper/generate_paper_figures.py",
        generator_command="uv run python scripts/paper/generate_paper_figures.py --figure dapt_v1_v5_smallmultiples",
        figure_generators_key="dapt_v1_v5_smallmultiples",
    ),
    FigureSpec(
        fig_id="fig10b",
        paper_label="图 10b",
        caption="Further Pre-training 下游评估",
        canonical_filename="further_pretraining.png",
        output_dir="paper/figures",
        generator_script="scripts/paper/generate_paper_figures.py",
        generator_command="uv run python scripts/paper/generate_paper_figures.py --figure further_pretraining",
        figure_generators_key="further_pretraining",
    ),
    FigureSpec(
        fig_id="fig11",
        paper_label="图 11",
        caption="推理延迟与模型规模对比",
        canonical_filename="inference_latency.png",
        output_dir="paper/figures",
        generator_script="scripts/paper/generate_paper_figures.py",
        generator_command="uv run python scripts/paper/generate_paper_figures.py --figure inference_latency",
        figure_generators_key="inference_latency",
    ),
    FigureSpec(
        fig_id="fig12",
        paper_label="图 12",
        caption="§3.7 探索性消融总览",
        canonical_filename="exploratory_ablation_overview.png",
        output_dir="paper/figures",
        generator_script="scripts/paper/generate_paper_figures.py",
        generator_command="uv run python scripts/paper/generate_paper_figures.py --figure exploratory_ablation_overview",
        figure_generators_key="exploratory_ablation_overview",
    ),
    FigureSpec(
        fig_id="fig_s1",
        paper_label="Figure S1",
        caption="Extra Sessions 评估策略对比",
        canonical_filename="extra_sessions_strategy_comparison.png",
        output_dir="paper/figures",
        generator_script="scripts/paper/generate_paper_figures.py",
        generator_command="uv run python scripts/paper/generate_paper_figures.py --figure extra_sessions_strategy",
        figure_generators_key="extra_sessions_strategy",
    ),
    FigureSpec(
        fig_id="fig_s2",
        paper_label="Figure S2",
        caption="21-被试 × 8-条件准确率热图",
        canonical_filename="subject_heatmap.png",
        output_dir="paper/figures",
        generator_script="scripts/paper/generate_paper_figures.py",
        generator_command="uv run python scripts/paper/generate_paper_figures.py --figure subject_heatmap",
        figure_generators_key="subject_heatmap",
    ),
]


FIGURES: dict[str, FigureSpec] = {f.fig_id: f for f in _FIGS}

# Reverse lookup: canonical_filename basename → fig_id.
# Used by history_cli.py import-snapshots to map files in legacy backup/snapshot
# dirs (which use canonical filenames, not fig_ids) into the right _history/<fig_id>/.
BY_CANONICAL_FILENAME: dict[str, str] = {f.canonical_filename: f.fig_id for f in _FIGS}


def all_figures() -> list[FigureSpec]:
    """Stable ordering for iteration (matches paper draft order)."""
    return list(_FIGS)


def get(fig_id: str) -> FigureSpec:
    if fig_id not in FIGURES:
        raise KeyError(f"unknown fig_id: {fig_id}")
    return FIGURES[fig_id]


def fig_id_for_filename(basename: str) -> Optional[str]:
    """Map a canonical filename basename to its fig_id, or None if unmapped."""
    return BY_CANONICAL_FILENAME.get(basename)


# Snapshot dir → (timestamp, tag) for history import.
# Order matters: earlier timestamp first.
SNAPSHOT_SOURCES: list[dict] = [
    {
        "path": "paper/figures_backup_20260512_pre_standardization",
        "timestamp": "20260512_0848",
        "tag": "pre_standardization",
        "has_results_subdir": True,  # nested results/ contains fig1/2/3c/6/6b
    },
    {
        "path": "paper/figures_snapshot_pre_colorblind_palette_20260512_1253",
        "timestamp": "20260512_1253",
        "tag": "pre_colorblind_palette",
        "has_results_subdir": False,
    },
    {
        "path": "paper/figures_snapshot_pre_colorblind_20260512_125351",
        "timestamp": "20260512_1254",
        "tag": "pre_colorblind",
        "has_results_subdir": False,
    },
    {
        "path": "paper/figures_snapshot_pre_dapt_30cell_20260512_1401",
        "timestamp": "20260512_1401",
        "tag": "pre_dapt_30cell",
        "has_results_subdir": False,
    },
    {
        "path": "paper/figures_snapshot_pre_style_overhaul_20260512_1613",
        "timestamp": "20260512_1613",
        "tag": "pre_style_overhaul",
        "has_results_subdir": False,
    },
    {
        "path": "paper/figures_snapshot_pre_violin_swap_20260512_1750",
        "timestamp": "20260512_1750",
        "tag": "pre_violin_swap",
        "has_results_subdir": False,
    },
    {
        "path": "paper/figures_snapshot_pre_40cell_matrix_update_20260512_1903",
        "timestamp": "20260512_1903",
        "tag": "pre_40cell_matrix_update",
        "has_results_subdir": False,
    },
]


HISTORY_ROOT = "paper/figures/_history"


if __name__ == "__main__":
    # Quick sanity check
    import json
    print(json.dumps(
        {f.fig_id: {"label": f.paper_label, "canonical": f.canonical_output_path,
                    "cmd": f.generator_command} for f in _FIGS},
        ensure_ascii=False, indent=2,
    ))
