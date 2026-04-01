#!/usr/bin/env python
"""
论文 v3 统计检验与数据提取脚本.

计算论文中引用的所有统计检验（配对 t 检验、观测范围等），
确保每个数值可追溯到原始实验结果文件。

设计目标：
  - 所有引用的 run 路径集中定义在 RUN_REGISTRY 中
  - 更换 run 时只需修改 RUN_REGISTRY 对应条目
  - 运行脚本即可重算所有统计量

Usage:
    uv run python scripts/paper/compute_paper_statistics.py
    uv run python scripts/paper/compute_paper_statistics.py --section 3.2
    uv run python scripts/paper/compute_paper_statistics.py --section all
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# RUN REGISTRY — 论文中引用的所有实验运行
# 更换 run 时只需修改此处路径
# =============================================================================

RUN_REGISTRY = {
    # Section 3.1 / 3.2: 被试内 & 跨被试对比 (128ch)
    "within_eegnet_binary": "results/20260316_1411_comparison_cache_imagery_binary.json",
    "within_cbramod_binary": "results/20260323_2237_comparison_cache_imagery_binary.json",
    "cross_eegnet_binary": "results/20260330_0709_cross_subject_cache_imagery_binary.json",
    "cross_cbramod_binary": "results/20260324_0023_cross_subject_cache_imagery_binary.json",
    # Section 3.4: 迁移学习 (128ch)
    "transfer_binary": "results/20260329_0507_transfer_cache_imagery_binary.json",
    "transfer_ternary": "results/20260329_0448_transfer_cache_imagery_ternary.json",
    "cross_cbramod_ternary": "results/20260324_0109_cross_subject_cache_imagery_ternary.json",
    # Section 3.5: Extra sessions
    "extra_sessions_binary": "results/20260324_2131_extra_sessions_cache_imagery_binary.json",
    "extra_sessions_ternary": "results/20260331_0827_extra_sessions_cache_imagery_ternary.json",
    "extra_sessions_cross_binary": "results/20260326_1409_cross_subject_extra_sessions_cache_imagery_binary.json",
    "extra_sessions_cross_ternary": "results/20260327_0303_cross_subject_extra_sessions_cache_imagery_ternary.json",
    "extra_sessions_transfer_binary": "results/20260329_1357_extra_sessions_cache_imagery_binary.json",
}

EXTRA_SESSION_STEPS = ["baseline", "sess03", "sess04", "sess05"]
EXTRA_SESSION_STEP_LABELS = {
    "baseline": "Baseline",
    "sess03": "+Sess03",
    "sess04": "+Sess04",
    "sess05": "+Sess05",
}


# =============================================================================
# Data Loading Helpers
# =============================================================================


def load_json(key_or_path: str) -> dict:
    """Load JSON from registry key or direct path."""
    path = RUN_REGISTRY.get(key_or_path, key_or_path)
    with open(PROJECT_ROOT / path) as f:
        return json.load(f)


def extract_per_subject_accs(
    cache: dict, model: str, step: Optional[str] = None
) -> Dict[str, float]:
    """Extract per-subject test accuracies (as percentages) from any JSON format.

    Returns dict of {subject_id: accuracy_percent}.

    Supports formats:
      - Cross-subject: results[model]['per_subject_test_acc'][subj] (0-1)
      - Cross-subject extra sessions: results[model][step]['per_subject_test_acc'][subj] (0-1)
      - Within/transfer: results[model][subj]['test_acc_majority'] (0-1)
      - Extra sessions: results[model][subj][step]['test_acc_majority'] (0-1)
    """
    results = cache.get("results", {})
    model_data = results.get(model, {})

    accs = {}

    # Format 1: per_subject_test_acc dict (cross-subject)
    psa = model_data.get("per_subject_test_acc", {})
    if psa and step is None:
        return {subj: acc * 100 for subj, acc in psa.items()}

    # Format 1b: cross-subject extra sessions cache
    if step is not None:
        step_results = model_data.get(step, {})
        if isinstance(step_results, dict):
            step_psa = step_results.get("per_subject_test_acc", {})
            if step_psa:
                return {subj: acc * 100 for subj, acc in step_psa.items()}

    # Format 2/3: direct subject keys
    for subj, subj_data in model_data.items():
        if not subj.startswith("S") or not isinstance(subj_data, dict):
            continue

        if step is not None:
            # Extra sessions format: subj_data[step]['test_acc_majority']
            step_data = subj_data.get(step, {})
            if isinstance(step_data, dict):
                acc = step_data.get("test_acc_majority", step_data.get("test_acc"))
                if acc is not None:
                    accs[subj] = acc * 100
        else:
            # Within/transfer format: subj_data['test_acc_majority']
            acc = subj_data.get("test_acc_majority", subj_data.get("test_acc"))
            if acc is not None:
                accs[subj] = acc * 100

    return accs


def paired_ttest(
    accs_a: Dict[str, float], accs_b: Dict[str, float], label: str
) -> dict:
    """Run paired t-test on aligned subject accuracies. Returns result dict."""
    common = sorted(set(accs_a.keys()) & set(accs_b.keys()))
    if len(common) < 3:
        logger.warning(f"  {label}: only {len(common)} common subjects, skipping")
        return {"label": label, "n": len(common), "error": "too few subjects"}

    a = np.array([accs_a[s] for s in common])
    b = np.array([accs_b[s] for s in common])
    t_stat, p_val = stats.ttest_rel(a, b)
    delta = (b - a).mean()

    return {
        "label": label,
        "n": len(common),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "delta_pp": float(delta),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
    }


def print_ttest(result: dict):
    """Pretty-print a paired t-test result."""
    if "error" in result:
        logger.info(f"  {result['label']}: ERROR — {result['error']}")
        return
    sig = "***" if result["p_value"] < 0.001 else (
        "**" if result["p_value"] < 0.01 else (
        "*" if result["p_value"] < 0.05 else "n.s."))
    logger.info(
        f"  {result['label']} (N={result['n']}): "
        f"{result['mean_a']:.2f}% → {result['mean_b']:.2f}% "
        f"(Δ = {result['delta_pp']:+.2f} pp, "
        f"t = {result['t_stat']:.3f}, p = {result['p_value']:.4f} {sig})"
    )


def describe_accs(accs: Dict[str, float], label: str) -> dict:
    """Compute descriptive statistics for per-subject accuracies."""
    vals = np.array(list(accs.values()))
    result = {
        "label": label,
        "n": len(vals),
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "min": float(vals.min()),
        "max": float(vals.max()),
        "min_subj": min(accs, key=accs.get),
        "max_subj": max(accs, key=accs.get),
    }
    return result


def print_describe(result: dict):
    """Pretty-print descriptive statistics."""
    logger.info(
        f"  {result['label']} (N={result['n']}): "
        f"{result['mean']:.2f} ± {result['std']:.2f}% "
        f"[{result['min']:.2f}% ({result['min_subj']}) – "
        f"{result['max']:.2f}% ({result['max_subj']})]"
    )


def extract_extra_session_step_series(cache: dict, model: str) -> Dict[str, Dict[str, float]]:
    """Extract aligned per-step per-subject accuracies for extra-session analyses."""
    return {
        step: extract_per_subject_accs(cache, model, step=step)
        for step in EXTRA_SESSION_STEPS
    }


def format_mean_std(accs: Dict[str, float]) -> str:
    """Format a per-subject accuracy dict as mean ± SD markdown text."""
    if not accs:
        return "—"
    desc = describe_accs(accs, "tmp")
    return f"{desc['mean']:.2f} ± {desc['std']:.2f}%"


def format_p_value(p_value: float) -> str:
    """Format p-values for paper tables."""
    return f"{p_value:.3f}"


# =============================================================================
# Section-specific statistics
# =============================================================================


def section_3_2():
    """Section 3.2: EEGNet within vs cross-subject (128ch binary).

    Tests whether EEGNet benefits from cross-subject data pooling.
    """
    logger.info("=" * 60)
    logger.info("Section 3.2: EEGNet within vs cross-subject (128ch binary)")
    logger.info("=" * 60)

    within_cache = load_json("within_eegnet_binary")
    cross_cache = load_json("cross_eegnet_binary")

    eegnet_within = extract_per_subject_accs(within_cache, "eegnet")
    eegnet_cross = extract_per_subject_accs(cross_cache, "eegnet")

    logger.info("\nDescriptive statistics:")
    print_describe(describe_accs(eegnet_within, "EEGNet within-subject"))
    print_describe(describe_accs(eegnet_cross, "EEGNet cross-subject"))

    logger.info("\nPaired t-test (within vs cross):")
    result = paired_ttest(
        eegnet_within, eegnet_cross, "EEGNet within → cross"
    )
    print_ttest(result)

    logger.info(f"\n  数据来源:")
    logger.info(f"    被试内: {RUN_REGISTRY['within_eegnet_binary']}")
    logger.info(f"    跨被试: {RUN_REGISTRY['cross_eegnet_binary']}")

    return result


def section_3_4():
    """Section 3.4: Transfer learning vs cross-subject (128ch).

    Tests whether transfer learning provides significant benefit over
    cross-subject training alone.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Section 3.4: Transfer learning vs cross-subject (128ch)")
    logger.info("=" * 60)

    results = {}

    for task, cross_key, transfer_key in [
        ("binary", "cross_cbramod_binary", "transfer_binary"),
        ("ternary", "cross_cbramod_ternary", "transfer_ternary"),
    ]:
        cross_cache = load_json(cross_key)
        transfer_cache = load_json(transfer_key)

        cross_accs = extract_per_subject_accs(cross_cache, "cbramod")
        transfer_accs = extract_per_subject_accs(transfer_cache, "cbramod")

        logger.info(f"\n{task.upper()}:")
        print_describe(describe_accs(cross_accs, f"CBraMod cross-subject {task}"))
        print_describe(describe_accs(transfer_accs, f"CBraMod transfer {task}"))

        result = paired_ttest(
            cross_accs, transfer_accs, f"cross → transfer ({task})"
        )
        print_ttest(result)
        results[task] = result

    logger.info(f"\n  数据来源:")
    logger.info(f"    跨被试二分类: {RUN_REGISTRY['cross_cbramod_binary']}")
    logger.info(f"    迁移二分类: {RUN_REGISTRY['transfer_binary']}")
    logger.info(f"    跨被试三分类: {RUN_REGISTRY['cross_cbramod_ternary']}")
    logger.info(f"    迁移三分类: {RUN_REGISTRY['transfer_ternary']}")

    return results


def section_3_5():
    """Section 3.5: Extra sessions — descriptive stats and subgroup analysis.

    Extracts min/max ranges (replacing normal distribution assumption),
    and subgroup sample sizes for low/high baseline.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Section 3.5: Extra Sessions descriptive statistics")
    logger.info("=" * 60)

    cache = load_json("extra_sessions_binary")

    for model in ["cbramod", "eegnet"]:
        logger.info(f"\n--- {model.upper()} ---")

        for step in ["baseline", "sess03", "sess04", "sess05"]:
            accs = extract_per_subject_accs(cache, model, step=step)
            if accs:
                step_label = "Baseline" if step == "baseline" else f"+{step.title()}"
                print_describe(describe_accs(accs, f"{model} {step_label}"))

        # Subgroup analysis: low vs high baseline
        baseline_accs = extract_per_subject_accs(cache, model, step="baseline")
        sess05_accs = extract_per_subject_accs(cache, model, step="sess05")

        if baseline_accs and sess05_accs:
            common = sorted(set(baseline_accs.keys()) & set(sess05_accs.keys()))
            low = [s for s in common if baseline_accs[s] < 80]
            high = [s for s in common if baseline_accs[s] > 90]
            mid = [s for s in common if 80 <= baseline_accs[s] <= 90]

            logger.info(f"\n  Subgroup analysis ({model}):")
            logger.info(f"    Low baseline (<80%): N={len(low)}, subjects={low}")
            if low:
                deltas = [sess05_accs[s] - baseline_accs[s] for s in low]
                logger.info(f"      Mean Δ: {np.mean(deltas):+.2f} pp, range: [{min(deltas):+.2f}, {max(deltas):+.2f}]")
            logger.info(f"    Mid baseline (80-90%): N={len(mid)}, subjects={mid}")
            if mid:
                deltas = [sess05_accs[s] - baseline_accs[s] for s in mid]
                logger.info(f"      Mean Δ: {np.mean(deltas):+.2f} pp, range: [{min(deltas):+.2f}, {max(deltas):+.2f}]")
            logger.info(f"    High baseline (>90%): N={len(high)}, subjects={high}")
            if high:
                deltas = [sess05_accs[s] - baseline_accs[s] for s in high]
                logger.info(f"      Mean Δ: {np.mean(deltas):+.2f} pp, range: [{min(deltas):+.2f}, {max(deltas):+.2f}]")

            # Paired t-test: baseline vs sess05
            logger.info(f"\n  Paired t-test baseline → +Sess05:")
            result = paired_ttest(baseline_accs, sess05_accs, f"{model} baseline → +sess05")
            print_ttest(result)

    logger.info(f"\n  数据来源: {RUN_REGISTRY['extra_sessions_binary']}")


def section_3_5_ternary():
    """Section 3.5.2: Extra sessions ternary descriptive stats."""
    logger.info("\n" + "=" * 60)
    logger.info("Section 3.5.2: Extra Sessions Ternary")
    logger.info("=" * 60)

    cache = load_json("extra_sessions_ternary")

    for model in ["cbramod", "eegnet"]:
        logger.info(f"\n--- {model.upper()} ---")
        for step in ["baseline", "sess03", "sess04", "sess05"]:
            accs = extract_per_subject_accs(cache, model, step=step)
            if accs:
                step_label = "Baseline" if step == "baseline" else f"+{step.title()}"
                print_describe(describe_accs(accs, f"{model} {step_label}"))

    logger.info(f"\n  数据来源: {RUN_REGISTRY['extra_sessions_ternary']}")


def section_3_5_4():
    """Section 3.5.4: Extra sessions across within/cross/transfer paradigms."""
    logger.info("\n" + "=" * 60)
    logger.info("Section 3.5.4: Extra Sessions across training paradigms")
    logger.info("=" * 60)

    configs = [
        ("被试内", "extra_sessions_binary", "cbramod"),
        ("跨被试（21-subj 训练）", "extra_sessions_cross_binary", "cbramod"),
        ("Transfer-init", "extra_sessions_transfer_binary", "cbramod"),
    ]

    step_series_by_label = {}
    endpoint_stats = {}

    for label, cache_key, model in configs:
        cache = load_json(cache_key)
        step_series = extract_extra_session_step_series(cache, model)
        step_series_by_label[label] = step_series
        endpoint_stats[label] = paired_ttest(
            step_series["baseline"],
            step_series["sess05"],
            f"{label} baseline → +Sess05",
        )

        logger.info(f"\n{label}:")
        for step in EXTRA_SESSION_STEPS:
            print_describe(
                describe_accs(
                    step_series[step],
                    f"{label} {EXTRA_SESSION_STEP_LABELS[step]}",
                )
            )
        print_ttest(endpoint_stats[label])

    logger.info("")
    logger.info(
        "**表 15. Extra sessions 在三种训练范式下的轨迹对比（CBraMod 二分类，N = 16）。**"
    )
    logger.info("")
    logger.info("| 阶段 | 被试内 | 跨被试（21-subj 训练） | Transfer-init |")
    logger.info("|------|--------|------------------------|---------------|")
    for step in EXTRA_SESSION_STEPS:
        row_label = EXTRA_SESSION_STEP_LABELS[step]
        row_values = [
            format_mean_std(step_series_by_label[label][step])
            for label, _, _ in configs
        ]
        logger.info(f"| {row_label} | {' | '.join(row_values)} |")

    delta_row = [
        f"{endpoint_stats[label]['delta_pp']:+.2f} pp"
        for label, _, _ in configs
    ]
    p_row = [
        format_p_value(endpoint_stats[label]["p_value"])
        for label, _, _ in configs
    ]
    logger.info(f"| Δ(BL→S05) | {' | '.join(delta_row)} |")
    logger.info(f"| paired p | {' | '.join(p_row)} |")
    logger.info("")
    logger.info(
        "> **数据来源**: "
        f"within-subject `20260324_2131`: `{RUN_REGISTRY['extra_sessions_binary']}`; "
        f"cross-subject `20260326_1409`: `{RUN_REGISTRY['extra_sessions_cross_binary']}`; "
        f"transfer-init `20260329_1357`: `{RUN_REGISTRY['extra_sessions_transfer_binary']}`"
    )

    return {
        "step_series_by_label": step_series_by_label,
        "endpoint_stats": endpoint_stats,
    }


def section_3_5_5():
    """Section 3.5.5: Cross-subject extra-session summary table."""
    logger.info("\n" + "=" * 60)
    logger.info("Section 3.5.5: Cross-subject extra-session summary")
    logger.info("=" * 60)

    row_configs = [
        ("CBraMod Binary", "extra_sessions_cross_binary", "cbramod"),
        ("EEGNet Binary", "extra_sessions_cross_binary", "eegnet"),
        ("CBraMod Ternary", "extra_sessions_cross_ternary", "cbramod"),
    ]

    rows = []
    for label, cache_key, model in row_configs:
        cache = load_json(cache_key)
        step_series = extract_extra_session_step_series(cache, model)
        result = paired_ttest(
            step_series["baseline"],
            step_series["sess05"],
            f"{label} baseline → +Sess05",
        )
        rows.append({
            "label": label,
            "baseline": format_mean_std(step_series["baseline"]),
            "sess05": format_mean_std(step_series["sess05"]),
            "delta": f"{result['delta_pp']:+.2f} pp",
            "p_value": format_p_value(result["p_value"]),
        })

        logger.info(f"\n{label}:")
        print_describe(describe_accs(step_series["baseline"], f"{label} Baseline"))
        print_describe(describe_accs(step_series["sess05"], f"{label} +Sess05"))
        print_ttest(result)

    logger.info("")
    logger.info("**表 15b. Cross-subject extra sessions 的边际收益摘要（N = 16）。**")
    logger.info("")
    logger.info("| 模型 / 任务 | Baseline | +Sess05 | Δ | paired p |")
    logger.info("|-------------|----------|---------|---|----------|")
    for row in rows:
        logger.info(
            f"| {row['label']} | {row['baseline']} | {row['sess05']} | "
            f"{row['delta']} | {row['p_value']} |"
        )
    logger.info("")
    logger.info(
        "> **数据来源**: "
        f"binary `20260326_1409`: `{RUN_REGISTRY['extra_sessions_cross_binary']}`; "
        f"ternary `20260327_0303`: `{RUN_REGISTRY['extra_sessions_cross_ternary']}`"
    )

    return rows


def supplementary_s3():
    """Table S3: Extra sessions per-subject data (binary, both models).

    Outputs markdown table for direct inclusion in paper.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Table S3: Extra Sessions per-subject (binary)")
    logger.info("=" * 60)

    cache = load_json("extra_sessions_binary")

    for model in ["cbramod", "eegnet"]:
        logger.info(f"\n### {model.upper()}")
        logger.info("")

        # Collect data
        subjects_data = {}
        results = cache.get("results", {}).get(model, {})
        for subj in sorted(results.keys()):
            if not subj.startswith("S"):
                continue
            sd = results[subj]
            if not isinstance(sd, dict):
                continue
            row = {step: None for step in EXTRA_SESSION_STEPS}
            for step in EXTRA_SESSION_STEPS:
                step_data = sd.get(step, {})
                if isinstance(step_data, dict):
                    acc = step_data.get("test_acc_majority")
                    row[step] = acc * 100 if acc is not None else None
            subjects_data[subj] = row

        # Print markdown table
        logger.info("| 被试 | Baseline | +Sess03 | +Sess04 | +Sess05 | Δ (总变化) |")
        logger.info("|------|---------|---------|---------|---------|-----------|")
        for subj in sorted(subjects_data.keys()):
            row = subjects_data[subj]
            # Calculate delta
            bl = row["baseline"]
            last_step = "sess05"
            while row.get(last_step) is None and last_step != "baseline":
                last_step = {
                    "sess05": "sess04",
                    "sess04": "sess03",
                    "sess03": "baseline",
                }[last_step]
            if bl is not None and last_step != "baseline" and row.get(last_step) is not None:
                delta = f"{row[last_step] - bl:+.2f} pp"
            else:
                delta = "—"

            formatted = {
                step: f"{row[step]:.2f}%" if row[step] is not None else "—"
                for step in EXTRA_SESSION_STEPS
            }
            logger.info(
                f"| {subj} | {formatted['baseline']} | {formatted['sess03']} | "
                f"{formatted['sess04']} | {formatted['sess05']} | {delta} |"
            )

    logger.info(f"\n> **数据来源**: `{RUN_REGISTRY['extra_sessions_binary']}`")


# =============================================================================
# Main
# =============================================================================

SECTIONS = {
    "3.2": section_3_2,
    "3.4": section_3_4,
    "3.5": section_3_5,
    "3.5.2": section_3_5_ternary,
    "3.5.4": section_3_5_4,
    "3.5.5": section_3_5_5,
    "s3": supplementary_s3,
}


def main():
    parser = argparse.ArgumentParser(
        description="论文 v3 统计检验与数据提取"
    )
    parser.add_argument(
        "--section",
        default="all",
        help="Section to compute: 3.2, 3.4, 3.5, 3.5.2, 3.5.4, 3.5.5, s3, or 'all'",
    )
    args = parser.parse_args()

    logger.info("论文 v3 统计检验与数据提取")
    logger.info(f"项目根目录: {PROJECT_ROOT}")
    logger.info("")

    # Verify all registry files exist
    missing = [k for k, v in RUN_REGISTRY.items() if not (PROJECT_ROOT / v).exists()]
    if missing:
        logger.warning(f"WARNING: 以下注册的结果文件不存在: {missing}")

    if args.section == "all":
        for name, func in SECTIONS.items():
            func()
    elif args.section in SECTIONS:
        SECTIONS[args.section]()
    else:
        logger.error(f"Unknown section: {args.section}")
        logger.info(f"Available: {', '.join(SECTIONS.keys())}, all")
        sys.exit(1)

    logger.info("\n" + "=" * 60)
    logger.info("完成. 所有统计量可通过修改 RUN_REGISTRY 重算.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
