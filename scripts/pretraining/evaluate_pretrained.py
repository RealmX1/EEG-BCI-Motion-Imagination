#!/usr/bin/env python3
"""
Further Pre-trained 权重下游评估脚本
=====================================
在 finger BCI cross-subject binary 任务上对比：
  - Baseline: 原始 TUEG 权重 → fine-tune
  - Ours: MI further-pretrained 权重 → fine-tune

通过临时替换预训练权重文件来复用现有 cross-subject 训练管线。

用法：
  uv run python scripts/pretraining/evaluate_pretrained.py --further-pretrained checkpoints/cbramod/further_pretrain_YYYYMMDD_HHMM/best_model.pth
  uv run python scripts/pretraining/evaluate_pretrained.py --further-pretrained checkpoints/cbramod/further_pretrain_YYYYMMDD_HHMM/best_model.pth --skip-baseline
"""

import os
import sys
import json
import shutil
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_default_baseline_weights() -> str | None:
    """查找原始 TUEG 预训练权重路径。"""
    candidates = [
        PROJECT_ROOT / "checkpoints" / "cbramod" / "pretrained_weights.pth",
        PROJECT_ROOT.parent / "CBraMod" / "pretrained_weights" / "pretrained_weights.pth",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def run_cross_subject_with_weights(
    pretrained_path: str,
    label: str,
    checkpoint_suffix: str,
) -> dict:
    """
    运行 cross-subject 评估。

    策略：将指定权重临时复制到标准位置，运行现有脚本后恢复。
    """
    logger.info(f"{'='*60}")
    logger.info(f"评估: {label}")
    logger.info(f"权重: {pretrained_path}")
    logger.info(f"{'='*60}")

    standard_path = PROJECT_ROOT / "checkpoints" / "cbramod" / "pretrained_weights.pth"
    backup_path = standard_path.with_suffix(".pth.bak")

    # 检测源和目标是否为同一文件（baseline 评估时无需替换）
    same_file = (
        standard_path.exists()
        and Path(pretrained_path).resolve() == standard_path.resolve()
    )

    start_time = datetime.now()

    try:
        if same_file:
            logger.info("权重已在标准位置，跳过替换")
        else:
            # 备份原始权重
            if standard_path.exists():
                shutil.copy2(str(standard_path), str(backup_path))
                logger.info(f"已备份原始权重到: {backup_path}")

            # 替换为评估权重
            shutil.copy2(pretrained_path, str(standard_path))
            logger.info(f"已替换权重: {pretrained_path} → {standard_path}")

        # 运行 cross-subject 脚本
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_cross_subject_comparison.py"),
            "--models", "cbramod",
            "--paradigm", "imagery",
            "--task", "binary",
            "--output-dir", str(
                PROJECT_ROOT / "checkpoints" / "cross_subject" / f"eval_{checkpoint_suffix}"
            ),
            "--force-retrain",
            "--no-wandb",
        ]
        logger.info(f"执行: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=7200,  # 2 小时超时
        )

        if result.returncode != 0:
            logger.error(f"脚本失败:\n{result.stderr[-2000:]}")
            return {
                "label": label,
                "pretrained_path": pretrained_path,
                "error": result.stderr[-500:],
            }

        # 从最新结果文件读取结果（只找本次运行后生成的）
        result_data = _find_latest_result(after=start_time)
        if result_data:
            result_data["label"] = label
            result_data["pretrained_path"] = pretrained_path
            return result_data
        else:
            return {
                "label": label,
                "pretrained_path": pretrained_path,
                "stdout_tail": result.stdout[-1000:],
                "status": "completed_but_no_result_file",
            }

    finally:
        # 恢复原始权重（os.replace 在 Windows 上可原子覆盖已存在文件）
        if backup_path.exists():
            os.replace(str(backup_path), str(standard_path))
            logger.info("已恢复原始权重")


def _find_latest_result(after: datetime | None = None) -> dict | None:
    """从 results/ 目录找到最新的 cross-subject 结果。"""
    results_dir = PROJECT_ROOT / "results"
    # 查找最近修改的 JSON 缓存文件
    json_files = sorted(
        results_dir.rglob("*cross*imagery_binary.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    # 过滤掉本次运行之前的旧文件
    if after is not None:
        after_ts = after.timestamp()
        json_files = [f for f in json_files if f.stat().st_mtime >= after_ts]
    if json_files:
        latest = json_files[0]
        logger.info(f"找到最新结果: {latest}")
        with open(latest, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Further Pre-trained 权重下游评估",
    )
    parser.add_argument(
        "--further-pretrained",
        type=str,
        required=True,
        help="Further pre-trained 权重路径",
    )
    parser.add_argument(
        "--baseline-weights",
        type=str,
        default=None,
        help="Baseline (TUEG) 权重路径（默认自动搜索）",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="跳过 baseline 评估（如已有结果）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="结果 JSON 输出路径",
    )
    args = parser.parse_args()

    # 检查权重文件
    further_path = Path(args.further_pretrained)
    if not further_path.exists():
        raise FileNotFoundError(f"Further pre-trained 权重不存在: {further_path}")

    baseline_path = args.baseline_weights or find_default_baseline_weights()
    if baseline_path is None and not args.skip_baseline:
        raise FileNotFoundError("找不到 baseline 权重，请用 --baseline-weights 指定")

    results = {}

    # 1. Baseline 评估（使用原始权重，无需替换）
    if not args.skip_baseline and baseline_path:
        results["baseline"] = run_cross_subject_with_weights(
            baseline_path, "Baseline (TUEG)", "baseline"
        )

    # 2. Further pre-trained 评估
    results["further_pretrained"] = run_cross_subject_with_weights(
        str(further_path), "Further Pre-trained (MI)", "further_pretrained"
    )

    # 3. 汇总对比
    print("\n" + "=" * 70)
    print("下游评估结果对比")
    print("=" * 70)
    print(f"{'配置':<30} {'权重来源'}")
    print("-" * 70)

    for key, res in results.items():
        label = res.get("label", key)
        if "error" in res:
            print(f"{label:<30} ERROR: {res['error'][:50]}")
        else:
            print(f"{label:<30} {res.get('pretrained_path', 'N/A')}")

    print("=" * 70)
    print("详细数值请查看各自的 JSON 结果文件。")

    # 保存汇总
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = PROJECT_ROOT / "results" / "pretraining"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_path = output_dir / f"{timestamp}_evaluation_comparison.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"评估结果已保存: {output_path}")


if __name__ == "__main__":
    main()
