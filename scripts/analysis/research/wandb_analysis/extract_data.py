"""
从 WandB 提取 S10/S20 及对照被试的训练历史数据。

Usage:
    uv run python scripts/research/wandb_analysis/extract_data.py \
        --project eeg-bci \
        --output analysis_data/

功能:
1. 连接 WandB API
2. 筛选目标 runs (S10, S20, S09, S19 的 EEGNet 和 CBraMod)
3. 提取完整 history 和 summary metrics
4. 保存为本地 JSON 供后续分析
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class SubjectRunData:
    """单个运行的分析数据容器"""

    # 基本信息
    run_id: str
    run_name: str
    subject_id: str
    model_type: str  # 'eegnet' or 'cbramod'
    task: str  # 'binary' or 'ternary'
    paradigm: str  # 'imagery' or 'movement'
    created_at: str

    # 配置
    config: Dict[str, Any] = field(default_factory=dict)

    # Summary metrics
    best_val_accuracy: float = 0.0
    test_accuracy: float = 0.0
    test_majority_accuracy: float = 0.0
    best_epoch: int = 0
    total_epochs: int = 0

    # Training history (per epoch)
    train_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    val_majority_acc: List[float] = field(default_factory=list)
    learning_rate: List[float] = field(default_factory=list)

    # 计算指标
    final_train_val_gap: float = 0.0
    val_loss_trend: str = "unknown"
    convergence_epoch: int = 0


class WandBDataExtractor:
    """WandB 数据提取器"""

    def __init__(self, entity: Optional[str] = None, project: str = "eeg-bci"):
        try:
            import wandb
        except ImportError:
            raise ImportError("wandb not installed. Run: pip install wandb")

        self.wandb = wandb
        self.api = wandb.Api()
        self.entity = entity
        self.project = project

    def get_runs(
        self,
        subject_ids: Optional[List[str]] = None,
        model_types: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
        paradigm: str = "imagery",
        limit: int = 500,
    ) -> List:
        """获取符合条件的运行"""
        path = f"{self.entity}/{self.project}" if self.entity else self.project

        # 基础过滤器
        filters = {"state": "finished"}

        logger.info(f"Querying runs from {path} with filters: {filters}")

        try:
            all_runs = list(self.api.runs(path=path, filters=filters, per_page=limit))
        except Exception as e:
            logger.error(f"Failed to fetch runs: {e}")
            return []

        logger.info(f"Found {len(all_runs)} total finished runs")

        # 手动过滤
        filtered_runs = []
        for run in all_runs:
            config = run.config

            # 提取配置值
            run_subject = config.get("subject_id", "")
            run_model = config.get("model_type", "")
            run_task = config.get("task", "")
            run_paradigm = config.get("paradigm", "")

            # 应用过滤
            if subject_ids and run_subject not in subject_ids:
                continue
            if model_types and run_model not in model_types:
                continue
            if tasks and run_task not in tasks:
                continue
            if paradigm and run_paradigm != paradigm:
                continue

            filtered_runs.append(run)

        logger.info(f"Filtered to {len(filtered_runs)} matching runs")
        return filtered_runs

    def extract_run_data(self, run) -> SubjectRunData:
        """提取单个 run 的完整数据"""
        config = dict(run.config)
        summary = dict(run.summary)

        # 获取 history
        try:
            history_df = run.history(samples=1000)
        except Exception as e:
            logger.warning(f"Failed to get history for {run.name}: {e}")
            history_df = None

        # 提取各指标序列
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        val_majority_acc = []
        learning_rate = []

        if history_df is not None and len(history_df) > 0:
            # 尝试多种可能的列名
            for col, target_list in [
                (["train/loss", "train_loss"], train_loss),
                (["train/accuracy", "train_acc"], train_acc),
                (["val/loss", "val_loss"], val_loss),
                (["val/accuracy", "val_acc"], val_acc),
                (["val/majority_accuracy", "val_majority_acc"], val_majority_acc),
                (["train/learning_rate", "learning_rate", "lr"], learning_rate),
            ]:
                for c in col:
                    if c in history_df.columns:
                        values = history_df[c].dropna().tolist()
                        target_list.extend(values)
                        break

        # 计算诊断指标
        final_train_val_gap = self._calc_train_val_gap(train_acc, val_acc)
        val_loss_trend = self._analyze_trend(val_loss)
        convergence_epoch = self._find_convergence_epoch(val_loss)

        return SubjectRunData(
            run_id=run.id,
            run_name=run.name,
            subject_id=config.get("subject_id", ""),
            model_type=config.get("model_type", ""),
            task=config.get("task", ""),
            paradigm=config.get("paradigm", ""),
            created_at=run.created_at,
            config=config,
            best_val_accuracy=summary.get("best_val_accuracy", 0),
            test_accuracy=summary.get("test/accuracy", summary.get("test_accuracy", 0)),
            test_majority_accuracy=summary.get(
                "test/majority_accuracy", summary.get("test_majority_accuracy", 0)
            ),
            best_epoch=summary.get("best_epoch", 0),
            total_epochs=len(train_loss),
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            val_majority_acc=val_majority_acc,
            learning_rate=learning_rate,
            final_train_val_gap=final_train_val_gap,
            val_loss_trend=val_loss_trend,
            convergence_epoch=convergence_epoch,
        )

    def _calc_train_val_gap(self, train_acc: List[float], val_acc: List[float]) -> float:
        """计算最终 train-val accuracy gap (过拟合指标)"""
        if not train_acc or not val_acc:
            return 0.0
        n = min(3, len(train_acc), len(val_acc))
        train_final = sum(train_acc[-n:]) / n
        val_final = sum(val_acc[-n:]) / n
        return train_final - val_final

    def _analyze_trend(self, values: List[float], window: int = 5) -> str:
        """分析序列趋势"""
        if len(values) < window * 2:
            return "unknown"

        first_half = sum(values[:window]) / window
        second_half = sum(values[-window:]) / window

        if first_half == 0:
            return "unknown"

        ratio = second_half / first_half

        if ratio < 0.9:
            return "decreasing"
        elif ratio > 1.1:
            return "increasing"
        else:
            return "stable"

    def _find_convergence_epoch(self, val_loss: List[float], threshold: float = 0.01) -> int:
        """找到验证 loss 收敛的 epoch"""
        if len(val_loss) < 3:
            return len(val_loss)

        for i in range(2, len(val_loss)):
            recent = val_loss[max(0, i - 3) : i + 1]
            min_val = min(recent)
            if min_val > 0 and max(recent) - min_val < threshold * min_val:
                return i

        return len(val_loss)

    def extract_all(
        self,
        target_subjects: List[str],
        control_subjects: List[str],
        model_types: List[str] = ["eegnet", "cbramod"],
        tasks: List[str] = ["binary", "ternary"],
        output_dir: Optional[Path] = None,
    ) -> Dict[str, List[SubjectRunData]]:
        """提取所有目标和对照被试数据"""

        all_subjects = target_subjects + control_subjects
        results: Dict[str, List[SubjectRunData]] = {}

        # 获取所有相关运行
        runs = self.get_runs(
            subject_ids=all_subjects,
            model_types=model_types,
            tasks=tasks,
        )

        logger.info(f"Processing {len(runs)} runs...")

        for run in runs:
            try:
                data = self.extract_run_data(run)
                key = f"{data.subject_id}_{data.model_type}_{data.task}"

                if key not in results:
                    results[key] = []
                results[key].append(data)

                logger.info(
                    f"  Extracted: {data.run_name} "
                    f"(test_acc={data.test_accuracy:.2%}, epochs={data.total_epochs})"
                )
            except Exception as e:
                logger.warning(f"Failed to extract {run.name}: {e}")

        # 按时间排序，最新的在前
        for key in results:
            results[key].sort(key=lambda x: x.created_at, reverse=True)

        # 保存数据
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "wandb_data.json"

            # 转换为可序列化格式
            serializable = {}
            for key, runs_list in results.items():
                serializable[key] = [asdict(r) for r in runs_list]

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(results)} run groups to {output_path}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Extract WandB training data for analysis")
    parser.add_argument("--project", default="eeg-bci", help="WandB project name")
    parser.add_argument("--entity", default=None, help="WandB entity (team/username)")
    parser.add_argument("--output", default="analysis_data", help="Output directory")
    parser.add_argument(
        "--target-subjects",
        nargs="+",
        default=["S10", "S20"],
        help="Target subjects with performance issues",
    )
    parser.add_argument(
        "--control-subjects",
        nargs="+",
        default=["S09", "S19", "S03", "S05"],
        help="Control subjects with good performance",
    )
    args = parser.parse_args()

    extractor = WandBDataExtractor(entity=args.entity, project=args.project)

    results = extractor.extract_all(
        target_subjects=args.target_subjects,
        control_subjects=args.control_subjects,
        output_dir=Path(args.output),
    )

    # 打印摘要
    print("\n" + "=" * 60)
    print("  提取结果摘要")
    print("=" * 60)

    for key, runs_list in sorted(results.items()):
        if runs_list:
            latest = runs_list[0]
            print(
                f"\n{key}:"
                f"\n  最新运行: {latest.run_name}"
                f"\n  测试准确率: {latest.test_accuracy:.2%}"
                f"\n  Majority 准确率: {latest.test_majority_accuracy:.2%}"
                f"\n  训练 epochs: {latest.total_epochs}"
                f"\n  历史运行数: {len(runs_list)}"
            )


if __name__ == "__main__":
    main()
