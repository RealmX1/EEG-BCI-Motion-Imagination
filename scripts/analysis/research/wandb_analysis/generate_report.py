#!/usr/bin/env python
"""
S10/S20 CBraMod 性能问题分析报告生成器。

Usage:
    # 完整流程 (提取数据 + 生成报告)
    uv run python scripts/research/wandb_analysis/generate_report.py

    # 仅生成报告 (使用已有数据)
    uv run python scripts/research/wandb_analysis/generate_report.py \
        --data analysis_data/wandb_data.json \
        --skip-extract

输出:
- reports/s10_s20_analysis/analysis_report.md
- reports/s10_s20_analysis/figures/*.png
- reports/s10_s20_analysis/diagnostic_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加项目根目录到 path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.research.wandb_analysis.diagnostic_metrics import (
    DiagnosticCalculator,
    DiagnosticMetrics,
    SubjectDiagnosticSummary,
)
from scripts.research.wandb_analysis.extract_data import SubjectRunData, WandBDataExtractor
from scripts.research.wandb_analysis.training_dynamics import (
    ModelComparisonReport,
    TrainingDynamicsAnalyzer,
    TrainingDynamicsReport,
)
from scripts.research.wandb_analysis.visualize import AnalysisVisualizer


class ReportGenerator:
    """分析报告生成器"""

    def __init__(
        self,
        output_dir: Path,
        target_subjects: List[str] = ["S10", "S20"],
        control_subjects: List[str] = ["S09", "S19", "S03", "S05"],
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.target_subjects = target_subjects
        self.control_subjects = control_subjects

        self.dynamics_analyzer = TrainingDynamicsAnalyzer()
        self.diagnostic_calc = DiagnosticCalculator()
        self.visualizer = AnalysisVisualizer(output_dir / "figures")

    def load_data(self, data_path: Path) -> Dict[str, List[SubjectRunData]]:
        """从 JSON 加载数据"""
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # 转换为 dataclass
        results = {}
        for key, runs_list in raw_data.items():
            results[key] = [SubjectRunData(**r) for r in runs_list]

        return results

    def generate(
        self,
        subjects_data: Dict[str, List[SubjectRunData]],
        tasks: List[str] = ["binary", "ternary"],
    ) -> Path:
        """生成完整分析报告"""
        print("\n" + "=" * 60)
        print("  开始生成分析报告")
        print("=" * 60)

        # 1. 运行分析
        dynamics_reports: Dict[str, TrainingDynamicsReport] = {}
        diagnostic_reports: Dict[str, DiagnosticMetrics] = {}
        comparison_reports: Dict[str, ModelComparisonReport] = {}
        subject_summaries: Dict[str, SubjectDiagnosticSummary] = {}

        for key, runs in subjects_data.items():
            if not runs:
                continue
            data = runs[0]  # 最新的运行

            # 训练动态分析
            try:
                dynamics_reports[key] = self.dynamics_analyzer.analyze(data)
            except Exception as e:
                print(f"  Warning: Failed to analyze dynamics for {key}: {e}")

            # 诊断指标
            try:
                diagnostic_reports[key] = self.diagnostic_calc.calculate(data)
            except Exception as e:
                print(f"  Warning: Failed to calculate diagnostics for {key}: {e}")

        # 模型对比分析
        for subject_id in self.target_subjects + self.control_subjects:
            for task in tasks:
                eegnet_key = f"{subject_id}_eegnet_{task}"
                cbramod_key = f"{subject_id}_cbramod_{task}"

                eegnet_data = subjects_data.get(eegnet_key, [None])[0] if eegnet_key in subjects_data else None
                cbramod_data = subjects_data.get(cbramod_key, [None])[0] if cbramod_key in subjects_data else None

                if eegnet_data and cbramod_data:
                    try:
                        comp_key = f"{subject_id}_{task}"
                        comparison_reports[comp_key] = self.dynamics_analyzer.compare_models(
                            eegnet_data, cbramod_data
                        )
                    except Exception as e:
                        print(f"  Warning: Failed to compare models for {subject_id} {task}: {e}")

                # 被试总结
                try:
                    summary_key = f"{subject_id}_{task}"
                    subject_summaries[summary_key] = self.diagnostic_calc.create_subject_summary(
                        subject_id, task, eegnet_data, cbramod_data
                    )
                except Exception as e:
                    print(f"  Warning: Failed to create summary for {subject_id} {task}: {e}")

        print(f"  分析完成: {len(dynamics_reports)} dynamics, {len(diagnostic_reports)} diagnostics")

        # 2. 生成可视化
        print("\n生成可视化图表...")
        for task in tasks:
            try:
                # Loss 曲线对比 - CBraMod
                self.visualizer.plot_loss_comparison(
                    subjects_data,
                    task=task,
                    problem_subjects=self.target_subjects,
                    control_subjects=self.control_subjects,
                    model_type="cbramod",
                )

                # Loss 曲线对比 - EEGNet
                self.visualizer.plot_loss_comparison(
                    subjects_data,
                    task=task,
                    problem_subjects=self.target_subjects,
                    control_subjects=self.control_subjects,
                    model_type="eegnet",
                )

                # 性能概览
                self.visualizer.plot_performance_overview(
                    subjects_data,
                    task=task,
                    highlight_subjects=self.target_subjects,
                )
            except Exception as e:
                print(f"  Warning: Failed to generate overview for {task}: {e}")

        # 模型对比图 (仅问题被试)
        for subject_id in self.target_subjects:
            for task in tasks:
                try:
                    self.visualizer.plot_model_comparison(subjects_data, subject_id, task)
                except Exception as e:
                    print(f"  Warning: Failed to generate model comparison for {subject_id} {task}: {e}")

        # 3. 生成 Markdown 报告
        print("\n生成 Markdown 报告...")
        report_path = self._generate_markdown(
            subjects_data,
            dynamics_reports,
            diagnostic_reports,
            comparison_reports,
            subject_summaries,
            tasks,
        )

        # 4. 保存 JSON 摘要
        self._save_summary_json(diagnostic_reports, comparison_reports, subject_summaries)

        print(f"\n报告生成完成: {report_path}")
        return report_path

    def _generate_markdown(
        self,
        subjects_data: Dict[str, List[SubjectRunData]],
        dynamics: Dict[str, TrainingDynamicsReport],
        diagnostics: Dict[str, DiagnosticMetrics],
        comparisons: Dict[str, ModelComparisonReport],
        summaries: Dict[str, SubjectDiagnosticSummary],
        tasks: List[str],
    ) -> Path:
        """生成 Markdown 报告"""
        lines = [
            "# S10/S20 CBraMod 性能问题分析报告",
            "",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
            "## 1. 执行摘要",
            "",
            "### 1.1 问题概述",
            "",
            "CBraMod 在 S10 和 S20 被试上的表现持续较差：",
            "",
            "| 被试 | 任务 | EEGNet | CBraMod | 差异 | 问题类型 |",
            "|------|------|--------|---------|------|----------|",
        ]

        # 添加性能对比表格
        for subject_id in self.target_subjects:
            for task in tasks:
                summary_key = f"{subject_id}_{task}"
                if summary_key in summaries:
                    s = summaries[summary_key]
                    diff = s.accuracy_gap
                    diff_str = f"+{diff:.1%}" if diff > 0 else f"{diff:.1%}"
                    lines.append(
                        f"| {subject_id} | {task} | {s.eegnet_accuracy:.1%} | "
                        f"{s.cbramod_accuracy:.1%} | {diff_str} | {s.problem_type} |"
                    )

        lines.extend(
            [
                "",
                "### 1.2 关键发现",
                "",
            ]
        )

        # S10 发现
        if "S10_binary" in summaries:
            s = summaries["S10_binary"]
            lines.extend(
                [
                    "**S10 - CBraMod 特有问题:**",
                    f"- EEGNet ({s.eegnet_accuracy:.1%}) 显著优于 CBraMod ({s.cbramod_accuracy:.1%})",
                    "- 说明该被试数据本身是可学习的，问题在于 CBraMod 的适配",
                    "- 可能原因：CBraMod 的预处理/频带特征与该被试不匹配",
                    "",
                ]
            )

        # S20 发现
        if "S20_binary" in summaries:
            s = summaries["S20_binary"]
            lines.extend(
                [
                    "**S20 - 两模型均表现不佳:**",
                    f"- CBraMod ({s.cbramod_accuracy:.1%}) 略优于 EEGNet ({s.eegnet_accuracy:.1%})",
                    "- 两个模型都低于正常水平",
                    "- 最近的调度器更改后 CBraMod 性能有所提升",
                    "",
                ]
            )

        # 详细分析部分
        lines.extend(
            [
                "---",
                "",
                "## 2. 详细分析",
                "",
            ]
        )

        for subject_id in self.target_subjects:
            lines.extend(
                [
                    f"### 2.{self.target_subjects.index(subject_id) + 1} {subject_id} 分析",
                    "",
                ]
            )

            for task in tasks:
                summary_key = f"{subject_id}_{task}"
                comp_key = f"{subject_id}_{task}"

                if summary_key in summaries:
                    s = summaries[summary_key]
                    lines.extend(
                        [
                            f"#### {task.title()} 任务",
                            "",
                        ]
                    )

                    # 性能对比
                    lines.extend(
                        [
                            "**性能对比:**",
                            f"- EEGNet: {s.eegnet_accuracy:.1%}",
                            f"- CBraMod: {s.cbramod_accuracy:.1%}",
                            f"- 问题类型: `{s.problem_type}`",
                            "",
                        ]
                    )

                    # 诊断详情
                    if s.cbramod_diagnosis:
                        d = s.cbramod_diagnosis
                        lines.extend(
                            [
                                "**CBraMod 诊断:**",
                                f"- 主要问题: {d.primary_issue}",
                                f"- 数据质量评分: {d.data_quality_score:.2f}",
                                f"- 泛化 Gap: {d.generalization_gap:.2%}",
                                f"- 训练稳定性: {d.training_stability:.2f}",
                                "",
                            ]
                        )

                    # 建议
                    if s.overall_recommendations:
                        lines.append("**建议:**")
                        for rec in s.overall_recommendations:
                            if rec:
                                lines.append(f"- {rec}")
                        lines.append("")

                    # 对比分析
                    if comp_key in comparisons:
                        comp = comparisons[comp_key]
                        lines.extend(
                            [
                                "**模型对比分析:**",
                                f"- {comp.diagnosis}",
                                "",
                            ]
                        )

        # 可视化部分
        lines.extend(
            [
                "---",
                "",
                "## 3. 可视化",
                "",
            ]
        )

        for task in tasks:
            lines.extend(
                [
                    f"### 3.{tasks.index(task) + 1} {task.title()} 任务",
                    "",
                    f"#### CBraMod Loss 曲线对比",
                    f"![CBraMod Loss](figures/loss_comparison_cbramod_{task}.png)",
                    "",
                    f"#### EEGNet Loss 曲线对比",
                    f"![EEGNet Loss](figures/loss_comparison_eegnet_{task}.png)",
                    "",
                    f"#### 性能概览",
                    f"![Performance Overview](figures/performance_overview_{task}.png)",
                    "",
                ]
            )

        for subject_id in self.target_subjects:
            for task in tasks:
                lines.extend(
                    [
                        f"#### {subject_id} 模型对比 ({task})",
                        f"![{subject_id} Comparison](figures/model_comparison_{subject_id}_{task}.png)",
                        "",
                    ]
                )

        # 改进建议部分
        lines.extend(
            [
                "---",
                "",
                "## 4. 改进建议",
                "",
                "### 4.1 针对 S10 的建议 (CBraMod 特有问题)",
                "",
                "1. **检查预处理配置:**",
                "   - 对比 S10 的 EEG 频带特征与其他被试的差异",
                "   - CBraMod 使用 0.3-75Hz 滤波，可能不适合 S10",
                "",
                "2. **调整模型策略:**",
                "   - 尝试冻结 CBraMod backbone，只微调分类器头",
                "   - 增加正则化 (dropout 从 0.1 提升到 0.3)",
                "",
                "3. **数据适配:**",
                "   - 检查 S10 的数据分布是否与预训练数据差异大",
                "   - 考虑对 S10 使用特定的归一化参数",
                "",
                "4. **模型选择:**",
                "   - 对于 S10，EEGNet 可能是更好的选择",
                "",
                "### 4.2 针对 S20 的建议 (两模型均较差)",
                "",
                "1. **数据质量检查:**",
                "   - 检查原始 EEG 数据的信噪比",
                "   - 检查是否有严重的伪迹",
                "",
                "2. **迁移学习:**",
                "   - 使用跨被试预训练的权重",
                "   - 在其他被试上预训练，然后在 S20 上微调",
                "",
                "3. **训练策略:**",
                "   - 调度器更改已经帮助了 CBraMod",
                "   - 继续优化超参数 (学习率、epochs)",
                "",
                "4. **数据增强:**",
                "   - 尝试时间域和频率域的数据增强",
                "   - 增加训练数据的多样性",
                "",
            ]
        )

        # 保存报告
        report_path = self.output_dir / "analysis_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return report_path

    def _save_summary_json(
        self,
        diagnostics: Dict[str, DiagnosticMetrics],
        comparisons: Dict[str, ModelComparisonReport],
        summaries: Dict[str, SubjectDiagnosticSummary],
    ):
        """保存 JSON 摘要"""
        summary_data = {
            "generated_at": datetime.now().isoformat(),
            "target_subjects": self.target_subjects,
            "control_subjects": self.control_subjects,
            "diagnostics": {},
            "comparisons": {},
            "summaries": {},
        }

        for key, d in diagnostics.items():
            summary_data["diagnostics"][key] = {
                "subject_id": d.subject_id,
                "model_type": d.model_type,
                "task": d.task,
                "primary_issue": d.primary_issue,
                "secondary_issues": d.secondary_issues,
                "data_quality_score": d.data_quality_score,
                "generalization_gap": d.generalization_gap,
                "recommendations": d.recommendations,
            }

        for key, c in comparisons.items():
            summary_data["comparisons"][key] = {
                "subject_id": c.subject_id,
                "task": c.task,
                "eegnet_test_acc": c.eegnet_test_acc,
                "cbramod_test_acc": c.cbramod_test_acc,
                "accuracy_diff": c.accuracy_diff,
                "better_model": c.better_model,
                "diagnosis": c.diagnosis,
                "recommendations": c.recommendations,
            }

        for key, s in summaries.items():
            summary_data["summaries"][key] = {
                "subject_id": s.subject_id,
                "task": s.task,
                "eegnet_accuracy": s.eegnet_accuracy,
                "cbramod_accuracy": s.cbramod_accuracy,
                "accuracy_gap": s.accuracy_gap,
                "problem_type": s.problem_type,
                "overall_recommendations": s.overall_recommendations,
            }

        json_path = self.output_dir / "diagnostic_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        print(f"  Saved: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate S10/S20 performance analysis report")
    parser.add_argument(
        "--project",
        default="eeg-bci",
        help="WandB project name",
    )
    parser.add_argument(
        "--entity",
        default=None,
        help="WandB entity (team/username)",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to existing data JSON (skip extraction if provided)",
    )
    parser.add_argument(
        "--output",
        default="reports/s10_s20_analysis",
        help="Output directory for report",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip data extraction (use with --data)",
    )
    parser.add_argument(
        "--target-subjects",
        nargs="+",
        default=["S10", "S20"],
        help="Target subjects to analyze",
    )
    parser.add_argument(
        "--control-subjects",
        nargs="+",
        default=["S09", "S19", "S03", "S05"],
        help="Control subjects for comparison",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)

    # 数据提取或加载
    if args.data and args.skip_extract:
        print(f"Loading existing data from {args.data}...")
        generator = ReportGenerator(
            output_dir,
            target_subjects=args.target_subjects,
            control_subjects=args.control_subjects,
        )
        subjects_data = generator.load_data(Path(args.data))
    else:
        print("Extracting data from WandB...")
        data_dir = output_dir / "data"
        extractor = WandBDataExtractor(entity=args.entity, project=args.project)
        subjects_data = extractor.extract_all(
            target_subjects=args.target_subjects,
            control_subjects=args.control_subjects,
            output_dir=data_dir,
        )

        generator = ReportGenerator(
            output_dir,
            target_subjects=args.target_subjects,
            control_subjects=args.control_subjects,
        )

    # 生成报告
    report_path = generator.generate(subjects_data)

    print("\n" + "=" * 60)
    print("  分析完成!")
    print("=" * 60)
    print(f"\n报告位置: {report_path}")
    print(f"图表位置: {output_dir / 'figures'}")
    print(f"JSON 摘要: {output_dir / 'diagnostic_summary.json'}")


if __name__ == "__main__":
    main()
