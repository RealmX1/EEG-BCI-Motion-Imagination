"""
可视化模块 - 生成分析图表。

图表类型:
1. Loss 曲线对比 (问题被试 vs 对照被试)
2. Train-Val Gap 演化
3. 模型性能对比 (EEGNet vs CBraMod)
4. 诊断指标热图
5. 被试性能分布
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# 使用非交互式后端
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class AnalysisVisualizer:
    """分析可视化器"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置中文字体支持
        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        # 颜色方案
        self.colors = {
            "S10": "#e74c3c",  # 红色 - 问题被试
            "S20": "#e67e22",  # 橙色 - 问题被试
            "S09": "#3498db",  # 蓝色 - 对照被试
            "S19": "#2ecc71",  # 绿色 - 对照被试
            "S03": "#9b59b6",  # 紫色
            "S05": "#1abc9c",  # 青色
            "eegnet": "#3498db",  # 蓝色
            "cbramod": "#e74c3c",  # 红色
        }

    def plot_loss_comparison(
        self,
        subjects_data: Dict[str, Any],
        task: str = "binary",
        problem_subjects: List[str] = ["S10", "S20"],
        control_subjects: List[str] = ["S09", "S19"],
        model_type: str = "cbramod",
        save_name: Optional[str] = None,
    ) -> Path:
        """
        绘制 loss 曲线对比图。

        对比问题被试 vs 对照被试的训练过程
        """
        if save_name is None:
            save_name = f"loss_comparison_{model_type}_{task}.png"

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        all_subjects = problem_subjects + control_subjects

        # Train Loss
        ax = axes[0, 0]
        for sid in all_subjects:
            key = f"{sid}_{model_type}_{task}"
            if key in subjects_data and subjects_data[key]:
                data = subjects_data[key][0]  # 最新的运行
                if data.train_loss:
                    linestyle = "--" if sid in problem_subjects else "-"
                    ax.plot(
                        data.train_loss,
                        label=sid,
                        color=self.colors.get(sid, "gray"),
                        linestyle=linestyle,
                        linewidth=2 if sid in problem_subjects else 1.5,
                    )
        ax.set_title("Train Loss", fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # Val Loss
        ax = axes[0, 1]
        max_val_loss = 0
        for sid in all_subjects:
            key = f"{sid}_{model_type}_{task}"
            if key in subjects_data and subjects_data[key]:
                data = subjects_data[key][0]
                if data.val_loss:
                    linestyle = "--" if sid in problem_subjects else "-"
                    ax.plot(
                        data.val_loss,
                        label=sid,
                        color=self.colors.get(sid, "gray"),
                        linestyle=linestyle,
                        linewidth=2 if sid in problem_subjects else 1.5,
                    )
                    max_val_loss = max(max_val_loss, max(data.val_loss))

        ax.set_title("Validation Loss", fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # 如果 val loss 差异很大，使用对数刻度
        if max_val_loss > 5:
            ax.set_yscale("log")
            ax.set_title("Validation Loss (log scale)", fontsize=12)

        # Train Accuracy
        ax = axes[1, 0]
        for sid in all_subjects:
            key = f"{sid}_{model_type}_{task}"
            if key in subjects_data and subjects_data[key]:
                data = subjects_data[key][0]
                if data.train_acc:
                    linestyle = "--" if sid in problem_subjects else "-"
                    ax.plot(
                        data.train_acc,
                        label=sid,
                        color=self.colors.get(sid, "gray"),
                        linestyle=linestyle,
                        linewidth=2 if sid in problem_subjects else 1.5,
                    )
        ax.set_title("Train Accuracy", fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.4, 1.0)

        # Val Accuracy
        ax = axes[1, 1]
        for sid in all_subjects:
            key = f"{sid}_{model_type}_{task}"
            if key in subjects_data and subjects_data[key]:
                data = subjects_data[key][0]
                if data.val_acc:
                    linestyle = "--" if sid in problem_subjects else "-"
                    ax.plot(
                        data.val_acc,
                        label=sid,
                        color=self.colors.get(sid, "gray"),
                        linestyle=linestyle,
                        linewidth=2 if sid in problem_subjects else 1.5,
                    )
        ax.set_title("Validation Accuracy", fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.4, 1.0)

        plt.suptitle(
            f"{model_type.upper()} Training Curves - {task.title()} Task\n"
            f"(Dashed: Problem Subjects, Solid: Control Subjects)",
            fontsize=14,
        )
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved: {save_path}")
        return save_path

    def plot_model_comparison(
        self,
        subjects_data: Dict[str, Any],
        subject_id: str,
        task: str = "binary",
        save_name: Optional[str] = None,
    ) -> Path:
        """
        绘制同一被试的 EEGNet vs CBraMod 对比图。

        用于分析 S10 (EEGNet 好) 和 S20 (CBraMod 好) 的差异
        """
        if save_name is None:
            save_name = f"model_comparison_{subject_id}_{task}.png"

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        eegnet_key = f"{subject_id}_eegnet_{task}"
        cbramod_key = f"{subject_id}_cbramod_{task}"

        eegnet_data = subjects_data.get(eegnet_key, [None])[0] if eegnet_key in subjects_data else None
        cbramod_data = subjects_data.get(cbramod_key, [None])[0] if cbramod_key in subjects_data else None

        # Val Loss 对比
        ax = axes[0, 0]
        if eegnet_data and eegnet_data.val_loss:
            ax.plot(
                eegnet_data.val_loss,
                label=f"EEGNet (test: {eegnet_data.test_accuracy:.1%})",
                color=self.colors["eegnet"],
                linewidth=2,
            )
        if cbramod_data and cbramod_data.val_loss:
            ax.plot(
                cbramod_data.val_loss,
                label=f"CBraMod (test: {cbramod_data.test_accuracy:.1%})",
                color=self.colors["cbramod"],
                linewidth=2,
            )
        ax.set_title("Validation Loss", fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Val Accuracy 对比
        ax = axes[0, 1]
        if eegnet_data and eegnet_data.val_acc:
            ax.plot(
                eegnet_data.val_acc,
                label=f"EEGNet",
                color=self.colors["eegnet"],
                linewidth=2,
            )
        if cbramod_data and cbramod_data.val_acc:
            ax.plot(
                cbramod_data.val_acc,
                label=f"CBraMod",
                color=self.colors["cbramod"],
                linewidth=2,
            )
        ax.set_title("Validation Accuracy", fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.4, 1.0)

        # Train-Val Gap 对比
        ax = axes[1, 0]
        if eegnet_data and eegnet_data.train_acc and eegnet_data.val_acc:
            min_len = min(len(eegnet_data.train_acc), len(eegnet_data.val_acc))
            gaps = [eegnet_data.train_acc[i] - eegnet_data.val_acc[i] for i in range(min_len)]
            ax.plot(gaps, label="EEGNet", color=self.colors["eegnet"], linewidth=2)
        if cbramod_data and cbramod_data.train_acc and cbramod_data.val_acc:
            min_len = min(len(cbramod_data.train_acc), len(cbramod_data.val_acc))
            gaps = [cbramod_data.train_acc[i] - cbramod_data.val_acc[i] for i in range(min_len)]
            ax.plot(gaps, label="CBraMod", color=self.colors["cbramod"], linewidth=2)
        ax.axhline(y=0.05, color="gray", linestyle=":", label="Normal gap (5%)")
        ax.axhline(y=0.15, color="red", linestyle=":", alpha=0.5, label="Warning (15%)")
        ax.set_title("Train-Val Gap (Overfitting Indicator)", fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gap (Train - Val)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 性能柱状图
        ax = axes[1, 1]
        models = []
        test_accs = []
        majority_accs = []
        colors_list = []

        if eegnet_data:
            models.append("EEGNet")
            test_accs.append(eegnet_data.test_accuracy)
            majority_accs.append(eegnet_data.test_majority_accuracy)
            colors_list.append(self.colors["eegnet"])
        if cbramod_data:
            models.append("CBraMod")
            test_accs.append(cbramod_data.test_accuracy)
            majority_accs.append(cbramod_data.test_majority_accuracy)
            colors_list.append(self.colors["cbramod"])

        if models:
            x = np.arange(len(models))
            width = 0.35
            ax.bar(x - width / 2, test_accs, width, label="Test Acc", color=colors_list, alpha=0.7)
            ax.bar(
                x + width / 2,
                majority_accs,
                width,
                label="Majority Acc",
                color=colors_list,
                alpha=0.4,
                hatch="//",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(models)
            ax.set_ylabel("Accuracy")
            ax.set_title("Test Performance Comparison", fontsize=12)
            ax.legend()
            ax.set_ylim(0, 1.0)

            # 添加数值标签
            for i, (test, maj) in enumerate(zip(test_accs, majority_accs)):
                ax.text(i - width / 2, test + 0.02, f"{test:.1%}", ha="center", fontsize=10)
                ax.text(i + width / 2, maj + 0.02, f"{maj:.1%}", ha="center", fontsize=10)

        plt.suptitle(
            f"{subject_id}: EEGNet vs CBraMod - {task.title()} Task",
            fontsize=14,
        )
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved: {save_path}")
        return save_path

    def plot_performance_overview(
        self,
        subjects_data: Dict[str, Any],
        task: str = "binary",
        highlight_subjects: List[str] = ["S10", "S20"],
        save_name: Optional[str] = None,
    ) -> Path:
        """
        绘制所有被试的性能概览，突出显示问题被试。
        """
        if save_name is None:
            save_name = f"performance_overview_{task}.png"

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 收集数据
        eegnet_data_dict = {}
        cbramod_data_dict = {}

        for key, runs in subjects_data.items():
            if not runs or not key.endswith(task):
                continue
            parts = key.split("_")
            subject_id = parts[0]
            model_type = parts[1]

            if model_type == "eegnet":
                eegnet_data_dict[subject_id] = runs[0].test_accuracy
            elif model_type == "cbramod":
                cbramod_data_dict[subject_id] = runs[0].test_accuracy

        # EEGNet 性能
        ax = axes[0]
        if eegnet_data_dict:
            subjects = sorted(eegnet_data_dict.keys())
            accuracies = [eegnet_data_dict[s] for s in subjects]
            colors = ["red" if s in highlight_subjects else "steelblue" for s in subjects]

            bars = ax.bar(subjects, accuracies, color=colors, alpha=0.7)
            ax.axhline(y=np.mean(accuracies), color="green", linestyle="--", label=f"Mean: {np.mean(accuracies):.1%}")

            ax.set_title(f"EEGNet - {task.title()}", fontsize=12)
            ax.set_xlabel("Subject")
            ax.set_ylabel("Test Accuracy")
            ax.set_ylim(0, 1.1)
            ax.legend()

            # 标注问题被试
            for i, (s, acc) in enumerate(zip(subjects, accuracies)):
                if s in highlight_subjects:
                    ax.annotate(
                        f"{acc:.1%}",
                        xy=(i, acc),
                        xytext=(i, acc + 0.05),
                        ha="center",
                        fontweight="bold",
                        color="red",
                    )

            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # CBraMod 性能
        ax = axes[1]
        if cbramod_data_dict:
            subjects = sorted(cbramod_data_dict.keys())
            accuracies = [cbramod_data_dict[s] for s in subjects]
            colors = ["red" if s in highlight_subjects else "steelblue" for s in subjects]

            bars = ax.bar(subjects, accuracies, color=colors, alpha=0.7)
            ax.axhline(y=np.mean(accuracies), color="green", linestyle="--", label=f"Mean: {np.mean(accuracies):.1%}")

            ax.set_title(f"CBraMod - {task.title()}", fontsize=12)
            ax.set_xlabel("Subject")
            ax.set_ylabel("Test Accuracy")
            ax.set_ylim(0, 1.1)
            ax.legend()

            for i, (s, acc) in enumerate(zip(subjects, accuracies)):
                if s in highlight_subjects:
                    ax.annotate(
                        f"{acc:.1%}",
                        xy=(i, acc),
                        xytext=(i, acc + 0.05),
                        ha="center",
                        fontweight="bold",
                        color="red",
                    )

            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        plt.suptitle(
            f"Performance Overview - {task.title()} Task\n(Red bars: Problem Subjects)",
            fontsize=14,
        )
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved: {save_path}")
        return save_path

    def plot_diagnostic_summary(
        self,
        diagnostics: Dict[str, Any],
        save_name: str = "diagnostic_summary.png",
    ) -> Path:
        """
        绘制诊断摘要图。
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # 收集数据
        subjects = []
        eegnet_accs = []
        cbramod_accs = []

        for key, diag in diagnostics.items():
            if hasattr(diag, "subject_id"):
                if diag.subject_id not in subjects:
                    subjects.append(diag.subject_id)
                if diag.model_type == "eegnet":
                    while len(eegnet_accs) < len(subjects):
                        eegnet_accs.append(0)
                    eegnet_accs[-1] = diag.test_val_consistency
                elif diag.model_type == "cbramod":
                    while len(cbramod_accs) < len(subjects):
                        cbramod_accs.append(0)
                    cbramod_accs[-1] = diag.test_val_consistency

        if subjects:
            x = np.arange(len(subjects))
            width = 0.35

            ax.bar(x - width / 2, eegnet_accs, width, label="EEGNet", color="steelblue")
            ax.bar(x + width / 2, cbramod_accs, width, label="CBraMod", color="coral")

            ax.set_xticks(x)
            ax.set_xticklabels(subjects)
            ax.set_ylabel("Test-Val Consistency")
            ax.set_title("Model Performance Consistency by Subject")
            ax.legend()

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved: {save_path}")
        return save_path
