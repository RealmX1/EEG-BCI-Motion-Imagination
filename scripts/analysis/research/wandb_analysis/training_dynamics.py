"""
训练动态分析模块 - 分析训练过程中的异常模式。

分析维度:
1. 过拟合检测: Train-Val accuracy gap 演化
2. 学习不足检测: Final train loss 是否高于正常范围
3. 收敛分析: Val loss 是否稳定收敛
4. 异常模式: Val loss 是否出现爆炸或不稳定
5. 模型对比: EEGNet vs CBraMod 在同一被试上的差异
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TrainingDynamicsReport:
    """训练动态分析报告"""

    subject_id: str
    model_type: str
    task: str

    # 过拟合检测
    overfitting_severity: str  # 'none', 'mild', 'moderate', 'severe'
    overfitting_start_epoch: int
    max_train_val_gap: float

    # 学习不足检测
    underfitting_detected: bool
    final_train_loss: float
    final_train_acc: float
    expected_train_loss_range: Tuple[float, float]

    # 收敛分析
    convergence_detected: bool
    convergence_epoch: int
    final_val_loss: float
    final_val_loss_stability: float  # std of last 5 epochs

    # 异常模式
    val_loss_explosion: bool  # 验证 loss 突然爆炸
    val_loss_max: float
    learning_rate_issues: bool

    # 性能指标
    test_accuracy: float
    test_majority_accuracy: float
    best_val_accuracy: float

    # 详细数据
    epoch_by_epoch_analysis: List[Dict] = field(default_factory=list)


@dataclass
class ModelComparisonReport:
    """EEGNet vs CBraMod 对比报告"""

    subject_id: str
    task: str

    # EEGNet 指标
    eegnet_test_acc: float
    eegnet_test_majority_acc: float
    eegnet_best_val_acc: float
    eegnet_epochs: int
    eegnet_final_train_val_gap: float

    # CBraMod 指标
    cbramod_test_acc: float
    cbramod_test_majority_acc: float
    cbramod_best_val_acc: float
    cbramod_epochs: int
    cbramod_final_train_val_gap: float

    # 对比分析
    accuracy_diff: float  # CBraMod - EEGNet
    better_model: str
    gap_diff: float  # CBraMod gap - EEGNet gap (正值表示 CBraMod 过拟合更严重)

    # 诊断
    diagnosis: str
    recommendations: List[str] = field(default_factory=list)


class TrainingDynamicsAnalyzer:
    """训练动态分析器"""

    def __init__(self, baseline_stats: Optional[Dict] = None):
        """
        Args:
            baseline_stats: 正常被试的统计基线
        """
        self.baseline = baseline_stats or self._get_default_baseline()

    def _get_default_baseline(self) -> Dict:
        """基于 S09, S19 等表现好的被试的统计"""
        return {
            "binary": {
                "mean_final_train_loss": 0.35,
                "std_final_train_loss": 0.10,
                "mean_final_val_loss": 0.50,
                "std_final_val_loss": 0.15,
                "mean_train_val_gap": 0.05,
                "std_train_val_gap": 0.03,
                "expected_test_acc": 0.85,
                "val_loss_upper_bound": 1.0,
            },
            "ternary": {
                "mean_final_train_loss": 0.50,
                "std_final_train_loss": 0.15,
                "mean_final_val_loss": 0.70,
                "std_final_val_loss": 0.20,
                "mean_train_val_gap": 0.08,
                "std_train_val_gap": 0.05,
                "expected_test_acc": 0.66,
                "val_loss_upper_bound": 1.2,
            },
        }

    def analyze(self, data: Any) -> TrainingDynamicsReport:
        """完整训练动态分析"""
        task = data.task
        ref = self.baseline.get(task, self.baseline["binary"])

        # 1. 过拟合检测
        overfitting = self._detect_overfitting(
            data.train_acc, data.val_acc, data.train_loss, data.val_loss
        )

        # 2. 学习不足检测
        underfitting = self._detect_underfitting(data.train_loss, data.train_acc, ref)

        # 3. 收敛分析
        convergence = self._analyze_convergence(data.val_loss)

        # 4. 异常模式检测
        val_explosion = self._detect_val_loss_explosion(data.val_loss, ref)
        lr_issues = self._detect_lr_issues(data.learning_rate, data.val_loss)

        # 5. Epoch-by-epoch 分析
        epoch_analysis = self._epoch_by_epoch(data)

        return TrainingDynamicsReport(
            subject_id=data.subject_id,
            model_type=data.model_type,
            task=data.task,
            overfitting_severity=overfitting["severity"],
            overfitting_start_epoch=overfitting["start_epoch"],
            max_train_val_gap=overfitting["max_gap"],
            underfitting_detected=underfitting["detected"],
            final_train_loss=underfitting["final_loss"],
            final_train_acc=underfitting["final_acc"],
            expected_train_loss_range=underfitting["expected_range"],
            convergence_detected=convergence["detected"],
            convergence_epoch=convergence["epoch"],
            final_val_loss=convergence["final_loss"],
            final_val_loss_stability=convergence["stability"],
            val_loss_explosion=val_explosion["detected"],
            val_loss_max=val_explosion["max_val"],
            learning_rate_issues=lr_issues,
            test_accuracy=data.test_accuracy,
            test_majority_accuracy=data.test_majority_accuracy,
            best_val_accuracy=data.best_val_accuracy,
            epoch_by_epoch_analysis=epoch_analysis,
        )

    def _detect_overfitting(
        self,
        train_acc: List[float],
        val_acc: List[float],
        train_loss: List[float],
        val_loss: List[float],
    ) -> Dict:
        """
        检测过拟合。

        S10 特征: train_acc ~70%, val_acc ~55%, gap ~15%
        正常特征: train_acc ~84%, val_acc ~79%, gap ~5%
        """
        if not train_acc or not val_acc:
            return {"severity": "unknown", "start_epoch": 0, "max_gap": 0}

        min_len = min(len(train_acc), len(val_acc))
        gaps = [train_acc[i] - val_acc[i] for i in range(min_len)]
        max_gap = max(gaps) if gaps else 0

        # 找到过拟合开始的 epoch
        start_epoch = len(gaps)
        for i in range(1, len(gaps)):
            if gaps[i] > gaps[i - 1] + 0.02:  # Gap 开始增大
                start_epoch = i
                break

        # 判断严重程度
        if max_gap < 0.05:
            severity = "none"
        elif max_gap < 0.10:
            severity = "mild"
        elif max_gap < 0.20:
            severity = "moderate"
        else:
            severity = "severe"

        return {
            "severity": severity,
            "start_epoch": start_epoch,
            "max_gap": max_gap,
        }

    def _detect_underfitting(
        self, train_loss: List[float], train_acc: List[float], ref: Dict
    ) -> Dict:
        """检测学习不足"""
        if not train_loss:
            return {
                "detected": False,
                "final_loss": 0,
                "final_acc": 0,
                "expected_range": (0, 1),
            }

        final_loss = float(np.mean(train_loss[-3:])) if train_loss else 0
        final_acc = float(np.mean(train_acc[-3:])) if train_acc else 0

        expected_low = ref["mean_final_train_loss"] - 2 * ref["std_final_train_loss"]
        expected_high = ref["mean_final_train_loss"] + 2 * ref["std_final_train_loss"]

        detected = final_loss > expected_high or final_acc < 0.70

        return {
            "detected": detected,
            "final_loss": final_loss,
            "final_acc": final_acc,
            "expected_range": (expected_low, expected_high),
        }

    def _analyze_convergence(self, val_loss: List[float]) -> Dict:
        """分析收敛情况"""
        if len(val_loss) < 5:
            return {
                "detected": False,
                "epoch": len(val_loss),
                "final_loss": val_loss[-1] if val_loss else 0,
                "stability": float("inf"),
            }

        final_loss = float(np.mean(val_loss[-3:]))
        stability = float(np.std(val_loss[-5:]))

        # 判断是否收敛
        detected = stability < 0.05

        # 找收敛点
        convergence_epoch = len(val_loss)
        for i in range(5, len(val_loss)):
            window_std = np.std(val_loss[i - 5 : i])
            if window_std < 0.05:
                convergence_epoch = i - 5
                break

        return {
            "detected": detected,
            "epoch": convergence_epoch,
            "final_loss": final_loss,
            "stability": stability,
        }

    def _detect_val_loss_explosion(self, val_loss: List[float], ref: Dict) -> Dict:
        """
        检测验证 loss 爆炸。

        S10 特征: val_loss 高达 9-11，远超正常范围 0.4-0.6
        """
        if not val_loss:
            return {"detected": False, "max_val": 0}

        max_val = max(val_loss)
        threshold = ref["val_loss_upper_bound"] * 5  # 5 倍于正常上界

        return {
            "detected": max_val > threshold,
            "max_val": max_val,
        }

    def _detect_lr_issues(self, learning_rate: List[float], val_loss: List[float]) -> bool:
        """检测学习率相关问题"""
        if len(learning_rate) < 5 or len(val_loss) < 5:
            return False

        # 如果 LR 降低但 loss 仍然上升，可能有问题
        lr_decreasing = learning_rate[-1] < learning_rate[0] * 0.5
        mid_idx = len(val_loss) // 2
        loss_increasing = val_loss[-1] > val_loss[mid_idx]

        return lr_decreasing and loss_increasing

    def _epoch_by_epoch(self, data: Any) -> List[Dict]:
        """生成逐 epoch 分析"""
        analysis = []
        n_epochs = len(data.train_loss)

        for i in range(n_epochs):
            epoch_data: Dict[str, Any] = {
                "epoch": i + 1,
                "train_loss": data.train_loss[i] if i < len(data.train_loss) else None,
                "train_acc": data.train_acc[i] if i < len(data.train_acc) else None,
                "val_loss": data.val_loss[i] if i < len(data.val_loss) else None,
                "val_acc": data.val_acc[i] if i < len(data.val_acc) else None,
            }

            if i < len(data.val_majority_acc):
                epoch_data["val_majority_acc"] = data.val_majority_acc[i]
            if i < len(data.learning_rate):
                epoch_data["lr"] = data.learning_rate[i]

            # 计算该 epoch 的特征
            if epoch_data["train_acc"] and epoch_data["val_acc"]:
                epoch_data["train_val_gap"] = epoch_data["train_acc"] - epoch_data["val_acc"]

            analysis.append(epoch_data)

        return analysis

    def compare_models(
        self,
        eegnet_data: Any,
        cbramod_data: Any,
    ) -> ModelComparisonReport:
        """
        对比 EEGNet 和 CBraMod 在同一被试上的表现。

        关键洞察:
        - S10: EEGNet (70%) >> CBraMod (55%) → CBraMod 特有问题
        - S20: CBraMod (68%) > EEGNet (60%) → 两者都较差，但 CBraMod 略好
        """
        subject_id = eegnet_data.subject_id
        task = eegnet_data.task

        # EEGNet 指标
        eegnet_test_acc = eegnet_data.test_accuracy
        eegnet_test_majority = eegnet_data.test_majority_accuracy
        eegnet_best_val = eegnet_data.best_val_accuracy
        eegnet_epochs = eegnet_data.total_epochs
        eegnet_gap = eegnet_data.final_train_val_gap

        # CBraMod 指标
        cbramod_test_acc = cbramod_data.test_accuracy
        cbramod_test_majority = cbramod_data.test_majority_accuracy
        cbramod_best_val = cbramod_data.best_val_accuracy
        cbramod_epochs = cbramod_data.total_epochs
        cbramod_gap = cbramod_data.final_train_val_gap

        # 对比
        acc_diff = cbramod_test_acc - eegnet_test_acc
        gap_diff = cbramod_gap - eegnet_gap

        if acc_diff > 0.05:
            better_model = "cbramod"
        elif acc_diff < -0.05:
            better_model = "eegnet"
        else:
            better_model = "similar"

        # 诊断
        diagnosis, recommendations = self._diagnose_model_difference(
            subject_id,
            eegnet_test_acc,
            cbramod_test_acc,
            eegnet_gap,
            cbramod_gap,
            task,
        )

        return ModelComparisonReport(
            subject_id=subject_id,
            task=task,
            eegnet_test_acc=eegnet_test_acc,
            eegnet_test_majority_acc=eegnet_test_majority,
            eegnet_best_val_acc=eegnet_best_val,
            eegnet_epochs=eegnet_epochs,
            eegnet_final_train_val_gap=eegnet_gap,
            cbramod_test_acc=cbramod_test_acc,
            cbramod_test_majority_acc=cbramod_test_majority,
            cbramod_best_val_acc=cbramod_best_val,
            cbramod_epochs=cbramod_epochs,
            cbramod_final_train_val_gap=cbramod_gap,
            accuracy_diff=acc_diff,
            better_model=better_model,
            gap_diff=gap_diff,
            diagnosis=diagnosis,
            recommendations=recommendations,
        )

    def _diagnose_model_difference(
        self,
        subject_id: str,
        eegnet_acc: float,
        cbramod_acc: float,
        eegnet_gap: float,
        cbramod_gap: float,
        task: str,
    ) -> Tuple[str, List[str]]:
        """诊断模型差异的原因"""

        recommendations = []

        # S10 模式: EEGNet 明显优于 CBraMod
        if eegnet_acc - cbramod_acc > 0.10:
            diagnosis = (
                f"CBraMod 在 {subject_id} 上存在特有问题。"
                f"EEGNet ({eegnet_acc:.1%}) 显著优于 CBraMod ({cbramod_acc:.1%})，"
                f"说明该被试数据本身是可学习的。"
            )
            recommendations = [
                "检查 CBraMod 的预处理配置是否适合该被试",
                "尝试增加正则化 (dropout, weight_decay)",
                "考虑冻结 backbone，只微调分类器",
                "检查该被试的 EEG 信号频带特征是否与 CBraMod 预训练数据不匹配",
            ]

        # S20 模式: CBraMod 略优于 EEGNet，但两者都较差
        elif cbramod_acc - eegnet_acc > 0.05 and cbramod_acc < 0.75:
            diagnosis = (
                f"{subject_id} 对两个模型都具有挑战性。"
                f"CBraMod ({cbramod_acc:.1%}) 略优于 EEGNet ({eegnet_acc:.1%})，"
                f"但两者均低于正常水平。"
            )
            recommendations = [
                "该被试可能存在数据质量问题或信号特征独特",
                "尝试数据增强策略",
                "考虑使用跨被试预训练进行迁移学习",
                "检查原始 EEG 数据的信噪比",
            ]

        # 两者相近
        elif abs(cbramod_acc - eegnet_acc) <= 0.05:
            diagnosis = (
                f"{subject_id} 上两个模型表现相近。"
                f"EEGNet: {eegnet_acc:.1%}, CBraMod: {cbramod_acc:.1%}。"
            )
            if cbramod_acc < 0.75:
                recommendations = [
                    "两个模型都未能很好地学习该被试的特征",
                    "建议进行数据质量检查",
                    "考虑被试特定的超参数调优",
                ]
            else:
                recommendations = ["性能正常，无需特殊处理"]

        # CBraMod 明显优于 EEGNet
        else:
            diagnosis = (
                f"CBraMod 在 {subject_id} 上表现更好。"
                f"CBraMod ({cbramod_acc:.1%}) 优于 EEGNet ({eegnet_acc:.1%})。"
            )
            recommendations = ["继续使用 CBraMod", "可考虑进一步微调以提升性能"]

        return diagnosis, recommendations
