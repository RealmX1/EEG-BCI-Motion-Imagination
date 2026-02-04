"""
诊断指标模块 - 量化问题程度并提供诊断建议。

诊断维度:
1. 数据质量指标
2. 模型训练指标
3. 泛化能力指标
4. 模型适配指标 (EEGNet vs CBraMod 对比)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DiagnosticMetrics:
    """诊断指标集合"""

    subject_id: str
    model_type: str
    task: str

    # === 数据质量指标 ===
    data_quality_score: float  # 0-1, 1 = 完美
    suspected_data_issue: bool
    data_issue_type: str  # 'noise', 'mislabel', 'distribution_shift', 'none'

    # === 训练效率指标 ===
    learning_efficiency: float  # 每 epoch 的准确率提升
    loss_reduction_rate: float  # Loss 下降速度
    epochs_to_best: int  # 达到最佳性能的 epoch 数

    # === 泛化能力指标 ===
    generalization_gap: float  # train-val gap
    test_val_consistency: float  # test 与 val 的一致性

    # === 稳定性指标 ===
    training_stability: float  # Loss 曲线平滑度 (0-1, 1=最稳定)
    val_loss_volatility: float  # 验证 loss 波动

    # === 综合诊断 ===
    primary_issue: str  # 主要问题
    secondary_issues: List[str] = field(default_factory=list)
    confidence: float = 0.0  # 诊断置信度
    recommendations: List[str] = field(default_factory=list)


@dataclass
class SubjectDiagnosticSummary:
    """被试诊断总结 (包含两个模型的对比)"""

    subject_id: str
    task: str

    # 性能对比
    eegnet_accuracy: float
    cbramod_accuracy: float
    accuracy_gap: float  # CBraMod - EEGNet

    # 问题类型
    problem_type: str  # 'cbramod_specific', 'both_low', 'normal', 'eegnet_specific'

    # 诊断
    eegnet_diagnosis: Optional[DiagnosticMetrics] = None
    cbramod_diagnosis: Optional[DiagnosticMetrics] = None

    # 综合建议
    overall_recommendations: List[str] = field(default_factory=list)


class DiagnosticCalculator:
    """诊断指标计算器"""

    def __init__(self, reference_metrics: Optional[Dict] = None):
        self.reference = reference_metrics or self._default_reference()

    def _default_reference(self) -> Dict:
        """默认参考值 (基于 S09, S19 等正常被试)"""
        return {
            "binary": {
                "expected_val_loss": 0.50,
                "expected_test_acc": 0.85,
                "expected_train_val_gap": 0.05,
                "val_loss_upper_bound": 1.0,
                "min_acceptable_acc": 0.70,
            },
            "ternary": {
                "expected_val_loss": 0.70,
                "expected_test_acc": 0.66,
                "expected_train_val_gap": 0.08,
                "val_loss_upper_bound": 1.2,
                "min_acceptable_acc": 0.50,
            },
        }

    def calculate(self, data: Any) -> DiagnosticMetrics:
        """计算完整诊断指标"""
        task = data.task
        ref = self.reference.get(task, self.reference["binary"])

        # 1. 数据质量评估
        data_quality = self._assess_data_quality(data, ref)

        # 2. 训练效率
        efficiency = self._calc_learning_efficiency(data)

        # 3. 泛化能力
        generalization = self._calc_generalization(data)

        # 4. 稳定性
        stability = self._calc_stability(data)

        # 5. 综合诊断
        diagnosis = self._synthesize_diagnosis(
            data, data_quality, efficiency, generalization, stability, ref
        )

        return DiagnosticMetrics(
            subject_id=data.subject_id,
            model_type=data.model_type,
            task=data.task,
            data_quality_score=data_quality["score"],
            suspected_data_issue=data_quality["suspected_issue"],
            data_issue_type=data_quality["issue_type"],
            learning_efficiency=efficiency["rate"],
            loss_reduction_rate=efficiency["loss_rate"],
            epochs_to_best=efficiency["epochs_to_best"],
            generalization_gap=generalization["gap"],
            test_val_consistency=generalization["consistency"],
            training_stability=stability["train_stability"],
            val_loss_volatility=stability["val_volatility"],
            primary_issue=diagnosis["primary"],
            secondary_issues=diagnosis["secondary"],
            confidence=diagnosis["confidence"],
            recommendations=diagnosis["recommendations"],
        )

    def _assess_data_quality(self, data: Any, ref: Dict) -> Dict:
        """
        评估数据质量。

        S10 特征: val_loss 异常高，暗示数据分布问题
        """
        if not data.val_loss:
            return {"score": 0.5, "suspected_issue": True, "issue_type": "unknown"}

        max_val_loss = max(data.val_loss)
        final_val_loss = float(np.mean(data.val_loss[-3:])) if data.val_loss else 0

        # 计算质量分数
        upper_bound = ref["val_loss_upper_bound"]
        if final_val_loss <= upper_bound:
            score = 1.0
        elif final_val_loss <= upper_bound * 2:
            score = 0.7
        elif final_val_loss <= upper_bound * 5:
            score = 0.3
        else:
            score = 0.1

        # 判断问题类型
        suspected_issue = score < 0.8

        if max_val_loss > upper_bound * 10:
            issue_type = "severe_distribution_shift"
        elif max_val_loss > upper_bound * 5:
            issue_type = "distribution_shift"
        elif data.final_train_val_gap > 0.20:
            issue_type = "possible_overfitting"
        elif final_val_loss > upper_bound * 2:
            issue_type = "noise"
        else:
            issue_type = "none"

        return {
            "score": score,
            "suspected_issue": suspected_issue,
            "issue_type": issue_type,
        }

    def _calc_learning_efficiency(self, data: Any) -> Dict:
        """计算学习效率"""
        if len(data.train_acc) < 2:
            return {"rate": 0, "loss_rate": 0, "epochs_to_best": 0}

        # 准确率提升率 (每 epoch)
        acc_improvement = data.train_acc[-1] - data.train_acc[0]
        n_epochs = len(data.train_acc)
        rate = acc_improvement / n_epochs if n_epochs > 0 else 0

        # Loss 下降率
        if data.train_loss:
            loss_reduction = data.train_loss[0] - data.train_loss[-1]
            loss_rate = loss_reduction / n_epochs if n_epochs > 0 else 0
        else:
            loss_rate = 0

        # 达到最佳的 epoch
        epochs_to_best = data.best_epoch if data.best_epoch > 0 else n_epochs

        return {"rate": rate, "loss_rate": loss_rate, "epochs_to_best": epochs_to_best}

    def _calc_generalization(self, data: Any) -> Dict:
        """计算泛化能力指标"""
        # Train-Val gap
        gap = data.final_train_val_gap

        # Test-Val consistency
        if data.val_acc and data.test_accuracy > 0:
            best_val = max(data.val_acc)
            consistency = 1.0 - abs(data.test_accuracy - best_val)
        else:
            consistency = 0.5

        return {"gap": gap, "consistency": max(0, consistency)}

    def _calc_stability(self, data: Any) -> Dict:
        """计算训练稳定性"""
        # 训练 loss 平滑度
        if len(data.train_loss) > 1:
            diffs = np.diff(data.train_loss)
            train_stability = 1.0 / (1.0 + np.std(diffs))
        else:
            train_stability = 0.5

        # 验证 loss 波动
        if len(data.val_loss) > 1:
            val_volatility = float(np.std(data.val_loss))
        else:
            val_volatility = 0

        return {
            "train_stability": float(min(1.0, train_stability)),
            "val_volatility": val_volatility,
        }

    def _synthesize_diagnosis(
        self,
        data: Any,
        data_quality: Dict,
        efficiency: Dict,
        generalization: Dict,
        stability: Dict,
        ref: Dict,
    ) -> Dict:
        """综合诊断"""
        issues = []
        recommendations = []

        # 检查数据质量问题
        if data_quality["suspected_issue"]:
            issues.append(("data_quality", data_quality["issue_type"], 0.9))
            if data_quality["issue_type"] == "severe_distribution_shift":
                recommendations.append("检查训练集和验证集的数据分布差异")
                recommendations.append("考虑对该被试使用不同的预处理参数")
            elif data_quality["issue_type"] == "noise":
                recommendations.append("检查原始 EEG 数据的信噪比")
                recommendations.append("尝试更强的数据清洗和伪迹去除")

        # 检查过拟合
        if generalization["gap"] > 0.15:
            issues.append(("overfitting", f"gap={generalization['gap']:.2f}", 0.8))
            recommendations.append("增加正则化 (dropout, weight_decay)")
            recommendations.append("减少训练 epochs 或使用更严格的 early stopping")

        # 检查学习不足
        if efficiency["rate"] < 0.005 and data.test_accuracy < ref["min_acceptable_acc"]:
            issues.append(("underfitting", f"rate={efficiency['rate']:.4f}", 0.7))
            recommendations.append("增加学习率或训练 epochs")
            recommendations.append("检查模型是否适合该被试的数据特征")

        # 检查不稳定
        if stability["val_volatility"] > 1.0:
            issues.append(
                ("instability", f"volatility={stability['val_volatility']:.2f}", 0.6)
            )
            recommendations.append("降低学习率")
            recommendations.append("使用更平滑的学习率调度器")

        # 检查低性能
        if data.test_accuracy < ref["min_acceptable_acc"]:
            issues.append(("low_performance", f"acc={data.test_accuracy:.2%}", 0.85))
            if not recommendations:
                recommendations.append("考虑使用迁移学习")
                recommendations.append("检查该被试的数据质量")

        # 排序并选择主要问题
        issues.sort(key=lambda x: x[2], reverse=True)

        if issues:
            primary = f"{issues[0][0]}: {issues[0][1]}"
            secondary = [f"{i[0]}: {i[1]}" for i in issues[1:]]
            confidence = issues[0][2]
        else:
            primary = "none"
            secondary = []
            confidence = 0.5
            recommendations = ["性能正常，无需特殊处理"]

        return {
            "primary": primary,
            "secondary": secondary,
            "confidence": confidence,
            "recommendations": recommendations,
        }

    def create_subject_summary(
        self,
        subject_id: str,
        task: str,
        eegnet_data: Optional[Any],
        cbramod_data: Optional[Any],
    ) -> SubjectDiagnosticSummary:
        """
        创建被试诊断总结，包含模型对比分析。

        关键洞察:
        - S10: EEGNet (70%) >> CBraMod (55%) → cbramod_specific 问题
        - S20: CBraMod (68%) > EEGNet (60%) → both_low 问题
        """
        eegnet_acc = eegnet_data.test_accuracy if eegnet_data else 0
        cbramod_acc = cbramod_data.test_accuracy if cbramod_data else 0
        accuracy_gap = cbramod_acc - eegnet_acc

        # 判断问题类型
        ref = self.reference.get(task, self.reference["binary"])
        min_acc = ref["min_acceptable_acc"]

        if eegnet_acc >= min_acc and cbramod_acc < min_acc - 0.10:
            problem_type = "cbramod_specific"
        elif eegnet_acc < min_acc - 0.10 and cbramod_acc >= min_acc:
            problem_type = "eegnet_specific"
        elif eegnet_acc < min_acc and cbramod_acc < min_acc:
            problem_type = "both_low"
        else:
            problem_type = "normal"

        # 计算各模型诊断
        eegnet_diagnosis = self.calculate(eegnet_data) if eegnet_data else None
        cbramod_diagnosis = self.calculate(cbramod_data) if cbramod_data else None

        # 综合建议
        recommendations = self._generate_subject_recommendations(
            subject_id, problem_type, eegnet_acc, cbramod_acc, task
        )

        return SubjectDiagnosticSummary(
            subject_id=subject_id,
            task=task,
            eegnet_accuracy=eegnet_acc,
            cbramod_accuracy=cbramod_acc,
            accuracy_gap=accuracy_gap,
            problem_type=problem_type,
            eegnet_diagnosis=eegnet_diagnosis,
            cbramod_diagnosis=cbramod_diagnosis,
            overall_recommendations=recommendations,
        )

    def _generate_subject_recommendations(
        self,
        subject_id: str,
        problem_type: str,
        eegnet_acc: float,
        cbramod_acc: float,
        task: str,
    ) -> List[str]:
        """生成被试级别的综合建议"""
        recommendations = []

        if problem_type == "cbramod_specific":
            recommendations = [
                f"【{subject_id} CBraMod 特有问题】",
                f"EEGNet ({eegnet_acc:.1%}) 显著优于 CBraMod ({cbramod_acc:.1%})",
                "说明该被试数据可学习，问题在于 CBraMod 的适配",
                "",
                "建议:",
                "1. 检查 CBraMod 预处理配置 (采样率、滤波参数)",
                "2. 该被试的 EEG 频带特征可能与 CBraMod 预训练数据不匹配",
                "3. 尝试冻结 CBraMod backbone，只微调分类器头",
                "4. 考虑使用 EEGNet 作为该被试的主模型",
            ]

        elif problem_type == "both_low":
            recommendations = [
                f"【{subject_id} 两模型均表现不佳】",
                f"EEGNet: {eegnet_acc:.1%}, CBraMod: {cbramod_acc:.1%}",
                "该被试对两个模型都具有挑战性",
                "",
                "建议:",
                "1. 检查原始 EEG 数据质量 (信噪比、伪迹)",
                "2. 该被试可能需要特殊的预处理参数",
                "3. 尝试数据增强策略",
                "4. 考虑使用跨被试预训练进行迁移学习",
                "5. 检查该被试的运动想象能力是否与其他被试不同",
            ]

        elif problem_type == "eegnet_specific":
            recommendations = [
                f"【{subject_id} EEGNet 特有问题】",
                f"CBraMod ({cbramod_acc:.1%}) 优于 EEGNet ({eegnet_acc:.1%})",
                "",
                "建议:",
                "1. 该被试更适合使用 CBraMod",
                "2. EEGNet 可能需要超参数调优",
            ]

        else:  # normal
            recommendations = [
                f"【{subject_id} 性能正常】",
                f"EEGNet: {eegnet_acc:.1%}, CBraMod: {cbramod_acc:.1%}",
                "无需特殊处理",
            ]

        return recommendations
