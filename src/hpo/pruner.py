"""
概率剪枝器 (Probabilistic Subject Pruner)

基于正态分布估计当前 trial 最终得分超过历史最优的概率。
当 P(final > best) < threshold 时剪枝。

用于 within-subject 和 transfer 范式的被试级别剪枝：
每训练完一个被试，report 累积均值，pruner 判断是否继续。
"""

import logging
import math
from typing import Optional

import numpy as np
import optuna
from scipy.stats import norm

log = logging.getLogger(__name__)


class ProbabilisticSubjectPruner(optuna.pruners.BasePruner):
    """
    估计 P(final_mean > best_historical) 并在概率过低时剪枝。

    工作原理：
    1. trial.report(cumulative_mean, step=i) 传入的是前 i+1 个被试的累积均值
    2. 从累积均值反推每个被试的独立得分，计算方差
    3. 用正态分布估计剩余被试的不确定性
    4. 计算 P(最终均值 > 历史最优) = 1 - Φ((best - current) / σ_final)
    5. 若概率 < threshold → 剪枝

    Args:
        n_total_steps: 总步数（被试数量）
        threshold: 剪枝概率阈值，P < threshold 时剪枝 (default: 0.1)
        min_steps: 至少完成多少步后才开始剪枝 (default: 3)
    """

    def __init__(
        self,
        n_total_steps: int,
        threshold: float = 0.1,
        min_steps: int = 3,
    ):
        if n_total_steps < 1:
            raise ValueError(f"n_total_steps must be >= 1, got {n_total_steps}")
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        if min_steps < 1:
            raise ValueError(f"min_steps must be >= 1, got {min_steps}")

        self.n_total_steps = n_total_steps
        self.threshold = threshold
        self.min_steps = min_steps

    def prune(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> bool:
        """判断是否应该剪枝当前 trial。"""
        # 获取已 report 的中间值 {step: cumulative_mean}
        intermediate_values = trial.intermediate_values
        if not intermediate_values:
            return False

        n_done = max(intermediate_values.keys()) + 1  # steps are 0-indexed

        # 太早不剪
        if n_done < self.min_steps:
            return False

        # 已经完成所有步骤，无需剪枝
        if n_done >= self.n_total_steps:
            return False

        # 获取历史最优值（已完成的 trials）
        best_value = self._get_best_value(study)
        if best_value is None:
            return False

        # 当前累积均值
        current_mean = intermediate_values[n_done - 1]

        # 从累积均值反推每步的独立值
        per_step_values = self._recover_per_step_values(intermediate_values, n_done)

        # 计算方差
        if len(per_step_values) < 2:
            return False
        sigma = float(np.std(per_step_values, ddof=1))

        # 方差接近零：无法估计不确定性，不剪
        if sigma < 1e-10:
            return False

        # 估计最终均值的标准差
        n_remaining = self.n_total_steps - n_done
        # final_mean = (n_done * current_mean + sum of remaining) / n_total
        # std of remaining sum = sigma * sqrt(n_remaining)
        # std of final_mean = sigma * sqrt(n_remaining) / n_total
        final_std = sigma * math.sqrt(n_remaining) / self.n_total_steps

        # P(final_mean > best) = 1 - Φ((best - current_mean) / final_std)
        z = (best_value - current_mean) / final_std
        prob_exceed = 1.0 - norm.cdf(z)

        should_prune = prob_exceed < self.threshold

        if should_prune:
            log.info(
                f"Trial {trial.number}: pruned at step {n_done}/{self.n_total_steps}. "
                f"P(final > {best_value:.4f}) = {prob_exceed:.4f} < {self.threshold}"
            )

        return should_prune

    def _get_best_value(self, study: optuna.Study) -> Optional[float]:
        """获取已完成 trials 中的最优值。"""
        completed_trials = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed_trials:
            return None

        if study.direction == optuna.study.StudyDirection.MAXIMIZE:
            return max(t.value for t in completed_trials)
        else:
            return min(t.value for t in completed_trials)

    @staticmethod
    def _recover_per_step_values(
        intermediate_values: dict, n_done: int
    ) -> np.ndarray:
        """
        从累积均值反推每步的独立值。

        cumulative_mean[i] = sum(values[0..i]) / (i+1)
        value[i] = cumulative_mean[i] * (i+1) - cumulative_mean[i-1] * i
        """
        per_step = []
        for step in range(n_done):
            cum_mean = intermediate_values.get(step)
            if cum_mean is None:
                continue
            if step == 0:
                per_step.append(cum_mean)
            else:
                prev_cum_mean = intermediate_values.get(step - 1)
                if prev_cum_mean is None:
                    continue
                value = cum_mean * (step + 1) - prev_cum_mean * step
                per_step.append(value)
        return np.array(per_step)
