"""
概率剪枝器 (Probabilistic Subject Pruner)

基于轨迹对比估计当前 trial 最终得分超过历史最优的概率。
当 P(final > best) < threshold 时剪枝。

核心思想：被试难度在不同超参配置间高度一致（subject consistency），
因此用 best trial 的逐步轨迹作为参考，比固定最终值更准确。

两种模式：
- **轨迹模式 [traj]**：将当前 trial 的累积均值与 best trial 在同一步的
  累积均值对比，使用逐被试 gap 的方差（σ_Δ）估计不确定性。
  σ_Δ 通常比绝对 σ 小 3-5x，剪枝更精准。
- **绝对回退 [abs]**：当 best trial 无中间值或步数不匹配时，
  回退到原始方法（与 best 最终值比较，用绝对方差）。

用于 within-subject 和 transfer 范式的被试级别剪枝：
每训练完一个被试，report 累积均值，pruner 判断是否继续。
"""

import logging
import math
from typing import Optional

import numpy as np
import optuna
from scipy.stats import norm

from src.utils.timing import Colors, colored

log = logging.getLogger(__name__)


class ProbabilisticSubjectPruner(optuna.pruners.BasePruner):
    """
    轨迹对比概率剪枝器。

    优先使用 best completed trial 的逐步轨迹作为参考（trajectory mode），
    利用逐被试 gap 方差（σ_Δ）估计剩余不确定性。当 best trial 无中间值时
    回退到绝对方差模式（absolute fallback）。

    工作原理（轨迹模式）：
    1. trial.report(cumulative_mean, step=i) 传入前 i+1 个被试的累积均值
    2. 找到 best completed trial 在同一步 i 的累积均值
    3. 计算 gap = best_at_step - current_mean（正值 = 落后）
    4. 从两条轨迹反推逐被试独立得分，计算逐被试 gap 的标准差 σ_Δ
    5. 估计最终 gap 的不确定性：σ_final = σ_Δ * √n_remaining / n_total
    6. P(最终均值 > best) = 1 - Φ(gap / σ_final)
    7. 若 P < threshold → 剪枝

    回退模式（无轨迹可用时）：
    - 与 best 最终值比较，用当前 trial 绝对方差估计不确定性

    Args:
        n_total_steps: 总步数（被试数量）
        threshold: 剪枝概率阈值，P < threshold 时剪枝 (default: 0.05)
        min_steps: 至少完成多少步后才开始剪枝 (default: 3)
    """

    def __init__(
        self,
        n_total_steps: int,
        threshold: float = 0.05,
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
        intermediate_values = trial.intermediate_values
        if not intermediate_values:
            return False

        n_done = len(intermediate_values)  # count of actually-reported steps

        if n_done < self.min_steps:
            return False

        if n_done >= self.n_total_steps:
            return False

        current_mean = intermediate_values[n_done - 1]
        n_remaining = self.n_total_steps - n_done

        # --- 轨迹对比模式 ---
        best_trial, best_intermediates = self._get_best_trajectory(study, required_step=n_done - 1)

        if best_trial is not None:
            best_at_step = best_intermediates[n_done - 1]
            gap = best_at_step - current_mean  # 正值 = 落后于 best

            cur_per_step = self._recover_per_step_values(intermediate_values, n_done)
            best_per_step = self._recover_per_step_values(best_intermediates, n_done)

            if len(cur_per_step) == len(best_per_step) >= 2:
                sigma_delta = self._compute_gap_sigma(cur_per_step, best_per_step)

                if sigma_delta < 1e-10:
                    # gap 方差为零：结果确定
                    should_prune = gap > 0
                    self._log_trajectory(
                        trial, n_done, current_mean, best_at_step,
                        gap, sigma_delta, best_trial.number,
                        prob_exceed=0.0 if should_prune else 1.0,
                    )
                    return should_prune

                final_std = sigma_delta * math.sqrt(n_remaining) / self.n_total_steps
                z = gap / final_std
                prob_exceed = 1.0 - norm.cdf(z)
                should_prune = prob_exceed < self.threshold

                self._log_trajectory(
                    trial, n_done, current_mean, best_at_step,
                    gap, sigma_delta, best_trial.number,
                    prob_exceed=prob_exceed,
                )
                return should_prune

        # --- 绝对回退模式 ---
        best_value = self._get_best_value(study)
        if best_value is None:
            return False

        per_step_values = self._recover_per_step_values(intermediate_values, n_done)

        if len(per_step_values) < 2:
            return False
        sigma = float(np.std(per_step_values, ddof=1))

        if sigma < 1e-10:
            return False

        final_std = sigma * math.sqrt(n_remaining) / self.n_total_steps
        z = (best_value - current_mean) / final_std
        prob_exceed = 1.0 - norm.cdf(z)
        should_prune = prob_exceed < self.threshold

        self._log_absolute(
            trial, n_done, current_mean, best_value, sigma,
            prob_exceed=prob_exceed,
        )
        return should_prune

    # ------------------------------------------------------------------
    # Reference methods
    # ------------------------------------------------------------------

    def _get_best_trajectory(
        self, study: optuna.Study, required_step: int
    ) -> tuple[Optional[optuna.trial.FrozenTrial], dict]:
        """获取有 required_step 数据的 best completed trial 及其中间值字典。"""
        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and required_step in t.intermediate_values
        ]
        if not completed:
            return None, {}
        if study.direction == optuna.study.StudyDirection.MAXIMIZE:
            best = max(completed, key=lambda t: t.value)
        else:
            best = min(completed, key=lambda t: t.value)
        return best, best.intermediate_values

    def _get_best_value(self, study: optuna.Study) -> Optional[float]:
        """获取已完成 trials 中的最优值（绝对回退用）。"""
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

    # ------------------------------------------------------------------
    # Statistical methods
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_gap_sigma(cur_per_step: np.ndarray, best_per_step: np.ndarray) -> float:
        """计算逐被试 gap 的标准差（σ_Δ）。"""
        deltas = best_per_step - cur_per_step
        return float(np.std(deltas, ddof=1))

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

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_trajectory(
        self, trial, n_done: int, current_mean: float, best_at_step: float,
        gap: float, sigma_delta: float,
        ref_trial_number: int, *, prob_exceed: float,
    ):
        tag = colored("[Pruner]", Colors.BRIGHT_MAGENTA, bold=True)
        mode = colored("[traj]", Colors.BRIGHT_CYAN)
        gap_sign = "+" if gap >= 0 else ""
        stats = (
            f"step {n_done}/{self.n_total_steps} | "
            f"cur={colored(f'{current_mean:.4f}', Colors.CYAN)} | "
            f"ref={colored(f'{best_at_step:.4f}', Colors.BRIGHT_CYAN)}@T{ref_trial_number} | "
            f"gap={gap_sign}{gap:.4f} | "
            f"\u03c3_\u0394={sigma_delta:.4f}"
        )
        if prob_exceed >= self.threshold:
            p_str = colored(f"P={prob_exceed:.4f}", Colors.BRIGHT_GREEN)
            decision = colored("CONTINUE", Colors.BRIGHT_GREEN, bold=True)
        else:
            p_str = colored(f"P={prob_exceed:.4f}", Colors.BRIGHT_RED)
            decision = colored("PRUNED", Colors.BRIGHT_RED, bold=True)
        print(f"  {tag} T{trial.number} {mode} | {stats} | {p_str} -> {decision}")

    def _log_absolute(
        self, trial, n_done: int, current_mean: float, best_value: float,
        sigma: float, *, prob_exceed: float,
    ):
        tag = colored("[Pruner]", Colors.BRIGHT_MAGENTA, bold=True)
        mode = colored("[abs]", Colors.YELLOW)
        stats = (
            f"step {n_done}/{self.n_total_steps} | "
            f"mean={colored(f'{current_mean:.4f}', Colors.CYAN)} | "
            f"best={colored(f'{best_value:.4f}', Colors.BRIGHT_CYAN)} | "
            f"\u03c3={sigma:.4f}"
        )
        if prob_exceed >= self.threshold:
            p_str = colored(f"P={prob_exceed:.4f}", Colors.BRIGHT_GREEN)
            decision = colored("CONTINUE", Colors.BRIGHT_GREEN, bold=True)
        else:
            p_str = colored(f"P={prob_exceed:.4f}", Colors.BRIGHT_RED)
            decision = colored("PRUNED", Colors.BRIGHT_RED, bold=True)
        print(f"  {tag} T{trial.number} {mode} | {stats} | {p_str} -> {decision}")
