"""
Milestone checkpoint comparison visualization.

Compares validation combined_score (from training) vs test accuracy
(majority voting, evaluated post-training) at each milestone checkpoint,
revealing whether validation improvements translate to test improvements
or indicate overfitting.
"""

import logging
from typing import Dict, List

import numpy as np

from ..config.constants import MODEL_COLORS
from ..utils.logging import SectionLogger

logger = logging.getLogger(__name__)
log_plot = SectionLogger(logger, 'plot')


def generate_milestone_plot(
    history: Dict,
    milestone_test_results: List[Dict],
    output_path: str,
    subject_id: str = '',
    model_type: str = 'eegnet',
) -> None:
    """
    生成里程碑检查点对比图.

    展示验证 combined_score 与测试准确率在各里程碑 epoch 的对比。

    布局：单面板折线图
    - 背景（灰色淡线）：完整训练过程的 val_combined_score 曲线
    - 蓝色折线+圆点：里程碑 epoch 的 val combined_score
    - 红色折线+方块：里程碑 epoch 的 test accuracy（majority voting）
    - 差异区域着色：红色 = val > test（过拟合信号），绿色 = test >= val

    Args:
        history: 训练历史字典（含 val_combined_score 等列表）
        milestone_test_results: 里程碑测试结果列表，每项含
            epoch, combined_score, test_accuracy
        output_path: 输出文件路径
        subject_id: 被试 ID（用于标题）
        model_type: 模型类型（用于颜色）
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        log_plot.warning("matplotlib not installed, skipping milestone plot")
        return

    if not milestone_test_results:
        log_plot.warning("No milestone results to plot")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # --- Background: full validation combined_score curve ---
    val_scores = history.get('val_combined_score', [])
    if val_scores:
        epochs_full = list(range(1, len(val_scores) + 1))
        ax.plot(epochs_full, val_scores,
                color='gray', alpha=0.25, linewidth=1.5,
                label='Val Combined Score (all epochs)')

    # --- Milestone lines ---
    ms_epochs = [ms['epoch'] for ms in milestone_test_results]
    ms_val_scores = [ms['combined_score'] for ms in milestone_test_results]
    ms_test_accs = [ms['test_accuracy'] for ms in milestone_test_results]

    color_val = MODEL_COLORS.get(model_type, '#2E86AB')

    # Validation combined_score at milestones
    ax.plot(ms_epochs, ms_val_scores,
            color=color_val, linewidth=2, marker='o', markersize=8,
            label='Val Combined Score (milestones)', zorder=5)

    # Test accuracy at milestones
    ax.plot(ms_epochs, ms_test_accs,
            color='#E94F37', linewidth=2, marker='s', markersize=8,
            label='Test Accuracy (milestones)', zorder=5)

    # Annotate each milestone point
    for ep, val, test in zip(ms_epochs, ms_val_scores, ms_test_accs):
        ax.annotate(f'{val:.3f}', (ep, val),
                    textcoords='offset points', xytext=(0, 10),
                    ha='center', fontsize=7, color=color_val)
        ax.annotate(f'{test:.3f}', (ep, test),
                    textcoords='offset points', xytext=(0, -14),
                    ha='center', fontsize=7, color='#E94F37')

    # Shaded region between val and test (overfitting indicator)
    if len(ms_epochs) > 1:
        ax.fill_between(ms_epochs, ms_val_scores, ms_test_accs,
                        alpha=0.08, color='red',
                        where=[v > t for v, t in zip(ms_val_scores, ms_test_accs)])
        ax.fill_between(ms_epochs, ms_val_scores, ms_test_accs,
                        alpha=0.08, color='green',
                        where=[v <= t for v, t in zip(ms_val_scores, ms_test_accs)])

    # --- Formatting ---
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Accuracy / Score', fontsize=11)
    title = 'Milestone Checkpoint Analysis'
    if subject_id:
        title += f' - {subject_id}'
    title += f' ({model_type.upper()})'
    ax.set_title(title, fontsize=13)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Y-axis range based on data
    all_values = ms_val_scores + ms_test_accs
    if val_scores:
        all_values.extend(val_scores)
    y_min = max(0, min(all_values) - 0.05)
    y_max = min(1.05, max(all_values) + 0.05)
    ax.set_ylim([y_min, y_max])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    log_plot.info(f"Milestone plot saved: {output_path}")
    plt.close()
