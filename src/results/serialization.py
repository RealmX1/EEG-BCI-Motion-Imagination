"""
Result serialization utilities.

This module provides functions for converting TrainingResult
objects to/from dictionaries for JSON serialization.
"""

from datetime import datetime
from typing import Optional

from .dataclasses import TrainingResult


def result_to_dict(result: TrainingResult) -> dict:
    """Convert TrainingResult to serializable dict."""
    return {
        'subject_id': result.subject_id,
        'task_type': result.task_type,
        'model_type': result.model_type,
        'best_val_acc': result.best_val_acc,
        'test_acc': result.test_acc,
        'test_acc_majority': result.test_acc_majority,
        'epochs_trained': result.epochs_trained,
        'training_time': result.training_time,
    }


def dict_to_result(d: dict) -> TrainingResult:
    """Convert dict back to TrainingResult."""
    return TrainingResult(
        subject_id=d['subject_id'],
        task_type=d.get('task_type', 'binary'),
        model_type=d['model_type'],
        best_val_acc=d['best_val_acc'],
        test_acc=d['test_acc'],
        test_acc_majority=d['test_acc_majority'],
        epochs_trained=d['epochs_trained'],
        training_time=d['training_time'],
    )


def generate_result_filename(
    prefix: str,
    paradigm: str,
    task: str,
    ext: str = 'json',
    run_tag: Optional[str] = None,
    is_cross_subject: bool = False,
) -> str:
    """
    生成统一格式的结果文件名: {timestamp}_{prefix}_{paradigm}_{task}.{ext}

    对于 cross-subject 结果，文件名格式为:
    {timestamp}_cross-subject_{prefix}_{paradigm}_{task}.{ext}

    Args:
        prefix: 文件名前缀 (如 'comparison', 'combined', 'eegnet', 'cbramod')
        paradigm: 范式 ('imagery' 或 'movement')
        task: 任务类型 ('binary', 'ternary', 'quaternary')
        ext: 文件扩展名 (默认 'json')
        run_tag: 可选的运行标签，如果提供则替代自动生成的时间戳
        is_cross_subject: 是否为跨被试训练结果 (默认 False)

    Returns:
        格式化的文件名，如:
        - '20260203_151711_comparison_imagery_binary.json' (within-subject)
        - '20260203_151711_cross-subject_eegnet_imagery_binary.json' (cross-subject)

    Examples:
        >>> generate_result_filename('eegnet', 'imagery', 'binary', run_tag='20260205_1737')
        '20260205_1737_eegnet_imagery_binary.json'

        >>> generate_result_filename('eegnet', 'imagery', 'binary', run_tag='20260205_1737', is_cross_subject=True)
        '20260205_1737_cross-subject_eegnet_imagery_binary.json'

        >>> generate_result_filename('combined', 'imagery', 'binary', ext='png', is_cross_subject=True)
        '20260205_123456_cross-subject_combined_imagery_binary.png'
    """
    if run_tag:
        timestamp = run_tag
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 添加 cross-subject 前缀
    if is_cross_subject:
        prefix = f"cross-subject_{prefix}"

    parts = [timestamp, prefix, paradigm, task]
    return "_".join(parts) + f".{ext}"
