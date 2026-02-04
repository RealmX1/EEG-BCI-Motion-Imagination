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
    run_tag: Optional[str] = None
) -> str:
    """
    生成统一格式的结果文件名: {timestamp}_{prefix}_{paradigm}_{task}.{ext}

    Args:
        prefix: 文件名前缀 (如 'comparison', 'eegnet', 'finetune_backbone')
        paradigm: 范式 ('imagery' 或 'movement')
        task: 任务类型 ('binary', 'ternary', 'quaternary')
        ext: 文件扩展名 (默认 'json')
        run_tag: 可选的运行标签，如果提供则替代自动生成的时间戳

    Returns:
        格式化的文件名，如 '20260203_151711_comparison_imagery_binary.json'
    """
    if run_tag:
        timestamp = run_tag
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parts = [timestamp, prefix, paradigm, task]
    return "_".join(parts) + f".{ext}"
