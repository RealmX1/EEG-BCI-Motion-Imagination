"""
Subject and session discovery utilities for FINGER-EEG-BCI dataset.

This module provides functions for discovering available subjects and sessions.
"""

import json
import logging
from pathlib import Path
from typing import List

from ..utils.logging import SectionLogger

logger = logging.getLogger(__name__)
log_load = SectionLogger(logger, 'load')


def get_session_folders_for_split(
    paradigm: str,
    task: str,
    split: str,
) -> List[str]:
    """
    Get the list of session folder names for a given data split.

    This follows the paper's experimental protocol:
    - For binary/ternary tasks:
        - Training: Offline + Online Session 1 (Base + Finetune) + Online Session 2 Base
        - Test: Online Session 2 Finetune (held out completely)
    - For quaternary (4-finger) task:
        - Only Offline data contains 4-finger trials (no Online 4class folders exist)
        - Both train and test splits use Offline data
        - Temporal split is handled by the caller

    Args:
        paradigm: 'imagery' or 'movement'
        task: 'binary', 'ternary', or 'quaternary'
        split: 'train' or 'test'

    Returns:
        List of folder names to include
    """
    # Map paradigm to prefix
    paradigm_prefix = 'Imagery' if paradigm == 'imagery' else 'Movement'
    offline = f'Offline{paradigm_prefix}'

    # Special case: quaternary task only has Offline data
    # No Online 4class folders exist in the dataset
    if task == 'quaternary':
        # Both train and test use Offline data; temporal split is done by caller
        return [offline]

    # Map task to n_class for binary/ternary
    task_to_nclass = {
        'binary': '2class',
        'ternary': '3class',
    }
    n_class = task_to_nclass.get(task, '2class')

    # Build folder names
    online_prefix = f'Online{paradigm_prefix}'

    if split == 'train':
        # Training: Offline + Sess01 Base + Sess01 Finetune + Sess02 Base
        folders = [
            offline,
            f'{online_prefix}_Sess01_{n_class}_Base',
            f'{online_prefix}_Sess01_{n_class}_Finetune',
            f'{online_prefix}_Sess02_{n_class}_Base',
        ]
    elif split == 'test':
        # Test: Sess02 Finetune only
        folders = [
            f'{online_prefix}_Sess02_{n_class}_Finetune',
        ]
    else:
        raise ValueError(f"Unknown split: {split}. Expected 'train' or 'test'.")

    return folders


def discover_available_subjects(
    data_root: str,
    paradigm: str = 'imagery',
    task: str = 'binary',
) -> List[str]:
    """
    Discover subjects that have the required data for both training and testing.

    Args:
        data_root: Root directory containing subject folders
        paradigm: 'imagery' or 'movement'
        task: 'binary', 'ternary', or 'quaternary'

    Returns:
        List of subject IDs (e.g., ['S01', 'S02', ...])
    """
    data_path = Path(data_root)
    subjects = []

    # Get required folders for test split (most restrictive)
    test_folders = get_session_folders_for_split(paradigm, task, 'test')

    for item in sorted(data_path.iterdir()):
        if item.is_dir() and item.name.startswith('S') and item.name[1:].isdigit():
            # Check if subject has required data folders
            # For binary/ternary: Session 2 Finetune
            # For quaternary: Offline data (only source of 4-finger trials)
            has_required_data = all(
                (item / folder).exists() for folder in test_folders
            )
            if has_required_data:
                subjects.append(item.name)

    return subjects


def discover_subjects_from_cache_index(
    cache_index_path: str = ".cache_index.json",
    paradigm: str = 'imagery',
    task: str = 'binary',
) -> List[str]:
    """
    从缓存索引中发现可用的被试。

    此函数读取预处理缓存索引，提取所有符合指定范式和任务的被试 ID。
    适用于数据已预处理但原始数据文件不在本地的场景。

    Args:
        cache_index_path: 缓存索引文件路径（默认：.cache_index.json）
        paradigm: 'imagery' 或 'movement'
        task: 'binary', 'ternary', 或 'quaternary'

    Returns:
        被试 ID 列表（如 ['S01', 'S02', ...]），按字母顺序排序

    Note:
        - Offline 数据的 n_classes 字段为 null，包含所有 4 个手指的数据
        - Binary/Ternary/Quaternary 任务都接受 n_classes == null 的条目
    """
    # 验证 paradigm 参数
    if paradigm not in ['imagery', 'movement']:
        logger.error(f"Invalid paradigm: {paradigm}. Must be 'imagery' or 'movement'")
        return []

    cache_path = Path(cache_index_path)

    # 检查缓存索引是否存在
    if not cache_path.exists():
        logger.warning(f"Cache index not found at {cache_index_path}, returning empty subject list")
        return []

    try:
        # 读取缓存索引
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)

        entries = cache_data.get('entries', {})
        if not entries:
            logger.warning(f"Cache index at {cache_index_path} contains no entries")
            return []

        # 确定任务对应的 n_classes
        task_to_n_classes = {
            'binary': [2, None],      # 接受 2-class 和 offline (null)
            'ternary': [3, None],     # 接受 3-class 和 offline (null)
            'quaternary': [4, None],  # 接受 4-class 和 offline (null)
        }

        if task not in task_to_n_classes:
            logger.error(f"Invalid task: {task}. Must be 'binary', 'ternary', or 'quaternary'")
            return []

        valid_n_classes = task_to_n_classes[task]

        # 提取符合条件的被试
        subjects_set = set()
        for entry_data in entries.values():
            # 检查 paradigm 匹配
            if entry_data.get('subject_task_type') != paradigm:
                continue

            # 检查 n_classes 匹配
            entry_n_classes = entry_data.get('n_classes')
            if entry_n_classes not in valid_n_classes:
                continue

            # 提取被试 ID
            subject_id = entry_data.get('subject')
            if subject_id:
                subjects_set.add(subject_id)

        subjects = sorted(list(subjects_set))

        if not subjects:
            logger.warning(f"No subjects found in cache index for paradigm={paradigm}, task={task}")
        else:
            logger.debug(f"Found {len(subjects)} subjects in cache index: {subjects}")

        return subjects

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse cache index at {cache_index_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error reading cache index: {e}")
        return []
