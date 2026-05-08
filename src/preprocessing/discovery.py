"""
Subject and session discovery utilities for FINGER-EEG-BCI dataset.

This module provides functions for discovering available subjects and sessions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

from ..config.constants import DEFAULT_CACHE_INDEX_PATH, DEFAULT_CACHE_INDEX_PATH_MOVEMENT
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
    online_prefix = f'Online{paradigm_prefix}'

    # Unified task: combine ALL available session types for training
    # Test returns empty — per-subtask evaluation is handled separately
    if task == 'unified':
        if split == 'train':
            return [
                offline,
                f'{online_prefix}_Sess01_2class_Base',
                f'{online_prefix}_Sess01_2class_Finetune',
                f'{online_prefix}_Sess02_2class_Base',
                f'{online_prefix}_Sess01_3class_Base',
                f'{online_prefix}_Sess01_3class_Finetune',
                f'{online_prefix}_Sess02_3class_Base',
            ]
        else:
            # Test: per-subtask evaluation loads each subtask's test set independently
            return []

    # Special case: quaternary task only has Offline data
    # No Online 4class folders exist in the dataset
    if task == 'quaternary':
        # Train uses Offline; test must be a holdout slice of Offline taken
        # from the train dataset (callers use temporal_split_with_offline_test
        # to reserve the holdout). Returning [] here prevents accidentally
        # reloading the full Offline pool as a "test set" — that path would
        # overlap with the train pool and inflate accuracy.
        if split == 'test':
            return []
        return [offline]

    # Map task to n_class for binary/ternary
    task_to_nclass = {
        'binary': '2class',
        'ternary': '3class',
    }
    n_class = task_to_nclass.get(task, '2class')

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

    # Unified task: check train folders instead (test is empty, evaluated per-subtask)
    if task == 'unified':
        check_folders = get_session_folders_for_split(paradigm, task, 'train')
    else:
        check_folders = test_folders

    for item in sorted(data_path.iterdir()):
        if item.is_dir() and item.name.startswith('S') and item.name[1:].isdigit():
            # Check if subject has required data folders
            # For binary/ternary: Session 2 Finetune
            # For quaternary: Offline data (only source of 4-finger trials)
            # For unified: at least some train folders exist
            has_required_data = any(
                (item / folder).exists() for folder in check_folders
            ) if task == 'unified' else all(
                (item / folder).exists() for folder in check_folders
            )
            if has_required_data:
                subjects.append(item.name)

    return subjects


def discover_extra_session_subjects_from_cache(
    paradigm: str = 'imagery',
    task: str = 'binary',
    cache_index_path: str = None,
) -> Dict[str, List[int]]:
    """
    从缓存索引发现拥有额外在线 session（Sess03+）的被试。

    用于 --cache-only 模式，原始数据文件不在本地时。

    Args:
        paradigm: 'imagery' 或 'movement'
        task: 'binary' 或 'ternary'
        cache_index_path: 缓存索引文件路径（None 时根据 paradigm 自动选择）

    Returns:
        字典 {subject_id: [available_session_numbers]}
    """
    import re

    if cache_index_path is None:
        cache_index_path = (
            DEFAULT_CACHE_INDEX_PATH_MOVEMENT if paradigm == 'movement'
            else DEFAULT_CACHE_INDEX_PATH
        )
    cache_path = Path(cache_index_path)
    if not cache_path.exists():
        logger.warning(f"Cache index not found: {cache_index_path}")
        return {}

    task_to_nclass = {'binary': '2class', 'ternary': '3class'}
    n_class = task_to_nclass.get(task)
    if n_class is None:
        logger.warning(f"Extra sessions experiment only supports binary/ternary, got: {task}")
        return {}

    with open(cache_path, 'r', encoding='utf-8') as f:
        cache_data = json.load(f)

    entries = cache_data.get('entries', {})
    result: Dict[str, set] = {}

    for entry in entries.values():
        subj = entry.get('subject', '')
        folder = entry.get('session_folder', '')
        entry_paradigm = entry.get('subject_task_type', '')
        n_classes = entry.get('n_classes')

        if entry_paradigm != paradigm:
            continue
        # Must have the right n_classes (2 for binary, 3 for ternary)
        expected_n = 2 if task == 'binary' else 3
        if n_classes != expected_n:
            continue

        # Extract session number from folder name
        m = re.search(r'Sess0(\d+)', folder)
        if not m:
            continue
        sess_num = int(m.group(1))
        if sess_num < 3:
            continue

        # Must be a Finetune folder (used as test set)
        if 'Finetune' not in folder:
            continue

        if subj not in result:
            result[subj] = set()
        result[subj].add(sess_num)

    final = {s: sorted(v) for s, v in result.items()}
    if final:
        logger.info(f"Found {len(final)} subjects with extra sessions in cache: "
                    f"{', '.join(f'{s}(Sess{min(v)}-{max(v)})' for s, v in sorted(final.items()))}")
    else:
        logger.warning(f"No subjects with extra sessions found in cache for {paradigm}/{task}")

    return final


def discover_extra_session_subjects(
    data_root: str,
    paradigm: str = 'imagery',
    task: str = 'binary',
    cache_only: bool = False,
    cache_index_path: str = None,
) -> Dict[str, List[int]]:
    """
    发现拥有额外在线 session（Sess03+）的被试。

    cache_only=True 时从缓存索引发现（用于原始文件不在本地的情况）。
    cache_only=False 时扫描文件系统（检查 Base+Finetune 文件夹是否存在）。

    Args:
        data_root: 数据根目录（cache_only=True 时可忽略）
        paradigm: 'imagery' 或 'movement'
        task: 'binary' 或 'ternary'
        cache_only: 若 True，从 cache_index_path 发现被试
        cache_index_path: 缓存索引路径（None 时根据 paradigm 自动选择，cache_only=True 时使用）

    Returns:
        字典 {subject_id: [available_session_numbers]}
        例: {'S02': [3, 4, 5], 'S03': [3, 4, 5]}
    """
    if cache_only:
        return discover_extra_session_subjects_from_cache(
            paradigm=paradigm, task=task, cache_index_path=cache_index_path
        )

    data_path = Path(data_root)
    paradigm_prefix = 'Imagery' if paradigm == 'imagery' else 'Movement'
    online_prefix = f'Online{paradigm_prefix}'

    task_to_nclass = {'binary': '2class', 'ternary': '3class'}
    n_class = task_to_nclass.get(task)
    if n_class is None:
        logger.warning(f"Extra sessions experiment only supports binary/ternary, got: {task}")
        return {}

    result = {}
    for item in sorted(data_path.iterdir()):
        if not (item.is_dir() and item.name.startswith('S') and item.name[1:].isdigit()):
            continue

        available_sessions = []
        for sess_num in range(3, 10):  # Sess03 through Sess09
            base_folder = f'{online_prefix}_Sess0{sess_num}_{n_class}_Base'
            finetune_folder = f'{online_prefix}_Sess0{sess_num}_{n_class}_Finetune'
            if (item / base_folder).exists() and (item / finetune_folder).exists():
                available_sessions.append(sess_num)

        if available_sessions:
            result[item.name] = available_sessions

    if result:
        logger.info(f"Found {len(result)} subjects with extra sessions: "
                     f"{', '.join(f'{s}(Sess{min(v)}-{max(v)})' for s, v in result.items())}")
    else:
        logger.warning(f"No subjects found with extra sessions for {paradigm}/{task}")

    return result


def get_progressive_session_folders(
    paradigm: str,
    task: str,
    up_to_session: int,
) -> Dict[str, List[str]]:
    """
    为渐进式数据实验生成 session folder 列表。

    训练集逐步添加前序 session 的全部数据（含 Finetune），
    测试集为目标 session 的 Finetune。

    Args:
        paradigm: 'imagery' 或 'movement'
        task: 'binary' 或 'ternary'
        up_to_session: 目标 session 编号 (3, 4, 5, ...)

    Returns:
        Dict with 'train' and 'test' folder lists.

    Example (binary, imagery, up_to_session=4):
        train: [OfflineImagery,
                OnlineImagery_Sess01_2class_Base, _Finetune,
                OnlineImagery_Sess02_2class_Base, _Finetune,
                OnlineImagery_Sess03_2class_Base, _Finetune,
                OnlineImagery_Sess04_2class_Base]
        test:  [OnlineImagery_Sess04_2class_Finetune]
    """
    if up_to_session < 3:
        raise ValueError(f"up_to_session must be >= 3, got {up_to_session}")

    paradigm_prefix = 'Imagery' if paradigm == 'imagery' else 'Movement'
    offline = f'Offline{paradigm_prefix}'
    online_prefix = f'Online{paradigm_prefix}'

    task_to_nclass = {'binary': '2class', 'ternary': '3class'}
    n_class = task_to_nclass.get(task)
    if n_class is None:
        raise ValueError(f"Progressive sessions only supports binary/ternary, got: {task}")

    # Start with standard training set
    train_folders = [
        offline,
        f'{online_prefix}_Sess01_{n_class}_Base',
        f'{online_prefix}_Sess01_{n_class}_Finetune',
    ]

    # Add ALL data (Base + Finetune) for sessions 2 through (up_to_session - 1)
    for sess in range(2, up_to_session):
        train_folders.append(f'{online_prefix}_Sess0{sess}_{n_class}_Base')
        train_folders.append(f'{online_prefix}_Sess0{sess}_{n_class}_Finetune')

    # Add only Base for target session
    train_folders.append(f'{online_prefix}_Sess0{up_to_session}_{n_class}_Base')

    # Test: Finetune of target session
    test_folders = [f'{online_prefix}_Sess0{up_to_session}_{n_class}_Finetune']

    return {'train': train_folders, 'test': test_folders}


def get_all_extra_session_folders(
    paradigm: str,
    task: str,
    available_sessions: List[int],
) -> List[str]:
    """
    获取包含所有额外 session 的完整 folder 列表。

    用于 fixed_combined 策略：加载全部数据到单一 dataset，
    再通过 index 控制 train/val/test 划分。

    Args:
        paradigm: 'imagery' 或 'movement'
        task: 'binary' 或 'ternary'
        available_sessions: 额外 session 编号列表 (e.g., [3, 4, 5])

    Returns:
        所有 session folders 的有序列表
    """
    paradigm_prefix = 'Imagery' if paradigm == 'imagery' else 'Movement'
    offline = f'Offline{paradigm_prefix}'
    online_prefix = f'Online{paradigm_prefix}'

    task_to_nclass = {'binary': '2class', 'ternary': '3class'}
    n_class = task_to_nclass.get(task)
    if n_class is None:
        raise ValueError(f"Only binary/ternary supported, got: {task}")

    folders = [
        offline,
        f'{online_prefix}_Sess01_{n_class}_Base',
        f'{online_prefix}_Sess01_{n_class}_Finetune',
        f'{online_prefix}_Sess02_{n_class}_Base',
        f'{online_prefix}_Sess02_{n_class}_Finetune',
    ]
    for sess in sorted(available_sessions):
        folders.append(f'{online_prefix}_Sess0{sess}_{n_class}_Base')
        folders.append(f'{online_prefix}_Sess0{sess}_{n_class}_Finetune')

    return folders


def get_progressive_session_folders_fixed_sess02(
    paradigm: str,
    task: str,
    up_to_session: int,
) -> Dict[str, List[str]]:
    """
    为 fixed_sess02 策略生成 session folder 列表。

    测试集始终为 Sess02_Finetune（跨所有 step 不变）。
    训练集逐步添加 Sess03-05 的 Base + Finetune，但不包含 Sess02_Finetune。

    Args:
        paradigm: 'imagery' 或 'movement'
        task: 'binary' 或 'ternary'
        up_to_session: 目标 session 编号 (3, 4, 5, ...)

    Returns:
        Dict with 'train' and 'test' folder lists.

    Example (binary, imagery, up_to_session=4):
        train: [OfflineImagery,
                OnlineImagery_Sess01_2class_Base, _Finetune,
                OnlineImagery_Sess02_2class_Base,
                OnlineImagery_Sess03_2class_Base, _Finetune,
                OnlineImagery_Sess04_2class_Base, _Finetune]
        test:  [OnlineImagery_Sess02_2class_Finetune]
    """
    if up_to_session < 3:
        raise ValueError(f"up_to_session must be >= 3, got {up_to_session}")

    paradigm_prefix = 'Imagery' if paradigm == 'imagery' else 'Movement'
    offline = f'Offline{paradigm_prefix}'
    online_prefix = f'Online{paradigm_prefix}'

    task_to_nclass = {'binary': '2class', 'ternary': '3class'}
    n_class = task_to_nclass.get(task)
    if n_class is None:
        raise ValueError(f"Only binary/ternary supported, got: {task}")

    # Standard baseline training set (Sess02_FT excluded — it's the test set)
    train_folders = [
        offline,
        f'{online_prefix}_Sess01_{n_class}_Base',
        f'{online_prefix}_Sess01_{n_class}_Finetune',
        f'{online_prefix}_Sess02_{n_class}_Base',
    ]

    # Add Sess03 through up_to_session: both Base + Finetune go to training
    for sess in range(3, up_to_session + 1):
        train_folders.append(f'{online_prefix}_Sess0{sess}_{n_class}_Base')
        train_folders.append(f'{online_prefix}_Sess0{sess}_{n_class}_Finetune')

    # Test: always Sess02 Finetune
    test_folders = [f'{online_prefix}_Sess02_{n_class}_Finetune']

    return {'train': train_folders, 'test': test_folders}


def discover_subjects_from_cache_index(
    paradigm: str = 'imagery',
    task: str = 'binary',
) -> List[str]:
    """
    从缓存索引中发现可用的被试。

    此函数读取预处理缓存索引，提取所有符合指定范式和任务的被试 ID。
    适用于数据已预处理但原始数据文件不在本地的场景。

    Args:
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

    _cache_index_path = (
        DEFAULT_CACHE_INDEX_PATH_MOVEMENT if paradigm == 'movement'
        else DEFAULT_CACHE_INDEX_PATH
    )
    cache_path = Path(_cache_index_path)

    # 检查缓存索引是否存在
    if not cache_path.exists():
        logger.warning(f"Cache index not found at {_cache_index_path}, returning empty subject list")
        return []

    try:
        # 读取缓存索引
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)

        entries = cache_data.get('entries', {})
        if not entries:
            logger.warning(f"Cache index at {_cache_index_path} contains no entries")
            return []

        # 确定任务对应的 n_classes
        task_to_n_classes = {
            'binary': [2, None],      # 接受 2-class 和 offline (null)
            'ternary': [3, None],     # 接受 3-class 和 offline (null)
            'quaternary': [4, None],  # 接受 4-class 和 offline (null)
            'unified': [2, 3, 4, None],  # 接受所有 n_classes
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
        logger.error(f"Failed to parse cache index at {_cache_index_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error reading cache index: {e}")
        return []
