"""
Training results cache management.

This module provides functions for caching and retrieving
training results to/from JSON files.
"""

import json
import logging
import os
import tempfile
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class SelectionStrategy(Enum):
    """历史数据选择策略."""
    NEWEST = "newest"           # 最新时间戳（用于训练恢复）
    BEST_ACCURACY = "best_acc"  # 最高准确率（用于图表生成）

from ..config.constants import CACHE_FILENAME, CACHE_FILENAME_WITH_TAG, CacheType, PARADIGM_CONFIG
from ..utils.logging import SectionLogger
from .dataclasses import ComparisonResult, TrainingResult, PlotDataSource
from .serialization import (
    dict_to_result, result_to_dict, generate_result_filename,
    cross_subject_result_to_training_results,
)
from .statistics import compute_model_statistics

logger = logging.getLogger(__name__)
log_cache = SectionLogger(logger, 'cache')


# ============================================================================
# Helper Functions (Reusable abstractions)
# ============================================================================

def _extract_subject_accuracy(subject_data: dict) -> float:
    """
    从被试数据字典中提取测试准确率.

    支持多种字段名以兼容不同格式:
    - test_acc_majority (首选)
    - test_accuracy_majority
    - test_accuracy
    - test_acc

    Args:
        subject_data: 被试数据字典

    Returns:
        测试准确率，找不到时返回 0
    """
    for key in ('test_acc_majority', 'test_accuracy_majority', 'test_accuracy', 'test_acc'):
        val = subject_data.get(key)
        if val is not None:
            return val
    return 0


def _compute_mean_accuracy(subjects_data: list) -> float:
    """
    计算被试列表的平均准确率.

    Args:
        subjects_data: 被试数据字典列表

    Returns:
        平均准确率，空列表返回 0.0
    """
    if not subjects_data:
        return 0.0
    accs = [_extract_subject_accuracy(s) for s in subjects_data]
    return float(np.mean(accs)) if accs else 0.0


def _filter_cache_type(files: List[Path], cache_type: str) -> List[Path]:
    """过滤 glob 结果，排除子串误匹配（如 comparison_cache 匹配到 transfer_comparison_cache）."""
    if cache_type == CacheType.WITHIN_SUBJECT:
        return [f for f in files if 'transfer_' not in f.name]
    return files


def get_cache_path(
    output_dir: str,
    paradigm: str,
    task: str,
    run_tag: Optional[str] = None,
    cache_type: str = CacheType.WITHIN_SUBJECT,
) -> Path:
    """Get path to cache file.

    Args:
        output_dir: 结果目录
        paradigm: 范式
        task: 任务类型
        run_tag: 可选运行标签
        cache_type: 缓存类型 (CacheType.WITHIN_SUBJECT 或 CacheType.TRANSFER)
    """
    if run_tag:
        filename = f'{run_tag}_{cache_type}_{paradigm}_{task}.json'
    else:
        filename = f'{cache_type}_{paradigm}_{task}.json'
    return Path(output_dir) / filename


def find_latest_cache(
    output_dir: str,
    paradigm: str,
    task: str,
    cache_type: str = CacheType.WITHIN_SUBJECT,
) -> Optional[Path]:
    """Find the latest cache file (tagged or untagged) for the given paradigm and task."""
    results_dir = Path(output_dir)
    if not results_dir.exists():
        return None

    # Pattern matches both tagged and untagged cache files
    pattern = f'*{cache_type}_{paradigm}_{task}.json'
    cache_files = _filter_cache_type(list(results_dir.glob(pattern)), cache_type)

    if not cache_files and cache_type == CacheType.WITHIN_SUBJECT:
        # Fallback: try old format without paradigm (only for within-subject)
        old_pattern = f'*comparison_cache_{task}.json'
        cache_files = list(results_dir.glob(old_pattern))

    if not cache_files:
        return None

    # Sort by modification time, newest first
    def safe_mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    cache_files.sort(key=safe_mtime, reverse=True)
    return cache_files[0]


def find_cache_by_tag(
    output_dir: str,
    paradigm: str,
    task: str,
    tag_substring: Optional[str] = None,
    cache_type: str = CacheType.WITHIN_SUBJECT,
) -> Optional[Tuple[Path, str]]:
    """
    根据时间戳子串查找缓存文件.

    Args:
        output_dir: 结果目录
        paradigm: 范式 ('imagery' 或 'movement')
        task: 任务类型 ('binary', 'ternary', 'quaternary')
        tag_substring: 时间戳子串（如 "20260205"）。如果为 None，返回最新的缓存。
        cache_type: 缓存类型 (CacheType.WITHIN_SUBJECT 或 CacheType.TRANSFER)

    Returns:
        Tuple of (cache_path, run_tag) 或 None（找不到时）

    Examples:
        # 恢复最新运行
        >>> find_cache_by_tag('results', 'imagery', 'binary')
        (Path('results/20260205_1430_comparison_cache_imagery_binary.json'), '20260205_1430')

        # 恢复特定日期的运行
        >>> find_cache_by_tag('results', 'imagery', 'binary', '20260205')
        (Path('results/20260205_1430_comparison_cache_imagery_binary.json'), '20260205_1430')
    """
    results_dir = Path(output_dir)
    if not results_dir.exists():
        return None

    # 搜索所有可能的缓存文件
    pattern = f'*{cache_type}_{paradigm}_{task}.json'
    cache_files = _filter_cache_type(list(results_dir.glob(pattern)), cache_type)

    if not cache_files:
        return None

    suffix = f'_{cache_type}_{paradigm}_{task}.json'

    def extract_run_tag(path: Path) -> Optional[str]:
        """从文件名中提取 run_tag."""
        name = path.name
        if name.endswith(suffix):
            tag = name[:-len(suffix)]
            return tag if tag else None
        return None

    def safe_mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    # 构建 (path, tag, mtime) 列表
    candidates = []
    for path in cache_files:
        tag = extract_run_tag(path)
        mtime = safe_mtime(path)
        candidates.append((path, tag, mtime))

    # 如果提供了 tag_substring，过滤匹配的文件
    if tag_substring:
        candidates = [(p, t, m) for p, t, m in candidates if t and tag_substring in t]

    if not candidates:
        return None

    # 按修改时间排序（最新优先）
    candidates.sort(key=lambda x: x[2], reverse=True)

    path, tag, _ = candidates[0]
    log_cache.debug(f"Found cache by tag: {path.name} (tag={tag})")
    return (path, tag) if tag else (path, '')


def load_cache(
    output_dir: str,
    paradigm: str,
    task: str,
    run_tag: Optional[str] = None,
    find_latest: bool = False,
    cache_type: str = CacheType.WITHIN_SUBJECT,
) -> Tuple[Dict[str, Dict[str, dict]], Dict]:
    """Load cached results with backward compatibility for old cache format.

    Args:
        output_dir: Directory containing cache files
        paradigm: 'imagery' or 'movement'
        task: 'binary', 'ternary', or 'quaternary'
        run_tag: Optional tag for specific run (e.g., '20260110_2317')
        find_latest: If True and run_tag is None, find the latest cache file
        cache_type: 缓存类型 (CacheType.WITHIN_SUBJECT 或 CacheType.TRANSFER)

    Returns:
        Tuple of (results_dict, metadata_dict) where metadata contains:
        - run_tag: Optional[str] - the run tag from cache
        - wandb_groups: Dict[str, str] - model_type -> wandb_group mapping
    """
    empty_metadata = {
        'run_tag': None,
        'wandb_groups': {},
        'metadata': {
            'timestamp': None,
            'n_subjects': None,
            'is_complete': False,
        },
        'summary': None,
        'comparison': None,
    }

    def extract_metadata(data: dict) -> Dict:
        """Extract metadata from cache data with backward compatibility.

        Handles both old format (flat structure) and new format (with metadata dict).
        """
        # New format: has 'metadata' field
        if 'metadata' in data:
            return {
                'run_tag': data.get('metadata', {}).get('run_tag'),
                'wandb_groups': data.get('wandb_groups', {}),
                'metadata': data.get('metadata', {}),
                'summary': data.get('summary'),
                'comparison': data.get('comparison'),
            }

        # Old format: migrate to new format
        return {
            'run_tag': data.get('run_tag'),
            'wandb_groups': data.get('wandb_groups', {}),
            'metadata': {
                'paradigm': data.get('paradigm'),
                'task': data.get('task'),
                'run_tag': data.get('run_tag'),
                'timestamp': data.get('last_updated'),  # Fallback to last_updated
                'n_subjects': None,
                'is_complete': False,
            },
            'summary': None,
            'comparison': None,
        }

    if find_latest and not run_tag:
        latest_cache = find_latest_cache(output_dir, paradigm, task, cache_type=cache_type)
        if latest_cache:
            try:
                with open(latest_cache, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                log_cache.info(f"Loaded latest cache: {latest_cache.name}")
                return data.get('results', {}), extract_metadata(data)
            except Exception as e:
                log_cache.warning(f"Failed to load latest cache: {e}")
        return {}, empty_metadata

    cache_path = get_cache_path(output_dir, paradigm, task, run_tag, cache_type=cache_type)
    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            log_cache.info(f"Loaded from {cache_path.name}")
            return data.get('results', {}), extract_metadata(data)
        except Exception as e:
            log_cache.warning(f"Failed to load: {e}")

    # Fallback: try old format without paradigm (backward compatibility)
    if not run_tag:
        old_format_path = Path(output_dir) / f'comparison_cache_{task}.json'
        if old_format_path.exists():
            try:
                with open(old_format_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                log_cache.info(f"Loaded legacy {old_format_path} -> new: {cache_path.name}")
                return data.get('results', {}), extract_metadata(data)
            except Exception as e:
                log_cache.warning(f"Failed to load legacy: {e}")

    return {}, empty_metadata


def save_cache(
    output_dir: str,
    paradigm: str,
    task: str,
    results: Dict[str, Dict[str, dict]],
    run_tag: Optional[str] = None,
    wandb_groups: Optional[Dict[str, str]] = None,
    summary: Optional[Dict[str, Dict[str, float]]] = None,
    comparison: Optional[Dict[str, Any]] = None,
    n_subjects: Optional[int] = None,
    is_complete: bool = False,
    existing_timestamp: Optional[str] = None,
    cache_type: str = CacheType.WITHIN_SUBJECT,
    extra_metadata: Optional[Dict] = None,
) -> Path:
    """Save results to cache using atomic write to prevent corruption.

    Args:
        output_dir: Directory to save cache files
        paradigm: 'imagery' or 'movement'
        task: 'binary', 'ternary', or 'quaternary'
        results: Training results dict
        run_tag: Optional tag for specific run
        wandb_groups: Optional dict mapping model_type to wandb_group name
        summary: Optional dict with model summaries (mean, std, min, max)
        comparison: Optional dict with model comparison statistics
        n_subjects: Optional total number of subjects
        is_complete: Whether all training is complete (default: False)
        existing_timestamp: Optional existing timestamp to preserve (for updates)
        cache_type: 缓存类型 (CacheType.WITHIN_SUBJECT 或 CacheType.TRANSFER)
        extra_metadata: 额外元数据字典，合并到 metadata 中（如 transfer_config）

    Returns:
        Path to saved cache file
    """
    cache_path = get_cache_path(output_dir, paradigm, task, run_tag, cache_type=cache_type)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Preserve existing timestamp or create new one
    timestamp = existing_timestamp or datetime.now().isoformat()

    # Create/update metadata structure
    metadata = {
        'paradigm': paradigm,
        'task': task,
        'run_tag': run_tag,
        'timestamp': timestamp,
        'n_subjects': n_subjects,
        'is_complete': is_complete,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    data = {
        'metadata': metadata,
        'wandb_groups': wandb_groups or {},
        'last_updated': datetime.now().isoformat(),
        'results': results,
    }

    # Add summary and comparison if provided
    if summary is not None:
        data['summary'] = summary
    if comparison is not None:
        data['comparison'] = comparison

    # Atomic write: write to temp file, then rename
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=cache_path.parent,
            suffix='.tmp',
            delete=False
        ) as f:
            json.dump(data, f, indent=2)
            temp_path = f.name
        os.replace(temp_path, cache_path)
    except Exception as e:
        # Fallback to direct write if atomic fails
        log_cache.warning(f"Atomic write failed, fallback: {e}")
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    return cache_path


def _collect_compatible_files(
    output_dir: str,
    paradigm: str,
    task: str,
    current_subjects: List[str],
    current_model: str,
    require_complete: bool = True,
) -> List[Dict]:
    """
    收集所有兼容的历史结果文件（核心搜索函数）.

    兼容性条件:
    1. 必须包含两个模型的结果
    2. 另一个模型的被试集合必须覆盖当前被试集合
    3. (可选) 必须是完整对比 (is_complete=True)

    Args:
        output_dir: 结果目录路径
        paradigm: 范式 ('imagery' 或 'movement')
        task: 任务类型 ('binary', 'ternary', 'quaternary')
        current_subjects: 当前运行的被试列表
        current_model: 当前运行的模型 ('eegnet' 或 'cbramod')
        require_complete: 是否要求 is_complete=True

    Returns:
        兼容文件信息列表，每个元素包含:
        {
            'file_path': Path,
            'timestamp': str,
            'data': dict (已转换为标准格式),
            'models_data': dict,
            'mean_accuracy': dict (各模型平均准确率)
        }
    """
    results_dir = Path(output_dir)
    if not results_dir.exists():
        return []

    # 搜索所有可能的结果文件
    cache_patterns = [
        f'*comparison_cache_{paradigm}_{task}.json',  # 新格式（带 tag）
        f'comparison_cache_{paradigm}_{task}.json',   # 新格式（不带 tag）
        f'*comparison_{paradigm}_{task}*.json',       # 旧格式
    ]

    all_files = []
    for pattern in cache_patterns:
        all_files.extend(results_dir.glob(pattern))

    # 去重并排除 transfer / cross-subject 缓存的误匹配
    all_files = list(set(all_files))
    all_files = [f for f in all_files if 'transfer_' not in f.name and 'cross-subject' not in f.name]

    if not all_files:
        return []

    other_model = 'cbramod' if current_model == 'eegnet' else 'eegnet'
    current_subjects_set = set(current_subjects)

    compatible_files = []
    for file_path in all_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            # 检查是否是新格式缓存文件
            if 'results' in raw_data and isinstance(raw_data['results'], dict):
                # 新格式：检查 is_complete
                metadata = raw_data.get('metadata', {})
                if require_complete and not metadata.get('is_complete', False):
                    continue  # 跳过未完成的训练
                timestamp = metadata.get('timestamp')
                # 转换为标准格式
                data = _convert_cache_to_comparison_format(raw_data)
            else:
                # 旧格式
                timestamp = raw_data.get('metadata', {}).get('timestamp')
                data = raw_data

            if not timestamp:
                continue

            models_data = data.get('models', {})

            # 检查是否包含两个模型
            if 'eegnet' not in models_data or 'cbramod' not in models_data:
                continue

            # 提取另一个模型的被试集合
            other_subjects_data = models_data[other_model].get('subjects', [])
            other_subjects_set = {s['subject_id'] for s in other_subjects_data}

            # 检查是否为超集或相同集合
            if not (current_subjects_set <= other_subjects_set):
                continue

            # 计算各模型平均准确率
            mean_accuracy = {}
            for model_type in ['eegnet', 'cbramod']:
                subjects_data = models_data.get(model_type, {}).get('subjects', [])
                mean_accuracy[model_type] = _compute_mean_accuracy(subjects_data)

            compatible_files.append({
                'file_path': file_path,
                'timestamp': timestamp,
                'data': data,
                'models_data': models_data,
                'mean_accuracy': mean_accuracy,
            })

        except (json.JSONDecodeError, KeyError, OSError) as e:
            log_cache.debug(f"Skipping {file_path.name}: {e}")
            continue

    return compatible_files


def _select_by_strategy(
    compatible_files: List[Dict],
    strategy: SelectionStrategy,
    target_model: str,
) -> Optional[Dict]:
    """
    根据策略从兼容文件中选择最佳文件.

    Args:
        compatible_files: _collect_compatible_files() 返回的兼容文件列表
        strategy: 选择策略 (NEWEST 或 BEST_ACCURACY)
        target_model: 目标模型（用于 BEST_ACCURACY 策略）

    Returns:
        选中的文件信息字典，或 None
    """
    if not compatible_files:
        return None

    if strategy == SelectionStrategy.NEWEST:
        # 按时间戳排序（最新优先）
        compatible_files.sort(key=lambda x: x['timestamp'], reverse=True)
        return compatible_files[0]

    elif strategy == SelectionStrategy.BEST_ACCURACY:
        # 按目标模型的平均准确率排序（最高优先）
        compatible_files.sort(
            key=lambda x: x['mean_accuracy'].get(target_model, 0.0),
            reverse=True
        )
        selected = compatible_files[0]
        best_acc = selected['mean_accuracy'].get(target_model, 0.0)
        log_cache.debug(
            f"Selected by best accuracy: {selected['file_path'].name} "
            f"({target_model} mean={best_acc:.4f})"
        )
        return selected

    return None


def find_compatible_historical_results(
    output_dir: str,
    paradigm: str,
    task: str,
    current_subjects: List[str],
    current_model: str,
    strategy: SelectionStrategy = SelectionStrategy.BEST_ACCURACY,
) -> Optional[Dict]:
    """
    搜索与当前运行兼容的历史完整对比结果.

    兼容性条件:
    1. 必须是完整对比文件（包含两个模型的结果）
    2. 另一个模型（非当前运行的模型）的被试集合必须覆盖当前被试集合

    选择策略:
    - BEST_ACCURACY (默认): 选择目标模型平均准确率最高的历史运行（用于图表生成）
    - NEWEST: 选择最新的兼容运行（用于训练恢复等场景）

    Args:
        output_dir: 结果目录路径
        paradigm: 范式 ('imagery' 或 'movement')
        task: 任务类型 ('binary', 'ternary', 'quaternary')
        current_subjects: 当前运行的被试列表
        current_model: 当前运行的模型 ('eegnet' 或 'cbramod')
        strategy: 选择策略 (默认 BEST_ACCURACY)

    Returns:
        包含历史数据的字典:
        {
            'source_file': str,           # 来源文件路径
            'timestamp': str,             # 历史运行时间戳
            'eegnet': {'subjects': [...], 'summary': {...}},
            'cbramod': {'subjects': [...], 'summary': {...}},
            'other_model': str,           # 另一个模型名称
            'mean_accuracy': dict,        # 各模型平均准确率
        }
        如果找不到兼容结果，返回 None
    """
    other_model = 'cbramod' if current_model == 'eegnet' else 'eegnet'

    # 收集所有兼容文件
    compatible_files = _collect_compatible_files(
        output_dir=output_dir,
        paradigm=paradigm,
        task=task,
        current_subjects=current_subjects,
        current_model=current_model,
        require_complete=True,
    )

    if not compatible_files:
        return None

    # 根据策略选择最佳文件
    # 对于图表生成，使用 "另一个模型" 的准确率作为选择依据
    selected = _select_by_strategy(compatible_files, strategy, target_model=other_model)

    if not selected:
        return None

    file_path = selected['file_path']
    models_data = selected['models_data']

    strategy_desc = "best accuracy" if strategy == SelectionStrategy.BEST_ACCURACY else "newest"
    log_cache.info(
        f"Found compatible historical data ({strategy_desc}): {file_path.name}"
    )

    return {
        'source_file': str(file_path),
        'timestamp': selected['data'].get('metadata', {}).get('timestamp', 'unknown'),
        'eegnet': models_data.get('eegnet', {}),
        'cbramod': models_data.get('cbramod', {}),
        'other_model': other_model,
        'mean_accuracy': selected['mean_accuracy'],
    }


def build_data_sources_from_historical(
    historical: Dict,
    current_results: Dict[str, List[TrainingResult]],
    subjects_list: List[str],
) -> List[PlotDataSource]:
    """
    从历史数据和当前结果构建绘图数据源列表.

    Args:
        historical: find_compatible_historical_results() 返回的历史数据字典
        current_results: 当前运行结果 {'eegnet': [...], 'cbramod': [...]}
        subjects_list: 要包含的被试列表

    Returns:
        PlotDataSource 列表，按顺序：历史EEGNet, 历史CBraMod, 当前EEGNet, 当前CBraMod
    """
    data_sources = []
    subjects_set = set(subjects_list)

    def _add_source(model_type: str, is_historical: bool):
        """辅助函数：添加单个数据源."""
        if is_historical:
            subjects_data = historical.get(model_type, {}).get('subjects', [])
            if subjects_data:
                results = [dict_to_result(s) for s in subjects_data]
                filtered = [r for r in results if r.subject_id in subjects_set]
                if filtered:
                    data_sources.append(PlotDataSource(
                        model_type=model_type,
                        results=filtered,
                        is_current_run=False,
                        label=f'{model_type.upper()} (hist)'
                    ))
        else:
            results = current_results.get(model_type, [])
            if results:
                filtered = [r for r in results if r.subject_id in subjects_set]
                if filtered:
                    data_sources.append(PlotDataSource(
                        model_type=model_type,
                        results=filtered,
                        is_current_run=True,
                        label=model_type.upper()
                    ))

    # 按正确顺序添加：历史EEGNet, 历史CBraMod, 当前EEGNet, 当前CBraMod
    _add_source('eegnet', is_historical=True)
    _add_source('cbramod', is_historical=True)
    _add_source('eegnet', is_historical=False)
    _add_source('cbramod', is_historical=False)

    return data_sources


def find_best_within_subject_for_model(
    output_dir: str,
    paradigm: str,
    task: str,
    model_type: str,
    subjects_set: set,
) -> Optional[List[TrainingResult]]:
    """
    为单个模型独立搜索最佳 within-subject 历史运行.

    在所有 comparison_cache 文件中查找该模型的数据，条件:
    - is_complete: true
    - 该模型的被试集合覆盖 subjects_set
    - 选择平均测试准确率最高的运行

    Args:
        output_dir: 结果目录
        paradigm: 范式
        task: 任务类型
        model_type: 'eegnet' 或 'cbramod'
        subjects_set: 需要覆盖的被试集合

    Returns:
        TrainingResult 列表（已过滤到 subjects_set），或 None
    """
    results_dir = Path(output_dir)
    if not results_dir.exists():
        return None

    pattern = f'*comparison_cache_{paradigm}_{task}.json'
    all_files = [
        f for f in results_dir.glob(pattern)
        if 'cross-subject' not in f.name and 'transfer' not in f.name
    ]

    best_results = None
    best_mean_acc = -1.0
    best_file = None

    for file_path in all_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            metadata = raw_data.get('metadata', {})
            if not metadata.get('is_complete', False):
                continue
            if not metadata.get('timestamp'):
                continue

            results_data = raw_data.get('results', {})
            model_data = results_data.get(model_type, {})
            if not model_data:
                continue

            # 检查被试覆盖
            if not (subjects_set <= set(model_data.keys())):
                continue

            # 转换并计算平均准确率
            converted = _convert_cache_to_comparison_format(raw_data)
            models_data = converted.get('models', {})
            subjects_data = models_data.get(model_type, {}).get('subjects', [])
            if not subjects_data:
                continue

            mean_acc = _compute_mean_accuracy(subjects_data)
            if mean_acc > best_mean_acc:
                best_mean_acc = mean_acc
                best_results = subjects_data
                best_file = file_path

        except (json.JSONDecodeError, KeyError, OSError):
            continue

    if best_results is None:
        return None

    log_cache.info(
        f"Best within-subject {model_type}: {best_file.name} "
        f"(mean={best_mean_acc*100:.1f}%)"
    )
    results = [dict_to_result(s) for s in best_results]
    return [r for r in results if r.subject_id in subjects_set]


def build_cross_subject_data_sources(
    current_results: Dict[str, Dict],
    output_dir: str = 'results',
    paradigm: str = 'imagery',
    task: str = 'binary',
    cross_subject_historical: Optional[Dict] = None,
    # Legacy parameter — ignored, kept for API compat
    within_subject_historical: Optional[Dict] = None,
) -> List[PlotDataSource]:
    """
    为 cross-subject 比较构建 PlotDataSource 列表.

    对每个模型独立搜索最佳 within-subject 历史运行（可来自不同文件）。

    顺序:
    1. 历史 Within-Subject EEGNet (alpha=0.4, 斜线填充)
    2. 历史 Within-Subject CBraMod (alpha=0.4, 斜线填充)
    3. 当前 Cross-Subject EEGNet (alpha=1.0)
    4. 当前 Cross-Subject CBraMod (alpha=1.0)
    5. (可选) 历史 Cross-Subject CBraMod (alpha=0.4)

    Args:
        current_results: 当前 cross-subject 训练结果 {model_type: train_cross_subject 返回值}
        output_dir: 结果目录路径（用于搜索历史数据）
        paradigm: 范式
        task: 任务类型
        cross_subject_historical: find_compatible_cross_subject_results() 返回值
        within_subject_historical: (已废弃，忽略) 保留用于 API 兼容

    Returns:
        PlotDataSource 列表，可直接传递给 generate_combined_plot()
    """
    data_sources = []

    # 收集所有被试（从当前结果）
    all_subjects = set()
    for result in current_results.values():
        per_subj = (
            result.get('per_subject_test_acc') or
            result.get('results', {}).get('per_subject_test_acc', {})
        )
        all_subjects.update(per_subj.keys())
    subjects_set = all_subjects

    # 1 & 2: 历史 Within-Subject 数据 — 每个模型独立搜索
    for model_type in ['eegnet', 'cbramod']:
        hist_results = find_best_within_subject_for_model(
            output_dir=output_dir,
            paradigm=paradigm,
            task=task,
            model_type=model_type,
            subjects_set=subjects_set,
        )
        if hist_results:
            data_sources.append(PlotDataSource(
                model_type=model_type,
                results=hist_results,
                is_current_run=False,
                label=f'{model_type.upper()} (Within)',
                hatch='///',
            ))

    # 3 & 4: 当前 Cross-Subject 数据 (EEGNet, CBraMod)
    for model_type in ['eegnet', 'cbramod']:
        if model_type not in current_results:
            continue
        result = current_results[model_type]

        training_results = cross_subject_result_to_training_results(result, model_type, task)
        if training_results:
            data_sources.append(PlotDataSource(
                model_type=model_type,
                results=training_results,
                is_current_run=True,
                label=f'{model_type.upper()} (Cross)'
            ))

    # 5: (可选) 历史 Cross-Subject CBraMod
    # 要求覆盖所有当前被试
    if cross_subject_historical:
        per_subj_hist = cross_subject_historical.get('per_subject_test_acc', {})
        hist_model = cross_subject_historical.get('model_type', 'cbramod')
        if per_subj_hist and subjects_set <= set(per_subj_hist.keys()):
            all_hist = cross_subject_result_to_training_results(
                cross_subject_historical, hist_model, task
            )
            training_results = [r for r in all_hist if r.subject_id in subjects_set]
            if training_results:
                data_sources.append(PlotDataSource(
                    model_type=hist_model,
                    results=training_results,
                    is_current_run=False,
                    label=f'{hist_model.upper()} (Cross-Hist)',
                    hatch='...',
                ))

    return data_sources


def build_transfer_data_sources(
    transfer_results: Dict[str, List[TrainingResult]],
    cross_subject_results: Dict[str, Dict],
    subjects: List[str],
    task: str,
    within_subject_results: Optional[Dict[str, List[TrainingResult]]] = None,
) -> List[PlotDataSource]:
    """
    为 transfer learning 比较构建 PlotDataSource 列表.

    顺序:
    1. Within-Subject EEGNet (baseline, alpha=0.4, hatch='///')
    2. Within-Subject CBraMod (baseline, alpha=0.4, hatch='///')
    3. Cross-Subject EEGNet (baseline, alpha=0.4, hatch='...')
    4. Cross-Subject CBraMod (baseline, alpha=0.4, hatch='...')
    5. Transfer EEGNet (current, alpha=1.0)
    6. Transfer CBraMod (current, alpha=1.0)

    Args:
        transfer_results: 迁移学习结果 {model_type: List[TrainingResult]}
        cross_subject_results: 跨被试基线结果 {model_type: cross_subject_result_dict}
        subjects: 被试列表
        task: 任务类型
        within_subject_results: 被试内基线结果 {model_type: List[TrainingResult]}

    Returns:
        PlotDataSource 列表，可直接传递给 generate_combined_plot()
    """
    subjects_set = set(subjects)
    data_sources = []

    # 1 & 2: Within-subject baselines
    if within_subject_results:
        for model_type in ['eegnet', 'cbramod']:
            ws_results = within_subject_results.get(model_type, [])
            filtered = [r for r in ws_results if r.subject_id in subjects_set]

            if filtered:
                data_sources.append(PlotDataSource(
                    model_type=model_type,
                    results=filtered,
                    is_current_run=False,
                    label=f'{model_type.upper()} (Within)',
                    hatch='///',
                ))

    # 3 & 4: Cross-subject baselines
    for model_type in ['eegnet', 'cbramod']:
        cross_result = cross_subject_results.get(model_type)
        if not cross_result:
            continue

        baseline_results = cross_subject_result_to_training_results(
            cross_result, model_type, task
        )
        filtered = [r for r in baseline_results if r.subject_id in subjects_set]

        if filtered:
            data_sources.append(PlotDataSource(
                model_type=model_type,
                results=filtered,
                is_current_run=False,
                label=f'{model_type.upper()} (Cross)',
                hatch='...',
            ))

    # 5 & 6: Transfer learning results (current)
    for model_type in ['eegnet', 'cbramod']:
        results = transfer_results.get(model_type, [])
        filtered = [r for r in results if r.subject_id in subjects_set]

        if filtered:
            data_sources.append(PlotDataSource(
                model_type=model_type,
                results=filtered,
                is_current_run=True,
                label=f'{model_type.upper()} (Transfer)',
            ))

    return data_sources


def prepare_combined_plot_data(
    output_dir: str,
    paradigm: str,
    task: str,
    current_results: Dict[str, List[TrainingResult]],
    current_model: Optional[str] = None,
) -> Tuple[Optional[List[PlotDataSource]], Optional[str]]:
    """
    准备组合图所需的数据源（自动检索历史数据）.

    Args:
        output_dir: 结果目录路径
        paradigm: 范式
        task: 任务类型
        current_results: 当前运行结果 {'eegnet': [...], 'cbramod': [...]}
        current_model: 当前运行的单个模型（可选，用于单模型模式）

    Returns:
        Tuple of (data_sources, historical_timestamp) or (None, None) if no historical data
    """
    # 确定被试列表（从任意有数据的模型获取）
    subjects_list = []
    for model_results in current_results.values():
        if model_results:
            subjects_list = [r.subject_id for r in model_results]
            break

    if not subjects_list:
        return None, None

    # 确定用于搜索的基准模型
    search_model = current_model or 'eegnet'

    # 搜索历史数据
    historical = find_compatible_historical_results(
        output_dir=output_dir,
        paradigm=paradigm,
        task=task,
        current_subjects=subjects_list,
        current_model=search_model,
    )

    if not historical:
        return None, None

    # 构建数据源
    data_sources = build_data_sources_from_historical(
        historical=historical,
        current_results=current_results,
        subjects_list=subjects_list,
    )

    if len(data_sources) < 2:
        return None, None

    timestamp = historical.get('timestamp', 'unknown')
    return data_sources, timestamp


# ============================================================================
# Full Comparison Results IO
# ============================================================================

def save_full_comparison_results(
    results: Dict[str, List[TrainingResult]],
    comparison: Optional[ComparisonResult],
    task_type: str,
    paradigm: str,
    output_dir: str,
    run_tag: Optional[str] = None,
) -> Path:
    """Save comprehensive comparison results to JSON.

    .. deprecated:: 2.0
        此函数已弃用，请使用 `save_cache()` 并传入 `summary` 和 `comparison` 参数。
        该函数将在未来版本中移除。

    Args:
        results: Dict mapping model_type to list of TrainingResult
        comparison: Optional ComparisonResult with statistics
        task_type: 'binary', 'ternary', or 'quaternary'
        paradigm: 'imagery' or 'movement'
        output_dir: Directory to save results
        run_tag: Optional run identifier

    Returns:
        Path to saved file
    """
    import warnings
    from dataclasses import asdict

    warnings.warn(
        "save_full_comparison_results() 已弃用，"
        "请使用 save_cache(summary=..., comparison=...) 替代。"
        "此函数将在 v2.0 中移除。",
        DeprecationWarning,
        stacklevel=2
    )

    output = {
        'metadata': {
            'paradigm': paradigm,
            'paradigm_description': PARADIGM_CONFIG[paradigm]['description'],
            'task_type': task_type,
            'run_tag': run_tag,
            'timestamp': datetime.now().isoformat(),
            'n_subjects': len(set(
                r.subject_id for model_results in results.values()
                for r in model_results
            )),
        },
        'models': {},
        'comparison': None,
    }

    for model_type, model_results in results.items():
        accs = [r.test_acc_majority for r in model_results]
        output['models'][model_type] = {
            'subjects': [result_to_dict(r) for r in model_results],
            'summary': {
                'mean': float(np.mean(accs)) if accs else 0,
                'std': float(np.std(accs)) if accs else 0,
                'min': float(np.min(accs)) if accs else 0,
                'max': float(np.max(accs)) if accs else 0,
            }
        }

    if comparison:
        output['comparison'] = asdict(comparison)

    filename = generate_result_filename('comparison', paradigm, task_type, 'json', run_tag)

    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    log_cache.info(f"Results saved: {output_path}")
    return output_path


def _convert_cache_to_comparison_format(cache: Dict) -> Dict:
    """将缓存格式转换为比较结果格式（内部使用）。

    Args:
        cache: 缓存文件数据

    Returns:
        比较结果格式的数据
    """
    metadata = cache.get('metadata', {})

    # 重建 models 结构（list 格式）
    models = {}
    for model_type, subjects_dict in cache.get('results', {}).items():
        # 将嵌套字典转换为列表
        subjects_list = list(subjects_dict.values()) if isinstance(subjects_dict, dict) else subjects_dict

        models[model_type] = {
            'subjects': subjects_list,
            'summary': cache.get('summary', {}).get(model_type, {})
        }

    return {
        'metadata': {
            'paradigm': metadata.get('paradigm'),
            'task_type': metadata.get('task'),
            'timestamp': metadata.get('timestamp'),
            'n_subjects': metadata.get('n_subjects'),
        },
        'models': models,
        'comparison': cache.get('comparison', {})
    }


def load_comparison_results(results_file: str) -> Dict[str, List[TrainingResult]]:
    """加载比较结果（支持缓存文件和旧格式结果文件）。

    Args:
        results_file: 结果或缓存 JSON 文件路径

    Returns:
        模型类型 → TrainingResult 列表的映射
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 检测文件类型：新格式缓存文件 vs 旧格式结果文件
    if 'results' in data and isinstance(data['results'], dict):
        # 新格式：缓存文件（嵌套字典结构）
        # 第一层是模型类型，第二层是被试 ID
        first_model = next(iter(data['results'].values()), {})
        if first_model and isinstance(next(iter(first_model.values()), None), dict):
            data = _convert_cache_to_comparison_format(data)

    # 解析为 TrainingResult 列表
    results = {}
    for model_type, model_data in data.get('models', data).items():
        subjects_data = model_data.get('subjects', model_data)
        if isinstance(subjects_data, dict):
            subjects_data = subjects_data.get('subjects', [])

        results[model_type] = [dict_to_result(s) for s in subjects_data]

    return results


# ============================================================================
# Single Model Results IO
# ============================================================================

def save_single_model_results(
    model_type: str,
    results: List[TrainingResult],
    statistics: Dict,
    paradigm: str,
    task: str,
    output_dir: str,
    run_tag: Optional[str] = None,
) -> Path:
    """Save single model results to JSON.

    Args:
        model_type: 'eegnet' or 'cbramod'
        results: List of TrainingResult
        statistics: Statistics dict from compute_model_statistics()
        paradigm: 'imagery' or 'movement'
        task: 'binary', 'ternary', or 'quaternary'
        output_dir: Directory to save results
        run_tag: Optional run identifier

    Returns:
        Path to saved file
    """
    output = {
        'metadata': {
            'model_type': model_type,
            'paradigm': paradigm,
            'paradigm_description': PARADIGM_CONFIG[paradigm]['description'],
            'task': task,
            'run_tag': run_tag,
            'timestamp': datetime.now().isoformat(),
            'n_subjects': statistics['n_subjects'],
        },
        'subjects': [result_to_dict(r) for r in results],
        'statistics': statistics,
    }

    filename = generate_result_filename(model_type, paradigm, task, 'json', run_tag)

    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    log_cache.info(f"Results saved: {output_path}")
    return output_path


def load_single_model_results(
    model_type: str,
    output_dir: str,
    paradigm: str,
    task: str,
    results_file: Optional[str] = None,
) -> Tuple[List[TrainingResult], Dict]:
    """Load single model results from cache or specific file.

    Args:
        model_type: 'eegnet' or 'cbramod'
        output_dir: Directory containing cache files
        paradigm: 'imagery' or 'movement'
        task: 'binary', 'ternary', or 'quaternary'
        results_file: Optional path to specific results file

    Returns:
        Tuple of (results_list, statistics_dict)
    """
    if results_file:
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle different file formats
        if 'subjects' in data:
            # Single model format
            results = [dict_to_result(s) for s in data['subjects']]
        elif 'models' in data and model_type in data['models']:
            # Full comparison format
            results = [dict_to_result(s) for s in data['models'][model_type]['subjects']]
        else:
            raise ValueError(f"Cannot find {model_type} results in {results_file}")

        stats = compute_model_statistics(results)
        return results, stats

    # Load from cache
    cache, _ = load_cache(output_dir, paradigm, task, find_latest=True)

    if model_type not in cache:
        return [], {}

    results = [dict_to_result(d) for d in cache[model_type].values()]
    stats = compute_model_statistics(results)
    return results, stats


# ============================================================================
# Cross-Subject Results Search
# ============================================================================

def find_compatible_within_subject_results(
    output_dir: str,
    paradigm: str,
    task: str,
    subjects: List[str],
    selection_strategy: SelectionStrategy = SelectionStrategy.BEST_ACCURACY,
) -> Optional[Dict]:
    """
    搜索兼容的 within-subject 历史结果（用于 cross-subject 对比图）.

    条件:
    - 文件模式: *comparison_cache_{paradigm}_{task}.json（排除 cross-subject 文件）
    - is_complete: true
    - 被试集合覆盖 subjects

    Args:
        output_dir: 结果目录路径
        paradigm: 范式 ('imagery' 或 'movement')
        task: 任务类型 ('binary', 'ternary', 'quaternary')
        subjects: 需要覆盖的被试列表
        selection_strategy: 选择策略 (BEST_ACCURACY 或 NEWEST)

    Returns:
        包含 eegnet 和 cbramod 结果的字典:
        {
            'source_file': str,
            'timestamp': str,
            'eegnet': {'subjects': [...], 'summary': {...}},
            'cbramod': {'subjects': [...], 'summary': {...}},
            'mean_accuracy': {'eegnet': float, 'cbramod': float},
        }
        如果找不到兼容结果，返回 None
    """
    results_dir = Path(output_dir)
    if not results_dir.exists():
        return None

    # 搜索 within-subject 缓存文件（排除 cross-subject）
    pattern = f'*comparison_cache_{paradigm}_{task}.json'
    all_files = list(results_dir.glob(pattern))

    # 排除 cross-subject 和 transfer 文件
    all_files = [
        f for f in all_files
        if 'cross-subject' not in f.name and 'transfer' not in f.name
    ]

    if not all_files:
        return None

    subjects_set = set(subjects)
    compatible_files = []

    for file_path in all_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            # 必须是新格式缓存文件
            if 'results' not in raw_data or not isinstance(raw_data['results'], dict):
                continue

            # 检查 is_complete
            metadata = raw_data.get('metadata', {})
            if not metadata.get('is_complete', False):
                continue

            timestamp = metadata.get('timestamp')
            if not timestamp:
                continue

            # 检查是否包含两个模型且被试集合覆盖
            results_data = raw_data.get('results', {})
            if 'eegnet' not in results_data or 'cbramod' not in results_data:
                continue

            # 获取 eegnet 被试集合
            eegnet_subjects = set(results_data.get('eegnet', {}).keys())
            if not (subjects_set <= eegnet_subjects):
                continue

            # 获取 cbramod 被试集合
            cbramod_subjects = set(results_data.get('cbramod', {}).keys())
            if not (subjects_set <= cbramod_subjects):
                continue

            # 转换为标准格式
            data = _convert_cache_to_comparison_format(raw_data)
            models_data = data.get('models', {})

            # 计算各模型平均准确率
            mean_accuracy = {}
            for model_type in ['eegnet', 'cbramod']:
                subjects_data = models_data.get(model_type, {}).get('subjects', [])
                mean_accuracy[model_type] = _compute_mean_accuracy(subjects_data)

            compatible_files.append({
                'file_path': file_path,
                'timestamp': timestamp,
                'data': data,
                'models_data': models_data,
                'mean_accuracy': mean_accuracy,
            })

        except (json.JSONDecodeError, KeyError, OSError) as e:
            log_cache.debug(f"Skipping {file_path.name}: {e}")
            continue

    if not compatible_files:
        return None

    # 根据策略选择
    if selection_strategy == SelectionStrategy.NEWEST:
        compatible_files.sort(key=lambda x: x['timestamp'], reverse=True)
    else:  # BEST_ACCURACY - 使用两个模型的平均准确率
        compatible_files.sort(
            key=lambda x: (x['mean_accuracy'].get('eegnet', 0) + x['mean_accuracy'].get('cbramod', 0)) / 2,
            reverse=True
        )

    selected = compatible_files[0]
    log_cache.info(f"Found compatible within-subject results: {selected['file_path'].name}")

    return {
        'source_file': str(selected['file_path']),
        'timestamp': selected['timestamp'],
        'eegnet': selected['models_data'].get('eegnet', {}),
        'cbramod': selected['models_data'].get('cbramod', {}),
        'mean_accuracy': selected['mean_accuracy'],
    }


def find_compatible_cross_subject_results(
    output_dir: str,
    paradigm: str,
    task: str,
    subjects: List[str],
    model_type: str = 'cbramod',
    exclude_run_tag: Optional[str] = None,
    selection_strategy: SelectionStrategy = SelectionStrategy.BEST_ACCURACY,
) -> Optional[Dict]:
    """
    搜索兼容的 cross-subject 历史结果.

    条件:
    - 文件模式: *cross-subject_*_{paradigm}_{task}.json
    - 被试集合匹配
    - 排除当前运行 (exclude_run_tag)

    Args:
        output_dir: 结果目录路径
        paradigm: 范式 ('imagery' 或 'movement')
        task: 任务类型 ('binary', 'ternary', 'quaternary')
        subjects: 需要匹配的被试列表
        model_type: 要搜索的模型类型 (默认 'cbramod')
        exclude_run_tag: 要排除的运行 tag (当前运行)
        selection_strategy: 选择策略 (BEST_ACCURACY 或 NEWEST)

    Returns:
        包含单个模型结果的字典:
        {
            'source_file': str,
            'timestamp': str,
            'run_tag': str,
            'model_type': str,
            'per_subject_test_acc': {'S01': 0.85, ...},
            'mean_test_acc': float,
            'std_test_acc': float,
            'subjects': [...] (被试列表)
        }
        如果找不到兼容结果，返回 None
    """
    results_dir = Path(output_dir)
    if not results_dir.exists():
        return None

    # 搜索 cross-subject 结果文件
    # 支持两种格式: *cross-subject_{model}_{paradigm}_{task}.json
    #               *cross-subject_comparison_cache_{paradigm}_{task}.json
    patterns = [
        f'*cross-subject_{model_type}_{paradigm}_{task}.json',
        f'*cross-subject_comparison_cache_{paradigm}_{task}.json',
    ]

    all_files = []
    for pattern in patterns:
        all_files.extend(results_dir.glob(pattern))

    # 去重
    all_files = list(set(all_files))

    if not all_files:
        return None

    subjects_set = set(subjects)
    compatible_files = []

    for file_path in all_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            metadata = data.get('metadata', {})
            run_tag = metadata.get('run_tag')
            timestamp = metadata.get('timestamp')

            # 排除当前运行
            if exclude_run_tag and run_tag == exclude_run_tag:
                continue

            if not timestamp:
                continue

            # 提取 per_subject_test_acc
            # 支持两种格式: 直接在 results 中或在嵌套结构中
            per_subject_acc = None
            if 'results' in data and isinstance(data['results'], dict):
                # 可能是 comparison cache 格式
                if 'per_subject_test_acc' in data.get('results', {}):
                    per_subject_acc = data['results']['per_subject_test_acc']
                elif model_type in data['results']:
                    # 模型在 results 字典中
                    model_data = data['results'][model_type]
                    if isinstance(model_data, dict):
                        # 可能是 subjects 字典
                        per_subject_acc = {
                            k: v.get('test_acc_majority', v.get('test_acc', 0))
                            for k, v in model_data.items()
                            if isinstance(v, dict)
                        }
            elif 'per_subject_test_acc' in data:
                per_subject_acc = data['per_subject_test_acc']

            if not per_subject_acc:
                continue

            # 检查被试集合是否匹配
            file_subjects = set(per_subject_acc.keys())
            if not (subjects_set <= file_subjects):
                continue

            # 计算平均准确率
            accs = list(per_subject_acc.values())
            mean_acc = np.mean(accs) if accs else 0.0
            std_acc = np.std(accs) if accs else 0.0

            compatible_files.append({
                'file_path': file_path,
                'timestamp': timestamp,
                'run_tag': run_tag,
                'per_subject_test_acc': per_subject_acc,
                'mean_test_acc': mean_acc,
                'std_test_acc': std_acc,
                'subjects': list(per_subject_acc.keys()),
            })

        except (json.JSONDecodeError, KeyError, OSError) as e:
            log_cache.debug(f"Skipping {file_path.name}: {e}")
            continue

    if not compatible_files:
        return None

    # 根据策略选择
    if selection_strategy == SelectionStrategy.NEWEST:
        compatible_files.sort(key=lambda x: x['timestamp'], reverse=True)
    else:  # BEST_ACCURACY
        compatible_files.sort(key=lambda x: x['mean_test_acc'], reverse=True)

    selected = compatible_files[0]
    log_cache.info(
        f"Found compatible cross-subject results: {selected['file_path'].name} "
        f"(mean_acc={selected['mean_test_acc']:.4f})"
    )

    return {
        'source_file': str(selected['file_path']),
        'timestamp': selected['timestamp'],
        'run_tag': selected['run_tag'],
        'model_type': model_type,
        'per_subject_test_acc': selected['per_subject_test_acc'],
        'mean_test_acc': selected['mean_test_acc'],
        'std_test_acc': selected['std_test_acc'],
        'subjects': selected['subjects'],
    }


def save_cross_subject_result(
    result: Dict,
    model_type: str,
    paradigm: str,
    task: str,
    output_dir: str,
    run_tag: Optional[str] = None,
) -> Path:
    """
    保存单模型跨被试训练结果到 JSON 文件.

    Args:
        result: train_cross_subject() 的返回值
        model_type: 'eegnet' 或 'cbramod'
        paradigm: 范式 ('imagery' 或 'movement')
        task: 任务类型 ('binary', 'ternary', 'quaternary')
        output_dir: 输出目录
        run_tag: 运行标签

    Returns:
        保存的文件路径
    """
    from .serialization import generate_result_filename

    output = {
        'metadata': {
            'type': 'cross-subject',
            'model_type': model_type,
            'paradigm': paradigm,
            'task': task,
            'subjects': result.get('subjects', list(result.get('per_subject_test_acc', {}).keys())),
            'n_subjects': len(result.get('per_subject_test_acc', {})),
            'run_tag': run_tag,
            'timestamp': datetime.now().isoformat(),
        },
        'results': {
            'per_subject_test_acc': result.get('per_subject_test_acc', {}),
            'mean_test_acc': result.get('mean_test_acc', 0),
            'std_test_acc': result.get('std_test_acc', 0),
            'best_val_acc': result.get('val_acc', 0),
            'best_epoch': result.get('best_epoch', 0),
        },
        'training_info': {
            'training_time': result.get('training_time', 0),
            'model_path': result.get('model_path', ''),
        },
    }

    filename = generate_result_filename(
        model_type, paradigm, task, 'json', run_tag, is_cross_subject=True
    )

    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    log_cache.info(f"Cross-subject results saved: {output_path}")
    return output_path
