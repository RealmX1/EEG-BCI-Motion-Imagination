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

from ..config.constants import CACHE_FILENAME, CACHE_FILENAME_WITH_TAG, PARADIGM_CONFIG
from ..utils.logging import SectionLogger
from .dataclasses import ComparisonResult, TrainingResult, PlotDataSource
from .serialization import dict_to_result, result_to_dict, generate_result_filename
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
    return (
        subject_data.get('test_acc_majority') or
        subject_data.get('test_accuracy_majority') or
        subject_data.get('test_accuracy') or
        subject_data.get('test_acc', 0)
    )


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


def get_cache_path(output_dir: str, paradigm: str, task: str, run_tag: Optional[str] = None) -> Path:
    """Get path to cache file."""
    if run_tag:
        filename = CACHE_FILENAME_WITH_TAG.format(tag=run_tag, paradigm=paradigm, task=task)
    else:
        filename = CACHE_FILENAME.format(paradigm=paradigm, task=task)
    return Path(output_dir) / filename


def find_latest_cache(output_dir: str, paradigm: str, task: str) -> Optional[Path]:
    """Find the latest cache file (tagged or untagged) for the given paradigm and task."""
    results_dir = Path(output_dir)
    if not results_dir.exists():
        return None

    # Pattern matches both tagged and untagged cache files
    pattern = f'*comparison_cache_{paradigm}_{task}.json'
    cache_files = list(results_dir.glob(pattern))

    if not cache_files:
        # Fallback: try old format without paradigm
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
) -> Optional[Tuple[Path, str]]:
    """
    根据时间戳子串查找缓存文件.

    Args:
        output_dir: 结果目录
        paradigm: 范式 ('imagery' 或 'movement')
        task: 任务类型 ('binary', 'ternary', 'quaternary')
        tag_substring: 时间戳子串（如 "20260205"）。如果为 None，返回最新的缓存。

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
    pattern = f'*comparison_cache_{paradigm}_{task}.json'
    cache_files = list(results_dir.glob(pattern))

    if not cache_files:
        return None

    def extract_run_tag(path: Path) -> Optional[str]:
        """从文件名中提取 run_tag."""
        # 文件名格式: {tag}_comparison_cache_{paradigm}_{task}.json
        # 或: comparison_cache_{paradigm}_{task}.json (无 tag)
        name = path.name
        suffix = f'_comparison_cache_{paradigm}_{task}.json'
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
    find_latest: bool = False
) -> Tuple[Dict[str, Dict[str, dict]], Dict]:
    """Load cached results with backward compatibility for old cache format.

    Args:
        output_dir: Directory containing cache files
        paradigm: 'imagery' or 'movement'
        task: 'binary', 'ternary', or 'quaternary'
        run_tag: Optional tag for specific run (e.g., '20260110_2317')
        find_latest: If True and run_tag is None, find the latest cache file

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
        latest_cache = find_latest_cache(output_dir, paradigm, task)
        if latest_cache:
            try:
                with open(latest_cache, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                log_cache.info(f"Loaded latest cache: {latest_cache.name}")
                return data.get('results', {}), extract_metadata(data)
            except Exception as e:
                log_cache.warning(f"Failed to load latest cache: {e}")
        return {}, empty_metadata

    cache_path = get_cache_path(output_dir, paradigm, task, run_tag)
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

    Returns:
        Path to saved cache file
    """
    cache_path = get_cache_path(output_dir, paradigm, task, run_tag)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Preserve existing timestamp or create new one
    timestamp = existing_timestamp or datetime.now().isoformat()

    # Create/update metadata structure
    data = {
        'metadata': {
            'paradigm': paradigm,
            'task': task,
            'run_tag': run_tag,
            'timestamp': timestamp,
            'n_subjects': n_subjects,
            'is_complete': is_complete,
        },
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

    # 去重
    all_files = list(set(all_files))

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


