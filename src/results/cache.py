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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config.constants import CACHE_FILENAME, CACHE_FILENAME_WITH_TAG, PARADIGM_CONFIG
from ..utils.logging import SectionLogger
from .dataclasses import ComparisonResult, TrainingResult, PlotDataSource
from .serialization import dict_to_result, result_to_dict, generate_result_filename
from .statistics import compute_model_statistics

logger = logging.getLogger(__name__)
log_cache = SectionLogger(logger, 'cache')


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
    empty_metadata = {'run_tag': None, 'wandb_groups': {}}

    def extract_metadata(data: dict) -> Dict:
        """Extract metadata from cache data with backward compatibility."""
        return {
            'run_tag': data.get('run_tag'),
            'wandb_groups': data.get('wandb_groups', {}),
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
):
    """Save results to cache using atomic write to prevent corruption.

    Args:
        output_dir: Directory to save cache files
        paradigm: 'imagery' or 'movement'
        task: 'binary', 'ternary', or 'quaternary'
        results: Training results dict
        run_tag: Optional tag for specific run
        wandb_groups: Optional dict mapping model_type to wandb_group name
    """
    cache_path = get_cache_path(output_dir, paradigm, task, run_tag)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'paradigm': paradigm,
        'task': task,
        'run_tag': run_tag,
        'wandb_groups': wandb_groups or {},
        'last_updated': datetime.now().isoformat(),
        'results': results,
    }

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


def find_compatible_historical_results(
    output_dir: str,
    paradigm: str,
    task: str,
    current_subjects: List[str],
    current_model: str,
) -> Optional[Dict]:
    """
    搜索与当前运行兼容的历史完整对比结果.

    兼容性条件:
    1. 必须是完整对比文件（包含两个模型的结果）
    2. 另一个模型（非当前运行的模型）的被试集合必须覆盖当前被试集合

    Args:
        output_dir: 结果目录路径
        paradigm: 范式 ('imagery' 或 'movement')
        task: 任务类型 ('binary', 'ternary', 'quaternary')
        current_subjects: 当前运行的被试列表
        current_model: 当前运行的模型 ('eegnet' 或 'cbramod')

    Returns:
        包含历史数据的字典:
        {
            'source_file': str,           # 来源文件路径
            'timestamp': str,             # 历史运行时间戳
            'eegnet': {'subjects': [...], 'summary': {...}},
            'cbramod': {'subjects': [...], 'summary': {...}},
            'other_model': str,           # 另一个模型名称
        }
        如果找不到兼容结果，返回 None
    """
    results_dir = Path(output_dir)
    if not results_dir.exists():
        return None

    # 匹配 comparison 结果文件（排除 cache 文件）
    pattern = f'*comparison_{paradigm}_{task}*.json'
    all_files = list(results_dir.glob(pattern))

    # 排除 cache 文件
    result_files = [f for f in all_files if 'cache' not in f.name.lower()]

    if not result_files:
        return None

    # 按修改时间排序（最新优先）
    def safe_mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    result_files.sort(key=safe_mtime, reverse=True)

    other_model = 'cbramod' if current_model == 'eegnet' else 'eegnet'
    current_subjects_set = set(current_subjects)

    for file_path in result_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            models_data = data.get('models', {})

            # 检查是否包含两个模型
            if 'eegnet' not in models_data or 'cbramod' not in models_data:
                continue

            # 提取另一个模型的被试集合
            other_subjects_data = models_data[other_model].get('subjects', [])
            other_subjects_set = {s['subject_id'] for s in other_subjects_data}

            # 检查是否为超集或相同集合
            if current_subjects_set <= other_subjects_set:
                log_cache.info(f"Found compatible historical data: {file_path.name}")
                return {
                    'source_file': str(file_path),
                    'timestamp': data.get('metadata', {}).get('timestamp', 'unknown'),
                    'eegnet': models_data.get('eegnet', {}),
                    'cbramod': models_data.get('cbramod', {}),
                    'other_model': other_model,
                }
        except (json.JSONDecodeError, KeyError, OSError) as e:
            log_cache.debug(f"Skipping {file_path.name}: {e}")
            continue

    return None


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
    from dataclasses import asdict

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


def load_comparison_results(results_file: str) -> Dict[str, List[TrainingResult]]:
    """Load comparison results from a previous run.

    Args:
        results_file: Path to results JSON file

    Returns:
        Dict mapping model_type to list of TrainingResult
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

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
