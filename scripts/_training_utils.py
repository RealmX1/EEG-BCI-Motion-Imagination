"""
Shared utilities for EEG-BCI training scripts.

This module contains common functions used by both run_single_model.py
and run_full_comparison.py, including:
- Cache management (load/save)
- Subject discovery
- Training result dataclass
- Training wrapper functions
"""

import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.device import set_seed
from src.utils.logging import SectionLogger, setup_logging
from src.preprocessing.data_loader import discover_available_subjects, PreprocessConfig
from src.training.train_within_subject import train_subject_simple


# Setup logging
setup_logging('training')
logger = logging.getLogger(__name__)
log_cache = SectionLogger(logger, 'cache')
log_train = SectionLogger(logger, 'train')


# ============================================================================
# Constants
# ============================================================================

CACHE_FILENAME = 'comparison_cache_{paradigm}_{task}.json'
CACHE_FILENAME_WITH_TAG = '{tag}_comparison_cache_{paradigm}_{task}.json'


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

PARADIGM_CONFIG = {
    'imagery': {
        'description': 'Motor Imagery (MI)',
    },
    'movement': {
        'description': 'Motor Execution (ME)',
    },
}

MODEL_COLORS = {
    'eegnet': '#2E86AB',   # Blue
    'cbramod': '#E94F37',  # Red/Coral
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TrainingResult:
    """Result from a single training run."""
    subject_id: str
    task_type: str
    model_type: str
    best_val_acc: float
    test_acc: float
    test_acc_majority: float
    epochs_trained: int
    training_time: float


@dataclass
class PlotDataSource:
    """绘图数据源配置，用于组合对比图."""
    model_type: str           # 'eegnet' 或 'cbramod'
    results: List[TrainingResult]
    is_current_run: bool      # True = 当前运行, False = 历史数据
    label: str                # 图例标签


# ============================================================================
# Subject Discovery
# ============================================================================

def discover_subjects(
    data_root: str,
    paradigm: str = 'imagery',
    task: str = 'binary',
    use_cache_index: bool = False,
    cache_index_path: str = ".cache_index.json"
) -> List[str]:
    """
    Discover all available subjects.

    Args:
        data_root: Root directory containing subject folders
        paradigm: 'imagery' or 'movement'
        task: 'binary', 'ternary', or 'quaternary'
        use_cache_index: If True, discover from cache index instead of filesystem
        cache_index_path: Path to cache index file (default: .cache_index.json)

    Returns:
        List of subject IDs (e.g., ['S01', 'S02', ...])
    """
    if use_cache_index:
        from src.preprocessing.data_loader import discover_subjects_from_cache_index
        return discover_subjects_from_cache_index(cache_index_path, paradigm, task)
    else:
        return discover_available_subjects(data_root, paradigm, task)


# ============================================================================
# Cache Management
# ============================================================================

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
# Result Serialization
# ============================================================================

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


# ============================================================================
# Training Functions
# ============================================================================

def print_subject_result(subject_id: str, model_type: str, result: TrainingResult):
    """Print formatted result for a single subject."""
    print("\n" + "=" * 60)
    print(f" {model_type.upper()} - {subject_id} COMPLETE")
    print("=" * 60)
    print(f"  Validation Accuracy:  {result.best_val_acc:.2%}")
    print(f"  Test Accuracy:        {result.test_acc_majority:.2%} (majority voting, Sess2 Finetune)")
    print(f"  Epochs Trained:       {result.epochs_trained}")
    print(f"  Training Time:        {result.training_time:.1f}s")
    print("=" * 60 + "\n")


def train_and_get_result(
    subject_id: str,
    model_type: str,
    task: str,
    paradigm: str,
    data_root: str,
    save_dir: str,
    no_wandb: bool = False,
    upload_model: bool = False,
    wandb_group: Optional[str] = None,
    preprocess_config: Optional[PreprocessConfig] = None,
    wandb_interactive: bool = False,
    wandb_metadata: Optional[Dict] = None,
    cache_only: bool = False,
    cache_index_path: str = ".cache_index.json",
    scheduler: Optional[str] = None,
) -> TrainingResult:
    """
    Train a model for a single subject and return TrainingResult.

    This is a thin wrapper around train_subject_simple from train_within_subject.py.

    Args:
        subject_id: Subject ID (e.g., 'S01')
        model_type: 'eegnet' or 'cbramod'
        task: 'binary', 'ternary', or 'quaternary'
        paradigm: 'imagery' or 'movement'
        data_root: Path to data directory
        save_dir: Path to save checkpoints
        no_wandb: Disable wandb logging
        upload_model: Upload model to WandB
        wandb_group: WandB run group
        preprocess_config: Optional custom PreprocessConfig for ML engineering experiments
        wandb_interactive: Prompt for run details interactively
        wandb_metadata: Pre-collected metadata (goal, hypothesis, notes) for batch training
        cache_only: If True, load data exclusively from cache index
        cache_index_path: Path to cache index file for cache_only mode
        scheduler: Learning rate scheduler type (e.g., 'wsd', 'cosine_annealing_warmup_decay')
    """
    # Build config overrides for scheduler
    config_overrides = None
    if scheduler is not None:
        config_overrides = {'training': {'scheduler': scheduler}}

    result_dict = train_subject_simple(
        subject_id=subject_id,
        model_type=model_type,
        task=task,
        paradigm=paradigm,
        data_root=data_root,
        save_dir=save_dir,
        no_wandb=no_wandb,
        upload_model=upload_model,
        wandb_group=wandb_group,
        preprocess_config=preprocess_config,
        wandb_interactive=wandb_interactive,
        wandb_metadata=wandb_metadata,
        cache_only=cache_only,
        cache_index_path=cache_index_path,
        config_overrides=config_overrides,
    )

    if not result_dict:
        raise ValueError(f"Training failed for {subject_id}")

    return TrainingResult(
        subject_id=subject_id,
        task_type=task,
        model_type=model_type,
        best_val_acc=result_dict.get('best_val_acc', result_dict.get('val_accuracy', 0.0)),
        test_acc=result_dict.get('test_accuracy', 0.0),
        test_acc_majority=result_dict.get('test_accuracy_majority', result_dict.get('test_accuracy', 0.0)),
        epochs_trained=result_dict.get('epochs_trained', result_dict.get('best_epoch', 0)),
        training_time=result_dict.get('training_time', 0.0),
    )


# ============================================================================
# Statistics Functions
# ============================================================================

def compute_model_statistics(results: List[TrainingResult]) -> Dict:
    """Compute summary statistics for a single model's results."""
    if not results:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0,
            'n_subjects': 0,
        }

    accs = [r.test_acc_majority for r in results]
    return {
        'mean': float(np.mean(accs)),
        'std': float(np.std(accs)),
        'min': float(np.min(accs)),
        'max': float(np.max(accs)),
        'median': float(np.median(accs)),
        'n_subjects': len(accs),
    }


def print_model_summary(model_type: str, stats: Dict, results: List[TrainingResult]):
    """Print formatted summary for a single model."""
    print("\n" + "=" * 70)
    print(f" {model_type.upper()} SUMMARY")
    print("=" * 70)

    # Per-subject table
    print(f"\n{'Subject':<10} {'Val Acc':<12} {'Test Acc':<12} {'Epochs':<10} {'Time (s)':<10}")
    print("-" * 54)
    for r in sorted(results, key=lambda x: x.subject_id):
        print(f"{r.subject_id:<10} {r.best_val_acc:<12.2%} {r.test_acc_majority:<12.2%} "
              f"{r.epochs_trained:<10} {r.training_time:<10.1f}")

    # Statistics
    print("\n" + "-" * 54)
    print(f"{'Statistic':<15} {'Value':<15}")
    print("-" * 30)
    print(f"{'N Subjects':<15} {stats['n_subjects']:<15}")
    print(f"{'Mean':<15} {stats['mean']:.2%}")
    print(f"{'Median':<15} {stats['median']:.2%}")
    print(f"{'Std':<15} {stats['std']:.2%}")
    print(f"{'Min':<15} {stats['min']:.2%}")
    print(f"{'Max':<15} {stats['max']:.2%}")
    print("=" * 70 + "\n")


# ============================================================================
# Combined Plotting (支持历史数据对比)
# ============================================================================

def generate_combined_plot(
    data_sources: List[PlotDataSource],
    output_path: str,
    task_type: str = 'binary',
    paradigm: str = 'imagery',
    historical_timestamp: Optional[str] = None,
):
    """
    生成组合对比图（支持混合新旧数据）.

    布局（2 行，第一行跨两列）:
    +------------------------------------------+
    |          条形图 (2x 宽度)                   |
    |   每被试 3 条: 历史数据(半透明) + 当前运行      |
    +--------------------+---------------------+
    |    箱线图(3 蜡烛)     |    配对对比图        |
    +--------------------+---------------------+

    视觉效果:
    - 历史数据: alpha=0.4, 斜线填充
    - 当前运行: alpha=1.0, 无填充, 粗边框

    Args:
        data_sources: 数据源列表（2-3 个 PlotDataSource）
        output_path: 输出文件路径
        task_type: 任务类型
        paradigm: 范式
        historical_timestamp: 历史数据时间戳（用于标题标注）
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ImportError:
        log_cache.warning("matplotlib not installed, skipping plot")
        return

    chance_levels = {'binary': 0.5, 'ternary': 1/3, 'quaternary': 0.25}
    chance_level = chance_levels.get(task_type, 0.5)

    colors = {'eegnet': '#2E86AB', 'cbramod': '#E94F37'}

    # 收集所有被试
    all_subjects = set()
    for source in data_sources:
        for r in source.results:
            all_subjects.add(r.subject_id)
    subjects = sorted(all_subjects)

    if not subjects:
        log_cache.warning("No subjects for plotting")
        return

    # 创建 2 行布局，第一行跨两列
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, height_ratios=[1.2, 1], hspace=0.3, wspace=0.25)

    ax_bar = fig.add_subplot(gs[0, :])      # 顶部条形图（跨两列）
    ax_box = fig.add_subplot(gs[1, 0])      # 左下箱线图
    ax_scatter = fig.add_subplot(gs[1, 1])  # 右下配对散点图

    # =========================================================================
    # Panel 1: 条形图
    # =========================================================================
    n_subjects = len(subjects)
    n_sources = len(data_sources)
    bar_width = 0.8 / n_sources
    x_base = np.arange(n_subjects)

    for i, source in enumerate(data_sources):
        x_positions = x_base + (i - (n_sources - 1) / 2) * bar_width

        # 按被试排序获取准确率
        result_by_subj = {r.subject_id: r.test_acc_majority for r in source.results}
        accs = [result_by_subj.get(s, 0) for s in subjects]

        alpha = 1.0 if source.is_current_run else 0.4
        edgecolor = 'black' if source.is_current_run else 'gray'
        linewidth = 1.5 if source.is_current_run else 0.5
        hatch = '' if source.is_current_run else '///'

        bars = ax_bar.bar(
            x_positions, accs, bar_width,
            label=source.label,
            color=colors[source.model_type],
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=linewidth,
            hatch=hatch,
        )

        # 仅为当前运行添加数值标签
        if source.is_current_run:
            for bar, val in zip(bars, accs):
                if val > 0:
                    ax_bar.text(
                        bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.01,
                        f'{val*100:.1f}',
                        ha='center', va='bottom', fontsize=7
                    )

    ax_bar.set_xlabel('Subject')
    ax_bar.set_ylabel('Test Accuracy')
    title = f'Per-Subject Accuracy Comparison ({paradigm.title()} {task_type.title()})'
    if historical_timestamp:
        title += f'\n(Historical data from: {historical_timestamp[:10]})'
    ax_bar.set_title(title)
    ax_bar.set_xticks(x_base)
    ax_bar.set_xticklabels(subjects, rotation=45, ha='right')
    ax_bar.axhline(y=chance_level, color='gray', linestyle='--', alpha=0.5,
                   label=f'Chance ({chance_level*100:.1f}%)')
    ax_bar.set_ylim([0, 1.05])
    ax_bar.legend(loc='upper right', fontsize=8)

    # =========================================================================
    # Panel 2: 箱线图
    # =========================================================================
    median_color = 'black'
    mean_color = '#E63946'

    box_data = []
    box_labels = []
    box_colors = []
    box_alphas = []

    for source in data_sources:
        accs = [r.test_acc_majority for r in source.results]
        if accs:
            box_data.append(accs)
            box_labels.append(source.label)
            box_colors.append(colors[source.model_type])
            box_alphas.append(1.0 if source.is_current_run else 0.4)

    if box_data:
        bp = ax_box.boxplot(
            box_data, labels=box_labels, patch_artist=True,
            showmeans=True, meanline=True,
            meanprops={'color': mean_color, 'linewidth': 2, 'linestyle': (0, (3, 2))}
        )

        for patch, color, alpha in zip(bp['boxes'], box_colors, box_alphas):
            patch.set_facecolor(color)
            patch.set_alpha(alpha)

        for median in bp['medians']:
            median.set_color(median_color)
            median.set_linewidth(2)

        # 添加统计标注
        for i, (source, accs_list) in enumerate(zip(data_sources, box_data)):
            mean_val = np.mean(accs_list)
            median_val = np.median(accs_list)
            x_offset = 0.35
            fontweight = 'bold' if source.is_current_run else 'normal'

            ax_box.text(i + 1 + x_offset, mean_val, f'{mean_val*100:.1f}',
                        ha='left', va='center', fontsize=7,
                        color=mean_color, fontweight=fontweight)
            ax_box.text(i + 1 + x_offset, median_val, f'{median_val*100:.1f}',
                        ha='left', va='center', fontsize=7,
                        color=median_color, fontweight=fontweight)

        legend_elements = [
            Line2D([0], [0], color=median_color, linewidth=2, linestyle='-', label='Median'),
            Line2D([0], [0], color=mean_color, linewidth=2, linestyle=(0, (3, 2)), label='Mean')
        ]
        ax_box.legend(handles=legend_elements, loc='upper right', fontsize=7)

    ax_box.set_ylabel('Test Accuracy')
    ax_box.set_title('Accuracy Distribution')
    ax_box.axhline(y=chance_level, color='gray', linestyle='--', alpha=0.5)

    # =========================================================================
    # Panel 3: 配对对比散点图（支持双配对：当前 vs 历史）
    # =========================================================================
    eegnet_sources = [s for s in data_sources if s.model_type == 'eegnet']
    cbramod_sources = [s for s in data_sources if s.model_type == 'cbramod']

    # 分离当前和历史数据源
    eegnet_current = next((s for s in eegnet_sources if s.is_current_run), None)
    eegnet_hist = next((s for s in eegnet_sources if not s.is_current_run), None)
    cbramod_current = next((s for s in cbramod_sources if s.is_current_run), None)
    cbramod_hist = next((s for s in cbramod_sources if not s.is_current_run), None)

    # 使用 EEGNet 作为 X 轴基准（优先使用当前数据）
    eegnet_baseline = eegnet_current or eegnet_hist

    # 配对颜色配置
    pair_colors = {
        'current': '#E94F37',   # 当前 CBraMod: 红色
        'historical': '#9B59B6',  # 历史 CBraMod: 紫色
    }

    all_accs = []  # 用于计算坐标轴范围
    has_any_pair = False

    if eegnet_baseline:
        eegnet_by_subj = {r.subject_id: r.test_acc_majority for r in eegnet_baseline.results}

        # 绘制当前配对：当前 CBraMod vs EEGNet
        if cbramod_current:
            cbramod_by_subj = {r.subject_id: r.test_acc_majority for r in cbramod_current.results}
            common = sorted(set(eegnet_by_subj.keys()) & set(cbramod_by_subj.keys()))

            if common:
                eegnet_accs = [eegnet_by_subj[s] for s in common]
                cbramod_accs = [cbramod_by_subj[s] for s in common]
                all_accs.extend(eegnet_accs + cbramod_accs)

                ax_scatter.scatter(eegnet_accs, cbramod_accs, s=100, alpha=0.9,
                                   c=pair_colors['current'], label='CBraMod (current)',
                                   edgecolors='black', linewidths=1)

                # 为当前运行添加被试标签
                for i, subj in enumerate(common):
                    ax_scatter.annotate(subj, (eegnet_accs[i], cbramod_accs[i]),
                                        xytext=(5, 5), textcoords='offset points', fontsize=7)
                has_any_pair = True

        # 绘制历史配对：历史 CBraMod vs EEGNet
        if cbramod_hist:
            cbramod_hist_by_subj = {r.subject_id: r.test_acc_majority for r in cbramod_hist.results}
            common_hist = sorted(set(eegnet_by_subj.keys()) & set(cbramod_hist_by_subj.keys()))

            if common_hist:
                eegnet_accs_hist = [eegnet_by_subj[s] for s in common_hist]
                cbramod_accs_hist = [cbramod_hist_by_subj[s] for s in common_hist]
                all_accs.extend(eegnet_accs_hist + cbramod_accs_hist)

                ax_scatter.scatter(eegnet_accs_hist, cbramod_accs_hist, s=80, alpha=0.5,
                                   c=pair_colors['historical'], label='CBraMod (hist)',
                                   edgecolors='gray', linewidths=0.5, marker='s')
                has_any_pair = True

        if has_any_pair and all_accs:
            lims = [min(all_accs) - 0.05, max(all_accs) + 0.05]
            ax_scatter.plot(lims, lims, 'k--', alpha=0.5, label='Equal')
            ax_scatter.set_xlim(lims)
            ax_scatter.set_ylim(lims)
            ax_scatter.set_xlabel(f'{eegnet_baseline.label} Accuracy')
            ax_scatter.set_ylabel('CBraMod Accuracy')
            ax_scatter.legend(loc='upper left', fontsize=7)
        else:
            ax_scatter.text(0.5, 0.5, 'No common subjects\nfor paired comparison',
                            ha='center', va='center', transform=ax_scatter.transAxes)
    else:
        ax_scatter.text(0.5, 0.5, 'Insufficient data\nfor paired comparison',
                        ha='center', va='center', transform=ax_scatter.transAxes)

    ax_scatter.set_title('CBraMod vs EEGNet (Paired Comparison)')

    # 保存
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    log_cache.info(f"Combined plot saved: {output_path}")
    plt.close()
