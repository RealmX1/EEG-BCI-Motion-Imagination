#!/usr/bin/env python
"""
Extra Sessions 图表重绘脚本（从 JSON cache 加载）.

由于 run_extra_sessions.py 不支持 --replot，此脚本从已有
JSON cache 文件加载结果并重新生成图表。

Usage:
    uv run python scripts/paper/generate_extra_sessions_plots.py --task binary --run-tag 20260324_2131
    uv run python scripts/paper/generate_extra_sessions_plots.py --task ternary --run-tag 20260331_0827
"""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def find_cache_file(run_tag: str, task: str, results_dir: str = 'results') -> Path:
    """查找匹配 run_tag 的 extra_sessions JSON cache 文件."""
    pattern = f'{run_tag}_extra_sessions_*_{task}.json'
    matches = list(Path(results_dir).glob(pattern))
    if not matches:
        # Try broader pattern
        pattern2 = f'{run_tag}_*_{task}.json'
        matches = list(Path(results_dir).glob(pattern2))
    if not matches:
        logger.error(f'No cache file found matching {pattern} in {results_dir}/')
        sys.exit(1)
    return matches[0]


def load_extra_sessions_cache(cache_path: Path) -> dict:
    """加载 extra sessions JSON cache 并提取绘图所需数据.

    Cache format: data['results'][model][subject][step]
    where step is 'baseline', 'sess03', 'sess04', 'sess05'
    """
    with open(cache_path) as f:
        cache = json.load(f)

    # The actual results are nested under 'results' key
    results_data = cache.get('results', cache)

    all_results = {}
    subjects_with_sessions = {}

    for model_type, model_data in results_data.items():
        if model_type in ('metadata', 'comparison', 'baseline_run_tags'):
            continue
        if not isinstance(model_data, dict):
            continue
        all_results[model_type] = {}
        for subject_id, subj_data in model_data.items():
            if not isinstance(subj_data, dict):
                continue
            all_results[model_type][subject_id] = subj_data
            # 从 subject 数据推断可用 sessions
            sessions = []
            for key in subj_data:
                if key.startswith('sess'):
                    sess_num = int(key.replace('sess', ''))
                    sessions.append(sess_num)
            if sessions:
                subjects_with_sessions[subject_id] = sorted(sessions)

    return all_results, subjects_with_sessions


def main():
    parser = argparse.ArgumentParser(description='Extra Sessions 图表重绘')
    parser.add_argument('--task', required=True, choices=['binary', 'ternary'])
    parser.add_argument('--run-tag', required=True, help='Run tag (e.g., 20260324_2131)')
    parser.add_argument('--paradigm', default='imagery')
    parser.add_argument('--output-dir', default='paper/figures')
    args = parser.parse_args()

    # 查找并加载 cache
    cache_path = find_cache_file(args.run_tag, args.task)
    logger.info(f'Loading cache: {cache_path}')
    all_results, subjects_with_sessions = load_extra_sessions_cache(cache_path)

    if not all_results:
        logger.error('No model results found in cache')
        sys.exit(1)

    logger.info(f'Models: {list(all_results.keys())}')
    logger.info(f'Subjects with sessions: {len(subjects_with_sessions)}')

    # 生成图表
    from src.visualization.extra_sessions import generate_extra_sessions_combined_plot

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_path = output_dir / f'extra_sessions_{args.task}.png'
    generate_extra_sessions_combined_plot(
        all_results=all_results,
        subjects_with_sessions=subjects_with_sessions,
        output_path=str(plot_path),
        paradigm=args.paradigm,
        task=args.task,
    )
    logger.info(f'Plot saved: {plot_path}')


if __name__ == '__main__':
    main()
