#!/usr/bin/env python3
"""
对比 CBraMod 不同学习率调度器的训练效果。

比较:
- Cosine Annealing (当前默认): T_max = total_steps // 2, 快速衰减
- WSD (Warmup-Stable-Decay): 10% warmup, 50% stable, 40% decay

使用方法:
    # 单被试对比 (快速验证)
    uv run python scripts/compare_schedulers.py --subject S01

    # 多被试对比 (完整实验)
    uv run python scripts/compare_schedulers.py --subjects S01 S02 S03

    # 全部被试
    uv run python scripts/compare_schedulers.py --all
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_within_subject import (
    train_subject_simple,
    get_default_config,
    discover_available_subjects,
)
import time
from src.utils.timing import Colors, colored


def run_comparison(
    subject: str,
    task: str = 'binary',
    paradigm: str = 'imagery',
    schedulers_to_run: list = None,
) -> dict:
    """
    运行单个被试的调度器对比实验。

    Args:
        subject: 被试 ID (如 'S01')
        task: 任务类型 ('binary', 'ternary', 'quaternary')
        paradigm: 范式 ('imagery' 或 'movement')

    Returns:
        包含两个调度器结果的字典
    """
    results = {
        'subject': subject,
        'task': task,
        'paradigm': paradigm,
        'schedulers': {},
    }

    # Scheduler configs: cosine uses 25 epochs, WSD uses 25 epochs
    all_scheduler_configs = [
        {'name': 'cosine', 'epochs': 25},
        {'name': 'wsd', 'epochs': 25},
    ]

    # Filter by requested schedulers
    if schedulers_to_run:
        scheduler_configs = [c for c in all_scheduler_configs if c['name'] in schedulers_to_run]
    else:
        scheduler_configs = all_scheduler_configs

    for sched_config in scheduler_configs:
        scheduler = sched_config['name']
        epochs = sched_config['epochs']

        print(f"\n{colored('=' * 60, Colors.CYAN)}")
        print(f"{colored(f'  {subject} | Scheduler: {scheduler.upper()} | Epochs: {epochs}', Colors.BRIGHT_CYAN, bold=True)}")
        print(f"{colored('=' * 60, Colors.CYAN)}\n")

        # 运行训练
        start_time = time.perf_counter()

        result = train_subject_simple(
            subject_id=subject,
            model_type='cbramod',
            task=task,
            paradigm=paradigm,
            config_overrides={'training': {'scheduler': scheduler, 'epochs': epochs}},
        )

        elapsed = time.perf_counter() - start_time

        # 保存结果 (keys match train_subject_simple return dict)
        results['schedulers'][scheduler] = {
            'epochs': epochs,
            'test_acc': result.get('test_accuracy', 0.0),
            'test_majority_acc': result.get('test_accuracy_majority', 0.0),
            'best_val_acc': result.get('best_val_acc', 0.0),
            'best_epoch': result.get('best_epoch', 0),
            'training_time': elapsed,
        }

        test_maj_acc = result.get('test_accuracy_majority', 0.0)
        print(f"\n{colored(f'[{scheduler.upper()}] Test Accuracy: {test_maj_acc:.4f}', Colors.BRIGHT_GREEN)}")

    return results


def print_comparison_table(all_results: list):
    """打印对比结果表格。"""
    print(f"\n{colored('=' * 80, Colors.CYAN)}")
    print(f"{colored('  学习率调度器对比结果 (CBraMod)', Colors.BRIGHT_CYAN, bold=True)}")
    print(f"{colored('  Cosine: 25 epochs | WSD: 50 epochs', Colors.DIM)}")
    print(f"{colored('=' * 80, Colors.CYAN)}\n")

    # 表头
    header = f"{'Subject':<10} | {'Cosine(25)':<12} | {'WSD(50)':<12} | {'Δ (WSD-Cosine)':<14} | {'Winner':<8}"
    print(colored(header, Colors.WHITE, bold=True))
    print("-" * 70)

    cosine_wins = 0
    wsd_wins = 0
    cosine_accs = []
    wsd_accs = []

    for result in all_results:
        subject = result['subject']
        cosine = result['schedulers'].get('cosine', {})
        wsd = result['schedulers'].get('wsd', {})

        cosine_acc = cosine.get('test_majority_acc', 0.0)
        wsd_acc = wsd.get('test_majority_acc', 0.0)

        cosine_accs.append(cosine_acc)
        wsd_accs.append(wsd_acc)

        delta = wsd_acc - cosine_acc
        winner = "WSD" if delta > 0.001 else ("Cosine" if delta < -0.001 else "Tie")

        if winner == "Cosine":
            cosine_wins += 1
            winner_color = Colors.BRIGHT_RED
        elif winner == "WSD":
            wsd_wins += 1
            winner_color = Colors.BRIGHT_GREEN
        else:
            winner_color = Colors.YELLOW

        delta_str = f"{delta:+.4f}"
        if delta > 0:
            delta_str = colored(delta_str, Colors.BRIGHT_GREEN)
        elif delta < 0:
            delta_str = colored(delta_str, Colors.BRIGHT_RED)

        print(f"{subject:<10} | {cosine_acc:<12.4f} | {wsd_acc:<12.4f} | {delta_str:<14} | {colored(winner, winner_color)}")

    # 汇总
    print("-" * 70)
    avg_cosine = sum(cosine_accs) / len(cosine_accs) if cosine_accs else 0
    avg_wsd = sum(wsd_accs) / len(wsd_accs) if wsd_accs else 0
    avg_delta = avg_wsd - avg_cosine

    print(f"{'Average':<10} | {avg_cosine:<12.4f} | {avg_wsd:<12.4f} | {avg_delta:+.4f}")
    print(f"\n{colored(f'Wins: Cosine={cosine_wins}, WSD={wsd_wins}', Colors.BRIGHT_YELLOW)}")


def main():
    parser = argparse.ArgumentParser(
        description="对比 CBraMod 学习率调度器 (Cosine vs WSD)"
    )
    parser.add_argument(
        '--subject', '-s',
        type=str,
        help='单个被试 ID (如 S01)'
    )
    parser.add_argument(
        '--subjects',
        nargs='+',
        type=str,
        help='多个被试 ID (如 S01 S02 S03)'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='运行所有可用被试'
    )
    parser.add_argument(
        '--task', '-t',
        type=str,
        default='binary',
        choices=['binary', 'ternary', 'quaternary'],
        help='分类任务 (默认: binary)'
    )
    parser.add_argument(
        '--paradigm', '-p',
        type=str,
        default='imagery',
        choices=['imagery', 'movement'],
        help='范式 (默认: imagery)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='输出 JSON 文件路径'
    )
    parser.add_argument(
        '--scheduler',
        type=str,
        nargs='+',
        choices=['cosine', 'wsd'],
        help='只运行指定的调度器 (如: --scheduler wsd)'
    )

    args = parser.parse_args()

    # 确定被试列表
    data_root = PROJECT_ROOT / 'data'
    if args.all:
        subjects = discover_available_subjects(data_root)
    elif args.subjects:
        subjects = args.subjects
    elif args.subject:
        subjects = [args.subject]
    else:
        parser.error("请指定 --subject, --subjects, 或 --all")

    print(f"\n{colored('学习率调度器对比实验', Colors.BRIGHT_CYAN, bold=True)}")
    print(f"被试: {', '.join(subjects)}")
    print(f"任务: {args.task}")
    print(f"范式: {args.paradigm}")
    schedulers_str = ', '.join(args.scheduler).upper() if args.scheduler else "Cosine vs WSD"
    print(f"调度器: {schedulers_str}")

    # 运行对比
    all_results = []
    total_start = time.perf_counter()

    for subject in subjects:
        result = run_comparison(
            subject=subject,
            task=args.task,
            paradigm=args.paradigm,
            schedulers_to_run=args.scheduler,
        )
        all_results.append(result)

    total_elapsed = time.perf_counter() - total_start

    # 打印对比表格
    print_comparison_table(all_results)

    print(f"\n{colored(f'总耗时: {total_elapsed:.1f}s', Colors.DIM)}")

    # 保存结果
    output_path = args.output
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_path = PROJECT_ROOT / 'results' / f'{timestamp}_scheduler_comparison.json'

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'timestamp': datetime.now().isoformat(),
        'task': args.task,
        'paradigm': args.paradigm,
        'subjects': subjects,
        'results': all_results,
        'summary': {
            'avg_cosine': sum(r['schedulers'].get('cosine', {}).get('test_majority_acc', 0) for r in all_results) / len(all_results),
            'avg_wsd': sum(r['schedulers'].get('wsd', {}).get('test_majority_acc', 0) for r in all_results) / len(all_results),
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"{colored(f'结果已保存: {output_path}', Colors.DIM)}")


if __name__ == '__main__':
    main()
