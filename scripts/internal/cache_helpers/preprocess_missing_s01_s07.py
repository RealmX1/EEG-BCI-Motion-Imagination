#!/usr/bin/env python
"""
针对性预处理脚本：填补 S01-S07 缺失的 3-class 配置

缺失配置：
- CBraMod online 3-class: S01-S07 (全部)
- EEGNet online 3-class: S05-S07 (部分)

用法：
    python scripts/preprocess_missing_s01_s07.py              # 预览模式
    python scripts/preprocess_missing_s01_s07.py --execute    # 实际预处理
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.data_loader import (
    FingerEEGDataset,
    PreprocessConfig,
    discover_available_subjects,
    get_session_folders_for_split,
)
from src.preprocessing.cache_manager import get_cache


def get_missing_configs():
    """返回需要预处理的配置列表"""
    configs = []

    # CBraMod 3-class for S01-S07
    for subject_num in range(1, 8):
        subject = f'S{subject_num:02d}'
        configs.append({
            'subject': subject,
            'model': 'cbramod',
            'task': 'ternary',
            'paradigm': 'imagery',
        })

    # EEGNet 3-class for S05-S07
    for subject_num in range(5, 8):
        subject = f'S{subject_num:02d}'
        configs.append({
            'subject': subject,
            'model': 'eegnet',
            'task': 'ternary',
            'paradigm': 'imagery',
        })

    return configs


def preprocess_config(subject, model, task, paradigm, dry_run=False):
    """预处理单个配置"""
    print(f"\n{'=' * 70}")
    print(f"Processing: {subject} | {model} | {task} | {paradigm}")
    print('=' * 70)

    # Determine target classes
    if task == 'binary':
        target_classes = [1, 4]  # Thumb vs Pinky
    elif task == 'ternary':
        target_classes = [1, 2, 4]  # Thumb, Index, Pinky
    else:
        target_classes = None  # All fingers

    # Create config based on model
    if model == 'cbramod':
        config = PreprocessConfig.for_cbramod()
    else:
        config = PreprocessConfig.for_eegnet()

    # Get cache
    cache = get_cache(cache_dir='caches/preprocessed', enabled=True)

    try:
        # Create dataset - this will trigger preprocessing if cache doesn't exist
        dataset = FingerEEGDataset(
            subject=subject,
            model_name=model,
            paradigm=paradigm,
            task=task,
            split='train',  # Use train split to get all training data
            use_cache=True,
            force_preprocess=not dry_run,  # Don't force in dry run mode
        )

        if dry_run:
            print(f"\n[DRY RUN] Dataset would be created with:")
            print(f"  Subject: {subject}")
            print(f"  Model: {model}")
            print(f"  Task: {task}")
            print(f"  Paradigm: {paradigm}")
            print(f"  Expected total samples: (would be calculated during creation)")
            return 1, 0
        else:
            print(f"\n[OK] Dataset created successfully")
            print(f"  Total samples: {len(dataset)}")
            return 1, 1

    except Exception as e:
        print(f"\n[ERROR] Failed to create dataset: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0


def main():
    parser = argparse.ArgumentParser(
        description="针对性预处理 S01-S07 缺失的 3-class 配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--execute',
        action='store_true',
        help="实际执行预处理（默认为预览模式）"
    )

    parser.add_argument(
        '--subjects',
        nargs='+',
        help="指定要处理的被试（默认全部）"
    )

    args = parser.parse_args()

    # Get configs to process
    all_configs = get_missing_configs()

    # Filter by subjects if specified
    if args.subjects:
        all_configs = [c for c in all_configs if c['subject'] in args.subjects]

    print("=" * 70)
    print("S01-S07 缺失配置预处理")
    print("=" * 70)
    print()

    if args.execute:
        print("[MODE] 实际预处理")
    else:
        print("[MODE] 预览模式 (使用 --execute 实际执行)")

    print()
    print(f"将处理 {len(all_configs)} 个配置:")

    # Group by subject
    from collections import defaultdict
    by_subject = defaultdict(list)
    for config in all_configs:
        by_subject[config['subject']].append(f"{config['model']} {config['task']}")

    for subject in sorted(by_subject.keys()):
        configs_str = ', '.join(by_subject[subject])
        print(f"  {subject}: {configs_str}")

    print()

    if not args.execute:
        confirm = input("继续预览？[y/N]: ").strip().lower()
        if confirm != 'y':
            print("操作已取消。")
            return

    # Process each config
    total_processed = 0
    total_files = 0

    for i, config_info in enumerate(all_configs, 1):
        print(f"\n[{i}/{len(all_configs)}]")

        n_files, n_cached = preprocess_config(
            subject=config_info['subject'],
            model=config_info['model'],
            task=config_info['task'],
            paradigm=config_info['paradigm'],
            dry_run=not args.execute
        )

        total_files += n_files
        total_processed += n_cached

    print()
    print("=" * 70)
    print("[SUMMARY]")
    print("=" * 70)
    print(f"Total configurations: {len(all_configs)}")
    print(f"Total files checked: {total_files}")
    print(f"Files cached/processed: {total_processed}")

    if args.execute:
        print()
        print("[NEXT STEPS]")
        print("1. Verify cache with: python scripts/cache_helper.py --stats")
        print("2. Test training with: python scripts/run_single_model.py --subject S01 --model cbramod --task ternary")
    else:
        print()
        print("Use --execute to actually preprocess the data")


if __name__ == '__main__':
    main()
