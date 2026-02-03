#!/usr/bin/env python
"""
填补 S01-S07 缺失的 3-class 缓存

通过调用 preprocess_zip.py 来完成预处理。

缺失配置：
- CBraMod online 3-class: S01-S07 (全部)
- EEGNet online 3-class: S05-S07 (部分)

用法：
    python scripts/fill_missing_caches.py           # 执行预处理
"""

import subprocess
import sys
from pathlib import Path

def run_preprocessing(subjects, models, tasks, paradigm='imagery'):
    """运行 preprocess_zip.py (每个被试单独运行)"""
    failed = []
    project_root = Path(__file__).parent.parent

    for subject in subjects:
        print(f"\n  Processing {subject}...")

        cmd = [
            sys.executable,
            'scripts/preprocess_zip.py',
            '--preprocess-only',
            '--subject', subject,
            '--data-root', 'data',  # Specify correct data directory
            '--models', *models,
            '--tasks', *tasks,
            '--paradigm', paradigm,
        ]

        result = subprocess.run(cmd, cwd=project_root)

        if result.returncode != 0:
            print(f"  [ERROR] {subject} failed with return code {result.returncode}")
            failed.append(subject)
        else:
            print(f"  [OK] {subject} completed")

    if failed:
        print(f"\n[WARNING] {len(failed)} subjects failed: {failed}")
        return 1

    return 0


def main():
    print("=" * 70)
    print("填补 S01-S07 缺失的 3-class 缓存")
    print("=" * 70)
    print()
    print("缺失配置:")
    print("  - CBraMod 3-class: S01-S07 (全部)")
    print("  - EEGNet 3-class: S05-S07 (部分)")
    print()

    # Step 1: CBraMod 3-class for all S01-S07
    print("[1/2] 预处理 CBraMod 3-class (S01-S07)")
    print("-" * 70)
    ret1 = run_preprocessing(
        subjects=['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07'],
        models=['cbramod'],
        tasks=['ternary'],
        paradigm='imagery'
    )

    if ret1 != 0:
        print(f"\n[ERROR] Step 1 failed with return code {ret1}")
        return ret1

    print()
    print("[OK] Step 1 completed")
    print()

    # Step 2: EEGNet 3-class for S05-S07
    print("[2/2] 预处理 EEGNet 3-class (S05-S07)")
    print("-" * 70)
    ret2 = run_preprocessing(
        subjects=['S05', 'S06', 'S07'],
        models=['eegnet'],
        tasks=['ternary'],
        paradigm='imagery'
    )

    if ret2 != 0:
        print(f"\n[ERROR] Step 2 failed with return code {ret2}")
        return ret2

    print()
    print("[OK] Step 2 completed")
    print()

    # Done
    print("=" * 70)
    print("[SUCCESS] 所有缺失缓存已填补！")
    print("=" * 70)
    print()
    print("后续步骤:")
    print("  1. 验证缓存: python scripts/cache_helper.py --stats")
    print("  2. 测试训练: python scripts/run_single_model.py --subject S01 --model cbramod --task ternary")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
