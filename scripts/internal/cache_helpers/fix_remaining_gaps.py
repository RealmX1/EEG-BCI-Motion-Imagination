#!/usr/bin/env python
"""
修复剩余的缓存缺口

剩余缺失：
- S01: CBraMod online 3-class
- S05: EEGNet online 3-class
"""

import subprocess
import sys
from pathlib import Path


def preprocess_subject(subject, model, task='ternary'):
    """预处理单个被试配置"""
    project_root = Path(__file__).parent.parent

    cmd = [
        sys.executable,
        'scripts/preprocess_zip.py',
        '--preprocess-only',
        '--subject', subject,
        '--data-root', 'data',
        '--models', model,
        '--tasks', task,
        '--paradigm', 'imagery',
    ]

    print(f"\n{'=' * 70}")
    print(f"处理: {subject} | {model} | {task}")
    print('=' * 70)

    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode


def main():
    print("=" * 70)
    print("修复剩余缓存缺口")
    print("=" * 70)
    print()
    print("将处理:")
    print("  1. S01 - CBraMod 3-class")
    print("  2. S05 - EEGNet 3-class")
    print()

    # Process S01 CBraMod
    ret1 = preprocess_subject('S01', 'cbramod', 'ternary')

    if ret1 != 0:
        print(f"\n[WARNING] S01 CBraMod 预处理失败 (返回码: {ret1})")
    else:
        print(f"\n[OK] S01 CBraMod 完成")

    # Process S05 EEGNet
    ret2 = preprocess_subject('S05', 'eegnet', 'ternary')

    if ret2 != 0:
        print(f"\n[WARNING] S05 EEGNet 预处理失败 (返回码: {ret2})")
    else:
        print(f"\n[OK] S05 EEGNet 完成")

    # Summary
    print()
    print("=" * 70)
    print("处理完成")
    print("=" * 70)
    print()

    if ret1 == 0 and ret2 == 0:
        print("[SUCCESS] 所有缺口已填补！")
        print()
        print("验证: python scripts/cache_helper.py --stats")
    else:
        print("[PARTIAL] 部分处理失败")
        print()
        if ret1 != 0:
            print("  - S01 CBraMod: 失败")
        if ret2 != 0:
            print("  - S05 EEGNet: 失败 (可能是内存问题)")
        print()
        print("这些配置可以跳过，不影响主要实验。")

    return 0 if (ret1 == 0 and ret2 == 0) else 1


if __name__ == '__main__':
    sys.exit(main())
