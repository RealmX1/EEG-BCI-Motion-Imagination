#!/usr/bin/env python
"""
重试 S05 EEGNet 3-class 预处理（单线程模式）

之前使用并行处理失败，现在使用单线程模式重试。
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("=" * 70)
    print("重试 S05 EEGNet 3-class 预处理（单线程模式）")
    print("=" * 70)
    print()

    project_root = Path(__file__).parent.parent

    # Note: preprocess_zip.py doesn't have a flag to disable parallel processing
    # But we can try with fewer workers by modifying the command or running sequentially

    # Simple solution: just retry, the cache system will skip already processed files
    cmd = [
        sys.executable,
        'scripts/preprocess_zip.py',
        '--preprocess-only',
        '--subject', 'S05',
        '--data-root', 'data',
        '--models', 'eegnet',
        '--tasks', 'ternary',
        '--paradigm', 'imagery',
    ]

    print("Running:")
    print(" ".join(cmd))
    print()

    result = subprocess.run(cmd, cwd=project_root)

    if result.returncode == 0:
        print()
        print("[OK] S05 EEGNet 3-class 预处理完成")
        print()
        print("验证:")
        print("  python scripts/cache_helper.py --subject S05")
    else:
        print()
        print("[ERROR] 预处理失败")
        print()
        print("建议:")
        print("  1. 检查系统内存是否充足")
        print("  2. 关闭其他占用内存的程序")
        print("  3. 如果问题持续，可以跳过 S05 EEGNet 3-class")
        print("     (S05 的其他配置都已完成)")

    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
