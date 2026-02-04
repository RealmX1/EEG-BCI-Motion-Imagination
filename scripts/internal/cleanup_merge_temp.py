#!/usr/bin/env python
"""
清理缓存合并产生的临时文件

用法:
    python scripts/cleanup_merge_temp.py              # 显示要删除的文件
    python scripts/cleanup_merge_temp.py --execute    # 实际删除
"""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="清理缓存合并临时文件")
    parser.add_argument(
        '--execute',
        action='store_true',
        help="实际删除文件（默认仅显示）"
    )
    args = parser.parse_args()

    root = Path(".")

    # 要清理的项目
    items_to_clean = [
        # 临时索引文件
        root / "caches" / ".cache_index_part2_cleaned.json",
        # dist 源目录（可选，约16GB）
        # root / "caches" / "dist",
    ]

    print("=" * 60)
    print("缓存合并临时文件清理")
    print("=" * 60)
    print()

    total_size = 0
    existing_items = []

    for item in items_to_clean:
        if item.exists():
            if item.is_file():
                size = item.stat().st_size
                total_size += size
                size_str = f"{size / 1024:.1f} KB" if size < 1024*1024 else f"{size / (1024*1024):.1f} MB"
                print(f"[文件] {item}")
                print(f"  大小: {size_str}")
            elif item.is_dir():
                # 计算目录大小
                size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                total_size += size
                size_str = f"{size / (1024**3):.2f} GB" if size > 1024**3 else f"{size / (1024**2):.1f} MB"
                print(f"[目录] {item}")
                print(f"  大小: {size_str}")

            existing_items.append(item)
            print()

    if not existing_items:
        print("没有需要清理的临时文件。")
        return

    print(f"总大小: {total_size / (1024**2):.1f} MB")
    print()

    if args.execute:
        print("开始删除...")
        for item in existing_items:
            try:
                if item.is_file():
                    item.unlink()
                    print(f"  [OK] 已删除: {item}")
                elif item.is_dir():
                    shutil.rmtree(item)
                    print(f"  [OK] 已删除: {item}")
            except Exception as e:
                print(f"  [ERROR] 删除失败 {item}: {e}")

        print()
        print("清理完成。")
    else:
        print("[预览模式] 使用 --execute 参数实际删除文件")
        print()
        print("注意: dist/ 目录未包含在清理列表中")
        print("      如需删除 caches/dist/，请手动删除或取消注释脚本中的对应行")


if __name__ == '__main__':
    main()
