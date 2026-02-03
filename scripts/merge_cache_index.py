#!/usr/bin/env python
"""
缓存索引合并工具 (Cache Index Merge Tool)

三步流程合并预处理缓存：
1. 清理 Part02 索引 - 移除不存在文件的引用
2. 合并索引 - 将清理后的 Part02 与原始索引合并
3. 移动文件 - 将 dist/ 中的文件移动到 caches/preprocessed/

用法 (Usage):
    python scripts/merge_cache_index.py validate   # 预检查
    python scripts/merge_cache_index.py clean      # 清理 Part02 索引
    python scripts/merge_cache_index.py merge      # 合并索引
    python scripts/merge_cache_index.py move       # 移动文件
    python scripts/merge_cache_index.py verify     # 验证结果
    python scripts/merge_cache_index.py full       # 完整流程 (推荐)
"""

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any


class CacheMergeManager:
    """缓存合并管理器"""

    CACHE_VERSION = "3.0"

    def __init__(self, dry_run: bool = False, no_interactive: bool = False):
        self.dry_run = dry_run
        self.no_interactive = no_interactive
        self.root = Path(".")

        # 关键路径
        self.orig_cache_dir = self.root / "caches" / "preprocessed"
        self.orig_index_path = self.orig_cache_dir / ".cache_index.json"

        self.dist_dir = self.root / "caches" / "dist"
        self.part1_cache_dir = self.dist_dir / "eeg_bci_cache_part01" / "caches" / "preprocessed"
        self.part2_cache_dir = self.dist_dir / "eeg_bci_cache_part02" / "caches" / "preprocessed"
        self.part2_index_path = self.part2_cache_dir / ".cache_index.json"

        # 临时文件
        self.cleaned_index_path = self.root / "caches" / ".cache_index_part2_cleaned.json"

        # 备份目录
        self.backup_dir = None

    # ==================== 工具函数 ====================

    def load_index(self, path: Path) -> Dict[str, Any]:
        """加载索引文件"""
        if not path.exists():
            raise FileNotFoundError(f"索引文件不存在: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_index(self, index: Dict[str, Any], path: Path) -> None:
        """保存索引文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

    def scan_h5_files(self, directory: Path) -> Set[str]:
        """扫描目录中所有 .h5 文件，返回文件名集合（不含扩展名）"""
        if not directory.exists():
            return set()
        return {f.stem for f in directory.glob("*.h5")}

    def normalize_path(self, path: str) -> str:
        """统一路径格式为 Unix 相对路径"""
        # 替换反斜杠为正斜杠
        path = path.replace('\\', '/')

        # 如果是绝对路径，提取 data/ 开始的部分
        if 'data/' in path:
            idx = path.index('data/')
            return path[idx:]

        return path

    def format_size(self, size_bytes: float) -> str:
        """格式化字节大小"""
        if size_bytes < 1024:
            return f"{size_bytes:.1f} B"
        elif size_bytes < 1024 ** 2:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 ** 3:
            return f"{size_bytes / (1024 ** 2):.1f} MB"
        else:
            return f"{size_bytes / (1024 ** 3):.2f} GB"

    def get_disk_space(self, path: Path) -> Tuple[int, int]:
        """获取磁盘空间信息（总空间，可用空间）"""
        import shutil
        stat = shutil.disk_usage(path)
        return stat.total, stat.free

    def create_backup(self) -> Path:
        """创建备份目录并备份原始索引"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.root / "caches" / f".backup_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)

        if self.orig_index_path.exists():
            shutil.copy2(
                self.orig_index_path,
                backup_dir / ".cache_index.json.orig"
            )
            print(f"[OK] 原始索引已备份到: {backup_dir}")

        self.backup_dir = backup_dir
        return backup_dir

    # ==================== 步骤 1: 预检查 ====================

    def validate(self) -> bool:
        """预检查所有条件"""
        print("=" * 70)
        print("缓存合并预检查 (Cache Merge Validation)")
        print("=" * 70)
        print()

        all_ok = True

        # 1. 检查索引文件
        print("[索引文件检查]")
        if self.orig_index_path.exists():
            orig_index = self.load_index(self.orig_index_path)
            orig_count = len(orig_index.get('entries', {}))
            print(f"  [OK] 原始索引: {orig_count} 条 ({self.orig_index_path})")
        else:
            print(f"  [FAIL] 原始索引不存在: {self.orig_index_path}")
            all_ok = False

        if self.part2_index_path.exists():
            part2_index = self.load_index(self.part2_index_path)
            part2_count = len(part2_index.get('entries', {}))
            print(f"  [OK] Part02 索引: {part2_count} 条 ({self.part2_index_path})")
        else:
            print(f"  [FAIL] Part02 索引不存在: {self.part2_index_path}")
            all_ok = False

        print()

        # 2. 检查目录和文件
        print("[缓存文件检查]")
        orig_files = self.scan_h5_files(self.orig_cache_dir)
        print(f"  原始目录: {len(orig_files)} 个 .h5 文件")

        part1_files = self.scan_h5_files(self.part1_cache_dir)
        part2_files = self.scan_h5_files(self.part2_cache_dir)
        print(f"  Part01 目录: {len(part1_files)} 个 .h5 文件")
        print(f"  Part02 目录: {len(part2_files)} 个 .h5 文件")
        print(f"  dist/ 总计: {len(part1_files) + len(part2_files)} 个 .h5 文件")
        print()

        # 3. 检查 Part02 索引与文件的对应关系
        if self.part2_index_path.exists():
            print("[Part02 索引与文件对应关系]")
            part2_index = self.load_index(self.part2_index_path)
            part2_entries = part2_index.get('entries', {})

            # 所有 dist/ 中的文件
            dist_files = part1_files | part2_files

            # 检查索引中有多少条目在 dist/ 中有文件
            entries_with_files = sum(1 for key in part2_entries if key in dist_files)
            entries_no_files = len(part2_entries) - entries_with_files

            print(f"  Part02 索引条目: {len(part2_entries)}")
            print(f"  有文件: {entries_with_files} ({entries_with_files / len(part2_entries) * 100:.1f}%)")
            print(f"  无文件: {entries_no_files} ({entries_no_files / len(part2_entries) * 100:.1f}%)")

            # 统计缺失文件的被试分布
            missing_by_subject = {}
            for key, entry in part2_entries.items():
                if key not in dist_files:
                    subject = entry.get('subject', 'unknown')
                    missing_by_subject[subject] = missing_by_subject.get(subject, 0) + 1

            if missing_by_subject:
                print(f"\n  缺失文件按被试分布:")
                for subject in sorted(missing_by_subject.keys()):
                    count = missing_by_subject[subject]
                    print(f"    {subject}: {count} 个")

            print()

        # 4. 预期结果
        print("[预期合并结果]")
        if self.orig_index_path.exists() and self.part2_index_path.exists():
            expected_cleaned = len(part1_files) + len(part2_files)
            expected_merged = orig_count + expected_cleaned
            expected_files = len(orig_files) + expected_cleaned

            print(f"  清理后 Part02 索引: {expected_cleaned} 条")
            print(f"  合并后总索引: {expected_merged} 条 ({orig_count} 原始 + {expected_cleaned} 新增)")
            print(f"  最终文件数: {expected_files} 个 ({len(orig_files)} 原始 + {expected_cleaned} 新增)")
            print()

        # 5. 磁盘空间检查
        print("[磁盘空间检查]")
        try:
            total, free = self.get_disk_space(self.orig_cache_dir)

            # 估算需要的空间（Part01 + Part02 文件总大小）
            part1_size = sum(f.stat().st_size for f in self.part1_cache_dir.glob("*.h5")) if self.part1_cache_dir.exists() else 0
            part2_size = sum(f.stat().st_size for f in self.part2_cache_dir.glob("*.h5")) if self.part2_cache_dir.exists() else 0
            needed = part1_size + part2_size

            print(f"  需要移动: {self.format_size(needed)}")
            print(f"  磁盘可用: {self.format_size(free)}")

            if free > needed * 1.2:  # 20% 安全边际
                print(f"  状态: [OK] 足够")
            else:
                print(f"  状态: [FAIL] 空间不足")
                all_ok = False
        except Exception as e:
            print(f"  无法检查磁盘空间: {e}")
            all_ok = False

        print()

        # 总结
        if all_ok:
            print("[OK] 所有检查通过，可以继续合并流程")
        else:
            print("[FAIL] 部分检查失败，请解决问题后再继续")

        print()
        return all_ok

    # ==================== 步骤 2: 清理 Part02 索引 ====================

    def clean_part2_index(self) -> Tuple[Dict[str, Any], int]:
        """清理 Part02 索引，只保留在 dist/ 中有文件的条目"""
        print("=" * 70)
        print("步骤 1: 清理 Part02 索引 (Clean Part02 Index)")
        print("=" * 70)
        print()

        # 加载 Part02 索引
        print(f"加载 Part02 索引: {self.part2_index_path}")
        part2_index = self.load_index(self.part2_index_path)
        part2_entries = part2_index.get('entries', {})
        print(f"  原始条目数: {len(part2_entries)}")
        print()

        # 扫描 dist/ 中所有文件
        print("扫描 dist/ 中的缓存文件...")
        part1_files = self.scan_h5_files(self.part1_cache_dir)
        part2_files = self.scan_h5_files(self.part2_cache_dir)
        dist_files = part1_files | part2_files

        print(f"  Part01: {len(part1_files)} 个文件")
        print(f"  Part02: {len(part2_files)} 个文件")
        print(f"  总计: {len(dist_files)} 个文件")
        print()

        # 过滤索引
        print("过滤索引条目...")
        cleaned_entries = {}
        removed_count = 0

        for key, entry in part2_entries.items():
            if key in dist_files:
                cleaned_entries[key] = entry
            else:
                removed_count += 1

        cleaned_index = {
            'version': self.CACHE_VERSION,
            'entries': cleaned_entries
        }

        print(f"  保留条目: {len(cleaned_entries)} (有文件)")
        print(f"  移除条目: {removed_count} (无文件)")
        print()

        # 保存清理后的索引到临时文件
        if not self.dry_run:
            self.save_index(cleaned_index, self.cleaned_index_path)
            print(f"[OK] 清理后索引已保存到: {self.cleaned_index_path}")
        else:
            print(f"[DRY RUN] 将保存清理后索引到: {self.cleaned_index_path}")

        print()
        return cleaned_index, removed_count

    # ==================== 步骤 3: 合并索引 ====================

    def merge_indices(self) -> Tuple[Dict[str, Any], int]:
        """合并原始索引和清理后的 Part02 索引"""
        print("=" * 70)
        print("步骤 2: 合并索引 (Merge Indices)")
        print("=" * 70)
        print()

        # 加载原始索引
        print(f"加载原始索引: {self.orig_index_path}")
        orig_index = self.load_index(self.orig_index_path)
        orig_entries = orig_index.get('entries', {})
        print(f"  原始条目数: {len(orig_entries)}")
        print()

        # 加载清理后的 Part02 索引
        print(f"加载清理后的 Part02 索引: {self.cleaned_index_path}")
        if not self.cleaned_index_path.exists():
            raise FileNotFoundError(
                f"清理后的索引不存在。请先运行 'clean' 步骤。\n"
                f"路径: {self.cleaned_index_path}"
            )

        cleaned_index = self.load_index(self.cleaned_index_path)
        cleaned_entries = cleaned_index.get('entries', {})
        print(f"  Part02 条目数: {len(cleaned_entries)}")
        print()

        # 合并索引
        print("合并索引...")
        merged_entries = {}
        overlap_count = 0

        # 1. 添加原始索引（标准化路径）
        for key, entry in orig_entries.items():
            entry_copy = entry.copy()
            entry_copy['source_file'] = self.normalize_path(entry['source_file'])
            merged_entries[key] = entry_copy

        # 2. 添加清理后的 Part02 索引（标准化路径）
        for key, entry in cleaned_entries.items():
            if key in merged_entries:
                overlap_count += 1
                print(f"  警告: 发现重复键 {key[:8]}，将保留原始索引中的条目")
            else:
                entry_copy = entry.copy()
                entry_copy['source_file'] = self.normalize_path(entry['source_file'])
                merged_entries[key] = entry_copy

        merged_index = {
            'version': self.CACHE_VERSION,
            'entries': merged_entries
        }

        print(f"  合并后条目数: {len(merged_entries)}")
        print(f"  重复键数: {overlap_count}")
        print()

        # 创建备份
        if not self.dry_run:
            backup_dir = self.create_backup()
            print()

        # 保存合并后的索引
        if not self.dry_run:
            # 原子性保存（先写临时文件，再替换）
            tmp_path = self.orig_cache_dir / ".cache_index.json.tmp"
            self.save_index(merged_index, tmp_path)
            tmp_path.replace(self.orig_index_path)
            print(f"[OK] 合并后索引已保存到: {self.orig_index_path}")
        else:
            print(f"[DRY RUN] 将保存合并后索引到: {self.orig_index_path}")

        print()
        return merged_index, overlap_count

    # ==================== 步骤 4: 移动文件 ====================

    def move_files(self) -> Tuple[int, int, List[str]]:
        """移动 Part01 和 Part02 文件到 caches/preprocessed/"""
        print("=" * 70)
        print("步骤 3: 移动文件 (Move Files)")
        print("=" * 70)
        print()

        moved_count = 0
        skipped_count = 0
        errors = []

        # 准备移动日志
        if not self.dry_run and self.backup_dir:
            move_log = self.backup_dir / "move_log.txt"
            log_file = open(move_log, 'w', encoding='utf-8')
        else:
            log_file = None

        # 移动 Part01 文件
        if self.part1_cache_dir.exists():
            print(f"移动 Part01 文件: {self.part1_cache_dir}")
            part1_files = list(self.part1_cache_dir.glob("*.h5"))
            print(f"  找到 {len(part1_files)} 个文件")

            for src in part1_files:
                dst = self.orig_cache_dir / src.name

                if dst.exists():
                    # 检查文件大小
                    if src.stat().st_size == dst.stat().st_size:
                        skipped_count += 1
                        if self.dry_run:
                            print(f"  [SKIP] {src.name} (已存在，大小相同)")
                        continue
                    else:
                        error_msg = f"文件大小不匹配: {src.name}"
                        errors.append(error_msg)
                        print(f"  [ERROR] {error_msg}")
                        continue

                if not self.dry_run:
                    try:
                        shutil.move(str(src), str(dst))
                        if log_file:
                            log_file.write(f"{src} -> {dst}\n")
                    except Exception as e:
                        error_msg = f"{src.name}: {e}"
                        errors.append(error_msg)
                        print(f"  [ERROR] {error_msg}")
                        continue

                moved_count += 1

                # 进度显示（每 100 个文件）
                if moved_count % 100 == 0:
                    print(f"  进度: {moved_count} 个文件已移动...")

            print(f"  Part01 完成: {moved_count} 个文件已移动")
            print()

        # 移动 Part02 文件
        if self.part2_cache_dir.exists():
            print(f"移动 Part02 文件: {self.part2_cache_dir}")
            part2_files = list(self.part2_cache_dir.glob("*.h5"))
            print(f"  找到 {len(part2_files)} 个文件")

            part2_moved = 0
            for src in part2_files:
                dst = self.orig_cache_dir / src.name

                if dst.exists():
                    # 检查文件大小
                    if src.stat().st_size == dst.stat().st_size:
                        skipped_count += 1
                        if self.dry_run:
                            print(f"  [SKIP] {src.name} (已存在，大小相同)")
                        continue
                    else:
                        error_msg = f"文件大小不匹配: {src.name}"
                        errors.append(error_msg)
                        print(f"  [ERROR] {error_msg}")
                        continue

                if not self.dry_run:
                    try:
                        shutil.move(str(src), str(dst))
                        if log_file:
                            log_file.write(f"{src} -> {dst}\n")
                    except Exception as e:
                        error_msg = f"{src.name}: {e}"
                        errors.append(error_msg)
                        print(f"  [ERROR] {error_msg}")
                        continue

                moved_count += 1
                part2_moved += 1

                # 进度显示（每 100 个文件）
                if part2_moved % 100 == 0:
                    print(f"  进度: {part2_moved} 个文件已移动...")

            print(f"  Part02 完成: {part2_moved} 个文件已移动")
            print()

        if log_file:
            log_file.close()
            print(f"[OK] 移动日志已保存到: {move_log}")
            print()

        # 总结
        print(f"文件移动完成:")
        print(f"  成功: {moved_count} 个")
        print(f"  跳过: {skipped_count} 个 (已存在)")
        print(f"  错误: {len(errors)} 个")

        if errors and len(errors) <= 5:
            print("\n错误列表:")
            for err in errors:
                print(f"  - {err}")
        elif errors:
            print(f"\n前 5 个错误:")
            for err in errors[:5]:
                print(f"  - {err}")
            print(f"  ... 以及其他 {len(errors) - 5} 个错误")

        print()
        return moved_count, skipped_count, errors

    # ==================== 步骤 5: 验证 ====================

    def verify(self) -> bool:
        """验证合并结果"""
        print("=" * 70)
        print("验证合并结果 (Verify Merge)")
        print("=" * 70)
        print()

        all_ok = True

        # 1. 检查索引
        print("[索引检查]")
        if not self.orig_index_path.exists():
            print(f"  [FAIL] 索引文件不存在: {self.orig_index_path}")
            return False

        index = self.load_index(self.orig_index_path)
        entries = index.get('entries', {})
        print(f"  [OK] 索引条目数: {len(entries)}")

        # 检查版本
        version = index.get('version')
        if version == self.CACHE_VERSION:
            print(f"  [OK] 索引版本: {version}")
        else:
            print(f"  [FAIL] 索引版本不匹配: {version} (期望 {self.CACHE_VERSION})")
            all_ok = False

        print()

        # 2. 检查文件存在性
        print("[文件存在性检查]")
        cache_files = self.scan_h5_files(self.orig_cache_dir)
        print(f"  缓存目录文件数: {len(cache_files)}")

        # 检查索引中的文件是否存在
        missing_files = []
        for key in entries:
            if key not in cache_files:
                missing_files.append(key)

        if not missing_files:
            print(f"  [OK] 所有索引条目都有对应文件")
        else:
            print(f"  [FAIL] {len(missing_files)} 个索引条目缺少文件")
            all_ok = False

            if len(missing_files) <= 5:
                for key in missing_files:
                    print(f"    - {key[:8]}")
            else:
                for key in missing_files[:5]:
                    print(f"    - {key[:8]}")
                print(f"    ... 以及其他 {len(missing_files) - 5} 个")

        # 额外文件（在目录中但不在索引中）
        extra_files = cache_files - set(entries.keys())
        if extra_files:
            print(f"  注意: {len(extra_files)} 个文件不在索引中（可能是旧缓存）")

        print()

        # 3. 检查路径格式
        print("[路径格式检查]")
        non_unix_paths = 0
        for entry in entries.values():
            source_file = entry.get('source_file', '')
            if '\\' in source_file:
                non_unix_paths += 1

        if non_unix_paths == 0:
            print(f"  [OK] 所有路径都是 Unix 格式")
        else:
            print(f"  [FAIL] {non_unix_paths} 个条目使用非 Unix 路径格式")
            all_ok = False

        print()

        # 4. 按被试统计
        print("[被试覆盖检查]")
        subjects = set()
        by_subject = {}
        for entry in entries.values():
            subject = entry.get('subject', 'unknown')
            subjects.add(subject)
            by_subject[subject] = by_subject.get(subject, 0) + 1

        print(f"  覆盖被试数: {len(subjects)}")
        print(f"  被试列表: {', '.join(sorted(subjects))}")
        print()

        # 显示前几个被试的统计
        print("  各被试缓存数（前 10）:")
        for i, (subject, count) in enumerate(sorted(by_subject.items())[:10]):
            print(f"    {subject}: {count} 个")

        print()

        # 5. 抽样加载测试
        print("[抽样加载测试]")
        import random
        import h5py

        sample_size = min(10, len(entries))
        sample_keys = random.sample(list(entries.keys()), sample_size)
        loaded = 0
        failed = 0

        for key in sample_keys:
            cache_file = self.orig_cache_dir / f"{key}.h5"
            try:
                with h5py.File(cache_file, 'r') as f:
                    # 检查数据集
                    if 'trials' in f and 'labels' in f:
                        loaded += 1
                    else:
                        print(f"  [FAIL] {key[:8]}: 缺少必需的数据集")
                        failed += 1
            except Exception as e:
                print(f"  [FAIL] {key[:8]}: 加载失败 - {e}")
                failed += 1

        print(f"  抽样测试: {loaded}/{sample_size} 成功")
        if failed > 0:
            print(f"  [FAIL] {failed} 个文件加载失败")
            all_ok = False
        else:
            print(f"  [OK] 所有抽样文件加载成功")

        print()

        # 总结
        if all_ok:
            print("[OK] 验证通过！合并成功完成。")
        else:
            print("[FAIL] 验证发现问题，请检查上述错误。")

        print()
        return all_ok

    # ==================== 完整流程 ====================

    def full_merge(self) -> bool:
        """执行完整的合并流程"""
        print("=" * 70)
        print("缓存索引合并 - 完整流程 (Full Merge Workflow)")
        print("=" * 70)
        print()

        if self.dry_run:
            print("[DRY RUN 模式] 所有操作仅模拟，不会实际修改文件")
            print()

        try:
            # 步骤 0: 预检查
            print("阶段 0: 预检查")
            print("-" * 70)
            if not self.validate():
                print("\n预检查失败，中止操作。")
                return False

            if not self.dry_run and not self.no_interactive:
                confirm = input("\n继续执行合并？[y/N]: ").strip().lower()
                if confirm != 'y':
                    print("操作已取消。")
                    return False
                print()

            # 步骤 1: 清理 Part02 索引
            print("\n阶段 1: 清理 Part02 索引")
            print("-" * 70)
            _, removed = self.clean_part2_index()

            if not self.dry_run and not self.no_interactive:
                confirm = input("\n继续合并索引？[y/N]: ").strip().lower()
                if confirm != 'y':
                    print("操作已取消。")
                    return False
                print()

            # 步骤 2: 合并索引
            print("\n阶段 2: 合并索引")
            print("-" * 70)
            _, overlap = self.merge_indices()

            if not self.dry_run and not self.no_interactive:
                confirm = input("\n继续移动文件？[y/N]: ").strip().lower()
                if confirm != 'y':
                    print("操作已取消。索引已合并，但文件尚未移动。")
                    return False
                print()

            # 步骤 3: 移动文件
            print("\n阶段 3: 移动文件")
            print("-" * 70)
            start_time = time.time()
            moved, skipped, errors = self.move_files()
            elapsed = time.time() - start_time

            print(f"文件移动耗时: {elapsed:.1f} 秒")
            print()

            if errors:
                print(f"警告: {len(errors)} 个文件移动失败")
                if not self.dry_run and not self.no_interactive:
                    confirm = input("\n继续验证？[y/N]: ").strip().lower()
                    if confirm != 'y':
                        print("操作已取消。")
                        return False
                print()

            # 步骤 4: 验证
            if not self.dry_run:
                print("\n阶段 4: 验证结果")
                print("-" * 70)
                verify_ok = self.verify()

                if verify_ok:
                    print("[SUCCESS] 合并流程完成！")
                    print()
                    print("建议后续步骤:")
                    print("1. 使用 cache_helper.py 查看合并后的缓存统计")
                    print("2. 测试加载不同被试的数据")
                    print("3. 确认无误后，可以删除 caches/dist/ 目录")
                    if self.backup_dir:
                        print(f"4. 备份保存在: {self.backup_dir}")
                else:
                    print("[WARNING]  验证发现问题，请检查后再使用。")
            else:
                print("\n[DRY RUN] 完整流程模拟完成")

            return True

        except KeyboardInterrupt:
            print("\n\n操作被用户中断。")
            return False
        except Exception as e:
            print(f"\n[FAIL] 错误: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(
        description="缓存索引合并工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        'command',
        choices=['validate', 'clean', 'merge', 'move', 'verify', 'full'],
        help=(
            "执行的命令:\n"
            "  validate - 预检查所有条件\n"
            "  clean    - 清理 Part02 索引（步骤 1）\n"
            "  merge    - 合并索引（步骤 2）\n"
            "  move     - 移动文件（步骤 3）\n"
            "  verify   - 验证结果（步骤 4）\n"
            "  full     - 执行完整流程（推荐）"
        )
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="模拟运行，不实际修改文件"
    )

    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help="非交互模式，自动执行所有步骤"
    )

    args = parser.parse_args()

    # 创建管理器
    manager = CacheMergeManager(dry_run=args.dry_run, no_interactive=args.no_interactive)

    # 执行命令
    try:
        if args.command == 'validate':
            success = manager.validate()
        elif args.command == 'clean':
            _, _ = manager.clean_part2_index()
            success = True
        elif args.command == 'merge':
            _, _ = manager.merge_indices()
            success = True
        elif args.command == 'move':
            _, _, errors = manager.move_files()
            success = len(errors) == 0
        elif args.command == 'verify':
            success = manager.verify()
        elif args.command == 'full':
            success = manager.full_merge()
        else:
            print(f"未知命令: {args.command}")
            success = False

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n操作被用户中断。")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
