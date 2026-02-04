"""
验证 trial-level split 无数据泄露

运行此脚本验证数据分割的正确性:
- 确保 trial indices 全局唯一
- 验证 train/val sets 无 trial 重叠
- 检查 segments 正确归属到各自的 trials

Usage:
    uv run python scripts/verify_trial_split.py --subject S01
"""

import argparse
import sys
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

# Fix Windows console encoding
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.data_loader import FingerEEGDataset, PreprocessConfig


def verify_trial_split(subject: str = 'S01'):
    """验证 trial-level split 的正确性"""

    print("=" * 70)
    print(f"验证 Trial-level Split - Subject {subject}")
    print("=" * 70)
    print()

    # Load dataset
    data_root = PROJECT_ROOT / 'data'
    elc_path = data_root / 'biosemi128.ELC'
    config = PreprocessConfig.paper_aligned(n_class=2)

    print("加载数据集...")
    dataset = FingerEEGDataset(
        str(data_root),
        [subject],
        config,
        task_types=['OfflineImagery'],
        target_classes=[1, 4],  # Thumb vs Pinky
        elc_path=str(elc_path)
    )

    print(f"[OK] 数据集加载完成")
    print()

    # Dataset statistics
    print("=" * 70)
    print("数据集统计")
    print("=" * 70)
    print(f"Total segments: {len(dataset)}")

    unique_trials = dataset.get_unique_trials()
    print(f"Total unique trials: {len(unique_trials)}")
    print(f"Trial IDs range: {min(unique_trials)} to {max(unique_trials)}")
    print()

    # Verify trial index uniqueness
    all_trial_ids = [info.trial_idx for info in dataset.trial_infos]
    if len(set(all_trial_ids)) == len(unique_trials):
        print(f"[OK] 所有 trial indices 在全局范围内唯一")
    else:
        print(f"[ERROR] Trial indices 有重复！")
        return False

    # Check segments per trial
    from collections import Counter
    segments_per_trial = Counter(all_trial_ids)
    avg_segments = sum(segments_per_trial.values()) / len(segments_per_trial)
    print(f"[OK] 平均每个 trial 有 {avg_segments:.1f} 个 segments")
    print()

    # Trial-level split
    print("=" * 70)
    print("Trial-level Split (80/20)")
    print("=" * 70)

    trial_labels = []
    for trial_idx in unique_trials:
        for i, info in enumerate(dataset.trial_infos):
            if info.trial_idx == trial_idx:
                trial_labels.append(dataset.labels[i])
                break

    train_trials, val_trials = train_test_split(
        unique_trials, test_size=0.2, stratify=trial_labels, random_state=42
    )

    train_indices = dataset.get_segment_indices_for_trials(train_trials)
    val_indices = dataset.get_segment_indices_for_trials(val_trials)

    print(f"Train trials: {len(train_trials)}")
    print(f"Val trials: {len(val_trials)}")
    print(f"Train segments: {len(train_indices)}")
    print(f"Val segments: {len(val_indices)}")
    print()

    # Verify no overlap in trials
    print("=" * 70)
    print("数据泄露检测")
    print("=" * 70)

    train_trial_set = set(train_trials)
    val_trial_set = set(val_trials)

    overlap = train_trial_set & val_trial_set
    if overlap:
        print(f"[ERROR] 发现 {len(overlap)} 个重叠的 trials!")
        print(f"   重叠的 trial IDs: {sorted(list(overlap)[:10])}...")
        return False
    else:
        print(f"[OK] 无 trial 重叠 (train vs val)")

    # Verify segment assignment
    train_trial_ids_from_segments = set(
        dataset.trial_infos[i].trial_idx for i in train_indices
    )
    val_trial_ids_from_segments = set(
        dataset.trial_infos[i].trial_idx for i in val_indices
    )

    if train_trial_ids_from_segments == train_trial_set:
        print(f"[OK] 所有 train segments 归属于 train trials")
    else:
        print(f"[ERROR] Train segments 归属错误!")
        return False

    if val_trial_ids_from_segments == val_trial_set:
        print(f"[OK] 所有 val segments 归属于 val trials")
    else:
        print(f"[ERROR] Val segments 归属错误!")
        return False

    # Check for cross-contamination
    cross_contamination = train_trial_ids_from_segments & val_trial_ids_from_segments
    if cross_contamination:
        print(f"[ERROR] 发现 {len(cross_contamination)} 个 trials 的 segments 同时出现在 train 和 val!")
        return False
    else:
        print(f"[OK] 无交叉污染 (segments 正确分配)")

    print()

    # Label distribution
    print("=" * 70)
    print("标签分布")
    print("=" * 70)

    train_labels = [dataset.labels[i] for i in train_indices]
    val_labels = [dataset.labels[i] for i in val_indices]

    train_label_counts = Counter(train_labels)
    val_label_counts = Counter(val_labels)

    print(f"Train set:")
    for label, count in sorted(train_label_counts.items()):
        print(f"  Label {label}: {count} segments ({count/len(train_labels)*100:.1f}%)")

    print(f"Val set:")
    for label, count in sorted(val_label_counts.items()):
        print(f"  Label {label}: {count} segments ({count/len(val_labels)*100:.1f}%)")

    print()
    print("=" * 70)
    print("[SUCCESS] 验证通过！数据分割正确，无数据泄露。")
    print("=" * 70)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="验证 trial-level split 无数据泄露"
    )
    parser.add_argument(
        '--subject', type=str, default='S01',
        help='Subject ID (e.g., S01)'
    )

    args = parser.parse_args()

    success = verify_trial_split(args.subject)

    if not success:
        print()
        print("验证失败！请检查数据加载和分割逻辑。")
        sys.exit(1)


if __name__ == '__main__':
    main()
