"""
分析 batch size 与训练集大小的比例。

研究问题:
1. 每个被试的训练集有多少 segments?
2. batch_size / training_set_size 的比例是多少?
3. 这个比例与原论文的设置相比如何?
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.preprocessing.data_loader import (
    FingerEEGDataset,
    PreprocessConfig,
    get_session_folders_for_split,
    discover_available_subjects,
)


def analyze_subject_data_size(
    data_root: Path,
    subject_id: str,
    task: str = 'binary',
    paradigm: str = 'imagery'
):
    """分析单个被试的数据集大小。"""

    elc_path = data_root / 'biosemi128.ELC'

    # 定义 target classes
    if task == 'binary':
        target_classes = [1, 2]  # Thumb vs Index
    elif task == 'ternary':
        target_classes = [1, 2, 3]  # Thumb, Index, Middle
    else:
        target_classes = [1, 2, 3, 4]  # All fingers

    # 获取训练/测试 session folders
    train_folders = get_session_folders_for_split(paradigm, task, 'train')
    test_folders = get_session_folders_for_split(paradigm, task, 'test')

    results = {}

    for model_name in ['eegnet', 'cbramod']:
        # 获取配置
        if model_name == 'eegnet':
            config = PreprocessConfig.paper_aligned()
        else:
            config = PreprocessConfig.for_cbramod_128ch()

        # 加载训练数据集
        train_dataset = FingerEEGDataset(
            str(data_root),
            [subject_id],
            config,
            session_folders=train_folders,
            target_classes=target_classes,
            elc_path=str(elc_path),
        )

        # 加载测试数据集
        test_dataset = FingerEEGDataset(
            str(data_root),
            [subject_id],
            config,
            session_folders=test_folders,
            target_classes=target_classes,
            elc_path=str(elc_path),
        )

        # 获取 unique trials
        train_trials = set(info.trial_idx for info in train_dataset.trial_infos)
        test_trials = set(info.trial_idx for info in test_dataset.trial_infos)

        results[model_name] = {
            'n_segments_train': len(train_dataset),
            'n_segments_test': len(test_dataset),
            'n_trials_train': len(train_trials),
            'n_trials_test': len(test_trials),
            'segments_per_trial': len(train_dataset) / len(train_trials) if train_trials else 0,
            'target_fs': config.target_fs,
            'segment_step_samples': config.segment_step_samples,
        }

    return results


def print_batch_ratio_analysis(data_root: Path, subjects: list):
    """打印 batch size 比例分析。"""

    print("=" * 80)
    print("Batch Size vs Training Set Size 分析")
    print("=" * 80)

    # 当前配置
    eegnet_batch_size = 64
    cbramod_batch_size = 128

    print("\n## 当前配置")
    print(f"- EEGNet batch_size: {eegnet_batch_size}")
    print(f"- CBraMod batch_size: {cbramod_batch_size}")

    print("\n## Per-Subject 数据量分析 (Binary task)")
    print("-" * 100)
    print(f"{'Subject':<8} {'Model':<10} {'Train Segs':<12} {'Test Segs':<10} {'Segs/Trial':<12} {'Batch Ratio':<12} {'Batches/Epoch':<15}")
    print("-" * 100)

    all_results = {}
    eegnet_train_sizes = []
    cbramod_train_sizes = []

    for subject in subjects[:7]:  # 分析前 7 个被试
        try:
            results = analyze_subject_data_size(data_root, subject, task='binary')
            all_results[subject] = results

            for model_name, data in results.items():
                batch_size = eegnet_batch_size if model_name == 'eegnet' else cbramod_batch_size
                n_train = data['n_segments_train']
                n_test = data['n_segments_test']
                batch_ratio = batch_size / n_train if n_train > 0 else 0
                batches_per_epoch = n_train / batch_size if batch_size > 0 else 0

                if model_name == 'eegnet':
                    eegnet_train_sizes.append(n_train)
                else:
                    cbramod_train_sizes.append(n_train)

                print(f"{subject:<8} {model_name:<10} {n_train:<12} {n_test:<10} {data['segments_per_trial']:<12.1f} "
                      f"{batch_ratio:<12.4f} {batches_per_epoch:<15.1f}")
        except Exception as e:
            print(f"{subject:<8} Error: {e}")
            import traceback
            traceback.print_exc()

    # 计算平均值
    print("-" * 100)

    if eegnet_train_sizes:
        avg_eegnet = np.mean(eegnet_train_sizes)
        avg_eegnet_ratio = eegnet_batch_size / avg_eegnet
        avg_eegnet_batches = avg_eegnet / eegnet_batch_size
        print(f"{'AVERAGE':<8} {'eegnet':<10} {avg_eegnet:<12.0f} {'-':<10} {'-':<12} {avg_eegnet_ratio:<12.4f} {avg_eegnet_batches:<15.1f}")

    if cbramod_train_sizes:
        avg_cbramod = np.mean(cbramod_train_sizes)
        avg_cbramod_ratio = cbramod_batch_size / avg_cbramod
        avg_cbramod_batches = avg_cbramod / cbramod_batch_size
        print(f"{'AVERAGE':<8} {'cbramod':<10} {avg_cbramod:<12.0f} {'-':<10} {'-':<12} {avg_cbramod_ratio:<12.4f} {avg_cbramod_batches:<15.1f}")

    # 原论文配置对比
    print("\n" + "=" * 80)
    print("## 原论文训练配置对比")
    print("=" * 80)

    print("""
### EEGNet 原论文 (Lawhern et al., 2018)
- 数据集: BCI Competition IV-2a (9 subjects, 288 trials/subject × 2 sessions = 576 trials)
- 采样率: 250 Hz
- Trial 长度: 4s (motor imagery onset 到结束)
- Sliding window: 未使用 (原论文直接用完整 trial)
- 训练样本数: ~288 trials (单 session) 或 ~576 trials (双 session)
- Batch size: 16 (论文默认)
- Epochs: 500
- **Batch ratio: 16 / 288 ≈ 0.056 (单 session)**
- **Batches per epoch: 288 / 16 = 18**

### 本项目 EEGNet 设置分析:""")

    if eegnet_train_sizes:
        print(f"- 训练样本数: ~{avg_eegnet:.0f} segments (因为使用了 sliding window)")
        print(f"- Batch size: {eegnet_batch_size}")
        print(f"- **Batch ratio: {avg_eegnet_ratio:.4f}**")
        print(f"- **Batches per epoch: {avg_eegnet_batches:.1f}**")
        print(f"- 对比: 本项目 batch ratio 比原论文{'小' if avg_eegnet_ratio < 0.056 else '大'} ({avg_eegnet_ratio/0.056:.1f}x)")

    print("""
### CBraMod 原论文 (ICLR 2025)
- **预训练 (TUEG)**:
  - 数据量: 14,000+ recordings, 每个数据集约几千到几万样本
  - Batch size: 128
  - Epochs: 40
  - 典型 batch ratio: 很小 (大规模数据)

- **下游任务微调**:
  - 数据量差异很大 (从几百到几万不等)
  - 典型 batch sizes: 32-128
  - Epochs: 30-50
  - 论文中使用的数据集:
    - SEED: 3394 samples, batch=32, ratio≈0.009
    - HGD: 880 samples/subject, batch=32, ratio≈0.036
    - BCI-IV-2a: 288 samples, batch=32, ratio≈0.111
  - **注意**: CBraMod 论文对小数据集也使用较小的 batch_size (32)

### 本项目 CBraMod 设置分析:""")

    if cbramod_train_sizes:
        print(f"- 训练样本数: ~{avg_cbramod:.0f} segments")
        print(f"- Batch size: {cbramod_batch_size}")
        print(f"- **Batch ratio: {avg_cbramod_ratio:.4f}**")
        print(f"- **Batches per epoch: {avg_cbramod_batches:.1f}**")

    print("""
### 结论与建议

1. **EEGNet**: 当前 batch ratio 较小，batches per epoch 较多
   - 这对于小模型是合理的，每个 epoch 有足够的梯度更新
   - 可以考虑增大 batch_size 到 128-256，加快训练同时可能略微提高泛化

2. **CBraMod**: 当前 batch_size=128 与数据量的比例
   - 对比 CBraMod 论文对小数据集 (BCI-IV-2a) 使用 batch=32
   - **建议测试 batch_size=32 或 64，可能更适合被试内训练**
   - 更小的 batch 提供更多的梯度更新，可能有助于微调预训练模型

3. **关键差异**: 原论文通常不使用 sliding window，我们使用了
   - Sliding window 显著增加了训练样本数 (~32x per offline trial)
   - 但这些样本高度相关 (同一 trial 的重叠 segments)
   - 这可能导致 effective batch size 比表面数字更大
""")


def print_segment_calculation():
    """打印 segment 计算细节。"""

    print("\n" + "=" * 80)
    print("## Segment 生成计算")
    print("=" * 80)

    # EEGNet
    print("\n### EEGNet (100 Hz)")
    print("- segment_length: 1.0s = 100 samples")
    print("- segment_step: 128 samples @ 1024 Hz → ceil(128 * 100 / 1024) = 13 samples @ 100 Hz")
    print("- 5s offline trial (500 samples):")
    print("  - n_segments = (500 - 100) / 13 + 1 = 31.8 → ~32 segments")
    print("- 3s online trial (300 samples):")
    print("  - n_segments = (300 - 100) / 13 + 1 = 16.4 → ~16 segments")

    # CBraMod
    print("\n### CBraMod (200 Hz)")
    print("- segment_length: 1.0s = 200 samples")
    print("- segment_step: 128 samples @ 1024 Hz → ceil(128 * 200 / 1024) = 25 samples @ 200 Hz")
    print("- 5s offline trial (1000 samples):")
    print("  - n_segments = (1000 - 200) / 25 + 1 = 33 segments")
    print("- 3s online trial (600 samples):")
    print("  - n_segments = (600 - 200) / 25 + 1 = 17 segments")


if __name__ == '__main__':
    print_segment_calculation()
    print("\n")

    # 检查是否可以访问数据
    data_root = PROJECT_ROOT / 'data'
    try:
        subjects = discover_available_subjects(str(data_root))
        if subjects:
            print(f"\n发现 {len(subjects)} 个被试: {subjects[:5]}...")
            print_batch_ratio_analysis(data_root, subjects)
        else:
            print("未发现被试数据目录")
    except Exception as e:
        print(f"无法访问数据: {e}")
        import traceback
        traceback.print_exc()
        print("\n仅打印理论计算...")
