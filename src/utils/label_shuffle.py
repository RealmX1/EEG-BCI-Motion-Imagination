"""
Within-subject trial-level label permutation for negative-control experiments.

Used by the cross-subject CBraMod label-shuffle robustness check (P0.3) to verify
that the headline cross-subject accuracy is driven by neural signal rather than
any hidden input-label shortcut. Shuffling within each subject preserves per-subject
class balance, so any above-chance accuracy after shuffling must come from leakage,
not from distributional shifts across folds.
"""
from collections import defaultdict
from typing import Optional

import numpy as np
import torch


def apply_within_subject_label_shuffle(dataset, seed: int, logger: Optional[object] = None):
    """
    在每个被试内部，按 trial 维度随机重排 label（in-place）。

    Args:
        dataset: 一个 ``FingerEEGDataset`` 实例，必须具备 ``trial_infos``、
            ``labels``、``label_to_idx``、``labels_tensor`` 属性。
        seed: ``np.random.RandomState`` 种子。
        logger: 可选的 ``SectionLogger`` / ``logging.Logger``，用于打印审计信息。

    Returns:
        修改过的 ``dataset`` 引用（in-place 改动 ``labels`` 与 ``labels_tensor``）。

    Notes:
        - 同一 ``(subject_id, trial_idx)`` 对应的所有 segment 共享同一新 label
          （trial-level 一致性，避免破坏 within-trial 时序结构）。
        - 全局 label 分布严格不变（multiset permutation）。
        - 仅依赖 ``dataset`` 的公开属性，不做 isinstance 检查，便于单元测试 mock。
    """
    n_segments = len(dataset.trial_infos)
    if n_segments == 0:
        if logger is not None:
            logger.info(f"[label-shuffle] seed={seed}, dataset empty — skipping")
        return dataset

    by_subject: "defaultdict[str, dict[int, object]]" = defaultdict(dict)
    for i in range(n_segments):
        info = dataset.trial_infos[i]
        by_subject[info.subject_id].setdefault(info.trial_idx, dataset.labels[i])

    rng = np.random.RandomState(seed)
    new_map: "dict[tuple[str, int], object]" = {}
    n_trials_total = 0
    n_changed = 0
    for subject in sorted(by_subject.keys()):
        trial2label = by_subject[subject]
        trials = list(trial2label.keys())
        labels = list(trial2label.values())
        permuted_idx = rng.permutation(len(labels))
        permuted_labels = [labels[j] for j in permuted_idx]
        for t, original, new in zip(trials, labels, permuted_labels):
            new_map[(subject, t)] = new
            n_trials_total += 1
            if new != original:
                n_changed += 1

    new_labels = [new_map[(info.subject_id, info.trial_idx)] for info in dataset.trial_infos]
    dataset.labels = new_labels
    dataset.labels_tensor = torch.tensor(
        [dataset.label_to_idx[l] for l in new_labels],
        dtype=torch.long,
    )

    if logger is not None:
        logger.info(
            f"[label-shuffle] seed={seed}, subjects={len(by_subject)}, "
            f"trials_with_changed_label={n_changed}/{n_trials_total} "
            f"({100.0 * n_changed / max(n_trials_total, 1):.1f}%), segments={n_segments}"
        )

    return dataset
