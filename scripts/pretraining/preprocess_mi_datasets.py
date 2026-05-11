#!/usr/bin/env python3
"""
MI 数据集统一预处理脚本 → LMDB
================================
将 MOABB Motor Imagery 数据集预处理为 CBraMod 预训练格式。

预处理管线（与 TUEG 对齐）：
  1. 加载原始连续 EEG（全部录制，非仅 trial）
  2. 选择 EEG 通道（排除 EOG/EMG/参考）
  3. 重采样到 200 Hz
  4. 带通滤波 0.3–75 Hz
  5. 陷波滤波（50 Hz 或 60 Hz）
  6. 切分为 30 秒段 → (ch_num, 30, 200)
  7. 质量过滤：丢弃首尾 5s + max(abs) ≥ 500 µV 的段

输出：每个数据集一个 LMDB，保持各自原生通道数。
"""

import os
import sys
import pickle
import traceback
import argparse
from pathlib import Path
from datetime import datetime

import json

import numpy as np
import lmdb
import mne
from scipy.stats import kurtosis as scipy_kurtosis

mne.set_log_level("ERROR")

# 设置 MNE 数据目录
MI_DATA_DIR = Path(r"D:\data\motion_imagination_datasets")
os.environ.setdefault("MNE_DATA", str(MI_DATA_DIR))

# 默认 LMDB 输出目录
DEFAULT_LMDB_DIR = Path(r"D:\data\motion_imagination_datasets\lmdb_pretrain")

# 数据集元数据（单位信息等），由论文/源码调研确定
METADATA_PATH = Path(__file__).parent / "dataset_metadata.json"

def _load_dataset_metadata() -> dict:
    """加载数据集元数据 JSON。"""
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("datasets", {})
    return {}

DATASET_METADATA = _load_dataset_metadata()

# CBraMod 预训练参数
TARGET_SFREQ = 200      # Hz
PATCH_DURATION = 30      # 秒
PATCH_SIZE = 200         # 每 patch 采样点 (1s × 200Hz)
N_PATCHES = 30           # 每段 patch 数 (30s)
SKIP_SECONDS = 5         # 首尾跳过秒数（MOABB 返回已切割的 run，无需长跳过）
AMP_THRESHOLD = 500      # µV，幅值阈值（宽松阈值，仅排除明显伪影）

# Strict filter（V4/V5 DAPT 用）：通用更严格的物理伪迹过滤
# 设计动机：替代 dataset-specific trial-level annotations（后者需要 trial 对齐重写整条
# pipeline）。300µV per-segment-max 比 500µV 更紧，能切掉残留眼动/肌电；per-channel
# kurtosis>10 切掉单通道脉冲性伪迹（断电极、接触不良）。
STRICT_AMP_THRESHOLD = 300   # µV
STRICT_KURT_THRESHOLD = 10   # per-channel kurtosis


# ─────────────────────────────────────────────
# 数据集特殊处理配置
# ─────────────────────────────────────────────

DATASET_PREPROCESS_CONFIG = {
    "Lee2019_MI": {
        "notch_freq": 50,
        "pick_types": {"eeg": True, "eog": False, "emg": False, "stim": False},
        "channel_rename": {
            "T7": "T3", "T8": "T4", "TP9": "T5", "TP10": "T6",
        },
    },
    "Stieger2021": {
        "notch_freq": 60,
        "pick_types": {"eeg": True, "eog": False, "stim": False},
    },
    "PhysionetMI": {
        "notch_freq": 60,
        "pick_types": {"eeg": True, "stim": False},
        "fix_channel_names": True,  # 移除通道名尾部 "."
    },
    "Cho2017": {
        "notch_freq": 60,
        "pick_types": {"eeg": True, "eog": False, "stim": False},
    },
    "Schirrmeister2017": {
        "notch_freq": 50,
        "pick_types": {"eeg": True, "eog": False, "emg": False, "stim": False},
    },
    "GrosseWentrup2009": {
        "notch_freq": 60,
        "pick_types": {"eeg": True, "eog": False, "stim": False},
    },
    "Ofner2017": {
        "notch_freq": 50,
        "pick_types": {"eeg": True, "eog": False, "stim": False},
    },
    "BNCI2015_004": {
        "notch_freq": 50,
        "pick_types": {"eeg": True, "eog": False, "stim": False},
    },
    "Shin2017A": {
        "notch_freq": 50,
        "pick_types": {"eeg": True, "fnirs": False, "stim": False},
    },
    "BNCI2014_001": {
        "notch_freq": 50,
        "pick_types": {"eeg": True, "eog": False, "stim": False},
    },
    # 以下数据集已知不可用，保留配置供参考但默认排除
    # Weibo2014: zip 解压错误
    # Dreyer2023: OSF 404
}

# 默认排除的数据集（下载/解压已知失败）
EXCLUDED_DATASETS = {"Weibo2014", "Dreyer2023"}


def fix_physionet_channel_names(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """PhysioNet 通道名尾部有 '.'，需要移除。"""
    rename_map = {}
    for ch in raw.info["ch_names"]:
        if ch.endswith("."):
            rename_map[ch] = ch.rstrip(".")
    if rename_map:
        raw.rename_channels(rename_map)
    return raw


def preprocess_raw(raw: mne.io.BaseRaw, config: dict) -> mne.io.BaseRaw | None:
    """
    对单个 Raw 对象执行统一预处理。

    Returns:
        预处理后的 Raw，如果数据不可用则返回 None。
    """
    try:
        # 1. 特殊通道名处理
        if config.get("fix_channel_names"):
            raw = fix_physionet_channel_names(raw)

        if config.get("channel_rename"):
            existing = raw.info["ch_names"]
            rename_map = {
                old: new
                for old, new in config["channel_rename"].items()
                if old in existing
            }
            if rename_map:
                raw.rename_channels(rename_map)

        # 2. 选择 EEG 通道
        pick_types = config.get("pick_types", {"eeg": True})
        try:
            raw.pick(picks="eeg")
        except Exception:
            # 回退到 pick_types
            raw.pick_types(**pick_types)

        n_eeg = len(raw.info["ch_names"])
        if n_eeg < 16:
            return None  # 通道太少

        # 3. 加载数据到内存
        if not raw.preload:
            raw.load_data()

        # 4. 重采样到 200 Hz
        if raw.info["sfreq"] != TARGET_SFREQ:
            raw.resample(TARGET_SFREQ)

        # 5. 带通滤波 0.3–75 Hz
        raw.filter(l_freq=0.3, h_freq=75.0)

        # 6. 陷波滤波
        notch = config.get("notch_freq", 50)
        raw.notch_filter(freqs=notch)

        return raw

    except Exception as e:
        print(f"    预处理失败: {e}")
        return None


def segment_raw_to_patches(
    raw: mne.io.BaseRaw,
    dataset_name: str = "",
) -> np.ndarray | None:
    """
    将预处理后的 Raw 切分为 30 秒段。

    Returns:
        shape (n_segments, n_channels, 30, 200) 或 None
    """
    data = raw.get_data()

    # 使用元数据中的 to_uV_factor（优先），否则回退到启发式检测
    meta = DATASET_METADATA.get(dataset_name, {})
    to_uv = meta.get("to_uV_factor")
    if to_uv is not None:
        data = data * to_uv
    else:
        # 启发式回退：根据 mean_abs 推断单位
        mean_abs = np.abs(data).mean()
        if mean_abs < 1e-2:
            data = data * 1e6  # V → µV
        elif mean_abs < 10:
            data = data * 1e3  # mV → µV
    n_channels, n_times = data.shape

    # 跳过首尾（SKIP_SECONDS 秒）
    skip_samples = SKIP_SECONDS * TARGET_SFREQ
    if n_times <= 2 * skip_samples + PATCH_DURATION * TARGET_SFREQ:
        return None  # 太短

    data = data[:, skip_samples:-skip_samples]
    n_times = data.shape[1]

    # 每段 30 秒 = 6000 样本
    segment_samples = PATCH_DURATION * TARGET_SFREQ
    n_segments = n_times // segment_samples
    if n_segments == 0:
        return None

    # 截断到整数段
    data = data[:, : n_segments * segment_samples]

    # reshape: (n_channels, n_segments * 30 * 200) → (n_segments, n_channels, 30, 200)
    data = data.reshape(n_channels, n_segments, N_PATCHES, PATCH_SIZE)
    data = data.transpose(1, 0, 2, 3)  # (n_segments, n_channels, 30, 200)

    return data


def filter_segments(segments: np.ndarray) -> np.ndarray:
    """质量过滤：丢弃 max(abs) >= AMP_THRESHOLD uV 的段。"""
    mask = np.max(np.abs(segments), axis=(1, 2, 3)) < AMP_THRESHOLD
    return segments[mask]


def filter_segments_strict(
    segments: np.ndarray,
    amp_thresh: float = STRICT_AMP_THRESHOLD,
    kurt_thresh: float = STRICT_KURT_THRESHOLD,
) -> tuple[np.ndarray, dict]:
    """更严格的质量过滤：amplitude + per-channel kurtosis。

    Returns:
        (filtered_segments, stats_dict) — stats 含各步骤丢弃数量
    """
    n0 = len(segments)
    if n0 == 0:
        return segments, {"n_in": 0, "dropped_amp": 0, "dropped_kurt": 0, "n_out": 0}

    # 1. Amplitude filter（更严阈值）
    amp_mask = np.max(np.abs(segments), axis=(1, 2, 3)) < amp_thresh
    n_after_amp = int(amp_mask.sum())

    # 2. Per-channel kurtosis filter
    # Reshape (n_seg, n_ch, 30, 200) → (n_seg, n_ch, 6000) for per-channel kurtosis
    flat = segments.reshape(segments.shape[0], segments.shape[1], -1)
    # scipy.stats.kurtosis 默认 Fisher（normal=0），nan_policy=propagate
    kurt = scipy_kurtosis(flat, axis=2, fisher=True, nan_policy="omit")  # (n_seg, n_ch)
    # 取每个 segment 中"最坏"的通道 kurtosis；NaN（恒定信号）按超阈处理
    max_kurt = np.nanmax(kurt, axis=1)
    kurt_mask = (max_kurt < kurt_thresh) & np.isfinite(max_kurt)

    combined = amp_mask & kurt_mask
    out = segments[combined]
    return out, {
        "n_in": n0,
        "dropped_amp": int(n0 - n_after_amp),
        "dropped_kurt": int(n_after_amp - len(out)),
        "n_out": int(len(out)),
    }


def process_single_dataset(
    dataset_name: str,
    lmdb_dir: Path,
    subjects: list | None = None,
    max_subjects: int | None = None,
    dry_run: bool = False,
    artifact_filter: str = "basic",
) -> dict:
    """
    处理单个 MOABB 数据集并写入 LMDB。

    Returns:
        统计信息字典
    """
    import moabb.datasets as moabb_ds

    config = DATASET_PREPROCESS_CONFIG.get(dataset_name)
    if config is None:
        return {"status": "skipped", "reason": f"无预处理配置: {dataset_name}"}

    stats = {
        "dataset": dataset_name,
        "status": "unknown",
        "artifact_filter": artifact_filter,
        "n_subjects_processed": 0,
        "n_subjects_failed": 0,
        "n_segments_total": 0,
        "n_segments_filtered": 0,
        "n_dropped_amp": 0,    # strict 模式下：被 amplitude 过滤掉的段数
        "n_dropped_kurt": 0,   # strict 模式下：被 kurtosis 过滤掉的段数
        "n_channels": 0,
        "total_duration_hours": 0,
        "lmdb_path": None,
    }

    try:
        ds_class = getattr(moabb_ds, dataset_name)
        dataset = ds_class()

        # 确定被试列表
        all_subjects = dataset.subject_list
        if subjects:
            all_subjects = [s for s in all_subjects if s in subjects]
        if max_subjects:
            all_subjects = all_subjects[:max_subjects]

        print(f"\n{'='*60}")
        print(f"[{dataset_name}] 开始处理 {len(all_subjects)} 个被试")
        print(f"{'='*60}")

        if dry_run:
            # 只处理第一个被试的第一个 run 做预检
            first_data = dataset.get_data(subjects=[all_subjects[0]])
            for subj_id, sessions in first_data.items():
                for sess_name, runs in sessions.items():
                    for run_name, raw in runs.items():
                        processed = preprocess_raw(raw.copy(), config)
                        if processed:
                            stats["n_channels"] = len(processed.info["ch_names"])
                            stats["status"] = "dry_run_ok"
                            print(
                                f"  预检通过: {stats['n_channels']} 通道, "
                                f"{processed.info['sfreq']} Hz"
                            )
                        else:
                            stats["status"] = "dry_run_failed"
                            print("  预检失败")
                        return stats

        # 准备 LMDB
        lmdb_suffix = "_pretrain_filtered" if artifact_filter == "strict" else "_pretrain"
        lmdb_path = lmdb_dir / f"{dataset_name}{lmdb_suffix}"
        lmdb_path.mkdir(parents=True, exist_ok=True)
        stats["lmdb_path"] = str(lmdb_path)

        # 动态估算 map_size：每段 ~700KB (128ch×30×200×4bytes)，×4 余量，上限 10GB
        n_est_segments = len(all_subjects) * 50  # 估计每被试 50 段
        est_bytes = n_est_segments * 700 * 1024  # ~700KB/段
        map_size = min(max(est_bytes * 4, 500 * 1024 * 1024), 10 * 1024 * 1024 * 1024)
        db = lmdb.open(str(lmdb_path), map_size=map_size)

        all_keys = []
        total_segments_before = 0
        total_segments_after = 0

        for subj_idx, subject in enumerate(all_subjects):
            try:
                print(
                    f"  [{subj_idx+1}/{len(all_subjects)}] 被试 {subject}...",
                    end="",
                    flush=True,
                )

                data = dataset.get_data(subjects=[subject])
                subj_segments = 0

                for subj_id, sessions in data.items():
                    for sess_name, runs in sessions.items():
                        for run_name, raw in runs.items():
                            # 预处理
                            processed = preprocess_raw(raw.copy(), config)
                            if processed is None:
                                continue

                            if stats["n_channels"] == 0:
                                stats["n_channels"] = len(
                                    processed.info["ch_names"]
                                )

                            # 切段
                            segments = segment_raw_to_patches(processed, dataset_name=dataset_name)
                            if segments is None:
                                continue

                            n_before = len(segments)
                            total_segments_before += n_before

                            # 质量过滤
                            if artifact_filter == "strict":
                                segments, fstats = filter_segments_strict(segments)
                                stats["n_dropped_amp"] += fstats["dropped_amp"]
                                stats["n_dropped_kurt"] += fstats["dropped_kurt"]
                            else:
                                segments = filter_segments(segments)
                            n_after = len(segments)
                            total_segments_after += n_after

                            # 写入 LMDB（带 MapFull 自动扩容）
                            seg_keys = []
                            for i, seg in enumerate(segments):
                                seg_keys.append(
                                    (f"{dataset_name}_s{subject}_{sess_name}_{run_name}_{i}",
                                     pickle.dumps(seg.astype(np.float32)))
                                )

                            written = False
                            for _attempt in range(5):
                                try:
                                    txn = db.begin(write=True)
                                    for key, val in seg_keys:
                                        txn.put(key.encode(), val)
                                    txn.commit()
                                    written = True
                                    break
                                except lmdb.MapFullError:
                                    txn.abort()
                                    db.close()
                                    map_size = int(map_size * 2)
                                    print(f" [MapFull, 扩大到 {map_size//1024//1024}MB]", end="")
                                    db = lmdb.open(str(lmdb_path), map_size=map_size)

                            if not written:
                                raise RuntimeError(f"LMDB 写入失败: 5 次扩容后仍 MapFull")

                            for key, _ in seg_keys:
                                all_keys.append(key)

                            subj_segments += n_after

                            # 释放内存
                            del processed, segments

                stats["n_subjects_processed"] += 1
                print(f" {subj_segments} 段")

            except Exception as e:
                stats["n_subjects_failed"] += 1
                print(f" 失败: {e}")
                traceback.print_exc()

        # 保存 keys 索引
        try:
            txn = db.begin(write=True)
            txn.put("__keys__".encode(), pickle.dumps(all_keys))
            txn.commit()
        except lmdb.MapFullError:
            txn.abort()
            db.close()
            map_size = int(map_size * 2)
            db = lmdb.open(str(lmdb_path), map_size=map_size)
            txn = db.begin(write=True)
            txn.put("__keys__".encode(), pickle.dumps(all_keys))
            txn.commit()
        db.close()

        stats["n_segments_total"] = total_segments_before
        stats["n_segments_filtered"] = total_segments_after
        stats["total_duration_hours"] = round(
            total_segments_after * PATCH_DURATION / 3600, 2
        )
        stats["status"] = "ok"

        print(f"\n  完成: {total_segments_after}/{total_segments_before} 段通过过滤")
        print(f"  时长: ~{stats['total_duration_hours']:.1f} 小时")
        print(f"  LMDB: {lmdb_path}")

    except Exception as e:
        stats["status"] = "error"
        stats["error"] = str(e)
        traceback.print_exc()

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="MI 数据集统一预处理 → LMDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="指定数据集名称（默认全部可用数据集）",
    )
    parser.add_argument(
        "--lmdb-dir",
        type=str,
        default=str(DEFAULT_LMDB_DIR),
        help=f"LMDB 输出目录（默认 {DEFAULT_LMDB_DIR}）",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="每个数据集最多处理的被试数（用于调试）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只处理每个数据集的第一个被试做预检",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="统计报告 JSON 输出路径",
    )
    parser.add_argument(
        "--artifact-filter",
        choices=["basic", "strict"],
        default="basic",
        help=(
            "过滤强度。basic = 现有 500 µV per-segment-max（默认，与 V1/V2/V3 一致）；"
            "strict = 300 µV + per-channel kurtosis>10（V4/V5 DAPT 用，输出到 _filtered 后缀目录）"
        ),
    )
    args = parser.parse_args()

    lmdb_dir = Path(args.lmdb_dir)
    lmdb_dir.mkdir(parents=True, exist_ok=True)

    # 选择数据集
    if args.datasets:
        dataset_names = args.datasets
    else:
        dataset_names = [k for k in DATASET_PREPROCESS_CONFIG.keys()
                         if k not in EXCLUDED_DATASETS]

    print(f"预处理管线配置:")
    print(f"  目标采样率: {TARGET_SFREQ} Hz")
    print(f"  段长: {PATCH_DURATION} 秒 ({N_PATCHES} patches × {PATCH_SIZE} 采样点)")
    if args.artifact_filter == "strict":
        print(f"  Filter: STRICT - {STRICT_AMP_THRESHOLD} uV amplitude + kurtosis<{STRICT_KURT_THRESHOLD}")
        print(f"  输出后缀: _pretrain_filtered")
    else:
        print(f"  Filter: BASIC - {AMP_THRESHOLD} uV per-segment-max only")
        print(f"  输出后缀: _pretrain")
    print(f"  首尾跳过: {SKIP_SECONDS} 秒")
    print(f"  LMDB 输出: {lmdb_dir}")
    print(f"  数据集: {dataset_names}")
    if args.dry_run:
        print(f"  模式: DRY RUN（仅预检）")
    print()

    all_stats = []
    for name in dataset_names:
        stats = process_single_dataset(
            name,
            lmdb_dir,
            max_subjects=args.max_subjects,
            dry_run=args.dry_run,
            artifact_filter=args.artifact_filter,
        )
        all_stats.append(stats)

    # 打印汇总
    print("\n" + "=" * 80)
    print("预处理汇总")
    print("=" * 80)
    print(
        f"{'数据集':<22} {'状态':<8} {'通道':<6} {'被试':<6} {'段数':<10} {'过滤后':<10} {'时长(h)':<10}"
    )
    print("-" * 80)

    total_segments = 0
    total_hours = 0
    for s in all_stats:
        status = s["status"]
        ch = s.get("n_channels", "?")
        n_sub = s.get("n_subjects_processed", 0)
        n_seg_total = s.get("n_segments_total", 0)
        n_seg_filtered = s.get("n_segments_filtered", 0)
        hours = s.get("total_duration_hours", 0)
        print(
            f"{s['dataset']:<22} {status:<8} {str(ch):<6} {n_sub:<6} "
            f"{n_seg_total:<10} {n_seg_filtered:<10} {hours:<10}"
        )
        total_segments += n_seg_filtered
        total_hours += hours

    print("-" * 80)
    print(
        f"{'合计':<22} {'':8} {'':6} {'':6} {'':10} {total_segments:<10} {total_hours:<10.1f}"
    )

    # 保存报告
    project_root = Path(__file__).parent.parent.parent
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = project_root / "results" / "pretraining"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "preprocess_report.json"

    report = {
        "preprocess_time": datetime.now().isoformat(),
        "config": {
            "target_sfreq": TARGET_SFREQ,
            "patch_duration": PATCH_DURATION,
            "amp_threshold": AMP_THRESHOLD,
            "skip_seconds": SKIP_SECONDS,
            "artifact_filter": args.artifact_filter,
            "strict_amp_threshold": STRICT_AMP_THRESHOLD,
            "strict_kurt_threshold": STRICT_KURT_THRESHOLD,
        },
        "datasets": all_stats,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n预处理报告已保存: {output_path}")


if __name__ == "__main__":
    main()
