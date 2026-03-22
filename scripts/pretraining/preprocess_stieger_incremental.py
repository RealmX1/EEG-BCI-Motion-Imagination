#!/usr/bin/env python3
"""
Stieger2021 增量预处理脚本
===========================
逐个被试下载、预处理、写入 LMDB，然后将原始数据从 D: 搬到 F: 归档。

注意: Stieger2021 数据托管在 Figshare (AWS S3 eu-west-1, 爱尔兰)。
      从中国大陆下载建议使用 VPN 连接西欧节点（英国/德国/法国），
      否则速度极慢且频繁超时。

流程（每个被试）：
  1. MOABB 从 Figshare 下载 S{id}_Session_{1-11}.mat 到 D: (~6.6 GB)
  2. 预处理 → 切段 → 写入 LMDB
  3. 将 D: 上的 .mat 文件搬到 F:/data/MNE-Stieger2021-data/ 归档
  4. D: 空间释放，继续下一个被试

特性：
  - 增量模式：读取已有 LMDB 中的 __keys__，跳过已处理的被试
  - 断点续传：每个被试完成后立即更新 __keys__，中断可恢复
  - 原始数据保留在 F: 上（不删除）

用法:
  # 处理所有剩余被试（自动跳过已完成的）
  uv run python scripts/pretraining/preprocess_stieger_incremental.py

  # 只处理指定被试
  uv run python scripts/pretraining/preprocess_stieger_incremental.py --subjects 15 16 17

  # 每处理 N 个被试后暂停（适合分批运行）
  uv run python scripts/pretraining/preprocess_stieger_incremental.py --batch-size 10

  # 不搬移原始数据（保留在 D: 上）
  uv run python scripts/pretraining/preprocess_stieger_incremental.py --no-move
"""

import os
import sys
import pickle
import shutil
import argparse
import traceback
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np
import lmdb
import mne

mne.set_log_level("ERROR")

# ─────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────

MI_DATA_DIR = Path(r"D:\data\motion_imagination_datasets")
os.environ.setdefault("MNE_DATA", str(MI_DATA_DIR))

DEFAULT_LMDB_PATH = MI_DATA_DIR / "lmdb_pretrain" / "Stieger2021_pretrain"
MOABB_RAW_DIR_D = MI_DATA_DIR / "MNE-Stieger2021-data"           # MOABB 下载到 D:
MOABB_RAW_DIR_F = Path(r"F:\data\MNE-Stieger2021-data")          # 归档到 F:

# ─────────────────────────────────────────────
# 预处理参数（与 preprocess_mi_datasets.py 一致）
# ─────────────────────────────────────────────

TARGET_SFREQ = 200
PATCH_DURATION = 30
PATCH_SIZE = 200
N_PATCHES = 30
SKIP_SECONDS = 5
AMP_THRESHOLD = 500

STIEGER_CONFIG = {
    "notch_freq": 60,
    "pick_types": {"eeg": True, "eog": False, "stim": False},
}

# 元数据（Stieger2021: V → uV）
TO_UV_FACTOR = 1e6

DATASET_NAME = "Stieger2021"
DISK_MIN_FREE_GB = 10  # 暂停下载的阈值


# ─────────────────────────────────────────────
# Figshare API 元数据缓存
# ─────────────────────────────────────────────
# MOABB 的 data_path() 每次调用都打 3 次 Figshare API（获取 682 个文件的
# 列表/哈希/ID），非常慢。我们只查一次，然后 monkey-patch data_path 使用缓存。

_figshare_cache: dict | None = None


def _warm_figshare_cache(dataset):
    """一次性获取 Figshare 元数据并缓存，然后 monkey-patch data_path。"""
    global _figshare_cache
    if _figshare_cache is not None:
        return

    import moabb.datasets.download as dl_mod
    from moabb.datasets.download import get_dataset_path
    import pooch as _pooch

    print("  缓存 Figshare 元数据 (一次性)...", end="", flush=True)
    file_list = dl_mod.fs_get_file_list(dataset.figshare_id)
    hash_file_list = dl_mod.fs_get_file_hash(file_list)
    id_file_list = dl_mod.fs_get_file_id(file_list)
    _figshare_cache = {
        "hash_file_list": hash_file_list,
        "id_file_list": id_file_list,
    }
    print(f" OK ({len(id_file_list)} 文件)", flush=True)

    # Monkey-patch: 替换 data_path 以使用缓存
    BASE_URL = "https://ndownloader.figshare.com/files/"

    def _cached_data_path(self, subject, path=None, force_update=False,
                          update_path=None, verbose=None):
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject {subject}")
        path = get_dataset_path(self.code, path)
        basepath = os.path.join(path, f"MNE-{self.code:s}-data")

        cached = _figshare_cache
        spath = []
        for file_name in cached["id_file_list"].keys():
            if ".mat" not in file_name:
                continue
            sub = int(file_name.split("_")[0][1:])
            ses = int(file_name.split("_")[-1].split(".")[0])
            if sub == subject:
                if self.sessions is not None and ses not in self.sessions:
                    continue
                fpath = os.path.join(basepath, file_name)
                if not os.path.exists(fpath):
                    _pooch.retrieve(
                        url=BASE_URL + cached["id_file_list"][file_name],
                        known_hash=cached["hash_file_list"][
                            cached["id_file_list"][file_name]
                        ],
                        fname=file_name,
                        path=basepath,
                        downloader=_pooch.HTTPDownloader(progressbar=True),
                    )
                spath.append(fpath)
        return spath

    # Bind to instance
    import types
    dataset.data_path = types.MethodType(_cached_data_path, dataset)


# ─────────────────────────────────────────────
# 预处理函数（复用自 preprocess_mi_datasets.py）
# ─────────────────────────────────────────────


def preprocess_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw | None:
    """对单个 Raw 执行标准预处理管线。"""
    try:
        try:
            raw.pick(picks="eeg")
        except Exception:
            raw.pick_types(**STIEGER_CONFIG["pick_types"])

        n_eeg = len(raw.info["ch_names"])
        if n_eeg < 16:
            return None

        if not raw.preload:
            raw.load_data()

        if raw.info["sfreq"] != TARGET_SFREQ:
            raw.resample(TARGET_SFREQ)

        raw.filter(l_freq=0.3, h_freq=75.0)
        raw.notch_filter(freqs=STIEGER_CONFIG["notch_freq"])

        return raw
    except Exception as e:
        print(f"    预处理失败: {e}")
        return None


def segment_raw_to_patches(raw: mne.io.BaseRaw) -> np.ndarray | None:
    """切分为 30 秒段，shape (n_segments, n_channels, 30, 200)。"""
    data = raw.get_data() * TO_UV_FACTOR  # V → uV

    n_channels, n_times = data.shape
    skip_samples = SKIP_SECONDS * TARGET_SFREQ

    if n_times <= 2 * skip_samples + PATCH_DURATION * TARGET_SFREQ:
        return None

    data = data[:, skip_samples:-skip_samples]
    n_times = data.shape[1]

    segment_samples = PATCH_DURATION * TARGET_SFREQ
    n_segments = n_times // segment_samples
    if n_segments == 0:
        return None

    data = data[:, :n_segments * segment_samples]
    data = data.reshape(n_channels, n_segments, N_PATCHES, PATCH_SIZE)
    data = data.transpose(1, 0, 2, 3)

    return data


def filter_segments(segments: np.ndarray) -> np.ndarray:
    """质量过滤：丢弃 max(abs) >= AMP_THRESHOLD 的段。"""
    mask = np.max(np.abs(segments), axis=(1, 2, 3)) < AMP_THRESHOLD
    return segments[mask]


# ─────────────────────────────────────────────
# LMDB 辅助
# ─────────────────────────────────────────────


def read_existing_keys(lmdb_path: Path) -> list[str]:
    """读取已有 LMDB 中的 __keys__ 列表。"""
    if not (lmdb_path / "data.mdb").exists():
        return []
    try:
        db = lmdb.open(str(lmdb_path), readonly=True, lock=False)
        with db.begin() as txn:
            raw = txn.get(b"__keys__")
            if raw is None:
                db.close()
                return []
            keys = pickle.loads(raw)
        db.close()
        return keys
    except Exception as e:
        print(f"  警告: 读取已有 keys 失败: {e}")
        return []


def extract_processed_subjects(keys: list[str]) -> set[int]:
    """从 key 列表中提取已处理的被试编号。

    Key 格式: Stieger2021_s{subj}_{sess}_{run}_{seg}
    """
    subjects = set()
    for k in keys:
        parts = k.split("_")
        for p in parts:
            if p.startswith("s") and p[1:].isdigit():
                subjects.add(int(p[1:]))
    return subjects


def move_subject_raw_to_f(subject: int):
    """将 MOABB 下载到 D: 的原始 .mat 文件搬到 F: 归档。

    Stieger2021 Figshare 下载结构:
      D:/.../MNE-Stieger2021-data/S{id}_Session_{1-11}.mat
    每个被试 11 个 session，每文件 ~600MB，共约 6.6GB。
    搬到 F:/data/MNE-Stieger2021-data/ 保留，释放 D: 空间。
    """
    if not MOABB_RAW_DIR_D.exists():
        print("    搬移跳过: D: 原始目录不存在")
        return

    MOABB_RAW_DIR_F.mkdir(parents=True, exist_ok=True)

    moved = 0
    total_bytes = 0
    for session in range(1, 12):  # Session 1-11
        fname = f"S{subject}_Session_{session}.mat"
        src = MOABB_RAW_DIR_D / fname
        dst = MOABB_RAW_DIR_F / fname
        if not src.exists():
            continue
        try:
            sz = src.stat().st_size
            mb = sz / (1024 ** 2)
            print(f"    搬移 {fname} ({mb:.0f} MB)...", end="", flush=True)
            shutil.move(str(src), str(dst))
            total_bytes += sz
            moved += 1
            print(" OK", flush=True)
        except Exception as e:
            print(f" 失败: {e}")

    if moved:
        gb = total_bytes / (1024 ** 3)
        print(f"    搬移完成: {moved} 文件, {gb:.1f} GB")


# ─────────────────────────────────────────────
# 主处理逻辑
# ─────────────────────────────────────────────


def check_disk_free_gb(path: str = "D:\\") -> float:
    """返回指定盘符剩余空间 (GB)。"""
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def download_subject(dataset, subject: int) -> dict:
    """下载并加载单个被试数据（可在后台线程运行）。

    会等待 D: 剩余空间 >= DISK_MIN_FREE_GB 才开始下载。
    """
    # 等待磁盘空间
    while True:
        free_gb = check_disk_free_gb("D:\\")
        if free_gb >= DISK_MIN_FREE_GB:
            break
        print(f"    [prefetch S{subject}] D: 仅剩 {free_gb:.1f} GB, "
              f"等待 >= {DISK_MIN_FREE_GB} GB...", flush=True)
        time.sleep(10)

    data = dataset.get_data(subjects=[subject])
    return data


def preprocess_subject_data(
    subject: int,
    data: dict,
    db,
    map_size: int,
    lmdb_path: Path,
) -> tuple[list[str], int, object, int]:
    """对已下载的被试数据做预处理+写LMDB，返回 (new_keys, map_size, db, n_segments)。"""
    new_keys = []
    subj_segments_after = 0

    # 收集所有 runs
    all_runs = []
    for subj_id, sessions in data.items():
        for sess_name, runs in sessions.items():
            for run_name, raw in runs.items():
                all_runs.append((sess_name, run_name, raw))

    for run_idx, (sess_name, run_name, raw) in enumerate(all_runs):
        print(
            f"    预处理 session {sess_name} [{run_idx + 1}/{len(all_runs)}]...",
            end="", flush=True,
        )

        processed = preprocess_raw(raw.copy())
        if processed is None:
            print(" 跳过 (通道不足)", flush=True)
            continue

        segments = segment_raw_to_patches(processed)
        if segments is None:
            print(" 跳过 (太短)", flush=True)
            continue

        n_before = len(segments)
        segments = filter_segments(segments)
        n_after = len(segments)
        subj_segments_after += n_after

        print(f" {n_after}/{n_before} 段", end="", flush=True)

        # 准备 key-value 对
        seg_kvs = []
        for i, seg in enumerate(segments):
            key = f"{DATASET_NAME}_s{subject}_{sess_name}_{run_name}_{i}"
            seg_kvs.append((key, pickle.dumps(seg.astype(np.float32))))

        # 写入 LMDB（带 MapFull 自动扩容）
        written = False
        for _attempt in range(5):
            try:
                txn = db.begin(write=True)
                for key, val in seg_kvs:
                    txn.put(key.encode(), val)
                txn.commit()
                written = True
                break
            except lmdb.MapFullError:
                txn.abort()
                db.close()
                map_size = int(map_size * 2)
                print(f" [MapFull, 扩到 {map_size // 1024 // 1024}MB]", end="")
                db = lmdb.open(str(lmdb_path), map_size=map_size)

        if not written:
            raise RuntimeError("LMDB 写入失败: 5 次扩容仍 MapFull")

        for key, _ in seg_kvs:
            new_keys.append(key)

        print(" -> LMDB OK", flush=True)
        del processed, segments

    print(f"    被试 {subject} 合计: {subj_segments_after} 段", flush=True)
    return new_keys, map_size, db, subj_segments_after


def update_keys_index(db, all_keys: list[str], map_size: int, lmdb_path: Path):
    """更新 LMDB 中的 __keys__ 索引。"""
    try:
        txn = db.begin(write=True)
        txn.put(b"__keys__", pickle.dumps(all_keys))
        txn.commit()
    except lmdb.MapFullError:
        txn.abort()
        db.close()
        map_size = int(map_size * 2)
        db = lmdb.open(str(lmdb_path), map_size=map_size)
        txn = db.begin(write=True)
        txn.put(b"__keys__", pickle.dumps(all_keys))
        txn.commit()
    return db, map_size


def main():
    parser = argparse.ArgumentParser(
        description="Stieger2021 增量预处理 → LMDB（逐被试下载+清理）",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        type=int,
        default=None,
        help="指定要处理的被试编号（默认: 所有未处理的）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="每批处理的被试数量（默认: 全部）",
    )
    parser.add_argument(
        "--lmdb-path",
        type=str,
        default=None,
        help=f"LMDB 路径（默认: {DEFAULT_LMDB_PATH}）",
    )
    parser.add_argument(
        "--no-move",
        action="store_true",
        help="处理后不将原始数据从 D: 搬到 F:（默认: 搬移）",
    )
    args = parser.parse_args()

    lmdb_path = Path(args.lmdb_path) if args.lmdb_path else DEFAULT_LMDB_PATH
    lmdb_path.mkdir(parents=True, exist_ok=True)

    # 1. 读取已有 keys
    existing_keys = read_existing_keys(lmdb_path)
    processed_subjects = extract_processed_subjects(existing_keys)

    print(f"LMDB 路径: {lmdb_path}")
    print(f"已有 {len(existing_keys)} 段, 来自被试: {sorted(processed_subjects)}")

    # 2. 获取完整被试列表
    import moabb.datasets as moabb_ds

    dataset = moabb_ds.Stieger2021()
    all_subjects = dataset.subject_list

    # 3. 确定要处理的被试
    if args.subjects:
        todo = [s for s in args.subjects if s in all_subjects]
        # 允许重新处理已有被试（用户显式指定时）
        skipped = [s for s in todo if s in processed_subjects]
        if skipped:
            print(f"注意: 被试 {skipped} 已存在于 LMDB，将重新处理并覆盖")
    else:
        todo = [s for s in all_subjects if s not in processed_subjects]

    if args.batch_size:
        todo = todo[:args.batch_size]

    if not todo:
        print("所有被试已处理完毕，无需继续。")
        return

    print(f"待处理被试 ({len(todo)}): {todo}")
    print(f"总被试数: {len(all_subjects)}, 已完成: {len(processed_subjects)}, "
          f"本次: {len(todo)}")

    # 一次性缓存 Figshare 元数据（省去每个被试 3 次 API 调用）
    _warm_figshare_cache(dataset)

    print("=" * 60)

    # 4. 打开 LMDB
    # 估算 map_size：已有数据 + 新被试估计
    # 每被试约 1000 段 × 700KB = ~700MB，但大多数不到这个数
    est_new = len(todo) * 200 * 700 * 1024  # 保守估计
    map_size = max(2 * 1024 * 1024 * 1024, est_new * 4)  # 至少 2GB
    map_size = min(map_size, 50 * 1024 * 1024 * 1024)  # 上限 50GB

    db = lmdb.open(str(lmdb_path), map_size=map_size)

    # 5. 流水线处理: 下载 N+1 与 预处理 N + 搬移 N-1 并行
    all_keys = list(existing_keys)
    total_new_segments = 0
    n_success = 0
    n_failed = 0

    # 线程池: 1 个下载线程 + 1 个搬移线程
    dl_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="download")
    mv_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="move")
    move_future: Future | None = None  # 上一个被试的搬移任务

    # 预取第一个被试
    print(f"  预取被试 {todo[0]} 下载...", flush=True)
    prefetch_future: Future | None = dl_pool.submit(download_subject, dataset, todo[0])

    for idx, subject in enumerate(todo):
        try:
            print(f"\n  [{idx + 1}/{len(todo)}] 被试 {subject}", flush=True)

            # 等待当前被试的下载完成（正常情况下已经在后台完成了）
            if prefetch_future is not None:
                print(f"    等待下载完成...", end="", flush=True)
                data = prefetch_future.result()
                print(" OK", flush=True)
                prefetch_future = None
            else:
                # 回退: 同步下载
                print(f"    下载中...", end="", flush=True)
                data = download_subject(dataset, subject)
                print(" OK", flush=True)

            # 立即启动下一个被试的预取下载（如果有）
            if idx + 1 < len(todo):
                next_subj = todo[idx + 1]
                print(f"    (后台预取被试 {next_subj})", flush=True)
                prefetch_future = dl_pool.submit(download_subject, dataset, next_subj)

            # 预处理当前被试（主线程，CPU/GPU 密集）
            new_keys, map_size, db, n_segments = preprocess_subject_data(
                subject, data, db, map_size, lmdb_path
            )
            del data  # 释放内存

            # 追加新 keys（如果是重处理，先移除旧 keys）
            if subject in processed_subjects:
                prefix = f"{DATASET_NAME}_s{subject}_"
                all_keys = [k for k in all_keys if not k.startswith(prefix)]

            all_keys.extend(new_keys)
            total_new_segments += n_segments
            n_success += 1

            # 更新 __keys__ 索引
            print(f"    更新 LMDB 索引 ({len(all_keys)} keys)...", end="", flush=True)
            db, map_size = update_keys_index(db, all_keys, map_size, lmdb_path)
            print(" OK", flush=True)

            # 等待上一个被试的搬移完成（如有）
            if move_future is not None:
                move_future.result()

            # 后台搬移当前被试的原始数据 D: → F:
            if not args.no_move:
                move_future = mv_pool.submit(move_subject_raw_to_f, subject)

        except Exception as e:
            n_failed += 1
            print(f" 失败: {e}")
            traceback.print_exc()

    # 等待最后一个搬移完成
    if move_future is not None:
        move_future.result()

    dl_pool.shutdown(wait=False)
    mv_pool.shutdown(wait=False)
    db.close()

    # 6. 汇总
    print("=" * 60)
    print(f"完成: {n_success} 成功, {n_failed} 失败")
    print(f"新增段数: {total_new_segments}")
    print(f"LMDB 总段数: {len(all_keys)}")
    print(f"LMDB 路径: {lmdb_path}")

    final_subjects = extract_processed_subjects(all_keys)
    remaining = [s for s in all_subjects if s not in final_subjects]
    if remaining:
        print(f"剩余未处理: {remaining}")
    else:
        print("所有 62 名被试已全部处理完毕!")


if __name__ == "__main__":
    main()
