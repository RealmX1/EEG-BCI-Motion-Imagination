#!/usr/bin/env python3
"""
Stieger2021 增量预处理 — GUI 版
================================
tkinter 图形界面，三条进度条实时显示下载/预处理/搬移状态。

注意: Stieger2021 数据托管在 Figshare (AWS S3 eu-west-1, 爱尔兰)。
      从中国大陆下载建议使用 VPN 连接西欧节点（英国/德国/法国），
      否则速度极慢且频繁超时。

用法:
  uv run python scripts/pretraining/preprocess_stieger_gui.py
"""

import os
import sys
import pickle
import shutil
import time
import threading
import queue
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future

import tkinter as tk
from tkinter import ttk, scrolledtext

import numpy as np
import lmdb
import mne
import requests

mne.set_log_level("ERROR")

# ─────────────────────────────────────────────
# 路径 & 常量
# ─────────────────────────────────────────────

MI_DATA_DIR = Path(r"D:\data\motion_imagination_datasets")
os.environ.setdefault("MNE_DATA", str(MI_DATA_DIR))

DEFAULT_LMDB_PATH = MI_DATA_DIR / "lmdb_pretrain" / "Stieger2021_pretrain"
MOABB_RAW_DIR_D = MI_DATA_DIR / "MNE-Stieger2021-data"
MOABB_RAW_DIR_F = Path(r"F:\data\MNE-Stieger2021-data")

TARGET_SFREQ = 200
PATCH_DURATION = 30
PATCH_SIZE = 200
N_PATCHES = 30
SKIP_SECONDS = 5
AMP_THRESHOLD = 500
DISK_MIN_FREE_GB = 10

STIEGER_CONFIG = {
    "notch_freq": 60,
    "pick_types": {"eeg": True, "eog": False, "stim": False},
}
TO_UV_FACTOR = 1e6
DATASET_NAME = "Stieger2021"

# 让 import 能找到同目录脚本
sys.path.insert(0, str(Path(__file__).parent))
from preprocess_stieger_incremental import (
    read_existing_keys,
    extract_processed_subjects,
)


# ─────────────────────────────────────────────
# Figshare 缓存 (复用)
# ─────────────────────────────────────────────

_figshare_cache: dict | None = None


def warm_figshare_cache(dataset) -> dict:
    """一次性获取 Figshare 元数据并 monkey-patch data_path（禁止额外下载）。"""
    global _figshare_cache
    if _figshare_cache is not None:
        return _figshare_cache

    import moabb.datasets.download as dl_mod
    from moabb.datasets.download import get_dataset_path
    import types

    file_list = dl_mod.fs_get_file_list(dataset.figshare_id)
    hash_file_list = dl_mod.fs_get_file_hash(file_list)
    id_file_list = dl_mod.fs_get_file_id(file_list)
    _figshare_cache = {
        "hash_file_list": hash_file_list,
        "id_file_list": id_file_list,
    }

    # Monkey-patch data_path: 只返回路径列表，不做任何 API 调用或下载。
    # 下载由 download_subject_files 独占控制。
    def _patched_data_path(self, subject, path=None, force_update=False,
                           update_path=None, verbose=None):
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject {subject}")
        path = get_dataset_path(self.code, path)
        basepath = os.path.join(path, f"MNE-{self.code:s}-data")
        spath = []
        for file_name in _figshare_cache["id_file_list"].keys():
            if ".mat" not in file_name:
                continue
            sub = int(file_name.split("_")[0][1:])
            ses = int(file_name.split("_")[-1].split(".")[0])
            if sub == subject:
                if self.sessions is not None and ses not in self.sessions:
                    continue
                fpath = os.path.join(basepath, file_name)
                # 不下载！只返回已存在的路径
                if os.path.exists(fpath):
                    spath.append(fpath)
        return spath

    dataset.data_path = types.MethodType(_patched_data_path, dataset)
    return _figshare_cache


# ─────────────────────────────────────────────
# 自定义下载器 (带字节级回调)
# ─────────────────────────────────────────────

BASE_URL = "https://ndownloader.figshare.com/files/"


MAX_RETRIES = 3
DOWNLOAD_TIMEOUT = (15, 30)  # (connect_timeout, read_timeout) — 30s 无数据即判定卡顿


class DownloadAborted(Exception):
    """用户按停止，中止下载。"""
    pass


class CallbackDownloader:
    """替代 pooch.HTTPDownloader，带字节级回调、自动重试、可中断。"""

    def __init__(self, callback=None, error_callback=None,
                 stop_event: threading.Event | None = None):
        self.callback = callback            # (downloaded, total, fname)
        self.error_callback = error_callback  # (msg)
        self.stop_event = stop_event

    def __call__(self, url, output_file, pooch_instance):
        fname = Path(output_file).name
        output_path = Path(output_file)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # 每次重试前检查停止信号
                if self.stop_event and self.stop_event.is_set():
                    raise DownloadAborted(f"{fname}: 用户停止")

                response = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))

                downloaded = 0
                with open(output_file, "wb") as f:
                    for chunk in response.iter_content(chunk_size=256 * 1024):
                        # 每个 chunk 之间检查停止信号
                        if self.stop_event and self.stop_event.is_set():
                            response.close()
                            f.close()
                            # 删除不完整文件
                            if output_path.exists():
                                output_path.unlink()
                            raise DownloadAborted(f"{fname}: 用户停止")

                        f.write(chunk)
                        downloaded += len(chunk)
                        if self.callback:
                            self.callback(downloaded, total, fname)
                return  # 成功

            except DownloadAborted:
                raise  # 不重试，直接上抛

            except (requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.ChunkedEncodingError) as e:
                # 删除不完整文件
                if output_path.exists():
                    output_path.unlink()

                msg = (f"{fname}: 第 {attempt}/{MAX_RETRIES} 次失败 "
                       f"({type(e).__name__}: {e})")
                if self.error_callback:
                    self.error_callback(msg)
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(5 * attempt)  # 递增等待


def download_subject_files(
    subject: int,
    cache: dict,
    basepath: str,
    sessions: list | None,
    file_callback=None,
    byte_callback=None,
    error_callback=None,
    stop_event: threading.Event | None = None,
) -> list[str]:
    """下载单个被试的所有文件，返回文件路径列表。

    file_callback(file_idx, n_files, fname): 每个文件开始时调用
    byte_callback(downloaded, total, fname): 每块数据时调用
    error_callback(msg): 下载重试时调用
    stop_event: 如果 set，当前文件下载完后停止，不下载更多文件
    """
    import pooch as _pooch

    os.makedirs(basepath, exist_ok=True)

    files_to_download = []
    for file_name in cache["id_file_list"].keys():
        if ".mat" not in file_name:
            continue
        sub = int(file_name.split("_")[0][1:])
        ses = int(file_name.split("_")[-1].split(".")[0])
        if sub == subject:
            if sessions is not None and ses not in sessions:
                continue
            files_to_download.append(file_name)

    spath = []
    downloader = CallbackDownloader(
        callback=byte_callback, error_callback=error_callback,
        stop_event=stop_event,
    )

    for file_idx, file_name in enumerate(files_to_download):
        # 每个文件之间检查停止信号
        if stop_event is not None and stop_event.is_set():
            break

        fpath = os.path.join(basepath, file_name)
        if file_callback:
            file_callback(file_idx, len(files_to_download), file_name)
        if not os.path.exists(fpath):
            try:
                _pooch.retrieve(
                    url=BASE_URL + cache["id_file_list"][file_name],
                    known_hash=cache["hash_file_list"][cache["id_file_list"][file_name]],
                    fname=file_name,
                    path=basepath,
                    downloader=downloader,
                )
            except DownloadAborted:
                break  # 用户停止，不是错误
        else:
            # 文件已存在，报告完整
            if byte_callback:
                sz = os.path.getsize(fpath)
                byte_callback(sz, sz, file_name)
        spath.append(fpath)
    return spath


def verify_subject_files(subject: int, cache: dict, basepath: str,
                         sessions: list | None) -> tuple[int, int]:
    """验证被试的所有文件都已下载，返回 (已有数, 应有数)。"""
    expected = []
    for file_name in cache["id_file_list"].keys():
        if ".mat" not in file_name:
            continue
        sub = int(file_name.split("_")[0][1:])
        ses = int(file_name.split("_")[-1].split(".")[0])
        if sub == subject:
            if sessions is not None and ses not in sessions:
                continue
            expected.append(file_name)
    found = sum(1 for f in expected if os.path.exists(os.path.join(basepath, f)))
    return found, len(expected)


def load_subject_data(dataset, subject: int) -> dict:
    """加载已下载的被试数据。data_path 已被 monkey-patch，不会触发下载。"""
    return dataset.get_data(subjects=[subject])


# ─────────────────────────────────────────────
# 预处理函数
# ─────────────────────────────────────────────


def preprocess_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw | None:
    try:
        try:
            raw.pick(picks="eeg")
        except Exception:
            raw.pick_types(**STIEGER_CONFIG["pick_types"])
        if len(raw.info["ch_names"]) < 16:
            return None
        if not raw.preload:
            raw.load_data()
        if raw.info["sfreq"] != TARGET_SFREQ:
            raw.resample(TARGET_SFREQ)
        raw.filter(l_freq=0.3, h_freq=75.0)
        raw.notch_filter(freqs=STIEGER_CONFIG["notch_freq"])
        return raw
    except Exception:
        return None


def segment_raw_to_patches(raw: mne.io.BaseRaw) -> np.ndarray | None:
    data = raw.get_data() * TO_UV_FACTOR
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
    return data.transpose(1, 0, 2, 3)


def filter_segments(segments: np.ndarray) -> np.ndarray:
    mask = np.max(np.abs(segments), axis=(1, 2, 3)) < AMP_THRESHOLD
    return segments[mask]


# ─────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────


class StiegerPreprocessGUI:
    """三进度条 GUI：下载 / 预处理 / 搬移。"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Stieger2021 增量预处理")
        self.root.geometry("720x560")
        self.root.resizable(True, True)

        self.msg_queue: queue.Queue = queue.Queue()
        self.running = False
        self.stop_flag = threading.Event()
        self._db = None  # LMDB handle，确保异常时能关闭

        self._build_ui()
        self._poll_queue()

    # ── UI 构建 ──

    def _build_ui(self):
        root = self.root
        root.columnconfigure(0, weight=1)

        # 总进度
        frm_overall = ttk.LabelFrame(root, text="总进度", padding=8)
        frm_overall.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        frm_overall.columnconfigure(1, weight=1)

        self.lbl_overall = ttk.Label(frm_overall, text="就绪")
        self.lbl_overall.grid(row=0, column=0, columnspan=2, sticky="w")
        self.pb_overall = ttk.Progressbar(frm_overall, mode="determinate")
        self.pb_overall.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))

        # 三条管线进度
        stages = [
            ("download", "下载"),
            ("preprocess", "预处理"),
            ("move", "搬移"),
        ]
        self.stage_labels: dict[str, ttk.Label] = {}
        self.stage_bars: dict[str, ttk.Progressbar] = {}
        self.stage_details: dict[str, ttk.Label] = {}

        for i, (key, label) in enumerate(stages):
            frm = ttk.LabelFrame(root, text=label, padding=6)
            frm.grid(row=1 + i, column=0, sticky="ew", padx=10, pady=2)
            frm.columnconfigure(1, weight=1)

            lbl = ttk.Label(frm, text="空闲", width=40, anchor="w")
            lbl.grid(row=0, column=0, columnspan=2, sticky="w")
            self.stage_labels[key] = lbl

            pb = ttk.Progressbar(frm, mode="determinate")
            pb.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 0))
            self.stage_bars[key] = pb

            detail = ttk.Label(frm, text="", foreground="gray", anchor="w")
            detail.grid(row=2, column=0, columnspan=2, sticky="w")
            self.stage_details[key] = detail

        # 日志
        frm_log = ttk.LabelFrame(root, text="日志", padding=4)
        frm_log.grid(row=4, column=0, sticky="nsew", padx=10, pady=(4, 4))
        root.rowconfigure(4, weight=1)

        self.log_text = scrolledtext.ScrolledText(
            frm_log, height=8, state="disabled", wrap="word", font=("Consolas", 9)
        )
        self.log_text.pack(fill="both", expand=True)

        # 按钮
        frm_btn = ttk.Frame(root, padding=4)
        frm_btn.grid(row=5, column=0, sticky="ew", padx=10, pady=(0, 10))

        self.btn_start = ttk.Button(frm_btn, text="开始", command=self._on_start)
        self.btn_start.pack(side="left", padx=4)

        self.btn_stop = ttk.Button(
            frm_btn, text="停止", command=self._on_stop, state="disabled"
        )
        self.btn_stop.pack(side="left", padx=4)

        self.lbl_disk = ttk.Label(frm_btn, text="", foreground="gray")
        self.lbl_disk.pack(side="right", padx=4)

    # ── 消息处理 ──

    def _poll_queue(self):
        """每 50ms 从 queue 取消息更新 UI。"""
        while True:
            try:
                msg = self.msg_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_msg(msg)
        # 更新磁盘空间
        try:
            usage = shutil.disk_usage("D:\\")
            free_gb = usage.free / (1024**3)
            self.lbl_disk.config(text=f"D: {free_gb:.1f} GB 剩余")
        except Exception:
            pass
        self.root.after(50, self._poll_queue)

    def _handle_msg(self, msg: dict):
        t = msg.get("type")

        if t == "log":
            self._append_log(msg["text"])

        elif t == "overall":
            done = msg.get("done", 0)
            total = msg.get("total", 1)
            self.pb_overall["maximum"] = total
            self.pb_overall["value"] = done
            self.lbl_overall.config(
                text=f"被试 {done}/{total} 完成  |  "
                f"总段数: {msg.get('segments', '?')}"
            )

        elif t == "stage":
            stage = msg["stage"]
            if "label" in msg:
                self.stage_labels[stage].config(text=msg["label"])
            if "progress" in msg and "maximum" in msg:
                self.stage_bars[stage]["maximum"] = msg["maximum"]
                self.stage_bars[stage]["value"] = msg["progress"]
            if "detail" in msg:
                self.stage_details[stage].config(text=msg["detail"])

        elif t == "done":
            self.running = False
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")
            self._append_log(msg.get("text", "完成"))

    def _append_log(self, text: str):
        self.log_text.config(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _send(self, **kwargs):
        self.msg_queue.put(kwargs)

    # ── 控制 ──

    def _on_start(self):
        if self.running:
            return
        self.running = True
        self.stop_flag.clear()
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        t = threading.Thread(target=self._pipeline_worker, daemon=True)
        t.start()

    def _on_stop(self):
        self.stop_flag.set()
        self._send(type="log",
                   text="正在停止: 等待当前文件下载完成 -> 预处理完成 -> 搬移完成...")
        self.btn_stop.config(state="disabled")

    # ── 管线工作线程 ──

    def _pipeline_worker(self):
        try:
            self._run_pipeline()
        except Exception as e:
            self._send(type="log", text=f"错误: {e}")
            traceback.print_exc()
        finally:
            # 确保 LMDB 关闭，否则重启时会 "already open"
            if self._db is not None:
                try:
                    self._db.close()
                except Exception:
                    pass
                self._db = None
            self._send(type="done", text="管线结束")

    def _run_pipeline(self):
        lmdb_path = DEFAULT_LMDB_PATH
        lmdb_path.mkdir(parents=True, exist_ok=True)

        # 1. 读取已有
        self._send(type="log", text="读取已有 LMDB...")
        existing_keys = read_existing_keys(lmdb_path)
        processed_subjects = extract_processed_subjects(existing_keys)
        self._send(
            type="log",
            text=f"已有 {len(existing_keys)} 段, 被试: {sorted(processed_subjects)}",
        )

        # 2. 获取被试列表
        import moabb.datasets as moabb_ds

        dataset = moabb_ds.Stieger2021()
        all_subjects = dataset.subject_list
        todo = [s for s in all_subjects if s not in processed_subjects]

        if not todo:
            self._send(type="log", text="所有 62 名被试已全部处理完毕!")
            return

        self._send(type="log", text=f"待处理: {len(todo)} 个被试")
        self._send(
            type="overall",
            done=len(processed_subjects),
            total=len(all_subjects),
            segments=len(existing_keys),
        )

        # 3. 缓存 Figshare 元数据
        self._send(type="log", text="缓存 Figshare 元数据...")
        self._send(type="stage", stage="download", label="获取文件列表...")
        cache = warm_figshare_cache(dataset)
        self._send(type="log", text=f"Figshare: {len(cache['id_file_list'])} 文件")

        # 4. 打开 LMDB
        est_new = len(todo) * 200 * 700 * 1024
        map_size = min(max(2 * 1024**3, est_new * 4), 50 * 1024**3)
        db = lmdb.open(str(lmdb_path), map_size=map_size)
        self._db = db  # 追踪，确保异常时能关闭

        all_keys = list(existing_keys)
        total_new_segments = 0
        n_done = len(processed_subjects)

        from moabb.datasets.download import get_dataset_path

        basepath = os.path.join(
            get_dataset_path(dataset.code, None), f"MNE-{dataset.code}-data"
        )

        # 线程池
        dl_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dl")
        mv_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mv")
        move_future: Future | None = None
        prefetch_future: Future | None = None

        # 预取第一个被试
        self._send(type="stage", stage="download", label=f"S{todo[0]} 下载中...")
        prefetch_future = dl_pool.submit(
            self._download_subject, dataset, todo[0], cache, basepath
        )

        for idx, subject in enumerate(todo):
            # 在被试之间检查停止信号（不是在被试处理中途）
            # stop_flag 会让 download 在当前文件完成后停止，
            # 但已开始的预处理和搬移会正常完成
            if self.stop_flag.is_set() and prefetch_future is None:
                # 上一轮预取已取消，无下载在进行，可以安全退出循环
                break

            # 等待当前被试下载完成
            if prefetch_future is not None:
                self._send(
                    type="stage",
                    stage="download",
                    label=f"S{subject} 等待下载完成...",
                )
                try:
                    prefetch_future.result()  # blocks
                except Exception as dl_err:
                    self._send(
                        type="log",
                        text=f"S{subject}: 下载失败 ({dl_err})，跳过",
                    )
                    prefetch_future = None
                    # 仍尝试启动下一个被试的预取
                    if idx + 1 < len(todo) and not self.stop_flag.is_set():
                        next_subj = todo[idx + 1]
                        prefetch_future = dl_pool.submit(
                            self._download_subject, dataset, next_subj,
                            cache, basepath
                        )
                    continue
                prefetch_future = None

            # stop 后不再启动新的预取
            if self.stop_flag.is_set():
                self._send(
                    type="stage",
                    stage="download",
                    label="已停止，不再下载新被试",
                    progress=0, maximum=1, detail="",
                )
            elif idx + 1 < len(todo):
                next_subj = todo[idx + 1]
                prefetch_future = dl_pool.submit(
                    self._download_subject, dataset, next_subj, cache, basepath
                )

            # 验证所有文件已下载
            found, expected = verify_subject_files(
                subject, cache, basepath, dataset.sessions
            )
            if found < expected:
                self._send(
                    type="log",
                    text=f"S{subject}: 文件不完整 ({found}/{expected})，跳过",
                )
                continue

            # 加载数据（data_path 已 monkey-patch，不会触发下载）
            self._send(
                type="stage",
                stage="preprocess",
                label=f"S{subject} 加载数据 ({found} 文件)...",
                progress=0, maximum=1, detail="",
            )
            data = load_subject_data(dataset, subject)

            # 预处理（即使 stop_flag set，也完成当前被试的预处理）
            if self.stop_flag.is_set():
                self._send(type="log",
                           text=f"停止中: 完成 S{subject} 的预处理...")
            new_keys, map_size, db, n_segments = self._preprocess_subject(
                subject, data, db, map_size, lmdb_path
            )
            del data

            # 更新 keys
            if subject in processed_subjects:
                prefix = f"{DATASET_NAME}_s{subject}_"
                all_keys = [k for k in all_keys if not k.startswith(prefix)]
            all_keys.extend(new_keys)
            total_new_segments += n_segments

            # 更新 __keys__ 索引
            self._send(
                type="stage",
                stage="preprocess",
                label=f"S{subject} 更新索引...",
                detail=f"{len(all_keys)} keys",
            )
            try:
                txn = db.begin(write=True)
                txn.put(b"__keys__", pickle.dumps(all_keys))
                txn.commit()
            except lmdb.MapFullError:
                txn.abort()
                db.close()
                map_size = int(map_size * 2)
                db = lmdb.open(str(lmdb_path), map_size=map_size)
                self._db = db
                txn = db.begin(write=True)
                txn.put(b"__keys__", pickle.dumps(all_keys))
                txn.commit()

            n_done += 1
            self._send(
                type="stage",
                stage="preprocess",
                label=f"S{subject} 完成 ({n_segments} 段)",
                progress=1, maximum=1, detail="",
            )
            self._send(
                type="overall",
                done=n_done,
                total=len(all_subjects),
                segments=len(all_keys),
            )
            self._send(
                type="log", text=f"S{subject}: {n_segments} 段完成"
            )

            # 等待上一个搬移完成
            if move_future is not None:
                move_future.result()

            # 启动当前被试的搬移
            move_future = mv_pool.submit(self._move_subject, subject)

            # stop 后: 当前被试预处理+搬移已启动，跳出循环
            if self.stop_flag.is_set():
                self._send(type="log",
                           text=f"停止中: 等待 S{subject} 搬移完成...")
                break

        # ── 优雅关闭 ──

        # 等待最后一个搬移完成
        if move_future is not None:
            self._send(
                type="stage", stage="move",
                label="等待搬移完成...",
            )
            move_future.result()

        # 如果有正在进行的预取，等它的当前文件完成
        if prefetch_future is not None:
            self._send(type="log", text="等待后台下载当前文件完成...")
            prefetch_future.result()

        dl_pool.shutdown(wait=True)
        mv_pool.shutdown(wait=True)
        db.close()
        self._db = None

        if self.stop_flag.is_set():
            self._send(
                type="log",
                text=f"已停止。本次新增 {total_new_segments} 段, "
                f"LMDB 共 {len(all_keys)} 段。下次运行自动继续。",
            )
        else:
            self._send(
                type="log",
                text=f"全部完成! 新增 {total_new_segments} 段, "
                f"LMDB 共 {len(all_keys)} 段",
            )

    # ── 下载 (在线程中运行) ──

    def _download_subject(self, dataset, subject, cache, basepath):
        """带进度回调的被试下载。"""
        # 等待磁盘空间（stop 时也跳出等待）
        while not self.stop_flag.is_set():
            try:
                free_gb = shutil.disk_usage("D:\\").free / (1024**3)
            except Exception:
                free_gb = 999
            if free_gb >= DISK_MIN_FREE_GB:
                break
            self._send(
                type="stage",
                stage="download",
                label=f"S{subject} 等待磁盘空间...",
                detail=f"D: {free_gb:.1f} GB < {DISK_MIN_FREE_GB} GB",
            )
            time.sleep(5)

        if self.stop_flag.is_set():
            self._send(
                type="stage", stage="download",
                label=f"S{subject} 下载已取消 (停止信号)",
                progress=0, maximum=1, detail="",
            )
            return

        current_file_idx = 0
        total_files = 0

        def on_file(file_idx, n_files, fname):
            nonlocal current_file_idx, total_files
            current_file_idx = file_idx
            total_files = n_files
            self._send(
                type="stage",
                stage="download",
                label=f"S{subject}  文件 {file_idx + 1}/{n_files}  {fname}",
                progress=file_idx,
                maximum=n_files,
            )

        def on_bytes(downloaded, total, fname):
            if total > 0:
                mb_dl = downloaded / (1024**2)
                mb_total = total / (1024**2)
                pct = downloaded * 100 // total
                self._send(
                    type="stage",
                    stage="download",
                    detail=f"{mb_dl:.0f}/{mb_total:.0f} MB  ({pct}%)",
                    progress=current_file_idx * 1000
                    + int(1000 * downloaded / total),
                    maximum=total_files * 1000,
                )

        def on_error(msg):
            self._send(type="log", text=msg)
            self._send(
                type="stage", stage="download",
                detail=f"重试中...",
            )

        download_subject_files(
            subject, cache, basepath,
            sessions=dataset.sessions,
            file_callback=on_file,
            byte_callback=on_bytes,
            error_callback=on_error,
            stop_event=self.stop_flag,
        )

        if self.stop_flag.is_set():
            self._send(
                type="stage", stage="download",
                label=f"S{subject} 下载中断 (当前文件已完成)",
                progress=0, maximum=1, detail="",
            )
        else:
            self._send(
                type="stage", stage="download",
                label=f"S{subject} 下载完成",
                progress=1, maximum=1, detail="",
            )

    # ── 预处理 (在主工作线程中运行) ──

    def _preprocess_subject(self, subject, data, db, map_size, lmdb_path):
        new_keys = []
        subj_segments_after = 0

        all_runs = []
        for subj_id, sessions in data.items():
            for sess_name, runs in sessions.items():
                for run_name, raw in runs.items():
                    all_runs.append((sess_name, run_name, raw))

        for run_idx, (sess_name, run_name, raw) in enumerate(all_runs):
            self._send(
                type="stage",
                stage="preprocess",
                label=f"S{subject}  session {sess_name}  [{run_idx + 1}/{len(all_runs)}]",
                progress=run_idx,
                maximum=len(all_runs),
                detail="预处理中...",
            )

            processed = preprocess_raw(raw.copy())
            if processed is None:
                self._send(
                    type="stage",
                    stage="preprocess",
                    detail="跳过 (通道不足)",
                )
                continue

            segments = segment_raw_to_patches(processed)
            if segments is None:
                self._send(
                    type="stage",
                    stage="preprocess",
                    detail="跳过 (太短)",
                )
                continue

            n_before = len(segments)
            segments = filter_segments(segments)
            n_after = len(segments)
            subj_segments_after += n_after

            self._send(
                type="stage",
                stage="preprocess",
                detail=f"{n_after}/{n_before} 段, 写入 LMDB...",
            )

            # 准备 key-value
            seg_kvs = []
            for i, seg in enumerate(segments):
                key = f"{DATASET_NAME}_s{subject}_{sess_name}_{run_name}_{i}"
                seg_kvs.append((key, pickle.dumps(seg.astype(np.float32))))

            # 写入 LMDB
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
                    db = lmdb.open(str(lmdb_path), map_size=map_size)
                    self._db = db

            if not written:
                raise RuntimeError("LMDB MapFull")

            for key, _ in seg_kvs:
                new_keys.append(key)

            self._send(
                type="stage",
                stage="preprocess",
                detail=f"{n_after} 段 -> LMDB OK",
                progress=run_idx + 1,
                maximum=len(all_runs),
            )
            del processed, segments

        self._send(
            type="stage",
            stage="preprocess",
            label=f"S{subject} 完成: {subj_segments_after} 段",
            progress=1,
            maximum=1,
            detail="",
        )
        return new_keys, map_size, db, subj_segments_after

    # ── 搬移 (在线程中运行) ──

    def _move_subject(self, subject: int):
        if not MOABB_RAW_DIR_D.exists():
            self._send(
                type="stage",
                stage="move",
                label=f"S{subject} 跳过 (D: 目录不存在)",
            )
            return

        MOABB_RAW_DIR_F.mkdir(parents=True, exist_ok=True)

        files = []
        for session in range(1, 12):
            fname = f"S{subject}_Session_{session}.mat"
            src = MOABB_RAW_DIR_D / fname
            if src.exists():
                files.append((fname, src))

        if not files:
            self._send(
                type="stage",
                stage="move",
                label=f"S{subject} 无文件需搬移",
                progress=1,
                maximum=1,
                detail="",
            )
            return

        total_bytes = 0
        for file_idx, (fname, src) in enumerate(files):
            dst = MOABB_RAW_DIR_F / fname
            sz = src.stat().st_size
            mb = sz / (1024**2)
            self._send(
                type="stage",
                stage="move",
                label=f"S{subject}  {fname}  [{file_idx + 1}/{len(files)}]",
                progress=file_idx,
                maximum=len(files),
                detail=f"{mb:.0f} MB",
            )
            try:
                shutil.move(str(src), str(dst))
                total_bytes += sz
            except Exception as e:
                self._send(type="log", text=f"搬移失败 {fname}: {e}")

        gb = total_bytes / (1024**3)
        self._send(
            type="stage",
            stage="move",
            label=f"S{subject} 搬移完成: {len(files)} 文件, {gb:.1f} GB",
            progress=1,
            maximum=1,
            detail="",
        )

    # ── 启动 ──

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = StiegerPreprocessGUI()
    app.run()
