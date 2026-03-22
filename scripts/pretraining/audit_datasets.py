#!/usr/bin/env python3
"""
MI 数据集审计脚本
================
对 D:/data/motion_imagination_datasets/ 下的所有 MOABB 数据集进行审计：
  - 通道数、通道名、采样率
  - 被试数、会话数
  - 可用录制时长估算
  - 数据质量预检

输出：审计报告 JSON + 终端汇总表格
"""

import os
import json
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np

# 设置 MNE_DATA 确保 MOABB 使用正确路径
MI_DATA_DIR = Path(r"D:\data\motion_imagination_datasets")
os.environ.setdefault("MNE_DATA", str(MI_DATA_DIR))

# 禁止 MNE 下载提示
os.environ.setdefault("MNE_DATASETS_EEGBCI_PATH", str(MI_DATA_DIR))

import mne

mne.set_log_level("ERROR")

# ─────────────────────────────────────────────
# 数据集配置
# ─────────────────────────────────────────────

# 每个数据集的元信息（MOABB 类名 → 审计配置）
DATASET_CONFIGS = {
    "Lee2019_MI": {
        "moabb_class": "Lee2019_MI",
        "expected_channels": 62,
        "expected_sfreq": 1000,
        "陷波": 50,
        "priority": "***",
        "notes": "最大 MI 数据集",
    },
    "Stieger2021": {
        "moabb_class": "Stieger2021",
        "expected_channels": 64,
        "expected_sfreq": 500,
        "陷波": 60,
        "priority": "***",
        "notes": "纵向多会话",
    },
    "PhysionetMI": {
        "moabb_class": "PhysionetMI",
        "expected_channels": 64,
        "expected_sfreq": 160,
        "陷波": 60,
        "priority": "***",
        "notes": "160Hz 需上采样",
    },
    "Cho2017": {
        "moabb_class": "Cho2017",
        "expected_channels": 64,
        "expected_sfreq": 512,
        "陷波": 60,
        "priority": "**",
        "notes": "",
    },
    "Schirrmeister2017": {
        "moabb_class": "Schirrmeister2017",
        "expected_channels": 128,
        "expected_sfreq": 500,
        "陷波": 50,
        "priority": "**",
        "notes": "排除 EOG/EMG",
    },
    "GrosseWentrup2009": {
        "moabb_class": "GrosseWentrup2009",
        "expected_channels": 128,
        "expected_sfreq": 500,
        "陷波": 60,
        "priority": "**",
        "notes": "",
    },
    "Ofner2017": {
        "moabb_class": "Ofner2017",
        "expected_channels": 61,
        "expected_sfreq": 512,
        "陷波": 50,
        "priority": "**",
        "notes": "上肢运动",
    },
    "BNCI2015_004": {
        "moabb_class": "BNCI2015_004",
        "expected_channels": 30,
        "expected_sfreq": 256,
        "陷波": 50,
        "priority": "*",
        "notes": "",
    },
    "Shin2017A": {
        "moabb_class": "Shin2017A",
        "expected_channels": 30,
        "expected_sfreq": 200,
        "陷波": 50,
        "priority": "*",
        "notes": "需分离 EEG/fNIRS",
    },
    "BNCI2014_001": {
        "moabb_class": "BNCI2014_001",
        "expected_channels": 22,
        "expected_sfreq": 250,
        "陷波": 50,
        "priority": "*",
        "notes": "经典 BCI-IV 2a",
    },
    "Weibo2014": {
        "moabb_class": "Weibo2014",
        "expected_channels": 60,
        "expected_sfreq": 200,
        "陷波": 50,
        "priority": "*",
        "notes": "检查 zip",
    },
    "Dreyer2023": {
        "moabb_class": "Dreyer2023",
        "expected_channels": 27,
        "expected_sfreq": 512,
        "陷波": 50,
        "priority": "*",
        "notes": "检查下载",
    },
}

# 排除的数据集
EXCLUDED_DATASETS = {
    "BNCI2014_004": "仅 3 通道，ACPE 效果极差",
    "Zhou2016": "仅 4 被试 14 通道，数据量过小",
    "AlexMI": "仅 8 被试 16 通道，数据量极小",
}


def audit_single_dataset(name: str, config: dict) -> dict:
    """审计单个 MOABB 数据集，返回审计结果字典。"""
    result = {
        "name": name,
        "moabb_class": config["moabb_class"],
        "priority": config["priority"],
        "notes": config["notes"],
        "status": "unknown",
        "error": None,
        "subjects": [],
        "n_subjects": 0,
        "channel_names": [],
        "n_channels_eeg": 0,
        "non_eeg_channels": [],
        "sfreq": None,
        "total_duration_seconds": 0,
        "total_duration_hours": 0,
        "n_sessions": 0,
        "n_runs": 0,
        "sample_amplitude_stats": {},
    }

    try:
        import moabb.datasets as moabb_ds

        # 获取 MOABB 数据集类
        ds_class = getattr(moabb_ds, config["moabb_class"])
        dataset = ds_class()

        # 获取被试列表
        subjects = dataset.subject_list
        result["subjects"] = [int(s) if isinstance(s, (int, np.integer)) else str(s) for s in subjects]
        result["n_subjects"] = len(subjects)

        # 只抽样第一个被试做详细检查
        first_subject = subjects[0]
        data = dataset.get_data(subjects=[first_subject])

        total_duration = 0.0
        n_sessions = 0
        n_runs = 0
        channel_names = None
        sfreq = None
        eeg_channels = []
        non_eeg_channels = []
        amplitude_samples = []

        for subj_id, sessions in data.items():
            for sess_name, runs in sessions.items():
                n_sessions += 1
                for run_name, raw in runs.items():
                    n_runs += 1
                    total_duration += raw.times[-1]

                    if channel_names is None:
                        sfreq = raw.info["sfreq"]
                        all_ch = raw.info["ch_names"]
                        ch_types = raw.get_channel_types()
                        for ch_name, ch_type in zip(all_ch, ch_types):
                            if ch_type == "eeg":
                                eeg_channels.append(ch_name)
                            else:
                                non_eeg_channels.append(f"{ch_name} ({ch_type})")
                        channel_names = all_ch

                    # 幅值抽样（取前 5 秒）
                    sample_len = min(int(raw.info["sfreq"] * 5), raw.n_times)
                    raw_data = raw.get_data(
                        picks="eeg", start=0, stop=sample_len
                    )
                    # MNE 返回 V，转 µV
                    raw_data_uv = raw_data * 1e6
                    amplitude_samples.append(np.max(np.abs(raw_data_uv)))

        result["channel_names"] = eeg_channels
        result["n_channels_eeg"] = len(eeg_channels)
        result["non_eeg_channels"] = non_eeg_channels
        result["sfreq"] = sfreq
        result["n_sessions_first_subject"] = n_sessions
        result["n_runs_first_subject"] = n_runs
        result["first_subject_duration_seconds"] = round(total_duration, 1)

        # 估算所有被试总时长（用第一个被试外推）
        est_total = total_duration * len(subjects)
        result["total_duration_seconds"] = round(est_total, 1)
        result["total_duration_hours"] = round(est_total / 3600, 2)

        if amplitude_samples:
            result["sample_amplitude_stats"] = {
                "max_abs_uV": round(float(np.max(amplitude_samples)), 1),
                "mean_max_abs_uV": round(float(np.mean(amplitude_samples)), 1),
            }

        result["status"] = "ok"

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    return result


def print_summary_table(results: list[dict]):
    """打印审计汇总表格。"""
    print("\n" + "=" * 120)
    print("MI 数据集审计汇总")
    print("=" * 120)
    header = f"{'数据集':<22} {'优先级':<6} {'状态':<8} {'通道(EEG)':<10} {'采样率':<8} {'被试':<6} {'估计时长(h)':<12} {'备注'}"
    print(header)
    print("-" * 120)

    total_hours = 0
    total_subjects = 0

    for r in results:
        status = r["status"]
        ch = r["n_channels_eeg"] if r["n_channels_eeg"] else "?"
        sfreq = r["sfreq"] if r["sfreq"] else "?"
        n_sub = r["n_subjects"]
        hours = r["total_duration_hours"]
        notes = r["notes"]
        if r["error"]:
            notes = f"ERROR: {r['error'][:50]}"

        print(
            f"{r['name']:<22} {r['priority']:<6} {status:<8} {str(ch):<10} {str(sfreq):<8} {n_sub:<6} {hours:<12} {notes}"
        )

        if status == "ok":
            total_hours += hours
            total_subjects += n_sub

    print("-" * 120)
    print(f"{'合计':<22} {'':6} {'':8} {'':10} {'':8} {total_subjects:<6} {total_hours:<12.1f}")
    print()

    # 排除数据集
    print("排除的数据集:")
    for name, reason in EXCLUDED_DATASETS.items():
        print(f"  [FAIL]{name}: {reason}")
    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MI 数据集审计")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="指定要审计的数据集名称（默认全部）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 JSON 路径（默认 results/pretraining/audit_report.json）",
    )
    args = parser.parse_args()

    # 确定输出路径
    project_root = Path(__file__).parent.parent.parent
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = project_root / "results" / "pretraining"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "audit_report.json"

    # 选择数据集
    if args.datasets:
        configs = {k: v for k, v in DATASET_CONFIGS.items() if k in args.datasets}
    else:
        configs = DATASET_CONFIGS

    print(f"准备审计 {len(configs)} 个数据集...")
    print(f"数据目录: {MI_DATA_DIR}")
    print()

    results = []
    for name, config in configs.items():
        print(f"[审计] {name} ({config['moabb_class']})...")
        r = audit_single_dataset(name, config)
        results.append(r)
        if r["status"] == "ok":
            print(
                f"  [OK]{r['n_channels_eeg']} EEG 通道, {r['sfreq']} Hz, "
                f"{r['n_subjects']} 被试, ~{r['total_duration_hours']:.1f}h"
            )
        else:
            print(f"  [FAIL]{r['error']}")

    print_summary_table(results)

    # 保存 JSON
    report = {
        "audit_time": datetime.now().isoformat(),
        "data_dir": str(MI_DATA_DIR),
        "excluded": EXCLUDED_DATASETS,
        "datasets": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"审计报告已保存: {output_path}")


if __name__ == "__main__":
    main()
