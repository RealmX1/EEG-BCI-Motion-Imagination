#!/usr/bin/env python3
"""
运动想象（Motor Imagery）EEG 公开数据集一键下载脚本
=====================================================
支持数据集：
  - MOABB集成（自动下载）：BNCI2014_001/004/002, BNCI2015_001/004,
    BNCI2003_004, PhysionetMI, Cho2017, Lee2019_MI, GrosseWentrup2009,
    Schirrmeister2017, Ofner2017, Weibo2014, Zhou2016, Shin2017A,
    AlexMI, Stieger2021, Dreyer2023
  - PhysioNet wget批量下载（备用）
  - BCI Competition官网链接提示

依赖安装：
  pip install moabb mne tqdm requests
"""

import os
import sys
import time
import argparse
import traceback

# 在所有库导入前设置 MNE_DATA，确保所有数据集使用正确路径
_DEFAULT_DATA_DIR = r"D:\data\motion_imagination_datasets"
os.environ.setdefault("MNE_DATA", _DEFAULT_DATA_DIR)

# ─────────────────────────────────────────────
# 数据集目录（MOABB支持的所有MI数据集）
# ─────────────────────────────────────────────
MOABB_DATASETS = {
    # 别名               : (MOABB类名,               中文说明,                          估计大小)
    "IV_2a"             : ("BNCI2014_001",  "BCI Comp IV Dataset 2a (4类, 9人, 22ch)",   "~420 MB"),
    "IV_2b"             : ("BNCI2014_004",  "BCI Comp IV Dataset 2b (2类, 9人, 3ch)",    "~216 MB"),
    "BNCI2014_002"      : ("BNCI2014_002",  "BNCI2014-002 (2类, 14人, 15ch)",            "~150 MB"),
    "BNCI2015_001"      : ("BNCI2015_001",  "BNCI2015-001 (2类, 12人, 13ch)",            "~100 MB"),
    "BNCI2015_004"      : ("BNCI2015_004",  "BNCI2015-004 (5类, 9人, 30ch)",             "~200 MB"),
    "III_IVa"           : ("BNCI2003_004",  "BCI Comp III Dataset IVa (2类, 5人, 118ch)","~300 MB"),
    "PhysionetMI"       : ("PhysionetMI",   "PhysioNet EEGMMIDB (4类, 109人, 64ch)",     "~3.4 GB"),
    "Cho2017"           : ("Cho2017",       "Cho2017 (2类, 52人, 64ch)",                  "~5.5 GB"),
    "Lee2019_MI"        : ("Lee2019_MI",    "OpenBMI / Lee2019 (2类, 54人, 62ch)",        "~61 GB"),
    "Schirrmeister2017" : ("Schirrmeister2017", "High Gamma (4类, 14人, 128ch)",          "~18 GB"),
    "GrosseWentrup2009" : ("GrosseWentrup2009", "MunichMI (2类, 10人, 128ch)",            "~7.3 GB"),
    "Ofner2017"         : ("Ofner2017",     "Ofner2017 上肢7类 (7类, 15人, 61ch)",        "~13 GB"),
    "Weibo2014"         : ("Weibo2014",     "Weibo2014 复合肢体 (7类, 10人, 60ch)",       "~1.5 GB"),
    "Zhou2016"          : ("Zhou2016",      "Zhou2016 (3类, 4人, 14ch)",                  "~50 MB"),
    "Shin2017A"         : ("Shin2017A",     "Shin2017A EEG+fNIRS (2类, 29人, 30ch)",      "~3 GB"),
    "AlexMI"            : ("AlexMI",        "AlexMI (3类, 8人, 16ch)",                    "~100 MB"),
    "Stieger2021"       : ("Stieger2021",   "Stieger2021 纵向 (4类, 62人, 7-11次会话)",  "~25 GB"),
    "Dreyer2023"        : ("Dreyer2023",    "Dreyer2023 BCI文盲研究 (2类, 87人, 27ch)",   "~10 GB"),
}

# 需要手动下载的数据集（仅提供链接说明）
MANUAL_DATASETS = {
    "BCI_Comp_II_III": {
        "name": "BCI Competition II - Dataset III",
        "url": "https://www.bbci.de/competition/ii/",
        "format": "MAT",
        "note": "需注册，下载后用 scipy.io.loadmat() 加载",
    },
    "BCI_Comp_III_IIIa": {
        "name": "BCI Competition III - Dataset IIIa (4类, 3人, 60ch)",
        "url": "https://www.bbci.de/competition/iii/download/",
        "format": "GDF / MAT / ASCII",
        "note": "需注册，GDF可用 mne.io.read_raw_gdf() 加载",
    },
    "BCI_Comp_IV_1": {
        "name": "BCI Competition IV - Dataset 1 (2类, 7人, 59ch)",
        "url": "https://www.bbci.de/competition/iv/download/",
        "format": "MAT",
        "note": "需注册，MAT格式用 scipy.io.loadmat() 加载",
    },
}

# ─────────────────────────────────────────────
# 颜色输出工具
# ─────────────────────────────────────────────
class C:
    HEADER  = "\033[95m"
    BLUE    = "\033[94m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    BOLD    = "\033[1m"
    RESET   = "\033[0m"

def log(msg, color=C.RESET):
    print(f"{color}{msg}{C.RESET}", flush=True)

def section(title):
    print()
    log("=" * 60, C.BOLD)
    log(f"  {title}", C.BOLD + C.HEADER)
    log("=" * 60, C.BOLD)

# ─────────────────────────────────────────────
# 依赖检查
# ─────────────────────────────────────────────
def check_dependencies():
    section("检查依赖")
    missing = []
    for pkg, import_name in [("moabb", "moabb"), ("mne", "mne"), ("tqdm", "tqdm")]:
        try:
            __import__(import_name)
            log(f"  [OK] {pkg}", C.GREEN)
        except ImportError:
            log(f"  [FAIL] {pkg} 未安装", C.RED)
            missing.append(pkg)
    if missing:
        log(f"\n请先安装缺失依赖：pip install {' '.join(missing)}", C.YELLOW)
        sys.exit(1)

# ─────────────────────────────────────────────
# 核心下载逻辑
# ─────────────────────────────────────────────
def download_moabb_dataset(alias, class_name, desc, est_size, subjects=None, save_dir=None):
    """通过MOABB下载单个数据集"""
    import moabb
    import moabb.datasets as moabb_datasets

    log(f"\n>>  {alias}  —  {desc}", C.BLUE + C.BOLD)
    log(f"   估计大小：{est_size}", C.YELLOW)

    try:
        cls = getattr(moabb_datasets, class_name)
        try:
            dataset = cls(accept=True)
        except TypeError:
            dataset = cls()

        # 获取受试者列表
        available = dataset.subject_list
        if subjects:
            target = [s for s in subjects if s in available]
            if not target:
                log(f"   [!]  指定受试者 {subjects} 不在可用列表 {available[:5]}... 中", C.YELLOW)
                return False
        else:
            target = available

        log(f"   受试者总数：{len(available)}，本次下载：{len(target)} 人", C.RESET)

        t0 = time.time()
        dataset.download(subject_list=target, path=save_dir, force_update=False, verbose=False)
        elapsed = time.time() - t0

        log(f"   [OK] 下载完成！耗时 {elapsed:.1f}s", C.GREEN)
        return True

    except AttributeError:
        log(f"   [FAIL] 未找到数据集类 '{class_name}'，请确认MOABB版本", C.RED)
        return False
    except Exception as e:
        log(f"   [FAIL] 下载失败：{e}", C.RED)
        if "--verbose" in sys.argv:
            traceback.print_exc()
        return False


def print_manual_download_info():
    """打印需要手动下载的数据集信息"""
    section("需手动注册下载的数据集")
    for key, info in MANUAL_DATASETS.items():
        log(f"\n* {info['name']}", C.BOLD)
        log(f"   URL    : {info['url']}", C.BLUE)
        log(f"   格式   : {info['format']}")
        log(f"   备注   : {info['note']}", C.YELLOW)


def print_summary(results):
    """打印下载汇总"""
    section("下载汇总")
    ok  = [k for k, v in results.items() if v is True]
    err = [k for k, v in results.items() if v is False]
    skp = [k for k, v in results.items() if v == "skip"]

    if ok:
        log(f"\n[OK] 成功 ({len(ok)})：", C.GREEN)
        for k in ok:
            log(f"   - {k}", C.GREEN)
    if skp:
        log(f"\n[SKIP]  跳过 ({len(skp)})：", C.YELLOW)
        for k in skp:
            log(f"   - {k}", C.YELLOW)
    if err:
        log(f"\n[FAIL] 失败 ({len(err)})：", C.RED)
        for k in err:
            log(f"   - {k}", C.RED)

    log(f"\n合计：{len(ok)} 成功 / {len(skp)} 跳过 / {len(err)} 失败", C.BOLD)


# ─────────────────────────────────────────────
# 参数解析
# ─────────────────────────────────────────────
def build_parser():
    parser = argparse.ArgumentParser(
        description="运动想象EEG数据集一键下载工具",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--datasets", "-d",
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "指定要下载的数据集别名（空格分隔），默认下载全部MOABB数据集。\n"
            "可用别名：\n" +
            "\n".join(f"  {k:20s} {v[2]}" for k, v in MOABB_DATASETS.items())
        ),
    )
    parser.add_argument(
        "--subjects", "-s",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="只下载指定受试者编号（如 --subjects 1 2 3），默认下载全部受试者",
    )
    parser.add_argument(
        "--save-dir", "-o",
        default=_DEFAULT_DATA_DIR,
        metavar="DIR",
        help=f"数据保存目录，默认：{_DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="只列出所有可用数据集，不执行下载",
    )
    parser.add_argument(
        "--skip-large",
        action="store_true",
        help="跳过大于5 GB的数据集（Lee2019_MI、Schirrmeister2017、GrosseWentrup2009、Ofner2017、Stieger2021）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细错误信息",
    )
    return parser


LARGE_DATASETS = {"Lee2019_MI", "Schirrmeister2017", "GrosseWentrup2009",
                  "Ofner2017", "Stieger2021"}

# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────
def main():
    parser = build_parser()
    args = parser.parse_args()

    # 只列出数据集
    if args.list:
        section("所有可用 MOABB MI 数据集")
        log(f"\n{'别名':<22} {'MOABB类名':<22} {'估计大小':<12} 说明", C.BOLD)
        log("-" * 90)
        for alias, (cls, desc, size) in MOABB_DATASETS.items():
            flag = " [!] 大" if alias in LARGE_DATASETS else ""
            log(f"{alias:<22} {cls:<22} {size:<12} {desc}{flag}")
        log("\n需手动下载的数据集：", C.YELLOW)
        for key, info in MANUAL_DATASETS.items():
            log(f"  {key:<28} {info['url']}", C.YELLOW)
        return

    check_dependencies()

    # 设置保存目录
    save_dir = args.save_dir
    if save_dir:
        save_dir = os.path.abspath(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        os.environ["MNE_DATA"] = save_dir
        import mne
        mne.set_config("MNE_DATA", save_dir)
        # 同步所有数据集专属路径（这些优先级高于 MNE_DATA）
        _dataset_keys = [
            "MNE_DATASETS_ALEXEEG_PATH", "MNE_DATASETS_BBCIFNIRS_PATH",
            "MNE_DATASETS_BNCI_PATH", "MNE_DATASETS_DREYER2023_PATH",
            "MNE_DATASETS_EEGBCI_PATH", "MNE_DATASETS_GIGADB_PATH",
            "MNE_DATASETS_LEE2019-MI_PATH", "MNE_DATASETS_MUNICHMI_PATH",
            "MNE_DATASETS_SCHIRRMEISTER2017_PATH", "MNE_DATASETS_STIEGER2021_PATH",
            "MNE_DATASETS_UPPERLIMB_PATH", "MNE_DATASETS_WEIBO_PATH",
            "MNE_DATASETS_ZHOU2016_PATH",
        ]
        for _key in _dataset_keys:
            mne.set_config(_key, save_dir)
        log(f"\n数据保存目录：{save_dir}", C.BLUE)

    # 确定要下载的数据集
    if args.datasets:
        to_download = {}
        for name in args.datasets:
            if name in MOABB_DATASETS:
                to_download[name] = MOABB_DATASETS[name]
            else:
                log(f"[!]  未知数据集别名 '{name}'，已跳过。可用别名见 --list", C.YELLOW)
    else:
        to_download = dict(MOABB_DATASETS)

    # 跳过大数据集
    if args.skip_large:
        skipped = [k for k in to_download if k in LARGE_DATASETS]
        if skipped:
            log(f"\n--skip-large 已跳过：{', '.join(skipped)}", C.YELLOW)
        to_download = {k: v for k, v in to_download.items() if k not in LARGE_DATASETS}

    section(f"开始下载 {len(to_download)} 个数据集")
    if args.subjects:
        log(f"  仅下载受试者：{args.subjects}", C.YELLOW)

    results = {}
    total_start = time.time()

    for alias, (class_name, desc, est_size) in to_download.items():
        ok = download_moabb_dataset(
            alias, class_name, desc, est_size,
            subjects=args.subjects,
            save_dir=save_dir,
        )
        results[alias] = ok

    total_elapsed = time.time() - total_start
    log(f"\nTime:  总耗时：{total_elapsed/60:.1f} 分钟", C.BOLD)

    print_manual_download_info()
    print_summary(results)


if __name__ == "__main__":
    main()