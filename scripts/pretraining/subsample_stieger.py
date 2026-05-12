"""一次性脚本：从 Stieger2021_pretrain LMDB 随机子采样，生成 V3 DAPT 用的小 LMDB。

目标：把 Stieger2021 在 V3 训练 sampler 中的 segment 占比从 V2 的 ~80% 降至 ~30%。

数学推导：
    V3 总 segments = K_stieger + N_other
    目标 share = K_stieger / (K_stieger + N_other) = target_share
    => K_stieger = N_other * target_share / (1 - target_share)

handoff 文档原文 "26K keys = 42% = 30% share" 的算术与实际数据不符
（26K / (26K + 16.7K) = 60.9%，非 30%）。本脚本按 share 反推 keys。
"""

import argparse
import pickle
import random
from pathlib import Path

import lmdb

DEFAULT_BASE = Path(r"D:/data/motion_imagination_datasets/lmdb_pretrain")
DEFAULT_SOURCE = DEFAULT_BASE / "Stieger2021_pretrain"
DEFAULT_OUTPUT = DEFAULT_BASE / "Stieger2021_subsampled_30pct"


def _calc_map_size(data_mdb: Path) -> int:
    file_size = data_mdb.stat().st_size
    gb = 1024 ** 3
    return max(2 * gb, ((file_size // gb) + 1) * gb)


def count_keys(lmdb_dir: Path) -> int:
    data_mdb = lmdb_dir / "data.mdb"
    map_size = _calc_map_size(data_mdb)
    env = lmdb.open(str(lmdb_dir), readonly=True, lock=False, map_size=map_size)
    try:
        with env.begin() as txn:
            blob = txn.get(b"__keys__")
            if blob is None:
                raise RuntimeError(f"{lmdb_dir} 缺少 __keys__")
            return len(pickle.loads(blob))
    finally:
        env.close()


def collect_other_total(base: Path, exclude_name_substr: str = "Stieger") -> tuple[int, dict[str, int]]:
    breakdown: dict[str, int] = {}
    for d in sorted(base.iterdir()):
        if not d.is_dir() or exclude_name_substr in d.name:
            continue
        if not (d / "data.mdb").exists():
            continue
        breakdown[d.name] = count_keys(d)
    return sum(breakdown.values()), breakdown


def subsample(source: Path, output: Path, n_keys: int, seed: int) -> None:
    if output.exists():
        raise SystemExit(f"输出目录已存在，拒绝覆盖：{output}")

    src_data = source / "data.mdb"
    src_size = src_data.stat().st_size
    src_map = _calc_map_size(src_data)
    src_env = lmdb.open(str(source), readonly=True, lock=False, map_size=src_map)
    try:
        with src_env.begin() as txn:
            all_keys = pickle.loads(txn.get(b"__keys__"))
    except Exception:
        src_env.close()
        raise

    if n_keys > len(all_keys):
        src_env.close()
        raise SystemExit(f"请求 {n_keys} keys 超过源 {len(all_keys)}")

    rng = random.Random(seed)
    selected = rng.sample(all_keys, n_keys)
    print(f"selected {len(selected)} keys from {len(all_keys)} (seed={seed})")

    avg_bytes = src_size / len(all_keys)
    est = int(avg_bytes * len(selected) * 1.5)
    dst_map = max(2 * 1024 ** 3, est)
    print(f"output map_size: {dst_map / 1024**3:.2f} GB (1.5x safety margin)")

    output.mkdir(parents=True, exist_ok=False)
    dst_env = lmdb.open(str(output), map_size=dst_map)

    try:
        chunk = 200
        with src_env.begin() as src_txn:
            for start in range(0, len(selected), chunk):
                batch = selected[start:start + chunk]
                with dst_env.begin(write=True) as dst_txn:
                    for key in batch:
                        kb = key.encode() if isinstance(key, str) else key
                        val = src_txn.get(kb)
                        if val is None:
                            raise SystemExit(f"源中找不到 key={key!r}")
                        dst_txn.put(kb, val)
                if start % (chunk * 5) == 0:
                    print(f"  copied {min(start + chunk, len(selected))}/{len(selected)}")

        with dst_env.begin(write=True) as txn:
            txn.put(b"__keys__", pickle.dumps(selected))
    finally:
        dst_env.close()
        src_env.close()

    out_size = (output / "data.mdb").stat().st_size
    print(f"\nwrote {output} size={out_size / 1024**3:.2f} GB")

    verify_env = lmdb.open(str(output), readonly=True, lock=False, map_size=_calc_map_size(output / "data.mdb"))
    try:
        with verify_env.begin() as txn:
            verify = pickle.loads(txn.get(b"__keys__"))
            n_kv_pairs = txn.stat()["entries"]
    finally:
        verify_env.close()
    assert len(verify) == n_keys, f"verify FAIL: keys list len {len(verify)} != {n_keys}"
    assert n_kv_pairs == n_keys + 1, f"verify FAIL: kv entries {n_kv_pairs} != {n_keys + 1}"
    print(f"verify OK: __keys__ list len={len(verify)}, kv entries={n_kv_pairs} (= n_keys + 1 for __keys__ itself)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--base", type=Path, default=DEFAULT_BASE, help="root containing 10 LMDBs")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--target-share", type=float, default=0.30,
                   help="Stieger 在 V3 训练总 segments 中目标占比（默认 0.30）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    other_total, breakdown = collect_other_total(args.base)
    print("other 9 datasets:")
    for name, n in sorted(breakdown.items()):
        print(f"  {name}: {n}")
    print(f"  TOTAL: {other_total}")

    src_keys = count_keys(args.source)
    target = int(other_total * args.target_share / (1 - args.target_share))
    print(f"\nStieger source: {src_keys} keys")
    print(f"target_share={args.target_share} => K_stieger={target} ({target / src_keys * 100:.2f}% of source)")
    new_share = target / (target + other_total)
    print(f"resulting share: {new_share * 100:.2f}%")

    if args.dry_run:
        print("--dry-run: stop before writing")
        return

    subsample(args.source, args.output, target, seed=args.seed)


if __name__ == "__main__":
    main()
