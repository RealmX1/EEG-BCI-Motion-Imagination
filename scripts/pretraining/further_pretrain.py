#!/usr/bin/env python3
"""
CBraMod Domain-Adaptive Further Pre-training
=============================================
从 TUEG 预训练权重出发，在 MI 数据集上继续自监督预训练。

核心设计：
  - 多数据集交替采样（各数据集通道数不同，不能混 batch）
  - 梯度累积模拟大 batch（有效 batch_size=128）
  - AMP 混合精度训练
  - Linear warmup + Constant LR (DAPT 风格)

用法：
  uv run python scripts/pretraining/further_pretrain.py
  uv run python scripts/pretraining/further_pretrain.py --epochs 5 --lr 5e-5
  uv run python scripts/pretraining/further_pretrain.py --lmdb-dirs D:/data/.../Lee2019_MI_pretrain D:/data/.../PhysionetMI_pretrain
"""

import os
import sys
import json
import time
import pickle
import random
import argparse
import logging
import importlib.util
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import lmdb
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────
# 项目路径设置
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
CBRAMOD_REPO = Path(os.environ.get(
    "CBRAMOD_REPO_PATH",
    str(PROJECT_ROOT.parent / "CBraMod"),
))

# Force unbuffered stderr (Windows nohup workaround)
class FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

_handler = FlushHandler(sys.stderr)
_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
logging.basicConfig(level=logging.INFO, handlers=[_handler])
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 默认配置
# ─────────────────────────────────────────────
DEFAULT_LMDB_DIR = Path(r"D:\data\motion_imagination_datasets\lmdb_pretrain")

DEFAULT_CONFIG = {
    # 训练超参数
    "max_epochs": 50,
    "patience": 5,                          # 连续 N epoch 无改善则停
    "min_delta": 1e-4,                      # 小于此的 loss 改善视为噪声
    "batch_size": 16,                       # 物理 batch 上限 (按通道数自动缩放)
    "effective_batch_size": 128,            # 目标有效 batch (样本数累积触发 optimizer step)
    "reference_channels": 64,              # batch_size 对应的参考通道数
    "lr": 5e-5,
    "weight_decay": 0.05,
    "warmup_epochs": 0.5,
    "eta_min": 1e-6,
    "scheduler": "warmup_constant",    # "warmup_constant" or "phased_cosine"
    "num_phases": 5,                   # for phased_cosine: number of cosine phases
    "phase_decay": 0.5,               # for phased_cosine: peak LR decay per phase
    "lr_ramp_ratio": 0.1,             # for phased_cosine: fraction of phase for ramp-up
    "clip_value": 1.0,
    "mask_ratio": 0.5,
    # AMP
    "amp_enabled": True,
    # 数据加载
    "num_workers": 0,              # Windows LMDB: workers 各自 mmap 整个 DB，10 数据集 × N workers 爆 RAM
    "pin_memory": True,
    # 模型
    "d_model": 200,
    "dim_feedforward": 800,
    "n_layer": 12,
    "nhead": 8,
    # 权重路径
    "pretrained_weights": None,  # 自动搜索
    # 保存
    "save_every_epoch": True,
    "checkpoint_dir": None,  # 自动设置
}


# ─────────────────────────────────────────────
# LMDB 数据集
# ─────────────────────────────────────────────


class LMDBPretrainingDataset(Dataset):
    """从 LMDB 读取预处理好的 EEG 段。

    Windows 上 LMDB 的 mmap 会持续占用物理内存（OS 积极 page-in），
    10 个数据库同时打开会导致 RAM 溢出。因此每次读取后关闭 LMDB 环境，
    仅保持一个短暂的 mmap 窗口。
    """

    def __init__(self, lmdb_path: str | Path):
        self.lmdb_path = str(lmdb_path)

        # 临时打开获取 keys 和元数据，然后关闭
        db = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get("__keys__".encode()))
            sample = pickle.loads(txn.get(self.keys[0].encode()))
        db.close()

        self.n_channels = sample.shape[0]
        self.n_patches = sample.shape[1]
        self.patch_size = sample.shape[2]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        db = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with db.begin(write=False) as txn:
            patch = pickle.loads(txn.get(key.encode()))
        db.close()
        return torch.from_numpy(patch).float()

    def __repr__(self):
        name = Path(self.lmdb_path).name
        return f"LMDBPretrainingDataset({name}, n={len(self)}, ch={self.n_channels})"


# ─────────────────────────────────────────────
# 多数据集交替采样器
# ─────────────────────────────────────────────


class MultiDatasetSampler:
    """
    管理多个 DataLoader（通道数不同，不能混 batch），
    按数据集大小加权随机采样。
    """

    def __init__(self, dataloaders: dict[str, DataLoader]):
        self.dataloaders = dataloaders
        self.names = list(dataloaders.keys())

        # 按数据集大小计算采样权重
        sizes = {name: len(dl.dataset) for name, dl in dataloaders.items()}
        total = sum(sizes.values())
        self.weights = {name: s / total for name, s in sizes.items()}

        # 每个 dataloader 的迭代器
        self.iterators: dict[str, iter] = {}

        logger.info("多数据集采样器初始化:")
        for name in self.names:
            logger.info(
                f"  {name}: {sizes[name]} 样本, 权重 {self.weights[name]:.3f}"
            )

        # 计算总步数（一个 epoch = 所有数据各过一遍）
        self.total_batches = sum(
            len(dl) for dl in dataloaders.values()
        )

    def _get_or_reset_iterator(self, name: str):
        """获取迭代器，如果耗尽则重置。"""
        if name not in self.iterators:
            self.iterators[name] = iter(self.dataloaders[name])
        try:
            return next(self.iterators[name])
        except StopIteration:
            self.iterators[name] = iter(self.dataloaders[name])
            return next(self.iterators[name])

    def sample_batch(self) -> tuple[str, torch.Tensor]:
        """按权重随机选一个数据集，返回 (dataset_name, batch)。"""
        name = random.choices(self.names, weights=[self.weights[n] for n in self.names])[0]
        batch = self._get_or_reset_iterator(name)
        return name, batch

    def reset(self):
        """重置所有迭代器（新 epoch）。"""
        self.iterators.clear()


# ─────────────────────────────────────────────
# 掩码生成（复用 CBraMod 的逻辑）
# ─────────────────────────────────────────────


def generate_mask(bz, ch_num, patch_num, mask_ratio, device):
    """生成随机二值掩码，1 表示被遮盖的位置。"""
    mask = torch.zeros((bz, ch_num, patch_num), dtype=torch.long, device=device)
    mask = mask.bernoulli_(mask_ratio)
    return mask


# ─────────────────────────────────────────────
# Warmup + Cosine 学习率调度
# ─────────────────────────────────────────────


class WarmupConstantScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup → Constant LR.

    Default for DAPT: the weights are already in a stable basin,
    short warmup then maintain constant LR.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            return list(self.base_lrs)


class PhasedCosineWarmupDecayScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Multi-phase cosine annealing with LR ramp-up and peak decay (step-based).

    Adapted from CosineAnnealingWarmupDecay (epoch-based, src/training/schedulers.py)
    for step-based pretraining loops.

    Each phase contains:
      - LR ramp-up (first lr_ramp_ratio fraction): linear rise to peak
      - Cosine decay (remaining fraction): cosine fall to eta_min

    Peak LR decays by phase_decay each phase.

    Args:
        optimizer: PyTorch optimizer
        total_steps: Total training steps
        num_phases: Number of cosine phases (default: 5)
        phase_decay: Peak LR decay factor between phases (default: 0.5)
        lr_ramp_ratio: Fraction of each phase for LR ramp-up (default: 0.1)
        eta_min: Minimum learning rate (default: 1e-6)
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        num_phases: int = 5,
        phase_decay: float = 0.5,
        lr_ramp_ratio: float = 0.1,
        eta_min: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.total_steps = total_steps
        self.num_phases = num_phases
        self.phase_decay = phase_decay
        self.lr_ramp_ratio = lr_ramp_ratio
        self.eta_min = eta_min
        self.steps_per_phase = total_steps // num_phases
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        phase_idx = min(step // max(1, self.steps_per_phase), self.num_phases - 1)
        step_in_phase = step - phase_idx * self.steps_per_phase
        progress = step_in_phase / max(1, self.steps_per_phase)

        peak_scale = self.phase_decay ** phase_idx

        if progress < self.lr_ramp_ratio:
            # LR ramp-up
            ramp_progress = progress / self.lr_ramp_ratio
            scale = ramp_progress * peak_scale
        else:
            # Cosine decay
            decay_progress = (progress - self.lr_ramp_ratio) / (1.0 - self.lr_ramp_ratio)
            import math
            cosine_factor = 0.5 * (1 + math.cos(math.pi * decay_progress))
            scale = peak_scale * cosine_factor

        return [
            self.eta_min + (base_lr - self.eta_min) * scale
            for base_lr in self.base_lrs
        ]


# ─────────────────────────────────────────────
# CBraMod 动态加载（避免 models 包名冲突）
# ─────────────────────────────────────────────

def _load_cbramod_class():
    """通过 importlib 从 CBraMod 仓库加载模型类，避免包名冲突。"""
    transformer_file = CBRAMOD_REPO / "models" / "criss_cross_transformer.py"
    cbramod_file = CBRAMOD_REPO / "models" / "cbramod.py"

    if not cbramod_file.exists():
        raise FileNotFoundError(f"CBraMod 模型文件不存在: {cbramod_file}")

    # 先加载 criss_cross_transformer 依赖
    if transformer_file.exists():
        spec = importlib.util.spec_from_file_location("criss_cross_transformer", transformer_file)
        transformer_module = importlib.util.module_from_spec(spec)
        sys.modules["criss_cross_transformer"] = transformer_module
        sys.modules["models.criss_cross_transformer"] = transformer_module
        spec.loader.exec_module(transformer_module)

    # 加载 cbramod
    spec = importlib.util.spec_from_file_location("cbramod", cbramod_file)
    cbramod_module = importlib.util.module_from_spec(spec)
    sys.modules["cbramod"] = cbramod_module
    sys.modules["models.cbramod"] = cbramod_module
    spec.loader.exec_module(cbramod_module)

    return cbramod_module.CBraMod


# ─────────────────────────────────────────────
# Batch size 自适应缩放
# ─────────────────────────────────────────────


def _batch_size_for_channels(
    n_channels: int, max_batch: int, reference_channels: int = 64
) -> int:
    """根据通道数缩放 batch size 以适配 GPU 显存。

    VRAM 占用与通道数近似成正比（CBraMod attention 沿通道维度计算）。
    以 reference_channels @ max_batch 为基准线性缩放，向下取 2 的幂。
    """
    scaled = max_batch * reference_channels / n_channels
    for bs in sorted({max_batch, max_batch // 2, max_batch // 4, max_batch // 8, 2},
                     reverse=True):
        if bs >= 2 and scaled >= bs:
            return bs
    return 2


# ─────────────────────────────────────────────
# 训练器
# ─────────────────────────────────────────────


class FurtherPretrainTrainer:
    """CBraMod domain-adaptive further pre-training 训练器。"""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("GPU 必须可用。请检查 CUDA 安装。")

        logger.info(f"设备: {self.device} ({torch.cuda.get_device_name()})")
        logger.info(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # 初始化模型
        self._init_model()

        # 初始化数据
        self._init_data()

        # 初始化优化器
        self._init_optimizer()

        # 混合精度
        self.scaler = torch.amp.GradScaler("cuda") if config["amp_enabled"] else None

        # Checkpoint 目录
        if config["checkpoint_dir"] is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            self.checkpoint_dir = (
                PROJECT_ROOT / "checkpoints" / "cbramod" / f"further_pretrain_{timestamp}"
            )
        else:
            self.checkpoint_dir = Path(config["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoint 目录: {self.checkpoint_dir}")

        # 保存配置
        with open(self.checkpoint_dir / "config.json", "w") as f:
            # 序列化 config（处理 Path 等不可序列化类型）
            serializable = {}
            for k, v in config.items():
                if isinstance(v, Path):
                    serializable[k] = str(v)
                elif isinstance(v, list) and v and isinstance(v[0], Path):
                    serializable[k] = [str(p) for p in v]
                else:
                    serializable[k] = v
            json.dump(serializable, f, indent=2, ensure_ascii=False)

    def _init_model(self):
        """加载 CBraMod 模型 + 预训练权重。"""
        CBraMod = _load_cbramod_class()

        self.model = CBraMod(
            in_dim=200,
            out_dim=200,
            d_model=self.config["d_model"],
            dim_feedforward=self.config["dim_feedforward"],
            seq_len=30,
            n_layer=self.config["n_layer"],
            nhead=self.config["nhead"],
        ).to(self.device)

        # 加载预训练权重
        weights_path = self.config["pretrained_weights"]
        candidates = [
            PROJECT_ROOT / "checkpoints" / "cbramod" / "pretrained_weights.pth",
            CBRAMOD_REPO / "pretrained_weights" / "pretrained_weights.pth",
        ]
        if weights_path is None:
            for p in candidates:
                if p.exists():
                    weights_path = str(p)
                    break

        if weights_path and Path(weights_path).exists():
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict, strict=True)
            logger.info(f"已加载预训练权重: {weights_path}")
        else:
            raise FileNotFoundError(
                f"找不到预训练权重。搜索路径: {[str(c) for c in candidates]}"
            )

        # 统计参数量
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"模型参数量: {n_params / 1e6:.2f}M")

    def _init_data(self):
        """初始化多数据集 DataLoader（batch size 按通道数自适应缩放）。"""
        lmdb_dirs = self.config.get("lmdb_dirs")
        if lmdb_dirs is None:
            # 自动扫描 LMDB 目录
            lmdb_base = Path(self.config.get("lmdb_base_dir", DEFAULT_LMDB_DIR))
            if not lmdb_base.exists():
                raise FileNotFoundError(f"LMDB 目录不存在: {lmdb_base}")
            lmdb_dirs = sorted(
                [d for d in lmdb_base.iterdir() if d.is_dir() and (d / "data.mdb").exists()]
            )
            if not lmdb_dirs:
                raise FileNotFoundError(f"未找到 LMDB 数据库: {lmdb_base}")

        max_batch = self.config["batch_size"]
        ref_channels = self.config["reference_channels"]
        self.effective_batch_size = self.config["effective_batch_size"]

        dataloaders = {}
        total_samples = 0

        for lmdb_path in lmdb_dirs:
            lmdb_path = Path(lmdb_path)
            if not (lmdb_path / "data.mdb").exists():
                logger.warning(f"跳过无效 LMDB: {lmdb_path}")
                continue

            dataset = LMDBPretrainingDataset(lmdb_path)
            bs = _batch_size_for_channels(dataset.n_channels, max_batch, ref_channels)
            dl = DataLoader(
                dataset,
                batch_size=bs,
                shuffle=True,
                num_workers=self.config["num_workers"],
                pin_memory=self.config["pin_memory"],
                drop_last=True,
                persistent_workers=self.config["num_workers"] > 0,
            )
            name = lmdb_path.name
            dataloaders[name] = dl
            total_samples += len(dataset)
            logger.info(f"  {name}: {len(dataset)} 样本, {dataset.n_channels} 通道, batch={bs}")

        self.sampler = MultiDatasetSampler(dataloaders)
        self.total_samples = total_samples
        logger.info(f"总样本数: {total_samples}")
        logger.info(f"有效 batch: {self.effective_batch_size} (按样本数累积)")

    def _init_optimizer(self):
        """初始化优化器 + 学习率调度器。"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
        )

        # 计算总步数（基于样本数累积）
        self.steps_per_epoch = self.total_samples // self.effective_batch_size
        max_steps = self.steps_per_epoch * self.config["max_epochs"]
        warmup_steps = int(self.steps_per_epoch * self.config["warmup_epochs"])

        scheduler_type = self.config.get("scheduler", "warmup_constant")
        if scheduler_type == "phased_cosine":
            self.scheduler = PhasedCosineWarmupDecayScheduler(
                self.optimizer,
                total_steps=max_steps,
                num_phases=self.config.get("num_phases", 5),
                phase_decay=self.config.get("phase_decay", 0.5),
                lr_ramp_ratio=self.config.get("lr_ramp_ratio", 0.1),
                eta_min=self.config.get("eta_min", 1e-6),
            )
            sched_desc = (
                f"PhasedCosine({self.config.get('num_phases', 5)} phases, "
                f"decay={self.config.get('phase_decay', 0.5)}, "
                f"ramp={self.config.get('lr_ramp_ratio', 0.1)})"
            )
        else:
            self.scheduler = WarmupConstantScheduler(
                self.optimizer,
                warmup_steps=warmup_steps,
            )
            sched_desc = f"Warmup({warmup_steps} steps) + Constant LR"

        logger.info(
            f"优化器: AdamW (lr={self.config['lr']}, wd={self.config['weight_decay']})"
        )
        logger.info(
            f"调度器: {sched_desc}"
        )
        logger.info(
            f"每 epoch {self.steps_per_epoch} 优化步, 最大 {self.config['max_epochs']} epochs, "
            f"patience={self.config['patience']}, min_delta={self.config['min_delta']}"
        )

    def train(self):
        """执行 further pre-training（early stopping）。"""
        criterion = nn.MSELoss(reduction="mean")
        best_loss = float("inf")
        eff_bs = self.effective_batch_size
        patience = self.config["patience"]
        min_delta = self.config["min_delta"]
        max_epochs = self.config["max_epochs"]
        epochs_without_improvement = 0
        history = []

        logger.info("=" * 60)
        logger.info("开始 further pre-training")
        logger.info(f"Early stopping: patience={patience}, min_delta={min_delta}, max_epochs={max_epochs}")
        logger.info("=" * 60)

        for epoch in range(max_epochs):
            self.model.train()
            self.sampler.reset()
            logger.info(f"Epoch {epoch+1} 开始...")

            epoch_losses = []
            accum_loss = 0.0
            accum_samples = 0
            accum_batches = 0
            step_count = 0

            t_start = time.time()

            for batch_idx in range(self.sampler.total_batches):
                # 从随机数据集取一个 batch
                ds_name, x = self.sampler.sample_batch()
                if batch_idx == 0:
                    logger.info(f"  首批: {ds_name} shape={x.shape}")
                x = x.to(self.device, non_blocking=True) / 100.0  # 归一化

                bz, ch_num, patch_num, patch_size = x.shape

                # 生成掩码
                mask = generate_mask(
                    bz, ch_num, patch_num,
                    mask_ratio=self.config["mask_ratio"],
                    device=self.device,
                )

                # loss 按 batch_size / effective_batch_size 缩放，
                # 使每个样本对梯度的贡献一致
                scale = bz / eff_bs

                # 前向
                if self.scaler is not None:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        y = self.model(x, mask=mask)
                        masked_x = x[mask == 1]
                        masked_y = y[mask == 1]
                        loss = criterion(masked_y, masked_x)
                        loss_scaled = loss * scale

                    self.scaler.scale(loss_scaled).backward()
                else:
                    y = self.model(x, mask=mask)
                    masked_x = x[mask == 1]
                    masked_y = y[mask == 1]
                    loss = criterion(masked_y, masked_x)
                    loss_scaled = loss * scale

                    loss_scaled.backward()

                accum_loss += loss.item()  # 未缩放的 loss 用于监控
                accum_samples += bz
                accum_batches += 1

                # 样本数累积达标 → optimizer step
                if accum_samples >= eff_bs:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config["clip_value"]
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config["clip_value"]
                        )
                        self.optimizer.step()

                    self.optimizer.zero_grad()
                    self.scheduler.step()

                    avg_loss = accum_loss / accum_batches
                    epoch_losses.append(avg_loss)
                    accum_loss = 0.0
                    accum_samples = 0
                    accum_batches = 0
                    step_count += 1

                    # 定期打印（前 5 步每步打印，之后每 50 步）
                    if step_count <= 5 or step_count % 50 == 0:
                        lr = self.optimizer.param_groups[0]["lr"]
                        logger.info(
                            f"  Epoch {epoch+1} Step {step_count}/{self.steps_per_epoch} "
                            f"Loss: {avg_loss:.6f} LR: {lr:.2e}"
                        )

            # Epoch 结束
            elapsed = time.time() - t_start
            mean_loss = np.mean(epoch_losses) if epoch_losses else float("inf")
            lr = self.optimizer.param_groups[0]["lr"]

            logger.info(
                f"Epoch {epoch+1}: "
                f"Loss={mean_loss:.6f}, LR={lr:.2e}, "
                f"Time={elapsed:.0f}s, "
                f"patience={epochs_without_improvement}/{patience}"
            )

            history.append({
                "epoch": epoch + 1,
                "loss": float(mean_loss),
                "lr": lr,
                "time_seconds": round(elapsed, 1),
            })

            # Early stopping 检查
            if mean_loss < best_loss - min_delta:
                best_loss = mean_loss
                epochs_without_improvement = 0
                best_path = self.checkpoint_dir / "best_model.pth"
                torch.save(self.model.state_dict(), best_path)
                logger.info(f"  ★ 最优模型已保存: {best_path} (loss={best_loss:.6f})")
            else:
                epochs_without_improvement += 1
                logger.info(
                    f"  无显著改善 ({epochs_without_improvement}/{patience}), "
                    f"best={best_loss:.6f}"
                )

            # 每 epoch 保存
            if self.config["save_every_epoch"]:
                epoch_path = self.checkpoint_dir / f"epoch{epoch+1}_loss{mean_loss:.6f}.pth"
                torch.save(self.model.state_dict(), epoch_path)

            # 触发 early stopping
            if epochs_without_improvement >= patience:
                logger.info(
                    f"Early stopping: {patience} epochs 无改善 (best={best_loss:.6f})"
                )
                break

        # 训练完成
        logger.info("=" * 60)
        logger.info(f"训练完成。最优 loss: {best_loss:.6f}")
        logger.info(f"权重保存于: {self.checkpoint_dir}")
        logger.info("=" * 60)

        # 保存训练历史
        with open(self.checkpoint_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        return best_loss, self.checkpoint_dir


def main():
    parser = argparse.ArgumentParser(
        description="CBraMod Domain-Adaptive Further Pre-training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 训练参数
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_CONFIG["max_epochs"])
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG["patience"],
                        help="连续 N epoch 无改善则停")
    parser.add_argument("--min-delta", type=float, default=DEFAULT_CONFIG["min_delta"],
                        help="小于此的 loss 改善视为噪声")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"],
                        help="物理 batch 上限 (按通道数自动缩放)")
    parser.add_argument("--effective-batch-size", type=int, default=DEFAULT_CONFIG["effective_batch_size"],
                        help="目标有效 batch (样本数累积触发 optimizer step)")
    parser.add_argument("--reference-channels", type=int, default=DEFAULT_CONFIG["reference_channels"],
                        help="batch_size 对应的参考通道数")
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--warmup-epochs", type=float, default=DEFAULT_CONFIG["warmup_epochs"])
    parser.add_argument("--clip-value", type=float, default=DEFAULT_CONFIG["clip_value"])
    parser.add_argument("--mask-ratio", type=float, default=DEFAULT_CONFIG["mask_ratio"])

    # AMP
    parser.add_argument("--no-amp", action="store_true", help="禁用混合精度")

    # 数据
    parser.add_argument(
        "--lmdb-dirs", nargs="+", type=str, default=None,
        help="LMDB 数据库路径列表（默认自动扫描）",
    )
    parser.add_argument(
        "--lmdb-base-dir", type=str, default=str(DEFAULT_LMDB_DIR),
        help="LMDB 基目录（自动扫描时使用）",
    )
    parser.add_argument("--num-workers", type=int, default=DEFAULT_CONFIG["num_workers"])

    # 权重
    parser.add_argument(
        "--pretrained-weights", type=str, default=None,
        help="预训练权重路径（默认自动搜索）",
    )

    # 保存
    parser.add_argument("--checkpoint-dir", type=str, default=None)

    args = parser.parse_args()

    config = {**DEFAULT_CONFIG}
    config["max_epochs"] = args.max_epochs
    config["patience"] = args.patience
    config["min_delta"] = args.min_delta
    config["batch_size"] = args.batch_size
    config["effective_batch_size"] = args.effective_batch_size
    config["reference_channels"] = args.reference_channels
    config["lr"] = args.lr
    config["weight_decay"] = args.weight_decay
    config["warmup_epochs"] = args.warmup_epochs
    config["clip_value"] = args.clip_value
    config["mask_ratio"] = args.mask_ratio
    config["amp_enabled"] = not args.no_amp
    config["num_workers"] = args.num_workers
    config["pretrained_weights"] = args.pretrained_weights
    config["checkpoint_dir"] = args.checkpoint_dir
    config["lmdb_base_dir"] = args.lmdb_base_dir

    if args.lmdb_dirs:
        config["lmdb_dirs"] = [Path(d) for d in args.lmdb_dirs]

    logger.info(f"物理 batch 上限: {config['batch_size']}, 有效 batch: {config['effective_batch_size']}")

    trainer = FurtherPretrainTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
