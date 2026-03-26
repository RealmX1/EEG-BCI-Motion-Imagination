# Cross-Subject Extra Sessions 训练性能分析

> **Run tag**: `20260326_1409`
> **配置**: 21 subjects 训练, 16 eval subjects, binary, imagery, 128 通道
> **GPU**: NVIDIA GeForce RTX 5070 (Blackwell sm_120, 12.8 GB VRAM)
> **数据来源**: `results/20260326_1409_cross_subject_extra_sessions_cache_imagery_binary.json`

## 1. 训练时间总览

Steps 1-7 使用优化前代码（FP16 + 双前向传播），Step 8 使用优化后代码（BF16 + 单次前向传播）。

| Model | Step | Segments | Epochs (total/best) | Epoch Time | Total Time | Mean Acc (16 eval) | 代码版本 |
|-------|------|----------|---------------------|------------|------------|-------------------|----------|
| EEGNet | baseline | 332,866 | 19 / 9 | ~1m 25s | 29m 51s | 81.45% | 优化前 |
| EEGNet | +Sess03 | 416,370 | 28 / 18 | ~1m 44s | 51m 44s | 81.84% | 优化前 |
| EEGNet | +Sess04 | 490,738 | 41 / 31 | ~2m 05s | 90m 57s | 82.54% | 优化前 |
| EEGNet | +Sess05 | 564,713 | 45 / 35 | ~2m 24s | 112m 49s | 81.33% | 优化前 |
| CBraMod | baseline | 93,243 | 50 / 30 | ~1m 01s | 51m 15s | 92.38% | 优化前 |
| CBraMod | +Sess03 | 115,733 | 38 / 18 | ~1m 17s | 49m 30s | 91.87% | 优化前 |
| CBraMod | +Sess04 | 135,742 | 40 / 20 | ~1m 26s | 58m 38s | 92.19% | 优化前 |
| CBraMod | +Sess05 | 155,656 | 39 / 19 | **~1m 12s** | **48m 03s** | **93.24%** | **优化后** |

**总计**: ~8h 53m (EEGNet 5h 25m + CBraMod 3h 28m)

---

## 2. 性能优化实测对比

### 2.1 优化内容 (2026-03-26)

| 编号 | 优化项 | 修改文件 |
|------|--------|---------|
| P0.1 | 合并 validate() + majority_vote() 为单次前向传播 | `trainer.py`, `evaluation.py` |
| P0.2 | AMP 从 FP16+GradScaler 切换为 BF16（无 GradScaler） | `trainer.py`, `evaluation.py` |
| P0.3 | 修复 `set_seed(deterministic=True)` 未被重置的冲突 | `common.py` |
| P1.4 | 验证 batch size 增大至 `min(512, 4*batch_size)` | `train_within_subject.py` |
| P1.6 | 预计算 trial-to-segment 分组映射（每次训练仅构建一次） | `trainer.py`, `evaluation.py` |

### 2.2 对比基准推导

CBraMod 在优化前代码的 epoch time 与 segments 呈线性关系：

```
epoch_time = 0.000591 × segments + 6.78  (R² from 3 data points)
```

| Step | Segments | 实测 epoch time (优化前) | 线性拟合 |
|------|----------|------------------------|---------|
| baseline | 93,243 | 61s | 61.9s |
| +Sess03 | 115,733 | 77s | 75.2s |
| +Sess04 | 135,742 | 86s | 87.0s |
| **+Sess05** | **155,656** | **(未实测)** | **98.7s (外推)** |

### 2.3 实测结果

CBraMod +Sess05, 155,656 segments, batch_size=256:

| 指标 | 旧代码外推 | 新代码实测 | 变化 |
|------|-----------|-----------|------|
| **稳态 epoch time** | ~99s | **72.3s** | **-27.0%** |
| **ms/batch** | ~203 ms | **~119 ms** | **-41.4%** |
| **训练总时间** (39 epochs) | ~64 min | **48m 03s** | **-25.0%** |

**关键确认**:
- `AMP enabled (BF16 — no GradScaler)` — BF16 生效
- `torch.use_deterministic_algorithms(False)` — 确定性约束已移除
- 验证阶段仅执行一次前向传播 — majority vote 使用已有 predictions 的 CPU 分组

### 2.4 Epoch-0 Profile 对比 (CBraMod +Sess05)

| 组件 | 优化后 | 占比 | 备注 |
|------|--------|------|------|
| data loading | 3.42s | 4% | 不变，in-memory 数据 |
| GPU transfer | 0.43s | 1% | non_blocking=True |
| forward | 33.76s | 39% | BF16 tensor cores |
| backward | 43.85s | 51% | 主要瓶颈 |
| optimizer | 5.23s | 6% | AdamW step |

### 2.5 Epoch Time 收敛曲线

```
Epoch  1: 99.5s  (exploration batch=128, + profiling + JIT warmup)
Epoch  2: 90.6s  (exploration batch=128)
Epoch  3: 80.6s  (exploration batch=128)
Epoch  4: 77.6s  ← 切换到 main batch=256
Epoch  5: 74.0s
Epoch  6: 73.2s
Epoch  7: 73.9s
Epoch  8: 72.3s  ← 稳态
...
Epoch 39: 72.1s  (early stopping)
```

Exploration phase (epochs 1-3) 使用 batch_size=128，batches/epoch 更多因此更慢。Epoch 4 切换到 batch_size=256 后迅速稳定在 ~72s。

---

## 3. 数据量与 Epoch 时间关系

### 3.1 EEGNet (batch_size=128, 优化前)

| Step | Segments | Batches/epoch | Epoch Time | 瓶颈 |
|------|----------|--------------|------------|------|
| baseline | 332,866 | ~2,600 | 1m 25s | backward (68%) |
| +Sess03 | 416,370 | ~3,253 | 1m 44s | backward (58%) |
| +Sess04 | 490,738 | ~3,834 | 2m 05s | backward (61%) |
| +Sess05 | 564,713 | ~4,412 | 2m 24s | backward (55%), data loading (23%) |

每增加 10 万 segments，epoch 时间约增加 15-20s。+Sess05 步的 data loading 占比升至 23%（52s/epoch），成为次要瓶颈。

### 3.2 CBraMod (batch_size=256)

| Step | Segments | Batches/epoch | Epoch Time | 瓶颈 | 代码版本 |
|------|----------|--------------|------------|------|----------|
| baseline | 93,243 | ~364 | 1m 01s | backward (48%), forward (39%) | 优化前 |
| +Sess03 | 115,733 | ~452 | 1m 17s | backward (51%), forward (36%) | 优化前 |
| +Sess04 | 135,742 | ~530 | 1m 26s | backward (52%), forward (35%) | 优化前 |
| +Sess05 | 155,656 | ~487* | **1m 12s** | backward (51%), forward (39%) | **优化后** |

*batch_size=256 的 80% 训练集 batches。

CBraMod 的 data loading 占比始终极低 (3-4%)，瓶颈在 forward+backward (87-90%)。这是因为 CBraMod 有 ~4M 参数（vs EEGNet ~2.5K），模型计算本身是主要开销。

---

## 4. 关键观察

1. **优化效果显著 (-27% epoch time)**: 数据量增加 14.7% 的情况下，epoch time 反而从 86s 降到 72s。三项 P0 优化（单次前向传播、BF16、deterministic fix）叠加效果超出单项预估。

2. **BF16 在 Blackwell 上表现良好**: RTX 5070 (sm_120) 的 BF16 tensor cores 原生支持，消除了 FP16 GradScaler 的 inf/nan 检查和 scale/unscale 开销。训练稳定性未受影响（39 epochs 无异常）。

3. **EEGNet 的 data loading 瓶颈**: 随着数据量增大，EEGNet 的 data loading 从 8% (baseline) 升至 23% (+Sess05)。可能的优化方向：增加 DataLoader workers、预取、或更大 batch size。CBraMod 不受此影响。

4. **EEGNet 收敛更慢**: EEGNet 在更大数据上需要更多 epochs（baseline 19 vs +Sess05 45），可能因为模型容量小（2.5K 参数），需要更多 passes 才能拟合。

5. **CBraMod +Sess05 准确率最高**: 93.24% mean test accuracy，比 baseline 92.38% 提升 +0.86%。额外 session 数据持续带来小幅提升。

---

## 5. 对比: 16-subject-only 运行 (run_tag=20260326_0345)

前一次运行仅用 16 个 extra-session subjects 训练（不含 S01/S05/S12/S20/S21）：

| Model | Step | 16-subj Segments | 21-subj Segments | 增幅 |
|-------|------|-----------------|-----------------|------|
| EEGNet | baseline | 252,858 | 332,866 | +31.6% |
| CBraMod | baseline | 71,738* | 93,243 | +30.0% |

*CBraMod 使用 200Hz 采样率（vs EEGNet 100Hz），segment 数量因 preprocessing 不同而不同。

21 subjects 的数据量增加约 30%，导致 epoch 时间对应增加。但模型质量略有提升（CBraMod baseline: 91.37% → 92.38%）。

---

## 6. 恢复命令（备忘）

实验已全部完成（8/8 步），无需恢复。如需重新运行 +Sess05 步骤：

```bash
uv run python scripts/experiments/run_cross_subject_extra_sessions.py --cache-only --no-wandb --resume 20260326_1409
```
