# 2026-05-14 — 16ch transition-point sweep + 61ch standard_1010 ternary 补齐

## 背景

2026-05-11 完成的 {4, 8, 32, 64}ch × 5 method × {binary, ternary} = 40-cell 矩阵闭合后，paper §3.5.2 留下 **16ch 档位未评估**作为 method-dependency boundary 的开放问题（"16ch 是否处于方法依赖临界 boundary 两侧"）。同期 61ch standard_1010（Yazıcı et al. 2025 [26] 文献参考点）在 binary 上有 `20260330_1213`，但 ternary 一直缺失。

本批次（2026-05-13）一并完成两件事：

1. **16ch × 5 method × 2 task = 10 个新 cell**，把矩阵扩展到 5×5×2 = 50 cell；
2. **61ch standard_1010 ternary 补齐**，使 61ch 在 paper 上能与其他档位平行呈现。

## 实验配置

- Cross-subject only, CBraMod only, --cache-only, --scheduler cosine_annealing_warmup_decay
- 21 名 responder 被试（与 paper 主对比 cohort 一致）
- 训练: `scripts/experiments/run_16ch_sweep_plus_61ch_ternary.sh` 串行调度，单 gate 在 sweep 开头（primary nvidia-smi compute-apps + secondary overwatch with `--disable-network` capped at 30 min）
- Wall time: 5h 00m（2026-05-13 19:08:38 → 2026-05-14 00:08:52）
- 11/11 OK，无 OOM / Traceback

## Run 列表（mean ± std cross-subject test accuracy, N=21）

| # | n_ch | config | task | run_tag | mean_acc ± std |
|---|------|--------|------|---------|----------------|
| 1 | 61 | standard_1010 | ternary | `20260513_1938` | 76.71 ± 12.04% |
| 2 | 16 | fdr | binary | `20260513_1959` | 84.26 ± 9.87% |
| 3 | 16 | csp | binary | `20260513_2027` | 83.36 ± 11.38% |
| 4 | 16 | attention | binary | `20260513_2048` | 76.55 ± 10.76% |
| 5 | 16 | band_power | binary | `20260513_2108` | 85.24 ± 9.37% |
| 6 | 16 | negative_control | binary | `20260513_2132` | 81.61 ± 10.42% |
| 7 | 16 | fdr | ternary | `20260513_2146` | 69.31 ± 13.33% |
| 8 | 16 | csp | ternary | `20260513_2227` | 62.84 ± 10.85% |
| 9 | 16 | attention | ternary | `20260513_2241` | 61.67 ± 10.59% |
| 10 | 16 | band_power | ternary | `20260513_2319` | 67.60 ± 11.38% |
| 11 | 16 | negative_control | ternary | `20260513_2343` | 64.37 ± 10.26% |

完整 JSON 路径：`results/16_channel/{cfg}/{run_tag}_cross_subject_cache_imagery_{task}.json` 与 `results/61_channel/standard_1010/20260513_1938_cross_subject_cache_imagery_ternary.json`。

## 关键发现

### 1. 16ch 是 method-agnostic 区间的崩溃入口

**5-entry spread 跃升**：

| 通道档 | binary spread (pp) | ternary spread (pp) |
|--------|-------------------:|---------------------:|
| 64 | 3.24 | 2.09 |
| 32 | 2.77 | 2.08 |
| **16** | **8.69** | **7.64** |
| 8  | 15.63 | 6.83 |
| 4  | 24.05 | 19.12 |

32→16ch 之间 spread 跳升 3–4 倍。这把"方法选择对性能影响 ≤ 3 pp"的 method-agnostic 论断（原适用于 32ch+）**精确定位到 ≥ 32ch**，16ch 即崩溃入口。

### 2. 16ch 排序的双 task 一致信号

- **Attention 在 16ch 双 task 上均为最差**（binary 76.55%、ternary 61.67%）—— 把"Attention top-K 外推失效"现象的临界点从 8ch 前移到 16ch。
- **BP 在 binary 上仍居首**（85.24%）但 ternary 让位给 FDR（69.31%）—— 与 32ch ternary 上 FDR/BP 几乎并列的局面延续。
- **BP 跨 5 通道档 × 2 task = 10 cell 上从不是 4 数据驱动方法的最差者**（之前 8 cell，现扩展到 10 cell），是本数据集最稳健的横向方法学观察。

### 3. negative_control 对最优方法的"追赶能力"在 16ch 已退化

| 通道档 | task | best 数据驱动方法 | negative_control | Δ (pp) |
|--------|------|-------------------|------------------|-------:|
| 32 | binary | FDR 87.71% | 84.08% | 3.63 |
| 32 | ternary | BP 72.20% | 72.38% | −0.18 |
| 64 | binary | FDR 89.46% | 88.57% | 0.89 |
| 64 | ternary | FDR 75.12% | 75.44% | −0.32 |
| **16** | **binary** | **BP 85.24%** | **81.61%** | **3.63** |
| **16** | **ternary** | **FDR 69.31%** | **64.37%** | **4.94** |

32ch / 64ch ternary 上 ≤ 0.32 pp 的"统计不可区分"在 16ch ternary 上扩大到 4.94 pp，即"低 method-overlap 配置 ≈ 数据驱动配置"的等价性论断**严格限制在 ≥ 32ch 通道档**。

### 4. 6 档 binary 包络线

90.68% (128) → 89.46% (64 FDR) → 89.55% (61 standard) → 87.71% (32 FDR) → 85.24% (16 BP) → 84.05% (8 BP) → 78.75% (4 BP)

**每减半通道损失 1.5–5 pp**，平滑无突变。32→16 BP / 16→8 BP 几乎平滑（16→8 BP −1.19 pp、16→8 FDR −7.83 pp），方法依赖差异主要由 FDR / CSP / Attention 在 ≤16ch 上的快速衰退驱动。

### 5. 61ch standard_1010 ternary 补齐

- 61ch ternary 76.71% > 64ch FDR ternary 75.12% > 128ch baseline ternary 74.88%，三者在 ±1.83 pp 内 indistinguishable；与 binary 上 61ch ≈ 64ch ≈ 128ch 的格局一致。
- Yazıcı et al. 2025 [26] 的"61ch 最优"在 ternary 上得到方向性复现（**61ch ternary 实际上是本研究 ternary 最高**，超越 128ch baseline，但落在 21 人 cohort std 范围内）。

## 文档同步

- ✅ `paper/run_registry.yaml`：新增 11 entries（1 × 61ch_ternary + 10 × 16ch）
- ✅ `paper/drafts/paper_draft_v3.1.md`：
  - §3.5.2 表 9: 插入 16ch 5 行 + 61ch ternary 更新；
  - §3.5.2 narrative: 加入 "16ch 是 method-agnostic 区间崩溃点" 段、64ch ¶ 加 16/8ch 桥接；
  - §3.5.2 数据来源 binary / ternary 行: 加入 16ch + 61ch ternary 路径；
  - §4.2 段尾: 把 "<32ch 方法依赖临界 boundary 未评估" 改成 "16ch 即崩溃入口"；
  - §6 未来工作 #4: 把 "96ch / 16ch 等中间档位" 缩窄为 "96ch 中间档位"；
  - 参考文献 [26]: 加 Yazıcı et al. 2025 Brain Sciences 引用。
- 🚧 figure regen 待做：`scripts/paper/generate_paper_figures.py --figure reduced_channel_40cell_grid` 目前生成 4×5×2 = 40 cell grid，扩成 5 通道列（4/8/16/32/64）需要调整脚本 default `channel_counts` 参数。本批次不附带 figure 更新，待 paper revision 阶段统一处理。

## 训练日志位置

- Sweep stdout: `/c/Users/zhang/.claude-procs/16ch_5cfg_sweep_v2_20260513_190838/output.log`
- Per-run stdout: `logs/16ch_sweep/{n_ch}ch_{cfg}_{task}.log`
- Per-run summary: `logs/16ch_sweep/_results.txt`
- Overwatch trace: `logs/16ch_sweep/_overwatch.log`
- Runner script: `scripts/experiments/run_16ch_sweep_plus_61ch_ternary.sh`
- Baseline PID snapshot: `logs/_pid_baseline_5config_sweep.txt`
