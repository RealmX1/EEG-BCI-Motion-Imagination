# Random-init CBraMod Ablation 交接 (2026-05-09)

## 目的

回应 paper draft v3 [Limitation #7](../../paper/drafts/paper_draft_v3.md#L875)（"仅测试了一个基座模型（CBraMod）和一种预训练目标（masked autoencoding）"）。新增 6 个 from-scratch CBraMod 运行（3 paradigm × 2 task），与历史 original-weights CBraMod baseline 和 EEGNet baseline 形成三方对比。

## 实验设置（与 baseline 唯一差异 = backbone init）

| 项 | 值 |
|---|---|
| 模型 | CBraMod 仅一个 |
| 通道数 | 128（与 baseline 一致）|
| 被试数 | n=21 |
| HP | `get_default_config()`（**无** `--config` flag — 与历史 original-weights baseline 完全相同）|
| Backbone init | PyTorch 默认随机初始化（`--no-pretrained`）|
| Transfer 第 2 阶段 | 用本次产出的 from-scratch cross-subject ckpt 作为起点（不混入原始 CBraMod 权重） |
| Wrapper | [scripts/experiments/run_random_init_ablation.sh](../../scripts/experiments/run_random_init_ablation.sh) |
| Wrapper 总时长 | 2h 13m（2026-05-08 23:38:42 → 2026-05-09 01:51:35）|

`--no-pretrained` 路由：[src/cli/experiment_utils.py:283](../../src/cli/experiment_utils.py#L283) → [src/models/cbramod_adapter.py:386-389](../../src/models/cbramod_adapter.py#L386-L389)。每个 run 的 stdout 包含 `WARN: No pretrained weights provided! CBraMod will be trained from scratch.`，wrapper 全程共触发 128 次（1+1+21+21+42+42，对应 3 paradigm × 2 task 的预期次数）。

## 三方对比表（n=21, n_channels=128）

| Paradigm × Task | random-init CBraMod | original-weights CBraMod | EEGNet |
|---|---|---|---|
| within / binary | **62.05% ± 17.68%** | 85.15% ± 11.00% | 78.10% ± 12.61% |
| within / ternary | **38.65% ± 14.07%** | 69.44% ± 15.42% | 66.81% ± 14.50% |
| cross / binary | **86.34% ± 9.41%** | 90.68% ± 9.31% | 76.67% ± 11.95% |
| cross / ternary | **73.06% ± 12.49%** | 74.88% ± 14.03% | 61.23% ± 11.28% |
| transfer / binary | **86.22% ± 9.46%** | 90.12% ± 8.98% | 82.05% ± 11.00% † |
| transfer / ternary | **73.43% ± 12.91%** | 75.04% ± 13.97% | 66.33% ± 12.65% † |

† EEGNet transfer 无 `is_baseline=1` 行；引用最近的 n=21 transfer 运行 `20260507_1835` / `20260507_1913` 作为参考。

### Δ（random-init − original-weights CBraMod）

| Cell | Δ pp |
|---|---|
| within binary | −23.10 |
| within ternary | −30.79 |
| cross binary | −4.34 |
| cross ternary | −1.82 |
| transfer binary | −3.90 |
| transfer ternary | −1.61 |

## 数据来源（run_tags）

### Random-init runs（本次新增）

| run_tag | paradigm | task | command 含 `--no-pretrained` |
|---|---|---|---|
| `20260508_2338` | cross_subject | binary | ✓ |
| `20260509_0014` | cross_subject | ternary | ✓ |
| `20260509_0047` | within_subject | binary | ✓ |
| `20260509_0102` | within_subject | ternary | ✓ |
| `20260509_0124` | transfer | binary | ✓（`--pretrained-cbramod` = `20260508_2338` 产出的 ckpt）|
| `20260509_0135` | transfer | ternary | ✓（`--pretrained-cbramod` = `20260509_0014` 产出的 ckpt）|

JSON cache（`results/`）：
- `20260508_2338_cross_subject_cache_imagery_binary.json`
- `20260509_0014_cross_subject_cache_imagery_ternary.json`
- `20260509_0047_within_subject_cache_imagery_binary.json`
- `20260509_0102_within_subject_cache_imagery_ternary.json`
- `20260509_0124_transfer_cache_imagery_binary.json`
- `20260509_0135_transfer_cache_imagery_ternary.json`

Cross-subject checkpoints（被 transfer 复用）：
- `checkpoints/cross_subject/20260508_2338_cbramod_imagery_binary/best.pt`
- `checkpoints/cross_subject/20260509_0014_cbramod_imagery_ternary/best.pt`

ExperimentDB 验证：`SELECT run_tag,experiment_type,task,n_subjects FROM runs WHERE command LIKE '%--no-pretrained%'` → 6 行，全 21 被试。

### 历史 baseline runs

| run_tag | paradigm | task | model | n_channels |
|---|---|---|---|---|
| `20260323_2237` | within_subject | binary | cbramod | 128 |
| `20260323_2320` | within_subject | ternary | cbramod | 128 |
| `20260324_0023` | cross_subject | binary | cbramod | 128 |
| `20260324_0109` | cross_subject | ternary | cbramod | 128 |
| `20260329_0507` | transfer | binary | cbramod | 128 |
| `20260329_0521` | transfer | ternary | cbramod | 128 |
| `20260316_1411` | within_subject | binary | eegnet | 128 |
| `20260329_0056` | within_subject | ternary | eegnet | 128 |
| `20260330_0709` | cross_subject | binary | eegnet | 128 |
| `20260330_0735` | cross_subject | ternary | eegnet | 128 |
| `20260507_1835` | transfer | binary | eegnet | 128 (无 is_baseline 标记) |
| `20260507_1913` | transfer | ternary | eegnet | 128 (无 is_baseline 标记) |

CBraMod baseline 全部 `is_baseline=1`；EEGNet baseline 在 within/cross 标记，transfer 未标记。

## Within / Ternary 单被试细节（random-init `20260509_0102`）

21 个被试中 **18 个** 测试准确率落在 chance 区间（33.33% ± 2pp），仅 3 个高于 chance：

| Subject | test_acc |
|---|---|
| S07 | 61.67% |
| S09 | 59.58% |
| S19 | 90.42% |
| 其余 18 个 | ≈ 33.33% |

来源：`SELECT subject_id, test_acc FROM subject_results WHERE run_id=(SELECT run_id FROM runs WHERE run_tag='20260509_0102') AND model_type='cbramod'`。

其它 paradigm × task cell 的 random-init 单被试明细未在本 handoff 列出；如需，可用同样查询替换 run_tag。

## Within / Ternary 跨 seed 复现性检查（2026-05-09 18:38 追加）

为排除 18/21 chance 塌陷是 seed=42 的运气特例，重跑了一次 within_subject ternary random-init，唯一改动是 `--seed 1234`（其余 HP 与 `20260509_0102` 完全一致）。

| 指标 | seed=42 (`20260509_0102`) | seed=1234 (`20260509_1838`) |
|---|---|---|
| Mean test_acc | 38.65% ± 14.07% | 39.25% ± 13.90% |
| Chance-collapsed 被试数 | 18/21 | 17/21 |
| Above-chance 被试集合 | {S07, S09, S19} | {S09, S13, S14, S19} |
| Max test_acc | 90.42% (S19) | 87.92% (S19) |
| `trained from scratch` warn 计数 | 21 | 21 |
| 运行时长 | 21m 29s | 26m 01s |

**两次 above-chance 被试交集**：{S09, S19}（两个种子下都逃出 chance）。
**仅在某一种子下逃出**：seed=42 的 {S07}；seed=1234 的 {S13, S14}。

> ⚠️ **判据警告**：上面的"above-chance"是按 `test_acc ≥ 0.36` 的宽松口径。下文 *Within / Ternary Loss 轨迹分析* 节用 **`max train_acc > 0.40`** 严格口径重新判定，发现 S09 (seed=1234) 与 S14 (seed=1234) 的训练阶段从未真正逃出 saddle —— 它们的 test_acc 仅是 uniform 预测撞上不平衡 test set 的统计噪声。**严格口径下，跨 seed 唯一稳定可学的被试只有 {S19}**。引用稳健性结论时应以严格口径为准。

新跑命令：

```bash
uv run python scripts/experiments/run_within_subject_comparison.py \
  --models cbramod --task ternary --no-pretrained --seed 1234 \
  --no-wandb --cache-only --force-retrain
```

JSON cache：`results/20260509_1838_within_subject_cache_imagery_ternary.json`。ExperimentDB run_tag：`20260509_1838`。

## Within / Ternary Loss 轨迹分析（LR 假设诊断）

### 数据来源

每被试的逐 epoch `train_loss / train_acc / val_loss / val_acc` 在 `results/<run_tag>_cbramod_within_subject/ternary/<sid>/history.json`。两次跑都有完整记录。

### 关键参考点

三分类 chance loss = ln(3) ≈ 1.0986。模型若**预测均匀分布**，cross-entropy loss 严格等于 ln(K)；偏离这个值意味着模型 logits 偏离均匀。

### 三类轨迹（以 seed=42 为代表）

**A. 完全塌陷型**（如 S01；两 seed 下 ep3 train_loss 几乎 bit-exact 一致 1.122 vs 1.122）

```
ep 1: train_loss=1.269  ← random init 自带 logits 偏移
ep 2: train_loss=1.149
ep 3: train_loss=1.122  ← 已逼近 ln(3)+0.023
ep 4–11: train_loss 在 1.104 ~ 1.111 之间窄幅游走（Δ ≈ ±0.005）
```
全程 train_acc ≈ 0.33；val_acc ≈ 0.32–0.34；patience=10 在 ep11 触发停训。

**B. 早期逃出型**（如 S07 seed=42）

```
ep 1: 1.729  → ep 2: 1.178  → ep 3: 1.059 (跨过 ln 3 阈值)  → ep 4: 0.996 → ... → ep 42: 0.420
```
train_acc 0.34 → 0.89；test_acc 0.617。

**C. 强逃出型**（S19 两 seed 下都是这个形态）

```
ep 1: 1.257 → ep 2: 1.020 → ep 3: 0.885 → ... 训练 27–34 epoch
```
train_acc 0.35 → 0.85+；test_acc 0.88–0.90。

### 用 train dynamics 判定"真实可学"被试（vs test_acc 误判）

`test_acc ≥ 0.36` 不一定意味着模型学到了东西 —— 类别不平衡的 test set 撞上 uniform 预测就能给出 0.4–0.5 的 test_acc。**严格判据是 max train_acc > 0.40**（即训练阶段曾真正逃出 saddle）。

| 判据 | seed=42 | seed=1234 |
|---|---|---|
| test_acc ≥ 0.36（subagent 原报告口径）| {S07, S09, S19} | {S09, S13, S14, S19} |
| **max train_acc > 0.40（严格口径）** | {S07, S09, S19} | {S13, S19} |
| 差异（仅 test_acc 高、train 仍卡 chance）| 无 | **{S09, S14}** |

S09 (seed=1234)：max_train_acc=0.378，train_loss 末段 1.108/1.108/1.106；S14 (seed=1234)：max_train_acc=0.345，train_loss 末段 1.105/1.106/1.105。这两个被试**实际上没有从 saddle 逃出**，test_acc 0.467/0.562 是 uniform 预测对上 test set 类别比例的统计噪声。

按严格口径：seed 之间唯一**稳定可学**的被试只有 **S19**。

### LR-deficiency 假设的诊断

**反对 "LR 太小是主因" 的证据（强）**：

1. **梯度流没冻结**：塌陷被试的 train_loss 每个 epoch 都在变化（±0.001~0.01），优化器在走步。如果 LR 真的太小，应该看到 bit-exact 的 train_loss 序列。
2. **部分被试在当前 HP 配置下仍能学到**：S19 在两 seed 下都达到 train_acc 0.85+，说明塌陷不是 categorical optimization 失败。
3. **Cross_subject (~5× 数据) 完美收敛**（train_acc 0.93+, run_tag `20260508_2338`/`20260509_0014`）。在更大数据规模上，优化器能把 over-parameterized backbone 拟合到接近完美 —— 表明 HP 层面没有结构性瓶颈，瓶颈出现在 within_subject 的小数据 regime。

   > 注：CBraMod 参数计数在跨文档间不一致——[CLAUDE.md](../../CLAUDE.md) 模型表写 `~4.0M`，[2026-05-09_eegnet_huge.md](2026-05-09_eegnet_huge.md) 写 `30,484,202`，runtime 日志报 `~10M`。引用具体参数数前请核对最新模型卡片，本节论证不依赖具体数值，仅依赖 "over-parameterized vs 数据量" 的相对关系。
4. **Saddle 标志**：train_loss → ln(K) 且 train_acc → 1/K 是 softmax+CE 著名的 uniform-logit saddle 指纹，是结构性 attractor，不是欠优化。

**支持 "LR / patience / warmup 是次要因素" 的证据（弱）**：

5. **patience=10 在 cosine cycle 还没跑完前就停训**。S07 seed=42 是 ep3 才逃出（train_loss 跨 ln3 阈值）—— 若某些塌陷被试需要 ep20+ 才能逃，patience 在 ep11 就杀掉了。
6. **`backbone_lr=2.9e-4` 偏 fine-tuning 量级**。From-scratch Transformer 经验值在 1e-3 ~ 3e-3。理论上更大 peak LR 提供更大随机扰动，**可能**踢出 saddle。
7. **LR 在 patience 触发时仅 ~50% peak**。S01 ep11 LR ≈ 1.07e-4，从未充分使用 peak。

### 概率判断

| 主要瓶颈假设 | 概率（基于现有证据）|
|---|---|
| 数据量 / 过参数化导致 saddle-lock（结构性，与 LR 量级关系弱）| **70–80%** |
| LR + patience + warmup 调优可救回 ≥ 5 个塌陷被试 | 15–25% |
| LR 是主因，提高 LR 能让 ≥ 10/18 塌陷被试学到 | < 5% |

核心论证：train_loss → ln(3) 后近乎冻结的形态是 saddle-lock-in 的指纹，不是 under-stepping。Saddle 在 over-parameterized + 小数据下是结构性的，调 LR 通常无法根除。

### 可选后续实验：高 LR + 长 patience retry（**未执行**）

如需在 paper 里把 LR 假设排除/确认到 95% 把握，可以做一次**仅在 within_ternary 单 seed**上的 HP ablation：

```bash
# 仅作为占位示意（实际需新增 YAML 或 CLI override），未跑过
uv run python scripts/experiments/run_within_subject_comparison.py \
  --models cbramod --task ternary --no-pretrained --seed 42 \
  --config configs/<新建_high_lr.yaml> \
  --no-wandb --cache-only --force-retrain
# 关键 HP 改动：
#   backbone_lr peak  : 2.9e-4 → 2.9e-3 (10×)
#   classifier_lr peak: 1.2e-3 → 1.2e-2 (10×) 或保持比例
#   patience          : 10 → 30
#   warmup            : 拉长到能让 LR 在 train_loss 决出胜负前持续在 peak
```

成本：~25 分钟（同 within_ternary 单跑量级，21 被试串行）。

预期判读：
- 塌陷计数仍 ≥ 14/21 → **LR 几乎排除**，主因为数据/saddle 假设升至 90%+
- 塌陷计数 ≤ 8/21 → LR + schedule **是显著贡献**，需重写论文 from-scratch 归因
- 中间值（9–13）→ LR 是次要但非可忽略因素

注：本 handoff **未执行**该实验；以上仅记录方法学路径，留给后续判断是否需要补做。

## 已知/未知 caveats

- **未做 random-init 专属 HPO**：random-init 复用了 original-weights CBraMod 的 HP（即 `get_default_config()`）。两者唯一变量是 backbone init。
- **Transfer 阶段的 init 来源**：是本次新跑出的 from-scratch cross ckpt，不是历史 original-weights cross ckpt — 即 transfer 也是 end-to-end from-scratch。
- **运行时间快于估计**：原估 9–13h，实际 2h 13m。原因是 random-init 训练集快速过拟合（train acc 升至 0.95+ 时 val 已高位震荡），patience=10 触发早停。
- **额外的 stray cache file**：`results/20260509_0016_within_subject_cache_imagery_binary.json`（761 B）非本批 6 个有效 run 的输出（within binary 真正的 cache 是 `20260509_0047_*`，7087 B）。可视为残留 placeholder，不在 ExperimentDB 中。

## 论文撰写时的引用模板

> **数据来源**：random-init CBraMod ablation，run_tag `20260509_0102`，cache `results/20260509_0102_within_subject_cache_imagery_ternary.json`。Baseline 对比来源：CBraMod `20260323_2320`，EEGNet `20260329_0056`。

## 不在本 handoff 范围

- 论文段落具体写法 / 章节插入位置
- 是否将本次任一 random-init run 升格为 baseline（按 [CLAUDE.md baseline 管理规范](../../CLAUDE.md)，需开发者明确确认）
- 与 EEGNet-Huge 实验（`docs/handoffs/2026-05-09_eegnet_huge.md`）的联合解读 — 那是另一个独立 ablation 维度
