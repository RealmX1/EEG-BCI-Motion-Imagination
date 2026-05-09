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

新跑命令：

```bash
uv run python scripts/experiments/run_within_subject_comparison.py \
  --models cbramod --task ternary --no-pretrained --seed 1234 \
  --no-wandb --cache-only --force-retrain
```

JSON cache：`results/20260509_1838_within_subject_cache_imagery_ternary.json`。ExperimentDB run_tag：`20260509_1838`。

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
