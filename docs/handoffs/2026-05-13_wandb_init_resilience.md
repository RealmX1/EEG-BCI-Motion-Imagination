# 2026-05-13: wandb 初始化容错补丁

## 背景

近期多次实验"训练根本起不来"。诊断后定位到 wandb 客户端在国内网络下偶发挂死，而非 wandb 包未安装。

## 故障现场（实证）

| 现象 | 日志位置 |
|------|----------|
| `wandb.init()` 卡在 Windows WMI 探测里，被 Ctrl+C 救出 | `wandb/run-20260125_194849-b79a14po/logs/debug.log`、`wandb/run-20260121_200112-sieypab2/logs/debug.log` |
| 训练中途 `file_stream` 返回 500 + `context deadline exceeded` | `wandb/run-20260512_211637-c2pp04vl/logs/debug-internal.log` |
| GraphQL POST `context canceled` / `unexpected EOF` | `wandb/run-20260114_211720-u3d1cjv6`、`wandb/run-20260203_062710-1rvxc791` 的 debug-internal.log |

诊断中已排除：
- wandb 包未装（`uv pip list` 显示 `0.23.1` 在 `.venv` 中）
- namespace 遮蔽（项目根 `wandb/` 无 `__init__.py`，site-packages 真包优先）

## 修复要点

**改动文件**：`src/utils/wandb_logger.py`

1. `WandbLogger.__init__`（约 line 204）：
   - 给 `wandb.init()` 加 `init_timeout=30`（默认 90 秒）
   - 用 `try/except Exception` 包裹；失败后 `self._enabled = False; self._run = None; return`
   - `wandb.define_metric()` 也加同样的 `try/except`，避免次生异常
2. `WandbLogger.finish`（约 line 421）：
   - `wandb.finish()` 用 `try/except/finally` 包裹
   - `finally` 中保证 `self._run = None`，防止半完成状态

宽 catch 是有意为之：wandb SDK 把 Go 后端错误翻译成多种形态（`CommError` / `UsageError` / `TimeoutError` 等），目标是"任何 wandb 失败都不影响训练"，所以一律降级为 no-op。

## 验证

- V1（单元冒烟）：`WandbLogger(entity='__nonexistent_entity_xyz__')` 应在 ≤ 35s 内返回 `enabled=False`
- V2（离线冒烟）：`$env:WANDB_MODE = 'disabled'` 应立即返回，无网络等待
- V3（完整训练）：`uv run python scripts/experiments/run_within_subject.py --model eegnet --task binary --paradigm imagery --subjects S01 --epochs 2 --cache-only` 应正常完成两 epoch

详见计划文件 `C:\Users\zhang\.claude\plans\there-seems-to-be-piped-owl.md`。

## 未来备选（如果国内网络持续恶化）

本次刻意没动这些；按风险递增排序：

1. **设置 `WANDB_BASE_URL` 走代理**：用户层设环境变量，repo 不引入 proxy 配置
2. **`WANDB_MODE=offline` 默认 + 训练结束后 `wandb sync wandb/offline-run-*`**：把网络从训练关键路径上彻底移除；适合无人值守长跑
3. **训练脚本顶层加 connectivity 预检**（如 `requests.head('https://api.wandb.ai', timeout=1)`，失败则强制 `os.environ['WANDB_MODE']='offline'`）：自动切换，但增加运行时复杂度
4. **整体废弃 wandb，只保留 TensorBoard / JSON / ExperimentDB**：极端选择；目前 wandb 在 dashboard 比较和远程查看上仍有价值

## 风险

宽 `except Exception` 可能吞掉真正的代码 bug（如 `entity` 拼错）。缓解：`logger.warning` 会把 `type(exc).__name__` 和 `exc` 都打印出来，终端能看到。
