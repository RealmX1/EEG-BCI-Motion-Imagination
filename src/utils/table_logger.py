"""
表格式训练日志器，提供清晰的 epoch 进度显示。

提供统一的可视化训练输出，包括：
- 固定宽度列的表格输出
- 颜色高亮（使用 ANSI 颜色码）
- 最佳值标记和条件色彩
- 进度条和 ETA 显示
- Majority Voting accuracy 支持
"""

import re
import time
from dataclasses import dataclass
from typing import List, Literal, Optional

from .timing import Colors, colored, format_time_plain

# ANSI 转义码正则表达式
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def visible_len(s: str) -> int:
    """计算字符串的可见长度（不包含 ANSI 转义码）."""
    return len(ANSI_ESCAPE.sub("", s))


def pad_right(s: str, width: int) -> str:
    """右填充字符串，正确处理 ANSI 转义码."""
    visible = visible_len(s)
    if visible >= width:
        return s
    return s + " " * (width - visible)


@dataclass
class EpochMetrics:
    """单个 epoch 的指标记录."""

    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    majority_acc: Optional[float] = None
    lr: Optional[float] = None
    epoch_time: Optional[float] = None
    is_best: bool = False
    event: Optional[str] = None  # "BEST", "STOP", "LR↓" 等


class TableEpochLogger:
    """
    表格式训练日志器，提供清晰的 epoch 进度显示。

    输出格式示例:
    Training: S01 | CBraMod | Binary                              [GPU: cuda:0]
    Progress: [████████████████████░░░░░░░░░░] 67% (20/30) ETA: 23s

     Epoch │    Loss (T/V)     │     Acc (T/V)     │ Maj Acc │    LR    │ Time
    ───────┼───────────────────┼───────────────────┼─────────┼──────────┼──────
         1 │  0.6932 /  0.7012 │  0.5234 /  0.5012 │  0.4923 │ 1.00e-03 │ 2.3s
         5 │  0.4521 /  0.4789 │  0.7123 /  0.6890 │  0.6512 │ 9.50e-04 │ 2.2s
     * 12  │  0.2456 /  0.2987 │  0.8567 / ↑0.8234 │ ↑0.8123 │ 7.00e-04 │ 2.3s  BEST
        15 │  0.1987 /  0.2654 │  0.8901 /  0.8156 │  0.8056 │ 5.00e-04 │ 2.2s
        20 │  0.1234 /  0.2456 │  0.9234 /  0.8123 │  0.7989 │ 3.00e-04 │ 2.3s  STOP

    ────────────────────────────────────────────────────────────────────────────
    Training Complete | Best: Epoch 12 | Val Acc: 0.8234 | Maj: 0.8123 | 46.2s

    特性:
    - 固定列宽，对齐输出
    - 最佳值使用 * 标记
    - 改进时显示 ↑ (绿色)，退步时显示 ↓ (红色)
    - 支持颜色输出 (可通过 use_color=False 禁用)
    - 动态覆盖 + 周期保留
    """

    # 默认列配置
    DEFAULT_COLUMNS = ["epoch", "loss", "acc", "majority_acc", "lr", "time"]

    # 列宽定义
    COLUMN_WIDTHS = {
        "epoch": 7,
        "loss": 19,  # "  0.6932 /  0.7012"
        "acc": 19,  # "  0.5234 /  0.5012"
        "majority_acc": 9,
        "lr": 10,
        "time": 6,
        "event": 6,
    }

    # 表格绘制字符 (Unicode box-drawing)
    BOX_H = "─"  # 水平线
    BOX_V = "│"  # 垂直线
    BOX_CROSS = "┼"  # 交叉

    def __init__(
        self,
        total_epochs: int,
        model_name: str = "",
        task_name: str = "",
        subject: str = "",
        device: str = "",
        keep_every: int = 1,
        header_every: int = 25,
        show_majority: bool = True,
        use_color: bool = True,
    ):
        """
        初始化表格式日志器.

        Args:
            total_epochs: 总 epoch 数
            model_name: 模型名称（显示在标题行）
            task_name: 任务名称
            subject: 被试 ID
            device: 设备名称（如 "cuda:0"）
            keep_every: 每 N 个 epoch 打印一行（默认 1 = 打印所有）
            header_every: 每 N 行重新打印表头
            show_majority: 是否显示 Majority Voting Accuracy 列
            use_color: 是否启用颜色输出
        """
        self.total_epochs = total_epochs
        self.model_name = model_name
        self.task_name = task_name
        self.subject = subject
        self.device = device
        self.keep_every = keep_every
        self.header_every = header_every
        self.show_majority = show_majority
        self.use_color = use_color

        # 状态追踪
        self.start_time: Optional[float] = None
        self.current_epoch = 0
        self.lines_since_header = 0

        # 最佳值追踪
        self.best_val_acc = 0.0
        self.best_val_loss = float("inf")
        self.best_majority_acc = 0.0
        self.best_epoch = 0

        # 上一个 epoch 的值（用于计算改进）
        self.prev_val_acc = 0.0
        self.prev_val_loss = float("inf")
        self.prev_majority_acc = 0.0

        # 历史记录
        self.history: List[EpochMetrics] = []

        # 是否已打印表头
        self._header_printed = False

    def _c(self, text: str, color: str, bold: bool = False) -> str:
        """应用颜色（如果启用）."""
        if not self.use_color:
            return text
        return colored(text, color, bold)

    def _format_value(
        self,
        value: float,
        prev_value: Optional[float] = None,
        best_value: Optional[float] = None,
        mode: Literal["max", "min"] = "max",
        width: int = 7,
        precision: int = 4,
    ) -> str:
        """
        格式化数值，添加改进/退步指示符和颜色.

        Args:
            value: 当前值
            prev_value: 上一个值（用于计算改进）
            best_value: 最佳值（用于判断是否为新最佳）
            mode: "max" 表示越大越好，"min" 表示越小越好
            width: 总宽度
            precision: 小数位数
        """
        # 基础格式化
        formatted = f"{value:.{precision}f}"

        # 确定是否改进
        indicator = " "
        color = Colors.WHITE

        if prev_value is not None:
            if mode == "max":
                improved = value > prev_value + 0.0001
                declined = value < prev_value - 0.0001
            else:
                improved = value < prev_value - 0.0001
                declined = value > prev_value + 0.0001

            if improved:
                indicator = "↑"
                color = Colors.BRIGHT_GREEN
            elif declined:
                indicator = "↓"
                color = Colors.BRIGHT_RED

        # 组合结果
        result = f"{indicator}{formatted}"
        result = result.rjust(width)

        return self._c(result, color)

    def _make_progress_bar(self, progress: float, width: int = 30) -> str:
        """创建进度条."""
        filled = int(progress * width)
        empty = width - filled

        # 使用 ASCII 兼容字符
        bar_filled = "#" * filled
        bar_empty = "-" * empty

        bar = self._c(bar_filled, Colors.GREEN) + self._c(bar_empty, Colors.DIM)
        return f"[{bar}]"

    def print_title(self):
        """打印训练标题行."""
        parts = []
        if self.subject:
            parts.append(self.subject)
        if self.model_name:
            parts.append(self.model_name)
        if self.task_name:
            parts.append(self.task_name)

        title = " | ".join(parts)
        title = self._c(f"Training: {title}", Colors.CYAN, bold=True)

        device_info = ""
        if self.device:
            device_info = self._c(f"[GPU: {self.device}]", Colors.DIM)

        # 计算右对齐
        total_width = 78
        title_plain_len = len(f"Training: {' | '.join(parts)}")
        device_plain_len = len(f"[GPU: {self.device}]") if self.device else 0
        padding = total_width - title_plain_len - device_plain_len

        print(f"{title}{' ' * max(1, padding)}{device_info}")
        self.start_time = time.perf_counter()

    def print_progress(self, epoch: int):
        """打印进度条行."""
        progress = epoch / self.total_epochs
        bar = self._make_progress_bar(progress)
        pct = int(progress * 100)

        # ETA 计算
        eta_str = ""
        if self.start_time and epoch > 0:
            elapsed = time.perf_counter() - self.start_time
            remaining = (elapsed / epoch) * (self.total_epochs - epoch)
            eta_str = f" ETA: {format_time_plain(remaining)}"

        progress_str = f"Progress: {bar} {pct}% ({epoch}/{self.total_epochs}){eta_str}"
        print(progress_str)

    def print_header(self):
        """打印表头."""
        # 列标题
        headers = [
            ("Epoch", self.COLUMN_WIDTHS["epoch"]),
            ("Loss (T/V)", self.COLUMN_WIDTHS["loss"]),
            ("Acc (T/V)", self.COLUMN_WIDTHS["acc"]),
        ]

        if self.show_majority:
            headers.append(("Maj Acc", self.COLUMN_WIDTHS["majority_acc"]))

        headers.extend(
            [
                ("LR", self.COLUMN_WIDTHS["lr"]),
                ("Time", self.COLUMN_WIDTHS["time"]),
            ]
        )

        # 构建表头行
        header_parts = []
        separator_parts = []

        for title, width in headers:
            header_parts.append(title.center(width))
            separator_parts.append(self.BOX_H * width)

        header_line = f" {self.BOX_V.join(header_parts)}"
        separator_line = self.BOX_H + self.BOX_CROSS.join(separator_parts) + self.BOX_H

        print()
        print(self._c(header_line, Colors.WHITE, bold=True))
        print(self._c(separator_line, Colors.DIM))

        self._header_printed = True
        self.lines_since_header = 0

    def print_footer(self):
        """打印表尾分隔线."""
        total_width = 78
        print(self._c(self.BOX_H * total_width, Colors.DIM))

    def on_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        majority_acc: Optional[float] = None,
        lr: Optional[float] = None,
        epoch_time: Optional[float] = None,
        is_best: bool = False,
        event: Optional[str] = None,
    ):
        """
        记录一个 epoch 的指标并打印.

        Args:
            epoch: 当前 epoch（1-indexed）
            train_loss: 训练损失
            train_acc: 训练准确率
            val_loss: 验证损失
            val_acc: 验证准确率
            majority_acc: Majority Voting 准确率（可选）
            lr: 当前学习率（可选）
            epoch_time: epoch 耗时（可选）
            is_best: 是否为最佳 epoch
            event: 事件标记（如 "BEST", "STOP", "LR↓"）
        """
        self.current_epoch = epoch

        # 保存历史
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            majority_acc=majority_acc,
            lr=lr,
            epoch_time=epoch_time,
            is_best=is_best,
            event=event,
        )
        self.history.append(metrics)

        # 更新最佳值 (仅依赖 is_best 标记，与 EEGTrainer 的 combined_score 逻辑保持一致)
        if is_best:
            self.best_val_acc = val_acc
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            if majority_acc is not None:
                self.best_majority_acc = majority_acc

        # 判断是否需要打印此行
        should_print = (
            epoch == 1
            or epoch == self.total_epochs
            or epoch % self.keep_every == 0
            or is_best
            or event is not None
        )

        if not should_print:
            # 更新上一个值
            self.prev_val_acc = val_acc
            self.prev_val_loss = val_loss
            if majority_acc is not None:
                self.prev_majority_acc = majority_acc
            return

        # 检查是否需要重新打印表头
        if not self._header_printed or self.lines_since_header >= self.header_every:
            self.print_header()

        # 构建行内容
        self._print_epoch_row(metrics)

        # 更新计数器
        self.lines_since_header += 1

        # 更新上一个值
        self.prev_val_acc = val_acc
        self.prev_val_loss = val_loss
        if majority_acc is not None:
            self.prev_majority_acc = majority_acc

    def _print_epoch_row(self, metrics: EpochMetrics):
        """打印单个 epoch 的行."""
        parts = []

        # Epoch 列 (带 * 标记最佳)
        epoch_str = f"{metrics.epoch:>5}"
        if metrics.is_best:
            epoch_str = self._c(f"*{metrics.epoch:>4}", Colors.BRIGHT_YELLOW, bold=True)
        else:
            epoch_str = f" {epoch_str}"
        parts.append(pad_right(epoch_str, self.COLUMN_WIDTHS["epoch"]))

        # Loss 列 (T/V)
        train_loss_str = f"{metrics.train_loss:.4f}"
        val_loss_str = self._format_value(
            metrics.val_loss,
            self.prev_val_loss if self.current_epoch > 1 else None,
            self.best_val_loss,
            mode="min",
            width=7,
        )
        loss_str = f" {train_loss_str} / {val_loss_str}"
        parts.append(pad_right(loss_str, self.COLUMN_WIDTHS["loss"]))

        # Acc 列 (T/V)
        train_acc_str = f"{metrics.train_acc:.4f}"
        val_acc_str = self._format_value(
            metrics.val_acc,
            self.prev_val_acc if self.current_epoch > 1 else None,
            self.best_val_acc,
            mode="max",
            width=7,
        )
        acc_str = f" {train_acc_str} / {val_acc_str}"
        parts.append(pad_right(acc_str, self.COLUMN_WIDTHS["acc"]))

        # Majority Acc 列
        if self.show_majority:
            if metrics.majority_acc is not None:
                maj_str = self._format_value(
                    metrics.majority_acc,
                    self.prev_majority_acc if self.current_epoch > 1 else None,
                    self.best_majority_acc,
                    mode="max",
                    width=8,
                )
            else:
                maj_str = self._c("   ---  ", Colors.DIM)
            parts.append(pad_right(maj_str, self.COLUMN_WIDTHS["majority_acc"]))

        # LR 列
        if metrics.lr is not None:
            lr_str = f" {metrics.lr:.2e}"
        else:
            lr_str = "    ---   "
        parts.append(pad_right(lr_str, self.COLUMN_WIDTHS["lr"]))

        # Time 列
        if metrics.epoch_time is not None:
            time_str = f" {format_time_plain(metrics.epoch_time)}"
        else:
            time_str = "  --- "
        parts.append(pad_right(time_str, self.COLUMN_WIDTHS["time"]))

        # 事件标记
        event_str = ""
        if metrics.event:
            if metrics.event == "BEST":
                event_str = self._c("  BEST", Colors.BRIGHT_GREEN, bold=True)
            elif metrics.event == "STOP":
                event_str = self._c("  STOP", Colors.BRIGHT_YELLOW, bold=True)
            elif "LR" in metrics.event:
                event_str = self._c(f"  {metrics.event}", Colors.CYAN)
            else:
                event_str = f"  {metrics.event}"

        # 组合并打印
        row = self.BOX_V.join(parts)
        print(f" {row}{event_str}")

    def log_epoch(self, epoch: int, metrics: dict):
        """
        简化的 epoch 记录接口.

        Args:
            epoch: 当前 epoch（1-indexed）
            metrics: 指标字典，支持的键：
                - train_loss, train_acc
                - val_loss, val_acc
                - majority_acc (可选)
                - lr (可选)
                - epoch_time (可选)
                - is_best (可选)
                - event (可选)
        """
        self.on_epoch_end(
            epoch=epoch,
            train_loss=metrics.get("train_loss", 0.0),
            train_acc=metrics.get("train_acc", 0.0),
            val_loss=metrics.get("val_loss", 0.0),
            val_acc=metrics.get("val_acc", 0.0),
            majority_acc=metrics.get("majority_acc"),
            lr=metrics.get("lr"),
            epoch_time=metrics.get("epoch_time"),
            is_best=metrics.get("is_best", False),
            event=metrics.get("event"),
        )

    def update_best(
        self, metric_name: str, value: float, mode: Literal["max", "min"] = "max"
    ) -> bool:
        """
        更新最佳值追踪.

        Args:
            metric_name: 指标名称 ("val_acc", "val_loss", "majority_acc")
            value: 新值
            mode: "max" 或 "min"

        Returns:
            是否更新了最佳值
        """
        if metric_name == "val_acc":
            if mode == "max" and value > self.best_val_acc:
                self.best_val_acc = value
                return True
        elif metric_name == "val_loss":
            if mode == "min" and value < self.best_val_loss:
                self.best_val_loss = value
                return True
        elif metric_name == "majority_acc":
            if mode == "max" and value > self.best_majority_acc:
                self.best_majority_acc = value
                return True
        return False

    def print_summary(self):
        """打印训练总结."""
        self.print_footer()

        # 计算总时间
        total_time = 0.0
        if self.start_time:
            total_time = time.perf_counter() - self.start_time

        # 总结行
        summary_parts = [
            self._c("Training Complete", Colors.CYAN, bold=True),
            f"Epochs: {self.current_epoch}/{self.total_epochs}",
            f"Best: Epoch {self.best_epoch}",
            f"Val Acc: {self.best_val_acc:.4f}",
        ]

        if self.best_majority_acc > 0:
            summary_parts.append(f"Maj: {self.best_majority_acc:.4f}")

        if total_time > 0:
            summary_parts.append(format_time_plain(total_time))

        summary = " | ".join(summary_parts)
        print(summary)
        print()

    def get_best_metrics(self) -> dict:
        """获取最佳指标."""
        return {
            "best_epoch": self.best_epoch,
            "best_val_acc": self.best_val_acc,
            "best_val_loss": self.best_val_loss,
            "best_majority_acc": self.best_majority_acc,
        }


def demo_table_logger():
    """演示 TableEpochLogger 的使用."""
    import random

    logger = TableEpochLogger(
        total_epochs=30,
        model_name="CBraMod",
        task_name="Binary",
        subject="S01",
        device="cuda:0",
        keep_every=5,
        show_majority=True,
    )

    logger.print_title()

    # 模拟训练
    val_acc = 0.5
    maj_acc = 0.48

    for epoch in range(1, 31):
        # 模拟指标
        train_loss = 0.7 - epoch * 0.02 + random.uniform(-0.02, 0.02)
        train_acc = 0.5 + epoch * 0.015 + random.uniform(-0.02, 0.02)
        val_loss = 0.7 - epoch * 0.015 + random.uniform(-0.03, 0.03)
        val_acc = min(0.95, val_acc + random.uniform(-0.01, 0.03))
        maj_acc = min(0.93, maj_acc + random.uniform(-0.01, 0.025))
        lr = 1e-3 * (0.95**epoch)

        is_best = val_acc > logger.best_val_acc
        event = "BEST" if is_best else None

        if epoch == 30:
            event = "STOP"

        logger.on_epoch_end(
            epoch=epoch,
            train_loss=max(0.1, train_loss),
            train_acc=min(0.98, train_acc),
            val_loss=max(0.15, val_loss),
            val_acc=val_acc,
            majority_acc=maj_acc if epoch % 5 == 0 or is_best else None,
            lr=lr,
            epoch_time=random.uniform(2.0, 2.5),
            is_best=is_best,
            event=event,
        )

        time.sleep(0.05)  # 模拟训练耗时

    logger.print_summary()


if __name__ == "__main__":
    demo_table_logger()
