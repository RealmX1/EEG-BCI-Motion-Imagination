"""
Timing utilities for performance profiling.

Provides colored console output and detailed timing breakdowns.
"""

import time
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict


# ANSI color codes (works on Windows 10+ and Unix)
class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"

    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


def enable_windows_ansi():
    """Enable ANSI escape sequences on Windows."""
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)


# Enable ANSI on import
try:
    enable_windows_ansi()
except Exception:
    pass


def colored(text: str, color: str, bold: bool = False) -> str:
    """Apply color to text."""
    prefix = Colors.BOLD if bold else ""
    return f"{prefix}{color}{text}{Colors.RESET}"


def format_time(seconds: float) -> str:
    """Format time in human-readable format with color coding."""
    if seconds < 0.001:
        return colored(f"{seconds*1000000:.0f}us", Colors.BRIGHT_GREEN)
    elif seconds < 0.1:
        return colored(f"{seconds*1000:.1f}ms", Colors.BRIGHT_GREEN)
    elif seconds < 1.0:
        return colored(f"{seconds*1000:.0f}ms", Colors.GREEN)
    elif seconds < 10.0:
        return colored(f"{seconds:.2f}s", Colors.YELLOW)
    elif seconds < 60.0:
        return colored(f"{seconds:.1f}s", Colors.BRIGHT_YELLOW)
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return colored(f"{mins}m {secs:.1f}s", Colors.RED, bold=True)


def format_time_plain(seconds: float) -> str:
    """Format time without colors."""
    if seconds < 0.001:
        return f"{seconds*1000000:.0f}us"
    elif seconds < 0.1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 1.0:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60.0:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"


@dataclass
class TimingStats:
    """Statistics for a timed operation."""
    name: str
    times: List[float] = field(default_factory=list)

    @property
    def total(self) -> float:
        return sum(self.times)

    @property
    def count(self) -> int:
        return len(self.times)

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0

    @property
    def min(self) -> float:
        return min(self.times) if self.times else 0.0

    @property
    def max(self) -> float:
        return max(self.times) if self.times else 0.0


class Timer:
    """
    Context manager and decorator for timing code blocks.

    Usage:
        # As context manager
        with Timer("data_loading"):
            load_data()

        # As decorator
        @Timer.decorator("forward_pass")
        def forward(x):
            return model(x)

        # Get stats
        Timer.print_summary()
    """

    _stats: Dict[str, TimingStats] = {}
    _stack: List[tuple] = []
    _enabled: bool = True

    def __init__(self, name: str, print_on_exit: bool = False, parent: Optional[str] = None):
        self.name = name
        self.print_on_exit = print_on_exit
        self.parent = parent
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        if Timer._enabled:
            self.start_time = time.perf_counter()
            Timer._stack.append((self.name, self.start_time))
        return self

    def __exit__(self, *args):
        if Timer._enabled and self.start_time is not None:
            self.elapsed = time.perf_counter() - self.start_time

            # Record stats
            full_name = self.name
            if self.parent:
                full_name = f"{self.parent}/{self.name}"

            if full_name not in Timer._stats:
                Timer._stats[full_name] = TimingStats(full_name)
            Timer._stats[full_name].times.append(self.elapsed)

            # Pop from stack
            if Timer._stack and Timer._stack[-1][0] == self.name:
                Timer._stack.pop()

            if self.print_on_exit:
                indent = "  " * len(Timer._stack)
                print(f"{indent}{colored('[TIME]', Colors.CYAN)} {self.name}: {format_time(self.elapsed)}")

    @classmethod
    def decorator(cls, name: str):
        """Use Timer as a decorator."""
        def wrapper(func):
            def wrapped(*args, **kwargs):
                with cls(name):
                    return func(*args, **kwargs)
            return wrapped
        return wrapper

    @classmethod
    def reset(cls):
        """Reset all timing statistics."""
        cls._stats.clear()
        cls._stack.clear()

    @classmethod
    def disable(cls):
        """Disable timing."""
        cls._enabled = False

    @classmethod
    def enable(cls):
        """Enable timing."""
        cls._enabled = True

    @classmethod
    def get_stats(cls) -> Dict[str, TimingStats]:
        """Get all timing statistics."""
        return cls._stats.copy()

    @classmethod
    def print_summary(cls, title: str = "Timing Summary"):
        """Print a summary of all timing statistics."""
        if not cls._stats:
            print(colored("No timing data collected.", Colors.DIM))
            return

        print()
        print(colored(f"{'='*60}", Colors.CYAN, bold=True))
        print(colored(f"  {title}", Colors.CYAN, bold=True))
        print(colored(f"{'='*60}", Colors.CYAN, bold=True))

        # Calculate total time
        root_times = [s for name, s in cls._stats.items() if '/' not in name]
        total_time = sum(s.total for s in root_times)

        # Group by parent
        grouped = defaultdict(list)
        for name, stats in sorted(cls._stats.items()):
            if '/' in name:
                parent, child = name.split('/', 1)
                grouped[parent].append((child, stats))
            else:
                grouped[''].append((name, stats))

        # Print root level first
        for name, stats in grouped.get('', []):
            pct = (stats.total / total_time * 100) if total_time > 0 else 0
            bar = cls._make_bar(pct)

            print(f"  {colored(name, Colors.WHITE, bold=True):40s} "
                  f"{format_time(stats.total):>12s} "
                  f"{colored(f'({stats.count}x)', Colors.DIM):>10s} "
                  f"{bar} {pct:5.1f}%")

            # Print children
            if name in grouped:
                for child_name, child_stats in grouped[name]:
                    child_pct = (child_stats.total / stats.total * 100) if stats.total > 0 else 0
                    child_bar = cls._make_bar(child_pct, width=15)
                    print(f"    {colored('└─', Colors.DIM)} {child_name:36s} "
                          f"{format_time(child_stats.total):>12s} "
                          f"{colored(f'({child_stats.count}x)', Colors.DIM):>10s} "
                          f"{child_bar} {child_pct:5.1f}%")

        print(colored(f"{'─'*60}", Colors.DIM))
        print(f"  {colored('TOTAL', Colors.WHITE, bold=True):40s} {format_time(total_time):>12s}")
        print(colored(f"{'='*60}", Colors.CYAN, bold=True))
        print()

    @staticmethod
    def _make_bar(pct: float, width: int = 20) -> str:
        """Create a visual progress bar (ASCII-safe for Windows)."""
        filled = int(pct / 100 * width)
        empty = width - filled

        if pct > 50:
            color = Colors.RED
        elif pct > 25:
            color = Colors.YELLOW
        else:
            color = Colors.GREEN

        # Use ASCII characters for Windows compatibility
        bar = colored('#' * filled, color) + colored('-' * empty, Colors.DIM)
        return f"[{bar}]"


@contextmanager
def timed_section(name: str, print_start: bool = True, print_end: bool = True):
    """
    Context manager for timing a section with start/end messages.

    Usage:
        with timed_section("Loading data"):
            data = load_data()
    """
    if print_start:
        print(f"{colored('[START]', Colors.BLUE)} {name}...")

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if print_end:
            print(f"{colored('[DONE]', Colors.GREEN)} {name}: {format_time(elapsed)}")


class EpochTimer:
    """
    Timer specifically for tracking epoch-level training metrics.

    Usage:
        epoch_timer = EpochTimer()

        for epoch in range(epochs):
            epoch_timer.start_epoch()

            with epoch_timer.phase("train"):
                train()

            with epoch_timer.phase("validate"):
                validate()

            epoch_timer.end_epoch()
            epoch_timer.print_epoch_summary()
    """

    def __init__(self):
        self.epoch_times: List[Dict[str, float]] = []
        self.current_epoch: Dict[str, float] = {}
        self.epoch_start: float = 0
        self.epoch_num: int = 0

    def start_epoch(self):
        """Mark the start of an epoch."""
        self.epoch_start = time.perf_counter()
        self.current_epoch = {}
        self.epoch_num += 1

    @contextmanager
    def phase(self, name: str):
        """Time a phase within an epoch."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.current_epoch[name] = time.perf_counter() - start

    def end_epoch(self):
        """Mark the end of an epoch."""
        self.current_epoch['total'] = time.perf_counter() - self.epoch_start
        self.epoch_times.append(self.current_epoch.copy())

    def print_epoch_summary(self, epoch_num: Optional[int] = None):
        """Print timing summary for an epoch."""
        if epoch_num is None:
            epoch_num = self.epoch_num

        if not self.current_epoch:
            return

        total = self.current_epoch.get('total', 0)
        parts = []

        for name, time_val in self.current_epoch.items():
            if name == 'total':
                continue
            pct = (time_val / total * 100) if total > 0 else 0

            # Color based on percentage
            if pct > 40:
                color = Colors.RED
            elif pct > 20:
                color = Colors.YELLOW
            else:
                color = Colors.GREEN

            parts.append(f"{name}={colored(format_time_plain(time_val), color)}({pct:.0f}%)")

        epoch_str = colored(f"Epoch {epoch_num}", Colors.CYAN, bold=True)
        total_str = format_time(total)
        parts_str = " | ".join(parts)

        print(f"  {epoch_str} [{total_str}] {parts_str}")

    def print_summary(self):
        """Print summary across all epochs."""
        if not self.epoch_times:
            return

        print()
        print(colored("Epoch Timing Summary", Colors.CYAN, bold=True))
        print(colored("─" * 50, Colors.DIM))

        # Aggregate stats
        all_phases = set()
        for epoch in self.epoch_times:
            all_phases.update(epoch.keys())
        all_phases.discard('total')

        for phase in sorted(all_phases):
            times = [e.get(phase, 0) for e in self.epoch_times]
            mean_time = sum(times) / len(times)
            total_time = sum(times)

            print(f"  {phase:20s} mean={format_time(mean_time):>10s}  total={format_time(total_time):>10s}")

        total_times = [e.get('total', 0) for e in self.epoch_times]
        mean_epoch = sum(total_times) / len(total_times)
        print(colored("─" * 50, Colors.DIM))
        print(f"  {'Epoch (mean)':20s} {format_time(mean_epoch):>10s}")
        print(f"  {'Total':20s} {format_time(sum(total_times)):>10s}")


def print_section_header(title: str):
    """Print a colored section header."""
    print()
    print(colored(f"{'='*60}", Colors.MAGENTA, bold=True))
    print(colored(f"  {title}", Colors.MAGENTA, bold=True))
    print(colored(f"{'='*60}", Colors.MAGENTA, bold=True))


def print_metric(name: str, value, color: str = Colors.WHITE):
    """Print a metric with formatting."""
    print(f"  {colored(name + ':', Colors.DIM)} {colored(str(value), color)}")
