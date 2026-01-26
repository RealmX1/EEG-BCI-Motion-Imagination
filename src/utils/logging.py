"""
Shared logging utilities for EEG-BCI project.

Provides colored, compact log formatting with section tags.
Different log levels have different colors for better visibility.

Usage:
    from src.utils.logging import get_logger, SectionLogger

    logger = get_logger(__name__)
    log_train = SectionLogger(logger, 'train')
    log_train.info("Training started")

Output format:
    12:34:56 INFO:[module:section] message
"""

import logging
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Formatter that outputs colored logs based on level with compact format."""
    # ANSI color codes
    RESET = '\033[0m'
    DIM = '\033[2m'
    BOLD = '\033[1m'

    # Level colors
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'WARN': '\033[93m',     # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[95m', # Magenta
    }

    def __init__(self, module_name: str = 'eeg'):
        super().__init__()
        self.module_name = module_name

    def format(self, record):
        section = getattr(record, 'section', 'main')
        time_str = self.formatTime(record, '%H:%M:%S')
        level = record.levelname[:4]
        msg = record.getMessage()
        color = self.COLORS.get(record.levelname, '\033[37m')  # Default: white
        return f"{color}{time_str} {level}:{self.DIM}[{self.module_name}:{section}]{self.RESET}{color} {msg}{self.RESET}"


# Alias for backwards compatibility
YellowFormatter = ColoredFormatter


class SectionLogger:
    """Logger wrapper that adds section info to all log calls."""

    def __init__(self, base_logger: logging.Logger, section: str):
        self._logger = base_logger
        self._section = section

    def _log(self, level: str, msg: str, *args, **kwargs):
        kwargs.setdefault('extra', {})['section'] = self._section
        getattr(self._logger, level)(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self._log('info', msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._log('warning', msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._log('error', msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        self._log('debug', msg, *args, **kwargs)


def setup_logging(module_name: str = 'eeg', level: int = logging.INFO) -> None:
    """
    Set up yellow logging formatter for the root logger.

    Args:
        module_name: Module name to show in log prefix (e.g., 'train', 'compare')
        level: Logging level (default: INFO)
    """
    handler = logging.StreamHandler()
    handler.setFormatter(YellowFormatter(module_name))

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


def create_section_loggers(
    logger: logging.Logger,
    sections: list[str]
) -> dict[str, SectionLogger]:
    """
    Create multiple section loggers at once.

    Args:
        logger: Base logger
        sections: List of section names

    Returns:
        Dict mapping section name to SectionLogger

    Example:
        logs = create_section_loggers(logger, ['data', 'train', 'eval'])
        logs['train'].info("Training started")
    """
    return {section: SectionLogger(logger, section) for section in sections}
