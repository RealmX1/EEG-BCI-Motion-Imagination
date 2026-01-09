# Utilities module
"""
Common utilities for EEG-BCI project.
"""

from .device import check_cuda_available, get_device

__all__ = [
    'check_cuda_available',
    'get_device',
]
