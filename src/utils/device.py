"""
Device utilities for EEG-BCI project.

Ensures NVIDIA GPU is available and used for training.
Provides reproducibility utilities (random seed setting).
"""

import sys
import os
import random
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value (default: 42)
        deterministic: If True, enable CUDA deterministic mode.
                      This may reduce performance but ensures reproducibility.

    Note:
        For full reproducibility on CUDA, you may also need to set:
        CUBLAS_WORKSPACE_CONFIG=:4096:8 environment variable
    """
    # Python built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuDNN deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # PyTorch 1.8+ deterministic algorithms
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
            except RuntimeError:
                # Some operations don't have deterministic implementations
                logger.warning(
                    "Could not enable fully deterministic algorithms. "
                    "Some operations may still be non-deterministic."
                )

    # Set environment variable for CUBLAS (PyTorch 1.8+)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

    logger.info(f"Random seed set to {seed} (deterministic={deterministic})")


def check_cuda_available(required: bool = True) -> bool:
    """
    Check if CUDA is available and working.

    Args:
        required: If True, exit program if CUDA not available

    Returns:
        True if CUDA is available and working

    Raises:
        SystemExit: If required=True and CUDA not available/working
    """
    if not torch.cuda.is_available():
        msg = (
            "CUDA is not available! "
            "This project requires an NVIDIA GPU for training. "
            "Please ensure:\n"
            "  1. NVIDIA GPU is installed\n"
            "  2. NVIDIA drivers are up to date\n"
            "  3. PyTorch is installed with CUDA support\n"
            "\n"
            "To install PyTorch with CUDA:\n"
            "  uv pip install torch --index-url https://download.pytorch.org/whl/cu124"
        )
        logger.error(msg)

        if required:
            print(f"\nERROR: {msg}", file=sys.stderr)
            sys.exit(1)

        return False

    # Test if CUDA actually works (check for compute capability support)
    try:
        # Try a simple CUDA operation
        test_tensor = torch.zeros(1, device='cuda')
        del test_tensor
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        cc = f"{props.major}.{props.minor}"

        msg = (
            f"CUDA device found but not compatible!\n"
            f"  GPU: {gpu_name}\n"
            f"  Compute Capability: {cc}\n"
            f"\n"
            f"Error: {e}\n"
            f"\n"
            f"This usually means your GPU is too new for the current PyTorch.\n"
            f"For RTX 50-series (Blackwell, sm_120), you may need to:\n"
            f"  1. Wait for official PyTorch support\n"
            f"  2. Build PyTorch from source with sm_120 support\n"
            f"  3. Check https://pytorch.org/get-started/locally/ for updates"
        )
        logger.error(msg)

        if required:
            print(f"\nERROR: {msg}", file=sys.stderr)
            sys.exit(1)

        return False


def get_device(allow_cpu: bool = False) -> torch.device:
    """
    Get the device to use for training.

    Args:
        allow_cpu: If True, fall back to CPU if CUDA not available.
                  If False (default), exit if CUDA not available.

    Returns:
        torch.device for training

    Raises:
        SystemExit: If allow_cpu=False and CUDA not available
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')

        # Log GPU info (debug level to reduce clutter)
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

        logger.debug(f"Using GPU: {gpu_name}")
        logger.debug(f"GPU Memory: {gpu_memory:.1f} GB")

        return device

    else:
        if allow_cpu:
            logger.warning("CUDA not available, using CPU (training will be slow)")
            return torch.device('cpu')
        else:
            check_cuda_available(required=True)  # This will exit
            return torch.device('cpu')  # Never reached


def is_blackwell_gpu() -> bool:
    """
    检测当前 GPU 是否为 Blackwell 架构 (sm_120+).

    Blackwell 架构的 GPU (如 RTX 50 系列) 目前 torch.compile/Triton 支持有限，
    需要跳过编译以避免兼容性问题。

    Returns:
        bool: True 如果是 Blackwell 架构 GPU
    """
    if not torch.cuda.is_available():
        return False

    try:
        props = torch.cuda.get_device_properties(0)
        # Blackwell 架构: compute capability >= 12.0 (sm_120)
        # 参考: https://developer.nvidia.com/cuda-gpus
        return props.major >= 12
    except Exception:
        return False


def print_gpu_info():
    """Print detailed GPU information."""
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return

    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch CUDA: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Total Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"  Multi-Processor Count: {props.multi_processor_count}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("GPU Information:")
    print("=" * 50)
    print_gpu_info()

    print("\n" + "=" * 50)
    print("Device check:")
    try:
        device = get_device(allow_cpu=False)
        print(f"Using device: {device}")
    except SystemExit:
        print("Exited due to no CUDA")
