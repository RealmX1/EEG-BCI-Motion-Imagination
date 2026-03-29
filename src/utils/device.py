"""
Device utilities for EEG-BCI project.

Ensures NVIDIA GPU is available and used for training.
Provides reproducibility utilities (random seed setting).
"""

import subprocess
import sys
import os
import random
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

# VRAM thresholds for the interactive check
_VRAM_WARN_RELATIVE = 0.40  # Warn if >40% VRAM used by other processes
_VRAM_WARN_ABSOLUTE_MB = 4096  # Warn if <4 GB free


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


def check_vram_utilization(interactive: bool = True) -> bool:
    """Check GPU VRAM utilization and warn if other processes consume too much.

    When VRAM usage exceeds thresholds (>40% used by others or <4 GB free),
    shows the offending processes and offers the user options:
    - Continue anyway
    - Abort
    - Close processes one by one

    Args:
        interactive: If True (default), prompt user for action. If False,
            just log a warning and continue.

    Returns:
        True if training should proceed, False if user chose to abort.
    """
    if not torch.cuda.is_available():
        return True

    props = torch.cuda.get_device_properties(0)
    total_mb = props.total_memory / (1024 ** 2)

    # Get current VRAM usage from nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--id=0', '--query-gpu=memory.used,memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return True
        parts = result.stdout.strip().split(',')
        used_mb = float(parts[0].strip())
        free_mb = float(parts[1].strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        return True

    used_ratio = used_mb / total_mb

    # Check thresholds
    if used_ratio <= _VRAM_WARN_RELATIVE and free_mb >= _VRAM_WARN_ABSOLUTE_MB:
        logger.debug(f"VRAM OK: {used_mb:.0f}/{total_mb:.0f} MB used ({used_ratio:.0%}), {free_mb:.0f} MB free")
        return True

    # Gather GPU process list
    processes = _get_gpu_processes()

    # Display warning
    print(f"\n{'='*60}")
    print(f"  [!] GPU VRAM usage high")
    print(f"{'='*60}")
    print(f"  Used: {used_mb:.0f} MB / {total_mb:.0f} MB ({used_ratio:.0%})")
    print(f"  Free: {free_mb:.0f} MB")
    print()

    if processes:
        print(f"  Processes using VRAM:")
        print(f"  {'PID':>8s}  {'VRAM (MB)':>10s}  Process")
        print(f"  {'-'*8}  {'-'*10}  {'-'*36}")
        for proc in processes:
            name = os.path.basename(proc['name']) if proc['name'] else '(unknown)'
            mem_str = f"{proc['mem_mb']:.0f}" if proc['mem_mb'] else '?'
            print(f"  {proc['pid']:>8d}  {mem_str:>10s}  {name}")
        print()

    if not interactive or not sys.stdin.isatty() or os.environ.get('EEG_NONINTERACTIVE'):
        print("  (non-interactive mode -- continuing)")
        print(f"{'='*60}\n")
        return True

    # Interactive menu
    print("  Options:")
    print("    [c] Continue (ignore warning)")
    print("    [a] Abort experiment")
    print("    [k] Kill processes one by one")
    print()

    while True:
        try:
            choice = input("  Choose [c/a/k]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False

        if choice == 'c':
            print(f"{'='*60}\n")
            return True
        elif choice == 'a':
            print("  Aborted.")
            print(f"{'='*60}\n")
            return False
        elif choice == 'k':
            if processes:
                _interactive_kill_processes(processes)
                # Re-check after killing
                return check_vram_utilization(interactive=True)
            else:
                print("  No processes to kill.")
        else:
            print("  Invalid option, enter c / a / k")


def _get_gpu_processes():
    """Get list of processes using the GPU via nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi',
             '--query-compute-apps=pid,used_gpu_memory,process_name',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    processes = []
    for line in result.stdout.strip().split('\n'):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        try:
            mem_mb = float(parts[1])
        except (ValueError, TypeError):
            mem_mb = None
        name = parts[2] if parts[2] not in ('[N/A]', '') else None
        processes.append({'pid': pid, 'mem_mb': mem_mb, 'name': name})

    return processes


def _interactive_kill_processes(processes):
    """Offer to terminate GPU processes one by one."""
    # Filter to only killable user processes (skip system processes)
    current_pid = os.getpid()

    for proc in processes:
        if proc['pid'] == current_pid:
            continue
        name = os.path.basename(proc['name']) if proc['name'] else '(unknown)'
        mem_str = f"{proc['mem_mb']:.0f} MB" if proc['mem_mb'] else '? MB'

        try:
            choice = input(f"  Kill PID {proc['pid']} ({name}, {mem_str})? [y/n/q]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if choice == 'q':
            return
        elif choice == 'y':
            try:
                if sys.platform == 'win32':
                    subprocess.run(['taskkill', '/PID', str(proc['pid']), '/F'],
                                   capture_output=True, timeout=5)
                else:
                    os.kill(proc['pid'], 15)  # SIGTERM
                print(f"    [OK] Terminated PID {proc['pid']}")
            except Exception as e:
                print(f"    [FAIL] Could not terminate PID {proc['pid']}: {e}")


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
