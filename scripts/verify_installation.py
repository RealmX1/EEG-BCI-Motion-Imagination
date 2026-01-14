#!/usr/bin/env python
"""
安装验证脚本 - 检查 EEG-BCI 项目环境是否正确配置

用法:
    uv run python scripts/verify_installation.py
"""

import sys
import platform
import importlib.util
import io
from pathlib import Path

# 修复 Windows 控制台 UTF-8 输出
if platform.system() == "Windows":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ANSI 颜色码 (Windows 10+ 支持)
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {Colors.GREEN}[OK]{Colors.END} {msg}")


def fail(msg: str) -> None:
    print(f"  {Colors.RED}[FAIL]{Colors.END} {msg}")


def warn(msg: str) -> None:
    print(f"  {Colors.YELLOW}[WARN]{Colors.END} {msg}")


def info(msg: str) -> None:
    print(f"  {Colors.BLUE}[INFO]{Colors.END} {msg}")


def header(title: str) -> None:
    print(f"\n{Colors.BOLD}[{title}]{Colors.END}")


def check_python_version() -> bool:
    """检查 Python 版本"""
    header("Python 版本")
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major == 3 and version.minor >= 9:
        ok(f"Python {version_str}")
        if version.minor == 11:
            info("推荐版本 3.11.x")
        return True
    else:
        fail(f"Python {version_str} (需要 >= 3.9)")
        return False


def check_platform() -> None:
    """显示平台信息"""
    header("系统信息")
    info(f"操作系统: {platform.system()} {platform.release()}")
    info(f"架构: {platform.machine()}")


def check_uv() -> bool:
    """检查 uv 是否可用"""
    header("包管理器")
    import shutil

    if shutil.which("uv"):
        ok("uv 已安装")
        return True
    else:
        warn("uv 未找到 (推荐安装: https://docs.astral.sh/uv/)")
        return True  # 不阻塞，因为可以用 pip


def check_core_dependencies() -> tuple[bool, list[str]]:
    """检查核心依赖"""
    header("核心依赖")

    required = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("scikit-learn", "sklearn"),
        ("pandas", "pandas"),
        ("PyYAML", "yaml"),
        ("mne", "mne"),
        ("h5py", "h5py"),
    ]

    all_ok = True
    missing = []

    for name, import_name in required:
        spec = importlib.util.find_spec(import_name)
        if spec is not None:
            try:
                module = importlib.import_module(import_name)
                version = getattr(module, "__version__", "?")
                ok(f"{name} ({version})")
            except Exception:
                ok(f"{name} (已安装)")
        else:
            fail(f"{name} 未安装")
            missing.append(name)
            all_ok = False

    return all_ok, missing


def check_pytorch() -> tuple[bool, dict]:
    """检查 PyTorch 和 CUDA"""
    header("PyTorch & CUDA")

    result = {
        "torch_installed": False,
        "cuda_available": False,
        "cuda_version": None,
        "gpu_name": None,
        "gpu_memory_gb": None,
    }

    try:
        import torch

        result["torch_installed"] = True
        ok(f"PyTorch {torch.__version__}")

        if torch.cuda.is_available():
            result["cuda_available"] = True
            result["cuda_version"] = torch.version.cuda
            ok(f"CUDA {torch.version.cuda}")

            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                result["gpu_name"] = props.name
                result["gpu_memory_gb"] = props.total_memory / (1024**3)
                ok(f"GPU {i}: {props.name} ({result['gpu_memory_gb']:.1f} GB)")

            # 验证 GPU 可以执行计算
            try:
                x = torch.randn(100, 100, device="cuda")
                y = torch.matmul(x, x)
                del x, y
                torch.cuda.empty_cache()
                ok("GPU 张量计算正常")
            except Exception as e:
                fail(f"GPU 计算测试失败: {e}")
                return False, result
        else:
            fail("CUDA 不可用")
            warn("本项目需要 NVIDIA GPU，CPU 模式已禁用")
            return False, result

        return True, result

    except ImportError:
        fail("PyTorch 未安装")
        info("安装命令:")
        info("  RTX 50 系列: uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128")
        info("  RTX 40/30:   uv pip install torch --index-url https://download.pytorch.org/whl/cu124")
        return False, result


def check_cbramod() -> bool:
    """检查 CBraMod 是否可用 (可选)"""
    header("CBraMod (可选)")

    try:
        from cbramod import CBraMod

        ok("CBraMod 已安装")
        return True
    except ImportError:
        warn("CBraMod 未安装 (可选，仅 CBraMod 模型需要)")
        info("安装命令:")
        info("  git clone https://github.com/wjq-learning/CBraMod.git")
        info("  uv pip install -e CBraMod/")
        return True  # 可选，不阻塞


def check_data_directory() -> bool:
    """检查数据目录结构"""
    header("数据目录")

    data_dir = Path("data")
    if not data_dir.exists():
        warn("data/ 目录不存在")
        info("请下载 FINGER-EEG-BCI 数据集并放置在 data/ 目录")
        return True  # 不阻塞

    # 检查被试目录
    subjects = list(data_dir.glob("S[0-9][0-9]"))
    if subjects:
        ok(f"找到 {len(subjects)} 个被试目录")
        for s in subjects[:3]:
            info(f"  {s.name}")
        if len(subjects) > 3:
            info(f"  ... 及其他 {len(subjects) - 3} 个")
    else:
        warn("未找到被试数据 (S01-S21)")

    # 检查电极文件
    elc_file = data_dir / "biosemi128.ELC"
    if elc_file.exists():
        ok("biosemi128.ELC 电极位置文件存在")
    else:
        warn("biosemi128.ELC 未找到")

    return True


def check_project_structure() -> bool:
    """检查项目结构"""
    header("项目结构")

    critical_files = [
        "pyproject.toml",
        "src/preprocessing/data_loader.py",
        "src/models/eegnet.py",
        "src/training/train_within_subject.py",
        "configs/eegnet_config.yaml",
    ]

    all_ok = True
    for f in critical_files:
        if Path(f).exists():
            ok(f)
        else:
            fail(f"{f} 不存在")
            all_ok = False

    return all_ok


def main() -> int:
    """运行所有检查"""
    print(f"\n{Colors.BOLD}{'=' * 50}{Colors.END}")
    print(f"{Colors.BOLD}  EEG-BCI 安装验证{Colors.END}")
    print(f"{Colors.BOLD}{'=' * 50}{Colors.END}")

    # 启用 Windows ANSI 支持
    if platform.system() == "Windows":
        import os

        os.system("")  # 启用 ANSI 转义序列

    checks = []

    check_platform()
    checks.append(("Python 版本", check_python_version()))
    checks.append(("uv 包管理器", check_uv()))

    deps_ok, missing = check_core_dependencies()
    checks.append(("核心依赖", deps_ok))

    pytorch_ok, pytorch_info = check_pytorch()
    checks.append(("PyTorch & CUDA", pytorch_ok))

    checks.append(("CBraMod", check_cbramod()))
    checks.append(("数据目录", check_data_directory()))
    checks.append(("项目结构", check_project_structure()))

    # 汇总
    print(f"\n{Colors.BOLD}{'=' * 50}{Colors.END}")
    print(f"{Colors.BOLD}  检查结果汇总{Colors.END}")
    print(f"{Colors.BOLD}{'=' * 50}{Colors.END}\n")

    critical_failed = False
    for name, passed in checks:
        if passed:
            ok(name)
        else:
            fail(name)
            if name in ["Python 版本", "核心依赖", "PyTorch & CUDA", "项目结构"]:
                critical_failed = True

    print()

    if critical_failed:
        print(f"{Colors.RED}{Colors.BOLD}环境检查失败{Colors.END}")
        print("请根据上述提示修复问题后重新运行此脚本。")
        print(f"故障排除指南: {Colors.BLUE}docs/TROUBLESHOOTING.md{Colors.END}\n")
        return 1
    else:
        print(f"{Colors.GREEN}{Colors.BOLD}环境检查通过{Colors.END}")
        print("可以开始运行实验了！\n")
        print("快速开始:")
        print("  uv run python scripts/run_full_comparison.py --help\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
