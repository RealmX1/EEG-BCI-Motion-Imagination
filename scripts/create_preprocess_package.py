#!/usr/bin/env python
"""
Create a distribution package for preprocessing on remote machines.

This script packages all necessary files to run preprocess_zip.py on a machine
with the complete EEG-BCI dataset (S01-S21).

The package includes:
- preprocess_zip.py and all its dependencies
- package_caches.py for splitting caches into uploadable chunks
- requirements.txt for pip installation
- README with instructions (Chinese)
- CLAUDE.md for AI coding assistants

Usage:
    uv run python scripts/create_preprocess_package.py

Output:
    dist/eeg_bci_preprocess_package.zip
"""

import sys
import zipfile
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent

# Files required for preprocess_zip.py
REQUIRED_FILES = [
    # Main scripts
    "scripts/preprocess_zip.py",
    "scripts/package_caches.py",
    # Preprocessing module
    "src/__init__.py",
    "src/preprocessing/__init__.py",
    "src/preprocessing/cache_manager.py",
    "src/preprocessing/channel_selection.py",
    "src/preprocessing/data_loader.py",
    "src/preprocessing/experiment_config.py",
    "src/preprocessing/filtering.py",
    "src/preprocessing/resampling.py",
    # Utils module (注意: __init__.py 使用精简版，见 UTILS_INIT_CONTENT)
    "src/utils/logging.py",
    "src/utils/timing.py",
    # Data files
    "data/biosemi128.ELC",
]

# Minimal requirements for preprocessing
REQUIREMENTS = """# EEG-BCI 预处理依赖
# 安装命令: pip install -r requirements.txt

numpy>=1.24.0
scipy>=1.10.0
h5py>=3.8.0

# PyTorch (仅用于 Dataset 类，不需要 GPU)
# 选择适合你平台的安装方式:
#
# macOS / Linux / Windows (CPU-only):
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
#
# Windows/Linux with NVIDIA GPU (可选，预处理不需要):
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
#
# 或使用默认安装 (自动选择):
#   pip install torch

torch>=2.0.0
"""

# Minimal utils/__init__.py for preprocessing (excludes training dependencies)
UTILS_INIT_CONTENT = """# Utilities module (preprocessing subset)
\"\"\"
Minimal utilities for EEG-BCI preprocessing.
This is a stripped-down version that excludes training dependencies.
\"\"\"

from .logging import SectionLogger
from .timing import Timer

__all__ = [
    'SectionLogger',
    'Timer',
]
"""

# Main README in Chinese
README_CONTENT = """# EEG-BCI 预处理缓存生成包

本包用于为 EEG-BCI 数据集生成预处理缓存，供后续模型训练使用。

## 项目背景

本项目对比验证 EEG 基座模型 (CBraMod) 与传统 CNN (EEGNet) 在单指级别运动解码任务中的性能。
数据集包含 21 名被试 (S01-S21) 的 EEG 数据，支持 Motor Imagery (MI) 和 Motor Execution (ME) 两种范式。

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境 (推荐)
python -m venv venv

# 激活虚拟环境
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\\Scripts\\activate

# 安装依赖
pip install -r requirements.txt
```

**macOS 用户注意**:
- 完全支持 macOS (Intel 和 Apple Silicon)
- 预处理使用 CPU，不需要 NVIDIA GPU
- Apple Silicon (M1/M2/M3/M4) 用户可使用默认 PyTorch 安装

### 2. 放置数据

有两种使用模式：

#### 模式 A: 标准模式 (ZIP 文件)

分发包解压后与被试 ZIP 文件放在同一目录：

```
<任意文件夹>/                      # 包的父目录 (放置 ZIP 文件和分发包的位置)
├── S01.zip                        # 被试 ZIP 文件
├── S02.zip
├── ...
├── S21.zip
├── S01/                           # 解压后的被试数据 (自动创建，与包同级)
├── S02/
├── ...
│
└── eeg_bci_preprocess_package/    # 解压后的分发包
    ├── data/
    │   └── biosemi128.ELC         # 电极文件 (已包含在分发包中)
    ├── caches/preprocessed/       # 缓存生成在这里
    ├── scripts/
    └── src/
```

#### 模式 B: 就地模式 (已解压数据)

如果数据已经解压，将分发包解压到数据目录内：

```
~/EEG_Data/                        # 数据目录
├── S01/                           # 已解压的被试数据
│   ├── OfflineImagery/
│   └── OnlineImagery_Sess01/
├── S02/
├── ...
├── S21/
│
└── eeg_bci_preprocess_package/    # 分发包解压到这里
    ├── data/
    │   └── biosemi128.ELC
    ├── scripts/
    └── ...
```

然后使用 `--in-place` 标志运行。

### 3. 生成缓存

```bash
# === 模式 A: 标准模式 (从 ZIP 文件) ===

# 处理所有 ZIP 文件 (Motor Imagery, 默认)
python scripts/preprocess_zip.py

# 处理 Motor Execution 范式
python scripts/preprocess_zip.py --paradigm movement

# 仅处理特定被试
python scripts/preprocess_zip.py ../S08.zip ../S09.zip

# === 模式 B: 就地模式 (已解压数据) ===

# 处理所有已解压的被试
python scripts/preprocess_zip.py --in-place

# 处理特定被试
python scripts/preprocess_zip.py --in-place --subject S01

# 强制重新生成 (清除已有缓存)
python scripts/preprocess_zip.py --in-place --force
```

### 4. 打包缓存

生成缓存后，使用以下命令打包成可上传的 ZIP 文件：

```bash
# 默认: 分割成 ≤9GB 的 ZIP 文件
python scripts/package_caches.py

# 自定义大小限制 (单位: GB)
python scripts/package_caches.py --max-size 4.5

# 预览模式 (不创建文件)
python scripts/package_caches.py --dry-run
```

输出文件在 `dist/` 目录：
```
dist/
├── eeg_bci_cache_part01.zip    # ≤9GB
├── eeg_bci_cache_part02.zip    # ≤9GB
├── ...
├── manifest.json               # 文件清单
└── README.md                   # 安装说明
```

## 命令参考

### preprocess_zip.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `zip_files` | 要处理的 ZIP 文件 (位置参数) | 处理父目录下所有 .zip |
| `--paradigm` | 实验范式: imagery 或 movement | imagery |
| `--subject` | 指定被试 ID | - |
| `--in-place` | 就地模式: 包在数据目录内，S01/ 等是 scripts/ 的兄弟目录 | False |
| `--preprocess-only` | 仅预处理 (假设数据在 data/ 子目录) | False |
| `--keep-extracted` | 保留解压后的文件 | False (删除以节省空间) |
| `--force` | 强制重新生成缓存 | False |
| `--tasks` | 任务类型: binary, ternary, all | all |
| `--models` | 模型类型: eegnet, cbramod, all | all |
| `--data-root` | 数据根目录 | data/ (--in-place 时为父目录) |
| `--cache-dir` | 缓存目录 | caches/preprocessed/ |

### package_caches.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--subjects` | 要打包的被试 (支持范围, 如 S08-S21) | 所有被试 |
| `--max-size` | 每个 ZIP 的最大大小 (GB) | 9.0 |
| `--output` | 输出目录 | dist/ |
| `--cache-dir` | 缓存目录 | caches/preprocessed/ |
| `--dry-run` | 预览模式 | False |
| `--no-compression` | 不压缩 (更快但更大) | False |

**被试范围示例**:
```bash
# 单个被试
python scripts/package_caches.py --subjects S08

# 多个被试
python scripts/package_caches.py --subjects S08 S09 S10

# 被试范围
python scripts/package_caches.py --subjects S08-S21

# 混合
python scripts/package_caches.py --subjects S01 S03 S08-S15
```

## 缓存系统说明

### 缓存格式 (v3.0)

- **格式**: HDF5 (.h5) + lzf 压缩
- **存储级别**: Trial 级别 (非 Segment 级别)
- **大小**: 约 50-100 MB/被试/模型
- **滑动窗口**: 加载时动态应用

### 缓存结构

```
caches/preprocessed/
├── .cache_index.json           # 元数据索引
├── {hash}.h5                   # 缓存文件
└── ...
```

每个缓存条目包含：
- `trials`: [n_trials × channels × samples] 数组
- `labels`: [n_trials] 手指标签 (1=拇指, 2=食指, 3=中指, 4=小指)
- 元数据: subject, session, model, version 等

### 缓存键生成规则

缓存键基于以下参数生成 MD5 哈希：
- 被试 ID (subject)
- Run 编号 (run)
- Session 文件夹 (session_folder)
- 目标模型 (eegnet/cbramod)
- 目标类别 (target_classes)
- 预处理配置 (采样率、滤波参数等)

**Offline 数据**: 缓存所有 4 个手指，加载时按 target_classes 过滤
**Online 数据**: 按 target_classes 分别缓存

## 预处理管线

### EEGNet 配置 (Paper-aligned)

| 参数 | 值 |
|------|-----|
| 采样率 | 100 Hz |
| 通道数 | 128 |
| 滤波 | 4-40 Hz Butterworth |
| 归一化 | Z-score (时间轴) |
| 滑动窗口 | 1s, 125ms 步长 |

### CBraMod 配置

| 参数 | 值 |
|------|-----|
| 采样率 | 200 Hz |
| 通道数 | 128 |
| 滤波 | 0.3-75 Hz Butterworth |
| 归一化 | ÷100 |
| 滑动窗口 | 1s, 500ms 步长 |

## 数据划分协议

遵循原论文实验设计：

| 数据来源 | 用途 |
|----------|------|
| Offline + Online_Sess01_* + Online_Sess02_*_Base | 训练 (时序分割 80/20) |
| Online_Sess02_*_Finetune | 测试 (完全独立) |

## 故障排除

### 缺少 biosemi128.ELC

```
错误: Electrode file not found
解决: 将 biosemi128.ELC 复制到 data/ 目录
```

### 内存不足

```bash
# 减少并行 worker 数量
# 编辑 preprocess_zip.py 中的 parallel_workers 参数
```

### 缓存损坏

```bash
# 强制重新生成
python scripts/preprocess_zip.py --force
```

### 磁盘空间不足

默认行为会在缓存生成后删除解压的文件夹。如果仍然空间不足：
1. 分批处理被试
2. 生成缓存后立即打包并删除

## 性能参考

基于测试环境 (19 并行 workers):
- **单被试预处理**: ~1.5 分钟
- **21 被试全量**: ~30 分钟
- **缓存大小**: ~2-3 GB/被试 (EEGNet + CBraMod, binary + ternary)
- **总缓存大小**: ~50-60 GB (21 被试, 双范式)

## 输出文件用途

生成的缓存 ZIP 文件应发送给主项目维护者，用于：
1. 训练 EEGNet 和 CBraMod 模型
2. 跨被试性能对比分析
3. 论文结果复现

## 联系方式

如有问题，请联系项目维护者或在 GitHub Issues 中提交。
"""

# CLAUDE.md for AI coding assistants
CLAUDE_MD_CONTENT = """# CLAUDE.md - AI 编码助手指南

本文件为 AI 编码助手 (如 Claude Code) 提供项目上下文。

## 语言规范

默认使用中文。技术术语可附带英文。

## 项目目的

本包是 EEG-BCI 项目的预处理子集，用于在远程机器上生成预处理缓存。

## 平台支持

| 平台 | 支持状态 | 说明 |
|------|----------|------|
| Windows | ✅ 完全支持 | 需 Python 3.9+ |
| Linux | ✅ 完全支持 | 需 Python 3.9+ |
| macOS Intel | ✅ 完全支持 | 需 Python 3.9+ |
| macOS Apple Silicon | ✅ 完全支持 | M1/M2/M3/M4 均支持 |

**GPU 要求**: 预处理**不需要** GPU。PyTorch 仅用于 Dataset 类 (CPU 友好)。

## 文件结构

### 标准模式 (ZIP 文件)
```
<parent_folder>/                   # 包的父目录 (--zip-dir AND --data-root 默认)
├── S01.zip, S02.zip, ...          # 被试 ZIP 文件放这里
├── S01/, S02/, ...                # 解压后的被试数据 (与包同级)
│
└── eeg_bci_preprocess_package/    # 解压后的分发包
    ├── scripts/
    ├── src/
    ├── data/
    │   └── biosemi128.ELC         # 电极文件 (已包含)
    └── caches/preprocessed/       # 缓存输出
```

### 就地模式 (--in-place)
```
~/EEG_Data/                        # 数据目录 (自动成为 data-root)
├── S01/, S02/, ..., S21/          # 已解压的被试数据
│
└── eeg_bci_preprocess_package/    # 分发包解压到这里
    ├── scripts/
    ├── src/
    ├── data/
    │   └── biosemi128.ELC
    └── caches/preprocessed/
```

## 关键命令

```bash
# 标准模式: 从 ZIP 文件生成缓存
python scripts/preprocess_zip.py

# 就地模式: 数据已解压，包在数据目录内
python scripts/preprocess_zip.py --in-place

# 指定范式
python scripts/preprocess_zip.py --in-place --paradigm movement

# 打包缓存
python scripts/package_caches.py
```

## 预处理管线概述

1. **ZIP 解压**: 从 data/*.zip 解压被试数据
2. **MAT 文件加载**: 读取 EEG 数据和事件标记
3. **Trial 提取**: 根据事件提取试验段
4. **CAR 参考**: 应用共同平均参考
5. **重采样**: 降采样到目标频率 (100/200 Hz)
6. **缓存保存**: 以 HDF5 格式保存到 caches/preprocessed/
7. **清理**: 删除解压的文件夹 (可选)

## 缓存系统 (v3.0)

- **存储级别**: Trial (非 Segment)
- **格式**: HDF5 + lzf 压缩
- **Offline 数据**: 缓存所有 4 个手指，加载时过滤
- **Online 数据**: 按 target_classes 分别缓存
- **滑动窗口**: 在 cache load 时动态应用

## 模型配置

| 模型 | 采样率 | 滤波 | 归一化 |
|------|--------|------|--------|
| EEGNet | 100 Hz | 4-40 Hz | Z-score |
| CBraMod | 200 Hz | 0.3-75 Hz | ÷100 |

## 常见任务

### 为新被试生成缓存
```bash
python scripts/preprocess_zip.py data/S15.zip
```

### 重新生成特定被试缓存
```bash
python scripts/preprocess_zip.py --subject S01 --preprocess-only --force
```

### 仅生成 CBraMod 缓存
```bash
python scripts/preprocess_zip.py --models cbramod
```

### 打包成小于 5GB 的文件
```bash
python scripts/package_caches.py --max-size 5
```

### 仅打包特定被试范围 (如 S08-S21)
```bash
python scripts/package_caches.py --subjects S08-S21
```

## 错误处理

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| Electrode file not found | 缺少 biosemi128.ELC | 复制到 data/ 目录 |
| Subject directory not found | 数据未解压或路径错误 | 检查数据位置 |
| Cache corrupted | 缓存文件损坏 | 使用 --force 重新生成 |

## 注意事项

1. 本包是完整 EEG-BCI 项目的子集，仅包含预处理所需文件
2. 生成的缓存需要发送给主项目维护者
3. 不要修改 src/ 下的代码，除非有特殊需求
4. biosemi128.ELC 文件必须与数据集一起提供
"""


def create_package():
    """Create the distribution package."""
    dist_dir = PROJECT_ROOT / "dist"
    dist_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d")
    package_name = f"eeg_bci_preprocess_package_{timestamp}.zip"
    package_path = dist_dir / package_name

    print(f"Creating package: {package_path}")

    # Check all required files exist
    missing = []
    for file_path in REQUIRED_FILES:
        full_path = PROJECT_ROOT / file_path
        if not full_path.exists():
            missing.append(file_path)

    if missing:
        print("ERROR: Missing required files:")
        for f in missing:
            print(f"  - {f}")
        return 1

    # Create ZIP package
    with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add Python files
        for file_path in REQUIRED_FILES:
            full_path = PROJECT_ROOT / file_path
            zf.write(full_path, file_path)
            print(f"  Added: {file_path}")

        # Add minimal utils/__init__.py (excludes training dependencies)
        zf.writestr("src/utils/__init__.py", UTILS_INIT_CONTENT)
        print("  Added: src/utils/__init__.py (minimal)")

        # Add requirements.txt
        zf.writestr("requirements.txt", REQUIREMENTS)
        print("  Added: requirements.txt")

        # Add README (Chinese)
        zf.writestr("README.md", README_CONTENT)
        print("  Added: README.md")

        # Add CLAUDE.md for AI assistants
        zf.writestr("CLAUDE.md", CLAUDE_MD_CONTENT)
        print("  Added: CLAUDE.md")

        # Create empty directories structure
        zf.writestr("data/.gitkeep", "# 放置 S*.zip 和 biosemi128.ELC\n")
        zf.writestr("caches/.gitkeep", "# 缓存将生成在此目录\n")
        zf.writestr("dist/.gitkeep", "# 打包的缓存文件将放置在此\n")

    # Get package size
    size_mb = package_path.stat().st_size / (1024 * 1024)
    print(f"\nPackage created: {package_path}")
    print(f"Size: {size_mb:.2f} MB")
    print(f"\nFiles included: {len(REQUIRED_FILES) + 3}")  # +3 for requirements, README, CLAUDE.md

    return 0


if __name__ == "__main__":
    sys.exit(create_package())
