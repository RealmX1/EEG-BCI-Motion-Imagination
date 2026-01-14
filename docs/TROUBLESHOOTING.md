# 故障排除指南

本文档收录 EEG-BCI 项目的常见问题及解决方案。

## 目录

- [CUDA / GPU 问题](#cuda--gpu-问题)
- [PyTorch 安装问题](#pytorch-安装问题)
- [uv 相关问题](#uv-相关问题)
- [依赖问题](#依赖问题)
- [数据相关问题](#数据相关问题)
- [训练问题](#训练问题)

---

## CUDA / GPU 问题

### CUDA 不可用 ("CUDA is not available")

**症状**:
```
RuntimeError: No CUDA GPUs are available
```

**排查步骤**:

1. **确认有 NVIDIA GPU**:
   ```bash
   nvidia-smi
   ```
   如果命令不存在，需要安装 NVIDIA 驱动。

2. **检查驱动版本**:
   - CUDA 12.8 需要驱动 >= 570.x
   - CUDA 12.4 需要驱动 >= 550.x

3. **确认 PyTorch 安装了 CUDA 版本**:
   ```python
   import torch
   print(torch.__version__)  # 应该包含 +cu12x
   print(torch.cuda.is_available())
   ```

**解决方案**:

重新安装正确的 PyTorch 版本:
```bash
# 先卸载
uv pip uninstall torch torchvision torchaudio

# RTX 50 系列 (5070/5080/5090)
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# RTX 40/30/20 系列
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

---

### CUDA out of memory

**症状**:
```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate X.XX GiB
```

**解决方案**:

1. **减小 batch size**:
   修改配置文件 `configs/eegnet_config.yaml` 或 `configs/cbramod_config.yaml`:
   ```yaml
   batch_size: 16  # 从 32 减小到 16
   ```

2. **使用 19 通道 CBraMod** (显存需求较低):
   ```bash
   uv run python -m src.training.train_within_subject --model cbramod --cbramod-channels 19
   ```

3. **关闭其他占用 GPU 的程序**:
   ```bash
   nvidia-smi  # 查看 GPU 使用情况
   ```

4. **清理 GPU 缓存** (在代码中):
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

---

### CUDA 版本不匹配

**症状**:
```
NVIDIA GeForce RTX XXXX with CUDA capability sm_XX is not compatible with the current PyTorch installation
```

**解决方案**:

| GPU 系列 | 推荐 CUDA 版本 | PyTorch 安装命令 |
|----------|---------------|-----------------|
| RTX 50xx | 12.8 (nightly) | `uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128` |
| RTX 40xx | 12.4 | `uv pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| RTX 30xx | 12.4 或 12.1 | `uv pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| RTX 20xx | 12.1 | `uv pip install torch --index-url https://download.pytorch.org/whl/cu121` |

---

## PyTorch 安装问题

### 安装了 CPU 版本

**症状**:
```python
>>> import torch
>>> torch.cuda.is_available()
False
>>> torch.__version__
'2.x.x+cpu'  # 注意 +cpu 后缀
```

**解决方案**:

1. 卸载当前版本:
   ```bash
   uv pip uninstall torch torchvision torchaudio
   ```

2. 安装 CUDA 版本 (见上方表格)

### torch 导入失败

**症状**:
```
ImportError: DLL load failed while importing torch
```

**解决方案** (Windows):

1. 安装 Microsoft Visual C++ Redistributable:
   https://aka.ms/vs/17/release/vc_redist.x64.exe

2. 重启终端后重试

---

## uv 相关问题

### uv 命令未找到

**症状**:
```
'uv' is not recognized as an internal or external command
```

**解决方案**:

**Windows (PowerShell)**:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

安装后重启终端。

### uv sync 失败

**症状**:
```
error: Failed to download package
```

**解决方案**:

1. 清理缓存:
   ```bash
   uv cache clean
   ```

2. 更新 uv:
   ```bash
   uv self update
   ```

3. 检查网络连接，必要时使用镜像源

---

## 依赖问题

### MNE 安装失败

**症状**:
```
error: Could not build wheels for mne
```

**解决方案**:

确保安装了构建工具:

**Windows**:
安装 Visual Studio Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/

**Linux**:
```bash
sudo apt-get install build-essential
```

### h5py 安装失败

**症状**:
```
ERROR: Could not build wheels for h5py
```

**解决方案**:

**Windows**: 使用预编译包
```bash
uv pip install h5py --only-binary :all:
```

**Linux**:
```bash
sudo apt-get install libhdf5-dev
uv pip install h5py
```

---

## 数据相关问题

### 找不到数据文件

**症状**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/S01/...'
```

**解决方案**:

1. 确认数据目录结构:
   ```
   data/
   ├── S01/
   │   ├── OfflineImagery/
   │   │   └── S01_OfflineImagery_R01.mat
   │   └── OnlineImagery_Sess01_2class_Base/
   ├── S02/
   └── ...
   ```

2. 确认从项目根目录运行命令:
   ```bash
   cd /path/to/EEG-BCI
   uv run python scripts/run_full_comparison.py
   ```

### MAT 文件读取错误

**症状**:
```
scipy.io.matlab._mio5.MatReadError: ...
```

**解决方案**:

数据文件可能是 MATLAB v7.3 格式 (HDF5)。确认 h5py 已安装:
```bash
uv pip install h5py
```

### 缓存问题

**症状**:
预处理结果与预期不符，或出现奇怪的数据错误。

**解决方案**:

清理预处理缓存:
```bash
# 查看缓存状态
uv run python scripts/cache_helper.py --stats

# 清理特定模型的缓存
uv run python scripts/cache_helper.py --model eegnet --execute

# 清理所有缓存
uv run python scripts/cache_helper.py --all --execute
```

---

## 训练问题

### 训练中断

**症状**:
训练突然停止，无错误信息。

**可能原因**:

1. **显存不足**: 减小 batch size
2. **系统休眠**: 禁用自动休眠
3. **终端断开**: 使用 `screen` 或 `tmux` 保持会话

### 验证准确率为 0 或 NaN

**症状**:
```
Val Acc: 0.0000 或 nan
```

**解决方案**:

1. 检查数据加载是否正确:
   ```bash
   uv run python -c "from src.preprocessing.data_loader import load_subject_data; print(load_subject_data('S01', 'binary', 'eegnet'))"
   ```

2. 检查类别是否平衡:
   二分类任务应该有大致相等的正负样本

3. 降低学习率:
   修改配置文件中的 `learning_rate`

### CBraMod 模型未找到

**症状**:
```
ModuleNotFoundError: No module named 'cbramod'
```

**解决方案**:

安装 CBraMod:
```bash
git clone https://github.com/wjq-learning/CBraMod.git
uv pip install -e CBraMod/
```

---

## 获取帮助

如果以上方案未能解决问题:

1. 运行环境验证脚本获取详细诊断:
   ```bash
   uv run python scripts/verify_installation.py
   ```

2. 在 GitHub Issues 中提问，并附上:
   - 完整的错误信息
   - `verify_installation.py` 的输出
   - 操作系统和 GPU 型号
