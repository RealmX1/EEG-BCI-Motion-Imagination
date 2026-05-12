"""
EEG 电极布局可视化模块.

提供 2D/3D 头部模型上的电极位置绘制功能，
用于可视化和对比不同通道选择配置.

使用 MNE-Python 的标准 montage 和坐标系:
- `make_standard_montage('biosemi128')` 加载内置 BioSemi 128 通道位置
- 方位角等距投影 (Azimuthal Equidistant) 用于 2D topomap
- MNE Info 对象可直接用于 MNE 生态系统 (plot_sensors, plot_topomap 等)

支持:
- 单配置电极布局图 (2D)
- 多配置并排对比图 (2D)
- 配置重叠分析热力图 (2D)
- 3D 头部模型视图 (matplotlib Axes3D)
"""

import logging
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np

from ..preprocessing.channel_selection import BIOSEMI_128_LABELS
from ..utils.logging import SectionLogger

logger = logging.getLogger(__name__)
log_plot = SectionLogger(logger, 'electrode')

# ============================================================================
# 常量
# ============================================================================

# 32ch 配置颜色映射 (视觉区分度高的色板)
CONFIG_COLORS: Dict[str, str] = {
    'motor_cortex':      '#2E86AB',   # 蓝 — 手动: 运动皮层
    'commercial':        '#A23B72',   # 紫红 — 手动: 商业 10-20
    'fdr':               '#F18F01',   # 橙 — 数据驱动: Fisher
    'csp':               '#C73E1D',   # 红 — 数据驱动: CSP
    'attention':         '#3B1F2B',   # 深棕 — 数据驱动: 注意力
    'band_power':        '#44BBA4',   # 青绿 — 数据驱动: 频带功率
    'fdr_complement':    '#7B7B7B',   # 灰 — 对照: FDR 补集
    'negative_control':  '#4A4A4A',   # 深灰 — 对照: 阴性对照
    'standard_1010':     '#6A0DAD',   # 紫 — 标准 10-10 系统 (61ch)
}

CONFIG_DISPLAY_NAMES: Dict[str, str] = {
    'motor_cortex':      'Motor Cortex',
    'commercial':        'Commercial 10-20',
    'fdr':               'FDR',
    'csp':               'CSP',
    'attention':         'Attention',
    'band_power':        'Band Power',
    'fdr_complement':    'FDR Complement',
    'negative_control':  'Negative Control',
    'standard_1010':     'Standard 10-10 (61ch)',
}

# 脑区定义: 名称 → (颜色, alpha)
BRAIN_REGIONS = {
    'Frontal':   ('#AEC6CF', 0.15),  # 淡蓝
    'Central':   ('#FFD1DC', 0.15),  # 淡粉
    'Parietal':  ('#B5EAD7', 0.15),  # 淡绿
    'Temporal':  ('#FFDAC1', 0.15),  # 淡橙
    'Occipital': ('#E2C2F8', 0.15),  # 淡紫
}

# 10-20 地标参考点 (用于标注在图上)
# 基于 MNE standard_1020 与 biosemi128 montage 的最近邻匹配验证
LANDMARK_1020 = {
    'Cz':  'A1',   'C3':  'D19',  'C4':  'B22',
    'Fz':  'C21',  'Pz':  'A19',  'Oz':  'A22',
    'Fp1': 'C29',  'Fp2': 'C16',
    'F3':  'D4',   'F4':  'C4',
    'T3':  'D24',  'T4':  'B14',
    'P3':  'A18',  'P4':  'B4',
    'O1':  'A16',  'O2':  'A29',
}


# ============================================================================
# MNE 基础设施
# ============================================================================

_BIOSEMI_128_LABELS_SET = set(BIOSEMI_128_LABELS)


def create_mne_montage(
    elc_path: Optional[str] = None,
) -> mne.channels.DigMontage:
    """
    创建 BioSemi 128 通道 montage.

    默认使用 MNE 内置 montage (无需文件), 可选传入自定义 ELC 路径.

    Args:
        elc_path: 可选的 biosemi128.ELC 文件路径. None 使用 MNE 内置.

    Returns:
        BioSemi 128 通道 DigMontage 对象
    """
    if elc_path is not None:
        return mne.channels.read_custom_montage(elc_path, head_size=0.095)
    return mne.channels.make_standard_montage('biosemi128')


def create_mne_info(
    elc_path: Optional[str] = None,
) -> mne.Info:
    """
    创建 BioSemi 128 通道 MNE Info 对象.

    Info 对象是 MNE 生态的核心数据结构, 可直接用于
    plot_sensors / plot_topomap / plot_alignment 等函数.

    Args:
        elc_path: 可选的 biosemi128.ELC 文件路径. None 使用 MNE 内置.

    Returns:
        MNE Info 对象 (128 EEG 通道, 带 montage)
    """
    montage = create_mne_montage(elc_path)
    info = mne.create_info(
        ch_names=montage.ch_names,
        sfreq=100.0,
        ch_types='eeg',
    )
    info.set_montage(montage)
    return info


# ============================================================================
# 坐标投影
# ============================================================================

def cartesian_to_2d(positions_3d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    3D 笛卡尔坐标 → 2D 方位角等距投影 (Azimuthal Equidistant).

    EEG topomap 标准投影: 从头顶 (Z+) 俯视,
    鼻子朝上 (Y+ → 2D 上方), 右耳朝右 (X+ → 2D 右方).

    兼容 MNE 坐标系 (米) 和 BioSemi ELC 坐标系 (毫米),
    因为内部做了归一化处理.

    Args:
        positions_3d: 电极标签 → 3D 坐标的映射

    Returns:
        电极标签 → 2D 坐标的映射 (归一化到 ~[-0.5, 0.5])
    """
    # 中心化 + 归一化到单位球
    all_pos = np.array(list(positions_3d.values()))
    center = all_pos.mean(axis=0)

    centered = {k: v - center for k, v in positions_3d.items()}
    max_r = max(np.linalg.norm(v) for v in centered.values())

    result = {}
    for label, pos in centered.items():
        pos_norm = pos / max_r
        x, y, z = pos_norm

        r_sphere = np.sqrt(x**2 + y**2 + z**2)
        if r_sphere < 1e-10:
            result[label] = np.array([0.0, 0.0])
            continue

        # 极角: 从顶点 (Z+) 的角距离
        polar_angle = np.arccos(np.clip(z / r_sphere, -1, 1))

        # 2D 半径与极角成正比 (方位角等距)
        r_2d = polar_angle / np.pi  # [0, 1]

        # 方位角: atan2(x, y) 使 Y+ (前方) 对应 2D 上方
        azimuth = np.arctan2(x, y)

        x_2d = r_2d * np.sin(azimuth)
        y_2d = r_2d * np.cos(azimuth)

        result[label] = np.array([x_2d, y_2d])

    return result


def load_electrode_positions_2d(
    elc_path: Optional[str] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    加载 BioSemi 128 通道位置并计算 2D 投影.

    使用 MNE 内置 biosemi128 montage (默认) 或自定义 ELC 文件.

    Args:
        elc_path: 可选的 biosemi128.ELC 文件路径. None 使用 MNE 内置.

    Returns:
        (positions_2d, positions_3d)
        - positions_2d: 标签 → 2D 坐标
        - positions_3d: 标签 → 3D 坐标 (MNE 坐标系, 米)
    """
    montage = create_mne_montage(elc_path)
    ch_pos = montage.get_positions()['ch_pos']

    # 仅保留 BioSemi 128 通道 (排除 fiducial 等额外点)
    positions_3d = {
        name: pos for name, pos in ch_pos.items()
        if name in _BIOSEMI_128_LABELS_SET
    }
    positions_2d = cartesian_to_2d(positions_3d)
    return positions_2d, positions_3d


# ============================================================================
# 脑区分类
# ============================================================================

def classify_electrode_region(x_2d: float, y_2d: float) -> str:
    """
    根据 2D 投影坐标将电极分配到脑区.

    区域边界基于标准 10-20 地标位置校准:
    - Temporal: |x| > 0.30 (T3/T4 在 |x|≈0.50)
    - Frontal:  y > 0.15 (Fz 在 y≈0.26)
    - Occipital: y < -0.32 (Oz 在 y≈-0.47)
    - Central:  y >= -0.05 (Cz 在 y≈0.00)
    - Parietal:  其余 (Pz 在 y≈-0.26)
    """
    if abs(x_2d) > 0.30:
        return 'Temporal'
    if y_2d > 0.15:
        return 'Frontal'
    if y_2d < -0.32:
        return 'Occipital'
    if y_2d >= -0.05:
        return 'Central'
    return 'Parietal'


def get_region_counts(
    positions_2d: Dict[str, np.ndarray],
    indices: List[int],
) -> Dict[str, int]:
    """统计选中通道在各脑区的数量."""
    labels = BIOSEMI_128_LABELS
    counts = {r: 0 for r in BRAIN_REGIONS}
    for i in indices:
        x, y = positions_2d[labels[i]]
        region = classify_electrode_region(x, y)
        counts[region] += 1
    return counts


# ============================================================================
# 10-20 地标 & 脑区着色
# ============================================================================

def draw_1020_landmarks(
    ax,
    positions_2d: Dict[str, np.ndarray],
    fontsize: float = 6.5,
) -> None:
    """在 2D 头部模型上标注 10-20 系统地标参考点."""
    labels_map = {biosemi: name_1020
                  for name_1020, biosemi in LANDMARK_1020.items()}

    for biosemi_label, name_1020 in labels_map.items():
        if biosemi_label not in positions_2d:
            continue
        x, y = positions_2d[biosemi_label]
        ax.plot(x, y, marker='d', markersize=4,
                color='black', markeredgecolor='black',
                markerfacecolor='white', zorder=5, alpha=0.7)
        ax.annotate(
            name_1020, (x, y),
            xytext=(0, -8), textcoords='offset points',
            fontsize=fontsize, ha='center', va='top',
            color='#444444', fontstyle='italic',
            zorder=5,
        )


def draw_brain_regions(
    ax,
    head_radius: float = 0.5,
) -> None:
    """
    在 2D 头部模型上绘制半透明脑区分区.

    使用 matplotlib 多边形填充，裁剪到头部圆内.
    """
    from matplotlib.patches import Circle, Polygon
    from matplotlib.collections import PatchCollection

    clip_circle = Circle((0, 0), head_radius, transform=ax.transData,
                         fill=False, visible=False)
    ax.add_patch(clip_circle)

    # 区域定义: (y_bottom, y_top, x_left, x_right)
    # 使用足够大的 x 范围，由 clip_circle 裁剪
    w = head_radius + 0.1  # 比头圆宽
    region_bounds = {
        'Frontal':   ( 0.15,  w,    -w, w),
        'Central':   (-0.05,  0.15, -0.30, 0.30),
        'Parietal':  (-0.32, -0.05, -0.30, 0.30),
        'Occipital': (-w,    -0.32, -w, w),
        'Temporal':  None,  # 特殊处理
    }

    for region_name, bounds in region_bounds.items():
        color, alpha = BRAIN_REGIONS[region_name]

        if region_name == 'Temporal':
            # 颞叶: 左右两个条带
            for x_sign in [-1, 1]:
                x_inner = x_sign * 0.30
                x_outer = x_sign * w
                rect_xy = [
                    (min(x_inner, x_outer), -0.32),
                    (max(x_inner, x_outer), -0.32),
                    (max(x_inner, x_outer),  0.15),
                    (min(x_inner, x_outer),  0.15),
                ]
                patch = Polygon(rect_xy, closed=True,
                                facecolor=color, alpha=alpha,
                                edgecolor='none', zorder=0)
                patch.set_clip_path(clip_circle)
                ax.add_patch(patch)
        else:
            y_bot, y_top, x_left, x_right = bounds
            rect_xy = [
                (x_left,  y_bot),
                (x_right, y_bot),
                (x_right, y_top),
                (x_left,  y_top),
            ]
            patch = Polygon(rect_xy, closed=True,
                            facecolor=color, alpha=alpha,
                            edgecolor='none', zorder=0)
            patch.set_clip_path(clip_circle)
            ax.add_patch(patch)

    # 区域标签 (小字, 边缘位置)
    region_label_pos = {
        'Frontal':   (0.0,  0.40),
        'Central':   (0.0,  0.07),
        'Parietal':  (0.0, -0.18),
        'Temporal':  (-0.42, 0.05),
        'Occipital': (0.0, -0.42),
    }
    for region_name, (lx, ly) in region_label_pos.items():
        color, _ = BRAIN_REGIONS[region_name]
        ax.text(lx, ly, region_name[0],  # 首字母缩写
                fontsize=14, ha='center', va='center',
                color=color, alpha=0.6, fontweight='bold',
                zorder=0)


# ============================================================================
# 脑区分布柱状图
# ============================================================================

def plot_region_distribution(
    configs: Dict[str, List[int]],
    positions_2d: Dict[str, np.ndarray],
    output_path: str,
) -> None:
    """
    生成各配置的脑区分布对比柱状图.

    分组柱状图: x 轴=脑区, 每组内多根柱子=各配置.

    Args:
        configs: 配置名称 → 通道索引列表
        positions_2d: 128 通道 2D 坐标
        output_path: 输出路径
    """
    import matplotlib.pyplot as plt

    region_names = list(BRAIN_REGIONS.keys())
    config_names = list(configs.keys())
    n_regions = len(region_names)
    n_configs = len(config_names)

    # 统计
    data = {}
    for cfg_name, indices in configs.items():
        data[cfg_name] = get_region_counts(positions_2d, indices)

    # 绘图
    fig, ax = plt.subplots(figsize=(max(10, n_configs * 1.5), 5))
    x = np.arange(n_regions)
    bar_width = 0.8 / n_configs
    offsets = np.linspace(-(n_configs - 1) / 2, (n_configs - 1) / 2, n_configs) * bar_width

    for i, cfg_name in enumerate(config_names):
        counts = [data[cfg_name][r] for r in region_names]
        color = CONFIG_COLORS.get(cfg_name, '#888888')
        display = CONFIG_DISPLAY_NAMES.get(cfg_name, cfg_name)
        bars = ax.bar(x + offsets[i], counts, bar_width * 0.9,
                      label=display, color=color, edgecolor='black',
                      linewidth=0.5, alpha=0.85)
        # 数值标签
        for bar, val in zip(bars, counts):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        str(val), ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(region_names, fontsize=10)
    ax.set_ylabel('Number of Channels', fontsize=11)
    ax.set_title('Channel Distribution by Brain Region', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right', ncol=2)
    ax.set_ylim(0, max(max(data[c][r] for r in region_names) for c in config_names) + 3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    log_plot.info(f"Region distribution plot saved: {output_path}")
    plt.close()


# ============================================================================
# 头部轮廓
# ============================================================================

def draw_head_outline(
    ax,
    head_radius: float = 0.5,
    linewidth: float = 1.5,
    color: str = 'black',
) -> None:
    """
    在 2D 轴上绘制头部轮廓 (圆 + 鼻子 + 耳朵).

    几何参数与 MNE `_make_head_outlines()` 保持一致:
    - 头圆: 101 个点的标准圆
    - 鼻子: 尖端在 radius × 1.15 处
    - 耳朵: MNE 标准归一化坐标 (缩放到 radius × 2)

    Args:
        ax: matplotlib Axes
        head_radius: 头部圆半径
        linewidth: 线宽
        color: 线颜色
    """
    # 头部圆 (MNE 使用 101 个点)
    theta = np.linspace(0, 2 * np.pi, 101)
    ax.plot(head_radius * np.cos(theta),
            head_radius * np.sin(theta),
            color=color, linewidth=linewidth, zorder=1)

    # 鼻子 (MNE 风格: 尖端在 radius * 1.15)
    nose_w = 0.08 * head_radius
    ax.plot([-nose_w, 0, nose_w],
            [head_radius, head_radius * 1.15, head_radius],
            color=color, linewidth=linewidth, zorder=1)

    # 耳朵 (MNE _make_head_outlines 归一化坐标)
    ear_x = np.array([.497, .510, .518, .5299, .5419,
                       .54, .547, .532, .510, .489])
    ear_y = np.array([.0555, .0775, .0783, .0746, .0555,
                       -.0055, -.0932, -.1313, -.1384, -.1199])
    scale = head_radius * 2
    # 左耳
    ax.plot(-ear_x * scale, ear_y * scale,
            color=color, linewidth=linewidth, zorder=1)
    # 右耳
    ax.plot(ear_x * scale, ear_y * scale,
            color=color, linewidth=linewidth, zorder=1)


# ============================================================================
# 2D 单配置绘制
# ============================================================================

def plot_electrode_placement_2d(
    ax,
    positions_2d: Dict[str, np.ndarray],
    selected_indices: List[int],
    config_name: str = '',
    show_all_128: bool = True,
    show_labels: bool = True,
    show_landmarks: bool = True,
    show_regions: bool = True,
    highlight_color: Optional[str] = None,
    marker_size_selected: float = 80,
    marker_size_background: float = 15,
) -> None:
    """
    在 2D 头部模型上绘制单个配置的电极布局.

    Args:
        ax: matplotlib Axes
        positions_2d: 全部 128 电极的 2D 坐标
        selected_indices: 选中通道的索引 (0-127)
        config_name: 配置名称
        show_all_128: 是否显示 128 通道背景点
        show_labels: 是否显示选中电极的标签
        show_landmarks: 是否显示 10-20 地标参考点
        show_regions: 是否显示脑区分区着色
        highlight_color: 选中电极颜色 (None 使用 CONFIG_COLORS)
        marker_size_selected: 选中电极 marker 大小
        marker_size_background: 背景电极 marker 大小
    """
    labels = BIOSEMI_128_LABELS
    selected_set = set(selected_indices)
    color = highlight_color or CONFIG_COLORS.get(config_name, '#E94F37')

    # 脑区着色 (最底层)
    if show_regions:
        draw_brain_regions(ax)

    draw_head_outline(ax)

    # 128ch 背景 (灰色小点)
    if show_all_128:
        bg_x = [positions_2d[labels[i]][0] for i in range(128) if i not in selected_set]
        bg_y = [positions_2d[labels[i]][1] for i in range(128) if i not in selected_set]
        ax.scatter(bg_x, bg_y, s=marker_size_background,
                   c='lightgray', edgecolors='gray', linewidths=0.3,
                   alpha=0.5, zorder=2)

    # 选中通道 (彩色大点)
    sel_x = [positions_2d[labels[i]][0] for i in selected_indices]
    sel_y = [positions_2d[labels[i]][1] for i in selected_indices]
    ax.scatter(sel_x, sel_y, s=marker_size_selected,
               c=color, edgecolors='black', linewidths=0.8,
               alpha=0.9, zorder=3)

    # 电极标签
    if show_labels:
        for i in selected_indices:
            x, y = positions_2d[labels[i]]
            ax.annotate(labels[i], (x, y),
                        xytext=(3, 3), textcoords='offset points',
                        fontsize=5, ha='left', va='bottom',
                        color='black', alpha=0.8)

    # 10-20 地标
    if show_landmarks:
        draw_1020_landmarks(ax, positions_2d)

    # 标题
    display_name = CONFIG_DISPLAY_NAMES.get(config_name, config_name)
    ax.set_title(f'{display_name}\n({len(selected_indices)} channels)',
                 fontsize=10, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xlim([-0.65, 0.65])
    ax.set_ylim([-0.65, 0.75])
    ax.axis('off')


# ============================================================================
# 2D 多配置网格
# ============================================================================

def plot_electrode_grid(
    configs: Dict[str, List[int]],
    positions_2d: Dict[str, np.ndarray],
    output_path: str,
    ncols: int = 3,
    show_labels: bool = False,
    suptitle: Optional[str] = None,
) -> None:
    """
    生成多配置并排网格对比图.

    Args:
        configs: 配置名称 → 通道索引列表
        positions_2d: 128 电极 2D 坐标
        output_path: 输出 PNG 路径
        ncols: 每行列数
        show_labels: 是否显示电极标签
        suptitle: 总标题
    """
    import matplotlib.pyplot as plt

    n = len(configs)
    nrows = (n + ncols - 1) // ncols
    remainder = n % ncols
    full_rows = n // ncols

    # 用 2*ncols 等宽子列的 gridspec：每个 panel 跨 2 列；
    # 最后一行不满时把 remainder 个 panel 水平偏移 (ncols - remainder) 列实现居中。
    fig = plt.figure(figsize=(5 * ncols, 5.5 * nrows))
    gs = fig.add_gridspec(nrows, 2 * ncols)

    axes_list = []
    for r in range(full_rows):
        for c in range(ncols):
            axes_list.append(fig.add_subplot(gs[r, 2 * c : 2 * c + 2]))
    if remainder > 0:
        offset = ncols - remainder
        for c in range(remainder):
            start = offset + 2 * c
            axes_list.append(fig.add_subplot(gs[full_rows, start : start + 2]))

    for ax, (config_name, indices) in zip(axes_list, configs.items()):
        plot_electrode_placement_2d(
            ax, positions_2d, indices,
            config_name=config_name,
            show_labels=show_labels,
        )

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    log_plot.info(f"Electrode grid plot saved: {output_path}")
    plt.close()


# ============================================================================
# 2D 重叠分析
# ============================================================================

def plot_electrode_overlap(
    configs: Dict[str, List[int]],
    positions_2d: Dict[str, np.ndarray],
    output_path: str,
) -> None:
    """
    生成通道重叠分析图 (2 面板).

    左: 热力图头部模型 — 颜色深度 = 被多少配置选中
    右: 配置间重叠矩阵 — 上三角=|A∩B|, 下三角=Jaccard, 对角=|config|

    Args:
        configs: 配置名称 → 通道索引列表
        positions_2d: 128 通道 2D 坐标
        output_path: 输出路径
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    fig, (ax_head, ax_matrix) = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={'width_ratios': [1, 1.2]},
    )

    labels = BIOSEMI_128_LABELS

    # --- 左面板: 热力图头部模型 ---
    overlap_count = np.zeros(128, dtype=int)
    for indices in configs.values():
        for idx in indices:
            overlap_count[idx] += 1

    draw_head_outline(ax_head)

    # 未被选中的通道
    unselected = [i for i in range(128) if overlap_count[i] == 0]
    if unselected:
        ux = [positions_2d[labels[i]][0] for i in unselected]
        uy = [positions_2d[labels[i]][1] for i in unselected]
        ax_head.scatter(ux, uy, s=15, c='lightgray', edgecolors='gray',
                        linewidths=0.3, alpha=0.4, zorder=2)

    # 被选中的通道 (按频次着色)
    selected = [i for i in range(128) if overlap_count[i] > 0]
    if selected:
        sx = [positions_2d[labels[i]][0] for i in selected]
        sy = [positions_2d[labels[i]][1] for i in selected]
        counts = [overlap_count[i] for i in selected]
        n_configs = len(configs)
        norm = Normalize(vmin=0.5, vmax=n_configs + 0.5)

        sc = ax_head.scatter(sx, sy, s=80, c=counts, cmap='YlOrRd', norm=norm,
                             edgecolors='black', linewidths=0.6, zorder=3)

        cbar = fig.colorbar(sc, ax=ax_head, shrink=0.7, pad=0.02)
        cbar.set_label('# configs selecting channel', fontsize=8)
        cbar.set_ticks(range(1, n_configs + 1))

    ax_head.set_title(f'Channel Selection Overlap\n({len(configs)} configs)', fontsize=11)
    ax_head.set_aspect('equal')
    ax_head.set_xlim([-0.65, 0.65])
    ax_head.set_ylim([-0.65, 0.75])
    ax_head.axis('off')

    # --- 右面板: 重叠矩阵 ---
    config_names = list(configs.keys())
    n = len(config_names)
    matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        set_i = set(configs[config_names[i]])
        for j in range(n):
            set_j = set(configs[config_names[j]])
            if i == j:
                matrix[i][j] = len(set_i)
            elif i < j:
                matrix[i][j] = len(set_i & set_j)
            else:
                union = len(set_i | set_j)
                matrix[i][j] = len(set_i & set_j) / union if union > 0 else 0

    display_labels = [CONFIG_DISPLAY_NAMES.get(c, c) for c in config_names]
    ax_matrix.imshow(matrix, cmap='Blues', aspect='auto')

    for i in range(n):
        for j in range(n):
            if i == j:
                text = f'{int(matrix[i][j])}'
            elif i < j:
                text = f'{int(matrix[i][j])}'
            else:
                text = f'{matrix[i][j]:.2f}'
            ax_matrix.text(j, i, text, ha='center', va='center',
                           fontsize=8, fontweight='bold' if i == j else 'normal')

    ax_matrix.set_xticks(range(n))
    ax_matrix.set_yticks(range(n))
    ax_matrix.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=8)
    ax_matrix.set_yticklabels(display_labels, fontsize=8)
    ax_matrix.set_title(
        'Overlap Matrix\n(upper: |A∩B|, lower: Jaccard, diag: |config|)',
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    log_plot.info(f"Overlap analysis plot saved: {output_path}")
    plt.close()


def plot_electrode_pairwise_overlap(
    config_a: Tuple[str, List[int]],
    config_b: Tuple[str, List[int]],
    positions_2d: Dict[str, np.ndarray],
    output_path: str,
    show_labels: bool = True,
    show_regions: bool = True,
) -> None:
    """
    生成两个配置之间的通道重叠对比图.

    使用三种颜色区分:
    - 仅 A: 配置 A 的颜色
    - 仅 B: 配置 B 的颜色
    - 重叠: 混合色 (绿色系)

    Args:
        config_a: (配置名称, 通道索引列表)
        config_b: (配置名称, 通道索引列表)
        positions_2d: 128 通道 2D 坐标
        output_path: 输出路径
        show_labels: 是否显示电极标签
        show_regions: 是否显示脑区着色
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    name_a, indices_a = config_a
    name_b, indices_b = config_b
    set_a = set(indices_a)
    set_b = set(indices_b)

    only_a = sorted(set_a - set_b)
    only_b = sorted(set_b - set_a)
    both = sorted(set_a & set_b)
    all_selected = set_a | set_b

    labels = BIOSEMI_128_LABELS
    color_a = CONFIG_COLORS.get(name_a, '#F18F01')
    color_b = CONFIG_COLORS.get(name_b, '#3B1F2B')
    color_both = '#2ECC71'  # 鲜绿色 — 重叠

    display_a = CONFIG_DISPLAY_NAMES.get(name_a, name_a)
    display_b = CONFIG_DISPLAY_NAMES.get(name_b, name_b)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8.5))

    # 脑区着色
    if show_regions:
        draw_brain_regions(ax)

    draw_head_outline(ax)

    # 128ch 背景 (灰色小点)
    bg_x = [positions_2d[labels[i]][0] for i in range(128) if i not in all_selected]
    bg_y = [positions_2d[labels[i]][1] for i in range(128) if i not in all_selected]
    ax.scatter(bg_x, bg_y, s=15, c='lightgray', edgecolors='gray',
               linewidths=0.3, alpha=0.5, zorder=2)

    # 仅 A
    if only_a:
        xa = [positions_2d[labels[i]][0] for i in only_a]
        ya = [positions_2d[labels[i]][1] for i in only_a]
        ax.scatter(xa, ya, s=100, c=color_a, edgecolors='black',
                   linewidths=0.8, alpha=0.9, zorder=3, marker='o')

    # 仅 B
    if only_b:
        xb = [positions_2d[labels[i]][0] for i in only_b]
        yb = [positions_2d[labels[i]][1] for i in only_b]
        ax.scatter(xb, yb, s=100, c=color_b, edgecolors='black',
                   linewidths=0.8, alpha=0.9, zorder=3, marker='o')

    # 重叠
    if both:
        xo = [positions_2d[labels[i]][0] for i in both]
        yo = [positions_2d[labels[i]][1] for i in both]
        ax.scatter(xo, yo, s=120, c=color_both, edgecolors='black',
                   linewidths=1.0, alpha=0.95, zorder=4, marker='o')

    # 电极标签
    if show_labels:
        for i in sorted(all_selected):
            x, y = positions_2d[labels[i]]
            ax.annotate(labels[i], (x, y),
                        xytext=(3, 3), textcoords='offset points',
                        fontsize=5, ha='left', va='bottom',
                        color='black', alpha=0.8)

    # 10-20 地标
    draw_1020_landmarks(ax, positions_2d)

    # 图例
    legend_items = [
        Patch(facecolor=color_a, edgecolor='black', linewidth=0.8,
              label=f'{display_a} only ({len(only_a)})'),
        Patch(facecolor=color_b, edgecolor='black', linewidth=0.8,
              label=f'{display_b} only ({len(only_b)})'),
        Patch(facecolor=color_both, edgecolor='black', linewidth=0.8,
              label=f'Both ({len(both)})'),
    ]
    ax.legend(handles=legend_items, loc='lower center',
              bbox_to_anchor=(0.5, -0.08), ncol=3,
              fontsize=9, framealpha=0.9, edgecolor='gray')

    # 标题
    total_union = len(all_selected)
    jaccard = len(both) / total_union if total_union > 0 else 0
    ax.set_title(
        f'{display_a} vs {display_b} — Channel Overlap\n'
        f'|A|={len(set_a)}  |B|={len(set_b)}  '
        f'|A∩B|={len(both)}  Jaccard={jaccard:.2f}',
        fontsize=11, fontweight='bold',
    )
    ax.set_aspect('equal')
    ax.set_xlim([-0.65, 0.65])
    ax.set_ylim([-0.65, 0.75])
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    log_plot.info(f"Pairwise overlap plot saved: {output_path}")
    plt.close()


# ============================================================================
# 3D 视图
# ============================================================================

def _draw_3d_electrodes(
    ax,
    config_name: str,
    indices: List[int],
    positions_3d: Dict[str, np.ndarray],
    show_background: bool = True,
    show_labels: bool = False,
    marker_size: float = 60,
) -> None:
    """3D 单配置渲染 (内部辅助函数)."""
    labels = BIOSEMI_128_LABELS
    color = CONFIG_COLORS.get(config_name, '#E94F37')
    selected_set = set(indices)

    # 参考球面线框
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    sphere_r = 0.95
    sx = sphere_r * np.outer(np.cos(u), np.sin(v))
    sy = sphere_r * np.outer(np.sin(u), np.sin(v))
    sz = sphere_r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(sx, sy, sz, color='lightgray', alpha=0.08, linewidth=0.3)

    # 鼻子指示器 (Y+ 方向)
    ax.plot([0, 0], [1.0, 1.12], [0, 0.03],
            color='black', linewidth=2, zorder=10)
    ax.text(0, 1.15, 0.03, 'N', fontsize=7, ha='center')

    # 归一化 3D 坐标
    all_pos = np.array(list(positions_3d.values()))
    center = all_pos.mean(axis=0)
    max_r = max(np.linalg.norm(v - center) for v in positions_3d.values())

    def norm_pos(label):
        return (positions_3d[label] - center) / max_r

    # 128ch 背景
    if show_background:
        bg = [i for i in range(128) if i not in selected_set]
        if bg:
            bg_coords = np.array([norm_pos(labels[i]) for i in bg])
            ax.scatter(bg_coords[:, 0], bg_coords[:, 1], bg_coords[:, 2],
                       s=8, c='lightgray', alpha=0.3, depthshade=True)

    # 选中通道
    sel_coords = np.array([norm_pos(labels[i]) for i in indices])
    ax.scatter(sel_coords[:, 0], sel_coords[:, 1], sel_coords[:, 2],
               s=marker_size, c=color, edgecolors='black', linewidths=0.5,
               alpha=0.9, depthshade=True,
               label=CONFIG_DISPLAY_NAMES.get(config_name, config_name))

    if show_labels:
        for i in indices:
            p = norm_pos(labels[i])
            ax.text(p[0], p[1], p[2] + 0.05, labels[i],
                    fontsize=4.5, ha='center', alpha=0.8)

    ax.set_xlabel('X (Right)')
    ax.set_ylabel('Y (Front)')
    ax.set_zlabel('Z (Up)')
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])


def plot_electrode_placement_3d(
    configs: Dict[str, List[int]],
    positions_3d: Dict[str, np.ndarray],
    output_path: str,
    elevation: float = 25.0,
    azimuth: float = -60.0,
) -> None:
    """
    在 3D 头部模型上叠加所有配置的电极布局.

    Args:
        configs: 配置名称 → 通道索引列表
        positions_3d: 128 通道 3D 坐标
        output_path: 输出路径
        elevation: 3D 仰角
        azimuth: 3D 方位角
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection='3d')

    labels = BIOSEMI_128_LABELS

    # 归一化
    all_pos = np.array(list(positions_3d.values()))
    center = all_pos.mean(axis=0)
    max_r = max(np.linalg.norm(v - center) for v in positions_3d.values())

    def norm_pos(label):
        return (positions_3d[label] - center) / max_r

    # 参考球面
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    sphere_r = 0.95
    ax.plot_wireframe(
        sphere_r * np.outer(np.cos(u), np.sin(v)),
        sphere_r * np.outer(np.sin(u), np.sin(v)),
        sphere_r * np.outer(np.ones_like(u), np.cos(v)),
        color='lightgray', alpha=0.08, linewidth=0.3,
    )

    # 鼻子
    ax.plot([0, 0], [1.0, 1.12], [0, 0.03], color='black', linewidth=2, zorder=10)
    ax.text(0, 1.15, 0.03, 'Nose', fontsize=7, ha='center')

    # 128ch 背景
    all_selected = set()
    for indices in configs.values():
        all_selected.update(indices)

    bg = [i for i in range(128) if i not in all_selected]
    if bg:
        bg_coords = np.array([norm_pos(labels[i]) for i in bg])
        ax.scatter(bg_coords[:, 0], bg_coords[:, 1], bg_coords[:, 2],
                   s=8, c='lightgray', alpha=0.3, depthshade=True)

    # 各配置叠加
    for config_name, indices in configs.items():
        color = CONFIG_COLORS.get(config_name, '#E94F37')
        coords = np.array([norm_pos(labels[i]) for i in indices])
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                   s=30, c=color, edgecolors='black', linewidths=0.3,
                   alpha=0.85, depthshade=True,
                   label=CONFIG_DISPLAY_NAMES.get(config_name, config_name))

    ax.set_xlabel('X (Right)')
    ax.set_ylabel('Y (Front)')
    ax.set_zlabel('Z (Up)')
    ax.view_init(elev=elevation, azim=azimuth)
    ax.legend(loc='lower right', fontsize=7, markerscale=1.5)
    ax.set_title(f'3D Electrode Placement — All Configs', fontsize=12)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    log_plot.info(f"3D electrode plot saved: {output_path}")
    plt.close()


def plot_electrode_3d_multiview(
    config_name: str,
    indices: List[int],
    positions_3d: Dict[str, np.ndarray],
    output_path: str,
) -> None:
    """
    单配置 4 视角 3D 组合图 (顶视/前视/右视/等距).

    Args:
        config_name: 配置名称
        indices: 通道索引
        positions_3d: 128 通道 3D 坐标
        output_path: 输出路径
    """
    import matplotlib.pyplot as plt

    views = [
        ('Top View',   90,   0),
        ('Front View',  0,  90),
        ('Right View',  0,   0),
        ('Isometric',  25, -60),
    ]

    fig = plt.figure(figsize=(14, 14))
    for idx, (view_name, elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        _draw_3d_electrodes(
            ax, config_name, indices, positions_3d,
            show_labels=(idx == 3),  # 仅等距视图显示标签
            marker_size=50,
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(view_name, fontsize=10)

    display_name = CONFIG_DISPLAY_NAMES.get(config_name, config_name)
    fig.suptitle(
        f'{display_name} — Multi-View ({len(indices)} channels)',
        fontsize=14, fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    log_plot.info(f"3D multi-view plot saved: {output_path}")
    plt.close()
