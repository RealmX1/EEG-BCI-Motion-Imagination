#!/usr/bin/env python
"""
32 通道 EEG 电极布局可视化.

在 2D/3D 头部模型上绘制各配置的电极位置, 支持:
- 多配置并排 2D 网格对比
- 重叠分析热力图 + Jaccard 矩阵
- 单配置 2D 详细图 (带标签)
- 3D 多配置叠加视图
- 单配置 4 视角 3D 组合图

Usage:
    uv run python scripts/analysis/visualize_electrode_placements.py
    uv run python scripts/analysis/visualize_electrode_placements.py --configs motor_cortex fdr
    uv run python scripts/analysis/visualize_electrode_placements.py --no-3d
    uv run python scripts/analysis/visualize_electrode_placements.py --show-labels
    uv run python scripts/analysis/visualize_electrode_placements.py --channels 8
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.channel_selection import (
    CHANNEL_32_CONFIGS,
    CHANNEL_32_CONFIG_NAMES,
    DATA_DRIVEN_CONFIG_NAMES,
    MOTOR_8_CHANNEL_INDICES,
    get_nch_indices,
    load_channel_selections,
)
from src.visualization.electrode_map import (
    CONFIG_DISPLAY_NAMES,
    load_electrode_positions_2d,
    plot_electrode_grid,
    plot_electrode_overlap,
    plot_electrode_placement_2d,
    plot_electrode_placement_3d,
    plot_electrode_3d_multiview,
    plot_region_distribution,
)


def collect_configs(args) -> dict:
    """收集所有可用的通道配置."""
    configs = {}
    n_ch = args.channels

    # 确定要加载的配置名称
    if args.configs:
        target_names = args.configs
    elif n_ch == 32:
        target_names = CHANNEL_32_CONFIG_NAMES
    elif n_ch == 8:
        target_names = ['motor_cortex'] + DATA_DRIVEN_CONFIG_NAMES
    else:
        target_names = DATA_DRIVEN_CONFIG_NAMES

    for name in target_names:
        try:
            indices = get_nch_indices(n_ch, name)
            configs[name] = indices
            print(f"  [OK] {name}: {len(indices)} channels")
        except (ValueError, FileNotFoundError) as e:
            print(f"  [SKIP] {name}: {e}")

    # 8ch motor_cortex 特殊处理 (硬编码)
    if n_ch == 8 and 'motor_cortex' not in configs:
        configs['motor_cortex'] = sorted(MOTOR_8_CHANNEL_INDICES)
        print(f"  [OK] motor_cortex: {len(MOTOR_8_CHANNEL_INDICES)} channels (硬编码)")

    return configs


def main():
    parser = argparse.ArgumentParser(
        description='EEG 电极布局可视化 (2D/3D 头部模型)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--channels', type=int, default=32,
        help='通道数 (默认: 32)',
    )
    parser.add_argument(
        '--configs', nargs='+', default=None,
        help='指定配置名称 (默认: 所有可用配置)',
    )
    parser.add_argument(
        '--elc-path', default=None,
        help='BioSemi ELC 文件路径 (默认: 使用 MNE 内置 biosemi128 montage)',
    )
    parser.add_argument(
        '--output-dir', default=None,
        help='输出目录 (默认: results/{N}_channel/electrode_placements)',
    )
    parser.add_argument(
        '--show-labels', action='store_true',
        help='在网格图中显示电极标签',
    )
    parser.add_argument(
        '--no-3d', action='store_true',
        help='跳过 3D 视图生成',
    )
    parser.add_argument(
        '--no-single', action='store_true',
        help='跳过单配置详细图',
    )
    args = parser.parse_args()

    # 输出目录
    if args.output_dir is None:
        output_dir = PROJECT_ROOT / 'results' / f'{args.channels}_channel' / 'electrode_placements'
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  EEG 电极布局可视化 ({args.channels} channels)")
    print(f"{'='*60}")

    # 加载电极位置 (MNE 内置 montage 或自定义 ELC)
    source = args.elc_path or 'MNE built-in biosemi128'
    print(f"\n加载电极位置: {source}")
    positions_2d, positions_3d = load_electrode_positions_2d(args.elc_path)
    print(f"  128 通道 2D/3D 坐标已加载")

    # 收集配置
    print(f"\n收集 {args.channels}ch 配置:")
    configs = collect_configs(args)

    if not configs:
        print("\n[ERROR] 没有找到可用的配置。")
        print("提示: 运行 `uv run python scripts/analysis/compute_32ch_selections.py` 生成数据驱动配置")
        sys.exit(1)

    print(f"\n共 {len(configs)} 个配置，输出目录: {output_dir}\n")

    # =========================================================================
    # 1. 2D 网格对比图
    # =========================================================================
    grid_path = str(output_dir / 'grid_all_configs_2d.png')
    print(f"[1/5] 生成 2D 网格对比图...")
    plot_electrode_grid(
        configs, positions_2d, grid_path,
        show_labels=args.show_labels,
        suptitle=f'{args.channels}-Channel Electrode Configurations',
    )
    print(f"  -> {grid_path}")

    # =========================================================================
    # 2. 重叠分析
    # =========================================================================
    if len(configs) >= 2:
        overlap_path = str(output_dir / 'overlap_analysis.png')
        print(f"[2/5] 生成重叠分析图...")
        plot_electrode_overlap(configs, positions_2d, overlap_path)
        print(f"  -> {overlap_path}")
    else:
        print(f"[2/5] 跳过重叠分析 (需至少 2 个配置)")

    # =========================================================================
    # 3. 脑区分布柱状图
    # =========================================================================
    if len(configs) >= 2:
        region_path = str(output_dir / 'region_distribution.png')
        print(f"[3/5] 生成脑区分布柱状图...")
        plot_region_distribution(configs, positions_2d, region_path)
        print(f"  -> {region_path}")
    else:
        print(f"[3/5] 跳过脑区分布图 (需至少 2 个配置)")

    # =========================================================================
    # 4. 单配置 2D 详细图 (带标签)
    # =========================================================================
    if not args.no_single:
        print(f"[4/5] 生成单配置 2D 详细图...")
        import matplotlib.pyplot as plt
        for name, indices in configs.items():
            fig, ax = plt.subplots(1, 1, figsize=(7, 7.5))
            plot_electrode_placement_2d(
                ax, positions_2d, indices,
                config_name=name,
                show_labels=True,
                marker_size_selected=100,
            )
            single_path = str(output_dir / f'single_{name}_2d.png')
            plt.tight_layout()
            plt.savefig(single_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  -> {single_path}")
    else:
        print(f"[4/5] 跳过单配置详细图 (--no-single)")

    # =========================================================================
    # 5. 3D 视图
    # =========================================================================
    if not args.no_3d:
        print(f"[5/5] 生成 3D 视图...")

        # 4a. 多配置叠加 3D
        all_3d_path = str(output_dir / 'all_configs_3d.png')
        plot_electrode_placement_3d(configs, positions_3d, all_3d_path)
        print(f"  -> {all_3d_path}")

        # 4b. 单配置多视角 3D
        if not args.no_single:
            for name, indices in configs.items():
                mv_path = str(output_dir / f'multiview_{name}_3d.png')
                plot_electrode_3d_multiview(name, indices, positions_3d, mv_path)
                print(f"  -> {mv_path}")
    else:
        print(f"[5/5] 跳过 3D 视图 (--no-3d)")

    # =========================================================================
    # 汇总
    # =========================================================================
    from pathlib import Path as P
    generated = list(output_dir.glob('*.png'))
    print(f"\n{'='*60}")
    print(f"  完成! 共生成 {len(generated)} 张图表")
    print(f"  输出目录: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
