"""
绘图工具模块

提供电路仿真结果的通用绘图函数
"""

from __future__ import annotations
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def align_yaxis_zeros(ax1: plt.Axes, ax2: plt.Axes) -> None:
    """
    对齐两个y轴的0点
    
    参数：
        ax1: 第一个y轴（左轴）
        ax2: 第二个y轴（右轴，通过twinx创建）
    """
    # 获取两个轴的y轴范围
    y1_min, y1_max = ax1.get_ylim()
    y2_min, y2_max = ax2.get_ylim()
    
    # 计算两个轴的最大绝对值范围
    y1_range = max(abs(y1_min), abs(y1_max))
    y2_range = max(abs(y2_min), abs(y2_max))
    
    # 设置两个轴的范围为对称的，使得0点对齐
    if y1_range > 0:
        ax1.set_ylim(-y1_range * 1.1, y1_range * 1.1)
    if y2_range > 0:
        ax2.set_ylim(-y2_range * 1.1, y2_range * 1.1)


def plot_circuit_results(
    t: np.ndarray, 
    results: dict, 
    stage_name: str, 
    u0: float,
    components: Sequence[tuple[str, str, str, str]] | None = None,
    summary_plots: Sequence[dict] | None = None,
    save_path: str | None = None
) -> None:
    """
    绘制电路仿真结果，每个元件单独一个图，显示该元件的电压和电流
    
    参数：
        t: 时间数组（秒）
        results: 包含各支路电流和电压的字典
        stage_name: 阶段名称（如 'Stage 1', 'Stage 2'）
        u0: 初始电压（用于标题）
        components: 要绘制的元件列表，每个元素为 (名称, 电流键, 电压键, 标签)
                    例如: [('电流源', 'i_s', 'u_s', '电流源'), ...]
                    如果为None，则自动从results中检测可用元件
        summary_plots: 汇总图表配置列表，每个元素为包含以下键的字典：
                      - 'title': 图表标题
                      - 'items': 要绘制的项目列表，每个元素为 (键, 标签, 线型, 线宽)
                      例如: [{'title': '电流汇总', 'items': [('i_s', 'i_s', '-', 2), ...]}, ...]
                      如果为None，则不绘制汇总图表
        save_path: 保存路径（可选）
    """
    t_ms = t * 1000  # 转换为毫秒
    
    # 如果没有提供components，则自动检测（向后兼容）
    if components is None:
        components = []
        # 自动检测可用的元件
        if 'i_s' in results and 'u_s' in results:
            components.append(('电流源', 'i_s', 'u_s', '电流源'))
        if 'i_l' in results and 'u_l' in results:
            components.append(('电感 L', 'i_l', 'u_l', '电感'))
        if 'i_bypass' in results and 'u_bypass' in results:
            components.append(('整流桥', 'i_bypass', 'u_bypass', '整流桥'))
        if 'i_d1' in results and 'u_d1' in results:
            components.append(('二极管 D1', 'i_d1', 'u_d1', 'D1'))
        if 'i_d3' in results and 'u_d3' in results:
            components.append(('二极管 D3', 'i_d3', 'u_d3', 'D3'))
        if 'i_scr1' in results and 'u_scr1' in results:
            components.append(('可控硅 SCR1', 'i_scr1', 'u_scr1', 'SCR1'))
        if 'i_r' in results and 'u_c' in results:
            components.append(('电阻 R', 'i_r', 'u_c', '电阻'))
        if 'i_c' in results and 'u_c' in results:
            components.append(('电容 C', 'i_c', 'u_c', '电容'))
        if 'i_rg' in results and 'u_rg' in results:
            components.append(('电阻 Rg', 'i_rg', 'u_rg', 'Rg'))
    
    # 计算需要的子图数量
    num_components = len(components)
    num_summary_plots = len(summary_plots) if summary_plots else 0
    total_plots = num_components + num_summary_plots
    cols = 3  # 每行3个图
    rows = (total_plots + cols - 1) // cols  # 向上取整
    
    # 创建子图
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    fig.suptitle(f'{stage_name} 仿真结果（初始电压: {u0}V）', fontsize=16, fontweight='bold')
    
    # 统一axes为二维数组格式
    # subplots返回的axes可能是单个对象、一维数组或二维数组
    if not isinstance(axes, np.ndarray):
        # 单个子图：转换为二维数组
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        # 一维数组：reshape为二维
        axes = axes.reshape(rows, cols)
    # 如果已经是二维数组，保持不变
    
    # 为每个元件绘制单独的图
    for idx, (comp_name, i_key, u_key, comp_label) in enumerate(components):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # 创建双y轴
        ax_twin = ax.twinx()
        
        # 绘制电流（左y轴）
        line1 = None
        if i_key in results:
            i_data = results[i_key] / 1000  # 转换为kA
            line1 = ax.plot(t_ms, i_data, 'b-', 
                           label=f'{comp_label}电流', linewidth=2)
            ax.set_ylabel('电流 (kA)', color='b', fontsize=10)
            ax.tick_params(axis='y', labelcolor='b')
        
        # 绘制电压（右y轴）
        line2 = None
        if u_key in results:
            u_data = results[u_key] / 1000  # 转换为kV
            line2 = ax_twin.plot(t_ms, u_data, 'r-', 
                                label=f'{comp_label}电压', linewidth=2)
            ax_twin.set_ylabel('电压 (kV)', color='r', fontsize=10)
            ax_twin.tick_params(axis='y', labelcolor='r')
        
        # 对齐0点：计算数据范围并设置y轴范围，使得0点对齐
        # 使用不同的缩放因子（电流1.1，电压1.2）避免曲线重叠
        if i_key in results and u_key in results:
            i_data = results[i_key] / 1000  # kA
            u_data = results[u_key] / 1000  # kV
            
            # 计算电流和电压的数据范围
            i_min, i_max = np.min(i_data), np.max(i_data)
            u_min, u_max = np.min(u_data), np.max(u_data)
            
            # 计算两个轴的最大绝对值范围（用于对齐0点）
            i_range = max(abs(i_min), abs(i_max))
            u_range = max(abs(u_min), abs(u_max))
            
            # 设置y轴范围，使得0点对齐
            # 使用对称范围，确保0点在中间
            # 电流使用1.1的缩放因子，电压使用1.2的缩放因子，避免曲线重叠
            if i_range > 0:
                ax.set_ylim(-i_range * 1.1, i_range * 1.1)  # 电流：1.1倍范围
            else:
                ax.set_ylim(-0.1, 0.1)  # 默认范围
            
            if u_range > 0:
                ax_twin.set_ylim(-u_range * 1.2, u_range * 1.2)  # 电压：1.2倍范围
            else:
                ax_twin.set_ylim(-0.1, 0.1)  # 默认范围
            
            # 使用辅助函数确保0点对齐（处理刻度不同的情况）
            align_yaxis_zeros(ax, ax_twin)
        elif i_key in results:
            # 只有电流数据
            i_data = results[i_key] / 1000  # kA
            i_min, i_max = np.min(i_data), np.max(i_data)
            i_range = max(abs(i_min), abs(i_max))
            if i_range > 0:
                ax.set_ylim(-i_range * 1.1, i_range * 1.1)
            else:
                ax.set_ylim(-0.1, 0.1)
        elif u_key in results:
            # 只有电压数据
            u_data = results[u_key] / 1000  # kV
            u_min, u_max = np.min(u_data), np.max(u_data)
            u_range = max(abs(u_min), abs(u_max))
            if u_range > 0:
                ax_twin.set_ylim(-u_range * 1.2, u_range * 1.2)
            else:
                ax_twin.set_ylim(-0.1, 0.1)
        
        # 设置标题和标签
        ax.set_xlabel('时间 (ms)', fontsize=10)
        ax.set_title(comp_name, fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # 添加0点参考线
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # 合并图例
        lines = []
        labels = []
        if line1 is not None:
            lines.extend(line1)
            labels.append(f'{comp_label}电流')
        if line2 is not None:
            lines.extend(line2)
            labels.append(f'{comp_label}电压')
        ax.legend(lines, labels, loc='upper left', fontsize=9)
    
    # 绘制汇总图表
    if summary_plots:
        for idx, summary_config in enumerate(summary_plots):
            summary_idx = num_components + idx
            row = summary_idx // cols
            col = summary_idx % cols
            ax_summary = axes[row, col]
            
            # 绘制每个项目
            for key, label, linestyle, linewidth in summary_config['items']:
                if key in results:
                    ax_summary.plot(t_ms, results[key] / 1000, label=label, 
                                   linewidth=linewidth, linestyle=linestyle)
            
            ax_summary.set_xlabel('时间 (ms)', fontsize=10)
            ax_summary.set_ylabel(summary_config.get('ylabel', ''), fontsize=10)
            ax_summary.set_title(summary_config['title'], fontsize=12, fontweight='bold')
            ax_summary.legend(fontsize=9)
            ax_summary.grid(True, linestyle='--', alpha=0.5)
    
    # 隐藏多余的子图
    for idx in range(total_plots, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    else:
        print("警告: 未指定保存路径，图表未保存")
    
    plt.close()  # 关闭图形，释放内存

