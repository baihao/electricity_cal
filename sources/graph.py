"""
绘图工具模块

提供电路仿真结果的通用绘图函数
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def plot_circuit_results(t: np.ndarray, results: dict, stage_name: str, 
                         u0: float, save_path: str | None = None) -> None:
    """
    绘制电路仿真结果，每个元件单独一个图，显示该元件的电压和电流
    
    参数：
        t: 时间数组（秒）
        results: 包含各支路电流和电压的字典
        stage_name: 阶段名称（如 'Stage 1', 'Stage 2'）
        u0: 初始电压（用于标题）
        save_path: 保存路径（可选）
    """
    t_ms = t * 1000  # 转换为毫秒
    
    # 确定需要绘制的元件列表
    components = []
    
    # 电流源（始终存在）
    components.append(('电流源', 'i_s', 'u_s', '电流源'))
    
    # Stage 2 特有的元件
    if 'i_l' in results and 'u_l' in results:
        components.append(('电感 L', 'i_l', 'u_l', '电感'))
    if 'i_bypass' in results and 'u_bypass' in results:
        components.append(('电子旁路 (D1-D2-SCR1)', 'i_bypass', 'u_bypass', '电子旁路'))
    
    # R, C, Rg（始终存在）
    components.append(('电阻 R', 'i_r', 'u_c', '电阻'))  # R的电压等于u_C
    components.append(('电容 C', 'i_c', 'u_c', '电容'))
    components.append(('电阻 Rg', 'i_rg', 'u_rg', 'Rg'))
    
    # 计算需要的子图数量
    # 元件图表 + 2个汇总图表（R/L/C/电流源的电流和电压汇总）
    num_components = len(components)
    num_summary_plots = 2  # 电流汇总图和电压汇总图
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
        if i_key in results:
            line1 = ax.plot(t_ms, results[i_key] / 1000, 'b-', 
                           label=f'{comp_label}电流', linewidth=2)
            ax.set_ylabel('电流 (kA)', color='b', fontsize=10)
            ax.tick_params(axis='y', labelcolor='b')
        
        # 绘制电压（右y轴）
        if u_key in results:
            line2 = ax_twin.plot(t_ms, results[u_key] / 1000, 'r-', 
                                label=f'{comp_label}电压', linewidth=2)
            ax_twin.set_ylabel('电压 (kV)', color='r', fontsize=10)
            ax_twin.tick_params(axis='y', labelcolor='r')
        
        # 设置标题和标签
        ax.set_xlabel('时间 (ms)', fontsize=10)
        ax.set_title(comp_name, fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # 合并图例
        lines = []
        labels = []
        if i_key in results:
            lines.extend(line1)
            labels.append(f'{comp_label}电流')
        if u_key in results:
            lines.extend(line2)
            labels.append(f'{comp_label}电压')
        ax.legend(lines, labels, loc='upper left', fontsize=9)
    
    # 添加汇总图表
    # 汇总图1：R, L, C 和电流源的电流变化
    summary_idx1 = num_components
    row1 = summary_idx1 // cols
    col1 = summary_idx1 % cols
    ax_summary1 = axes[row1, col1]
    
    ax_summary1.plot(t_ms, results['i_s'] / 1000, label='i_s (电流源)', linewidth=2)
    ax_summary1.plot(t_ms, results['i_r'] / 1000, label='i_R (电阻)', linewidth=1.5)
    ax_summary1.plot(t_ms, results['i_c'] / 1000, label='i_C (电容)', linewidth=1.5)
    if 'i_l' in results:
        ax_summary1.plot(t_ms, results['i_l'] / 1000, label='i_L (电感)', linewidth=1.5)
    ax_summary1.set_xlabel('时间 (ms)', fontsize=10)
    ax_summary1.set_ylabel('电流 (kA)', fontsize=10)
    ax_summary1.set_title('R, L, C 和电流源电流变化', fontsize=12, fontweight='bold')
    ax_summary1.legend(fontsize=9)
    ax_summary1.grid(True, linestyle='--', alpha=0.5)
    
    # 汇总图2：R, L, C 和电流源的电压变化
    summary_idx2 = num_components + 1
    row2 = summary_idx2 // cols
    col2 = summary_idx2 % cols
    ax_summary2 = axes[row2, col2]
    
    ax_summary2.plot(t_ms, results['u_s'] / 1000, label='u_s (电流源)', linewidth=2)
    # R和C的电压都是u_C（因为它们并联）
    ax_summary2.plot(t_ms, results['u_c'] / 1000, label='u_R = u_C (电阻和电容)', linewidth=1.5)
    if 'u_l' in results:
        ax_summary2.plot(t_ms, results['u_l'] / 1000, label='u_L (电感)', linewidth=1.5)
    ax_summary2.set_xlabel('时间 (ms)', fontsize=10)
    ax_summary2.set_ylabel('电压 (kV)', fontsize=10)
    ax_summary2.set_title('R, L, C 和电流源电压变化', fontsize=12, fontweight='bold')
    ax_summary2.legend(fontsize=9)
    ax_summary2.grid(True, linestyle='--', alpha=0.5)
    
    # 隐藏多余的子图
    for idx in range(total_plots, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    plt.show()

