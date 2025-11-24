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
    绘制电路仿真结果
    
    参数：
        t: 时间数组（秒）
        results: 包含各支路电流和电压的字典，必须包含以下键：
            - 'i_s': 电流源电流
            - 'i_r': 电阻电流
            - 'i_c': 电容电流
            - 'i_rg': Rg电流
            - 'u_s': 电流源电压
            - 'u_c': 电容电压
            - 'u_rg': Rg电压
        stage_name: 阶段名称（如 'Stage 1', 'Stage 2'）
        u0: 初始电压（用于标题）
        save_path: 保存路径（可选）
    """
    t_ms = t * 1000  # 转换为毫秒
    
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{stage_name} 仿真结果（初始电压: {u0}V）', fontsize=14, fontweight='bold')
    
    # 子图1：各支路电流
    ax1 = axes[0, 0]
    ax1.plot(t_ms, results['i_s'] / 1000, label='i_s (电流源)', linewidth=2)
    ax1.plot(t_ms, results['i_r'] / 1000, label='i_R (电阻)', linewidth=1.5)
    ax1.plot(t_ms, results['i_c'] / 1000, label='i_C (电容)', linewidth=1.5)
    ax1.plot(t_ms, results['i_rg'] / 1000, label='i_Rg', linewidth=1.5, linestyle='--')
    ax1.set_xlabel('时间 (ms)')
    ax1.set_ylabel('电流 (kA)')
    ax1.set_title('各支路电流随时间变化')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 子图2：各元件电压
    ax2 = axes[0, 1]
    ax2.plot(t_ms, results['u_s'] / 1000, label='u_s (电流源)', linewidth=2)
    ax2.plot(t_ms, results['u_c'] / 1000, label='u_C (电容)', linewidth=1.5)
    ax2.plot(t_ms, results['u_rg'] / 1000, label='u_Rg', linewidth=1.5, linestyle='--')
    ax2.set_xlabel('时间 (ms)')
    ax2.set_ylabel('电压 (kV)')
    ax2.set_title('各元件电压随时间变化')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # 子图3：电流源电流和电压（验证）
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()
    line1 = ax3.plot(t_ms, results['i_s'] / 1000, 'b-', label='i_s (电流)', linewidth=2)
    line2 = ax3_twin.plot(t_ms, results['u_s'] / 1000, 'r-', label='u_s (电压)', linewidth=2)
    ax3.set_xlabel('时间 (ms)')
    ax3.set_ylabel('电流 (kA)', color='b')
    ax3_twin.set_ylabel('电压 (kV)', color='r')
    ax3.set_title('电流源电流和电压')
    ax3.tick_params(axis='y', labelcolor='b')
    ax3_twin.tick_params(axis='y', labelcolor='r')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # 子图4：R∥C 支路电流分解
    ax4 = axes[1, 1]
    ax4.plot(t_ms, results['i_r'] / 1000, label='i_R (电阻)', linewidth=1.5)
    ax4.plot(t_ms, results['i_c'] / 1000, label='i_C (电容)', linewidth=1.5)
    ax4.plot(t_ms, (results['i_r'] + results['i_c']) / 1000, 
             label='i_R + i_C (验证)', linewidth=1.5, linestyle='--', alpha=0.7)
    ax4.set_xlabel('时间 (ms)')
    ax4.set_ylabel('电流 (kA)')
    ax4.set_title('R∥C 支路电流分解（验证KCL）')
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    plt.show()

