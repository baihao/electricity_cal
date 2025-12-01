"""
Stage All: 连续进行三个阶段的仿真并验证能量守恒

该脚本按顺序执行：
1. Stage 1: 0-200微秒电路仿真
2. Stage 2: 200微秒-50毫秒电路仿真（从Stage 1继承初始条件）
3. Stage 3: 50毫秒-200毫秒电路仿真（从Stage 2继承初始条件）

并对每个阶段进行能量守恒验证。

================================================================================
能量守恒验证思路和步骤
================================================================================

1. 能量守恒原理
   ------------
   根据能量守恒定律，电路中输入的能量等于消耗的能量加上存储能量的变化：
   
   E_source = E_consumed + ΔE_stored
   
   其中：
   - E_source：电流源提供的总能量
   - E_consumed：所有耗能元件消耗的总能量（电阻、整流桥等）
   - ΔE_stored：储能元件（电感和电容）存储能量的变化

2. 能量计算方法
   ------------
   
   2.1 电流源提供的能量（输入能量）
       E_source = ∫[t0 to t1] u_s(t) · i_s(t) dt
       
       物理意义：电流源在时间[t0, t1]内提供的总能量
       计算方法：使用数值积分（trapz）计算功率对时间的积分
   
   2.2 电阻R消耗的能量
       E_R = ∫[t0 to t1] u_R(t) · i_R(t) dt
            = ∫[t0 to t1] (u_C²(t) / R) dt
            = (1/R) · ∫[t0 to t1] u_C²(t) dt
       
       物理意义：电阻R在时间[t0, t1]内消耗的总能量（转化为热能）
       计算方法：使用数值积分计算 u_C²(t) / R 对时间的积分
   
   2.3 电阻Rg消耗的能量
       E_Rg = ∫[t0 to t1] u_Rg(t) · i_Rg(t) dt
             = ∫[t0 to t1] (i_s²(t) · Rg) dt
             = Rg · ∫[t0 to t1] i_s²(t) dt
       
       物理意义：电阻Rg在时间[t0, t1]内消耗的总能量（转化为热能）
       计算方法：使用数值积分计算 i_s²(t) · Rg 对时间的积分
   
   2.4 整流桥消耗的能量（仅Stage 2和Stage 3）
       E_bypass = ∫[t0 to t1] u_bypass(t) · i_bypass(t) dt
       
       物理意义：整流桥（二极管和SCR）在时间[t0, t1]内消耗的总能量
                包括阈值电压损耗和等效电阻损耗
       计算方法：使用数值积分计算 u_bypass(t) · i_bypass(t) 对时间的积分
       
       注意：整流桥的损耗包括：
             - 阈值电压损耗：U_threshold · |i_bypass|
             - 等效电阻损耗：R_total · i_bypass²
   
   2.5 电感存储能量的变化
       ΔE_L = ∫[t0 to t1] u_L(t) · i_L(t) dt
             = (1/2) · L · [i_L²(t1) - i_L²(t0)]
       
       物理意义：电感在时间[t0, t1]内存储能量的变化
       计算方法：
          - 方法1（功率积分）：∫ u_L(t) · i_L(t) dt（更准确，适用于非理想情况）
          - 方法2（能量差）：(1/2) · L · [i_L²(t1) - i_L²(t0)]（理想情况）
       
       注意：如果 ΔE_L > 0，表示电感存储能量增加；如果 ΔE_L < 0，表示释放能量
   
   2.6 电容存储能量的变化
       ΔE_C = ∫[t0 to t1] u_C(t) · i_C(t) dt
             = (1/2) · C · [u_C²(t1) - u_C²(t0)]
       
       物理意义：电容在时间[t0, t1]内存储能量的变化
       计算方法：
          - 方法1（功率积分）：∫ u_C(t) · i_C(t) dt（更准确，适用于非理想情况）
          - 方法2（能量差）：(1/2) · C · [u_C²(t1) - u_C²(t0)]（理想情况）
       
       注意：如果 ΔE_C > 0，表示电容存储能量增加；如果 ΔE_C < 0，表示释放能量

3. 能量守恒验证步骤
   ---------------
   
   步骤1：计算输入能量
          E_source = ∫ u_s(t) · i_s(t) dt
   
   步骤2：计算消耗能量
          E_consumed = E_R + E_Rg + E_bypass
   
   步骤3：计算存储能量变化
          ΔE_stored = ΔE_L + ΔE_C
   
   步骤4：计算总能量
          E_total = E_consumed + ΔE_stored
   
   步骤5：计算误差
          error = E_source - E_total
          error_percent = (error / E_source) × 100%
   
   步骤6：判断验证结果
          - 如果 |error_percent| < 1%，认为能量守恒验证通过
          - 如果 1% ≤ |error_percent| < 5%，认为能量守恒基本满足，但存在数值误差
          - 如果 |error_percent| ≥ 5%，可能存在能量计算错误或数值问题

4. 注意事项
   ---------
   
   4.1 为什么不能直接用电流平方积分？
       电流平方积分 ∫ i²(t) dt 本身没有直接的物理意义。
       能量是功率对时间的积分：E = ∫ P(t) dt = ∫ u(t) · i(t) dt
       
       对于电阻：P = i²R，所以 E = R · ∫ i²(t) dt
       但对于不同支路，电压不同，不能直接比较电流平方积分。
   
   4.2 数值积分方法
       使用 scipy.integrate.trapz 进行梯形积分，这是数值积分中最常用的方法。
       对于等间隔采样数据，trapz 方法精度较高且计算效率好。
   
   4.3 误差来源
       - 数值积分误差：trapz方法的离散化误差
       - 数值求解器误差：ODE求解器的截断误差和舍入误差
       - 非线性器件建模误差：二极管和SCR的简化模型可能不完全准确
       - 时间采样间隔：采样间隔过大可能导致能量计算不准确
   
   4.4 各阶段的特殊性
       - Stage 1：只有R、C、Rg，没有L和整流桥
       - Stage 2：有L、整流桥、R、C、Rg
       - Stage 3：有L、整流桥、R、C、Rg，还有K3（但K3电阻很小，能量损耗可忽略）

5. 验证输出格式
   ------------
   
   对每个阶段输出：
   - 输入能量（E_source）
   - 各耗能元件的能量（E_R, E_Rg, E_bypass）
   - 各储能元件的能量变化（ΔE_L, ΔE_C）
   - 总消耗能量（E_consumed）
   - 总存储能量变化（ΔE_stored）
   - 总能量（E_total）
   - 误差（error）和误差百分比（error_percent）
   - 验证结果（通过/基本满足/失败）

================================================================================
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapz, cumtrapz
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 导入仿真函数
from stage_1 import simulate_stage1, U0_CASE1, U0_CASE2
from stage_2 import simulate_stage2
from stage_3 import simulate_stage3
from reader import get_stage1_final_time_and_u_c, get_stage2_final_values
from circuit_params import R, Rg, L, C
from serialize import save_results_to_csv
from graph import plot_circuit_results


def verify_energy_conservation(t: np.ndarray, results: dict, stage_name: str, 
                               u0: float) -> dict:
    """
    验证能量守恒
    
    能量守恒方程：
    E_source = E_R + E_Rg + E_bypass + ΔE_L + ΔE_C
    
    参数：
        t: 时间数组（秒）
        results: 仿真结果字典，包含各支路电流和电压
        stage_name: 阶段名称（'Stage 1', 'Stage 2', 'Stage 3'）
        u0: 初始电压（V）
    
    返回：
        包含能量计算结果和验证结果的字典
    """
    # 1. 电流源提供的能量
    u_s = results['u_s']
    i_s = results['i_s']
    E_source = trapz(u_s * i_s, t)
    
    # 2. 电阻R消耗的能量
    u_c = results['u_c']
    E_R = trapz(u_c**2 / R, t)
    
    # 3. 电阻Rg消耗的能量
    E_Rg = trapz(i_s**2 * Rg, t)
    
    # 4. 整流桥消耗的能量（Stage 2和Stage 3才有）
    E_bypass = 0.0
    if 'u_bypass' in results and 'i_bypass' in results:
        u_bypass = results['u_bypass']
        i_bypass = results['i_bypass']
        E_bypass = trapz(u_bypass * i_bypass, t)
    
    # 5. 电感存储能量变化（Stage 2和Stage 3才有）
    ΔE_L = 0.0
    ΔE_L_power = 0.0
    ΔE_L_energy = 0.0
    if 'i_l' in results and 'u_l' in results:
        i_l = results['i_l']
        u_l = results['u_l']
        # 方法1：通过功率积分
        ΔE_L_power = trapz(u_l * i_l, t)
        # 方法2：通过能量差
        ΔE_L_energy = 0.5 * L * (i_l[-1]**2 - i_l[0]**2)
        ΔE_L = ΔE_L_power  # 使用功率积分更准确
    
    # 6. 电容存储能量变化
    i_c = results['i_c']
    # 方法1：通过功率积分
    ΔE_C_power = trapz(u_c * i_c, t)
    # 方法2：通过能量差
    ΔE_C_energy = 0.5 * C * (u_c[-1]**2 - u_c[0]**2)
    ΔE_C = ΔE_C_power  # 使用功率积分更准确
    
    # 计算总消耗和存储能量
    E_consumed = E_R + E_Rg + E_bypass
    E_stored_change = ΔE_L + ΔE_C
    E_total = E_consumed + E_stored_change
    
    # 计算误差
    error = E_source - E_total
    error_percent = (error / E_source * 100) if abs(E_source) > 1e-10 else 0.0
    
    # 判断验证结果
    if abs(error_percent) < 1.0:
        status = "通过"
    elif abs(error_percent) < 5.0:
        status = "基本满足"
    else:
        status = "失败"
    
    return {
        'stage_name': stage_name,
        'u0': u0,
        'E_source': E_source,
        'E_R': E_R,
        'E_Rg': E_Rg,
        'E_bypass': E_bypass,
        'ΔE_L_power': ΔE_L_power,
        'ΔE_L_energy': ΔE_L_energy,
        'ΔE_L': ΔE_L,
        'ΔE_C_power': ΔE_C_power,
        'ΔE_C_energy': ΔE_C_energy,
        'ΔE_C': ΔE_C,
        'E_consumed': E_consumed,
        'E_stored_change': E_stored_change,
        'E_total': E_total,
        'error': error,
        'error_percent': error_percent,
        'status': status
    }


def calculate_energy_over_time(t: np.ndarray, results: dict, stage_name: str) -> dict:
    """
    计算各元器件能量随时间的变化（累积能量）
    
    参数：
        t: 时间数组（秒）
        results: 仿真结果字典，包含各支路电流和电压
        stage_name: 阶段名称（'Stage 1', 'Stage 2', 'Stage 3'）
    
    返回：
        包含各元器件能量随时间变化的字典
    """
    # 计算时间间隔（用于累积积分）
    dt = np.diff(t)
    
    # 1. 电流源提供的累积能量
    u_s = results['u_s']
    i_s = results['i_s']
    P_source = u_s * i_s  # 瞬时功率
    E_source_cum = np.concatenate([[0], cumtrapz(P_source, t)])  # 累积能量
    
    # 2. 电阻R消耗的累积能量
    u_c = results['u_c']
    P_R = u_c**2 / R  # 瞬时功率
    E_R_cum = np.concatenate([[0], cumtrapz(P_R, t)])  # 累积能量
    
    # 3. 电阻Rg消耗的累积能量
    P_Rg = i_s**2 * Rg  # 瞬时功率
    E_Rg_cum = np.concatenate([[0], cumtrapz(P_Rg, t)])  # 累积能量
    
    # 4. 整流桥消耗的累积能量（Stage 2和Stage 3才有）
    E_bypass_cum = np.zeros_like(t)
    P_bypass = None
    if 'u_bypass' in results and 'i_bypass' in results:
        u_bypass = results['u_bypass']
        i_bypass = results['i_bypass']
        P_bypass = u_bypass * i_bypass  # 瞬时功率
        E_bypass_cum = np.concatenate([[0], cumtrapz(P_bypass, t)])  # 累积能量
    
    # 5. 电感存储能量的累积变化（Stage 2和Stage 3才有）
    E_L_cum = np.zeros_like(t)
    P_L = None
    if 'i_l' in results and 'u_l' in results:
        i_l = results['i_l']
        u_l = results['u_l']
        P_L = u_l * i_l  # 瞬时功率
        E_L_cum = np.concatenate([[0], cumtrapz(P_L, t)])  # 累积能量变化
    
    # 6. 电容存储能量的累积变化
    i_c = results['i_c']
    P_C = u_c * i_c  # 瞬时功率
    E_C_cum = np.concatenate([[0], cumtrapz(P_C, t)])  # 累积能量变化
    
    # 计算总消耗能量和总存储能量变化
    E_consumed_cum = E_R_cum + E_Rg_cum + E_bypass_cum
    E_stored_cum = E_L_cum + E_C_cum
    E_total_cum = E_consumed_cum + E_stored_cum
    
    return {
        't': t,
        'E_source_cum': E_source_cum,
        'E_R_cum': E_R_cum,
        'E_Rg_cum': E_Rg_cum,
        'E_bypass_cum': E_bypass_cum,
        'E_L_cum': E_L_cum,
        'E_C_cum': E_C_cum,
        'E_consumed_cum': E_consumed_cum,
        'E_stored_cum': E_stored_cum,
        'E_total_cum': E_total_cum,
        'P_source': P_source,
        'P_R': P_R,
        'P_Rg': P_Rg,
        'P_bypass': P_bypass if 'u_bypass' in results else None,
        'P_L': P_L if 'i_l' in results else None,
        'P_C': P_C
    }


def plot_energy_over_time(energy_time: dict, stage_name: str, u0: float, 
                          output_dir: Path) -> None:
    """
    绘制各元器件能量随时间变化的图表
    
    参数：
        energy_time: 能量随时间变化的字典（来自calculate_energy_over_time）
        stage_name: 阶段名称
        u0: 初始电压（V）
        output_dir: 输出目录
    """
    t = energy_time['t']
    t_ms = t * 1000  # 转换为毫秒
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{stage_name} 各元器件能量变化（初始电压: {u0}V）', 
                 fontsize=16, fontweight='bold')
    
    # 子图1：输入能量和总能量
    ax1 = axes[0, 0]
    ax1.plot(t_ms, energy_time['E_source_cum'] / 1e6, 'b-', 
             label='E_source (电流源)', linewidth=2)
    ax1.plot(t_ms, energy_time['E_total_cum'] / 1e6, 'r--', 
             label='E_total (总能量)', linewidth=2)
    ax1.set_xlabel('时间 (ms)', fontsize=12)
    ax1.set_ylabel('累积能量 (MJ)', fontsize=12)
    ax1.set_title('输入能量和总能量', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 子图2：消耗能量
    ax2 = axes[0, 1]
    ax2.plot(t_ms, energy_time['E_R_cum'] / 1e6, '-', 
             label='E_R (电阻R)', linewidth=2)
    ax2.plot(t_ms, energy_time['E_Rg_cum'] / 1e6, '-', 
             label='E_Rg (电阻Rg)', linewidth=2)
    if np.any(energy_time['E_bypass_cum'] != 0):
        ax2.plot(t_ms, energy_time['E_bypass_cum'] / 1e6, '-', 
                 label='E_bypass (整流桥)', linewidth=2)
    ax2.plot(t_ms, energy_time['E_consumed_cum'] / 1e6, 'k--', 
             label='E_consumed (总消耗)', linewidth=2)
    ax2.set_xlabel('时间 (ms)', fontsize=12)
    ax2.set_ylabel('累积能量 (MJ)', fontsize=12)
    ax2.set_title('消耗能量', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 子图3：存储能量变化
    ax3 = axes[1, 0]
    if np.any(energy_time['E_L_cum'] != 0):
        ax3.plot(t_ms, energy_time['E_L_cum'] / 1e6, '-', 
                 label='ΔE_L (电感)', linewidth=2)
    ax3.plot(t_ms, energy_time['E_C_cum'] / 1e6, '-', 
             label='ΔE_C (电容)', linewidth=2)
    ax3.plot(t_ms, energy_time['E_stored_cum'] / 1e6, 'k--', 
             label='ΔE_stored (总存储变化)', linewidth=2)
    ax3.set_xlabel('时间 (ms)', fontsize=12)
    ax3.set_ylabel('累积能量变化 (MJ)', fontsize=12)
    ax3.set_title('存储能量变化', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # 子图4：能量平衡（输入能量 vs 总能量）
    ax4 = axes[1, 1]
    ax4.plot(t_ms, energy_time['E_source_cum'] / 1e6, 'b-', 
             label='E_source (输入)', linewidth=2)
    ax4.plot(t_ms, energy_time['E_total_cum'] / 1e6, 'r--', 
             label='E_total (消耗+存储)', linewidth=2)
    # 计算误差
    error_cum = energy_time['E_source_cum'] - energy_time['E_total_cum']
    ax4_twin = ax4.twinx()
    ax4_twin.plot(t_ms, error_cum / 1e6, 'g:', 
                  label='误差', linewidth=1.5, alpha=0.7)
    ax4.set_xlabel('时间 (ms)', fontsize=12)
    ax4.set_ylabel('累积能量 (MJ)', fontsize=12, color='b')
    ax4_twin.set_ylabel('误差 (MJ)', fontsize=12, color='g')
    ax4.set_title('能量平衡验证', fontsize=13, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=10)
    ax4_twin.legend(loc='upper right', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4_twin.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    # 保存图表
    stage_suffix = stage_name.lower().replace(' ', '_')
    img_path = output_dir / f"{stage_suffix}_energy_over_time_u0_{u0:.0f}V.png"
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"能量变化图表已保存至: {img_path}")
    plt.close()


def print_energy_verification(energy_results: dict) -> None:
    """
    打印能量守恒验证结果
    
    参数：
        energy_results: 能量验证结果字典
    """
    print(f"\n{'='*80}")
    print(f"{energy_results['stage_name']} 能量守恒验证结果（初始电压: {energy_results['u0']}V）")
    print(f"{'='*80}")
    
    print(f"\n1. 输入能量:")
    print(f"   E_source = {energy_results['E_source']:.6e} J")
    
    print(f"\n2. 消耗能量:")
    print(f"   E_R      = {energy_results['E_R']:.6e} J (电阻R)")
    print(f"   E_Rg     = {energy_results['E_Rg']:.6e} J (电阻Rg)")
    if energy_results['E_bypass'] != 0:
        print(f"   E_bypass = {energy_results['E_bypass']:.6e} J (整流桥)")
    print(f"   E_consumed = {energy_results['E_consumed']:.6e} J (总消耗)")
    
    print(f"\n3. 存储能量变化:")
    if energy_results['ΔE_L'] != 0:
        print(f"   ΔE_L (功率积分) = {energy_results['ΔE_L_power']:.6e} J")
        print(f"   ΔE_L (能量差)   = {energy_results['ΔE_L_energy']:.6e} J")
        print(f"   ΔE_L (使用)     = {energy_results['ΔE_L']:.6e} J (电感)")
    print(f"   ΔE_C (功率积分) = {energy_results['ΔE_C_power']:.6e} J")
    print(f"   ΔE_C (能量差)   = {energy_results['ΔE_C_energy']:.6e} J")
    print(f"   ΔE_C (使用)     = {energy_results['ΔE_C']:.6e} J (电容)")
    print(f"   ΔE_stored       = {energy_results['E_stored_change']:.6e} J (总存储变化)")
    
    print(f"\n4. 能量守恒验证:")
    print(f"   E_source         = {energy_results['E_source']:.6e} J")
    print(f"   E_total          = {energy_results['E_total']:.6e} J")
    print(f"   误差             = {energy_results['error']:.6e} J")
    print(f"   误差百分比       = {energy_results['error_percent']:.6f}%")
    print(f"   验证结果         = {energy_results['status']}")
    
    print(f"\n5. 能量分配比例:")
    if abs(energy_results['E_source']) > 1e-10:
        print(f"   消耗能量占比    = {energy_results['E_consumed']/energy_results['E_source']*100:.2f}%")
        print(f"   存储能量变化占比 = {energy_results['E_stored_change']/energy_results['E_source']*100:.2f}%")
    
    print(f"{'='*80}\n")


def get_stage_plot_config(stage_num: int):
    """
    获取指定阶段的绘图配置
    
    参数：
        stage_num: 阶段编号（1, 2, 或 3）
    
    返回：
        (components, summary_plots) 元组
    """
    if stage_num == 1:
        components = [
            ('电流源', 'i_s', 'u_s', '电流源'),
            ('电阻 R', 'i_r', 'u_c', '电阻'),
            ('电容 C', 'i_c', 'u_c', '电容'),
            ('电阻 Rg', 'i_rg', 'u_rg', 'Rg'),
        ]
        summary_plots = [
            {
                'title': 'R, C 和电流源电流变化',
                'ylabel': '电流 (kA)',
                'items': [
                    ('i_s', 'i_s (电流源)', '-', 2),
                    ('i_r', 'i_R (电阻)', '-', 1.5),
                    ('i_c', 'i_C (电容)', '-', 1.5),
                ]
            },
            {
                'title': 'R, C 和电流源电压变化',
                'ylabel': '电压 (kV)',
                'items': [
                    ('u_s', 'u_s (电流源)', '-', 2),
                    ('u_c', 'u_R = u_C (电阻和电容)', '-', 1.5),
                ]
            }
        ]
    elif stage_num == 2:
        components = [
            ('电流源', 'i_s', 'u_s', '电流源'),
            ('电感 L', 'i_l', 'u_l', '电感'),
            ('电阻 R', 'i_r', 'u_c', '电阻'),
            ('电容 C', 'i_c', 'u_c', '电容'),
            ('电阻 Rg', 'i_rg', 'u_rg', 'Rg'),
            ('整流桥', 'i_bypass', 'u_bypass', '整流桥'),
        ]
        summary_plots = [
            {
                'title': 'R, L, C 和电流源电流变化',
                'ylabel': '电流 (kA)',
                'items': [
                    ('i_s', 'i_s (电流源)', '-', 2),
                    ('i_r', 'i_R (电阻)', '-', 1.5),
                    ('i_c', 'i_C (电容)', '-', 1.5),
                    ('i_l', 'i_L (电感)', '-', 1.5),
                    ('i_bypass', 'i_bypass (整流桥)', ':', 1.5),
                    ('i_d1', 'i_D1 (正向路径)', ':', 1.5),
                    ('i_d2', 'i_D2 (正向路径)', ':', 1.5),
                    ('i_d3', 'i_D3 (反向路径)', ':', 1.5),
                    ('i_d4', 'i_D4 (反向路径)', ':', 1.5),
                    ('i_scr1', 'i_SCR1 (可控硅)', '--', 1.5),
                ]
            },
            {
                'title': 'R, L, C 和电流源电压变化',
                'ylabel': '电压 (kV)',
                'items': [
                    ('u_s', 'u_s (电流源)', '-', 2),
                    ('u_c', 'u_R = u_C (电阻和电容)', '-', 1.5),
                    ('u_l', 'u_L (电感)', '-', 1.5),
                    ('u_d1', 'u_D1 (正向路径)', ':', 1.5),
                    ('u_d2', 'u_D2 (正向路径)', ':', 1.5),
                    ('u_d3', 'u_D3 (反向路径)', ':', 1.5),
                    ('u_d4', 'u_D4 (反向路径)', ':', 1.5),
                    ('u_scr1', 'u_SCR1 (可控硅)', '--', 1.5),
                ]
            }
        ]
    elif stage_num == 3:
        components = [
            ('电流源', 'i_s', 'u_s', '电流源'),
            ('K3回路', 'i_k3', 'u_k3', 'K3'),
            ('电阻 R', 'i_r', 'u_c', '电阻'),
            ('电容 C', 'i_c', 'u_c', '电容'),
            ('电感 L', 'i_l', 'u_l', '电感'),
            ('电阻 Rg', 'i_rg', 'u_rg', 'Rg'),
        ]
        summary_plots = [
            {
                'title': 'K3, R, C, L, Rg 和电流源电流变化',
                'ylabel': '电流 (kA)',
                'items': [
                    ('i_s', 'i_s (电流源)', '-', 2),
                    ('i_k3', 'i_K3 (K3回路)', '-', 2),
                    ('i_rg', 'i_Rg (Rg)', '-', 2),
                    ('i_r', 'i_R (电阻)', '-', 1.5),
                    ('i_c', 'i_C (电容)', '-', 1.5),
                    ('i_l', 'i_L (电感)', '-', 1.5),
                ]
            },
            {
                'title': 'R, C 和电流源电压变化',
                'ylabel': '电压 (kV)',
                'items': [
                    ('u_s', 'u_s (电流源)', '-', 2),
                    ('u_c', 'u_C (电容和电阻)', '-', 1.5),
                ]
            },
        ]
    else:
        raise ValueError(f"无效的阶段编号: {stage_num}")
    
    return components, summary_plots


def run_stage_with_verification(stage_num: int, u0: float, 
                                prev_stage_results: dict | None = None,
                                results_dir: Path | None = None) -> dict:
    """
    运行指定阶段并验证能量守恒
    
    参数：
        stage_num: 阶段编号（1, 2, 或 3）
        u0: 初始电压（V）
        prev_stage_results: 前一阶段的结果（用于继承初始条件）
    
    返回：
        包含仿真结果和能量验证结果的字典
    """
    print(f"\n{'#'*80}")
    print(f"运行 Stage {stage_num}（初始电压: {u0}V）")
    print(f"{'#'*80}\n")
    
    if stage_num == 1:
        # Stage 1 仿真
        t, u_c, results = simulate_stage1(u0)
        
    elif stage_num == 2:
        # Stage 2 仿真（从Stage 1继承初始条件）
        if prev_stage_results:
            u_c_stage1_final = prev_stage_results['u_c'][-1]
            t_stage1_end = prev_stage_results['t'][-1]
        else:
            # 尝试从CSV文件读取
            results_dir = Path(__file__).parent.parent / "results"
            try:
                t_stage1_end, u_c_stage1_final = get_stage1_final_time_and_u_c(u0, results_dir)
            except FileNotFoundError:
                print(f"警告: 未找到Stage 1的结果文件，使用默认值")
                u_c_stage1_final = u0
                t_stage1_end = 200e-6
        
        t, u_c, i_l, results = simulate_stage2(
            u_c_stage1_final,
            i_l0=0.0,
            t_start=t_stage1_end
        )
        # 将i_l添加到results中以便后续使用
        results['i_l'] = i_l
        
    elif stage_num == 3:
        # Stage 3 仿真（从Stage 2继承初始条件）
        if prev_stage_results:
            u_c_stage2_final = prev_stage_results['u_c'][-1]
            # Stage 2返回的results中包含i_l
            i_l_stage2_final = prev_stage_results['results'].get('i_l', [0.0])[-1]
        else:
            # 尝试从CSV文件读取
            results_dir = Path(__file__).parent.parent / "results"
            try:
                u_c_stage2_final, i_l_stage2_final = get_stage2_final_values(u0, results_dir)
            except FileNotFoundError:
                print(f"警告: 未找到Stage 2的结果文件，使用默认值")
                u_c_stage2_final = u0
                i_l_stage2_final = 0.0
        
        t, u_c, i_l, results = simulate_stage3(
            u_c_stage2_final,
            i_l0=i_l_stage2_final
        )
        # results字典中已经包含i_l，无需重复添加
    else:
        raise ValueError(f"无效的阶段编号: {stage_num}")
    
    # 将t和u_c添加到results中以便后续使用
    results['t'] = t
    results['u_c'] = u_c
    
    # 能量守恒验证
    energy_results = verify_energy_conservation(t, results, f'Stage {stage_num}', u0)
    print_energy_verification(energy_results)
    
    # 计算能量随时间的变化
    energy_time = calculate_energy_over_time(t, results, f'Stage {stage_num}')
    
    # 保存CSV文件
    if results_dir is not None:
        stage_suffix = f"stage_{stage_num}"
        csv_path = results_dir / f"{stage_suffix}_results_u0_{u0:.0f}V.csv"
        save_results_to_csv(t, results, f'Stage {stage_num}', u0, filename=str(csv_path))
        print(f"CSV结果已保存至: {csv_path}")
    
    # 保存PNG图表
    if results_dir is not None:
        components, summary_plots = get_stage_plot_config(stage_num)
        if stage_num == 1:
            img_filename = f"stage1_case{'1' if u0 == U0_CASE1 else '2'}_results.png"
        elif stage_num == 2:
            img_filename = f"stage2_case{'1' if u0 == U0_CASE1 else '2'}_results.png"
        else:  # stage_num == 3
            img_filename = f"stage3_case{'1' if u0 == U0_CASE1 else '2'}_results.png"
        
        img_path = results_dir / img_filename
        plot_circuit_results(t, results, f'Stage {stage_num}', u0,
                            components=components,
                            summary_plots=summary_plots,
                            save_path=str(img_path))
        print(f"电路结果图表已保存至: {img_path}")
    
    return {
        't': t,
        'u_c': u_c,
        'results': results,
        'energy_results': energy_results,
        'energy_time': energy_time
    }


if __name__ == "__main__":
    print(f"\n{'#'*80}")
    print("开始连续仿真：运行所有三个阶段并进行能量守恒验证")
    print(f"{'#'*80}\n")
    
    # 创建 results 文件夹
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # 存储所有能量验证结果
    all_energy_results = []
    
    # 情况1：初始电压为 0V
    print(f"\n{'='*80}")
    print("情况1：初始电压 = 0V")
    print(f"{'='*80}\n")
    
    stage1_results_case1 = run_stage_with_verification(1, U0_CASE1, results_dir=results_dir)
    stage2_results_case1 = run_stage_with_verification(2, U0_CASE1, stage1_results_case1, results_dir=results_dir)
    stage3_results_case1 = run_stage_with_verification(3, U0_CASE1, stage2_results_case1, results_dir=results_dir)
    
    all_energy_results.extend([
        stage1_results_case1['energy_results'],
        stage2_results_case1['energy_results'],
        stage3_results_case1['energy_results']
    ])
    
    # 绘制能量变化图表（情况1）
    print(f"\n{'='*80}")
    print("绘制能量变化图表（情况1：初始电压 = 0V）")
    print(f"{'='*80}\n")
    plot_energy_over_time(stage1_results_case1['energy_time'], 'Stage 1', U0_CASE1, results_dir)
    plot_energy_over_time(stage2_results_case1['energy_time'], 'Stage 2', U0_CASE1, results_dir)
    plot_energy_over_time(stage3_results_case1['energy_time'], 'Stage 3', U0_CASE1, results_dir)
    
    # 情况2：初始电压为 750V
    print(f"\n{'='*80}")
    print("情况2：初始电压 = 750V")
    print(f"{'='*80}\n")
    
    stage1_results_case2 = run_stage_with_verification(1, U0_CASE2, results_dir=results_dir)
    stage2_results_case2 = run_stage_with_verification(2, U0_CASE2, stage1_results_case2, results_dir=results_dir)
    stage3_results_case2 = run_stage_with_verification(3, U0_CASE2, stage2_results_case2, results_dir=results_dir)
    
    all_energy_results.extend([
        stage1_results_case2['energy_results'],
        stage2_results_case2['energy_results'],
        stage3_results_case2['energy_results']
    ])
    
    # 绘制能量变化图表（情况2）
    print(f"\n{'='*80}")
    print("绘制能量变化图表（情况2：初始电压 = 750V）")
    print(f"{'='*80}\n")
    plot_energy_over_time(stage1_results_case2['energy_time'], 'Stage 1', U0_CASE2, results_dir)
    plot_energy_over_time(stage2_results_case2['energy_time'], 'Stage 2', U0_CASE2, results_dir)
    plot_energy_over_time(stage3_results_case2['energy_time'], 'Stage 3', U0_CASE2, results_dir)
    
    # 保存能量验证结果到CSV
    energy_df = pd.DataFrame(all_energy_results)
    energy_csv_path = results_dir / "energy_conservation_verification.csv"
    energy_df.to_csv(energy_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n能量守恒验证结果已保存至: {energy_csv_path}")
    
    # 打印总结
    print(f"\n{'#'*80}")
    print("所有仿真和能量守恒验证已完成！")
    print(f"{'#'*80}\n")
    
    print("能量守恒验证总结:")
    for result in all_energy_results:
        print(f"  {result['stage_name']} (u0={result['u0']}V): "
              f"误差={result['error_percent']:.3f}%, 状态={result['status']}")
