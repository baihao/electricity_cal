"""
Stage 2: 200微秒-50毫秒电路仿真

拓扑说明：
- 电流源 i_s(t) 注入
- 电流分流到三个并联支路：
  1. L支路（电感L）
  2. R支路（电阻R）
  3. C支路（电容C）
- 这些支路汇合后流过Rg
- 拓扑：电流源 → [L | R | C] → Rg → 地

电路参数：
- C = 36 mF = 0.036 F
- R = 500 Ω
- Rg = 0.5 Ω
- L = 30 µH = 0.00003 H

计算方法：
1. 状态变量选择：
   - u_C：电容电压（也是R两端电压，以及L支路电压）
   - i_L：电感电流

2. KCL方程（在电流源节点）：
   i_s(t) = i_L + i_R + i_C
   其中：
   - i_L：电感电流
   - i_R = u_C / R：电阻电流
   - i_C = C · du_C/dt：电容电流

3. KVL方程（L支路）：
   由于L与R、C并联：
   u_C = u_L = L · di_L/dt
   因此：di_L/dt = u_C / L

4. 电容支路方程：
   i_C = C · du_C/dt = i_s - i_L - i_R
   因此：du_C/dt = (i_s - i_L - u_C/R) / C

5. 状态方程：
   du_C/dt = (i_s(t) - i_L - u_C/R) / C
   di_L/dt = u_C / L

6. Rg支路：
   i_Rg = i_s（由KCL）
   u_Rg = i_Rg · Rg = i_s · Rg

7. 各支路电压：
   - u_C：C两端电压（也是R两端电压，以及L支路电压）
   - u_L = u_C：电感两端电压（与C、R并联）
   - u_Rg：Rg两端电压
   - u_s = u_C + u_Rg：电流源两端电压（由KVL）
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from pathlib import Path

from source import neutral_current
from graph import plot_circuit_results
from serialize import save_results_to_csv
from reader import get_stage1_final_values, get_stage1_final_time_and_u_c
from circuit_params import (
    C, R, Rg, L,
    U0_CASE1, U0_CASE2
)

# ==================== 时间参数 ====================
# 时间范围：从Stage 1结束时间到50毫秒
# T_START 将从Stage 1的CSV文件中读取（动态设置）
T_END = 50e-3  # 50 毫秒 = 0.05 秒
DT = 10e-6  # 采样间隔：10 微秒 = 0.00001 秒


def stage2_ode(t: float, y: np.ndarray) -> np.ndarray:
    """
    状态方程：
        du_C/dt = (i_s(t) - i_L - u_C/R) / C
        di_L/dt = u_C / L
    
    参数：
        t: 时间（秒）
        y: 状态变量数组 [u_C, i_L]
    
    返回：
        [du_C/dt, di_L/dt]
    """
    u_c = y[0]
    i_l = y[1]
    
    # 电流源电流
    i_s = neutral_current(t)
    
    # 电阻电流
    i_r = u_c / R
    
    # 电容电压变化率（KCL：i_C = i_s - i_L - i_R）
    du_c_dt = (i_s - i_l - i_r) / C
    
    # 电感电流变化率
    # KVL: u_C = u_L（L与C、R并联）
    # u_L = L · di_L/dt
    # 因此：di_L/dt = u_C / L
    di_l_dt = u_c / L
    
    return np.array([du_c_dt, di_l_dt])


def simulate_stage2(u0: float, i_l0: float = 0.0, t_start: float | None = None, 
                    t_end: float = T_END, dt: float = DT, method: str = 'Radau', 
                    rtol: float = 1e-8, atol: float = 1e-10) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    仿真 Stage 2（从Stage 1结束时间到50毫秒）的电路响应
    
    参数：
        u0: 初始电容电压（V），从Stage 1的最终值继承
        i_l0: 初始电感电流（A），默认0
        t_start: 仿真开始时间（秒），如果为None则从Stage 1的CSV文件读取
        t_end: 仿真结束时间（秒），默认50毫秒
        dt: 采样间隔（秒），默认10微秒
        method: 求解器方法
        rtol: 相对容差
        atol: 绝对容差
    
    返回：
        t: 时间数组（秒）
        u_c: 电容电压数组（V）
        i_l: 电感电流数组（A）
        results: 字典，包含各支路电流和电压
    """
    # 如果t_start为None，尝试从Stage 1的CSV文件读取
    if t_start is None:
        try:
            results_dir = Path(__file__).parent.parent / "results"
            t_start, _ = get_stage1_final_time_and_u_c(u0, results_dir)
            print(f"从Stage 1 CSV文件读取开始时间: t_start = {t_start*1000:.6f} ms")
        except (FileNotFoundError, KeyError) as e:
            print(f"警告: 无法从Stage 1 CSV文件读取开始时间: {e}")
            print(f"使用默认值: t_start = 200e-6 (200微秒)")
            t_start = 200e-6
    
    # 时间采样点（确保不超过t_end）
    t_eval = np.arange(t_start, t_end + dt / 2, dt)
    t_eval = t_eval[t_eval <= t_end]  # 确保所有值都在t_span范围内
    
    # 求解ODE
    sol = solve_ivp(
        fun=stage2_ode,
        t_span=(t_start, t_end),
        y0=[u0, i_l0],
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol
    )
    
    # 检查求解器是否成功
    if not sol.success:
        print(f"警告: ODE求解器失败!")
        print(f"  消息: {sol.message}")
        print(f"  返回的时间点数: {len(sol.t)}")
        if hasattr(sol, 't_events') and sol.t_events:
            print(f"  事件时间: {sol.t_events}")
    
    t = sol.t
    u_c = sol.y[0]
    i_l = sol.y[1]
    
    # 检查结果是否为空
    if len(t) == 0:
        raise RuntimeError(f"求解器返回空结果！时间范围: {t_start} 到 {t_end}")
    
    if len(t) == 1:
        print(f"警告: 求解器只返回了1个时间点!")
        print(f"  t_start = {t_start}, t_end = {t_end}, dt = {dt}")
        print(f"  预期时间点数: {len(t_eval)}")
        print(f"  实际返回点数: {len(t)}")
        print(f"  求解器成功: {sol.success}")
        if hasattr(sol, 'message'):
            print(f"  消息: {sol.message}")
    
    # 计算各支路电流
    i_s = neutral_current(t)  # 电流源电流
    i_r = u_c / R  # 电阻电流
    
    # 电容电流（由KCL：i_C = i_s - i_L - i_R）
    i_c = i_s - i_l - i_r
    
    i_rg = i_s  # Rg电流（由KCL）
    
    # 计算各元件电压
    u_rg = i_rg * Rg  # Rg两端电压
    u_s = u_c + u_rg  # 电流源两端电压（由KVL）
    
    # 计算电感电压
    # L与C、R并联，所以 u_L = u_C
    u_l = u_c
    
    results = {
        'i_s': i_s,
        'i_l': i_l,
        'i_r': i_r,
        'i_c': i_c,
        'i_rg': i_rg,
        'u_c': u_c,
        'u_l': u_l,  # 电感两端电压（等于u_C）
        'u_rg': u_rg,
        'u_s': u_s,
    }
    
    return t, u_c, i_l, results


def print_summary(t: np.ndarray, u_c: np.ndarray, i_l: np.ndarray, results: dict, u0: float) -> None:
    """
    打印仿真结果摘要
    """
    print(f"\n{'='*60}")
    print(f"Stage 2 仿真结果摘要（初始电压: {u0}V）")
    print(f"{'='*60}")
    print(f"时间范围: {t[0]*1000:.3f} ms 到 {t[-1]*1000:.3f} ms")
    print(f"采样点数: {len(t)}")
    print(f"\n初始值:")
    print(f"  u_C(0) = {u_c[0]:.6f} V")
    print(f"  i_L(0) = {i_l[0]:.6f} A")
    print(f"  i_s(0) = {results['i_s'][0]:.6f} A")
    print(f"\n最终值（t = {t[-1]*1000:.3f} ms）:")
    print(f"  u_C = {u_c[-1]:.6f} V")
    print(f"  i_L = {i_l[-1]:.6f} A")
    print(f"  i_s = {results['i_s'][-1]:.6f} A")
    print(f"  i_R = {results['i_r'][-1]:.6f} A")
    print(f"  i_C = {results['i_c'][-1]:.6f} A")
    print(f"  i_Rg = {results['i_rg'][-1]:.6f} A")
    print(f"  u_s = {results['u_s'][-1]:.6f} V")
    print(f"  u_C = {u_c[-1]:.6f} V")
    print(f"  u_L = {results['u_l'][-1]:.6f} V")
    print(f"  u_Rg = {results['u_rg'][-1]:.6f} V")
    print(f"\n验证（KCL）:")
    print(f"  i_s = i_L + i_R + i_C:")
    print(f"    {results['i_s'][-1]:.6f} ≈ {i_l[-1] + results['i_r'][-1] + results['i_c'][-1]:.6f}")
    print(f"  误差: {abs(results['i_s'][-1] - (i_l[-1] + results['i_r'][-1] + results['i_c'][-1])):.6e} A")
    print(f"\n注意：")
    print(f"  - 拓扑：电流源 → [L | R | C] → Rg → 地")
    print(f"  - L、R、C并联，所以 u_L = u_C = u_R")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # 创建 results 文件夹（如果不存在）
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    print(f"输出目录: {results_dir}\n")
    
    # 从Stage 1继承初始条件
    print("开始仿真 Stage 2：200微秒-50毫秒")
    print("从Stage 1的最终值继承初始条件\n")
    
    # 情况1：初始电压为 0V（从Stage 1继承）
    stage1_u0_case1 = U0_CASE1
    try:
        t_stage1_end, u_c_stage1_final = get_stage1_final_time_and_u_c(stage1_u0_case1, results_dir)
        print(f"从Stage 1 (u0={stage1_u0_case1}V) 继承:")
        print(f"  结束时间: t = {t_stage1_end*1000:.6f} ms")
        print(f"  u_C = {u_c_stage1_final:.6f} V")
        u0_case1 = u_c_stage1_final
        t_start_case1 = t_stage1_end
    except FileNotFoundError:
        print(f"警告: 未找到Stage 1的结果文件 (u0={stage1_u0_case1}V)，使用默认值")
        u0_case1 = U0_CASE1
        t_start_case1 = 200e-6  # 默认200微秒
    
    t1, u_c1, i_l1, results1 = simulate_stage2(u0_case1, t_start=t_start_case1)
    print_summary(t1, u_c1, i_l1, results1, u0_case1)
    
    # 定义Stage 2的绘图配置
    stage2_components = [
        ('电流源', 'i_s', 'u_s', '电流源'),
        ('电感 L', 'i_l', 'u_l', '电感'),
        ('电阻 R', 'i_r', 'u_c', '电阻'),
        ('电容 C', 'i_c', 'u_c', '电容'),
        ('电阻 Rg', 'i_rg', 'u_rg', 'Rg'),
    ]
    
    stage2_summary_plots = [
        {
            'title': 'R, L, C 和电流源电流变化',
            'ylabel': '电流 (kA)',
            'items': [
                ('i_s', 'i_s (电流源)', '-', 2),
                ('i_r', 'i_R (电阻)', '-', 1.5),
                ('i_c', 'i_C (电容)', '-', 1.5),
                ('i_l', 'i_L (电感)', '-', 1.5),
            ]
        },
        {
            'title': 'R, L, C 和电流源电压变化',
            'ylabel': '电压 (kV)',
            'items': [
                ('u_s', 'u_s (电流源)', '-', 2),
                ('u_c', 'u_R = u_C = u_L (电阻、电容和电感)', '-', 1.5),
            ]
        }
    ]
    
    # 保存结果
    csv_path1 = results_dir / f"stage_2_results_u0_{stage1_u0_case1:.0f}V.csv"
    save_results_to_csv(t1, results1, 'Stage 2', stage1_u0_case1, filename=str(csv_path1))
    
    # 绘制结果图表
    img_path1 = results_dir / f"stage2_case1_results.png"
    plot_circuit_results(t1, results1, 'Stage 2', stage1_u0_case1,
                        components=stage2_components,
                        summary_plots=stage2_summary_plots,
                        save_path=str(img_path1))
    
    # 情况2：初始电压为 750V（从Stage 1继承）
    stage1_u0_case2 = U0_CASE2
    try:
        t_stage1_end, u_c_stage1_final = get_stage1_final_time_and_u_c(stage1_u0_case2, results_dir)
        print(f"\n从Stage 1 (u0={stage1_u0_case2}V) 继承:")
        print(f"  结束时间: t = {t_stage1_end*1000:.6f} ms")
        print(f"  u_C = {u_c_stage1_final:.6f} V")
        u0_case2 = u_c_stage1_final
        t_start_case2 = t_stage1_end
    except FileNotFoundError:
        print(f"警告: 未找到Stage 1的结果文件 (u0={stage1_u0_case2}V)，使用默认值")
        u0_case2 = U0_CASE2
        t_start_case2 = 200e-6  # 默认200微秒
    
    t2, u_c2, i_l2, results2 = simulate_stage2(u0_case2, t_start=t_start_case2)
    print_summary(t2, u_c2, i_l2, results2, u0_case2)
    
    csv_path2 = results_dir / f"stage_2_results_u0_{stage1_u0_case2:.0f}V.csv"
    save_results_to_csv(t2, results2, 'Stage 2', stage1_u0_case2, filename=str(csv_path2))
    
    # 绘制结果图表
    img_path2 = results_dir / f"stage2_case2_results.png"
    plot_circuit_results(t2, results2, 'Stage 2', stage1_u0_case2,
                        components=stage2_components,
                        summary_plots=stage2_summary_plots,
                        save_path=str(img_path2))

