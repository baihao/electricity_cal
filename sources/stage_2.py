"""
Stage 2: 200微秒-50毫秒电路仿真

拓扑说明：
- 电流源 i_s(t) 注入
- 电流分流到三个并联支路：
  1. 整流桥+L支路（L在整流桥内部，与SCR串联）
  2. R支路（电阻R）
  3. C支路（电容C）
- 这些支路汇合后流过Rg
- 拓扑：电流源 → [整流桥(L在内部) | R | C] → Rg → 地

整流桥电路（L在内部）：
- 包含4个二极管（D1, D2, D3, D4）、1个可控硅（SCR1）和1个电感（L）
- L在整流桥内部，与SCR串联
- 当 i_bypass > 0 时：电流路径为 电流源 → D1 → SCR1 → L → D2 → Rg → GND
- 当 i_bypass < 0 时：电流路径为 GND → Rg → D4(反接) → SCR1(反接) → D3(反接) → L(反接) → 电流源
- 整流桥实现全波整流，L的电流 i_L = |i_bypass| 恒为正

电路参数：
- C = 36 mF = 0.036 F
- R = 500 Ω
- Rg = 0.5 Ω
- L = 30 µH = 0.00003 H

非线性器件伏安特性（导通时）：
- D1, D2, D3, D4：U(D) = 0.75 + 0.00007*i(t)（参数相同）
- SCR1：U(SCR1) = 0.88 + 0.000052*i(t)

计算方法：
1. 状态变量选择：
   - u_C：电容电压（也是R两端电压，以及整流桥+L支路总电压）
   - i_bypass：整流桥+L支路电流（可正可负）
   - i_L：L的电流（恒为正，i_L = |i_bypass|，因为L在整流桥内部）

2. KCL方程（在电流源节点）：
   i_s(t) = i_bypass + i_R + i_C
   其中：
   - i_bypass：整流桥+L支路电流（可正可负）
   - i_R = u_C / R：电阻电流
   - i_C = C · du_C/dt：电容电流

3. KVL方程（整流桥+L支路）：
   由于整流桥+L支路与R、C并联：
   u_bypass = u_C
   
   整流桥+L支路内部KVL（考虑L在整流桥内部）：
   - 当 i_bypass > 0 时：电流路径为 电流源 → D1 → SCR1 → L → D2 → Rg → GND
     u_C = U(D1) + U(SCR1) + U(L) + U(D2)
         = (0.75 + 0.00007*i_bypass) + (0.88 + 0.000052*i_bypass) 
           + L·di_bypass/dt + (0.75 + 0.00007*i_bypass)
         = 2.38 + 0.000192*i_bypass + L·di_bypass/dt
     其中 i_L = i_bypass > 0（L的电流恒为正）
   
   - 当 i_bypass < 0 时：电流路径为 GND → Rg → D4(反接) → SCR1(反接) → D3(反接) → L(反接) → 电流源
     u_C = -[U(D4) + U(SCR1) + U(L) + U(D3)]
         = -[(0.75 + 0.00007*|i_bypass|) + (0.88 + 0.000052*|i_bypass|)
           + L·d|i_bypass|/dt + (0.75 + 0.00007*|i_bypass|)]
         = -[2.38 + 0.000192*|i_bypass| + L·d|i_bypass|/dt]
         = -2.38 - 0.000192*|i_bypass| - L·d|i_bypass|/dt
     由于 i_L = |i_bypass| = -i_bypass > 0，所以：
     d|i_bypass|/dt = d(-i_bypass)/dt = -di_bypass/dt
     因此：u_C = -2.38 - 0.000192*|i_bypass| + L·di_bypass/dt
              = -2.38 + 0.000192*i_bypass + L·di_bypass/dt
     其中 i_L = |i_bypass| = -i_bypass > 0（L的电流恒为正）
   
   - 统一表达式：u_C = sign(i_bypass) · 2.38 + 0.000192·i_bypass + L·di_bypass/dt
   
   整理得到：L·di_bypass/dt = u_C - sign(i_bypass)·2.38 - 0.000192·i_bypass
   因此：di_bypass/dt = (u_C - sign(i_bypass)·2.38 - 0.000192·i_bypass) / L

4. 电容支路方程：
   i_C = C · du_C/dt = i_s - i_bypass - i_R
   因此：du_C/dt = (i_s - i_bypass - u_C/R) / C

5. 状态方程（最终形式）：
   du_C/dt = (i_s(t) - i_bypass - u_C/R) / C
   di_bypass/dt = (u_C - sign(i_bypass)·2.38 - 0.000192·i_bypass) / L
   其中：
   - i_bypass：整流桥+L支路电流（可正可负）
   - i_L = |i_bypass|：L的电流（恒为正）

6. Rg支路：
   i_Rg = i_s（由KCL）
   u_Rg = i_Rg · Rg = i_s · Rg

7. 各支路电压：
   - u_C：C两端电压（也是R两端电压，以及整流桥+L支路总电压）
   - u_bypass = u_C：整流桥+L支路总电压（因为并联）
   - u_L：电感两端电压
     * 当 i_bypass > 0 时：L正向接，u_L = L·di_bypass/dt
     * 当 i_bypass < 0 时：L反接，u_L的参考方向相反，u_L = -L·di_bypass/dt
     * 统一表达式：u_L = sign(i_bypass) · L · di_bypass/dt
   - u_Rg：Rg两端电压
   - u_s = u_C + u_Rg：电流源两端电压（由KVL）

注意：
- L在整流桥内部，与SCR串联
- i_bypass：整流桥+L支路的总电流（可正可负）
- i_L = |i_bypass|：L的电流（恒为正，因为L在整流桥内部）
- u_bypass = u_C：因为整流桥+L支路与R、C并联
- 整流桥实现全波整流，保证L的电流恒为正
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
    U_D1_THRESHOLD, U_D2_THRESHOLD, U_SCR_THRESHOLD,
    R_D1, R_D2, R_SCR,
    U_BYPASS_THRESHOLD, R_BYPASS_TOTAL,
    U0_CASE1, U0_CASE2
)

# ==================== 时间参数 ====================
# 时间范围：从Stage 1结束时间到50毫秒
# T_START 将从Stage 1的CSV文件中读取（动态设置）
T_END = 50e-3  # 50 毫秒 = 0.05 秒
DT = 10e-6  # 采样间隔：10 微秒 = 0.00001 秒


def diode_voltage(i: float, u_threshold: float, r: float) -> float:
    """
    计算单个二极管的电压
    
    参数：
        i: 通过二极管的电流（A），正值表示正向电流（从阳极到阴极）
        u_threshold: 二极管阈值电压（V）
        r: 二极管等效电阻（Ω）
    
    返回：
        二极管两端电压（V），定义为"从阳极到阴极"的电压降
        - 当导通时（i > 0）：U = U_threshold + R * i（正值，正向压降）
        - 当不导通时（i <= 0）：返回NaN（电压由外部电路决定）
    
    注意：
        - 电压参考方向：从阳极到阴极（沿着正向电流方向）
        - 当二极管正向导通时，电压为正值（正向压降）
        - 当二极管不导通时，电压由外部电路决定，不能简单地设为0（那意味着短路）
    """
    if abs(i) < 1e-10:  # 电流接近0
        return np.nan  # 不导通，电压由外部电路决定，用NaN表示未定义
    
    if i > 0:
        # 正向导通
        return u_threshold + r * i
    else:
        # 反向不导通（理想二极管反向不导通）
        # 不导通的二极管电压由外部电路决定，不能设为0（那意味着短路）
        return np.nan  # 不导通，电压由外部电路决定，用NaN表示未定义


def scr_voltage(i: float) -> float:
    """
    计算SCR1的电压
    
    参数：
        i: 通过SCR1的电流（A），应该总是大于0（因为SCR1在导通路径上）
    
    返回：
        SCR1两端电压（V）
    
    注意：
        - 当导通时（i > 0）：U = U_threshold + R * i（正向压降）
        - 在整流桥中，SCR1总是在导通路径上，所以i应该总是大于0
    """
    if abs(i) < 1e-10:  # 电流接近0
        return np.nan  # 不导通，电压未定义
    
    if i > 0:
        # 正向导通
        return U_SCR_THRESHOLD + R_SCR * i
    else:
        # 反向不导通（理想情况下）
        # 在整流桥中，SCR1应该在导通路径上，如果i < 0说明逻辑有问题
        return np.nan  # 不导通，电压未定义


def rectifier_bridge_voltage_only(i_bypass: float, epsilon: float = 1e-6) -> float:
    """
    计算整流桥部分（D1-D4 + SCR1，不包括L）的电压
    
    注意：L在整流桥内部，与SCR串联。整流桥+L支路的总电压 u_bypass = u_C。
    整流桥部分的电压 = u_C - u_L = u_C - L·di_bypass/dt
    
    整流桥的伏安特性（不包括L）：
    - 当 i_bypass > 0 时：电流路径为 电流源 → D1 → SCR1 → L → D2 → Rg → GND
      整流桥部分（D1+SCR1+D2）的电压：
      u_bridge = U(D1) + U(SCR1) + U(D2)
               = (0.75 + 0.00007*i_bypass) + (0.88 + 0.000052*i_bypass) + (0.75 + 0.00007*i_bypass)
               = 2.38 + 0.000192*i_bypass
    
    - 当 i_bypass < 0 时：电流路径为 GND → Rg → D4(反接) → SCR1(反接) → D3(反接) → L(反接) → 电流源
      整流桥部分（D4+SCR1+D3）的电压（相对于i_bypass方向）：
      u_bridge = -[U(D4) + U(SCR1) + U(D3)]
               = -[(0.75 + 0.00007*|i_bypass|) + (0.88 + 0.000052*|i_bypass|) + (0.75 + 0.00007*|i_bypass|)]
               = -(2.38 + 0.000192*|i_bypass|)
               = -2.38 + 0.000192*i_bypass  （因为 i_bypass < 0，所以 |i_bypass| = -i_bypass）
    
    - 统一表达式：u_bridge = sign(i_bypass) · U_threshold + R_total · i_bypass
    
    使用平滑过渡函数避免在 i_bypass = 0 处的不连续性，提高数值稳定性。
    
    参数：
        i_bypass: 整流桥+L支路电流（A），可正可负
        epsilon: 平滑过渡参数（默认1e-6），用于tanh函数
    
    返回：
        整流桥部分（不包括L）的电压（V）
    """
    # 使用平滑的符号函数 tanh(i_bypass/epsilon) 代替 sign(i_bypass)
    sign_i_smooth = np.tanh(i_bypass / epsilon)
    
    # 统一表达式：u_bridge = sign(i_bypass) · U_threshold + R_total · i_bypass
    u_bridge = sign_i_smooth * U_BYPASS_THRESHOLD + R_BYPASS_TOTAL * i_bypass
    
    return u_bridge


def calculate_individual_device_voltages(i_l: float, u_c: float) -> dict[str, float]:
    """
    计算各个器件的电压（D1, D2, D3, D4, SCR1）
    
    注意：L在整流桥内部，与SCR串联。i_L恒为正。
    路径判断在外层完成，此函数只计算导通器件的电压。
    
    参数：
        i_l: L的电流（A），恒为正（i_L = |i_bypass|）
        u_c: 电容电压（V）
    
    返回：
        包含各个器件电压的字典：
        - 'u_d1': D1的电压（V），定义为"从阳极到阴极"的电压降
        - 'u_d2': D2的电压（V），定义为"从阳极到阴极"的电压降
        - 'u_d3': D3的电压（V），定义为"从阳极到阴极"的电压降
        - 'u_d4': D4的电压（V），定义为"从阳极到阴极"的电压降
        - 'u_scr1': SCR1的电压（V），定义为"从阳极到阴极"的电压降
    
    注意：
        - i_L恒为正，路径判断在外层根据i_bypass的方向完成
        - 此函数只计算导通器件的电压，不导通的器件电压在外层设为0
    """
    if abs(i_l) < 1e-10:
        # i_L = 0时，所有器件都不导通，电压设为0
        return {
            'u_d1': 0.0, 'u_d2': 0.0, 'u_d3': 0.0, 'u_d4': 0.0, 
            'u_scr1': 0.0
        }
    
    # i_L恒为正，计算器件电压（路径判断在外层完成）
    u_d1 = diode_voltage(i_l, U_D1_THRESHOLD, R_D1)
    u_d2 = diode_voltage(i_l, U_D2_THRESHOLD, R_D2)
    u_d3 = diode_voltage(i_l, U_D1_THRESHOLD, R_D1)
    u_d4 = diode_voltage(i_l, U_D2_THRESHOLD, R_D2)
    u_scr1 = scr_voltage(i_l)
    
    # 处理NaN值
    u_d1 = u_d1 if not np.isnan(u_d1) else 0.0
    u_d2 = u_d2 if not np.isnan(u_d2) else 0.0
    u_d3 = u_d3 if not np.isnan(u_d3) else 0.0
    u_d4 = u_d4 if not np.isnan(u_d4) else 0.0
    u_scr1 = u_scr1 if not np.isnan(u_scr1) else 0.0
    
    return {'u_d1': u_d1, 'u_d2': u_d2, 'u_d3': u_d3, 'u_d4': u_d4, 'u_scr1': u_scr1}


def stage2_ode(t: float, y: np.ndarray) -> np.ndarray:
    """
    状态方程（L在整流桥内部）：
        du_C/dt = (i_s(t) - i_bypass - u_C/R) / C
        di_bypass/dt = (u_C - sign(i_bypass)·2.38 - 0.000192·i_bypass) / L
    
    其中：
    - i_bypass：整流桥+L支路电流（可正可负）
    - i_L = |i_bypass|：L的电流（恒为正）
    - u_bypass = u_C：因为整流桥+L支路与R、C并联
    
    参数：
        t: 时间（秒）
        y: 状态变量数组 [u_C, i_bypass]
    
    返回：
        [du_C/dt, di_bypass/dt]
    """
    u_c = y[0]
    i_bypass = y[1]
    
    # 电流源电流
    i_s = neutral_current(t)
    
    # 电阻电流
    i_r = u_c / R
    
    # 电容电压变化率（KCL：i_C = i_s - i_bypass - i_R）
    du_c_dt = (i_s - i_bypass - i_r) / C
    
    # 计算整流桥部分的电压（不包括L）
    # u_C = u_bridge + u_L = u_bridge + L·di_bypass/dt
    # 其中 u_bridge = sign(i_bypass)·2.38 + 0.000192·i_bypass
    # 因此：L·di_bypass/dt = u_C - u_bridge
    sign_i_smooth = np.tanh(i_bypass / 1e-6)  # 平滑符号函数
    u_bridge = sign_i_smooth * U_BYPASS_THRESHOLD + R_BYPASS_TOTAL * i_bypass
    
    # 整流桥+L支路电流变化率
    # u_C = u_bridge + L·di_bypass/dt
    # 因此：di_bypass/dt = (u_C - u_bridge) / L
    di_bypass_dt = (u_c - u_bridge) / L
    
    return np.array([du_c_dt, di_bypass_dt])


def simulate_stage2(u0: float, i_bypass0: float = 0.0, t_start: float | None = None, 
                    t_end: float = T_END, dt: float = DT, method: str = 'Radau', 
                    rtol: float = 1e-8, atol: float = 1e-10) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    仿真 Stage 2（从Stage 1结束时间到50毫秒）的电路响应
    
    参数：
        u0: 初始电容电压（V），从Stage 1的最终值继承
        i_bypass0: 初始整流桥+L支路电流（A），默认0
        t_start: 仿真开始时间（秒），如果为None则从Stage 1的CSV文件读取
        t_end: 仿真结束时间（秒），默认50毫秒
        dt: 采样间隔（秒），默认10微秒
        method: 求解器方法
        rtol: 相对容差
        atol: 绝对容差
    
    返回：
        t: 时间数组（秒）
        u_c: 电容电压数组（V）
        i_bypass: 整流桥+L支路电流数组（A，可正可负）
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
        y0=[u0, i_bypass0],
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
    i_bypass = sol.y[1]  # 整流桥+L支路电流（可正可负）
    i_l = np.abs(i_bypass)  # L的电流（恒为正）
    
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
    
    # i_bypass：整流桥+L支路电流（可正可负）
    # i_L：L的电流（恒为正，i_L = |i_bypass|）
    
    # 电容电流（由KCL：i_C = i_s - i_bypass - i_R）
    i_c = i_s - i_bypass - i_r
    
    i_rg = i_s  # Rg电流（由KCL）
    
    # 计算各元件电压
    u_rg = i_rg * Rg  # Rg两端电压
    u_s = u_c + u_rg  # 电流源两端电压（由KVL）
    
    # 计算整流桥部分电压和电感电压
    # u_C = u_bridge + u_L（KVL，因为整流桥+L支路与R、C并联）
    # u_bridge = sign(i_bypass)·2.38 + 0.000192·i_bypass
    # u_bypass = u_C（整流桥+L支路总电压）
    
    # 计算di_bypass/dt（从ODE函数中推导）
    sign_i_smooth = np.tanh(i_bypass / 1e-6)
    u_bridge = sign_i_smooth * U_BYPASS_THRESHOLD + R_BYPASS_TOTAL * i_bypass
    di_bypass_dt = (u_c - u_bridge) / L
    
    # 计算u_L：当i_bypass < 0时，L反接，u_L的参考方向相反
    # 当 i_bypass > 0 时：L正向接，u_L = L·di_bypass/dt
    # 当 i_bypass < 0 时：L反接，u_L = -L·di_bypass/dt（参考方向相反）
    # 统一表达式：u_L = sign(i_bypass) · L · di_bypass/dt
    u_l = sign_i_smooth * L * di_bypass_dt  # 电感两端电压（考虑L的接法）
    u_bypass = u_c  # 整流桥+L支路总电压（因为并联）
    
    # 计算各个器件的电压和电流
    # 正向导通路径（i_bypass > 0）：电流源 → D1 → SCR1 → L → D2 → Rg → GND
    # 反向导通路径（i_bypass < 0）：GND → Rg → D4(反接) → SCR1(反接) → D3(反接) → L(反接) → 电流源
    # 注意：i_L = |i_bypass| 恒为正
    i_d1 = np.where(i_bypass > 0, i_bypass, 0.0)  # D1只在i_bypass > 0时导通（正向路径）
    i_d2 = np.where(i_bypass > 0, i_bypass, 0.0)  # D2只在i_bypass > 0时导通（正向路径）
    i_d3 = np.where(i_bypass < 0, -i_bypass, 0.0)  # D3只在i_bypass < 0时导通（反向路径）
    i_d4 = np.where(i_bypass < 0, -i_bypass, 0.0)  # D4只在i_bypass < 0时导通（反向路径）
    i_scr1 = i_l  # SCR1的电流等于L的电流（恒为正，因为串联）
    
    # 计算各个器件的电压
    # 在整流桥中，不导通的二极管与导通路径并联，其电压等于u_C（反向电压）
    u_d1 = np.zeros_like(i_bypass)
    u_d2 = np.zeros_like(i_bypass)
    u_d3 = np.zeros_like(i_bypass)
    u_d4 = np.zeros_like(i_bypass)
    u_scr1 = np.zeros_like(i_bypass)
    
    for idx, i_bypass_val in enumerate(i_bypass):
        # i_L = |i_bypass| 恒为正
        i_l_val = abs(i_bypass_val)
        # 使用calculate_individual_device_voltages函数计算各个器件的电压
        # 注意：这个函数需要i_L的值（恒为正），但我们需要根据i_bypass的方向来判断路径
        if i_bypass_val > 0:
            # 正向路径：D1 → SCR1 → L → D2
            device_voltages = calculate_individual_device_voltages(i_l_val, u_c[idx])
            u_d1[idx] = device_voltages['u_d1']
            u_d2[idx] = device_voltages['u_d2']
            u_scr1[idx] = device_voltages['u_scr1']
            u_d3[idx] = 0.0
            u_d4[idx] = 0.0
        elif i_bypass_val < 0:
            # 反向路径：D4(反接) → SCR1(反接) → D3(反接) → L(反接)
            device_voltages = calculate_individual_device_voltages(-i_bypass_val, u_c[idx])
            u_d3[idx] = device_voltages['u_d3']
            u_d4[idx] = device_voltages['u_d4']
            u_scr1[idx] = device_voltages['u_scr1']
            u_d1[idx] = 0.0
            u_d2[idx] = 0.0
        else:
            # i_bypass = 0
            u_d1[idx] = 0.0
            u_d2[idx] = 0.0
            u_d3[idx] = 0.0
            u_d4[idx] = 0.0
            u_scr1[idx] = 0.0
    
    results = {
        'i_s': i_s,
        'i_l': i_l,  # L的电流（恒为正，i_L = |i_bypass|）
        'i_bypass': i_bypass,  # 整流桥+L支路电流（可正可负）
        'i_r': i_r,
        'i_c': i_c,
        'i_rg': i_rg,
        'i_d1': i_d1,  # D1的电流（正向导通路径：D1-SCR1-L-D2）
        'i_d2': i_d2,  # D2的电流（正向导通路径：D1-SCR1-L-D2）
        'i_d3': i_d3,  # D3的电流（反向导通路径：D4-SCR1-L-D3）
        'i_d4': i_d4,  # D4的电流（反向导通路径：D4-SCR1-L-D3）
        'i_scr1': i_scr1,  # SCR1的电流（等于i_L，恒为正）
        'u_c': u_c,
        'u_l': u_l,  # 电感两端电压
        'u_bypass': u_bypass,  # 整流桥+L支路总电压（等于u_C）
        'u_rg': u_rg,
        'u_s': u_s,
        'u_d1': u_d1,  # D1的电压
        'u_d2': u_d2,  # D2的电压
        'u_d3': u_d3,  # D3的电压
        'u_d4': u_d4,  # D4的电压
        'u_scr1': u_scr1  # SCR1的电压
    }
    
    return t, u_c, i_bypass, results


def print_summary(t: np.ndarray, u_c: np.ndarray, i_bypass: np.ndarray, results: dict, u0: float) -> None:
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
    print(f"  i_bypass(0) = {i_bypass[0]:.6f} A")
    print(f"  i_L(0) = {results['i_l'][0]:.6f} A (恒为正)")
    print(f"  i_s(0) = {results['i_s'][0]:.6f} A")
    print(f"\n最终值（t = {t[-1]*1000:.3f} ms）:")
    print(f"  u_C = {u_c[-1]:.6f} V")
    print(f"  i_bypass = {i_bypass[-1]:.6f} A")
    print(f"  i_L = {results['i_l'][-1]:.6f} A (恒为正)")
    print(f"  i_s = {results['i_s'][-1]:.6f} A")
    print(f"  i_R = {results['i_r'][-1]:.6f} A")
    print(f"  i_C = {results['i_c'][-1]:.6f} A")
    if 'i_d1' in results:
        print(f"  i_D1 = {results['i_d1'][-1]:.6f} A (正向路径：D1-SCR1-L-D2)")
    if 'i_d2' in results:
        print(f"  i_D2 = {results['i_d2'][-1]:.6f} A (正向路径：D1-SCR1-L-D2)")
    if 'i_d3' in results:
        print(f"  i_D3 = {results['i_d3'][-1]:.6f} A (反向路径：D4-SCR1-L-D3)")
    if 'i_d4' in results:
        print(f"  i_D4 = {results['i_d4'][-1]:.6f} A (反向路径：D4-SCR1-L-D3)")
    if 'i_scr1' in results:
        print(f"  i_SCR1 = {results['i_scr1'][-1]:.6f} A (等于i_L，恒为正)")
    print(f"  i_Rg = {results['i_rg'][-1]:.6f} A")
    print(f"  u_s = {results['u_s'][-1]:.6f} V")
    print(f"  u_Rg = {results['u_rg'][-1]:.6f} V")
    print(f"  u_bypass = {results['u_bypass'][-1]:.6f} V (等于u_C)")
    if 'u_d1' in results:
        print(f"  u_D1 = {results['u_d1'][-1]:.6f} V")
    if 'u_d2' in results:
        print(f"  u_D2 = {results['u_d2'][-1]:.6f} V")
    if 'u_d3' in results:
        print(f"  u_D3 = {results['u_d3'][-1]:.6f} V")
    if 'u_d4' in results:
        print(f"  u_D4 = {results['u_d4'][-1]:.6f} V")
    if 'u_scr1' in results:
        print(f"  u_SCR1 = {results['u_scr1'][-1]:.6f} V")
    print(f"\n验证（KCL）:")
    print(f"  i_s = i_bypass + i_R + i_C:")
    print(f"    {results['i_s'][-1]:.6f} ≈ {i_bypass[-1] + results['i_r'][-1] + results['i_c'][-1]:.6f}")
    print(f"  误差: {abs(results['i_s'][-1] - (i_bypass[-1] + results['i_r'][-1] + results['i_c'][-1])):.6e} A")
    print(f"\n注意：")
    print(f"  - i_bypass：整流桥+L支路电流（可正可负）")
    print(f"  - i_L = |i_bypass|：L的电流（恒为正，因为L在整流桥内部）")
    print(f"  - u_bypass = u_C：因为整流桥+L支路与R、C并联")
    print(f"  - 正向导通路径（i_bypass > 0）：电流源 → D1 → SCR1 → L → D2 → Rg → GND")
    print(f"  - 反向导通路径（i_bypass < 0）：GND → Rg → D4(反接) → SCR1(反接) → D3(反接) → L(反接) → 电流源")
    print(f"  - 不导通的二极管承受反向电压u_C（与导通路径并联）")
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
    
    t1, u_c1, i_bypass1, results1 = simulate_stage2(u0_case1, t_start=t_start_case1)
    print_summary(t1, u_c1, i_bypass1, results1, u0_case1)
    
    # 定义Stage 2的绘图配置
    stage2_components = [
        ('电流源', 'i_s', 'u_s', '电流源'),
        ('电感 L', 'i_l', 'u_l', '电感'),
        ('整流桥', 'i_bypass', 'u_bypass', '整流桥'),
        ('二极管 D1', 'i_d1', 'u_d1', 'D1'),
        ('二极管 D2', 'i_d2', 'u_d2', 'D2'),
        ('二极管 D3', 'i_d3', 'u_d3', 'D3'),
        ('二极管 D4', 'i_d4', 'u_d4', 'D4'),
        ('可控硅 SCR1', 'i_scr1', 'u_scr1', 'SCR1'),
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
    
    t2, u_c2, i_bypass2, results2 = simulate_stage2(u0_case2, t_start=t_start_case2)
    print_summary(t2, u_c2, i_bypass2, results2, u0_case2)
    
    csv_path2 = results_dir / f"stage_2_results_u0_{stage1_u0_case2:.0f}V.csv"
    save_results_to_csv(t2, results2, 'Stage 2', stage1_u0_case2, filename=str(csv_path2))
    
    # 绘制结果图表
    img_path2 = results_dir / f"stage2_case2_results.png"
    plot_circuit_results(t2, results2, 'Stage 2', stage1_u0_case2,
                        components=stage2_components,
                        summary_plots=stage2_summary_plots,
                        save_path=str(img_path2))

