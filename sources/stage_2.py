"""
Stage 2: 200微秒-50毫秒电路仿真

拓扑说明：
- 电流源 i_s(t) 注入
- 电流分流到三个并联支路：
  1. L-整流桥支路（电感L和整流桥串联）
  2. R支路（电阻R）
  3. C支路（电容C）
- 这些支路汇合后流过Rg
- 拓扑：电流源 → [L-整流桥 | R | C] → Rg → 地

整流桥电路：
- 包含4个二极管（D1, D2, D3, D4）和1个可控硅（SCR1）
- 当 i_L > 0 时：电流路径为 L → D1 → SCR1 → D2
- 当 i_L < 0 时：电流路径为 L → D3（反接）→ SCR1（反接）→ D4（反接）
- 整流桥实现全波整流，无论电流方向如何都能导通

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
   - u_C：电容电压（也是R两端电压，以及L-D1-D2-SCR1支路总电压）
   - i_L：电感电流（也是L-D1-D2-SCR1支路电流，因为串联）

2. KCL方程（在电流源节点）：
   i_s(t) = i_L + i_R + i_C
   其中：
   - i_L：电感电流（也是L-D1-D2-SCR1支路电流，因为串联）
   - i_R = u_C / R：电阻电流
   - i_C = C · du_C/dt：电容电流

3. KVL方程（L-整流桥支路）：
   由于L和整流桥串联，且与R、C并联：
   u_C = u_L + u_bypass
   其中：
   - u_L = L · di_L/dt：电感两端电压
   - u_bypass：整流桥两端电压（根据电流方向计算）
   
   整流桥的伏安特性（考虑电流方向）：
   - 当 i_L > 0 时：电流路径为 L → D1(正接) → SCR1(正接) → D2(正接) → 电流源
     正接：L接D的正极，D的负极接另一端
     u_bypass = U(D1) + U(SCR1) + U(D2)
              = (0.75 + 0.00007*i_L) + (0.88 + 0.000052*i_L) + (0.75 + 0.00007*i_L)
              = 2.38 + 0.000192*i_L
   
   - 当 i_L < 0 时：电流路径为 GND → Rg → D4(反接) → SCR1(反接) → D3(反接) → L → 电流源
     反接：L接D的负极，D的正极接另一端
     因为 i_L < 0，电流从GND流向电流源（通过整流桥），所以D4、SCR1、D3需要反接才能导通
     由于电流方向相反，整流桥的电压降方向也相反，u_bypass为负值：
     u_bypass = -[U(D4) + U(SCR1) + U(D3)]
              = -[(0.75 + 0.00007*|i_L|) + (0.88 + 0.000052*|i_L|) + (0.75 + 0.00007*|i_L|)]
              = -(2.38 + 0.000192*|i_L|)
              = -2.38 - 0.000192*|i_L|
              = -2.38 + 0.000192*i_L  （因为 i_L < 0，所以 |i_L| = -i_L，因此 -|i_L| = i_L）
     注意：u_bypass为负值，表示电压降方向与i_L > 0时相反
   
   - 统一表达式：u_bypass = sign(i_L) * 2.38 + 0.000192 * i_L
   
   因此：u_C = L · di_L/dt + u_bypass
   整理得到：di_L/dt = (u_C - u_bypass) / L

4. 电容支路方程：
   i_C = C · du_C/dt = i_s - i_L - i_R
   因此：du_C/dt = (i_s - i_L - u_C/R) / C

5. 状态方程（考虑整流桥的电流方向）：
   du_C/dt = (i_s(t) - i_L - u_C/R) / C
   di_L/dt = (u_C - u_bypass) / L
   其中 u_bypass 根据 i_L 的方向计算：
   - i_L > 0：u_bypass = 2.38 + 0.000192*i_L（D1-D2-SCR1路径）
   - i_L < 0：u_bypass = -2.38 + 0.000192*i_L（D3-D4-SCR1路径）
   - i_L = 0：u_bypass = 0

6. Rg支路：
   i_Rg = i_s（由KCL）
   u_Rg = i_Rg · Rg = i_s · Rg

7. 各支路电压：
   - u_C：C两端电压（也是R两端电压，以及L-整流桥支路总电压）
   - u_bypass：整流桥两端电压（根据电流方向计算）
     * i_L > 0：u_bypass = 2.38 + 0.000192*i_L（D1-D2-SCR1路径）
     * i_L < 0：u_bypass = -2.38 + 0.000192*i_L（D3-D4-SCR1路径）
     * i_L = 0：u_bypass = 0
   - u_L = u_C - u_bypass：电感两端电压
   - u_Rg：Rg两端电压
   - u_s = u_C + u_Rg：电流源两端电压（由KVL）

注意：
- 由于L和整流桥串联，它们的电流相同（i_L = i_bypass）
- 整流桥实现全波整流：无论电流方向如何，都能保证电流单向通过
- 当 i_L > 0 时，使用 D1-D2-SCR1 路径
- 当 i_L < 0 时，使用 D3-D4-SCR1 路径（反接）
- 整流桥的电压降方向与电流方向一致
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from pathlib import Path

from source import neutral_current
from graph import plot_circuit_results
from serialize import save_results_to_csv
from reader import get_stage1_final_values

# ==================== 电路参数 ====================
C = 36e-3  # 36 mF = 0.036 F
R = 500.0  # 500 Ω
Rg = 0.5  # 0.5 Ω
L = 30e-6  # 30 µH = 0.00003 H

# 非线性器件参数
U_D1_THRESHOLD = 0.75  # D1、D2导通阈值电压（V）
U_D2_THRESHOLD = 0.75  # D2导通阈值电压（V）
U_SCR_THRESHOLD = 0.88  # SCR1导通阈值电压（V）
R_D1 = 0.00007  # D1、D2等效电阻（Ω）
R_D2 = 0.00007  # D2等效电阻（Ω）
R_SCR = 0.000052  # SCR1等效电阻（Ω）

# 电子旁路总阈值电压（D1 + D2 + SCR1）
U_BYPASS_THRESHOLD = U_D1_THRESHOLD + U_D2_THRESHOLD + U_SCR_THRESHOLD  # 2.38 V
R_BYPASS_TOTAL = R_D1 + R_D2 + R_SCR  # 0.000192 Ω

# 时间范围：200微秒到50毫秒
T_START = 200e-6  # 200 微秒 = 0.0002 秒
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


def rectifier_bridge_voltage(i_l: float, epsilon: float = 1e-6) -> float:
    """
    计算整流桥（D1-D4 + SCR1）的电压，考虑电流方向
    
    整流桥的伏安特性（u_bypass定义为从L端到GND端的电压）：
    - 当 i_L > 0 时：电流路径为 L → D1(正接) → SCR1(正接) → D2(正接) → 电流源
      电流从L流向GND，u_bypass为正值：
      u_bypass = 2.38 + 0.000192*i_L
    
    - 当 i_L < 0 时：电流路径为 GND → Rg → D4(反接) → SCR1(反接) → D3(反接) → L → 电流源
      电流从GND流向L，u_bypass为负值（电压降方向相反）：
      u_bypass = -2.38 + 0.000192*i_L
      注意：当i_L < 0时，u_bypass为负值，因为电压降方向与i_L > 0时相反
    
    - 当 i_L = 0 时：u_bypass = 0
    
    统一表达式：u_bypass = sign(i_L) * U_threshold + R_total * i_L
    
    使用平滑过渡函数避免在 i_L = 0 处的不连续性，提高数值稳定性。
    
    参数：
        i_l: 通过整流桥的电流（A）
        epsilon: 平滑过渡参数（默认1e-6），用于tanh函数
    
    返回：
        整流桥两端电压（V），从L端到GND端
    """
    # 使用平滑的符号函数 tanh(i_L/epsilon) 代替 sign(i_L)
    # 这样可以避免在 i_L = 0 处的不连续性
    sign_i_smooth = np.tanh(i_l / epsilon)
    
    # 统一表达式：u_bypass = sign(i_L) * U_threshold + R_total * i_L
    # 使用平滑的符号函数
    u_bypass = sign_i_smooth * U_BYPASS_THRESHOLD + R_BYPASS_TOTAL * i_l
    
    return u_bypass


def calculate_individual_device_voltages(i_l: float, u_c: float) -> dict[str, float]:
    """
    计算各个器件的电压（D1, D2, D3, D4, SCR1）
    
    参数：
        i_l: 通过整流桥的电流（A）
        u_c: 电容电压（V），用于计算不导通二极管的反向电压（已不使用）
    
    返回：
        包含各个器件电压的字典：
        - 'u_d1': D1的电压（V），定义为"从阳极到阴极"的电压降
        - 'u_d2': D2的电压（V），定义为"从阳极到阴极"的电压降
        - 'u_d3': D3的电压（V），定义为"从阳极到阴极"的电压降
        - 'u_d4': D4的电压（V），定义为"从阳极到阴极"的电压降
        - 'u_scr1': SCR1的电压（V），定义为"从阳极到阴极"的电压降
    
    注意：
        - 电压参考方向：从阳极到阴极（沿着正向电流方向）
        - 正向导通路径（i_L > 0）：L → D1(正接) → SCR1(正接) → D2(正接) → 电流源
          * D1、D2、SCR1导通，电压为正值（正向压降）
          * D3、D4不导通，电压设为0（因为不导通的器件是串联拓扑，单个器件电压无意义）
        - 反向导通路径（i_L < 0）：电流源 → D4(反接) → SCR1(反接) → D3(反接) → L
          * D3、D4、SCR1导通，电压为正值（相对于负电流方向是正向）
          * D1、D2不导通，电压设为0（因为不导通的器件是串联拓扑，单个器件电压无意义）
        - 不导通的二极管和SCR是串联拓扑，单个器件的电压没有意义，因此设为0
    """
    if abs(i_l) < 1e-10:
        # i_L = 0时，所有器件都不导通，电压设为0
        return {
            'u_d1': 0.0, 'u_d2': 0.0, 'u_d3': 0.0, 'u_d4': 0.0, 
            'u_scr1': 0.0  # SCR1不导通时电压为0
        }
    
    if i_l > 0:
        # 正向导通路径：L → D1(正接) → SCR1(正接) → D2(正接) → 电流源
        u_d1 = diode_voltage(i_l, U_D1_THRESHOLD, R_D1)  # D1导通
        u_scr1 = scr_voltage(i_l)  # SCR1导通
        u_d2 = diode_voltage(i_l, U_D2_THRESHOLD, R_D2)  # D2导通
        # D3和D4不导通，电压设为0（串联拓扑，单个器件电压无意义）
        u_d3 = 0.0
        u_d4 = 0.0
    else:
        # 反向导通路径：电流源 → D4(反接) → SCR1(反接) → D3(反接) → L
        # 因为i_L < 0，电流从电流源流向L，D4、SCR1、D3反接才能导通
        u_d4 = diode_voltage(-i_l, U_D2_THRESHOLD, R_D2)  # D4导通（反接，相对于负电流是正向）
        u_scr1 = scr_voltage(-i_l)  # SCR1导通（反接，相对于负电流是正向）
        u_d3 = diode_voltage(-i_l, U_D1_THRESHOLD, R_D1)  # D3导通（反接，相对于负电流是正向）
        # D1和D2不导通，电压设为0（串联拓扑，单个器件电压无意义）
        u_d1 = 0.0
        u_d2 = 0.0
    
    return {'u_d1': u_d1, 'u_d2': u_d2, 'u_d3': u_d3, 'u_d4': u_d4, 'u_scr1': u_scr1}


def stage2_ode(t: float, y: np.ndarray) -> np.ndarray:
    """
    状态方程，考虑整流桥的电流方向：
        du_C/dt = (i_s(t) - i_L - u_C/R) / C
        di_L/dt = (u_C - u_bypass) / L
    
    其中 u_bypass 根据电流方向计算：
    - 当 i_L > 0 时：u_bypass = 2.38 + 0.000192*i_L（D1-D2-SCR1路径）
    - 当 i_L < 0 时：u_bypass = -2.38 + 0.000192*i_L（D3-D4-SCR1路径）
    - 当 i_L = 0 时：u_bypass = 0
    
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
    
    # 计算整流桥电压（考虑电流方向）
    u_bypass = rectifier_bridge_voltage(i_l)
    
    # 电感电流变化率
    # KVL: u_C = u_L + u_bypass
    # u_L = L · di_L/dt
    # 因此：di_L/dt = (u_C - u_bypass) / L
    di_l_dt = (u_c - u_bypass) / L
    
    return np.array([du_c_dt, di_l_dt])


def simulate_stage2(u0: float, i_l0: float = 0.0, t_start: float = T_START, 
                    t_end: float = T_END, dt: float = DT, method: str = 'Radau', 
                    rtol: float = 1e-8, atol: float = 1e-10) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    仿真 Stage 2（200微秒-50毫秒）的电路响应
    
    参数：
        u0: 初始电容电压（V），从Stage 1的最终值继承
        i_l0: 初始电感电流（A），默认0
        t_start: 仿真开始时间（秒），默认200微秒
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
    # 时间采样点
    t_eval = np.arange(t_start, t_end + dt / 2, dt)
    
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
    
    # L和D1-D2-SCR1串联，电流相同：i_L就是L-D1-D2-SCR1支路电流
    i_bypass = i_l  # 电子旁路电流等于电感电流（串联）
    
    # 电容电流（由KCL：i_C = i_s - i_L - i_R）
    i_c = i_s - i_l - i_r
    
    i_rg = i_s  # Rg电流（由KCL）
    
    # 计算各元件电压
    u_rg = i_rg * Rg  # Rg两端电压
    u_s = u_c + u_rg  # 电流源两端电压（由KVL）
    
    # 计算电感电压和整流桥电压
    # u_C = u_L + u_bypass（KVL）
    # u_bypass 根据电流方向计算（考虑整流桥的电流方向）
    # u_L = u_C - u_bypass
    
    u_bypass = np.array([rectifier_bridge_voltage(i) for i in i_l])  # 对每个时间点计算
    u_l = u_c - u_bypass
    
    # 计算各个器件的电压和电流
    # 正向导通路径（i_L > 0）：L → D1(正接) → SCR1(正接) → D2(正接) → 电流源
    # 反向导通路径（i_L < 0）：电流源 → D4(反接) → SCR1(反接) → D3(反接) → L
    i_d1 = np.where(i_l > 0, i_l, 0.0)  # D1只在i_L > 0时导通（正向路径）
    i_d2 = np.where(i_l > 0, i_l, 0.0)  # D2只在i_L > 0时导通（正向路径）
    i_d3 = np.where(i_l < 0, -i_l, 0.0)  # D3只在i_L < 0时导通（反向路径：D4-SCR1-D3）
    i_d4 = np.where(i_l < 0, -i_l, 0.0)  # D4只在i_L < 0时导通（反向路径：D4-SCR1-D3）
    i_scr1 = np.abs(i_l)  # SCR1在两种情况下都导通（相对于电流方向）
    
    # 计算各个器件的电压
    # 在整流桥中，不导通的二极管与导通路径并联，其电压等于u_C（反向电压）
    u_d1 = np.zeros_like(i_l)
    u_d2 = np.zeros_like(i_l)
    u_d3 = np.zeros_like(i_l)
    u_d4 = np.zeros_like(i_l)
    u_scr1 = np.zeros_like(i_l)
    
    for idx, i_val in enumerate(i_l):
        # 使用calculate_individual_device_voltages函数计算各个器件的电压
        device_voltages = calculate_individual_device_voltages(i_val, u_c[idx])
        u_d1[idx] = device_voltages['u_d1']
        u_d2[idx] = device_voltages['u_d2']
        u_d3[idx] = device_voltages['u_d3']
        u_d4[idx] = device_voltages['u_d4']
        u_scr1[idx] = device_voltages['u_scr1']
    
    results = {
        'i_s': i_s,
        'i_l': i_l,
        'i_r': i_r,
        'i_c': i_c,
        'i_bypass': i_bypass,  # 等于i_L（串联）
        'i_rg': i_rg,
        'i_d1': i_d1,  # D1的电流（正向导通路径：D1-SCR1-D2）
        'i_d2': i_d2,  # D2的电流（正向导通路径：D1-SCR1-D2）
        'i_d3': i_d3,  # D3的电流（反向导通路径：D3-SCR1-D4）
        'i_d4': i_d4,  # D4的电流（反向导通路径：D3-SCR1-D4）
        'i_scr1': i_scr1,  # SCR1的电流
        'u_c': u_c,
        'u_l': u_l,  # 电感两端电压
        'u_bypass': u_bypass,  # 整流桥两端电压
        'u_rg': u_rg,
        'u_s': u_s,
        'u_d1': u_d1,  # D1的电压
        'u_d2': u_d2,  # D2的电压
        'u_d3': u_d3,  # D3的电压
        'u_d4': u_d4,  # D4的电压
        'u_scr1': u_scr1  # SCR1的电压
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
    print(f"  i_bypass = {results['i_bypass'][-1]:.6f} A")
    if 'i_d1' in results:
        print(f"  i_D1 = {results['i_d1'][-1]:.6f} A (正向路径：D1-SCR1-D2)")
    if 'i_d2' in results:
        print(f"  i_D2 = {results['i_d2'][-1]:.6f} A (正向路径：D1-SCR1-D2)")
    if 'i_d3' in results:
        print(f"  i_D3 = {results['i_d3'][-1]:.6f} A (反向路径：D3-SCR1-D4)")
    if 'i_d4' in results:
        print(f"  i_D4 = {results['i_d4'][-1]:.6f} A (反向路径：D3-SCR1-D4)")
    if 'i_scr1' in results:
        print(f"  i_SCR1 = {results['i_scr1'][-1]:.6f} A")
    print(f"  i_Rg = {results['i_rg'][-1]:.6f} A")
    print(f"  u_s = {results['u_s'][-1]:.6f} V")
    print(f"  u_Rg = {results['u_rg'][-1]:.6f} V")
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
    print(f"  i_s = i_L + i_R + i_C:")
    print(f"    {results['i_s'][-1]:.6f} ≈ {i_l[-1] + results['i_r'][-1] + results['i_c'][-1]:.6f}")
    print(f"  误差: {abs(results['i_s'][-1] - (i_l[-1] + results['i_r'][-1] + results['i_c'][-1])):.6e} A")
    print(f"\n注意：")
    print(f"  - i_bypass = i_L（L和整流桥串联，电流相同）")
    print(f"  - 正向导通路径（i_L > 0）：L → D1(正接) → SCR1(正接) → D2(正接) → 电流源")
    print(f"  - 反向导通路径（i_L < 0）：电流源 → D4(反接) → SCR1(反接) → D3(反接) → L")
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
    stage1_u0_case1 = 0.0
    try:
        u_c_stage1_final, _ = get_stage1_final_values(stage1_u0_case1, results_dir)
        print(f"从Stage 1 (u0={stage1_u0_case1}V) 继承: u_C(200µs) = {u_c_stage1_final:.6f} V")
        u0_case1 = u_c_stage1_final
    except FileNotFoundError:
        print(f"警告: 未找到Stage 1的结果文件 (u0={stage1_u0_case1}V)，使用默认值 0.0V")
        u0_case1 = 0.0
    
    t1, u_c1, i_l1, results1 = simulate_stage2(u0_case1)
    print_summary(t1, u_c1, i_l1, results1, u0_case1)
    
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
    stage1_u0_case2 = 750.0
    try:
        u_c_stage1_final, _ = get_stage1_final_values(stage1_u0_case2, results_dir)
        print(f"\n从Stage 1 (u0={stage1_u0_case2}V) 继承: u_C(200µs) = {u_c_stage1_final:.6f} V")
        u0_case2 = u_c_stage1_final
    except FileNotFoundError:
        print(f"警告: 未找到Stage 1的结果文件 (u0={stage1_u0_case2}V)，使用默认值 750.0V")
        u0_case2 = 750.0
    
    t2, u_c2, i_l2, results2 = simulate_stage2(u0_case2)
    print_summary(t2, u_c2, i_l2, results2, u0_case2)
    
    csv_path2 = results_dir / f"stage_2_results_u0_{stage1_u0_case2:.0f}V.csv"
    save_results_to_csv(t2, results2, 'Stage 2', stage1_u0_case2, filename=str(csv_path2))
    
    # 绘制结果图表
    img_path2 = results_dir / f"stage2_case2_results.png"
    plot_circuit_results(t2, results2, 'Stage 2', stage1_u0_case2,
                        components=stage2_components,
                        summary_plots=stage2_summary_plots,
                        save_path=str(img_path2))

