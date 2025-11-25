"""
Stage 3: 50毫秒-200毫秒电路仿真

拓扑说明：
- 在50ms时，K3闭合
- K3闭合后，电路拓扑变为：K3 | (L + bypass) | R | C（四个支路并联）
- 在L回路中，晶闸管因为电流过小而关闭之后，将不再开启
- 当i_L < I_SCR_OFF_THRESHOLD时，(L + bypass)支路断开，拓扑变为：K3 | R | C（三个支路并联）
- 大部分电流由K3接管
- 拓扑：电流源 → [K3 | (L + bypass) | R | C] → Rg → 地
- 由于K3电阻很小，各并联支路电压接近0

电路参数：
- C = 36 mF = 0.036 F
- R = 500 Ω
- Rg = 0.5 Ω
- L = 30 µH = 0.00003 H
- R_K3 = 0.000001 Ω（K3闭合时的等效电阻，假设很小）

计算方法：
1. 状态变量选择：
   - u_C：电容电压（被K3短接，会快速衰减到接近0）
   - i_L：电感电流（L回路被短接，会快速衰减到0）

2. KCL方程（在电流源节点）：
   根据电路拓扑 K3 | (L + bypass) | R | C（并联）：
   
   当L回路导通（|i_L| ≥ I_SCR_OFF_THRESHOLD）：
   i_s(t) = i_K3 + i_L + i_R + i_C
   其中：
   - i_K3 = u_C / R_K3：K3回路电流（由于K3与C并联，电压相同）
   - i_L：电感电流（L + bypass支路电流）
   - i_R = u_C / R：电阻电流
   - i_C = C · du_C/dt：电容电流
   
   当L回路断开（|i_L| < I_SCR_OFF_THRESHOLD）：
   i_s(t) = i_K3 + i_R + i_C
   其中：
   - i_K3 = u_C / R_K3：K3回路电流
   - i_R = u_C / R：电阻电流
   - i_C = C · du_C/dt：电容电流
   - i_L：快速衰减到0（L回路断开，不再参与KCL）
   
   电感电流i_L的计算方式：
   * 当晶闸管关闭（|i_L| < I_SCR_OFF_THRESHOLD）：
     (L + bypass)支路断开，i_L通过L和R_K3的串联回路衰减
     di_L/dt = -i_L / τ_L
     其中 τ_L = L / R_K3 为L回路的衰减时间常数
     解为：i_L(t) = i_L(0) · exp(-t/τ_L)，指数衰减到0
   * 当晶闸管还导通（|i_L| ≥ I_SCR_OFF_THRESHOLD）：
     KVL：(L + bypass)支路：u_C = u_L + u_bypass
     其中 u_L = L · di_L/dt，u_bypass = rectifier_bridge_voltage(i_L)
     因此：di_L/dt = (u_C - u_bypass) / L
     由于K3短接，u_C快速衰减到接近0，u_bypass也接近0
     所以i_L主要通过u_C的衰减而衰减

3. KVL方程（并联支路电压关系）：
   由于K3、R、C、(L + bypass)并联，它们两端的电压相同：
   - u_K3 = u_C = u_R = u_L + u_bypass（当L回路导通时）
   - u_K3 = u_C = u_R（当L回路断开时）
   - 由于R_K3很小，u_K3 = R_K3 · i_K3 ≈ 0（因为R_K3很小）
   - 所以u_C ≈ 0，各支路电压都接近0

4. 简化模型：
   由于K3短接了L、R、C支路，可以假设：
   - u_C会快速衰减到接近0（时间常数由R、C、L和K3电阻决定）
   - i_L会快速衰减到0（时间常数由L和K3电阻决定）
   - i_R = u_C / R，会衰减到0
   - i_C = C · du_C/dt，会衰减到0
   - i_K3 = i_s - i_L - i_R - i_C ≈ i_s（大部分电流）

5. 状态方程：
   根据KCL和KVL推导：
   
   当L回路导通（|i_L| ≥ I_SCR_OFF_THRESHOLD）：
   - KCL：i_s = i_K3 + i_L + i_R + i_C
     其中：i_K3 = u_C / R_K3, i_R = u_C / R, i_C = C · du_C/dt
     因此：i_s = u_C/R_K3 + i_L + u_C/R + C·du_C/dt
     整理得到：du_C/dt = (i_s - i_L - u_C/R - u_C/R_K3) / C
   
   - KVL：(L + bypass)支路：u_C = u_L + u_bypass
     其中：u_L = L · di_L/dt, u_bypass = rectifier_bridge_voltage(i_L)
     因此：di_L/dt = (u_C - u_bypass) / L
   
   当L回路断开（|i_L| < I_SCR_OFF_THRESHOLD）：
   - KCL：i_s = i_K3 + i_R + i_C
     其中：i_K3 = u_C / R_K3, i_R = u_C / R, i_C = C · du_C/dt
     因此：i_s = u_C/R_K3 + u_C/R + C·du_C/dt
     整理得到：du_C/dt = (i_s - u_C/R - u_C/R_K3) / C
   
   - i_L衰减：di_L/dt = -i_L / (L / R_K3) = -i_L · R_K3 / L

6. 简化处理：
   由于K3短接，u_C和i_L都会快速衰减，可以采用简化模型：
   - u_C的衰减时间常数：τ_C ≈ R_K3 * C（如果K3电阻很小，衰减很快）
   - i_L的衰减时间常数：τ_L ≈ L / R_K3（如果K3电阻很小，衰减很快）
   
   为了简化，假设K3是理想短接（R_K3 = 0），那么：
   - u_C = 0
   - i_L = 0
   - i_R = 0
   - i_C = 0
   - i_K3 = i_s

   但实际上，K3可能有很小的电阻，所以u_C和i_L会快速衰减但不完全为0。
   我们使用一个很小的R_K3来模拟这个过程。
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from pathlib import Path

from source import neutral_current
from graph import plot_circuit_results
from serialize import save_results_to_csv
from reader import get_stage2_final_values
from circuit_params import (
    C, R, Rg, L, R_K3, I_SCR_OFF_THRESHOLD, U0_CASE1, U0_CASE2,
    U_BYPASS_THRESHOLD, R_BYPASS_TOTAL
)

# ==================== 时间参数 ====================
# 时间范围：50毫秒到200毫秒
T_START = 50e-3  # 50 毫秒 = 0.05 秒
T_END = 200e-3  # 200 毫秒 = 0.2 秒
DT = 10e-6  # 采样间隔：10 微秒 = 0.00001 秒


def rectifier_bridge_voltage(i_l: float, epsilon: float = 1e-6) -> float:
    """
    计算整流桥（D1-D4 + SCR1）的电压，考虑电流方向
    
    参数：
        i_l: 通过整流桥的电流（A）
        epsilon: 平滑过渡参数（默认1e-6），用于tanh函数
    
    返回：
        整流桥两端电压（V），从L端到GND端
    """
    # 使用平滑的符号函数 tanh(i_L/epsilon) 代替 sign(i_L)
    sign_i_smooth = np.tanh(i_l / epsilon)
    
    # 统一表达式：u_bypass = sign(i_L) * U_threshold + R_total * i_L
    u_bypass = sign_i_smooth * U_BYPASS_THRESHOLD + R_BYPASS_TOTAL * i_l
    
    return u_bypass


def stage3_ode(t: float, y: np.ndarray) -> np.ndarray:
    """
    Stage 3状态方程（K3闭合后）
    
    参数：
        t: 时间（秒）
        y: 状态变量数组 [u_C, i_L]
    
    返回：
        [du_C/dt, di_L/dt]
    
    注意：
        - 拓扑：K3 | (L + bypass) | R | C（并联）
        - 当L回路断开时：K3 | R | C（并联）
        - u_C会快速衰减到接近0（因为R_K3很小）
    """
    u_c = y[0]
    i_l = y[1]
    
    # 电流源电流
    i_s = neutral_current(t)
    
    # 检查晶闸管是否关闭（i_L < 阈值）
    scr_off = abs(i_l) < I_SCR_OFF_THRESHOLD
    
    # 各支路电流（由于并联，电压都是u_C）
    i_k3 = u_c / R_K3  # K3回路电流
    i_r = u_c / R  # 电阻电流
    
    if scr_off:
        # L回路断开，拓扑变为：K3 | R | C
        # KCL：i_s = i_K3 + i_R + i_C
        # 其中：i_C = C · du_C/dt
        # 因此：i_s = u_C/R_K3 + u_C/R + C·du_C/dt
        # 整理得到：du_C/dt = (i_s - u_C/R - u_C/R_K3) / C
        du_c_dt = (i_s - i_r - i_k3) / C
        
        # i_L衰减：L回路已断开，i_L应该快速衰减到0
        # 为了避免数值问题，使用一个很小的固定衰减时间常数
        # 当i_L已经很小时，直接设置为0以避免数值不稳定
        if abs(i_l) < 1e-6:
            # i_L已经非常小，直接设置为0，避免数值问题
            di_l_dt = 0.0
        else:
            # 使用一个很小的固定衰减时间常数（例如1微秒），让i_L快速衰减
            # 这样既能快速衰减到0，又不会导致数值不稳定
            tau_l_fast = 1e-6  # 1微秒的衰减时间常数
            di_l_dt = -i_l / tau_l_fast
    else:
        # L回路导通，拓扑：K3 | (L + bypass) | R | C
        # KCL：i_s = i_K3 + i_L + i_R + i_C
        # 其中：i_C = C · du_C/dt
        # 因此：i_s = u_C/R_K3 + i_L + u_C/R + C·du_C/dt
        # 整理得到：du_C/dt = (i_s - i_L - u_C/R - u_C/R_K3) / C
        du_c_dt = (i_s - i_l - i_r - i_k3) / C
        
        # KVL：(L + bypass)支路：u_C = u_L + u_bypass
        # 其中：u_L = L · di_L/dt, u_bypass = rectifier_bridge_voltage(i_L)
        # 因此：di_L/dt = (u_C - u_bypass) / L
        u_bypass = rectifier_bridge_voltage(i_l)
        di_l_dt = (u_c - u_bypass) / L
    
    return np.array([du_c_dt, di_l_dt])


def simulate_stage3(u0: float, i_l0: float = 0.0, t_start: float = T_START, 
                    t_end: float = T_END, dt: float = DT, method: str = 'Radau', 
                    rtol: float = 1e-8, atol: float = 1e-10) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    仿真 Stage 3（50毫秒-200毫秒）的电路响应
    
    参数：
        u0: 初始电容电压（V），从Stage 2的最终值继承
        i_l0: 初始电感电流（A），从Stage 2的最终值继承
        t_start: 仿真开始时间（秒），默认50毫秒
        t_end: 仿真结束时间（秒），默认200毫秒
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
    # 时间采样点（确保不超过t_end）
    t_eval = np.arange(t_start, t_end + dt / 2, dt)
    t_eval = t_eval[t_eval <= t_end]  # 确保不超过t_end
    
    # 求解ODE
    sol = solve_ivp(
        fun=stage3_ode,
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
    
    t = sol.t
    u_c = sol.y[0]
    i_l = sol.y[1]
    
    # 检查结果是否为空
    if len(t) == 0:
        raise RuntimeError(f"求解器返回空结果！时间范围: {t_start} 到 {t_end}")
    
    # 计算各支路电流
    i_s = neutral_current(t)  # 电流源电流
    
    # 由于K3、R、C、(L + bypass)并联，它们两端的电压都是u_C
    i_k3 = u_c / R_K3  # K3回路电流（根据欧姆定律）
    i_r = u_c / R  # 电阻电流
    i_c = np.gradient(u_c, t) * C  # 电容电流（数值微分）
    
    # 验证KCL：i_s = i_K3 + i_L + i_R + i_C（当L回路导通时）
    # 或 i_s = i_K3 + i_R + i_C（当L回路断开时）
    # 由于数值误差，可能不完全相等，但应该非常接近
    
    i_rg = i_s  # Rg电流（由KCL）
    
    # 计算各元件电压
    # 由于K3、R、C、(L + bypass)并联，它们两端的电压都是u_C
    u_k3 = u_c  # K3两端电压（与R、C并联，电压相同）
    u_rg = i_rg * Rg  # Rg两端电压
    u_s = u_c + u_rg  # 电流源两端电压（由KVL：u_s = u_C + u_Rg）
    
    # 由于K3短接（R_K3很小），u_C会快速衰减到接近0
    # 所以u_s ≈ u_Rg
    
    results = {
        'i_s': i_s,
        'i_k3': i_k3,  # K3回路电流
        'i_l': i_l,
        'i_r': i_r,
        'i_c': i_c,
        'i_rg': i_rg,
        'u_c': u_c,
        'u_k3': u_k3,  # K3两端电压（等于u_C）
        'u_rg': u_rg,
        'u_s': u_s,
    }
    
    return t, u_c, i_l, results


def print_summary(t: np.ndarray, u_c: np.ndarray, i_l: np.ndarray, results: dict, u0: float) -> None:
    """
    打印仿真结果摘要
    """
    print(f"\n{'='*60}")
    print(f"Stage 3 仿真结果摘要（初始电压: {u0}V）")
    print(f"{'='*60}")
    print(f"时间范围: {t[0]*1000:.3f} ms 到 {t[-1]*1000:.3f} ms")
    print(f"采样点数: {len(t)}")
    print(f"\n初始值:")
    print(f"  u_C(50ms) = {u_c[0]:.6f} V")
    print(f"  i_L(50ms) = {i_l[0]:.6f} A")
    print(f"  i_s(50ms) = {results['i_s'][0]:.6f} A")
    print(f"\n最终值（t = {t[-1]*1000:.3f} ms）:")
    print(f"  u_C = {u_c[-1]:.6f} V")
    print(f"  i_L = {i_l[-1]:.6f} A")
    print(f"  i_s = {results['i_s'][-1]:.6f} A")
    print(f"  i_K3 = {results['i_k3'][-1]:.6f} A")
    print(f"  i_R = {results['i_r'][-1]:.6f} A")
    print(f"  i_C = {results['i_c'][-1]:.6f} A")
    print(f"  i_Rg = {results['i_rg'][-1]:.6f} A")
    print(f"  u_s = {results['u_s'][-1]:.6f} V")
    print(f"  u_Rg = {results['u_rg'][-1]:.6f} V")
    print(f"\n验证（KCL）:")
    if abs(i_l[-1]) < I_SCR_OFF_THRESHOLD:
        # L回路断开
        print(f"  i_s = i_K3 + i_R + i_C (L回路断开):")
        kcl_sum = results['i_k3'][-1] + results['i_r'][-1] + results['i_c'][-1]
        print(f"    {results['i_s'][-1]:.6f} ≈ {kcl_sum:.6f}")
        print(f"  误差: {abs(results['i_s'][-1] - kcl_sum):.6e} A")
    else:
        # L回路导通
        print(f"  i_s = i_K3 + i_L + i_R + i_C (L回路导通):")
        kcl_sum = results['i_k3'][-1] + i_l[-1] + results['i_r'][-1] + results['i_c'][-1]
        print(f"    {results['i_s'][-1]:.6f} ≈ {kcl_sum:.6f}")
        print(f"  误差: {abs(results['i_s'][-1] - kcl_sum):.6e} A")
    print(f"\n注意：")
    print(f"  - 拓扑：K3 | (L + bypass) | R | C（并联）")
    print(f"  - 当i_L < {I_SCR_OFF_THRESHOLD} A时，L回路断开，拓扑变为：K3 | R | C")
    print(f"  - 由于R_K3很小，u_C快速衰减到接近0，大部分电流由K3接管（i_K3 ≈ i_s）")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # 创建 results 文件夹（如果不存在）
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    print(f"输出目录: {results_dir}\n")
    
    # 从Stage 2继承初始条件
    print("开始仿真 Stage 3：50毫秒-200毫秒")
    print("从Stage 2的最终值继承初始条件\n")
    
    # 情况1：初始电压为 0V（从Stage 2继承）
    stage2_u0_case1 = U0_CASE1
    try:
        u_c_stage2_final, i_l_stage2_final = get_stage2_final_values(stage2_u0_case1, results_dir)
        print(f"从Stage 2 (u0={stage2_u0_case1}V) 继承:")
        print(f"  u_C(50ms) = {u_c_stage2_final:.6f} V")
        print(f"  i_L(50ms) = {i_l_stage2_final:.6f} A")
        u0_case1 = u_c_stage2_final
        i_l0_case1 = i_l_stage2_final
    except FileNotFoundError:
        print(f"警告: 未找到Stage 2的结果文件 (u0={stage2_u0_case1}V)，使用默认值")
        u0_case1 = U0_CASE1
        i_l0_case1 = 0.0
    
    t1, u_c1, i_l1, results1 = simulate_stage3(u0_case1, i_l0_case1)
    print_summary(t1, u_c1, i_l1, results1, u0_case1)
    
    # 定义Stage 3的绘图配置
    stage3_components = [
        ('电流源', 'i_s', 'u_s', '电流源'),
        ('K3回路', 'i_k3', 'u_k3', 'K3'),  # K3电压应该等于u_C（并联）
        ('电阻 R', 'i_r', 'u_c', '电阻'),
        ('电容 C', 'i_c', 'u_c', '电容'),
        ('电感 L', 'i_l', 'u_c', '电感'),
        ('电阻 Rg', 'i_rg', 'u_rg', 'Rg'),
    ]
    
    stage3_summary_plots = [
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
    
    # 保存结果
    csv_path1 = results_dir / f"stage_3_results_u0_{stage2_u0_case1:.0f}V.csv"
    save_results_to_csv(t1, results1, 'Stage 3', stage2_u0_case1, filename=str(csv_path1))
    
    # 绘制结果图表
    img_path1 = results_dir / f"stage3_case1_results.png"
    plot_circuit_results(t1, results1, 'Stage 3', stage2_u0_case1,
                        components=stage3_components,
                        summary_plots=stage3_summary_plots,
                        save_path=str(img_path1))
    
    # 情况2：初始电压为 750V（从Stage 2继承）
    stage2_u0_case2 = U0_CASE2
    try:
        u_c_stage2_final, i_l_stage2_final = get_stage2_final_values(stage2_u0_case2, results_dir)
        print(f"\n从Stage 2 (u0={stage2_u0_case2}V) 继承:")
        print(f"  u_C(50ms) = {u_c_stage2_final:.6f} V")
        print(f"  i_L(50ms) = {i_l_stage2_final:.6f} A")
        u0_case2 = u_c_stage2_final
        i_l0_case2 = i_l_stage2_final
    except FileNotFoundError:
        print(f"警告: 未找到Stage 2的结果文件 (u0={stage2_u0_case2}V)，使用默认值")
        u0_case2 = U0_CASE2
        i_l0_case2 = 0.0
    
    t2, u_c2, i_l2, results2 = simulate_stage3(u0_case2, i_l0_case2)
    print_summary(t2, u_c2, i_l2, results2, u0_case2)
    
    csv_path2 = results_dir / f"stage_3_results_u0_{stage2_u0_case2:.0f}V.csv"
    save_results_to_csv(t2, results2, 'Stage 3', stage2_u0_case2, filename=str(csv_path2))
    
    # 绘制结果图表
    img_path2 = results_dir / f"stage3_case2_results.png"
    plot_circuit_results(t2, results2, 'Stage 3', stage2_u0_case2,
                        components=stage3_components,
                        summary_plots=stage3_summary_plots,
                        save_path=str(img_path2))

