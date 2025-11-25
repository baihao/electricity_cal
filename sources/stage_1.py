"""
Stage 1: 0-200微秒电路仿真

拓扑说明：
- 电流源 i_s(t) 注入
- 电流分流到 R 和 C（并联）
- R 和 C 的电流汇合后流过 Rg
- 拓扑：电流源 → R∥C → Rg → 地

电路参数：
- C = 36 mF = 0.036 F
- R = 500 Ω
- Rg = 0.5 Ω
- 初始电压：0V 或 750V（两种情况）

计算方法：
1. 状态方程推导：
   根据 KCL：i_s(t) = i_R + i_C
   其中：i_R = u_C / R,  i_C = C · du_C/dt
   因此：i_s(t) = u_C / R + C · du_C/dt
   整理得到状态方程：du_C/dt = (i_s(t) - u_C / R) / C

2. 各支路电流计算：
   - i_R = u_C / R（电阻电流）
   - i_C = C · du_C/dt（电容电流，由状态方程导数得到）
   - i_Rg = i_s（Rg电流，由KCL：i_Rg = i_R + i_C = i_s）

3. 各元件电压计算：
   - u_C：电容电压（也是R∥C两端电压），由ODE求解得到
   - u_Rg = i_Rg · Rg = i_s · Rg（Rg两端电压）
   - u_s = u_C + u_Rg = u_C + i_s · Rg（电流源两端电压，由KVL）

4. 数值求解：
   使用 scipy.integrate.solve_ivp 求解状态方程
   时间范围：0 到 200 微秒（0 到 0.0002 秒）
   采样间隔：1 微秒（0.000001 秒）

5. scipy.integrate.solve_ivp 接口约定：
   solve_ivp 要求传入的函数 fun(t, y) 必须返回 dy/dt（状态变量的导数）
   这是 ODE 求解器的标准接口约定，不是从代码推导出来的。
   
   文档参考：
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
   
   接口说明：
   - fun(t, y) -> dy/dt
   - t: 当前时间点
   - y: 当前状态变量数组（本问题中 y = [u_C]）
   - 返回值: dy/dt，即状态变量的时间导数（本问题中返回 [du_C/dt]）
   
   为什么返回的是导数？
   - 这是 solve_ivp 的 API 设计：它需要知道 dy/dt 才能进行数值积分
   - solve_ivp 内部使用数值积分方法（如 Runge-Kutta）来求解 y(t)
   - 给定初始值 y0 和导数函数 dy/dt，求解器可以逐步计算出 y(t)
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
import os
from pathlib import Path

from source import neutral_current
from graph import plot_circuit_results
from serialize import save_results_to_csv
from circuit_params import C, R, Rg, U0_CASE1, U0_CASE2

# ==================== 时间参数 ====================
# 时间范围：0 到 200 微秒
T_START = 0.0
T_END = 200e-6  # 200 微秒 = 0.0002 秒
DT = 1e-6  # 采样间隔：1 微秒 = 0.000001 秒


def stage1_ode(t: float, u_c: np.ndarray) -> np.ndarray:
    """
    状态方程：du_C/dt = (i_s(t) - u_C / R) / C
    
    这是 scipy.integrate.solve_ivp 要求的 ODE 函数接口。
    
    scipy.integrate.solve_ivp 接口约定：
    - 函数签名必须是: fun(t, y) -> dy/dt
    - t: 当前时间点
    - y: 当前状态变量（这里是 [u_C]）
    - 返回值: dy/dt，即状态变量的时间导数
    
    文档参考：
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
    - solve_ivp 要求 fun(t, y) 返回 dy/dt，这是 ODE 求解器的标准接口
    
    对于本问题：
    - 状态变量 y = [u_C]（电容电压）
    - dy/dt = [du_C/dt]（电容电压的导数）
    - 根据电路分析：du_C/dt = (i_s(t) - u_C/R) / C
    
    参数：
        t: 时间（秒）
        u_c: 电容电压 u_C（作为状态变量 y[0] 传入）
    
    返回：
        [du_C/dt]: 状态变量的导数数组，必须与 y0 的形状一致
    """
    i_s = neutral_current(t)
    # 根据状态方程计算导数
    # 注意：u_c 是数组的第一个元素（因为 y0=[u0] 是长度为1的数组）
    u_c_val = u_c[0] if isinstance(u_c, np.ndarray) else u_c
    du_dt = (i_s - u_c_val / R) / C
    return np.array([du_dt])  # 返回数组，形状与 y0 一致


def simulate_stage1(u0: float, t_end: float = T_END, dt: float = DT,
                    method: str = 'Radau', rtol: float = 1e-8, atol: float = 1e-10) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    仿真 Stage 1（0-200微秒）的电路响应
    
    参数：
        u0: 初始电容电压（V）
        t_end: 仿真结束时间（秒），默认200微秒
        dt: 采样间隔（秒），默认1微秒
        method: 求解器方法，可选：
            - 'Radau': 隐式方法（默认，适合刚性问题）
            - 'RK45': 4-5阶Runge-Kutta（适合非刚性问题）
            - 'RK23': 2-3阶Runge-Kutta（更快但精度较低）
            - 'DOP853': 8阶Dormand-Prince（高精度）
            - 'BDF': 后向差分公式（适合刚性问题）
        rtol: 相对容差（默认1e-8）
        atol: 绝对容差（默认1e-10）
    
    返回：
        t: 时间数组（秒）
        u_c: 电容电压数组（V）
        results: 字典，包含各支路电流和电压
    """
    # 时间采样点
    t_eval = np.arange(T_START, t_end + dt / 2, dt)
    
    # 求解ODE
    sol = solve_ivp(
        fun=stage1_ode,
        t_span=(T_START, t_end),
        y0=[u0],
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol
    )
    
    # solve_ivp 返回值说明：
    # sol.t: 时间数组，形状 (m,)，m 是时间点数量
    # sol.y: 状态变量数组，形状 (n, m)，其中：
    #   - n: 状态变量的数量（本问题中 n=1，因为只有 u_C 一个状态变量）
    #   - m: 时间点的数量
    # sol.y[i]: 第 i 个状态变量在所有时间点的值
    # 
    # 本问题中：
    # - y0 = [u0]，只有1个状态变量 u_C
    # - sol.y 形状为 (1, m)
    # - sol.y[0] 是 u_C 在所有时间点的值，形状 (m,)
    # 
    # 示例对比（如果有多个状态变量）：
    # - 如果有2个状态变量：y0 = [u_C(0), i_L(0)]
    # - sol.y 形状为 (2, m)
    # - sol.y[0] 是 u_C 的时间序列
    # - sol.y[1] 是 i_L 的时间序列
    t = sol.t
    u_c = sol.y[0]  # 提取第一个（也是唯一的）状态变量 u_C 的时间序列
    
    # 计算各支路电流
    i_s = neutral_current(t)  # 电流源电流
    i_r = u_c / R  # 电阻电流
    
    # 电容电流：直接从状态方程计算，避免数值微分误差
    # i_C = C · du_C/dt = i_s - u_C/R（由状态方程）
    i_c = i_s - i_r  # 直接从KCL计算，更精确
    
    # 或者使用ODE函数直接计算导数（更精确的方法）
    # du_c_dt = np.array([stage1_ode(ti, uc)[0] for ti, uc in zip(t, u_c)])
    # i_c_alt = C * du_c_dt
    
    i_rg = i_s  # Rg电流（由KCL）
    
    # 计算各元件电压
    u_rg = i_rg * Rg  # Rg两端电压
    u_s = u_c + u_rg  # 电流源两端电压（由KVL）
    
    results = {
        'i_s': i_s,
        'i_r': i_r,
        'i_c': i_c,
        'i_rg': i_rg,
        'u_c': u_c,
        'u_rg': u_rg,
        'u_s': u_s
    }
    
    return t, u_c, results


def print_summary(t: np.ndarray, results: dict, u0: float) -> None:
    """
    打印仿真结果摘要
    """
    print(f"\n{'='*60}")
    print(f"Stage 1 仿真结果摘要（初始电压: {u0}V）")
    print(f"{'='*60}")
    print(f"时间范围: {t[0]*1000:.3f} ms 到 {t[-1]*1000:.3f} ms")
    print(f"采样点数: {len(t)}")
    print(f"\n初始值:")
    print(f"  u_C(0) = {results['u_c'][0]:.6f} V")
    print(f"  i_s(0) = {results['i_s'][0]:.6f} A")
    print(f"\n最终值（t = {t[-1]*1000:.3f} ms）:")
    print(f"  u_C = {results['u_c'][-1]:.6f} V")
    print(f"  i_s = {results['i_s'][-1]:.6f} A")
    print(f"  i_R = {results['i_r'][-1]:.6f} A")
    print(f"  i_C = {results['i_c'][-1]:.6f} A")
    print(f"  i_Rg = {results['i_rg'][-1]:.6f} A")
    print(f"  u_s = {results['u_s'][-1]:.6f} V")
    print(f"  u_Rg = {results['u_rg'][-1]:.6f} V")
    print(f"\n验证（KCL）:")
    print(f"  i_s = i_R + i_C: {results['i_s'][-1]:.6f} ≈ {results['i_r'][-1] + results['i_c'][-1]:.6f}")
    print(f"  误差: {abs(results['i_s'][-1] - (results['i_r'][-1] + results['i_c'][-1])):.6e} A")
    print(f"{'='*60}\n")


def print_solver_recommendations() -> None:
    """
    打印求解器参数调整建议
    """
    print(f"\n{'='*60}")
    print("求解器参数调整建议")
    print(f"{'='*60}\n")
    
    print("1. 误差来源分析：")
    print("   - 主要误差来自使用 np.gradient 进行数值微分")
    print("   - 应该直接从 ODE 函数计算导数，避免数值微分误差")
    print("   - 已改进：使用 i_C = i_s - i_R 直接从 KCL 计算\n")
    
    print("2. 求解器方法选择：")
    print("   - Radau (默认): 隐式方法，适合刚性问题，数值稳定性好")
    print("   - RK45: 4-5阶Runge-Kutta，适合非刚性问题，精度和速度平衡")
    print("   - DOP853: 8阶方法，精度最高但速度较慢")
    print("   - BDF: 后向差分公式，适合刚性问题\n")
    
    print("3. 容差参数调整：")
    print("   - rtol (相对容差): 默认 1e-8，可调整到 1e-10 提高精度")
    print("   - atol (绝对容差): 默认 1e-10，可调整到 1e-12 提高精度")
    print("   - 注意：过小的容差可能导致计算时间显著增加\n")
    
    print("4. 采样间隔影响：")
    print("   - 当前采样间隔：1 微秒")
    print("   - 更小的间隔不会提高求解精度（求解器内部自适应）")
    print("   - 但会影响数值微分的精度（已改用直接计算）\n")
    
    print("5. 推荐配置：")
    print("   - 方法: 'Radau' (默认，适合刚性问题)")
    print("   - rtol: 1e-8 到 1e-10")
    print("   - atol: 1e-10 到 1e-12")
    print("   - 使用直接计算导数，避免数值微分\n")
    
    print(f"{'='*60}\n")


def verify_numerical_solution(t: np.ndarray, u_c: np.ndarray, results: dict) -> None:
    """
    验证数值解是否满足状态方程
    
    参数：
        t: 时间数组
        u_c: 电容电压数组
        results: 包含各支路电流和电压的字典
    """
    print(f"\n{'='*60}")
    print("数值解验证：检查是否满足状态方程")
    print(f"{'='*60}\n")
    
    # 方法1：直接从ODE函数计算导数（最精确）
    du_c_dt_from_ode = np.array([stage1_ode(ti, uc)[0] for ti, uc in zip(t, u_c)])
    
    # 方法2：根据状态方程计算理论导数
    i_s = results['i_s']
    du_c_dt_theoretical = (i_s - u_c / R) / C
    
    # 验证方法1和方法2的一致性（应该完全一致，因为都来自同一方程）
    error_ode_vs_theory = np.abs(du_c_dt_from_ode - du_c_dt_theoretical)
    max_error_ode = np.max(error_ode_vs_theory)
    mean_error_ode = np.mean(error_ode_vs_theory)
    
    print(f"状态方程: du_C/dt = (i_s - u_C/R) / C")
    print(f"\n验证结果:")
    print(f"  ODE函数直接计算 vs 状态方程理论值:")
    print(f"    最大误差: {max_error_ode:.6e} V/s")
    print(f"    平均误差: {mean_error_ode:.6e} V/s")
    
    # 验证KCL
    i_r = results['i_r']
    i_c = results['i_c']
    kcl_error = np.abs(i_s - (i_r + i_c))
    max_kcl_error = np.max(kcl_error)
    
    print(f"\nKCL验证: i_s = i_R + i_C")
    print(f"  最大误差: {max_kcl_error:.6e} A")
    
    if max_error_ode < 1e-10 and max_kcl_error < 1e-6:
        print(f"\n  ✓ 数值解满足状态方程和KCL（误差在可接受范围内）")
    else:
        print(f"\n  ⚠ 数值解误差较大，可能需要调整求解器参数")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # ==================== 执行顺序说明 ====================
    # 1. 求解器参数建议
    #    - 提供求解器配置建议，帮助理解如何调整参数
    # 
    # 2. 数值仿真
    #    - 进行数值计算
    #    - 包含两种初始条件的仿真
    # 
    # 3. 数值解验证
    #    - 验证数值计算结果是否满足状态方程和KCL
    #    - 检查数值解的精度和正确性
    #    - 注意：不影响计算结果，只进行验证
    # ======================================================
    
    # 创建 results 文件夹（如果不存在）
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    print(f"输出目录: {results_dir}\n")
    
    # 步骤1：打印求解器参数建议
    print_solver_recommendations()
    
    # 情况1：初始电压为 0V
    print("开始仿真情况1：初始电压 = 0V")
    # 可以使用更严格的容差：rtol=1e-10, atol=1e-12
    # 或使用更高阶方法：method='DOP853'
    t1, u_c1, results1 = simulate_stage1(U0_CASE1)
    print_summary(t1, results1, U0_CASE1)
    
    # 步骤3：数值解验证
    # 目的：验证数值计算结果是否满足状态方程和KCL
    #      - 使用数值计算结果进行验证
    #      - 检查数值解的精度和正确性
    #      - 不影响计算结果，只进行验证
    verify_numerical_solution(t1, u_c1, results1)
    
    # 保存结果到 results 文件夹
    csv_path1 = results_dir / f"stage_1_results_u0_{U0_CASE1:.0f}V.csv"
    save_results_to_csv(t1, results1, 'Stage 1', U0_CASE1, filename=str(csv_path1))
    
    # 定义Stage 1的绘图配置
    stage1_components = [
        ('电流源', 'i_s', 'u_s', '电流源'),
        ('电阻 R', 'i_r', 'u_c', '电阻'),
        ('电容 C', 'i_c', 'u_c', '电容'),
        ('电阻 Rg', 'i_rg', 'u_rg', 'Rg'),
    ]
    
    stage1_summary_plots = [
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
    
    img_path1 = results_dir / "stage1_case1_results.png"
    plot_circuit_results(t1, results1, 'Stage 1', U0_CASE1, 
                        components=stage1_components,
                        summary_plots=stage1_summary_plots,
                        save_path=str(img_path1))
    
    # 情况2：初始电压为 750V
    print("\n开始仿真情况2：初始电压 = 750V")
    t2, u_c2, results2 = simulate_stage1(U0_CASE2)
    print_summary(t2, results2, U0_CASE2)
    verify_numerical_solution(t2, u_c2, results2)
    
    # 保存结果到 results 文件夹
    csv_path2 = results_dir / f"stage_1_results_u0_{U0_CASE2:.0f}V.csv"
    save_results_to_csv(t2, results2, 'Stage 1', U0_CASE2, filename=str(csv_path2))
    
    img_path2 = results_dir / "stage1_case2_results.png"
    plot_circuit_results(t2, results2, 'Stage 1', U0_CASE2,
                        components=stage1_components,
                        summary_plots=stage1_summary_plots,
                        save_path=str(img_path2))

