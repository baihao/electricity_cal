"""
L值参数扫描脚本

在5-60µH范围内调整L的值，步长为5µH，对每个L值进行三个stage的仿真，
提取每个stage的u_C, u_L, i_C, i_L的绝对值最大值，并生成图表和CSV文件。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path

# 导入仿真模块
import circuit_params
from stage_1 import simulate_stage1, U0_CASE1
# 注意：stage_2和stage_3需要在每次修改L后重新导入
# 因为它们使用 from circuit_params import L，会在导入时复制L的值
import stage_2
import stage_3

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def run_simulation_with_L(L_value: float) -> dict:
    """
    使用指定的L值运行三个stage的仿真
    
    参数：
        L_value: L的值（H）
    
    返回：
        包含各stage最大值的字典
    """
    # 修改circuit_params中的L值
    original_L = circuit_params.L
    circuit_params.L = L_value
    
    # 重新导入stage_2和stage_3，使它们使用新的L值
    # 因为stage_2和stage_3使用 from circuit_params import L，
    # 会在导入时复制L的值，所以需要重新导入
    import importlib
    importlib.reload(stage_2)
    importlib.reload(stage_3)
    
    print(f"\n{'='*60}")
    print(f"L = {L_value*1e6:.1f} µH")
    print(f"{'='*60}")
    
    results = {}
    
    try:
        # Stage 1 仿真
        print(f"\n运行 Stage 1...")
        t1, u_c1, results1 = simulate_stage1(U0_CASE1)
        
        # Stage 1 提取最大值（Stage 1没有L，所以u_L和i_L为0）
        results['stage1'] = {
            'max_u_C': np.max(np.abs(u_c1)),
            'max_u_L': 0.0,  # Stage 1没有L
            'max_i_C': np.max(np.abs(results1['i_c'])),
            'max_i_L': 0.0,  # Stage 1没有L
        }
        
        # Stage 2 仿真（从Stage 1继承初始条件）
        print(f"\n运行 Stage 2...")
        u_c_stage1_final = u_c1[-1]
        t_stage1_end = t1[-1]
        t2, u_c2, i_l2, results2 = stage_2.simulate_stage2(
            u_c_stage1_final, 
            i_l0=0.0, 
            t_start=t_stage1_end
        )
        
        # Stage 2 提取最大值
        # i_L 恒为正（因为L在整流桥内部）
        results['stage2'] = {
            'max_u_C': np.max(np.abs(u_c2)),
            'max_u_L': np.max(np.abs(results2['u_l'])),
            'max_i_C': np.max(np.abs(results2['i_c'])),
            'max_i_L': np.max(np.abs(i_l2)),
        }
        
        # Stage 3 仿真（从Stage 2继承初始条件）
        print(f"\n运行 Stage 3...")
        u_c_stage2_final = u_c2[-1]
        i_l_stage2_final = i_l2[-1]
        t3, u_c3, i_l3, results3 = stage_3.simulate_stage3(
            u_c_stage2_final,
            i_l0=i_l_stage2_final
        )
        
        # Stage 3 提取最大值
        # i_L 恒为正（因为L在整流桥内部）
        results['stage3'] = {
            'max_u_C': np.max(np.abs(u_c3)),
            'max_u_L': np.max(np.abs(results3['u_l'])),
            'max_i_C': np.max(np.abs(results3['i_c'])),
            'max_i_L': np.max(i_l3),
        }
        
        print(f"\nL = {L_value*1e6:.1f} µH 仿真完成")
        print(f"  Stage 1: max|u_C|={results['stage1']['max_u_C']:.2f}V, max|i_C|={results['stage1']['max_i_C']:.2f}A")
        print(f"  Stage 2: max|u_C|={results['stage2']['max_u_C']:.2f}V, max|u_L|={results['stage2']['max_u_L']:.2f}V, "
              f"max|i_C|={results['stage2']['max_i_C']:.2f}A, max|i_L|={results['stage2']['max_i_L']:.2f}A")
        print(f"  Stage 3: max|u_C|={results['stage3']['max_u_C']:.2f}V, max|u_L|={results['stage3']['max_u_L']:.2f}V, "
              f"max|i_C|={results['stage3']['max_i_C']:.2f}A, max|i_L|={results['stage3']['max_i_L']:.2f}A")
        
    except Exception as e:
        print(f"错误: L = {L_value*1e6:.1f} µH 时仿真失败: {e}")
        # 返回NaN值
        results = {
            'stage1': {'max_u_C': np.nan, 'max_u_L': np.nan, 'max_i_C': np.nan, 'max_i_L': np.nan},
            'stage2': {'max_u_C': np.nan, 'max_u_L': np.nan, 'max_i_C': np.nan, 'max_i_L': np.nan},
            'stage3': {'max_u_C': np.nan, 'max_u_L': np.nan, 'max_i_C': np.nan, 'max_i_L': np.nan},
        }
    finally:
        # 恢复原始L值
        circuit_params.L = original_L
    
    return results


def plot_results(L_values: np.ndarray, all_results: list[dict], output_dir: Path) -> None:
    """
    绘制结果图表
    
    参数：
        L_values: L值数组（µH）
        all_results: 所有L值对应的结果列表
        output_dir: 输出目录
    """
    # 提取数据
    stage1_u_C = [r['stage1']['max_u_C'] for r in all_results]
    stage1_i_C = [r['stage1']['max_i_C'] for r in all_results]
    
    stage2_u_C = [r['stage2']['max_u_C'] for r in all_results]
    stage2_u_L = [r['stage2']['max_u_L'] for r in all_results]
    stage2_i_C = [r['stage2']['max_i_C'] for r in all_results]
    stage2_i_L = [r['stage2']['max_i_L'] for r in all_results]
    
    stage3_u_C = [r['stage3']['max_u_C'] for r in all_results]
    stage3_u_L = [r['stage3']['max_u_L'] for r in all_results]
    stage3_i_C = [r['stage3']['max_i_C'] for r in all_results]
    stage3_i_L = [r['stage3']['max_i_L'] for r in all_results]
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('L值参数扫描结果：各Stage的最大值', fontsize=16, fontweight='bold')
    
    # 子图1：u_C最大值
    ax1 = axes[0, 0]
    ax1.plot(L_values, stage1_u_C, 'o-', label='Stage 1', linewidth=2, markersize=6)
    ax1.plot(L_values, stage2_u_C, 's-', label='Stage 2', linewidth=2, markersize=6)
    ax1.set_xlabel('L (µH)', fontsize=12)
    ax1.set_ylabel('max|u_C| (V)', fontsize=12)
    ax1.set_title('电容电压最大值', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 子图2：u_L最大值
    ax2 = axes[0, 1]
    ax2.plot(L_values, stage2_u_L, 's-', label='Stage 2', linewidth=2, markersize=6, color='orange')
    ax2.set_xlabel('L (µH)', fontsize=12)
    ax2.set_ylabel('max|u_L| (V)', fontsize=12)
    ax2.set_title('电感电压最大值', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 子图3：i_C最大值
    ax3 = axes[1, 0]
    ax3.plot(L_values, stage1_i_C, 'o-', label='Stage 1', linewidth=2, markersize=6)
    ax3.plot(L_values, stage2_i_C, 's-', label='Stage 2', linewidth=2, markersize=6)
    ax3.set_xlabel('L (µH)', fontsize=12)
    ax3.set_ylabel('max|i_C| (A)', fontsize=12)
    ax3.set_title('电容电流最大值', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 子图4：i_L最大值
    ax4 = axes[1, 1]
    ax4.plot(L_values, stage2_i_L, 's-', label='Stage 2', linewidth=2, markersize=6, color='orange')
    ax4.set_xlabel('L (µH)', fontsize=12)
    ax4.set_ylabel('max|i_L| (A)', fontsize=12)
    ax4.set_title('电感电流最大值', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    img_path = output_dir / "L_parameter_scan_results.png"
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存至: {img_path}")
    plt.close()


def save_results_to_csv(L_values: np.ndarray, all_results: list[dict], output_dir: Path) -> None:
    """
    保存结果到CSV文件
    
    参数：
        L_values: L值数组（µH）
        all_results: 所有L值对应的结果列表
        output_dir: 输出目录
    """
    csv_path = output_dir / "L_parameter_scan_results.csv"
    
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        
        # 写入表头
        writer.writerow([
            'L (µH)',
            'Stage 1 max|u_C| (V)', 'Stage 1 max|i_C| (A)',
            'Stage 2 max|u_C| (V)', 'Stage 2 max|u_L| (V)', 'Stage 2 max|i_C| (A)', 'Stage 2 max|i_L| (A)',
            'Stage 3 max|u_C| (V)', 'Stage 3 max|u_L| (V)', 'Stage 3 max|i_C| (A)', 'Stage 3 max|i_L| (A)'
        ])
        
        # 写入数据
        for L_val, result in zip(L_values, all_results):
            writer.writerow([
                f"{L_val:.1f}",
                f"{result['stage1']['max_u_C']:.6f}",
                f"{result['stage1']['max_i_C']:.6f}",
                f"{result['stage2']['max_u_C']:.6f}",
                f"{result['stage2']['max_u_L']:.6f}",
                f"{result['stage2']['max_i_C']:.6f}",
                f"{result['stage2']['max_i_L']:.6f}",
                f"{result['stage3']['max_u_C']:.6f}",
                f"{result['stage3']['max_u_L']:.6f}",
                f"{result['stage3']['max_i_C']:.6f}",
                f"{result['stage3']['max_i_L']:.6f}",
            ])
    
    print(f"CSV文件已保存至: {csv_path}")


if __name__ == "__main__":
    # 创建输出目录
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    print(f"输出目录: {output_dir}\n")
    
    # L值范围：1-100 µH，步长1 µH
    L_min = 1e-6  # 1 µH
    L_max = 100e-6  # 100 µH
    L_step = 1e-6  # 1 µH
    
    L_values_H = np.arange(L_min, L_max + L_step/2, L_step)  # 单位：H
    L_values_uH = L_values_H * 1e6  # 单位：µH
    
    print(f"L值扫描范围: {L_min*1e6:.0f} - {L_max*1e6:.0f} µH，步长: {L_step*1e6:.0f} µH")
    print(f"共 {len(L_values_H)} 个L值需要仿真\n")
    
    # 存储所有结果
    all_results = []
    
    # 对每个L值进行仿真
    for i, L_val in enumerate(L_values_H):
        print(f"\n进度: {i+1}/{len(L_values_H)}")
        result = run_simulation_with_L(L_val)
        all_results.append(result)
    
    # 绘制结果
    print(f"\n{'='*60}")
    print("生成图表...")
    print(f"{'='*60}")
    plot_results(L_values_uH, all_results, output_dir)
    
    # 保存CSV
    print(f"\n{'='*60}")
    print("保存CSV文件...")
    print(f"{'='*60}")
    save_results_to_csv(L_values_uH, all_results, output_dir)
    
    print(f"\n{'='*60}")
    print("L值参数扫描完成！")
    print(f"{'='*60}\n")

