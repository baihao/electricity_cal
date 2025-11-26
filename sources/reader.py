"""
结果读取工具模块

提供从CSV文件读取仿真结果的功能，特别是用于获取初始条件
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional


def read_csv_final_values(filename: str | Path) -> dict[str, float]:
    """
    读取CSV文件的最终值（最后一行数据）
    
    参数：
        filename: CSV文件路径
    
    返回：
        字典，包含各字段的最终值
    """
    filename = Path(filename)
    
    if not filename.exists():
        raise FileNotFoundError(f"文件不存在: {filename}")
    
    with open(filename, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        raise ValueError(f"CSV文件为空: {filename}")
    
    # 获取最后一行数据
    final_row = rows[-1]
    
    # 转换数值字段
    result = {}
    for key, value in final_row.items():
        try:
            result[key] = float(value)
        except ValueError:
            # 如果无法转换为浮点数，保持原值
            result[key] = value
    
    return result


def get_stage_final_values(stage_name: str, u0: float, 
                           results_dir: Path | None = None) -> dict[str, float]:
    """
    根据阶段名称和初始电压获取CSV文件的最终值
    
    参数：
        stage_name: 阶段名称（如 'Stage 1', 'Stage 2'）
        u0: 初始电压（V）
        results_dir: results文件夹路径（可选，默认从当前文件位置推断）
    
    返回：
        字典，包含各字段的最终值
    """
    if results_dir is None:
        # 从当前文件位置推断results目录
        results_dir = Path(__file__).parent.parent / "results"
    
    # 生成文件名
    stage_suffix = stage_name.lower().replace(' ', '_')
    filename = results_dir / f"{stage_suffix}_results_u0_{u0:.0f}V.csv"
    
    return read_csv_final_values(filename)


def get_stage1_final_values(u0: float, results_dir: Path | None = None) -> tuple[float, float]:
    """
    获取Stage 1的最终值（u_C和i_L）
    
    注意：Stage 1没有i_L，所以i_L返回0
    
    参数：
        u0: 初始电压（V）
        results_dir: results文件夹路径（可选）
    
    返回：
        (u_C_final, i_L_final) 元组
    """
    final_values = get_stage_final_values('Stage 1', u0, results_dir)
    
    u_c_final = final_values.get('u_C (V)', 0.0)
    i_l_final = 0.0  # Stage 1没有电感，所以i_L为0
    
    return u_c_final, i_l_final


def get_stage1_final_time_and_u_c(u0: float, results_dir: Path | None = None) -> tuple[float, float]:
    """
    获取Stage 1的最终时间和u_C值
    
    参数：
        u0: 初始电压（V）
        results_dir: results文件夹路径（可选）
    
    返回：
        (t_final, u_C_final) 元组，t_final单位为秒
    """
    final_values = get_stage_final_values('Stage 1', u0, results_dir)
    
    # 从CSV文件读取时间（秒）
    t_final = final_values.get('时间 (s)', 0.0)
    u_c_final = final_values.get('u_C (V)', 0.0)
    
    return t_final, u_c_final


def get_stage2_final_values(u0: float, results_dir: Path | None = None) -> tuple[float, float]:
    """
    获取Stage 2的最终值（u_C和i_L）
    
    参数：
        u0: 初始电压（V）
        results_dir: results文件夹路径（可选）
    
    返回：
        (u_C_final, i_L_final) 元组
    """
    final_values = get_stage_final_values('Stage 2', u0, results_dir)
    
    u_c_final = final_values.get('u_C (V)', 0.0)
    i_l_final = final_values.get('i_L (A)', 0.0)
    
    return u_c_final, i_l_final


def print_final_values(stage_name: str, u0: float, results_dir: Path | None = None) -> None:
    """
    打印指定阶段的最终值
    
    参数：
        stage_name: 阶段名称
        u0: 初始电压（V）
        results_dir: results文件夹路径（可选）
    """
    try:
        final_values = get_stage_final_values(stage_name, u0, results_dir)
        
        print(f"\n{'='*60}")
        print(f"{stage_name} 最终值（初始电压: {u0}V）")
        print(f"{'='*60}")
        
        # 打印关键值
        if 'u_C (V)' in final_values:
            print(f"  u_C (最终) = {final_values['u_C (V)']:.6f} V")
        if 'i_L (A)' in final_values:
            print(f"  i_L (最终) = {final_values['i_L (A)']:.6f} A")
        
        # 打印其他值
        print(f"\n其他最终值:")
        for key, value in final_values.items():
            if key not in ['时间 (ms)', '时间 (s)', 'u_C (V)', 'i_L (A)']:
                if isinstance(value, float):
                    print(f"  {key} = {value:.6f}")
                else:
                    print(f"  {key} = {value}")
        
        print(f"{'='*60}\n")
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print(f"请确保已运行 {stage_name} 的仿真并生成了结果文件\n")
    except Exception as e:
        print(f"读取文件时出错: {e}\n")


if __name__ == "__main__":
    # 示例：读取Stage 1的最终值
    print("读取Stage 1的最终值:")
    try:
        u_c1, i_l1 = get_stage1_final_values(0.0)
        print(f"  u_C (最终) = {u_c1:.6f} V")
        print(f"  i_L (最终) = {i_l1:.6f} A (Stage 1没有电感)\n")
    except Exception as e:
        print(f"  错误: {e}\n")
    
    try:
        u_c2, i_l2 = get_stage1_final_values(750.0)
        print(f"  u_C (最终) = {u_c2:.6f} V")
        print(f"  i_L (最终) = {i_l2:.6f} A (Stage 1没有电感)\n")
    except Exception as e:
        print(f"  错误: {e}\n")
    
    # 示例：打印Stage 1的完整最终值
    print_final_values('Stage 1', 0.0)
    print_final_values('Stage 1', 750.0)

