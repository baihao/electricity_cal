"""
序列化工具模块

提供电路仿真结果的CSV保存功能
"""

from __future__ import annotations

import numpy as np
import csv


def save_results_to_csv(t: np.ndarray, results: dict, stage_name: str, 
                        u0: float, filename: str | None = None) -> None:
    """
    将仿真结果保存为CSV文件
    
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
        stage_name: 阶段名称（如 'Stage 1', 'Stage 2'），用于生成默认文件名
        u0: 初始电压（用于生成文件名）
        filename: 输出文件名（可选，默认自动生成）
    """
    if filename is None:
        # 生成默认文件名，将 stage_name 转换为小写并替换空格
        stage_suffix = stage_name.lower().replace(' ', '_')
        filename = f"{stage_suffix}_results_u0_{u0:.0f}V.csv"
    
    # 准备数据
    t_ms = t * 1000  # 转换为毫秒
    
    # CSV表头 - 根据results字典中可用的键动态生成
    base_headers = ['时间 (ms)', '时间 (s)']
    
    # 定义字段顺序和显示名称
    field_mapping = [
        ('i_s', 'i_s (A)'),
        ('i_k3', 'i_K3 (A)'),
        ('i_l', 'i_L (A)'),
        ('i_r', 'i_R (A)'),
        ('i_c', 'i_C (A)'),
        ('i_bypass', 'i_bypass (A)'),
        ('i_d1', 'i_D1 (A)'),
        ('i_d2', 'i_D2 (A)'),
        ('i_d3', 'i_D3 (A)'),
        ('i_d4', 'i_D4 (A)'),
        ('i_scr1', 'i_SCR1 (A)'),
        ('i_rg', 'i_Rg (A)'),
        ('u_c', 'u_C (V)'),
        ('u_l', 'u_L (V)'),
        ('u_bypass', 'u_bypass (V)'),
        ('u_d1', 'u_D1 (V)'),
        ('u_d2', 'u_D2 (V)'),
        ('u_d3', 'u_D3 (V)'),
        ('u_d4', 'u_D4 (V)'),
        ('u_scr1', 'u_SCR1 (V)'),
        ('u_rg', 'u_Rg (V)'),
        ('u_s', 'u_s (V)'),
    ]
    
    # 构建表头和数据字段
    headers = base_headers.copy()
    available_fields = []
    for key, label in field_mapping:
        if key in results:
            headers.append(label)
            available_fields.append(key)
    
    # 写入CSV文件
    with open(filename, 'w', newline='', encoding='utf-8-sig') as f:  # utf-8-sig支持Excel中文显示
        writer = csv.writer(f)
        writer.writerow(headers)
        
        # 写入数据行
        for i in range(len(t)):
            row = [
                f"{t_ms[i]:.6f}",
                f"{t[i]:.12e}"
            ]
            # 添加各字段的值
            for key in available_fields:
                row.append(f"{results[key][i]:.6f}")
            writer.writerow(row)
    
    print(f"结果已保存至CSV文件: {filename}")
    print(f"  共 {len(t)} 行数据\n")

