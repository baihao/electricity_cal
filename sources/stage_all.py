"""
Stage All: 连续进行三个阶段的仿真

该脚本按顺序执行：
1. Stage 1: 0-200微秒电路仿真
2. Stage 2: 200微秒-50毫秒电路仿真（从Stage 1继承初始条件）
3. Stage 3: 50毫秒-200毫秒电路仿真（从Stage 2继承初始条件）

直接调用三个脚本的 main 函数来执行仿真。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_script(script_name: str) -> None:
    """
    运行指定的脚本
    
    参数：
        script_name: 脚本名称（如 'stage_1.py'）
    """
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        raise FileNotFoundError(f"脚本不存在: {script_path}")
    
    print(f"\n{'='*80}")
    print(f"运行 {script_name}")
    print(f"{'='*80}\n")
    
    # 运行脚本
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(Path(__file__).parent),
        check=False  # 不抛出异常，允许继续执行
    )
    
    if result.returncode != 0:
        print(f"\n警告: {script_name} 执行失败，退出码: {result.returncode}")
    else:
        print(f"\n{script_name} 执行完成")


if __name__ == "__main__":
    print(f"\n{'#'*80}")
    print("开始连续仿真：运行所有三个阶段")
    print(f"{'#'*80}\n")
    
    # 按顺序运行三个脚本
    # 注意：每个脚本都会运行两种情况（0V 和 750V）
    run_script("stage_1.py")
    run_script("stage_2.py")
    run_script("stage_3.py")
    
    print(f"\n{'#'*80}")
    print("所有仿真已完成！")
    print(f"{'#'*80}\n")

