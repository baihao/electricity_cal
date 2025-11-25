# 整流桥电路建模说明

## 1. 整流桥电路拓扑

完整的整流桥电路包含：
- **4个二极管**：D1, D2, D3, D4（参数相同）
- **1个可控硅**：SCR1
- **1个电感**：L（与整流桥串联）

## 2. 整流桥的工作原理

整流桥电路能够实现**全波整流**，无论输入电流方向如何，都能保证电流单向通过负载。

### 2.1 当支路电流为正（i_L > 0）时

**电流路径**：L → D1 → SCR1 → D2

- D1 正向导通（正向电流通过）
- SCR1 正向导通
- D2 正向导通
- D3 和 D4 反向截止（不导通）

**电压降**：
```
u_bypass = U(D1) + U(SCR1) + U(D2)
         = (0.75 + 0.00007*i_L) + (0.88 + 0.000052*i_L) + (0.75 + 0.00007*i_L)
         = 2.38 + 0.000192*i_L
```

### 2.2 当支路电流为负（i_L < 0）时

**电流路径**：L → D3（反接）→ SCR1（反接）→ D4（反接）

- D3 正向导通（相对于负电流方向）
- SCR1 正向导通（相对于负电流方向）
- D4 正向导通（相对于负电流方向）
- D1 和 D2 反向截止（不导通）

**电压降**：
由于电流为负，但二极管和SCR的压降方向与电流方向一致：
```
u_bypass = U(D3) + U(SCR1) + U(D4)
         = (0.75 + 0.00007*|i_L|) + (0.88 + 0.000052*|i_L|) + (0.75 + 0.00007*|i_L|)
         = 2.38 + 0.000192*|i_L|
```

由于 i_L < 0，所以 |i_L| = -i_L，因此：
```
u_bypass = 2.38 + 0.000192*(-i_L)
         = 2.38 - 0.000192*i_L
```

## 3. 统一的伏安特性表达式

整流桥的伏安特性可以统一表示为：

```
u_bypass = sign(i_L) * (U_threshold + R_total * |i_L|)
```

其中：
- `sign(i_L)`：电流的符号函数（+1 当 i_L > 0，-1 当 i_L < 0）
- `U_threshold = 2.38 V`：总阈值电压
- `R_total = 0.000192 Ω`：总等效电阻

**简化形式**：
```
u_bypass = 2.38 * sign(i_L) + 0.000192 * i_L
```

当 i_L > 0：`u_bypass = 2.38 + 0.000192*i_L`
当 i_L < 0：`u_bypass = -2.38 + 0.000192*i_L`（因为 sign(i_L) = -1）

**更精确的形式**（考虑二极管的非线性）：
```
u_bypass = sign(i_L) * U_threshold + R_total * i_L
```

这可以简化为：
```
u_bypass = sign(i_L) * 2.38 + 0.000192 * i_L
```

## 4. 在状态方程中的体现

### 4.1 KVL方程

```
u_C = u_L + u_bypass
```

其中：
- `u_L = L · di_L/dt`：电感电压
- `u_bypass`：整流桥电压（根据电流方向计算）

### 4.2 电感电流方程

```
u_L = u_C - u_bypass
L · di_L/dt = u_C - u_bypass
di_L/dt = (u_C - u_bypass) / L
```

其中 `u_bypass` 根据电流方向计算：
- 当 i_L > 0：`u_bypass = 2.38 + 0.000192*i_L`
- 当 i_L < 0：`u_bypass = -2.38 + 0.000192*i_L`
- 当 i_L = 0：`u_bypass = 0`（理想情况）

## 5. 代码实现

### 5.1 整流桥电压函数

```python
def rectifier_bridge_voltage(i_l: float) -> float:
    """
    计算整流桥（D1-D4 + SCR1）的电压，考虑电流方向
    
    参数：
        i_l: 通过整流桥的电流（A）
    
    返回：
        整流桥两端电压（V）
    """
    if abs(i_l) < 1e-10:  # 电流接近0
        return 0.0
    
    # 使用统一的表达式：u_bypass = sign(i_L) * U_threshold + R_total * i_L
    sign_i = 1.0 if i_l > 0 else -1.0
    u_bypass = sign_i * U_BYPASS_THRESHOLD + R_BYPASS_TOTAL * i_l
    
    return u_bypass
```

### 5.2 状态方程中的使用

```python
def stage2_ode(t: float, y: np.ndarray) -> np.ndarray:
    u_c = y[0]
    i_l = y[1]
    
    # 计算整流桥电压（考虑电流方向）
    u_bypass = rectifier_bridge_voltage(i_l)
    
    # 电感电流变化率
    di_l_dt = (u_c - u_bypass) / L
    
    return np.array([du_c_dt, di_l_dt])
```

## 6. 物理意义

### 6.1 整流桥的作用

整流桥电路确保：
- **无论电流方向如何，都能保证电流单向通过**
- **电压降的方向与电流方向一致**
- **当电流为正时，使用D1-D2-SCR1路径**
- **当电流为负时，使用D3-D4-SCR1路径**

### 6.2 电压降的方向性

- **i_L > 0**：电压降为正，u_bypass > 0
- **i_L < 0**：电压降为负，u_bypass < 0（但绝对值相同）
- **i_L = 0**：电压降为0

这确保了整流桥的电压降总是**阻碍电流变化**，符合物理规律。

## 7. 数值稳定性

### 7.1 平滑过渡

在 i_L = 0 附近，可以使用平滑过渡函数避免不连续性：

```python
def rectifier_bridge_voltage_smooth(i_l: float, epsilon: float = 1e-6) -> float:
    """
    使用平滑过渡的整流桥电压计算
    """
    # 使用 tanh 函数实现平滑过渡
    sign_i = np.tanh(i_l / epsilon)
    u_bypass = sign_i * U_BYPASS_THRESHOLD + R_BYPASS_TOTAL * i_l
    return u_bypass
```

### 7.2 事件检测

可以使用 `solve_ivp` 的事件检测功能，在 i_L = 0 处切换方程，但这会增加计算复杂度。

## 8. 总结

整流桥电路的建模要点：

1. **双向导通**：无论电流方向如何，整流桥都能导通
2. **路径切换**：根据电流方向自动切换导通路径（D1-D2 或 D3-D4）
3. **电压方向**：电压降的方向与电流方向一致
4. **统一表达式**：`u_bypass = sign(i_L) * U_threshold + R_total * i_L`

这种建模方式准确体现了整流桥的全波整流特性，使得仿真结果更符合实际电路的物理行为。

