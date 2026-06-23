# MCU 协议套件实现说明

基于 MCU 论文（Yang et al., *MCU: Exact and Constant-Round Nonlinear Function
Evaluation in MPC without Preprocessing*）与项目技术方案第二章实现的安全多方计算
协议套件。三方角色：`P0`（用户方）、`P1`（服务商）、`HP`（非合谋辅助方）。

## 已实现协议

| 文件 | 协议 | 论文对应 | 轮数 | 验证指标 | 实测 |
| --- | --- | --- | --- | --- | --- |
| `multiply.py` | Π_mul 安全乘法 | Protocol 1 | 2 | 与明文一致 | 精确 |
| `wrap_detect.py` | Wrap 溢出检测 | Protocol 4 | ~3 | w 完全正确 | 8/8 |
| `exponential.py` | Π_exp 安全指数 | Protocol 5 | 4 | 平均误差 < 1e-4 | ~1.7e-11 |
| `softmax.py` | Π_softmax | Protocol 8 | 6 | 最大误差 < 1e-4 | ~1.3e-14 |
| `gelu.py` (Sigmoid) | Π_sigmoid | Protocol 7 | 6 | 最大误差 < 1e-4 | ~7.9e-15 |
| `gelu.py` (GeLU) | Π_gelu | §6.2 | 8 | 平均误差 < 1e-3 | ~2.1e-12 |

## 运行测试

```bash
cd mcu_core
pip install pycryptodome -i https://pypi.tuna.tsinghua.edu.cn/simple

# Windows 控制台需设 UTF-8，否则中文/符号显示异常
set PYTHONIOENCODING=utf-8        # PowerShell: $env:PYTHONIOENCODING="utf-8"

# 一键运行全部协议验证
python -m mcu_core.protocols.run_all_tests

# 或单独运行某个协议
python -m mcu_core.protocols.wrap_detect
python -m mcu_core.protocols.exponential
python -m mcu_core.protocols.softmax
python -m mcu_core.protocols.gelu
```

## 核心设计

### MCU 三步范式（Mask-Compute-Unmask）
1. **Mask**：各方用 PRG0 同步生成的掩码对输入加性掩码，发给 HP。
2. **Compute**：HP 在聚合后的掩码数据上**精确**计算目标函数，加性分享结果。
3. **Unmask**：各方用本地已知的掩码恢复最终份额。

无需任何离线预处理（Beaver 三元组），随机性全部由同步 PRG 在线生成。

### 代数函数 vs 超越函数
- **代数函数**（乘法）：在整数环 `Z_{2^64}` 上运算，结果**精确**。
- **超越函数**（exp/sigmoid/softmax/gelu）：按论文在**浮点域**（IEEE 754）计算，
  只有浮点舍入误差，无多项式近似误差。

### 关键恒等式
- 指数：`e^x = e^((x+r) mod M) · e^(w·M − r)`，`w = wrap(x, r, M)`
- Softmax：`softmax(x_m) = [e^{x_m} · e^t] / [e^t · Σ_j e^{x_j}]`（分母用 `e^t` 隐藏，避免除法）
- Sigmoid：`σ(x) = [e^x · e^t] / [e^t · (1 + e^x)]`
- GeLU：`GeLU(x) = x · σ(1.702x)`（先 sigmoid 再实数域安全乘法）

### 数值参数
超越函数采用适中的实数模 `MOD = 256`（见 `exponential.py`）：
- 保证 `e^R`、修正因子在 float64 范围内不溢出（x∈[−20,20] 时 `e^256 ≈ 1.5e111`）；
- HP 把结果拆成两个**正**份额（`s0 = u·E`、`s1 = (1−u)·E`），去掩码时乘以正因子，
  两份额同号同量级，避免灾难性相消，精度仅受浮点舍入限制。

## 安全性与简化说明

- 所有协议遵循 MCU 的半诚实 + HP 非合谋安全模型，HP 仅见加性/乘性掩码后的数据。
- `wrap_detect.py` 为**简化但正确**的实例化：在 `|x| < M` 假设下（ML 场景成立），
  用两次"秘密值 vs 公开阈值"的符号比较得到 `w ∈ {−1,0,1}`，结构忠实于论文
  （论文 Wrap 依赖 Bicoptor 2.0 的 Sign 协议）。生产环境可替换为完整 Sign 协议。
- 浮点域的加性分享存在与论文一致的统计隐藏特性；整数环上的乘法分享则为完美隐藏。

## 测试方法

每个协议的 `verify_xxx()` 使用 `mock_comm.make_mock_comm()` + 多线程模拟三方并发，
通过阻塞队列保证多轮协议的消息顺序，无需启动真实进程即可验证正确性与精度。
