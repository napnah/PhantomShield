"""
Π_gelu：MCU 安全 GeLU 协议（对应 MCU 论文 6.2 节 GeLU 支持 / 技术方案 2.6 节）

GeLU 采用标准 sigmoid 近似（Hendrycks & Gimpel）：
    GeLU(x) = x · sigmoid(1.702 · x),    sigmoid(z) = e^z / (1 + e^z)

本文件实现两个协议：
  1. Π_sigmoid（对应论文 Protocol 7）：复用 Π_exp 计算 e^z，再用"分母公开 + e^t 隐藏"
     的技巧避免除法，得到 [[sigmoid(z)]]。
  2. Π_gelu：先本地缩放 1.702·[x]_i，调用 Π_sigmoid，再用一次实数域安全乘法
     把 [[x]] 与 [[sigmoid(1.702x)]] 相乘。

Sigmoid 流程
------------
1. 调用 Π_exp 得到 [e^z]_i。
2. 分母份额 d_i = [e^z]_i + (i==0 ? 1 : 0)，使 Σ d_i = 1 + e^z。
3. 各方用同步随机 t，发送 e^t·d_i 给 HP；HP 聚合 D_pub = e^t(1+e^z) 广播。
4. 本地：[sigmoid(z)]_i = [e^z]_i · e^t / D_pub。
   Σ_i = e^z·e^t / (e^t(1+e^z)) = e^z/(1+e^z) ✓

实数域安全乘法（容斥恒等式，与整数 Π_mul 同构，但在实数上运算）
--------------------------------------------------------------
    a·b = (a+r_a)(b+r_b) - a·r_b - b·r_a - r_a·r_b
仅 P0 叠加掩码并修正三项，P1 修正两项；HP 在掩码值上求积并加性分享。
此处 a=x、b=sigmoid(1.702x) 取值均有界，浮点运算精确（仅舍入误差）。

通信轮次：Sigmoid（指数 4 轮 + 掩码求和 1 轮 ≈ 6 轮）+ 乘法（2 轮）≈ 8 轮，与论文一致。
"""
import math

from mcu_core.protocols.exponential import ExpParty, ExpHP, MOD, _run_three_party

# GeLU 的 sigmoid 近似系数
GELU_COEF = 1.702


# --------------------------------------------------------------------------- #
# Π_sigmoid
# --------------------------------------------------------------------------- #
class SigmoidParty:
    """P0 / P1 执行的 Sigmoid 协议逻辑。"""

    def __init__(self, party_id: int, prg0, comm):
        self.id = party_id
        self.prg0 = prg0
        self.comm = comm
        self._exp = ExpParty(party_id, prg0, comm)

    def sigmoid(self, share_z: float) -> float:
        """输入本方 [z]_i，输出本方 [sigmoid(z)]_i。"""
        # 1. 指数
        e_i = self._exp.exp(share_z)

        # 2. 分母份额 (1 + e^z)，常数 1 仅由 P0 承担
        d_i = e_i + (1.0 if self.id == 0 else 0.0)

        # 3. 掩码求和
        t = self.prg0.next_real(MOD)
        et = math.exp(t)
        self.comm.send_to_hp({'u': et * d_i})
        d_pub = self.comm.recv_from_hp()['D']

        # 4. 本地相除
        return e_i * et / d_pub


class SigmoidHP:
    """HP 执行的 Sigmoid 协议逻辑。"""

    def __init__(self, asprg_p0, comm):
        self.asprg_p0 = asprg_p0
        self.comm = comm
        self._exp = ExpHP(asprg_p0, comm)

    def serve_sigmoid(self):
        # 1. 协助指数
        self._exp.serve_exp()
        # 2. 聚合掩码后的分母并广播
        u0 = self.comm.recv_from_p0()['u']
        u1 = self.comm.recv_from_p1()['u']
        d_pub = u0 + u1                    # = e^t (1 + e^z)
        self.comm.send_to_p0({'D': d_pub})
        self.comm.send_to_p1({'D': d_pub})


# --------------------------------------------------------------------------- #
# Π_gelu
# --------------------------------------------------------------------------- #
class GeluParty:
    """P0 / P1 执行的 GeLU 协议逻辑。"""

    def __init__(self, party_id: int, prg0, comm):
        self.id = party_id
        self.prg0 = prg0
        self.comm = comm
        self._sigmoid = SigmoidParty(party_id, prg0, comm)

    def gelu(self, share_x: float) -> float:
        """输入本方 [x]_i，输出本方 [GeLU(x)]_i。"""
        # 1. 本地缩放 1.702·x（无需通信）
        scaled = GELU_COEF * share_x

        # 2. 调用 Sigmoid
        sig_i = self._sigmoid.sigmoid(scaled)

        # 3. 实数域安全乘法 x · sigmoid(1.702x)
        r_a = self.prg0.next_real(MOD)
        r_b = self.prg0.next_real(MOD)
        if self.id == 0:
            self.comm.send_to_hp({'ma': share_x + r_a, 'mb': sig_i + r_b})
        else:
            self.comm.send_to_hp({'ma': share_x, 'mb': sig_i})

        s_i = self.comm.recv_from_hp()['s']

        if self.id == 0:
            correction = share_x * r_b + sig_i * r_a + r_a * r_b
        else:
            correction = share_x * r_b + sig_i * r_a
        return s_i - correction


class GeluHP:
    """HP 执行的 GeLU 协议逻辑。"""

    def __init__(self, asprg_p0, comm):
        self.asprg_p0 = asprg_p0
        self.comm = comm
        self._sigmoid = SigmoidHP(asprg_p0, comm)

    def serve_gelu(self):
        # 1. 协助 Sigmoid
        self._sigmoid.serve_sigmoid()
        # 2. 实数乘法：聚合掩码值求积并加性分享
        m0 = self.comm.recv_from_p0()
        m1 = self.comm.recv_from_p1()
        ma = m0['ma'] + m1['ma']          # = x + r_a
        mb = m0['mb'] + m1['mb']          # = sigmoid + r_b
        product = ma * mb
        # 拆成同号同量级份额，避免相消（u∈[0,1)）
        u = self.asprg_p0.next_unit()
        s0 = u * product
        s1 = product - s0
        self.comm.send_to_p0({'s': s0})
        self.comm.send_to_p1({'s': s1})


# --------------------------------------------------------------------------- #
# 验证
# --------------------------------------------------------------------------- #
def _plaintext_sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def _plaintext_gelu(x):
    return x * _plaintext_sigmoid(GELU_COEF * x)


def verify_sigmoid(num_tests: int = 500, verbose_samples: int = 5):
    """[-8, 8] 采样验证 sigmoid 精度。"""
    import random
    from mcu_core.prg_sync import PRGSync
    from mcu_core.mock_comm import make_mock_comm

    shared_seed = bytes(range(16))
    seed_hp_p0 = bytes(range(16, 32))

    print('=== Π_sigmoid 安全 Sigmoid 协议验证 ===')
    print(f'测试 {num_tests} 个样本，z ~ U[-8, 8]\n')

    max_abs_err = 0.0
    sum_abs_err = 0.0
    shown = 0
    for _ in range(num_tests):
        z = random.uniform(-8, 8)
        z0 = random.uniform(-50, 50)
        z1 = z - z0
        expected = _plaintext_sigmoid(z)

        prg0_p0 = PRGSync(shared_seed)
        prg0_p1 = PRGSync(shared_seed)
        asprg_p0 = PRGSync(seed_hp_p0)
        comm_p0, comm_p1, comm_hp = make_mock_comm()

        sp0 = SigmoidParty(0, prg0_p0, comm_p0)
        sp1 = SigmoidParty(1, prg0_p1, comm_p1)
        shp = SigmoidHP(asprg_p0, comm_hp)

        out = {}
        _run_three_party(
            lambda: out.__setitem__('s0', sp0.sigmoid(z0)),
            lambda: out.__setitem__('s1', sp1.sigmoid(z1)),
            lambda: shp.serve_sigmoid(),
        )
        result = out['s0'] + out['s1']
        err = abs(result - expected)
        max_abs_err = max(max_abs_err, err)
        sum_abs_err += err
        if shown < verbose_samples:
            print(f'  z={z:8.4f}  sigmoid={expected:.6f}  '
                  f'MPC={result:.6f}  |误差|={err:.2e}')
            shown += 1

    avg = sum_abs_err / num_tests
    print(f'\n平均绝对误差 = {avg:.3e}')
    print(f'最大绝对误差 = {max_abs_err:.3e}')
    ok = max_abs_err < 1e-4
    print(f'指标 (最大绝对误差 < 1e-4): {"[OK] 通过" if ok else "[FAIL] 未达标"}')
    return ok


def verify_gelu(num_tests: int = 1000, verbose_samples: int = 8):
    """[-5, 5] 采样验证 GeLU 精度（技术方案 4.2 节指标：平均绝对误差 < 1e-3）。"""
    import random
    from mcu_core.prg_sync import PRGSync
    from mcu_core.mock_comm import make_mock_comm

    shared_seed = bytes(range(16))
    seed_hp_p0 = bytes(range(16, 32))

    print('=== Π_gelu 安全 GeLU 协议验证 ===')
    print(f'测试 {num_tests} 个样本，x ~ U[-5, 5]\n')

    max_abs_err = 0.0
    sum_abs_err = 0.0
    shown = 0
    for _ in range(num_tests):
        x = random.uniform(-5, 5)
        x0 = random.uniform(-50, 50)
        x1 = x - x0
        expected = _plaintext_gelu(x)

        prg0_p0 = PRGSync(shared_seed)
        prg0_p1 = PRGSync(shared_seed)
        asprg_p0 = PRGSync(seed_hp_p0)
        comm_p0, comm_p1, comm_hp = make_mock_comm()

        gp0 = GeluParty(0, prg0_p0, comm_p0)
        gp1 = GeluParty(1, prg0_p1, comm_p1)
        ghp = GeluHP(asprg_p0, comm_hp)

        out = {}
        _run_three_party(
            lambda: out.__setitem__('g0', gp0.gelu(x0)),
            lambda: out.__setitem__('g1', gp1.gelu(x1)),
            lambda: ghp.serve_gelu(),
        )
        result = out['g0'] + out['g1']
        err = abs(result - expected)
        max_abs_err = max(max_abs_err, err)
        sum_abs_err += err
        if shown < verbose_samples:
            print(f'  x={x:8.4f}  GeLU={expected:10.6f}  '
                  f'MPC={result:10.6f}  |误差|={err:.2e}')
            shown += 1

    avg = sum_abs_err / num_tests
    print(f'\n平均绝对误差 = {avg:.3e}')
    print(f'最大绝对误差 = {max_abs_err:.3e}')
    ok = avg < 1e-3
    print(f'指标 (平均绝对误差 < 1e-3): {"[OK] 通过" if ok else "[FAIL] 未达标"}')
    return ok


if __name__ == '__main__':
    verify_sigmoid()
    print()
    verify_gelu()
