"""
Π_exp：MCU 安全指数协议（对应 MCU 论文 Protocol 5 / 技术方案 2.4 节）

功能：输入 [[x]]（x 加性秘密共享），输出 [[e^x]]（e^x 加性秘密共享）。

核心恒等式（Mask-Compute-Unmask）
--------------------------------
    e^x = e^((x+r) mod M) · e^(w·M - r),   w = wrap(x, r, M) = floor((x+r)/M)

推导：设 R = (x+r) mod M，则 x + r = R + w·M，故
    e^R · e^(w·M - r) = e^(R + w·M - r) = e^(x + r - r) = e^x。

数值说明
--------
论文对超越函数在**浮点域**计算（IEEE 754），只有舍入误差、无近似误差。
为保证 e^(·) 不溢出，本实现采用一个适中的实数模 M（默认 256）：
  - x ∈ [-20, 20] 时 e^R ≤ e^256 ≈ 1.5e111，远小于 float64 上限 ~1.8e308；
  - 修正因子 e^(w·M - r) 同样在可表示范围内；
  - HP 把 E = e^R 拆成正份额 s0 = u·E、s1 = (1-u)·E（u∈[0,1)），
    两份额均落在 [0, E]，乘以正修正因子后求和得到 e^x，避免灾难性相消，
    精度仅受浮点舍入限制。

通信轮次
--------
  第 1 轮：Mask + Compute（各方发掩码值，HP 回份额）
  第 2-3 轮：Wrap 检测（两次符号比较，可并行为 1 轮）
共约 3-4 轮，与论文"4 轮"一致量级。

参与方约定：仅 P0 叠加掩码 r，P1 发送原始份额；两方均通过 PRG0 获知 r，
HP 对各份额取模求和得到 R=(x+r) mod M，无法获知 wrap。
"""
import math

from mcu_core.protocols.wrap_detect import WrapParty, WrapHP

# 超越函数协议统一使用的实数模（需满足 M > 取值范围，且 e^M 不溢出）
MOD = 256.0


class ExpParty:
    """P0 / P1 执行的指数协议逻辑。"""

    def __init__(self, party_id: int, prg0, comm):
        self.id = party_id
        self.prg0 = prg0
        self.comm = comm
        self._wrap = WrapParty(party_id, prg0, comm)

    def exp(self, share_x: float) -> float:
        """输入本方 [x]_i，输出本方 [e^x]_i。"""
        # --- 掩码（两方 PRG0 同步生成相同的实数掩码 r） ---
        r = self.prg0.next_real(MOD)

        # 第 1 轮：发送掩码后的份额（各自取模，保证 HP 求和后得 R=(x+r) mod M）
        if self.id == 0:
            m_i = (share_x + r) % MOD
        else:
            m_i = share_x % MOD
        self.comm.send_to_hp({'m': m_i})

        # 接收 HP 分发的 e^R 份额
        s_i = self.comm.recv_from_hp()['s']

        # 第 2-3 轮：Wrap 检测得到 w = floor((x+r)/M)
        w = self._wrap.wrap(share_x, r, MOD)

        # 去掩码：[e^x]_i = s_i · e^(w·M - r)
        correction = math.exp(w * MOD - r)
        return s_i * correction


class ExpHP:
    """HP 执行的指数协议逻辑。"""

    def __init__(self, asprg_p0, comm):
        self.asprg_p0 = asprg_p0     # 用于生成分发份额的随机性
        self.comm = comm
        self._wrap = WrapHP(comm)

    def serve_exp(self):
        # 第 1 轮：聚合掩码值，计算 R 与 E = e^R
        m0 = self.comm.recv_from_p0()['m']
        m1 = self.comm.recv_from_p1()['m']
        R = (m0 + m1) % MOD
        E = math.exp(R)

        # 拆成两个**正**份额，避免去掩码时灾难性相消
        u = self.asprg_p0.next_unit()
        s0 = u * E
        s1 = E - s0
        self.comm.send_to_p0({'s': s0})
        self.comm.send_to_p1({'s': s1})

        # 第 2-3 轮：协助完成 wrap 检测
        self._wrap.serve_wrap()


# --------------------------------------------------------------------------- #
# 验证
# --------------------------------------------------------------------------- #
def _run_three_party(fn_p0, fn_p1, fn_hp):
    """启动三方线程并等待完成。"""
    import threading
    errors = []

    def wrap(fn):
        def inner():
            try:
                fn()
            except Exception as e:           # noqa: BLE001
                errors.append(e)
                raise
        return inner

    threads = [
        threading.Thread(target=wrap(fn_p0)),
        threading.Thread(target=wrap(fn_p1)),
        threading.Thread(target=wrap(fn_hp)),
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    if errors:
        raise errors[0]


def verify_exp(num_tests: int = 1000, verbose_samples: int = 8):
    """[-10, 10] 均匀采样测试 e^x 精度（技术方案 4.2 节指标：平均绝对误差 < 1e-4）。"""
    import random
    from mcu_core.prg_sync import PRGSync
    from mcu_core.mock_comm import make_mock_comm

    shared_seed = bytes(range(16))
    seed_hp_p0 = bytes(range(16, 32))

    print('=== Π_exp 安全指数协议验证 ===')
    print(f'模数 M = {MOD}，测试样本 {num_tests} 个，x ~ U[-10, 10]\n')

    max_abs_err = 0.0
    sum_abs_err = 0.0
    max_rel_err = 0.0
    shown = 0

    for t in range(num_tests):
        x = random.uniform(-10, 10)
        x0 = random.uniform(-100, 100)
        x1 = x - x0
        expected = math.exp(x)

        prg0_p0 = PRGSync(shared_seed)
        prg0_p1 = PRGSync(shared_seed)
        asprg_p0 = PRGSync(seed_hp_p0)

        comm_p0, comm_p1, comm_hp = make_mock_comm()
        ep0 = ExpParty(0, prg0_p0, comm_p0)
        ep1 = ExpParty(1, prg0_p1, comm_p1)
        ehp = ExpHP(asprg_p0, comm_hp)

        out = {}
        _run_three_party(
            lambda: out.__setitem__('e0', ep0.exp(x0)),
            lambda: out.__setitem__('e1', ep1.exp(x1)),
            lambda: ehp.serve_exp(),
        )

        result = out['e0'] + out['e1']
        abs_err = abs(result - expected)
        rel_err = abs_err / max(abs(expected), 1e-12)
        max_abs_err = max(max_abs_err, abs_err)
        sum_abs_err += abs_err
        max_rel_err = max(max_rel_err, rel_err)

        if shown < verbose_samples:
            print(f'  x={x:8.4f}  e^x={expected:14.6e}  '
                  f'MPC={result:14.6e}  |误差|={abs_err:.2e}')
            shown += 1

    avg_abs_err = sum_abs_err / num_tests
    print(f'\n平均绝对误差 = {avg_abs_err:.3e}')
    print(f'最大绝对误差 = {max_abs_err:.3e}')
    print(f'最大相对误差 = {max_rel_err:.3e}')
    ok = avg_abs_err < 1e-4
    print(f'指标 (平均绝对误差 < 1e-4): {"[OK] 通过" if ok else "[FAIL] 未达标"}')
    return ok


if __name__ == '__main__':
    verify_exp()
