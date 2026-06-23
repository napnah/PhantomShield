"""
Π_softmax：MCU 安全 Softmax 协议（对应 MCU 论文 Protocol 8 / 技术方案 2.5 节）

功能：输入 [[x_1]], ..., [[x_k]]，输出 [[softmax(x_m)]]，其中
    softmax(x_m) = e^{x_m} / Σ_j e^{x_j}

协议流程
--------
1. 并行指数：对每个 x_j 调用 Π_exp，得到各方持有的 [e^{x_j}]_i（已 wrap 修正、精确）。
2. 掩码求和：各方本地求分母份额 p_i = Σ_j [e^{x_j}]_i，用 PRG0 同步的随机数 t，
   把 e^t · p_i 发给 HP；HP 聚合得到公开分母 D_pub = e^t · Σ_j e^{x_j}，广播回各方。
3. 本地相除：各方计算 [softmax(x_m)]_i = [e^{x_m}]_i · e^t / D_pub。

正确性：Σ_i [softmax(x_m)]_i = (e^t / D_pub) · Σ_i [e^{x_m}]_i
        = (e^t / (e^t Σ_j e^{x_j})) · e^{x_m} = e^{x_m} / Σ_j e^{x_j} ✓

安全性（技术方案 定理 3）：HP 只看到 D_pub = e^t · Σ_j e^{x_j}，由于随机掩码 t
对 HP 未知，乘积在 HP 视角下被随机缩放，无法反推任何单个 e^{x_j}，从而不泄露 x_j。

为避免除法（MPC 中昂贵），核心技巧是"把分母公开但用 e^t 隐藏"。

通信轮次：k 路指数（可并行，约 4 轮）+ 掩码求和（1 轮）≈ 6 轮，与论文一致。
"""
import math

from mcu_core.protocols.exponential import ExpParty, ExpHP, MOD, _run_three_party


class SoftmaxParty:
    """P0 / P1 执行的 Softmax 协议逻辑。"""

    def __init__(self, party_id: int, prg0, comm):
        self.id = party_id
        self.prg0 = prg0
        self.comm = comm
        self._exp = ExpParty(party_id, prg0, comm)

    def softmax(self, shares: list, m: int) -> float:
        """输入本方持有的各分量份额列表 shares=[[x_1]_i, ...]，
        返回 [softmax(x_m)]_i。"""
        # 1. 对每个分量并行（此处顺序模拟）执行指数协议
        exp_shares = [self._exp.exp(s) for s in shares]

        # 2. 掩码求和
        t = self.prg0.next_real(MOD)        # 两方同步的随机掩码
        et = math.exp(t)
        p_i = sum(exp_shares)               # 分母份额
        self.comm.send_to_hp({'u': et * p_i})
        d_pub = self.comm.recv_from_hp()['D']

        # 3. 本地相除
        return exp_shares[m] * et / d_pub


class SoftmaxHP:
    """HP 执行的 Softmax 协议逻辑。"""

    def __init__(self, asprg_p0, comm):
        self.asprg_p0 = asprg_p0
        self.comm = comm
        self._exp = ExpHP(asprg_p0, comm)

    def serve_softmax(self, k: int):
        # 1. 协助 k 路指数
        for _ in range(k):
            self._exp.serve_exp()

        # 2. 聚合掩码后的分母并广播
        u0 = self.comm.recv_from_p0()['u']
        u1 = self.comm.recv_from_p1()['u']
        d_pub = u0 + u1                     # = e^t · Σ_j e^{x_j}
        self.comm.send_to_p0({'D': d_pub})
        self.comm.send_to_p1({'D': d_pub})


# --------------------------------------------------------------------------- #
# 验证
# --------------------------------------------------------------------------- #
def _plaintext_softmax(xs):
    mx = max(xs)
    exps = [math.exp(v - mx) for v in xs]
    s = sum(exps)
    return [e / s for e in exps]


def verify_softmax(num_tests: int = 200, k: int = 8, verbose_samples: int = 5):
    """随机向量上验证 softmax 精度（技术方案 4.2 节指标：最大绝对误差 < 1e-4）。"""
    import random
    from mcu_core.prg_sync import PRGSync
    from mcu_core.mock_comm import make_mock_comm

    shared_seed = bytes(range(16))
    seed_hp_p0 = bytes(range(16, 32))

    print('=== Π_softmax 安全 Softmax 协议验证 ===')
    print(f'模数 M = {MOD}，测试 {num_tests} 组，每组向量维度 k = {k}\n')

    max_abs_err = 0.0
    sum_abs_err = 0.0
    count = 0
    shown = 0

    for t in range(num_tests):
        xs = [random.uniform(-8, 8) for _ in range(k)]
        # 加性秘密共享
        xs0 = [random.uniform(-50, 50) for _ in range(k)]
        xs1 = [xs[j] - xs0[j] for j in range(k)]
        expected = _plaintext_softmax(xs)

        # 对每个目标分量 m 验证（这里验证全部分量，确保整向量正确）
        mpc_probs = []
        for m in range(k):
            prg0_p0 = PRGSync(shared_seed)
            prg0_p1 = PRGSync(shared_seed)
            asprg_p0 = PRGSync(seed_hp_p0)

            comm_p0, comm_p1, comm_hp = make_mock_comm()
            sp0 = SoftmaxParty(0, prg0_p0, comm_p0)
            sp1 = SoftmaxParty(1, prg0_p1, comm_p1)
            shp = SoftmaxHP(asprg_p0, comm_hp)

            out = {}
            _run_three_party(
                lambda: out.__setitem__('s0', sp0.softmax(xs0, m)),
                lambda: out.__setitem__('s1', sp1.softmax(xs1, m)),
                lambda: shp.serve_softmax(k),
            )
            mpc_probs.append(out['s0'] + out['s1'])

        for m in range(k):
            err = abs(mpc_probs[m] - expected[m])
            max_abs_err = max(max_abs_err, err)
            sum_abs_err += err
            count += 1

        if shown < verbose_samples:
            ssum = sum(mpc_probs)
            worst = max(abs(mpc_probs[j] - expected[j]) for j in range(k))
            print(f'  组{t}: Σprob={ssum:.6f}  最大分量误差={worst:.2e}')
            shown += 1

    avg_abs_err = sum_abs_err / count
    print(f'\n平均绝对误差 = {avg_abs_err:.3e}')
    print(f'最大绝对误差 = {max_abs_err:.3e}')
    ok = max_abs_err < 1e-4
    print(f'指标 (最大绝对误差 < 1e-4): {"[OK] 通过" if ok else "[FAIL] 未达标"}')
    return ok


if __name__ == '__main__':
    verify_softmax()
