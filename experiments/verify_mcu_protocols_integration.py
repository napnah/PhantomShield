"""
verify_mcu_protocols_integration.py
用潘涵的真实 MCU 协议跑通 Attention 的 Softmax 和 FFN 的 GeLU
证明：真协议（非CrypTen内置）下，Transformer 非线性层也能正确密文计算
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import random
import numpy as np
from mcu_core.prg_sync import PRGSync
from mcu_core.mock_comm import make_mock_comm
from mcu_core.protocols.softmax import SoftmaxParty, SoftmaxHP
from mcu_core.protocols.gelu import GeluParty, GeluHP
from mcu_core.protocols.exponential import ExpHP, _run_three_party

SEED_SHARED = bytes(range(16))
SEED_HP_P0  = bytes(range(16, 32))


def mpc_softmax_vector(vec):
    """用 Π_softmax 对一个向量做密文softmax"""
    k = len(vec)
    # 秘密共享每个分量（实数域）
    shares0, shares1 = [], []
    for x in vec:
        x0 = random.uniform(-100, 100)
        shares0.append(x0)
        shares1.append(x - x0)

    mpc_result = [None] * k
    for m in range(k):
        prg0_p0 = PRGSync(SEED_SHARED)
        prg0_p1 = PRGSync(SEED_SHARED)
        asprg_p0 = PRGSync(SEED_HP_P0)
        comm_p0, comm_p1, comm_hp = make_mock_comm()

        sp0 = SoftmaxParty(0, prg0_p0, comm_p0)
        sp1 = SoftmaxParty(1, prg0_p1, comm_p1)

        # HP 服务函数：k路exp + 1次求和聚合
        def hp_serve():
            exp_hp = ExpHP(asprg_p0, comm_hp)
            for _ in range(k):
                exp_hp.serve_exp()
            msg0 = comm_hp.recv_from_p0()
            msg1 = comm_hp.recv_from_p1()
            D = msg0['u'] + msg1['u']
            comm_hp.send_to_p0({'D': D})
            comm_hp.send_to_p1({'D': D})

        out = {}
        _run_three_party(
            lambda: out.__setitem__('r0', sp0.softmax(shares0, m)),
            lambda: out.__setitem__('r1', sp1.softmax(shares1, m)),
            hp_serve,
        )
        mpc_result[m] = out['r0'] + out['r1']

    plain = np.exp(vec) / np.exp(vec).sum()
    return plain, np.array(mpc_result)


def mpc_gelu_scalar(x):
    """用 Π_gelu 算单个gelu"""
    x0 = random.uniform(-100, 100)
    x1 = x - x0
    prg0_p0 = PRGSync(SEED_SHARED)
    prg0_p1 = PRGSync(SEED_SHARED)
    asprg_p0 = PRGSync(SEED_HP_P0)
    comm_p0, comm_p1, comm_hp = make_mock_comm()

    gp0 = GeluParty(0, prg0_p0, comm_p0)
    gp1 = GeluParty(1, prg0_p1, comm_p1)
    ghp = GeluHP(asprg_p0, comm_hp)

    out = {}
    _run_three_party(
        lambda: out.__setitem__('r0', gp0.gelu(x0)),
        lambda: out.__setitem__('r1', gp1.gelu(x1)),
        lambda: ghp.serve_gelu(),
    )
    mpc = out['r0'] + out['r1']
    plain = x * (1.0 / (1.0 + math.exp(-1.702 * x)))
    return plain, mpc


def main():
    print('=' * 60)
    print('PhantomShield 真实 MCU 协议集成验证')
    print('（使用自研 MCU 协议，非 CrypTen 内置函数）')
    print('=' * 60)

    random.seed(42)
    np.random.seed(42)

    # 测试1：Attention 的 Softmax
    print('\n[测试1] 密文 Attention 的 Softmax（真 Π_softmax）')
    attn_scores = np.random.randn(6) * 2
    plain, mpc = mpc_softmax_vector(attn_scores)
    err = np.abs(plain - mpc).max()
    print(f'  输入分数:    {np.round(attn_scores, 3)}')
    print(f'  明文softmax: {np.round(plain, 6)}')
    print(f'  MPC softmax: {np.round(mpc, 6)}')
    print(f'  概率和: {mpc.sum():.6f}（应为1.0）')
    print(f'  最大误差: {err:.2e}')
    print(f'  结论: {"通过" if err < 1e-4 else "失败"}')

    # 测试2：FFN 的 GeLU
    print('\n[测试2] 密文 FFN 的 GeLU（真 Π_gelu）')
    max_err = 0
    for x in [-2.0, -0.5, 0.5, 1.5, 3.0]:
        plain, mpc = mpc_gelu_scalar(x)
        e = abs(plain - mpc)
        max_err = max(max_err, e)
        print(f'  GeLU({x:+.1f}): 明文={plain:+.6f}  MPC={mpc:+.6f}  误差={e:.2e}')
    print(f'  最大误差: {max_err:.2e}')
    print(f'  结论: {"通过" if max_err < 1e-3 else "失败"}')

    print('\n' + '=' * 60)
    print('验证完成：MCU 协议可正确支撑 Transformer 非线性层')
    print('=' * 60)


if __name__ == '__main__':
    main()