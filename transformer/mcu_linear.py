"""
mcu_linear.py —— 密文线性层
实现 Y = XW 在秘密共享下的计算（三方MCU协议）
"""
import torch
import sys
import os
import threading
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcu_core.prg_sync import PRGSync

L = 2**64
SIGNED_L = 2**63
SCALE = 2**16


def float_to_fixed(x: torch.Tensor):
    """浮点转定点整数列表"""
    return [round(v * SCALE) for v in x.reshape(-1).tolist()]


def share_value(v: int, prg: PRGSync):
    """单个整数秘密共享"""
    r = prg.next(SIGNED_L)
    return r, (v - r)


def share_tensor(x: torch.Tensor, prg: PRGSync):
    """张量秘密共享，返回 (share0_list, share1_list, shape)"""
    fixed = float_to_fixed(x)
    s0, s1 = [], []
    for v in fixed:
        a, b = share_value(v, prg)
        s0.append(a)
        s1.append(b)
    return s0, s1, x.shape


def to_signed(v: int) -> int:
    """无符号转有符号"""
    v = v % L
    return v - L if v > SIGNED_L else v


class MCULinearParty:
    """P0 / P1 的密文线性层"""

    def __init__(self, party_id: int, prg0: PRGSync, comm):
        self.id = party_id
        self.prg0 = prg0
        self.comm = comm

    def forward(self, share_x, share_w, x_shape, w_shape):
        """
        share_x, share_w: 一维份额列表
        x_shape: (batch, seq, d_in)
        w_shape: (d_in, d_out)
        返回：结果份额列表（一维）
        """
        batch, seq, d_in = x_shape
        d_in_w, d_out = w_shape

        # reshape 成易索引的嵌套结构
        def idx_x(b, s, k): return (b * seq + s) * d_in + k
        def idx_w(k, j):    return k * d_out + j

        result = []
        for b in range(batch):
            for s in range(seq):
                for j in range(d_out):
                    acc = 0
                    for k in range(d_in):
                        sx = share_x[idx_x(b, s, k)]
                        sw = share_w[idx_w(k, j)]
                        acc += self._mul(sx, sw)
                    result.append(acc % L)
        return result

    def _mul(self, share_x: int, share_w: int) -> int:
        """单标量MCU乘法"""
        r_x = self.prg0.next(L)
        r_w = self.prg0.next(L)

        if self.id == 0:
            self.comm.send_to_hp({
                'mx': (share_x + r_x) % L,
                'my': (share_w + r_w) % L
            })
        else:
            self.comm.send_to_hp({
                'mx': share_x % L,
                'my': share_w % L
            })

        s_i = self.comm.recv_from_hp()['share']

        if self.id == 0:
            corr = (share_x * r_w + share_w * r_x + r_x * r_w) % L
        else:
            corr = (share_x * r_w + share_w * r_x) % L

        return (s_i - corr) % L


class MCULinearHP:
    """HP 的密文线性层辅助"""

    def __init__(self, asprg_p0: PRGSync, asprg_p1: PRGSync, comm):
        self.asprg_p0 = asprg_p0
        self.comm = comm

    def handle(self, count: int):
        for _ in range(count):
            m0 = self.comm.recv_from_p0()
            m1 = self.comm.recv_from_p1()
            mx = (m0['mx'] + m1['mx']) % L
            my = (m0['my'] + m1['my']) % L
            # 重建成有符号整数再相乘
            product = (to_signed(mx) * to_signed(my)) % L
            s0 = self.asprg_p0.next(L)
            s1 = (product - s0) % L
            self.comm.send_to_p0({'share': s0})
            self.comm.send_to_p1({'share': s1})


def reconstruct(s0_list, s1_list, shape):
    """合并份额，还原浮点矩阵乘法结果（精度 SCALE^2）"""
    combined = []
    for a, b in zip(s0_list, s1_list):
        v = (a + b) % L
        v = to_signed(v)
        combined.append(v / (SCALE * SCALE))
    return torch.tensor(combined).reshape(shape)


def verify_mcu_linear():
    from mcu_core.mock_comm import make_mock_comm

    print('=== 验证密文线性层 ===')
    torch.manual_seed(42)

    X = torch.randn(1, 2, 4) * 0.1
    W = torch.randn(4, 3) * 0.1
    expected = X @ W
    print(f'期望结果 XW:\n{expected}\n')

    # 秘密共享
    x0, x1, xs = share_tensor(X, PRGSync(bytes(range(48, 64))))
    w0, w1, ws = share_tensor(W, PRGSync(bytes(range(64, 80))))

    comm_p0, comm_p1, comm_hp = make_mock_comm()
    p0 = MCULinearParty(0, PRGSync(bytes(range(16))), comm_p0)
    p1 = MCULinearParty(1, PRGSync(bytes(range(16))), comm_p1)
    hp = MCULinearHP(PRGSync(bytes(range(16, 32))),
                     PRGSync(bytes(range(32, 48))), comm_hp)

    batch, seq, d_in = xs
    d_out = ws[1]
    mul_count = batch * seq * d_out * d_in
    print(f'需要 {mul_count} 次标量乘法...\n')

    results = {}
    t0 = threading.Thread(target=lambda: results.__setitem__('r0', p0.forward(x0, w0, xs, ws)))
    t1 = threading.Thread(target=lambda: results.__setitem__('r1', p1.forward(x1, w1, xs, ws)))
    th = threading.Thread(target=lambda: hp.handle(mul_count))

    t0.start(); t1.start(); th.start()
    t0.join(); t1.join(); th.join()

    mpc_result = reconstruct(results['r0'], results['r1'], (batch, seq, d_out))
    print(f'MPC结果:\n{mpc_result}\n')

    error = (mpc_result - expected).abs().max().item()
    print(f'最大误差: {error:.8f}')
    print(f'验证: {"通过" if error < 0.01 else "失败"}')


if __name__ == '__main__':
    verify_mcu_linear()