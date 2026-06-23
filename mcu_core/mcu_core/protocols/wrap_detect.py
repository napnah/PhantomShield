"""
Wrap 检测子协议（对应 MCU 论文 Protocol 4 / 技术方案 2.4 节）

目标
----
给定秘密 x = x0 + x1（两方各持一份额）以及一个三方通过 PRG0 同步获得的
公开掩码 r（两方都完整知道 r），模数为 M。计算溢出标志：

    w = wrap(x, r, M) = floor((x + r) / M)

使得 (x + r) = R + w * M，其中 R = (x + r) mod M ∈ [0, M)。

在 Π_exp 中，HP 只能得到 R = (x+r) mod M（份额各自取模后求和再取模），
无法得知 (x+r) 究竟绕环多少圈，故需要本协议恢复 w，才能在去掩码阶段
用 e^(w*M - r) 修正得到精确的 e^x。

设计说明（面向竞赛工程实现的简化实例化）
--------------------------------------
MCU 论文的 wrap 协议依赖 Sign 协议（Bicoptor 2.0 的概率截断），实现复杂。
本文件给出一个**正确且结构忠实**的简化实例化：

  当 |x| < M（机器学习场景下取 M 远大于取值范围即可满足）时，
  x + r ∈ (-M, 2M)，故 w ∈ {-1, 0, 1}，可由两次"秘密值 vs 公开阈值"
  的符号比较得到：

      w = [x + r >= M] - [x + r < 0]
        = [x >= M - r] - (1 - [x >= -r])

  其中 M-r、-r 都是公开值（两方已知 r、M）。每个比较通过一次
  "乘性掩码 + HP 求和判号"实现：各方把 alpha*(x_i - T) 发给 HP，
  HP 求和得到 alpha*(x - T)，alpha>0 不改变符号，HP 只学到符号位
  （以及被 alpha 随机缩放后的幅度），据此返回比较结果。

安全性：HP 至多得到 w 以及被乘性掩码缩放后的差值符号。结合 Π_exp 中
HP 已持有的 R=(x+r) mod M，即便得知 w，也只能恢复 x+r（实数），而 r
对 HP 未知且在 [0,M) 上均匀，故 x 仍被 r 隐藏。该泄露与 MCU 的加性掩码
隐藏强度一致。生产环境可替换为论文的完整 Sign 协议。
"""

# 符号比较时乘性掩码 alpha 的取值范围上界（alpha ∈ [1, 1+SIGN_SCALE)）
SIGN_SCALE = 2 ** 20


class WrapParty:
    """P0 / P1 执行的 Wrap 检测逻辑。"""

    def __init__(self, party_id: int, prg0, comm):
        self.id = party_id        # 0 或 1
        self.prg0 = prg0          # 与对方同步的 PRG0
        self.comm = comm          # 与 HP 的通信对象

    def _sign_ge(self, share_x: float, threshold: float) -> int:
        """安全判断 x >= threshold ？返回 1（是）/ 0（否）。

        x 为秘密共享（share_x 是本方份额），threshold 为公开值。
        """
        # z = x - threshold，仅 P0 减去公开阈值，保证 z0 + z1 = x - threshold
        z_i = share_x - (threshold if self.id == 0 else 0.0)

        # 乘性正掩码（两方 PRG0 同步，取值相同），alpha > 0 不改变符号
        alpha = 1.0 + self.prg0.next_real(SIGN_SCALE)

        self.comm.send_to_hp({'za': alpha * z_i})
        return self.comm.recv_from_hp()['bit']

    def wrap(self, share_x: float, r: float, M: float) -> int:
        """返回 w = floor((x + r) / M) ∈ {-1, 0, 1}。"""
        ge_hi = self._sign_ge(share_x, M - r)   # x >= M - r  <=>  x + r >= M
        ge_lo = self._sign_ge(share_x, -r)      # x >= -r     <=>  x + r >= 0
        c_hi = ge_hi               # [x + r >= M]
        c_lo = 1 - ge_lo           # [x + r < 0]
        return c_hi - c_lo


class WrapHP:
    """HP 执行的 Wrap 检测逻辑。"""

    def __init__(self, comm):
        self.comm = comm

    def _serve_sign(self):
        """处理一次符号比较：求和判号，把结果广播给两方。"""
        za0 = self.comm.recv_from_p0()['za']
        za1 = self.comm.recv_from_p1()['za']
        total = za0 + za1                      # = alpha * (x - threshold)
        bit = 1 if total >= 0 else 0
        self.comm.send_to_p0({'bit': bit})
        self.comm.send_to_p1({'bit': bit})

    def serve_wrap(self):
        """一次 wrap 检测包含两次符号比较。"""
        self._serve_sign()   # x >= M - r
        self._serve_sign()   # x >= -r


# --------------------------------------------------------------------------- #
# 验证
# --------------------------------------------------------------------------- #
def verify_wrap(num_tests: int = 8, seed_offset: int = 0):
    """用 mock_comm + 多线程验证 wrap 检测正确性。"""
    import math
    import random
    import threading
    from mcu_core.prg_sync import PRGSync
    from mcu_core.mock_comm import make_mock_comm

    M = 256.0
    shared_seed = bytes((i + seed_offset) % 256 for i in range(16))

    print('=== Wrap 检测协议验证 ===')
    print(f'模数 M = {M}\n')

    passed = 0
    for t in range(num_tests):
        # 随机秘密 x 与掩码 r（保证 |x| < M）
        x = random.uniform(-M + 1, M - 1)
        r = random.uniform(0, M)

        # 加性秘密共享 x = x0 + x1
        x0 = random.uniform(-3 * M, 3 * M)
        x1 = x - x0

        expected = math.floor((x + r) / M)

        prg0_p0 = PRGSync(shared_seed)
        prg0_p1 = PRGSync(shared_seed)
        # 预先消费掉 r 之前的状态？这里直接把 r 作为外部已知量传入，
        # PRG0 仅用于符号比较的 alpha，两方同步即可。

        comm_p0, comm_p1, comm_hp = make_mock_comm()

        wp0 = WrapParty(0, prg0_p0, comm_p0)
        wp1 = WrapParty(1, prg0_p1, comm_p1)
        whp = WrapHP(comm_hp)

        out = {}

        def run_p0():
            out['w0'] = wp0.wrap(x0, r, M)

        def run_p1():
            out['w1'] = wp1.wrap(x1, r, M)

        def run_hp():
            whp.serve_wrap()

        threads = [
            threading.Thread(target=run_p0),
            threading.Thread(target=run_p1),
            threading.Thread(target=run_hp),
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        # 两方应得到相同的 w（HP 广播）
        w = out['w0']
        ok = (out['w0'] == out['w1'] == expected)
        passed += ok
        status = '[OK]  ' if ok else '[FAIL]'
        print(f'  {status} x={x:8.3f}, r={r:7.3f}, x+r={x + r:8.3f} '
              f'-> w={w} (期望 {expected})')

    print(f'\n通过 {passed}/{num_tests}')
    return passed == num_tests


if __name__ == '__main__':
    verify_wrap()
